# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument(
    "--best_video_interval",
    type=int,
    default=500,
    help="Create a best-checkpoint debug video every N training iterations. Set <= 0 to disable.",
)
parser.add_argument(
    "--best_video_length",
    type=int,
    default=600,
    help="Video length in environment steps for each best-checkpoint debug rollout.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import shutil
import statistics
import os
import subprocess
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import pongbot_w.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _render_best_video(checkpoint_path: str, output_path: str, task: str, video_length: int):
    """Render one headless video using the current best checkpoint and store it at the requested path."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    play_video_dir = os.path.join(os.path.dirname(checkpoint_path), "videos", "play")
    os.makedirs(play_video_dir, exist_ok=True)
    existing_mp4s = {
        os.path.abspath(os.path.join(play_video_dir, file_name))
        for file_name in os.listdir(play_video_dir)
        if file_name.endswith(".mp4")
    }

    cmd = [
        os.path.join(repo_root, "isaaclab.sh"),
        "-p",
        os.path.join(repo_root, "scripts/reinforcement_learning/rsl_rl/play.py"),
        "--task",
        task,
        "--checkpoint",
        checkpoint_path,
        "--num_envs",
        "1",
        "--headless",
        "--video",
        "--video_length",
        str(video_length),
    ]
    print(f"[INFO] Rendering best-checkpoint video: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=repo_root)

    new_mp4s = [
        os.path.abspath(os.path.join(play_video_dir, file_name))
        for file_name in os.listdir(play_video_dir)
        if file_name.endswith(".mp4")
        and os.path.abspath(os.path.join(play_video_dir, file_name)) not in existing_mp4s
    ]
    if not new_mp4s:
        raise RuntimeError(f"No mp4 generated in '{play_video_dir}' after play.py execution.")

    latest_mp4 = max(new_mp4s, key=os.path.getmtime)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    shutil.copy2(latest_mp4, output_path)
    print(f"[INFO] Saved best-checkpoint video to: {output_path}")


def _attach_best_checkpoint_video_hook(runner: OnPolicyRunner, log_dir: str):
    """Attach post-log hook that tracks best reward, saves best.pt, and records periodic videos."""
    if args_cli.best_video_interval <= 0:
        print("[INFO] Best-checkpoint video hook disabled (--best_video_interval <= 0).")
        return

    best_ckpt_path = os.path.join(log_dir, "best.pt")
    best_video_dir = os.path.join(log_dir, "videos")
    os.makedirs(best_video_dir, exist_ok=True)

    state = {"best_mean_reward": float("-inf")}
    original_log = runner.log

    def hooked_log(locs: dict, width: int = 80, pad: int = 35):
        original_log(locs, width, pad)
        rewbuffer = locs.get("rewbuffer")
        if not rewbuffer:
            return

        mean_reward = statistics.mean(rewbuffer)
        iteration = int(locs["it"]) + 1

        if mean_reward > state["best_mean_reward"]:
            state["best_mean_reward"] = mean_reward
            runner.save(best_ckpt_path)
            print(
                f"[INFO] New best checkpoint at iter={iteration}: "
                f"mean_reward={mean_reward:.4f} -> {best_ckpt_path}"
            )

        if iteration % args_cli.best_video_interval != 0:
            return
        if not os.path.exists(best_ckpt_path):
            print(f"[WARN] Skipping best video at iter={iteration}: missing {best_ckpt_path}")
            return

        video_path = os.path.join(best_video_dir, f"iter_{iteration:04d}_best.mp4")
        try:
            _render_best_video(
                checkpoint_path=best_ckpt_path,
                output_path=video_path,
                task=args_cli.task,
                video_length=args_cli.best_video_length,
            )
        except Exception as error:
            print(f"[WARN] Failed to render best video at iter={iteration}: {error}")

    runner.log = hooked_log
    print(
        "[INFO] Best-checkpoint video hook enabled: "
        f"interval={args_cli.best_video_interval}, length={args_cli.best_video_length}"
    )


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner_cfg = agent_cfg.to_dict()
    for key in ("wandb_entity", "wandb_group", "wandb_tags", "wandb_run_name"):
        value = getattr(agent_cfg, key, None)
        if value is not None:
            runner_cfg[key] = value
    runner = OnPolicyRunner(env, runner_cfg, log_dir=log_dir, device=agent_cfg.device)
    _attach_best_checkpoint_video_hook(runner, log_dir)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
