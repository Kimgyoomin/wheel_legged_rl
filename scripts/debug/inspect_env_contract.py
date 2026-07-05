#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG_SRC = PROJECT_ROOT / "source" / "pongbot_w"

if str(PKG_SRC) not in sys.path:
    sys.path.insert(0, str(PKG_SRC))


parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

# Critical: register external PongbotW tasks after SimulationApp/AppLauncher is ready.
import pongbot_w.tasks.manager_based.pongbot_w  # noqa: F401,E402

from isaaclab_tasks.utils import parse_env_cfg


def flatten_dim(x) -> int:
    if isinstance(x, torch.Tensor):
        if x.ndim >= 2:
            return int(x[0].numel())
        return int(x.numel())

    if isinstance(x, dict):
        return sum(flatten_dim(v) for v in x.values())

    raise TypeError(f"Unsupported observation type: {type(x)}")


def print_obs_tree(prefix: str, obj) -> None:
    if isinstance(obj, torch.Tensor):
        per_env_dim = obj[0].numel() if obj.ndim >= 2 else obj.numel()
        print(f"{prefix}: shape={tuple(obj.shape)}, per_env_dim={per_env_dim}")
    elif isinstance(obj, dict):
        for key, value in obj.items():
            print_obs_tree(f"{prefix}.{key}", value)
    else:
        print(f"{prefix}: type={type(obj)}")


def print_task_spec(task_id: str) -> None:
    spec = gym.spec(task_id)
    print("\n=== Gymnasium spec ===")
    print("id         :", spec.id)
    print("entry_point:", spec.entry_point)
    print("kwargs     :", spec.kwargs)


def main() -> None:
    print("[INFO] PROJECT_ROOT:", PROJECT_ROOT)
    print("[INFO] PKG_SRC     :", PKG_SRC)

    print_task_spec(args_cli.task)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, info = env.reset()

    print("\n=== Raw reset observation tree ===")
    print_obs_tree("obs", obs)

    print("\n=== Spaces ===")
    print("observation_space:", env.observation_space)
    print("action_space     :", env.action_space)

    print("\n=== Action manager ===")
    action_manager = env.unwrapped.action_manager
    print("total_action_dim:", action_manager.total_action_dim)

    for term_name, term in action_manager._terms.items():
        print(f"action term: {term_name}")
        print(f"  class     : {term.__class__.__name__}")
        print(f"  action_dim: {term.action_dim}")

        for attr_name in ["joint_names", "_joint_names"]:
            if hasattr(term, attr_name):
                print(f"  {attr_name}: {getattr(term, attr_name)}")

    print("\n=== Observation manager ===")
    obs_manager = env.unwrapped.observation_manager
    print("active terms:", obs_manager.active_terms)

    if hasattr(obs_manager, "group_obs_dim"):
        print("group_obs_dim:", obs_manager.group_obs_dim)

    if hasattr(obs_manager, "_group_obs_dim"):
        print("_group_obs_dim:", obs_manager._group_obs_dim)

    print("\n=== Final inferred dims ===")
    policy_obs = obs["policy"] if isinstance(obs, dict) and "policy" in obs else obs
    print("policy_obs_per_env_dim:", flatten_dim(policy_obs))
    print("action_dim:", action_manager.total_action_dim)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
