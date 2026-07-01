import argparse
import inspect

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect PongbotW robot.data fields and action/command managers.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-PongbotW-Baseline-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Isaac Sim / Omni modules must be loaded first.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

# Register external project task.
import pongbot_w.tasks.manager_based.pongbot_w  # noqa: F401

from isaaclab_tasks.utils import parse_env_cfg


def print_tensor_summary(name, value, env_idx=0, max_items=12):
    print(f"\n--- {name} ---")
    try:
        print("type:", type(value))
        print("shape:", getattr(value, "shape", None))
        print("dtype:", getattr(value, "dtype", None))
        print("device:", getattr(value, "device", None))

        if torch.is_tensor(value):
            sample = value
            if value.ndim >= 1:
                sample = value[env_idx]
            sample_cpu = sample.detach().cpu().flatten()
            n = min(max_items, sample_cpu.numel())
            print(f"sample env_{env_idx} first {n}:", sample_cpu[:n].tolist())
        else:
            print("value:", value)
    except Exception as e:
        print("summary failed:", repr(e))


def safe_get(obj, name):
    try:
        return getattr(obj, name)
    except Exception as e:
        return f"<ERROR: {repr(e)}>"


def main():
    print("\n=== Args ===")
    print(args_cli)

    device = getattr(args_cli, "device", "cuda:0")
    parse_kwargs = {
        "device": device,
        "num_envs": args_cli.num_envs,
    }

    sig = inspect.signature(parse_env_cfg)
    if "use_fabric" in sig.parameters:
        parse_kwargs["use_fabric"] = not getattr(args_cli, "disable_fabric", False)

    env_cfg = parse_env_cfg(args_cli.task, **parse_kwargs)

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped

    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    print("\n=== Env ===")
    print("env:", env)
    print("unwrapped:", unwrapped)
    print("device:", getattr(unwrapped, "device", None))
    print("num_envs:", getattr(unwrapped, "num_envs", None))
    print("obs type:", type(obs))
    if isinstance(obs, dict):
        print("obs keys:", obs.keys())
        for k, v in obs.items():
            print_tensor_summary(f"obs[{k}]", v)
    else:
        print_tensor_summary("obs", obs)

    print("\n=== Scene keys ===")
    try:
        print(list(unwrapped.scene.keys()))
    except Exception as e:
        print("scene keys failed:", repr(e))

    robot = unwrapped.scene["robot"]

    print("\n=== Robot ===")
    print("robot:", robot)
    print("robot type:", type(robot))
    print("robot prim path:", getattr(robot, "prim_path", None))

    print("\n=== Joint names ===")
    joint_names = None
    for candidate in ["joint_names"]:
        if hasattr(robot, candidate):
            joint_names = getattr(robot, candidate)
            print(f"robot.{candidate}:", joint_names)
    if hasattr(robot, "data") and hasattr(robot.data, "joint_names"):
        print("robot.data.joint_names:", robot.data.joint_names)
        joint_names = robot.data.joint_names

    print("\n=== Body names ===")
    for candidate in ["body_names"]:
        if hasattr(robot, candidate):
            print(f"robot.{candidate}:", getattr(robot, candidate))
    if hasattr(robot, "data") and hasattr(robot.data, "body_names"):
        print("robot.data.body_names:", robot.data.body_names)

    print("\n=== Action Manager ===")
    am = unwrapped.action_manager
    print("action_manager:", am)
    print("total_action_dim:", safe_get(am, "total_action_dim"))
    print("action_term_names:", safe_get(am, "active_terms"))
    print("available attributes containing action/term/target:")
    for name in dir(am):
        lname = name.lower()
        if any(k in lname for k in ["action", "term", "target"]):
            if not name.startswith("__"):
                value = safe_get(am, name)
                print(f"  {name}: type={type(value)} value={value if isinstance(value, (int, float, str, list, tuple)) else ''}")

    # Print internal action terms if available.
    for internal_name in ["_terms", "_term_cfgs"]:
        terms = getattr(am, internal_name, None)
        if terms is not None:
            print(f"\n{internal_name}:")
            if isinstance(terms, dict):
                iterator = terms.items()
            else:
                iterator = enumerate(terms)
            for key, term in iterator:
                print(f"  term {key}: {term}")
                for attr in dir(term):
                    lattr = attr.lower()
                    if any(k in lattr for k in ["action", "target", "joint", "scale", "offset"]):
                        if not attr.startswith("__"):
                            try:
                                val = getattr(term, attr)
                                shape = getattr(val, "shape", None)
                                print(f"    {attr}: type={type(val)} shape={shape}")
                            except Exception:
                                pass

    print("\n=== Command Manager ===")
    cm = getattr(unwrapped, "command_manager", None)
    print("command_manager:", cm)
    if cm is not None:
        for name in dir(cm):
            lname = name.lower()
            if any(k in lname for k in ["command", "term", "active"]):
                if not name.startswith("__"):
                    try:
                        val = getattr(cm, name)
                        print(f"  {name}: type={type(val)}")
                    except Exception:
                        pass
        try:
            cmd = cm.get_command("base_velocity")
            print_tensor_summary("command base_velocity", cmd)
        except Exception as e:
            print("get_command('base_velocity') failed:", repr(e))

    print("\n=== robot.data fields containing joint/torque/effort/root/body/vel/acc ===")
    data = robot.data
    for name in dir(data):
        lname = name.lower()
        if any(k in lname for k in ["joint", "torque", "effort", "root", "body", "vel", "acc", "quat", "pos"]):
            if name.startswith("__"):
                continue
            try:
                value = getattr(data, name)
                shape = getattr(value, "shape", None)
                dtype = getattr(value, "dtype", None)
                device = getattr(value, "device", None)
                print(f"{name:45s} shape={str(shape):25s} dtype={str(dtype):18s} device={device}")
            except Exception as e:
                print(f"{name:45s} ERROR {repr(e)}")

    print("\n=== Step with zero action ===")
    total_action_dim = int(unwrapped.action_manager.total_action_dim)
    zero_action = torch.zeros((args_cli.num_envs, total_action_dim), device=unwrapped.device)

    for i in range(args_cli.steps):
        out = env.step(zero_action)
        obs, rew, terminated, truncated, info = out

        done = terminated | truncated
        print(
            f"step={i:03d} "
            f"rew_mean={rew.mean().item(): .4f} "
            f"terminated_rate={terminated.float().mean().item(): .3f} "
            f"truncated_rate={truncated.float().mean().item(): .3f} "
            f"done_rate={done.float().mean().item(): .3f}"
        )

        if cm is not None:
            try:
                cmd = cm.get_command("base_velocity")
                print_tensor_summary("command base_velocity after step", cmd)
            except Exception:
                pass

        for field in ["root_pos_w", "root_lin_vel_w", "root_ang_vel_w", "joint_pos", "joint_vel"]:
            if hasattr(data, field):
                print_tensor_summary(field, getattr(data, field))

        for field in ["applied_torque", "computed_torque", "joint_acc"]:
            if hasattr(data, field):
                print_tensor_summary(field, getattr(data, field))

    print("\n=== Selected final env_0 tensors ===")
    selected_fields = [
        "joint_pos",
        "joint_vel",
        "joint_acc",
        "applied_torque",
        "computed_torque",
        "root_pos_w",
        "root_quat_w",
        "root_lin_vel_w",
        "root_ang_vel_w",
        "body_pos_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ]

    for field in selected_fields:
        if hasattr(data, field):
            print_tensor_summary(field, getattr(data, field))

    print("\nInspection complete.")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
