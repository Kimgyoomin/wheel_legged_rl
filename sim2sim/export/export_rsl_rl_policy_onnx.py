"""Export an RSL-RL actor policy checkpoint to ONNX for MuJoCo sim2sim.

Expected policy contract:
    obs:    [1, 62]
    action: [1, 16]
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description="Export PongbotW RSL-RL policy to ONNX.")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--filename", type=str, default="policy.onnx")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--obs_dim", type=int, default=62)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def find_actor_critic(runner):
    """Find actor-critic module across rsl_rl version differences."""
    candidates = []

    for obj_name, obj in [("runner", runner), ("runner.alg", runner.alg)]:
        for attr in ["actor_critic", "policy", "actor_critic_module", "_actor_critic", "_policy"]:
            if hasattr(obj, attr):
                candidates.append((f"{obj_name}.{attr}", getattr(obj, attr)))

    for name, module in candidates:
        if hasattr(module, "act_inference"):
            print(f"[INFO] Using actor-critic from {name}: {type(module)}")
            return module

    print("[ERROR] Could not find actor-critic module with act_inference().")
    print("[DEBUG] Candidates:")
    for name, module in candidates:
        print(" ", name, type(module), "has act_inference:", hasattr(module, "act_inference"))
    raise RuntimeError("Actor-critic module not found. Run inspect_rsl_rl_runner.py for details.")


def find_normalizer(runner):
    """Find observation normalizer if empirical normalization is used."""
    for attr in ["obs_normalizer", "normalizer", "_obs_normalizer"]:
        if hasattr(runner, attr):
            value = getattr(runner, attr)
            if value is not None:
                print(f"[INFO] Found normalizer at runner.{attr}: {type(value)}")
                return value
    print("[INFO] No observation normalizer found.")
    return None


def main():
    args_cli = parse_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym

    import pongbot_w.tasks.manager_based.pongbot_w  # noqa: F401

    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    try:
        from isaaclab_rl.rsl_rl import export_policy_as_onnx
    except Exception as exc:
        raise ImportError(
            "Could not import export_policy_as_onnx from isaaclab_rl.rsl_rl. "
            "Search your Isaac Lab install for export_policy_as_onnx and update this script."
        ) from exc

    task = args_cli.task
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print("Task:", task)
    print("Checkpoint:", checkpoint)
    print("Output dir:", output_dir)

    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    spec = gym.spec(task)
    rsl_rl_cfg_entry_point = spec.kwargs.get("rsl_rl_cfg_entry_point")
    if rsl_rl_cfg_entry_point is None:
        raise RuntimeError(f"No rsl_rl_cfg_entry_point found in task registry for {task}")

    module_name, class_name = rsl_rl_cfg_entry_point.split(":")
    agent_cfg_cls = getattr(importlib.import_module(module_name), class_name)
    agent_cfg = agent_cfg_cls()

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
    runner.load(str(checkpoint))

    # Initialize inference policy path.
    _ = runner.get_inference_policy(device=args_cli.device)

    actor_critic = find_actor_critic(runner)
    normalizer = find_normalizer(runner)

    print("Actor-Critic module:", actor_critic)
    print("Normalizer:", normalizer)

    export_policy_as_onnx(
        actor_critic,
        path=str(output_dir),
        normalizer=normalizer,
        filename=args_cli.filename,
    )

    onnx_path = output_dir / args_cli.filename
    print("Exported ONNX:", onnx_path)

    # ONNX sanity check.
    try:
        import onnx

        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        print("ONNX check: OK")
        print("Inputs:")
        for i in model.graph.input:
            dims = []
            for d in i.type.tensor_type.shape.dim:
                dims.append(d.dim_value if d.dim_value else d.dim_param)
            print(" ", i.name, dims)
        print("Outputs:")
        for o in model.graph.output:
            dims = []
            for d in o.type.tensor_type.shape.dim:
                dims.append(d.dim_value if d.dim_value else d.dim_param)
            print(" ", o.name, dims)
    except Exception as exc:
        print("[WARN] ONNX sanity check failed:", repr(exc))

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
