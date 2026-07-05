from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib
import gymnasium as gym

import pongbot_w.tasks.manager_based.pongbot_w  # noqa: F401

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner


def show_attrs(label, obj):
    print("\n" + "=" * 100)
    print(label, type(obj))
    print("=" * 100)
    for name in dir(obj):
        lname = name.lower()
        if any(k in lname for k in ["actor", "critic", "policy", "normal", "obs", "alg"]):
            if name.startswith("__"):
                continue
            try:
                value = getattr(obj, name)
                print(f"{name:35s} type={type(value)} value={value if isinstance(value, (int, float, str, bool)) else ''}")
            except Exception as e:
                print(f"{name:35s} ERROR {repr(e)}")


task = args_cli.task
checkpoint = Path(args_cli.checkpoint).expanduser().resolve()

env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=args_cli.num_envs)
env = gym.make(task, cfg=env_cfg)
env = RslRlVecEnvWrapper(env)

spec = gym.spec(task)
entry = spec.kwargs["rsl_rl_cfg_entry_point"]
module_name, class_name = entry.split(":")
agent_cfg_cls = getattr(importlib.import_module(module_name), class_name)
agent_cfg = agent_cfg_cls()

runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
runner.load(str(checkpoint))

policy_callable = runner.get_inference_policy(device=args_cli.device)
print("\nInference policy callable:", policy_callable)
print("callable type:", type(policy_callable))

show_attrs("runner", runner)
show_attrs("runner.alg", runner.alg)

# Try likely actor-critic module locations.
print("\n=== Likely actor-critic candidates ===")
candidates = []
for obj_name, obj in [("runner", runner), ("runner.alg", runner.alg)]:
    for attr in ["actor_critic", "policy", "actor_critic_module", "_actor_critic", "_policy"]:
        if hasattr(obj, attr):
            value = getattr(obj, attr)
            candidates.append((f"{obj_name}.{attr}", value))

for name, value in candidates:
    print(name, type(value))
    print("  has act_inference:", hasattr(value, "act_inference"))
    print("  has actor:", hasattr(value, "actor"))
    print("  has critic:", hasattr(value, "critic"))

env.close()
simulation_app.close()
