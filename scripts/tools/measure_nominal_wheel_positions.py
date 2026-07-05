import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Velocity-Flat-PongbotW-Baseline-v0")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app = AppLauncher(args_cli)
simulation_app = app.app

import gymnasium as gym
import torch

import pongbot_w.tasks.manager_based.pongbot_w  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_rotate_inverse

WHEEL_BODY_NAMES = ["FL_WHEEL", "FR_WHEEL", "RL_WHEEL", "RR_WHEEL"]

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
env = gym.make(args_cli.task, cfg=env_cfg)
env.reset()

robot = env.unwrapped.scene["robot"]

body_ids = [robot.body_names.index(name) for name in WHEEL_BODY_NAMES]

root_pos_w = robot.data.root_pos_w[:, None, :]
root_quat_w = robot.data.root_quat_w[:, None, :].expand(-1, len(body_ids), -1)
wheel_pos_w = robot.data.body_pos_w[:, body_ids, :]

rel_w = wheel_pos_w - root_pos_w

q = root_quat_w.reshape(-1, 4)
v = rel_w.reshape(-1, 3)
rel_b = quat_rotate_inverse(q, v).reshape(args_cli.num_envs, len(body_ids), 3)

print("robot.body_names =", robot.body_names)
print("wheel body order =", WHEEL_BODY_NAMES)
print("nominal wheel positions in base frame, env_0:")
for name, pos in zip(WHEEL_BODY_NAMES, rel_b[0].detach().cpu().tolist()):
    print(f'    "{name}": ({pos[0]: .6f}, {pos[1]: .6f}, {pos[2]: .6f}),')

print("\nCopy this list into NOMINAL_WHEEL_POS_B in pongbot_w_env_cfg.py:")
print("NOMINAL_WHEEL_POS_B = [")
for name, pos in zip(WHEEL_BODY_NAMES, rel_b[0].detach().cpu().tolist()):
    print(f"    ({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}),  # {name}")
print("]")

env.close()
simulation_app.close()
