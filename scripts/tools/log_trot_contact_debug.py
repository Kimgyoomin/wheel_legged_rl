import argparse
import csv
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Velocity-Flat-PongbotW-Baseline-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--period", type=float, default=0.72)
parser.add_argument("--margin", type=float, default=0.35)
parser.add_argument("--threshold", type=float, default=1.0)
parser.add_argument("--out", default="/tmp/pongbot_trot_contact_debug.csv")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app = AppLauncher(args_cli)
simulation_app = app.app

import gymnasium as gym
import torch

import pongbot_w.tasks.manager_based.pongbot_w  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

WHEEL_BODY_NAMES = ["FL_WHEEL", "FR_WHEEL", "RL_WHEEL", "RR_WHEEL"]
BAD_BODY_NAMES = ["BASE", "FL_HIP", "FR_HIP", "RL_HIP", "RR_HIP",
                  "FL_THIGH", "FR_THIGH", "RL_THIGH", "RR_THIGH",
                  "FL_CALF", "FR_CALF", "RL_CALF", "RR_CALF"]


def get_step_dt(env):
    if hasattr(env.unwrapped, "step_dt"):
        return float(env.unwrapped.step_dt)
    return float(env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation)


def phase_masks(step_idx, step_dt, period, margin, device):
    t = torch.tensor(step_idx * step_dt, device=device, dtype=torch.float32)
    phase = torch.remainder(t / period, 1.0)

    # order: FL, FR, RL, RR
    offsets = torch.tensor([0.0, 0.5, 0.5, 0.0], device=device)
    leg_phase = torch.remainder(phase + offsets, 1.0)
    g = torch.sin(2.0 * math.pi * leg_phase)

    stance = g > margin
    swing = g < -margin
    return phase.item(), g.detach().cpu().tolist(), stance.detach().cpu().tolist(), swing.detach().cpu().tolist()


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    robot = env.unwrapped.scene["robot"]
    sensor = env.unwrapped.scene.sensors["contact_forces"]
    step_dt = get_step_dt(env)

    wheel_ids = [sensor.body_names.index(n) for n in WHEEL_BODY_NAMES if n in sensor.body_names]
    bad_ids = [sensor.body_names.index(n) for n in BAD_BODY_NAMES if n in sensor.body_names]

    print("sensor body names:", sensor.body_names)
    print("wheel ids:", wheel_ids)
    print("bad ids:", bad_ids)
    print("out:", args_cli.out)

    action_dim = env.unwrapped.action_manager.total_action_dim
    action = torch.zeros((args_cli.num_envs, action_dim), device=env.unwrapped.device)

    with open(args_cli.out, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "step", "time", "phase",
            "FL_g", "FR_g", "RL_g", "RR_g",
            "FL_des_stance", "FR_des_stance", "RL_des_stance", "RR_des_stance",
            "FL_des_swing", "FR_des_swing", "RL_des_swing", "RR_des_swing",
            "FL_contact", "FR_contact", "RL_contact", "RR_contact",
            "bad_contact_count",
            "root_z",
            "base_vx",
            "FL_wheel_z", "FR_wheel_z", "RL_wheel_z", "RR_wheel_z",
            "FL_wheel_vel", "FR_wheel_vel", "RL_wheel_vel", "RR_wheel_vel",
        ]
        writer.writerow(header)

        for step in range(args_cli.steps):
            env.step(action)

            phase, g, stance, swing = phase_masks(
                step, step_dt, args_cli.period, args_cli.margin, env.unwrapped.device
            )

            forces = sensor.data.net_forces_w[0]
            wheel_forces = forces[wheel_ids]
            wheel_contact = (torch.norm(wheel_forces, dim=-1) > args_cli.threshold).detach().cpu().tolist()

            bad_contact_count = 0
            if len(bad_ids) > 0:
                bad_forces = forces[bad_ids]
                bad_contact_count = int((torch.norm(bad_forces, dim=-1) > args_cli.threshold).sum().item())

            wheel_body_ids = [robot.body_names.index(n) for n in WHEEL_BODY_NAMES]
            wheel_z = robot.data.body_pos_w[0, wheel_body_ids, 2].detach().cpu().tolist()

            joint_names = robot.joint_names
            wheel_joint_ids = [joint_names.index(n.replace("WHEEL", "WHEEL_JOINT")) if n.replace("WHEEL", "WHEEL_JOINT") in joint_names else None for n in WHEEL_BODY_NAMES]
            # fallback exact names
            exact_wheel_joint_names = ["FL_WHEEL_JOINT", "FR_WHEEL_JOINT", "RL_WHEEL_JOINT", "RR_WHEEL_JOINT"]
            wheel_joint_ids = [joint_names.index(n) for n in exact_wheel_joint_names]
            wheel_vel = robot.data.joint_vel[0, wheel_joint_ids].detach().cpu().tolist()

            row = (
                [step, step * step_dt, phase]
                + g
                + [int(x) for x in stance]
                + [int(x) for x in swing]
                + [int(x) for x in wheel_contact]
                + [
                    bad_contact_count,
                    float(robot.data.root_pos_w[0, 2].item()),
                    float(robot.data.root_lin_vel_b[0, 0].item()) if hasattr(robot.data, "root_lin_vel_b") else float(robot.data.root_lin_vel_w[0, 0].item()),
                ]
                + wheel_z
                + wheel_vel
            )
            writer.writerow(row)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
