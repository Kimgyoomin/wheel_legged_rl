# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import math

from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)

def base_height_tolerance_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    tolerance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base height error outside a tolerance band.

    This is an AOW-style body-height shaping term.

    It does not penalize small deviations inside the tolerance band.
    This is safer than a pure L2 height penalty at the beginning because
    the robot can still settle naturally on its wheels.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    base_z = asset.data.root_pos_w[:, 2]
    height_error = torch.abs(base_z - target_height)

    # No penalty inside tolerance.
    violation = torch.clamp(height_error - tolerance, min=0.0)
    return violation * violation


def joint_deviation_from_default_l1(
    env: ManagerBasedRLEnv,
    default_joint_pos: dict[str, float],
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize selected joints deviating from nominal standing configuration.

    `default_joint_pos` uses suffix matching, for example:
        {
            "HR_JOINT": 0.0,
            "HP_JOINT": 0.716,
            "KN_JOINT": -1.396,
        }

    This is useful for discouraging low-crawling or over-folded postures
    without forcing a handcrafted gait.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]

    target_values = []
    for joint_id in asset_cfg.joint_ids:
        joint_name = asset.data.joint_names[int(joint_id)]

        matched = False
        for suffix, value in default_joint_pos.items():
            if joint_name.endswith(suffix):
                target_values.append(float(value))
                matched = True
                break

        if not matched:
            raise ValueError(
                f"No default joint position specified for joint '{joint_name}'. "
                f"Available default suffixes: {list(default_joint_pos.keys())}"
            )

    target = torch.tensor(target_values, device=joint_pos.device, dtype=joint_pos.dtype).unsqueeze(0)
    return torch.sum(torch.abs(joint_pos - target), dim=1)


def knee_upper_constraint_l2(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_KN_JOINT"]),
) -> torch.Tensor:
    """Penalize knees that move above a safe upper threshold.

    Our nominal knee angle is negative. If the knee approaches zero or positive
    angles, the leg may flip or take an unsafe posture.

    Use this only if the learned gait shows knee flipping or over-extension.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    violation = torch.clamp(joint_pos - threshold, min=0.0)
    return torch.sum(violation * violation, dim=1)

def _get_step_dt(env: ManagerBasedRLEnv) -> float:
    if hasattr(env, "step_dt"):
        return float(env.step_dt)
    try:
        return float(env.cfg.sim.dt * env.cfg.decimation)
    except Exception:
        return 0.02


def _trot_phase_masks(
    env: ManagerBasedRLEnv,
    period: float,
    margin: float,
    device,
):
    """Return stance/swing masks for wheel order [FL, FR, RL, RR].

    Trot diagonal:
        FL + RR in phase
        FR + RL out of phase by 0.5
    """
    step_dt = _get_step_dt(env)
    t = env.episode_length_buf.to(dtype=torch.float32, device=device) * step_dt
    phase = torch.remainder(t / period, 1.0)

    # order: FL, FR, RL, RR
    offsets = torch.tensor([0.0, 0.5, 0.5, 0.0], device=device).unsqueeze(0)
    leg_phase = torch.remainder(phase.unsqueeze(1) + offsets, 1.0)
    g = torch.sin(2.0 * math.pi * leg_phase)

    # soft gated regions:
    # stance when g > margin
    # swing  when g < -margin
    # neutral zone otherwise
    stance = g > margin
    swing = g < -margin
    valid = stance | swing
    return stance, swing, valid, g


def _contact_state_from_sensor(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Return bool contact state [num_envs, num_bodies]."""
    sensor = env.scene.sensors[sensor_cfg.name]

    # Current net contact forces: [num_envs, num_bodies, 3]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    force_norm = torch.norm(forces, dim=-1)
    return force_norm > threshold


def trot_contact_match_reward(
    env: ManagerBasedRLEnv,
    period: float,
    margin: float,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward matching a soft diagonal trot contact schedule.

    Wheel/body order in sensor_cfg must be:
        [FL_WHEEL, FR_WHEEL, RL_WHEEL, RR_WHEEL]
    """
    contact = _contact_state_from_sensor(env, sensor_cfg, threshold)
    stance, swing, valid, _ = _trot_phase_masks(env, period, margin, contact.device)

    match = (stance & contact) | (swing & (~contact))
    match = match & valid

    denom = torch.clamp(valid.float().sum(dim=1), min=1.0)
    return match.float().sum(dim=1) / denom


def trot_swing_contact_penalty(
    env: ManagerBasedRLEnv,
    period: float,
    margin: float,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize wheel contact during enforced swing phase."""
    contact = _contact_state_from_sensor(env, sensor_cfg, threshold)
    _, swing, _, _ = _trot_phase_masks(env, period, margin, contact.device)
    return (swing & contact).float().sum(dim=1)


def trot_stance_miss_penalty(
    env: ManagerBasedRLEnv,
    period: float,
    margin: float,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize missing wheel contact during enforced stance phase."""
    contact = _contact_state_from_sensor(env, sensor_cfg, threshold)
    stance, _, _, _ = _trot_phase_masks(env, period, margin, contact.device)
    return (stance & (~contact)).float().sum(dim=1)


def trot_swing_clearance_l2(
    env: ManagerBasedRLEnv,
    period: float,
    margin: float,
    target_height: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize swing wheels below target height.

    asset_cfg body order must be:
        [FL_WHEEL, FR_WHEEL, RL_WHEEL, RR_WHEEL]
    """
    asset = env.scene[asset_cfg.name]
    wheel_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    _, swing, _, _ = _trot_phase_masks(env, period, margin, wheel_z.device)

    violation = torch.clamp(target_height - wheel_z, min=0.0)
    return torch.sum((violation * swing.float()) ** 2, dim=1)