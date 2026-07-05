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
# from isaaclab.utils.math import wrap_to_pi
from isaaclab.utils.math import wrap_to_pi, quat_rotate_inverse

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

def raibert_swing_wheel_placement_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    period: float,
    margin: float,
    nominal_wheel_pos_b: list[tuple[float, float, float]],
    x_gain: float,
    y_gain: float,
    yaw_gain: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Command-conditioned Raibert-style swing wheel placement penalty.

    This term does not replace trot phase rewards.
    It adds "where to place the swing wheel" information.

    Wheel/body order in asset_cfg must be:
        [FL_WHEEL, FR_WHEEL, RL_WHEEL, RR_WHEEL]

    Target in base frame:
        p_target_xy = p_nominal_xy
                    + [x_gain * vx, y_gain * vy]
                    + yaw_gain * [-wz * y_nominal, wz * x_nominal]
    """
    asset: Articulation = env.scene[asset_cfg.name]

    wheel_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    root_pos_w = asset.data.root_pos_w[:, None, :]
    rel_pos_w = wheel_pos_w - root_pos_w

    num_envs = rel_pos_w.shape[0]
    num_wheels = rel_pos_w.shape[1]

    root_quat_w = asset.data.root_quat_w[:, None, :].expand(-1, num_wheels, -1)

    rel_pos_b = quat_rotate_inverse(
        root_quat_w.reshape(-1, 4),
        rel_pos_w.reshape(-1, 3),
    ).reshape(num_envs, num_wheels, 3)

    cmd = env.command_manager.get_command(command_name)
    vx = cmd[:, 0]
    vy = cmd[:, 1]
    wz = cmd[:, 2]

    nominal = torch.tensor(
        nominal_wheel_pos_b,
        device=rel_pos_b.device,
        dtype=rel_pos_b.dtype,
    ).unsqueeze(0).expand(num_envs, -1, -1)

    target_xy = nominal[:, :, :2].clone()

    # Linear velocity placement offset.
    target_xy[:, :, 0] += x_gain * vx.unsqueeze(1)
    target_xy[:, :, 1] += y_gain * vy.unsqueeze(1)

    # Yaw-conditioned tangential placement offset.
    # omega_z x r = [-wz*y, wz*x]
    target_xy[:, :, 0] += yaw_gain * (-wz.unsqueeze(1) * nominal[:, :, 1])
    target_xy[:, :, 1] += yaw_gain * ( wz.unsqueeze(1) * nominal[:, :, 0])

    _, swing, _, _ = _trot_phase_masks(env, period, margin, rel_pos_b.device)

    error_xy = rel_pos_b[:, :, :2] - target_xy
    loss = torch.sum((error_xy ** 2) * swing.float().unsqueeze(-1), dim=(1, 2))
    denom = torch.clamp(swing.float().sum(dim=1), min=1.0)
    return loss / denom

def raibert_swing_wheel_placement_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    period: float,
    margin: float,
    nominal_wheel_pos_b: list[tuple[float, float, float]],
    x_gain: float,
    y_gain: float,
    yaw_gain: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Command-conditioned Raibert-style swing wheel placement penalty.

    This term complements the trot phase/contact reward.

    The phase reward tells the policy:
        - when each wheel should be stance or swing.

    This placement reward tells the policy:
        - where the swing wheel should move according to vx, vy, and yaw command.

    Wheel/body order in asset_cfg and nominal_wheel_pos_b must be:
        [FL_WHEEL, FR_WHEEL, RL_WHEEL, RR_WHEEL]
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Current wheel body positions in world frame.
    wheel_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]

    # Convert wheel position from world frame to base frame.
    root_pos_w = asset.data.root_pos_w[:, None, :]
    rel_pos_w = wheel_pos_w - root_pos_w

    num_envs = rel_pos_w.shape[0]
    num_wheels = rel_pos_w.shape[1]

    root_quat_w = asset.data.root_quat_w[:, None, :].expand(-1, num_wheels, -1)

    rel_pos_b = quat_rotate_inverse(
        root_quat_w.reshape(-1, 4),
        rel_pos_w.reshape(-1, 3),
    ).reshape(num_envs, num_wheels, 3)

    # Command: [vx, vy, wz]
    cmd = env.command_manager.get_command(command_name)
    vx = cmd[:, 0]
    vy = cmd[:, 1]
    wz = cmd[:, 2]

    nominal = torch.tensor(
        nominal_wheel_pos_b,
        device=rel_pos_b.device,
        dtype=rel_pos_b.dtype,
    ).unsqueeze(0).expand(num_envs, -1, -1)

    target_xy = nominal[:, :, :2].clone()

    # Linear command-conditioned placement.
    # vx < 0 moves target backward, vx > 0 moves target forward.
    target_xy[:, :, 0] += x_gain * vx.unsqueeze(1)

    # vy < 0 / vy > 0 shifts target laterally.
    target_xy[:, :, 1] += y_gain * vy.unsqueeze(1)

    # Yaw-conditioned tangential placement.
    # omega_z x r = [-wz * y, wz * x]
    target_xy[:, :, 0] += yaw_gain * (-wz.unsqueeze(1) * nominal[:, :, 1])
    target_xy[:, :, 1] += yaw_gain * (wz.unsqueeze(1) * nominal[:, :, 0])

    # Only apply this penalty to swing wheels.
    _, swing, _, _ = _trot_phase_masks(env, period, margin, rel_pos_b.device)

    error_xy = rel_pos_b[:, :, :2] - target_xy
    loss = torch.sum((error_xy ** 2) * swing.float().unsqueeze(-1), dim=(1, 2))

    denom = torch.clamp(swing.float().sum(dim=1), min=1.0)
    return loss / denom