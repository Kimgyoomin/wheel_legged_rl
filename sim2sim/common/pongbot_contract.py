"""Shared PongbotW policy and low-level control contract for MuJoCo sim2sim."""

from __future__ import annotations

JOINT_ORDER = [
    "FL_HR_JOINT", "FL_HP_JOINT", "FL_KN_JOINT", "FL_WHEEL_JOINT",
    "FR_HR_JOINT", "FR_HP_JOINT", "FR_KN_JOINT", "FR_WHEEL_JOINT",
    "RL_HR_JOINT", "RL_HP_JOINT", "RL_KN_JOINT", "RL_WHEEL_JOINT",
    "RR_HR_JOINT", "RR_HP_JOINT", "RR_KN_JOINT", "RR_WHEEL_JOINT",
]

LEG_JOINT_NAMES = [
    "FL_HR_JOINT", "FL_HP_JOINT", "FL_KN_JOINT",
    "FR_HR_JOINT", "FR_HP_JOINT", "FR_KN_JOINT",
    "RL_HR_JOINT", "RL_HP_JOINT", "RL_KN_JOINT",
    "RR_HR_JOINT", "RR_HP_JOINT", "RR_KN_JOINT",
]

WHEEL_JOINT_NAMES = [
    "FL_WHEEL_JOINT", "FR_WHEEL_JOINT", "RL_WHEEL_JOINT", "RR_WHEEL_JOINT",
]

WHEEL_BODY_NAMES = ["FL_WHEEL", "FR_WHEEL", "RL_WHEEL", "RR_WHEEL"]

DEFAULT_JOINT_POS_BY_SUFFIX = {
    "HR_JOINT": 0.0,
    "HP_JOINT": 0.716,
    "KN_JOINT": -1.396,
    "WHEEL_JOINT": 0.0,
}

LEG_ACTION_SCALE = 0.5
WHEEL_ACTION_SCALE = 8.0

GAIT_PERIOD = 0.72
POLICY_DT = 0.02

KP_BY_SUFFIX = {
    "HR_JOINT": 200.0,
    "HP_JOINT": 200.0,
    "KN_JOINT": 200.0,
}

KD_BY_SUFFIX = {
    "HR_JOINT": 5.0,
    "HP_JOINT": 5.0,
    "KN_JOINT": 5.0,
}

TORQUE_LIMIT_BY_SUFFIX = {
    "HR_JOINT": 80.0,
    "HP_JOINT": 160.0,
    "KN_JOINT": 280.0,
    "WHEEL_JOINT": 9.0,
}

WHEEL_KV = 1.0

OBS_DIM = 62
ACTION_DIM = 16
BASE_RESET_POS_Z = 0.63
BASE_RESET_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)

# Policy output order differs from JOINT_ORDER: 12 leg actions first, 4 wheel actions last.
POLICY_ACTION_ORDER = LEG_JOINT_NAMES + WHEEL_JOINT_NAMES


def joint_suffix(name: str) -> str:
    for suffix in DEFAULT_JOINT_POS_BY_SUFFIX:
        if name.endswith(suffix):
            return suffix
    raise KeyError(f"Unsupported joint name '{name}'.")


def default_joint_position(name: str) -> float:
    return DEFAULT_JOINT_POS_BY_SUFFIX[joint_suffix(name)]


def kp_for_joint(name: str) -> float:
    return KP_BY_SUFFIX.get(joint_suffix(name), 0.0)


def kd_for_joint(name: str) -> float:
    return KD_BY_SUFFIX.get(joint_suffix(name), 0.0)


def torque_limit_for_joint(name: str) -> float:
    return TORQUE_LIMIT_BY_SUFFIX[joint_suffix(name)]
