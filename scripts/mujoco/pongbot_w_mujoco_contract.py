#!/usr/bin/env python3

import numpy as np


# ============================================================
# PongbotW Isaac Lab <-> MuJoCo contract
# ============================================================

PHYSICS_DT = 1.0 / 200.0
DECIMATION = 4
CONTROL_DT = PHYSICS_DT * DECIMATION

BASE_INIT_POS = np.array([0.0, 0.0, 0.63], dtype=np.float64)
BASE_INIT_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

LEG_JOINTS = [
    "FL_HR_JOINT", "FL_HP_JOINT", "FL_KN_JOINT",
    "FR_HR_JOINT", "FR_HP_JOINT", "FR_KN_JOINT",
    "RL_HR_JOINT", "RL_HP_JOINT", "RL_KN_JOINT",
    "RR_HR_JOINT", "RR_HP_JOINT", "RR_KN_JOINT",
]

WHEEL_JOINTS = [
    "FL_WHEEL_JOINT",
    "FR_WHEEL_JOINT",
    "RL_WHEEL_JOINT",
    "RR_WHEEL_JOINT",
]

ALL_ACTUATED_JOINTS = LEG_JOINTS + WHEEL_JOINTS

DEFAULT_JOINT_POS = {
    "FL_HR_JOINT": 0.0,
    "FL_HP_JOINT": 0.716,
    "FL_KN_JOINT": -1.396,

    "FR_HR_JOINT": 0.0,
    "FR_HP_JOINT": 0.716,
    "FR_KN_JOINT": -1.396,

    "RL_HR_JOINT": 0.0,
    "RL_HP_JOINT": 0.716,
    "RL_KN_JOINT": -1.396,

    "RR_HR_JOINT": 0.0,
    "RR_HP_JOINT": 0.716,
    "RR_KN_JOINT": -1.396,

    "FL_WHEEL_JOINT": 0.0,
    "FR_WHEEL_JOINT": 0.0,
    "RL_WHEEL_JOINT": 0.0,
    "RR_WHEEL_JOINT": 0.0,
}

LEG_KP = 200.0
LEG_KD = 5.0

TORQUE_LIMIT = {
    "HR": 80.0,
    "HP": 160.0,
    "KN": 280.0,
    "WHEEL": 9.0,
}

WHEEL_KV = 1.0
WHEEL_VEL_LIMIT = 19.0

LEG_ACTION_SCALE = 0.5
WHEEL_ACTION_SCALE = 15.0

OBS_DIM = 60
ACTION_DIM = 16


def joint_kind(joint_name: str) -> str:
    if "_HR_JOINT" in joint_name:
        return "HR"
    if "_HP_JOINT" in joint_name:
        return "HP"
    if "_KN_JOINT" in joint_name:
        return "KN"
    if "_WHEEL_JOINT" in joint_name:
        return "WHEEL"
    raise ValueError(f"Unknown PongbotW joint kind: {joint_name}")
