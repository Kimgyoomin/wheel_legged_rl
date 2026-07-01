#!/usr/bin/env python3

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from pongbot_w_mujoco_contract import (
    PHYSICS_DT,
    BASE_INIT_POS,
    BASE_INIT_QUAT_WXYZ,
    LEG_JOINTS,
    WHEEL_JOINTS,
    DEFAULT_JOINT_POS,
    LEG_KP,
    LEG_KD,
    TORQUE_LIMIT,
    WHEEL_KV,
    WHEEL_VEL_LIMIT,
    joint_kind,
)


# Wheel order here is fixed as:
# [FL, FR, RL, RR]
#
# MuJoCo joint internal order is NOT trusted.
# We always map by joint name.
WHEEL_SIGN_PATTERNS = [
    np.array([+1.0, +1.0, +1.0, +1.0]),
    np.array([-1.0, -1.0, -1.0, -1.0]),
    np.array([+1.0, -1.0, +1.0, -1.0]),
    np.array([-1.0, +1.0, -1.0, +1.0]),
    np.array([+1.0, +1.0, -1.0, -1.0]),
    np.array([-1.0, -1.0, +1.0, +1.0]),
]


@dataclass
class TeleopState:
    forward_cmd: float = 0.0
    yaw_cmd: float = 0.0
    pattern_index: int = 0
    paused: bool = False
    reset_requested: bool = False


def name_or_none(model: mujoco.MjModel, obj_type: mujoco.mjtObj, obj_id: int) -> str | None:
    return mujoco.mj_id2name(model, obj_type, obj_id)


def build_joint_maps(model: mujoco.MjModel) -> dict[str, dict[str, int]]:
    maps: dict[str, dict[str, int]] = {}

    for jid in range(model.njnt):
        name = name_or_none(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name is None:
            continue

        maps[name] = {
            "jid": jid,
            "qposadr": int(model.jnt_qposadr[jid]),
            "dofadr": int(model.jnt_dofadr[jid]),
            "type": int(model.jnt_type[jid]),
        }

    required = LEG_JOINTS + WHEEL_JOINTS
    missing = [joint_name for joint_name in required if joint_name not in maps]
    if missing:
        raise RuntimeError(
            "Missing expected PongbotW joints:\n"
            + "\n".join(f"  - {name}" for name in missing)
        )

    return maps


def find_free_joint(model: mujoco.MjModel) -> tuple[int, int]:
    for jid in range(model.njnt):
        if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])

    raise RuntimeError(
        "No free joint found. This is not a floating-base locomotion model."
    )


def reset_robot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_maps: dict[str, dict[str, int]],
) -> None:
    free_qposadr, _ = find_free_joint(model)

    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.qfrc_applied[:] = 0.0

    # Free joint qpos layout:
    # [x, y, z, qw, qx, qy, qz]
    data.qpos[free_qposadr + 0] = BASE_INIT_POS[0]
    data.qpos[free_qposadr + 1] = BASE_INIT_POS[1]
    data.qpos[free_qposadr + 2] = BASE_INIT_POS[2]

    data.qpos[free_qposadr + 3] = BASE_INIT_QUAT_WXYZ[0]
    data.qpos[free_qposadr + 4] = BASE_INIT_QUAT_WXYZ[1]
    data.qpos[free_qposadr + 5] = BASE_INIT_QUAT_WXYZ[2]
    data.qpos[free_qposadr + 6] = BASE_INIT_QUAT_WXYZ[3]

    for joint_name, q_default in DEFAULT_JOINT_POS.items():
        qposadr = joint_maps[joint_name]["qposadr"]
        data.qpos[qposadr] = q_default

    mujoco.mj_forward(model, data)


def apply_leg_pd(
    data: mujoco.MjData,
    joint_maps: dict[str, dict[str, int]],
) -> None:
    for joint_name in LEG_JOINTS:
        qposadr = joint_maps[joint_name]["qposadr"]
        dofadr = joint_maps[joint_name]["dofadr"]

        q = data.qpos[qposadr]
        qd = data.qvel[dofadr]
        q_des = DEFAULT_JOINT_POS[joint_name]

        tau = LEG_KP * (q_des - q) - LEG_KD * qd

        kind = joint_kind(joint_name)
        tau = float(np.clip(tau, -TORQUE_LIMIT[kind], TORQUE_LIMIT[kind]))

        data.qfrc_applied[dofadr] = tau


def apply_wheel_velocity_servo(
    data: mujoco.MjData,
    joint_maps: dict[str, dict[str, int]],
    state: TeleopState,
) -> None:
    pattern = WHEEL_SIGN_PATTERNS[state.pattern_index]

    # Rough differential yaw command.
    # This is only for manual testing, not the final policy mapping.
    yaw_pattern = np.array([-1.0, +1.0, -1.0, +1.0])

    wheel_v_des = pattern * state.forward_cmd + yaw_pattern * state.yaw_cmd
    wheel_v_des = np.clip(wheel_v_des, -WHEEL_VEL_LIMIT, WHEEL_VEL_LIMIT)

    for i, joint_name in enumerate(WHEEL_JOINTS):
        dofadr = joint_maps[joint_name]["dofadr"]
        qd = data.qvel[dofadr]

        tau = WHEEL_KV * (wheel_v_des[i] - qd)
        tau = float(np.clip(tau, -TORQUE_LIMIT["WHEEL"], TORQUE_LIMIT["WHEEL"]))

        data.qfrc_applied[dofadr] = tau


def apply_control(
    data: mujoco.MjData,
    joint_maps: dict[str, dict[str, int]],
    state: TeleopState,
) -> None:
    data.qfrc_applied[:] = 0.0
    apply_leg_pd(data, joint_maps)
    apply_wheel_velocity_servo(data, joint_maps, state)


def print_status(state: TeleopState) -> None:
    pattern = WHEEL_SIGN_PATTERNS[state.pattern_index].astype(int).tolist()
    print(
        f"[teleop] forward={state.forward_cmd:+.2f} rad/s | "
        f"yaw={state.yaw_cmd:+.2f} rad/s | "
        f"pattern#{state.pattern_index}={pattern} | "
        f"paused={state.paused}"
    )


def print_joint_mapping(model: mujoco.MjModel, joint_maps: dict[str, dict[str, int]]) -> None:
    print("\n=== PongbotW joint mapping used by teleop ===")
    for joint_name in LEG_JOINTS + WHEEL_JOINTS:
        m = joint_maps[joint_name]
        print(
            f"{joint_name:18s} | "
            f"qposadr={m['qposadr']:2d} | dofadr={m['dofadr']:2d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to MuJoCo MJCF XML or floating URDF")
    args = parser.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    print(f"[INFO] Loading MuJoCo model: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.opt.timestep = PHYSICS_DT

    data = mujoco.MjData(model)

    joint_maps = build_joint_maps(model)
    find_free_joint(model)

    reset_robot(model, data, joint_maps)
    print_joint_mapping(model, joint_maps)

    state = TeleopState()

    print("\n=== PongbotW MuJoCo Keyboard Teleop ===")
    print("W / S : increase / decrease forward wheel velocity")
    print("A / D : yaw command")
    print("X     : stop wheel command")
    print("P     : cycle wheel sign pattern")
    print("R     : reset robot")
    print("Space : pause / resume")
    print("Close viewer window to exit")
    print_status(state)

    def key_callback(keycode: int) -> None:
        try:
            key = chr(keycode).lower()
        except ValueError:
            return

        if key == "w":
            state.forward_cmd += 1.0
        elif key == "s":
            state.forward_cmd -= 1.0
        elif key == "a":
            state.yaw_cmd += 1.0
        elif key == "d":
            state.yaw_cmd -= 1.0
        elif key == "x":
            state.forward_cmd = 0.0
            state.yaw_cmd = 0.0
        elif key == "p":
            state.pattern_index = (state.pattern_index + 1) % len(WHEEL_SIGN_PATTERNS)
        elif key == "r":
            state.reset_requested = True
        elif key == " ":
            state.paused = not state.paused
        else:
            return

        state.forward_cmd = float(np.clip(state.forward_cmd, -WHEEL_VEL_LIMIT, WHEEL_VEL_LIMIT))
        state.yaw_cmd = float(np.clip(state.yaw_cmd, -WHEEL_VEL_LIMIT, WHEEL_VEL_LIMIT))
        print_status(state)

    # Avoid `with ...` because this machine currently segfaults during viewer teardown.
    viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)

    try:
        while viewer.is_running():
            step_start = time.time()

            if state.reset_requested:
                reset_robot(model, data, joint_maps)
                state.reset_requested = False

            if not state.paused:
                apply_control(data, joint_maps, state)
                mujoco.mj_step(model, data)

            viewer.sync()

            elapsed = time.time() - step_start
            sleep_time = model.opt.timestep - elapsed
            if sleep_time > 0.0:
                time.sleep(sleep_time)

        print("[INFO] Viewer closed. Forcing process exit to avoid GLFW teardown segfault.")
        sys.stdout.flush()
        os._exit(0)

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt. Forcing process exit to avoid GLFW teardown segfault.")
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
