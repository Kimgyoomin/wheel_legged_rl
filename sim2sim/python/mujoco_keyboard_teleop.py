from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim2sim.common.model_resolver import make_mujoco_resolved_xml
from sim2sim.common.pongbot_contract import (
    BASE_RESET_POS_Z,
    BASE_RESET_QUAT_WXYZ,
    JOINT_ORDER,
    LEG_JOINT_NAMES,
    WHEEL_ACTION_SCALE,
    WHEEL_JOINT_NAMES,
    WHEEL_KV,
    default_joint_position,
    kd_for_joint,
    kp_for_joint,
    torque_limit_for_joint,
)


def _resolve_joint_map(mujoco, model):
    joint_map = {}
    for name in JOINT_ORDER:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise KeyError(f"Joint '{name}' not found in MuJoCo model.")
        joint_map[name] = {
            "joint_id": joint_id,
            "qposadr": int(model.jnt_qposadr[joint_id]),
            "dofadr": int(model.jnt_dofadr[joint_id]),
        }
    return joint_map


def _find_freejoint(mujoco, model):
    for joint_id in range(model.njnt):
        if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return {
                "joint_id": joint_id,
                "qposadr": int(model.jnt_qposadr[joint_id]),
                "dofadr": int(model.jnt_dofadr[joint_id]),
            }
    return None


def _reset_pose(mujoco, model, data, joint_map, freejoint):
    mujoco.mj_resetData(model, data)
    if freejoint is not None:
        qadr = freejoint["qposadr"]
        dadr = freejoint["dofadr"]
        data.qpos[qadr : qadr + 3] = np.array([0.0, 0.0, BASE_RESET_POS_Z], dtype=np.float64)
        data.qpos[qadr + 3 : qadr + 7] = np.array(BASE_RESET_QUAT_WXYZ, dtype=np.float64)
        data.qvel[dadr : dadr + 6] = 0.0
    for name, entry in joint_map.items():
        data.qpos[entry["qposadr"]] = default_joint_position(name)
        data.qvel[entry["dofadr"]] = 0.0
    data.qfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)


def _key_callback_factory(state):
    def _callback(keycode):
        ch = chr(keycode).lower() if 32 <= keycode <= 126 else ""
        if ch == "w":
            state["cmd_vx"] += 0.2 * state["cmd_scale"]
        elif ch == "s":
            state["cmd_vx"] -= 0.2 * state["cmd_scale"]
        elif ch == "a":
            state["cmd_vy"] += 0.2 * state["cmd_scale"]
        elif ch == "d":
            state["cmd_vy"] -= 0.2 * state["cmd_scale"]
        elif ch == "q":
            state["cmd_wz"] += 0.3 * state["cmd_scale"]
        elif ch == "e":
            state["cmd_wz"] -= 0.3 * state["cmd_scale"]
        elif ch == "x":
            state["cmd_vx"] = 0.0
            state["cmd_vy"] = 0.0
            state["cmd_wz"] = 0.0
        elif ch == "r":
            state["reset_requested"] = True
        elif ch == "1":
            state["cmd_scale"] = 0.25
        elif ch == "2":
            state["cmd_scale"] = 0.5
        elif ch == "3":
            state["cmd_scale"] = 1.0
        elif keycode == 32:
            state["paused"] = not state["paused"]

    return _callback


def _compute_wheel_targets(cmd_vx: float, cmd_wz: float, wheelbase_half: float = 0.2):
    # Lateral command is intentionally not mapped here. Stable vy control is policy/gait-dependent.
    left = WHEEL_ACTION_SCALE * (cmd_vx - wheelbase_half * cmd_wz)
    right = WHEEL_ACTION_SCALE * (cmd_vx + wheelbase_half * cmd_wz)
    return {
        "FL_WHEEL_JOINT": left,
        "RL_WHEEL_JOINT": left,
        "FR_WHEEL_JOINT": right,
        "RR_WHEEL_JOINT": right,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="URDF or MJCF path")
    parser.add_argument("--mesh-dir", type=str, default=None, help="Optional explicit mesh directory")
    parser.add_argument("--keep-resolved-model", action="store_true", default=True)
    args = parser.parse_args()

    try:
        import mujoco
        from mujoco import viewer
    except ImportError as exc:
        raise SystemExit("mujoco is required. Install with: python -m pip install mujoco") from exc

    resolved_model_path = make_mujoco_resolved_xml(
        model_path=args.model,
        mesh_dir=args.mesh_dir,
        keep=args.keep_resolved_model,
    )
    print("[INFO] Original model:", args.model)
    print("[INFO] Resolved model:", resolved_model_path)

    model = mujoco.MjModel.from_xml_path(str(resolved_model_path))
    data = mujoco.MjData(model)
    joint_map = _resolve_joint_map(mujoco, model)
    freejoint = _find_freejoint(mujoco, model)
    if freejoint is None:
        raise RuntimeError("This model has no freejoint. Use build_freebase_mjcf.py first.")
    _reset_pose(mujoco, model, data, joint_map, freejoint)

    state = {
        "cmd_vx": 0.0,
        "cmd_vy": 0.0,
        "cmd_wz": 0.0,
        "cmd_scale": 1.0,
        "paused": False,
        "reset_requested": False,
    }

    with viewer.launch_passive(model, data, key_callback=_key_callback_factory(state)) as viewer_handle:
        last_print = time.time()
        while viewer_handle.is_running():
            if state["reset_requested"]:
                _reset_pose(mujoco, model, data, joint_map, freejoint)
                state["reset_requested"] = False

            max_torque = 0.0
            if not state["paused"]:
                data.qfrc_applied[:] = 0.0
                mujoco.mj_step1(model, data)
                wheel_targets = _compute_wheel_targets(state["cmd_vx"], state["cmd_wz"])
                for name in LEG_JOINT_NAMES:
                    entry = joint_map[name]
                    q = float(data.qpos[entry["qposadr"]])
                    qd = float(data.qvel[entry["dofadr"]])
                    tau = kp_for_joint(name) * (default_joint_position(name) - q) - kd_for_joint(name) * qd
                    tau = float(np.clip(tau, -torque_limit_for_joint(name), torque_limit_for_joint(name)))
                    data.qfrc_applied[entry["dofadr"]] = tau
                    max_torque = max(max_torque, abs(tau))
                for name in WHEEL_JOINT_NAMES:
                    entry = joint_map[name]
                    qd = float(data.qvel[entry["dofadr"]])
                    tau = WHEEL_KV * (wheel_targets[name] - qd)
                    tau = float(np.clip(tau, -torque_limit_for_joint(name), torque_limit_for_joint(name)))
                    data.qfrc_applied[entry["dofadr"]] = tau
                    max_torque = max(max_torque, abs(tau))
                mujoco.mj_step2(model, data)

            viewer_handle.sync()
            now = time.time()
            if now - last_print > 0.5:
                print(
                    f"[teleop] t={data.time:7.3f} cmd=({state['cmd_vx']:+.2f}, "
                    f"{state['cmd_vy']:+.2f}, {state['cmd_wz']:+.2f}) "
                    f"scale={state['cmd_scale']:.2f} paused={state['paused']} max_tau={max_torque:.2f}"
                )
                last_print = now
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
