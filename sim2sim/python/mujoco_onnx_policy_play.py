from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim2sim.common.math_utils import projected_gravity, quat_rotate_inverse
from sim2sim.common.model_resolver import make_mujoco_resolved_xml
from sim2sim.common.pongbot_contract import (
    ACTION_DIM,
    BASE_RESET_POS_Z,
    BASE_RESET_QUAT_WXYZ,
    GAIT_PERIOD,
    JOINT_ORDER,
    LEG_ACTION_SCALE,
    LEG_JOINT_NAMES,
    OBS_DIM,
    POLICY_ACTION_ORDER,
    POLICY_DT,
    WHEEL_ACTION_SCALE,
    WHEEL_BODY_NAMES,
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


def _resolve_body_map(mujoco, model):
    body_map = {}
    for name in WHEEL_BODY_NAMES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise KeyError(f"Body '{name}' not found in MuJoCo model.")
        body_map[name] = body_id
    return body_map


def _find_freejoint(mujoco, model):
    for joint_id in range(model.njnt):
        if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return {
                "joint_id": joint_id,
                "qposadr": int(model.jnt_qposadr[joint_id]),
                "dofadr": int(model.jnt_dofadr[joint_id]),
                "body_id": int(model.jnt_bodyid[joint_id]),
            }
    return None


def _reset_pose(mujoco, model, data, joint_map, freejoint):
    mujoco.mj_resetData(model, data)
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


def _freejoint_base_vel_b(data, freejoint):
    qadr = freejoint["qposadr"]
    dadr = freejoint["dofadr"]
    quat_wxyz = np.array(data.qpos[qadr + 3 : qadr + 7], dtype=np.float64)
    lin_vel_w = np.array(data.qvel[dadr : dadr + 3], dtype=np.float64)
    ang_vel_w = np.array(data.qvel[dadr + 3 : dadr + 6], dtype=np.float64)
    return quat_rotate_inverse(quat_wxyz, lin_vel_w), quat_rotate_inverse(quat_wxyz, ang_vel_w)


def _phase_terms(time_s: float):
    phase = 2.0 * math.pi * ((time_s % GAIT_PERIOD) / GAIT_PERIOD)
    return math.sin(phase), math.cos(phase)


def _joint_pos_rel(data, joint_map):
    return np.array(
        [float(data.qpos[joint_map[name]["qposadr"]] - default_joint_position(name)) for name in JOINT_ORDER],
        dtype=np.float32,
    )


def _joint_vel(data, joint_map):
    return np.array([float(data.qvel[joint_map[name]["dofadr"]]) for name in JOINT_ORDER], dtype=np.float32)


def _build_observation(mujoco, model, data, joint_map, freejoint, previous_action, command):
    base_lin_vel_b, base_ang_vel_b = _freejoint_base_vel_b(data, freejoint)
    qadr = freejoint["qposadr"]
    gravity_b = projected_gravity(np.array(data.qpos[qadr + 3 : qadr + 7], dtype=np.float64)).astype(np.float32)
    phase_sin, phase_cos = _phase_terms(float(data.time))
    obs = np.concatenate(
        [
            base_lin_vel_b.astype(np.float32),
            base_ang_vel_b.astype(np.float32),
            gravity_b,
            np.asarray(command, dtype=np.float32),
            np.array([phase_sin, phase_cos], dtype=np.float32),
            _joint_pos_rel(data, joint_map),
            _joint_vel(data, joint_map),
            previous_action.astype(np.float32),
        ],
        axis=0,
    )
    if obs.shape != (OBS_DIM,):
        raise RuntimeError(f"Expected observation shape ({OBS_DIM},), got {obs.shape}")
    return obs


def _decode_action(action):
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape != (ACTION_DIM,):
        raise RuntimeError(f"Expected policy action shape ({ACTION_DIM},), got {action.shape}")
    leg_target = np.array(
        [default_joint_position(name) + LEG_ACTION_SCALE * action[i] for i, name in enumerate(LEG_JOINT_NAMES)],
        dtype=np.float32,
    )
    wheel_target = WHEEL_ACTION_SCALE * action[len(LEG_JOINT_NAMES) :]
    return leg_target, wheel_target


def _apply_action(data, joint_map, leg_target, wheel_target):
    torques = np.zeros(ACTION_DIM, dtype=np.float32)
    data.qfrc_applied[:] = 0.0
    for i, name in enumerate(LEG_JOINT_NAMES):
        entry = joint_map[name]
        q = float(data.qpos[entry["qposadr"]])
        qd = float(data.qvel[entry["dofadr"]])
        tau = kp_for_joint(name) * (float(leg_target[i]) - q) - kd_for_joint(name) * qd
        tau = float(np.clip(tau, -torque_limit_for_joint(name), torque_limit_for_joint(name)))
        data.qfrc_applied[entry["dofadr"]] = tau
        torques[i] = tau
    for local_i, name in enumerate(WHEEL_JOINT_NAMES):
        entry = joint_map[name]
        qd = float(data.qvel[entry["dofadr"]])
        tau = WHEEL_KV * (float(wheel_target[local_i]) - qd)
        tau = float(np.clip(tau, -torque_limit_for_joint(name), torque_limit_for_joint(name)))
        data.qfrc_applied[entry["dofadr"]] = tau
        torques[len(LEG_JOINT_NAMES) + local_i] = tau
    return torques


def _csv_fieldnames():
    names = ["time", "cmd_vx", "cmd_vy", "cmd_wz", "base_vx", "base_vy", "base_wz"]
    names += [f"projected_gravity_{i}" for i in range(3)]
    names += [f"joint_pos_{name}" for name in JOINT_ORDER]
    names += [f"joint_vel_{name}" for name in JOINT_ORDER]
    names += [f"action_{name}" for name in POLICY_ACTION_ORDER]
    names += [f"leg_target_{name}" for name in LEG_JOINT_NAMES]
    names += [f"wheel_vel_target_{name}" for name in WHEEL_JOINT_NAMES]
    names += [f"torque_{name}" for name in POLICY_ACTION_ORDER]
    return names


def _csv_row(time_s, command, base_lin_vel_b, base_ang_vel_b, gravity_b, joint_pos, joint_vel, action, leg_target, wheel_target, torques):
    row = {
        "time": float(time_s),
        "cmd_vx": float(command[0]),
        "cmd_vy": float(command[1]),
        "cmd_wz": float(command[2]),
        "base_vx": float(base_lin_vel_b[0]),
        "base_vy": float(base_lin_vel_b[1]),
        "base_wz": float(base_ang_vel_b[2]),
    }
    for i in range(3):
        row[f"projected_gravity_{i}"] = float(gravity_b[i])
    for name, value in zip(JOINT_ORDER, joint_pos):
        row[f"joint_pos_{name}"] = float(value)
    for name, value in zip(JOINT_ORDER, joint_vel):
        row[f"joint_vel_{name}"] = float(value)
    for name, value in zip(POLICY_ACTION_ORDER, action):
        row[f"action_{name}"] = float(value)
    for name, value in zip(LEG_JOINT_NAMES, leg_target):
        row[f"leg_target_{name}"] = float(value)
    for name, value in zip(WHEEL_JOINT_NAMES, wheel_target):
        row[f"wheel_vel_target_{name}"] = float(value)
    for name, value in zip(POLICY_ACTION_ORDER, torques):
        row[f"torque_{name}"] = float(value)
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="URDF or MJCF path")
    parser.add_argument("--onnx", required=True, help="Path to policy.onnx")
    parser.add_argument("--dt-policy", type=float, default=POLICY_DT, help="Policy update period [s]")
    parser.add_argument("--log-csv", default=None, help="Optional CSV log path")
    parser.add_argument("--duration", type=float, default=120.0, help="Max duration [s]")
    parser.add_argument("--mesh-dir", type=str, default=None, help="Optional explicit mesh directory")
    parser.add_argument("--keep-resolved-model", action="store_true", default=True)
    args = parser.parse_args()

    try:
        import mujoco
        from mujoco import viewer
    except ImportError as exc:
        raise SystemExit("mujoco is required. Install with: python -m pip install mujoco") from exc
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit("onnxruntime is required. Install with: python -m pip install onnxruntime") from exc

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
    _resolve_body_map(mujoco, model)
    freejoint = _find_freejoint(mujoco, model)
    if freejoint is None:
        raise SystemExit("MuJoCo model has no free joint. PongbotW sim2sim expects a floating base.")

    session = ort.InferenceSession(str(Path(args.onnx).expanduser()), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"[INFO] ONNX input={input_name} output={output_name}")

    _reset_pose(mujoco, model, data, joint_map, freejoint)
    state = {"cmd_vx": 0.0, "cmd_vy": 0.0, "cmd_wz": 0.0, "cmd_scale": 1.0, "paused": False, "reset_requested": False}
    previous_action = np.zeros(ACTION_DIM, dtype=np.float32)
    current_action = np.zeros(ACTION_DIM, dtype=np.float32)
    leg_target = np.array([default_joint_position(name) for name in LEG_JOINT_NAMES], dtype=np.float32)
    wheel_target = np.zeros(len(WHEEL_JOINT_NAMES), dtype=np.float32)
    next_policy_time = 0.0

    csv_file = None
    csv_writer = None
    if args.log_csv:
        csv_path = Path(args.log_csv).expanduser()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path.open("w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=_csv_fieldnames())
        csv_writer.writeheader()
        print(f"[INFO] CSV logging to {csv_path}")

    with viewer.launch_passive(model, data, key_callback=_key_callback_factory(state)) as viewer_handle:
        last_print = time.time()
        while viewer_handle.is_running() and data.time < args.duration:
            if state["reset_requested"]:
                _reset_pose(mujoco, model, data, joint_map, freejoint)
                previous_action[:] = 0.0
                current_action[:] = 0.0
                next_policy_time = data.time
                state["reset_requested"] = False

            torques = np.zeros(ACTION_DIM, dtype=np.float32)
            if not state["paused"]:
                mujoco.mj_step1(model, data)
                command = np.array([state["cmd_vx"], state["cmd_vy"], state["cmd_wz"]], dtype=np.float32)
                if data.time + 1.0e-9 >= next_policy_time:
                    obs = _build_observation(mujoco, model, data, joint_map, freejoint, previous_action, command)
                    current_action = session.run([output_name], {input_name: obs.reshape(1, OBS_DIM)})[0].astype(np.float32).reshape(-1)
                    if current_action.shape != (ACTION_DIM,):
                        raise RuntimeError(f"Expected ONNX output shape ({ACTION_DIM},), got {current_action.shape}")
                    leg_target, wheel_target = _decode_action(current_action)
                    previous_action = current_action.copy()
                    next_policy_time += args.dt_policy
                torques = _apply_action(data, joint_map, leg_target, wheel_target)
                mujoco.mj_step2(model, data)

            base_lin_vel_b, base_ang_vel_b = _freejoint_base_vel_b(data, freejoint)
            qadr = freejoint["qposadr"]
            gravity_b = projected_gravity(np.array(data.qpos[qadr + 3 : qadr + 7], dtype=np.float64))
            joint_pos = _joint_pos_rel(data, joint_map)
            joint_vel = _joint_vel(data, joint_map)

            if csv_writer is not None:
                csv_writer.writerow(
                    _csv_row(
                        data.time,
                        np.array([state["cmd_vx"], state["cmd_vy"], state["cmd_wz"]], dtype=np.float32),
                        base_lin_vel_b,
                        base_ang_vel_b,
                        gravity_b,
                        joint_pos,
                        joint_vel,
                        current_action,
                        leg_target,
                        wheel_target,
                        torques,
                    )
                )

            viewer_handle.sync()
            now = time.time()
            if now - last_print > 0.5:
                print(
                    f"[policy] t={data.time:7.3f} cmd=({state['cmd_vx']:+.2f}, "
                    f"{state['cmd_vy']:+.2f}, {state['cmd_wz']:+.2f}) "
                    f"base_vel=({base_lin_vel_b[0]:+.2f}, {base_lin_vel_b[1]:+.2f}, {base_ang_vel_b[2]:+.2f}) "
                    f"action_norm={np.linalg.norm(current_action):.3f} max_torque={np.max(np.abs(torques)):.3f}"
                )
                last_print = now

    if csv_file is not None:
        csv_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
