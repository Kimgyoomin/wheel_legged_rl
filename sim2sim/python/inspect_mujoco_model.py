from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim2sim.common.model_resolver import make_mujoco_resolved_xml
from sim2sim.common.pongbot_contract import JOINT_ORDER, WHEEL_BODY_NAMES


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="URDF or MJCF path")
    parser.add_argument("--mesh-dir", type=str, default=None, help="Optional explicit mesh directory")
    parser.add_argument("--keep-resolved-model", action="store_true", default=True)
    parser.add_argument("--require-freejoint", action="store_true", help="Fail if the model is fixed-base")
    args = parser.parse_args()

    try:
        import mujoco
    except ImportError as exc:
        raise SystemExit("mujoco is required. Install with: python -m pip install mujoco") from exc

    model_path = Path(args.model).expanduser()
    if not model_path.is_file():
        raise SystemExit(f"Model file not found: {model_path}")

    resolved_model_path = make_mujoco_resolved_xml(
        model_path=model_path,
        mesh_dir=args.mesh_dir,
        keep=args.keep_resolved_model,
    )
    print("[INFO] Original model:", model_path)
    print("[INFO] Resolved model:", resolved_model_path)

    model = mujoco.MjModel.from_xml_path(str(resolved_model_path))
    print(f"[INFO] Model: {resolved_model_path}")
    print(f"[INFO] nq={model.nq} nv={model.nv} nu={model.nu} timestep={model.opt.timestep}")

    freejoint_found = False
    hinge_joint_count = 0
    wheel_joint_count = 0
    print("[INFO] Joints:")
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"<unnamed:{joint_id}>"
        jtype = int(model.jnt_type[joint_id])
        qposadr = int(model.jnt_qposadr[joint_id])
        dofadr = int(model.jnt_dofadr[joint_id])
        if jtype == int(mujoco.mjtJoint.mjJNT_FREE):
            freejoint_found = True
        if jtype == int(mujoco.mjtJoint.mjJNT_HINGE):
            hinge_joint_count += 1
        if name in {"FL_WHEEL_JOINT", "FR_WHEEL_JOINT", "RL_WHEEL_JOINT", "RR_WHEEL_JOINT"}:
            wheel_joint_count += 1
        print(f"  - id={joint_id} name={name} type={jtype} qposadr={qposadr} dofadr={dofadr}")

    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "BASE")
    print("[INFO] Bodies:")
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"<unnamed:{body_id}>"
        print(f"  - id={body_id} name={name}")

    missing_joints = [name for name in JOINT_ORDER if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) < 0]
    if missing_joints:
        raise SystemExit(f"Missing required joints: {missing_joints}")

    missing_bodies = [name for name in WHEEL_BODY_NAMES if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) < 0]
    if missing_bodies:
        raise SystemExit(f"Missing required wheel bodies: {missing_bodies}")

    print(f"[INFO] freejoint_present={freejoint_found}")
    print(f"[INFO] BASE body exists={base_body_id >= 0}")
    print(f"[INFO] hinge_joint_count={hinge_joint_count}")
    print(f"[INFO] wheel_joint_count={wheel_joint_count}")
    if args.require_freejoint and not freejoint_found:
        raise RuntimeError("Freejoint is required for locomotion sim2sim, but model is fixed-base.")
    if args.require_freejoint and base_body_id < 0:
        raise RuntimeError("BASE body is required for PongbotW locomotion sim2sim, but it is missing.")
    print("[INFO] MuJoCo model inspection passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
