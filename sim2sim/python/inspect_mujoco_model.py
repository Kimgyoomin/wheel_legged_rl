from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SIM2SIM_ROOT = Path(__file__).resolve().parents[1]
if str(_SIM2SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(_SIM2SIM_ROOT))

from common.pongbot_contract import JOINT_ORDER, WHEEL_BODY_NAMES


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="URDF or MJCF path")
    args = parser.parse_args()

    try:
        import mujoco
    except ImportError as exc:
        raise SystemExit("mujoco is required. Install with: python -m pip install mujoco") from exc

    model_path = Path(args.model).expanduser()
    if not model_path.is_file():
        raise SystemExit(f"Model file not found: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    print(f"[INFO] Model: {model_path}")
    print(f"[INFO] nq={model.nq} nv={model.nv} nu={model.nu} timestep={model.opt.timestep}")

    freejoint_found = False
    print("[INFO] Joints:")
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"<unnamed:{joint_id}>"
        jtype = int(model.jnt_type[joint_id])
        qposadr = int(model.jnt_qposadr[joint_id])
        dofadr = int(model.jnt_dofadr[joint_id])
        if jtype == int(mujoco.mjtJoint.mjJNT_FREE):
            freejoint_found = True
        print(f"  - id={joint_id} name={name} type={jtype} qposadr={qposadr} dofadr={dofadr}")

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
    print("[INFO] MuJoCo model inspection passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
