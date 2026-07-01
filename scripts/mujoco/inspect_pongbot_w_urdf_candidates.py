#!/usr/bin/env python3

from pathlib import Path
import mujoco


URDF_DIR = Path("/home/kim/isaac_projects/pongbot_w/assets/robots/pongbot_w/urdf")

CANDIDATES = [
    URDF_DIR / "PONGBOT_W.urdf",
    URDF_DIR / "PONGBOT_W_isaaclab_recommended.urdf",
]

EXPECTED_JOINTS = [
    "FL_HR_JOINT", "FL_HP_JOINT", "FL_KN_JOINT", "FL_WHEEL_JOINT",
    "FR_HR_JOINT", "FR_HP_JOINT", "FR_KN_JOINT", "FR_WHEEL_JOINT",
    "RL_HR_JOINT", "RL_HP_JOINT", "RL_KN_JOINT", "RL_WHEEL_JOINT",
    "RR_HR_JOINT", "RR_HP_JOINT", "RR_KN_JOINT", "RR_WHEEL_JOINT",
]


def name_or_empty(model, obj_type, obj_id: int) -> str:
    name = mujoco.mj_id2name(model, obj_type, obj_id)
    return "" if name is None else name


def joint_type_name(joint_type: int) -> str:
    table = {
        int(mujoco.mjtJoint.mjJNT_FREE): "free",
        int(mujoco.mjtJoint.mjJNT_BALL): "ball",
        int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
        int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
    }
    return table.get(int(joint_type), f"unknown({joint_type})")


def inspect_model(path: Path) -> None:
    print("\n" + "=" * 90)
    print(f"[MODEL] {path}")
    print("=" * 90)

    if not path.exists():
        print("[FAIL] file does not exist")
        return

    try:
        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
    except Exception as exc:
        print("[FAIL] MuJoCo failed to compile this URDF")
        print(type(exc).__name__ + ":", exc)
        return

    print("\n=== Summary ===")
    print(f"nq     : {model.nq}")
    print(f"nv     : {model.nv}")
    print(f"nu     : {model.nu}")
    print(f"nbody  : {model.nbody}")
    print(f"njnt   : {model.njnt}")
    print(f"ngeom  : {model.ngeom}")
    print(f"dt     : {model.opt.timestep}")

    print("\n=== Joints ===")
    found = set()
    has_free = False

    for jid in range(model.njnt):
        name = name_or_empty(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        found.add(name)

        jtype = int(model.jnt_type[jid])
        if jtype == int(mujoco.mjtJoint.mjJNT_FREE):
            has_free = True

        axis = model.jnt_axis[jid]
        print(
            f"{jid:02d} | {name:28s} | "
            f"type={joint_type_name(jtype):5s} | "
            f"qposadr={int(model.jnt_qposadr[jid]):2d} | "
            f"dofadr={int(model.jnt_dofadr[jid]):2d} | "
            f"axis=[{axis[0]: .3f}, {axis[1]: .3f}, {axis[2]: .3f}]"
        )

    print("\n=== Expected joint check ===")
    missing = []
    for joint_name in EXPECTED_JOINTS:
        if joint_name in found:
            print(f"[OK]      {joint_name}")
        else:
            missing.append(joint_name)
            print(f"[MISSING] {joint_name}")

    print("\n=== Floating base check ===")
    if has_free:
        print("[OK] free joint exists")
    else:
        print("[FAIL] no free joint found")

    print("\n=== Geom / collision name scan ===")
    calf_count = 0
    wheel_count = 0
    for gid in range(model.ngeom):
        geom_name = name_or_empty(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        body_id = int(model.geom_bodyid[gid])
        body_name = name_or_empty(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        contype = int(model.geom_contype[gid])
        conaffinity = int(model.geom_conaffinity[gid])

        label = f"{geom_name} {body_name}".lower()
        if "calf" in label:
            calf_count += 1
        if "wheel" in label:
            wheel_count += 1

        if any(key in label for key in ["base", "hip", "thigh", "calf", "wheel"]):
            print(
                f"{gid:03d} | geom={geom_name:35s} | "
                f"body={body_name:28s} | "
                f"contype={contype} | conaffinity={conaffinity}"
            )

    print("\n=== Verdict ===")
    if missing:
        print("[FAIL] expected actuated joint names are missing")
    elif not has_free:
        print("[FAIL] actuated joints exist, but floating base is missing")
    else:
        print("[OK] candidate is usable for next MuJoCo inspection stage")

    print(f"[INFO] calf-related geom/body hits : {calf_count}")
    print(f"[INFO] wheel-related geom/body hits: {wheel_count}")


def main() -> None:
    for path in CANDIDATES:
        inspect_model(path)


if __name__ == "__main__":
    main()
