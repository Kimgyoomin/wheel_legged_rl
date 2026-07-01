import argparse
import math
import os

from pxr import Usd, UsdGeom, UsdPhysics, Gf


THIGH_LINKS = ["FL_THIGH", "FR_THIGH", "RL_THIGH", "RR_THIGH"]

# wheel joint y offset from URDF:
# FL/RL: +0.0405
# FR/RR: -0.0405
CALF_Y_SIGN = {
    "FL_CALF": +1.0,
    "RL_CALF": +1.0,
    "FR_CALF": -1.0,
    "RR_CALF": -1.0,
}

# Manual refined collider dimensions.
# These are deliberately smaller than the visual mesh.
THIGH_CENTER = (0.0, 0.0, -0.13)
THIGH_SIZE = (0.07, 0.045, 0.22)

CALF_LENGTH = 0.22
CALF_SIZE_X = 0.045
CALF_SIZE_Y = 0.035
CALF_CENTER_Z = -0.165
CALF_HALF_Y_OFFSET = 0.02025  # 0.0405 / 2

# Align CALF box roughly from knee to wheel.
# direction = (0, sign*0.0405, -0.3292)
CALF_ANGLE_DEG = math.degrees(math.atan2(0.0405, 0.3292))


def ensure_scope(stage, path: str):
    prim = stage.GetPrimAtPath(path)
    if not prim:
        prim = UsdGeom.Scope.Define(stage, path).GetPrim()
    return prim


def remove_collision_api_prims_under(stage, collisions_path: str):
    """Remove all CollisionAPI prims under the given collisions scope."""
    to_remove = []
    for prim in stage.TraverseAll():
        path = str(prim.GetPath())
        if path.startswith(collisions_path + "/") and prim.HasAPI(UsdPhysics.CollisionAPI):
            to_remove.append(prim.GetPath())

    for path in sorted(to_remove, key=lambda p: len(str(p)), reverse=True):
        print("remove:", path)
        stage.RemovePrim(path)


def add_box_collision(stage, path, center, size, rpy_deg=(0.0, 0.0, 0.0)):
    cube = UsdGeom.Cube.Define(stage, path)
    prim = cube.GetPrim()
    cube.CreateSizeAttr(1.0)

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(float(center[0]), float(center[1]), float(center[2])))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(float(rpy_deg[0]), float(rpy_deg[1]), float(rpy_deg[2])))
    xform.AddScaleOp().Set(Gf.Vec3f(float(size[0]), float(size[1]), float(size[2])))

    UsdPhysics.CollisionAPI.Apply(prim)
    return prim


def count_collision(stage):
    collision_api = []
    thigh_collision_api = []
    calf_collision_api = []
    wheel_collision_api = []
    joints = []
    wheel_joints = []
    fixed_joints = []
    art_roots = []
    rigid_bodies = []

    for prim in stage.TraverseAll():
        path = str(prim.GetPath())
        upath = path.upper()

        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            art_roots.append(path)

        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_bodies.append(path)

        if (
            prim.IsA(UsdPhysics.RevoluteJoint)
            or prim.IsA(UsdPhysics.FixedJoint)
            or prim.IsA(UsdPhysics.PrismaticJoint)
            or prim.IsA(UsdPhysics.SphericalJoint)
        ):
            joints.append(path)
            if prim.IsA(UsdPhysics.FixedJoint):
                fixed_joints.append(path)
            if "WHEEL" in upath:
                wheel_joints.append(path)

        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_api.append(path)
            if "THIGH" in upath:
                thigh_collision_api.append(path)
            if "CALF" in upath:
                calf_collision_api.append(path)
            if "WHEEL" in upath:
                wheel_collision_api.append(path)

    return {
        "art_roots": art_roots,
        "rigid_bodies": rigid_bodies,
        "joints": joints,
        "fixed_joints": fixed_joints,
        "wheel_joints": wheel_joints,
        "collision_api": collision_api,
        "thigh_collision_api": thigh_collision_api,
        "calf_collision_api": calf_collision_api,
        "wheel_collision_api": wheel_collision_api,
    }


def print_counts(label, counts):
    print(f"\n=== {label} ===")
    print("articulation roots:", len(counts["art_roots"]))
    print("rigid bodies:", len(counts["rigid_bodies"]))
    print("joints:", len(counts["joints"]))
    print("fixed joints:", len(counts["fixed_joints"]))
    print("wheel joints:", len(counts["wheel_joints"]))
    print("CollisionAPI prims:", len(counts["collision_api"]))
    print("THIGH CollisionAPI prims:", len(counts["thigh_collision_api"]))
    print("CALF CollisionAPI prims:", len(counts["calf_collision_api"]))
    print("Wheel CollisionAPI prims:", len(counts["wheel_collision_api"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source USD path")
    parser.add_argument("--dst", required=True, help="Destination USD path")
    parser.add_argument("--thigh-scale", type=float, default=1.0)
    parser.add_argument("--calf-scale", type=float, default=1.0)
    args = parser.parse_args()

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)

    print("SRC:", src)
    print("DST:", dst)
    print("thigh-scale:", args.thigh_scale)
    print("calf-scale:", args.calf_scale)

    if not os.path.exists(src):
        raise FileNotFoundError(src)

    os.makedirs(os.path.dirname(dst), exist_ok=True)

    src_stage = Usd.Stage.Open(src)
    if src_stage is None:
        raise RuntimeError(f"Failed to open source USD: {src}")

    # Flatten/export so destination is editable and self-contained.
    flat_layer = src_stage.Flatten()
    flat_layer.Export(dst)

    stage = Usd.Stage.Open(dst)
    if stage is None:
        raise RuntimeError(f"Failed to open destination USD: {dst}")

    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        raise RuntimeError("Destination USD has no default prim.")

    robot_root = str(default_prim.GetPath())
    print("Robot root:", robot_root)

    print_counts("Before manual THIGH/CALF refinement", count_collision(stage))

    print("\n=== Replacing THIGH collisions ===")
    for link in THIGH_LINKS:
        link_path = f"{robot_root}/{link}"
        if not stage.GetPrimAtPath(link_path):
            print("[WARN] missing link:", link_path)
            continue

        collisions_path = f"{link_path}/collisions"
        ensure_scope(stage, collisions_path)
        remove_collision_api_prims_under(stage, collisions_path)

        size = tuple(s * args.thigh_scale for s in THIGH_SIZE)
        coll_path = f"{collisions_path}/auto_{link}_manual_collision_0"

        prim = add_box_collision(
            stage,
            coll_path,
            center=THIGH_CENTER,
            size=size,
            rpy_deg=(0.0, 0.0, 0.0),
        )

        print(
            f"{link}: {coll_path}, center={THIGH_CENTER}, size={size}, "
            f"type={prim.GetTypeName()}, schemas={list(prim.GetAppliedSchemas())}"
        )

    print("\n=== Replacing/adding CALF collisions ===")
    for link, sign in CALF_Y_SIGN.items():
        link_path = f"{robot_root}/{link}"
        if not stage.GetPrimAtPath(link_path):
            print("[WARN] missing link:", link_path)
            continue

        collisions_path = f"{link_path}/collisions"
        ensure_scope(stage, collisions_path)
        remove_collision_api_prims_under(stage, collisions_path)

        center = (
            0.0,
            sign * CALF_HALF_Y_OFFSET,
            CALF_CENTER_Z,
        )
        size = (
            CALF_SIZE_X * args.calf_scale,
            CALF_SIZE_Y * args.calf_scale,
            CALF_LENGTH * args.calf_scale,
        )
        rpy_deg = (
            sign * CALF_ANGLE_DEG,
            0.0,
            0.0,
        )

        coll_path = f"{collisions_path}/auto_{link}_manual_collision_0"

        prim = add_box_collision(
            stage,
            coll_path,
            center=center,
            size=size,
            rpy_deg=rpy_deg,
        )

        print(
            f"{link}: {coll_path}, center={center}, size={size}, rpy_deg={rpy_deg}, "
            f"type={prim.GetTypeName()}, schemas={list(prim.GetAppliedSchemas())}"
        )

    stage.GetRootLayer().Save()

    stage = Usd.Stage.Open(dst)
    counts = count_collision(stage)
    print_counts("After manual THIGH/CALF refinement", counts)

    print("\nTHIGH CollisionAPI prims:")
    for p in counts["thigh_collision_api"]:
        prim = stage.GetPrimAtPath(p)
        print(" ", p, "| type=", prim.GetTypeName(), "| schemas=", list(prim.GetAppliedSchemas()))

    print("\nCALF CollisionAPI prims:")
    for p in counts["calf_collision_api"]:
        prim = stage.GetPrimAtPath(p)
        print(" ", p, "| type=", prim.GetTypeName(), "| schemas=", list(prim.GetAppliedSchemas()))

    print("\n=== Checks ===")
    if len(counts["collision_api"]) != 17:
        print(f"[WARN] Expected 17 CollisionAPI prims, got {len(counts['collision_api'])}.")
    if len(counts["thigh_collision_api"]) != 4:
        print(f"[WARN] Expected 4 THIGH CollisionAPI prims, got {len(counts['thigh_collision_api'])}.")
    if len(counts["calf_collision_api"]) != 4:
        print(f"[WARN] Expected 4 CALF CollisionAPI prims, got {len(counts['calf_collision_api'])}.")
    if len(counts["wheel_collision_api"]) != 4:
        print(f"[WARN] Expected 4 wheel CollisionAPI prims, got {len(counts['wheel_collision_api'])}.")
    if len(counts["fixed_joints"]) != 0:
        print(f"[WARN] Expected 0 fixed joints, got {len(counts['fixed_joints'])}.")

    print("\nSaved:", dst)


if __name__ == "__main__":
    main()
