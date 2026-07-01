import argparse
import os
from pxr import Usd, UsdPhysics


def count_collision(stage):
    collision_api = []
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
            if "WHEEL" in upath:
                wheel_collision_api.append(path)

    return {
        "articulation_roots": art_roots,
        "rigid_bodies": rigid_bodies,
        "joints": joints,
        "fixed_joints": fixed_joints,
        "wheel_joints": wheel_joints,
        "collision_api": collision_api,
        "wheel_collision_api": wheel_collision_api,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source USD path")
    parser.add_argument("--dst", required=True, help="Destination USD path")
    args = parser.parse_args()

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)

    print("SRC:", src)
    print("DST:", dst)

    if not os.path.exists(src):
        raise FileNotFoundError(src)

    os.makedirs(os.path.dirname(dst), exist_ok=True)

    src_stage = Usd.Stage.Open(src)
    if src_stage is None:
        raise RuntimeError(f"Failed to open source USD: {src}")

    # Flatten to make all prim specs editable in the destination layer.
    flat_layer = src_stage.Flatten()
    flat_layer.Export(dst)

    stage = Usd.Stage.Open(dst)
    if stage is None:
        raise RuntimeError(f"Failed to open destination USD: {dst}")

    before = count_collision(stage)
    print("\n=== Before cleanup ===")
    print("articulation roots:", len(before["articulation_roots"]))
    print("rigid bodies:", len(before["rigid_bodies"]))
    print("joints:", len(before["joints"]))
    print("fixed joints:", len(before["fixed_joints"]))
    print("wheel joints:", len(before["wheel_joints"]))
    print("CollisionAPI prims:", len(before["collision_api"]))
    print("Wheel CollisionAPI prims:", len(before["wheel_collision_api"]))

    # Keep only generated primitive colliders whose path contains '/auto_'.
    # Remove importer-generated collision prims:
    # - wheel spheres
    # - duplicated base/hip/thigh boxes
    # - calf mesh collisions
    to_remove = []
    for prim in stage.TraverseAll():
        path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            if "/collisions/" in path and "/auto_" not in path:
                to_remove.append(prim.GetPath())

    # Remove deepest paths first.
    to_remove = sorted(to_remove, key=lambda p: len(str(p)), reverse=True)

    print("\n=== Removing non-auto CollisionAPI prims ===")
    for path in to_remove:
        print("remove:", path)
        stage.RemovePrim(path)

    stage.GetRootLayer().Save()

    # Reopen and verify.
    stage = Usd.Stage.Open(dst)
    after = count_collision(stage)

    print("\n=== After cleanup ===")
    print("articulation roots:", len(after["articulation_roots"]))
    print("rigid bodies:", len(after["rigid_bodies"]))
    print("joints:", len(after["joints"]))
    print("fixed joints:", len(after["fixed_joints"]))
    print("wheel joints:", len(after["wheel_joints"]))
    print("CollisionAPI prims:", len(after["collision_api"]))
    print("Wheel CollisionAPI prims:", len(after["wheel_collision_api"]))

    print("\nWheel CollisionAPI prims:")
    for p in after["wheel_collision_api"]:
        prim = stage.GetPrimAtPath(p)
        print(" ", p, "| type=", prim.GetTypeName(), "| schemas=", list(prim.GetAppliedSchemas()))

    print("\nAll CollisionAPI prims:")
    for p in after["collision_api"]:
        prim = stage.GetPrimAtPath(p)
        print(" ", p, "| type=", prim.GetTypeName(), "| schemas=", list(prim.GetAppliedSchemas()))

    print("\n=== Checks ===")
    if len(after["articulation_roots"]) < 1:
        print("[WARN] No articulation root.")
    if len(after["wheel_joints"]) != 4:
        print(f"[WARN] Expected 4 wheel joints, got {len(after['wheel_joints'])}.")
    if len(after["fixed_joints"]) != 0:
        print(f"[WARN] Expected 0 fixed joints, got {len(after['fixed_joints'])}.")
    if len(after["collision_api"]) != 13:
        print(f"[WARN] Expected 13 CollisionAPI prims for v0, got {len(after['collision_api'])}.")
    if len(after["wheel_collision_api"]) != 4:
        print(f"[WARN] Expected 4 wheel CollisionAPI prims for v0, got {len(after['wheel_collision_api'])}.")

    print("\nSaved clean collision v0 USD:", dst)


if __name__ == "__main__":
    main()
