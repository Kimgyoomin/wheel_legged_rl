import argparse
import os
from pxr import Usd, UsdGeom, UsdPhysics, Gf


THIGH_LINKS = ["FL_THIGH", "FR_THIGH", "RL_THIGH", "RR_THIGH"]
CALF_LINKS = ["FL_CALF", "FR_CALF", "RL_CALF", "RR_CALF"]


def ensure_scope(stage, path: str):
    prim = stage.GetPrimAtPath(path)
    if not prim:
        prim = UsdGeom.Scope.Define(stage, path).GetPrim()
    return prim


def remove_collision_children(stage, link_path: str):
    """Remove all existing CollisionAPI prims under link_path/collisions.

    We only use this for THIGH and CALF. BASE/HIP/WHEEL are left untouched.
    """
    collisions_path = f"{link_path}/collisions"
    collisions_prim = stage.GetPrimAtPath(collisions_path)
    if not collisions_prim:
        return

    to_remove = []
    for prim in stage.TraverseAll():
        path = str(prim.GetPath())
        if path.startswith(collisions_path + "/") and prim.HasAPI(UsdPhysics.CollisionAPI):
            to_remove.append(prim.GetPath())

    # Remove deepest prims first.
    to_remove = sorted(to_remove, key=lambda p: len(str(p)), reverse=True)
    for path in to_remove:
        print("remove collision:", path)
        stage.RemovePrim(path)


def get_local_bbox(stage, link_path: str):
    link_prim = stage.GetPrimAtPath(link_path)
    if not link_prim:
        raise RuntimeError(f"Missing link prim: {link_path}")

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )

    bbox = bbox_cache.ComputeLocalBound(link_prim)
    rng = bbox.GetRange()
    mn = rng.GetMin()
    mx = rng.GetMax()

    center = Gf.Vec3d(
        0.5 * (mn[0] + mx[0]),
        0.5 * (mn[1] + mx[1]),
        0.5 * (mn[2] + mx[2]),
    )
    size = Gf.Vec3d(
        mx[0] - mn[0],
        mx[1] - mn[1],
        mx[2] - mn[2],
    )
    return center, size


def add_refined_box(
    stage,
    path: str,
    center: Gf.Vec3d,
    size: Gf.Vec3d,
    long_axis_scale: float,
    cross_axis_scale: float,
    center_offset_long_axis: float = 0.0,
):
    """Create a refined box collider.

    The longest local bbox axis is assumed to be the link length axis.
    We shrink that axis more strongly to avoid covering joint regions.
    cross_axis_scale shrinks the other two axes.
    center_offset_long_axis can move the box along the long axis if needed.
    """
    dims = [float(size[0]), float(size[1]), float(size[2])]
    long_axis = max(range(3), key=lambda i: dims[i])

    final_size = []
    for i, d in enumerate(dims):
        if i == long_axis:
            final_size.append(max(d * long_axis_scale, 0.015))
        else:
            final_size.append(max(d * cross_axis_scale, 0.015))

    final_center = Gf.Vec3d(center)
    final_center[long_axis] += float(size[long_axis]) * float(center_offset_long_axis)

    cube = UsdGeom.Cube.Define(stage, path)
    prim = cube.GetPrim()
    cube.CreateSizeAttr(1.0)

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(final_center)
    xform.AddScaleOp().Set(Gf.Vec3f(final_size[0], final_size[1], final_size[2]))

    UsdPhysics.CollisionAPI.Apply(prim)

    return prim, long_axis, final_center, final_size


def count_collision(stage):
    collision_api = []
    wheel_collision_api = []
    thigh_collision_api = []
    calf_collision_api = []
    joints = []
    fixed_joints = []
    wheel_joints = []
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
            if "THIGH" in upath:
                thigh_collision_api.append(path)
            if "CALF" in upath:
                calf_collision_api.append(path)

    return {
        "articulation_roots": art_roots,
        "rigid_bodies": rigid_bodies,
        "joints": joints,
        "fixed_joints": fixed_joints,
        "wheel_joints": wheel_joints,
        "collision_api": collision_api,
        "wheel_collision_api": wheel_collision_api,
        "thigh_collision_api": thigh_collision_api,
        "calf_collision_api": calf_collision_api,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)

    # These are intentionally conservative. If contact does not trigger enough,
    # increase long_axis_scale slightly. If normal stance contacts occur, decrease it.
    parser.add_argument("--thigh-long-scale", type=float, default=0.60)
    parser.add_argument("--thigh-cross-scale", type=float, default=0.85)
    parser.add_argument("--calf-long-scale", type=float, default=0.65)
    parser.add_argument("--calf-cross-scale", type=float, default=0.75)

    # Optional offsets along the local longest axis.
    # Default 0 means shrink symmetrically around center.
    parser.add_argument("--thigh-offset", type=float, default=0.0)
    parser.add_argument("--calf-offset", type=float, default=0.0)

    args = parser.parse_args()

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)

    print("SRC:", src)
    print("DST:", dst)
    print("thigh long/cross/offset:", args.thigh_long_scale, args.thigh_cross_scale, args.thigh_offset)
    print("calf  long/cross/offset:", args.calf_long_scale, args.calf_cross_scale, args.calf_offset)

    if not os.path.exists(src):
        raise FileNotFoundError(src)

    os.makedirs(os.path.dirname(dst), exist_ok=True)

    src_stage = Usd.Stage.Open(src)
    if src_stage is None:
        raise RuntimeError(f"Failed to open source USD: {src}")

    # Flatten to create an editable destination.
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

    before = count_collision(stage)
    print("\n=== Before refinement ===")
    print("articulation roots:", len(before["articulation_roots"]))
    print("rigid bodies:", len(before["rigid_bodies"]))
    print("joints:", len(before["joints"]))
    print("fixed joints:", len(before["fixed_joints"]))
    print("wheel joints:", len(before["wheel_joints"]))
    print("CollisionAPI prims:", len(before["collision_api"]))
    print("Wheel CollisionAPI prims:", len(before["wheel_collision_api"]))
    print("THIGH CollisionAPI prims:", len(before["thigh_collision_api"]))
    print("CALF CollisionAPI prims:", len(before["calf_collision_api"]))

    print("\n=== Refining THIGH collisions ===")
    for link in THIGH_LINKS:
        link_path = f"{robot_root}/{link}"
        if not stage.GetPrimAtPath(link_path):
            print("[WARN] missing link:", link_path)
            continue

        ensure_scope(stage, f"{link_path}/collisions")
        remove_collision_children(stage, link_path)

        center, size = get_local_bbox(stage, link_path)
        coll_path = f"{link_path}/collisions/auto_{link}_refined_collision_0"

        prim, long_axis, final_center, final_size = add_refined_box(
            stage,
            coll_path,
            center,
            size,
            long_axis_scale=args.thigh_long_scale,
            cross_axis_scale=args.thigh_cross_scale,
            center_offset_long_axis=args.thigh_offset,
        )

        print(
            f"{link}: path={coll_path}, long_axis={long_axis}, "
            f"bbox_size=({size[0]:.4f},{size[1]:.4f},{size[2]:.4f}), "
            f"center=({center[0]:.4f},{center[1]:.4f},{center[2]:.4f}), "
            f"final_center=({final_center[0]:.4f},{final_center[1]:.4f},{final_center[2]:.4f}), "
            f"final_size=({final_size[0]:.4f},{final_size[1]:.4f},{final_size[2]:.4f}), "
            f"type={prim.GetTypeName()}, schemas={list(prim.GetAppliedSchemas())}"
        )

    print("\n=== Refining CALF collisions ===")
    for link in CALF_LINKS:
        link_path = f"{robot_root}/{link}"
        if not stage.GetPrimAtPath(link_path):
            print("[WARN] missing link:", link_path)
            continue

        ensure_scope(stage, f"{link_path}/collisions")
        remove_collision_children(stage, link_path)

        center, size = get_local_bbox(stage, link_path)
        coll_path = f"{link_path}/collisions/auto_{link}_refined_collision_0"

        prim, long_axis, final_center, final_size = add_refined_box(
            stage,
            coll_path,
            center,
            size,
            long_axis_scale=args.calf_long_scale,
            cross_axis_scale=args.calf_cross_scale,
            center_offset_long_axis=args.calf_offset,
        )

        print(
            f"{link}: path={coll_path}, long_axis={long_axis}, "
            f"bbox_size=({size[0]:.4f},{size[1]:.4f},{size[2]:.4f}), "
            f"center=({center[0]:.4f},{center[1]:.4f},{center[2]:.4f}), "
            f"final_center=({final_center[0]:.4f},{final_center[1]:.4f},{final_center[2]:.4f}), "
            f"final_size=({final_size[0]:.4f},{final_size[1]:.4f},{final_size[2]:.4f}), "
            f"type={prim.GetTypeName()}, schemas={list(prim.GetAppliedSchemas())}"
        )

    stage.GetRootLayer().Save()

    stage = Usd.Stage.Open(dst)
    after = count_collision(stage)

    print("\n=== After refinement ===")
    print("articulation roots:", len(after["articulation_roots"]))
    print("rigid bodies:", len(after["rigid_bodies"]))
    print("joints:", len(after["joints"]))
    print("fixed joints:", len(after["fixed_joints"]))
    print("wheel joints:", len(after["wheel_joints"]))
    print("CollisionAPI prims:", len(after["collision_api"]))
    print("Wheel CollisionAPI prims:", len(after["wheel_collision_api"]))
    print("THIGH CollisionAPI prims:", len(after["thigh_collision_api"]))
    print("CALF CollisionAPI prims:", len(after["calf_collision_api"]))

    print("\nTHIGH CollisionAPI prims:")
    for p in after["thigh_collision_api"]:
        prim = stage.GetPrimAtPath(p)
        print(" ", p, "| type=", prim.GetTypeName(), "| schemas=", list(prim.GetAppliedSchemas()))

    print("\nCALF CollisionAPI prims:")
    for p in after["calf_collision_api"]:
        prim = stage.GetPrimAtPath(p)
        print(" ", p, "| type=", prim.GetTypeName(), "| schemas=", list(prim.GetAppliedSchemas()))

    print("\n=== Checks ===")
    if len(after["collision_api"]) != 17:
        print(f"[WARN] Expected 17 CollisionAPI prims, got {len(after['collision_api'])}.")
    if len(after["wheel_collision_api"]) != 4:
        print(f"[WARN] Expected 4 wheel CollisionAPI prims, got {len(after['wheel_collision_api'])}.")
    if len(after["thigh_collision_api"]) != 4:
        print(f"[WARN] Expected 4 THIGH CollisionAPI prims, got {len(after['thigh_collision_api'])}.")
    if len(after["calf_collision_api"]) != 4:
        print(f"[WARN] Expected 4 CALF CollisionAPI prims, got {len(after['calf_collision_api'])}.")
    if len(after["fixed_joints"]) != 0:
        print(f"[WARN] Expected 0 fixed joints, got {len(after['fixed_joints'])}.")

    print("\nSaved:", dst)


if __name__ == "__main__":
    main()
