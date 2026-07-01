import argparse
from pxr import Usd, UsdPhysics

try:
    from pxr import PhysxSchema
except Exception:
    PhysxSchema = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", required=True, help="Path to USD file")
    args = parser.parse_args()

    usd_path = args.usd
    print("USD:", usd_path)

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Failed to open USD: {usd_path}")

    default_prim = stage.GetDefaultPrim()
    print("Default prim:", default_prim.GetPath() if default_prim else None)

    art_roots = []
    rigid_bodies = []
    joints = []
    fixed_joints = []
    wheel_joints = []
    collision_api = []
    wheel_collision_api = []
    physx_collision = []
    collision_like = []
    wheel_collision_like = []

    for prim in stage.TraverseAll():
        path = str(prim.GetPath())
        type_name = prim.GetTypeName()
        schemas = list(prim.GetAppliedSchemas())
        attrs = [a.GetName() for a in prim.GetAttributes()]
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

        if PhysxSchema is not None:
            for api_name in ["PhysxCollisionAPI", "PhysxMeshCollisionAPI"]:
                if hasattr(PhysxSchema, api_name):
                    try:
                        if prim.HasAPI(getattr(PhysxSchema, api_name)):
                            physx_collision.append(path)
                    except Exception:
                        pass

        tokens = []
        for s in schemas:
            if "collision" in s.lower() or "physx" in s.lower():
                tokens.append(f"schema:{s}")
        for a in attrs:
            if "collision" in a.lower() or "physx" in a.lower():
                tokens.append(f"attr:{a}")
        if "collision" in path.lower() or "collider" in path.lower():
            tokens.append("path")

        if tokens:
            collision_like.append((path, type_name, schemas, tokens))
            if "WHEEL" in upath:
                wheel_collision_like.append((path, type_name, schemas, tokens))

    print("\n--- Summary ---")
    print("articulation roots:", len(art_roots))
    print("rigid bodies:", len(rigid_bodies))
    print("real joints:", len(joints))
    print("fixed joints:", len(fixed_joints))
    print("wheel joints:", len(wheel_joints))
    print("UsdPhysics.CollisionAPI prims:", len(collision_api))
    print("Wheel CollisionAPI prims:", len(wheel_collision_api))
    print("Physx collision API prims:", len(set(physx_collision)))
    print("collision-like prims:", len(collision_like))
    print("wheel collision-like prims:", len(wheel_collision_like))

    print("\n--- Articulation Roots ---")
    for p in art_roots:
        print(" ", p)

    print("\n--- Wheel Joints ---")
    for p in wheel_joints:
        prim = stage.GetPrimAtPath(p)
        axis = prim.GetAttribute("physics:axis").Get() if prim.HasAttribute("physics:axis") else None
        lower = prim.GetAttribute("physics:lowerLimit").Get() if prim.HasAttribute("physics:lowerLimit") else None
        upper = prim.GetAttribute("physics:upperLimit").Get() if prim.HasAttribute("physics:upperLimit") else None
        print(f" {p} | type={prim.GetTypeName()} | axis={axis} | lower={lower} | upper={upper}")

    print("\n--- Fixed Joints ---")
    for p in fixed_joints:
        print(" ", p)

    print("\n--- Wheel CollisionAPI prims ---")
    for p in wheel_collision_api:
        prim = stage.GetPrimAtPath(p)
        print(f" {p} | type={prim.GetTypeName()} | schemas={list(prim.GetAppliedSchemas())}")

    print("\n--- All CollisionAPI prims ---")
    for p in collision_api:
        prim = stage.GetPrimAtPath(p)
        print(f" {p} | type={prim.GetTypeName()} | schemas={list(prim.GetAppliedSchemas())}")

    print("\n--- Warnings ---")
    if len(art_roots) == 0:
        print("[WARN] No articulation root.")
    if len(wheel_joints) < 4:
        print(f"[WARN] Expected 4 wheel joints, found {len(wheel_joints)}.")
    if len(collision_api) == 0:
        print("[WARN] No real CollisionAPI prims found.")
    if len(wheel_collision_api) < 4:
        print(f"[WARN] Expected at least 4 wheel CollisionAPI prims, found {len(wheel_collision_api)}.")
    if fixed_joints:
        print("[WARN] Fixed joints exist. Make sure wheel joints are not fixed.")

    print("\nInspection done.")


if __name__ == "__main__":
    main()
