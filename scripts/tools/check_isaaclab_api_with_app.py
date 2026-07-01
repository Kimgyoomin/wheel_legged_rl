import argparse
import inspect

from isaaclab.app import AppLauncher

# Launch Isaac Sim first.
parser = argparse.ArgumentParser(description="Check Isaac Lab API after launching SimulationApp.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import Isaac/Omni-dependent modules only after SimulationApp is launched.
import isaaclab.envs.mdp as mdp
from isaaclab.actuators import ImplicitActuatorCfg


def check_symbol(name: str):
    obj = getattr(mdp, name, None)
    print(f"{name:36s} exists={obj is not None} object={obj}")


print("\n=== MDP symbols ===")
for name in [
    "JointPositionActionCfg",
    "JointVelocityActionCfg",
    "UniformVelocityCommandCfg",
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "generated_commands",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
    "track_lin_vel_xy_exp",
    "track_ang_vel_z_exp",
    "lin_vel_z_l2",
    "ang_vel_xy_l2",
    "flat_orientation_l2",
    "joint_torques_l2",
    "joint_vel_l2",
    "joint_acc_l2",
    "action_rate_l2",
    "is_terminated",
    "time_out",
    "bad_orientation",
    "root_height_below_minimum",
    "reset_root_state_uniform",
    "reset_joints_by_offset",
]:
    check_symbol(name)


print("\n=== ImplicitActuatorCfg fields ===")
print("__annotations__:")
print(getattr(ImplicitActuatorCfg, "__annotations__", None))

print("\n__dataclass_fields__ keys:")
fields = getattr(ImplicitActuatorCfg, "__dataclass_fields__", {})
print(list(fields.keys()))


print("\n=== Action cfg signatures ===")
if hasattr(mdp, "JointPositionActionCfg"):
    print("JointPositionActionCfg:")
    print(inspect.signature(mdp.JointPositionActionCfg))

if hasattr(mdp, "JointVelocityActionCfg"):
    print("JointVelocityActionCfg:")
    print(inspect.signature(mdp.JointVelocityActionCfg))

if hasattr(mdp, "UniformVelocityCommandCfg"):
    print("UniformVelocityCommandCfg:")
    print(inspect.signature(mdp.UniformVelocityCommandCfg))


print("\n=== Try actuator keyword variants ===")
variants = [
    dict(joint_names_expr=[".*"], stiffness=40.0, damping=2.0, effort_limit_sim=120.0, velocity_limit_sim=20.0),
    dict(joint_names_expr=[".*"], stiffness=40.0, damping=2.0, effort_limit=120.0, velocity_limit=20.0),
]

for i, kwargs in enumerate(variants):
    try:
        cfg = ImplicitActuatorCfg(**kwargs)
        print(f"variant {i} OK:", kwargs)
        print(cfg)
    except Exception as e:
        print(f"variant {i} FAILED:", kwargs)
        print(repr(e))


print("\nAPI check complete.")
simulation_app.close()
