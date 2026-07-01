# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
# import isaaclab.envs.mdp as mdp
from . import mdp
# add contact sensor
from isaaclab.sensors import ContactSensorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass


##
# Constants
##

PONGBOT_W_USD_PATH = (
    "/home/kim/isaac_projects/pongbot_w/assets/robots/pongbot_w/usd/"
    "PONGBOT_W_collision_v4_manual_contact/PONGBOT_W.usd"
)

LEG_JOINT_NAMES = [".*_HR_JOINT", ".*_HP_JOINT", ".*_KN_JOINT"]
WHEEL_JOINT_NAMES = [".*_WHEEL_JOINT"]

LEG_DEFAULT_JOINT_POS = {
    ".*_HR_JOINT": 0.0,
    ".*_HP_JOINT": 0.716,
    ".*_KN_JOINT": -1.396,
}

WHEEL_DEFAULT_JOINT_POS = {
    ".*_WHEEL_JOINT": 0.0,
}

WHEEL_BODY_NAMES = ["FL_WHEEL", "FR_WHEEL", "RL_WHEEL", "RR_WHEEL"]

GAIT_PERIOD = 0.72
GAIT_MARGIN = 0.35
WHEEL_CONTACT_THRESHOLD = 1.0
SWING_WHEEL_HEIGHT = 0.16

##
# Robot config
##

PONGBOT_W_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=PONGBOT_W_USD_PATH,
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.63),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            ".*_HR_JOINT": 0.0,
            ".*_HP_JOINT": 0.716,
            ".*_KN_JOINT": -1.396,
            ".*_WHEEL_JOINT": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Hip-roll output torque limit: 80 Nm
        "hip_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*_HR_JOINT"],
            stiffness=200.0,
            damping=5.0,
            effort_limit_sim=80.0,
        ),

        # Hip-pitch output torque limit: 160 Nm
        "hip_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_HP_JOINT"],
            stiffness=200.0,
            damping=5.0,
            effort_limit_sim=160.0,
        ),

        # Knee output torque limit: 280 Nm
        "knee": ImplicitActuatorCfg(
            joint_names_expr=[".*_KN_JOINT"],
            stiffness=200.0,
            damping=5.0,
            effort_limit_sim=280.0,
        ),

        # Wheel velocity control.
        # The former Isaac Gym setup used approximately:
        # wheel velocity scale = 15 rad/s
        # wheel torque limit = 9 Nm
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_WHEEL_JOINT"],
            stiffness=0.0,
            damping=1.0,
            effort_limit_sim=9.0,
            velocity_limit_sim=19.0,
        ),

        # Conservative first values. We will tune after the first reset/action-probe.
    #     "legs": ImplicitActuatorCfg(
    #         joint_names_expr=LEG_JOINT_NAMES,
    #         stiffness=200.0,
    #         damping=5.0,
    #         effort_limit_sim=300.0,
    #         velocity_limit_sim=30.0,
    #         armature=0.0,
    #         friction=0.0,
    #     ),
    #     # Wheel joints are velocity-driven. Stiffness must be zero.
    #     "wheels": ImplicitActuatorCfg(
    #         joint_names_expr=WHEEL_JOINT_NAMES,
    #         stiffness=0.0,
    #         damping=1.0,
    #         effort_limit_sim=120.0,
    #         velocity_limit_sim=25.0,
    #         armature=0.0,
    #         friction=0.0,
    #     ),
    },
)

##
# MDP settings
##

@configclass
class CommandsCfg:
    """Velocity command for hybrid wheel+trot locomotion.

    The policy should learn a common command space:
    - forward / backward
    - left / right lateral motion
    - yaw turning

    This is not a wheel-only driving task. The gait phase reward remains active
    so lateral and yaw commands should be solved with hybrid wheel+trot motion.
    """

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.05,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.8),
            lin_vel_y=(-0.25, 0.25),
            ang_vel_z=(-0.8, 0.8),
            heading=(-math.pi, math.pi),
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for PongbotW."""

    # 12 leg joint position targets around default pose.
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        scale=0.5,
        offset=LEG_DEFAULT_JOINT_POS,
        use_default_offset=False,
    )

    # 4 wheel joint velocity targets.
    wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=WHEEL_JOINT_NAMES,
        scale=8.0,
    )


@configclass
class ObservationsCfg:
    """Observation specifications."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observation group.

        Expected dimension:
        base_lin_vel      3
        base_ang_vel      3
        projected_gravity 3
        velocity_command  3
        phase             2  
        joint_pos         16
        joint_vel         16
        previous_action   16
        total             62
        """

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        phase = ObsTerm(
                    func=mdp.gait_phase_sin_cos,
                    params={"period": GAIT_PERIOD},
                )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class PongbotWSceneCfg(InteractiveSceneCfg):
    """Flat-ground PongbotW scene."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    robot: ArticulationCfg = PONGBOT_W_CFG

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        update_period=0.0,
        track_air_time=False,
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

@configclass
class EventCfg:
    """Reset events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "yaw": (-0.1, 0.1),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.02, 0.02),
            "velocity_range": (-0.05, 0.05),
        },
    )


@configclass
class RewardsCfg:
    """Minimal flat locomotion reward."""
    # Survival
    alive = RewTerm(
        func=mdp.is_alive,
        weight=0.2,
    )


    # Task rewards
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # Stability penalties
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.15)
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    base_height = RewTerm(
        func=mdp.base_height_tolerance_l2,
        weight=-3.0,
        params={
            "target_height": 0.58,
            "tolerance": 0.05,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # joint deviation
    joint_deviation = RewTerm(
        func=mdp.joint_deviation_from_default_l1,
        weight=-0.03,
        params={
            "default_joint_pos": {
                "HR_JOINT": 0.0,
                "HP_JOINT": 0.716,
                "KN_JOINT": -1.396,
            },
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_HR_JOINT", ".*_HP_JOINT", ".*_KN_JOINT"],
            ),
        },
    )

    # Contact
    bad_body_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["BASE", ".*_HIP", ".*_THIGH", ".*_CALF"],
            ),
            "threshold": 1.0,
        },
    )

    # Hybrid wheel+trot style
    trot_contact_match = RewTerm(
        func=mdp.trot_contact_match_reward,
        weight=0.5,
        params={
            "period": GAIT_PERIOD,
            "margin": GAIT_MARGIN,
            "threshold": WHEEL_CONTACT_THRESHOLD,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=WHEEL_BODY_NAMES,
            ),
        },
    )

    trot_swing_contact = RewTerm(
        func=mdp.trot_swing_contact_penalty,
        weight=-2.0,
        params={
            "period": GAIT_PERIOD,
            "margin": GAIT_MARGIN,
            "threshold": WHEEL_CONTACT_THRESHOLD,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=WHEEL_BODY_NAMES,
            ),
        },
    )

    trot_stance_miss = RewTerm(
        func=mdp.trot_stance_miss_penalty,
        weight=-1.0,
        params={
            "period": GAIT_PERIOD,
            "margin": GAIT_MARGIN,
            "threshold": WHEEL_CONTACT_THRESHOLD,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=WHEEL_BODY_NAMES,
            ),
        },
    )

    trot_swing_clearance = RewTerm(
        func=mdp.trot_swing_clearance_l2,
        weight=-2.0,
        params={
            "period": GAIT_PERIOD,
            "margin": GAIT_MARGIN,
            "target_height": SWING_WHEEL_HEIGHT,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=WHEEL_BODY_NAMES,
            ),
        },
    )
    
    # Regularization
    leg_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )

    wheel_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
    )

    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-4)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-10.0)

    


@configclass
class TerminationsCfg:
    """Minimal terminations."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    root_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"asset_cfg": SceneEntityCfg("robot"), "minimum_height": 0.25},
    )

    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 1.1},
    )


##
# Environment configuration
##

@configclass
class PongbotWEnvCfg(ManagerBasedRLEnvCfg):
    scene: PongbotWSceneCfg = PongbotWSceneCfg(num_envs=4096, env_spacing=4.0)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 10.0

        self.viewer.eye = (4.0, -4.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)

        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation