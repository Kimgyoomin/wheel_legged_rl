# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


# @configclass
# class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
#     num_steps_per_env = 24
#     max_iterations = 2000
#     save_interval = 50
#     experiment_name = "pongbot_w_flat_baseline"
#     empirical_normalization = False
#     policy = RslRlPpoActorCriticCfg(
#         init_noise_std=0.3,
#         actor_hidden_dims=[256, 256, 128],
#         critic_hidden_dims=[256, 256, 128],
#         activation="elu",
#     )
#     algorithm = RslRlPpoAlgorithmCfg(
#         value_loss_coef=1.0,
#         use_clipped_value_loss=True,
#         clip_param=0.2,
#         entropy_coef=0.005,
#         num_learning_epochs=5,
#         num_mini_batches=4,
#         learning_rate=3.0e-4,
#         schedule="adaptive",
#         gamma=0.99,
#         lam=0.95,
#         desired_kl=0.01,
#         max_grad_norm=1.0,
#     )
@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # Hybrid trot uses an explicit phase/contact schedule.
    # With sim.dt=1/200 and decimation=4, one policy step is 0.02 s.
    # num_steps_per_env=48 gives 0.96 s rollout, which covers at least one gait cycle
    # for GAIT_PERIOD ~= 0.72 s.
    # num_steps_per_env = 48
    num_steps_per_env = 50

    max_iterations = 2000
    save_interval = 50

    # Separate this experiment from the previous wheel-driving baseline.
    experiment_name = "pongbot_w_flat_hybrid_trot"

    # Keep TensorBoard logging explicit.
    logger = "tensorboard"

    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        # Hybrid trot may need more exploration than pure driving.
        # Start with 0.5. If bad_orientation/root_height becomes unstable,
        # reduce back to 0.3.
        init_noise_std=0.5,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )