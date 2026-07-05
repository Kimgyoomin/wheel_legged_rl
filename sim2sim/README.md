# PongbotW MuJoCo Sim2Sim

This folder runs an exported Isaac Lab / RSL-RL PongbotW policy in Python MuJoCo first, then provides a C++ scaffold for a later ONNX Runtime port.

## Policy Contract

- Observation input: `62D`
- Action output: `16D`
- ONNX example: `sim2sim/exported/pongbot_w_flat_hybrid_trot_2026-07-02_best/policy.onnx`

Observation layout:

- `base_lin_vel_b`: 3
- `base_ang_vel_b`: 3
- `projected_gravity_b`: 3
- `command`: 3
- `phase sin/cos`: 2
- `joint_pos_rel`: 16
- `joint_vel`: 16
- `previous_action`: 16

Action decode:

- `action[0:12]`: leg residual position targets
- `action[12:16]`: wheel velocity targets

## Python Install

```bash
python -m pip install mujoco onnxruntime onnx
```

## Inspect ONNX

```bash
python sim2sim/python/inspect_onnx_policy.py \
  --onnx sim2sim/exported/pongbot_w_flat_hybrid_trot_2026-07-02_best/policy.onnx
```

## Inspect MuJoCo Model

```bash
python sim2sim/python/inspect_mujoco_model.py \
  --model assets/robots/pongbot_w/urdf/PONGBOT_W_isaaclab_recommended.urdf
```

## Keyboard Teleop

```bash
python sim2sim/python/mujoco_keyboard_teleop.py \
  --model assets/robots/pongbot_w/urdf/PONGBOT_W_isaaclab_recommended.urdf
```

Keys:

- `W/S`: increase/decrease `vx`
- `A/D`: increase/decrease `vy`
- `Q/E`: increase/decrease `wz`
- `X`: zero command
- `R`: reset
- `Space`: pause
- `1/2/3`: command scale `0.25/0.5/1.0`

Teleop focuses on `vx/wz`. Lateral command is policy/gait dependent and is not fully implemented in the wheel-only teleop controller.

## ONNX Policy Play

```bash
python sim2sim/python/mujoco_onnx_policy_play.py \
  --model assets/robots/pongbot_w/urdf/PONGBOT_W_isaaclab_recommended.urdf \
  --onnx sim2sim/exported/pongbot_w_flat_hybrid_trot_2026-07-02_best/policy.onnx \
  --log-csv /tmp/pongbot_mujoco_policy.csv
```

## CSV Logging

`mujoco_onnx_policy_play.py` logs:

- time
- command
- base velocity
- projected gravity
- joint position / velocity
- policy action
- leg targets
- wheel velocity targets
- applied torques

## Known Caveats

- MuJoCo URDF collision can differ from Isaac Lab USD collision.
- Wheel sign convention must be checked.
- Observation frame mismatch is the most likely failure source.
- Assets and exported model binaries are not tracked by Git.
