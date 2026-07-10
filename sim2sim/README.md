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
conda activate env_isaaclab
python -m pip install mujoco onnxruntime onnx numpy
```

## Dependency Check

```bash
cd ~/isaac_projects/pongbot_w
conda activate env_isaaclab
python sim2sim/python/check_dependencies.py
```

## Build Free-Base MJCF

Do not use the Isaac Lab recommended URDF directly for locomotion sim2sim. MuJoCo loads it as fixed-base. Generate a free-base MJCF first:

```bash
python sim2sim/python/build_freebase_mjcf.py \
  --urdf assets/robots/pongbot_w/urdf/PONGBOT_W_isaaclab_recommended.urdf \
  --mesh-dir assets/robots/pongbot_w/meshes \
  --output sim2sim/.cache/generated_models/PONGBOT_W_freebase.xml
```

## Inspect ONNX

```bash
python sim2sim/python/inspect_onnx_policy.py \
  --onnx sim2sim/exported/pongbot_w_flat_hybrid_trot_2026-07-02_best/policy.onnx
```

## Inspect MuJoCo Model

```bash
python sim2sim/python/inspect_mujoco_model.py \
  --model sim2sim/.cache/generated_models/PONGBOT_W_freebase.xml \
  --require-freejoint
```

## Keyboard Teleop

```bash
python sim2sim/python/mujoco_keyboard_teleop.py \
  --model sim2sim/.cache/generated_models/PONGBOT_W_freebase.xml
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
  --model sim2sim/.cache/generated_models/PONGBOT_W_freebase.xml \
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
- MuJoCo resolves URDF mesh paths relative to the model XML path.
- PongbotW URDF files may reference bare STL filenames while actual meshes live in `assets/robots/pongbot_w/meshes/`.
- The sim2sim scripts preprocess the model XML and rewrite mesh paths to absolute paths under `sim2sim/.cache/resolved_models/`.
- PongbotW locomotion sim2sim requires a freejoint. A fixed-base model must fail inspection/play when `--require-freejoint` is used.
