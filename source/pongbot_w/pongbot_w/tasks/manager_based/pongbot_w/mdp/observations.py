from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_step_dt(env: ManagerBasedRLEnv) -> float:
    """Get policy/control step dt."""
    if hasattr(env, "step_dt"):
        return float(env.step_dt)

    # Fallback for ManagerBasedRLEnvCfg-style envs.
    try:
        return float(env.cfg.sim.dt * env.cfg.decimation)
    except Exception:
        return 0.02


def gait_phase_sin_cos(env: ManagerBasedRLEnv, period: float = 0.72) -> torch.Tensor:
    """Return global gait phase as [sin(phase), cos(phase)].

    period:
        Gait period in seconds. 0.72 s is a reasonable initial trot-like value.
    """
    step_dt = _get_step_dt(env)
    t = env.episode_length_buf.to(dtype=torch.float32) * step_dt
    phase = torch.remainder(t / period, 1.0)
    angle = 2.0 * math.pi * phase
    return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
