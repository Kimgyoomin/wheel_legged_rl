"""Math utilities for PongbotW MuJoCo sim2sim."""

from __future__ import annotations

import numpy as np


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert MuJoCo free-joint qpos quaternion [qw, qx, qy, qz] to a rotation matrix."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), got {q.shape}.")
    norm = np.linalg.norm(q)
    if norm <= 0.0:
        raise ValueError("Quaternion norm must be positive.")
    qw, qx, qy, qz = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a world-frame vector into body frame using MuJoCo qpos quaternion [qw, qx, qy, qz]."""
    return quat_wxyz_to_rotmat(q).T @ np.asarray(v, dtype=np.float64)


def projected_gravity(q: np.ndarray) -> np.ndarray:
    """Project world gravity [0, 0, -1] into body frame using MuJoCo qpos quaternion [qw, qx, qy, qz]."""
    return quat_rotate_inverse(q, np.array([0.0, 0.0, -1.0], dtype=np.float64))
