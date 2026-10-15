#!/usr/bin/env python3
"""Shared 6-DoF motion-policy model and quaternion utilities.

The policy predicts the next head-centred local linear velocity (m/s) and
local angular velocity (rad/s) from a short velocity history.  It is a
behaviour-cloning motion model, not a goal-conditioned navigator.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn


FEATURE_NAMES = (
    "local_linear_velocity_x_mps",
    "local_linear_velocity_y_mps",
    "local_linear_velocity_z_mps",
    "local_angular_velocity_x_radps",
    "local_angular_velocity_y_radps",
    "local_angular_velocity_z_radps",
)


class ExplorationLSTM(nn.Module):
    """Small LSTM for one-step 6-DoF head-motion behaviour cloning."""

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 64,
        output_dim: int = 6,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected (batch, sequence, feature), got {tuple(x.shape)}")
        sequence, _ = self.lstm(x)
        return self.fc(sequence[:, -1, :])


def build_model(config: dict[str, Any]) -> ExplorationLSTM:
    return ExplorationLSTM(
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        output_dim=int(config["output_dim"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
    )


def load_checkpoint(path: str) -> dict[str, Any]:
    """Load the project checkpoint without accepting arbitrary Python objects."""

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Model checkpoint must be a dictionary")
    required = {
        "schema_version",
        "state_dict",
        "model_config",
        "normalization",
        "median_dt_seconds",
        "action_clip_low",
        "action_clip_high",
        "demo_seed_velocity",
        "data_source",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Model checkpoint is missing: {sorted(missing)}")
    return checkpoint


def normalize_quaternions(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms < 1e-12):
        raise ValueError("Quaternion contains a zero or non-finite norm")
    return values / norms


def quaternion_conjugate(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = values.copy()
    result[..., :3] *= -1.0
    return result


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product for quaternions stored as (x, y, z, w)."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    lx, ly, lz, lw = np.moveaxis(left, -1, 0)
    rx, ry, rz, rw = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        axis=-1,
    )


def rotate_vectors(quaternions: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    quaternions = normalize_quaternions(quaternions)
    vectors = np.asarray(vectors, dtype=np.float64)
    vector_quaternions = np.concatenate(
        (vectors, np.zeros(vectors.shape[:-1] + (1,), dtype=np.float64)), axis=-1
    )
    return quaternion_multiply(
        quaternion_multiply(quaternions, vector_quaternions),
        quaternion_conjugate(quaternions),
    )[..., :3]


def relative_quaternions(current: np.ndarray, following: np.ndarray) -> np.ndarray:
    current = normalize_quaternions(current)
    following = normalize_quaternions(following)
    delta = normalize_quaternions(
        quaternion_multiply(quaternion_conjugate(current), following)
    )
    delta = delta.copy()
    delta[delta[..., 3] < 0.0] *= -1.0
    return delta


def quaternion_to_rotation_vector(quaternions: np.ndarray) -> np.ndarray:
    """Map shortest-arc unit quaternions to axis-angle vectors in radians."""

    quaternions = normalize_quaternions(quaternions).copy()
    quaternions[quaternions[..., 3] < 0.0] *= -1.0
    xyz = quaternions[..., :3]
    xyz_norm = np.linalg.norm(xyz, axis=-1)
    angle = 2.0 * np.arctan2(xyz_norm, np.clip(quaternions[..., 3], -1.0, 1.0))
    scale = np.ones_like(angle)
    nonzero = xyz_norm > 1e-10
    scale[nonzero] = angle[nonzero] / xyz_norm[nonzero]
    scale[~nonzero] = 2.0
    return xyz * scale[..., None]


def rotation_vector_to_quaternion(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    angle = np.linalg.norm(vectors, axis=-1)
    half = angle * 0.5
    scale = np.empty_like(angle)
    nonzero = angle > 1e-10
    scale[nonzero] = np.sin(half[nonzero]) / angle[nonzero]
    scale[~nonzero] = 0.5
    quaternion = np.concatenate(
        (vectors * scale[..., None], np.cos(half)[..., None]), axis=-1
    )
    return normalize_quaternions(quaternion)


def integrate_local_velocity(
    position: np.ndarray,
    orientation: np.ndarray,
    local_linear_velocity: np.ndarray,
    local_angular_velocity: np.ndarray,
    dt_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate one local-frame velocity step into a world pose."""

    orientation = normalize_quaternions(np.asarray(orientation, dtype=np.float64))
    world_velocity = rotate_vectors(orientation, np.asarray(local_linear_velocity))
    next_position = np.asarray(position, dtype=np.float64) + world_velocity * dt_seconds
    delta = rotation_vector_to_quaternion(
        np.asarray(local_angular_velocity, dtype=np.float64) * dt_seconds
    )
    next_orientation = normalize_quaternions(quaternion_multiply(orientation, delta))
    return next_position, next_orientation

