#!/usr/bin/env python3
"""Shared math and I/O for the independent Normal HMD Agent v2 pipeline.

The module deliberately does not import the original one-step exploration policy.
Normal HMD v2 has a different state definition and data contract.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


STATE_FEATURE_NAMES = (
    "root_right_velocity_mps",
    "root_forward_velocity_mps",
    "root_yaw_rate_radps_rh",
    "head_local_right_m",
    "head_local_up_offset_m",
    "head_local_forward_m",
    "head_local_rotvec_x_rad_rh",
    "head_local_rotvec_y_rad_rh",
    "head_local_rotvec_z_rad_rh",
)

COORDINATE_SYSTEM = {
    "training_handedness": "right-handed",
    "up_axis": "+Y",
    "camera_forward_axis": "-Z",
    "root_right_velocity": (
        "Semantic velocity toward body-right. Unity may use it directly as local +X."
    ),
    "root_forward_velocity": (
        "Semantic velocity toward body-forward (-Z in training coordinates). "
        "Unity may use it directly as local +Z."
    ),
    "root_yaw_rate": (
        "Right-hand rotation about +Y; positive turns camera forward -Z toward -X "
        "(left). Negate for Unity yaw degrees/radians."
    ),
    "head_local_position": (
        "Semantic right/up/forward components. Unity local XYZ = "
        "(right, up, forward)."
    ),
    "head_local_rotvec": (
        "Right-handed Cartesian body-local rotation vector with z pointing backward. "
        "Map to Unity axial vector as (-rx, -ry, +rz)."
    ),
}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_quaternions(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < eps):
        raise ValueError("zero-length quaternion")
    return q / norm


def align_quaternion_signs(q: np.ndarray) -> np.ndarray:
    """Make an XYZW quaternion sequence continuous without changing rotations."""

    result = normalize_quaternions(q).copy()
    for index in range(1, len(result)):
        if float(np.dot(result[index - 1], result[index])) < 0.0:
            result[index] *= -1.0
    return result


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    result = np.asarray(q, dtype=np.float64).copy()
    result[..., :3] *= -1.0
    return result


def quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product for XYZW quaternions with NumPy broadcasting."""

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        axis=-1,
    )


def rotate_vectors(q: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Rotate vectors by normalized XYZW quaternions."""

    q = normalize_quaternions(q)
    vectors = np.asarray(vectors, dtype=np.float64)
    q_xyz = q[..., :3]
    q_w = q[..., 3:4]
    cross = np.cross(q_xyz, vectors)
    return vectors + 2.0 * (q_w * cross + np.cross(q_xyz, cross))


def relative_quaternions(parent: np.ndarray, child: np.ndarray) -> np.ndarray:
    return normalize_quaternions(
        quaternion_multiply(quaternion_conjugate(parent), child)
    )


def quaternion_to_rotation_vector(q: np.ndarray) -> np.ndarray:
    """Convert XYZW quaternions to principal rotation vectors in radians."""

    q = normalize_quaternions(q).copy()
    q = np.where(q[..., 3:4] < 0.0, -q, q)
    xyz = q[..., :3]
    xyz_norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(xyz_norm, np.clip(q[..., 3:4], -1.0, 1.0))
    scale = np.where(xyz_norm > 1e-10, angle / np.maximum(xyz_norm, 1e-10), 2.0)
    return xyz * scale


def rotation_vector_to_quaternion(rotation_vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotation_vector, dtype=np.float64)
    angle = np.linalg.norm(vector, axis=-1, keepdims=True)
    half = angle * 0.5
    scale = np.where(
        angle > 1e-10,
        np.sin(half) / np.maximum(angle, 1e-10),
        0.5 - angle * angle / 48.0,
    )
    return normalize_quaternions(
        np.concatenate((vector * scale, np.cos(half)), axis=-1)
    )


def yaw_quaternions(yaw: np.ndarray) -> np.ndarray:
    yaw = np.asarray(yaw, dtype=np.float64)
    zeros = np.zeros_like(yaw)
    return np.stack((zeros, np.sin(yaw * 0.5), zeros, np.cos(yaw * 0.5)), axis=-1)


def quaternion_yaw(q: np.ndarray) -> np.ndarray:
    """Yaw of camera forward (-Z), positive by the RH +Y convention."""

    q = normalize_quaternions(q)
    forward = rotate_vectors(q, np.broadcast_to(np.array([0.0, 0.0, -1.0]), q[..., :3].shape))
    return np.arctan2(-forward[..., 0], -forward[..., 2])


def slerp_pairs(q0: np.ndarray, q1: np.ndarray, amount: np.ndarray) -> np.ndarray:
    """Vectorized shortest-arc SLERP for paired XYZW quaternions."""

    q0 = normalize_quaternions(q0)
    q1 = normalize_quaternions(q1)
    amount = np.asarray(amount, dtype=np.float64)[..., None]
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    angle = np.arccos(dot)
    sin_angle = np.sin(angle)
    use_linear = sin_angle < 1e-6
    safe_sin = np.where(use_linear, 1.0, sin_angle)
    spherical = (
        np.sin((1.0 - amount) * angle) / safe_sin * q0
        + np.sin(amount * angle) / safe_sin * q1
    )
    linear = (1.0 - amount) * q0 + amount * q1
    return normalize_quaternions(np.where(use_linear, linear, spherical))


def resample_pose(
    timestamps_seconds: np.ndarray,
    positions: np.ndarray,
    quaternions: np.ndarray,
    fixed_dt_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearly interpolate positions and SLERP orientations onto a fixed clock."""

    timestamps_seconds = np.asarray(timestamps_seconds, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    quaternions = align_quaternion_signs(quaternions)
    if len(timestamps_seconds) < 2 or np.any(np.diff(timestamps_seconds) <= 0.0):
        raise ValueError("resample_pose requires a strictly increasing run")
    first = math.ceil(timestamps_seconds[0] / fixed_dt_seconds) * fixed_dt_seconds
    last = math.floor(timestamps_seconds[-1] / fixed_dt_seconds) * fixed_dt_seconds
    if last <= first:
        raise ValueError("pose run is shorter than one fixed time step")
    count = int(math.floor((last - first) / fixed_dt_seconds + 1e-8)) + 1
    target_time = first + np.arange(count, dtype=np.float64) * fixed_dt_seconds
    target_position = np.column_stack(
        [np.interp(target_time, timestamps_seconds, positions[:, axis]) for axis in range(3)]
    )
    upper = np.searchsorted(timestamps_seconds, target_time, side="right")
    upper = np.clip(upper, 1, len(timestamps_seconds) - 1)
    lower = upper - 1
    denominator = timestamps_seconds[upper] - timestamps_seconds[lower]
    amount = (target_time - timestamps_seconds[lower]) / denominator
    target_q = slerp_pairs(quaternions[lower], quaternions[upper], amount)
    return target_time, target_position, align_quaternion_signs(target_q)


def exponential_alpha(dt_seconds: float, time_constant_seconds: float) -> float:
    if time_constant_seconds <= 0.0:
        return 1.0
    return 1.0 - math.exp(-dt_seconds / time_constant_seconds)


def derive_body_head_state(
    positions: np.ndarray,
    head_quaternions: np.ndarray,
    dt_seconds: float,
    root_position_tau_seconds: float,
    root_height_tau_seconds: float,
    root_yaw_tau_seconds: float,
    movement_heading_threshold_mps: float,
    calibration_seconds: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create the 9-D BodyRoot-proxy + bounded HeadLocal state.

    EyeNavGS has no pelvis/feet labels.  The BodyRoot is therefore explicitly a
    causal low-pass proxy, not ground-truth anatomy.  Absolute scene height is
    removed using only the start-of-trace calibration interval.
    """

    positions = np.asarray(positions, dtype=np.float64)
    head_quaternions = align_quaternion_signs(head_quaternions)
    if len(positions) < 3:
        raise ValueError("at least three resampled poses are required")

    position_alpha = exponential_alpha(dt_seconds, root_position_tau_seconds)
    height_alpha = exponential_alpha(dt_seconds, root_height_tau_seconds)
    yaw_alpha = exponential_alpha(dt_seconds, root_yaw_tau_seconds)
    root_xz = np.empty((len(positions), 2), dtype=np.float64)
    root_xz[0] = positions[0, (0, 2)]
    for index in range(1, len(positions)):
        root_xz[index] = (
            root_xz[index - 1]
            + position_alpha * (positions[index, (0, 2)] - root_xz[index - 1])
        )

    raw_head_yaw = np.unwrap(quaternion_yaw(head_quaternions))
    root_yaw = np.empty(len(positions), dtype=np.float64)
    root_yaw[0] = raw_head_yaw[0]
    displacement = np.diff(positions[:, (0, 2)], axis=0, prepend=positions[:1, (0, 2)])
    speed = np.linalg.norm(displacement, axis=1) / dt_seconds
    movement_yaw = np.arctan2(-displacement[:, 0], -displacement[:, 1])
    movement_yaw = np.unwrap(movement_yaw)
    for index in range(1, len(positions)):
        candidate = raw_head_yaw[index]
        if speed[index] >= movement_heading_threshold_mps:
            # A conservative blend avoids assuming that every user walks exactly
            # where the HMD faces, while still letting the proxy follow locomotion.
            movement_near_head = candidate + math.atan2(
                math.sin(movement_yaw[index] - candidate),
                math.cos(movement_yaw[index] - candidate),
            )
            candidate = 0.75 * candidate + 0.25 * movement_near_head
        candidate_near_root = root_yaw[index - 1] + math.atan2(
            math.sin(candidate - root_yaw[index - 1]),
            math.cos(candidate - root_yaw[index - 1]),
        )
        root_yaw[index] = root_yaw[index - 1] + yaw_alpha * (
            candidate_near_root - root_yaw[index - 1]
        )

    root_delta = np.diff(root_xz, axis=0, prepend=root_xz[:1])
    root_world_velocity = root_delta / dt_seconds
    cosine = np.cos(root_yaw)
    sine = np.sin(root_yaw)
    # Body right=(cos, -sin) in world XZ. Body forward=(-sin, -cos)
    # follows from the RH camera forward axis -Z and the yaw definition above.
    right_velocity = cosine * root_world_velocity[:, 0] - sine * root_world_velocity[:, 1]
    forward_velocity = -sine * root_world_velocity[:, 0] - cosine * root_world_velocity[:, 1]
    yaw_rate = np.diff(root_yaw, prepend=root_yaw[:1]) / dt_seconds

    calibration_count = max(1, min(len(positions), int(round(calibration_seconds / dt_seconds))))
    height_zero = float(np.median(positions[:calibration_count, 1]))
    # Scene-aligned traces can include slow terrain/floor-height changes.  EyeNavGS
    # has no feet or floor-contact labels, so treating that slow component as a
    # HeadLocal offset would make a flat-floor Unity avatar appear to sink or
    # float by metres.  Estimate a causal low-frequency height baseline solely
    # for detrending; it is deliberately not exported as a BodyRoot vertical DOF.
    root_height = np.empty(len(positions), dtype=np.float64)
    root_height[0] = height_zero
    for index in range(1, len(positions)):
        root_height[index] = root_height[index - 1] + height_alpha * (
            positions[index, 1] - root_height[index - 1]
        )
    offset_world = np.column_stack(
        (
            positions[:, 0] - root_xz[:, 0],
            positions[:, 1] - root_height,
            positions[:, 2] - root_xz[:, 1],
        )
    )
    head_right = cosine * offset_world[:, 0] - sine * offset_world[:, 2]
    head_forward = -sine * offset_world[:, 0] - cosine * offset_world[:, 2]
    head_up = offset_world[:, 1]

    body_q = yaw_quaternions(root_yaw)
    local_q = relative_quaternions(body_q, head_quaternions)
    local_rotvec = quaternion_to_rotation_vector(local_q)

    state = np.column_stack(
        (
            right_velocity,
            forward_velocity,
            yaw_rate,
            head_right,
            head_up,
            head_forward,
            local_rotvec,
        )
    )
    return state.astype(np.float32), {
        "height_zero_aligned_units": height_zero,
        "calibration_frames": calibration_count,
        "root_position_tau_seconds": root_position_tau_seconds,
        "root_height_detrend_tau_seconds": root_height_tau_seconds,
        "root_yaw_tau_seconds": root_yaw_tau_seconds,
        "movement_heading_threshold_mps": movement_heading_threshold_mps,
    }


def orientation_geodesic_degrees(
    predicted_rotvec: np.ndarray, target_rotvec: np.ndarray
) -> np.ndarray:
    predicted_q = rotation_vector_to_quaternion(predicted_rotvec)
    target_q = rotation_vector_to_quaternion(target_rotvec)
    dot = np.abs(np.sum(predicted_q * target_q, axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))
