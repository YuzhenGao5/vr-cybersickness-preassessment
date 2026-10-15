#!/usr/bin/env python3
"""Shared local feature extraction helpers for VR.net sessions.

Image flow/depth caches are accepted explicitly so a prior, fully audited
image pass can be reused without silently substituting a proxy for depth.
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd


LOGGER = logging.getLogger("vrnet_feature_extractor")


def _seconds(values: Sequence[float] | pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    # VR.net's raw timestamp is Unix milliseconds; allow elapsed seconds too.
    if np.nanmedian(np.abs(values)) > 1e11:
        return values / 1000.0
    return values


def _numeric_frame(values: pd.DataFrame | np.ndarray | Sequence[Sequence[float]], columns: list[str]) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        frame = values.copy()
        if not set(columns).issubset(frame.columns):
            if frame.shape[1] < len(columns):
                raise ValueError(f"Expected at least {len(columns)} pose columns, got {frame.shape[1]}")
            frame = frame.iloc[:, :len(columns)].copy()
            frame.columns = columns
        return frame[columns].apply(pd.to_numeric, errors="coerce")
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != len(columns):
        raise ValueError(f"Expected N x {len(columns)} values, got shape {array.shape}")
    return pd.DataFrame(array, columns=columns)


def _cache_for_window_features(cache_path: str | Path, prefixes: tuple[str, ...], required: tuple[str, ...] = ()) -> pd.DataFrame:
    path = Path(cache_path)
    if not path.is_file():
        raise FileNotFoundError(f"Required local feature cache not found: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "window_index" not in frame:
        raise ValueError(f"Feature cache has no window_index: {path}")
    selected = [column for column in frame.columns if column == "window_index" or column.startswith(prefixes) or column in required]
    if not set(required).issubset(selected):
        raise ValueError(f"Feature cache {path} lacks required columns {sorted(set(required) - set(selected))}")
    result = frame[selected].copy()
    LOGGER.info("Reused local precomputed features from %s: rows=%d columns=%s", path, len(result), selected)
    return result


def extract_optical_flow(
    frames_dir: str | Path,
    window_size: int = 30,
    *,
    cached_features: str | Path | None = None,
) -> pd.DataFrame:
    """Return flow features by 30-second cached window or frame-pair block.

    If a cache is supplied, its existing window_index is retained. Otherwise
    Farneback flow is computed over numeric-named JPG pairs and summarized in
    consecutive blocks of ``window_size`` pairs. No timestamp unit is inferred
    in that standalone block mode.
    """
    directory = Path(frames_dir)
    if cached_features is not None:
        return _cache_for_window_features(
            cached_features,
            ("flow_",),
            ("flow_mean_magnitude_px_analysis",),
        )
    if not directory.is_dir():
        raise FileNotFoundError(f"Frame directory not found: {directory}")
    if window_size < 1:
        raise ValueError("window_size must be positive")
    files = []
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}:
            match = re.search(r"(\d+)$", path.stem)
            if match:
                files.append((int(match.group(1)), path))
    files.sort(key=lambda item: item[0])
    if len(files) < 2:
        raise ValueError(f"At least two numeric-named JPGs are required; found {len(files)} in {directory}")
    chunks: list[list[dict[str, float]]] = []
    current: list[dict[str, float]] = []
    previous_gray = None
    for _, image_path in files:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            LOGGER.warning("Skipping undecodable JPG: %s", image_path)
            previous_gray = None
            continue
        h, w = image.shape[:2]
        scale = min(1.0, 320.0 / max(h, w))
        if scale < 1:
            image = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if previous_gray is not None and previous_gray.shape == gray.shape:
            flow = cv2.calcOpticalFlowFarneback(previous_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude = np.hypot(flow[:, :, 0], flow[:, :, 1])
            angles = np.mod(np.arctan2(flow[:, :, 1], flow[:, :, 0]), 2 * np.pi)
            hist, _ = np.histogram(angles, bins=16, range=(0, 2 * np.pi))
            prob = hist[hist > 0].astype(float)
            entropy = float(-(prob / prob.sum() * np.log2(prob / prob.sum())).sum()) if prob.size else 0.0
            current.append({
                "flow_mean_magnitude_px_analysis": float(magnitude.mean()),
                "flow_max_magnitude_px_analysis": float(magnitude.max()),
                "flow_direction_entropy_bits": entropy,
            })
            if len(current) >= window_size:
                chunks.append(current)
                current = []
        previous_gray = gray
    if current:
        chunks.append(current)
    if not chunks:
        raise ValueError(f"No adjacent JPG pairs could be processed in {directory}")
    rows = []
    for index, chunk in enumerate(chunks):
        row: dict[str, float | int] = {"window_index": index, "source_pair_count": len(chunk)}
        for key in chunk[0]:
            values = np.asarray([item[key] for item in chunk], dtype=float)
            row[key] = float(np.nanmean(values))
        rows.append(row)
    return pd.DataFrame(rows)


def extract_camera_speed(poses: pd.DataFrame | np.ndarray | Sequence[Sequence[float]], timestamps: Sequence[float]) -> pd.DataFrame:
    """Calculate translational speed from 3-D pose samples.

    Output units remain source-position-units/second. For this project the
    available source is HMD ``head_pos`` (user motion), not the virtual camera.
    """
    xyz = _numeric_frame(poses, ["x", "y", "z"]).to_numpy(dtype=float)
    times = _seconds(timestamps)
    if len(xyz) != len(times):
        raise ValueError("poses and timestamps must have equal lengths")
    speed = np.full(len(xyz), np.nan, dtype=float)
    if len(xyz) > 1:
        dt = np.diff(times)
        distance = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        valid = (dt > 0) & np.isfinite(distance)
        speed[1:][valid] = distance[valid] / dt[valid]
    return pd.DataFrame({"timestamp": np.asarray(timestamps), "camera_speed_source_units_s": speed})


def extract_rotation_axes(quaternions: pd.DataFrame | np.ndarray | Sequence[Sequence[float]], threshold: float = 0.5) -> pd.DataFrame:
    """Estimate active rotation components between XYZW quaternion samples.

    ``threshold`` is radians for each relative rotation-vector component.
    """
    q = _numeric_frame(quaternions, ["qx", "qy", "qz", "qw"]).to_numpy(dtype=float)
    axis_delta = np.full((len(q), 3), np.nan, dtype=float)
    for i in range(1, len(q)):
        prev, curr = q[i - 1], q[i]
        if not np.isfinite(prev).all() or not np.isfinite(curr).all():
            continue
        prev_norm, curr_norm = np.linalg.norm(prev), np.linalg.norm(curr)
        if prev_norm < 1e-12 or curr_norm < 1e-12:
            continue
        prev, curr = prev / prev_norm, curr / curr_norm
        # q_delta = inverse(previous) * current, quaternion order XYZW.
        pxyz, pw = -prev[:3], prev[3]
        cxyz, cw = curr[:3], curr[3]
        dxyz = pw * cxyz + cw * pxyz + np.cross(pxyz, cxyz)
        dw = pw * cw - np.dot(pxyz, cxyz)
        d = np.r_[dxyz, dw]
        if d[3] < 0:
            d = -d
        d /= max(np.linalg.norm(d), 1e-12)
        angle = 2.0 * math.atan2(float(np.linalg.norm(d[:3])), float(np.clip(d[3], -1, 1)))
        axis = d[:3] / max(float(np.linalg.norm(d[:3])), 1e-12)
        axis_delta[i] = axis * angle
    count = (np.abs(axis_delta) >= float(threshold)).sum(axis=1)
    return pd.DataFrame({
        "rotation_x_delta_rad": axis_delta[:, 0],
        "rotation_y_delta_rad": axis_delta[:, 1],
        "rotation_z_delta_rad": axis_delta[:, 2],
        "rotation_axes_count": count.astype(float),
    })


def extract_path_curvature(positions: pd.DataFrame | np.ndarray | Sequence[Sequence[float]]) -> pd.DataFrame:
    """Estimate discrete 3-D path curvature in inverse source-position units."""
    xyz = _numeric_frame(positions, ["x", "y", "z"]).to_numpy(dtype=float)
    curvature = np.full(len(xyz), np.nan, dtype=float)
    for i in range(1, len(xyz) - 1):
        a, b = xyz[i] - xyz[i - 1], xyz[i + 1] - xyz[i]
        la, lb = np.linalg.norm(a), np.linalg.norm(b)
        if la <= 1e-12 or lb <= 1e-12:
            continue
        angle = math.acos(float(np.clip(np.dot(a, b) / (la * lb), -1.0, 1.0)))
        curvature[i] = angle / ((la + lb) / 2.0)
    return pd.DataFrame({"path_curvature_source_units_inv": curvature})


def extract_depth_change_rate(
    frames_dir: str | Path,
    window_size: int = 30,
    *,
    cached_features: str | Path | None = None,
) -> pd.DataFrame:
    """Reuse an explicit depth-feature cache; do not infer depth from RGB.

    The Mars Miners source has JPGs but no depth maps. Its existing depth rate
    was computed by a local monocular depth model and is reused by window id.
    """
    if cached_features is None:
        raise FileNotFoundError(
            f"No depth-map input or precomputed depth cache supplied for {frames_dir}; RGB images alone are not treated as depth"
        )
    return _cache_for_window_features(cached_features, ("__none__",), ("depth_change_rate",))[["window_index", "depth_change_rate"]]


def aggregate_to_windows(features: pd.DataFrame, timestamps: Sequence[float], window_sec: float = 30) -> pd.DataFrame:
    """Aggregate per-sample numeric features into half-open elapsed-time windows."""
    if window_sec <= 0:
        raise ValueError("window_sec must be positive")
    values = features.reset_index(drop=True).copy()
    times = _seconds(timestamps)
    if len(values) != len(times):
        raise ValueError("features and timestamps must have equal lengths")
    if not len(times):
        return pd.DataFrame(columns=["window_index", "window_start_sec", "window_end_sec"])
    elapsed = times - np.nanmin(times)
    values["_elapsed_sec"] = elapsed
    values["window_index"] = np.floor(elapsed / window_sec + 1e-10).astype(int)
    feature_cols = [c for c in values.columns if c not in {"_elapsed_sec", "window_index", "timestamp"}]
    rows = []
    for window_index, group in values.groupby("window_index", sort=True):
        start = float(window_index * window_sec)
        row: dict[str, float | int] = {
            "window_index": int(window_index), "window_start_sec": start,
            "window_end_sec": min(start + float(window_sec), float(np.nanmax(elapsed))),
            "window_sample_count": int(len(group)),
        }
        for column in feature_cols:
            numeric = pd.to_numeric(group[column], errors="coerce")
            row[column] = float(numeric.mean()) if numeric.notna().any() else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)
