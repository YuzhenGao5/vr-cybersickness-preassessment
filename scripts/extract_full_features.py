#!/usr/bin/env python
"""Stream image pairs and emit windowed, auditable VR scene features.

This script intentionally separates scene-image measurements from participant
head-pose telemetry and sparse MS_rating labels. It never reads MS_rating.
Metric units and non-equivalences are documented in results/required_metrics.md.
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

LOG = logging.getLogger("extract_full_features")

STATIC_METRICS = {
    "edge_density": "fraction",
    "color_entropy_bits": "bits",
    "laplacian_variance": "intensity^2",
    "brightness_mean_0_255": "gray_level_0_255",
    "brightness_std_0_255": "gray_level_0_255",
    "peripheral_edge_density": "fraction",
}
DEPTH_MODEL_INPUT_SIZE = 336
PAIR_METRICS = {
    "flow_mean_magnitude_px_analysis": "analysis_px_per_sampled_pair",
    "flow_max_magnitude_px_analysis": "analysis_px_per_sampled_pair",
    "flow_direction_change_rate_deg_s": "degrees_per_second",
    "flow_direction_entropy_bits": "bits",
    "flow_mean_abs_divergence": "analysis_px_per_pair_per_analysis_px",
    "flow_mean_abs_curl": "analysis_px_per_pair_per_analysis_px",
    "homography_translation_speed_proxy_px_s": "analysis_px_per_second",
    "homography_rotation_proxy_deg_s": "degrees_per_second",
    "homography_acceleration_proxy_px_s2": "analysis_px_per_second_squared",
    "homography_path_curvature_proxy_px_inv": "inverse_analysis_px",
    "depth_change_rate": "normalized_relative_depth_per_second",
    "depth_occlusion_boundary_proxy_density": "fraction",
}
RULE_METRIC_COLUMNS = [
    "virtual_translation_speed_m_s",
    "oscillation_frequency_hz",
    "camera_speed_m_s",
    "camera_acceleration_m_s2",
    "rotation_axes",
    "camera_fov_deg",
    "has_reference_frame",
    "controllability",
    "expected_duration_min",
    "particle_emission_rate",
    "dynamic_light_count",
    "peripheral_geometry_complexity",
]


def parse_float_list(value: Any) -> list[float]:
    if value is None:
        return []
    return [float(x) for x in re.findall(r"[-+]?(?:\d*\.)?\d+(?:[eE][-+]?\d+)?", str(value))]


def parse_frame_value(value: Any) -> int | None:
    if value is None:
        return None
    match = re.search(r"(-?\d+)(?:\.\d+)?", str(value))
    return int(match.group(1)) if match else None


def fov_from_projection(value: Any) -> float | None:
    values = parse_float_list(value)
    if len(values) < 6:
        return None
    # VR.net stores a 4x4 projection matrix. For the common Unity/OpenGL
    # perspective convention, element 5 is the vertical focal scale.
    focal_y = abs(values[5])
    if not math.isfinite(focal_y) or focal_y <= 1e-8:
        return None
    return math.degrees(2.0 * math.atan(1.0 / focal_y))


def read_session_metadata(csv_path: Path) -> tuple[dict[int, int], set[int], dict[int, float]]:
    """Read only frame/time/video/projection columns; never parse sickness labels."""
    csv.field_size_limit(20_000_000)
    timestamps: dict[int, int] = {}
    expected_frames: set[int] = set()
    fov_by_frame: dict[int, float] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {"frame", "timestamp", "video"}
        missing = required - fields
        if missing:
            raise ValueError(f"{csv_path} lacks required columns: {sorted(missing)}")
        has_projection = "p_matrix" in fields
        for row in reader:
            frame = parse_frame_value(row.get("frame"))
            stamp = parse_frame_value(row.get("timestamp"))
            if frame is None:
                continue
            if stamp is not None:
                timestamps[frame] = stamp
            video_value = (row.get("video") or "").strip()
            if video_value and video_value.lower() not in {"nan", "none"}:
                video_frame = parse_frame_value(Path(video_value.replace("\\", "/")).name)
                if video_frame is not None:
                    expected_frames.add(video_frame)
            if has_projection and frame not in fov_by_frame:
                fov = fov_from_projection(row.get("p_matrix"))
                if fov is not None:
                    fov_by_frame[frame] = fov
    return timestamps, expected_frames, fov_by_frame


def static_features(image_bgr: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    channel_entropies = []
    for channel in cv2.split(image_bgr):
        hist = np.histogram(channel, bins=64, range=(0, 256))[0].astype(np.float64)
        probs = hist[hist > 0] / max(1.0, hist.sum())
        channel_entropies.append(float(-(probs * np.log2(probs)).sum()) if probs.size else float("nan"))
    color_entropy = float(np.nanmean(channel_entropies))
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt(((xx - (w - 1) / 2) / max(w, 1)) ** 2 + ((yy - (h - 1) / 2) / max(h, 1)) ** 2)
    center_radius = 0.30
    peripheral = radius >= center_radius
    periph_count = int(peripheral.sum())
    values = {
        "edge_density": float(np.count_nonzero(edges) / edges.size),
        "color_entropy_bits": color_entropy,
        "laplacian_variance": float(lap.var()),
        "brightness_mean_0_255": float(gray.mean()),
        "brightness_std_0_255": float(gray.std()),
        "peripheral_edge_density": float(np.count_nonzero(edges[peripheral]) / periph_count) if periph_count else float("nan"),
    }
    return gray, values


def flow_features(flow: np.ndarray) -> dict[str, float]:
    u, v = flow[:, :, 0], flow[:, :, 1]
    magnitude = np.hypot(u, v)
    moving = magnitude > 1e-6
    if moving.any():
        angle = np.mod(np.arctan2(v[moving], u[moving]), 2 * np.pi)
        hist = np.histogram(angle, bins=16, range=(0, 2 * np.pi))[0].astype(np.float64)
        p = hist[hist > 0] / hist.sum()
        entropy = float(-(p * np.log2(p)).sum())
    else:
        entropy = 0.0
    du_dy, du_dx = np.gradient(u)
    dv_dy, dv_dx = np.gradient(v)
    return {
        "flow_mean_magnitude_px_analysis": float(magnitude.mean()),
        "flow_max_magnitude_px_analysis": float(magnitude.max()),
        "flow_direction_entropy_bits": entropy,
        "flow_mean_abs_divergence": float(np.mean(np.abs(du_dx + dv_dy))),
        "flow_mean_abs_curl": float(np.mean(np.abs(dv_dx - du_dy))),
    }


def direction_change(previous: np.ndarray | None, current: np.ndarray, delta_t: float) -> float:
    if previous is None or delta_t <= 0 or previous.shape != current.shape:
        return float("nan")
    p_mag = np.hypot(previous[:, :, 0], previous[:, :, 1])
    c_mag = np.hypot(current[:, :, 0], current[:, :, 1])
    mask = (p_mag > 1e-4) & (c_mag > 1e-4)
    if not mask.any():
        return float("nan")
    p_ang = np.arctan2(previous[:, :, 1][mask], previous[:, :, 0][mask])
    c_ang = np.arctan2(current[:, :, 1][mask], current[:, :, 0][mask])
    delta = np.arctan2(np.sin(c_ang - p_ang), np.cos(c_ang - p_ang))
    return float(np.degrees(np.abs(delta).mean()) / delta_t)


def homography_proxy(previous: np.ndarray, current: np.ndarray, delta_t: float, prior_speed: float | None) -> dict[str, float]:
    result = {
        "homography_translation_speed_proxy_px_s": float("nan"),
        "homography_rotation_proxy_deg_s": float("nan"),
        "homography_acceleration_proxy_px_s2": float("nan"),
        "homography_path_curvature_proxy_px_inv": float("nan"),
    }
    if delta_t <= 0:
        return result
    points0 = cv2.goodFeaturesToTrack(previous, maxCorners=500, qualityLevel=0.01, minDistance=7)
    if points0 is None or len(points0) < 8:
        return result
    points1, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points0, None)
    if points1 is None or status is None:
        return result
    good = status.ravel() == 1
    if int(good.sum()) < 8:
        return result
    matrix, inliers = cv2.findHomography(
        points0[good].reshape(-1, 2), points1[good].reshape(-1, 2),
        method=cv2.RANSAC, ransacReprojThreshold=3.0,
    )
    if matrix is None or abs(matrix[2, 2]) < 1e-9:
        return result
    matrix = matrix / matrix[2, 2]
    translation_px = float(np.hypot(matrix[0, 2], matrix[1, 2]))
    speed = translation_px / delta_t
    rotation_deg_s = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])) / delta_t
    result["homography_translation_speed_proxy_px_s"] = speed
    result["homography_rotation_proxy_deg_s"] = float(rotation_deg_s)
    if prior_speed is not None:
        result["homography_acceleration_proxy_px_s2"] = float((speed - prior_speed) / delta_t)
    if speed > 1e-9:
        result["homography_path_curvature_proxy_px_inv"] = float(abs(math.radians(rotation_deg_s)) / speed)
    return result


class LocalDepthEstimator:
    """Optional local-only Hugging Face depth model; never downloads weights."""

    def __init__(self, model_dir: Path | None):
        self.model = None
        self.processor = None
        self.torch = None
        self.reason = "not requested: no --depth-model-dir supplied"
        if model_dir is None:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            self.torch = torch
            self.processor = AutoImageProcessor.from_pretrained(str(model_dir), local_files_only=True)
            # Input images are already capped at 320 px by the scene pipeline.
            # Avoid upscaling every frame to the checkpoint's 518 px default.
            self.processor.size = {"height": DEPTH_MODEL_INPUT_SIZE, "width": DEPTH_MODEL_INPUT_SIZE}
            self.model = AutoModelForDepthEstimation.from_pretrained(str(model_dir), local_files_only=True)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device).eval()
            self.reason = f"loaded local checkpoint: {model_dir}"
        except Exception as exc:
            self.reason = f"unavailable local depth model: {type(exc).__name__}: {exc}"
            LOG.warning(self.reason)

    def predict(self, image_bgr: np.ndarray) -> np.ndarray | None:
        if self.model is None:
            return None
        try:
            from PIL import Image

            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            inputs = self.processor(images=Image.fromarray(rgb), return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self.torch.inference_mode():
                outputs = self.model(**inputs)
                depth = getattr(outputs, "predicted_depth", None)
                if depth is None:
                    depth = getattr(outputs, "depth", None)
                if depth is None:
                    raise ValueError("model output contains no predicted_depth or depth tensor")
                depth = self.torch.nn.functional.interpolate(
                    depth.unsqueeze(1) if depth.ndim == 3 else depth,
                    size=image_bgr.shape[:2], mode="bicubic", align_corners=False,
                )[0, 0].float().cpu().numpy()
            finite = np.isfinite(depth)
            if not finite.any():
                return None
            lo, hi = np.quantile(depth[finite], [0.05, 0.95])
            if hi <= lo:
                return None
            return np.clip((depth - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
        except Exception as exc:
            LOG.warning("Depth inference failed for one frame: %s: %s", type(exc).__name__, exc)
            return None


def depth_features(depth: np.ndarray) -> dict[str, float]:
    gy, gx = np.gradient(depth)
    gradient = np.hypot(gx, gy)
    cutoff = float(np.quantile(gradient, 0.90))
    return {
        "relative_depth_mean_unitless": float(depth.mean()),
        "relative_depth_std_unitless": float(depth.std()),
        "depth_occlusion_boundary_proxy_density": float(np.mean(gradient >= cutoff)),
    }


def append_values(bucket: dict[str, Any], values: dict[str, float], elapsed: float) -> None:
    for metric, value in values.items():
        if value is not None and math.isfinite(float(value)):
            bucket["values"][metric].append(float(value))
    if "flow_mean_magnitude_px_analysis" in values and math.isfinite(values["flow_mean_magnitude_px_analysis"]):
        bucket["spectrum"].append((elapsed, values["flow_mean_magnitude_px_analysis"]))


def fft_peak_frequency(samples: list[tuple[float, float]]) -> float:
    if len(samples) < 16:
        return float("nan")
    times = np.asarray([x[0] for x in samples], dtype=float)
    signal = np.asarray([x[1] for x in samples], dtype=float)
    order = np.argsort(times)
    times, signal = times[order], signal[order]
    dt = np.diff(times)
    dt = dt[dt > 0]
    if not len(dt) or float(np.std(signal)) < 1e-9:
        return float("nan")
    median_dt = float(np.median(dt))
    uniform_times = np.arange(times[0], times[-1], median_dt)
    if len(uniform_times) < 16:
        return float("nan")
    # Timestamps are slightly irregular. Interpolate the scalar flow series to
    # the median observed cadence before applying a Hann-windowed FFT.
    uniform_signal = np.interp(uniform_times, times, signal)
    uniform_signal = uniform_signal - uniform_signal.mean()
    spectrum = np.abs(np.fft.rfft(uniform_signal * np.hanning(len(uniform_signal))))
    frequencies = np.fft.rfftfreq(len(uniform_signal), d=median_dt)
    if len(spectrum) < 2:
        return float("nan")
    spectrum[0] = 0.0
    return float(frequencies[int(np.argmax(spectrum))])


def new_bucket() -> dict[str, Any]:
    return {"values": defaultdict(list), "spectrum": [], "image_count": 0, "pair_count": 0, "fovs": []}


def summarize(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=0)), float(np.max(arr))


def process_session(
    session_dir: Path,
    output_rows: list[dict[str, Any]],
    stats_by_metric: dict[str, list[float]],
    window_sec: float,
    max_dimension: int,
    depth_model_dir: Path | None,
) -> dict[str, Any]:
    csv_path = session_dir / "data_file.csv"
    if not csv_path.is_file():
        LOG.error("Skip %s: data_file.csv not found", session_dir)
        return {"session": session_dir.name, "status": "skipped_no_data_file"}
    timestamps, expected_frames, fov_by_frame = read_session_metadata(csv_path)
    image_dir = session_dir / "video" if (session_dir / "video").is_dir() else session_dir
    images = sorted(
        [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}],
        key=lambda p: (parse_frame_value(p.stem) is None, parse_frame_value(p.stem) or -1, p.name.lower()),
    )
    frame_to_path = {parse_frame_value(path.stem): path for path in images if parse_frame_value(path.stem) is not None}
    missing_frames = sorted(expected_frames - set(frame_to_path))
    unexpected_frames = sorted(set(frame_to_path) - expected_frames) if expected_frames else []
    if missing_frames:
        LOG.warning("%s missing %d expected image files; first IDs: %s", session_dir.name, len(missing_frames), missing_frames[:20])
    if unexpected_frames:
        LOG.warning("%s has %d JPG files absent from CSV video index; first IDs: %s", session_dir.name, len(unexpected_frames), unexpected_frames[:20])
    available = [frame for frame in sorted(frame_to_path) if frame in timestamps]
    if not available:
        return {"session": session_dir.name, "status": "skipped_no_mapped_jpg"}
    t0 = min(timestamps[frame] for frame in timestamps)
    observed_duration_sec = (max(timestamps.values()) - t0) / 1000.0
    buckets: dict[int, dict[str, Any]] = defaultdict(new_bucket)
    depth = LocalDepthEstimator(depth_model_dir)
    LOG.info(
        "Session %s: rows in telemetry=%d, JPGs=%d, indexed JPGs=%d, missing JPGs=%d; depth=%s",
        session_dir.name, len(timestamps), len(images), len(expected_frames), len(missing_frames), depth.reason,
    )
    previous_gray: np.ndarray | None = None
    previous_flow: np.ndarray | None = None
    previous_pair_mid_sec: float | None = None
    previous_homography_speed: float | None = None
    previous_depth: np.ndarray | None = None
    previous_depth_sec: float | None = None
    previous_frame_id: int | None = None
    decoded_images = 0
    valid_pairs = 0
    failed_images: list[int] = []
    all_timestamps = sorted(timestamps.values())
    total_observed_sec = (max(all_timestamps) - min(all_timestamps)) / 1000.0 if all_timestamps else 0.0
    for frame_id in available:
        path = frame_to_path[frame_id]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            failed_images.append(frame_id)
            LOG.error("Failed image decode: %s", path)
            previous_gray = None
            previous_flow = None
            previous_depth = None
            continue
        decoded_images += 1
        original_h, original_w = image.shape[:2]
        scale = min(1.0, max_dimension / max(original_h, original_w))
        if scale < 1.0:
            image = cv2.resize(
                image,
                (max(1, round(original_w * scale)), max(1, round(original_h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        gray, static = static_features(image)
        stamp_sec = (timestamps[frame_id] - t0) / 1000.0
        window_index = max(0, int(stamp_sec // window_sec))
        bucket = buckets[window_index]
        bucket["image_count"] += 1
        append_values(bucket, static, stamp_sec)
        fov = fov_by_frame.get(frame_id)
        if fov is not None:
            bucket["fovs"].append(fov)
        current_depth = depth.predict(image)
        if current_depth is not None:
            append_values(bucket, depth_features(current_depth), stamp_sec)
            if previous_depth is not None and previous_depth.shape == current_depth.shape:
                dt_depth = stamp_sec - float(previous_depth_sec)
                if dt_depth > 0:
                    change = float(np.mean(np.abs(current_depth - previous_depth)) / dt_depth)
                    append_values(bucket, {"depth_change_rate": change}, stamp_sec)
        current_gray = gray
        if previous_gray is not None and previous_frame_id is not None:
            dt = (timestamps[frame_id] - timestamps[previous_frame_id]) / 1000.0
            if dt > 0:
                flow = cv2.calcOpticalFlowFarneback(
                    previous_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0,
                )
                mid_sec = (stamp_sec + (timestamps[previous_frame_id] - t0) / 1000.0) / 2.0
                pair_window = max(0, int(mid_sec // window_sec))
                pair_bucket = buckets[pair_window]
                flow_values = flow_features(flow)
                between_pair_sec = mid_sec - previous_pair_mid_sec if previous_pair_mid_sec is not None else float("nan")
                flow_values["flow_direction_change_rate_deg_s"] = direction_change(previous_flow, flow, between_pair_sec)
                homography_values = homography_proxy(previous_gray, current_gray, dt, previous_homography_speed)
                speed = homography_values["homography_translation_speed_proxy_px_s"]
                previous_homography_speed = speed if math.isfinite(speed) else None
                append_values(pair_bucket, flow_values, mid_sec)
                append_values(pair_bucket, homography_values, mid_sec)
                pair_bucket["pair_count"] += 1
                valid_pairs += 1
                previous_flow = flow
                previous_pair_mid_sec = mid_sec
        previous_gray = current_gray
        previous_frame_id = frame_id
        previous_depth = current_depth
        previous_depth_sec = stamp_sec if current_depth is not None else None
        if decoded_images % 250 == 0:
            LOG.info("%s progress: decoded_frames=%d/%d; flow_pairs=%d", session_dir.name, decoded_images, len(available), valid_pairs)

    if failed_images:
        LOG.warning("%s failed to decode %d JPGs; first frame IDs: %s", session_dir.name, len(failed_images), failed_images[:20])
    number_of_windows = max(1, int(math.ceil(max(0.0, total_observed_sec) / window_sec)))
    for window_index in sorted(buckets):
        bucket = buckets[window_index]
        row: dict[str, Any] = {
            "session_id": session_dir.name,
            "window_index": window_index,
            "window_start_sec": window_index * window_sec,
            "window_end_sec": min((window_index + 1) * window_sec, total_observed_sec),
            "window_center_sec": min((window_index + 0.5) * window_sec, total_observed_sec),
            "observed_session_duration_sec": total_observed_sec,
            "observed_session_duration_min": total_observed_sec / 60.0,
            "source_frame_count": bucket["image_count"],
            "source_pair_count": bucket["pair_count"],
            "window_status": "complete_base_features" if bucket["image_count"] and bucket["pair_count"] else "partial",
            "analysis_max_dimension_px": max_dimension,
            "expected_jpg_count_from_index": len(expected_frames),
            "found_jpg_count": len(frame_to_path),
            "decoded_jpg_count": decoded_images,
            "missing_expected_jpg_count": len(missing_frames),
            "failed_jpg_count": len(failed_images),
            "camera_fov_deg": float(np.median(bucket["fovs"])) if bucket["fovs"] else float("nan"),
        }
        for metric in (*STATIC_METRICS.keys(), *PAIR_METRICS.keys()):
            mean, std, maximum = summarize(bucket["values"].get(metric, []))
            row[metric] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_max"] = maximum
            if math.isfinite(mean):
                stats_by_metric[metric].append(mean)
        row["flow_magnitude_dominant_frequency_hz"] = fft_peak_frequency(bucket["spectrum"])
        # Do not map an image-texture spectrum onto a physical scene oscillation rule.
        row["oscillation_frequency_hz"] = float("nan")
        # Explicitly retain rulebook metric names while refusing unsupported proxies.
        for metric in RULE_METRIC_COLUMNS:
            if metric == "camera_fov_deg":
                continue
            if metric in {"has_reference_frame", "controllability"}:
                row[metric] = "unknown"
            else:
                row[metric] = float("nan")
        output_rows.append(row)
    return {
        "session": session_dir.name,
        "status": "processed",
        "telemetry_rows": len(timestamps),
        "indexed_jpgs": len(expected_frames),
        "found_jpgs": len(frame_to_path),
        "decoded_jpgs": decoded_images,
        "missing_jpgs": missing_frames,
        "unexpected_jpgs": unexpected_frames,
        "failed_jpgs": failed_images,
        "flow_pairs": valid_pairs,
        "duration_sec": total_observed_sec,
        "duration_min": total_observed_sec / 60.0,
        "window_count": len(buckets),
        "first_timestamp_ms": min(timestamps.values()),
        "last_timestamp_ms": max(timestamps.values()),
        "depth_status": depth.reason,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No window feature rows were produced")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_statistics(path: Path, rows: list[dict[str, Any]]) -> None:
    numeric_metrics = [*STATIC_METRICS, *PAIR_METRICS, "flow_magnitude_dominant_frequency_hz", "camera_fov_deg"]
    units = {**STATIC_METRICS, **PAIR_METRICS, "flow_magnitude_dominant_frequency_hz": "Hz (visual-flow magnitude spectrum)", "camera_fov_deg": "degrees"}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["metric", "unit", "n_windows", "n_nonmissing", "mean", "std", "median", "min", "max", "missing_fraction"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metric in numeric_metrics:
            vals = []
            for row in rows:
                value = row.get(metric)
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    vals.append(value)
            array = np.asarray(vals, dtype=float)
            n = len(rows)
            writer.writerow({
                "metric": metric,
                "unit": units[metric],
                "n_windows": n,
                "n_nonmissing": len(vals),
                "mean": float(array.mean()) if len(array) else "",
                "std": float(array.std()) if len(array) else "",
                "median": float(np.median(array)) if len(array) else "",
                "min": float(array.min()) if len(array) else "",
                "max": float(array.max()) if len(array) else "",
                "missing_fraction": 1.0 - (len(vals) / n) if n else 1.0,
            })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", type=Path, required=True, help="Folder containing one subfolder per session")
    parser.add_argument("--output", type=Path, required=True, help="Windowed feature matrix CSV")
    parser.add_argument("--stats", type=Path, required=True, help="Per-feature descriptive statistics CSV")
    parser.add_argument("--window-sec", type=float, default=30.0)
    parser.add_argument("--max-dimension", type=int, default=320)
    parser.add_argument("--depth-model-dir", type=Path, default=None, help="Optional local MiDaS/Depth Anything Transformers checkpoint directory; local-only, no downloads")
    parser.add_argument("--log", type=Path, default=Path("results/full_feature_extraction_log.txt"))
    args = parser.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    if args.window_sec <= 0 or args.max_dimension < 64:
        parser.error("--window-sec must be positive and --max-dimension must be at least 64")
    sessions = sorted(
        [path for path in args.samples_dir.iterdir() if path.is_dir() and (path / "data_file.csv").is_file()],
        key=lambda path: path.name.lower(),
    ) if args.samples_dir.is_dir() else []
    if not sessions:
        parser.error(f"No session directories containing data_file.csv under {args.samples_dir}")
    LOG.info("Start: samples_dir=%s output=%s stats=%s window_sec=%s max_dimension=%s depth_model_dir=%s",
             args.samples_dir, args.output, args.stats, args.window_sec, args.max_dimension, args.depth_model_dir or "not supplied")
    output_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    stats_by_metric: dict[str, list[float]] = defaultdict(list)
    for session_dir in sessions:
        try:
            summaries.append(process_session(session_dir, output_rows, stats_by_metric, args.window_sec, args.max_dimension, args.depth_model_dir))
        except Exception:
            LOG.exception("Session processing failed: %s", session_dir)
            summaries.append({"session": session_dir.name, "status": "failed"})
    write_csv(args.output, output_rows)
    write_statistics(args.stats, output_rows)
    total_images = sum(int(s.get("decoded_jpgs", 0)) for s in summaries)
    total_pairs = sum(int(s.get("flow_pairs", 0)) for s in summaries)
    total_missing = sum(len(s.get("missing_jpgs", [])) for s in summaries)
    total_failed = sum(len(s.get("failed_jpgs", [])) for s in summaries)
    LOG.info("Complete: sessions=%d windows=%d decoded_images=%d flow_pairs=%d missing_expected_jpgs=%d failed_jpgs=%d",
             len(sessions), len(output_rows), total_images, total_pairs, total_missing, total_failed)
    for summary in summaries:
        LOG.info("Session summary: %s", summary)
    print(f"windows={len(output_rows)} sessions={len(sessions)} decoded_images={total_images} flow_pairs={total_pairs} missing_expected_jpgs={total_missing} failed_jpgs={total_failed}")
    print(f"features={args.output} statistics={args.stats} log={args.log}")
    return 0 if output_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
