#!/usr/bin/env python3
"""Extract explicit video-only baseline features from clip files.

No MS/FMS/SSQ, VR.net matrix telemetry, metric camera/object motion, or learned
depth model is used. Optical-flow and homography quantities are image-space
proxies and must not be described as metres or physical scene motion.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd


ID_COLUMNS = ["clip_id", "session_id", "game_name"]
MANIFEST_COLUMNS = [*ID_COLUMNS, "clip_path", "start_time", "end_time"]

STATIC_BASES = [
    "edge_density",
    "color_entropy_bits",
    "laplacian_variance",
    "brightness_mean_0_255",
    "brightness_std_0_255",
]
STATIC_FEATURES = [f"{base}_{stat}" for base in STATIC_BASES for stat in ("mean", "max", "std")]
FLOW_FEATURES = [
    "flow_magnitude_mean_px_per_sample",
    "flow_magnitude_max_px_per_sample",
    "flow_magnitude_std_px_per_sample",
    "flow_magnitude_entropy_bits",
    "flow_direction_entropy_bits",
    "flow_direction_circular_mean_deg",
    "flow_direction_circular_concentration",
]
HOMOGRAPHY_FEATURES = [
    f"homography_translation_proxy_px_s_{stat}" for stat in ("mean", "max", "std")
] + [f"homography_rotation_proxy_deg_s_{stat}" for stat in ("mean", "max", "std")]
VIDEO_FEATURE_COLUMNS = [*STATIC_FEATURES, *FLOW_FEATURES, *HOMOGRAPHY_FEATURES]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clips-manifest", type=Path, default=Path("features/video_clips_manifest.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("features/lightgbm_features.csv")
    )
    parser.add_argument(
        "--log", type=Path, default=Path("results/lightgbm_feature_extraction_log.txt")
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=5.0,
        help="Decode the full clip but analyze approximately this many frames/s.",
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=320,
        help="Downscale frames so their longest dimension is at most this value.",
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, expected: Iterable[str], source: Path) -> None:
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def resolve_clip_path(value: object, manifest_path: Path) -> Path:
    raw = Path(str(value))
    if raw.is_absolute():
        return raw
    candidates = [Path.cwd() / raw, manifest_path.parent / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def resize_frame(frame: np.ndarray, max_dimension: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(1.0, max_dimension / max(height, width))
    if scale >= 1.0:
        return frame
    return cv2.resize(
        frame,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def entropy_from_counts(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total <= 0:
        return math.nan
    probabilities = counts[counts > 0].astype(float) / total
    return float(-np.sum(probabilities * np.log2(probabilities)))


def color_entropy(frame: np.ndarray) -> float:
    values: list[float] = []
    for channel in cv2.split(frame):
        counts = cv2.calcHist([channel], [0], None, [64], [0, 256]).ravel()
        values.append(entropy_from_counts(counts))
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else math.nan


def frame_metrics(frame: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return gray, {
        "edge_density": float(np.mean(edges > 0)),
        "color_entropy_bits": color_entropy(frame),
        "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "brightness_mean_0_255": float(gray.mean()),
        "brightness_std_0_255": float(gray.std()),
    }


def homography_proxy(
    previous_gray: np.ndarray, current_gray: np.ndarray, delta_seconds: float
) -> tuple[float, float] | None:
    if not math.isfinite(delta_seconds) or delta_seconds <= 0:
        return None
    points = cv2.goodFeaturesToTrack(
        previous_gray, maxCorners=500, qualityLevel=0.01, minDistance=7, blockSize=7
    )
    if points is None or len(points) < 6:
        return None
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if tracked is None or status is None:
        return None
    keep = status.reshape(-1).astype(bool)
    if int(keep.sum()) < 6:
        return None
    transform, _ = cv2.estimateAffinePartial2D(
        points.reshape(-1, 2)[keep],
        tracked.reshape(-1, 2)[keep],
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    if transform is None or not np.isfinite(transform).all():
        return None
    translation_px_s = float(np.linalg.norm(transform[:, 2]) / delta_seconds)
    rotation_deg_s = float(
        abs(math.degrees(math.atan2(float(transform[1, 0]), float(transform[0, 0]))))
        / delta_seconds
    )
    return translation_px_s, rotation_deg_s


def aggregate(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return math.nan, math.nan, math.nan
    return float(array.mean()), float(array.max()), float(array.std(ddof=0))


def extract_clip(path: Path, sample_fps: float, max_dimension: int) -> tuple[dict[str, float], dict[str, object]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError("OpenCV could not open the clip")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frame_count = int(max(0, round(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
    if not math.isfinite(source_fps) or source_fps <= 0:
        capture.release()
        raise RuntimeError(f"Invalid source FPS: {source_fps!r}")
    step = max(1, int(round(source_fps / sample_fps)))

    static_values: dict[str, list[float]] = {name: [] for name in STATIC_BASES}
    magnitude_sum = 0.0
    magnitude_sumsq = 0.0
    magnitude_count = 0
    magnitude_max = -math.inf
    # Fixed log-magnitude bins keep the definition reproducible. Values above
    # 100 px/sample enter the final bin; this remains an image-space proxy.
    magnitude_edges = np.linspace(0.0, math.log1p(100.0), 65)
    magnitude_histogram = np.zeros(64, dtype=np.float64)
    direction_edges = np.linspace(-math.pi, math.pi, 37)
    direction_histogram = np.zeros(36, dtype=np.float64)
    direction_sin = 0.0
    direction_cos = 0.0
    direction_weight = 0.0
    homography_translation: list[float] = []
    homography_rotation: list[float] = []

    decoded = 0
    sampled = 0
    flow_pairs = 0
    previous_gray: np.ndarray | None = None
    previous_sampled_index: int | None = None

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index = decoded
        decoded += 1
        if frame_index % step != 0:
            continue
        sampled += 1
        resized = resize_frame(frame, max_dimension)
        gray, metrics = frame_metrics(resized)
        for name, value in metrics.items():
            static_values[name].append(value)

        if previous_gray is not None and previous_sampled_index is not None:
            flow = cv2.calcOpticalFlowFarneback(
                previous_gray,
                gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
            finite = np.isfinite(magnitude) & np.isfinite(angle)
            if finite.any():
                mag = magnitude[finite].astype(np.float64)
                ang = angle[finite].astype(np.float64)
                # cartToPolar returns [0, 2pi); convert for symmetric bins.
                ang = (ang + math.pi) % (2.0 * math.pi) - math.pi
                magnitude_sum += float(mag.sum())
                magnitude_sumsq += float(np.square(mag).sum())
                magnitude_count += int(mag.size)
                magnitude_max = max(magnitude_max, float(mag.max()))
                magnitude_histogram += np.histogram(
                    np.minimum(np.log1p(mag), magnitude_edges[-1]), bins=magnitude_edges
                )[0]
                direction_histogram += np.histogram(
                    ang, bins=direction_edges, weights=mag
                )[0]
                direction_sin += float(np.sum(mag * np.sin(ang)))
                direction_cos += float(np.sum(mag * np.cos(ang)))
                direction_weight += float(mag.sum())
                flow_pairs += 1

            delta_seconds = (frame_index - previous_sampled_index) / source_fps
            proxy = homography_proxy(previous_gray, gray, delta_seconds)
            if proxy is not None:
                homography_translation.append(proxy[0])
                homography_rotation.append(proxy[1])

        previous_gray = gray
        previous_sampled_index = frame_index

    capture.release()
    if sampled == 0:
        raise RuntimeError("No frame could be decoded at the requested sampling rate")

    features: dict[str, float] = {}
    for base in STATIC_BASES:
        mean, maximum, std = aggregate(static_values[base])
        features[f"{base}_mean"] = mean
        features[f"{base}_max"] = maximum
        features[f"{base}_std"] = std

    if magnitude_count:
        magnitude_mean = magnitude_sum / magnitude_count
        magnitude_variance = max(0.0, magnitude_sumsq / magnitude_count - magnitude_mean**2)
        features.update(
            {
                "flow_magnitude_mean_px_per_sample": magnitude_mean,
                "flow_magnitude_max_px_per_sample": magnitude_max,
                "flow_magnitude_std_px_per_sample": math.sqrt(magnitude_variance),
                "flow_magnitude_entropy_bits": entropy_from_counts(magnitude_histogram),
                "flow_direction_entropy_bits": entropy_from_counts(direction_histogram),
            }
        )
        if direction_weight > 0:
            features["flow_direction_circular_mean_deg"] = math.degrees(
                math.atan2(direction_sin, direction_cos)
            )
            features["flow_direction_circular_concentration"] = float(
                math.hypot(direction_sin, direction_cos) / direction_weight
            )
        else:
            features["flow_direction_circular_mean_deg"] = math.nan
            features["flow_direction_circular_concentration"] = math.nan
    else:
        for name in FLOW_FEATURES:
            features[name] = math.nan

    translation_mean, translation_max, translation_std = aggregate(homography_translation)
    rotation_mean, rotation_max, rotation_std = aggregate(homography_rotation)
    features.update(
        {
            "homography_translation_proxy_px_s_mean": translation_mean,
            "homography_translation_proxy_px_s_max": translation_max,
            "homography_translation_proxy_px_s_std": translation_std,
            "homography_rotation_proxy_deg_s_mean": rotation_mean,
            "homography_rotation_proxy_deg_s_max": rotation_max,
            "homography_rotation_proxy_deg_s_std": rotation_std,
        }
    )
    metadata: dict[str, object] = {
        "source_fps": source_fps,
        "source_frame_count": source_frame_count,
        "decoded_frame_count": decoded,
        "sampled_frame_count": sampled,
        "flow_pair_count": flow_pairs,
        "homography_pair_count": len(homography_translation),
        "sampling_step_frames": step,
        "analysis_max_dimension_px": max_dimension,
        "extraction_status": "success" if sampled >= 2 else "partial_single_frame",
    }
    return features, metadata


def main() -> None:
    args = parse_args()
    if args.sample_fps <= 0:
        raise ValueError("--sample-fps must be positive")
    if args.max_dimension < 32:
        raise ValueError("--max-dimension must be at least 32")
    if not args.clips_manifest.exists():
        raise FileNotFoundError(args.clips_manifest)

    manifest = pd.read_csv(args.clips_manifest, encoding="utf-8-sig")
    require_columns(manifest, MANIFEST_COLUMNS, args.clips_manifest)
    if manifest[ID_COLUMNS].isna().any().any():
        raise ValueError("Manifest ID columns contain missing values")
    if manifest.duplicated(ID_COLUMNS).any():
        duplicates = manifest.loc[manifest.duplicated(ID_COLUMNS, keep=False), ID_COLUMNS]
        raise ValueError(f"Manifest IDs are not unique:\n{duplicates.to_string(index=False)}")

    output_rows: list[dict[str, object]] = []
    log_lines = [
        "LightGBM video-only feature extraction log",
        f"manifest={args.clips_manifest}",
        f"output={args.output}",
        f"sample_fps={args.sample_fps}",
        f"max_dimension={args.max_dimension}",
        "label/outcome columns used=none",
        "depth features=SKIPPED (no external depth model supplied; no depth model is trained here)",
        "metric camera/object features=SKIPPED (require independent matrix supervision and are not video inputs)",
        "homography translation/rotation=IMAGE-SPACE PROXIES only; not physical motion",
        "",
    ]
    started = time.perf_counter()
    for index, row in manifest.iterrows():
        clip_path = resolve_clip_path(row["clip_path"], args.clips_manifest)
        output: dict[str, object] = {column: row[column] for column in MANIFEST_COLUMNS}
        output["resolved_clip_path"] = str(clip_path)
        clip_started = time.perf_counter()
        try:
            if not clip_path.exists():
                raise FileNotFoundError(clip_path)
            features, metadata = extract_clip(clip_path, args.sample_fps, args.max_dimension)
            output.update(metadata)
            output.update(features)
            log_lines.append(
                f"SUCCESS clip_id={row['clip_id']} session={row['session_id']} "
                f"sampled={metadata['sampled_frame_count']} flow_pairs={metadata['flow_pair_count']} "
                f"homography_pairs={metadata['homography_pair_count']} "
                f"elapsed_sec={time.perf_counter()-clip_started:.3f} path={clip_path}"
            )
        except Exception as exc:
            output.update(
                {
                    "extraction_status": "failed",
                    "extraction_error": f"{type(exc).__name__}: {exc}",
                }
            )
            for feature in VIDEO_FEATURE_COLUMNS:
                output[feature] = math.nan
            log_lines.append(
                f"FAILED clip_id={row['clip_id']} session={row['session_id']} "
                f"error={type(exc).__name__}: {exc} path={clip_path}"
            )
        output_rows.append(output)
        print(f"[{index+1}/{len(manifest)}] {row['clip_id']}: {output.get('extraction_status')}")

    ordered = [
        *MANIFEST_COLUMNS,
        "resolved_clip_path",
        "source_fps",
        "source_frame_count",
        "decoded_frame_count",
        "sampled_frame_count",
        "flow_pair_count",
        "homography_pair_count",
        "sampling_step_frames",
        "analysis_max_dimension_px",
        "extraction_status",
        "extraction_error",
        *VIDEO_FEATURE_COLUMNS,
    ]
    result = pd.DataFrame(output_rows)
    for column in ordered:
        if column not in result.columns:
            result[column] = math.nan
    result = result[ordered]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig", float_format="%.12g")

    succeeded = int(result["extraction_status"].isin(["success", "partial_single_frame"]).sum())
    log_lines.extend(
        [
            "",
            f"rows={len(result)} succeeded={succeeded} failed={len(result)-succeeded}",
            f"elapsed_sec={time.perf_counter()-started:.3f}",
        ]
    )
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(result)} rows to {args.output}; log={args.log}")


if __name__ == "__main__":
    main()
