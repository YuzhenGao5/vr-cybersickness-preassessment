#!/usr/bin/env python3
"""Stream local VR.net matrices into five-second windows paired to real reports."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from parse_vrnet_matrices import choose_camera, literal_list, numeric_matrix_list, timestamp_scale

MEASURES = ("camera_speed", "camera_acceleration", "camera_rotation_rate", "fov_horizontal", "fov_vertical", "object_count")
SOURCE_FIELDS = ("frame", "timestamp", "camera_name", "p_matrix", "v_matrix", "m_matrix", "MS_rating")
Z_FLIP = np.diag([1.0, 1.0, -1.0])


class MissingCameraObservation(Exception):
    """An interleaved sensor row, not a broken camera measurement."""


def finite_number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Moments:
    def __init__(self):
        self.n, self.mean, self.m2, self.maximum = 0, 0.0, 0.0, -math.inf

    def add(self, value):
        if value is None or not math.isfinite(value):
            return
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)
        self.maximum = max(self.maximum, value)

    def output(self, prefix):
        return {prefix + "_mean": self.mean if self.n else np.nan,
                prefix + "_max": self.maximum if self.n else np.nan,
                prefix + "_std": math.sqrt(max(0.0, self.m2 / (self.n - 1))) if self.n > 1 else np.nan,
                prefix + "_n": self.n}


def full_fov(matrix, rounding):
    """Full opening for the observed transposed asymmetric perspective matrix."""
    if abs(matrix[3, 3]) >= 1e-4 or abs(abs(matrix[2, 3]) - 1.0) > 1e-4:
        raise ValueError("not a supported perspective projection")
    # Skew/oblique projection needs a separate ray-based convention audit.
    if max(abs(matrix[0, 1]), abs(matrix[1, 0]), abs(matrix[0, 2]), abs(matrix[1, 2])) > 1e-6:
        raise ValueError("skew/oblique projection not supported")
    result = []
    for axis in (0, 1):
        scale = abs(float(matrix[axis, axis]))
        offset = float(matrix[2, axis])
        if scale <= 1e-12:
            raise ValueError("zero projection scale")
        opening = math.degrees(math.atan((1.0 + offset) / scale) - math.atan((-1.0 + offset) / scale))
        if not 0.0 < opening < 180.0:
            raise ValueError("invalid projection opening")
        result.append(round(opening / rounding) * rounding)
    return result


def parse_pose(view):
    world = np.linalg.inv(view)
    position = world[3, :3].copy()
    rotation = (Z_FLIP @ world[:3, :3]).T
    if not np.isfinite(world).all() or abs(np.linalg.det(rotation) - 1.0) > 1e-3 or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
        raise ValueError("view does not yield a finite rigid camera pose")
    return position, rotation


def process_session(path, config, logger, csv_reader=None):
    # Inject a reader for an already-local ZIP member without extracting images.
    # The default preserves the existing standalone CSV command.
    read_csv = csv_reader if csv_reader is not None else pd.read_csv
    started = time.perf_counter()
    session, game = path.parent.name, path.parent.parent.name
    inventory = {"session_id": session, "game_name": game, "source_file": str(path.resolve()), "size_bytes": path.stat().st_size}
    header = read_csv(path, nrows=0, encoding="utf-8-sig").columns.tolist()
    inventory["columns"] = header
    missing = [c for c in ("timestamp", "camera_name", "p_matrix", "v_matrix", "m_matrix", "MS_rating") if c not in header]
    if missing:
        inventory.update(status="skipped_missing_columns", missing_columns=missing)
        logger.info("SKIP %s missing=%s", session, missing)
        return [], inventory
    width = float(config["window_seconds"])
    max_gap = float(config["maximum_derivative_interval_seconds"])
    jump_limit = float(config["camera_jump_displacement_source_units"])
    columns = [c for c in SOURCE_FIELDS if c in header]
    windows = {}
    reports = []
    counters = Counter()
    error_examples = []
    origin_epoch = None
    last_elapsed = 0.0
    scale, unit = None, None
    selected_name = None
    previous = None
    previous_velocity = None
    record_number = 0
    last_progress = time.perf_counter()
    for chunk in read_csv(path, usecols=columns, dtype=str, chunksize=16, keep_default_na=False, encoding="utf-8-sig"):
        for source_row in chunk.to_dict("records"):
            record_number += 1
            raw_time = finite_number(source_row.get("timestamp"))
            if raw_time is None:
                counters["invalid_timestamp"] += 1
                previous = previous_velocity = None
                continue
            if origin_epoch is None:
                scale, unit = timestamp_scale(raw_time)
                origin_epoch = raw_time * scale
            elapsed = raw_time * scale - origin_epoch
            if elapsed < 0:
                counters["before_session_origin"] += 1
                previous = previous_velocity = None
                continue
            last_elapsed = max(last_elapsed, elapsed)
            index = int(math.floor(elapsed / width))
            if index not in windows:
                windows[index] = {"moments": {x: Moments() for x in MEASURES}, "raw_rows": 0, "time_min": elapsed, "time_max": elapsed}
            bucket = windows[index]
            bucket["raw_rows"] += 1
            bucket["time_min"] = min(bucket["time_min"], elapsed)
            bucket["time_max"] = max(bucket["time_max"], elapsed)
            score = finite_number(source_row.get("MS_rating"))
            if score is not None:
                reports.append({"rating_time_sec": elapsed, "ms_rating": score, "rating_event_id": f"modern:{session}:record:{record_number}", "rating_record_number": record_number})
            elif source_row.get("MS_rating", "").strip():
                counters["invalid_nonempty_rating"] += 1
            values = {x: None for x in MEASURES}
            # Missing object data is missing, not a zero-object observation.
            object_raw = source_row.get("m_matrix", "")
            if object_raw.strip():
                try:
                    values["object_count"] = float(len(numeric_matrix_list(object_raw)))
                    counters["valid_object_rows"] += 1
                except Exception as exc:
                    counters["invalid_object_rows"] += 1
                    if len(error_examples) < 10:
                        error_examples.append(f"record {record_number}: object: {type(exc).__name__}: {exc}")
            try:
                if not source_row.get("v_matrix", "").strip():
                    raise MissingCameraObservation("camera not sampled in this source row")
                names = literal_list(source_row.get("camera_name", ""))
                views = numeric_matrix_list(source_row.get("v_matrix", ""))
                projections = numeric_matrix_list(source_row.get("p_matrix", ""))
                if not len(views):
                    raise MissingCameraObservation("empty camera observation")
                if not names or len(views) != len(names):
                    raise ValueError("missing camera or unequal camera/view list lengths")
                candidate_name = selected_name if selected_name is not None else str(names[choose_camera(names)])
                indices = [i for i, name in enumerate(names) if str(name) == candidate_name]
                if len(indices) != 1:
                    raise ValueError("fixed selected camera absent or duplicated")
                selected = indices[0]
                position, rotation = parse_pose(views[selected])
                selected_name = candidate_name
                counters["valid_camera_rows"] += 1
                if len(projections) == len(names):
                    try:
                        values["fov_horizontal"], values["fov_vertical"] = full_fov(projections[selected], float(config["fov_rounding_degrees"]))
                        counters["valid_fov_rows"] += 1
                    except Exception as exc:
                        counters["invalid_fov_rows"] += 1
                        if len(error_examples) < 10:
                            error_examples.append(f"record {record_number}: projection: {type(exc).__name__}: {exc}")
                else:
                    counters["missing_or_misaligned_projection"] += 1
                current = (elapsed, position, rotation)
                if previous is not None:
                    dt = elapsed - previous[0]
                    displacement = position - previous[1]
                    if dt <= 0 or dt > max_gap:
                        counters["derivative_interval_resets"] += 1
                        previous_velocity = None
                    elif np.linalg.norm(displacement) > jump_limit:
                        counters["position_jumps"] += 1
                        previous_velocity = None
                    else:
                        velocity = displacement / dt
                        midpoint = (elapsed + previous[0]) / 2.0
                        values["camera_speed"] = float(np.linalg.norm(velocity))
                        relative = previous[2].T @ rotation
                        rate = float(np.degrees(Rotation.from_matrix(relative).magnitude()) / dt)
                        values["camera_rotation_rate"] = rate
                        if rate > config["rotation_rate_flag_degrees_per_second"]:
                            counters["rotation_rate_over_flag"] += 1
                        if previous_velocity is not None:
                            velocity_dt = midpoint - previous_velocity[0]
                            if velocity_dt > 0:
                                values["camera_acceleration"] = float(np.linalg.norm((velocity - previous_velocity[1]) / velocity_dt))
                        previous_velocity = (midpoint, velocity)
                previous = current
            except MissingCameraObservation:
                counters["unsampled_camera_rows"] += 1
            except Exception as exc:
                counters["invalid_or_missing_camera_rows"] += 1
                previous = previous_velocity = None
                if len(error_examples) < 10:
                    error_examples.append(f"record {record_number}: camera: {type(exc).__name__}: {exc}")
            for measure, value in values.items():
                bucket["moments"][measure].add(value)
        if time.perf_counter() - last_progress > 25:
            logger.info("PROGRESS %s raw_rows=%s elapsed_work_seconds=%.1f", session, record_number, time.perf_counter() - started)
            last_progress = time.perf_counter()
    reports.sort(key=lambda x: (x["rating_time_sec"], x["rating_record_number"]))
    report_times = np.array([x["rating_time_sec"] for x in reports])
    output = []
    if origin_epoch is not None:
        for index in range(int(math.floor(last_elapsed / width)) + 1):
            start, end = index * width, min((index + 1) * width, last_elapsed)
            if end <= start:
                counters["zero_duration_terminal_windows_skipped"] += 1
                continue
            center = (start + end) / 2.0
            row = {"window_id": f"{session}:window:{index}", "game_name": game, "session_id": session,
                   "participant_id": "modern:" + session.split("_", 1)[0], "window_index": index,
                   "start_time_sec": start, "end_time_sec": end, "window_center_sec": center,
                   "window_duration_sec": end - start, "partial_window": end - start < width,
                   "session_origin_epoch_seconds": origin_epoch, "timestamp_unit": unit,
                   "selected_camera": selected_name, "source_file": str(path.resolve()),
                   "camera_speed_units": "source_coordinate_units_per_second",
                   "camera_acceleration_units": "source_coordinate_units_per_second_squared",
                   "camera_rotation_rate_units": "degrees_per_second", "pure_design_status": "not_certified"}
            bucket = windows.get(index)
            row["n_raw_rows"] = bucket["raw_rows"] if bucket else 0
            row["observed_span_sec"] = bucket["time_max"] - bucket["time_min"] if bucket else 0.0
            for measure in MEASURES:
                row.update((bucket["moments"][measure] if bucket else Moments()).output(measure))
            row["window_has_any_feature"] = any(row[x + "_n"] > 0 for x in MEASURES)
            if len(reports):
                insertion = int(np.searchsorted(report_times, center, side="left"))
                candidate_indices = [j for j in (insertion - 1, insertion) if 0 <= j < len(reports)]
                nearest = min(candidate_indices, key=lambda j: (abs(report_times[j] - center), report_times[j], reports[j]["rating_record_number"]))
                row.update(reports[nearest])
                delta = reports[nearest]["rating_time_sec"] - center
                row.update(rating_time_delta_sec=delta, rating_abs_distance_sec=abs(delta), label_is_future=delta > 0, label_source="nearest_real_report")
            else:
                row.update(ms_rating=np.nan, rating_event_id="", rating_time_sec=np.nan, rating_time_delta_sec=np.nan,
                           rating_abs_distance_sec=np.nan, label_is_future=False, label_source="missing")
            output.append(row)
    inventory.update(status="processed", raw_rows=record_number, timestamp_unit=unit, session_duration_sec=last_elapsed,
                     selected_camera=selected_name, real_rating_events=len(reports), rating_distribution=dict(Counter(str(x["ms_rating"]) for x in reports)),
                     output_windows=len(output), counters=dict(counters), error_examples=error_examples,
                     processing_seconds=round(time.perf_counter() - started, 3))
    logger.info("DONE %s raw_rows=%s windows=%s reports=%s jumps=%s duration=%.1fs", session, record_number, len(output), len(reports), counters["position_jumps"], last_elapsed)
    return output, inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/vrnet/sampled_csv"))
    parser.add_argument("--output", type=Path, default=Path("features/param_rating_matrix.csv"))
    parser.add_argument("--config", type=Path, default=Path("configs/param_rating_analysis_2026-10-10.json"))
    parser.add_argument("--manifest", type=Path, default=Path("results/param_rating_preparation_manifest.json"))
    parser.add_argument("--log", type=Path, default=Path("results/param_rating_extraction_log.txt"))
    args = parser.parse_args()
    if not Path("PROTOCOL.md").is_file():
        raise SystemExit("PROTOCOL.md missing: experiments forbidden")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for path in (args.output, args.manifest, args.log):
        path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(args.log, encoding="utf-8"), logging.StreamHandler()])
    logger = logging.getLogger("param_rating")
    paths = sorted(args.input_dir.rglob("data_file.csv"))
    if not paths:
        raise SystemExit("No local data_file.csv found")
    all_rows, inventory = [], []
    started = time.perf_counter()
    for path in paths:
        logger.info("START %s bytes=%s", path.parent.name, path.stat().st_size)
        try:
            rows, record = process_session(path, config, logger)
            all_rows.extend(rows)
            inventory.append(record)
        except Exception as exc:
            logger.exception("FAILED %s", path)
            inventory.append({"session_id": path.parent.name, "game_name": path.parent.parent.name, "source_file": str(path.resolve()),
                              "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    frame = pd.DataFrame(all_rows)
    if frame.empty:
        summary = {"session_inventory": inventory, "output_windows": 0, "status": "no_extractable_windows"}
        args.manifest.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit("No extractable windows; see manifest/log")
    frame.to_csv(args.output, index=False, encoding="utf-8-sig")
    usable = frame[frame["window_has_any_feature"] & frame["ms_rating"].notna()]
    distance = usable["rating_abs_distance_sec"]
    feature_summary = {}
    for column in config["feature_columns"]:
        values = pd.to_numeric(usable[column], errors="coerce")
        feature_summary[column] = {"nonempty": int(values.notna().sum()), "missing_fraction": float(values.isna().mean()),
                                   "min": float(values.min()) if values.notna().any() else None,
                                   "max": float(values.max()) if values.notna().any() else None,
                                   "unique_nonempty": int(values.nunique())}
    summary = {"status": "complete", "scope": str(args.input_dir.resolve()), "config": config, "config_sha256": sha256(args.config),
               "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
               "source_csv_count": len(paths), "source_game_count": len({p.parent.parent.name for p in paths}),
               "processed_session_count": sum(x["status"] == "processed" for x in inventory),
               "skipped_session_count": sum(x["status"].startswith("skipped") for x in inventory),
               "failed_session_count": sum(x["status"] == "failed" for x in inventory),
               "output_windows": len(frame), "usable_windows": len(usable), "usable_sessions": int(usable.session_id.nunique()),
               "usable_games": int(usable.game_name.nunique()), "real_rating_events_in_processed_sources": sum(x.get("real_rating_events", 0) for x in inventory),
               "paired_unique_rating_events": int(usable.rating_event_id.nunique()),
               "window_score_distribution": {str(k): int(v) for k, v in usable.ms_rating.value_counts().sort_index().items()},
               "paired_event_score_distribution": {str(k): int(v) for k, v in usable.drop_duplicates("rating_event_id").ms_rating.value_counts().sort_index().items()},
               "label_distance_seconds": {"median": float(distance.median()), "p90": float(distance.quantile(.9)), "max": float(distance.max()),
                                          "over_30s": int((distance > 30).sum()), "over_60s": int((distance > 60).sum()), "future_windows": int(usable.label_is_future.sum())},
               "feature_summary": feature_summary, "session_inventory": inventory, "matrix_sha256": sha256(args.output),
               "elapsed_seconds": round(time.perf_counter() - started, 3)}
    args.manifest.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    logger.info("COMPLETE windows=%s usable=%s sessions=%s games=%s real_reports=%s", len(frame), len(usable), usable.session_id.nunique(), usable.game_name.nunique(), summary["real_rating_events_in_processed_sources"])
    print(json.dumps({k: summary[k] for k in ("output_windows", "usable_windows", "usable_sessions", "usable_games", "real_rating_events_in_processed_sources", "paired_unique_rating_events", "label_distance_seconds")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
