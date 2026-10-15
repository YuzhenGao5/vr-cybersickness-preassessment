#!/usr/bin/env python3
"""Prepare strictly historical, event-level modern MS_rating regression data.

This script only prepares data. Runtime graphics are exploratory inputs; no
numeric field here has been certified as a user-independent design setting.
The input timestamps retain Unix epoch milliseconds. Positions retain source
coordinate units, and linear derivatives must not be described as m/s.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


FRAME_NUMERIC = [
    "frame_id", "timestamp", "fov_horizontal", "fov_vertical", "camera_pos_x",
    "camera_pos_y", "camera_pos_z", "camera_speed", "camera_acceleration",
    "camera_rotation_rate", "object_count", "mean_object_speed", "max_object_speed",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def numeric_quality(series: pd.Series) -> dict:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values)
    return {
        "finite_rows": int(finite.sum()),
        "missing_rows": int(np.isnan(values).sum()),
        "nonfinite_not_nan_rows": int((~finite & ~np.isnan(values)).sum()),
        "negative_rows": int((values[finite] < 0).sum()),
        "minimum": float(values[finite].min()) if finite.any() else None,
        "maximum": float(values[finite].max()) if finite.any() else None,
    }


def finite_nonnegative(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.where(np.isfinite(values) & values.ge(0))


def summarize_motion(values: pd.Series) -> tuple[float, float]:
    usable = finite_nonnegative(values).dropna()
    if usable.empty:
        return np.nan, np.nan
    return float(usable.median()), float(usable.quantile(0.9))


def clean_session(session: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Recompute derivatives on adjacent valid camera positions within a session.

    A jump starts a new coordinate segment. The next speed may be computed in
    that new segment; acceleration requires two valid adjacent speed intervals.
    Object speed is invalidated at the jump and its next valid position point.
    """
    ordered = session.sort_values(["timestamp", "frame_id", "_source_order"], kind="stable").copy()
    ordered["_scene"] = ordered[["fov_horizontal", "fov_vertical", "object_count"]].notna().any(axis=1)
    ordered["_speed"] = np.nan
    ordered["_acceleration"] = np.nan
    jump = truthy(ordered["camera_jump_flag"])
    positions = ordered[["camera_pos_x", "camera_pos_y", "camera_pos_z"]]
    valid_positions = np.isfinite(positions.to_numpy(dtype=float)).all(axis=1)
    camera = ordered.loc[valid_positions].copy()
    dt = camera["timestamp"].diff() / 1000.0
    displacement = np.sqrt(camera[["camera_pos_x", "camera_pos_y", "camera_pos_z"]].diff().pow(2).sum(axis=1, min_count=3))
    camera_jump = truthy(camera["camera_jump_flag"]) | displacement.gt(cfg["camera_jump_displacement_source_units"])
    derivative_valid = dt.gt(0) & dt.le(cfg["maximum_derivative_interval_seconds"]) & ~camera_jump
    speed = (displacement / dt).where(derivative_valid)
    acceleration = ((speed - speed.shift()).abs() / dt).where(derivative_valid & speed.notna() & speed.shift().notna())
    ordered.loc[camera.index, "_speed"] = speed
    ordered.loc[camera.index, "_acceleration"] = acceleration

    object_boundary = jump.copy()
    jump_indices = np.flatnonzero(camera_jump.to_numpy())
    if len(jump_indices):
        object_boundary.loc[camera.index[jump_indices]] = True
        following = jump_indices + 1
        following = following[following < len(camera)]
        object_boundary.loc[camera.index[following]] = True
    ordered["_object_speed"] = finite_nonnegative(ordered["mean_object_speed"]).where(~object_boundary)
    ordered["_object_count"] = finite_nonnegative(ordered["object_count"])
    ordered["_rotation_rate"] = finite_nonnegative(ordered["camera_rotation_rate"]).where(
        ~jump & ordered["camera_rotation_rate"].le(cfg["rotation_screen_degrees_per_second"])
    )
    for raw, derived in (("fov_horizontal", "_fov_h"), ("fov_vertical", "_fov_v")):
        step = cfg["fov_rounding_degrees"]
        values = finite_nonnegative(ordered[raw])
        ordered[derived] = np.round(values / step) * step
    stats = {
        "session_id": str(ordered["session_id"].iloc[0]),
        "rows": int(len(ordered)),
        "scene_rows": int(ordered["_scene"].sum()),
        "camera_position_rows": int(len(camera)),
        "duplicate_timestamps": int(ordered["timestamp"].diff().eq(0).sum()),
        "camera_intervals_nonpositive": int(dt.le(0).sum()),
        "camera_intervals_over_maximum": int(dt.gt(cfg["maximum_derivative_interval_seconds"]).sum()),
        "camera_source_jump_rows": int(jump.sum()),
        "camera_coordinate_discontinuities": int(camera_jump.sum()),
        "object_speed_boundary_rows": int(object_boundary.sum()),
        "object_speed_finite_values_invalidated_at_boundary": int((object_boundary & np.isfinite(ordered["mean_object_speed"])).sum()),
        "recomputed_speed_finite_rows": int(ordered["_speed"].notna().sum()),
        "recomputed_acceleration_finite_rows": int(ordered["_acceleration"].notna().sum()),
    }
    return ordered, stats


def dictionary_rows(cfg: dict) -> list[dict]:
    definitions = {
        "fov_horizontal_equivalent_mean": ("runtime_projection", "degrees", "fov_horizontal rounded to configured degrees, window mean", "symmetric-equivalent horizontal projection FOV"),
        "fov_vertical_equivalent_mean": ("runtime_projection", "degrees", "fov_vertical rounded to configured degrees, window mean", "symmetric-equivalent vertical projection FOV"),
        "visible_object_count_median": ("runtime_visible_objects", "visible renderer entries", "object_count window median", "visible renderer count, not total GameObject density"),
        "visible_object_count_p90": ("runtime_visible_objects", "visible renderer entries", "object_count window 90th percentile", "visible renderer count, not total GameObject density"),
        "object_speed_median": ("runtime_visible_objects", "source_coordinate_units/second", "cleaned mean_object_speed window median", "movement of uniquely named objects visible in adjacent scene records"),
        "object_speed_p90": ("runtime_visible_objects", "source_coordinate_units/second", "cleaned mean_object_speed window 90th percentile", "movement of uniquely named objects visible in adjacent scene records"),
        "exposure_elapsed_seconds": ("exposure", "seconds", "report timestamp minus earliest raw row timestamp in session", "elapsed logged session exposure at report"),
        "mixed_camera_speed_median": ("mixed_camera", "source_coordinate_units/second", "recomputed position displacement/time, window median", "rendered camera movement mixes navigation and user head motion"),
        "mixed_camera_speed_p90": ("mixed_camera", "source_coordinate_units/second", "recomputed position displacement/time, window 90th percentile", "rendered camera movement mixes navigation and user head motion"),
        "mixed_camera_acceleration_median": ("mixed_camera", "source_coordinate_units/second^2", "absolute adjacent scalar-speed difference/time, window median", "scalar-speed change, not vector acceleration"),
        "mixed_camera_acceleration_p90": ("mixed_camera", "source_coordinate_units/second^2", "absolute adjacent scalar-speed difference/time, window 90th percentile", "scalar-speed change, not vector acceleration"),
        "mixed_camera_rotation_rate_median": ("mixed_camera", "degrees/second", "screened camera_rotation_rate window median", "shortest relative orientation angle/time, mixed user/navigation motion"),
        "mixed_camera_rotation_rate_p90": ("mixed_camera", "degrees/second", "screened camera_rotation_rate window 90th percentile", "shortest relative orientation angle/time, mixed user/navigation motion"),
    }
    features = list(dict.fromkeys(feature for group in cfg["feature_sets"].values() for feature in group))
    rows = []
    for feature in features:
        if feature not in definitions:
            raise ValueError(f"No documented feature definition for {feature}")
        category, unit, source, meaning = definitions[feature]
        rows.append({"feature": feature, "category": category, "unit": unit, "source": source,
                     "meaning": meaning, "pure_design_status": "not_certified",
                     "source_url": "https://arxiv.org/html/2306.03381v1",
                     "source_feature_file": cfg["frame_features"],
                     "aggregation_definition": "self_defined_for_this_pilot; frozen configuration, not a paper-derived sickness threshold",
                     "feature_sets": ";".join(name for name, cols in cfg["feature_sets"].items() if feature in cols)})
    return rows


def prepare(config_path: Path, output_dir: Path) -> dict:
    cfg = json.loads(config_path.read_text(encoding="utf-8-sig"))
    for key in ("history_seconds", "fov_rounding_degrees", "minimum_history_span_seconds", "minimum_graphics_rows",
                "maximum_final_graphics_age_seconds", "minimum_time_span_fraction", "maximum_derivative_interval_seconds",
                "camera_jump_displacement_source_units", "rotation_screen_degrees_per_second"):
        if not isinstance(cfg[key], (int, float)) or cfg[key] <= 0:
            raise ValueError(f"Invalid positive configuration value: {key}")
    frame_path, rating_path = Path(cfg["frame_features"]), Path(cfg["rating_events"])
    frame = pd.read_csv(frame_path, encoding="utf-8-sig", low_memory=False)
    required = {"session_id", "game_name", "camera_jump_flag", *FRAME_NUMERIC}
    if required.difference(frame.columns):
        raise ValueError(f"Missing frame columns: {sorted(required.difference(frame.columns))}")
    for column in FRAME_NUMERIC:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame["timestamp"]).all() or not frame["timestamp"].between(1e11, 1e14, inclusive="left").all():
        raise ValueError("Frame timestamps must be finite epoch milliseconds; cannot safely align this input")
    if not frame["timestamp"].eq(frame["timestamp"].round()).all():
        raise ValueError("Frame timestamps contain fractional milliseconds")
    frame["timestamp"] = frame["timestamp"].astype("int64")
    frame["_source_order"] = np.arange(len(frame))
    mapping = frame[["session_id", "game_name"]].drop_duplicates()
    if mapping["session_id"].duplicated().any():
        raise ValueError("A frame session maps to multiple games")
    events_all = pd.read_csv(rating_path, encoding="utf-8-sig", low_memory=False,
                             dtype={"report_timestamp_raw": str, "participant_id": str, "session_id": str})
    events = events_all.loc[events_all["dataset_family"].eq("modern")].copy()
    if events["event_id"].duplicated().any():
        raise ValueError("Modern event IDs are not unique")
    report_ms = pd.to_numeric(events["report_timestamp_raw"], errors="coerce")
    events["report_time"] = report_ms
    events["report_time_unit"] = "epoch_milliseconds"
    events["report_time_epoch_seconds"] = report_ms / 1000.0
    feature_definitions = dictionary_rows(cfg)
    feature_columns = [row["feature"] for row in feature_definitions]

    raw_quality = {column: numeric_quality(frame[column]) for column in FRAME_NUMERIC}
    raw_quality["rotation_over_screen_rows"] = int(frame["camera_rotation_rate"].gt(cfg["rotation_screen_degrees_per_second"]).sum())
    raw_quality["rotation_over_screen_without_jump_rows"] = int((frame["camera_rotation_rate"].gt(cfg["rotation_screen_degrees_per_second"]) & ~truthy(frame["camera_jump_flag"])).sum())
    raw_quality["input_order_backward_timestamps"] = int(frame.groupby("session_id", sort=False)["timestamp"].diff().lt(0).sum())
    clean_sessions, clean_stats = {}, []
    for session_id, session in frame.groupby("session_id", sort=False):
        clean_sessions[str(session_id)], stats = clean_session(session, cfg)
        clean_stats.append(stats)

    training, excluded, quality = [], [], []
    for _, event in events.iterrows():
        metadata = event.to_dict()
        reasons = []
        q = {key: metadata[key] for key in ("event_id", "session_id", "game_name", "participant_id", "report_time", "score_value")}
        q.update(window_start_time=np.nan, window_end_time=np.nan, raw_window_rows=0, scene_window_rows=0,
                 first_scene_time=np.nan, last_scene_time=np.nan, history_span_seconds=np.nan,
                 expected_history_seconds=np.nan, history_span_fraction=np.nan, final_graphics_age_seconds=np.nan,
                 exposure_elapsed_seconds=np.nan)
        session = clean_sessions.get(str(event["session_id"]))
        if session is None:
            reasons.append("no_frame_feature_cache")
        elif not np.isfinite(event["report_time"]) or event["timestamp_quality"] != "integer_epoch_milliseconds":
            reasons.append("invalid_report_timestamp")
        elif not truthy(pd.Series([event["score_numeric_valid"]])).iloc[0] or not np.isfinite(event["score_value"]):
            reasons.append("invalid_score")
        elif str(session["game_name"].iloc[0]) != str(event["game_name"]):
            reasons.append("game_session_mismatch")
        else:
            report = int(event["report_time"])
            elapsed = (report - int(session["timestamp"].iloc[0])) / 1000.0
            start = report - cfg["history_seconds"] * 1000.0
            times = session["timestamp"].to_numpy()
            lo, hi = np.searchsorted(times, [start, report], side="left")
            raw_window = session.iloc[lo:hi]
            scene = raw_window.loc[raw_window["_scene"]]
            expected = min(cfg["history_seconds"], elapsed)
            span = (float(scene["timestamp"].iloc[-1]) - float(scene["timestamp"].iloc[0])) / 1000.0 if len(scene) else 0.0
            final_age = (report - float(scene["timestamp"].iloc[-1])) / 1000.0 if len(scene) else np.nan
            fraction = span / expected if expected > 0 else np.nan
            q.update(window_start_time=start, window_end_time=report, raw_window_rows=len(raw_window),
                     scene_window_rows=len(scene), first_scene_time=float(scene["timestamp"].iloc[0]) if len(scene) else np.nan,
                     last_scene_time=float(scene["timestamp"].iloc[-1]) if len(scene) else np.nan, history_span_seconds=span,
                     expected_history_seconds=expected, history_span_fraction=fraction,
                     final_graphics_age_seconds=final_age, exposure_elapsed_seconds=elapsed)
            if elapsed <= 0:
                reasons.append("nonpositive_exposure")
            if span < cfg["minimum_history_span_seconds"]:
                reasons.append("history_span_below_minimum")
            if len(scene) < cfg["minimum_graphics_rows"]:
                reasons.append("scene_rows_below_minimum")
            if not np.isfinite(final_age) or final_age > cfg["maximum_final_graphics_age_seconds"]:
                reasons.append("last_scene_too_old_or_missing")
            if not np.isfinite(fraction) or fraction < cfg["minimum_time_span_fraction"]:
                reasons.append("history_span_fraction_below_minimum")
            if not reasons:
                features = {
                    "fov_horizontal_equivalent_mean": float(scene["_fov_h"].mean()),
                    "fov_vertical_equivalent_mean": float(scene["_fov_v"].mean()),
                    "exposure_elapsed_seconds": elapsed,
                }
                for source, prefix in (("_object_count", "visible_object_count"), ("_object_speed", "object_speed"),
                                       ("_speed", "mixed_camera_speed"), ("_acceleration", "mixed_camera_acceleration"),
                                       ("_rotation_rate", "mixed_camera_rotation_rate")):
                    features[prefix + "_median"], features[prefix + "_p90"] = summarize_motion(scene[source])
                    q[prefix + "_valid_rows"] = int(scene[source].notna().sum())
                if set(features) != set(feature_columns):
                    raise ValueError("Constructed feature set does not match the frozen configuration")
                training.append({**metadata, **{name: features[name] for name in feature_columns}})
        q["included"] = not reasons
        q["exclusion_reasons"] = ";".join(reasons)
        quality.append(q)
        if reasons:
            excluded.append({**metadata, "exclusion_reasons": ";".join(reasons)})

    output_dir.mkdir(parents=True, exist_ok=True)
    train_df = pd.DataFrame(training, columns=[*events.columns, *feature_columns])
    excluded_df = pd.DataFrame(excluded, columns=[*events.columns, "exclusion_reasons"])
    quality_df = pd.DataFrame(quality)
    files = {
        "training_matrix.csv": train_df, "excluded_events.csv": excluded_df,
        "window_quality.csv": quality_df, "feature_dictionary.csv": pd.DataFrame(feature_definitions),
    }
    for name, df in files.items():
        # Preserve millisecond epoch audit timestamps and derived feature values.
        df.to_csv(output_dir / name, index=False, encoding="utf-8-sig", float_format="%.17g")
    reason_counts = Counter(reason for item in excluded for reason in item["exclusion_reasons"].split(";"))
    first_reason_counts = Counter(item["exclusion_reasons"].split(";")[0] for item in excluded)
    fov = {}
    for column in ("fov_horizontal_equivalent_mean", "fov_vertical_equivalent_mean"):
        values = train_df[column].dropna()
        fov[column] = {"finite_samples": int(len(values)), "rounded_unique_values": [float(v) for v in np.unique(np.round(values, 9))],
                       "constant_after_rounding": bool(len(np.unique(np.round(values, 9))) <= 1)}
    manifest = {
        "status": "success", "preparation_only": True, "configuration": cfg,
        "input_hashes": {str(config_path): sha256(config_path), str(frame_path): sha256(frame_path), str(rating_path): sha256(rating_path),
                         str(Path(__file__).resolve()): sha256(Path(__file__))},
        "input_frame_rows": int(len(frame)), "input_frame_sessions": int(frame["session_id"].nunique()),
        "input_frame_games": int(frame["game_name"].nunique()), "all_score_events": int(len(events_all)),
        "modern_score_events": int(len(events)), "ignored_nonmodern_score_events": int(len(events_all) - len(events)),
        "raw_frame_quality": raw_quality, "per_session_cleaning": clean_stats,
        "timestamp_unit": "epoch_milliseconds; integer modern report and frame timestamps checked",
        "window_interval": "[report_time-history_seconds, report_time); all feature rows strictly precede score",
        "window_quality_rules": {key: cfg[key] for key in ("minimum_history_span_seconds", "minimum_graphics_rows", "maximum_final_graphics_age_seconds", "minimum_time_span_fraction")},
        "exclusion_counts_multiple_reasons_possible": dict(reason_counts),
        "exclusion_counts_first_reason_mutually_exclusive": dict(first_reason_counts),
        "excluded_event_count": int(len(excluded_df)), "training_sample_count": int(len(train_df)),
        "training_session_count": int(train_df["session_id"].nunique()), "training_game_count": int(train_df["game_name"].nunique()),
        "training_participant_count": int(train_df["participant_id"].nunique()),
        "training_events_by_game": {str(k): int(v) for k, v in train_df["game_name"].value_counts().sort_index().items()},
        "training_label_distribution": {str(k): int(v) for k, v in train_df["score_value"].value_counts().sort_index().items()},
        "modern_label_distribution": {str(k): int(v) for k, v in events["score_value"].value_counts().sort_index().items()},
        "feature_columns": feature_columns, "missing_feature_values": {name: int(train_df[name].isna().sum()) for name in feature_columns},
        "fov_constant_check": fov, "feature_imputation": "none; NaN preserved for training-fold preprocessing",
        "limitations": cfg["limitations"],
        "output_hashes": {name: sha256(output_dir / name) for name in files},
    }
    (output_dir / "preparation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/score_regression_pilot_2026-10-10.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("features/score_regression_pilot_2026-10-10"))
    args = parser.parse_args()
    try:
        manifest = prepare(args.config, args.output_dir)
    except Exception as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise
    print(json.dumps({key: manifest[key] for key in ("status", "modern_score_events", "input_frame_rows", "input_frame_sessions", "training_sample_count", "training_session_count", "training_game_count", "training_participant_count", "excluded_event_count", "exclusion_counts_first_reason_mutually_exclusive", "training_label_distribution", "fov_constant_check")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
