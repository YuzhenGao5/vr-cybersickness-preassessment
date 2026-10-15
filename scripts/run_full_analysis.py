#!/usr/bin/env python3
"""Run the scene/video feature tools and literature-rule report pipeline.

Source groups are joined only when identifiers support the join. Otherwise
they are kept as separate rows in the union feature matrix and the mismatch is
recorded; synthetic fixtures are never attached to unrelated real sessions.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import subprocess
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


LOG = logging.getLogger("run_full_analysis")
UNKNOWN = "unknown"
RULE_SUPPORT_COLUMNS = [
    "virtual_translation_speed_m_s", "oscillation_frequency_hz", "camera_speed_m_s",
    "camera_acceleration_m_s2", "rotation_axes", "camera_fov_deg",
    "has_reference_frame", "controllability", "expected_duration_min",
    "particle_emission_rate", "dynamic_light_count", "peripheral_geometry_complexity",
    "depth_change_rate",
]
UNITY_FIELDS = {
    "camera_fov": "unity_scene_camera_fov",
    "camera_near_clip": "unity_scene_camera_near_clip",
    "camera_far_clip": "unity_scene_camera_far_clip",
    "particle_system_count": "unity_scene_particle_system_count",
    "particle_emission_rate": "unity_scene_particle_emission_rate",
    "dynamic_light_count": "unity_scene_dynamic_light_count",
    "light_type_distribution": "unity_scene_light_type_distribution",
    "object_count": "unity_scene_object_count",
    "transform_depth_max": "unity_scene_transform_depth_max",
    "triangle_count": "unity_scene_triangle_count",
    "mesh_filter_count": "unity_scene_mesh_filter_count",
    "mesh_renderer_count": "unity_scene_mesh_renderer_count",
    "has_reference_frame": "unity_scene_has_reference_frame",
    "has_horizon": "unity_scene_has_horizon",
    "has_virtual_nose": "unity_scene_has_virtual_nose",
    "controllability": "unity_scene_controllability",
    "expected_duration": "unity_scene_expected_duration",
}


class PipelineError(RuntimeError):
    pass


def absolute_path(raw: str | Path, root: Path) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_step(label: str, command: list[str], cwd: Path) -> None:
    printable = subprocess.list2cmdline(command)
    LOG.info("STEP START: %s", label)
    LOG.info("COMMAND: %s", printable)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    result = subprocess.run(
        command, cwd=cwd, env=env, text=True, encoding="utf-8", errors="replace",
        capture_output=True, check=False,
    )
    if result.stdout:
        LOG.info("STDOUT [%s]:\n%s", label, result.stdout.rstrip())
    if result.stderr:
        (LOG.error if result.returncode else LOG.warning)("STDERR [%s]:\n%s", label, result.stderr.rstrip())
    if result.returncode != 0:
        LOG.error("STEP FAILED: %s exit_code=%d", label, result.returncode)
        raise PipelineError(f"{label} failed with exit code {result.returncode}")
    LOG.info("STEP SUCCEEDED: %s", label)


def count_sessions(samples_root: Path) -> list[Path]:
    if not samples_root.is_dir():
        return []
    return sorted(
        [item for item in samples_root.iterdir() if item.is_dir() and (item / "data_file.csv").is_file()],
        key=lambda item: item.name.casefold(),
    )


def prepare_video_samples_root(video_arg: Path, work_dir: Path) -> tuple[Path, Path, Path | None]:
    """Return extractor root, selected session dir, and an optional temp scope link."""
    if not video_arg.exists():
        raise FileNotFoundError(f"--video-dir does not exist: {video_arg}")
    if (video_arg / "data_file.csv").is_file():
        session_dir = video_arg
    elif (video_arg.parent / "data_file.csv").is_file():
        # The caller passed the session's video/ subdirectory.
        session_dir = video_arg.parent
    else:
        candidates = count_sessions(video_arg)
        if len(candidates) == 1:
            return video_arg, candidates[0], None
        raise ValueError(
            f"--video-dir must be a session directory (with data_file.csv), its video/ directory, "
            f"or a root containing exactly one session; found {len(candidates)} session(s): {video_arg}"
        )

    parent = session_dir.parent
    siblings = count_sessions(parent)
    if len(siblings) == 1 and siblings[0].resolve() == session_dir.resolve():
        return parent, session_dir, None

    # The extractor accepts a parent containing session folders. Scope it to the
    # requested session with a symlink, avoiding copies or edits to source data.
    scope = work_dir / f"session_scope_{uuid.uuid4().hex}"
    scope.mkdir(parents=True, exist_ok=False)
    link = scope / session_dir.name
    try:
        link.symlink_to(session_dir.resolve(), target_is_directory=True)
    except OSError:
        try:
            scope.rmdir()
        except OSError:
            pass
        raise
    return scope, session_dir, scope


def synthetic_marker(scene_path: Path) -> bool:
    start = scene_path.read_text(encoding="utf-8-sig", errors="replace")[:3000]
    return "SYNTHETIC PARSER FIXTURE ONLY" in start


def rule_metric_names(rule_path: Path) -> list[str]:
    with rule_path.open("r", encoding="utf-8-sig") as handle:
        rules_doc = yaml.safe_load(handle) or {}
    names = list(RULE_SUPPORT_COLUMNS)
    for rule in rules_doc.get("risk_factors", []):
        threshold = rule.get("threshold") or {}
        if threshold.get("condition") == "and":
            names.extend(str(part.get("metric", "")) for part in threshold.get("conditions", []))
        elif rule.get("metric"):
            names.append(str(rule["metric"]))
    return list(dict.fromkeys(name for name in names if name))


def clean_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return UNKNOWN
    return value


def make_scene_record(
    scene: dict[str, str], scene_path: Path, synthetic: bool,
    video_session_ids: set[str],
) -> tuple[dict[str, Any], bool, str]:
    name = scene.get("scene_name", scene_path.stem)
    can_join = bool(name and name in video_session_ids and not synthetic)
    if not video_session_ids:
        alignment = "scene-only input; no video rows to align"
    elif synthetic:
        alignment = "unmatched synthetic fixture; not assigned to any real video session"
    elif can_join:
        alignment = "exact scene_name/session_id match"
    else:
        alignment = "unmatched: scene_name does not equal any video session_id"

    record: dict[str, Any] = {key: clean_value(value) for key, value in scene.items()}
    for source_key, value in scene.items():
        record[f"unity_scene_{source_key}"] = clean_value(value)
    for source_key, merged_key in UNITY_FIELDS.items():
        record[merged_key] = clean_value(scene.get(source_key, UNKNOWN))
    record.update({
        "analysis_record_id": f"unity:{name}",
        "record_type": "unity_scene",
        "session_id": name if can_join else f"unity_scene:{name}",
        "scene_id": name,
        "window_index": "",
        "source_dataset": "synthetic Unity fixture" if synthetic else "Unity scene",
        "source_scene_file": str(scene_path),
        "source_video_dir": "",
        "scene_video_alignment_status": alignment,
        "scene_video_alignment_reason": alignment,
        "metadata_join_status": "not supplied",
        "hmd_metadata_row_count": 0,
        "hmd_metadata_alignment_level": UNKNOWN,
        "camera_fov_deg": clean_value(scene.get("camera_fov", UNKNOWN)),
        "has_reference_frame": clean_value(scene.get("has_reference_frame", UNKNOWN)),
        "controllability": clean_value(scene.get("controllability", UNKNOWN)),
        "expected_duration_min": clean_value(scene.get("expected_duration", UNKNOWN)),
        "particle_emission_rate": clean_value(scene.get("particle_emission_rate", UNKNOWN)),
        "dynamic_light_count": clean_value(scene.get("dynamic_light_count", UNKNOWN)),
    })
    return record, can_join, alignment


def find_column(headers: list[str], candidates: tuple[str, ...]) -> str | None:
    by_lower = {header.strip().casefold(): header for header in headers}
    for candidate in candidates:
        if candidate in by_lower:
            return by_lower[candidate]
    return None


def is_outcome_column(name: str) -> bool:
    key = re.sub(r"[^a-z0-9]+", "", name.casefold())
    exact = {"np", "op", "dp", "tsp", "msrating", "overallfms", "firstfms", "secondfms"}
    return key in exact or any(token in key for token in ("fms", "ssq", "sickness", "cybersickness", "symptom", "label"))


def mean_or_unique(values: list[str]) -> str:
    numeric: list[float] = []
    for value in values:
        try:
            number = float(value)
            if math.isfinite(number):
                numeric.append(number)
        except (TypeError, ValueError):
            pass
    if numeric:
        return str(sum(numeric) / len(numeric))
    unique = list(dict.fromkeys(value for value in values if value not in (None, "")))
    if not unique:
        return ""
    return unique[0] if len(unique) == 1 else f"mixed ({len(unique)} values)"


def merge_hmd_metadata(
    path: Path | None, rows: list[dict[str, Any]], video_headers: list[str], scene_name: str | None,
) -> tuple[list[str], str]:
    if path is None:
        for row in rows:
            row["metadata_join_status"] = "not supplied"
            row["hmd_metadata_row_count"] = 0
            row["hmd_metadata_alignment_level"] = UNKNOWN
        return [], "No HMD metadata CSV supplied."

    headers, metadata_rows = read_csv(path)
    id_col = find_column(headers, ("session_id", "session", "scene_id", "scene", "session_name"))
    window_col = find_column(headers, ("window_index", "window_id"))
    filtered = [header for header in headers if not is_outcome_column(header)]
    excluded = [header for header in headers if is_outcome_column(header)]
    data_cols = [header for header in filtered if header != id_col and header != window_col]
    prefixed = {header: f"hmd_{header}" for header in filtered}

    if not id_col:
        for row in rows:
            row["metadata_join_status"] = "unmatched: metadata CSV has no session_id/scene_id key"
            row["hmd_metadata_row_count"] = 0
            row["hmd_metadata_alignment_level"] = UNKNOWN
        note = f"Read {len(metadata_rows)} HMD metadata rows; no recognized session/scene key, so no merge was performed."
        if excluded:
            note += f" Excluded outcome/label columns: {', '.join(excluded)}."
        return list(prefixed.values()), note

    groups: dict[tuple[str, str | None], list[dict[str, str]]] = defaultdict(list)
    for item in metadata_rows:
        identity = (item.get(id_col) or "").strip()
        if not identity:
            continue
        window = (item.get(window_col) or "").strip() if window_col else None
        groups[(identity, window or None)].append(item)

    joined = 0
    for row in rows:
        candidates = [str(row.get("session_id", "")), str(row.get("scene_id", ""))]
        found: list[dict[str, str]] = []
        level = UNKNOWN
        for identity in candidates:
            if not identity:
                continue
            if window_col and row.get("window_index") not in (None, ""):
                found = groups.get((identity, str(row["window_index"])), [])
                if found:
                    level = "window_index"
                    break
            if not window_col:
                found = groups.get((identity, None), [])
                if found:
                    level = "session/scene aggregate"
                    break
        if found:
            joined += 1
            row["metadata_join_status"] = "matched"
            row["hmd_metadata_row_count"] = len(found)
            row["hmd_metadata_alignment_level"] = level
            for original in filtered:
                values = [(item.get(original) or "").strip() for item in found]
                row[prefixed[original]] = mean_or_unique(values)
        else:
            row["metadata_join_status"] = "unmatched: no ID/window key match"
            row["hmd_metadata_row_count"] = 0
            row["hmd_metadata_alignment_level"] = UNKNOWN

    note = (
        f"Read {len(metadata_rows)} HMD metadata rows; {joined}/{len(rows)} analysis rows matched by "
        f"{id_col}" + (f" + {window_col}." if window_col else " at session/scene level (no window-level join).")
    )
    if excluded:
        note += f" Excluded outcome/label columns to avoid label leakage: {', '.join(excluded)}."
    return list(prefixed.values()), note


def build_merged_matrix(
    scene_csv: Path | None, video_csv: Path | None, metadata_csv: Path | None,
    scene_path: Path | None, video_dir: Path | None, rules_path: Path, matrix_path: Path,
) -> tuple[list[dict[str, Any]], str, str]:
    scene_row: dict[str, str] | None = None
    if scene_csv:
        scene_headers, scene_rows = read_csv(scene_csv)
        if len(scene_rows) != 1:
            raise ValueError(f"Expected one row from Unity scene parser; found {len(scene_rows)}")
        scene_row = scene_rows[0]
    video_headers: list[str] = []
    video_rows: list[dict[str, str]] = []
    if video_csv:
        video_headers, video_rows = read_csv(video_csv)
        if not video_rows:
            raise ValueError(f"Video extractor produced no feature windows: {video_csv}")
    video_sessions = {row.get("session_id", "").strip() for row in video_rows if row.get("session_id")}
    synthetic = bool(scene_path and synthetic_marker(scene_path))

    rows: list[dict[str, Any]] = []
    alignment_notes: list[str] = []
    scene_can_join = False
    scene_data: dict[str, Any] | None = None
    scene_alignment = "not supplied"
    scene_name = scene_row.get("scene_name", scene_path.stem) if scene_row and scene_path else None
    if scene_row and scene_path:
        scene_data, scene_can_join, scene_alignment = make_scene_record(scene_row, scene_path, synthetic, video_sessions)
        if scene_can_join:
            alignment_notes.append(f"Unity scene `{scene_name}` exact ID match to video session; static metadata will be joined to its windows.")
        elif video_sessions:
            alignment_notes.append(
                f"Unity scene `{scene_name}` and video session(s) {', '.join(sorted(video_sessions))} do not have a valid join. "
                f"The Unity scene is {'synthetic and intentionally not assigned to a real video session' if synthetic else 'kept as a separate scene record'}; "
                "no scene-to-video feature alignment is claimed."
            )
        else:
            alignment_notes.append(f"Unity scene `{scene_name}` is a standalone record; no video source was supplied.")

    for original in video_rows:
        row: dict[str, Any] = {key: clean_value(value) for key, value in original.items()}
        sid = str(original.get("session_id", "unknown_session"))
        row.update({
            "analysis_record_id": f"video:{sid}:{original.get('window_index', '')}",
            "record_type": "video_window",
            "scene_id": sid,
            "source_dataset": "VR.net video frames",
            "source_scene_file": str(scene_path) if scene_path else "",
            "source_video_dir": str(video_dir) if video_dir else "",
            "metadata_join_status": "not processed",
            "hmd_metadata_row_count": 0,
            "hmd_metadata_alignment_level": UNKNOWN,
        })
        if scene_can_join and scene_data:
            row["scene_video_alignment_status"] = "exact scene_id/session_id match"
            row["scene_video_alignment_reason"] = "scene_name equals video session_id"
            for field, value in scene_data.items():
                if field.startswith("unity_scene_"):
                    row[field] = value
            # These are scene-level attributes joined only after an exact, non-synthetic ID match.
            for metric in ("camera_fov_deg", "has_reference_frame", "controllability", "expected_duration_min",
                           "particle_emission_rate", "dynamic_light_count"):
                scene_value = scene_data.get(metric, UNKNOWN)
                if scene_value not in ("", None, UNKNOWN, "nan"):
                    row[metric] = scene_value
        elif scene_row:
            row["scene_video_alignment_status"] = "unmatched sources; kept separate"
            row["scene_video_alignment_reason"] = scene_alignment
        else:
            row["scene_video_alignment_status"] = "video-only; no Unity scene supplied"
            row["scene_video_alignment_reason"] = "no scene input"
        rows.append(row)

    if scene_data and not scene_can_join:
        rows.append(scene_data)

    if not rows and metadata_csv:
        meta_headers, meta_rows = read_csv(metadata_csv)
        id_col = find_column(meta_headers, ("session_id", "session", "scene_id", "scene", "session_name"))
        kept = [field for field in meta_headers if not is_outcome_column(field)]
        excluded = [field for field in meta_headers if is_outcome_column(field)]
        hmd_columns = [f"hmd_{field}" for field in kept]
        for index, item in enumerate(meta_rows):
            identity = (item.get(id_col) or "").strip() if id_col else ""
            record = {
                "analysis_record_id": f"hmd:{identity or 'unmatched'}:{index}",
                "record_type": "hmd_metadata_only",
                "session_id": identity or f"hmd_metadata:{index}",
                "scene_id": identity,
                "source_dataset": "HMD metadata CSV",
                "source_scene_file": "",
                "source_video_dir": "",
                "scene_video_alignment_status": "no scene/video source supplied",
                "scene_video_alignment_reason": "HMD metadata only",
                "metadata_join_status": "metadata-only source row",
                "hmd_metadata_row_count": 1,
                "hmd_metadata_alignment_level": "source row",
            }
            for field in kept:
                record[f"hmd_{field}"] = (item.get(field) or "").strip()
            rows.append(record)
        hmd_columns = [f"hmd_{field}" for field in kept]
        metadata_note = f"Read {len(meta_rows)} HMD metadata rows; no scene/video source was supplied, so rows remain standalone."
        if excluded:
            metadata_note += f" Excluded outcome/label columns: {', '.join(excluded)}."
        alignment_notes.append(metadata_note)
    elif metadata_csv:
        hmd_columns, metadata_note = merge_hmd_metadata(metadata_csv, rows, video_headers, scene_name)
    else:
        hmd_columns, metadata_note = merge_hmd_metadata(None, rows, video_headers, scene_name)
    if rows and metadata_note not in alignment_notes:
        alignment_notes.append(metadata_note)

    if not rows:
        raise ValueError("No scene, video, or metadata records are available to merge")

    # Create every rule metric as an explicit value column so missing evidence
    # remains unknown instead of silently becoming a no-trigger.
    required_columns = rule_metric_names(rules_path)
    for row in rows:
        for metric in required_columns:
            if metric not in row or row[metric] in (None, ""):
                row[metric] = UNKNOWN
        if "session_id" not in row or row["session_id"] in (None, ""):
            row["session_id"] = row.get("scene_id") or "unknown_session"
        if "record_type" not in row:
            row["record_type"] = "metadata_only"
        if "scene_video_alignment_status" not in row:
            row["scene_video_alignment_status"] = "unknown"
        if "scene_video_alignment_reason" not in row:
            row["scene_video_alignment_reason"] = "unknown"

    # Stable, readable columns first; preserve all extractor fields and unioned metadata fields.
    leading = [
        "analysis_record_id", "record_type", "session_id", "scene_id", "window_index",
        "window_start_sec", "window_end_sec", "window_center_sec", "source_dataset",
        "source_scene_file", "source_video_dir", "scene_video_alignment_status",
        "scene_video_alignment_reason", "metadata_join_status", "hmd_metadata_row_count",
        "hmd_metadata_alignment_level",
    ]
    video_metric_cols = [column for column in video_headers if column != "session_id"]
    other_cols = [key for row in rows for key in row if key not in leading and key not in video_metric_cols]
    all_cols = list(dict.fromkeys([*leading, *video_metric_cols, *UNITY_FIELDS.values(), *hmd_columns, *required_columns, *other_cols]))
    for row in rows:
        for column in all_cols:
            row.setdefault(column, "")
    write_csv(matrix_path, rows, all_cols)
    return rows, "\n".join(f"- {note}" for note in alignment_notes), metadata_note


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", help="Optional .unity/.prefab scene file")
    parser.add_argument("--video-dir", help="Optional video session directory or its video/ subdirectory")
    parser.add_argument("--metadata-csv", help="Optional HMD metadata CSV")
    parser.add_argument("--rules", default="configs/risk_factor_rules.yaml")
    parser.add_argument("--output", default="results/full_analysis_report.md")
    parser.add_argument("--matrix-output", default="features/merged_feature_matrix.csv")
    parser.add_argument("--detections-output", default="features/full_analysis_rule_detections.csv")
    parser.add_argument("--log", default="results/full_analysis_log.txt")
    args = parser.parse_args()
    if not any((args.scene, args.video_dir, args.metadata_csv)):
        parser.error("provide at least one of --scene, --video-dir, or --metadata-csv")

    root = Path(__file__).resolve().parents[1]
    log_path = absolute_path(args.log, root)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    work_dir = absolute_path("features/full_analysis_components", root)
    work_dir.mkdir(parents=True, exist_ok=True)
    scene_path = absolute_path(args.scene, root) if args.scene else None
    video_arg = absolute_path(args.video_dir, root) if args.video_dir else None
    metadata_path = absolute_path(args.metadata_csv, root) if args.metadata_csv else None
    rules_path = absolute_path(args.rules, root)
    report_path = absolute_path(args.output, root)
    matrix_path = absolute_path(args.matrix_output, root)
    detections_path = absolute_path(args.detections_output, root)

    scene_csv: Path | None = None
    video_csv: Path | None = None
    video_stats: Path | None = None
    scope_to_remove: Path | None = None
    try:
        for path, label in ((scene_path, "scene"), (video_arg, "video directory"), (metadata_path, "metadata CSV"), (rules_path, "rules file")):
            if path is not None and not path.exists():
                raise FileNotFoundError(f"{label} does not exist: {path}")
        if scene_path:
            scene_csv = work_dir / "unity_scene_features.csv"
            parser_log = absolute_path("results/full_analysis_unity_parser.log", root)
            run_step("Unity scene parser", [
                sys.executable, str(root / "scripts/parse_unity_scene.py"), "--scene", str(scene_path),
                "--output", str(scene_csv), "--log", str(parser_log),
            ], root)
            headers, parsed = read_csv(scene_csv)
            if len(parsed) != 1:
                raise PipelineError(f"Unity parser output expected 1 row; found {len(parsed)}")
            LOG.info("Unity parser produced columns=%s", headers)

        video_scope: Path | None = None
        selected_session: Path | None = None
        if video_arg:
            video_scope, selected_session, scope_to_remove = prepare_video_samples_root(video_arg, work_dir)
            video_csv = work_dir / "video_window_features.csv"
            video_stats = work_dir / "video_feature_statistics.csv"
            extractor_log = absolute_path("results/full_analysis_video_extractor.log", root)
            run_step("Video feature extractor", [
                sys.executable, str(root / "scripts/extract_full_features.py"),
                "--samples-dir", str(video_scope), "--output", str(video_csv),
                "--stats", str(video_stats), "--window-sec", "30", "--max-dimension", "320",
                "--log", str(extractor_log),
            ], root)
            video_headers, extracted = read_csv(video_csv)
            if not extracted:
                raise PipelineError("Video extractor produced no windows")
            session_ids = {row.get("session_id", "") for row in extracted}
            if session_ids != {selected_session.name}:
                raise PipelineError(
                    f"Video extractor output session IDs {sorted(session_ids)} do not match selected session {selected_session.name}"
                )
            LOG.info("Video extractor produced windows=%d columns=%d session=%s", len(extracted), len(video_headers), selected_session.name)
            if scope_to_remove:
                link = scope_to_remove / selected_session.name
                if link.is_symlink():
                    link.unlink()
                scope_to_remove.rmdir()
                scope_to_remove = None

        if metadata_path:
            # Validate/read immediately, before later pipeline stages.
            meta_headers, meta_rows = read_csv(metadata_path)
            LOG.info("HMD metadata read rows=%d columns=%s", len(meta_rows), meta_headers)

        merged_rows, alignment_note, metadata_note = build_merged_matrix(
            scene_csv, video_csv, metadata_path, scene_path, video_arg, rules_path, matrix_path,
        )
        LOG.info("Feature merge succeeded: rows=%d columns=%d matrix=%s", len(merged_rows), len(read_csv(matrix_path)[0]), matrix_path)
        LOG.info("Alignment notes:\n%s", alignment_note)

        missing_report = absolute_path("results/full_analysis_missing_metrics.md", root)
        run_step("Rule detector", [
            sys.executable, str(root / "scripts/detect_risk_factors.py"),
            "--features", str(matrix_path), "--rules", str(rules_path),
            "--output", str(detections_path), "--missing-report", str(missing_report),
        ], root)
        detection_headers, detection_rows = read_csv(detections_path)
        if not detection_rows:
            raise PipelineError("Rule detector produced no rows")
        LOG.info("Rule detector produced rows=%d columns=%s", len(detection_rows), detection_headers)

        scene_is_synthetic = bool(scene_path and synthetic_marker(scene_path))
        generator_cmd = [
            sys.executable, str(root / "scripts/generate_scene_report.py"),
            "--detections", str(detections_path), "--features", str(matrix_path),
            "--rules", str(rules_path), "--output", str(report_path),
            "--alignment-note", alignment_note, "--metadata-note", metadata_note,
        ]
        if scene_path:
            generator_cmd.extend(["--scene", str(scene_path)])
        if video_arg:
            generator_cmd.extend(["--video-dir", str(video_arg)])
        if metadata_path:
            generator_cmd.extend(["--metadata-csv", str(metadata_path)])
        if scene_is_synthetic:
            generator_cmd.append("--synthetic-scene")
        run_step("Scene report generator", generator_cmd, root)
        if not report_path.is_file() or report_path.stat().st_size == 0:
            raise PipelineError(f"Report generator returned success but report is missing/empty: {report_path}")
        LOG.info("PIPELINE COMPLETE: matrix=%s report=%s", matrix_path, report_path)
        print(f"matrix={matrix_path}")
        print(f"report={report_path}")
        return 0
    except Exception as exc:
        LOG.exception("PIPELINE STOPPED: %s: %s", type(exc).__name__, exc)
        print(f"PIPELINE STOPPED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if scope_to_remove and scope_to_remove.exists():
            link = next(iter(scope_to_remove.iterdir()), None)
            if link is not None and link.is_symlink():
                link.unlink()
            try:
                scope_to_remove.rmdir()
            except OSError:
                LOG.warning("Could not remove temporary input scope directory: %s", scope_to_remove)


if __name__ == "__main__":
    raise SystemExit(main())
