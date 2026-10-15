#!/usr/bin/env python3
"""Parse VR.net scene matrices into auditable per-frame graphics features.

The source CSV stores Unity matrices as Python-literal strings. Matrix values
are treated as row-major after reshaping the serialized 16-value tuple; this
places translation in row 3, columns 0:3, matching the observed files.
"""

from __future__ import annotations

import argparse
import ast
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


MATRIX_COLUMNS = ("p_matrix", "v_matrix", "m_matrix")
SOURCE_COLUMNS = (
    "frame",
    "timestamp",
    "camera_name",
    "p_matrix",
    "v_matrix",
    "object_name",
    "bounds",
    "m_matrix",
)
OUTPUT_COLUMNS = (
    "session_id",
    "frame_id",
    "timestamp",
    "fov_horizontal",
    "fov_vertical",
    "camera_pos_x",
    "camera_pos_y",
    "camera_pos_z",
    "camera_yaw",
    "camera_pitch",
    "camera_roll",
    "camera_speed",
    "camera_rotation_rate",
    "camera_acceleration",
    "object_count",
    "mean_object_speed",
    "max_object_speed",
)
Z_FLIP = np.diag([1.0, 1.0, -1.0])
MATRIX_TRANSLATION = str.maketrans({"[": " ", "]": " ", "(": " ", ")": " ", ",": " "})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/vrnet/sampled_csv/Pirate_Life"),
        help="Directory containing <session>/data_file.csv files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("features/vrnet_matrix_features.csv"),
        help="Combined output CSV",
    )
    parser.add_argument(
        "--per-session-dir",
        type=Path,
        default=Path("features/vrnet_matrix_features_by_session"),
        help="Directory for one feature CSV per session",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("results/matrix_parsing_log.txt"),
        help="Parsing log",
    )
    parser.add_argument(
        "--format-report",
        type=Path,
        default=Path("results/matrix_format_check.md"),
        help="Matrix-format inspection report",
    )
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument(
        "--session",
        action="append",
        default=[],
        help="Optional session directory name; may be repeated",
    )
    return parser.parse_args()


def literal_list(value: Any) -> list[Any]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if not isinstance(value, str) or not value.strip():
        return []
    parsed = ast.literal_eval(value)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, tuple):
        # A bare 16-value tuple is one matrix; other tuples are one value.
        return [parsed]
    return [parsed]


def matrix4(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size != 16:
        raise ValueError(f"expected 16 matrix values, got {array.size}")
    matrix = array.reshape(4, 4)
    if not np.isfinite(matrix).all():
        raise ValueError("matrix contains NaN or infinity")
    return matrix


def numeric_matrix_list(value: Any) -> np.ndarray:
    """Parse a serialized list of 16-value numeric tuples without Python AST.

    Scene rows can contain hundreds of matrices and occupy many megabytes.
    ``numpy.fromstring`` avoids constructing one Python float object per value.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.empty((0, 4, 4), dtype=np.float64)
    if not isinstance(value, str) or not value.strip():
        return np.empty((0, 4, 4), dtype=np.float64)
    flat = np.fromstring(value.translate(MATRIX_TRANSLATION), sep=" ", dtype=np.float64)
    if flat.size % 16 != 0:
        raise ValueError(f"matrix field has {flat.size} numeric values, not a multiple of 16")
    matrices = flat.reshape(-1, 4, 4)
    if not np.isfinite(matrices).all():
        raise ValueError("matrix field contains NaN or infinity")
    return matrices


def choose_camera(names: list[Any]) -> int | None:
    normalized = [str(name).lower() for name in names]
    for token in ("main", "center"):
        for index, name in enumerate(normalized):
            if token in name:
                return index
    return 0 if names else None


def projection_type(matrix: np.ndarray) -> str:
    # Matrices are the transpose of the conventional column-vector layout.
    if abs(matrix[3, 3]) < 1e-4 and abs(matrix[2, 3]) > 0.5:
        return "perspective"
    if abs(matrix[3, 3] - 1.0) < 1e-4 and abs(matrix[2, 3]) < 1e-4:
        return "orthographic"
    return "unknown"


def extract_fov(matrix: np.ndarray) -> tuple[float, float]:
    kind = projection_type(matrix)
    if kind != "perspective":
        raise ValueError(f"FOV formula is not applicable to {kind} projection")
    scale_x = abs(float(matrix[0, 0]))
    scale_y = abs(float(matrix[1, 1]))
    if scale_x <= 1e-12 or scale_y <= 1e-12:
        raise ValueError("projection scale is zero")
    horizontal = math.degrees(2.0 * math.atan(1.0 / scale_x))
    vertical = math.degrees(2.0 * math.atan(1.0 / scale_y))
    return horizontal, vertical


def camera_pose(view_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_world = np.linalg.inv(view_matrix)
    position = camera_world[3, 0:3].copy()
    # Unity view matrices use OpenGL camera space (forward=-Z). The serialized
    # row-vector layout gives an improper 3x3 matrix after inversion. Remove
    # that fixed Z reflection and transpose to SciPy's column-vector rotation.
    rotation_matrix = (Z_FLIP @ camera_world[0:3, 0:3]).T
    determinant = float(np.linalg.det(rotation_matrix))
    if not np.isfinite(determinant) or abs(determinant - 1.0) > 1e-3:
        raise ValueError(f"camera rotation determinant is {determinant:.6g}, expected +1")
    rotation = Rotation.from_matrix(rotation_matrix)
    # Extrinsic Y-X-Z: yaw about Y, pitch about X, roll about Z.
    yaw, pitch, roll = rotation.as_euler("yxz", degrees=True)
    return position, rotation_matrix, np.array([yaw, pitch, roll], dtype=float)


def timestamp_scale(first_timestamp: float) -> tuple[float, str]:
    magnitude = abs(float(first_timestamp))
    if magnitude >= 1e14:
        return 1e-6, "microseconds"
    if magnitude >= 1e11:
        return 1e-3, "milliseconds"
    return 1.0, "seconds"


def markdown_code(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "null"
    return str(value)


def write_format_report(csv_path: Path, report_path: Path) -> None:
    physical = pd.read_csv(csv_path, usecols=list(SOURCE_COLUMNS), nrows=3, encoding="utf-8-sig")
    scene_rows: list[pd.Series] = []
    for chunk in pd.read_csv(
        csv_path,
        usecols=list(SOURCE_COLUMNS),
        chunksize=64,
        encoding="utf-8-sig",
        low_memory=False,
    ):
        selected = chunk.loc[
            chunk[list(MATRIX_COLUMNS)].notna().any(axis=1)
        ]
        for _, row in selected.iterrows():
            scene_rows.append(row)
            if len(scene_rows) == 3:
                break
        if len(scene_rows) == 3:
            break

    first_scene = scene_rows[0]
    camera_names = literal_list(first_scene["camera_name"])
    object_names = literal_list(first_scene["object_name"])
    p_values = literal_list(first_scene["p_matrix"])
    v_values = literal_list(first_scene["v_matrix"])
    m_values = literal_list(first_scene["m_matrix"])
    bounds_values = literal_list(first_scene["bounds"])

    lines = [
        "# VR.net 矩阵格式检查",
        "",
        f"样例文件：`{csv_path.as_posix()}`",
        "",
        "## 文件前 3 个物理行的原始矩阵字段",
        "",
    ]
    for _, row in physical.iterrows():
        lines.extend(
            [
                f"### frame={row['frame']}, timestamp={row['timestamp']}",
                "",
                "`p_matrix`：",
                "```text",
                markdown_code(row["p_matrix"]),
                "```",
                "`v_matrix`：",
                "```text",
                markdown_code(row["v_matrix"]),
                "```",
                "`m_matrix`：",
                "```text",
                markdown_code(row["m_matrix"]),
                "```",
                "",
            ]
        )

    lines.extend(["## 前 3 个包含矩阵的行", ""])
    for row in scene_rows:
        lines.extend(
            [
                f"### frame={row['frame']}, timestamp={row['timestamp']}",
                "",
                f"- `camera_name`: `{markdown_code(row['camera_name'])}`",
                f"- `object_name`: `{markdown_code(row['object_name'])}`",
                f"- `bounds`: `{markdown_code(row['bounds'])}`",
                "",
                "`p_matrix`：",
                "```text",
                markdown_code(row["p_matrix"]),
                "```",
                "`v_matrix`：",
                "```text",
                markdown_code(row["v_matrix"]),
                "```",
                "`m_matrix`：",
                "```text",
                markdown_code(row["m_matrix"]),
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## 格式判断",
            "",
            "- 每个字段是一个 CSV 字符串，其中包含 Python literal。",
            "- 最外层使用方括号 `[]`，表示同一记录中的多个相机或多个可见物体。",
            "- 相机名和物体名是字符串列表。",
            "- 每个矩阵使用圆括号 `()`，内部有 16 个逗号分隔浮点数；按 4×4 reshape 后，平移位于 `[3, 0:3]`。",
            "- `bounds` 是六元组列表；样例中前三个值为中心，后三个值为 extents/尺寸相关量，但本任务不据此推断单位。",
            "- 一条 CSV 记录可以同时含多个相机和多个物体；不是“每个物体一条 CSV 行”。名称列表与矩阵列表按索引对应。",
            "- 主相机选择顺序：名称含 `Main`，其次含 `Center`，否则取第一个。",
            "",
            "## 首个矩阵行的结构核对",
            "",
            f"- camera 数：{len(camera_names)}；p_matrix 数：{len(p_values)}；v_matrix 数：{len(v_values)}",
            f"- object 数：{len(object_names)}；bounds 数：{len(bounds_values)}；m_matrix 数：{len(m_values)}",
            "- 格式可识别：**是**。后续解析可以继续。",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_session(csv_path: Path, output_path: Path, chunk_size: int) -> dict[str, Any]:
    header = pd.read_csv(csv_path, nrows=0, encoding="utf-8-sig").columns.tolist()
    missing = [column for column in SOURCE_COLUMNS if column not in header]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    stats: Counter[str] = Counter()
    start = time.perf_counter()
    session_id = csv_path.parent.name
    first_timestamp: float | None = None
    time_factor = 1.0
    time_unit = "unknown"
    previous_camera: tuple[float, np.ndarray, np.ndarray, float | None] | None = None
    previous_objects: dict[str, np.ndarray] = {}
    previous_object_timestamp: float | None = None
    outputs: list[dict[str, Any]] = []

    for chunk in pd.read_csv(
        csv_path,
        usecols=list(SOURCE_COLUMNS),
        chunksize=chunk_size,
        encoding="utf-8-sig",
        low_memory=False,
    ):
        for row in chunk.itertuples(index=False):
            stats["rows"] += 1
            timestamp_raw = float(row.timestamp) if pd.notna(row.timestamp) else math.nan
            if first_timestamp is None and math.isfinite(timestamp_raw):
                first_timestamp = timestamp_raw
                time_factor, time_unit = timestamp_scale(timestamp_raw)
            timestamp_sec = timestamp_raw * time_factor if math.isfinite(timestamp_raw) else math.nan
            result: dict[str, Any] = {column: math.nan for column in OUTPUT_COLUMNS}
            result["session_id"] = session_id
            result["frame_id"] = row.frame
            result["timestamp"] = row.timestamp

            # Camera projection and view matrices.
            if pd.notna(row.camera_name) and pd.notna(row.p_matrix) and pd.notna(row.v_matrix):
                try:
                    names = literal_list(row.camera_name)
                    projections = numeric_matrix_list(row.p_matrix)
                    views = numeric_matrix_list(row.v_matrix)
                    if not (len(names) == len(projections) == len(views)):
                        stats["camera_length_mismatch"] += 1
                    camera_index = choose_camera(names)
                    if camera_index is None or camera_index >= min(len(projections), len(views)):
                        raise ValueError("selected camera has no matching projection/view matrix")
                    if len(names) > 1:
                        stats["multi_camera_rows"] += 1
                    projection = matrix4(projections[camera_index])
                    view = matrix4(views[camera_index])
                    kind = projection_type(projection)
                    stats[f"projection_{kind}"] += 1
                    if kind == "perspective":
                        fov_h, fov_v = extract_fov(projection)
                        result["fov_horizontal"] = fov_h
                        result["fov_vertical"] = fov_v
                        stats["fov_rows"] += 1

                    position, rotation_matrix, euler = camera_pose(view)
                    result["camera_pos_x"], result["camera_pos_y"], result["camera_pos_z"] = position
                    result["camera_yaw"], result["camera_pitch"], result["camera_roll"] = euler
                    stats["camera_pose_rows"] += 1

                    current_speed: float | None = None
                    if previous_camera is not None and math.isfinite(timestamp_sec):
                        prev_time, prev_position, prev_rotation, prev_speed = previous_camera
                        delta_time = timestamp_sec - prev_time
                        if 0.0 < delta_time <= 5.0:
                            current_speed = float(np.linalg.norm(position - prev_position) / delta_time)
                            relative = prev_rotation.T @ rotation_matrix
                            rotation_angle = float(Rotation.from_matrix(relative).magnitude())
                            result["camera_speed"] = current_speed
                            result["camera_rotation_rate"] = math.degrees(rotation_angle) / delta_time
                            stats["camera_speed_rows"] += 1
                            stats["camera_rotation_rate_rows"] += 1
                            if prev_speed is not None:
                                result["camera_acceleration"] = abs(current_speed - prev_speed) / delta_time
                                stats["camera_acceleration_rows"] += 1
                        else:
                            stats["camera_nonadjacent_or_bad_dt"] += 1
                    if math.isfinite(timestamp_sec):
                        previous_camera = (timestamp_sec, position, rotation_matrix, current_speed)
                except Exception as exc:  # continue at row granularity
                    stats["camera_parse_errors"] += 1
                    if stats["camera_parse_errors"] <= 5:
                        stats[f"camera_error_example::{type(exc).__name__}: {exc}"] += 1

            # Visible object matrices. Missing scene fields remain NaN rather than
            # being treated as a zero-object frame.
            if pd.notna(row.object_name) and pd.notna(row.m_matrix):
                try:
                    names = [str(value) for value in literal_list(row.object_name)]
                    matrices = numeric_matrix_list(row.m_matrix)
                    if len(names) != len(matrices):
                        stats["object_length_mismatch"] += 1
                    usable = min(len(names), len(matrices))
                    current_positions: dict[str, np.ndarray] = {}
                    name_counts = Counter(names[:usable])
                    duplicate_names = {name for name, count in name_counts.items() if count > 1}
                    if duplicate_names:
                        stats["rows_with_duplicate_object_names"] += 1
                        stats["duplicate_object_instances"] += sum(name_counts[name] for name in duplicate_names)
                    valid_positions: list[tuple[str, np.ndarray]] = []
                    for index in range(usable):
                        try:
                            model = matrix4(matrices[index])
                            position = model[3, 0:3].copy()
                            valid_positions.append((names[index], position))
                        except Exception:
                            stats["invalid_object_matrices"] += 1
                    result["object_count"] = len(valid_positions)
                    stats["object_count_rows"] += 1

                    object_speeds: list[float] = []
                    delta_time = (
                        timestamp_sec - previous_object_timestamp
                        if previous_object_timestamp is not None and math.isfinite(timestamp_sec)
                        else math.nan
                    )
                    for name, position in valid_positions:
                        if name in duplicate_names:
                            continue
                        current_positions[name] = position
                        if name in previous_objects and math.isfinite(delta_time) and 0.0 < delta_time <= 5.0:
                            speed = float(np.linalg.norm(position - previous_objects[name]) / delta_time)
                            if math.isfinite(speed):
                                object_speeds.append(speed)
                    if object_speeds:
                        result["mean_object_speed"] = float(np.mean(object_speeds))
                        result["max_object_speed"] = float(np.max(object_speeds))
                        stats["object_speed_rows"] += 1
                        stats["matched_object_instances"] += len(object_speeds)
                    previous_objects = current_positions
                    previous_object_timestamp = timestamp_sec if math.isfinite(timestamp_sec) else None
                except Exception as exc:
                    stats["object_parse_errors"] += 1
                    if stats["object_parse_errors"] <= 5:
                        stats[f"object_error_example::{type(exc).__name__}: {exc}"] += 1

            outputs.append(result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_frame = pd.DataFrame(outputs, columns=list(OUTPUT_COLUMNS))
    output_frame.to_csv(output_path, index=False, encoding="utf-8-sig", float_format="%.9g")
    elapsed = time.perf_counter() - start
    return {
        "session_id": session_id,
        "source": str(csv_path),
        "output": str(output_path),
        "time_unit": time_unit,
        "elapsed_sec": elapsed,
        **dict(stats),
    }


def main() -> int:
    args = parse_args()
    root = Path.cwd()
    input_dir = args.input_dir.resolve()
    csv_paths = sorted(input_dir.glob("*/data_file.csv"))
    if args.session:
        requested = set(args.session)
        csv_paths = [path for path in csv_paths if path.parent.name in requested]
    if not csv_paths:
        raise FileNotFoundError(f"No <session>/data_file.csv found under {input_dir}")

    # Freeze the format report before feature extraction.
    write_format_report(csv_paths[0], args.format_report.resolve())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.per_session_dir.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    log_lines = [
        "VR.net matrix parsing log",
        f"Input directory: {input_dir}",
        f"Session count discovered: {len(csv_paths)}",
        "No downloads, model training, or Git operations were performed.",
        "Matrix convention: serialized 16-value tuple reshaped row-major; translation [3,0:3].",
        "Camera Euler convention: extrinsic Y-X-Z => yaw, pitch, roll.",
        "Camera-space handedness: fixed Z reflection removed before Euler decomposition.",
        "Object speed matching: only unique object names present in consecutive scene-labelled rows.",
        "Position unit is retained as source coordinate units; m/s interpretation is conditional on 1 Unity unit = 1 metre.",
        "",
    ]
    per_session_outputs: list[Path] = []
    summaries: list[dict[str, Any]] = []
    for csv_path in csv_paths:
        session_output = args.per_session_dir / f"{csv_path.parent.name}.csv"
        try:
            summary = parse_session(csv_path, session_output, args.chunk_size)
            summaries.append(summary)
            per_session_outputs.append(session_output)
            print(
                f"{summary['session_id']}: rows={summary.get('rows', 0)}, "
                f"FOV={summary.get('fov_rows', 0)}, camera_speed={summary.get('camera_speed_rows', 0)}, "
                f"object_speed={summary.get('object_speed_rows', 0)}, elapsed={summary['elapsed_sec']:.1f}s",
                flush=True,
            )
            log_lines.extend([f"[{summary['session_id']}] SUCCESS"])
            for key, value in sorted(summary.items()):
                log_lines.append(f"{key}: {value}")
            log_lines.append("")
        except Exception as exc:
            print(f"{csv_path.parent.name}: FAILED: {type(exc).__name__}: {exc}", flush=True)
            log_lines.extend(
                [
                    f"[{csv_path.parent.name}] FAILED",
                    f"{type(exc).__name__}: {exc}",
                    "",
                ]
            )

    if not per_session_outputs:
        args.log.write_text("\n".join(log_lines), encoding="utf-8")
        raise RuntimeError("All sessions failed; no combined output was created")

    combined = pd.concat(
        [pd.read_csv(path, encoding="utf-8-sig") for path in per_session_outputs],
        ignore_index=True,
    )
    combined.to_csv(args.output, index=False, encoding="utf-8-sig", float_format="%.9g")
    log_lines.extend(
        [
            "[COMBINED] SUCCESS",
            f"successful_sessions: {len(per_session_outputs)}",
            f"failed_sessions: {len(csv_paths) - len(per_session_outputs)}",
            f"combined_rows: {len(combined)}",
            f"combined_output: {args.output.resolve()}",
            "",
        ]
    )
    args.log.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"Combined {len(combined)} rows from {len(per_session_outputs)} sessions -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
