#!/usr/bin/env python3
"""Create 30-second, carry-forward label windows and merge existing features."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import yaml


WINDOW_FIELDS = [
    "window_id", "start_frame", "end_frame", "start_time", "end_time",
    "fms_raw", "fms_label", "label_source",
]
OUTCOME_TOKENS = ("fms", "ssq", "sickness", "ms_rating", "label", "symptom")


def load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    csv.field_size_limit(200_000_000)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def class_for(value: str, mapping: dict[str, Any]) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "unknown"
    classes = mapping["classes"]
    for label in ("low", "medium", "high"):
        limits = classes[label]
        low, high = limits.get("min_inclusive"), limits.get("max_inclusive")
        if (low is None or number >= float(low)) and (high is None or number <= float(high)):
            return label
    return "unknown"


def create_windows(raw_csv: Path, mapping_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with mapping_path.open("r", encoding="utf-8-sig") as handle:
        mapping = yaml.safe_load(handle)
    if mapping.get("frozen") is not True:
        raise ValueError("VR.net label mapping is not frozen; refusing to create training labels")
    headers, rows = load_rows(raw_csv)
    timestamp_col = mapping["timestamp_column"]
    frame_col = mapping["frame_column"]
    label_col = mapping["source_label_column"]
    missing = [column for column in (timestamp_col, frame_col, label_col) if column not in headers]
    if missing:
        raise ValueError(f"Raw CSV missing required columns: {missing}")

    samples: list[tuple[int, int, str]] = []
    events: list[tuple[int, float, str]] = []
    for row in rows:
        try:
            stamp = int(float(row[timestamp_col]))
            frame = int(float(row[frame_col]))
        except (TypeError, ValueError):
            continue
        raw_label = (row.get(label_col) or "").strip()
        samples.append((stamp, frame, raw_label))
        if raw_label and raw_label.casefold() not in {"nan", "none", "null"}:
            events.append((stamp, float(frame), raw_label))
    if not samples:
        raise ValueError("No rows with valid frame and timestamp values")

    t0 = min(stamp for stamp, _, _ in samples)
    t1 = max(stamp for stamp, _, _ in samples)
    unit = str(mapping.get("timestamp_unit", "milliseconds")).casefold()
    divisor = 1000.0 if unit in {"millisecond", "milliseconds", "ms"} else 1.0
    duration = (t1 - t0) / divisor
    window_seconds = float(mapping["window_seconds"])
    number_of_windows = max(1, int(math.ceil(duration / window_seconds)))
    per_window: list[list[tuple[int, int]]] = [[] for _ in range(number_of_windows)]
    for stamp, frame, _ in samples:
        relative = (stamp - t0) / divisor
        index = min(number_of_windows - 1, max(0, int(relative // window_seconds)))
        per_window[index].append((stamp, frame))

    event_windows: dict[int, list[tuple[int, str]]] = {}
    for stamp, _frame, value in events:
        index = min(number_of_windows - 1, max(0, int(((stamp - t0) / divisor) // window_seconds)))
        event_windows.setdefault(index, []).append((stamp, value))

    output: list[dict[str, Any]] = []
    prior: tuple[int, str] | None = None
    for index in range(number_of_windows):
        current_events = sorted(event_windows.get(index, []), key=lambda item: item[0])
        if current_events:
            prior = current_events[-1]
            raw, source = prior[1], "within_window"
        elif prior is not None:
            raw, source = prior[1], "previous"
        else:
            raw, source = "", "missing"
        start_sec = index * window_seconds
        end_sec = min((index + 1) * window_seconds, duration)
        window_samples = per_window[index]
        output.append({
            "window_id": f"window_{index:03d}",
            "start_frame": min((frame for _, frame in window_samples), default=""),
            "end_frame": max((frame for _, frame in window_samples), default=""),
            "start_time": round(start_sec, 3),
            "end_time": round(end_sec, 3),
            "fms_raw": raw,
            "fms_label": class_for(raw, mapping) if source != "missing" else "missing",
            "label_source": source,
            "window_index": index,
        })
    summary = {
        "raw_rows": len(rows), "valid_samples": len(samples), "nonempty_events": len(events),
        "duration_sec": duration, "window_seconds": window_seconds,
        "windows": len(output), "source_label_column": label_col,
        "event_value_counts": {},
    }
    for _, _, value in events:
        summary["event_value_counts"][value] = summary["event_value_counts"].get(value, 0) + 1
    return output, summary


def merge_features(
    windows: list[dict[str, Any]], feature_path: Path, metadata_path: Path | None,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    feature_headers, feature_rows = load_rows(feature_path)
    if "window_index" not in feature_headers:
        raise ValueError(f"Video features need window_index: {feature_path}")
    by_index: dict[int, dict[str, str]] = {}
    for row in feature_rows:
        try:
            by_index[int(float(row["window_index"]))] = row
        except (TypeError, ValueError):
            continue
    merged: list[dict[str, Any]] = []
    for window in windows:
        index = int(window["window_index"])
        row = {key: value for key, value in window.items() if key != "window_index"}
        row["window_index"] = index
        row.update(by_index.get(index, {}))
        row["window_id"] = window["window_id"]
        row["fms_raw"] = window["fms_raw"]
        row["fms_label"] = window["fms_label"]
        row["label_source"] = window["label_source"]
        merged.append(row)

    note: list[str] = []
    note.append(f"使用视频特征文件 `{feature_path}`：{len(feature_rows)} 行，按 window_index 与标签窗关联。")
    if metadata_path is None or not metadata_path.is_file():
        note.append(f"未找到可选的 HMD 元数据特征文件 `{metadata_path or 'features/mars_miners_metadata_features.csv'}`；未创建或估算速度、旋转轴、角速度、路径曲率等字段。")
        metadata_headers: list[str] = []
    else:
        metadata_headers, metadata_rows = load_rows(metadata_path)
        key = "window_id" if "window_id" in metadata_headers else "window_index" if "window_index" in metadata_headers else None
        if key is None:
            note.append(f"HMD 元数据特征存在但缺少 window_id/window_index：{metadata_path}；为避免错配而跳过。")
        else:
            safe_headers = [h for h in metadata_headers if not any(token in h.casefold() for token in OUTCOME_TOKENS)]
            meta_by_key = {(row.get(key) or "").strip(): row for row in metadata_rows}
            joined = 0
            for row in merged:
                join_key = row["window_id"] if key == "window_id" else str(row["window_index"])
                match = meta_by_key.get(join_key)
                if match:
                    joined += 1
                    for header in safe_headers:
                        if header != key:
                            row[f"metadata_{header}"] = match.get(header, "")
            note.append(f"读取 HMD 特征 {len(metadata_rows)} 行，按 `{key}` 成功关联 {joined}/{len(merged)} 个窗；结果列加前缀 metadata_。")

    feature_value_columns = [
        header for header in feature_headers
        if header not in {"session_id", "window_index"}
        and not any(token in header.casefold() for token in OUTCOME_TOKENS)
    ]
    metadata_value_columns = sorted({key for row in merged for key in row if key.startswith("metadata_")})
    fieldnames = list(dict.fromkeys([*WINDOW_FIELDS, "window_index", *feature_value_columns, *metadata_value_columns]))
    return merged, fieldnames, note


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv", default="data/raw/vrnet/samples_full/11_2_Mars_Miners/data_file.csv")
    parser.add_argument("--mapping", default="configs/label_mapping_vrnet.yaml")
    parser.add_argument("--video-features", default="features/mars_miners_full_features.csv")
    parser.add_argument("--metadata-features", default="features/mars_miners_metadata_features.csv")
    parser.add_argument("--windows-output", default="features/training_windows_vrnet.csv")
    parser.add_argument("--matrix-output", default="features/training_feature_matrix_vrnet.csv")
    parser.add_argument("--missing-note", default="results/missing_features_note.md")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    resolve = lambda raw: (root / raw).resolve() if not Path(raw).is_absolute() else Path(raw)
    raw_path, mapping_path = resolve(args.raw_csv), resolve(args.mapping)
    video_path, metadata_path = resolve(args.video_features), resolve(args.metadata_features)
    windows_path, matrix_path, note_path = resolve(args.windows_output), resolve(args.matrix_output), resolve(args.missing_note)
    windows, summary = create_windows(raw_path, mapping_path)
    windows_path.parent.mkdir(parents=True, exist_ok=True)
    with windows_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=WINDOW_FIELDS)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in WINDOW_FIELDS} for row in windows])
    merged, columns, notes = merge_features(windows, video_path, metadata_path if metadata_path.is_file() else None)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    with matrix_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)
    labeled = [row for row in windows if row["fms_label"] in {"low", "medium", "high"}]
    counts = {level: sum(row["fms_label"] == level for row in windows) for level in ("low", "medium", "high")}
    notes = [
        "# VR.net 缺失特征与数据准备说明", "",
        "标签按冻结映射生成；标签字段源名为 `MS_rating`。未读取或合并任何未说明的推断标签。", "",
        f"- 原始 CSV：`{raw_path}`（{summary['raw_rows']} 行；有效时间/帧样本 {summary['valid_samples']} 行）",
        f"- 非空 `{summary['source_label_column']}` 事件：{summary['nonempty_events']}；原始值分布：`{json.dumps(summary['event_value_counts'], ensure_ascii=False, sort_keys=True)}`",
        f"- 时间窗：{summary['window_seconds']} 秒；总录制时长 {summary['duration_sec']:.3f} 秒；窗口数 {summary['windows']}。",
        f"- 标签窗：有分类标签 {len(labeled)}；低 {counts['low']}，中 {counts['medium']}，高 {counts['high']}；其余为 missing/unknown。",
        f"- 输出：`{windows_path}`、`{matrix_path}`。",
        "", "## 特征覆盖", "", *notes,
        "", "## 标签来源说明", "",
        "`MS_rating` 是源 CSV 的字段名。本项目按用户冻结的分箱把该字段作为窗口级标签输入；此操作不证明它与其他数据集的 FMS 量表可互换。一个窗有多个事件时取窗内时间最新的非空值；无窗内事件时以前值延续；首个事件之前记 missing。",
        "",
    ]
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text("\n".join(notes), encoding="utf-8")
    print(f"windows={len(windows)} labeled={len(labeled)} classes={json.dumps(counts)} matrix_columns={len(columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
