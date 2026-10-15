#!/usr/bin/env python3
"""
check_phase2.py — Phase 2 数据核验
检查：文件存在性、标签 CSV、映射文件、配置、元数据。
输出：results/phase2_check_report.md
"""

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent

REQUIRED_FILES = {
    "config": ROOT / "configs" / "label_mapping.yaml",
    "preprocess_script": ROOT / "scripts" / "preprocess.py",
    "csv": ROOT / "data" / "raw" / "vrwalking" / "SSQ_scores_NASA_Reaction_Time.csv",
    "video_dir": ROOT / "data" / "raw" / "vrwalking" / "videos",
    "mapping": ROOT / "data" / "raw" / "vrwalking" / "participant_video_mapping.csv",
    "labels_out": ROOT / "data" / "processed" / "vrwalking" / "labels.csv",
    "manifest_out": ROOT / "data" / "processed" / "vrwalking" / "video_manifest.csv",
    "metadata_out": ROOT / "data" / "processed" / "vrwalking" / "preprocess_metadata.json",
    "log": ROOT / "results" / "preprocess_log.txt",
}

EXPECTED_LABEL_COLS = ["n_p", "o_p", "d_p", "TS_p", "First_FMS", "Second_FMS", "Overall_FMS"]


def check_file_exists(name, path):
    if not path.exists():
        return f"| {name} | MISSING | {path} |"
    if path.is_dir():
        n = len([p for p in path.iterdir() if p.is_file()])
        return f"| {name} | DIR ({n} files) | {path} |"
    size = path.stat().st_size
    return f"| {name} | OK ({size} bytes) | {path} |"


def check_csv(csv_path):
    lines = []
    if not csv_path.exists():
        lines.append("CSV 不存在，无法检查。")
        return lines
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception as e:
        lines.append(f"CSV 读取失败: {e}")
        return lines

    lines.append(f"- 行数: {len(df)}")
    lines.append(f"- 列数: {len(df.columns)}")
    lines.append(f"- 列名: {list(df.columns)}")
    lines.append("")
    lines.append("### 标签列检查")
    for col in EXPECTED_LABEL_COLS:
        if col in df.columns:
            n_missing = int(df[col].isna().sum())
            lines.append(f"- `{col}`: 存在, 缺失 {n_missing}/{len(df)}")
        else:
            lines.append(f"- `{col}`: **缺失**")
    lines.append("")
    for col in ["TS_p", "Overall_FMS"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            lines.append(f"### `{col}` 描述性统计")
            lines.append(f"- min: {s.min()}")
            lines.append(f"- max: {s.max()}")
            lines.append(f"- mean: {s.mean():.3f}")
            lines.append(f"- median: {s.median():.3f}")
            lines.append(f"- std: {s.std():.3f}")
            lines.append(f"- 缺失: {int(s.isna().sum())}")
            lines.append("")
    return lines


def check_mapping(mapping_path, video_dir):
    lines = []
    if not mapping_path.exists():
        lines.append("映射文件不存在，需要人工建立。")
        return lines
    try:
        m = pd.read_csv(mapping_path, encoding="utf-8-sig")
    except Exception as e:
        lines.append(f"映射文件读取失败: {e}")
        return lines

    lines.append(f"- 行数: {len(m)}")
    lines.append(f"- 列名: {list(m.columns)}")
    required = {"participant_id", "video_file"}
    if not required.issubset(set(m.columns)):
        lines.append(f"- **缺少列**: {required - set(m.columns)}")
        return lines
    lines.append(f"- participant_id 范围: {m['participant_id'].min()} - {m['participant_id'].max()}")
    lines.append(f"- 重复 participant_id: {int(m['participant_id'].duplicated().sum())}")
    lines.append(f"- 重复 video_file: {int(m['video_file'].duplicated().sum())}")
    if video_dir.exists():
        available = {p.name for p in video_dir.iterdir() if p.is_file()}
        missing = [f for f in m["video_file"] if f not in available]
        lines.append(f"- 视频目录文件数: {len(available)}")
        lines.append(f"- 映射中不存在的视频: {len(missing)}")
        if missing:
            lines.append(f"- 缺失文件示例: {missing[:10]}")
    else:
        lines.append("- 视频目录不存在，无法验证")
    return lines


def check_metadata(metadata_path):
    lines = []
    if not metadata_path.exists():
        lines.append("元数据文件不存在。")
        return lines
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        lines.append(f"元数据读取失败: {e}")
        return lines
    for key in ["feature_source_type", "proxy_label", "analysis_unit",
                "evaluation_protocol", "leave_one_scene_out", "n_participants"]:
        lines.append(f"- `{key}`: {meta.get(key, 'MISSING')}")
    if meta.get("feature_source_type") != "path_dependent":
        lines.append("- **警告**: feature_source_type 不是 path_dependent")
    if meta.get("proxy_label") is not False:
        lines.append("- **警告**: proxy_label 不是 false")
    if meta.get("evaluation_protocol") != "LOSO_only":
        lines.append("- **警告**: evaluation_protocol 不是 LOSO_only")
    return lines


def check_config(config_path):
    lines = []
    if not config_path.exists():
        lines.append("配置文件不存在。")
        return lines
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        lines.append(f"配置读取失败: {e}")
        return lines
    for key in ["label_source", "primary_label", "analysis_unit", "evaluation_protocol"]:
        lines.append(f"- `{key}`: {cfg.get(key, 'MISSING')}")
    fs = cfg.get("feature_source", {})
    lines.append(f"- `feature_source.type`: {fs.get('type', 'MISSING')}")
    pl = cfg.get("proxy_label", {})
    lines.append(f"- `proxy_label.enabled`: {pl.get('enabled', 'MISSING')}")
    th = cfg.get("low_medium_high_mapping", {}).get("thresholds", {})
    lines.append(f"- thresholds: {th}")
    return lines


def main():
    out = []
    out.append("# Phase 2 数据核验报告")
    out.append("")
    out.append("## 1. 文件存在性")
    out.append("")
    out.append("| 文件 | 状态 | 路径 |")
    out.append("|---|---|---|")
    for name, path in REQUIRED_FILES.items():
        out.append(check_file_exists(name, path))
    out.append("")
    out.append("## 2. 标签 CSV 检查")
    out.append("")
    out.extend(check_csv(REQUIRED_FILES["csv"]))
    out.append("")
    out.append("## 3. 映射文件检查")
    out.append("")
    out.extend(check_mapping(REQUIRED_FILES["mapping"], REQUIRED_FILES["video_dir"]))
    out.append("")
    out.append("## 4. 配置检查")
    out.append("")
    out.extend(check_config(REQUIRED_FILES["config"]))
    out.append("")
    out.append("## 5. 预处理元数据检查")
    out.append("")
    out.extend(check_metadata(REQUIRED_FILES["metadata_out"]))
    out.append("")
    out.append("## 6. 门控判定")
    out.append("")
    if not REQUIRED_FILES["csv"].exists():
        out.append("- **门控未通过**: 标签 CSV 不存在")
    elif not REQUIRED_FILES["mapping"].exists():
        out.append("- **门控未通过**: 映射文件不存在，需要人工建立")
    elif not REQUIRED_FILES["video_dir"].exists():
        out.append("- **门控未通过**: 视频目录不存在，需要人工确认访问权限")
    elif not REQUIRED_FILES["metadata_out"].exists():
        out.append("- **门控未通过**: 预处理尚未运行，先运行 preprocess.py")
    else:
        out.append("- **门控通过**: 所有必要文件存在，可进入 Phase 3")
    out.append("")

    report_path = ROOT / "results" / "phase2_check_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"报告写入: {report_path}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
