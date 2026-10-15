#!/usr/bin/env python3
"""Render rule detections and their missing-evidence context as Markdown."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import yaml


UNKNOWN_VALUES = {"", "unknown", "tbd", "na", "n/a", "none", "null", "nan"}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def md(value: Any) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip()


def observed(value: Any) -> bool:
    return str(value if value is not None else "").strip().casefold() not in UNKNOWN_VALUES


def get_metrics(rule: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    threshold = rule.get("threshold") or {}
    if threshold.get("condition") == "and":
        return [(str(part.get("metric", "")), part) for part in threshold.get("conditions", [])]
    return [(str(rule.get("metric", "")), threshold)]


def requirement_context(
    detection: dict[str, str], rule: dict[str, Any], rows_for_session: list[dict[str, str]], feature_headers: list[str],
) -> tuple[str, str]:
    missing: list[str] = []
    needed: list[str] = []
    for metric, threshold in get_metrics(rule):
        if not metric:
            continue
        values = [row.get(metric, "") for row in rows_for_session]
        if metric not in feature_headers or not any(observed(value) for value in values):
            missing.append(metric)
            needed.append(f"采集/确认 `{metric}`，并核对其单位与窗口定义")
        if threshold.get("condition") == "unknown" or threshold.get("value") is None and threshold.get("condition") not in {"range", "and"}:
            needed.append(f"为 `{metric}` 提供有证据支持的操作化阈值（当前 cutoff 未知/TBD）")
    threshold = rule.get("threshold") or {}
    if threshold.get("condition") == "and":
        for part in threshold.get("conditions", []):
            if part.get("condition") == "unknown" or part.get("value") is None:
                metric = part.get("metric", "rule")
                needed.append(f"核实联合条件 `{metric}` 的证据与阈值")
    if not missing:
        missing_text = "无缺失指标（但阈值或组合条件可能仍未核实）"
    else:
        missing_text = ", ".join(dict.fromkeys(missing))
    if not needed:
        needed_text = md(rule.get("caveat", "核实规则定义与输入证据。"))
    else:
        needed_text = "; ".join(dict.fromkeys(needed))
        caveat = str(rule.get("caveat", "")).strip()
        if caveat:
            needed_text += "; Caveat: " + caveat
    return missing_text, needed_text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", required=True, type=Path, help="CSV from detect_risk_factors.py")
    parser.add_argument("--features", required=True, type=Path, help="Merged feature matrix CSV")
    parser.add_argument("--rules", required=True, type=Path, help="YAML literature rulebook")
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument("--synthetic-scene", action="store_true")
    parser.add_argument("--alignment-note", default="Alignment not reported.")
    parser.add_argument("--metadata-note", default="No HMD metadata integration status supplied.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    detection_headers, detections = read_csv(args.detections)
    feature_headers, feature_rows = read_csv(args.features)
    with args.rules.open("r", encoding="utf-8-sig") as handle:
        rulebook = yaml.safe_load(handle) or {}
    rules = {str(item.get("id", "")): item for item in rulebook.get("risk_factors", [])}
    by_session: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in feature_rows:
        by_session[row.get("session_id", "unknown_session")].append(row)

    yes_rows, no_rows, unknown_rows = [], [], []
    status_counts: Counter[str] = Counter()
    for row in detections:
        status = (row.get("triggered") or "unknown").strip().casefold()
        status_counts[status] += 1
        if status == "yes":
            yes_rows.append(row)
        elif status == "no":
            no_rows.append(row)
        else:
            unknown_rows.append(row)

    session_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in detections:
        status = (row.get("triggered") or "unknown").strip().casefold()
        session_counts[row.get("session_id", "unknown_session")][status] += 1

    scene_path = str(args.scene) if args.scene else "未提供"
    video_path = str(args.video_dir) if args.video_dir else "未提供"
    metadata_path = str(args.metadata_csv) if args.metadata_csv else "未提供"
    if args.synthetic_scene and args.video_dir:
        source_label = f"合成场景（非真实数据） + 真实 VR.net 视频帧（{args.video_dir.name}）"
    elif args.synthetic_scene:
        source_label = "合成场景（非真实数据）"
    elif args.scene and args.video_dir:
        source_label = "Unity 场景 + 视频帧（请按路径和对齐状态核验来源关系）"
    elif args.scene:
        source_label = "Unity 场景文件"
    elif args.video_dir:
        source_label = f"视频帧（{args.video_dir.name}）"
    else:
        source_label = "HMD 元数据 CSV"

    lines = [
        "# 场景风险分析报告", "", "## 场景信息", "",
        f"- 场景文件：`{md(scene_path)}`",
        f"- 视频来源：`{md(video_path)}`",
        f"- HMD 元数据：`{md(metadata_path)}`",
        f"- 分析日期：{date.today().isoformat()}",
        f"- 数据来源标注：{source_label}",
        "", "## 对齐与输入处理", "",
        str(args.alignment_note).strip(), "",
        str(args.metadata_note).strip(),
        "", "当前报告按 `session_id` 分组报告规则结果；没有对不匹配的数据源做隐式拼接。",
        "", "## 检测到的风险因素", "", "### 触发规则", "",
        "| 规则 | 来源组 | 严重程度 | 证据强度 | 修改建议 |", "|---|---|---|---|---|",
    ]
    if yes_rows:
        for row in yes_rows:
            lines.append("| {} | {} | {} | {} | {} |".format(
                md(row.get("risk_factor_name") or row.get("risk_factor_id")), md(row.get("session_id")),
                md(row.get("severity", UNKNOWN)), md(row.get("evidence_strength", UNKNOWN)),
                md(row.get("recommendation", "")),
            ))
    else:
        lines.append("| 无 | — | — | — | 没有规则在当前数据中被判为触发；这不代表晕动症风险为零。 |")

    lines.extend(["", "### 未触发规则", "", "| 规则 | 来源组 | 状态 | 原因 |", "|---|---|---|---|"])
    if no_rows:
        for row in no_rows:
            lines.append("| {} | {} | 未触发 | 在可计算窗口中，操作化条件未满足。 |".format(
                md(row.get("risk_factor_name") or row.get("risk_factor_id")), md(row.get("session_id"))))
    else:
        lines.append("| 无 | — | — | 当前没有可可靠判为未触发的规则。 |")

    lines.extend(["", "### 无法判定规则", "", "| 规则 | 来源组 | 缺失指标 | 需要什么 |", "|---|---|---|---|"])
    if unknown_rows:
        for row in unknown_rows:
            rule = rules.get(row.get("risk_factor_id", ""), {})
            missing, needed = requirement_context(
                row, rule, by_session.get(row.get("session_id", ""), []), feature_headers,
            )
            lines.append("| {} | {} | {} | {} |".format(
                md(row.get("risk_factor_name") or row.get("risk_factor_id")), md(row.get("session_id")),
                md(missing), md(needed),
            ))
    else:
        lines.append("| 无 | — | — | — | 当前规则均已得到可计算结果。 |")

    lines.extend(["", "## 汇总", "", f"- 触发规则数：{status_counts.get('yes', 0)}",
                  f"- 未触发规则数：{status_counts.get('no', 0)}",
                  f"- 无法判定规则数：{status_counts.get('unknown', 0)}",
                  f"- 检测结果记录总数：{len(detections)}", "", "### 按来源组汇总", "",
                  "| 来源组 | 触发 | 未触发 | unknown |", "|---|---:|---:|---:|"])
    for session, counts in sorted(session_counts.items()):
        lines.append(f"| {md(session)} | {counts.get('yes', 0)} | {counts.get('no', 0)} | {counts.get('unknown', 0)} |")

    lines.extend(["", "## 修改建议", ""])
    suggestions = list(dict.fromkeys(str(row.get("recommendation", "")).strip() for row in yes_rows if row.get("recommendation")))
    if suggestions:
        lines.extend(f"- {item}" for item in suggestions)
    else:
        lines.append("本次没有规则被判为触发，因此不提供基于触发结果的场景修改建议。可优先补齐上表所列测量并复核阈值，再重新运行分析。")

    lines.extend([
        "", "## 免责声明", "",
        "本报告检测的是已知风险因素，不预测晕动症。",
        "所有建议基于文献，未经用户验证。",
        "合成场景数据仅用于流程演示，不代表真实风险结论。",
        "视频图像特征与合成 Unity 场景不匹配时按独立来源处理；unknown 表示证据不足，不表示未触发。",
        "规则严重程度没有经验证的距离分级时保留为 unknown。",
        "",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"report={args.output} triggers={status_counts.get('yes', 0)} nontriggers={status_counts.get('no', 0)} unknown={status_counts.get('unknown', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
