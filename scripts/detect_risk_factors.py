#!/usr/bin/env python
"""Detect literature-based VR scene risk-factor conditions from a feature CSV.

Rows with skipped extraction status are excluded. A session is marked triggered
if any usable window satisfies an operationalized rule. Unknown thresholds,
missing metrics, and unit mismatches produce triggered=unknown; they are never
silently converted into non-triggers. Severity remains unknown unless a rule
defines validated severity bands.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

OUTPUT_FIELDS = [
    "session_id", "risk_factor_id", "risk_factor_name", "triggered",
    "severity", "evidence_strength", "recommendation", "caveat",
]


def as_number(value: Any):
    try:
        if value is None or str(value).strip() == "":
            return None
        number = float(value)
        return number if number == number else None
    except (TypeError, ValueError):
        return None


def is_unknown_value(value: Any) -> bool:
    return value is None or str(value).strip().lower() in {
        "", "unknown", "tbd", "na", "n/a", "null", "none", "nan",
    }


def has_observed_value(value: Any) -> bool:
    return not is_unknown_value(value)


def get_requirements(rule: dict) -> list[tuple[str, dict]]:
    threshold = rule.get("threshold") or {}
    if threshold.get("condition") == "and":
        return [
            (part.get("metric", ""), part)
            for part in threshold.get("conditions", [])
        ]
    return [(rule.get("metric", ""), threshold)]


def compare(value: Any, threshold: dict):
    """Return True/False/None; None means metric or cutoff cannot be evaluated."""
    if is_unknown_value(value):
        return None
    condition = str(threshold.get("condition", "unknown")).lower()
    if condition == "and":
        outcomes = [compare(value, item) for item in threshold.get("conditions", [])]
        if any(outcome is False for outcome in outcomes):
            return False
        if outcomes and all(outcome is True for outcome in outcomes):
            return True
        return None
    if condition == "unknown":
        return None

    expected = threshold.get("value")
    if condition == "range":
        low, high = as_number(threshold.get("low")), as_number(threshold.get("high"))
        actual = as_number(value)
        if actual is None or low is None or high is None:
            return None
        return low <= actual <= high

    if condition in {">", ">=", "<", "<="}:
        actual, expected = as_number(value), as_number(expected)
        if actual is None or expected is None:
            return None
        return {
            ">": actual > expected,
            ">=": actual >= expected,
            "<": actual < expected,
            "<=": actual <= expected,
        }[condition]

    if condition in {"==", "!="}:
        if expected is None:
            return None
        if isinstance(expected, bool):
            actual = str(value).strip().lower()
            if actual in {"true", "1", "yes"}:
                actual_value = True
            elif actual in {"false", "0", "no"}:
                actual_value = False
            else:
                return None
            return actual_value is expected if condition == "==" else actual_value is not expected
        if isinstance(expected, (int, float)):
            actual = as_number(value)
            if actual is None:
                return None
            return actual == expected if condition == "==" else actual != expected
        actual = str(value).strip().lower()
        expected_text = str(expected).strip().lower()
        return actual == expected_text if condition == "==" else actual != expected_text
    return None


def load_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Feature CSV has no header: {path}")
        rows = list(reader)
    return reader.fieldnames, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--missing-report", type=Path, default=Path("results/missing_metrics.md"))
    args = parser.parse_args()

    with args.rules.open("r", encoding="utf-8-sig") as f:
        rulebook = yaml.safe_load(f)
    rules = rulebook.get("risk_factors", [])
    if not rules:
        raise ValueError("Rule file has no risk_factors")

    columns, all_rows = load_rows(args.features)
    session_col = next((c for c in ("session_id", "session", "scene_id", "scene") if c in columns), None)
    if not session_col:
        raise ValueError("Feature CSV needs session_id, session, scene_id, or scene column")

    status_col = "status" if "status" in columns else None
    usable_rows, skipped = [], 0
    for row in all_rows:
        status = (row.get(status_col, "") if status_col else "").strip().lower()
        if status.startswith("skipped") or status in {"error", "failed"}:
            skipped += 1
            continue
        usable_rows.append(row)

    grouped = defaultdict(list)
    for row in usable_rows:
        sid = row.get(session_col, "").strip() or "unknown_session"
        grouped[sid].append(row)

    # Mark absent metrics and unsupported thresholds explicitly.
    missing_entries = []
    for rule in rules:
        threshold = rule.get("threshold") or {}
        for metric, subthreshold in get_requirements(rule):
            if metric not in columns:
                reason = "metric column absent"
                # Distinguish non-equivalent image flow from physical speed.
                if metric == "virtual_translation_speed_m_s" and any(c.startswith("flow_mean_magnitude") for c in columns):
                    reason += "; available flow magnitudes are pixel displacement per sampled image pair, not m/s"
                if metric == "expected_duration_min" and "event_time_sec" in columns:
                    reason += "; event_time_sec is an epoch timestamp in the current matrix, not session duration"
                missing_entries.append((rule["id"], metric, reason))
            elif not any(has_observed_value(row.get(metric)) for row in usable_rows):
                missing_entries.append((rule["id"], metric, "metric column present but all window values are missing or explicitly unknown"))
            if subthreshold.get("condition") == "unknown":
                missing_entries.append((rule["id"], metric, "threshold/cutoff is unsupported or TBD"))
            if subthreshold.get("condition") == "and":
                for part in subthreshold.get("conditions", []):
                    if part.get("condition") == "unknown" or part.get("value") is None:
                        missing_entries.append((rule["id"], part.get("metric", metric), "combined cutoff is unsupported or TBD"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for sid in sorted(grouped):
            session_rows = grouped[sid]
            for rule in rules:
                requirements = get_requirements(rule)
                outcomes = []
                for row in session_rows:
                    if any(metric not in columns for metric, _ in requirements):
                        outcomes.append(None)
                        continue
                    if any(thr.get("condition") == "unknown" for _, thr in requirements):
                        outcomes.append(None)
                        continue
                    # Composite thresholds use their own metrics on this row.
                    if (rule.get("threshold") or {}).get("condition") == "and":
                        parts = (rule["threshold"].get("conditions") or [])
                        part_results = []
                        for part in parts:
                            if part.get("metric") not in columns:
                                part_results.append(None)
                            else:
                                part_results.append(compare(row.get(part["metric"]), part))
                        if any(x is False for x in part_results):
                            outcomes.append(False)
                        elif part_results and all(x is True for x in part_results):
                            outcomes.append(True)
                        else:
                            outcomes.append(None)
                    else:
                        metric, threshold = requirements[0]
                        outcomes.append(compare(row.get(metric), threshold))

                if any(x is True for x in outcomes):
                    triggered = "yes"
                elif outcomes and all(x is False for x in outcomes):
                    triggered = "no"
                else:
                    triggered = "unknown"

                # The rulebook intentionally has no validated severity-distance bands.
                severity = "unknown"
                writer.writerow({
                    "session_id": sid,
                    "risk_factor_id": rule.get("id", ""),
                    "risk_factor_name": rule.get("name", ""),
                    "triggered": triggered,
                    "severity": severity,
                    "evidence_strength": rule.get("evidence_strength", "unknown"),
                    "recommendation": rule.get("recommendation", ""),
                    "caveat": rule.get("caveat", ""),
                })

    args.missing_report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 缺失指标与不可判定规则",
        "",
        f"- 输入：{args.features}",
        f"- 保留窗口：{len(usable_rows)}；因提取状态跳过：{skipped}；session 数：{len(grouped)}。",
        "- triggered=unknown 表示现有数据/阈值无法判断，不表示未触发。",
        "- 本次没有规则可对当前矩阵作出有证据的 yes/no 判定；severity 也为 unknown，因为规则库没有文献支持的距离分箱。",
        "",
        "| 规则 ID | 需要的指标 | 原因 |",
        "|---|---|---|",
    ]
    for rid, metric, reason in sorted(set(missing_entries)):
        lines.append(f"| {rid} | {metric} | {reason} |")
    if not missing_entries:
        lines.append("| none | none | All rule metrics and cutoffs are operationalized in the input |")
    args.missing_report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"usable_windows={len(usable_rows)} skipped_rows={skipped} sessions={len(grouped)} rules={len(rules)} detections={len(grouped)*len(rules)}")
    print(f"output={args.output} missing_report={args.missing_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
