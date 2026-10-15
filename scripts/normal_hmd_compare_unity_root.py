#!/usr/bin/env python3
"""Compare applied BodyRoot dynamics from two sets of Unity telemetry logs."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from normal_hmd_common import atomic_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = (
    ROOT / "results/normal_hmd_unity_rollout_logs/20261011_145758_396_seed_20261011/telemetry.csv",
    ROOT / "results/normal_hmd_unity_rollout_logs/20261011_145801_496_seed_20261012/telemetry.csv",
)
DEFAULT_CANDIDATE = (
    ROOT / "results/normal_hmd_unity_rollout_logs/20261012_212754_809_seed_20261011/telemetry.csv",
    ROOT / "results/normal_hmd_unity_rollout_logs/20261012_212757_912_seed_20261012/telemetry.csv",
)
DEFAULT_JSON = ROOT / "results/normal_hmd_v4_smooth_walk_unity_dynamics.json"
DEFAULT_MD = ROOT / "results/normal_hmd_v4_smooth_walk_unity_dynamics.md"
DT = 0.027


def read_columns(path: Path) -> tuple[np.ndarray, np.ndarray]:
    planar: list[list[float]] = []
    yaw: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            planar.append([
                float(row["applied_root_right_velocity_mps"]),
                float(row["applied_root_forward_velocity_mps"]),
            ])
            yaw.append(float(row["applied_root_yaw_rate_radps_rh"]))
    return np.asarray(planar, dtype=np.float64), np.asarray(yaw, dtype=np.float64)


def metrics(path: Path) -> dict[str, Any]:
    planar, yaw = read_columns(path)
    acceleration = np.diff(planar, axis=0) / DT
    jerk = np.diff(acceleration, axis=0) / DT
    yaw_acceleration = np.diff(yaw) / DT
    yaw_jerk = np.diff(yaw_acceleration) / DT
    return {
        "telemetry": str(path.resolve()),
        "steps": int(len(planar)),
        "linear_acceleration_p95_mps2": float(
            np.quantile(np.linalg.norm(acceleration, axis=1), .95)
        ),
        "linear_jerk_p95_mps3": float(
            np.quantile(np.linalg.norm(jerk, axis=1), .95)
        ),
        "yaw_acceleration_p95_radps2": float(
            np.quantile(np.abs(yaw_acceleration), .95)
        ),
        "yaw_jerk_p95_radps3": float(np.quantile(np.abs(yaw_jerk), .95)),
    }


def summarize(items: list[dict[str, Any]]) -> dict[str, float]:
    names = (
        "linear_acceleration_p95_mps2",
        "linear_jerk_p95_mps3",
        "yaw_acceleration_p95_radps2",
        "yaw_jerk_p95_radps3",
    )
    return {
        name: float(np.median([float(item[name]) for item in items]))
        for name in names
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Normal HMD v4 丝滑走路 Unity 对照",
        "",
        f"状态：**{'PASS' if report['acceptance']['passed'] else 'FAIL'}**",
        "",
        "| 指标 | 原 Demo median | 丝滑版 median | 新/原 |",
        "|---|---:|---:|---:|",
    ]
    for name, ratio in report["ratios"].items():
        metric = name.removesuffix("_ratio")
        lines.append(
            f"| `{metric}` | {report['baseline']['summary'][metric]:.5g} | "
            f"{report['candidate']['summary'][metric]:.5g} | {ratio:.1%} |"
        )
    lines.extend(["", "## 验收", ""])
    for name, passed in report["acceptance"]["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    baseline_items = [metrics(path) for path in args.baseline]
    candidate_items = [metrics(path) for path in args.candidate]
    baseline_summary = summarize(baseline_items)
    candidate_summary = summarize(candidate_items)
    ratios = {
        f"{name}_ratio": candidate_summary[name] / baseline_summary[name]
        for name in baseline_summary
    }
    checks = {
        "two_60_second_runs": all(item["steps"] == 2223 for item in candidate_items),
        "linear_acceleration_reduced_at_least_35pct": ratios[
            "linear_acceleration_p95_mps2_ratio"
        ] <= .65,
        "linear_jerk_reduced_at_least_50pct": ratios[
            "linear_jerk_p95_mps3_ratio"
        ] <= .50,
        "yaw_acceleration_reduced_at_least_35pct": ratios[
            "yaw_acceleration_p95_radps2_ratio"
        ] <= .65,
        "yaw_jerk_reduced_at_least_50pct": ratios[
            "yaw_jerk_p95_radps3_ratio"
        ] <= .50,
        "linear_acceleration_below_human_validation_p90": candidate_summary[
            "linear_acceleration_p95_mps2"
        ] <= 2.0542,
        "linear_jerk_below_human_validation_p90": candidate_summary[
            "linear_jerk_p95_mps3"
        ] <= 32.6643,
    }
    report = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "success",
        "baseline": {"runs": baseline_items, "summary": baseline_summary},
        "candidate": {"runs": candidate_items, "summary": candidate_summary},
        "ratios": ratios,
        "human_validation_reference_p90": {
            "linear_acceleration_p95_mps2": 2.0542,
            "linear_jerk_p95_mps3": 32.6643,
        },
        "acceptance": {"checks": checks, "passed": bool(all(checks.values()))},
    }
    atomic_json(args.output_json, report)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_md.with_suffix(args.output_md.suffix + ".tmp")
    temporary.write_text(markdown(report), encoding="utf-8")
    temporary.replace(args.output_md)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--baseline", nargs="+", type=Path, default=list(DEFAULT_BASELINE))
    result.add_argument("--candidate", nargs="+", type=Path, default=list(DEFAULT_CANDIDATE))
    result.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    result.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    return result


def main() -> int:
    args = parser().parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "passed": report["acceptance"]["passed"],
                "baseline": report["baseline"]["summary"],
                "candidate": report["candidate"]["summary"],
                "ratios": report["ratios"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if report["acceptance"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
