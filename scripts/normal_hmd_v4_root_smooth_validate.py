#!/usr/bin/env python3
"""Compare BodyRoot smoothness before and after v4 Root fine-tuning."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from normal_hmd_common import atomic_json, sha256_file
from normal_hmd_v4_data import load_split, reconstruct_trajectories
from normal_hmd_v4_validate import choose_histories, load_checkpoint


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "models" / "normal_hmd_v4.pth"
DEFAULT_CANDIDATE = ROOT / "models" / "normal_hmd_v4_smooth_root.pth"
DEFAULT_DATA = ROOT / "data" / "processed" / "normal_hmd_v2"
DEFAULT_JSON = ROOT / "results" / "normal_hmd_v4_smooth_root_validation.json"
DEFAULT_MD = ROOT / "results" / "normal_hmd_v4_smooth_root_validation.md"
DT = 0.027
STEPS = 2223


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def lag1(values: np.ndarray) -> float | None:
    if len(values) < 3:
        return None
    left = values[:-1]
    right = values[1:]
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def high_frequency_fraction(values: np.ndarray, cutoff_hz: float = 2.0) -> float:
    centered = np.asarray(values, dtype=np.float64) - float(np.mean(values))
    if len(centered) < 8:
        return 0.0
    power = np.abs(np.fft.rfft(centered)) ** 2
    frequencies = np.fft.rfftfreq(len(centered), d=DT)
    total = float(np.sum(power[1:]))
    if total <= 1e-18:
        return 0.0
    return float(np.sum(power[frequencies >= cutoff_hz]) / total)


def trajectory_metrics(state: np.ndarray) -> dict[str, float | None]:
    root = np.asarray(state[:, :3], dtype=np.float64)
    linear_velocity = root[:, :2]
    speed = np.linalg.norm(linear_velocity, axis=1)
    linear_acceleration = np.diff(linear_velocity, axis=0) / DT
    linear_acceleration_magnitude = np.linalg.norm(linear_acceleration, axis=1)
    linear_jerk = np.diff(linear_acceleration, axis=0) / DT
    linear_jerk_magnitude = np.linalg.norm(linear_jerk, axis=1)
    yaw_acceleration = np.diff(root[:, 2]) / DT
    yaw_jerk = np.diff(yaw_acceleration) / DT

    direction_change: list[float] = []
    for previous, current in zip(linear_velocity[:-1], linear_velocity[1:]):
        product = float(np.linalg.norm(previous) * np.linalg.norm(current))
        if product > 1e-6:
            cosine = float(np.clip(np.dot(previous, current) / product, -1.0, 1.0))
            direction_change.append(math.acos(cosine))

    return {
        "distance_m": float(np.sum(speed) * DT),
        "speed_mean_mps": float(np.mean(speed)),
        "speed_p95_mps": float(np.quantile(speed, 0.95)),
        "root_right_velocity_lag1": lag1(root[:, 0]),
        "root_forward_velocity_lag1": lag1(root[:, 1]),
        "root_yaw_rate_lag1": lag1(root[:, 2]),
        "linear_acceleration_rms_mps2": float(np.sqrt(np.mean(linear_acceleration_magnitude**2))),
        "linear_acceleration_p95_mps2": float(np.quantile(linear_acceleration_magnitude, 0.95)),
        "linear_jerk_rms_mps3": float(np.sqrt(np.mean(linear_jerk_magnitude**2))),
        "linear_jerk_p95_mps3": float(np.quantile(linear_jerk_magnitude, 0.95)),
        "yaw_acceleration_rms_radps2": float(np.sqrt(np.mean(yaw_acceleration**2))),
        "yaw_acceleration_p95_radps2": float(np.quantile(np.abs(yaw_acceleration), 0.95)),
        "yaw_jerk_rms_radps3": float(np.sqrt(np.mean(yaw_jerk**2))),
        "yaw_jerk_p95_radps3": float(np.quantile(np.abs(yaw_jerk), 0.95)),
        "direction_change_p95_degrees": float(np.degrees(np.quantile(direction_change, 0.95))) if direction_change else 0.0,
        "right_high_frequency_power_fraction": high_frequency_fraction(root[:, 0]),
        "forward_high_frequency_power_fraction": high_frequency_fraction(root[:, 1]),
        "yaw_high_frequency_power_fraction": high_frequency_fraction(root[:, 2]),
    }


def summarize(items: list[dict[str, float | None]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name in items[0]:
        values = np.asarray(
            [item[name] for item in items if item[name] is not None], dtype=np.float64
        )
        result[name] = {
            "median": float(np.median(values)),
            "p10": float(np.quantile(values, 0.10)),
            "p90": float(np.quantile(values, 0.90)),
        }
    return result


@torch.no_grad()
def generate_variant(
    model_path: Path,
    histories: np.ndarray,
    seed: int,
    root_noise_std: float,
    root_noise_rho: float,
    device: torch.device,
) -> tuple[list[dict[str, float | None]], dict[str, Any]]:
    model, checkpoint = load_checkpoint(
        model_path,
        device,
        horizontal_duration_scale=1.0,
        vertical_duration_scale=1.0,
        roll_limit_rad=0.12,
    )
    mean = np.asarray(checkpoint["normalization"]["state_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalization"]["state_std"], dtype=np.float32)
    normalized = torch.from_numpy(
        ((histories.astype(np.float32) - mean) / std).astype(np.float32)
    ).to(device)
    generated, _ = model.runtime_rollout(
        normalized,
        STEPS,
        seed=seed,
        root_noise_std_normalized=root_noise_std,
        root_noise_rho=root_noise_rho,
    )
    physical = generated.cpu().numpy() * std[None, None, :] + mean[None, None, :]
    items = [trajectory_metrics(value) for value in physical]
    return items, {
        "model": str(model_path.resolve()),
        "model_sha256": sha256_file(model_path),
        "root_noise_std_normalized": root_noise_std,
        "root_noise_rho": root_noise_rho,
        "summary": summarize(items),
        "rollouts": items,
    }


def human_reference(data_dir: Path) -> dict[str, dict[str, float]]:
    split = load_split(data_dir, "val")
    fragments = reconstruct_trajectories(split)
    metrics = [trajectory_metrics(fragment.state) for fragment in fragments if len(fragment.state) >= 16]
    return summarize(metrics)


def ratio(candidate: dict[str, Any], baseline: dict[str, Any], name: str) -> float:
    denominator = float(baseline["summary"][name]["median"])
    return float(candidate["summary"][name]["median"] / denominator) if denominator > 1e-12 else math.inf


def markdown(report: dict[str, Any]) -> str:
    names = (
        "distance_m",
        "linear_acceleration_p95_mps2",
        "linear_jerk_p95_mps3",
        "yaw_acceleration_p95_radps2",
        "yaw_jerk_p95_radps3",
        "direction_change_p95_degrees",
        "right_high_frequency_power_fraction",
        "forward_high_frequency_power_fraction",
        "yaw_high_frequency_power_fraction",
    )
    labels = list(report["variants"])
    lines = [
        "# Normal HMD v4 Root 丝滑度对照",
        "",
        f"状态：**{'PASS' if report['acceptance']['passed'] else 'FAIL'}**",
        "",
        "| 指标 | " + " | ".join(labels) + " |",
        "|---|" + "---:|" * len(labels),
    ]
    for name in names:
        values = [report["variants"][label]["summary"][name]["median"] for label in labels]
        lines.append("| `" + name + "` | " + " | ".join(f"{value:.5g}" for value in values) + " |")
    lines.extend(["", "## 验收", ""])
    for name, passed in report["acceptance"]["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    with np.load(args.data_dir / "normal_hmd_val.npz", allow_pickle=False) as archive:
        histories, rows, trajectories = choose_histories(archive, args.max_seeds)

    definitions = {
        # The shipped asset used stationary std .15 with default scale .9, so
        # the effective normalized deployment standard deviation was .135.
        "baseline_production": (args.baseline_model, 0.135, 0.95),
        "baseline_no_noise": (args.baseline_model, 0.0, 0.95),
        "baseline_smooth_runtime": (
            args.baseline_model,
            args.smooth_noise_std,
            args.smooth_noise_rho,
        ),
        "candidate_same_noise": (args.candidate_model, 0.15, 0.95),
        "candidate_no_noise": (args.candidate_model, 0.0, 0.95),
        "candidate_smooth_runtime": (
            args.candidate_model,
            args.smooth_noise_std,
            args.smooth_noise_rho,
        ),
    }
    variants: dict[str, Any] = {}
    for name, (path, noise_std, noise_rho) in definitions.items():
        _, variants[name] = generate_variant(
            path,
            histories,
            args.seed,
            noise_std,
            noise_rho,
            device,
        )

    baseline = variants["baseline_production"]
    baseline_no_noise = variants["baseline_no_noise"]
    baseline_smooth = variants["baseline_smooth_runtime"]
    same_noise = variants["candidate_same_noise"]
    candidate_no_noise = variants["candidate_no_noise"]
    smooth = variants["candidate_smooth_runtime"]
    human = human_reference(args.data_dir)
    fine_tune_checks = {
        "trained_candidate_reduces_linear_jerk_same_noise": ratio(
            same_noise, baseline, "linear_jerk_p95_mps3"
        ) < 0.95,
        "trained_candidate_reduces_yaw_jerk_same_noise": ratio(
            same_noise, baseline, "yaw_jerk_p95_radps3"
        ) < 0.95,
        "trained_candidate_no_noise_not_rougher_than_parent": (
            ratio(candidate_no_noise, baseline_no_noise, "linear_jerk_p95_mps3")
            <= 1.05
            and ratio(candidate_no_noise, baseline_no_noise, "yaw_jerk_p95_radps3")
            <= 1.05
        ),
    }
    checks = {
        "deployment_linear_jerk_at_least_35pct_lower": ratio(
            baseline_smooth, baseline, "linear_jerk_p95_mps3"
        ) <= 0.65,
        "deployment_yaw_jerk_at_least_35pct_lower": ratio(
            baseline_smooth, baseline, "yaw_jerk_p95_radps3"
        ) <= 0.65,
        "deployment_distance_in_human_band": 4.82
        <= baseline_smooth["summary"]["distance_m"]["median"]
        <= 33.34,
        "deployment_linear_acceleration_below_human_p90": (
            baseline_smooth["summary"]["linear_acceleration_p95_mps2"]["median"]
            <= human["linear_acceleration_p95_mps2"]["p90"]
        ),
        "deployment_linear_jerk_below_human_p90": (
            baseline_smooth["summary"]["linear_jerk_p95_mps3"]["median"]
            <= human["linear_jerk_p95_mps3"]["p90"]
        ),
        "all_outputs_finite": all(
            math.isfinite(float(item[name]))
            for variant in variants.values()
            for item in variant["rollouts"]
            for name in item
            if item[name] is not None
        ),
    }
    report = {
        "schema_version": "1.0.0",
        "generated_at_utc": utc_now(),
        "status": "success",
        "evaluation_split": "validation",
        "histories": len(histories),
        "history_rows": rows,
        "trajectory_indices": trajectories,
        "steps": STEPS,
        "seconds": STEPS * DT,
        "human_validation_fragments": human,
        "variants": variants,
        "ratios_vs_baseline_production": {
            "candidate_same_noise_linear_jerk": ratio(
                same_noise, baseline, "linear_jerk_p95_mps3"
            ),
            "candidate_same_noise_yaw_jerk": ratio(
                same_noise, baseline, "yaw_jerk_p95_radps3"
            ),
            "candidate_smooth_runtime_linear_jerk": ratio(
                smooth, baseline, "linear_jerk_p95_mps3"
            ),
            "candidate_smooth_runtime_yaw_jerk": ratio(
                smooth, baseline, "yaw_jerk_p95_radps3"
            ),
            "baseline_smooth_runtime_linear_jerk": ratio(
                baseline_smooth, baseline, "linear_jerk_p95_mps3"
            ),
            "baseline_smooth_runtime_yaw_jerk": ratio(
                baseline_smooth, baseline, "yaw_jerk_p95_radps3"
            ),
        },
        "fine_tune_candidate": {
            "accepted": bool(all(fine_tune_checks.values())),
            "checks": fine_tune_checks,
            "decision": "reject; do not export to Unity unless every fine-tune check passes",
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
    result.add_argument("--baseline-model", type=Path, default=DEFAULT_BASELINE)
    result.add_argument("--candidate-model", type=Path, default=DEFAULT_CANDIDATE)
    result.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    result.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    result.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    result.add_argument("--max-seeds", type=int, default=16)
    result.add_argument("--seed", type=int, default=20261013)
    result.add_argument("--smooth-noise-std", type=float, default=0.05)
    result.add_argument("--smooth-noise-rho", type=float, default=0.99)
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--smoke", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.smoke:
        args.max_seeds = min(args.max_seeds, 2)
        args.output_json = ROOT / "tmp" / "normal_hmd_v4_smooth_root_validation_smoke.json"
        args.output_md = ROOT / "tmp" / "normal_hmd_v4_smooth_root_validation_smoke.md"
    report = run(args)
    print(
        json.dumps(
            {
                "passed": report["acceptance"]["passed"],
                "checks": report["acceptance"]["checks"],
                "ratios": report["ratios_vs_baseline_production"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if report["acceptance"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
