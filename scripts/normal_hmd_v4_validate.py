#!/usr/bin/env python3
"""Validate 60-second Normal HMD v4 rollouts against frozen validation bands.

This is a distributional generation check.  A sampled future has no unique
frame-wise ground truth, so this script deliberately does not report onset
classification accuracy.  Predictive hazard metrics live in the separate v4
evaluation report.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from normal_hmd_common import STATE_FEATURE_NAMES, atomic_json, sha256_file
from normal_hmd_event_axis_analysis import Sequence, make_labels
from normal_hmd_v4_model import NormalHMDV4


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "normal_hmd_v4.pth"
DEFAULT_DATA = ROOT / "data" / "processed" / "normal_hmd_v2"
DEFAULT_JSON = ROOT / "results" / "normal_hmd_v4_rollout_validation.json"
DEFAULT_MD = ROOT / "results" / "normal_hmd_v4_rollout_validation.md"
DT_SECONDS = 0.027
ROLLOUT_STEPS = 2223
HEAD_INDICES = (3, 4, 5, 6, 7, 8)
VALIDATION_BANDS = {
    "horizontal_occupancy": (0.238, 0.767),
    "vertical_occupancy": (0.207, 0.777),
    "vertical_horizontal_ratio": (0.573, 1.388),
    "overlap_lift": (0.892, 1.405),
    "horizontal_rate_per_minute": (14.61, 34.03),
    "vertical_rate_per_minute": (20.98, 49.16),
    "root_distance_m": (4.82, 33.34),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_checkpoint(
    path: Path,
    device: torch.device,
    *,
    horizontal_duration_scale: float,
    vertical_duration_scale: float,
    roll_limit_rad: float,
) -> tuple[NormalHMDV4, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    normalization = checkpoint["normalization"]
    bounds = checkpoint["bounds"]
    correlation = checkpoint["parameter_correlation"]
    model = NormalHMDV4(
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        state_mean=normalization["state_mean"],
        state_std=normalization["state_std"],
        horizontal_parameter_center=normalization["horizontal_parameter_center"],
        horizontal_parameter_scale=normalization["horizontal_parameter_scale"],
        vertical_parameter_center=normalization["vertical_parameter_center"],
        vertical_parameter_scale=normalization["vertical_parameter_scale"],
        horizontal_parameter_low=bounds["horizontal_parameter_p05_p95"][0],
        horizontal_parameter_high=bounds["horizontal_parameter_p05_p95"][1],
        vertical_parameter_low=bounds["vertical_parameter_p05_p95"][0],
        vertical_parameter_high=bounds["vertical_parameter_p05_p95"][1],
        horizontal_pose_low=bounds["horizontal_absolute_pose_p01_p99"][0],
        horizontal_pose_high=bounds["horizontal_absolute_pose_p01_p99"][1],
        vertical_pose_low=bounds["vertical_absolute_pose_p01_p99"][0],
        vertical_pose_high=bounds["vertical_absolute_pose_p01_p99"][1],
        horizontal_parameter_correlation=correlation["horizontal"],
        vertical_parameter_correlation=correlation["vertical"],
        cooldown_steps=int(config["cooldown_steps"]),
        minimum_duration_steps=int(config["duration_bounds_steps"][0]),
        maximum_duration_steps=int(config["duration_bounds_steps"][1]),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    # Deployment calibration is selected on validation rollouts.  The learned
    # duration is a hysteresis-event duration, while a quintic transition only
    # exceeds the diagnostic velocity threshold in its middle.  These frozen
    # scale factors compensate that representation mismatch and also make the
    # visible motion slower.  They do not add per-frame noise.
    model.horizontal_parameter_center[-1] += math.log(horizontal_duration_scale)
    model.vertical_parameter_center[-1] += math.log(vertical_duration_scale)
    # A comfort envelope keeps the horizon stable.  It is tighter than the
    # dataset P01/P99 safety envelope and must also be applied by Unity.
    model.horizontal_parameter_low[3] = max(
        float(model.horizontal_parameter_low[3]), -roll_limit_rad
    )
    model.horizontal_parameter_high[3] = min(
        float(model.horizontal_parameter_high[3]), roll_limit_rad
    )
    model.horizontal_pose_low[3] = max(float(model.horizontal_pose_low[3]), -roll_limit_rad)
    model.horizontal_pose_high[3] = min(float(model.horizontal_pose_high[3]), roll_limit_rad)
    return model, checkpoint


def choose_histories(archive: Any, maximum: int) -> tuple[np.ndarray, list[int], list[int]]:
    trajectory = np.asarray(archive["trajectory_index"], dtype=np.int64)
    selected: list[int] = []
    trajectory_ids: list[int] = []
    for value in np.unique(trajectory):
        rows = np.flatnonzero(trajectory == value)
        if len(rows):
            selected.append(int(rows[len(rows) // 2]))
            trajectory_ids.append(int(value))
    if maximum > 0 and len(selected) > maximum:
        positions = np.linspace(0, len(selected) - 1, maximum, dtype=np.int64)
        selected = [selected[int(index)] for index in positions]
        trajectory_ids = [trajectory_ids[int(index)] for index in positions]
    return np.asarray(archive["X"])[selected], selected, trajectory_ids


def rising_count(mask: np.ndarray) -> int:
    values = np.asarray(mask, dtype=bool)
    if not len(values):
        return 0
    return int(values[0]) + int(np.sum(values[1:] & ~values[:-1]))


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 1e-12 else None


def dynamics(values: np.ndarray, dt: float) -> dict[str, float | None]:
    velocity = np.diff(values) / dt
    acceleration = np.diff(velocity) / dt
    jerk = np.diff(acceleration) / dt
    lag = None
    if len(velocity) > 2 and np.std(velocity[:-1]) > 1e-12 and np.std(velocity[1:]) > 1e-12:
        lag = float(np.corrcoef(velocity[:-1], velocity[1:])[0, 1])
    reversals = int(np.sum(velocity[1:] * velocity[:-1] < 0.0)) if len(velocity) > 1 else 0
    duration = max((len(values) - 1) * dt, dt)
    return {
        "velocity_lag1": lag,
        "direction_reversals_per_second": reversals / duration,
        "absolute_acceleration_p95": float(np.quantile(np.abs(acceleration), 0.95)) if len(acceleration) else 0.0,
        "absolute_jerk_p95": float(np.quantile(np.abs(jerk), 0.95)) if len(jerk) else 0.0,
    }


def segment_ratios(state: np.ndarray, thresholds: dict[str, Any], frames: int) -> dict[str, float | None]:
    first = state[:frames]
    last = state[-frames:]
    result: dict[str, float | None] = {}
    for label, values in (("first", first), ("last", last)):
        labelled = make_labels(Sequence(label, values, DT_SECONDS), thresholds)
        result[f"{label}_horizontal_rate"] = rising_count(labelled["masks"]["horizontal_any"])
        result[f"{label}_vertical_rate"] = rising_count(labelled["masks"]["vertical_any"])
        result[f"{label}_pose_variance"] = float(np.mean(np.var(values[:, HEAD_INDICES], axis=0)))
        result[f"{label}_delta_rms"] = float(np.sqrt(np.mean(np.square(np.diff(values[:, HEAD_INDICES], axis=0)))))
    for name in ("horizontal_rate", "vertical_rate", "pose_variance", "delta_rms"):
        result[f"last_first_{name}_ratio"] = safe_ratio(
            float(result[f"last_{name}"]), float(result[f"first_{name}"])
        )
    return result


def evaluate_rollout(
    state: np.ndarray,
    diagnostics: list[Any],
    batch_index: int,
    thresholds: dict[str, Any],
    bounds: dict[int, tuple[float, float]],
    seed: int,
    row: int,
    trajectory: int,
) -> dict[str, Any]:
    labelled = make_labels(Sequence(str(seed), state, DT_SECONDS), thresholds)
    horizontal = labelled["masks"]["horizontal_any"]
    vertical = labelled["masks"]["vertical_any"]
    horizontal_occupancy = float(np.mean(horizontal))
    vertical_occupancy = float(np.mean(vertical))
    overlap = float(np.mean(horizontal & vertical))
    minutes = len(state) * DT_SECONDS / 60.0
    h_triggers = int(sum(bool(item.horizontal_trigger[batch_index]) for item in diagnostics))
    v_triggers = int(sum(bool(item.vertical_trigger[batch_index]) for item in diagnostics))
    boundary: dict[str, float] = {}
    for index in HEAD_INDICES:
        low, high = bounds[index]
        margin = 0.005 * (high - low)
        boundary[STATE_FEATURE_NAMES[index]] = float(
            np.mean((state[:, index] <= low + margin) | (state[:, index] >= high - margin))
        )
    axis_dynamics = {
        STATE_FEATURE_NAMES[index]: dynamics(state[:, index], DT_SECONDS)
        for index in HEAD_INDICES
    }
    return {
        "seed": seed,
        "npz_row": row,
        "trajectory_index": trajectory,
        "finite": bool(np.isfinite(state).all()),
        "steps": int(len(state)),
        "controller_horizontal_events": h_triggers,
        "controller_vertical_events": v_triggers,
        "horizontal_occupancy": horizontal_occupancy,
        "vertical_occupancy": vertical_occupancy,
        "vertical_horizontal_ratio": safe_ratio(vertical_occupancy, horizontal_occupancy),
        "overlap_fraction": overlap,
        "overlap_lift": safe_ratio(overlap, horizontal_occupancy * vertical_occupancy),
        "horizontal_rate_per_minute": rising_count(horizontal) / minutes,
        "vertical_rate_per_minute": rising_count(vertical) / minutes,
        "root_distance_m": float(np.sum(np.hypot(state[:, 0], state[:, 1])) * DT_SECONDS),
        "maximum_absolute_roll_degrees": float(np.degrees(np.max(np.abs(state[:, 8])))),
        "near_safety_boundary_fraction": boundary,
        "maximum_near_safety_boundary_fraction": max(boundary.values()),
        "stationarity": segment_ratios(state, thresholds, min(741, len(state) // 3)),
        "head_axis_dynamics": axis_dynamics,
    }


def summarize(rollouts: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    for name, (low, high) in VALIDATION_BANDS.items():
        values = np.asarray([item[name] for item in rollouts], dtype=np.float64)
        fraction = float(np.mean((values >= low) & (values <= high)))
        median = float(np.median(values))
        metrics[name] = {
            "median": median,
            "p10": float(np.quantile(values, 0.10)),
            "p90": float(np.quantile(values, 0.90)),
            "band": [low, high],
            "fraction_inside_band": fraction,
        }
        gates[name] = bool(low <= median <= high and fraction >= 0.60)
    zero_h = float(np.mean([item["controller_horizontal_events"] == 0 for item in rollouts]))
    zero_v = float(np.mean([item["controller_vertical_events"] == 0 for item in rollouts]))
    max_boundary = max(item["maximum_near_safety_boundary_fraction"] for item in rollouts)
    max_roll = max(item["maximum_absolute_roll_degrees"] for item in rollouts)
    finite_steps = all(item["finite"] and item["steps"] == ROLLOUT_STEPS for item in rollouts)
    stationarity_values: list[float] = []
    for item in rollouts:
        for key, value in item["stationarity"].items():
            if key.startswith("last_first_") and value is not None and math.isfinite(value):
                stationarity_values.append(float(value))
    # Event counts can be zero in one 20-second third; aggregate ratios are a
    # diagnostic and only finite ratios take part in this provisional gate.
    stationarity_fraction = float(np.mean([
        0.5 <= value <= 2.0 for value in stationarity_values
    ])) if stationarity_values else 0.0
    gates.update({
        "zero_horizontal_seed_fraction_below_5pct": zero_h < 0.05,
        "zero_vertical_seed_fraction_below_5pct": zero_v < 0.05,
        "near_safety_boundary_below_1pct": max_boundary < 0.01,
        "maximum_roll_below_10deg": max_roll <= 10.0,
        "finite_exact_2223_steps": finite_steps,
        "stationarity_ratio_fraction_at_least_60pct": stationarity_fraction >= 0.60,
    })
    metrics["zero_horizontal_seed_fraction"] = zero_h
    metrics["zero_vertical_seed_fraction"] = zero_v
    metrics["maximum_near_safety_boundary_fraction"] = max_boundary
    metrics["maximum_absolute_roll_degrees"] = max_roll
    metrics["stationarity_ratio_fraction_in_0.5_to_2"] = stationarity_fraction
    return metrics, {"checks": gates, "passed": bool(all(gates.values()))}


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Normal HMD Agent v4 — 60 秒闭环分布验收",
        "",
        f"状态：**{'通过' if report['acceptance']['passed'] else '未通过'}**",
        "",
        f"模型 SHA-256：`{report['model_sha256']}`",
        "",
        f"Validation rollouts：{len(report['rollouts'])} × {report['steps']} steps。",
        "",
        "随机生成没有唯一正确 future；这里比较分布，不把 onset F1 当作生成质量。",
        "",
        "| 指标 | 生成 median | 真人 validation band | band 内 rollout |",
        "|---|---:|---:|---:|",
    ]
    for name in VALIDATION_BANDS:
        item = report["summary"][name]
        lines.append(
            f"| `{name}` | {item['median']:.4g} | "
            f"[{item['band'][0]:.4g}, {item['band'][1]:.4g}] | "
            f"{item['fraction_inside_band']:.1%} |"
        )
    lines.extend(["", "## 硬门槛", ""])
    for name, passed in report["acceptance"]["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
    lines.extend([
        "",
        "说明：frozen v2 NPZ 只能恢复无重复连续片段，不能声称恢复了 raw EyeNavGS 的所有完整 run；片段边界事件已删失。",
        "",
    ])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    model, checkpoint = load_checkpoint(
        args.model,
        device,
        horizontal_duration_scale=args.horizontal_duration_scale,
        vertical_duration_scale=args.vertical_duration_scale,
        roll_limit_rad=args.roll_limit_rad,
    )
    with np.load(args.data_dir / "normal_hmd_val.npz", allow_pickle=False) as archive:
        histories, rows, trajectories = choose_histories(archive, args.max_seeds)
    mean = np.asarray(checkpoint["normalization"]["state_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalization"]["state_std"], dtype=np.float32)
    normalized = torch.from_numpy(((histories.astype(np.float32) - mean) / std)).to(device)
    normalized[:, -1, 8] = torch.clamp(
        normalized[:, -1, 8],
        (-args.roll_limit_rad - float(mean[8])) / float(std[8]),
        (args.roll_limit_rad - float(mean[8])) / float(std[8]),
    )
    generated, diagnostics = model.runtime_rollout(
        normalized,
        ROLLOUT_STEPS,
        seed=args.seed,
        root_noise_std_normalized=args.root_noise_std,
        root_noise_rho=args.root_noise_rho,
    )
    physical = generated.cpu().numpy() * std[None, None, :] + mean[None, None, :]
    horizontal_bounds = checkpoint["bounds"]["horizontal_absolute_pose_p01_p99"]
    vertical_bounds = checkpoint["bounds"]["vertical_absolute_pose_p01_p99"]
    bounds: dict[int, tuple[float, float]] = {}
    for offset, index in enumerate((3, 5, 7, 8)):
        bounds[index] = (float(horizontal_bounds[0][offset]), float(horizontal_bounds[1][offset]))
    for offset, index in enumerate((4, 6)):
        bounds[index] = (float(vertical_bounds[0][offset]), float(vertical_bounds[1][offset]))
    rollouts = [
        evaluate_rollout(
            physical[index], diagnostics, index,
            checkpoint["event_definition"]["thresholds"], bounds,
            args.seed + index, rows[index], trajectories[index],
        )
        for index in range(len(physical))
    ]
    summary, acceptance = summarize(rollouts)
    report = {
        "schema_version": "4.0.0",
        "generated_at_utc": utc_now(),
        "status": "success",
        "evaluation_split": "validation",
        "model": str(args.model.resolve()),
        "model_sha256": sha256_file(args.model),
        "steps": ROLLOUT_STEPS,
        "seconds": ROLLOUT_STEPS * DT_SECONDS,
        "root_noise_std_normalized": args.root_noise_std,
        "root_noise_rho": args.root_noise_rho,
        "horizontal_duration_scale": args.horizontal_duration_scale,
        "vertical_duration_scale": args.vertical_duration_scale,
        "roll_limit_rad": args.roll_limit_rad,
        "seed": args.seed,
        "validation_bands_status": "provisional bands from reconstructed validation fragments",
        "summary": summary,
        "acceptance": acceptance,
        "rollouts": rollouts,
    }
    atomic_json(args.output_json, report)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_md.with_suffix(args.output_md.suffix + ".tmp")
    temporary.write_text(markdown(report), encoding="utf-8")
    temporary.replace(args.output_md)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    result.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    result.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    result.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    result.add_argument("--max-seeds", type=int, default=16)
    result.add_argument("--seed", type=int, default=20261012)
    result.add_argument("--root-noise-std", type=float, default=0.15)
    result.add_argument("--root-noise-rho", type=float, default=0.95)
    result.add_argument("--horizontal-duration-scale", type=float, default=1.0)
    result.add_argument("--vertical-duration-scale", type=float, default=1.0)
    result.add_argument("--roll-limit-rad", type=float, default=0.12)
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--smoke", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.smoke:
        args.max_seeds = min(args.max_seeds, 2)
        args.output_json = ROOT / "tmp" / "normal_hmd_v4_validation_smoke.json"
        args.output_md = ROOT / "tmp" / "normal_hmd_v4_validation_smoke.md"
    report = run(args)
    print(json.dumps({
        "passed": report["acceptance"]["passed"],
        "rollouts": len(report["rollouts"]),
        "json": str(args.output_json.resolve()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
