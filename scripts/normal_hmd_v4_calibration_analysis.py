#!/usr/bin/env python3
"""Diagnose Normal HMD v4 rollout occupancy without changing the model.

The report compares reconstructed human validation events with the frozen
scale=1.0 rollout policy at three levels:

* frame occupancy under the exact SG+hysteresis event definition;
* detected-event duration and endpoint displacement/rotation;
* latent controller events, including whether a minimum-jerk transition is
  theoretically capable of crossing the diagnostic high threshold.

This is a read-only calibration diagnostic.  It does not modify the checkpoint,
the core model, or Unity assets.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from normal_hmd_common import sha256_file
from normal_hmd_event_axis_analysis import Sequence, make_labels
from normal_hmd_v4_data import (
    HORIZONTAL_INDICES,
    VERTICAL_INDICES,
    label_trajectories,
    load_split,
    reconstruct_trajectories,
)
from normal_hmd_v4_validate import (
    DT_SECONDS,
    ROLLOUT_STEPS,
    VALIDATION_BANDS,
    choose_histories,
    load_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "normal_hmd_v4.pth"
DEFAULT_DATA = ROOT / "data" / "processed" / "normal_hmd_v2"
DEFAULT_REFERENCE = ROOT / "tmp" / "normal_hmd_v4_scale_1.0.json"
DEFAULT_JSON = ROOT / "results" / "normal_hmd_v4_calibration_analysis.json"
DEFAULT_MD = ROOT / "results" / "normal_hmd_v4_calibration_analysis.md"

BRANCHES: dict[str, dict[str, Any]] = {
    "horizontal": {
        "indices": HORIZONTAL_INDICES,
        "signals": ("lateral_sway_speed_mps", "look_turn_rate_radps"),
        "detect_indices": (3, 7),
        "endpoint_metrics": {
            "abs_lateral_endpoint_delta_m": lambda d: abs(d[0]),
            "translation_xz_endpoint_delta_m": lambda d: float(np.hypot(d[0], d[1])),
            "abs_yaw_endpoint_delta_deg": lambda d: abs(math.degrees(d[2])),
            "rotation_yaw_roll_endpoint_delta_deg": lambda d: float(
                np.hypot(math.degrees(d[2]), math.degrees(d[3]))
            ),
        },
    },
    "vertical": {
        "indices": VERTICAL_INDICES,
        "signals": ("vertical_bob_speed_mps", "nod_rate_radps"),
        "detect_indices": (4, 6),
        "endpoint_metrics": {
            "abs_vertical_endpoint_delta_m": lambda d: abs(d[0]),
            "abs_pitch_endpoint_delta_deg": lambda d: abs(math.degrees(d[1])),
        },
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def mask_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    edges = np.diff(np.concatenate(([0], values.astype(np.int8), [0])))
    return [
        (int(start), int(stop))
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
    ]


def quantiles(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "p10": None, "median": None, "p90": None, "mean": None}
    return {
        "count": int(len(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "mean": float(np.mean(array)),
    }


def event_records(
    smooth: np.ndarray,
    mask: np.ndarray,
    branch_name: str,
) -> list[dict[str, float]]:
    definition = BRANCHES[branch_name]
    indices = tuple(definition["indices"])
    records: list[dict[str, float]] = []
    for start, stop in mask_runs(mask):
        # Match v4 parameter supervision: exclude events censored at either
        # reconstructed-fragment/rollout boundary.
        if start <= 0 or stop >= len(smooth):
            continue
        delta = smooth[stop - 1, indices] - smooth[start - 1, indices]
        record: dict[str, float] = {
            "duration_frames": float(stop - start),
            "duration_seconds": float((stop - start) * DT_SECONDS),
        }
        for name, function in definition["endpoint_metrics"].items():
            record[name] = float(function(delta))
        records.append(record)
    return records


def collect_source(
    labelled: Iterable[tuple[np.ndarray, Mapping[str, np.ndarray], Mapping[str, np.ndarray]]],
) -> dict[str, Any]:
    items = list(labelled)
    total_frames = sum(len(smooth) for smooth, _, _ in items)
    total_seconds = total_frames * DT_SECONDS
    result: dict[str, Any] = {"frames": total_frames, "seconds": total_seconds}
    for branch_name, definition in BRANCHES.items():
        signal_a, signal_b = definition["signals"]
        masks = [np.asarray(masks[f"{branch_name}_any"], dtype=bool) for _, _, masks in items]
        high_masks = [
            (np.asarray(signals[signal_a]) >= float(thresholds_global[signal_a]["high"]))
            | (np.asarray(signals[signal_b]) >= float(thresholds_global[signal_b]["high"]))
            for _, signals, _ in items
        ]
        low_masks = [
            (np.asarray(signals[signal_a]) >= float(thresholds_global[signal_a]["low"]))
            | (np.asarray(signals[signal_b]) >= float(thresholds_global[signal_b]["low"]))
            for _, signals, _ in items
        ]
        records: list[dict[str, float]] = []
        run_count = 0
        known_onset_count = 0
        active_frames = 0
        for (smooth, _, _), mask in zip(items, masks):
            runs = mask_runs(mask)
            run_count += len(runs)
            known_onset_count += sum(start > 0 for start, _ in runs)
            active_frames += int(mask.sum())
            records.extend(event_records(smooth, mask, branch_name))
        rate = known_onset_count * 60.0 / max(total_seconds, 1e-12)
        occupancy = active_frames / max(total_frames, 1)
        endpoint_names = tuple(definition["endpoint_metrics"])
        result[branch_name] = {
            "hysteresis_occupancy": float(occupancy),
            "raw_high_threshold_occupancy": float(
                sum(int(value.sum()) for value in high_masks) / max(total_frames, 1)
            ),
            "raw_low_threshold_occupancy": float(
                sum(int(value.sum()) for value in low_masks) / max(total_frames, 1)
            ),
            "event_runs_including_censored": int(run_count),
            "known_onset_count": int(known_onset_count),
            "complete_event_count": int(len(records)),
            "events_per_minute": float(rate),
            "occupancy_implied_mean_event_seconds": float(
                occupancy * 60.0 / rate if rate > 0.0 else 0.0
            ),
            "duration_seconds": quantiles(record["duration_seconds"] for record in records),
            "endpoint": {
                name: quantiles(record[name] for record in records)
                for name in endpoint_names
            },
        }
    return result


def controller_records(
    histories: np.ndarray,
    generated: np.ndarray,
    diagnostics: list[Any],
    generated_labels: list[dict[str, Any]],
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    cooldown = int(checkpoint["model_config"]["cooldown_steps"])
    total_seconds = generated.shape[0] * generated.shape[1] * DT_SECONDS
    report: dict[str, Any] = {}
    for branch_name, definition in BRANCHES.items():
        indices = tuple(definition["indices"])
        signal_names = tuple(definition["signals"])
        trigger_name = f"{branch_name}_trigger"
        eligible_name = f"{branch_name}_eligible"
        records: list[dict[str, Any]] = []
        total_triggers = 0
        for batch_index in range(len(generated)):
            trigger = np.asarray(
                [bool(getattr(item, trigger_name)[batch_index]) for item in diagnostics]
            )
            eligible = np.asarray(
                [bool(getattr(item, eligible_name)[batch_index]) for item in diagnostics]
            )
            total_triggers += int(trigger.sum())
            labels = generated_labels[batch_index]
            diagnostic_mask = np.asarray(labels["masks"][f"{branch_name}_any"], dtype=bool)
            signals = labels["signals"]
            for start in np.flatnonzero(trigger):
                later_eligible = np.flatnonzero(eligible[start + 1 :])
                if not len(later_eligible):
                    continue
                next_eligible = int(start + 1 + later_eligible[0])
                duration = next_eligible - int(start) - cooldown
                if duration < 2 or start + duration > generated.shape[1]:
                    continue
                previous = (
                    histories[batch_index, -1]
                    if start == 0
                    else generated[batch_index, start - 1]
                )
                endpoint = generated[batch_index, start + duration - 1]
                delta = endpoint[list(indices)] - previous[list(indices)]
                transition_seconds = max((duration - 1) * DT_SECONDS, DT_SECONDS)
                theoretical_peaks: dict[str, float] = {}
                for local_offset, signal_name in enumerate(signal_names):
                    # Detection axes are the first/third horizontal branch
                    # coordinates (x/yaw), and both vertical coordinates.
                    state_index = definition["detect_indices"][local_offset]
                    branch_offset = indices.index(state_index)
                    theoretical_peaks[signal_name] = float(
                        1.875 * abs(delta[branch_offset]) / transition_seconds
                    )
                high_capable = any(
                    theoretical_peaks[name] >= float(thresholds_global[name]["high"])
                    for name in signal_names
                )
                low_capable = any(
                    theoretical_peaks[name] >= float(thresholds_global[name]["low"])
                    for name in signal_names
                )
                event_slice = slice(int(start), int(start + duration))
                observed_high = any(
                    bool(np.any(np.asarray(signals[name])[event_slice] >= float(thresholds_global[name]["high"])))
                    for name in signal_names
                )
                record: dict[str, Any] = {
                    "duration_frames": int(duration),
                    "duration_seconds": float(duration * DT_SECONDS),
                    "theoretical_high_capable": bool(high_capable),
                    "theoretical_low_capable": bool(low_capable),
                    "observed_high_crossing": bool(observed_high),
                    "any_hysteresis_frame": bool(np.any(diagnostic_mask[event_slice])),
                    "hysteresis_fraction_during_controller_event": float(
                        np.mean(diagnostic_mask[event_slice])
                    ),
                    "theoretical_peak_to_high_ratio": float(max(
                        theoretical_peaks[name] / float(thresholds_global[name]["high"])
                        for name in signal_names
                    )),
                }
                for name, function in definition["endpoint_metrics"].items():
                    record[name] = float(function(delta))
                records.append(record)
        endpoint_names = tuple(definition["endpoint_metrics"])
        diagnostic_count = sum(
            sum(start > 0 for start, _ in mask_runs(item["masks"][f"{branch_name}_any"]))
            for item in generated_labels
        )
        report[branch_name] = {
            "trigger_count": int(total_triggers),
            "complete_controller_event_count": int(len(records)),
            "triggers_per_minute": float(total_triggers * 60.0 / total_seconds),
            "diagnostic_known_onsets_per_controller_trigger": float(
                diagnostic_count / max(total_triggers, 1)
            ),
            "duration_seconds": quantiles(record["duration_seconds"] for record in records),
            "endpoint": {
                name: quantiles(record[name] for record in records)
                for name in endpoint_names
            },
            "theoretical_high_capable_fraction": float(
                np.mean([record["theoretical_high_capable"] for record in records])
            ),
            "theoretical_low_capable_fraction": float(
                np.mean([record["theoretical_low_capable"] for record in records])
            ),
            "observed_high_crossing_fraction": float(
                np.mean([record["observed_high_crossing"] for record in records])
            ),
            "any_hysteresis_frame_fraction": float(
                np.mean([record["any_hysteresis_frame"] for record in records])
            ),
            "hysteresis_fraction_during_controller_event": quantiles(
                record["hysteresis_fraction_during_controller_event"] for record in records
            ),
            "theoretical_peak_to_high_ratio": quantiles(
                record["theoretical_peak_to_high_ratio"] for record in records
            ),
        }
    return report


def scale_sweep(directory: Path, model_sha256: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("normal_hmd_v4_scale_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("model_sha256") != model_sha256:
            continue
        rows.append({
            "path": str(path.resolve()),
            "horizontal_duration_scale": float(payload["horizontal_duration_scale"]),
            "vertical_duration_scale": float(payload["vertical_duration_scale"]),
            "horizontal_occupancy_median": float(payload["summary"]["horizontal_occupancy"]["median"]),
            "vertical_occupancy_median": float(payload["summary"]["vertical_occupancy"]["median"]),
            "horizontal_rate_per_minute_median": float(payload["summary"]["horizontal_rate_per_minute"]["median"]),
            "vertical_rate_per_minute_median": float(payload["summary"]["vertical_rate_per_minute"]["median"]),
        })
    return rows


def generated_velocity_lag1(generated: np.ndarray) -> dict[str, Any]:
    names = (
        "head_local_right_m",
        "head_local_up_offset_m",
        "head_local_forward_m",
        "head_local_rotvec_x_rad_rh",
        "head_local_rotvec_y_rad_rh",
        "head_local_rotvec_z_rad_rh",
    )
    result: dict[str, Any] = {}
    medians: list[float] = []
    for offset, name in enumerate(names, start=3):
        values: list[float] = []
        for rollout in generated:
            velocity = np.diff(rollout[:, offset]) / DT_SECONDS
            if (
                len(velocity) > 2
                and np.std(velocity[:-1]) > 1e-12
                and np.std(velocity[1:]) > 1e-12
            ):
                values.append(float(np.corrcoef(velocity[:-1], velocity[1:])[0, 1]))
        result[name] = quantiles(values)
        if result[name]["median"] is not None:
            medians.append(float(result[name]["median"]))
    result["axis_median_range"] = [min(medians), max(medians)] if medians else None
    return result


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def markdown(report: Mapping[str, Any]) -> str:
    real = report["comparison"]["real_validation"]
    generated = report["comparison"]["generated_scale_1_0"]
    controller = report["controller_scale_1_0"]
    lines = [
        "# Normal HMD v4 生成校准分析",
        "",
        f"生成时间：`{report['generated_at_utc']}`",
        "",
        "## 结论",
        "",
        "- **保持 duration scale = 1.0。** 已有的 1.0/1.4/1.8/2.2 同 checkpoint 扫描表明，拉长最小加加速轨迹会降低峰值速度、延长控制器 busy 时间，使占空比和事件率更低，不是修复方向。",
        "- **不先上调 hazard 概率。** scale=1.0 的控制器触发率已接近或高于真人 validation 事件率；主要损失发生在“触发→速度过阈值”的可见性转换。",
        "- **本轮不启用 visibility gate 或全局幅度下限。** controller 的时长和终点幅度中位数已接近真人；强制每次触发都跨过 high threshold 会删除真人也存在的微动作，并把幅度分布条件化偏大。visibility gate 仅作后续离线 ablation，不改 runtime。",
        "- 当前 scale=1.0 是最低风险部署基线：保持现有概率、有界终点抽样、minimum-jerk 和无逐帧 HeadLocal 噪声。occupancy 低说明“真人 hysteresis duration →单段 minimum-jerk →同一检测器重检测”的表示不对齐，不等于轨迹有逐帧抖动，但也不应宣称已通过人类 occupancy band。",
        "",
        "## 帧级与事件级对比",
        "",
        "| 分支 | 来源 | hysteresis occupancy | raw high 过阈占空比 | 事件/分钟 | occupancy 反推平均事件秒数 | 完整事件时长中位数 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for branch in ("horizontal", "vertical"):
        for label, source in (("真人 validation", real), ("scale=1.0 生成", generated)):
            item = source[branch]
            lines.append(
                f"| {branch} | {label} | {fmt(item['hysteresis_occupancy'])} | "
                f"{fmt(item['raw_high_threshold_occupancy'])} | {fmt(item['events_per_minute'])} | "
                f"{fmt(item['occupancy_implied_mean_event_seconds'])} | "
                f"{fmt(item['duration_seconds']['median'])} |"
            )
    lines.extend([
        "",
        "注：occupancy 是原验收脚本的 SG+hysteresis 定义；`raw high` 是平滑导数直接超过 high threshold 的帧比例。",
        "",
        "## 终点幅度（完整事件，中位数 [P10, P90]）",
        "",
        "| 分支 | 指标 | 真人 validation | scale=1.0 诊断事件 | scale=1.0 控制器事件 |",
        "|---|---|---:|---:|---:|",
    ])
    for branch in ("horizontal", "vertical"):
        for name in BRANCHES[branch]["endpoint_metrics"]:
            cells = []
            for source in (real[branch]["endpoint"], generated[branch]["endpoint"], controller[branch]["endpoint"]):
                item = source[name]
                cells.append(f"{fmt(item['median'])} [{fmt(item['p10'])}, {fmt(item['p90'])}]")
            lines.append(f"| {branch} | `{name}` | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "## 控制器可见性",
        "",
        "| 分支 | 触发/分钟 | 真人事件/分钟 | 诊断 onset / 触发 | 理论可达 high | 实测达 high | 触发期间有 hysteresis 帧 | 控制器时长中位数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for branch in ("horizontal", "vertical"):
        item = controller[branch]
        lines.append(
            f"| {branch} | {fmt(item['triggers_per_minute'])} | {fmt(real[branch]['events_per_minute'])} | "
            f"{fmt(item['diagnostic_known_onsets_per_controller_trigger'])} | "
            f"{fmt(item['theoretical_high_capable_fraction'])} | {fmt(item['observed_high_crossing_fraction'])} | "
            f"{fmt(item['any_hysteresis_frame_fraction'])} | {fmt(item['duration_seconds']['median'])} |"
        )
    lines.extend([
        "",
        "minimum-jerk 的理论峰值速度为 `1.875 * |Δ| / ((D-1)*dt)`。因此拉长 D 而不同时增大终点幅度，会把更多事件压到阈值下。",
        "",
        "## 平滑性复核",
        "",
        "| HeadLocal 轴 | scale=1.0 velocity lag-1 中位数 |",
        "|---|---:|",
    ])
    for name, item in report["smoothness_scale_1_0"].items():
        if name != "axis_median_range":
            lines.append(f"| `{name}` | {fmt(item['median'])} |")
    lag_range = report["smoothness_scale_1_0"]["axis_median_range"]
    lines.extend([
        "",
        f"六轴中位数范围为 **{fmt(lag_range[0])}–{fmt(lag_range[1])}**，与连续平滑速度一致。occupancy 差异是事件表示经同一阈值器重检测后的差异，不是逐帧噪声/抖动证据。",
        "",
        "## 已有 duration scale 扫描",
        "",
        "| scale | H occupancy median | V occupancy median | H 事件/分钟 median | V 事件/分钟 median |",
        "|---:|---:|---:|---:|---:|",
    ])
    for row in report["existing_duration_scale_sweep"]:
        lines.append(
            f"| {fmt(row['horizontal_duration_scale'], 1)} | {fmt(row['horizontal_occupancy_median'])} | "
            f"{fmt(row['vertical_occupancy_median'])} | {fmt(row['horizontal_rate_per_minute_median'])} | "
            f"{fmt(row['vertical_rate_per_minute_median'])} |"
        )
    lines.extend([
        "",
        "## 原因判定",
        "",
        "1. checkpoint 的 hazard bias 是在 teacher-forced validation 的 eligible risk set 上校准的；闭环控制器触发率并不稀疏。",
        "2. 训练的 duration 是真人 hysteresis mask 的总长度，而生成器用一段单调 minimum-jerk 转移来表示它。只有转移中段速度超阈，两端为零速；真人事件则可包含持续超阈、反向和被 merge-gap 合并的多个子运动。",
        "3. 终点是有界的绝对 pose，幅度由 `target-current` 导出。若抽到的绝对终点靠近当前 pose，控制器仍记一次触发，但轨迹可能无法超过 high threshold。",
        "4. horizontal 控制器还生成 z/roll，但 horizontal 诊断只看 x/yaw；分配在 z/roll 的幅度不会提高 H occupancy。",
        "",
        "## 建议的部署校准顺序",
        "",
        "1. 本轮部署保持 `duration_scale=1.0`、当前 hazard bias、现有 endpoint 分布、roll comfort limit 和无逐帧 HeadLocal 噪声约定；**不改 runtime**。",
        "2. 不使用全局幅度乘数或强制过阈 gate。它们会把本来合理的微动作删除/放大，而已检出事件的终点幅度并不普遍小于真人。",
        "3. 不上调原始 Bernoulli 概率；controller 触发率已高于真人 pooled 事件率，再调高会优先增加微动作/控制器 busy，未必修复重检测 occupancy。",
        "4. 将 bounded visibility gate（固定原 duration，仅在 onset 重采 endpoint）保留为后续离线 ablation；必须同时比较幅度分布偏移、加速度/jerk、边界占空和主观视觉舒适性，不应只优化 occupancy。",
        "",
        "## 口径与限制",
        "",
        "- 真人数据是 frozen v2 NPZ 重建的无重复连续片段，不是 raw EyeNavGS 完整 run；边界事件不进入 duration/endpoint 分布。",
        "- SG 滤波为离线居中滤波，occupancy 是诊断伪标签，不是真实人工动作标注。",
        "- 本脚本/报告不修改 checkpoint、core model 或 Unity。",
        "",
    ])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    global thresholds_global
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    model, checkpoint = load_checkpoint(
        args.model,
        device,
        horizontal_duration_scale=1.0,
        vertical_duration_scale=1.0,
        roll_limit_rad=args.roll_limit_rad,
    )
    thresholds_global = checkpoint["event_definition"]["thresholds"]

    validation_split = load_split(args.data_dir, "val")
    fragments = reconstruct_trajectories(validation_split)
    labelled_fragments = label_trajectories(
        fragments,
        thresholds_global,
        dt_seconds=DT_SECONDS,
        cooldown_steps=int(checkpoint["model_config"]["cooldown_steps"]),
    )
    real_items: list[tuple[np.ndarray, Mapping[str, np.ndarray], Mapping[str, np.ndarray]]] = []
    for item in labelled_fragments:
        labels = make_labels(
            Sequence(item.fragment.trajectory_id, item.fragment.state, DT_SECONDS),
            thresholds_global,
        )
        real_items.append((labels["smooth"], labels["signals"], labels["masks"]))

    histories, rows, trajectories = choose_histories(validation_split, args.max_seeds)
    mean = np.asarray(checkpoint["normalization"]["state_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalization"]["state_std"], dtype=np.float32)
    normalized = torch.from_numpy((histories.astype(np.float32) - mean) / std).to(device)
    normalized[:, -1, 8] = torch.clamp(
        normalized[:, -1, 8],
        (-args.roll_limit_rad - float(mean[8])) / float(std[8]),
        (args.roll_limit_rad - float(mean[8])) / float(std[8]),
    )
    generated_tensor, diagnostics = model.runtime_rollout(
        normalized,
        ROLLOUT_STEPS,
        seed=args.seed,
        root_noise_std_normalized=args.root_noise_std,
        root_noise_rho=args.root_noise_rho,
    )
    generated = generated_tensor.cpu().numpy() * std[None, None, :] + mean[None, None, :]
    generated_labels = [
        make_labels(Sequence(str(args.seed + index), state, DT_SECONDS), thresholds_global)
        for index, state in enumerate(generated)
    ]
    generated_items = [
        (labels["smooth"], labels["signals"], labels["masks"])
        for labels in generated_labels
    ]

    model_hash = sha256_file(args.model)
    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "generated_at_utc": utc_now(),
        "status": "success",
        "scope": "read-only calibration analysis; no core model or Unity changes",
        "model": str(args.model.resolve()),
        "model_sha256": model_hash,
        "validation_data": str((args.data_dir / "normal_hmd_val.npz").resolve()),
        "validation_data_sha256": sha256_file(args.data_dir / "normal_hmd_val.npz"),
        "dt_seconds": DT_SECONDS,
        "rollout_steps": ROLLOUT_STEPS,
        "rollout_count": int(len(generated)),
        "seed": args.seed,
        "duration_scale": 1.0,
        "root_noise_std_normalized": args.root_noise_std,
        "root_noise_rho": args.root_noise_rho,
        "roll_limit_rad": args.roll_limit_rad,
        "selected_npz_rows": rows,
        "selected_trajectory_indices": trajectories,
        "thresholds": thresholds_global,
        "validation_bands": {name: list(bounds) for name, bounds in VALIDATION_BANDS.items()},
        "comparison": {
            "real_validation": collect_source(real_items),
            "generated_scale_1_0": collect_source(generated_items),
        },
        "controller_scale_1_0": controller_records(
            histories, generated, diagnostics, generated_labels, checkpoint
        ),
        "existing_duration_scale_sweep": scale_sweep(args.reference.parent, model_hash),
        "smoothness_scale_1_0": generated_velocity_lag1(generated),
        "recommendation": {
            "duration_scale": 1.0,
            "hazard_probability_action": "keep current bias initially; controller trigger rates are already near/above real validation event rates",
            "deployment_action": "no runtime calibration this round; keep the scale=1.0 smooth baseline",
            "amplitude_action": "do not impose a global multiplier or visibility floor; endpoint medians are already close to human validation",
            "future_ablation_only": "bounded onset-time endpoint visibility gate with duration held fixed",
            "visibility_formula": "1.875 * abs(endpoint_delta) / ((duration_steps - 1) * dt) >= selected diagnostic high threshold on at least one branch axis",
            "per_frame_headlocal_noise": False,
            "deployment_status": "smooth low-occupancy baseline; occupancy band remains failed and must not be claimed as matched",
        },
    }

    if args.reference.is_file():
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        report["reference_scale_1_0_artifact"] = {
            "path": str(args.reference.resolve()),
            "model_sha256_matches": reference.get("model_sha256") == model_hash,
            "horizontal_occupancy_median": reference["summary"]["horizontal_occupancy"]["median"],
            "vertical_occupancy_median": reference["summary"]["vertical_occupancy"]["median"],
            "horizontal_rate_per_minute_median": reference["summary"]["horizontal_rate_per_minute"]["median"],
            "vertical_rate_per_minute_median": reference["summary"]["vertical_rate_per_minute"]["median"],
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary_json = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    temporary_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_json.replace(args.output_json)
    temporary_md = args.output_md.with_suffix(args.output_md.suffix + ".tmp")
    temporary_md.write_text(markdown(report), encoding="utf-8")
    temporary_md.replace(args.output_md)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    result.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    result.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    result.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    result.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    result.add_argument("--max-seeds", type=int, default=16)
    result.add_argument("--seed", type=int, default=20261012)
    result.add_argument("--root-noise-std", type=float, default=0.15)
    result.add_argument("--root-noise-rho", type=float, default=0.95)
    result.add_argument("--roll-limit-rad", type=float, default=0.12)
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return result


def main() -> int:
    args = parser().parse_args()
    report = run(args)
    print(json.dumps({
        "status": report["status"],
        "json": str(args.output_json.resolve()),
        "markdown": str(args.output_md.resolve()),
    }, ensure_ascii=False))
    return 0


thresholds_global: Mapping[str, Mapping[str, float]] = {}


if __name__ == "__main__":
    raise SystemExit(main())
