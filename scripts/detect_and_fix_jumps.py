#!/usr/bin/env python3
"""Detect camera coordinate jumps and invalidate derivatives across boundaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


POSITION_COLUMNS = ["camera_pos_x", "camera_pos_y", "camera_pos_z"]
DERIVATIVE_COLUMNS = ["camera_speed", "camera_acceleration"]


def describe(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "count": float(values.size),
        "mean": float(values.mean()) if not values.empty else np.nan,
        "max": float(values.max()) if not values.empty else np.nan,
        "std": float(values.std(ddof=1)) if values.size > 1 else np.nan,
    }


def fmt(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.6g}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("features/vrnet_matrix_features.csv"))
    parser.add_argument("--output", type=Path, default=Path("features/vrnet_matrix_features_clean.csv"))
    parser.add_argument("--report", type=Path, default=Path("results/jump_detection_report.md"))
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--dataset-name", default="Pirate Life")
    args = parser.parse_args()
    if args.threshold <= 0:
        raise ValueError("Jump threshold must be positive")
    if not args.input.is_file():
        raise FileNotFoundError(f"Input does not exist: {args.input}")

    frame = pd.read_csv(args.input, low_memory=False)
    required = {"session_id", "frame_id", "timestamp", *POSITION_COLUMNS, *DERIVATIVE_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    session_count = int(frame["session_id"].nunique(dropna=True))
    if session_count < 1:
        raise ValueError("Input contains no sessions")

    original_order = np.arange(len(frame))
    frame["_original_order"] = original_order
    frame["camera_displacement"] = np.nan
    frame["camera_jump_flag"] = False
    frame["camera_segment_id"] = 0

    jump_records: list[dict[str, object]] = []
    for session_id, indices in frame.groupby("session_id", sort=True).groups.items():
        session = frame.loc[indices].sort_values(["timestamp", "frame_id", "_original_order"])
        positions = session[POSITION_COLUMNS].apply(pd.to_numeric, errors="coerce")
        valid_mask = positions.notna().all(axis=1)
        valid_positions = positions.loc[valid_mask]
        displacement = np.sqrt(valid_positions.diff().pow(2).sum(axis=1, min_count=3))
        frame.loc[displacement.index, "camera_displacement"] = displacement
        jump_indices = displacement.index[displacement > args.threshold]
        frame.loc[jump_indices, "camera_jump_flag"] = True

        ordered_valid_indices = list(valid_positions.index)
        index_positions = {index: position for position, index in enumerate(ordered_valid_indices)}
        for jump_index in jump_indices:
            position = index_positions[jump_index]
            previous_index = ordered_valid_indices[position - 1]
            jump_records.append(
                {
                    "session_id": session_id,
                    "previous_frame": frame.at[previous_index, "frame_id"],
                    "jump_frame": frame.at[jump_index, "frame_id"],
                    "timestamp": frame.at[jump_index, "timestamp"],
                    "displacement": frame.at[jump_index, "camera_displacement"],
                }
            )
            # Speed and acceleration at the boundary cross two coordinate segments.
            frame.loc[jump_index, DERIVATIVE_COLUMNS] = np.nan
            # Acceleration on the next valid row was derived from the invalid jump speed.
            if position + 1 < len(ordered_valid_indices):
                frame.at[ordered_valid_indices[position + 1], "camera_acceleration"] = np.nan

        ordered_all_indices = list(session.index)
        flags = frame.loc[ordered_all_indices, "camera_jump_flag"].astype(int)
        frame.loc[ordered_all_indices, "camera_segment_id"] = flags.cumsum().to_numpy()

    before = pd.read_csv(args.input, usecols=DERIVATIVE_COLUMNS)
    after = frame[DERIVATIVE_COLUMNS]
    comparisons = {
        column: (describe(before[column]), describe(after[column]))
        for column in DERIVATIVE_COLUMNS
    }
    jumps = pd.DataFrame(jump_records)

    frame = frame.sort_values("_original_order").drop(columns=["_original_order"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, encoding="utf-8-sig", float_format="%.12g")

    per_session = (
        jumps.groupby("session_id").size().rename("jumps").reset_index()
        if not jumps.empty
        else pd.DataFrame(columns=["session_id", "jumps"])
    )
    lines = [
        f"# {args.dataset_name} 坐标跳变检测与处理",
        "",
        "**本处理用于探索性流程验证，非最终数据清洗结论。**",
        "",
        f"- 输入：`{args.input.as_posix()}`",
        f"- 输出：`{args.output.as_posix()}`",
        f"- 阈值：相邻有效相机矩阵记录的位置位移 > {args.threshold:g} 源坐标单位",
        "- 单位说明：VR.net 文件未独立确认 1 Unity unit = 1 m，因此报告使用“源坐标单位”，不把阈值直接称为米。",
        f"- 检测到的跳变：{len(jumps)}",
        f"- session 数：{session_count}",
        "",
        "## 处理策略",
        "",
        "1. 每个 session 按时间戳和帧号排序，只比较相邻的有效相机位置。",
        "2. 位移超过阈值的当前行标记 `camera_jump_flag=true`，并开始新的 `camera_segment_id`。",
        "3. 跳变行的 `camera_speed` 与 `camera_acceleration` 设为 NaN。",
        "4. 跳变后第一个有效行的 `camera_acceleration` 也设为 NaN，因为它依赖跨段速度差。",
        "5. 保留原始位置、位移和跳变标记以便审计；session 聚合时整行排除跳变帧。",
        "",
        "## 各 session 跳变数",
        "",
        "| session | 跳变数 |",
        "|---|---:|",
    ]
    jump_map = dict(zip(per_session.get("session_id", []), per_session.get("jumps", [])))
    for session_id in sorted(frame["session_id"].astype(str).unique()):
        lines.append(f"| {session_id} | {int(jump_map.get(session_id, 0))} |")

    lines.extend(
        [
            "",
            "## 处理前后对比",
            "",
            "| 特征 | 阶段 | 有效数 | mean | max | std |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for column, (raw_stats, clean_stats) in comparisons.items():
        lines.append(
            f"| {column} | 处理前 | {int(raw_stats['count'])} | {fmt(raw_stats['mean'])} | "
            f"{fmt(raw_stats['max'])} | {fmt(raw_stats['std'])} |"
        )
        lines.append(
            f"| {column} | 处理后 | {int(clean_stats['count'])} | {fmt(clean_stats['mean'])} | "
            f"{fmt(clean_stats['max'])} | {fmt(clean_stats['std'])} |"
        )

    lines.extend(["", "## 最大跳变", ""])
    if jumps.empty:
        lines.append("未检测到超过阈值的跳变。")
    else:
        lines.extend(
            [
                "| session | 前一帧 | 跳变帧 | timestamp | 位移 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for _, row in jumps.nlargest(20, "displacement").iterrows():
            lines.append(
                f"| {row['session_id']} | {row['previous_frame']} | {row['jump_frame']} | "
                f"{row['timestamp']} | {float(row['displacement']):.6g} |"
            )
    lines.extend(
        [
            "",
            "## 限制",
            "",
            "阈值 10 是本次流程预先指定的工程阈值，未经游戏内传送事件真值验证。被标记事件可能是传送、加载、坐标系切换或记录异常；本报告不对原因作无证据判断。",
        ]
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Detected {len(jumps)} jumps; wrote {args.output} and {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
