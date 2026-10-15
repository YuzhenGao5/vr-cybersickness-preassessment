#!/usr/bin/env python3
"""Audit residual rotation-rate anomalies and rebuild affected session aggregates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROTATION_COLUMNS = [
    "camera_rotation_rate_mean", "camera_rotation_rate_max", "camera_rotation_rate_std"
]


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=Path("features/vrnet_training_matrix_v6.csv"))
    parser.add_argument("--frames", type=Path, default=Path("features/vrnet_matrix_features_v6.csv"))
    parser.add_argument("--threshold", type=float, default=720.0)
    parser.add_argument("--clean-output", type=Path, default=Path("features/vrnet_training_matrix_v7_clean.csv"))
    parser.add_argument("--report", type=Path, default=Path("results/rotation_anomaly_check.md"))
    args = parser.parse_args()
    matrix = pd.read_csv(args.matrix, encoding="utf-8-sig")
    frames = pd.read_csv(
        args.frames,
        usecols=["session_id", "game_name", "frame_id", "timestamp", "camera_rotation_rate", "camera_jump_flag"],
        low_memory=False,
    )
    rate = pd.to_numeric(frames["camera_rotation_rate"], errors="coerce")
    jumps = truthy(frames["camera_jump_flag"])
    over = rate > args.threshold
    residual = over & ~jumps
    matrix_max = pd.to_numeric(matrix["camera_rotation_rate_max"], errors="coerce")
    aggregate_anomaly = matrix_max > args.threshold

    usable = frames.loc[~jumps, ["session_id", "camera_rotation_rate"]].copy()
    usable["camera_rotation_rate"] = pd.to_numeric(usable["camera_rotation_rate"], errors="coerce")
    usable.loc[usable["camera_rotation_rate"] > args.threshold, "camera_rotation_rate"] = np.nan
    recomputed = usable.groupby("session_id")["camera_rotation_rate"].agg(["mean", "max", "std"]).reset_index()
    recomputed = recomputed.rename(columns={
        "mean": "camera_rotation_rate_mean_clean",
        "max": "camera_rotation_rate_max_clean",
        "std": "camera_rotation_rate_std_clean",
    })
    clean = matrix.merge(recomputed, on="session_id", how="left", validate="one_to_one")
    before = clean[["session_id", *ROTATION_COLUMNS]].copy()
    for column in ROTATION_COLUMNS:
        clean_name = f"{column}_clean"
        clean[column] = clean[clean_name]
        clean = clean.drop(columns=clean_name)
    if len(clean) != len(matrix) or clean["session_id"].duplicated().any():
        raise RuntimeError("rotation cleanup changed the session identity set")
    args.clean_output.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.clean_output, index=False, encoding="utf-8-sig", float_format="%.12g")

    after = clean.set_index("session_id")
    anomaly_rows = []
    for _, row in matrix.loc[aggregate_anomaly, ["game_name", "session_id", "camera_rotation_rate_max"]].iterrows():
        session = str(row["session_id"])
        residual_rows = int((residual & frames["session_id"].astype(str).eq(session)).sum())
        anomaly_rows.append(
            f"| {row['game_name']} | {session} | {float(row['camera_rotation_rate_max']):.6f} | "
            f"{float(after.at[session, 'camera_rotation_rate_max']):.6f} | {residual_rows} |"
        )
    frame_sessions = int(frames.loc[over, "session_id"].nunique())
    residual_sessions = int(frames.loc[residual, "session_id"].nunique())
    changed = 0
    for column in ROTATION_COLUMNS:
        left = pd.to_numeric(before[column], errors="coerce").to_numpy()
        right = pd.to_numeric(clean[column], errors="coerce").to_numpy()
        changed = max(changed, int((~np.isclose(left, right, equal_nan=True)).sum()))
    lines = [
        "# v7 旋转率异常与残留跳变检查", "",
        "**探索性清理，非最终物理有效性结论。720°/s 是本任务指定的合理性阈值。**", "",
        f"- session 聚合矩阵中 `camera_rotation_rate_max > {args.threshold:g}`：{int(aggregate_anomaly.sum())} 个 session",
        f"- 帧级超阈值：{int(over.sum())} 行，涉及 {frame_sessions} 个 session",
        f"- 其中已被位置跳变规则标记：{int((over & jumps).sum())} 行",
        f"- 未被位置跳变规则标记的残留：{int(residual.sum())} 行，涉及 {residual_sessions} 个 session",
        f"- 清理后旋转率聚合发生变化：{changed} 个 session", "",
        "## 异常 session", "",
        "| 游戏 | session | 原 max (°/s) | 清理后 max (°/s) | 未标记超阈值帧 |",
        "|---|---|---:|---:|---:|", *anomaly_rows, "",
        "## 处理", "",
        "1. 继续排除原 `camera_jump_flag=true` 的帧。",
        f"2. 对其余帧中 `camera_rotation_rate > {args.threshold:g}` 的值设为缺失，不截断、不替换为人为数值。",
        "3. 从保留帧重新计算每个 session 的 rotation rate mean/max/std。",
        f"4. 清理矩阵写入 `{args.clean_output.as_posix()}`；标签与其他特征保持原值。", "",
        "## 判断", "",
        "位置跳变处理未完全覆盖旋转突变：残留的 6 个超阈值帧分布在 6 个 session。v6 中 rotation max/std 的 SHAP 方向可能受到这些极值和共线性共同影响。",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"aggregate_anomaly_sessions={int(aggregate_anomaly.sum())} frame_over={int(over.sum())} "
        f"residual_rows={int(residual.sum())} residual_sessions={residual_sessions} changed_sessions={changed}"
    )
    print(f"Wrote {args.clean_output} and {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
