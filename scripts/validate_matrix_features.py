#!/usr/bin/env python3
"""Validate frame-level features parsed from VR.net scene matrices."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = [
    "fov_horizontal",
    "fov_vertical",
    "camera_pos_x",
    "camera_pos_y",
    "camera_pos_z",
    "camera_yaw",
    "camera_pitch",
    "camera_roll",
    "camera_speed",
    "camera_rotation_rate",
    "camera_acceleration",
    "object_count",
    "mean_object_speed",
    "max_object_speed",
]

RANGES = {
    "fov_horizontal": (60.0, 120.0, "degrees"),
    "fov_vertical": (60.0, 120.0, "degrees"),
    "camera_speed": (0.0, 20.0, "source-units/s"),
    "camera_rotation_rate": (0.0, 720.0, "degrees/s"),
    "object_count": (1.0, 1000.0, "objects"),
}


def fmt(value: float | int) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.6g}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("features/vrnet_matrix_features.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/matrix_features_validation.md"),
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input feature CSV not found: {args.input}")

    df = pd.read_csv(args.input, low_memory=False)
    missing_columns = [column for column in FEATURES if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Required feature columns missing: {missing_columns}")

    lines: list[str] = [
        "# VR.net 矩阵特征合理性检查",
        "",
        f"- 输入文件：`{args.input.as_posix()}`",
        f"- 总行数：{len(df)}",
        f"- session 数：{df['session_id'].nunique(dropna=True)}",
        "- 位置和线速度单位：保留 VR.net 源坐标单位；只有在 Unity 场景采用 1 unit = 1 m 时，速度才可解释为 m/s。",
        "- 本报告检查数值和典型范围，不构成物理单位或语义正确性的外部校准。",
        "",
        "## 每个 session 的可用行数",
        "",
        "| session | 总行数 | FOV 行 | 相机运动行 | 物体计数行 | 物体速度行 |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for session_id, group in df.groupby("session_id", sort=True):
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                session_id,
                len(group),
                int(group["fov_horizontal"].notna().sum()),
                int(group["camera_speed"].notna().sum()),
                int(group["object_count"].notna().sum()),
                int(group["mean_object_speed"].notna().sum()),
            )
        )

    lines.extend(
        [
            "",
            "## 特征统计",
            "",
            "| 特征 | 有限值数 | 缺失率 | inf 数 | min | max | mean | std | 全零 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )

    feature_summaries: dict[str, dict[str, float | int | bool]] = {}
    for column in FEATURES:
        numeric = pd.to_numeric(df[column], errors="coerce")
        inf_count = int(np.isinf(numeric.to_numpy(dtype=float)).sum())
        finite = numeric[np.isfinite(numeric.to_numpy(dtype=float))]
        summary = {
            "count": int(finite.size),
            "missing_rate": 1.0 - (finite.size / len(df) if len(df) else 0.0),
            "inf_count": inf_count,
            "min": finite.min() if not finite.empty else np.nan,
            "max": finite.max() if not finite.empty else np.nan,
            "mean": finite.mean() if not finite.empty else np.nan,
            "std": finite.std(ddof=1) if finite.size > 1 else np.nan,
            "all_zero": bool(not finite.empty and np.allclose(finite.to_numpy(), 0.0)),
        }
        feature_summaries[column] = summary
        lines.append(
            "| {} | {} | {:.2%} | {} | {} | {} | {} | {} | {} |".format(
                column,
                summary["count"],
                summary["missing_rate"],
                summary["inf_count"],
                fmt(summary["min"]),
                fmt(summary["max"]),
                fmt(summary["mean"]),
                fmt(summary["std"]),
                "是" if summary["all_zero"] else "否",
            )
        )

    lines.extend(
        [
            "",
            "## 典型范围检查",
            "",
            "| 特征 | 检查范围 | 有限值数 | 异常数 | 异常比例 | 结论 |",
            "|---|---|---:|---:|---:|---|",
        ]
    )

    range_results: dict[str, tuple[int, int]] = {}
    for column, (lower, upper, unit) in RANGES.items():
        numeric = pd.to_numeric(df[column], errors="coerce")
        finite = numeric[np.isfinite(numeric.to_numpy(dtype=float))]
        outlier_count = int(((finite < lower) | (finite > upper)).sum())
        valid_count = int(finite.size)
        range_results[column] = (valid_count, outlier_count)
        ratio = outlier_count / valid_count if valid_count else np.nan
        if valid_count == 0:
            conclusion = "无可检查值"
        elif outlier_count == valid_count:
            conclusion = "全部超界，需检查公式或单位"
        elif ratio <= 0.05:
            conclusion = "大部分位于典型范围"
        else:
            conclusion = "存在较多超界值，需检查数据与单位"
        lines.append(
            f"| {column} | {fmt(lower)}–{fmt(upper)} {unit} | {valid_count} | "
            f"{outlier_count} | {ratio:.2%} | {conclusion} |"
            if valid_count
            else f"| {column} | {fmt(lower)}–{fmt(upper)} {unit} | 0 | 0 | NA | {conclusion} |"
        )

    fov_checks = [range_results["fov_horizontal"], range_results["fov_vertical"]]
    fov_all_out = any(valid > 0 and outliers == valid for valid, outliers in fov_checks)
    any_inf = any(int(summary["inf_count"]) > 0 for summary in feature_summaries.values())

    lines.extend(
        [
            "",
            "## 可信度判断",
            "",
        ]
    )
    if fov_all_out:
        lines.append("- **FOV：不通过。** 至少一个 FOV 维度全部超出 60°–120°，投影矩阵公式或布局需要复核。")
    else:
        lines.append("- **FOV：通过基本合理性检查。** 有效值并非全部超出 60°–120°；这只能验证量级，不能证明投影约定完全正确。")
    lines.append(
        "- **相机旋转：有限制。** 视图矩阵先求逆，并校正 Unity 相机空间的 Z 轴反射；报告的 yaw/pitch/roll 使用外旋 Y-X-Z 顺序。欧拉角存在表示歧义，运动强度主要依据相邻旋转矩阵的相对角度。"
    )
    lines.append(
        "- **相机速度与加速度：有限制。** 时间戳被换算为秒，但位置单位未经独立校准，阈值 0–20 仅在 1 Unity unit = 1 m 时具有 m/s 含义。"
    )
    lines.append(
        "- **物体运动：保守提取。** 只在相邻场景记录中同名且唯一的物体之间计算速度；`object_name` 不是稳定实例 ID，同名重复物体不会被强制匹配。"
    )
    lines.append(
        "- **投影偏移：** FOV 由投影矩阵对角缩放项计算。矩阵中可能存在非对称投影偏移，因此输出是由缩放项得到的对称等效 FOV。"
    )
    lines.append(f"- **数值稳定性：** {'发现 inf，需要复核。' if any_inf else '未发现 inf。'}")
    if not fov_all_out and not any_inf and len(df) > 0:
        lines.append("- **总体结论：** 特征可用于后续流程和探索性分析；物理单位、物体身份和欧拉角约定仍需作者文档或引擎内真值校准。")
    else:
        lines.append("- **总体结论：** 当前结果未通过基本检查，不能用于后续分析，需先修正上述问题。")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
