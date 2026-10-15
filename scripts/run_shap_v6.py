#!/usr/bin/env python3
"""Compute exploratory SHAP attribution and direction checks for VR.net v6."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


EXPECTED_MEDIUM_PREFIXES = (
    "fov_horizontal_", "fov_vertical_", "camera_speed_",
    "camera_acceleration_", "object_count_",
)


def direction_label(correlation: float) -> str:
    if not np.isfinite(correlation) or abs(correlation) < 1e-12:
        return "方向无法稳定判定"
    return "较高值倾向推向 medium" if correlation > 0 else "较高值倾向推向 low"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("features/vrnet_training_matrix_v6.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/vrnet_model_v6.pkl"))
    parser.add_argument("--output", type=Path, default=Path("features/shap_values_v6.csv"))
    parser.add_argument("--summary-plot", type=Path, default=Path("figures/shap_summary_v6.png"))
    parser.add_argument("--bar-plot", type=Path, default=Path("figures/shap_bar_v6.png"))
    parser.add_argument("--report", type=Path, default=Path("results/shap_interpretation_v6.md"))
    args = parser.parse_args()
    data = pd.read_csv(args.input, encoding="utf-8-sig")
    bundle = joblib.load(args.model)
    pipeline = bundle["pipeline"]
    model_type = bundle["model_type"]
    feature_names = list(bundle["feature_names"])
    X = data[feature_names].apply(pd.to_numeric, errors="coerce")
    transformed = np.asarray(pipeline[:-1].transform(X), dtype=float)
    classifier = pipeline.named_steps["classifier"]

    if model_type == "lightgbm":
        explanation = shap.TreeExplainer(classifier)(transformed)
        values = np.asarray(explanation.values)
        if values.ndim == 3:
            values = values[:, :, 1]
        method = "TreeSHAP"
    else:
        explanation = shap.LinearExplainer(classifier, transformed)(transformed)
        values = np.asarray(explanation.values)
        method = "LinearExplainer"
    if values.shape != (len(data), len(feature_names)):
        raise RuntimeError(f"unexpected SHAP shape: {values.shape}")

    correlations: list[float] = []
    for index in range(len(feature_names)):
        correlation = pd.Series(transformed[:, index]).corr(
            pd.Series(values[:, index]), method="spearman"
        )
        correlations.append(float(correlation) if pd.notna(correlation) else np.nan)
    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.mean(np.abs(values), axis=0),
        "spearman_feature_shap": correlations,
    }).sort_values("mean_abs_shap", ascending=False, ignore_index=True)

    probability = pipeline.predict_proba(X)[:, 1]
    output = pd.DataFrame({
        "session_id": data["session_id"], "game_name": data["game_name"],
        "probability_medium": probability,
    })
    for index, feature in enumerate(feature_names):
        output[f"shap_{feature}"] = values[:, index]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig", float_format="%.12g")

    args.summary_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 9))
    shap.summary_plot(values, transformed, feature_names=feature_names, max_display=len(feature_names), show=False)
    plt.title("VR.net v6 SHAP summary — exploratory only")
    plt.tight_layout()
    plt.savefig(args.summary_plot, dpi=200, bbox_inches="tight")
    plt.close()
    display = importance.sort_values("mean_abs_shap")
    figure, axis = plt.subplots(figsize=(11, 9))
    axis.barh(display["feature"], display["mean_abs_shap"], color="#1688e8")
    axis.set_xlabel("mean(|SHAP value|)")
    axis.set_title("VR.net v6 mean |SHAP| — exploratory only")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(args.bar_plot, dpi=200, bbox_inches="tight")
    plt.close(figure)

    top_rows = []
    for rank, row in importance.head(5).iterrows():
        top_rows.append(
            f"| {rank + 1} | {row['feature']} | {row['mean_abs_shap']:.6g} | "
            f"{direction_label(float(row['spearman_feature_shap']))} |"
        )
    expected = importance[
        importance["feature"].map(lambda name: str(name).startswith(EXPECTED_MEDIUM_PREFIXES))
    ].copy()
    check_rows = []
    reversed_count = 0
    for _, row in expected.iterrows():
        correlation = float(row["spearman_feature_shap"])
        if not np.isfinite(correlation) or abs(correlation) < 1e-12:
            status = "无法判定"
        elif correlation > 0:
            status = "符合预期"
        else:
            status = "方向相反"
            reversed_count += 1
        check_rows.append(
            f"| {row['feature']} | {correlation:.6g} | {direction_label(correlation)} | {status} |"
        )
    lines = [
        "# VR.net v6 SHAP 解释", "",
        "**探索性结果，非最终结论。61 个 session 的 SHAP 排名和方向仍可能受样本量、游戏差异、坐标跳变残留及标签噪声影响。**", "",
        f"- 模型：{model_type}", f"- 解释器：{method}", f"- session：{len(data)}",
        f"- 特征：{len(feature_names)}", "- 方向判定：特征值与该特征 SHAP 值的 Spearman 相关符号", "",
        "## Top-5", "",
        "| 排名 | 特征 | mean(|SHAP|) | 探索性方向 |",
        "|---:|---|---:|---|", *top_rows, "",
        "## 预期方向检查", "",
        "预设直觉为较高 FOV、相机速度、加速度和物体数量倾向 medium。", "",
        "| 特征 | Spearman(feature, SHAP) | 当前方向 | 检查 |",
        "|---|---:|---|---|", *check_rows, "",
        f"方向相反的预期特征数：{reversed_count}/{len(expected)}。", "",
    ]
    if reversed_count:
        lines.append(
            "方向相反可能来自小样本、跨游戏尺度差异、坐标跳变残留、特征共线性或稀疏 self-report 标签噪声；本结果不作因果解释。"
        )
    else:
        lines.append("当前方向未发现与预设相反的项，但这不构成验证证据。")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Top-5:")
    print(importance.head(5).to_string(index=False))
    print(f"Expected-direction reversals: {reversed_count}/{len(expected)}")
    print(f"Wrote SHAP outputs for {len(data)} sessions using {method}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
