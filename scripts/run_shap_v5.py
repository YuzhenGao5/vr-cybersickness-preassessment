#!/usr/bin/env python3
"""Compute exploratory SHAP attribution for the final VR.net v5 model."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("features/vrnet_training_matrix_v5.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/vrnet_model_v5.pkl"))
    parser.add_argument("--output", type=Path, default=Path("features/shap_values_v5.csv"))
    parser.add_argument("--summary-plot", type=Path, default=Path("figures/shap_summary_v5.png"))
    parser.add_argument("--bar-plot", type=Path, default=Path("figures/shap_bar_v5.png"))
    parser.add_argument("--report", type=Path, default=Path("results/shap_interpretation_v5.md"))
    args = parser.parse_args()
    data = pd.read_csv(args.input, encoding="utf-8-sig")
    bundle = joblib.load(args.model)
    pipeline = bundle["pipeline"]
    model_type = bundle["model_type"]
    feature_names = list(bundle["feature_names"])
    X = data[feature_names].apply(pd.to_numeric, errors="coerce")
    preprocessing = pipeline[:-1]
    classifier = pipeline.named_steps["classifier"]
    transformed = preprocessing.transform(X)

    if model_type == "lightgbm":
        explainer = shap.TreeExplainer(classifier)
        explanation = explainer(transformed)
        values = np.asarray(explanation.values)
        if values.ndim == 3:
            values = values[:, :, 1]
        method = "TreeSHAP"
        importance_direction = np.full(len(feature_names), np.nan)
    else:
        explainer = shap.LinearExplainer(classifier, transformed)
        explanation = explainer(transformed)
        values = np.asarray(explanation.values)
        method = "LinearExplainer"
        importance_direction = classifier.coef_[0]
    if values.shape != (len(data), len(feature_names)):
        raise RuntimeError(f"Unexpected SHAP shape: {values.shape}")

    probability = pipeline.predict_proba(X)[:, 1]
    output = pd.DataFrame(
        {
            "session_id": data["session_id"],
            "game_name": data["game_name"],
            "probability_medium": probability,
        }
    )
    for index, feature in enumerate(feature_names):
        output[f"shap_{feature}"] = values[:, index]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig", float_format="%.12g")

    args.summary_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 9))
    shap.summary_plot(
        values,
        transformed,
        feature_names=feature_names,
        max_display=len(feature_names),
        plot_size=(11, 9),
        show=False,
    )
    plt.title("VR.net v5 SHAP summary — exploratory only")
    plt.tight_layout()
    plt.savefig(args.summary_plot, dpi=200, bbox_inches="tight")
    plt.close()

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.mean(np.abs(values), axis=0),
            "direction_value": importance_direction,
        }
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    display = importance.sort_values("mean_abs_shap", ascending=True)
    figure, axis = plt.subplots(figsize=(11, 9))
    axis.barh(display["feature"], display["mean_abs_shap"], color="#1688e8")
    axis.set_xlabel("mean(|SHAP value|)")
    axis.set_title("VR.net v5 mean |SHAP| — exploratory only")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(args.bar_plot, dpi=200, bbox_inches="tight")
    plt.close(figure)

    rows = []
    for rank, row in importance.head(5).iterrows():
        if model_type == "logistic_regression":
            direction = "较高值推向 medium" if row["direction_value"] > 0 else "较高值推向 low"
        else:
            feature_index = feature_names.index(row["feature"])
            correlation = np.corrcoef(transformed[:, feature_index], values[:, feature_index])[0, 1]
            direction = "较高值倾向推向 medium" if correlation > 0 else "较高值倾向推向 low"
        rows.append(f"| {rank+1} | {row['feature']} | {row['mean_abs_shap']:.6g} | {direction} |")
    lines = [
        "# VR.net v5 SHAP 解释",
        "",
        "**探索性结果，非最终结论。只有 21 个 session，SHAP 排名不稳定，不能作为论文中的稳定风险因素结论。**",
        "",
        f"- 模型：{model_type}",
        f"- 解释器：{method}",
        f"- session：{len(data)}",
        f"- 特征：{len(feature_names)}",
        "",
        "## Top-5",
        "",
        "| 排名 | 特征 | mean(|SHAP|) | 探索性方向 |",
        "|---:|---|---:|---|",
        *rows,
        "",
        "方向只描述当前最终拟合模型的关联。跨游戏样本极少，部分评分事件也很稀疏；不能解释为因果关系或跨游戏泛化证据。",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Top-5:")
    print(importance.head(5).to_string(index=False))
    print(f"Wrote SHAP outputs for {len(data)} sessions using {method}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
