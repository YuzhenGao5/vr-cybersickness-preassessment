#!/usr/bin/env python3
"""Compute exploratory linear SHAP values for the Pirate_Life pipeline model."""

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
    parser.add_argument("--input", type=Path, default=Path("features/pipeline_training_matrix.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/pipeline_model.pkl"))
    parser.add_argument("--output", type=Path, default=Path("features/shap_values_pipeline.csv"))
    parser.add_argument("--summary-plot", type=Path, default=Path("figures/shap_summary_pipeline.png"))
    parser.add_argument("--bar-plot", type=Path, default=Path("figures/shap_bar_pipeline.png"))
    parser.add_argument("--report", type=Path, default=Path("results/shap_interpretation_pipeline.md"))
    args = parser.parse_args()

    if not args.input.is_file() or not args.model.is_file():
        raise FileNotFoundError("Training matrix or fitted model is missing")
    data = pd.read_csv(args.input, encoding="utf-8-sig")
    bundle = joblib.load(args.model)
    pipeline = bundle["pipeline"]
    feature_names = list(bundle["feature_names"])
    X = data[feature_names].apply(pd.to_numeric, errors="coerce")
    X_train = X.iloc[bundle["train_indices"]]

    preprocessing = pipeline[:-1]
    classifier = pipeline.named_steps["classifier"]
    transformed_train = preprocessing.transform(X_train)
    transformed_all = preprocessing.transform(X)
    explainer = shap.LinearExplainer(classifier, transformed_train)
    explanation = explainer(transformed_all)
    values = np.asarray(explanation.values, dtype=float)
    if values.ndim != 2 or values.shape != (len(data), len(feature_names)):
        raise RuntimeError(f"Unexpected SHAP shape: {values.shape}")

    probabilities = pipeline.predict_proba(X)[:, 1]
    output = pd.DataFrame({"session_id": data["session_id"], "probability_medium": probabilities})
    for index, feature in enumerate(feature_names):
        output[f"shap_{feature}"] = values[:, index]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig", float_format="%.12g")

    args.summary_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 9))
    shap.summary_plot(
        values,
        transformed_all,
        feature_names=feature_names,
        max_display=len(feature_names),
        plot_size=(11, 9),
        show=False,
    )
    plt.title("Pirate Life SHAP summary — pipeline validation only")
    plt.tight_layout()
    plt.savefig(args.summary_plot, dpi=200, bbox_inches="tight")
    plt.close()

    args.bar_plot.parent.mkdir(parents=True, exist_ok=True)
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.mean(np.abs(values), axis=0),
            "coefficient": classifier.coef_[0],
        }
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    display = importance.sort_values("mean_abs_shap", ascending=True)
    figure, axis = plt.subplots(figsize=(11, 9))
    axis.barh(display["feature"], display["mean_abs_shap"], color="#1688e8")
    axis.set_xlabel("mean(|SHAP value|)")
    axis.set_title("Pirate Life mean |SHAP| — pipeline validation only")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(args.bar_plot, dpi=200, bbox_inches="tight")
    plt.close(figure)

    top = importance.head(5)
    rows = []
    for rank, row in top.iterrows():
        direction = "较高值推向 medium" if row["coefficient"] > 0 else "较高值推向 low"
        rows.append(
            f"| {rank + 1} | {row['feature']} | {row['mean_abs_shap']:.6g} | "
            f"{row['coefficient']:.6g} | {direction} |"
        )
    lines = [
        "# Pirate Life SHAP 解释",
        "",
        "**警告：仅 11 个 session，且模型只用其中 8 个训练。SHAP 值极不稳定，仅验证解释代码可以运行，不能作为论文结论。**",
        "",
        "## 方法",
        "",
        "模型因样本量过小采用逻辑回归，因此使用与线性模型匹配的 `shap.LinearExplainer`，没有强行使用 TreeSHAP。解释对象是经过中位数填补和标准化后的特征。正 SHAP 值增加 medium 类别的 log-odds。",
        "",
        "## Top-5",
        "",
        "| 排名 | 特征 | mean(|SHAP|) | 逻辑回归系数 | 探索性方向 |",
        "|---:|---|---:|---:|---|",
        *rows,
        "",
        "## 限制",
        "",
        "这些排名可能由一次划分、极端矩阵值和单一游戏内的偶然差异驱动。它们只能说明代码输出了可审计的贡献值，不能说明特征具有稳定因果关系或跨游戏预测能力。",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Top-5:")
    print(top.to_string(index=False))
    print(f"Wrote SHAP values and plots for {len(data)} sessions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
