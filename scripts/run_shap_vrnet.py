#!/usr/bin/env python3
"""Compute window-level SHAP explanations when a valid fitted model exists."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


SHAP_FIELDS = ["window_id", "predicted_risk_level", "target_class", "feature", "feature_value", "shap_value"]
TOP_FIELDS = ["rank", "feature", "mean_abs_shap", "mean_shap_by_class", "direction_by_class"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="features/training_feature_matrix_vrnet.csv")
    parser.add_argument("--model", default="models/risk_model_vrnet.pkl")
    parser.add_argument("--metadata", default="models/risk_model_vrnet_metadata.json")
    parser.add_argument("--output", default="features/shap_values_vrnet.csv")
    parser.add_argument("--top5", default="features/shap_top5_vrnet.csv")
    parser.add_argument("--interpretation", default="results/shap_interpretation_vrnet.md")
    parser.add_argument("--summary-plot", default="figures/shap_summary_vrnet.png")
    parser.add_argument("--bar-plot", default="figures/shap_bar_vrnet.png")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    resolve = lambda value: (root / value).resolve() if not Path(value).is_absolute() else Path(value)
    paths = {name: resolve(value) for name, value in vars(args).items()}
    with paths["metadata"].open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    paths["top5"].parent.mkdir(parents=True, exist_ok=True)
    if metadata.get("fit_status") != "trained" or not metadata.get("model_path") or not resolve(metadata["model_path"]).exists():
        with paths["output"].open("w", encoding="utf-8-sig", newline="") as handle:
            csv.writer(handle).writerow(SHAP_FIELDS)
        with paths["top5"].open("w", encoding="utf-8-sig", newline="") as handle:
            csv.writer(handle).writerow(TOP_FIELDS)
        note = [
            "# VR.net SHAP 解释", "", "**流程验证，非最终结论。**", "",
            "未计算 SHAP：训练阶段未拟合有效分类器。当前 25 个有标签窗口全部属于 low 类，单类目标无法训练 low/medium/high 分类器。",
            "`features/shap_values_vrnet.csv` 与 `features/shap_top5_vrnet.csv` 仅含列头；没有生成 SHAP 图，也没有可报告的 Top-5。",
            "因此不能描述特征对低/中/高风险的影响方向。不能把特征与模型输出的关系解释为因果关系或个体风险预测。", "",
        ]
        paths["interpretation"].parent.mkdir(parents=True, exist_ok=True)
        paths["interpretation"].write_text("\n".join(note), encoding="utf-8")
        print("shap_status=skipped reason=no_valid_fitted_model")
        return 0

    from joblib import load
    import shap
    import matplotlib.pyplot as plt

    payload = load(resolve(metadata["model_path"]))
    frame = pd.read_csv(paths["input"], encoding="utf-8-sig")
    features = payload["feature_names"]
    model = payload["pipeline"]
    x = frame.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
    imputer = model.named_steps["imputer"]
    classifier = model.named_steps["classifier"]
    transformed = imputer.transform(x)
    classes = [str(value) for value in payload["classes"]]
    explainer = shap.TreeExplainer(classifier)
    explanation = explainer(transformed)
    raw_values = np.asarray(explanation.values)
    if raw_values.ndim == 2:
        raw_values = raw_values[:, :, None]
    elif raw_values.ndim == 3 and raw_values.shape[1] == len(classes):
        raw_values = np.transpose(raw_values, (0, 2, 1))
    if raw_values.ndim != 3 or raw_values.shape[:2] != transformed.shape:
        raise ValueError(f"Unexpected SHAP array shape {raw_values.shape} for input {transformed.shape}")
    if raw_values.shape[2] != len(classes):
        if raw_values.shape[2] == 1 and len(classes) == 2:
            raw_values = np.concatenate([-raw_values, raw_values], axis=2)
        else:
            raise ValueError(f"SHAP output count {raw_values.shape[2]} does not match model classes {classes}")
    probabilities = model.predict_proba(x)
    prediction = np.asarray(model.classes_)[probabilities.argmax(axis=1)]
    class_names = [classes[int(value)] for value in model.classes_]
    rows: list[dict[str, object]] = []
    aggregate: dict[str, dict[str, object]] = {}
    for fi, feature in enumerate(features):
        per_class_mean = {}
        per_class_direction = {}
        for ci, class_name in enumerate(class_names):
            vals = raw_values[:, fi, ci]
            mean = float(np.mean(vals))
            per_class_mean[class_name] = mean
            per_class_direction[class_name] = "positive class score" if mean > 0 else "negative class score" if mean < 0 else "near zero"
            for i in range(len(frame)):
                rows.append({
                    "window_id": frame.iloc[i].get("window_id", f"window_{i:03d}"),
                    "predicted_risk_level": str(prediction[i]),
                    "target_class": class_name,
                    "feature": feature,
                    "feature_value": float(transformed[i, fi]),
                    "shap_value": float(vals[i]),
                })
        aggregate[feature] = {
            "mean_abs": float(np.mean(np.abs(raw_values[:, fi, :]))),
            "means": per_class_mean,
            "directions": per_class_direction,
        }
    pd.DataFrame(rows, columns=SHAP_FIELDS).to_csv(paths["output"], index=False, encoding="utf-8-sig")
    top = sorted(aggregate, key=lambda feature: aggregate[feature]["mean_abs"], reverse=True)[:5]
    top_rows = [{
        "rank": rank, "feature": feature, "mean_abs_shap": aggregate[feature]["mean_abs"],
        "mean_shap_by_class": json.dumps(aggregate[feature]["means"], ensure_ascii=False),
        "direction_by_class": json.dumps(aggregate[feature]["directions"], ensure_ascii=False),
    } for rank, feature in enumerate(top, 1)]
    pd.DataFrame(top_rows, columns=TOP_FIELDS).to_csv(paths["top5"], index=False, encoding="utf-8-sig")

    plot_dir = paths["summary_plot"].parent
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot([raw_values[:, :, ci] for ci in range(len(class_names))], transformed,
                      feature_names=features, class_names=class_names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(paths["summary_plot"], dpi=160, bbox_inches="tight")
    plt.close("all")
    plt.figure(figsize=(8, max(3, 0.4 * len(top))))
    vals = [aggregate[feature]["mean_abs"] for feature in top]
    plt.barh(top[::-1], vals[::-1])
    plt.xlabel("Mean |SHAP value| across model classes")
    plt.title("Global feature importance (process validation only)")
    plt.tight_layout()
    plt.savefig(paths["bar_plot"], dpi=160, bbox_inches="tight")
    plt.close()

    lines = ["# VR.net SHAP 解释", "", "**流程验证，非最终结论；样本量小。** SHAP 描述已拟合模型输出的归因，不表示因果，也不代表个体晕动症风险。", "",
             f"- 解释窗口：{len(frame)}；模型类别：{', '.join(class_names)}", "- 方向指分类器对应类别的原始 SHAP 贡献方向，不是对真实症状的因果影响。", "",
             "## 全局 Top-5", "", "| 排名 | 特征 | 平均 |SHAP| | 类别方向 |", "|---:|---|---:|---|"]
    for row in top_rows:
        lines.append(f"| {row['rank']} | `{row['feature']}` | {row['mean_abs_shap']:.6g} | `{row['direction_by_class']}` |")
    lines.extend(["", f"摘要图：`{paths['summary_plot'].relative_to(root)}`；条形图：`{paths['bar_plot'].relative_to(root)}`。", ""])
    paths["interpretation"].parent.mkdir(parents=True, exist_ok=True)
    paths["interpretation"].write_text("\n".join(lines), encoding="utf-8")
    print(f"shap_status=complete rows={len(rows)} top5={','.join(top)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
