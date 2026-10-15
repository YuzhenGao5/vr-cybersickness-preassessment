#!/usr/bin/env python3
"""Explain trained rule-proxy classifiers without inventing missing models.

The targets in this pipeline are deterministic, rule-generated proxy labels.
SHAP therefore explains how a fitted classifier reproduces those rules; it is
not evidence that a feature predicts, or causes, participant cybersickness.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr


WARNING = "探索性，非最终结论；规则生成代理标签；规则复现不等于晕动症风险验证。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/risk_factor_detector.pkl"),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("features/risk_factor_training_matrix.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("features/shap_risk_factors.csv"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("figures"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/shap_risk_factors.md"),
    )
    return parser.parse_args()


def positive_class_shap(values: Any, n_samples: int, n_features: int) -> np.ndarray:
    """Normalize SHAP's version-dependent binary-class return shape."""
    if isinstance(values, list):
        if not values:
            raise ValueError("SHAP returned an empty list")
        values = values[1] if len(values) > 1 else values[0]

    array = np.asarray(values, dtype=float)
    if array.ndim == 2 and array.shape == (n_samples, n_features):
        return array
    if array.ndim == 3:
        # Recent SHAP: samples x features x outputs.
        if array.shape[:2] == (n_samples, n_features):
            return array[:, :, 1 if array.shape[2] > 1 else 0]
        # Older alternative: outputs x samples x features.
        if array.shape[1:] == (n_samples, n_features):
            return array[1 if array.shape[0] > 1 else 0, :, :]
    raise ValueError(
        f"Unsupported SHAP value shape {array.shape}; expected "
        f"({n_samples}, {n_features}) or a binary-output 3-D equivalent"
    )


def safe_name(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label)


def direction(feature_values: np.ndarray, shap_values: np.ndarray) -> tuple[float, str]:
    valid = np.isfinite(feature_values) & np.isfinite(shap_values)
    if valid.sum() < 3 or np.unique(feature_values[valid]).size < 2:
        return math.nan, "无法判定（取值不足或为常数）"
    finite_shap = shap_values[valid]
    scale = max(1.0, float(np.max(np.abs(finite_shap))))
    tolerance = 1e-10 * scale
    if float(np.max(np.abs(finite_shap))) <= 1e-12:
        return math.nan, "无可见贡献（mean |SHAP| 近似 0）"
    # TreeSHAP can leave machine-precision differences between conceptually
    # identical leaf values. Quantize them before rank correlation so those
    # numerical residues do not create a spurious direction.
    robust_shap = np.round(finite_shap / tolerance) * tolerance
    if np.unique(robust_shap).size < 2:
        return math.nan, "无法判定（SHAP 值为常数）"
    rho = float(spearmanr(feature_values[valid], robust_shap).statistic)
    if not math.isfinite(rho):
        return math.nan, "无法判定（相关未定义）"
    if abs(rho) < 0.1:
        return rho, "方向很弱/不明确"
    return rho, "高值推向标签 1" if rho > 0 else "高值推离标签 1"


def extract_estimator_and_values(model: Any, values: pd.DataFrame) -> tuple[Any, pd.DataFrame]:
    """Apply fitted preprocessing and return the tree estimator plus model input."""
    if hasattr(model, "named_steps"):
        steps = model.named_steps
        estimator = steps.get("classifier", list(steps.values())[-1])
        transformed: Any = values
        for name, step in steps.items():
            if step is estimator or name == "classifier":
                break
            transformed = step.transform(transformed)
        transformed_array = np.asarray(transformed, dtype=float)
    else:
        estimator = model
        transformed_array = values.to_numpy(dtype=float)
    if transformed_array.shape != values.shape:
        raise ValueError(
            f"Preprocessing changed shape from {values.shape} to {transformed_array.shape}; "
            "feature names cannot be aligned safely"
        )
    return estimator, pd.DataFrame(
        transformed_array,
        columns=values.columns,
        index=values.index,
    )


def main() -> None:
    args = parse_args()
    for path in (args.model, args.data):
        if not path.exists():
            raise FileNotFoundError(path)

    bundle = joblib.load(args.model)
    if not isinstance(bundle, dict) or not isinstance(bundle.get("labels"), dict):
        raise TypeError("Model bundle must be a dict containing a labels dict")
    data = pd.read_csv(args.data, encoding="utf-8-sig")

    id_columns = [column for column in ("session_id", "game_name") if column in data.columns]
    result_rows: list[dict[str, Any]] = []
    report_sections: list[str] = []
    trained_count = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    for label, entry in bundle["labels"].items():
        status = str(entry.get("status", "missing_status"))
        model = entry.get("model")
        n_positive = entry.get("n_positive", "NA")
        n_negative = entry.get("n_negative", "NA")
        n_unknown = entry.get("n_unknown", "NA")
        direct_feature = entry.get("direct_feature")

        if status != "trained" or model is None:
            if status == "untrainable_all_unknown":
                reason = "61 个标签均为 unknown，没有可训练模型"
            elif status == "untrainable_global_single_class":
                reason = "已知标签只有一个类别，没有可训练模型"
            else:
                reason = f"模型状态为 {status}"
            report_sections.extend(
                [
                    f"## `{label}`",
                    "",
                    f"- 状态：`{status}`",
                    f"- 0/1/unknown：{n_negative}/{n_positive}/{n_unknown}",
                    f"- SHAP：未生成；{reason}。",
                    "",
                ]
            )
            continue

        feature_names = list(entry.get("feature_names") or bundle.get("feature_names") or [])
        missing = [column for column in feature_names if column not in data.columns]
        if missing:
            raise KeyError(f"{label}: data is missing model features {missing}")
        trained_count += 1

        original = data[feature_names].apply(pd.to_numeric, errors="coerce")
        estimator, model_values = extract_estimator_and_values(model, original)
        explainer = shap.TreeExplainer(estimator)
        shap_array = positive_class_shap(
            explainer.shap_values(model_values), len(model_values), len(feature_names)
        )

        abs_mean = np.mean(np.abs(shap_array), axis=0)
        ranking = np.argsort(abs_mean)[::-1]
        top_importance = float(abs_mean[ranking[0]]) if len(ranking) else 0.0
        interpretability_floor = max(1e-12, top_importance * 1e-9)
        meaningful_indices = [
            int(index) for index in ranking if float(abs_mean[int(index)]) > interpretability_floor
        ]
        top_three: list[tuple[str, float, float, str]] = []
        for feature_index in meaningful_indices[:3]:
            feature = feature_names[feature_index]
            rho, feature_direction = direction(
                model_values[feature].to_numpy(dtype=float),
                shap_array[:, feature_index],
            )
            top_three.append((feature, float(abs_mean[feature_index]), rho, feature_direction))

        for row_index, data_index in enumerate(data.index):
            identifiers = {column: data.at[data_index, column] for column in id_columns}
            label_value = data.at[data_index, label] if label in data.columns else math.nan
            for feature_index, feature in enumerate(feature_names):
                result_rows.append(
                    {
                        **identifiers,
                        "label": label,
                        "label_value": label_value,
                        "feature": feature,
                        "feature_value": float(model_values.iloc[row_index, feature_index]),
                        "shap_value_label_1": float(shap_array[row_index, feature_index]),
                        "mean_abs_shap": float(abs_mean[feature_index]),
                        "direct_rule_feature": feature == direct_feature,
                        "label_source": "rule_generated_proxy",
                    }
                )

        figure_path = args.figures_dir / f"shap_risk_factors_{safe_name(label)}.png"
        if meaningful_indices:
            plot_features = [feature_names[index] for index in meaningful_indices]
            plt.figure()
            shap.summary_plot(
                shap_array[:, meaningful_indices],
                model_values.iloc[:, meaningful_indices],
                feature_names=plot_features,
                max_display=min(12, len(plot_features)),
                show=False,
                plot_size=(10, 7),
            )
            plt.title(f"SHAP summary: {label}\nrule-generated proxy; exploratory only")
            plt.tight_layout()
            plt.savefig(figure_path, dpi=180, bbox_inches="tight")
            plt.close("all")

        report_sections.extend(
            [
                f"## `{label}`",
                "",
                f"- 状态：`{status}`",
                f"- 0/1/unknown：{n_negative}/{n_positive}/{n_unknown}",
                f"- 直接生成标签的特征：`{direct_feature}`",
                f"- 可解释贡献阈值：`mean |SHAP| > {interpretability_floor:.6g}` "
                f"(取 max(1e-12, top mean |SHAP| × 1e-9))",
                f"- 高于该阈值的特征：{len(meaningful_indices)}/{len(feature_names)}",
                f"- 图：`{figure_path.as_posix()}`"
                if meaningful_indices
                else "- 图：未生成；所有 SHAP 贡献均低于可解释阈值。",
                "",
            ]
        )
        if top_three:
            report_sections.extend(
                [
                    "| 排名 | 特征 | mean_abs_SHAP | Spearman(feature, SHAP) | 高值方向 |",
                    "|---:|---|---:|---:|---|",
                ]
            )
            for rank, (feature, importance, rho, feature_direction) in enumerate(top_three, 1):
                rho_text = "NA" if not math.isfinite(rho) else f"{rho:.6f}"
                report_sections.append(
                    f"| {rank} | `{feature}` | {importance:.6f} | {rho_text} | {feature_direction} |"
                )
        if len(top_three) < 3:
            report_sections.extend(
                [
                    "",
                    f"**只有 {len(top_three)} 项特征贡献高于可解释阈值，不足 3 项。**"
                    " 其余排名只是机器精度数值噪声，不作为 Top-3 发现。",
                ]
            )
        report_sections.extend(
            [
                "",
                "该方向表示模型如何复现代理规则，不表示特征对真实晕动症的因果方向。",
                "",
            ]
        )

    output_columns = [
        *id_columns,
        "label",
        "label_value",
        "feature",
        "feature_value",
        "shap_value_label_1",
        "mean_abs_shap",
        "direct_rule_feature",
        "label_source",
    ]
    pd.DataFrame(result_rows, columns=output_columns).to_csv(
        args.output, index=False, encoding="utf-8-sig", float_format="%.12g"
    )

    report_lines = [
        "# 风险因素规则复现模型 SHAP 报告",
        "",
        f"**{WARNING}**",
        "",
        "## 方法与输出",
        "",
        f"- 模型 bundle：`{args.model.as_posix()}`",
        f"- 训练矩阵：`{args.data.as_posix()}`",
        f"- SHAP 长表：`{args.output.as_posix()}`",
        f"- bundle 中实际可训练且可解释的标签：{trained_count}/{len(bundle['labels'])}",
        "- 对 LightGBM 使用 TreeSHAP；方向由特征值与标签 1 SHAP 值的 Spearman 相关表示。",
        "- 单类标签、全 unknown 标签不生成虚假 SHAP 值或图。",
        "",
        *report_sections,
        "## 解释边界",
        "",
        "1. 目标是由同一特征矩阵的阈值生成的代理标签，SHAP 只能解释规则复现模型。",
        "2. 直接规则特征及其强相关代理特征会导致同源泄漏；高 |SHAP| 不是独立风险证据。",
        "3. 当前唯一可拟合标签只有 1 个正例，特征排名和方向极不稳定。",
        "4. 本报告不解释 MS_rating/FMS/SSQ，不证明个体或场景的晕动症风险。",
        "",
        f"**{WARNING}**",
        "",
    ]
    args.report.write_text("\n".join(report_lines), encoding="utf-8")
    print(
        f"Wrote {len(result_rows)} SHAP rows for {trained_count} trained labels; "
        f"report={args.report}"
    )


if __name__ == "__main__":
    main()
