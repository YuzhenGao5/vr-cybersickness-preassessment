#!/usr/bin/env python3
"""Run TreeSHAP only for actually fitted video-only LightGBM label models."""

from __future__ import annotations

import argparse
import math
import re
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr


WARNING = "探索性，非最终结论；规则生成代理标签；因素检测不等于晕动症风险验证。"
ID_COLUMNS = ["clip_id", "session_id", "game_name"]
UNKNOWN_TOKENS = {"", "unknown", "na", "n/a", "nan", "none", "null", "tbd"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=Path("models/lightgbm_baseline.pkl")
    )
    parser.add_argument(
        "--data", type=Path, default=Path("features/lightgbm_training_matrix.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("features/shap_lightgbm_values.csv")
    )
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument(
        "--report", type=Path, default=Path("results/shap_lightgbm_summary.md")
    )
    parser.add_argument(
        "--log", type=Path, default=Path("results/shap_lightgbm_log.txt")
    )
    return parser.parse_args()


def normalize_target(series: pd.Series, label: str) -> pd.Series:
    result = pd.Series(np.nan, index=series.index, dtype=float)
    for index, value in series.items():
        if pd.isna(value) or str(value).strip().lower() in UNKNOWN_TOKENS:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} has invalid value {value!r} at row {index}") from exc
        if numeric not in (0.0, 1.0):
            raise ValueError(f"{label} has non-binary value {value!r} at row {index}")
        result.at[index] = numeric
    return result


def positive_class_shap(values: Any, n_samples: int, n_features: int) -> np.ndarray:
    if isinstance(values, list):
        if not values:
            raise ValueError("SHAP returned an empty list")
        values = values[1] if len(values) > 1 else values[0]
    array = np.asarray(values, dtype=float)
    if array.ndim == 2 and array.shape == (n_samples, n_features):
        return array
    if array.ndim == 3:
        if array.shape[:2] == (n_samples, n_features):
            return array[:, :, 1 if array.shape[2] > 1 else 0]
        if array.shape[1:] == (n_samples, n_features):
            return array[1 if array.shape[0] > 1 else 0, :, :]
    raise ValueError(
        f"Unsupported SHAP shape {array.shape}; expected ({n_samples}, {n_features})"
    )


def estimator_and_values(model: Any, values: pd.DataFrame) -> tuple[Any, pd.DataFrame]:
    if hasattr(model, "named_steps"):
        steps = model.named_steps
        estimator = steps.get("classifier", list(steps.values())[-1])
        transformed: Any = values
        for name, step in steps.items():
            if step is estimator or name == "classifier":
                break
            transformed = step.transform(transformed)
        array = np.asarray(transformed, dtype=float)
    else:
        estimator = model
        array = values.to_numpy(dtype=float)
    if array.shape != values.shape:
        raise ValueError(
            f"Preprocessing changed shape from {values.shape} to {array.shape}; "
            "cannot align SHAP names safely"
        )
    return estimator, pd.DataFrame(array, columns=values.columns, index=values.index)


def robust_direction(values: np.ndarray, shap_values: np.ndarray) -> tuple[float, str]:
    finite = np.isfinite(values) & np.isfinite(shap_values)
    if finite.sum() < 3 or np.unique(values[finite]).size < 2:
        return math.nan, "无法判定（特征取值不足）"
    contribution = shap_values[finite]
    scale = max(1.0, float(np.max(np.abs(contribution))))
    tolerance = 1e-10 * scale
    if float(np.max(np.abs(contribution))) <= 1e-12:
        return math.nan, "贡献近零，不解释方向"
    quantized = np.round(contribution / tolerance) * tolerance
    if np.unique(quantized).size < 2:
        return math.nan, "SHAP 值为常数，不解释方向"
    rho = float(spearmanr(values[finite], quantized).statistic)
    if not math.isfinite(rho):
        return math.nan, "方向未定义"
    if abs(rho) < 0.1:
        return rho, "方向很弱/不明确"
    return rho, "高值推向标签 1" if rho > 0 else "高值推离标签 1"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> None:
    args = parse_args()
    for path in (args.model, args.data):
        if not path.exists():
            raise FileNotFoundError(path)
    bundle = joblib.load(args.model)
    if not isinstance(bundle, dict) or not isinstance(bundle.get("labels"), dict):
        raise TypeError("Expected a joblib dict bundle containing labels")
    feature_names = list(bundle.get("feature_names") or [])
    if not feature_names:
        raise ValueError("Bundle has no explicit feature_names whitelist")
    data = pd.read_csv(args.data, encoding="utf-8-sig", low_memory=False)
    missing = [column for column in [*ID_COLUMNS, *feature_names] if column not in data.columns]
    if missing:
        raise ValueError(f"Training matrix is missing bundle fields: {missing}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    # Keep the figure directory consistent with this bundle. Only remove this
    # script's own prior outputs; unrelated figures are untouched.
    for stale_figure in args.figures_dir.glob("shap_lightgbm_*.png"):
        stale_figure.unlink()

    started = time.perf_counter()
    result_rows: list[dict[str, Any]] = []
    sections: list[str] = []
    log_lines = [
        "LightGBM video-only baseline SHAP log",
        f"model={args.model}",
        f"data={args.data}",
        f"output={args.output}",
        "only status=trained models are explained",
        "machine-noise floor=max(1e-12, top mean_abs_SHAP * 1e-9)",
        "",
    ]
    figure_count = 0
    explained_labels = 0

    for label, entry in bundle["labels"].items():
        status = str(entry.get("status", "missing_status"))
        model = entry.get("model")
        if status != "trained" or model is None:
            sections.extend(
                [
                    f"## `{label}`",
                    "",
                    f"- 状态：`{status}`",
                    f"- 0/1/unknown：{entry.get('n_negative', 'NA')}/"
                    f"{entry.get('n_positive', 'NA')}/{entry.get('n_unknown', 'NA')}",
                    "- SHAP：未生成；该标签没有实际拟合的双类模型。",
                    "",
                ]
            )
            log_lines.append(f"SKIP {label}: status={status} model={model is not None}")
            continue
        if label not in data.columns:
            raise ValueError(f"Training matrix is missing fitted label column {label}")
        target = normalize_target(data[label], label)
        names = list(entry.get("feature_names") or feature_names)
        missing_label_features = [column for column in names if column not in data.columns]
        if missing_label_features:
            raise ValueError(f"{label} is missing features {missing_label_features}")
        numeric_features = data[names].apply(pd.to_numeric, errors="coerce")
        known = target.notna() & numeric_features.notna().any(axis=1)
        if "extraction_status" in data.columns:
            known &= data["extraction_status"].isin(["success", "partial_single_frame"])
        if known.sum() == 0:
            raise ValueError(f"Bundle says {label} is trained but data has no known usable targets")
        original = numeric_features.loc[known]
        estimator, model_values = estimator_and_values(model, original)
        explainer = shap.TreeExplainer(estimator)
        shap_array = positive_class_shap(
            explainer.shap_values(model_values), len(model_values), len(names)
        )
        abs_mean = np.mean(np.abs(shap_array), axis=0)
        ranking = np.argsort(abs_mean)[::-1]
        top = float(abs_mean[ranking[0]]) if len(ranking) else 0.0
        floor = max(1e-12, top * 1e-9)
        meaningful = [int(index) for index in ranking if float(abs_mean[int(index)]) > floor]
        top_five: list[tuple[str, float, float, str]] = []
        for feature_index in meaningful[:5]:
            feature = names[feature_index]
            rho, interpretation = robust_direction(
                model_values[feature].to_numpy(dtype=float), shap_array[:, feature_index]
            )
            top_five.append((feature, float(abs_mean[feature_index]), rho, interpretation))

        known_indices = data.index[known]
        for row_position, data_index in enumerate(known_indices):
            for feature_index, feature in enumerate(names):
                result_rows.append(
                    {
                        "clip_id": data.at[data_index, "clip_id"],
                        "session_id": data.at[data_index, "session_id"],
                        "game_name": data.at[data_index, "game_name"],
                        "label": label,
                        "label_value": int(target.at[data_index]),
                        "feature": feature,
                        "feature_value": float(model_values.iloc[row_position, feature_index]),
                        "shap_value_label_1": float(shap_array[row_position, feature_index]),
                        "mean_abs_shap": float(abs_mean[feature_index]),
                        "label_source": "matrix_rule_generated_proxy",
                    }
                )

        figure_path = args.figures_dir / f"shap_lightgbm_{safe_name(label)}.png"
        if meaningful:
            plot_names = [names[index] for index in meaningful]
            plt.figure()
            shap.summary_plot(
                shap_array[:, meaningful],
                model_values.iloc[:, meaningful],
                feature_names=plot_names,
                max_display=min(15, len(plot_names)),
                show=False,
                plot_size=(10, 7),
            )
            plt.title(f"Video-only LightGBM SHAP: {label}\nrule proxy; exploratory only")
            plt.tight_layout()
            plt.savefig(figure_path, dpi=180, bbox_inches="tight")
            plt.close("all")
            figure_count += 1
        explained_labels += 1

        sections.extend(
            [
                f"## `{label}`",
                "",
                f"- 状态：`{status}`",
                f"- 已知标签：{int(known.sum())}；0/1："
                f"{entry.get('n_negative', 'NA')}/{entry.get('n_positive', 'NA')}",
                f"- 可解释阈值：`mean_abs_SHAP > {floor:.6g}`",
                f"- 高于阈值的特征：{len(meaningful)}/{len(names)}",
                f"- 图：`{figure_path.as_posix()}`"
                if meaningful
                else "- 图：未生成；全部特征贡献均低于机器噪声阈值。",
                "",
            ]
        )
        if top_five:
            sections.extend(
                [
                    "| 排名 | 特征 | mean_abs_SHAP | Spearman(feature, SHAP) | 方向 |",
                    "|---:|---|---:|---:|---|",
                ]
            )
            for rank, (feature, importance, rho, interpretation) in enumerate(top_five, 1):
                sections.append(
                    f"| {rank} | `{feature}` | {importance:.6f} | {fmt(rho)} | {interpretation} |"
                )
        else:
            sections.append("没有高于可解释阈值的 SHAP 特征。")
        sections.extend(
            [
                "",
                "这些方向只解释视频代理模型对规则标签的复现，不是晕动症因果方向。",
                "",
            ]
        )
        log_lines.append(
            f"EXPLAIN {label}: rows={int(known.sum())} features={len(names)} "
            f"meaningful={len(meaningful)} figure={meaningful != []}"
        )

    output_columns = [
        *ID_COLUMNS,
        "label",
        "label_value",
        "feature",
        "feature_value",
        "shap_value_label_1",
        "mean_abs_shap",
        "label_source",
    ]
    pd.DataFrame(result_rows, columns=output_columns).to_csv(
        args.output, index=False, encoding="utf-8-sig", float_format="%.12g"
    )
    report_lines = [
        "# LightGBM 视频代理特征 SHAP 总结",
        "",
        f"**{WARNING}**",
        "",
        "## 门控与方法",
        "",
        f"- 模型 bundle：`{args.model.as_posix()}`",
        f"- 训练矩阵：`{args.data.as_posix()}`",
        f"- SHAP 长表：`{args.output.as_posix()}`",
        f"- 实际有拟合模型并运行 SHAP 的标签：{explained_labels}/{len(bundle['labels'])}",
        f"- 实际生成图数：{figure_count}",
        "- 只对 bundle 中 `status=trained` 且 `model` 存在的标签运行 TreeSHAP。",
        "- 单类、全 unknown 或其他不可训练标签不生成虚假 SHAP 值或图。",
        "- 用 `max(1e-12, top mean_abs_SHAP × 1e-9)` 过滤机器精度噪声。",
        "",
        *sections,
        "## 解释边界",
        "",
        "1. 标签是场景/矩阵规则代理，不是 MS_rating、FMS 或 SSQ。",
        "2. 这一步不能验证特征与真实晕动症的关联、因果性或预测能力。",
        "3. 光流与单应性依赖视频采样和像素坐标；不是真实米制相机/物体运动。",
        "4. 没有可训练标签时，空 SHAP CSV 和零张图是真实的门控结果，不应用人工数值补齐。",
        "",
        f"**{WARNING}**",
        "",
    ]
    args.report.write_text("\n".join(report_lines), encoding="utf-8")
    log_lines.extend(
        [
            "",
            f"explained_labels={explained_labels}",
            f"figures={figure_count}",
            f"shap_rows={len(result_rows)}",
            f"elapsed_sec={time.perf_counter()-started:.3f}",
        ]
    )
    args.log.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(result_rows)} SHAP rows for {explained_labels} labels; "
        f"figures={figure_count}; report={args.report}"
    )


if __name__ == "__main__":
    main()
