#!/usr/bin/env python3
"""Diagnose core-model SHAP directions with class and per-game distributions."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import yaml


def fmt(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("features/vrnet_training_matrix_v7_normalized.csv"))
    parser.add_argument("--raw", type=Path, default=Path("features/vrnet_training_matrix_v7_clean.csv"))
    parser.add_argument("--config", type=Path, default=Path("configs/feature_set_v7.yaml"))
    parser.add_argument("--model", type=Path, default=Path("models/vrnet_model_v7_core.pkl"))
    parser.add_argument("--values-output", type=Path, default=Path("features/shap_values_v7_core.csv"))
    parser.add_argument("--report", type=Path, default=Path("results/shap_diagnosis_v7.md"))
    args = parser.parse_args()
    data = pd.read_csv(args.input, encoding="utf-8-sig")
    raw = pd.read_csv(args.raw, encoding="utf-8-sig")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    features = list(config["feature_sets"]["core"])
    bundle = joblib.load(args.model)
    if list(bundle["feature_names"]) != features:
        raise ValueError("core model features differ from frozen v7 core configuration")
    pipeline = bundle["pipeline"]
    X = data[features].apply(pd.to_numeric, errors="coerce")
    transformed = np.asarray(pipeline[:-1].transform(X), dtype=float)
    classifier = pipeline.named_steps["classifier"]
    explanation = shap.TreeExplainer(classifier)(transformed)
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[:, :, 1]
    if values.shape != (len(data), len(features)):
        raise RuntimeError(f"unexpected SHAP shape: {values.shape}")
    probability = pipeline.predict_proba(X)[:, 1]
    output = pd.DataFrame({
        "session_id": data["session_id"], "game_name": data["game_name"],
        "fms_label": data["fms_label"], "probability_medium": probability,
    })
    for index, feature in enumerate(features):
        output[f"shap_{feature}"] = values[:, index]
    args.values_output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.values_output, index=False, encoding="utf-8-sig", float_format="%.12g")

    rows = []
    detail_sections = []
    fixed = 0
    reversed_features: list[str] = []
    for index, feature in enumerate(features):
        normalized = pd.to_numeric(data[feature], errors="coerce")
        shap_feature = pd.Series(values[:, index])
        correlation = normalized.corr(shap_feature, method="spearman")
        correlation = float(correlation) if pd.notna(correlation) else np.nan
        direction = "medium" if np.isfinite(correlation) and correlation > 0 else "low" if np.isfinite(correlation) and correlation < 0 else "unknown"
        status = "符合预期" if direction == "medium" else "方向相反" if direction == "low" else "无法判定"
        fixed += int(status == "符合预期")
        if status == "方向相反":
            reversed_features.append(feature)
        y = data["fms_label_binary"].astype(int)
        low = normalized[y == 0]
        medium = normalized[y == 1]
        raw_values = pd.to_numeric(raw[feature], errors="coerce")
        raw_low = raw_values[y == 0]
        raw_medium = raw_values[y == 1]
        mean_difference = float(medium.mean() - low.mean())
        game_differences = []
        for game, subset in data.assign(_value=normalized).groupby("game_name"):
            if subset["fms_label_binary"].nunique() < 2:
                continue
            game_low = subset.loc[subset["fms_label_binary"].eq(0), "_value"].mean()
            game_medium = subset.loc[subset["fms_label_binary"].eq(1), "_value"].mean()
            game_differences.append((str(game), float(game_medium - game_low)))
        supports = sum(delta > 0 for _, delta in game_differences)
        opposes = sum(delta < 0 for _, delta in game_differences)
        largest_game, largest_delta = max(game_differences, key=lambda item: abs(item[1])) if game_differences else ("NA", np.nan)
        if status == "方向相反":
            if mean_difference < 0 and opposes >= supports:
                reason = "观察到的 medium 组分布更低，且多数可比较游戏同向；标签关联本身与预期相反"
            elif mean_difference >= 0:
                reason = "类别均值未反向，但树模型 SHAP 反向；可能来自非线性交互、共线性或小样本不稳定"
            else:
                reason = "游戏内方向混合，可能由游戏异质性和少数游戏主导"
        elif status == "符合预期":
            reason = "当前 SHAP 方向与预期一致，但不构成因果或外部验证证据"
        else:
            reason = "特征或 SHAP 近似常数，无法稳定判断"
        rows.append(
            f"| {feature} | {float(np.mean(np.abs(values[:, index]))):.6g} | {fmt(correlation)} | "
            f"{direction} | {status} | {mean_difference:.6f} | {supports}/{opposes} | {largest_game} ({fmt(largest_delta)}) |"
        )
        detail_sections.extend([
            "", f"### {feature}", "",
            f"- 标准化 low：mean={fmt(float(low.mean()))}, median={fmt(float(low.median()))}, std={fmt(float(low.std()))}",
            f"- 标准化 medium：mean={fmt(float(medium.mean()))}, median={fmt(float(medium.median()))}, std={fmt(float(medium.std()))}",
            f"- 原尺度 low：mean={fmt(float(raw_low.mean()))}, median={fmt(float(raw_low.median()))}",
            f"- 原尺度 medium：mean={fmt(float(raw_medium.mean()))}, median={fmt(float(raw_medium.median()))}",
            f"- 可比较游戏：{len(game_differences)}；medium > low：{supports}；medium < low：{opposes}",
            f"- 最大游戏内差异：{largest_game}，medium-low={fmt(largest_delta)}",
            f"- 诊断：{reason}。",
        ])

    lines = [
        "# VR.net v7 核心集 SHAP 方向诊断", "",
        "**探索性结果，非最终结论。SHAP 解释的是当前 LightGBM 模型关联，不表示因果风险因素。**", "",
        f"- session：{len(data)}", f"- 核心特征：{len(features)}",
        "- 输入：清理旋转异常后按游戏 z-score 的矩阵。",
        "- 方向定义：特征值与该特征 SHAP 值的 Spearman 相关符号。",
        "- 预期：六个核心特征高值均倾向 medium。", "",
        "## 汇总", "",
        "| 特征 | mean(|SHAP|) | Spearman(feature, SHAP) | 当前高值方向 | 检查 | 标准化均值差 medium-low | 游戏支持/反对 | 最大游戏内差异 |",
        "|---|---:|---:|---|---|---:|---:|---|", *rows, "",
        f"- 符合预期：{fixed}/{len(features)}",
        f"- 方向相反：{len(reversed_features)}/{len(features)}（{', '.join(reversed_features) if reversed_features else '无'}）", "",
        "## 分布与游戏主导诊断", *detail_sections, "",
        "## 结论", "",
        (
            "核心集 SHAP 方向未完全修复。反向项需要结合类别分布与逐游戏差异理解，不能通过删掉不符合直觉的结果来调整特征集。"
            if reversed_features else
            "六个核心特征方向均与预期一致，但小样本和 transductive 标准化仍限制解释。"
        ),
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"direction_expected={fixed}/{len(features)} reversed={reversed_features}")
    print(f"Wrote {args.values_output} and {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
