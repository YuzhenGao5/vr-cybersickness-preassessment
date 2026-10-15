#!/usr/bin/env python3
"""Analyze Pearson feature correlation and recommend a reduced feature representation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def value(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("features/vrnet_training_matrix_v7_clean.csv"))
    parser.add_argument("--config", type=Path, default=Path("configs/feature_set_v7.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/feature_correlation_matrix.md"))
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()
    data = pd.read_csv(args.input, encoding="utf-8-sig")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    features = list(config["feature_sets"]["full"])
    core = set(config["feature_sets"]["core"])
    extended = set(config["feature_sets"]["extended"])
    numeric = data[features].apply(pd.to_numeric, errors="coerce")
    correlation = numeric.corr(method="pearson")
    pairs = []
    for index, left in enumerate(features):
        for right in features[index + 1:]:
            r = correlation.at[left, right]
            if pd.notna(r) and abs(float(r)) > args.threshold:
                pairs.append((left, right, float(r)))
    pairs.sort(key=lambda item: abs(item[2]), reverse=True)

    matrix_header = "| feature | " + " | ".join(features) + " |"
    matrix_rule = "|---|" + "---:|" * len(features)
    matrix_rows = [
        "| " + feature + " | " + " | ".join(value(float(correlation.at[feature, other])) for other in features) + " |"
        for feature in features
    ]
    pair_rows = []
    for left, right, r in pairs:
        if left in core and right not in core:
            keep, drop = left, right
        elif right in core and left not in core:
            keep, drop = right, left
        elif left in extended and right not in extended:
            keep, drop = left, right
        elif right in extended and left not in extended:
            keep, drop = right, left
        else:
            keep, drop = left, right
        pair_rows.append(f"| {left} | {right} | {r:.6f} | {keep} | {drop} |")
    constant = [feature for feature in features if numeric[feature].nunique(dropna=True) <= 1]
    lines = [
        "# v7 特征 Pearson 相关性", "",
        f"- session：{len(data)}", f"- 特征：{len(features)}",
        f"- 高相关阈值：`|r| > {args.threshold:g}`", "- 输入为清理后的未标准化 session 特征。", "",
        "## 相关矩阵", "", matrix_header, matrix_rule, *matrix_rows, "",
        "## 高度相关特征对", "",
        "| 特征 A | 特征 B | Pearson r | 建议保留 | 建议去掉 |",
        "|---|---|---:|---|---|",
        *(pair_rows or ["| 无 | 无 | NA | NA | NA |"]), "",
        "## 建议", "",
        "- 核心集优先保留每类指标的 mean，减少同一统计量族的 max/std 共线性。",
        "- 扩展集只加入旋转率 mean/std 与 FOV std，用于检验动态变化是否提供额外信号。",
        "- 完整集保留全部 18 个特征作为敏感性对照，不作为默认解释集。",
        f"- 常数或近似常数检查（严格唯一值数 <= 1）：{', '.join(constant) if constant else '无'}。",
        "- 表中的保留/去掉建议按 core > extended > full 的预先优先级生成，不依据 AUC 调整特征集。",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"high_correlation_pairs={len(pairs)} constant_features={constant}")
    for left, right, r in pairs:
        print(f"{left} <-> {right}: r={r:.6f}")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
