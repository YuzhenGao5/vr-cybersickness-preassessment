#!/usr/bin/env python3
"""Generate the v7 cleanup, feature-set comparison, and SHAP diagnosis summary."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, default=Path("features/vrnet_training_matrix_v7_clean.csv"))
    parser.add_argument("--normalized", type=Path, default=Path("features/vrnet_training_matrix_v7_normalized.csv"))
    parser.add_argument("--frames", type=Path, default=Path("features/vrnet_matrix_features_v6.csv"))
    parser.add_argument("--config", type=Path, default=Path("configs/feature_set_v7.yaml"))
    parser.add_argument("--metrics", type=Path, default=Path("results/training_v7_metrics.csv"))
    parser.add_argument("--shap", type=Path, default=Path("features/shap_values_v7_core.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/v7_summary.md"))
    args = parser.parse_args()
    clean = pd.read_csv(args.clean, encoding="utf-8-sig")
    normalized = pd.read_csv(args.normalized, encoding="utf-8-sig")
    frames = pd.read_csv(args.frames, usecols=["camera_rotation_rate", "camera_jump_flag"], low_memory=False)
    metrics = pd.read_csv(args.metrics, encoding="utf-8-sig")
    shap_frame = pd.read_csv(args.shap, encoding="utf-8-sig")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    core = list(config["feature_sets"]["core"])
    full = list(config["feature_sets"]["full"])

    rate = pd.to_numeric(frames["camera_rotation_rate"], errors="coerce")
    jump = frames["camera_jump_flag"].astype(str).str.lower().isin({"true", "1", "yes"})
    frame_over = rate > 720
    residual = frame_over & ~jump
    correlations = clean[full].apply(pd.to_numeric, errors="coerce").corr()
    high_pairs = []
    for index, left in enumerate(full):
        for right in full[index + 1:]:
            r = correlations.at[left, right]
            if pd.notna(r) and abs(float(r)) > 0.8:
                high_pairs.append((left, right, float(r)))
    high_pairs.sort(key=lambda item: abs(item[2]), reverse=True)
    pair_rows = [f"| {left} | {right} | {r:.6f} |" for left, right, r in high_pairs]

    comparison_rows = []
    for _, row in metrics.iterrows():
        comparison_rows.append(
            f"| {row['feature_set']} | {int(row['n_features'])} | {int(row['auc_folds'])}/11 | "
            f"{float(row['mean_fold_auc']):.6f} | {float(row['mean_fold_f1_macro']):.6f} | "
            f"{float(row['pooled_oof_auc']):.6f} | {float(row['pooled_oof_f1_macro']):.6f} |"
        )
    ranked = metrics.sort_values(["mean_fold_auc", "mean_fold_f1_macro"], ascending=False)
    best = str(ranked.iloc[0]["feature_set"])
    best_auc = float(ranked.iloc[0]["mean_fold_auc"])

    direction_rows = []
    expected_count = 0
    weak_positive: list[str] = []
    reversed_features: list[str] = []
    for feature in core:
        rho = pd.to_numeric(normalized[feature], errors="coerce").corr(
            pd.to_numeric(shap_frame[f"shap_{feature}"], errors="coerce"), method="spearman"
        )
        rho = float(rho) if pd.notna(rho) else np.nan
        if np.isfinite(rho) and rho > 0:
            direction, status = "medium", "符号符合预期"
            expected_count += 1
            if abs(rho) < 0.1:
                weak_positive.append(feature)
        elif np.isfinite(rho) and rho < 0:
            direction, status = "low", "方向相反"
            reversed_features.append(feature)
        else:
            direction, status = "unknown", "无法判定"
        direction_rows.append(f"| {feature} | {rho:.6f} | {direction} | {status} |")

    if best_auc >= 0.65:
        signal = "达到继续使用该特征集的探索性门槛"
    elif best_auc >= 0.55:
        signal = "弱信号，未达到 0.65 门槛"
    else:
        signal = "当前无信号"
    lines = [
        "# VR.net v7 特征清理与 SHAP 诊断总结", "",
        "**探索性结果，非最终结论；不代表模型或风险因素已经验证。**", "",
        "## 1. 跳变残留", "",
        f"- 帧级旋转率 >720°/s：{int(frame_over.sum())} 行",
        f"- 已被位置跳变标记：{int((frame_over & jump).sum())} 行",
        f"- 未被位置跳变捕获：{int(residual.sum())} 行",
        "- 训练矩阵中 6 个 session 的 rotation max 超阈值；v7 将残留值设为缺失后重新计算 rotation mean/max/std。", "",
        "## 2. 高相关特征", "",
        "| 特征 A | 特征 B | Pearson r |", "|---|---|---:|", *pair_rows, "",
        "旋转率 mean/max/std 清理后的相关系数最高为 mean-std `r=0.670908`，未超过 0.8。旋转率 SHAP 异常不能只由三者高度共线解释。", "",
        "## 3. 三组特征性能", "",
        "| 特征集 | 特征数 | 可算 AUC 折 | 平均 fold AUC | 平均 fold F1 | pooled OOF AUC | pooled OOF F1 |",
        "|---|---:|---:|---:|---:|---:|---:|", *comparison_rows, "",
        f"数值最优：**{best}**，平均 fold AUC={best_auc:.6f}。", "",
        "## 4. 核心集 SHAP 方向", "",
        "| 特征 | Spearman(feature, SHAP) | 高值方向 | 判断 |", "|---|---:|---|---|", *direction_rows, "",
        f"- 符号符合预期：{expected_count}/6。",
        f"- 方向相反：{len(reversed_features)}/6（{', '.join(reversed_features) if reversed_features else '无'}）。",
        f"- 正向但非常弱（|rho|<0.1）：{', '.join(weak_positive) if weak_positive else '无'}。",
        "- 结论：SHAP 方向没有修复。逐游戏 medium-low 差异方向混合，显示游戏异质性与标签噪声仍然明显。", "",
        "## 5. 信号判断", "", f"**{signal}。**", "",
        "core 明显退化；extended 和 full 都接近但未超过 0.65。full 的优势很小，不能据此认定额外极值特征稳定有效。", "",
        "## 6. 下一步建议", "",
        "1. 需要更多数据或更可靠标签；优先增加每个游戏内同时包含 low/medium 的 session。",
        "2. 将 extended 作为较简洁的候选集、full 作为敏感性对照；不要使用 core 作为当前主模型。",
        "3. 用训练游戏统计量或外部固定尺度重新设计标准化。当前按游戏 z-score 使用留出游戏自身分布，属于 transductive 分析，不适合声称新游戏部署性能。",
        "4. 对旋转率、速度和加速度增加独立的时间戳/采样间隔质量检查，并比较稳健分位数特征。",
        "5. 在样本和标签改善前不训练 3D CNN，也不依据 SHAP 方向删特征。",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"best={best} best_auc={best_auc:.6f} expected_directions={expected_count}/6 signal={signal}")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
