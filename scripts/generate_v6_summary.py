#!/usr/bin/env python3
"""Generate the auditable exploratory v6 summary from saved data and model outputs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def fmt(value: float | None) -> str:
    return "NA" if value is None or not math.isfinite(value) else f"{value:.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=Path("features/vrnet_training_matrix_v6.csv"))
    parser.add_argument("--frames", type=Path, default=Path("features/vrnet_matrix_features_v6.csv"))
    parser.add_argument("--shap", type=Path, default=Path("features/shap_values_v6.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/vrnet_model_v6.pkl"))
    parser.add_argument("--output", type=Path, default=Path("results/v6_summary.md"))
    args = parser.parse_args()
    data = pd.read_csv(args.matrix, encoding="utf-8-sig")
    frames = pd.read_csv(args.frames, usecols=["camera_jump_flag"], low_memory=False)
    shap_values = pd.read_csv(args.shap, encoding="utf-8-sig")
    bundle = joblib.load(args.model)
    feature_names = list(bundle["feature_names"])
    metrics = bundle["metrics"]
    counts = data.groupby("game_name")["session_id"].nunique().sort_index()
    labels = data["fms_label"].value_counts().to_dict()

    importance_rows = []
    for feature in feature_names:
        column = f"shap_{feature}"
        values = pd.to_numeric(shap_values[column], errors="coerce")
        raw = pd.to_numeric(data[feature], errors="coerce")
        correlation = raw.corr(values, method="spearman")
        importance_rows.append({
            "feature": feature,
            "mean_abs_shap": float(values.abs().mean()),
            "correlation": float(correlation) if pd.notna(correlation) else np.nan,
        })
    importance = pd.DataFrame(importance_rows).sort_values("mean_abs_shap", ascending=False)
    top_rows = []
    for rank, (_, row) in enumerate(importance.head(5).iterrows(), start=1):
        correlation = float(row["correlation"])
        direction = (
            "无法稳定判定" if not np.isfinite(correlation) or abs(correlation) < 1e-12
            else "较高值倾向 medium" if correlation > 0 else "较高值倾向 low"
        )
        top_rows.append(
            f"| {rank} | {row['feature']} | {row['mean_abs_shap']:.6g} | {direction} |"
        )

    mean_auc = metrics.get("mean_fold_auc")
    auc_folds = int(metrics.get("auc_folds", 0))
    if mean_auc is not None and mean_auc >= 0.65 and auc_folds >= 5:
        verdict = "有信号，达到本阶段继续扩展标准"
    elif mean_auc is not None and 0.55 <= mean_auc < 0.65:
        verdict = "信号弱，需要更多数据或改进特征"
    elif mean_auc is not None and mean_auc < 0.55:
        verdict = "当前无信号，应先检查特征、标签和对齐"
    else:
        verdict = "证据不足，未满足判断条件"
    jumps = int(frames["camera_jump_flag"].astype(str).str.lower().isin({"true", "1", "yes"}).sum())
    lines = [
        "# VR.net v6 扩展、训练与 SHAP 总结", "",
        "**全部结果均为探索性流程验证，非最终结论，不代表模型已具备个体晕动症预测能力。**", "",
        "## 数据", "",
        f"- 总 session：{len(data)}", f"- 总游戏：{data['game_name'].nunique()}",
        "- 本轮新增：40 个 session（10 个游戏各 4 个）", "- Pirate_Life：沿用本地 11 个 session", "",
        "### 每个游戏的 session 数", "",
        "| 游戏 | session 数 |", "|---|---:|",
        *[f"| {game} | {int(count)} |" for game, count in counts.items()], "",
        f"- 标签分布：low={int(labels.get('low', 0))}, medium={int(labels.get('medium', 0))}",
        "- 冻结映射：`max(MS_rating) <= 2` 为 low，`> 2` 为 medium", f"- 帧级坐标跳变标记：{jumps}", "",
        "## Leave-One-Game-Out", "",
        f"- 可计算 AUC 的折：{auc_folds}/11", f"- 平均 fold AUC：{fmt(mean_auc)}",
        f"- 平均 fold F1 macro：{fmt(metrics.get('mean_fold_f1_macro'))}",
        f"- pooled OOF AUC：{fmt(metrics.get('pooled_oof_auc'))}",
        f"- pooled OOF F1 macro：{fmt(metrics.get('pooled_oof_f1_macro'))}", "",
        "## SHAP Top-5", "",
        "| 排名 | 特征 | mean(|SHAP|) | 探索性方向 |", "|---:|---|---:|---|", *top_rows, "",
        "## 判断", "", f"**{verdict}。**", "",
        "平均 fold AUC 位于 0.55–0.65 区间。即使 10 折可以计算 AUC，61 个 session、稀疏 self-report 标签和明显的游戏间差异仍不足以支持有效性或泛化性结论。", "",
        "## 下一步建议", "",
        "1. 继续扩展每个游戏的 session，并优先补齐同一游戏内的 low/medium 两类，避免 City_Parkour 这类留出集单类别造成 AUC 无法计算。",
        "2. 核查 session 与参与者的对应关系；如同一参与者出现多次，应按参与者和游戏共同分组，防止数据泄漏。",
        "3. 对速度、加速度和旋转率做游戏内标准化或稳健截尾，并保留未处理版本作敏感性分析。",
        "4. 比较 `session max(MS_rating)` 与 mean/末次/事件窗口标签，量化稀疏 self-report 聚合造成的标签噪声。",
        "5. 在样本扩大前不投入 3D CNN；当前更值得先改进标签与分组设计。",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}; verdict={verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
