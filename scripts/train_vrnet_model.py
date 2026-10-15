#!/usr/bin/env python3
"""Run cautious local LightGBM feasibility evaluation and event-level correlations."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
import scipy
import sklearn
from joblib import dump
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier


FEATURE_PREFIXES = ("flow_", "depth_")
FEATURE_EXACT = {
    "hmd_head_translation_speed_source_units_s", "rotation_axes_count",
    "rotation_x_delta_rad", "rotation_y_delta_rad", "rotation_z_delta_rad",
    "path_curvature_source_units_inv",
}
LEVELS = ["low", "medium", "high"]


def write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="features/vrnet_training_matrix.csv")
    parser.add_argument("--model", default="models/vrnet_model.pkl")
    parser.add_argument("--report", default="results/vrnet_training_report.md")
    parser.add_argument("--correlation-report", default="results/vrnet_feature_correlation.md")
    parser.add_argument("--decision-report", default="results/local_validation_decision.md")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    resolve = lambda value: (root / value).resolve() if not Path(value).is_absolute() else Path(value)
    paths = {name: resolve(value) for name, value in vars(args).items()}
    frame = pd.read_csv(paths["input"], encoding="utf-8-sig")
    if "fms_label" not in frame or "fms_raw" not in frame:
        raise ValueError("Input must contain fms_label and fms_raw")
    feature_cols = [column for column in frame.columns
                    if (column.startswith(FEATURE_PREFIXES) or column in FEATURE_EXACT)
                    and pd.to_numeric(frame[column], errors="coerce").notna().any()]
    if not feature_cols:
        raise ValueError("No usable optical-flow, depth, pose, rotation, or curvature features")
    labeled = frame.loc[frame["fms_label"].isin(LEVELS)].copy().reset_index(drop=True)
    labels = labeled["fms_label"].astype(str)
    counts = {level: int((labels == level).sum()) for level in LEVELS}
    unique = labeled.drop_duplicates("fms_event_index") if "fms_event_index" in labeled else labeled
    correlations = []
    for column in feature_cols:
        x = pd.to_numeric(unique[column], errors="coerce")
        valid = x.notna() & pd.to_numeric(unique["fms_raw"], errors="coerce").notna()
        n = int(valid.sum())
        rho = pvalue = None
        status = "computed"
        if n < 3:
            status = "insufficient_n"
        elif x[valid].nunique() < 2 or unique.loc[valid, "fms_raw"].nunique() < 2:
            status = "constant_input_or_label"
        else:
            result = spearmanr(x[valid], pd.to_numeric(unique.loc[valid, "fms_raw"], errors="coerce"))
            rho, pvalue = float(result.statistic), float(result.pvalue)
        correlations.append({"feature": column, "n_unique_events": n, "spearman_rho": rho,
                             "p_value_unadjusted": pvalue, "status": status,
                             "unadjusted_p_lt_0_05": bool(pvalue is not None and pvalue < 0.05)})
    corr_df = pd.DataFrame(correlations).sort_values(
        "spearman_rho", key=lambda values: values.abs(), ascending=False, na_position="last")
    corr_df["p_value_holm"] = np.nan
    valid_p = corr_df["p_value_unadjusted"].notna()
    valid_indices = corr_df.index[valid_p].tolist()
    if valid_indices:
        p_values = corr_df.loc[valid_indices, "p_value_unadjusted"].astype(float).to_numpy()
        order = np.argsort(p_values)
        m = len(p_values)
        adjusted_sorted = np.maximum.accumulate((m - np.arange(m)) * p_values[order])
        adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
        adjusted = np.empty(m, dtype=float)
        adjusted[order] = adjusted_sorted
        corr_df.loc[valid_indices, "p_value_holm"] = adjusted
    corr_df["holm_p_lt_0_05"] = corr_df["p_value_holm"].lt(0.05)
    nominal_significant = corr_df.loc[corr_df["unadjusted_p_lt_0_05"]]
    corrected_significant = corr_df.loc[corr_df["holm_p_lt_0_05"]]
    corr_lines = [
        "# Mars Miners 特征与 MS_rating 的 Spearman 相关", "", "**流程验证，非最终结论。**",
        f"- 分析单位：按 `fms_event_index` 去重后的实际稀疏事件；n={len(unique)}（窗口重复沿用标签未作为独立事件重复计数）。",
        f"- 目标：源字段 `MS_rating` 的原始数值；本 session 观测值为 {sorted(pd.to_numeric(unique['fms_raw'], errors='coerce').dropna().unique().tolist())}。",
        "- 表中同时列原始未校正 p 值与 Holm 多重比较校正 p 值；事件少且标签仅有两个取值，显著性结果只作探索性描述。Spearman 相关不表示因果。", "",
        "## 结果", "", "| 特征 | n | Spearman ρ | p（未校正） | Holm 校正 p | 状态 |", "|---|---:|---:|---:|---:|---|"]
    for row in corr_df.to_dict("records"):
        rho = "未计算" if pd.isna(row["spearman_rho"]) else f"{row['spearman_rho']:.4f}"
        p = "未计算" if pd.isna(row["p_value_unadjusted"]) else f"{row['p_value_unadjusted']:.4g}"
        p_holm = "未计算" if pd.isna(row["p_value_holm"]) else f"{row['p_value_holm']:.4g}"
        status = "Holm 校正后 p<0.05" if row["holm_p_lt_0_05"] else "名义显著（未校正）" if row["unadjusted_p_lt_0_05"] else row["status"]
        corr_lines.append(f"| `{row['feature']}` | {row['n_unique_events']} | {rho} | {p} | {p_holm} | {status} |")
    corr_lines.extend(["", f"名义 p<0.05 特征数：{len(nominal_significant)} / {len(corr_df)}（未校正）。Holm 多重比较校正后 p<0.05 特征数：{len(corrected_significant)} / {len(corr_df)}。", ""])
    write_text(paths["correlation_report"], corr_lines)

    fit_status = "skipped"
    reasons = []
    metrics: dict[str, float | None] = {"accuracy": None, "f1_macro": None, "auc_ovr_macro": None}
    confusion = None
    feature_importance: list[tuple[str, float]] = []
    evaluation = "未执行"
    model_version = lightgbm.__version__
    if len(labeled) < 2:
        reasons.append("有标签窗口少于 2。")
    elif labels.nunique() < 2:
        reasons.append(f"所有 {len(labeled)} 个有标签窗口均属于 {labels.iloc[0]}；单类别不能拟合或评估多类别分类器。")
    else:
        x = labeled[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        y = labels.to_numpy()
        n_classes = len(np.unique(y))
        stratify = y if min(Counter(y).values()) >= 2 and math.ceil(len(y) * 0.2) >= n_classes else None
        x_train, x_valid, y_train, y_valid = train_test_split(
            x, y, test_size=0.2, random_state=42, stratify=stratify,
        )
        if len(np.unique(y_train)) < 2:
            reasons.append("80/20 拆分后的训练集缺少类别，LightGBM 无法拟合。")
        else:
            model = LGBMClassifier(
                objective="multiclass", num_class=len(np.unique(y_train)), n_estimators=80,
                learning_rate=0.03, max_depth=3, num_leaves=7, random_state=42, n_jobs=1, verbosity=-1,
            )
            model.fit(x_train, y_train)
            pred = model.predict(x_valid)
            probability = model.predict_proba(x_valid)
            classes = model.classes_
            metrics["accuracy"] = float(accuracy_score(y_valid, pred))
            metrics["f1_macro"] = float(f1_score(y_valid, pred, labels=classes, average="macro", zero_division=0))
            if len(np.unique(y_valid)) >= 2:
                if len(classes) == 2:
                    positive_index = list(classes).index("high") if "high" in classes else 1
                    metrics["auc_ovr_macro"] = float(roc_auc_score(y_valid == classes[positive_index], probability[:, positive_index]))
                else:
                    metrics["auc_ovr_macro"] = float(roc_auc_score(y_valid, probability, labels=classes, multi_class="ovr", average="macro"))
            confusion = confusion_matrix(y_valid, pred, labels=classes).tolist()
            feature_importance = sorted(zip(feature_cols, model.feature_importances_.astype(float)), key=lambda item: item[1], reverse=True)
            paths["model"].parent.mkdir(parents=True, exist_ok=True)
            dump({"model": model, "features": feature_cols, "classes": list(map(str, classes)), "random_seed": 42}, paths["model"])
            fit_status = "trained"
            evaluation = "80/20 随机窗口划分；非严格评估，同一 session 相邻窗相关，训练/验证不独立。"
    if fit_status == "skipped":
        # No placeholder/fabricated model artifact is written.
        if paths["model"].exists():
            reasons.append(f"当前运行未拟合模型；没有覆盖或宣称旧文件有效：{paths['model']}")

    cm_lines = ["未计算：当前数据不能形成有效分类验证集。"] if confusion is None else ["标签顺序按模型类别；行是真值，列是预测。", "", "```text"] + [" ".join(map(str, row)) for row in confusion] + ["```"]
    reason_lines = [f"- {reason}" for reason in reasons] if reasons else ["- 无"]
    train_lines = [
        "# Mars Miners LightGBM 本地训练报告", "", "**流程验证，非最终结论。指标不可靠，不表示已验证工具或个体风险预测。**", "",
        f"- 数据：`{paths['input'].relative_to(root)}`；窗口 {len(frame)}；有标签窗口 {len(labeled)}。",
        f"- 标签分布：low={counts['low']}, medium={counts['medium']}, high={counts['high']}。",
        f"- 训练状态：{fit_status}；LightGBM {model_version}。",
        f"- 评估：{evaluation}",
        f"- Accuracy：{metrics['accuracy'] if metrics['accuracy'] is not None else '未计算'}",
        f"- F1 (macro)：{metrics['f1_macro'] if metrics['f1_macro'] is not None else '未计算'}",
        f"- AUC（OVR macro）：{metrics['auc_ovr_macro'] if metrics['auc_ovr_macro'] is not None else '未计算'}",
        f"- 特征数：{len(feature_cols)}；种子：42。", "", "## 混淆矩阵", "", *cm_lines, "", "## 跳过原因", "",
        *reason_lines, "",
        "单 session 随机窗口 80/20 划分可能泄漏邻近时间结构；不能替代按参与者/游戏分组的外部验证。", "",
    ]
    write_text(paths["report"], train_lines)

    decision = [
        "# 本地最小验证决策", "", "**流程验证，非最终结论。**", "",
        f"- 来源：单个 VR.net Mars Miners (11_2) session；窗口 {len(frame)}；标签窗 {len(labeled)}。",
        f"- 标签：low={counts['low']}, medium={counts['medium']}, high={counts['high']}。",
        f"- 实际独立稀疏事件数（相关分析）：{len(unique)}；MS_rating 原始取值：{sorted(pd.to_numeric(unique['fms_raw'], errors='coerce').dropna().unique().tolist())}。",
        f"- 名义显著相关（p<0.05 未校正）：{len(nominal_significant)} / {len(corr_df)}；Holm 校正后：{len(corrected_significant)} / {len(corr_df)}。详见 `{paths['correlation_report'].relative_to(root)}`。",
        f"- 模型状态：{fit_status}；Accuracy={metrics['accuracy'] if metrics['accuracy'] is not None else '未计算'}；F1={metrics['f1_macro'] if metrics['f1_macro'] is not None else '未计算'}；AUC={metrics['auc_ovr_macro'] if metrics['auc_ovr_macro'] is not None else '未计算'}。",
    ]
    if feature_importance:
        decision.extend(["", "## 特征重要性（模型内置）", "", "| 排名 | 特征 | Importance |", "|---:|---|---:|"])
        decision.extend(f"| {i} | `{name}` | {score:.4g} |" for i, (name, score) in enumerate(feature_importance, 1))
    else:
        decision.extend(["", "## 特征重要性", "", "无：模型未拟合，因此没有 LightGBM 特征重要性。"])
    decision.extend([
        "", "## 是否值得上服务器", "",
        "**目前不建议直接投入服务器大规模训练。** 30 秒定义对应约 27 个窗（不是约 78 个）；本次最近事件标注后虽可有标签覆盖，但全部都落在 low 档，无法估计分类 AUC 或检验跨等级判别信号。事件级相关分析只有约 13 个稀疏观测，原始评分只含 1、2 两个值；2 个名义相关在 Holm 多重比较校正后均不显著，只能作为未经确认的探索线索。",
        "这不等于证明视觉特征无信号；当前设计没有足够标签变化来回答“是否值得扩展分类训练”。下一步优先获取覆盖多个评分等级、多个独立 session/游戏的标签，再按参与者/场景划分验证。", "",
        "## 下一步建议", "",
        "1. 不按本次 AUC 空缺结果做服务器训练决策；先确认 MS_rating 的量表定义及 1/2 的含义。",
        "2. 收集更多带中、高评分的独立 session，并采用按参与者/场景分组验证。",
        "3. 将 HMD 头部运动与场景相机运动分开；当前 `head_pos/head_dir` 是参与者 HMD 运动，可能是症状反应或行为变量，不应冒充预暴露场景设计特征。",
        "4. 相关分析按唯一稀疏事件去重；显著性未校正，不能当作确认性证据。", "",
    ])
    write_text(paths["decision_report"], decision)
    print(f"windows={len(frame)} labeled={len(labeled)} event_rows={len(unique)} labels={counts} features={len(feature_cols)} fit_status={fit_status} metric_auc={metrics['auc_ovr_macro']} nominal_correlations={len(nominal_significant)} holm_correlations={len(corrected_significant)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
