#!/usr/bin/env python3
"""Compare frozen core, extended, and full LightGBM feature sets with LOGO."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline


def model(seed: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("classifier", LGBMClassifier(
            n_estimators=100, learning_rate=0.05, num_leaves=7,
            min_child_samples=5, class_weight="balanced",
            random_state=seed, verbosity=-1,
        )),
    ])


def fmt(value: float | None) -> str:
    return "NA" if value is None or not math.isfinite(value) else f"{value:.6f}"


def evaluate(data: pd.DataFrame, features: list[str], seed: int) -> tuple[list[dict[str, object]], dict[str, float | int | None]]:
    X = data[features].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(data["fms_label_binary"], errors="raise").astype(int)
    groups = data["game_name"].astype(str)
    oof_probability = np.full(len(data), np.nan)
    oof_prediction = np.full(len(data), -1, dtype=int)
    records: list[dict[str, object]] = []
    for fold, (train_idx, test_idx) in enumerate(LeaveOneGroupOut().split(X, y, groups), start=1):
        held_out = str(groups.iloc[test_idx[0]])
        estimator = model(seed)
        estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
        prediction = estimator.predict(X.iloc[test_idx])
        probability = estimator.predict_proba(X.iloc[test_idx])[:, 1]
        oof_prediction[test_idx] = prediction
        oof_probability[test_idx] = probability
        y_test = y.iloc[test_idx]
        can_auc = y_test.nunique() == 2
        records.append({
            "fold": fold, "held_out_game": held_out, "n_test": len(test_idx),
            "test_low": int((y_test == 0).sum()), "test_medium": int((y_test == 1).sum()),
            "accuracy": float(accuracy_score(y_test, prediction)),
            "f1_macro": float(f1_score(y_test, prediction, labels=[0, 1], average="macro", zero_division=0)),
            "auc": float(roc_auc_score(y_test, probability)) if can_auc else None,
            "auc_reason": "可计算" if can_auc else "测试集只有一个类别",
            "confusion": str(confusion_matrix(y_test, prediction, labels=[0, 1]).tolist()),
        })
    if not (oof_prediction >= 0).all():
        raise RuntimeError("incomplete OOF predictions")
    valid_auc = [float(row["auc"]) for row in records if row["auc"] is not None]
    summary: dict[str, float | int | None] = {
        "auc_folds": len(valid_auc),
        "mean_fold_auc": float(np.mean(valid_auc)) if valid_auc else None,
        "mean_fold_f1_macro": float(np.mean([float(row["f1_macro"]) for row in records])),
        "mean_fold_accuracy": float(np.mean([float(row["accuracy"]) for row in records])),
        "pooled_oof_auc": float(roc_auc_score(y, oof_probability)),
        "pooled_oof_f1_macro": float(f1_score(y, oof_prediction, labels=[0, 1], average="macro", zero_division=0)),
        "pooled_oof_accuracy": float(accuracy_score(y, oof_prediction)),
    }
    return records, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("features/vrnet_training_matrix_v7_normalized.csv"))
    parser.add_argument("--config", type=Path, default=Path("configs/feature_set_v7.yaml"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--report", type=Path, default=Path("results/training_v7.md"))
    parser.add_argument("--metrics-output", type=Path, default=Path("results/training_v7_metrics.csv"))
    args = parser.parse_args()
    data = pd.read_csv(args.input, encoding="utf-8-sig")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    feature_sets = {name: list(values) for name, values in config["feature_sets"].items()}
    seed = int(config.get("random_seed", 42))
    if len(data) != 61 or data["game_name"].nunique() != 11 or data["fms_label_binary"].nunique() != 2:
        raise ValueError("expected 61 sessions, 11 games, and two labels")

    all_records: dict[str, list[dict[str, object]]] = {}
    summaries: list[dict[str, object]] = []
    args.model_dir.mkdir(parents=True, exist_ok=True)
    for name in ("core", "extended", "full"):
        features = feature_sets[name]
        records, summary = evaluate(data, features, seed)
        final_model = model(seed)
        final_model.fit(data[features].apply(pd.to_numeric, errors="coerce"), data["fms_label_binary"].astype(int))
        model_path = args.model_dir / f"vrnet_model_v7_{name}.pkl"
        joblib.dump({
            "pipeline": final_model, "model_type": "lightgbm", "feature_set": name,
            "feature_names": features, "random_seed": seed,
            "class_mapping": {0: "low", 1: "medium"}, "fold_records": records,
            "metrics": summary, "normalization": "within_game_zscore_transductive",
        }, model_path)
        all_records[name] = records
        summaries.append({"feature_set": name, "n_features": len(features), **summary, "model_path": str(model_path)})
        print(
            f"{name}: n={len(features)} auc_folds={summary['auc_folds']} "
            f"mean_auc={fmt(summary['mean_fold_auc'])} mean_f1={fmt(summary['mean_fold_f1_macro'])} "
            f"pooled_auc={fmt(summary['pooled_oof_auc'])}"
        )

    metrics = pd.DataFrame(summaries)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.metrics_output, index=False, encoding="utf-8-sig", float_format="%.12g")
    ranking = metrics.sort_values(
        ["mean_fold_auc", "mean_fold_f1_macro"], ascending=[False, False], na_position="last"
    )
    best = str(ranking.iloc[0]["feature_set"])
    lines = [
        "# VR.net v7 特征集对比训练", "",
        "**探索性结果，非最终结论。未训练深度模型。标签阈值保持 `max(MS_rating) <= 2` / `> 2`。**", "",
        "## 方法", "",
        "- 模型：LightGBM，三个特征集使用相同超参数与随机种子 42。",
        "- 验证：Leave-One-Game-Out，共 11 折。",
        "- AUC：仅测试集同时包含 low 和 medium 时计算。",
        "- 输入：清除 >720°/s 残留旋转率后，按游戏分别做 z-score。",
        "- 限制：按游戏标准化使用了每个留出游戏自身的无标签分布，属于 transductive 探索；不能代表对单个全新游戏 session 的部署性能。", "",
        "## 汇总", "",
        "| 特征集 | 特征数 | 可算 AUC 折 | 平均 fold AUC | 平均 fold F1 | pooled OOF AUC | pooled OOF F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics.iterrows():
        lines.append(
            f"| {row['feature_set']} | {int(row['n_features'])} | {int(row['auc_folds'])}/11 | "
            f"{fmt(float(row['mean_fold_auc']))} | {fmt(float(row['mean_fold_f1_macro']))} | "
            f"{fmt(float(row['pooled_oof_auc']))} | {fmt(float(row['pooled_oof_f1_macro']))} |"
        )
    for name in ("core", "extended", "full"):
        lines.extend([
            "", f"## {name} 每折结果", "",
            "| fold | 留出游戏 | n | low | medium | AUC | F1 macro | 混淆矩阵 | AUC 状态 |",
            "|---:|---|---:|---:|---:|---:|---:|---|---|",
        ])
        for row in all_records[name]:
            lines.append(
                f"| {row['fold']} | {row['held_out_game']} | {row['n_test']} | {row['test_low']} | "
                f"{row['test_medium']} | {fmt(row['auc'])} | {fmt(row['f1_macro'])} | "
                f"`{row['confusion']}` | {row['auc_reason']} |"
            )
    lines.extend(["", "## 数值最优", "", f"按平均 fold AUC（并以 F1 处理并列）排序，数值最优特征集为 **{best}**。该选择仍是探索性比较。"])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"best={best}; wrote {args.report}, {args.metrics_output}, and 3 model bundles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
