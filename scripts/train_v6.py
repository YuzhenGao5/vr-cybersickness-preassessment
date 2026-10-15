#!/usr/bin/env python3
"""Train exploratory VR.net v6 binary model with Leave-One-Game-Out validation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


NON_FEATURE_COLUMNS = {
    "session_id", "game_name", "ms_rating_non_null", "ms_rating_max",
    "fms_label", "fms_label_binary",
}


def build_model(n: int, seed: int) -> tuple[str, Pipeline]:
    if n > 30:
        from lightgbm import LGBMClassifier
        return "lightgbm", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("classifier", LGBMClassifier(
                n_estimators=100, learning_rate=0.05, num_leaves=7,
                min_child_samples=5, class_weight="balanced",
                random_state=seed, verbosity=-1,
            )),
        ])
    return "logistic_regression", Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            solver="liblinear", class_weight="balanced", random_state=seed, max_iter=2000,
        )),
    ])


def fmt(value: float | None) -> str:
    return "NA" if value is None or not math.isfinite(value) else f"{value:.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("features/vrnet_training_matrix_v6.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/vrnet_model_v6.pkl"))
    parser.add_argument("--report", type=Path, default=Path("results/training_v6.md"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    data = pd.read_csv(args.input, encoding="utf-8-sig")
    feature_names = [column for column in data.columns if column not in NON_FEATURE_COLUMNS]
    X = data[feature_names].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(data["fms_label_binary"], errors="raise").astype(int)
    groups = data["game_name"].astype(str)
    if len(data) != 61 or groups.nunique() != 11 or y.nunique() != 2:
        raise ValueError(
            f"expected 61 sessions, 11 games, 2 labels; got {len(data)}, {groups.nunique()}, {y.nunique()}"
        )

    records: list[dict[str, object]] = []
    oof_prob = np.full(len(data), np.nan)
    oof_pred = np.full(len(data), -1, dtype=int)
    for fold, (train_idx, test_idx) in enumerate(LeaveOneGroupOut().split(X, y, groups), start=1):
        held_out = str(groups.iloc[test_idx[0]])
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model_type, model = build_model(len(train_idx), args.seed)
        if y_train.nunique() != 2:
            records.append({
                "fold": fold, "held_out_game": held_out, "n_train": len(train_idx),
                "n_test": len(test_idx), "test_low": int((y_test == 0).sum()),
                "test_medium": int((y_test == 1).sum()),
                "accuracy": None, "f1_macro": None, "auc": None,
                "auc_reason": "训练集只有一个类别，未训练", "confusion": "NA",
                "status": "skipped_training_one_class",
            })
            continue
        model.fit(X.iloc[train_idx], y_train)
        pred = model.predict(X.iloc[test_idx])
        prob = model.predict_proba(X.iloc[test_idx])[:, 1]
        oof_pred[test_idx] = pred
        oof_prob[test_idx] = prob
        has_two_test_classes = y_test.nunique() == 2
        auc = float(roc_auc_score(y_test, prob)) if has_two_test_classes else None
        records.append({
            "fold": fold, "held_out_game": held_out, "n_train": len(train_idx),
            "n_test": len(test_idx), "test_low": int((y_test == 0).sum()),
            "test_medium": int((y_test == 1).sum()),
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1_macro": float(f1_score(y_test, pred, labels=[0, 1], average="macro", zero_division=0)),
            "auc": auc,
            "auc_reason": "可计算" if has_two_test_classes else "测试集只有一个类别",
            "confusion": str(confusion_matrix(y_test, pred, labels=[0, 1]).tolist()),
            "status": "success",
        })

    valid = oof_pred >= 0
    if not valid.all():
        raise RuntimeError("one or more LOGO observations have no OOF prediction")
    pooled_accuracy = float(accuracy_score(y, oof_pred))
    pooled_f1 = float(f1_score(y, oof_pred, labels=[0, 1], average="macro", zero_division=0))
    pooled_auc = float(roc_auc_score(y, oof_prob)) if y.nunique() == 2 else None
    successful = [record for record in records if record["status"] == "success"]
    auc_values = [float(record["auc"]) for record in successful if record["auc"] is not None]
    mean_auc = float(np.mean(auc_values)) if auc_values else None
    mean_f1 = float(np.mean([float(record["f1_macro"]) for record in successful]))
    mean_accuracy = float(np.mean([float(record["accuracy"]) for record in successful]))

    model_type, final_model = build_model(len(data), args.seed)
    final_model.fit(X, y)
    bundle = {
        "pipeline": final_model, "model_type": model_type, "feature_names": feature_names,
        "random_seed": args.seed, "class_mapping": {0: "low", 1: "medium"},
        "fold_records": records,
        "metrics": {
            "mean_fold_accuracy": mean_accuracy, "mean_fold_f1_macro": mean_f1,
            "mean_fold_auc": mean_auc, "auc_folds": len(auc_values),
            "pooled_oof_accuracy": pooled_accuracy, "pooled_oof_f1_macro": pooled_f1,
            "pooled_oof_auc": pooled_auc,
        },
    }
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model)

    fold_rows = []
    for record in records:
        fold_rows.append(
            f"| {record['fold']} | {record['held_out_game']} | {record['n_test']} | "
            f"{record['test_low']} | {record['test_medium']} | {fmt(record['accuracy'])} | "
            f"{fmt(record['f1_macro'])} | {fmt(record['auc'])} | {record['auc_reason']} | "
            f"`{record['confusion']}` |"
        )
    label_counts = y.value_counts().sort_index().to_dict()
    game_counts = data.groupby("game_name")["session_id"].nunique().sort_index()
    lines = [
        "# VR.net v6 Leave-One-Game-Out 训练报告", "",
        "**探索性结果，非最终结论。61 个 session 仍是小样本，指标和 SHAP 均不代表已验证预测能力。**", "",
        "## 数据与模型", "",
        f"- session：{len(data)}", f"- 游戏：{groups.nunique()}", f"- 特征：{len(feature_names)}",
        f"- 标签：low={int(label_counts.get(0, 0))}, medium={int(label_counts.get(1, 0))}",
        f"- 模型：{model_type}", "- 验证：Leave-One-Game-Out", f"- 随机种子：{args.seed}", "",
        "### 每个游戏的 session 数", "",
        *[f"- {game}: {int(count)}" for game, count in game_counts.items()], "",
        "## 每折结果", "",
        "混淆矩阵格式为 `[[TN, FP], [FN, TP]]`。测试集同时含 low 和 medium 时才计算 AUC。", "",
        "| fold | 留出游戏 | n_test | low | medium | Accuracy | F1 macro | AUC | AUC 状态/原因 | 混淆矩阵 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|---|", *fold_rows, "",
        "## 汇总", "",
        f"- 可计算 AUC 的折：{len(auc_values)}/{len(records)}",
        f"- 平均 fold AUC：{fmt(mean_auc)}（仅统计可计算折）",
        f"- 平均 fold F1 macro：{fmt(mean_f1)}",
        f"- 平均 fold Accuracy：{fmt(mean_accuracy)}",
        f"- pooled OOF AUC：{fmt(pooled_auc)}",
        f"- pooled OOF F1 macro：{fmt(pooled_f1)}",
        f"- pooled OOF Accuracy：{fmt(pooled_accuracy)}", "",
        "pooled OOF 指标合并了由不同训练折产生的预测，仅作探索性流程验证。",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"model={model_type} sessions={len(data)} games={groups.nunique()} "
        f"auc_folds={len(auc_values)} mean_auc={fmt(mean_auc)} mean_f1={mean_f1:.6f} "
        f"pooled_auc={fmt(pooled_auc)} pooled_f1={pooled_f1:.6f}"
    )
    print(f"Wrote {args.model} and {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
