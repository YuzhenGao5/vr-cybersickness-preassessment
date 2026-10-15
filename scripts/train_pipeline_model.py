#!/usr/bin/env python3
"""Train one exploratory 80/20 logistic-regression split for pipeline validation."""

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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


NON_FEATURE_COLUMNS = {
    "session_id",
    "ms_rating_non_null",
    "ms_rating_max",
    "fms_label",
    "fms_label_binary",
}


def metric(value: float | None) -> str:
    return "不可计算" if value is None or not math.isfinite(value) else f"{value:.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("features/pipeline_training_matrix.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/pipeline_model.pkl"))
    parser.add_argument("--report", type=Path, default=Path("results/pipeline_training.md"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Training matrix does not exist: {args.input}")
    data = pd.read_csv(args.input, encoding="utf-8-sig")
    if len(data) != 11:
        raise ValueError(f"Expected 11 sessions, found {len(data)}")

    feature_names = [column for column in data.columns if column not in NON_FEATURE_COLUMNS]
    if not feature_names:
        raise ValueError("No feature columns found")
    X = data[feature_names].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(data["fms_label_binary"], errors="raise").astype(int)
    class_counts = y.value_counts().sort_index()
    if set(class_counts.index) != {0, 1}:
        raise ValueError(f"Binary training requires both labels, found {class_counts.to_dict()}")

    test_count = int(math.ceil(0.2 * len(data)))
    can_stratify = int(class_counts.min()) >= 2 and test_count >= 2
    indices = np.arange(len(data))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=args.seed,
        stratify=y if can_stratify else None,
    )
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    if y_train.nunique() != 2:
        raise ValueError("The frozen split left only one class in training; model cannot be fitted")

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=args.seed,
                    max_iter=2000,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    prediction = pipeline.predict(X_test)
    probability = pipeline.predict_proba(X_test)[:, 1]
    accuracy = float(accuracy_score(y_test, prediction))
    f1_binary = float(f1_score(y_test, prediction, pos_label=1, zero_division=0))
    f1_macro = float(f1_score(y_test, prediction, average="macro", zero_division=0))
    auc = float(roc_auc_score(y_test, probability)) if y_test.nunique() == 2 else None
    matrix = confusion_matrix(y_test, prediction, labels=[0, 1])

    bundle = {
        "pipeline": pipeline,
        "feature_names": feature_names,
        "random_seed": args.seed,
        "train_indices": train_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "train_session_ids": data.iloc[train_idx]["session_id"].tolist(),
        "test_session_ids": data.iloc[test_idx]["session_id"].tolist(),
        "class_mapping": {0: "low", 1: "medium"},
        "metrics": {
            "accuracy": accuracy,
            "f1_binary": f1_binary,
            "f1_macro": f1_macro,
            "auc": auc,
        },
        "test_predictions": prediction.tolist(),
        "test_probabilities_medium": probability.tolist(),
    }
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model)

    test_rows = []
    for position, row_index in enumerate(test_idx):
        test_rows.append(
            "| {} | {} | {} | {:.6f} |".format(
                data.iloc[row_index]["session_id"],
                int(y.iloc[row_index]),
                int(prediction[position]),
                float(probability[position]),
            )
        )
    lines = [
        "# Pirate Life 流程验证：二分类训练",
        "",
        "**警告：仅 11 个 session。本结果仅验证代码流程，极不稳定，不是最终结论，不能用于论文结论。**",
        "",
        "## 设置",
        "",
        "- 模型：逻辑回归（样本量过小，因此未使用 LightGBM）",
        "- 预处理：中位数缺失值填补 + 标准化",
        "- 划分：一次 80/20 train/test split，无交叉验证",
        f"- 随机种子：{args.seed}",
        f"- 分层划分：{'是' if can_stratify else '否'}",
        f"- 训练 session：{len(train_idx)}",
        f"- 测试 session：{len(test_idx)}",
        f"- 特征数：{len(feature_names)}",
        f"- 标签分布：low={int(class_counts.get(0, 0))}, medium={int(class_counts.get(1, 0))}",
        "",
        "## 指标",
        "",
        f"- Accuracy：{metric(accuracy)}",
        f"- F1（二分类，medium 为正类）：{metric(f1_binary)}",
        f"- F1（macro）：{metric(f1_macro)}",
        f"- AUC：{metric(auc)}",
        "",
        "AUC 只有在三个测试样本同时包含 low 和 medium 时才计算。即使可计算，也会因单个样本改变而剧烈波动。",
        "",
        "## 混淆矩阵",
        "",
        "行是真实标签，列是预测标签。",
        "",
        "|  | 预测 low | 预测 medium |",
        "|---|---:|---:|",
        f"| 真实 low | {int(matrix[0, 0])} | {int(matrix[0, 1])} |",
        f"| 真实 medium | {int(matrix[1, 0])} | {int(matrix[1, 1])} |",
        "",
        "## 测试样本",
        "",
        "| session | 真实标签 | 预测标签 | P(medium) |",
        "|---|---:|---:|---:|",
        *test_rows,
        "",
        "## 结论限制",
        "",
        "此次运行仅证明特征矩阵能够进入传统分类器并输出评估指标。一个游戏的 11 个 session、一次划分和三个测试样本不足以估计泛化能力。",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Accuracy={accuracy:.6f} F1={f1_binary:.6f} AUC={metric(auc)}")
    print(f"Wrote model to {args.model} and report to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
