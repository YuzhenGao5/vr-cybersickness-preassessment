#!/usr/bin/env python3
"""Train and evaluate an exploratory VR.net binary model with LOGO validation."""

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
    "session_id",
    "game_name",
    "ms_rating_non_null",
    "ms_rating_max",
    "fms_label",
    "fms_label_binary",
}


def build_model(sample_count: int, seed: int) -> tuple[str, Pipeline]:
    if sample_count > 30:
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError("LightGBM is required when sample_count > 30") from exc
        return (
            "lightgbm",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    (
                        "classifier",
                        LGBMClassifier(
                            n_estimators=100,
                            learning_rate=0.05,
                            num_leaves=7,
                            min_child_samples=5,
                            class_weight="balanced",
                            random_state=seed,
                            verbosity=-1,
                        ),
                    ),
                ]
            ),
        )
    return (
        "logistic_regression",
        Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        solver="liblinear",
                        class_weight="balanced",
                        random_state=seed,
                        max_iter=2000,
                    ),
                ),
            ]
        ),
    )


def fmt(value: float | None) -> str:
    return "NA" if value is None or not math.isfinite(value) else f"{value:.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("features/vrnet_training_matrix_v5.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/vrnet_model_v5.pkl"))
    parser.add_argument("--report", type=Path, default=Path("results/training_v5.md"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    data = pd.read_csv(args.input, encoding="utf-8-sig")
    feature_names = [column for column in data.columns if column not in NON_FEATURE_COLUMNS]
    X = data[feature_names].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(data["fms_label_binary"], errors="raise").astype(int)
    groups = data["game_name"].astype(str)
    if y.nunique() != 2:
        raise ValueError(f"Both labels are required, found {y.value_counts().to_dict()}")
    if groups.nunique() < 2:
        raise ValueError("Leave-One-Game-Out requires at least two games")

    logo = LeaveOneGroupOut()
    fold_records: list[dict[str, object]] = []
    oof_probability = np.full(len(data), np.nan)
    oof_prediction = np.full(len(data), -1, dtype=int)
    for fold_number, (train_index, test_index) in enumerate(logo.split(X, y, groups), start=1):
        held_out = str(groups.iloc[test_index[0]])
        model_type, model = build_model(len(train_index), args.seed)
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]
        if y_train.nunique() != 2:
            fold_records.append(
                {
                    "fold": fold_number,
                    "held_out_game": held_out,
                    "n_train": len(train_index),
                    "n_test": len(test_index),
                    "status": "skipped_training_has_one_class",
                    "accuracy": None,
                    "f1_macro": None,
                    "f1_binary": None,
                    "auc": None,
                    "confusion": "NA",
                }
            )
            continue
        model.fit(X.iloc[train_index], y_train)
        prediction = model.predict(X.iloc[test_index])
        probability = model.predict_proba(X.iloc[test_index])[:, 1]
        oof_prediction[test_index] = prediction
        oof_probability[test_index] = probability
        accuracy = float(accuracy_score(y_test, prediction))
        f1_macro = float(f1_score(y_test, prediction, labels=[0, 1], average="macro", zero_division=0))
        f1_binary = float(f1_score(y_test, prediction, pos_label=1, zero_division=0))
        auc = float(roc_auc_score(y_test, probability)) if y_test.nunique() == 2 else None
        matrix = confusion_matrix(y_test, prediction, labels=[0, 1]).tolist()
        fold_records.append(
            {
                "fold": fold_number,
                "held_out_game": held_out,
                "n_train": len(train_index),
                "n_test": len(test_index),
                "test_low": int((y_test == 0).sum()),
                "test_medium": int((y_test == 1).sum()),
                "status": "success",
                "accuracy": accuracy,
                "f1_macro": f1_macro,
                "f1_binary": f1_binary,
                "auc": auc,
                "confusion": str(matrix),
            }
        )

    successful = [record for record in fold_records if record["status"] == "success"]
    valid_predictions = oof_prediction >= 0
    pooled_accuracy = float(accuracy_score(y[valid_predictions], oof_prediction[valid_predictions]))
    pooled_f1_macro = float(
        f1_score(y[valid_predictions], oof_prediction[valid_predictions], labels=[0, 1], average="macro", zero_division=0)
    )
    pooled_f1_binary = float(
        f1_score(y[valid_predictions], oof_prediction[valid_predictions], pos_label=1, zero_division=0)
    )
    pooled_auc = (
        float(roc_auc_score(y[valid_predictions], oof_probability[valid_predictions]))
        if y[valid_predictions].nunique() == 2
        else None
    )
    fold_auc_values = [float(record["auc"]) for record in successful if record["auc"] is not None]
    mean_fold_auc = float(np.mean(fold_auc_values)) if fold_auc_values else None
    mean_fold_f1 = float(np.mean([float(record["f1_macro"]) for record in successful]))
    mean_fold_accuracy = float(np.mean([float(record["accuracy"]) for record in successful]))

    model_type, final_model = build_model(len(data), args.seed)
    final_model.fit(X, y)
    bundle = {
        "pipeline": final_model,
        "model_type": model_type,
        "feature_names": feature_names,
        "random_seed": args.seed,
        "class_mapping": {0: "low", 1: "medium"},
        "fold_records": fold_records,
        "metrics": {
            "mean_fold_accuracy": mean_fold_accuracy,
            "mean_fold_f1_macro": mean_fold_f1,
            "mean_fold_auc": mean_fold_auc,
            "auc_folds": len(fold_auc_values),
            "pooled_oof_accuracy": pooled_accuracy,
            "pooled_oof_f1_macro": pooled_f1_macro,
            "pooled_oof_f1_binary": pooled_f1_binary,
            "pooled_oof_auc": pooled_auc,
        },
    }
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model)

    rows = []
    for record in fold_records:
        rows.append(
            "| {fold} | {held_out_game} | {n_test} | {test_low} | {test_medium} | "
            "{accuracy} | {f1_macro} | {auc} | `{confusion}` | {status} |".format(
                fold=record["fold"],
                held_out_game=record["held_out_game"],
                n_test=record["n_test"],
                test_low=record.get("test_low", "NA"),
                test_medium=record.get("test_medium", "NA"),
                accuracy=fmt(record["accuracy"]),
                f1_macro=fmt(record["f1_macro"]),
                auc=fmt(record["auc"]),
                confusion=record["confusion"],
                status=record["status"],
            )
        )
    label_counts = y.value_counts().sort_index().to_dict()
    lines = [
        "# VR.net v5 按游戏分组训练",
        "",
        "**探索性结果，非最终结论。样本量小且十个游戏只有一个 session，不能用于论文效果结论。**",
        "",
        "## 数据与模型",
        "",
        f"- session：{len(data)}",
        f"- 游戏：{groups.nunique()}",
        f"- 特征：{len(feature_names)}",
        f"- 标签：low={int(label_counts.get(0, 0))}, medium={int(label_counts.get(1, 0))}",
        f"- 模型：{model_type}",
        "- 验证：Leave-One-Game-Out",
        "- 随机种子：42",
        "",
        "## 每折结果",
        "",
        "混淆矩阵格式为 `[[TN, FP], [FN, TP]]`。测试集只有一个类别时 AUC 记为 NA。",
        "",
        "| fold | 留出游戏 | n_test | low | medium | Accuracy | F1 macro | AUC | 混淆矩阵 | 状态 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|---|",
        *rows,
        "",
        "## 汇总",
        "",
        f"- 平均 fold Accuracy：{fmt(mean_fold_accuracy)}",
        f"- 平均 fold F1 macro：{fmt(mean_fold_f1)}",
        f"- 平均 fold AUC：{fmt(mean_fold_auc)}（仅 {len(fold_auc_values)} 个测试集含两个类别）",
        f"- pooled OOF Accuracy：{fmt(pooled_accuracy)}",
        f"- pooled OOF F1 macro：{fmt(pooled_f1_macro)}",
        f"- pooled OOF F1，medium 正类：{fmt(pooled_f1_binary)}",
        f"- pooled OOF AUC：{fmt(pooled_auc)}",
        "",
        "平均 fold AUC 可能只来自 Pirate_Life 这一折，因为其他游戏各只有一个测试 session。pooled OOF AUC 合并了由不同训练折产生的概率，也只能作探索性参考。",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"model={model_type} folds={len(fold_records)} mean_f1={mean_fold_f1:.6f} "
        f"mean_auc={fmt(mean_fold_auc)} pooled_auc={fmt(pooled_auc)}"
    )
    print(f"Wrote {args.model} and {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
