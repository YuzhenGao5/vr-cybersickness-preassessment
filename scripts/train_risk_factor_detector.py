#!/usr/bin/env python3
"""Train exploratory classifiers that reproduce six deterministic risk-factor labels.

This script intentionally uses an explicit 18-feature allowlist from the VR.net v6
schema.  Outcome, FMS, MS_rating, and other label columns are never selected as
inputs.  The resulting models reproduce rule-generated labels; they do not predict
cybersickness.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline


# Frozen from features/vrnet_training_matrix_v6.csv: the 18 columns between
# game_name and ms_rating_non_null.  Do not replace this with automatic selection.
EXPECTED_FEATURES = [
    "fov_horizontal_mean",
    "fov_horizontal_max",
    "fov_horizontal_std",
    "fov_vertical_mean",
    "fov_vertical_max",
    "fov_vertical_std",
    "camera_speed_mean",
    "camera_speed_max",
    "camera_speed_std",
    "camera_rotation_rate_mean",
    "camera_rotation_rate_max",
    "camera_rotation_rate_std",
    "camera_acceleration_mean",
    "camera_acceleration_max",
    "object_count_mean",
    "object_count_max",
    "object_speed_mean",
    "object_speed_max",
]

LABEL_DIRECT_FEATURE = {
    "fast_camera_speed": "camera_speed_max",
    "high_camera_acceleration": "camera_acceleration_max",
    "multi_axis_rotation": None,
    "wide_fov": "fov_horizontal_mean",
    "high_object_density": "object_count_mean",
    "high_object_speed": "object_speed_mean",
}

FORBIDDEN_INPUT_TOKENS = ("ms_rating", "fms", "label")


def build_model(seed: int) -> Pipeline:
    """Return the fixed LightGBM pipeline used in every trainable fold/label."""
    return Pipeline(
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
    )


def normalize_label(series: pd.Series) -> pd.Series:
    """Normalize yes/no rule labels to nullable 0/1; preserve unknown as missing."""
    true_values = {"1", "true", "yes", "y", "positive", "triggered", "high"}
    false_values = {"0", "false", "no", "n", "negative", "not_triggered", "low"}
    unknown_values = {"", "na", "n/a", "nan", "none", "null", "unknown", "tbd", "?"}

    def convert(value: object) -> object:
        if pd.isna(value):
            return pd.NA
        if isinstance(value, (bool, np.bool_)):
            return int(value)
        if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
            return int(value)
        if isinstance(value, (float, np.floating)) and float(value) in (0.0, 1.0):
            return int(value)
        text = str(value).strip().lower()
        if text in true_values:
            return 1
        if text in false_values:
            return 0
        if text in unknown_values:
            return pd.NA
        raise ValueError(f"unrecognized rule label value: {value!r}")

    return series.map(convert).astype("Int64")


def safe_auc(y_true: pd.Series, probability: np.ndarray) -> float | None:
    if y_true.nunique(dropna=True) != 2:
        return None
    return float(roc_auc_score(y_true.astype(int), probability))


def finite_mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def evaluate_logo(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate one binary target with LOGO, preserving skipped-fold reasons."""
    records: list[dict[str, Any]] = []
    oof_probability = np.full(len(y), np.nan, dtype=float)
    oof_prediction = np.full(len(y), np.nan, dtype=float)

    if groups.nunique() < 2:
        return records, {
            "evaluation_status": "unavailable_fewer_than_two_games",
            "successful_folds": 0,
            "total_folds": int(groups.nunique()),
            "auc_folds": 0,
            "mean_fold_accuracy": None,
            "mean_fold_f1_macro": None,
            "mean_fold_auc": None,
            "pooled_oof_n": 0,
            "pooled_oof_coverage": 0.0,
            "pooled_oof_accuracy": None,
            "pooled_oof_f1_macro": None,
            "pooled_oof_auc": None,
        }

    logo = LeaveOneGroupOut()
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
        held_out_game = str(groups.iloc[test_idx[0]])
        y_train = y.iloc[train_idx].astype(int)
        y_test = y.iloc[test_idx].astype(int)
        base = {
            "fold": fold,
            "held_out_game": held_out_game,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "train_negative": int((y_train == 0).sum()),
            "train_positive": int((y_train == 1).sum()),
            "test_negative": int((y_test == 0).sum()),
            "test_positive": int((y_test == 1).sum()),
        }
        if y_train.nunique() < 2:
            records.append(
                {
                    **base,
                    "status": "skipped_training_one_class",
                    "accuracy": None,
                    "f1_macro": None,
                    "auc": None,
                    "auc_reason": "训练集只有一个类别；按协议跳过训练和预测",
                    "confusion_matrix": None,
                }
            )
            continue

        model = build_model(seed)
        model.fit(X.iloc[train_idx], y_train)
        prediction = model.predict(X.iloc[test_idx]).astype(int)
        probability = model.predict_proba(X.iloc[test_idx])[:, 1]
        oof_prediction[test_idx] = prediction
        oof_probability[test_idx] = probability
        auc = safe_auc(y_test, probability)
        records.append(
            {
                **base,
                "status": "success",
                "accuracy": float(accuracy_score(y_test, prediction)),
                "f1_macro": float(
                    f1_score(y_test, prediction, labels=[0, 1], average="macro", zero_division=0)
                ),
                "auc": auc,
                "auc_reason": "可计算" if auc is not None else "测试集只有一个类别，AUC 不可计算",
                "confusion_matrix": confusion_matrix(y_test, prediction, labels=[0, 1]).tolist(),
            }
        )

    successful = [record for record in records if record["status"] == "success"]
    covered = np.isfinite(oof_prediction)
    covered_y = y.iloc[np.flatnonzero(covered)].astype(int)
    covered_prediction = oof_prediction[covered].astype(int)
    covered_probability = oof_probability[covered]

    pooled_accuracy: float | None = None
    pooled_f1: float | None = None
    pooled_auc: float | None = None
    if covered.any():
        pooled_accuracy = float(accuracy_score(covered_y, covered_prediction))
        pooled_f1 = float(
            f1_score(covered_y, covered_prediction, labels=[0, 1], average="macro", zero_division=0)
        )
        pooled_auc = safe_auc(covered_y, covered_probability)

    auc_values = [record["auc"] for record in successful]
    metrics = {
        "evaluation_status": "complete" if covered.all() else "partial_skipped_folds",
        "successful_folds": len(successful),
        "total_folds": len(records),
        "auc_folds": sum(value is not None for value in auc_values),
        "mean_fold_accuracy": finite_mean([record["accuracy"] for record in successful]),
        "mean_fold_f1_macro": finite_mean([record["f1_macro"] for record in successful]),
        "mean_fold_auc": finite_mean(auc_values),
        "pooled_oof_n": int(covered.sum()),
        "pooled_oof_coverage": float(covered.mean()) if len(covered) else 0.0,
        "pooled_oof_accuracy": pooled_accuracy,
        "pooled_oof_f1_macro": pooled_f1,
        "pooled_oof_auc": pooled_auc,
    }
    return records, metrics


def fmt(value: object) -> str:
    if value is None:
        return "NA"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:.6f}" if math.isfinite(numeric) else "NA"


def fold_table(records: list[dict[str, Any]]) -> list[str]:
    rows = [
        "| fold | 留出游戏 | n_train | n_test | test 0/1 | 状态 | Accuracy | F1 macro | AUC | AUC 状态 | 混淆矩阵 |",
        "|---:|---|---:|---:|---:|---|---:|---:|---:|---|---|",
    ]
    for record in records:
        confusion = "NA" if record["confusion_matrix"] is None else f"`{record['confusion_matrix']}`"
        rows.append(
            f"| {record['fold']} | {record['held_out_game']} | {record['n_train']} | "
            f"{record['n_test']} | {record['test_negative']}/{record['test_positive']} | "
            f"{record['status']} | {fmt(record['accuracy'])} | {fmt(record['f1_macro'])} | "
            f"{fmt(record['auc'])} | {record['auc_reason']} | {confusion} |"
        )
    return rows


def metric_summary_rows(metrics: dict[str, Any]) -> list[str]:
    return [
        f"- LOGO 状态：`{metrics['evaluation_status']}`",
        f"- 成功预测折：{metrics['successful_folds']}/{metrics['total_folds']}",
        f"- 可计算 AUC 折：{metrics['auc_folds']}/{metrics['total_folds']}",
        f"- 平均 fold Accuracy：{fmt(metrics['mean_fold_accuracy'])}",
        f"- 平均 fold F1 macro：{fmt(metrics['mean_fold_f1_macro'])}",
        f"- 平均 fold AUC：{fmt(metrics['mean_fold_auc'])}（仅统计可计算折）",
        f"- pooled OOF 覆盖：{metrics['pooled_oof_n']} 条，{metrics['pooled_oof_coverage']:.2%}",
        f"- pooled OOF Accuracy：{fmt(metrics['pooled_oof_accuracy'])}",
        f"- pooled OOF F1 macro：{fmt(metrics['pooled_oof_f1_macro'])}",
        f"- pooled OOF AUC：{fmt(metrics['pooled_oof_auc'])}",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("features/risk_factor_training_matrix.csv"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/risk_factor_detector.pkl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/risk_factor_detector_training.md"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"input matrix does not exist: {args.input}")
    data = pd.read_csv(args.input, encoding="utf-8-sig")
    required_identifiers = ["session_id", "game_name"]
    missing_identifiers = [column for column in required_identifiers if column not in data.columns]
    missing_features = [column for column in EXPECTED_FEATURES if column not in data.columns]
    if missing_identifiers or missing_features:
        raise ValueError(
            f"schema mismatch; missing identifiers={missing_identifiers}, missing features={missing_features}"
        )
    forbidden_selected = [
        column
        for column in EXPECTED_FEATURES
        if any(token in column.lower() for token in FORBIDDEN_INPUT_TOKENS)
    ]
    if forbidden_selected:
        raise RuntimeError(f"forbidden leakage columns selected as features: {forbidden_selected}")

    X_all = data[EXPECTED_FEATURES].apply(pd.to_numeric, errors="coerce")
    groups_all = data["game_name"].astype(str)
    ignored_forbidden_columns = [
        column
        for column in data.columns
        if column not in LABEL_DIRECT_FEATURE
        and column not in EXPECTED_FEATURES
        and any(token in column.lower() for token in FORBIDDEN_INPUT_TOKENS)
    ]

    bundle: dict[str, Any] = {
        "schema_version": "risk_factor_detector_v1",
        "task": "deterministic_risk_factor_rule_reproduction",
        "warning": "Exploratory, not a final conclusion; rule reproduction is not cybersickness prediction.",
        "input_path": str(args.input),
        "random_seed": args.seed,
        "feature_names": EXPECTED_FEATURES.copy(),
        "label_columns": list(LABEL_DIRECT_FEATURE),
        "direct_feature_map": LABEL_DIRECT_FEATURE.copy(),
        "labels": {},
        "ignored_forbidden_columns": ignored_forbidden_columns,
    }

    report = [
        "# VR.net 风险因素规则复现训练报告",
        "",
        "**探索性，非最终结论；规则复现不等于晕动症预测。**",
        "",
        "这些标签由输入特征阈值直接生成。包含直接规则特征时的高指标主要反映模型能否复现阈值，不能作为场景特征能够预测真实晕动症的证据。",
        "",
        "## 数据和泄漏防护",
        "",
        f"- 输入：`{args.input}`",
        f"- 行数：{len(data)}",
        f"- 游戏数：{groups_all.nunique()}",
        f"- 固定输入特征：{len(EXPECTED_FEATURES)}",
        f"- 随机种子：{args.seed}",
        "- 模型：LightGBM；缺失值仅在各训练折内用中位数填补。",
        "- 验证：Leave-One-Game-Out。",
        "- F1 使用固定标签 `[0,1]` 的 macro F1；测试折只有一个类别时仍如实计算 F1，但 AUC 记为 NA。",
        "- 训练折只有一个类别时不训练、不预测，并在 pooled OOF 覆盖率中保留缺口。",
        "- 18 个特征由 v6 schema 显式白名单定义；没有自动选择列。",
        f"- 被忽略的 MS_rating/FMS/其他 label 列：{ignored_forbidden_columns or '无'}",
        "",
        "### 固定 18 项输入特征",
        "",
        *[f"- `{feature}`" for feature in EXPECTED_FEATURES],
        "",
    ]

    per_label_summary_rows: list[str] = []
    for label, direct_feature in LABEL_DIRECT_FEATURE.items():
        report.extend([f"## 标签：`{label}`", ""])
        if label not in data.columns:
            entry = {
                "status": "untrainable_missing_label_column",
                "model": None,
                "feature_names": EXPECTED_FEATURES.copy(),
                "direct_feature": direct_feature,
                "n_total": len(data),
                "n_known": 0,
                "n_unknown": len(data),
                "n_negative": 0,
                "n_positive": 0,
                "fold_metrics": [],
                "metrics": None,
                "leakage_diagnostic": None,
            }
            bundle["labels"][label] = entry
            report.extend(["- 状态：不可训练，输入缺少该标签列。", ""])
            per_label_summary_rows.append(
                f"| {label} | {direct_feature or '无'} | 0 | {len(data)} | 0 | 0 | 不可训练：缺列 | NA | NA | NA | NA |"
            )
            continue

        normalized = normalize_label(data[label])
        known_mask = normalized.notna()
        y = normalized.loc[known_mask].astype(int).reset_index(drop=True)
        X = X_all.loc[known_mask].reset_index(drop=True)
        groups = groups_all.loc[known_mask].reset_index(drop=True)
        n_known = int(known_mask.sum())
        n_unknown = int((~known_mask).sum())
        n_negative = int((y == 0).sum())
        n_positive = int((y == 1).sum())

        entry = {
            "status": None,
            "model": None,
            "feature_names": EXPECTED_FEATURES.copy(),
            "direct_feature": direct_feature,
            "n_total": len(data),
            "n_known": n_known,
            "n_unknown": n_unknown,
            "n_negative": n_negative,
            "n_positive": n_positive,
            "fold_metrics": [],
            "metrics": None,
            "leakage_diagnostic": None,
        }
        report.extend(
            [
                f"- direct rule feature：`{direct_feature}`" if direct_feature else "- direct rule feature：无",
                f"- 已知：{n_known}；unknown：{n_unknown}",
                f"- 0：{n_negative}；1：{n_positive}",
                "",
            ]
        )

        if n_known == 0:
            entry["status"] = "untrainable_all_unknown"
            bundle["labels"][label] = entry
            report.extend(["状态：**不可训练；全部标签均为 unknown。**", ""])
            per_label_summary_rows.append(
                f"| {label} | {direct_feature or '无'} | 0 | {n_unknown} | 0 | 0 | 全部 unknown | NA | NA | NA | NA |"
            )
            continue
        if y.nunique() < 2:
            entry["status"] = "untrainable_global_single_class"
            bundle["labels"][label] = entry
            report.extend(["状态：**不可训练；全部已知标签只有一个类别。**", ""])
            per_label_summary_rows.append(
                f"| {label} | {direct_feature or '无'} | {n_known} | {n_unknown} | {n_negative} | {n_positive} | 全局单一类别 | NA | NA | NA | NA |"
            )
            continue

        fold_records, metrics = evaluate_logo(X, y, groups, args.seed)
        final_model = build_model(args.seed)
        final_model.fit(X, y)
        entry.update(
            {
                "status": "trained",
                "model": final_model,
                "fold_metrics": fold_records,
                "metrics": metrics,
            }
        )
        report.extend(["### 包含全部 18 项特征", "", *fold_table(fold_records), "", *metric_summary_rows(metrics), ""])

        if direct_feature is None:
            leakage_diagnostic = {
                "status": "not_applicable_no_single_direct_feature",
                "excluded_feature": None,
                "feature_names": EXPECTED_FEATURES.copy(),
                "fold_metrics": [],
                "metrics": None,
            }
            report.extend(
                [
                    "### Direct-rule-feature excluded 诊断",
                    "",
                    "该标签没有单一直接特征，诊断不适用。",
                    "",
                ]
            )
        else:
            diagnostic_features = [feature for feature in EXPECTED_FEATURES if feature != direct_feature]
            diagnostic_records, diagnostic_metrics = evaluate_logo(
                X[diagnostic_features], y, groups, args.seed
            )
            leakage_diagnostic = {
                "status": "evaluated",
                "excluded_feature": direct_feature,
                "feature_names": diagnostic_features,
                "fold_metrics": diagnostic_records,
                "metrics": diagnostic_metrics,
            }
            report.extend(
                [
                    "### Direct-rule-feature excluded 诊断",
                    "",
                    f"移除直接生成规则标签的 `{direct_feature}`。该诊断只能显示其余相关特征能否代理该阈值，仍不等于预测晕动症。",
                    "",
                    *fold_table(diagnostic_records),
                    "",
                    *metric_summary_rows(diagnostic_metrics),
                    "",
                    "#### 与完整输入对比",
                    "",
                    "| 指标 | 全部特征 | 移除直接特征 |",
                    "|---|---:|---:|",
                    f"| 平均 fold AUC | {fmt(metrics['mean_fold_auc'])} | {fmt(diagnostic_metrics['mean_fold_auc'])} |",
                    f"| 平均 fold F1 macro | {fmt(metrics['mean_fold_f1_macro'])} | {fmt(diagnostic_metrics['mean_fold_f1_macro'])} |",
                    f"| pooled OOF AUC | {fmt(metrics['pooled_oof_auc'])} | {fmt(diagnostic_metrics['pooled_oof_auc'])} |",
                    f"| pooled OOF F1 macro | {fmt(metrics['pooled_oof_f1_macro'])} | {fmt(diagnostic_metrics['pooled_oof_f1_macro'])} |",
                    "",
                ]
            )
        entry["leakage_diagnostic"] = leakage_diagnostic
        bundle["labels"][label] = entry
        per_label_summary_rows.append(
            f"| {label} | {direct_feature or '无'} | {n_known} | {n_unknown} | {n_negative} | {n_positive} | trained | "
            f"{fmt(metrics['mean_fold_auc'])} | {fmt(metrics['mean_fold_f1_macro'])} | "
            f"{fmt(metrics['pooled_oof_auc'])} | {fmt(metrics['pooled_oof_f1_macro'])} |"
        )

    trained_entries = [
        entry for entry in bundle["labels"].values() if entry["status"] == "trained"
    ]
    macro_metrics = {
        "trained_labels": len(trained_entries),
        "total_labels": len(LABEL_DIRECT_FEATURE),
        "macro_mean_fold_auc": finite_mean(
            [entry["metrics"]["mean_fold_auc"] for entry in trained_entries]
        ),
        "macro_mean_fold_f1": finite_mean(
            [entry["metrics"]["mean_fold_f1_macro"] for entry in trained_entries]
        ),
        "macro_pooled_oof_auc": finite_mean(
            [entry["metrics"]["pooled_oof_auc"] for entry in trained_entries]
        ),
        "macro_pooled_oof_f1": finite_mean(
            [entry["metrics"]["pooled_oof_f1_macro"] for entry in trained_entries]
        ),
    }
    bundle["macro_metrics"] = macro_metrics

    report.extend(
        [
            "## 标签级汇总",
            "",
            "宏平均仅使用有可计算值的已训练标签；不可训练标签不以零值代替。",
            "",
            "| 标签 | direct feature | known | unknown | 0 | 1 | 状态 | mean fold AUC | mean fold F1 | pooled AUC | pooled F1 |",
            "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|",
            *per_label_summary_rows,
            "",
            f"- 已训练标签：{macro_metrics['trained_labels']}/{macro_metrics['total_labels']}",
            f"- 标签宏平均 mean-fold AUC：{fmt(macro_metrics['macro_mean_fold_auc'])}",
            f"- 标签宏平均 mean-fold F1：{fmt(macro_metrics['macro_mean_fold_f1'])}",
            f"- 标签宏平均 pooled OOF AUC：{fmt(macro_metrics['macro_pooled_oof_auc'])}",
            f"- 标签宏平均 pooled OOF F1：{fmt(macro_metrics['macro_pooled_oof_f1'])}",
            "",
            "## 解释限制",
            "",
            "1. 这些目标是确定性规则标签，完整模型包含产生标签的直接特征；接近完美的结果属于阈值复现，不能当作真实风险预测性能。",
            "2. 移除直接特征后仍可能通过强相关代理特征复现规则，因此 excluded 诊断也不是独立风险验证。",
            "3. `multi_axis_rotation` 若全部 unknown，会保留不可训练状态，不创建虚假模型或指标。",
            "4. 结果不证明任一因素导致晕动症，也不能据此验证 low/medium/high 风险等级。",
            "",
            "**探索性，非最终结论；规则复现不等于晕动症预测。**",
        ]
    )

    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model)
    args.report.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(
        f"rows={len(data)} games={groups_all.nunique()} "
        f"trained_labels={macro_metrics['trained_labels']}/{macro_metrics['total_labels']} "
        f"macro_mean_fold_auc={fmt(macro_metrics['macro_mean_fold_auc'])} "
        f"macro_mean_fold_f1={fmt(macro_metrics['macro_mean_fold_f1'])}"
    )
    for label, entry in bundle["labels"].items():
        metrics = entry.get("metrics")
        if metrics is None:
            print(
                f"{label}: status={entry['status']} known={entry['n_known']} "
                f"unknown={entry['n_unknown']} 0={entry['n_negative']} 1={entry['n_positive']}"
            )
        else:
            print(
                f"{label}: status={entry['status']} known={entry['n_known']} "
                f"unknown={entry['n_unknown']} 0={entry['n_negative']} 1={entry['n_positive']} "
                f"mean_auc={fmt(metrics['mean_fold_auc'])} "
                f"mean_f1={fmt(metrics['mean_fold_f1_macro'])} "
                f"pooled_auc={fmt(metrics['pooled_oof_auc'])} "
                f"pooled_f1={fmt(metrics['pooled_oof_f1_macro'])}"
            )
    print(f"Wrote {args.model} and {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
