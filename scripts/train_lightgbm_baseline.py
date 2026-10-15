#!/usr/bin/env python3
"""Train video-only one-vs-rest LightGBM clip-label baselines.

Outer validation is leave-one-game-out. Model iteration count and probability
threshold are selected only inside each outer training set with a grouped
session split. Rule-generated labels are proxies, not symptom outcomes.
"""

from __future__ import annotations

import argparse
import math
import re
import time
from pathlib import Path
from typing import Any, Iterable

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline


SEED = 42
WARNING = (
    "Exploratory, not a final conclusion. Targets are rule-generated proxy labels; "
    "rule-factor detection is not validation of cybersickness risk prediction."
)
ID_COLUMNS = ["clip_id", "session_id", "game_name"]
LABEL_COLUMNS = [
    "fast_camera_rotation",
    "high_camera_acceleration",
    "multi_axis_rotation",
    "wide_fov",
    "high_object_density",
    "high_object_speed",
    "high_optical_flow",
]
STATIC_BASES = [
    "edge_density",
    "color_entropy_bits",
    "laplacian_variance",
    "brightness_mean_0_255",
    "brightness_std_0_255",
]
STATIC_FEATURES = [f"{base}_{stat}" for base in STATIC_BASES for stat in ("mean", "max", "std")]
FLOW_FEATURES = [
    "flow_magnitude_mean_px_per_sample",
    "flow_magnitude_max_px_per_sample",
    "flow_magnitude_std_px_per_sample",
    "flow_magnitude_entropy_bits",
    "flow_direction_entropy_bits",
    "flow_direction_circular_mean_deg",
    "flow_direction_circular_concentration",
]
HOMOGRAPHY_FEATURES = [
    f"homography_translation_proxy_px_s_{stat}" for stat in ("mean", "max", "std")
] + [f"homography_rotation_proxy_deg_s_{stat}" for stat in ("mean", "max", "std")]
# Explicit whitelist: no symptom outcome and no direct VR.net matrix threshold variable.
FEATURE_COLUMNS = [*STATIC_FEATURES, *FLOW_FEATURES, *HOMOGRAPHY_FEATURES]
FORBIDDEN = re.compile(
    r"(^|_)(ms_rating|fms|ssq|camera_speed|camera_acceleration|object_count|object_speed)($|_)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("features/lightgbm_training_matrix.csv")
    )
    parser.add_argument(
        "--model", type=Path, default=Path("models/lightgbm_baseline.pkl")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("results/lightgbm_baseline_training.md")
    )
    parser.add_argument(
        "--log", type=Path, default=Path("results/lightgbm_baseline_training_log.txt")
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, expected: Iterable[str], source: Path) -> None:
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def normalized_target(series: pd.Series, label: str) -> pd.Series:
    result = pd.Series(np.nan, index=series.index, dtype=float)
    unknown_tokens = {"", "unknown", "na", "n/a", "nan", "none", "null", "tbd"}
    for index, value in series.items():
        if pd.isna(value) or str(value).strip().lower() in unknown_tokens:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} contains invalid value {value!r} at row {index}") from exc
        if numeric not in (0.0, 1.0):
            raise ValueError(f"{label} contains non-binary value {value!r} at row {index}")
        result.at[index] = numeric
    return result


def classifier(n_estimators: int, seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=max(1, int(n_estimators)),
        learning_rate=0.05,
        num_leaves=7,
        min_child_samples=5,
        subsample=1.0,
        colsample_bytree=1.0,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )


def fit_pipeline(X: pd.DataFrame, y: np.ndarray, n_estimators: int, seed: int) -> Pipeline:
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    transformed = imputer.fit_transform(X)
    model = classifier(n_estimators, seed)
    model.fit(transformed, y)
    return Pipeline([("imputer", imputer), ("classifier", model)])


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    candidates = np.round(np.arange(0.05, 0.951, 0.05), 2)
    scored: list[tuple[float, float, float]] = []
    for threshold in candidates:
        score = float(
            f1_score(y_true, probabilities >= threshold, average="macro", labels=[0, 1], zero_division=0)
        )
        # Prefer a threshold closest to 0.5, then the lower threshold, on ties.
        scored.append((score, -abs(float(threshold) - 0.5), -float(threshold)))
    best = max(scored)
    return -best[2], best[0]


def inner_settings(
    X: pd.DataFrame,
    y: np.ndarray,
    session_groups: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    fallback = {
        "status": "fallback_no_valid_grouped_inner_split",
        "n_estimators": 100,
        "threshold": 0.5,
        "inner_f1_macro": None,
        "inner_train_sessions": None,
        "inner_validation_sessions": None,
    }
    unique_sessions = np.unique(session_groups)
    if len(unique_sessions) < 2 or len(np.unique(y)) < 2:
        return fallback
    splitter = GroupShuffleSplit(n_splits=32, test_size=0.2, random_state=seed)
    for train_index, validation_index in splitter.split(X, y, groups=session_groups):
        y_train = y[train_index]
        y_validation = y[validation_index]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_validation)) < 2:
            continue
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        X_train = imputer.fit_transform(X.iloc[train_index])
        X_validation = imputer.transform(X.iloc[validation_index])
        model = classifier(500, seed)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_validation, y_validation)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)],
        )
        best_iteration = int(model.best_iteration_ or model.n_estimators)
        probabilities = model.predict_proba(X_validation, num_iteration=best_iteration)[:, 1]
        threshold, inner_f1 = choose_threshold(y_validation, probabilities)
        return {
            "status": "grouped_session_inner_split",
            "n_estimators": best_iteration,
            "threshold": threshold,
            "inner_f1_macro": inner_f1,
            "inner_train_sessions": int(len(np.unique(session_groups[train_index]))),
            "inner_validation_sessions": int(len(np.unique(session_groups[validation_index]))),
        }
    return fallback


def metrics_or_none(
    y_true: np.ndarray, probabilities: np.ndarray, predictions: np.ndarray
) -> dict[str, float | None]:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {
            "roc_auc": None,
            "average_precision": None,
            "f1_macro": None,
            "precision": None,
            "recall": None,
        }
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "f1_macro": float(
            f1_score(y_true, predictions, average="macro", labels=[0, 1], zero_division=0)
        ),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
    }


def mean_defined(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def evaluate_label(
    data: pd.DataFrame,
    X: pd.DataFrame,
    target: pd.Series,
    label: str,
    valid_feature_rows: pd.Series,
    seed: int,
) -> dict[str, Any]:
    known = target.notna() & valid_feature_rows
    subset = data.loc[known].copy()
    X_known = X.loc[known].copy()
    y = target.loc[known].astype(int).to_numpy()
    n_unknown = int(target.isna().sum())
    n_feature_invalid = int((target.notna() & ~valid_feature_rows).sum())
    base: dict[str, Any] = {
        "status": None,
        "model": None,
        "threshold": None,
        "feature_names": list(FEATURE_COLUMNS),
        "n_total": int(len(data)),
        "n_known_usable": int(len(subset)),
        "n_unknown": n_unknown,
        "n_feature_invalid": n_feature_invalid,
        "n_negative": int((y == 0).sum()),
        "n_positive": int((y == 1).sum()),
        "n_games": int(subset["game_name"].nunique()) if len(subset) else 0,
        "n_sessions": int(subset["session_id"].nunique()) if len(subset) else 0,
        "fold_metrics": [],
        "metrics": None,
        "final_tuning": None,
    }
    if len(subset) == 0:
        base["status"] = "untrainable_no_known_usable_rows"
        return base
    if len(np.unique(y)) < 2:
        base["status"] = "untrainable_global_single_class"
        return base

    fold_rows: list[dict[str, Any]] = []
    oof_y: list[int] = []
    oof_probability: list[float] = []
    oof_prediction: list[int] = []
    baseline_probability: list[float] = []
    baseline_prediction: list[int] = []
    games = sorted(subset["game_name"].astype(str).unique())
    if len(games) > 1:
        for fold_number, held_out_game in enumerate(games, 1):
            test_mask = subset["game_name"].astype(str).eq(held_out_game).to_numpy()
            train_mask = ~test_mask
            y_train = y[train_mask]
            y_test = y[test_mask]
            row: dict[str, Any] = {
                "fold": fold_number,
                "held_out_game": held_out_game,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "train_negative": int((y_train == 0).sum()),
                "train_positive": int((y_train == 1).sum()),
                "test_negative": int((y_test == 0).sum()),
                "test_positive": int((y_test == 1).sum()),
            }
            if len(np.unique(y_train)) < 2:
                row.update(
                    {
                        "status": "skipped_outer_train_single_class",
                        "threshold": None,
                        "n_estimators": None,
                        "inner_status": "not_run",
                        "metrics": metrics_or_none(np.array([]), np.array([]), np.array([])),
                        "majority_metrics": metrics_or_none(
                            np.array([]), np.array([]), np.array([])
                        ),
                    }
                )
                fold_rows.append(row)
                continue

            settings = inner_settings(
                X_known.loc[train_mask],
                y_train,
                subset.loc[train_mask, "session_id"].astype(str).to_numpy(),
                seed,
            )
            model = fit_pipeline(X_known.loc[train_mask], y_train, settings["n_estimators"], seed)
            probabilities = model.predict_proba(X_known.loc[test_mask])[:, 1]
            predictions = (probabilities >= settings["threshold"]).astype(int)
            model_metrics = metrics_or_none(y_test, probabilities, predictions)

            train_prevalence = float(np.mean(y_train))
            majority_class = int(train_prevalence > 0.5)
            base_probabilities = np.full(len(y_test), train_prevalence, dtype=float)
            base_predictions = np.full(len(y_test), majority_class, dtype=int)
            base_metrics = metrics_or_none(y_test, base_probabilities, base_predictions)
            row.update(
                {
                    "status": "success_two_class_test"
                    if len(np.unique(y_test)) == 2
                    else "success_single_class_test_metrics_na",
                    "threshold": settings["threshold"],
                    "n_estimators": settings["n_estimators"],
                    "inner_status": settings["status"],
                    "metrics": model_metrics,
                    "majority_class": majority_class,
                    "majority_metrics": base_metrics,
                }
            )
            fold_rows.append(row)
            oof_y.extend(y_test.tolist())
            oof_probability.extend(probabilities.tolist())
            oof_prediction.extend(predictions.tolist())
            baseline_probability.extend(base_probabilities.tolist())
            baseline_prediction.extend(base_predictions.tolist())

    pooled_metrics = metrics_or_none(
        np.asarray(oof_y, dtype=int),
        np.asarray(oof_probability, dtype=float),
        np.asarray(oof_prediction, dtype=int),
    )
    pooled_baseline = metrics_or_none(
        np.asarray(oof_y, dtype=int),
        np.asarray(baseline_probability, dtype=float),
        np.asarray(baseline_prediction, dtype=int),
    )
    valid_fold_metrics = [row["metrics"] for row in fold_rows if row["status"].startswith("success")]
    valid_baselines = [
        row["majority_metrics"] for row in fold_rows if row["status"].startswith("success")
    ]
    metrics = {
        "evaluation_status": "logo_complete_or_partially_evaluable" if len(games) > 1 else "no_logo_one_game",
        "total_game_folds": len(games) if len(games) > 1 else 0,
        "predicted_folds": int(sum(row["status"].startswith("success") for row in fold_rows)),
        "two_class_test_folds": int(
            sum(row["status"] == "success_two_class_test" for row in fold_rows)
        ),
        "mean_fold_roc_auc": mean_defined(valid_fold_metrics, "roc_auc"),
        "mean_fold_average_precision": mean_defined(valid_fold_metrics, "average_precision"),
        "mean_fold_f1_macro": mean_defined(valid_fold_metrics, "f1_macro"),
        "mean_fold_precision": mean_defined(valid_fold_metrics, "precision"),
        "mean_fold_recall": mean_defined(valid_fold_metrics, "recall"),
        "pooled_oof_n": len(oof_y),
        "pooled_oof_coverage": float(len(oof_y) / len(y)) if len(y) else 0.0,
        "pooled_oof": pooled_metrics,
        "majority_mean_fold_roc_auc": mean_defined(valid_baselines, "roc_auc"),
        "majority_mean_fold_average_precision": mean_defined(
            valid_baselines, "average_precision"
        ),
        "majority_mean_fold_f1_macro": mean_defined(valid_baselines, "f1_macro"),
        "majority_mean_fold_precision": mean_defined(valid_baselines, "precision"),
        "majority_mean_fold_recall": mean_defined(valid_baselines, "recall"),
        "majority_pooled_oof": pooled_baseline,
    }

    final_settings = inner_settings(
        X_known,
        y,
        subset["session_id"].astype(str).to_numpy(),
        seed,
    )
    final_model = fit_pipeline(X_known, y, final_settings["n_estimators"], seed)
    base.update(
        {
            "status": "trained",
            "model": final_model,
            "threshold": final_settings["threshold"],
            "fold_metrics": fold_rows,
            "metrics": metrics,
            "final_tuning": final_settings,
        }
    )
    return base


def label_report(label: str, entry: dict[str, Any]) -> list[str]:
    lines = [
        f"## `{label}`",
        "",
        f"- 状态：`{entry['status']}`",
        f"- 可用已知标签：{entry['n_known_usable']}；unknown：{entry['n_unknown']}"
        f"；特征全缺失/解析失败：{entry['n_feature_invalid']}",
        f"- 0/1：{entry['n_negative']}/{entry['n_positive']}",
        f"- 游戏/session：{entry['n_games']}/{entry['n_sessions']}",
        "",
    ]
    if entry["status"] != "trained":
        lines.append("没有拟合模型，不生成指标。")
        lines.append("")
        return lines

    lines.extend(
        [
            f"- 最终模型阈值：{fmt(entry['threshold'])}",
            f"- 最终 inner 策略：`{entry['final_tuning']['status']}`；"
            f"n_estimators={entry['final_tuning']['n_estimators']}",
            "",
            "| 折 | 留出游戏 | train 0/1 | test 0/1 | 状态 | threshold | ROC-AUC | AP | F1 macro | Precision | Recall | 多数类 F1 |",
            "|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in entry["fold_metrics"]:
        metrics = row["metrics"]
        majority = row["majority_metrics"]
        lines.append(
            f"| {row['fold']} | {row['held_out_game']} | "
            f"{row['train_negative']}/{row['train_positive']} | "
            f"{row['test_negative']}/{row['test_positive']} | `{row['status']}` | "
            f"{fmt(row.get('threshold'))} | {fmt(metrics['roc_auc'])} | "
            f"{fmt(metrics['average_precision'])} | {fmt(metrics['f1_macro'])} | "
            f"{fmt(metrics['precision'])} | {fmt(metrics['recall'])} | "
            f"{fmt(majority['f1_macro'])} |"
        )
    metrics = entry["metrics"]
    pooled = metrics["pooled_oof"]
    baseline = metrics["majority_pooled_oof"]
    lines.extend(
        [
            "",
            f"- 可计算指标的双类测试折：{metrics['two_class_test_folds']}/{metrics['total_game_folds']}",
            f"- 平均 fold ROC-AUC/AP/F1：{fmt(metrics['mean_fold_roc_auc'])} / "
            f"{fmt(metrics['mean_fold_average_precision'])} / {fmt(metrics['mean_fold_f1_macro'])}",
            f"- pooled OOF ROC-AUC/AP/F1/Precision/Recall：{fmt(pooled['roc_auc'])} / "
            f"{fmt(pooled['average_precision'])} / {fmt(pooled['f1_macro'])} / "
            f"{fmt(pooled['precision'])} / {fmt(pooled['recall'])}",
            f"- pooled OOF 覆盖：{metrics['pooled_oof_n']}/{entry['n_known_usable']} "
            f"({metrics['pooled_oof_coverage']:.2%})",
            f"- 多数类 pooled OOF ROC-AUC/AP/F1/Precision/Recall：{fmt(baseline['roc_auc'])} / "
            f"{fmt(baseline['average_precision'])} / {fmt(baseline['f1_macro'])} / "
            f"{fmt(baseline['precision'])} / {fmt(baseline['recall'])}",
            "",
        ]
    )
    return lines


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    data = pd.read_csv(args.input, encoding="utf-8-sig", low_memory=False)
    require_columns(data, [*ID_COLUMNS, *FEATURE_COLUMNS, *LABEL_COLUMNS], args.input)
    forbidden_features = [feature for feature in FEATURE_COLUMNS if FORBIDDEN.search(feature)]
    if forbidden_features:
        raise RuntimeError(f"Internal feature whitelist contains forbidden fields: {forbidden_features}")
    if data[ID_COLUMNS].isna().any().any():
        raise ValueError("Training matrix has missing clip/session/game identifiers")
    if data[ID_COLUMNS].duplicated().any():
        raise ValueError("Training matrix has duplicate clip/session/game identifiers")

    X = data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    valid_feature_rows = X.notna().any(axis=1)
    if "extraction_status" in data.columns:
        valid_feature_rows &= data["extraction_status"].isin(["success", "partial_single_frame"])

    started = time.perf_counter()
    label_entries: dict[str, dict[str, Any]] = {}
    log_lines = [
        "LightGBM video-only baseline training log",
        f"input={args.input}",
        f"model={args.model}",
        f"report={args.report}",
        f"seed={args.seed}",
        f"rows={len(data)} games={data['game_name'].nunique()} sessions={data['session_id'].nunique()}",
        f"feature_count={len(FEATURE_COLUMNS)} valid_feature_rows={int(valid_feature_rows.sum())}",
        "feature whitelist=video-only optical-flow/appearance/homography image-space proxies",
        "forbidden inputs=MS_rating/FMS/SSQ and direct matrix camera/object threshold variables",
        "unknown labels=excluded per target, never converted to negative",
        "outer split=leave-one-game-out; inner split=session-grouped, training-fold only",
        "default inner fallback=n_estimators 100, threshold 0.5",
        "",
    ]
    for label in LABEL_COLUMNS:
        target = normalized_target(data[label], label)
        entry = evaluate_label(data, X, target, label, valid_feature_rows, args.seed)
        label_entries[label] = entry
        metrics = entry.get("metrics") or {}
        log_lines.append(
            f"{label}: status={entry['status']} known={entry['n_known_usable']} "
            f"0={entry['n_negative']} 1={entry['n_positive']} "
            f"two_class_folds={metrics.get('two_class_test_folds', 'NA')} "
            f"mean_auc={metrics.get('mean_fold_roc_auc', 'NA')}"
        )

    trained = [entry for entry in label_entries.values() if entry["status"] == "trained"]
    defined_auc = [
        entry["metrics"]["mean_fold_roc_auc"]
        for entry in trained
        if entry["metrics"]["mean_fold_roc_auc"] is not None
    ]
    defined_f1 = [
        entry["metrics"]["mean_fold_f1_macro"]
        for entry in trained
        if entry["metrics"]["mean_fold_f1_macro"] is not None
    ]
    bundle = {
        "schema_version": "1.0",
        "task": "video_only_multilabel_rule_proxy_baseline",
        "warning": WARNING,
        "input_path": str(args.input),
        "random_seed": args.seed,
        "feature_names": list(FEATURE_COLUMNS),
        "label_columns": list(LABEL_COLUMNS),
        "labels": label_entries,
        "macro_metrics": {
            "trained_labels": len(trained),
            "total_labels": len(LABEL_COLUMNS),
            "labels_with_defined_mean_fold_auc": len(defined_auc),
            "macro_mean_fold_auc": float(np.mean(defined_auc)) if defined_auc else None,
            "macro_mean_fold_f1": float(np.mean(defined_f1)) if defined_f1 else None,
        },
        "input_guards": {
            "explicit_video_feature_whitelist": True,
            "symptom_outcomes_forbidden": True,
            "direct_matrix_threshold_features_forbidden": True,
            "unknown_is_negative": False,
        },
    }
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model)

    report_lines = [
        "# LightGBM 视频代理特征多标签基线",
        "",
        "**探索性，非最终结论；规则生成代理标签；因素规则复现不等于晕动症风险验证。**",
        "",
        "## 方法",
        "",
        f"- 输入：`{args.input.as_posix()}`，{len(data)} 个 clip。",
        f"- 游戏/session：{data['game_name'].nunique()}/{data['session_id'].nunique()}。",
        f"- 视频特征白名单：{len(FEATURE_COLUMNS)} 项。",
        "- 输入只含外观、Farneback 光流和图像单应性代理特征。单应性平移/旋转是像素空间代理，不是米制场景运动。",
        "- 没有输入 MS_rating/FMS/SSQ、矩阵直接阈值特征或学习式深度特征。",
        "- 每个已知且包含两类的标签拟合一个 LightGBM；unknown 按标签独立排除。",
        "- 外层是 Leave-One-Game-Out；在每个外层训练集内，用 session 分组划分选择 early-stopping 轮数和固定网格概率阈值。",
        "- 测试折只有一类时，ROC-AUC、AP、F1、Precision 和 Recall 全部记为 NA。",
        "- 多数类 baseline 只使用当前外层训练折的类别分布。",
        "",
        "### 显式输入白名单",
        "",
        *[f"- `{feature}`" for feature in FEATURE_COLUMNS],
        "",
    ]
    for label in LABEL_COLUMNS:
        report_lines.extend(label_report(label, label_entries[label]))
    macro = bundle["macro_metrics"]
    report_lines.extend(
        [
            "## 标签级汇总",
            "",
            f"- 实际拟合标签：{macro['trained_labels']}/{macro['total_labels']}",
            f"- 有可定义 mean-fold AUC 的标签：{macro['labels_with_defined_mean_fold_auc']}",
            f"- macro mean-fold ROC-AUC：{fmt(macro['macro_mean_fold_auc'])}",
            f"- macro mean-fold F1：{fmt(macro['macro_mean_fold_f1'])}",
            "",
            "## 限制",
            "",
            "1. 标签是矩阵/规则生成的因素代理，不是参与者症状结局。",
            "2. 视频代理能否识别规则标签，不能证明这些因素会导致晕动症。",
            "3. 光流和单应性值依赖视频分辨率、帧率、采样和画面内容，不是物理速度/加速度。",
            "4. 没有深度模型时不生成深度特征；本基线未训练视频深度模型。",
            "",
            "**探索性，非最终结论；规则生成代理标签；因素规则复现不等于晕动症风险验证。**",
            "",
        ]
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report_lines), encoding="utf-8")
    log_lines.extend(
        [
            "",
            f"trained_labels={len(trained)}/{len(LABEL_COLUMNS)}",
            f"macro_mean_fold_auc={macro['macro_mean_fold_auc']}",
            f"elapsed_sec={time.perf_counter()-started:.3f}",
        ]
    )
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(
        f"Saved {len(trained)} trained label models to {args.model}; report={args.report}"
    )


if __name__ == "__main__":
    main()
