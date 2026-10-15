#!/usr/bin/env python3
"""Train two video-only baselines for the fixed VR.net arXiv v1 rules.

The input must already align real clips and paper-rule labels. This script does
not generate labels, tune their cutoffs, join matrices, or download data. Each
label is evaluated independently with leave-one-game-out validation. Unknown
labels are excluded per target; all learned preprocessing uses training rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 42
THRESHOLD = 0.5
LABEL_COLUMNS = ["fast_camera_speed", "high_camera_acceleration"]
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
# Exact schema from extract_lightgbm_features.py. No numeric-column discovery.
VIDEO_FEATURE_COLUMNS = [*STATIC_FEATURES, *FLOW_FEATURES, *HOMOGRAPHY_FEATURES]
UNKNOWN_TOKENS = {"", "unknown", "na", "n/a", "nan", "none", "null", "tbd"}
FORBIDDEN_FEATURE = re.compile(
    r"(^|_)(camera_speed|camera_acceleration|speed|velocity|acceleration|accel|"
    r"ms_rating|fms|ssq|session_id|game_name|clip_id|window_id|windowid|timestamp|"
    r"start_time|end_time|path|label)($|_)",
    re.IGNORECASE,
)
RULE_ORIGIN = {
    "label_name": "按论文规则生成的风险因素标签",
    "paper": "VR.net: A Real-world Dataset for Virtual Reality Motion Sickness Research",
    "version": "arXiv:2306.03381v1 (2023-06-06)",
    "url": "https://arxiv.org/abs/2306.03381v1",
    "fast_camera_speed": {"operator": ">", "threshold": 3.0, "assumed_unit": "m/s"},
    "high_camera_acceleration": {
        "operator": ">", "threshold": 0.7, "assumed_unit": "m/s^2"
    },
    "label_generation": "Supplied upstream paper-rule clip labels; not recomputed by this script.",
    "assumptions": [
        "Upstream camera reconstruction, units, valid intervals, and clip aggregation must match the paper-rule label specification.",
        "Matrix-derived motion is treated as a rule-label proxy; metric calibration and deployment are not validated here.",
        "Labels are rule-factor states, not participant symptom outcomes or individual predictions.",
    ],
}
LGBM_PARAMETERS = {
    "objective": "binary",
    "n_estimators": 100,
    "learning_rate": 0.05,
    "num_leaves": 7,
    "min_child_samples": 5,
    "max_depth": 3,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "class_weight": "balanced",
    "n_jobs": 1,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "features/paper_rule_training_matrix.csv")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models/paper_rule_baseline.pkl")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "results/paper_rule_baseline_training.md")
    parser.add_argument("--oof", type=Path, default=PROJECT_ROOT / "results/paper_rule_oof_predictions.csv")
    parser.add_argument("--metrics", type=Path, default=PROJECT_ROOT / "results/paper_rule_baseline_metrics.json")
    parser.add_argument("--log", type=Path, default=PROJECT_ROOT / "results/paper_rule_baseline_training_log.txt")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def setup_logging(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("paper_rule_baseline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(path, mode="w", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def normalize_target(values: pd.Series, label: str) -> pd.Series:
    target = pd.Series(np.nan, index=values.index, dtype=float)
    for index, value in values.items():
        if pd.isna(value) or str(value).strip().lower() in UNKNOWN_TOKENS:
            continue
        try:
            number = float(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{label}: invalid value {value!r} at input row {index + 2}") from exc
        if number not in (0.0, 1.0):
            raise ValueError(f"{label}: expected 0, 1, or unknown; got {value!r} at input row {index + 2}")
        target.at[index] = number
    return target


def read_input(path: Path) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    # Reject duplicate header names before pandas can silently rename them.
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if len(header) != len(set(header)):
        raise ValueError("Input CSV contains duplicate column names")
    data = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if data.empty:
        raise ValueError("Training matrix has zero rows")
    required = ["session_id", "game_name", *LABEL_COLUMNS]
    missing = [name for name in required if name not in data.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    identifier_columns = ["session_id", "game_name"]
    if "clip_id" in data.columns:
        identifier_columns.append("clip_id")
    elif "windowid" in data.columns:
        identifier_columns.append("windowid")
    elif "window_id" in data.columns:
        identifier_columns.append("window_id")
    else:
        interval = next(
            (list(pair) for pair in [("start_time", "end_time"), ("start_time_sec", "end_time_sec"),
                                     ("start_sec", "end_sec"), ("window_start_sec", "window_end_sec")]
             if all(name in data.columns for name in pair)),
            None,
        )
        if interval is None:
            raise ValueError("Input requires clip_id, windowid/window_id, or a recognized start/end interval")
        identifier_columns.extend(interval)
    for column in identifier_columns:
        if data[column].isna().any() or data[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Input contains missing/blank identifier: {column}")
    if data[identifier_columns].duplicated().any():
        raise ValueError(f"Input contains duplicate real clip/window keys: {identifier_columns}")
    for column in ("session_id", "game_name"):
        data[column] = data[column].astype(str).str.strip()
    if data.groupby("session_id")["game_name"].nunique().gt(1).any():
        raise ValueError("A session_id maps to more than one game_name")
    selected = [name for name in VIDEO_FEATURE_COLUMNS if name in data.columns]
    if not selected:
        raise ValueError("No recognized RGB-only features. Expected the extract_lightgbm_features.py schema")
    forbidden = [name for name in selected if FORBIDDEN_FEATURE.search(name)]
    if forbidden:
        raise RuntimeError(f"Video whitelist unexpectedly contains a forbidden feature: {forbidden}")
    schema = {
        "version": "paper_rule_video_baseline_v1",
        "identifier_columns": identifier_columns,
        "feature_source": "scripts/extract_lightgbm_features.py RGB-only whitelist",
        "available_video_features": selected,
        "absent_video_features": [name for name in VIDEO_FEATURE_COLUMNS if name not in data.columns],
        "excluded_input_columns": [name for name in data.columns if name not in selected],
        "excluded_motion_or_outcome_columns": [
            name for name in data.columns if name not in selected and FORBIDDEN_FEATURE.search(name)
        ],
        "numeric_conversion": "Non-numeric and +/-infinite feature values become missing; median fitted on training fold only.",
        "labels": LABEL_COLUMNS.copy(),
    }
    return data, selected, schema


def counts(y: np.ndarray) -> dict[str, Any]:
    return {
        "known_support": int(len(y)),
        "negative": int(np.sum(y == 0)),
        "positive": int(np.sum(y == 1)),
        "classes": [int(value) for value in np.unique(y)],
        "positive_prevalence": float(np.mean(y)) if len(y) else None,
    }


def score(y: np.ndarray, probability: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    if not len(y):
        return {"confusion_matrix": None, "accuracy": None, "f1_binary": None,
                "f1_macro": None, "roc_auc": None, "precision": None, "recall": None,
                "known_support": 0, "positive_class": 1, "auc_status": "no_test_rows"}
    return {
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
        "accuracy": float(accuracy_score(y, prediction)),
        "f1_binary": float(f1_score(y, prediction, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y, prediction, labels=[0, 1], average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
        "precision": float(precision_score(y, prediction, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y, prediction, pos_label=1, zero_division=0)),
        "known_support": int(len(y)),
        "positive_class": 1,
        "auc_status": "defined" if len(np.unique(y)) == 2 else "NA_single_class_test",
    }


def fit_model(X_train: pd.DataFrame, y_train: np.ndarray, seed: int) -> tuple[Pipeline, dict[str, Any]]:
    if not len(y_train):
        raise ValueError("Cannot fit with zero known training labels")
    # These decisions inspect only this training set. Test values never affect
    # medians, retained features, class choice, or LightGBM parameters.
    all_missing = [name for name in X_train.columns if X_train[name].notna().sum() == 0]
    constant = [name for name in X_train.columns if name not in all_missing and X_train[name].nunique(dropna=True) <= 1]
    retained = [name for name in X_train.columns if name not in all_missing and name not in constant]
    transform = ColumnTransformer(
        [("median_imputer", SimpleImputer(strategy="median"), retained)] if retained else [],
        remainder="drop", verbose_feature_names_out=False,
    )
    classes = np.unique(y_train)
    if len(classes) == 1:
        estimator: Any = DummyClassifier(strategy="constant", constant=int(classes[0]))
        model_kind = "DummyClassifier_constant_single_class"
        lightgbm_status = "infeasible_single_class_train"
        constant_class: int | None = int(classes[0])
    elif not retained:
        constant_class = int(np.sum(y_train == 1) > np.sum(y_train == 0))
        estimator = DummyClassifier(strategy="constant", constant=constant_class)
        model_kind = "DummyClassifier_constant_no_variable_video_features"
        lightgbm_status = "infeasible_no_variable_train_features"
    else:
        estimator = lgb.LGBMClassifier(**LGBM_PARAMETERS, random_state=seed)
        model_kind = "LightGBM"
        lightgbm_status = "fitted"
        constant_class = None
    pipeline = Pipeline([("video_preprocessing", transform), ("classifier", estimator)])
    pipeline.fit(X_train, y_train)
    metadata = {
        "model_kind": model_kind,
        "lightgbm_status": lightgbm_status,
        "constant_class": constant_class,
        "retained_features": retained,
        "dropped_all_missing_features": all_missing,
        "dropped_constant_features": constant,
        "n_train_rows_all_video_features_missing": int(X_train.isna().all(axis=1).sum()),
        "train_classes": [int(value) for value in classes],
        "threshold": THRESHOLD,
    }
    return pipeline, metadata


def predict(pipeline: Pipeline, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    probabilities = pipeline.predict_proba(X_test)
    classes = pipeline.named_steps["classifier"].classes_
    positive_position = np.flatnonzero(classes == 1)
    probability = probabilities[:, int(positive_position[0])] if len(positive_position) else np.zeros(len(X_test))
    return probability.astype(float), (probability >= THRESHOLD).astype(int)


def defined_mean(rows: list[dict[str, Any]], metric: str) -> float | None:
    values = [row["metrics"][metric] for row in rows if row["metrics"][metric] is not None]
    return float(np.mean(values)) if values else None


def evaluate_label(
    data: pd.DataFrame, X: pd.DataFrame, target: pd.Series, label: str,
    identifier_columns: list[str], seed: int, logger: logging.Logger,
) -> tuple[dict[str, Any], Pipeline | None, pd.DataFrame]:
    known_mask = target.notna()
    y_known = target.loc[known_mask].astype(int).to_numpy()
    games = sorted(data["game_name"].unique())
    oof = data[identifier_columns].copy()
    oof.insert(0, "input_row", np.arange(len(data)))
    oof["label"] = label
    oof["y_true"] = target
    oof["positive_class"] = 1
    oof["probability_class_1"] = np.nan
    oof["prediction"] = pd.Series(pd.NA, index=data.index, dtype="Int64")
    oof["threshold"] = THRESHOLD
    oof["fold"] = pd.Series(pd.NA, index=data.index, dtype="Int64")
    oof["held_out_game"] = pd.Series(pd.NA, index=data.index, dtype="string")
    oof["model_kind"] = pd.Series(pd.NA, index=data.index, dtype="string")
    oof["oof_status"] = np.where(known_mask, "no_logo_one_game", "unknown_excluded")
    entry: dict[str, Any] = {
        "label": label,
        "n_total": int(len(data)),
        "n_known": int(known_mask.sum()),
        "n_unknown": int((~known_mask).sum()),
        "known_counts": counts(y_known),
        "n_games_total": len(games),
        "n_sessions_total": int(data["session_id"].nunique()),
        "n_games_known": int(data.loc[known_mask, "game_name"].nunique()),
        "n_sessions_known": int(data.loc[known_mask, "session_id"].nunique()),
        "n_known_rows_all_video_features_missing": int(X.loc[known_mask].isna().all(axis=1).sum()),
        "fold_metrics": [],
        "final_fit": None,
    }
    if len(games) >= 2:
        for fold_number, held_out in enumerate(games, 1):
            train_mask = known_mask & data["game_name"].ne(held_out)
            test_mask = known_mask & data["game_name"].eq(held_out)
            y_train = target.loc[train_mask].astype(int).to_numpy()
            y_test = target.loc[test_mask].astype(int).to_numpy()
            fold: dict[str, Any] = {
                "fold": fold_number,
                "held_out_game": held_out,
                "train": counts(y_train),
                "test": counts(y_test),
                "n_train_games": int(data.loc[train_mask, "game_name"].nunique()),
                "n_train_sessions": int(data.loc[train_mask, "session_id"].nunique()),
                "n_test_sessions": int(data.loc[test_mask, "session_id"].nunique()),
                "test_unknown_support": int((data["game_name"].eq(held_out) & ~known_mask).sum()),
                "model_kind": None,
                "lightgbm_status": "not_fitted",
                "metrics": score(np.array([], dtype=int), np.array([]), np.array([], dtype=int)),
                "train_majority_baseline_metrics": score(np.array([], dtype=int), np.array([]), np.array([], dtype=int)),
            }
            oof.loc[test_mask, "fold"] = fold_number
            oof.loc[test_mask, "held_out_game"] = held_out
            if not len(y_test):
                fold["status"] = "no_known_test_labels"
            elif not len(y_train):
                fold["status"] = "infeasible_no_known_train_labels"
                oof.loc[test_mask, "oof_status"] = fold["status"]
            else:
                pipeline, fit_metadata = fit_model(X.loc[train_mask], y_train, seed)
                probability, prediction = predict(pipeline, X.loc[test_mask])
                fold.update(fit_metadata)
                fold["status"] = "evaluated"
                fold["metrics"] = score(y_test, probability, prediction)
                prevalence = float(np.mean(y_train))
                majority = int(prevalence > 0.5)
                fold["train_majority_baseline_metrics"] = score(
                    y_test, np.full(len(y_test), prevalence), np.full(len(y_test), majority, dtype=int)
                )
                fold["train_majority_class"] = majority
                fold["train_prevalence_probability"] = prevalence
                oof.loc[test_mask, "probability_class_1"] = probability
                oof.loc[test_mask, "prediction"] = prediction
                oof.loc[test_mask, "model_kind"] = fit_metadata["model_kind"]
                oof.loc[test_mask, "oof_status"] = "evaluated"
            entry["fold_metrics"].append(fold)
            logger.info(
                "%s fold=%s held_out=%s train=%s test=%s status=%s model=%s accuracy=%s f1_binary=%s auc=%s",
                label, fold_number, held_out, len(y_train), len(y_test), fold["status"],
                fold["model_kind"], fold["metrics"]["accuracy"], fold["metrics"]["f1_binary"], fold["metrics"]["roc_auc"],
            )
    evaluated = oof["oof_status"].eq("evaluated")
    pooled = score(
        oof.loc[evaluated, "y_true"].astype(int).to_numpy(),
        oof.loc[evaluated, "probability_class_1"].to_numpy(dtype=float),
        oof.loc[evaluated, "prediction"].to_numpy(dtype=int),
    )
    entry["evaluation"] = {
        "protocol": "leave-one-game-out",
        "status": "logo" if len(games) >= 2 else "not_evaluable_only_one_game",
        "total_game_folds": len(games) if len(games) >= 2 else 0,
        "evaluated_folds": int(sum(row["status"] == "evaluated" for row in entry["fold_metrics"])),
        "folds_with_defined_auc": int(sum(row["metrics"]["roc_auc"] is not None for row in entry["fold_metrics"])),
        "mean_fold_accuracy": defined_mean(entry["fold_metrics"], "accuracy"),
        "mean_fold_f1_binary": defined_mean(entry["fold_metrics"], "f1_binary"),
        "mean_fold_f1_macro": defined_mean(entry["fold_metrics"], "f1_macro"),
        "mean_fold_roc_auc": defined_mean(entry["fold_metrics"], "roc_auc"),
        "pooled_oof": pooled,
        "pooled_oof_coverage": float(evaluated.sum() / known_mask.sum()) if known_mask.sum() else None,
    }
    if not len(y_known):
        final_model = None
        entry["status"] = "infeasible_all_labels_unknown"
    else:
        final_model, final_metadata = fit_model(X.loc[known_mask], y_known, seed)
        entry["final_fit"] = final_metadata
        entry["status"] = "lightgbm_fitted" if final_metadata["lightgbm_status"] == "fitted" else "constant_baseline_fitted_lightgbm_infeasible"
    logger.info("%s final=%s known=%s unknown=%s 0=%s 1=%s pooled_auc=%s", label, entry["status"],
                entry["n_known"], entry["n_unknown"], entry["known_counts"]["negative"],
                entry["known_counts"]["positive"], pooled["roc_auc"])
    return entry, final_model, oof


def fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def label_report(label: str, entry: dict[str, Any]) -> list[str]:
    count = entry["known_counts"]
    lines = [
        f"## `{label}`", "",
        f"- 全量片段：{entry['n_total']}；known：{entry['n_known']}；unknown：{entry['n_unknown']}；0/1：{count['negative']}/{count['positive']}。",
        f"- 已知标签游戏/session：{entry['n_games_known']}/{entry['n_sessions_known']}；全量游戏/session：{entry['n_games_total']}/{entry['n_sessions_total']}。",
        f"- 正类比例：{fmt(count['positive_prevalence'])}；已知标签且视频输入全缺失的片段：{entry['n_known_rows_all_video_features_missing']}。",
        f"- 全数据最终拟合：`{entry['status']}`。", "",
        "| 留出游戏 | train known (0/1), classes | test known (0/1), classes | unknown test | 模型/状态 | TN, FP, FN, TP | Accuracy | F1 binary | F1 macro | AUC (positive=1) |",
        "|---|---|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for fold in entry["fold_metrics"]:
        train, test, metrics = fold["train"], fold["test"], fold["metrics"]
        confusion = metrics["confusion_matrix"]
        confusion_text = "NA" if confusion is None else str([*confusion[0], *confusion[1]])
        lines.append(
            f"| {fmt(fold['held_out_game'])} | {train['known_support']} ({train['negative']}/{train['positive']}), {train['classes']} | "
            f"{test['known_support']} ({test['negative']}/{test['positive']}), {test['classes']} | {fold['test_unknown_support']} | "
            f"{fmt(fold['model_kind'] or fold['status'])} | {confusion_text} | {fmt(metrics['accuracy'])} | "
            f"{fmt(metrics['f1_binary'])} | {fmt(metrics['f1_macro'])} | {fmt(metrics['roc_auc'])} |"
        )
    evaluation = entry["evaluation"]
    pooled = evaluation["pooled_oof"]
    lines.extend([
        "", f"- LOGO 状态：`{evaluation['status']}`；已评估折：{evaluation['evaluated_folds']}/{evaluation['total_game_folds']}；有定义 AUC 的折：{evaluation['folds_with_defined_auc']}。",
        f"- mean fold Accuracy/F1 binary/F1 macro/AUC：{fmt(evaluation['mean_fold_accuracy'])} / {fmt(evaluation['mean_fold_f1_binary'])} / {fmt(evaluation['mean_fold_f1_macro'])} / {fmt(evaluation['mean_fold_roc_auc'])}。",
        f"- pooled OOF known support：{pooled['known_support']}/{entry['n_known']}；覆盖率：{fmt(evaluation['pooled_oof_coverage'])}。",
        f"- pooled OOF Accuracy/F1 binary/F1 macro/AUC：{fmt(pooled['accuracy'])} / {fmt(pooled['f1_binary'])} / {fmt(pooled['f1_macro'])} / {fmt(pooled['roc_auc'])}。",
        f"- pooled OOF 混淆矩阵（行 true 0/1，列 predicted 0/1）：`{pooled['confusion_matrix']}`。",
        "", "### 各训练折预处理与拟合可行性", "",
    ])
    for fold in entry["fold_metrics"]:
        lines.append(f"- `{fmt(fold['held_out_game'])}`：`{fold['lightgbm_status']}`；train/test sessions：{fold['n_train_sessions']}/{fold['n_test_sessions']}。")
        if fold["status"] == "evaluated":
            lines.extend([
                f"  - 保留特征：{len(fold['retained_features'])}；训练折全缺失剔除：`{fold['dropped_all_missing_features']}`；训练折常数剔除：`{fold['dropped_constant_features']}`。",
                f"  - 当前训练折多数类基线 Accuracy/F1 binary/F1 macro/AUC：{fmt(fold['train_majority_baseline_metrics']['accuracy'])} / {fmt(fold['train_majority_baseline_metrics']['f1_binary'])} / {fmt(fold['train_majority_baseline_metrics']['f1_macro'])} / {fmt(fold['train_majority_baseline_metrics']['roc_auc'])}。",
            ])
    final = entry["final_fit"]
    if final:
        lines.extend(["", f"- 最终模型类型：`{final['model_kind']}`；LightGBM 可行性：`{final['lightgbm_status']}`。",
                      f"- 全量已知训练行拟合后保留列：`{final['retained_features']}`。",
                      f"- 最终全缺失列：`{final['dropped_all_missing_features']}`；常数列：`{final['dropped_constant_features']}`。"])
    lines.append("")
    return lines


def write_report(path: Path, summary: dict[str, Any], args: argparse.Namespace) -> None:
    schema = summary["schema"]
    lines = [
        "# 论文规则标签的 LightGBM 视频特征基线", "",
        "**探索性：本结果评估 RGB 视频代理特征对两项论文规则标签的识别能力；标签不是参与者症状结局，不提供个体晕动症预测。**", "",
        "## 规则来源和标签假设", "",
        "标签统一命名为“按论文规则生成的风险因素标签”。",
        "规则来源：[VR.net arXiv:2306.03381v1 (2023-06-06)](https://arxiv.org/abs/2306.03381v1)。",
        "- `fast_camera_speed`：camera speed > 3，论文单位假设 m/s。",
        "- `high_camera_acceleration`：camera acceleration > 0.7，论文单位假设 m/s²。",
        "- 标签由上游真实 clip 对齐矩阵提供；本脚本直接读取 0/1/unknown，不重新生成标签、不搜索分位数、不改规则阈值。",
        "- 上游相机重建、单位、有效导数时间间隔和片段聚合定义应与标签说明一致。矩阵度量的物理标定、跨设备适用性与部署均未在本训练中验证。", "",
        "## 数据、模型和输入", "",
        f"- 输入：`{args.input}`；片段 {summary['n_rows']}；游戏 {summary['n_games']}；session {summary['n_sessions']}。",
        f"- 主模型只使用现有 `extract_lightgbm_features.py` 显式白名单中的 {len(schema['available_video_features'])} 项 RGB 视频特征。",
        "- 输入包括外观统计、Farneback 光流和图像单应性代理。光流/单应性以像素或图像角度为单位，不能视为矩阵中的物理速度/加速度。",
        "- 所有非白名单列均排除，包含矩阵速度、加速度及其等价遥测/阈值字段、MS/FMS/SSQ、其他标签、ID、时间和路径。",
        f"- 可用视频白名单列：`{schema['available_video_features']}`。",
        f"- schema 不含的视频白名单列：`{schema['absent_video_features']}`。",
        f"- 排除的输入列：`{schema['excluded_input_columns']}`。",
        f"- 数值转换后各列缺失数：`{summary['missing_values_by_feature']}`。",
        f"- 非数值或无限数转换为缺失的各列数量：`{summary['invalid_values_by_feature']}`。",
        f"- 参数固定：`{summary['lightgbm_parameters']}`；seed={summary['random_seed']}；预测正类阈值固定 {THRESHOLD}。",
        "- 每个标签单独排除 unknown；保留已知但视频值缺失的行，训练折中位数填补。全缺失/常数列仅根据当前训练折判断并剔除；最终模型使用全量该标签已知行重新拟合预处理。",
        "- 验证使用 Leave-One-Game-Out，无随机 clip 划分、参数搜索、early stopping 或概率阈值调优。",
        "- 单类训练折 LightGBM 不可行，以 DummyClassifier(strategy=constant) 预测该唯一类别；训练折无变量视频特征时以训练多数类常数基线替代。训练标签为空时没有可拟合模型，该折保留缺口。",
        "- 测试单类仅 AUC 为 NA，Accuracy/F1/混淆矩阵仍计算。F1 binary 正类=1；F1 macro 固定 labels=[0,1]，zero_division=0；混淆矩阵固定 labels=[0,1]。",
        "- mean fold AUC 只平均有定义的游戏折；pooled OOF 指标使用全部已评估预测，并保留实际 OOF 覆盖率。", "",
    ]
    for label in LABEL_COLUMNS:
        lines.extend(label_report(label, summary["labels"][label]))
    lines.extend([
        "## 解释限制和交付", "",
        f"- 当前仅 {summary['n_games']} 个游戏。仅两游戏时只有两个 LOGO 折，每折只在另一游戏上训练，无法稳定估计泛化范围。",
        "- 同一 session 的片段共享画面、游戏内容与采集条件，存在相关性；片段数量不能作为独立参与者或独立场景数量。LOGO 隔离游戏，无法消除游戏内相关性。",
        "- 类别不平衡和单类游戏会使 Accuracy、binary F1 与 macro F1 呈现不同结果；AUC 缺失不是零。常数基线的指标不应解释为成功训练了 LightGBM。",
        "- pooled OOF AUC 比较不同训练折输出的分数，可能受游戏/标签分布差异影响；应连同逐游戏结果查看。",
        "- 规则标签识别不能建立因素的因果作用、通用安全阈值、low/medium/high 等级或参与者症状预测能力。",
        "- 已保存的最终模型在各标签全量已知数据上拟合，不能用其训练集输出报告泛化性能；部署、跨视频采样设置和用户级应用未验证。",
        f"- 模型 bundle：`{args.model}`；机器可读指标：`{args.metrics}`；逐片段逐标签 OOF：`{args.oof}`；真实运行日志：`{args.log}`。",
        "- bundle 的 `models[label]` 为 sklearn Pipeline，输入应为包含 `feature_names` 的数值 DataFrame；预先将非数值/无限数转换为缺失。各模型保存自己的列选择器、imputer 和分类器。`imputers[label]` 同时提供最终中位数 imputer；无保留列时为 None。", "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace, logger: logging.Logger) -> None:
    started = time.perf_counter()
    logger.info("START input=%s seed=%s", args.input, args.seed)
    data, features, schema = read_input(args.input)
    numeric = data[features].apply(pd.to_numeric, errors="coerce")
    invalid = (data[features].notna() & numeric.isna()) | numeric.isin([np.inf, -np.inf])
    X = numeric.replace([np.inf, -np.inf], np.nan)
    labels: dict[str, Any] = {}
    models: dict[str, Pipeline | None] = {}
    imputers: dict[str, SimpleImputer | None] = {}
    oof_tables: list[pd.DataFrame] = []
    logger.info("rows=%s games=%s sessions=%s available_video_features=%s absent_video_features=%s",
                len(data), data["game_name"].nunique(), data["session_id"].nunique(), len(features), schema["absent_video_features"])
    for label in LABEL_COLUMNS:
        target = normalize_target(data[label], label)
        entry, final_model, oof = evaluate_label(
            data, X, target, label, schema["identifier_columns"], args.seed, logger
        )
        labels[label] = entry
        models[label] = final_model
        imputers[label] = (
            final_model.named_steps["video_preprocessing"].named_transformers_.get("median_imputer")
            if final_model is not None and entry["final_fit"]["retained_features"] else None
        )
        oof_tables.append(oof)
    summary = {
        "schema_version": schema["version"],
        "task": "RGB_video_baseline_for_two_fixed_paper_rules",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "random_seed": args.seed,
        "threshold": THRESHOLD,
        "n_rows": int(len(data)),
        "n_games": int(data["game_name"].nunique()),
        "n_sessions": int(data["session_id"].nunique()),
        "schema": schema,
        "rule_origin": RULE_ORIGIN,
        "labels": labels,
        "missing_values_by_feature": {name: int(X[name].isna().sum()) for name in features},
        "invalid_values_by_feature": {name: int(invalid[name].sum()) for name in features},
        "lightgbm_parameters": {**LGBM_PARAMETERS, "random_state": args.seed},
        "package_versions": {"pandas": pd.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__, "lightgbm": lgb.__version__},
        "deployment_verified": False,
        "individual_prediction_validated": False,
        "input_guards": {"explicit_video_whitelist": True, "unknown_is_negative": False,
                         "direct_rule_telemetry_used": False, "symptom_outcomes_used": False,
                         "preprocessing_fit_scope": "current training fold; final fit on all known rows per label"},
    }
    definition_path = PROJECT_ROOT / 'results/paper_rule_definition.json'
    if definition_path.is_file():
        summary['label_definition'] = json.loads(definition_path.read_text(encoding='utf-8'))
        summary['label_definition_sha256'] = hashlib.sha256(definition_path.read_bytes()).hexdigest()
    bundle = {**summary, "feature_names": features, "label_columns": LABEL_COLUMNS.copy(),
              "models": models, "imputers": imputers, "model_output_positive_class": 1}
    for output in (args.model, args.metrics, args.oof):
        output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model)
    args.metrics.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    pd.concat(oof_tables, ignore_index=True).to_csv(args.oof, index=False, encoding="utf-8-sig")
    write_report(args.report, summary, args)
    logger.info("SUCCESS elapsed_seconds=%.3f model=%s report=%s metrics=%s oof=%s",
                time.perf_counter() - started, args.model, args.report, args.metrics, args.oof)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    logger = setup_logging(args.log)
    try:
        run(args, logger)
    except Exception:
        logger.exception("FAILED: no successful training completion; see traceback")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
