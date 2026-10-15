#!/usr/bin/env python3
"""Train a small LightGBM risk-level classifier when labels permit it.

This script intentionally skips fitting when the available window labels do
not contain at least two classes. It never manufactures validation metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm
import numpy as np
import pandas as pd
import sklearn
import yaml
from joblib import dump
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier


LEVELS = ["low", "medium", "high"]
EXCLUDE_COLUMNS = {
    "window_id", "window_index", "window_start_sec", "window_end_sec", "window_center_sec",
    "start_frame", "end_frame", "start_time", "end_time",
    "fms_raw", "fms_label", "label_source", "session_id", "scene_id", "scene_name",
    "observed_session_duration_sec", "observed_session_duration_min", "source_frame_count",
    "source_pair_count", "window_status", "analysis_max_dimension_px",
    "expected_jpg_count_from_index", "found_jpg_count", "decoded_jpg_count",
    "missing_expected_jpg_count", "failed_jpg_count",
}
PREDICTION_FIELDS = ["window_id", "fms_label", "predicted_risk_level", "split_role"]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def number_or_none(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def auc_macro(y_true: np.ndarray, probs: np.ndarray, labels: np.ndarray) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    try:
        if len(labels) == 2:
            return float(roc_auc_score(y_true, probs[:, 1]))
        return float(roc_auc_score(y_true, probs, labels=labels, multi_class="ovr", average="macro"))
    except ValueError:
        return None


def write_report(path: Path, metadata: dict[str, Any]) -> None:
    counts = metadata["label_counts"]
    metrics = metadata.get("metrics") or {}
    cm = metadata.get("confusion_matrix")
    show_metric = lambda key: "未计算" if metrics.get(key) is None else metrics[key]
    lines = [
        "# VR.net 风险等级模型训练记录", "",
        f"日期：{metadata['run_date']}", "",
        "**流程验证，非最终结论。** 本模型（若成功拟合）只针对窗口级标签，不预测个体晕动症。", "",
        "## 输入与标签", "",
        f"- 输入：`{metadata['input_path']}`",
        f"- 有效标签窗口：{metadata['n_labeled']} / {metadata['n_windows']}",
        f"- 标签来源字段：`{metadata['source_label_column']}`；分箱配置：`{metadata['mapping_path']}`",
        f"- 类别分布：低 {counts.get('low', 0)}，中 {counts.get('medium', 0)}，高 {counts.get('high', 0)}",
        f"- 可用数值特征：{metadata['n_features']}",
        f"- 排除字段：{', '.join(metadata.get('excluded_nonfeatures', [])) or '无额外字段'}",
        f"- 因全空/非数值跳过字段：{', '.join(metadata.get('dropped_feature_columns', [])) or '无'}",
        "", "## 拟合与评估", "",
        f"- 状态：**{metadata['fit_status']}**",
        f"- 模型：{metadata.get('model_name', 'LightGBM（未拟合）')}",
        f"- 评估方案：{metadata['evaluation_plan']}",
        f"- Accuracy：{show_metric('accuracy')}",
        f"- F1 (macro)：{show_metric('f1_macro')}",
        f"- AUC (one-vs-rest macro)：{show_metric('auc_ovr_macro')}",
        f"- 主要类基线 AUC：{show_metric('majority_baseline_auc_ovr_macro')}",
        f"- 随机种子：{metadata['random_seed']}",
        f"- 类别支持数：训练 {metadata.get('train_class_counts', {})}；验证 {metadata.get('validation_class_counts', {})}",
        "", "### 混淆矩阵", "",
    ]
    if cm:
        lines.extend(["标签/预测顺序：`low, medium, high`", "", "```text"])
        lines.extend(" ".join(str(value) for value in row) for row in cm)
        lines.append("```")
    else:
        lines.append("未计算：当前标签不能构成有效的多类别训练/验证划分。")
    lines.extend(["", "## 跳过或限制原因", ""])
    reasons = metadata.get("reasons", [])
    lines.extend(f"- {reason}" for reason in reasons) if reasons else lines.append("- 未发现阻止拟合的原因。")
    lines.extend([
        "", "## 解释边界", "",
        "窗口由同一 VR.net session 按时间切分；随机 80/20 窗口划分不是独立参与者/场景泛化评估。当前数据不能替代 PROTOCOL.md 的 LOSO 与 Leave-One-Scene-Out 验证。",
        "样本量小，任何可计算的指标都不可靠；本报告不构成模型有效性、因果关系或个体风险结论。",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="features/training_feature_matrix_vrnet.csv")
    parser.add_argument("--mapping", default="configs/label_mapping_vrnet.yaml")
    parser.add_argument("--model", default="models/risk_model_vrnet.pkl")
    parser.add_argument("--metadata", default="models/risk_model_vrnet_metadata.json")
    parser.add_argument("--predictions", default="features/risk_predictions_vrnet.csv")
    parser.add_argument("--report", default="results/risk_model_training_vrnet.md")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = lambda raw: (root / raw).resolve() if not Path(raw).is_absolute() else Path(raw)
    input_path, mapping_path = path(args.input), path(args.mapping)
    model_path, metadata_path = path(args.model), path(args.metadata)
    predictions_path, report_path = path(args.predictions), path(args.report)
    with mapping_path.open("r", encoding="utf-8-sig") as handle:
        mapping = yaml.safe_load(handle)
    frame = pd.read_csv(input_path, encoding="utf-8-sig")
    if "fms_label" not in frame.columns or "window_id" not in frame.columns:
        raise ValueError("Training matrix must contain window_id and fms_label")
    labels = frame["fms_label"].astype(str).str.casefold()
    labeled_mask = labels.isin(LEVELS)
    labeled = frame.loc[labeled_mask].copy()
    y_text = labels.loc[labeled_mask].to_numpy()
    counts = Counter(y_text.tolist())
    for level in LEVELS:
        counts.setdefault(level, 0)

    dropped: list[str] = []
    feature_names: list[str] = []
    all_feature_frame = pd.DataFrame(index=frame.index)
    for column in frame.columns:
        if column in EXCLUDE_COLUMNS or any(token in column.casefold() for token in ("fms", "ssq", "sickness", "label")):
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            feature_names.append(column)
            all_feature_frame[column] = values.replace([np.inf, -np.inf], np.nan)
        else:
            if column not in {"session_id", "scene_id"}:
                dropped.append(column)
    reasons: list[str] = []
    fit_status = "trained"
    evaluation_plan = "5-fold stratified cross-validation" if len(labeled) > 50 else "80/20 train/validation split (requested small-sample procedure)"
    metrics: dict[str, Any] = {}
    confusion: list[list[int]] | None = None
    train_counts: dict[str, int] = {}
    validation_counts: dict[str, int] = {}
    predictions: list[dict[str, Any]] = []
    model_classes: list[str] = []

    if len(labeled) < 2:
        fit_status = "skipped"
        reasons.append("有效标签窗口少于 2。")
    elif len(set(y_text.tolist())) < 2:
        fit_status = "skipped"
        reasons.append(
            "所有有标签窗口都落在同一个风险等级（" + str(next(iter(set(y_text.tolist())))) + "）；"
            "单类别数据不能训练或验证 low/medium/high 分类器，也不能计算 F1/AUC。"
        )
    elif not feature_names:
        fit_status = "skipped"
        reasons.append("没有可用的非标签数值特征。")

    model_payload: dict[str, Any] | None = None
    if fit_status == "trained":
        ordered_classes = [level for level in LEVELS if level in set(y_text.tolist())]
        class_to_index = {level: index for index, level in enumerate(ordered_classes)}
        y = np.asarray([class_to_index[level] for level in y_text], dtype=int)
        X = all_feature_frame.loc[labeled.index, feature_names]
        n_classes = len(encoder.classes_)
        estimator = Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("classifier", LGBMClassifier(
                objective="multiclass", num_class=n_classes, n_estimators=50,
                learning_rate=0.03, max_depth=3, num_leaves=7,
                random_state=int(mapping.get("random_seed", 42)), n_jobs=1, verbosity=-1,
            )),
        ])
        if len(labeled) > 50:
            class_counts = Counter(y.tolist())
            if min(class_counts.values()) < 5:
                fit_status = "skipped"
                reasons.append("样本量虽超过 50，但至少一个类别少于 5 个窗口，无法执行 5 折分层评估。")
            else:
                folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(mapping.get("random_seed", 42)))
                cv_prob = cross_val_predict(estimator, X, y, cv=folds, method="predict_proba", n_jobs=1)
                cv_pred = cv_prob.argmax(axis=1)
                metrics["accuracy"] = float(accuracy_score(y, cv_pred))
                metrics["f1_macro"] = float(f1_score(y, cv_pred, average="macro", labels=np.arange(n_classes), zero_division=0))
                metrics["auc_ovr_macro"] = auc_macro(y, cv_prob, np.arange(n_classes))
                majority = DummyClassifier(strategy="most_frequent")
                base_prob = cross_val_predict(majority, X, y, cv=folds, method="predict_proba", n_jobs=1)
                # Align the baseline probability columns to LabelEncoder classes.
                base_full = np.zeros((len(y), n_classes), dtype=float)
                for j, cls in enumerate(majority.classes_):
                    base_full[:, int(cls)] = base_prob[:, j]
                metrics["majority_baseline_auc_ovr_macro"] = auc_macro(y, base_full, np.arange(n_classes))
                confusion = confusion_matrix(y, cv_pred, labels=np.arange(n_classes)).tolist()
                train_counts = dict(Counter(ordered_classes[int(x)] for x in y))
                validation_counts = dict(Counter(ordered_classes[int(x)] for x in y))
                estimator.fit(X, y)
        else:
            class_counts = Counter(y.tolist())
            test_size = max(n_classes, int(math.ceil(len(y) * 0.2))) / len(y)
            test_size = min(0.49, test_size)
            stratify = y if min(class_counts.values()) >= 2 and int(math.ceil(len(y) * 0.2)) >= n_classes else None
            try:
                X_train, X_valid, y_train, y_valid = train_test_split(
                    X, y, test_size=test_size, random_state=int(mapping.get("random_seed", 42)), stratify=stratify,
                )
            except ValueError as exc:
                fit_status = "skipped"
                reasons.append(f"无法构造有效 80/20 训练/验证拆分：{exc}")
            if fit_status == "trained":
                estimator.fit(X_train, y_train)
                valid_prob = estimator.predict_proba(X_valid)
                valid_pred = valid_prob.argmax(axis=1)
                metrics["accuracy"] = float(accuracy_score(y_valid, valid_pred))
                metrics["f1_macro"] = float(f1_score(y_valid, valid_pred, average="macro", labels=np.arange(n_classes), zero_division=0))
                metrics["auc_ovr_macro"] = auc_macro(y_valid, valid_prob, np.arange(n_classes))
                majority = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
                base_prob = majority.predict_proba(X_valid)
                base_full = np.zeros((len(y_valid), n_classes), dtype=float)
                for j, cls in enumerate(majority.classes_):
                    base_full[:, int(cls)] = base_prob[:, j]
                metrics["majority_baseline_auc_ovr_macro"] = auc_macro(y_valid, base_full, np.arange(n_classes))
                confusion = confusion_matrix(y_valid, valid_pred, labels=np.arange(n_classes)).tolist()
                train_counts = {ordered_classes[int(k)]: int(v) for k, v in Counter(y_train).items()}
                validation_counts = {ordered_classes[int(k)]: int(v) for k, v in Counter(y_valid).items()}
                # Refit on all labeled windows for later explanation; validation metrics remain holdout-only.
                estimator.fit(X, y)

        if fit_status == "trained":
            model_classes = ordered_classes
            model_payload = {
                "pipeline": estimator,
                "feature_names": feature_names,
                "classes": model_classes,
                "random_seed": int(mapping.get("random_seed", 42)),
                "mapping_path": str(mapping_path),
                "training_rows": int(len(labeled)),
            }
            if not model_path.exists():
                model_path.parent.mkdir(parents=True, exist_ok=True)
                dump(model_payload, model_path)
            else:
                reasons.append(f"模型文件已存在，未覆盖：{model_path}")
            full_prob = estimator.predict_proba(all_feature_frame[feature_names])
            predicted = estimator.classes_[full_prob.argmax(axis=1)]
            for idx, label in enumerate(predicted):
                predictions.append({
                    "window_id": frame.iloc[idx].get("window_id", f"window_{idx:03d}"),
                    "fms_label": frame.iloc[idx].get("fms_label", ""),
                    "predicted_risk_level": ordered_classes[int(label)],
                    "split_role": "all_windows_final_model",
                })

    if fit_status == "skipped":
        # Do not load or replace any existing artifact when this data cannot fit a classifier.
        for metric in ("accuracy", "f1_macro", "auc_ovr_macro", "majority_baseline_auc_ovr_macro"):
            metrics[metric] = None
        evaluation_plan = "80/20 split not run: label/class support is insufficient"
    else:
        for metric in ("accuracy", "f1_macro", "auc_ovr_macro", "majority_baseline_auc_ovr_macro"):
            metrics.setdefault(metric, None)
        if metrics.get("auc_ovr_macro") is None:
            reasons.append("AUC 未计算：验证集没有足够类别支持或指标数学上未定义。")
        if metrics.get("majority_baseline_auc_ovr_macro") is None:
            reasons.append("多数类基线 AUC 未计算：验证标签类别不足。")
    write_predictions(predictions_path, predictions)
    metadata = {
        "fit_status": fit_status,
        "run_date": datetime.now(timezone.utc).date().isoformat(),
        "input_path": str(input_path), "mapping_path": str(mapping_path),
        "source_label_column": mapping.get("source_label_column", "unknown"),
        "n_windows": int(len(frame)), "n_labeled": int(len(labeled)),
        "label_counts": {level: int(counts[level]) for level in LEVELS},
        "n_features": len(feature_names), "feature_names": feature_names,
        "dropped_feature_columns": dropped, "excluded_nonfeatures": sorted(EXCLUDE_COLUMNS),
        "model_path": str(model_path) if fit_status == "trained" and model_path.exists() else None,
        "model_name": f"LightGBM {lightgbm.__version__}" if fit_status == "trained" else "LightGBM (not fitted)",
        "metrics": metrics, "confusion_matrix": confusion,
        "train_class_counts": train_counts, "validation_class_counts": validation_counts,
        "evaluation_plan": evaluation_plan,
        "random_seed": int(mapping.get("random_seed", 42)),
        "majority_baseline_included": True,
        "protocol_gate_status": "not_evaluated" if fit_status == "skipped" or metrics.get("auc_ovr_macro") is None else "see AUC results",
        "warnings": ["流程验证，非最终结论。", "样本量小，指标不可靠。", "单一 session/场景不能验证跨参与者/场景泛化。"],
        "reasons": reasons,
        "software": {"lightgbm": lightgbm.__version__, "scikit_learn": sklearn.__version__},
        "predictions_path": str(predictions_path),
    }
    write_json(metadata_path, metadata)
    write_report(report_path, metadata)
    print(f"fit_status={fit_status} windows={len(frame)} labeled={len(labeled)} classes={dict(counts)} features={len(feature_names)}")
    if fit_status == "trained":
        print(f"metrics={json.dumps(metrics, ensure_ascii=False)} model={model_path}")
    else:
        print("training_skipped=" + " ".join(reasons))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
