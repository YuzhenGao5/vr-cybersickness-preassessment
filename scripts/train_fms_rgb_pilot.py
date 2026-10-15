#!/usr/bin/env python3
"""Fixed, exploratory FMS regression with paired 3/30-second RGB windows.

Use only extraction status='ok' and finite raw FMS targets. The primary LOGO
evaluation uses the intersection of eligible events in both windows. Every
model uses the same training-label-range prediction limits. No hyperparameter
selection, target binning, subject validation, or threshold search is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SEED = 42
WINDOWS = (3, 30)
EVALUATIONS = ("primary_logo", "proxy_purged_logo")
N_BOOTSTRAP = 2000
METHODS = (
    "train_mean", "train_median", "elapsed_only_ridge", "rgb_only_ridge",
    "rgb_only_lightgbm", "rgb_plus_elapsed_ridge",
)
CONSTANT_METHODS = {"train_mean", "train_median"}
RGB_METHODS = ("rgb_only_ridge", "rgb_only_lightgbm", "rgb_plus_elapsed_ridge")
RIDGE_PARAMETERS = {"alpha": 10.0}
LGBM_PARAMETERS = {
    "objective": "regression", "n_estimators": 150, "max_depth": 3,
    "num_leaves": 7, "learning_rate": 0.05, "min_child_samples": 20,
    "reg_lambda": 5.0, "feature_fraction": 0.9, "verbosity": -1,
    "n_jobs": 2, "random_state": SEED,
}
REQUIRED = [
    "event_id", "session_id", "game_name", "window_sec", "event_time_sec",
    "fms_raw", "status", "elapsed_sec",
]
# Schema-level exclusion only. Missing-feature decisions are training-fold only.
FORBIDDEN_RGB = re.compile(
    r"(^|_)(id|path|game|session|subject|participant|event|timestamp|time|elapsed|"
    r"fms|ssq|score|rating|label|target|coverage|count|counts|support|duration)($|_)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "features/fms_rgb_event_features.csv")
    parser.add_argument("--oof", type=Path, default=ROOT / "features/fms_rgb_oof_predictions.csv")
    parser.add_argument("--metrics", type=Path, default=ROOT / "results/fms_rgb_metrics.json")
    parser.add_argument("--fold-metrics", type=Path, default=ROOT / "results/fms_rgb_fold_metrics.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "results/fms_rgb_training_report.md")
    parser.add_argument("--model", type=Path, default=ROOT / "models/fms_rgb_pilot.pkl")
    parser.add_argument("--log", type=Path, default=ROOT / "results/fms_rgb_training_log.txt")
    parser.add_argument("--definition", type=Path, default=ROOT / "audit/fms_rgb_training_definition.json")
    return parser.parse_args()


def setup_logging(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fms_rgb_pilot")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    for handler in (logging.FileHandler(path, mode="a", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logging.captureWarnings(True)
    warning_logger = logging.getLogger("py.warnings")
    warning_logger.handlers = logger.handlers.copy()
    warning_logger.setLevel(logging.WARNING)
    warning_logger.propagate = False
    return logger


def finite_numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def read_input(path: Path) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if len(header) != len(set(header)):
        raise ValueError("Input has duplicate CSV column names")
    data = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    missing = [name for name in REQUIRED if name not in data.columns]
    if missing:
        raise ValueError(f"Input missing required columns: {missing}")
    if data.empty:
        raise ValueError("Input has zero event/window rows")
    for name in ("event_id", "session_id", "game_name"):
        if data[name].isna().any() or data[name].astype(str).str.strip().eq("").any():
            raise ValueError(f"Input has blank/missing {name}")
        data[name] = data[name].astype(str).str.strip()
    data["window_sec"] = finite_numeric(data["window_sec"])
    if data["window_sec"].isna().any() or not data["window_sec"].isin(WINDOWS).all():
        raise ValueError("window_sec must contain only the prespecified windows 3 and 30")
    data["window_sec"] = data["window_sec"].astype(int)
    if set(data["window_sec"].unique()) != set(WINDOWS):
        raise ValueError("Both 3-second and 30-second window rows are required")
    if data.duplicated(["event_id", "window_sec"]).any():
        raise ValueError("Duplicate event_id/window_sec rows; cannot silently pool duplicate events")
    for name in ("session_id", "game_name"):
        if data.groupby("event_id")[name].nunique().gt(1).any():
            raise ValueError(f"An event_id maps to multiple {name} values")
    if data.groupby("session_id")["game_name"].nunique().gt(1).any():
        raise ValueError("A session_id maps to more than one game")
    data["status"] = data["status"].fillna("missing").astype(str).str.strip().str.lower()
    rgb_candidates = [name for name in data.columns if name.startswith("rgb_")]
    excluded_rgb = [name for name in rgb_candidates if FORBIDDEN_RGB.search(name)]
    rgb_features = [name for name in rgb_candidates if name not in excluded_rgb]
    if not rgb_features:
        raise ValueError("No allowed rgb_ feature columns remain after schema exclusions")
    numeric_columns = [*rgb_features, "elapsed_sec", "event_time_sec", "fms_raw"]
    invalid_values: dict[str, int] = {}
    for name in numeric_columns:
        converted = finite_numeric(data[name])
        invalid_values[name] = int((data[name].notna() & converted.isna()).sum())
        data[name] = converted
    data["eligible"] = data["status"].eq("ok") & data["fms_raw"].notna()
    valid_event_sets = {
        window: set(data.loc[data["window_sec"].eq(window) & data["eligible"], "event_id"])
        for window in WINDOWS
    }
    common_events = valid_event_sets[3] & valid_event_sets[30]
    data["primary_common_event"] = data["event_id"].isin(common_events) & data["eligible"]
    # The same target and metadata must be preserved across window comparisons.
    paired = data.loc[data["primary_common_event"]]
    for name in ("fms_raw", "event_time_sec", "elapsed_sec"):
        pivot = paired.pivot(index="event_id", columns="window_sec", values=name)
        if len(pivot) and not np.allclose(pivot[3], pivot[30], rtol=0, atol=1e-6, equal_nan=True):
            raise ValueError(f"Paired windows disagree on {name}; fix alignment before training")
    data = data.sort_values(["window_sec", "game_name", "session_id", "event_time_sec", "event_id"]).reset_index(drop=True)
    by_window: dict[str, Any] = {}
    by_game: list[dict[str, Any]] = []
    for window in WINDOWS:
        rows = data.loc[data["window_sec"].eq(window)]
        eligible = rows.loc[rows["eligible"]]
        primary = rows.loc[rows["primary_common_event"]]
        by_window[str(window)] = {
            "n_input_rows": int(len(rows)), "n_status_ok": int(rows["status"].eq("ok").sum()),
            "n_status_ok_invalid_target": int((rows["status"].eq("ok") & rows["fms_raw"].isna()).sum()),
            "n_eligible": int(len(eligible)), "n_common_events": int(len(primary)),
            "eligible_coverage_of_input": float(len(eligible) / len(rows)) if len(rows) else None,
            "primary_coverage_of_eligible": float(len(primary) / len(eligible)) if len(eligible) else None,
            "status_counts": {str(k): int(v) for k, v in rows["status"].value_counts().items()},
            "n_eligible_games": int(eligible["game_name"].nunique()),
            "n_eligible_sessions": int(eligible["session_id"].nunique()),
            "n_eligible_rows_all_rgb_missing": int(eligible[rgb_features].isna().all(axis=1).sum()),
        }
        for game, group in rows.groupby("game_name", sort=True):
            eligible_group = group.loc[group["eligible"]]
            primary_group = group.loc[group["primary_common_event"]]
            y = primary_group["fms_raw"].to_numpy(dtype=float)
            by_game.append({
                "window_sec": window, "game_name": str(game), "n_input": int(len(group)),
                "n_eligible": int(len(eligible_group)), "n_common": int(len(primary_group)),
                "n_common_sessions": int(primary_group["session_id"].nunique()),
                "fms_min": float(np.min(y)) if len(y) else None,
                "fms_max": float(np.max(y)) if len(y) else None,
                "fms_mean": float(np.mean(y)) if len(y) else None,
                "fms_median": float(np.median(y)) if len(y) else None,
                "fms_zero_fraction": float(np.mean(y == 0)) if len(y) else None,
                "fms_value_counts": {str(k): int(v) for k, v in primary_group["fms_raw"].value_counts().sort_index().items()},
            })
    schema = {
        "rgb_feature_names": rgb_features,
        "excluded_rgb_columns": excluded_rgb,
        "excluded_non_rgb_columns": [name for name in header if name not in rgb_features and name != "elapsed_sec"],
        "invalid_numeric_values": invalid_values,
        "coverage": {
            "n_total_events": int(data["event_id"].nunique()), "n_total_rows": int(len(data)),
            "n_primary_common_events": len(common_events), "by_window": by_window, "by_game": by_game,
        },
    }
    return data, rgb_features, schema


def features_for(method: str, rgb_features: list[str]) -> list[str]:
    if method in CONSTANT_METHODS:
        return []
    if method == "elapsed_only_ridge":
        return ["elapsed_sec"]
    if method == "rgb_plus_elapsed_ridge":
        return [*rgb_features, "elapsed_sec"]
    return rgb_features.copy()


def fit_model(method: str, data: pd.DataFrame, rgb_features: list[str]) -> dict[str, Any]:
    y = data["fms_raw"].to_numpy(dtype=float)
    if not len(y):
        raise ValueError("Cannot fit a model with zero known training targets")
    requested = features_for(method, rgb_features)
    dropped = [name for name in requested if data[name].isna().all()]
    retained = [name for name in requested if name not in dropped]
    imputer: SimpleImputer | None = None
    scaler: StandardScaler | None = None
    if method in CONSTANT_METHODS or not retained:
        strategy = "median" if method == "train_median" else "mean"
        estimator: Any = DummyRegressor(strategy=strategy)
        X_train = np.zeros((len(y), 1), dtype=float)
        kind = f"DummyRegressor_{strategy}"
        status = "fitted_constant_baseline" if method in CONSTANT_METHODS else "fallback_no_nonmissing_train_features"
        input_mode = "dummy_zero_column"
    else:
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(data[retained])
        if method == "rgb_only_lightgbm":
            estimator = lgb.LGBMRegressor(**LGBM_PARAMETERS)
            kind = "LightGBMRegressor"
        else:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            estimator = Ridge(**RIDGE_PARAMETERS)
            kind = "Ridge"
        status = "fitted"
        input_mode = "imputed_retained_features"
    estimator.fit(X_train, y)
    return {
        "estimator": estimator, "imputer": imputer, "scaler": scaler,
        "metadata": {
            "method": method, "model_kind": kind, "fit_status": status, "input_mode": input_mode,
            "requested_features": requested, "retained_features": retained,
            "dropped_all_missing_train_features": dropped,
            "n_train": int(len(data)), "n_train_games": int(data["game_name"].nunique()),
            "n_train_sessions": int(data["session_id"].nunique()),
            "n_train_rows_all_requested_features_missing": int(data[requested].isna().all(axis=1).sum()) if requested else None,
            "train_fms_min": float(np.min(y)), "train_fms_max": float(np.max(y)),
            "train_fms_mean": float(np.mean(y)), "train_fms_median": float(np.median(y)),
            "train_fms_unique_values": int(len(np.unique(y))),
            "prediction_limit_policy": "clip to this training set's raw FMS minimum/maximum",
        },
    }


def predict_model(fitted: dict[str, Any], data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    metadata = fitted["metadata"]
    if metadata["input_mode"] == "dummy_zero_column":
        transformed = np.zeros((len(data), 1), dtype=float)
    else:
        transformed = fitted["imputer"].transform(data[metadata["retained_features"]])
        if fitted["scaler"] is not None:
            transformed = fitted["scaler"].transform(transformed)
    raw = np.asarray(fitted["estimator"].predict(transformed), dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError(f"{metadata['method']} produced nonfinite predictions")
    limited = np.clip(raw, metadata["train_fms_min"], metadata["train_fms_max"])
    return raw, limited


def regression_metrics(y: np.ndarray, prediction: np.ndarray, force_constant_na: bool = False) -> dict[str, Any]:
    if not len(y):
        return {"n": 0, "mae": None, "rmse": None, "spearman": None, "spearman_status": "no_predictions"}
    error = prediction - y
    if force_constant_na:
        correlation, reason = None, "NA_constant_baseline"
    elif len(y) < 2:
        correlation, reason = None, "NA_fewer_than_two_rows"
    elif len(np.unique(y)) < 2:
        correlation, reason = None, "NA_constant_target"
    elif len(np.unique(prediction)) < 2:
        correlation, reason = None, "NA_constant_prediction"
    else:
        coefficient = float(spearmanr(y, prediction).statistic)
        correlation = coefficient if math.isfinite(coefficient) else None
        reason = "descriptive_no_p_value" if correlation is not None else "NA_nonfinite"
    return {"n": int(len(y)), "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))), "spearman": correlation, "spearman_status": reason}


def macro_metrics(folds: list[dict[str, Any]], view: str = "metrics") -> dict[str, Any]:
    result: dict[str, Any] = {"n_evaluated_games": sum(fold["status"] == "evaluated" for fold in folds)}
    for key in ("mae", "rmse", "spearman"):
        values = [fold[view][key] for fold in folds if fold[view][key] is not None]
        result[key] = float(np.mean(values)) if values else None
        result[f"n_games_with_defined_{key}"] = len(values)
    return result


def evaluate_window(
    data: pd.DataFrame, window: int, rgb_features: list[str], logger: logging.Logger,
    evaluation: str, fit_demo: bool,
) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    rows = data.loc[data["window_sec"].eq(window) & data["primary_common_event"]].copy()
    games = sorted(rows["game_name"].unique())
    prediction_rows: list[dict[str, Any]] = []
    fold_csv_rows: list[dict[str, Any]] = []
    by_method: dict[str, Any] = {}
    fitted_models: dict[str, Any] = {}
    for method in METHODS:
        folds: list[dict[str, Any]] = []
        logger.info("evaluation=%s window=%s method=%s primary_common_events=%s games=%s", evaluation, window, method, len(rows), len(games))
        for fold_number, held_out in enumerate(games, 1):
            train = rows.loc[rows["game_name"].ne(held_out)]
            test = rows.loc[rows["game_name"].eq(held_out)]
            train_before_purge = train
            test_prefixes = sorted(set(test["session_id"].str.split("_", n=1).str[0]))
            if evaluation == "proxy_purged_logo":
                overlap = train["session_id"].str.split("_", n=1).str[0].isin(test_prefixes)
                train = train.loc[~overlap]
            y = test["fms_raw"].to_numpy(dtype=float)
            fold: dict[str, Any] = {
                "evaluation": evaluation, "fold": fold_number, "held_out_game": str(held_out), "n_train": int(len(train)),
                "n_train_before_purge": int(len(train_before_purge)),
                "n_purged_train_events": int(len(train_before_purge) - len(train)),
                "n_purged_train_sessions": int(train_before_purge["session_id"].nunique() - train["session_id"].nunique()),
                "test_session_prefixes": test_prefixes,
                "n_test": int(len(test)), "n_train_games": int(train["game_name"].nunique()),
                "n_train_sessions": int(train["session_id"].nunique()),
                "n_test_sessions": int(test["session_id"].nunique()), "fit": None,
                "metrics": regression_metrics(np.array([]), np.array([])),
                "raw_metrics": regression_metrics(np.array([]), np.array([])), "n_clipped": 0,
            }
            if train.empty:
                fold["status"] = "not_evaluable_no_training_game"
            else:
                fitted = fit_model(method, train, rgb_features)
                raw, limited = predict_model(fitted, test)
                fold["fit"] = fitted["metadata"]
                fold["status"] = "evaluated"
                fold["metrics"] = regression_metrics(y, limited, method in CONSTANT_METHODS)
                fold["raw_metrics"] = regression_metrics(y, raw, method in CONSTANT_METHODS)
                fold["n_clipped"] = int(np.sum(raw != limited))
                for position, event in enumerate(test.to_dict("records")):
                    prediction_rows.append({
                        "evaluation": evaluation, "analysis_scope": "paired_common_events", "window_sec": window, "method": method,
                        "event_id": event["event_id"], "session_id": event["session_id"], "game_name": event["game_name"],
                        "event_time_sec": event["event_time_sec"], "elapsed_sec": event["elapsed_sec"], "fms_raw": event["fms_raw"],
                        "fold": fold_number, "held_out_game": str(held_out), "prediction_raw": float(raw[position]),
                        "prediction": float(limited[position]), "was_clipped": bool(raw[position] != limited[position]),
                        "train_fms_min": fitted["metadata"]["train_fms_min"], "train_fms_max": fitted["metadata"]["train_fms_max"],
                        "n_train": len(train), "n_retained_features": len(fitted["metadata"]["retained_features"]),
                        "n_purged_train_events": fold["n_purged_train_events"],
                        "n_purged_train_sessions": fold["n_purged_train_sessions"],
                        "model_kind": fitted["metadata"]["model_kind"], "fit_status": fitted["metadata"]["fit_status"],
                    })
            folds.append(fold)
            logger.info("evaluation=%s window=%s method=%s held_out=%s status=%s train=%s test=%s purged=%s MAE=%s RMSE=%s rho=%s",
                        evaluation, window, method, held_out, fold["status"], len(train), len(test), fold["n_purged_train_events"],
                        fold["metrics"]["mae"], fold["metrics"]["rmse"], fold["metrics"]["spearman"])
            fold_csv_rows.append({
                "evaluation": evaluation, "window_sec": window, "method": method, "fold": fold_number, "held_out_game": str(held_out),
                "status": fold["status"], "n_train": fold["n_train"], "n_test": fold["n_test"],
                "n_train_before_purge": fold["n_train_before_purge"],
                "n_purged_train_events": fold["n_purged_train_events"],
                "n_purged_train_sessions": fold["n_purged_train_sessions"],
                "test_session_prefixes": json.dumps(test_prefixes, ensure_ascii=False),
                "n_train_games": fold["n_train_games"], "n_train_sessions": fold["n_train_sessions"],
                "n_test_sessions": fold["n_test_sessions"], **fold["metrics"],
                "raw_mae": fold["raw_metrics"]["mae"], "raw_rmse": fold["raw_metrics"]["rmse"],
                "raw_spearman": fold["raw_metrics"]["spearman"], "n_clipped": fold["n_clipped"],
                "train_fms_min": fold["fit"]["train_fms_min"] if fold["fit"] else None,
                "train_fms_max": fold["fit"]["train_fms_max"] if fold["fit"] else None,
                "model_kind": fold["fit"]["model_kind"] if fold["fit"] else None,
                "fit_status": fold["fit"]["fit_status"] if fold["fit"] else None,
                "retained_features": json.dumps(fold["fit"]["retained_features"], ensure_ascii=False) if fold["fit"] else "[]",
                "dropped_all_missing_train_features": json.dumps(fold["fit"]["dropped_all_missing_train_features"], ensure_ascii=False) if fold["fit"] else "[]",
            })
        method_predictions = [row for row in prediction_rows if row["method"] == method]
        y_oof = np.asarray([row["fms_raw"] for row in method_predictions], dtype=float)
        pred_oof = np.asarray([row["prediction"] for row in method_predictions], dtype=float)
        raw_oof = np.asarray([row["prediction_raw"] for row in method_predictions], dtype=float)
        only_fold_constants = bool(folds) and all(
            fold["metrics"]["spearman_status"] in {"NA_constant_baseline", "NA_constant_prediction", "no_predictions"}
            for fold in folds
        )
        by_method[method] = {
            "feature_names": features_for(method, rgb_features), "folds": folds,
            "pooled_oof": regression_metrics(y_oof, pred_oof, method in CONSTANT_METHODS or only_fold_constants),
            "macro_game": macro_metrics(folds), "raw_pooled_oof": regression_metrics(y_oof, raw_oof, method in CONSTANT_METHODS),
            "raw_macro_game": macro_metrics(folds, "raw_metrics"),
            "n_clipped_oof": sum(fold["n_clipped"] for fold in folds),
            "oof_coverage_of_common_events": float(len(method_predictions) / len(rows)) if len(rows) else None,
        }
        if fit_demo:
            final_train = data.loc[data["window_sec"].eq(window) & data["eligible"]]
            if final_train.empty:
                fitted_models[method] = None
                by_method[method]["final_demo_fit"] = {"fit_status": "infeasible_no_eligible_events", "n_train": 0}
            else:
                final = fit_model(method, final_train, rgb_features)
                fitted_models[method] = final
                by_method[method]["final_demo_fit"] = final["metadata"]
    oof_columns = [
        "evaluation", "analysis_scope", "window_sec", "method", "event_id", "session_id", "game_name",
        "event_time_sec", "elapsed_sec", "fms_raw", "fold", "held_out_game", "prediction_raw",
        "prediction", "was_clipped", "train_fms_min", "train_fms_max", "n_train",
        "n_retained_features", "n_purged_train_events", "n_purged_train_sessions", "model_kind", "fit_status",
    ]
    return {"n_common_events": int(len(rows)), "n_common_games": len(games), "models": by_method}, pd.DataFrame(prediction_rows, columns=oof_columns), fold_csv_rows, fitted_models


def paired_comparison(model: pd.DataFrame, mae_reference: pd.DataFrame, rmse_reference: pd.DataFrame) -> dict[str, Any]:
    keys = ["event_id", "game_name"]
    model = model[keys + ["fms_raw", "prediction"]].rename(columns={"prediction": "model_prediction"})
    mae_reference = mae_reference[keys + ["fms_raw", "prediction"]].rename(columns={"fms_raw": "mae_y", "prediction": "mae_reference_prediction"})
    rmse_reference = rmse_reference[keys + ["fms_raw", "prediction"]].rename(columns={"fms_raw": "rmse_y", "prediction": "rmse_reference_prediction"})
    paired = model.merge(mae_reference, on=keys, validate="one_to_one").merge(rmse_reference, on=keys, validate="one_to_one")
    if len(paired) != len(model) or len(paired) != len(mae_reference) or len(paired) != len(rmse_reference):
        raise ValueError("Baseline/model OOF supports differ; paired comparisons require exactly the same events")
    if paired.empty:
        return {"n_events": 0, "n_games": 0, "status": "no_paired_oof", "pooled": None, "macro_game": None, "bootstrap": None}
    if not np.allclose(paired["fms_raw"], paired["mae_y"]) or not np.allclose(paired["fms_raw"], paired["rmse_y"]):
        raise ValueError("Paired comparison targets disagree")
    game_statistics: list[tuple[float, ...]] = []
    by_game: list[dict[str, Any]] = []
    for game, group in paired.groupby("game_name", sort=True):
        y = group["fms_raw"].to_numpy(dtype=float)
        model_error = group["model_prediction"].to_numpy(dtype=float) - y
        mae_error = group["mae_reference_prediction"].to_numpy(dtype=float) - y
        rmse_error = group["rmse_reference_prediction"].to_numpy(dtype=float) - y
        game_statistics.append((len(group), np.abs(model_error).sum(), np.abs(mae_error).sum(),
                                np.square(model_error).sum(), np.square(rmse_error).sum()))
        model_mae, reference_mae = float(np.abs(model_error).mean()), float(np.abs(mae_error).mean())
        model_rmse = float(np.sqrt(np.square(model_error).mean()))
        reference_rmse = float(np.sqrt(np.square(rmse_error).mean()))
        by_game.append({"game_name": str(game), "n_events": int(len(group)),
                        "model_mae": model_mae, "reference_mae": reference_mae,
                        "model_rmse": model_rmse, "reference_rmse": reference_rmse,
                        "delta_mae": model_mae - reference_mae,
                        "delta_rmse": model_rmse - reference_rmse})
    statistics = np.asarray(game_statistics, dtype=float)
    n = statistics[:, 0]
    pooled_model_mae = float(statistics[:, 1].sum() / n.sum())
    pooled_reference_mae = float(statistics[:, 2].sum() / n.sum())
    pooled_model_rmse = float(np.sqrt(statistics[:, 3].sum() / n.sum()))
    pooled_reference_rmse = float(np.sqrt(statistics[:, 4].sum() / n.sum()))
    mae_game_delta = (statistics[:, 1] - statistics[:, 2]) / n
    rmse_game_delta = np.sqrt(statistics[:, 3] / n) - np.sqrt(statistics[:, 4] / n)
    result: dict[str, Any] = {
        "status": "paired", "n_events": int(len(paired)), "n_games": len(statistics),
        "difference_direction": "model minus reference; negative means lower error",
        "pooled": {"delta_mae": pooled_model_mae - pooled_reference_mae,
                   "delta_rmse": pooled_model_rmse - pooled_reference_rmse,
                   "mae_relative_reduction": (pooled_reference_mae - pooled_model_mae) / pooled_reference_mae if pooled_reference_mae else None,
                   "rmse_relative_reduction": (pooled_reference_rmse - pooled_model_rmse) / pooled_reference_rmse if pooled_reference_rmse else None},
        "macro_game": {"delta_mae": float(mae_game_delta.mean()), "delta_rmse": float(rmse_game_delta.mean())},
        "by_game": by_game,
    }
    if len(statistics) < 2:
        result["bootstrap"] = {"status": "not_estimable_fewer_than_two_game_clusters", "n_resamples": 0}
        return result
    # Each selected game contributes every event in that game, including repeats
    # when that game is selected again. Never resample individual events.
    rng = np.random.default_rng(SEED)
    selection = rng.integers(0, len(statistics), size=(N_BOOTSTRAP, len(statistics)))
    sampled = statistics[selection]
    totals = sampled.sum(axis=1)
    bootstrap_values = {
        "pooled_delta_mae": (totals[:, 1] - totals[:, 2]) / totals[:, 0],
        "pooled_delta_rmse": np.sqrt(totals[:, 3] / totals[:, 0]) - np.sqrt(totals[:, 4] / totals[:, 0]),
        "macro_game_delta_mae": mae_game_delta[selection].mean(axis=1),
        "macro_game_delta_rmse": rmse_game_delta[selection].mean(axis=1),
    }
    result["bootstrap"] = {
        "status": "game_cluster_percentile_interval_exploratory", "n_resamples": N_BOOTSTRAP,
        "seed": SEED, "unit": "entire game", "confidence_level": 0.95,
        "retraining_performed": False,
        **{f"{name}_ci95": [float(v) for v in np.quantile(values, [0.025, 0.975])]
           for name, values in bootstrap_values.items()},
    }
    return result


def add_comparisons(windows: dict[str, Any], oof: pd.DataFrame) -> dict[str, Any]:
    for window in WINDOWS:
        for evaluation in EVALUATIONS:
            rows = oof.loc[oof["window_sec"].eq(window) & oof["evaluation"].eq(evaluation)]
            median = rows.loc[rows["method"].eq("train_median")]
            mean = rows.loc[rows["method"].eq("train_mean")]
            elapsed = rows.loc[rows["method"].eq("elapsed_only_ridge")]
            for method in METHODS:
                model = rows.loc[rows["method"].eq(method)]
                entry = windows[str(window)]["evaluations"][evaluation]["models"][method]
                entry["versus_train_baselines"] = paired_comparison(model, median, mean)
                if method in RGB_METHODS:
                    entry["versus_elapsed_only"] = paired_comparison(model, elapsed, elapsed)
    window_comparisons: dict[str, Any] = {}
    for evaluation in EVALUATIONS:
        window_comparisons[evaluation] = {}
        for method in METHODS:
            model = oof.loc[oof["window_sec"].eq(30) & oof["method"].eq(method) & oof["evaluation"].eq(evaluation)]
            reference = oof.loc[oof["window_sec"].eq(3) & oof["method"].eq(method) & oof["evaluation"].eq(evaluation)]
            window_comparisons[evaluation][method] = paired_comparison(model, reference, reference)
            window_comparisons[evaluation][method]["difference_direction"] = "30-second minus 3-second error; negative means lower error for 30 seconds"
    return window_comparisons


def fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def interval_text(comparison: dict[str, Any], metric: str, macro: bool = False) -> str:
    bootstrap = comparison.get("bootstrap") or {}
    prefix = "macro_game" if macro else "pooled"
    interval = bootstrap.get(f"{prefix}_delta_{metric}_ci95")
    return "NA" if interval is None else f"[{fmt(interval[0])}, {fmt(interval[1])}]"


def delta_value(comparison: dict[str, Any], metric: str, macro: bool = False) -> float | None:
    metrics = comparison.get("macro_game" if macro else "pooled") or {}
    return metrics.get(f"delta_{metric}")


def report_lines(summary: dict[str, Any], args: argparse.Namespace) -> list[str]:
    coverage = summary["schema"]["coverage"]
    lines = [
        "# FMS：RGB 事件窗口探索性回归试点", "",
        "用户授权的原始 FMS 回归探索。它位于原 PROTOCOL 的分类和 subject 门控之外；参与者身份尚未认证，本次做 Leave-One-Game-Out 及前缀清除敏感性评估。", "",
        "## 固定设置与样本", "",
        f"- 输入：`{args.input}`；SHA256：`{summary['input_sha256']}`。",
        f"- 训练前固定定义：`{args.definition}`；SHA256：`{summary['training_definition_sha256']}`。固定模型、窗口和敏感性设置均在首个 fit 之前保存。",
        f"- 全量事件：{coverage['n_total_events']}；event/window 行：{coverage['n_total_rows']}；3/30 秒共同有效事件：{coverage['n_primary_common_events']}。",
        "- 仅 status=ok 且 fms_raw 有限的行可用于训练；FMS 保持原始数值，不分箱、不移动标签、不搜索阈值。",
        "- 两窗口主对照只使用共同有效 event_id，并检查 game/session、FMS、事件时刻、elapsed 在两个窗口一致。各窗口全量有效样本仅用于最终演示模型拟合，不报告其训练集性能。",
        "- `primary_logo`：完整的其他游戏共同事件作训练。`proxy_purged_logo`：先留出游戏，再从训练中清除 `session_id.split('_')[0]` 与测试重叠的全部事件；测试集保持一致，全部模型执行同样清除。",
        "- 数字前缀只是保守 identity proxy，未认证为参与者。敏感性评估不称 subject LOSO；其结果与主结果一并报告。",
        f"- seed={SEED}；Ridge 参数 `{RIDGE_PARAMETERS}`；LightGBM 参数 `{LGBM_PARAMETERS}`；无调参和 early stopping。",
        "- 模型输入只含允许的 rgb_ 数值列及 elapsed-only/组合模型明确指定的 elapsed_sec。ID/game/session、FMS/score、coverage/count、路径、绝对事件时刻和最终时长均不进入模型。",
        "- 每一训练折独立剔除全缺失输入列、拟合 median imputer；Ridge 另拟合 StandardScaler。全缺失输入无可拟合特征时使用训练均值 DummyRegressor 并明确标记 fallback。",
        "- 所有模型统一将预测限制到当前训练标签 min/max。表中主指标使用限制后的预测；原始预测及其指标保存在 OOF、JSON 和 fold CSV。",
        "- macro-game MAE/RMSE 为各游戏 MAE/RMSE 的等权平均；macro-game Spearman 仅平均有定义游戏并报告个数。Spearman 为描述性相关，不报告事件独立 p 值。",
        "- 常数基线的 Spearman 记 NA；即使不同 LOGO 折的常数不同，也不把这种游戏先验差异计算为基线 pooled 相关。",
        f"- RGB feature list：`{summary['schema']['rgb_feature_names']}`。",
        f"- 排除的 rgb_ 元数据/疑似泄漏列：`{summary['schema']['excluded_rgb_columns']}`。",
        f"- 非数值或无限值转换为缺失的计数：`{summary['schema']['invalid_numeric_values']}`。", "",
        "### 覆盖", "",
        "| 窗口秒 | 输入行 | status ok | 有效目标 | 共同事件 | 有效/输入 | 共同/有效 | 有效 games/sessions |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for window in WINDOWS:
        c = coverage["by_window"][str(window)]
        lines.append(f"| {window} | {c['n_input_rows']} | {c['n_status_ok']} | {c['n_eligible']} | {c['n_common_events']} | {fmt(c['eligible_coverage_of_input'])} | {fmt(c['primary_coverage_of_eligible'])} | {c['n_eligible_games']}/{c['n_eligible_sessions']} |")
    lines.extend(["", *[f"- {window} 秒提取状态：`{coverage['by_window'][str(window)]['status_counts']}`。" for window in WINDOWS],
                  "", "| 游戏 | 窗口秒 | 输入/有效/共同 | 共同 sessions | FMS min/max | FMS mean/median | FMS 值频数 |",
                  "|---|---:|---|---:|---|---|---|"])
    for c in coverage["by_game"]:
        lines.append(f"| {fmt(c['game_name'])} | {c['window_sec']} | {c['n_input']}/{c['n_eligible']}/{c['n_common']} | {c['n_common_sessions']} | {fmt(c['fms_min'])}/{fmt(c['fms_max'])} | {fmt(c['fms_mean'])}/{fmt(c['fms_median'])} | {fmt(c['fms_value_counts'])} |")
    for window, evaluation in [(w, e) for w in WINDOWS for e in EVALUATIONS]:
        entry = summary["windows"][str(window)]["evaluations"][evaluation]
        lines.extend(["", f"## {window} 秒窗口 / {evaluation}：全部模型", "",
                      "| 模型 | OOF n | pooled MAE | pooled RMSE | pooled Spearman | macro-game MAE | macro-game RMSE | macro-game Spearman (定义游戏数) | 限制预测数 |",
                      "|---|---:|---:|---:|---:|---:|---:|---|---:|"])
        for method in METHODS:
            model = entry["models"][method]
            pooled, macro = model["pooled_oof"], model["macro_game"]
            lines.append(f"| {method} | {pooled['n']} | {fmt(pooled['mae'])} | {fmt(pooled['rmse'])} | {fmt(pooled['spearman'])} | {fmt(macro['mae'])} | {fmt(macro['rmse'])} | {fmt(macro['spearman'])} ({macro['n_games_with_defined_spearman']}) | {model['n_clipped_oof']} |")
        lines.extend(["", "### 相对训练基线的配对差值", "",
                      "ΔMAE 相对 train_median；ΔRMSE 相对 train_mean；均为 model−reference，负值代表误差更小。括号为整游戏聚类 95% bootstrap 区间。", "",
                      "| 模型 | pooled ΔMAE [CI] | pooled ΔRMSE [CI] | macro-game ΔMAE [CI] | macro-game ΔRMSE [CI] |",
                      "|---|---|---|---|---|"])
        for method in METHODS:
            c = entry["models"][method]["versus_train_baselines"]
            lines.append(f"| {method} | {fmt(delta_value(c, 'mae'))} {interval_text(c, 'mae')} | {fmt(delta_value(c, 'rmse'))} {interval_text(c, 'rmse')} | {fmt(delta_value(c, 'mae', True))} {interval_text(c, 'mae', True)} | {fmt(delta_value(c, 'rmse', True))} {interval_text(c, 'rmse', True)} |")
        lines.extend(["", "### RGB 模型相对 elapsed-only", "",
                      "这是额外的预定时间混杂诊断；两项差值均相对 elapsed_only_ridge。", "",
                      "| 模型 | pooled ΔMAE [CI] | pooled ΔRMSE [CI] |",
                      "|---|---|---|"])
        for method in RGB_METHODS:
            c = entry["models"][method]["versus_elapsed_only"]
            lines.append(f"| {method} | {fmt(delta_value(c, 'mae'))} {interval_text(c, 'mae')} | {fmt(delta_value(c, 'rmse'))} {interval_text(c, 'rmse')} |")
        lines.extend(["", "### 全部游戏折", "",
                      "| 模型 | 留出游戏 | train/test | train/test sessions | 清除 events/sessions | 状态 | MAE | RMSE | Spearman | 限制数 |",
                      "|---|---|---|---|---|---|---:|---:|---:|---:|"])
        for method in METHODS:
            for fold in entry["models"][method]["folds"]:
                m = fold["metrics"]
                status = fold["fit"]["fit_status"] if fold["fit"] else fold["status"]
                lines.append(f"| {method} | {fmt(fold['held_out_game'])} | {fold['n_train']}/{fold['n_test']} | {fold['n_train_sessions']}/{fold['n_test_sessions']} | {fold['n_purged_train_events']}/{fold['n_purged_train_sessions']} | {status} | {fmt(m['mae'])} | {fmt(m['rmse'])} | {fmt(m['spearman'])} | {fold['n_clipped']} |")
        lines.extend(["", "### RGB 相对基线：每游戏改善", "",
                      "ΔMAE 相对该折 train_median；ΔRMSE 相对该折 train_mean。负值为误差降低。", "",
                      "| 模型 | 游戏 | n | ΔMAE | ΔRMSE |", "|---|---|---:|---:|---:|"])
        for method in RGB_METHODS:
            for game in entry["models"][method]["versus_train_baselines"].get("by_game", []):
                lines.append(f"| {method} | {fmt(game['game_name'])} | {game['n_events']} | {fmt(game['delta_mae'])} | {fmt(game['delta_rmse'])} |")
    for window in WINDOWS:
        lines.extend(["", f"## {window} 秒全量有效行上的演示模型", ""])
        for method in METHODS:
            fit = summary["windows"][str(window)]["final_demo_fits"][method]
            lines.append(f"- `{method}`：`{fit['fit_status']}`；n={fit['n_train']}；保留列 `{fit.get('retained_features', [])}`；剔除全缺失列 `{fit.get('dropped_all_missing_train_features', [])}`。")
    lines.extend(["", "## 两窗口共同事件的配对对照", "",
                  "差值为 30 秒−3 秒，负值表示 30 秒误差更小。每个模型使用相同事件，不挑选最佳窗口作为最终结论。", "",
                  "| 评估 | 模型 | pooled ΔMAE [CI] | pooled ΔRMSE [CI] | macro-game ΔMAE | macro-game ΔRMSE |",
                  "|---|---|---|---|---:|---:|"])
    for evaluation, method in [(e, m) for e in EVALUATIONS for m in METHODS]:
        c = summary["window_comparisons"][evaluation][method]
        lines.append(f"| {evaluation} | {method} | {fmt(delta_value(c, 'mae'))} {interval_text(c, 'mae')} | {fmt(delta_value(c, 'rmse'))} {interval_text(c, 'rmse')} | {fmt(delta_value(c, 'mae', True))} | {fmt(delta_value(c, 'rmse', True))} |")
    lines.extend(["", "## 解释与交付", "",
                  f"- 游戏聚类 bootstrap 固定 {N_BOOTSTRAP} 次，seed={SEED}；每次抽取整个游戏并保留其全部事件。游戏仅约六个时区间解释有限，不能把许多相关事件当成独立证据。",
                  "- 区间只描述既有 OOF 误差的游戏重抽样差值；没有 bootstrap 重训，不涵盖重训、参与者、新游戏族等全部不确定性。不报告事件独立显著性检验。",
                  "- 共同有效事件筛选可能改变 FMS 与游戏分布；覆盖表保留筛选范围，结果仅适用于已成功提取的共同事件。",
                  "- session/participant 身份未认证，LOGO 不能替代 subject LOSO。来自同一录制的事件相关，且训练折彼此共享部分训练游戏。",
                  "- 前缀清除只排除已观察到的代理重叠，无法发现不同前缀代表同一参与者；清除减少训练数据，也可能改变训练分布。它不能证明身份泛化。",
                  "- elapsed-only 用于检查记录经过时间的预测贡献。RGB 优于常数基线而未优于 elapsed-only 时，不能据此声称视觉信息提供额外作用。",
                  "- FMS 原始量表具有有序性质，本次 MAE/RMSE 回归将数值距离作为探索目标，不证明因果机制或修改建议有效。",
                  "- 模型 bundle 为演示用途；部署、身份泛化与 Unity 场景迁移尚未验证。所有模型、所有窗口均报告，无 AUC/分类分箱/SHAP。",
                  f"- OOF：`{args.oof}`；完整指标/设置：`{args.metrics}`；各折：`{args.fold_metrics}`；模型：`{args.model}`；日志：`{args.log}`。",
                  "- bundle.windows[窗口秒字符串][方法] 保存 estimator/imputer/scaler/metadata；推理按 retained_features 中位数填补、可选缩放，再预测并限制到该模型 metadata 的 train_fms_min/max。常数模型使用任意 n×1 零矩阵。", ""])
    return lines


def run(args: argparse.Namespace, logger: logging.Logger) -> None:
    started = time.perf_counter()
    command = subprocess.list2cmdline([sys.executable, *sys.argv])
    logger.info("COMMAND %s", command)
    logger.info("WORKDIR %s", Path.cwd())
    logger.info("START input=%s seed=%s windows=%s evaluations=%s", args.input, SEED, WINDOWS, EVALUATIONS)
    data, rgb_features, schema = read_input(args.input)
    logger.info("rows=%s events=%s paired_common_events=%s rgb_features=%s excluded_rgb=%s",
                len(data), data["event_id"].nunique(), schema["coverage"]["n_primary_common_events"], len(rgb_features), schema["excluded_rgb_columns"])
    source_definition_path = ROOT / "audit/fms_rgb_pilot_definition.json"
    source_definition: dict[str, Any] | None = None
    if source_definition_path.is_file():
        source_definition = {
            "path": str(source_definition_path.resolve()),
            "sha256": hashlib.sha256(source_definition_path.read_bytes()).hexdigest(),
            "content": json.loads(source_definition_path.read_text(encoding="utf-8-sig")),
        }
    definition = {
        "created_utc_before_any_model_fit": datetime.now(timezone.utc).isoformat(),
        "input_path": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "training_script_path": str(Path(__file__).resolve()),
        "training_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "command": command, "source_extraction_definition": source_definition,
        "target": "original fms_raw unchanged; authorized exploratory regression outside original classification/subject gates",
        "eligible": "status=ok and finite fms_raw", "window_seconds": list(WINDOWS),
        "evaluation_events": "intersection of eligible event IDs across 3/30-second windows",
        "evaluations": {
            "primary_logo": "Leave-One-Game-Out; all other-game common events in training",
            "proxy_purged_logo": "Leave-One-Game-Out; remove all training events with session_id.split('_')[0] in held-out test prefix set; test support unchanged",
        },
        "identity_proxy": {"expression": "session_id.split('_')[0]", "subject_identity_verified": False,
                           "not_subject_loso": True, "apply_to_all_models": True},
        "random_seed": SEED, "model_names": list(METHODS),
        "feature_lists": {method: features_for(method, rgb_features) for method in METHODS},
        "hyperparameters": {"ridge": RIDGE_PARAMETERS, "lightgbm": LGBM_PARAMETERS},
        "preprocessing": "train-fold allmissing-column drop and median imputation; Ridge train-fold StandardScaler",
        "prediction_limits": "training-label min/max for every method",
        "comparisons": "MAE versus train_median; RMSE versus train_mean; RGB versus elapsed_only; 30sec versus 3sec",
        "bootstrap": {"seed": SEED, "resamples": N_BOOTSTRAP, "unit": "whole game", "paired": True,
                      "confidence_level": 0.95, "retraining": False, "event_independent_ci": False},
        "tuning": False, "classification_auc": False, "individual_p_values": False,
    }
    args.definition.parent.mkdir(parents=True, exist_ok=True)
    args.definition.write_text(json.dumps(definition, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    definition_sha256 = hashlib.sha256(args.definition.read_bytes()).hexdigest()
    logger.info("PRE_FIT_DEFINITION_SAVED path=%s sha256=%s", args.definition, definition_sha256)
    window_results: dict[str, Any] = {}
    fitted: dict[str, Any] = {}
    oof_tables: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    for window in WINDOWS:
        evaluations: dict[str, Any] = {}
        final_demo_fits: dict[str, Any] = {}
        for evaluation in EVALUATIONS:
            result, oof, folds, final_models = evaluate_window(
                data, window, rgb_features, logger, evaluation, fit_demo=evaluation == "primary_logo",
            )
            evaluations[evaluation] = result
            if evaluation == "primary_logo":
                fitted[str(window)] = final_models
                final_demo_fits = {method: result["models"][method]["final_demo_fit"] for method in METHODS}
            oof_tables.append(oof)
            fold_rows.extend(folds)
        window_results[str(window)] = {
            "n_common_events": evaluations["primary_logo"]["n_common_events"],
            "n_common_games": evaluations["primary_logo"]["n_common_games"],
            "evaluations": evaluations, "final_demo_fits": final_demo_fits,
        }
    oof = pd.concat(oof_tables, ignore_index=True)
    comparisons = add_comparisons(window_results, oof)
    summary = {
        "schema_version": "fms_rgb_event_regression_pilot_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(), "input_path": str(args.input.resolve()),
        "input_sha256": definition["input_sha256"],
        "training_definition_path": str(args.definition.resolve()),
        "training_definition_sha256": definition_sha256, "training_definition": definition,
        "task": "raw_FMS_exploratory_regression", "random_seed": SEED,
        "protocol": "User-authorized exploration outside original classification/subject validation gates",
        "subject_identity_verified": False, "deployment_verified": False,
        "evaluation_protocol": definition["evaluations"], "identity_proxy": definition["identity_proxy"],
        "target": "fms_raw unchanged; no target threshold or binning",
        "prediction_limits": "training-fold target min/max for all methods; final demo uses its full-fit target range",
        "model_names": list(METHODS), "window_seconds": list(WINDOWS),
        "hyperparameters": {"ridge": RIDGE_PARAMETERS, "lightgbm": LGBM_PARAMETERS},
        "preprocessing": {"numeric_invalid_to_missing": True, "drop_all_missing_scope": "training fold only",
                          "imputer": "training-fold median", "ridge_scaler": "training-fold StandardScaler",
                          "constant_feature_dropping": False, "tuning": False},
        "schema": schema, "windows": window_results, "window_comparisons": comparisons,
        "bootstrap_settings": {"n_resamples": N_BOOTSTRAP, "seed": SEED, "unit": "entire game",
                               "paired": True, "retraining": False, "confidence_level": 0.95,
                               "p_values_reported": False, "few_cluster_limit": "Approximately six games; exploratory intervals only"},
        "package_versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                             "scikit_learn": sklearn.__version__, "lightgbm": lgb.__version__},
    }
    bundle = {"schema_version": summary["schema_version"], "settings_and_metrics": summary,
              "feature_lists": {name: features_for(name, rgb_features) for name in METHODS},
              "windows": fitted, "demo_fit_scope": "all eligible events separately for each window; not evaluation predictions"}
    for path in (args.oof, args.metrics, args.fold_metrics, args.report, args.model):
        path.parent.mkdir(parents=True, exist_ok=True)
    oof.to_csv(args.oof, index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(args.fold_metrics, index=False, encoding="utf-8-sig")
    args.metrics.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    joblib.dump(bundle, args.model)
    args.report.write_text("\n".join(report_lines(summary, args)), encoding="utf-8")
    logger.info("SUCCESS seconds=%.3f oof=%s metrics=%s folds=%s report=%s model=%s",
                time.perf_counter() - started, args.oof, args.metrics, args.fold_metrics, args.report, args.model)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    logger = setup_logging(args.log)
    try:
        run(args, logger)
    except Exception:
        logger.exception("FAILED: FMS regression did not successfully complete")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
