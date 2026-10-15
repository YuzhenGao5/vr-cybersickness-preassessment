#!/usr/bin/env python3
"""Fixed raw-FMS regression within one game, holding out whole recordings.

Sessions with the same numeric prefix are kept in one conservative proxy group.
This is not authenticated participant LOSO. Evaluation compares 3/30-second
windows on common eligible events and never splits a recording's events at random.
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
FORBIDDEN_RGB = re.compile(
    r"(^|_)(id|path|game|session|subject|participant|event|timestamp|time|elapsed|"
    r"fms|ms|ssq|score|rating|label|target|coverage|count|counts|support|duration|"
    r"history|hmd|controller|headpose|head_pose|matrix|matrices)($|_)", re.IGNORECASE,
)
EVALUATION = "leave_one_session_or_prefix_group_out"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--game", required=True)
    parser.add_argument("--project-dir", type=Path, default=ROOT)
    parser.add_argument("--output-prefix", default="fms_full_game")
    args = parser.parse_args()
    args.game = args.game.strip()
    if not args.game:
        parser.error("--game cannot be blank")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,79}", args.output_prefix):
        parser.error("--output-prefix must be a plain name, using letters/numbers/_/-")
    if args.output_prefix in {"fms_rgb", "fms_rgb_pilot"}:
        parser.error("--output-prefix cannot overwrite previous pilot artifact names")
    args.project_dir = args.project_dir.resolve()
    args.features = args.features.resolve()
    return args


def output_paths(args: argparse.Namespace) -> dict[str, Path]:
    root, prefix = args.project_dir, args.output_prefix
    return {
        "oof": root / "features" / f"{prefix}_oof_predictions.csv",
        "metrics": root / "results" / f"{prefix}_metrics.json",
        "fold_metrics": root / "results" / f"{prefix}_fold_metrics.csv",
        "report": root / "results" / f"{prefix}_training_report.md",
        "model": root / "models" / f"{prefix}.pkl",
        "definition": root / "audit" / f"{prefix}_training_definition.json",
        "group_map": root / "audit" / f"{prefix}_session_group_map.csv",
        "log": root / "results" / f"{prefix}_training_log.txt",
    }


def setup_logging(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fms_full_game")
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


def group_for_session(session_id: str) -> str:
    prefix = session_id.split("_", 1)[0]
    return f"numeric_prefix:{prefix}" if prefix.isdigit() else f"session:{session_id}"


def verify_complete_game(project: Path, game: str, data: pd.DataFrame, rgb_features: list[str]) -> dict[str, Any]:
    summary_path = project / "results/fms_full_game_download_summary.json"
    audit_path = project / "results/fms_full_game_session_audit.csv"
    log_path = project / "results/fms_full_game_extraction_log.txt"
    for path in (summary_path, audit_path, log_path):
        if not path.is_file():
            raise ValueError(f"Complete-game audit prerequisite missing: {path}")
    acquisition = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    if acquisition.get("complete_game") is not True or acquisition.get("game_name") != game:
        raise ValueError("Download summary must declare complete_game=true and match --game")
    sources = acquisition.get("sessions", [])
    expected = {source["session_id"] for source in sources}
    if not expected or len(expected) != len(sources) or len(expected) != acquisition.get("verified_index_session_count"):
        raise ValueError("Download summary session list/count is missing, duplicated or inconsistent")
    if any(source.get("complete_by_csv_references") is not True or source.get("missing_references") != 0 for source in sources):
        raise ValueError("Download audit reports incomplete/missing JPG references")
    audit = pd.read_csv(audit_path, encoding="utf-8-sig", low_memory=False)
    required = {"game_name", "session_id", "status", "events", "missing_jpg_references"}
    if not required.issubset(audit.columns):
        raise ValueError(f"Session audit lacks columns: {sorted(required - set(audit.columns))}")
    if audit["session_id"].duplicated().any() or set(audit["session_id"]) != expected:
        raise ValueError("Extraction session audit must cover every downloaded session exactly once")
    if not audit["game_name"].eq(game).all() or not audit["status"].eq("ok").all():
        raise ValueError("Every complete-game extraction session must have matching game and status=ok")
    if not finite_numeric(audit["missing_jpg_references"]).eq(0).all():
        raise ValueError("Extraction audit has missing or unknown JPG references")
    if set(data["session_id"]) != expected:
        raise ValueError("Feature CSV session set differs from complete-game download/extraction audit")
    for row in audit.to_dict("records"):
        expected_events = row["events"]
        if not isinstance(expected_events, (int, float)) or not math.isfinite(expected_events) or expected_events < 0 or int(expected_events) != expected_events:
            raise ValueError("Session audit event counts must be finite nonnegative integers")
        for window in WINDOWS:
            count = len(data.loc[data["session_id"].eq(row["session_id"]) & data["window_sec"].eq(window)])
            if count != int(expected_events):
                raise ValueError(f"Partial/mismatched feature CSV: {row['session_id']} window={window}, {count} rows versus audit={expected_events}")
    log_lines = [line for line in log_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not log_lines or f"DONE full game: {len(expected)} sessions," not in log_lines[-1]:
        raise ValueError("Extraction log must end with the matching complete-game DONE record")
    inventory_path = project / "results/fms_full_game_archive_inventory.csv"
    expanded_bytes = None
    if inventory_path.is_file():
        inventory = pd.read_csv(inventory_path, encoding="utf-8-sig")
        if "bytes" in inventory:
            expanded_bytes = int(finite_numeric(inventory["bytes"]).sum())
    prior_path = project / "features/fms_rgb_event_features.csv"
    prior_alignment = None
    if prior_path.is_file():
        prior = pd.read_csv(prior_path, encoding="utf-8-sig", low_memory=False)
        if {"game_name", "session_id", "event_id", "window_sec", "status", "fms_raw", *rgb_features}.issubset(prior.columns):
            old_game = prior.loc[prior["game_name"].eq(game)]
            aligned = data.merge(old_game, on=["event_id", "window_sec"], suffixes=("_new", "_old"), validate="one_to_one")
            both_ok = aligned.loc[aligned["status_new"].eq("ok") & aligned["status_old"].eq("ok")]
            differences = [float((finite_numeric(both_ok[f"{name}_new"]) - finite_numeric(both_ok[f"{name}_old"])).abs().max()) for name in rgb_features] if len(both_ok) else []
            prior_alignment = {
                "prior_path": str(prior_path), "prior_sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest(),
                "prior_game_sessions": int(old_game["session_id"].nunique()),
                "newly_added_sessions": sorted(expected - set(old_game["session_id"])),
                "overlap_event_window_rows": len(aligned), "overlap_both_ok_rows": len(both_ok),
                "rgb_max_absolute_difference": max(differences) if differences and all(math.isfinite(v) for v in differences) else None,
                "overlap_raw_labels_equal": bool(np.allclose(finite_numeric(aligned["fms_raw_new"]), finite_numeric(aligned["fms_raw_old"]), equal_nan=True)),
            }
    return {
        "verified_before_fit": True, "download_summary_path": str(summary_path),
        "download_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "session_audit_path": str(audit_path), "session_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "extraction_log_path": str(log_path), "extraction_log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
        "complete_game": True, "game_name": game, "acquisition": acquisition.get("acquisition"),
        "archive_bytes": acquisition.get("archive_bytes"), "expanded_bytes": expanded_bytes,
        "session_count": len(expected), "csv_count": acquisition.get("retained_csv_count"),
        "jpg_count": acquisition.get("retained_jpg_count"), "missing_references": 0,
        "prior_pilot_alignment": prior_alignment,
    }


def read_input(path: Path, game: str) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if len(header) != len(set(header)):
        raise ValueError("Duplicate CSV column names")
    data = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    missing = [name for name in REQUIRED if name not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    input_rows = len(data)
    for name in ("event_id", "session_id", "game_name"):
        if data[name].isna().any() or data[name].astype(str).str.strip().eq("").any():
            raise ValueError(f"Blank/missing {name}")
        data[name] = data[name].astype(str).str.strip()
    input_games = sorted(data["game_name"].unique().tolist())
    data = data.loc[data["game_name"].eq(game)].copy()
    if data.empty:
        raise ValueError(f"No rows for game={game!r}; input games={input_games}")
    data["window_sec"] = finite_numeric(data["window_sec"])
    if data["window_sec"].isna().any() or not data["window_sec"].isin(WINDOWS).all():
        raise ValueError("window_sec must contain only the prespecified 3 and 30 seconds")
    data["window_sec"] = data["window_sec"].astype(int)
    if set(data["window_sec"].unique()) != set(WINDOWS):
        raise ValueError("Both 3-second and 30-second rows are required")
    if data.duplicated(["event_id", "window_sec"]).any():
        raise ValueError("Duplicate event_id/window_sec rows")
    if data.groupby("event_id")["session_id"].nunique().gt(1).any():
        raise ValueError("An event_id maps to multiple sessions")
    data["status"] = data["status"].fillna("missing").astype(str).str.strip().str.lower()
    rgb_candidates = [name for name in header if name.startswith("rgb_")]
    excluded_rgb = [name for name in rgb_candidates if FORBIDDEN_RGB.search(name)]
    rgb_features = [name for name in rgb_candidates if name not in excluded_rgb]
    if not rgb_features:
        raise ValueError("No allowed rgb_ numerical feature columns")
    invalid_values: dict[str, int] = {}
    for name in [*rgb_features, "elapsed_sec", "event_time_sec", "fms_raw"]:
        converted = finite_numeric(data[name])
        invalid_values[name] = int((data[name].notna() & converted.isna()).sum())
        data[name] = converted
    data["eligible"] = data["status"].eq("ok") & data["fms_raw"].notna()
    eligible_sets = {
        window: set(data.loc[data["window_sec"].eq(window) & data["eligible"], "event_id"])
        for window in WINDOWS
    }
    common_events = eligible_sets[3] & eligible_sets[30]
    data["common_event"] = data["event_id"].isin(common_events) & data["eligible"]
    data["holdout_group"] = data["session_id"].map(group_for_session)
    common = data.loc[data["common_event"]]
    for name in ("fms_raw", "event_time_sec", "elapsed_sec"):
        paired = common.pivot(index="event_id", columns="window_sec", values=name)
        if len(paired) and not np.allclose(paired[3], paired[30], rtol=0, atol=1e-6, equal_nan=True):
            raise ValueError(f"Common windows disagree on {name}")
    n_sessions = common["session_id"].nunique()
    n_groups = common["holdout_group"].nunique()
    if n_sessions < 3 or n_groups < 3:
        raise ValueError(f"Need >=3 sessions AND >=3 holdout groups with common events; got {n_sessions} sessions, {n_groups} groups")
    by_window: dict[str, Any] = {}
    by_session: list[dict[str, Any]] = []
    for window in WINDOWS:
        rows = data.loc[data["window_sec"].eq(window)]
        valid = rows.loc[rows["eligible"]]
        paired = rows.loc[rows["common_event"]]
        by_window[str(window)] = {
            "n_input_rows": len(rows), "n_status_ok": int(rows["status"].eq("ok").sum()),
            "n_eligible": len(valid), "n_common_events": len(paired),
            "n_eligible_sessions": int(valid["session_id"].nunique()),
            "n_common_sessions": int(paired["session_id"].nunique()),
            "n_common_groups": int(paired["holdout_group"].nunique()),
            "eligible_coverage_of_input": len(valid) / len(rows) if len(rows) else None,
            "common_coverage_of_eligible": len(paired) / len(valid) if len(valid) else None,
            "status_counts": {str(k): int(v) for k, v in rows["status"].value_counts().items()},
            "common_fms_value_counts": {str(k): int(v) for k, v in paired["fms_raw"].value_counts().sort_index().items()},
            "n_eligible_rows_all_rgb_missing": int(valid[rgb_features].isna().all(axis=1).sum()),
        }
        for session, group in rows.groupby("session_id", sort=True):
            paired_group = group.loc[group["common_event"]]
            by_session.append({
                "session_id": str(session), "holdout_group": group_for_session(str(session)),
                "window_sec": window, "n_input": len(group), "n_eligible": int(group["eligible"].sum()),
                "n_common": len(paired_group),
                "common_fms_value_counts": {str(k): int(v) for k, v in paired_group["fms_raw"].value_counts().sort_index().items()},
            })
    group_map = data[["session_id", "game_name", "holdout_group"]].drop_duplicates().sort_values("session_id")
    mapping = []
    for row in group_map.to_dict("records"):
        session_rows = data.loc[data["session_id"].eq(row["session_id"]) & data["window_sec"].eq(3)]
        mapping.append({**row, "numeric_prefix": row["session_id"].split("_", 1)[0] if row["session_id"].split("_", 1)[0].isdigit() else None,
                        "n_total_events": len(session_rows), "n_common_events": int(session_rows["common_event"].sum())})
    event_rows = data.loc[data["window_sec"].eq(3) & data["fms_raw"].notna()]
    session_means = event_rows.groupby("session_id")["fms_raw"].mean()
    event_intervals = [float(value) for _, rows in event_rows.groupby("session_id")
                       for value in np.diff(np.sort(rows["event_time_sec"].dropna().to_numpy(dtype=float))) if value > 0]
    schema = {
        "rgb_feature_names": rgb_features, "excluded_rgb_columns": excluded_rgb,
        "excluded_non_rgb_columns": [name for name in header if name not in rgb_features and name != "elapsed_sec"],
        "invalid_numeric_values": invalid_values, "session_group_map": mapping,
        "coverage": {
            "n_original_input_rows": input_rows, "input_games": input_games,
            "n_rows_excluded_other_games": input_rows - len(data),
            "n_selected_game_rows": len(data), "n_total_events": int(data["event_id"].nunique()),
            "n_total_sessions": int(data["session_id"].nunique()),
            "n_common_events": len(common_events), "n_common_sessions": int(n_sessions),
            "n_common_groups": int(n_groups), "by_window": by_window, "by_session": by_session,
            "raw_fms_value_counts": {str(k): int(v) for k, v in event_rows["fms_raw"].value_counts().sort_index().items()},
            "raw_session_fms_mean_min": float(session_means.min()), "raw_session_fms_mean_max": float(session_means.max()),
            "median_positive_within_session_score_interval_sec": float(np.median(event_intervals)) if event_intervals else None,
        },
    }
    data = data.sort_values(["window_sec", "holdout_group", "session_id", "event_time_sec", "event_id"]).reset_index(drop=True)
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
        raise ValueError("No known training targets")
    requested = features_for(method, rgb_features)
    dropped = [name for name in requested if data[name].isna().all()]
    retained = [name for name in requested if name not in dropped]
    imputer, scaler = None, None
    if method in CONSTANT_METHODS or not retained:
        strategy = "median" if method == "train_median" else "mean"
        estimator: Any = DummyRegressor(strategy=strategy)
        X = np.zeros((len(data), 1), dtype=float)
        kind = f"DummyRegressor_{strategy}"
        status = "fitted_constant_baseline" if method in CONSTANT_METHODS else "fallback_no_nonmissing_train_features"
        input_mode = "dummy_zero_column"
    else:
        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(data[retained])
        if method == "rgb_only_lightgbm":
            estimator = lgb.LGBMRegressor(**LGBM_PARAMETERS)
            kind = "LightGBMRegressor"
        else:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            estimator = Ridge(**RIDGE_PARAMETERS)
            kind = "Ridge"
        status, input_mode = "fitted", "imputed_retained_features"
    estimator.fit(X, y)
    return {"estimator": estimator, "imputer": imputer, "scaler": scaler, "metadata": {
        "method": method, "model_kind": kind, "fit_status": status, "input_mode": input_mode,
        "requested_features": requested, "retained_features": retained,
        "dropped_all_missing_train_features": dropped, "n_train": len(data),
        "n_train_sessions": int(data["session_id"].nunique()), "n_train_groups": int(data["holdout_group"].nunique()),
        "train_fms_min": float(y.min()), "train_fms_max": float(y.max()),
        "train_fms_mean": float(y.mean()), "train_fms_median": float(np.median(y)),
        "train_fms_unique_values": int(len(np.unique(y))),
        "n_train_rows_all_requested_features_missing": int(data[requested].isna().all(axis=1).sum()) if requested else None,
        "prediction_limits": "training-label minimum/maximum",
    }}


def predict_model(fitted: dict[str, Any], data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    meta = fitted["metadata"]
    if meta["input_mode"] == "dummy_zero_column":
        X = np.zeros((len(data), 1), dtype=float)
    else:
        X = fitted["imputer"].transform(data[meta["retained_features"]])
        if fitted["scaler"] is not None:
            X = fitted["scaler"].transform(X)
    raw = np.asarray(fitted["estimator"].predict(X), dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError(f"Nonfinite predictions from {meta['method']}")
    return raw, np.clip(raw, meta["train_fms_min"], meta["train_fms_max"])


def regression_metrics(y: np.ndarray, prediction: np.ndarray, constant_baseline: bool = False) -> dict[str, Any]:
    if not len(y):
        return {"n": 0, "mae": None, "rmse": None, "spearman": None, "spearman_status": "no_predictions"}
    error = prediction - y
    coefficient = None
    if constant_baseline:
        reason = "NA_constant_baseline"
    elif len(y) < 2:
        reason = "NA_fewer_than_two_events"
    elif len(np.unique(y)) < 2:
        reason = "NA_constant_target"
    elif len(np.unique(prediction)) < 2:
        reason = "NA_constant_prediction"
    else:
        value = float(spearmanr(y, prediction).statistic)
        coefficient = value if math.isfinite(value) else None
        reason = "descriptive_no_p_value" if coefficient is not None else "NA_nonfinite"
    return {"n": len(y), "mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2))),
            "spearman": coefficient, "spearman_status": reason}


def macro_group_metrics(folds: list[dict[str, Any]], view: str = "test_metrics") -> dict[str, Any]:
    result: dict[str, Any] = {"n_evaluated_groups": len(folds)}
    for metric in ("mae", "rmse", "spearman"):
        values = [fold[view][metric] for fold in folds if fold[view][metric] is not None]
        result[metric] = float(np.mean(values)) if values else None
        result[f"n_groups_with_defined_{metric}"] = len(values)
    return result


def evaluate_window(data: pd.DataFrame, window: int, rgb_features: list[str], logger: logging.Logger) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    rows = data.loc[data["window_sec"].eq(window) & data["common_event"]].copy()
    groups = sorted(rows["holdout_group"].unique())
    predictions: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    models: dict[str, Any] = {}
    final_models: dict[str, Any] = {}
    for method in METHODS:
        folds = []
        for fold_number, group in enumerate(groups, 1):
            train = rows.loc[rows["holdout_group"].ne(group)]
            test = rows.loc[rows["holdout_group"].eq(group)]
            if set(train["session_id"]) & set(test["session_id"]):
                raise AssertionError("Session leakage across train/test")
            fitted = fit_model(method, train, rgb_features)
            raw, limited = predict_model(fitted, test)
            train_raw, train_limited = predict_model(fitted, train)
            y, train_y = test["fms_raw"].to_numpy(dtype=float), train["fms_raw"].to_numpy(dtype=float)
            constant = method in CONSTANT_METHODS
            test_metrics = regression_metrics(y, limited, constant)
            train_metrics = regression_metrics(train_y, train_limited, constant)
            meta = fitted["metadata"]
            fold = {
                "fold": fold_number, "held_out_group": str(group), "status": "evaluated",
                "test_sessions": sorted(test["session_id"].unique().tolist()),
                "n_train": len(train), "n_test": len(test),
                "n_train_sessions": int(train["session_id"].nunique()), "n_test_sessions": int(test["session_id"].nunique()),
                "n_train_groups": int(train["holdout_group"].nunique()), "fit": meta,
                "test_metrics": test_metrics, "raw_test_metrics": regression_metrics(y, raw, constant),
                "train_metrics_in_sample": train_metrics,
                "raw_train_metrics_in_sample": regression_metrics(train_y, train_raw, constant),
                "test_minus_train_mae": test_metrics["mae"] - train_metrics["mae"],
                "test_minus_train_rmse": test_metrics["rmse"] - train_metrics["rmse"],
                "n_test_clipped": int(np.sum(raw != limited)), "n_train_clipped": int(np.sum(train_raw != train_limited)),
                "n_test_labels_outside_training_range": int(np.sum((y < meta["train_fms_min"]) | (y > meta["train_fms_max"]))),
                "test_fms_min": float(y.min()), "test_fms_max": float(y.max()), "test_fms_mean": float(y.mean()),
                "train_fms_value_counts": {str(k): int(v) for k, v in train["fms_raw"].value_counts().sort_index().items()},
                "test_fms_value_counts": {str(k): int(v) for k, v in test["fms_raw"].value_counts().sort_index().items()},
            }
            folds.append(fold)
            for position, event in enumerate(test.to_dict("records")):
                predictions.append({
                    "evaluation": EVALUATION, "window_sec": window, "method": method,
                    "event_id": event["event_id"], "session_id": event["session_id"], "game_name": event["game_name"],
                    "holdout_group": event["holdout_group"], "event_time_sec": event["event_time_sec"],
                    "elapsed_sec": event["elapsed_sec"], "fms_raw": event["fms_raw"], "fold": fold_number,
                    "prediction_raw": float(raw[position]), "prediction": float(limited[position]),
                    "was_clipped": bool(raw[position] != limited[position]),
                    "train_fms_min": meta["train_fms_min"], "train_fms_max": meta["train_fms_max"],
                    "n_train": len(train), "n_train_groups": meta["n_train_groups"],
                    "n_retained_features": len(meta["retained_features"]), "fit_status": meta["fit_status"],
                })
            fold_rows.append({
                "evaluation": EVALUATION, "window_sec": window, "method": method,
                "fold": fold_number, "held_out_group": str(group), "status": fold["status"],
                "test_sessions": json.dumps(fold["test_sessions"], ensure_ascii=False),
                "n_train": len(train), "n_test": len(test), "n_train_groups": fold["n_train_groups"],
                "n_train_sessions": fold["n_train_sessions"], "n_test_sessions": fold["n_test_sessions"],
                **{f"test_{k}": v for k, v in test_metrics.items()},
                **{f"raw_test_{k}": v for k, v in fold["raw_test_metrics"].items()},
                **{f"train_in_sample_{k}": v for k, v in train_metrics.items()},
                **{f"raw_train_in_sample_{k}": v for k, v in fold["raw_train_metrics_in_sample"].items()},
                "test_minus_train_mae": fold["test_minus_train_mae"], "test_minus_train_rmse": fold["test_minus_train_rmse"],
                "n_test_clipped": fold["n_test_clipped"], "n_train_clipped": fold["n_train_clipped"],
                "n_test_labels_outside_training_range": fold["n_test_labels_outside_training_range"],
                "train_fms_min": meta["train_fms_min"], "train_fms_max": meta["train_fms_max"],
                "train_fms_mean": meta["train_fms_mean"], "test_fms_min": fold["test_fms_min"],
                "test_fms_max": fold["test_fms_max"], "test_fms_mean": fold["test_fms_mean"],
                "train_fms_value_counts": json.dumps(fold["train_fms_value_counts"]),
                "test_fms_value_counts": json.dumps(fold["test_fms_value_counts"]),
                "fit_status": meta["fit_status"], "model_kind": meta["model_kind"],
                "retained_features": json.dumps(meta["retained_features"], ensure_ascii=False),
                "dropped_all_missing_train_features": json.dumps(meta["dropped_all_missing_train_features"], ensure_ascii=False),
            })
            logger.info("window=%s method=%s group=%s train=%s test=%s fit=%s train_MAE=%.6f test_MAE=%.6f train_RMSE=%.6f test_RMSE=%.6f",
                        window, method, group, len(train), len(test), meta["fit_status"], train_metrics["mae"], test_metrics["mae"], train_metrics["rmse"], test_metrics["rmse"])
        method_rows = [row for row in predictions if row["method"] == method]
        y_oof = np.asarray([row["fms_raw"] for row in method_rows], dtype=float)
        pred_oof = np.asarray([row["prediction"] for row in method_rows], dtype=float)
        raw_oof = np.asarray([row["prediction_raw"] for row in method_rows], dtype=float)
        constant_folds = all(len(np.unique([row["prediction"] for row in method_rows if row["fold"] == fold["fold"]])) < 2 for fold in folds)
        final_data = data.loc[data["window_sec"].eq(window) & data["eligible"]]
        final = fit_model(method, final_data, rgb_features)
        final_models[method] = final
        models[method] = {
            "feature_names": features_for(method, rgb_features), "folds": folds,
            "pooled_oof": regression_metrics(y_oof, pred_oof, method in CONSTANT_METHODS or constant_folds),
            "raw_pooled_oof": regression_metrics(y_oof, raw_oof, method in CONSTANT_METHODS),
            "macro_group": macro_group_metrics(folds), "raw_macro_group": macro_group_metrics(folds, "raw_test_metrics"),
            "n_clipped_oof": sum(fold["n_test_clipped"] for fold in folds),
            "fit_diagnostics": {
                "scope": "each outer fold's training events predicted by the fitted outer-fold model; in-sample only",
                "mean_fold_train_mae": float(np.mean([fold["train_metrics_in_sample"]["mae"] for fold in folds])),
                "mean_fold_train_rmse": float(np.mean([fold["train_metrics_in_sample"]["rmse"] for fold in folds])),
                "mean_fold_test_minus_train_mae": float(np.mean([fold["test_minus_train_mae"] for fold in folds])),
                "mean_fold_test_minus_train_rmse": float(np.mean([fold["test_minus_train_rmse"] for fold in folds])),
            }, "final_demo_fit": final["metadata"],
        }
    return {"n_common_events": len(rows), "n_groups": len(groups), "models": models}, pd.DataFrame(predictions), fold_rows, final_models


def paired_comparison(model: pd.DataFrame, mae_reference: pd.DataFrame, rmse_reference: pd.DataFrame) -> dict[str, Any]:
    keys = ["event_id", "holdout_group"]
    model = model[keys + ["fms_raw", "prediction"]].rename(columns={"prediction": "model_prediction"})
    mae_reference = mae_reference[keys + ["fms_raw", "prediction"]].rename(columns={"fms_raw": "mae_y", "prediction": "mae_reference_prediction"})
    rmse_reference = rmse_reference[keys + ["fms_raw", "prediction"]].rename(columns={"fms_raw": "rmse_y", "prediction": "rmse_reference_prediction"})
    paired = model.merge(mae_reference, on=keys, validate="one_to_one").merge(rmse_reference, on=keys, validate="one_to_one")
    if not len(paired) or len(paired) != len(model) or len(paired) != len(mae_reference) or len(paired) != len(rmse_reference):
        raise ValueError("Paired comparison needs identical, nonempty OOF event support")
    if not np.allclose(paired["fms_raw"], paired["mae_y"]) or not np.allclose(paired["fms_raw"], paired["rmse_y"]):
        raise ValueError("Paired OOF targets disagree")
    statistics, by_group = [], []
    for group_id, rows in paired.groupby("holdout_group", sort=True):
        y = rows["fms_raw"].to_numpy(dtype=float)
        error = rows["model_prediction"].to_numpy(dtype=float) - y
        mae_error = rows["mae_reference_prediction"].to_numpy(dtype=float) - y
        rmse_error = rows["rmse_reference_prediction"].to_numpy(dtype=float) - y
        statistics.append((len(rows), np.abs(error).sum(), np.abs(mae_error).sum(), np.square(error).sum(), np.square(rmse_error).sum()))
        by_group.append({"holdout_group": str(group_id), "n_events": len(rows),
                         "delta_mae": float(np.abs(error).mean() - np.abs(mae_error).mean()),
                         "delta_rmse": float(np.sqrt(np.square(error).mean()) - np.sqrt(np.square(rmse_error).mean()))})
    stats = np.asarray(statistics, dtype=float)
    n = stats[:, 0]
    model_mae, reference_mae = stats[:, 1].sum() / n.sum(), stats[:, 2].sum() / n.sum()
    model_rmse, reference_rmse = np.sqrt(stats[:, 3].sum() / n.sum()), np.sqrt(stats[:, 4].sum() / n.sum())
    group_mae_delta = (stats[:, 1] - stats[:, 2]) / n
    group_rmse_delta = np.sqrt(stats[:, 3] / n) - np.sqrt(stats[:, 4] / n)
    selection = np.random.default_rng(SEED).integers(0, len(stats), size=(N_BOOTSTRAP, len(stats)))
    totals = stats[selection].sum(axis=1)
    samples = {
        "pooled_delta_mae": (totals[:, 1] - totals[:, 2]) / totals[:, 0],
        "pooled_delta_rmse": np.sqrt(totals[:, 3] / totals[:, 0]) - np.sqrt(totals[:, 4] / totals[:, 0]),
        "macro_group_delta_mae": group_mae_delta[selection].mean(axis=1),
        "macro_group_delta_rmse": group_rmse_delta[selection].mean(axis=1),
    }
    return {
        "n_events": len(paired), "n_groups": len(stats), "difference_direction": "model minus reference; negative means lower error",
        "pooled": {"delta_mae": float(model_mae - reference_mae), "delta_rmse": float(model_rmse - reference_rmse),
                   "mae_relative_reduction": float((reference_mae - model_mae) / reference_mae) if reference_mae else None,
                   "rmse_relative_reduction": float((reference_rmse - model_rmse) / reference_rmse) if reference_rmse else None},
        "macro_group": {"delta_mae": float(group_mae_delta.mean()), "delta_rmse": float(group_rmse_delta.mean())},
        "by_group": by_group,
        "bootstrap": {"unit": "whole held-out session-or-prefix group", "n_resamples": N_BOOTSTRAP, "seed": SEED,
                      "confidence_level": 0.95, "retraining": False, "status": "exploratory_group_cluster_percentile",
                      **{f"{name}_ci95": [float(v) for v in np.quantile(values, [0.025, 0.975])] for name, values in samples.items()}},
    }


def add_comparisons(windows: dict[str, Any], oof: pd.DataFrame) -> dict[str, Any]:
    for window in WINDOWS:
        rows = oof.loc[oof["window_sec"].eq(window)]
        median, mean = rows.loc[rows["method"].eq("train_median")], rows.loc[rows["method"].eq("train_mean")]
        elapsed = rows.loc[rows["method"].eq("elapsed_only_ridge")]
        for method in METHODS:
            model = rows.loc[rows["method"].eq(method)]
            entry = windows[str(window)]["models"][method]
            entry["versus_train_baselines"] = paired_comparison(model, median, mean)
            if method in RGB_METHODS:
                entry["versus_elapsed_only"] = paired_comparison(model, elapsed, elapsed)
    comparisons = {}
    for method in METHODS:
        model = oof.loc[oof["method"].eq(method) & oof["window_sec"].eq(30)]
        reference = oof.loc[oof["method"].eq(method) & oof["window_sec"].eq(3)]
        comparisons[method] = paired_comparison(model, reference, reference)
        comparisons[method]["difference_direction"] = "30-second minus 3-second error"
    return comparisons


def fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def comparison_cell(comparison: dict[str, Any], metric: str, macro: bool = False) -> str:
    scope = "macro_group" if macro else "pooled"
    interval = comparison["bootstrap"][f"{scope}_delta_{metric}_ci95"]
    return f"{fmt(comparison[scope][f'delta_{metric}'])} [{fmt(interval[0])}, {fmt(interval[1])}]"


def report_lines(summary: dict[str, Any], paths: dict[str, Path]) -> list[str]:
    coverage = summary["schema"]["coverage"]
    acquisition = summary["acquisition_audit"]
    lines = [f"# 完整游戏 FMS RGB 回归：{summary['game_name']}", "",
             "原始 FMS 用户授权探索回归；未通过原 PROTOCOL 的分类/subject 门控。单游戏验证不能证明跨游戏或 Unity 迁移。", "",
             "## 固定设计与覆盖", "",
             f"- 输入 `{summary['input_path']}`；SHA256 `{summary['input_sha256']}`。",
             f"- 首个 fit 之前保存定义 `{paths['definition']}`，SHA256 `{summary['definition_sha256']}`。",
             f"- {coverage['n_total_events']} 原始事件 / {coverage['n_total_sessions']} sessions；共同有效 {coverage['n_common_events']} events / {coverage['n_common_sessions']} sessions / {coverage['n_common_groups']} groups。",
             f"- 完整游戏审计：fresh download `{acquisition['archive_bytes']}` bytes；展开 `{acquisition['expanded_bytes']}` bytes；{acquisition['csv_count']} CSV、{acquisition['jpg_count']} JPG；全部 session 提取 ok、missing JPG references=0，提取日志以 DONE 结束。",
             f"- 原始 FMS 值频数 `{coverage['raw_fms_value_counts']}`；各 session 评分均值范围 {fmt(coverage['raw_session_fms_mean_min'])}–{fmt(coverage['raw_session_fms_mean_max'])}；session 内正评分间隔中位 {fmt(coverage['median_positive_within_session_score_interval_sec'])} 秒，评分事件稀疏。",
             f"- 与旧 pilot 重叠审计 `{acquisition['prior_pilot_alignment']}`。新增 session 全部保留，不按分数、时长或训练结果选择。",
             "- 上轮 6-game LOGO 检验跨游戏泛化，本轮单游戏 session 留出检验同游戏新录制。两者任务、训练与测试分布均不同，性能差异不能直接归因于数据量增加。",
             "- 同 session 的全部事件同折；多个 session 的数字前缀相同则合并 group，否则 session 单独留出。前缀不是认证的 participant ID，此评估不称 subject LOSO。无随机 event split。",
             "- 两窗口共同事件仅取 status=ok 且原始 FMS 有限的交集；目标不分箱、不填充、不平移。",
             f"- 六固定模型，seed={SEED}；Ridge `{RIDGE_PARAMETERS}`；LightGBM `{LGBM_PARAMETERS}`；不调参、不 early stopping。",
             "- 只使用允许的 rgb_ 列及指定 elapsed_sec；评分历史、session/group/game、controller/headpose/HMD、矩阵、路径、覆盖及最终时长不进入模型。",
             "- 每训练折独立 drop allmissing 列、median imputer；Ridge 独立 StandardScaler。没有保留列时明确降为训练均值 Dummy。",
             "- 全部模型用训练标签 min/max 限制预测；主表用限制后值，OOF/JSON/每fold表亦保留原始值。",
             "- pooled 指标按事件汇总；macro-group 按留出 group 等权平均。常数模型 Spearman 为 NA，其他未定义相关单独记录；不报事件独立 p 值。",
             f"- RGB 特征 `{summary['schema']['rgb_feature_names']}`；排除 rgb_ 列 `{summary['schema']['excluded_rgb_columns']}`。", "",
             "| 窗口 | 输入 | status ok | 有效 | 共同 | 有效/输入 | 共同/有效 | 状态 |", "|---:|---:|---:|---:|---:|---:|---:|---|"]
    for window in WINDOWS:
        c = coverage["by_window"][str(window)]
        lines.append(f"| {window} | {c['n_input_rows']} | {c['n_status_ok']} | {c['n_eligible']} | {c['n_common_events']} | {fmt(c['eligible_coverage_of_input'])} | {fmt(c['common_coverage_of_eligible'])} | {fmt(c['status_counts'])} |")
    lines.extend(["", "### 全部 session 覆盖与分组", "", "| Session | Group | 窗口 | 输入/有效/共同 | 共同 FMS 频数 |", "|---|---|---:|---|---|"])
    for row in coverage["by_session"]:
        lines.append(f"| {fmt(row['session_id'])} | {fmt(row['holdout_group'])} | {row['window_sec']} | {row['n_input']}/{row['n_eligible']}/{row['n_common']} | {fmt(row['common_fms_value_counts'])} |")
    for window in WINDOWS:
        models = summary["windows"][str(window)]["models"]
        lines.extend(["", f"## {window} 秒：全部模型", "", "| 模型 | OOF n | pooled MAE | RMSE | Spearman | macro MAE | RMSE | Spearman (定义groups) | clipped |", "|---|---:|---:|---:|---:|---:|---:|---|---:|"])
        for method in METHODS:
            m = models[method]
            p, a = m["pooled_oof"], m["macro_group"]
            lines.append(f"| {method} | {p['n']} | {fmt(p['mae'])} | {fmt(p['rmse'])} | {fmt(p['spearman'])} | {fmt(a['mae'])} | {fmt(a['rmse'])} | {fmt(a['spearman'])} ({a['n_groups_with_defined_spearman']}) | {m['n_clipped_oof']} |")
        lines.extend(["", "### 配对基线差值与 95% group 聚类区间", "", "ΔMAE 对 train_median；ΔRMSE 对 train_mean；负值代表更低误差。", "", "| 模型 | pooled ΔMAE [CI] | pooled ΔRMSE [CI] | macro ΔMAE [CI] | macro ΔRMSE [CI] |", "|---|---|---|---|---|"])
        for method in METHODS:
            c = models[method]["versus_train_baselines"]
            lines.append(f"| {method} | {comparison_cell(c, 'mae')} | {comparison_cell(c, 'rmse')} | {comparison_cell(c, 'mae', True)} | {comparison_cell(c, 'rmse', True)} |")
        lines.extend(["", "### RGB 相对 elapsed-only", "", "| 模型 | pooled ΔMAE [CI] | pooled ΔRMSE [CI] |", "|---|---|---|"])
        for method in RGB_METHODS:
            c = models[method]["versus_elapsed_only"]
            lines.append(f"| {method} | {comparison_cell(c, 'mae')} | {comparison_cell(c, 'rmse')} |")
        lines.extend(["", "### 全部折及训练拟合诊断", "", "训练值是同一 outer fold 已拟合模型对其训练事件的 in-sample 重代入误差。test−train 差值只诊断拟合差距，可来自拟合或评分分布变化；训练集事件重复用于不同折，不能当作独立泛化样本或改善证明。各折 train/test 评分频数和范围另存 JSON/CSV。", "", "| 模型 | 留出 group | train/test events | test sessions | train MAE/RMSE | test MAE/RMSE | test−train MAE/RMSE | train/test FMS mean | test Spearman | fit |", "|---|---|---|---|---|---|---|---|---:|---|"])
        for method in METHODS:
            for fold in models[method]["folds"]:
                tr, te = fold["train_metrics_in_sample"], fold["test_metrics"]
                lines.append(f"| {method} | {fmt(fold['held_out_group'])} | {fold['n_train']}/{fold['n_test']} | {fmt(fold['test_sessions'])} | {fmt(tr['mae'])}/{fmt(tr['rmse'])} | {fmt(te['mae'])}/{fmt(te['rmse'])} | {fmt(fold['test_minus_train_mae'])}/{fmt(fold['test_minus_train_rmse'])} | {fmt(fold['fit']['train_fms_mean'])}/{fmt(fold['test_fms_mean'])} | {fmt(te['spearman'])} | {fold['fit']['fit_status']} |")
        lines.extend(["", "### 全量有效数据演示模型", ""])
        for method in METHODS:
            fit = models[method]["final_demo_fit"]
            lines.append(f"- `{method}`：{fit['fit_status']}；n={fit['n_train']}，sessions={fit['n_train_sessions']}，groups={fit['n_train_groups']}；retained `{fit['retained_features']}`；dropped `{fit['dropped_all_missing_train_features']}`。")
    lines.extend(["", "## 30 秒对 3 秒共同事件", "", "差值为 30 秒−3 秒，负值表示 30 秒更低误差；报告全部模型，不按结果调整窗口。", "", "| 模型 | pooled ΔMAE [CI] | pooled ΔRMSE [CI] | macro ΔMAE [CI] | macro ΔRMSE [CI] |", "|---|---|---|---|---|"])
    for method in METHODS:
        c = summary["window_comparisons"][method]
        lines.append(f"| {method} | {comparison_cell(c, 'mae')} | {comparison_cell(c, 'rmse')} | {comparison_cell(c, 'mae', True)} | {comparison_cell(c, 'rmse', True)} |")
    lines.extend(["", "## 解释与交付", "",
                  f"- 固定 {N_BOOTSTRAP} 次 whole-group paired bootstrap，seed={SEED}；抽到某 group 时保留其全部事件。不同 session 同数字前缀时以合并 group 为重抽样单位。",
                  "- 这是既有 OOF 误差的条件性区间，未 bootstrap 重训；少 group 时区间非常不稳定，参与者独立性也未认证。不按 event 独立重采样，不报伪独立 p 值。",
                  "- 同录制事件相关，outer-fold 训练数据重叠；覆盖筛选可改变评分分布。目标是有序原始评分，本次回归用数值距离作探索损失。",
                  "- 只有 RGB 持续优于固定常数和 elapsed-only 对照，且多个留出 group 支持相同方向，才可能形成额外视觉信息的证据；不能仅凭低训练误差或某一优胜窗口声称有效。",
                  "- 本脚本在训练前核对 complete-game 下载审计、全部 session 提取状态、feature session 集与逐窗口事件数、提取 DONE；未重新读取全部原始 JPG/CSV。模型为未部署验证的演示，不证明个体、跨游戏或 Unity 泛化。",
                  "- bundle.windows[窗口秒字符串][方法] 含 estimator/imputer/scaler/metadata；按 retained_features 填补、可选缩放、预测后按 train_fms_min/max 限制。Dummy 使用 n×1 零矩阵。",
                  *[f"- {name}：`{path}`。" for name, path in paths.items()], ""])
    return lines


def run(args: argparse.Namespace, paths: dict[str, Path], logger: logging.Logger) -> None:
    started = time.perf_counter()
    command = subprocess.list2cmdline(getattr(sys, "orig_argv", [sys.executable, *sys.argv]))
    logger.info("COMMAND %s", command)
    logger.info("WORKDIR %s", Path.cwd())
    data, rgb_features, schema = read_input(args.features, args.game)
    acquisition_audit = verify_complete_game(args.project_dir, args.game, data, rgb_features)
    logger.info("COMPLETE_GAME_GATE_PASSED sessions=%s JPG=%s missing=%s", acquisition_audit["session_count"], acquisition_audit["jpg_count"], acquisition_audit["missing_references"])
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    definition = {
        "created_utc_before_any_fit": datetime.now(timezone.utc).isoformat(), "command": command,
        "input_path": str(args.features), "input_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
        "training_script_path": str(Path(__file__).resolve()), "training_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "game_name": args.game, "output_prefix": args.output_prefix, "random_seed": SEED,
        "target": "unchanged raw fms_raw; no bins/history/interpolation",
        "protocol": "user-authorized exploratory regression outside original classification/subject gates",
        "eligibility": "status=ok and finite fms_raw; primary paired comparison uses common 3/30 event IDs",
        "windows_sec": list(WINDOWS), "evaluation": EVALUATION,
        "grouping": "same numeric session_id.split('_')[0] -> one group; nonnumeric prefix -> session-specific group; all same-session events remain together",
        "subject_identity_verified": False, "subject_loso_claim": False, "minimum_common_sessions": 3, "minimum_common_groups": 3,
        "model_names": list(METHODS), "feature_lists": {method: features_for(method, rgb_features) for method in METHODS},
        "hyperparameters": {"ridge": RIDGE_PARAMETERS, "lightgbm": LGBM_PARAMETERS},
        "preprocessing": "training-fold allmissing-column drop, median imputation, Ridge StandardScaler; no constant-column or target-based feature selection",
        "prediction_limits": "every method clipped to its training-label min/max",
        "comparisons": "MAE versus train_median; RMSE versus train_mean; RGB versus elapsed-only; 30sec versus 3sec",
        "fit_diagnostic": "outer-fold model predicts its own training events; descriptive in-sample only, not validation",
        "bootstrap": {"unit": "entire held-out session-or-prefix group", "seed": SEED, "n_resamples": N_BOOTSTRAP, "paired": True,
                      "confidence_level": 0.95, "retraining": False, "independent_event_resampling": False},
        "session_group_map": schema["session_group_map"], "tuning": False,
        "acquisition_audit": acquisition_audit,
        "deployment_verified": False, "classification_auc": False, "p_values": False,
    }
    paths["definition"].write_text(json.dumps(definition, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pd.DataFrame(schema["session_group_map"]).to_csv(paths["group_map"], index=False, encoding="utf-8-sig")
    definition_sha = hashlib.sha256(paths["definition"].read_bytes()).hexdigest()
    logger.info("PRE_FIT_FROZEN definition=%s sha256=%s events=%s common=%s sessions=%s groups=%s features=%s",
                paths["definition"], definition_sha, schema["coverage"]["n_total_events"], schema["coverage"]["n_common_events"],
                schema["coverage"]["n_common_sessions"], schema["coverage"]["n_common_groups"], len(rgb_features))
    windows, final_models, tables, fold_rows = {}, {}, [], []
    for window in WINDOWS:
        result, oof, folds, fitted = evaluate_window(data, window, rgb_features, logger)
        windows[str(window)], final_models[str(window)] = result, fitted
        tables.append(oof)
        fold_rows.extend(folds)
    oof = pd.concat(tables, ignore_index=True)
    comparisons = add_comparisons(windows, oof)
    if hashlib.sha256(args.features.read_bytes()).hexdigest() != definition["input_sha256"]:
        raise RuntimeError("Input file changed during training; refusing to publish inconsistent artifacts")
    summary = {
        "schema_version": "fms_full_game_regression_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "game_name": args.game, "input_path": str(args.features), "input_sha256": definition["input_sha256"],
        "definition_path": str(paths["definition"]), "definition_sha256": definition_sha, "definition": definition,
        "schema": schema, "windows": windows, "window_comparisons": comparisons,
        "acquisition_audit": acquisition_audit,
        "subject_identity_verified": False, "deployment_verified": False,
        "package_versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                             "scikit_learn": sklearn.__version__, "lightgbm": lgb.__version__},
    }
    bundle = {"schema_version": summary["schema_version"], "settings_and_metrics": summary,
              "feature_lists": definition["feature_lists"], "windows": final_models,
              "demo_fit_scope": "all eligible events per window; not OOF evaluation"}
    oof.to_csv(paths["oof"], index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(paths["fold_metrics"], index=False, encoding="utf-8-sig")
    paths["metrics"].write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    joblib.dump(bundle, paths["model"])
    paths["report"].write_text("\n".join(report_lines(summary, paths)), encoding="utf-8")
    logger.info("SUCCESS seconds=%.3f OOF_rows=%s fold_rows=%s outputs=%s", time.perf_counter() - started, len(oof), len(fold_rows), paths)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    paths = output_paths(args)
    logger = setup_logging(paths["log"])
    try:
        run(args, paths, logger)
    except Exception:
        logger.exception("FAILED: full-game FMS regression did not successfully complete")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
