#!/usr/bin/env python3
"""Bounded nested game-CV binary-signal exploration; never tune on outer games.

Config methods: {name: {input, feature_columns, log1p_columns, description}}.
The exact event cohort is read from cohort_reference_csv (not rediscovered).
All preprocessing, sample/class weighting and candidate choice use training only.
No training is performed on import. See --help for the command-line interface.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, f1_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from train_param_rating_model import ROOT, VARIANCE_THRESHOLD, digest, save_json, table as helper_table

NAME = "auc_signal_round2_2026-10-10"
IDS = ["game_name", "session_id", "participant_id", "rating_event_id"]
META = [*IDS, "ms_rating"]
TARGET_THRESHOLD = 2.0
CLASS_THRESHOLD = 0.5
SEED = 42
ELAPSED = {"exposure_elapsed_seconds", "elapsed_seconds", "elapsed_time_seconds"}
FORBIDDEN = re.compile(r"(?:rating|label|future|timestamp|event_id|session_id|participant_id|game_name|source_file|time_delta|matching_distance)", re.I)
UNKNOWN_PEOPLE = {"", "unknown", "nan", "none", "null", "<na>"}
SELECTOR_POLICY = ("within each outer training set select the method/family pair having the highest "
                   "inner mean validation-game AUC after within-family tuning; reuse that pair's outer "
                   "predictions; ties follow config method order then config family order then fixed "
                   "candidate order; no valid inner AUC uses first pair; no outer scores consulted")

# These finite grids are fixed before a run. Class balancing is implemented via
# training-only sample weights, identically for all estimator families.
GRIDS = {
    "logistic": [dict(C=c, class_balance=balance)
                 for c in [0.1, 1.0, 10.0] for balance in ["none", "balanced"]],
    "lightgbm": [dict(num_leaves=leaves, n_estimators=trees,
                      min_child_samples=child, class_balance="balanced")
                 for leaves in [3, 7] for trees in [100, 200] for child in [20, 40]],
    "extra_trees": [dict(min_samples_leaf=5, max_features=0.5, class_balance="balanced"),
                    dict(min_samples_leaf=15, max_features=1.0, class_balance="balanced")],
}


def table(rows, columns):
    return "\n".join(helper_table(rows, [item[0] for item in columns], [item[1] for item in columns]))


def root_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def known_people(frame):
    return {str(value) for value in frame["participant_id"].dropna()
            if str(value).lower() not in UNKNOWN_PEOPLE}


def session_weights(frame):
    counts = frame.groupby("session_id")["rating_event_id"].transform("size").to_numpy(dtype=float)
    weights = 1.0 / counts
    return weights / weights.mean()


def fit_weights(frame, y, balance):
    weights = session_weights(frame)
    masses = [float(weights[y == label].sum()) for label in [0, 1]]
    factors = [1.0, 1.0]
    if balance == "balanced" and min(masses) > 0:
        factors = [float(weights.sum() / (2 * mass)) for mass in masses]
        weights *= np.asarray(factors)[y]
    return weights, dict(base_weight_policy="equal total weight per session, equal events within session",
                         class_balance=balance, training_weighted_class_mass=masses,
                         class_factors=factors, final_class_mass=[float(weights[y == label].sum()) for label in [0, 1]])


def estimator(kind, candidate):
    parameters = {name: value for name, value in candidate.items() if name != "class_balance"}
    if kind == "logistic":
        return LogisticRegression(**parameters, solver="lbfgs", penalty="l2", max_iter=3000,
                                  random_state=SEED, class_weight=None)
    if kind == "lightgbm":
        return LGBMClassifier(**parameters, objective="binary", max_depth=3,
                              learning_rate=0.05, reg_lambda=1.0, random_state=SEED,
                              n_jobs=2, verbosity=-1, class_weight=None)
    if kind == "extra_trees":
        return ExtraTreesClassifier(**parameters, n_estimators=300, criterion="gini",
                                    random_state=SEED, n_jobs=2, class_weight=None,
                                    bootstrap=False)
    raise ValueError(f"unknown model family: {kind}")


def fit_model(frame, columns, log_columns, kind, candidate):
    y = frame["binary_true"].to_numpy(dtype=int)
    usable = [name for name in columns if frame[name].notna().any()]
    weights, weight_detail = fit_weights(frame, y, candidate["class_balance"])
    prior = float(np.average(y, weights=session_weights(frame)))
    detail = dict(input_feature_cols=usable, fitted_feature_cols=[],
                  requested_feature_columns=columns, log1p_columns=log_columns,
                  dropped_all_missing=[name for name in columns if name not in usable],
                  dropped_constant=[], candidate=candidate, weight_detail=weight_detail,
                  training_events=len(frame), training_sessions=int(frame["session_id"].nunique()),
                  training_games=int(frame["game_name"].nunique()),
                  training_positive_events=int(y.sum()), training_prevalence=prior,
                  constant_probability=None, status="fitted")
    if len(np.unique(y)) < 2 or not usable:
        detail.update(status="prior_only_single_train_class" if len(np.unique(y)) < 2 else "prior_only_no_features",
                      constant_probability=prior)
        return None, detail
    fitted_logs = [name for name in log_columns if name in usable]
    # ColumnTransformer puts the explicit log group first, then the passthrough
    # group. Track that exact output order when interpreting variance support.
    transformed_columns = fitted_logs + [name for name in usable if name not in fitted_logs]
    log_transform = ColumnTransformer(
        [("log1p", FunctionTransformer(np.log1p), [usable.index(name) for name in fitted_logs])],
        remainder="passthrough")
    steps = [("imputer", SimpleImputer(strategy="median")),
             ("registered_log1p", log_transform),
             ("variance", VarianceThreshold(threshold=VARIANCE_THRESHOLD))]
    # Check constant columns with training only. A constant-feature predictor is
    # represented explicitly; no validation/test values are consulted here.
    prepared = steps[0][1].fit_transform(frame[usable])
    prepared = steps[1][1].fit_transform(prepared)
    try:
        steps[2][1].fit(prepared)
    except ValueError:
        detail.update(status="prior_only_constant_features", constant_probability=prior,
                      dropped_constant=usable)
        return None, detail
    support = steps[2][1].get_support()
    detail.update(transformed_feature_order=transformed_columns,
                  fitted_feature_cols=[name for name, keep in zip(transformed_columns, support) if keep],
                  dropped_constant=[name for name, keep in zip(transformed_columns, support) if not keep])
    if kind == "logistic":
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator(kind, candidate)))
    pipe = Pipeline(steps)
    pipe.fit(frame[usable], y, model__sample_weight=weights)
    return pipe, detail


def predict_model(pipe, detail, frame):
    if pipe is None:
        return np.full(len(frame), detail["constant_probability"], dtype=float)
    index = list(pipe.named_steps["model"].classes_).index(1)
    return pipe.predict_proba(frame[detail["input_feature_cols"]])[:, index]


def selection_value(choice):
    value = choice.get("inner_mean_game_auc")
    return float(value) if value is not None else -np.inf


def feature_registration(definition):
    columns = definition.get("feature_columns")
    logs = definition.get("log1p_columns", [])
    if not isinstance(columns, list) or not columns or len(columns) != len(set(columns)):
        raise ValueError("each method needs unique nonempty feature_columns")
    if not isinstance(logs, list) or len(logs) != len(set(logs)) or not set(logs) <= set(columns):
        raise ValueError("log1p_columns must be an explicit unique subset of feature_columns")
    invalid = [name for name in columns if not isinstance(name, str) or
               (name not in ELAPSED and (name in META or FORBIDDEN.search(name) or
                name.startswith("nearest_") or name in {"window_id", "start_time_sec", "end_time_sec",
                "window_center_sec", "common_eligible", "window_has_any_feature", "dataset_family"}))]
    if invalid:
        raise ValueError(f"identity/score/matching-time/post-rating features forbidden: {invalid}")
    if definition.get("contains_post_rating", False):
        raise ValueError("round2 requires strictly pre-rating inputs")
    return columns, logs


def read_identity(path):
    frame = pd.read_csv(path, dtype={name: str for name in IDS}, encoding="utf-8-sig")
    missing = set(META) - set(frame)
    if missing:
        raise ValueError(f"missing identity/score fields: {sorted(missing)}")
    if frame[IDS].isna().any().any() or not frame["rating_event_id"].is_unique:
        raise ValueError("one complete identity row per unique rating_event_id required")
    frame["ms_rating"] = pd.to_numeric(frame["ms_rating"], errors="coerce")
    if not np.isfinite(frame["ms_rating"]).all():
        raise ValueError("fixed cohort contains nonfinite score")
    if not frame["ms_rating"].isin([1, 2, 3, 4, 5]).all():
        raise ValueError("expected unchanged integer MS_rating in 1..5")
    if frame.groupby("session_id")["game_name"].nunique().gt(1).any():
        raise ValueError("a session belongs to more than one game")
    if frame.groupby("session_id")["participant_id"].nunique().gt(1).any():
        raise ValueError("a session belongs to more than one participant")
    return frame


def same_identity(left, right):
    # CSV encoders may emit 1 versus 1.0; score equality must not depend on dtype.
    return len(left) == len(right) and all(np.array_equal(left[name].to_numpy(), right[name].to_numpy()) for name in META)


def load_data(cfg, input_dir):
    if "cohort_reference_csv" not in cfg:
        raise ValueError("config.cohort_reference_csv is mandatory: keep the exact prior cohort")
    reference_path = root_path(cfg["cohort_reference_csv"])
    if cfg.get("cohort_reference_sha256") and digest(reference_path) != cfg["cohort_reference_sha256"]:
        raise ValueError("cohort reference hash mismatch")
    reference = read_identity(reference_path)[META].sort_values("rating_event_id").reset_index(drop=True)
    if len(reference) != int(cfg.get("expected_cohort_events", 1024)):
        raise ValueError("fixed cohort event count differs from registration")
    definitions = cfg.get("methods")
    if not isinstance(definitions, dict) or not definitions:
        raise ValueError("config.methods must be a nonempty map")
    cache, frames, audits = {}, {}, {}
    for method, definition in definitions.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", method):
            raise ValueError("method names must be safe filename components")
        columns, logs = feature_registration(definition)
        source = Path(definition["input"])
        if not source.is_absolute():
            source = input_dir / source
        source = source.resolve()
        if source not in cache:
            cache[source] = read_identity(source)
        original = cache[source]
        if definition.get("input_sha256") and digest(source) != definition["input_sha256"]:
            raise ValueError(f"{method}: input hash mismatch")
        if set(reference["rating_event_id"]) - set(original["rating_event_id"]):
            raise ValueError(f"{method}: missing registered cohort events; do not silently reduce cohort")
        missing = set(columns + ["common_eligible"]) - set(original)
        if missing:
            raise ValueError(f"{method}: missing {sorted(missing)}")
        frame = original.set_index("rating_event_id", drop=False).loc[reference["rating_event_id"]].reset_index(drop=True).copy()
        if not same_identity(frame, reference):
            raise ValueError(f"{method}: fixed cohort identities/labels mismatch")
        eligible = frame["common_eligible"].astype(str).str.lower().isin({"true", "1", "1.0", "yes"})
        if not eligible.all():
            raise ValueError(f"{method}: some fixed cohort events fail preparation common_eligible")
        frame[columns] = frame[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if not frame[columns].notna().any(axis=1).all():
            raise ValueError(f"{method}: all-missing event input; fixed cohort cannot be changed")
        negative = [name for name in logs if (frame[name].dropna() < 0).any()]
        if negative:
            raise ValueError(f"{method}: registered nonnegative log1p columns have negative values: {negative}")
        frame["binary_true"] = (frame["ms_rating"] > TARGET_THRESHOLD).astype(int)
        frames[method] = frame
        audits[method] = dict(input=str(source), input_sha256=digest(source), input_rows=len(original),
                              cohort_events=len(frame), feature_columns=columns, log1p_columns=logs,
                              description=definition.get("description", method),
                              temporal_policy=definition.get("temporal_policy", "strictly before rating"),
                              missing_values=int(frame[columns].isna().sum().sum()))
    reference["binary_true"] = (reference["ms_rating"] > TARGET_THRESHOLD).astype(int)
    return frames, reference, audits, dict(path=str(reference_path.resolve()), sha256=digest(reference_path))


def inner_plan(frame, double_purge):
    games = frame["game_name"].to_numpy()
    unique_games = np.unique(games)
    plans, audit = [], []
    if len(unique_games) < 2:
        return plans, audit
    splits = GroupKFold(n_splits=min(3, len(unique_games))).split(frame, frame["binary_true"], games)
    for inner_fold, (train_idx, validation_idx) in enumerate(splits, 1):
        before = frame.iloc[train_idx]
        validation = frame.iloc[validation_idx]
        shared = sorted(known_people(before) & known_people(validation))
        if double_purge:
            validation_people = known_people(validation)
            train_idx = train_idx[~frame.iloc[train_idx]["participant_id"].isin(validation_people).to_numpy()]
        train = frame.iloc[train_idx]
        if set(train["game_name"]) & set(validation["game_name"]) or set(train["session_id"]) & set(validation["session_id"]):
            raise ValueError("inner group/session leakage")
        if double_purge and known_people(train) & known_people(validation):
            raise ValueError("inner participant purge failed")
        usable_train = len(train) > 0 and train["binary_true"].nunique() == 2
        valid_games = [str(game) for game, part in validation.groupby("game_name", sort=True)
                       if part["binary_true"].nunique() == 2 and usable_train]
        detail = dict(inner_fold=inner_fold, training_events_before_purge=len(before), training_events=len(train),
                      purged_events=len(before)-len(train), validation_events=len(validation),
                      validation_games=sorted(validation["game_name"].unique()),
                      valid_auc_games=valid_games, shared_participants_before_purge=shared,
                      shared_participants_after_purge=sorted(known_people(train) & known_people(validation)),
                      status="usable" if valid_games else "unavailable", reason=None if valid_games else "single-class/empty inner train or single-class validation games")
        audit.append(detail)
        plans.append((inner_fold, train_idx, validation_idx, valid_games))
    return plans, audit


def tune(frame, columns, log_columns, kind, double_purge, context):
    plans, plan_audit = inner_plan(frame, double_purge)
    expected_games = sum(len(plan[3]) for plan in plans)
    candidates, game_rows, score_rows = GRIDS[kind], [], []
    for candidate_index, candidate in enumerate(candidates):
        aucs = []
        for inner_fold, train_idx, validation_idx, valid_games in plans:
            if not valid_games:
                continue
            pipe, detail = fit_model(frame.iloc[train_idx], columns, log_columns, kind, candidate)
            validation = frame.iloc[validation_idx]
            prediction = predict_model(pipe, detail, validation)
            for game in valid_games:
                mask = validation["game_name"].eq(game).to_numpy()
                auc = float(roc_auc_score(validation.loc[mask, "binary_true"], prediction[mask]))
                aucs.append(auc)
                game_rows.append(dict(**context, model=kind, candidate_index=candidate_index,
                                      candidate_id=f"{kind}_{candidate_index:02d}", inner_fold=inner_fold,
                                      validation_game=game, validation_events=int(mask.sum()),
                                      inner_training_events=len(train_idx), inner_auc=auc,
                                      fit_status=detail["status"], parameters_json=json.dumps(candidate, sort_keys=True)))
        complete = expected_games > 0 and len(aucs) == expected_games
        score_rows.append(dict(**context, model=kind, candidate_index=candidate_index,
                               candidate_id=f"{kind}_{candidate_index:02d}",
                               inner_mean_game_auc=float(np.mean(aucs)) if complete else None,
                               valid_auc_games=len(aucs), expected_auc_games=expected_games,
                               selection_eligible=complete, selected=False,
                               parameters_json=json.dumps(candidate, sort_keys=True)))
    if expected_games:
        selected = max(range(len(score_rows)), key=lambda index: (score_rows[index]["inner_mean_game_auc"], -index))
        selection_reason = "maximum equally weighted valid validation-game AUC; ties use fixed candidate order"
    else:
        selected = 0
        selection_reason = "no valid inner game AUC; use preregistered first candidate without consulting outer labels"
    score_rows[selected]["selected"] = True
    choice = dict(candidate_index=selected, candidate_id=f"{kind}_{selected:02d}",
                  candidate=candidates[selected], inner_mean_game_auc=score_rows[selected]["inner_mean_game_auc"],
                  selection_reason=selection_reason, inner_plan=plan_audit)
    return candidates[selected], choice, score_rows, game_rows


def metric_row(rows, protocol, method, model, fold=None, held_out_game=None):
    y = rows["binary_true"].to_numpy(dtype=int)
    p = rows["probability"].to_numpy(dtype=float)
    complete = len(p) > 0 and np.isfinite(p).all()
    both_classes = len(np.unique(y)) == 2
    if complete and ((p < 0).any() or (p > 1).any()):
        raise ValueError("classifier probability outside [0,1]")
    classification = (p >= CLASS_THRESHOLD).astype(int) if complete else None
    reason = None if complete and both_classes else "incomplete OOF" if not complete else "single-class target"
    return dict(protocol=protocol, method=method, model=model, fold=fold, held_out_game=held_out_game,
                events=len(rows), sessions=int(rows["session_id"].nunique()), games=int(rows["game_name"].nunique()),
                positive_events=int(y.sum()), negative_events=int((1-y).sum()), prevalence=float(y.mean()) if len(y) else None,
                binary_auc=float(roc_auc_score(y, p)) if complete and both_classes else None,
                average_precision=float(average_precision_score(y, p)) if complete and both_classes else None,
                ranking_metric_reason=reason,
                accuracy=float(accuracy_score(y, classification)) if complete else None,
                f1_macro=float(f1_score(y, classification, labels=[0, 1], average="macro", zero_division=0)) if complete else None,
                brier=float(brier_score_loss(y, p)) if complete else None,
                mean_probability=float(p.mean()) if complete else None,
                mean_probability_positive=float(p[y == 1].mean()) if complete and (y == 1).any() else None,
                mean_probability_negative=float(p[y == 0].mean()) if complete and (y == 0).any() else None,
                complete_oof=complete)


def grouped_auc_bootstrap(rows, draws):
    if draws <= 0 or not np.isfinite(rows["probability"]).all():
        return dict(auc_ci_low=None, auc_ci_high=None, bootstrap_valid_draws=0)
    y, p = rows["binary_true"].to_numpy(dtype=int), rows["probability"].to_numpy(dtype=float)
    grouped_indices = list(rows.reset_index(drop=True).groupby("session_id", sort=True).indices.values())
    rng, aucs = np.random.default_rng(SEED), []
    for _ in range(draws):
        index = np.concatenate([grouped_indices[i] for i in rng.integers(0, len(grouped_indices), len(grouped_indices))])
        if len(np.unique(y[index])) == 2:
            aucs.append(float(roc_auc_score(y[index], p[index])))
    return dict(auc_ci_low=float(np.percentile(aucs, 2.5)) if aucs else None,
                auc_ci_high=float(np.percentile(aucs, 97.5)) if aucs else None,
                bootstrap_valid_draws=len(aucs))


def calibration_rows(predictions):
    outputs = []
    for (method, model), group in predictions.groupby(["method", "model"], sort=False):
        if not np.isfinite(group["probability"]).all():
            continue
        bins = np.minimum((group["probability"].to_numpy() * 10).astype(int), 9)
        for number in range(10):
            part = group.loc[bins == number]
            outputs.append(dict(protocol=str(group["protocol"].iloc[0]), method=method, model=model,
                                bin=number, lower=number/10, upper=(number+1)/10, events=len(part),
                                mean_probability=float(part["probability"].mean()) if len(part) else None,
                                observed_positive_rate=float(part["binary_true"].mean()) if len(part) else None))
    return outputs


def probability_score_groups(predictions):
    outputs = []
    for (method, model, score), group in predictions.groupby(["method", "model", "ms_rating"], sort=False):
        p = group["probability"].to_numpy(dtype=float)
        complete = np.isfinite(p).all()
        outputs.append(dict(protocol=str(group["protocol"].iloc[0]), method=method, model=model,
                            actual_ms_rating=float(score), events=len(group),
                            mean_probability=float(p.mean()) if complete else None,
                            predicted_positive_events=int((p >= CLASS_THRESHOLD).sum()) if complete else None))
    return outputs


def write_csv(path, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.17g")


def run_protocol(frames, cohort, audits, kinds, protocol, methods, results_dir, log, draws):
    double_purge = protocol == "leave_game_and_participant_out"
    outputs, folds, candidates, inner_games, fit_audit = [], [], [], [], []
    groups = cohort["game_name"].to_numpy()
    for fold, (train_idx, test_idx) in enumerate(LeaveOneGroupOut().split(cohort, cohort["binary_true"], groups), 1):
        test_meta = cohort.iloc[test_idx]
        heldout = str(test_meta["game_name"].iloc[0])
        train_before = cohort.iloc[train_idx]
        shared_before = sorted(known_people(train_before) & known_people(test_meta))
        if double_purge:
            train_idx = train_idx[~cohort.iloc[train_idx]["participant_id"].isin(known_people(test_meta)).to_numpy()]
        train_meta = cohort.iloc[train_idx]
        if set(train_meta["session_id"]) & set(test_meta["session_id"]) or set(train_meta["rating_event_id"]) & set(test_meta["rating_event_id"]):
            raise ValueError("outer event/session leakage")
        if double_purge and known_people(train_meta) & known_people(test_meta):
            raise ValueError("outer participant purge failed")
        outer_detail = dict(training_events_before_purge=len(train_before), training_events=len(train_meta),
                            purged_training_events=len(train_before)-len(train_meta),
                            participant_overlap_before_purge=len(shared_before),
                            participant_overlap_after_purge=len(known_people(train_meta) & known_people(test_meta)))
        if len(train_meta):
            prior = float(np.average(train_meta["binary_true"], weights=session_weights(train_meta)))
        else:
            prior = np.nan
        variants = [("baseline", "train_prevalence", np.full(len(test_meta), prior),
                     dict(status="training_prior" if len(train_meta) else "unavailable_empty_train"), {})]
        for method in methods:
            for kind in kinds:
                context = dict(protocol=protocol, phase="outer", method=method, fold=fold, held_out_game=heldout)
                if len(train_meta):
                    candidate, choice, scores, game_scores = tune(frames[method].iloc[train_idx],
                        audits[method]["feature_columns"], audits[method]["log1p_columns"], kind, double_purge, context)
                    candidates.extend(scores)
                    inner_games.extend(game_scores)
                    pipe, detail = fit_model(frames[method].iloc[train_idx], audits[method]["feature_columns"],
                                             audits[method]["log1p_columns"], kind, candidate)
                    p = predict_model(pipe, detail, frames[method].iloc[test_idx])
                else:
                    detail, choice = dict(status="unavailable_empty_train"), dict(selection_reason="outer training is empty")
                    p = np.full(len(test_idx), np.nan)
                fit_audit.append(dict(**context, outer_split=outer_detail, choice=choice, preprocessing=detail))
                variants.append((method, kind, p, detail, choice))
        # All pair choices use the same predefined inner validation games.
        # This selector was specified before any round2 metrics, so its outer
        # predictions evaluate one complete selection procedure rather than a
        # winner chosen using the resulting outer table.
        selected_index = max(range(1, len(variants)), key=lambda index: (selection_value(variants[index][4]), -index))
        selected_method, selected_kind, selected_p, selected_detail, selected_choice = variants[selected_index]
        selected_choice = dict(selected_choice, selected_input_method=selected_method, selected_model_family=selected_kind,
                               pair_selector_policy=SELECTOR_POLICY)
        variants.append(("nested_selected", "selected", selected_p, selected_detail, selected_choice))
        fit_audit.append(dict(protocol=protocol, phase="outer", method="nested_selected", model="selected", fold=fold,
                              held_out_game=heldout, outer_split=outer_detail, choice=selected_choice,
                              preprocessing=selected_detail, prediction_reused_without_fit=True))
        for method, kind, p, detail, choice in variants:
            rows = test_meta.copy()
            rows["protocol"], rows["method"], rows["model"] = protocol, method, kind
            rows["fold"], rows["held_out_game"] = fold, heldout
            rows["probability"] = p
            rows["predicted_class"] = pd.Series(np.where(np.isfinite(p), (p >= CLASS_THRESHOLD).astype(float), np.nan), index=rows.index)
            rows["selected_candidate_id"] = choice.get("candidate_id")
            rows["selected_input_method"] = choice.get("selected_input_method", method if method != "baseline" else None)
            rows["selected_model_family"] = choice.get("selected_model_family", kind if method != "baseline" else None)
            rows["inner_mean_game_auc"] = choice.get("inner_mean_game_auc")
            rows["fit_status"] = detail["status"]
            outputs.append(rows)
            metric = metric_row(rows, protocol, method, kind, fold, heldout)
            metric.update(**outer_detail, selected_candidate_id=choice.get("candidate_id"),
                          selected_input_method=choice.get("selected_input_method", method if method != "baseline" else None),
                          selected_model_family=choice.get("selected_model_family", kind if method != "baseline" else None),
                          selected_parameters_json=json.dumps(choice.get("candidate", {}), sort_keys=True),
                          inner_mean_game_auc=choice.get("inner_mean_game_auc"),
                          fit_status=detail["status"], fitted_feature_columns_json=json.dumps(detail.get("fitted_feature_cols", [])))
            folds.append(metric)
        log.info("%s fold %s/%s game=%s train=%s test=%s purged=%s", protocol, fold,
                 cohort["game_name"].nunique(), heldout, len(train_meta), len(test_meta), outer_detail["purged_training_events"])
    predictions = pd.concat(outputs, ignore_index=True)
    metrics = []
    for (method, kind), rows in predictions.groupby(["method", "model"], sort=False):
        if len(rows) != len(cohort) or not rows["rating_event_id"].is_unique:
            raise ValueError("OOF event coverage differs from registered cohort")
        row = metric_row(rows, protocol, method, kind)
        matching = [part for part in folds if part["method"] == method and part["model"] == kind]
        valid_auc = [part["binary_auc"] for part in matching if part["binary_auc"] is not None]
        valid_ap = [part["average_precision"] for part in matching if part["average_precision"] is not None]
        row.update(mean_valid_fold_auc=float(np.mean(valid_auc)) if valid_auc else None,
                   valid_auc_folds=len(valid_auc), total_folds=len(matching),
                   mean_valid_fold_average_precision=float(np.mean(valid_ap)) if valid_ap else None,
                   **grouped_auc_bootstrap(rows, draws))
        metrics.append(row)
    prefix = "double_holdout_" if double_purge else ""
    write_csv(results_dir / f"{prefix}oof_predictions.csv", predictions)
    write_csv(results_dir / f"{prefix}fold_metrics.csv", folds)
    write_csv(results_dir / f"{prefix}metrics.csv", metrics)
    write_csv(results_dir / f"{prefix}inner_candidate_scores.csv", candidates)
    write_csv(results_dir / f"{prefix}inner_game_scores.csv", inner_games)
    write_csv(results_dir / f"{prefix}calibration.csv", calibration_rows(predictions))
    write_csv(results_dir / f"{prefix}score_group_probability.csv", probability_score_groups(predictions))
    return metrics, folds, fit_audit


def previous_references(cfg, cohort, results_dir):
    rows, audits = [], []
    defaults = [dict(name=f"previous_{method}_ridge", input=f"results/rating_aggregation_comparison_2026-10-10/oof_predictions.csv",
                     filter={"method": method, "model": "ridge"}, prediction_column="prediction")
                for method in ["multi_scene", "multi_scene_head", "elapsed_only"]]
    for definition in cfg.get("previous_oof_references", defaults):
        source = root_path(definition["input"])
        frame = pd.read_csv(source, dtype={name: str for name in IDS}, encoding="utf-8-sig")
        for name, value in definition.get("filter", {}).items():
            if name not in frame:
                raise ValueError(f"reference filter missing column {name}")
            frame = frame.loc[frame[name].eq(value)]
        if not frame["rating_event_id"].is_unique:
            raise ValueError("previous OOF reference must have unique event IDs after filtering")
        if not set(cohort["rating_event_id"]) <= set(frame["rating_event_id"]):
            raise ValueError("previous reference does not cover exact common cohort")
        frame = frame.set_index("rating_event_id", drop=False).loc[cohort["rating_event_id"]].reset_index(drop=True)
        if not same_identity(frame, cohort):
            raise ValueError("previous reference identity/target mismatch")
        score = pd.to_numeric(frame[definition.get("prediction_column", "prediction")], errors="coerce").to_numpy(dtype=float)
        y = cohort["binary_true"].to_numpy(dtype=int)
        valid = np.isfinite(score).all() and len(np.unique(y)) == 2
        fold_aucs = [float(roc_auc_score(y[index], score[index])) for index in cohort.groupby("game_name", sort=True).indices.values()
                     if valid and len(np.unique(y[index])) == 2]
        rows.append(dict(reference=definition["name"], events=len(cohort), target="original score regression ranking reference",
                         binary_auc=float(roc_auc_score(y, score)) if valid else None,
                         average_precision=float(average_precision_score(y, score)) if valid else None,
                         mean_valid_fold_auc=float(np.mean(fold_aucs)) if fold_aucs else None,
                         mean_raw_score=float(score.mean()) if np.isfinite(score).all() else None,
                         probability_calibration="not applicable: regression scores are not probabilities"))
        audits.append(dict(**definition, input_sha256=digest(source), events=len(cohort)))
    write_csv(results_dir / "previous_regression_references.csv", rows)
    return rows, audits


def write_report(path, manifest, primary, sensitivity, references):
    columns = [("method", "输入组"), ("model", "模型族"), ("binary_auc", "合并OOF AUC"),
               ("auc_ci_low", "AUC CI下界"), ("auc_ci_high", "上界"),
               ("mean_valid_fold_auc", "游戏折均值AUC"), ("average_precision", "AP"),
               ("f1_macro", "F1 macro"), ("accuracy", "准确率"), ("brier", "Brier"),
               ("mean_probability", "平均输出概率")]
    info = manifest.get("cohort", {})
    text = [f"# 直接二分类信号探索：{NAME}", "", f"状态：**{manifest['status']}**。本轮在上一轮结果之后提出，是有限、预先固定的探索；没有达到0.70即停止时的性能保证。",
            "", f"固定共同事件：{info.get('events', 0)} 次真实评分，{info.get('sessions', 0)} sessions，{info.get('games', 0)} 游戏，{info.get('participants', 0)} 参与者编号。真实 MS_rating>2 固定为正类；原评分不作为输入。",
            f"正类比例 {info.get('prevalence')}；AP 的无技巧参考为这个正类比例。评分分布 {info.get('rating_distribution')}。",
            "", "## 输入与输出", "",
            "每行对应一次原始真实评分事件，输入为配置注册的严格评分前统计及累计暴露时间。身份、评分、历史评分、评分配对距离和最近分组含未来测量均禁止作为输入。场景相机运动仍可能混合用户头动和导航；用户变量并非无人开发期设计输入。",
            "输出为 MS_rating>2 的连续分类分数，固定0.5用于补充分类统计。类别加权改变训练先验，输出未另做概率校准，不能直接解释为可靠的个人患病风险。",
            "", "## 训练和评估", "",
            "外层 Leave-One-Game-Out：测试游戏及其评分从不参与该折预处理、权重、候选参数选择或阈值选择。内层在外层训练游戏上 GroupKFold3；每个可计算的验证游戏单独计算AUC，再等权平均选参数。各模型族分别调参并全部报告；内层平局取固定候选顺序第一项。无可用内层AUC时使用预设第一项并记录原因。",
            "另有预设 nested_selected 决策程序：每个外层训练集仅按内层游戏AUC选择输入组及模型族，复用所选组合的外层预测；不按本表外层表现选择。平局按配置输入组/模型族及固定候选顺序处理。两协议分别执行选择。",
            "中位数补缺、常量筛选及 Logistic 标准化只拟合当前训练折；显式注册的非负量使用固定 log1p。基权重每 session 等权，balanced 候选再按当前训练折的加权类别质量乘类别权重。",
            "候选规模：Logistic 6、LightGBM 8、ExtraTrees 2（如注册启用）；完整固定参数与每折选择记录在 manifest/inner_candidate_scores.csv。没有搜索到某个AUC就追加候选。",
            "合并OOF AUC与游戏折AUC均值是不同统计；AP明确使用 average_precision_score，并非梯形PR积分。阈值0.5未按外层结果调整。F1 macro固定包含0/1两类。",
            "", "## 主协议：留游戏", "", table(primary, columns) if primary else "没有可报告结果。",
            "", "AUC区间按session重采样固定OOF；不重训，条件于现有游戏。重复评分和共享参与者意味着它不覆盖未知游戏/参与者以及多轮方法选择的全部不确定性。最高AUC是探索性名次，不是独立验证。0.70应同时查看合并值、游戏差异、区间、AP和校准，不能作为事后门槛挑选结果。",
            "", "## 敏感性：留游戏并删除测试参与者", "",
            "若启用，外层训练删除测试游戏出现的所有参与者；内层也删除各验证游戏参与者，使用相同固定候选及游戏AUC选择规则。训练不可用折记录原因为空训练/单类/常量特征；常量或单类训练只返回训练先验，不能解释为训练成功。",
            table(sensitivity, columns) if sensitivity else "配置未启用。",
            "", "## 前一轮评分回归参考", "",
            "下表仅比较相同事件上既有OOF评分预测的排名。目标、训练和调参方式不同；它们不是概率，不计算Brier或概率校准。",
            table(references, [("reference", "参考"), ("binary_auc", "合并AUC"), ("mean_valid_fold_auc", "游戏折均值AUC"), ("average_precision", "AP")]) if references else "未注册旧参考。",
            "", "## 文件与限制", "",
            "oof_predictions.csv保存原评分、binary_true、概率、固定阈值类别以及每折选择；fold_metrics.csv保留游戏差异及参与者交叉；calibration.csv是固定0.1宽度概率分箱；score_group_probability.csv检查真实3/4/5的输出。double_holdout_*为独立敏感性输出。",
            "完整数据模型用于复现/推理，其候选通过全数据的内层分组CV选择后全量拟合；报告性能仅来自外层OOF。运行中的头动/控制器可能反映参与者反应或录分行为，场景单位及源字段含义的限制仍适用；本次没有证明可控设计改动产生因果效果。",
            "本轮二分类AUC不等同原三分类macro AUC，也没有验证原三分类协议门控。"]
    if manifest.get("reason"):
        text += ["", f"不可用原因：{manifest['reason']}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / f"{NAME}.json")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "features" / NAME)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / NAME)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models" / NAME)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.bootstrap_draws < 0:
        raise ValueError("bootstrap draws must be nonnegative")
    cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if cfg.get("status") not in {"fixed_before_round2_training", "fixed_before_training", "fixed_before_comparison_training"}:
        raise ValueError("config must declare a fixed-before-training status")
    if float(cfg.get("binary_threshold", TARGET_THRESHOLD)) != TARGET_THRESHOLD or float(cfg.get("class_threshold", CLASS_THRESHOLD)) != CLASS_THRESHOLD:
        raise ValueError("target>2 and classification probability>=0.5 are fixed")
    if int(cfg.get("seed", SEED)) != SEED or int(cfg.get("inner_splits", 3)) != 3:
        raise ValueError("registered seed42 and inner GroupKFold3 are fixed")
    if "fixed_candidate_grids" in cfg and cfg["fixed_candidate_grids"] != GRIDS:
        raise ValueError("registered fixed_candidate_grids differs from the fixed script grids or tie order")
    protocol_file = root_path(cfg.get("protocol_file", "PROTOCOL.md"))
    if not protocol_file.is_file():
        raise ValueError("protocol file not found")
    if cfg.get("protocol_sha256") and cfg["protocol_sha256"] != digest(protocol_file):
        raise ValueError("protocol hash mismatch")
    preparation_path = root_path(cfg["preparation_manifest"])
    preparation = json.loads(preparation_path.read_text(encoding="utf-8-sig"))
    if preparation.get("status") != "complete":
        raise ValueError("preparation manifest is not complete")
    if cfg.get("preparation_sha256") and cfg["preparation_sha256"] != digest(preparation_path):
        raise ValueError("preparation hash mismatch")
    frames, cohort, audits, cohort_reference = load_data(cfg, args.input_dir)
    kinds = cfg.get("models", ["logistic", "lightgbm", "extra_trees"])
    if not isinstance(kinds, list) or not kinds or len(kinds) != len(set(kinds)) or not set(kinds) <= set(GRIDS):
        raise ValueError("models must be unique registered model families")
    sensitivity_cfg = cfg.get("double_holdout", {})
    sensitivity_enabled = bool(sensitivity_cfg.get("enabled", False))
    sensitivity_methods = sensitivity_cfg.get("methods", list(frames))
    sensitivity_models = sensitivity_cfg.get("models", kinds)
    if not isinstance(sensitivity_methods, list) or not set(sensitivity_methods) <= set(frames) or not sensitivity_methods:
        raise ValueError("double_holdout.methods must be a nonempty subset of methods")
    if not isinstance(sensitivity_models, list) or not set(sensitivity_models) <= set(kinds) or not sensitivity_models:
        raise ValueError("double_holdout.models must be a nonempty subset of models")
    if sensitivity_enabled and any(str(value).lower() in UNKNOWN_PEOPLE for value in cohort["participant_id"]):
        raise ValueError("double holdout requires known participant IDs for every cohort event")
    # Preserve any completed run: changing method/model selection needs a new directory.
    if any((args.results_dir / name).is_file() for name in ["training_manifest.json", "planned_training_manifest.json"]):
        raise ValueError("results directory already has a frozen/training manifest; use a fresh directory")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report or args.results_dir / "report.md"
    log = logging.getLogger(NAME)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in [logging.StreamHandler(), logging.FileHandler(args.results_dir / "training_log.txt", encoding="utf-8")]:
        handler.setFormatter(formatter)
        log.addHandler(handler)
    write_csv(args.results_dir / "cohort.csv", cohort)
    # Reference scores and hashes are fixed before the first model fit too.
    references, reference_audit = previous_references(cfg, cohort, args.results_dir)
    manifest = dict(status="planned", analysis=NAME, created_utc=datetime.now(timezone.utc).isoformat(),
                    config_path=str(args.config.resolve()), config_sha256=digest(args.config), config=cfg,
                    training_script_sha256=digest(Path(__file__)), helper_script_sha256=digest(ROOT / "scripts" / "train_param_rating_model.py"),
                    preparation_path=str(preparation_path.resolve()), preparation_sha256=digest(preparation_path), preparation_manifest=preparation,
                    protocol_path=str(protocol_file.resolve()), protocol_sha256=digest(protocol_file),
                    cohort_reference=cohort_reference, cohort_sha256=digest(args.results_dir / "cohort.csv"), inputs=audits,
                    model_families=kinds, fixed_candidate_grids={kind: GRIDS[kind] for kind in kinds},
                    full_estimator_parameters={kind: [estimator(kind, candidate).get_params(deep=False) for candidate in GRIDS[kind]] for kind in kinds},
                    target="unchanged original MS_rating>2", binary_threshold=TARGET_THRESHOLD, classification_threshold=CLASS_THRESHOLD,
                    preprocessing="inner/outer training-only median imputation, registered stateless log1p, variance filter; Logistic training-only standardization",
                    class_weight_policy="equal-session base sample weights; balanced candidates equalize weighted train class masses; no test-label class weighting",
                    inner_selection="GroupKFold3 over train games; equal mean AUC of valid validation games; same predefined eligible games for all candidates; fixed first candidate fallback/ties",
                    nested_pair_selector=dict(policy=SELECTOR_POLICY, planned_before_any_round2_metric=True,
                                              methods=list(frames), families=kinds, outer_refits=0,
                                              full_fit_policy="same inner game-AUC method/family selection over the full cohort; reuse selected full fitted bundle"),
                    outer_protocol="LeaveOneGameOut", double_holdout=dict(enabled=sensitivity_enabled, methods=sensitivity_methods, models=sensitivity_models, inner_participant_purge=True),
                    previous_references=reference_audit,
                    bootstrap=dict(draws=args.bootstrap_draws, seed=SEED, unit="session", interpretation="fixed OOF conditional on observed games; no model refits; excludes method-selection uncertainty"),
                    cohort=dict(events=len(cohort), sessions=int(cohort["session_id"].nunique()), games=int(cohort["game_name"].nunique()),
                                participants=int(cohort["participant_id"].nunique()), prevalence=float(cohort["binary_true"].mean()),
                                rating_distribution=cohort["ms_rating"].value_counts().sort_index().to_dict(), events_by_game=cohort.groupby("game_name").size().to_dict()),
                    environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                                     numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, lightgbm=lightgbm.__version__, joblib=joblib.__version__),
                    iteration="follow-up exploratory study after earlier regression results; bounded grids; not an independent confirmatory or original three-class gate test")
    manifest["frozen_before_fit_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(args.results_dir / "planned_training_manifest.json", manifest)
    if cohort["game_name"].nunique() < 3 or cohort["binary_true"].nunique() < 2:
        manifest.update(status="unavailable", reason="need at least three games and two binary target classes")
        save_json(args.results_dir / "training_manifest.json", manifest)
        write_report(report_path, manifest, [], [], [])
        return 2
    try:
        primary, primary_folds, primary_audit = run_protocol(frames, cohort, audits, kinds, "leave_game_out", list(frames), args.results_dir, log, args.bootstrap_draws)
        sensitivity, sensitivity_folds, sensitivity_audit = [], [], []
        if sensitivity_enabled:
            sensitivity, sensitivity_folds, sensitivity_audit = run_protocol(frames, cohort, audits, sensitivity_models,
                "leave_game_and_participant_out", sensitivity_methods, args.results_dir, log, args.bootstrap_draws)
        full_models, full_scores, full_games, full_bundles = {}, [], [], {}
        for method, frame in frames.items():
            for kind in kinds:
                context = dict(protocol="full_fit", phase="full_fit", method=method, fold=0, held_out_game=None)
                candidate, choice, scores, games = tune(frame, audits[method]["feature_columns"], audits[method]["log1p_columns"], kind, False, context)
                full_scores.extend(scores)
                full_games.extend(games)
                pipe, detail = fit_model(frame, audits[method]["feature_columns"], audits[method]["log1p_columns"], kind, candidate)
                destination = args.models_dir / f"{method}_{kind}.pkl"
                bundle = dict(pipeline=pipe, constant_probability=detail["constant_probability"], input_feature_cols=detail["input_feature_cols"],
                                 fitted_feature_cols=detail["fitted_feature_cols"], requested_feature_columns=audits[method]["feature_columns"],
                                 preprocessing=detail, selection=choice, method=method, model=kind,
                                 target="MS_rating>2", class_threshold=CLASS_THRESHOLD, target_threshold=TARGET_THRESHOLD,
                                 warning="exploratory full fit; class-weighted output is uncalibrated; performance from outer OOF only",
                                 config_sha256=manifest["config_sha256"], cohort_sha256=manifest["cohort_sha256"], input_sha256=audits[method]["input_sha256"])
                joblib.dump(bundle, destination)
                full_bundles[f"{method}/{kind}"] = bundle
                full_models[f"{method}/{kind}"] = dict(path=str(destination.resolve()), selection=choice, preprocessing=detail)
        full_keys = list(full_models)
        chosen_key = max(full_keys, key=lambda key: (selection_value(full_models[key]["selection"]), -full_keys.index(key)))
        chosen_method, chosen_kind = chosen_key.split("/", 1)
        selected_bundle = dict(full_bundles[chosen_key], method="nested_selected", model="selected",
                               selected_input_method=chosen_method, selected_model_family=chosen_kind,
                               pair_selector_policy=SELECTOR_POLICY)
        destination = args.models_dir / "nested_selected.pkl"
        joblib.dump(selected_bundle, destination)
        full_models["nested_selected/selected"] = dict(path=str(destination.resolve()), selected_input_method=chosen_method,
            selected_model_family=chosen_kind, prediction_reused_without_fit=True,
            selection=full_models[chosen_key]["selection"], preprocessing=full_models[chosen_key]["preprocessing"])
        write_csv(args.results_dir / "full_fit_candidate_scores.csv", full_scores)
        write_csv(args.results_dir / "full_fit_inner_game_scores.csv", full_games)
        complete = all(row["complete_oof"] for row in primary + sensitivity)
        manifest.update(status="complete" if complete else "partial_unavailable_folds", metrics=primary, fold_metrics=primary_folds,
                        fold_training_audit=primary_audit, double_holdout_metrics=sensitivity,
                        double_holdout_fold_metrics=sensitivity_folds, double_holdout_training_audit=sensitivity_audit,
                        previous_references=reference_audit, full_models=full_models, finished_utc=datetime.now(timezone.utc).isoformat())
        save_json(args.results_dir / "training_manifest.json", manifest)
        write_report(report_path, manifest, primary, sensitivity, references)
        for row in primary:
            log.info("%s/%s AUC=%s foldmean=%s AP=%s Brier=%s", row["method"], row["model"], row["binary_auc"], row["mean_valid_fold_auc"], row["average_precision"], row["brier"])
        return 0 if complete else 2
    except Exception as error:
        manifest.update(status="failed", reason=f"{type(error).__name__}: {error}", finished_utc=datetime.now(timezone.utc).isoformat())
        save_json(args.results_dir / "training_manifest.json", manifest)
        write_report(report_path, manifest, [], [], [])
        log.exception("training failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
