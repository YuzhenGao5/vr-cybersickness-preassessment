#!/usr/bin/env python3
"""Independently audit round-2 preparation and fixed OOF predictions; never fit.

The checker does not import either preparer or trainer. Participant-bootstrap
increments are descriptive comparisons of already-produced OOF predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
NAME = "auc_signal_round2_2026-10-10"
IDS = ["game_name", "session_id", "participant_id", "rating_event_id"]
META = [*IDS, "ms_rating"]
UNKNOWN = {"", "unknown", "nan", "none", "null", "<na>"}
FORBIDDEN = re.compile(r"rating|label|future|timestamp|event_id|session_id|participant_id|game_name|source_file|time_delta|matching_distance", re.I)
ELAPSED = {"exposure_elapsed_seconds", "elapsed_seconds", "elapsed_time_seconds"}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve(value, base=ROOT):
    path = Path(value)
    return path if path.is_absolute() else base / path


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def truth(value):
    return str(value).lower() in {"true", "1", "1.0", "yes"}


def people(frame):
    return {str(v) for v in frame["participant_id"].dropna() if str(v).lower() not in UNKNOWN}


class Audit:
    def __init__(self):
        self.checks = []
        self.warnings = []
        self.details = {}

    def check(self, condition, name, detail=None, fatal=False):
        self.checks.append(dict(name=name, passed=bool(condition), detail=clean(detail)))
        if not condition and fatal:
            raise ValueError(name + (": " + str(detail) if detail is not None else ""))

    def close(self, actual, expected, name, tolerance=1e-9):
        if expected is None:
            ok = actual is None or (isinstance(actual, (float, np.floating)) and np.isnan(actual))
        elif isinstance(expected, (bool, np.bool_)):
            ok = truth(actual) == bool(expected)
        else:
            try:
                ok = bool(np.isfinite(float(actual)) and np.isclose(float(actual), float(expected), rtol=tolerance, atol=tolerance))
            except (ValueError, TypeError):
                ok = False
        self.check(ok, name, None if ok else dict(actual=actual, expected=expected))


def read_csv(path):
    return pd.read_csv(path, dtype={k: str for k in IDS}, encoding="utf-8-sig")


def read_identity(path, audit, name):
    frame = read_csv(path)
    audit.check(set(META) <= set(frame), name + ": metadata columns", fatal=True)
    audit.check(frame[IDS].notna().all().all() and frame.rating_event_id.is_unique, name + ": complete unique IDs", fatal=True)
    frame["ms_rating"] = pd.to_numeric(frame.ms_rating, errors="coerce")
    audit.check(frame.ms_rating.isin([1, 2, 3, 4, 5]).all(), name + ": unchanged 1..5 scores", fatal=True)
    audit.check(not frame.groupby("session_id").game_name.nunique().gt(1).any(), name + ": game identity", fatal=True)
    audit.check(not frame.groupby("session_id").participant_id.nunique().gt(1).any(), name + ": participant identity", fatal=True)
    return frame


def same_identity(frame, reference, audit, name):
    audit.check(set(frame.rating_event_id) == set(reference.rating_event_id) and frame.rating_event_id.is_unique,
                name + ": exact frozen event IDs", fatal=True)
    aligned = frame.set_index("rating_event_id", drop=False).loc[reference.rating_event_id].reset_index(drop=True)
    audit.check(all(np.array_equal(aligned[k].to_numpy(), reference[k].to_numpy()) for k in META),
                name + ": exact frozen metadata/labels", fatal=True)
    return aligned


def optional_hash(path, expected, audit, name):
    if expected:
        audit.check(digest(path) == expected, name + ": SHA256", fatal=True)


def check_preparation(cfg, input_dir, inventory_path, audit):
    reference_path = resolve(cfg["cohort_reference_csv"])
    optional_hash(reference_path, cfg.get("cohort_reference_sha256"), audit, "frozen reference")
    reference = read_identity(reference_path, audit, "reference")[META].sort_values("rating_event_id").reset_index(drop=True)
    audit.check(len(reference) == 1024 == cfg.get("expected_cohort_events", 1024), "frozen 1024 events", fatal=True)
    preparation_path = resolve(cfg["preparation_manifest"])
    preparation = json.loads(preparation_path.read_text(encoding="utf-8-sig"))
    audit.check(preparation.get("status") == "complete", "preparation complete", fatal=True)
    audit.check(preparation["config_sha256"] == digest(audit.details["config_path"]), "preparation frozen config hash", fatal=True)
    optional_hash(preparation_path, cfg.get("preparation_sha256"), audit, "preparation manifest")
    events_path = input_dir / "events.csv"
    optional_hash(events_path, preparation.get("events_sha256"), audit, "prepared events")
    optional_hash(reference_path, preparation.get("cohort_sha256"), audit, "preparation reference")
    events = same_identity(read_identity(events_path, audit, "prepared events"), reference, audit, "prepared events")
    audit.check(events["common_eligible"].map(truth).all(), "all frozen events common_eligible", fatal=True)
    for name in ["binary_label", "binary_true"]:
        if name in events:
            audit.check(np.array_equal(events[name].to_numpy(), (events.ms_rating > 2).astype(int).to_numpy()), "prepared " + name)
    frames = {}
    for method, spec in cfg["methods"].items():
        path = resolve(spec["input"], input_dir)
        original = read_identity(path, audit, method + " input")
        frame = same_identity(original, reference, audit, method)
        cols, logs = spec["feature_columns"], spec.get("log1p_columns", [])
        audit.check(len(cols) == len(set(cols)) and set(cols) <= set(frame), method + ": unique available features", fatal=True)
        prohibited = [c for c in cols if c not in ELAPSED and (FORBIDDEN.search(c) or c.startswith(("nearest_", "coverage")) or c.endswith(("_coverage", "_valid_duration")) or c in {"binary_true", "common_eligible"})]
        audit.check(not prohibited, method + ": no identity/target/QC/post-report inputs", prohibited, fatal=True)
        values = frame[cols].apply(pd.to_numeric, errors="coerce")
        audit.check(not np.isinf(values.to_numpy(dtype=float)).any(), method + ": no infinite feature values")
        audit.check(values.notna().any(axis=1).all(), method + ": every event has finite feature", fatal=True)
        audit.check(set(logs) <= set(cols) and all(not values[c].dropna().lt(0).any() for c in logs), method + ": registered log1p nonnegative", fatal=True)
        frame[cols] = values
        frames[method] = frame
    quality_path = input_dir / "support_quality.csv"
    optional_hash(quality_path, preparation.get("support_quality_sha256"), audit, "support quality")
    quality = read_csv(quality_path)
    audit.check(quality.rating_event_id.is_unique and set(quality.rating_event_id) == set(reference.rating_event_id), "QC exact frozen IDs", fatal=True)
    quality = quality.set_index("rating_event_id").loc[reference.rating_event_id]
    audit.check(np.array_equal(quality.session_id.to_numpy(), reference.session_id.to_numpy()), "QC session identity")
    elapsed = pd.to_numeric(events["exposure_elapsed_seconds"], errors="coerce").to_numpy(dtype=float)
    audit.check(np.isfinite(elapsed).all() and (elapsed > 0).all(), "positive finite elapsed", fatal=True)
    coverage = [c for c in quality if c.startswith("coverage") or c.endswith("_coverage")]
    durations = [c for c in quality if c.endswith("_valid_duration")]
    coverage_stats = {}
    for c in coverage:
        x = pd.to_numeric(quality[c], errors="coerce").to_numpy(dtype=float)
        audit.check(np.isfinite(x).all() and ((x >= -1e-6) & (x <= 1 + 1e-6)).all(), c + ": coverage in [0,1]", dict(min=np.nanmin(x), max=np.nanmax(x)))
        coverage_stats[c] = dict(min=np.nanmin(x), median=np.nanmedian(x), p90=np.nanquantile(x, .9), max=np.nanmax(x))
    for c in durations:
        x = pd.to_numeric(quality[c], errors="coerce").to_numpy(dtype=float)
        audit.check(np.isfinite(x).all() and (x >= -1e-6).all() and (x <= elapsed + 1e-5).all(), c + ": duration <= elapsed", dict(max_duration_minus_elapsed=np.nanmax(x - elapsed)))
        prefix = c.removesuffix("_valid_duration")
        if prefix + "_coverage" in quality:
            audit.check(np.allclose(x / elapsed, quality[prefix + "_coverage"], rtol=1e-8, atol=1e-7), prefix + ": coverage=duration/elapsed")
        if prefix + "_integral" in events and prefix + "_mean" in events:
            finite = x > 0
            audit.check(np.allclose(events.loc[finite, prefix + "_integral"].to_numpy() / x[finite], events.loc[finite, prefix + "_mean"], rtol=1e-8, atol=1e-8), prefix + ": integral/duration=mean")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8-sig"))
    optional_hash(inventory_path, cfg.get("source_inventory_sha256"), audit, "source inventory")
    sources = {r["session_id"]: r for r in inventory["sources"]}
    sessions = preparation.get("sessions_metadata", [])
    audit.check(len(sessions) == reference.session_id.nunique() and {m["session_id"] for m in sessions} == set(reference.session_id), "all cohort sessions prepared exactly once", fatal=True)
    counts = reference.groupby("session_id").size()
    for meta in sessions:
        sid, source = meta["session_id"], sources[meta["session_id"]]
        audit.check(meta["rows"] == source["raw_rows_expected"] and meta["real_reports"] == source["rating_events_expected"], sid + ": raw rows/reports match canonical inventory", dict(rows=meta["rows"], reports=meta["real_reports"]))
        audit.check(meta["events"] == counts[sid] and meta["status"] == "complete", sid + ": cohort events/status")
        identity = meta["identity"]
        audit.check(all(identity[k] == source[k] for k in ["source_locator", "known_sha256", "size_bytes"]), sid + ": source identity")
        audit.check(identity["config_sha256"] == digest(audit.details["config_path"]), sid + ": frozen cache config")
        p = input_dir / "by_session" / (sid + ".csv")
        q = input_dir / "by_session" / (sid + "_qc.csv")
        optional_hash(p, meta.get("output_sha256"), audit, sid + " feature cache")
        optional_hash(q, meta.get("qc_sha256"), audit, sid + " QC cache")
    audit.details.update(preparation=dict(events=len(reference), sessions=reference.session_id.nunique(), games=reference.game_name.nunique(), participants=reference.participant_id.nunique(), raw_rows=sum(m["rows"] for m in sessions), raw_reports=sum(m["real_reports"] for m in sessions), coverage_summary=coverage_stats), preparation_path=str(preparation_path), events_path=str(events_path))
    audit.warnings.append("Raw temporal intervals are not saved; coverage/duration invariants and source counts are checked without rereading large source CSVs. Strict endpoint timing remains supported by preparer static review, not independent full raw re-extraction.")
    reference["binary_true"] = (reference.ms_rating > 2).astype(int)
    return reference, frames, preparation


def compute_metrics(rows):
    y = rows.binary_true.to_numpy(dtype=int)
    p = rows.probability.to_numpy(dtype=float)
    both = np.unique(y).size == 2
    pred = (p >= .5).astype(int)
    return dict(events=len(rows), sessions=rows.session_id.nunique(), games=rows.game_name.nunique(), positive_events=int(y.sum()), negative_events=int((1-y).sum()), prevalence=float(y.mean()), binary_auc=float(roc_auc_score(y, p)) if both else None, average_precision=float(average_precision_score(y, p)) if both else None, accuracy=float(accuracy_score(y, pred)), f1_macro=float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)), brier=float(brier_score_loss(y, p)), mean_probability=float(p.mean()), mean_probability_positive=float(p[y == 1].mean()) if (y == 1).any() else None, mean_probability_negative=float(p[y == 0].mean()) if (y == 0).any() else None, complete_oof=True)


def compare_metrics(actual, expected, audit, context):
    for key, value in expected.items():
        audit.check(key in actual, context + ": field " + key, fatal=True)
        audit.close(actual[key], value, context + ": " + key)


def expected_inner(train, purge):
    games = train.game_name.to_numpy()
    n = len(np.unique(games))
    if n < 2:
        return []
    out = []
    for fold, (a, b) in enumerate(GroupKFold(n_splits=min(3, n)).split(train, train.binary_true, games), 1):
        before, validation = train.iloc[a], train.iloc[b]
        if purge:
            a = a[~before.participant_id.isin(people(validation)).to_numpy()]
        retained = train.iloc[a]
        usable = len(retained) > 0 and retained.binary_true.nunique() == 2
        valid = [str(g) for g, part in validation.groupby("game_name", sort=True) if usable and part.binary_true.nunique() == 2]
        out.append(dict(inner_fold=fold, training_events_before_purge=len(before), training_events=len(retained), purged_events=len(before)-len(retained), validation_events=len(validation), validation_games=sorted(validation.game_name.unique()), valid_auc_games=valid, shared_participants_before_purge=sorted(people(before)&people(validation)), shared_participants_after_purge=sorted(people(retained)&people(validation))))
    return out


def check_training_audits(cfg, cohort, frames, manifest, purge, audit):
    key = "double_holdout_training_audit" if purge else "fold_training_audit"
    all_records = manifest[key]
    records = []
    # Original-pair audit records omit a top-level model field. Their family is
    # explicitly encoded by choice.candidate_id; certify that ID and parameter
    # combination against the manifest's frozen grid before normalizing locally.
    normalized = 0
    for original in all_records:
        if original["method"] == "nested_selected":
            continue
        record = dict(original)
        choice = record["choice"]
        index = int(choice["candidate_index"])
        matches = [kind for kind, grid in manifest["fixed_candidate_grids"].items()
                   if 0 <= index < len(grid)
                   and choice["candidate_id"] == f"{kind}_{index:02d}"
                   and choice["candidate"] == grid[index]]
        audit.check(len(matches) == 1, key + ": exact candidate ID/grid family", dict(candidate_id=choice["candidate_id"], matches=matches), fatal=True)
        if "model" in record:
            audit.check(record["model"] == matches[0], key + ": recorded family agrees with candidate ID", fatal=True)
        else:
            record["model"] = matches[0]
            normalized += 1
        records.append(record)
    audit.details.setdefault("schema_compatibility", {})[key] = dict(normalized_family_records=normalized, family_source="choice.candidate_id and exact candidate parameters, validated against fixed_candidate_grids; training manifest unchanged")
    selector_records = [r for r in all_records if r["method"] == "nested_selected"]
    audit.check(len(selector_records) == cohort.game_name.nunique(), key + ": selector audit for every fold")
    audit.check(all(r.get("prediction_reused_without_fit") is True for r in selector_records), key + ": selector has no extra fit")
    definitions = cfg.get("double_holdout", {}) if purge else {}
    methods = definitions.get("methods", list(cfg["methods"]))
    kinds = definitions.get("models", cfg["models"])
    expected = {(method, kind, i, game) for i, game in enumerate(sorted(cohort.game_name.unique()), 1) for method in methods for kind in kinds}
    actual = {(r["method"], r["model"], int(r["fold"]), r["held_out_game"]) for r in records}
    audit.check(len(records) == len(expected) and actual == expected, key + ": every outer model/fold audited", fatal=True)
    for r in records:
        context = f"{'double' if purge else 'primary'}/{r['method']}/{r['model']}/{r['held_out_game']}"
        test = cohort[cohort.game_name.eq(r["held_out_game"])]
        before = cohort[~cohort.game_name.eq(r["held_out_game"])]
        train = before[~before.participant_id.isin(people(test))] if purge else before
        details = r["outer_split"]
        want = dict(training_events_before_purge=len(before), training_events=len(train), purged_training_events=len(before)-len(train), participant_overlap_before_purge=len(people(before)&people(test)), participant_overlap_after_purge=len(people(train)&people(test)))
        compare_metrics(details, want, audit, context + " split")
        if purge:
            audit.check(not people(train)&people(test), context + ": no shared participants")
        for name, event_frame in [("train_event_ids", train), ("test_event_ids", test)]:
            if name in details:
                audit.check(set(details[name]) == set(event_frame.rating_event_id), context + ": recorded " + name)
        audit.check(r["choice"].get("inner_plan", []) == [dict(x, status="usable" if x["valid_auc_games"] else "unavailable", reason=None if x["valid_auc_games"] else "single-class/empty inner train or single-class validation games") for x in expected_inner(train, purge)], context + ": independent inner game/participant splits")
        detail = r["preprocessing"]
        compare_metrics(detail, dict(training_events=len(train), training_sessions=train.session_id.nunique(), training_games=train.game_name.nunique(), training_positive_events=int(train.binary_true.sum())), audit, context + " fitted training counts")
        registered = cfg["methods"][r["method"]]["feature_columns"]
        audit.check(detail["requested_feature_columns"] == registered, context + ": registered features")
        train_features = frames[r["method"]].set_index("rating_event_id", drop=False).loc[train.rating_event_id]
        usable = [c for c in registered if train_features[c].notna().any()]
        audit.check(detail["input_feature_cols"] == usable and detail["dropped_all_missing"] == [c for c in registered if c not in usable], context + ": train-only empty-feature removal")
        counts = train.groupby("session_id")["rating_event_id"].transform("size").to_numpy(dtype=float)
        weights = 1 / counts
        weights /= weights.mean()
        y = train.binary_true.to_numpy(dtype=int)
        audit.close(detail["training_prevalence"], float(np.average(y, weights=weights)), context + ": train-only prior")
        if detail["status"] == "fitted":
            logs = [c for c in cfg["methods"][r["method"]].get("log1p_columns", []) if c in usable]
            transformed = logs + [c for c in usable if c not in logs]
            matrix = train_features[transformed].fillna(train_features[transformed].median()).to_numpy(dtype=float)
            matrix[:, :len(logs)] = np.log1p(matrix[:, :len(logs)])
            active = [c for c, var in zip(transformed, np.var(matrix, axis=0)) if var > 1e-10]
            audit.check(detail["fitted_feature_cols"] == active, context + ": independent train-only median/log1p/variance columns")
        choice = r["choice"]
        grids = manifest["fixed_candidate_grids"][r["model"]]
        index = int(choice["candidate_index"])
        audit.check(choice["candidate"] == grids[index] and choice["candidate_id"] == f"{r['model']}_{index:02d}", context + ": registered candidate")
    return records


def participant_increment(predictions, draws):
    outputs = []
    for (method, kind), rows in predictions.groupby(["method", "model"], sort=False):
        if method in {"baseline", "elapsed_only"}:
            continue
        family = kind if method != "nested_selected" else None
        if family is None:
            # A family-varying selector has no single family matched elapsed baseline.
            families = ["logistic", "lightgbm", "extra_trees"]
        else:
            families = [family]
        for base_kind in families:
            baseline = predictions[predictions.method.eq("elapsed_only") & predictions.model.eq(base_kind)]
            if baseline.empty:
                continue
            base = baseline.set_index("rating_event_id").loc[rows.rating_event_id]
            y = rows.binary_true.to_numpy(dtype=int)
            p, b = rows.probability.to_numpy(dtype=float), base.probability.to_numpy(dtype=float)
            clusters = list(rows.reset_index(drop=True).groupby("participant_id", sort=True).indices.values())
            rng, differences = np.random.default_rng(42), []
            for _ in range(draws):
                selected = np.concatenate([clusters[i] for i in rng.integers(0, len(clusters), len(clusters))])
                if np.unique(y[selected]).size == 2:
                    differences.append(float(roc_auc_score(y[selected], p[selected])-roc_auc_score(y[selected], b[selected])))
            model_auc, baseline_auc = roc_auc_score(y, p), roc_auc_score(y, b)
            per_game = []
            for game, part in rows.groupby("game_name", sort=True):
                if part.binary_true.nunique() == 2:
                    indices = rows.reset_index(drop=True).groupby("game_name").indices[game]
                    per_game.append(dict(game_name=game, auc_difference=float(roc_auc_score(y[indices], p[indices])-roc_auc_score(y[indices], b[indices]))))
            outputs.append(dict(protocol=str(rows.protocol.iloc[0]), method=method, model=kind, elapsed_baseline_model=base_kind, comparison="same-family elapsed_only" if family else "selector versus separately reported elapsed family; not a selected best baseline", events=len(rows), participant_clusters=len(clusters), binary_auc=float(model_auc), elapsed_binary_auc=float(baseline_auc), auc_difference=float(model_auc-baseline_auc), mean_valid_game_auc_difference=float(np.mean([x["auc_difference"] for x in per_game])) if per_game else None, difference_ci_low=float(np.percentile(differences, 2.5)) if differences else None, difference_ci_high=float(np.percentile(differences, 97.5)) if differences else None, bootstrap_draws=draws, valid_draws=len(differences), per_game_difference=per_game, interpretation="paired participant-cluster bootstrap of fixed OOF; conditional on observed games; no refit or method-selection uncertainty; association not causal"))
    return outputs


def check_protocol(cfg, cohort, frames, manifest, results_dir, purge, audit, draws):
    prefix = "double_holdout_" if purge else ""
    protocol = "leave_game_and_participant_out" if purge else "leave_game_out"
    predictions = read_csv(results_dir / (prefix + "oof_predictions.csv"))
    metrics = read_csv(results_dir / (prefix + "metrics.csv"))
    folds = read_csv(results_dir / (prefix + "fold_metrics.csv"))
    specification = cfg.get("double_holdout", {}) if purge else {}
    methods, kinds = specification.get("methods", list(cfg["methods"])), specification.get("models", cfg["models"])
    combinations = {(m, k) for m in methods for k in kinds} | {("baseline", "train_prevalence")}
    if (predictions.method.eq("nested_selected") & predictions.model.eq("selected")).any():
        combinations.add(("nested_selected", "selected"))
    audit.check(set(zip(predictions.method, predictions.model)) == combinations, protocol + ": all registered combinations", fatal=True)
    audit.check(set(predictions.protocol) == {protocol}, protocol + ": protocol identity", fatal=True)
    audit.check("nested_pair_selector" in manifest and manifest["nested_pair_selector"].get("planned_before_any_round2_metric") is True, protocol + ": selector preplanned", fatal=True)
    audit.check(set(zip(metrics.method, metrics.model)) == combinations and len(metrics) == len(combinations), protocol + ": summary combinations", fatal=True)
    games = sorted(cohort.game_name.unique())
    audit.check(len(folds) == len(combinations)*len(games) and not folds.duplicated(["method", "model", "fold"]).any(), protocol + ": every fold metric", fatal=True)
    records = check_training_audits(cfg, cohort, frames, manifest, purge, audit)
    by_pair = {(r["method"], r["model"], r["held_out_game"]): r for r in records}
    for (method, kind), raw in predictions.groupby(["method", "model"], sort=False):
        context = f"{protocol}/{method}/{kind}"
        rows = same_identity(raw, cohort, audit, context)
        p = rows.probability.to_numpy(dtype=float)
        audit.check(np.isfinite(p).all() and ((p >= 0)&(p <= 1)).all(), context + ": complete [0,1] probabilities", fatal=True)
        audit.check(np.array_equal(rows.binary_true.to_numpy(), cohort.binary_true.to_numpy()), context + ": original >2 target")
        audit.check(np.array_equal(rows.predicted_class.to_numpy(), (p >= .5).astype(int)), context + ": fixed 0.5 classes")
        computed = compute_metrics(rows)
        game_metrics = []
        for i, game in enumerate(games, 1):
            part = rows[rows.game_name.eq(game)]
            audit.check(part.fold.eq(i).all() and part.held_out_game.eq(game).all(), context + ": heldout " + game)
            values = compute_metrics(part)
            observed = folds[folds.method.eq(method)&folds.model.eq(kind)&folds.fold.eq(i)]
            audit.check(len(observed) == 1, context + ": unique fold row", fatal=True)
            compare_metrics(observed.iloc[0].to_dict(), values, audit, context + "/" + game)
            game_metrics.append(values)
            if method == "baseline":
                training = cohort[~cohort.game_name.eq(game)]
                if purge:
                    training = training[~training.participant_id.isin(people(part))]
                counts = training.groupby("session_id")["rating_event_id"].transform("size").to_numpy(dtype=float)
                prior = float(np.average(training.binary_true, weights=1/counts))
                audit.check(np.allclose(part.probability, prior, rtol=1e-12, atol=1e-12), context + ": independently reconstructed training prior " + game)
            if method == "nested_selected":
                scores = [by_pair[(m, k, game)] for m in methods for k in kinds]
                selected = max(range(len(scores)), key=lambda j: (scores[j]["choice"].get("inner_mean_game_auc") if scores[j]["choice"].get("inner_mean_game_auc") is not None else -np.inf, -j))
                chosen = scores[selected]
                audit.check(part.selected_input_method.eq(chosen["method"]).all() and part.selected_model_family.eq(chosen["model"]).all(), context + ": inner-only pair selection " + game)
                original = predictions[predictions.method.eq(chosen["method"])&predictions.model.eq(chosen["model"])&predictions.game_name.eq(game)].set_index("rating_event_id").loc[part.rating_event_id]
                audit.check(np.array_equal(part.probability.to_numpy(), original.probability.to_numpy()), context + ": selector reuses chosen OOF " + game)
        valid_auc = [x["binary_auc"] for x in game_metrics if x["binary_auc"] is not None]
        valid_ap = [x["average_precision"] for x in game_metrics if x["average_precision"] is not None]
        computed.update(mean_valid_fold_auc=float(np.mean(valid_auc)) if valid_auc else None, valid_auc_folds=len(valid_auc), total_folds=len(games), mean_valid_fold_average_precision=float(np.mean(valid_ap)) if valid_ap else None)
        compare_metrics(metrics[metrics.method.eq(method)&metrics.model.eq(kind)].iloc[0].to_dict(), computed, audit, context + " summary")
    candidates = read_csv(results_dir / (prefix + "inner_candidate_scores.csv"))
    inner_games = read_csv(results_dir / (prefix + "inner_game_scores.csv"))
    for record in records:
        method, kind, fold = record["method"], record["model"], int(record["fold"])
        subset = candidates[candidates.method.eq(method)&candidates.model.eq(kind)&candidates.fold.eq(fold)].sort_values("candidate_index")
        grid = manifest["fixed_candidate_grids"][kind]
        audit.check(len(subset) == len(grid) and subset.candidate_index.tolist() == list(range(len(grid))), f"{protocol}/{method}/{kind}/{fold}: full candidate grid", fatal=True)
        scores = []
        for _, row in subset.iterrows():
            index = int(row.candidate_index)
            audit.check(json.loads(row.parameters_json) == grid[index], "registered inner candidate parameters")
            detail = inner_games[inner_games.method.eq(method)&inner_games.model.eq(kind)&inner_games.fold.eq(fold)&inner_games.candidate_index.eq(index)]
            wanted = {(x["inner_fold"], g) for x in record["choice"]["inner_plan"] for g in x["valid_auc_games"]}
            audit.check(set(zip(detail.inner_fold, detail.validation_game)) == wanted and len(detail) == len(wanted), "same inner validation games for every candidate")
            mean = float(detail.inner_auc.mean()) if wanted else None
            audit.close(row.inner_mean_game_auc, mean, "inner mean is per-game AUC mean")
            scores.append(mean if mean is not None else -np.inf)
        selected = max(range(len(scores)), key=lambda j: (scores[j], -j))
        flags = subset.selected.map(truth).to_numpy()
        audit.check(flags.sum() == 1 and flags[selected] and record["choice"]["candidate_index"] == selected, f"{protocol}/{method}/{kind}/{fold}: inner-only max/tie/fallback selection")
    return participant_increment(predictions, draws)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / (NAME + ".json"))
    parser.add_argument("--input-dir", type=Path, default=ROOT / "features" / NAME)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / NAME)
    parser.add_argument("--source-inventory", type=Path, default=ROOT / "results/param_rating_local_all_2026-10-10/source_inventory.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--preparation-only", action="store_true")
    args = parser.parse_args()
    output = args.output or args.results_dir / "independent_check.json"
    audit = Audit()
    audit.details["config_path"] = str(args.config.resolve())
    increments = []
    try:
        if args.bootstrap_draws < 0:
            raise ValueError("bootstrap draws must be nonnegative")
        cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))
        cohort, frames, preparation = check_preparation(cfg, args.input_dir, args.source_inventory, audit)
        if not args.preparation_only:
            manifest_path = args.results_dir / "training_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            audit.check(manifest.get("status") == "complete", "training manifest complete", fatal=True)
            optional_hash(args.config, manifest.get("config_sha256"), audit, "training config")
            optional_hash(Path(__file__).with_name("train_auc_signal_round2.py"), manifest.get("training_script_sha256"), audit, "training script")
            audit.check(manifest["fixed_candidate_grids"] == {k: cfg["fixed_candidate_grids"][k] for k in cfg["models"]}, "training grid registration", fatal=True)
            same_identity(read_identity(args.results_dir / "cohort.csv", audit, "training cohort"), cohort, audit, "training cohort")
            optional_hash(args.results_dir / "cohort.csv", manifest.get("cohort_sha256"), audit, "training cohort")
            increments.extend(check_protocol(cfg, cohort, frames, manifest, args.results_dir, False, audit, args.bootstrap_draws))
            if cfg.get("double_holdout", {}).get("enabled"):
                increments.extend(check_protocol(cfg, cohort, frames, manifest, args.results_dir, True, audit, args.bootstrap_draws))
            audit.warnings.append("Actual training IDs are verified only when recorded. Otherwise purge membership/counts and inner split plans are independently reconstructed from frozen cohort and compared to manifest; fitted fold pipelines/raw inner predictions are not saved, so their internal execution cannot be proved solely from outputs.")
            audit.details["training_manifest_sha256"] = digest(manifest_path)
    except Exception as exc:
        audit.check(False, "checker halted", dict(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()))
    failures = [x for x in audit.checks if not x["passed"]]
    result = clean(dict(status="passed" if not failures else "failed", checker_path=str(Path(__file__).resolve()), checker_sha256=digest(__file__), preparation_only=args.preparation_only, checks_passed=len(audit.checks)-len(failures), checks_failed=len(failures), failures=failures, checks=audit.checks, details=audit.details, warnings=audit.warnings, paired_participant_bootstrap_vs_elapsed= increments))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({k:result[k] for k in ["status", "checks_passed", "checks_failed", "failures"]}, ensure_ascii=False))
    print(str(output.resolve()))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
