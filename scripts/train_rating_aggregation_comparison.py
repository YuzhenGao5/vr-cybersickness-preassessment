#!/usr/bin/env python3
"""Compare preregistered event-level rating feature aggregations on one cohort."""
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
import scipy
import sklearn
from lightgbm import LGBMRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_param_rating_model import (
    BINARY_SCORE_THRESHOLD, MODEL_PARAMETERS, ROOT, VARIANCE_THRESHOLD,
    clean_json, digest, format_value, measures, save_json, table,
)
from train_param_rating_local_all import session_bootstrap, weighted_median

NAME = "rating_aggregation_comparison_2026-10-10"
IDS = ["game_name", "session_id", "participant_id", "rating_event_id"]
METADATA = [*IDS, "ms_rating"]
RIDGE_PARAMETERS = dict(alpha=10.0)
LIGHTGBM_PARAMETERS = dict(MODEL_PARAMETERS, objective="regression")
ELAPSED_FEATURES = {"exposure_elapsed_seconds", "elapsed_seconds", "elapsed_time_seconds"}
FORBIDDEN_PATTERN = re.compile(r"(?:rating|label|future|timestamp|event_id|session_id|participant_id|game_name|source_file|time_delta|matching_distance)", re.I)
DOUBLE_HOLDOUT_METHODS = ["multi_scene", "multi_user", "multi_joint", "multi_scene_head"]
DESCRIPTIONS = {
    "nearest_mean": "最近评分分组内原测量池的六项均值（诊断）",
    "nearest_pooled": "最近评分分组内原测量池的均值/最大值/标准差（诊断）",
    "past5_scene": "评分前5秒场景运行统计",
    "past30_scene": "评分前30秒场景运行统计",
    "past60_scene": "评分前60秒场景运行统计",
    "past30_mean_only": "评分前30秒场景运行六项均值",
    "multi_scene": "评分前5/30/60秒场景统计及累计暴露时间",
    "multi_user": "评分前5/30/60秒头动/控制器统计及累计暴露时间",
    "multi_joint": "评分前5/30/60秒场景及用户统计、累计暴露时间",
    "multi_joint_noelapsed": "评分前5/30/60秒场景及用户统计",
    "multi_scene_head": "评分前5/30/60秒场景及头动统计、累计暴露时间，无控制器",
    "elapsed_only": "仅累计体验秒数",
}


def train_weights(frame):
    # One row per rating event: each session has the same total training weight.
    counts = frame.groupby("session_id")["rating_event_id"].transform("size").to_numpy(dtype=float)
    weights = 1.0 / counts
    return weights / weights.mean()


def registered_features(definition):
    columns = definition.get("feature_columns")
    if not isinstance(columns, list) or not columns or len(columns) != len(set(columns)):
        raise ValueError("each method requires a nonempty unique feature_columns list")
    forbidden = [name for name in columns if name not in ELAPSED_FEATURES and
                 (name in METADATA or FORBIDDEN_PATTERN.search(name) or name in {
                     "start_time_sec", "end_time_sec", "window_center_sec", "window_id", "window_has_any_feature", "dataset_family"})]
    if forbidden:
        raise ValueError(f"identity/target/matching-time inputs forbidden: {forbidden}")
    return columns


def load_methods(cfg, input_dir):
    methods = cfg["methods"]
    if not isinstance(methods, dict) or not methods:
        raise ValueError("config.methods must map method name to input and feature_columns")
    frames, audits, identities = {}, {}, []
    for method, definition in methods.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", method):
            raise ValueError(f"method name must be a safe filename component: {method}")
        feature_cols = registered_features(definition)
        input_path = Path(definition["input"])
        if not input_path.is_absolute():
            input_path = input_dir / input_path
        frame = pd.read_csv(input_path, dtype={name: str for name in IDS}, encoding="utf-8-sig")
        missing = sorted(set(METADATA + feature_cols + ["common_eligible"]) - set(frame.columns))
        if missing:
            raise ValueError(f"{method}: missing columns {missing}")
        if frame["rating_event_id"].isna().any() or not frame["rating_event_id"].is_unique:
            raise ValueError(f"{method}: one unique nonmissing row per rating_event_id required")
        if frame[["session_id", "game_name"]].isna().any().any():
            raise ValueError(f"{method}: missing game/session identity")
        frame["ms_rating"] = pd.to_numeric(frame["ms_rating"], errors="coerce")
        frame[feature_cols] = frame[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        # Pairing eligibility is established before producing any model metric.
        common_eligible = frame["common_eligible"].astype(str).str.lower().isin({"true", "1", "1.0", "yes"})
        eligible = common_eligible & np.isfinite(frame["ms_rating"]) & frame[feature_cols].notna().any(axis=1)
        identities.append(frame.loc[np.isfinite(frame["ms_rating"]), METADATA])
        frames[method] = frame.loc[eligible].set_index("rating_event_id", drop=False)
        audits[method] = dict(input_path=str(input_path.resolve()), input_sha256=digest(input_path),
                              input_rows=len(frame), eligible_events=int(eligible.sum()),
                              excluded_nonfinite_target=int((~np.isfinite(frame["ms_rating"])).sum()),
                              excluded_all_missing_features=int((~frame[feature_cols].notna().any(axis=1)).sum()),
                              excluded_preparation_common_eligible=int((~common_eligible).sum()),
                              requested_feature_columns=feature_cols,
                              description=definition.get("description", DESCRIPTIONS.get(method, method)),
                              temporal_policy=definition.get("temporal_policy", "含评分后测量，诊断性关联" if method.startswith("nearest_") else "严格评分前历史；累计暴露对照见注册配置"))
    identities = pd.concat(identities, ignore_index=True)
    for name in ["ms_rating", "game_name", "session_id", "participant_id"]:
        if identities.groupby("rating_event_id", dropna=False)[name].nunique(dropna=False).gt(1).any():
            raise ValueError(f"shared rating_event_id has inconsistent {name} across methods")
    if identities.groupby("session_id")["game_name"].nunique().gt(1).any():
        raise ValueError("session has different games across methods")
    common = set.intersection(*(set(frame.index) for frame in frames.values()))
    ordered_ids = sorted(common)
    frames = {name: frame.loc[ordered_ids].reset_index(drop=True) for name, frame in frames.items()}
    cohort = next(iter(frames.values()))[METADATA].copy()
    for method, audit in audits.items():
        audit["common_cohort_events"] = len(cohort)
        audit["eligible_events_outside_common_cohort"] = audit["eligible_events"] - len(cohort)
    return frames, cohort, audits


def delta_bootstrap(rows, reference, draws=1000, seed=42):
    """Paired session-group MAE difference; avoids recomputing Spearman."""
    if draws <= 0:
        return dict(status="disabled")
    ref = reference.set_index("rating_event_id").loc[rows["rating_event_id"]]
    y = rows["ms_rating"].to_numpy(dtype=float)
    if not np.array_equal(y, ref["ms_rating"].to_numpy(dtype=float)):
        raise ValueError("reference score identity mismatch")
    pred = rows["prediction"].to_numpy(dtype=float)
    baseline = ref["prediction"].to_numpy(dtype=float)
    if not np.isfinite(pred).all() or not np.isfinite(baseline).all():
        return dict(status="unavailable", reason="incomplete paired OOF")
    delta = np.abs(y - baseline) - np.abs(y - pred)
    group = pd.DataFrame({"session": rows["session_id"].to_numpy(), "delta": delta}).groupby("session")["delta"].agg(["sum", "count"]).to_numpy()
    rng = np.random.default_rng(seed)
    sampled = group[rng.integers(0, len(group), size=(draws, len(group)))]
    statistics = sampled[:, :, 0].sum(axis=1) / sampled[:, :, 1].sum(axis=1)
    return dict(status="complete", draws=draws, seed=seed,
                mae_improvement_over_nearest_mean=float(delta.mean()),
                nearest_mean_mae_improvement_ci_low=float(np.percentile(statistics, 2.5)),
                nearest_mean_mae_improvement_ci_high=float(np.percentile(statistics, 97.5)),
                method="paired session bootstrap of fixed OOF; matching model kind; conditional on observed games")


def legacy_reference(cfg, cohort, results_dir):
    source = cfg.get("legacy_oof_reference", "results/param_rating_local_all_2026-10-10/event_oof_predictions.csv")
    if not source:
        return dict(status="not_requested"), []
    path = Path(source)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        return dict(status="unavailable", source=str(path), reason="existing legacy OOF file not found"), []
    reference = pd.read_csv(path, dtype={name: str for name in IDS}, encoding="utf-8-sig")
    if "analysis" in reference:
        reference = reference.loc[reference["analysis"].eq("unlimited_nearest")]
    if "model" in reference:
        reference = reference.loc[reference["model"].eq("lightgbm")]
    if not reference["rating_event_id"].is_unique:
        raise ValueError("legacy reference event ids are not unique after fixed filtering")
    reference = reference.set_index("rating_event_id", drop=False)
    available = cohort["rating_event_id"].isin(reference.index)
    meta = cohort.loc[available].copy()
    aligned = reference.loc[meta["rating_event_id"]]
    if not np.array_equal(meta["ms_rating"].to_numpy(dtype=float), aligned["ms_rating"].to_numpy(dtype=float)):
        raise ValueError("legacy reference targets differ on shared event IDs")
    meta["prediction"] = aligned["prediction"].to_numpy(dtype=float)
    rows = [metric_row(meta, "legacy_window_training_reference", "lightgbm")] if len(meta) else []
    pd.DataFrame(rows).to_csv(results_dir / "legacy_reference_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    return dict(status="complete" if available.all() else "partial_cohort", source=str(path.resolve()), source_sha256=digest(path),
                matched_events=len(meta), common_cohort_events=len(cohort),
                interpretation="reference only: trained on windows and then averaged OOF predictions; not the same event-level training grain"), rows


def score_group_rows(predictions):
    rows = []
    for (method, kind, score), group in predictions.groupby(["method", "model", "ms_rating"], sort=False):
        predicted = group["prediction"].to_numpy(dtype=float)
        complete = np.isfinite(predicted).all()
        rows.append(dict(method=method, model=kind, actual_ms_rating=float(score), events=len(group),
                         available_predictions=int(np.isfinite(predicted).sum()),
                         mean_prediction=float(predicted.mean()) if complete else None,
                         median_prediction=float(np.median(predicted)) if complete else None,
                         mean_bias=float((predicted-score).mean()) if complete else None,
                         mae=float(np.abs(predicted-score).mean()) if complete else None,
                         rmse=float(np.sqrt(np.mean((predicted-score)**2))) if complete else None))
    return rows


def fit(frame, target, columns, kind):
    usable = [name for name in columns if frame[name].notna().any()]
    detail = dict(input_feature_cols=usable, fitted_feature_cols=[],
                  dropped_all_missing=[name for name in columns if name not in usable],
                  dropped_constant=[], status="unavailable")
    if not usable:
        detail["reason"] = "all feature columns missing in training fold"
        return None, detail
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(frame[usable])
    variance = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    try:
        X = variance.fit_transform(X)
    except ValueError:
        detail.update(reason="all training-fold feature columns constant", dropped_constant=usable)
        return None, detail
    active = np.array(usable)[variance.get_support()].tolist()
    steps = [("imputer", imputer), ("variance", variance)]
    if kind == "ridge":
        scale = StandardScaler()
        X = scale.fit_transform(X)
        steps.append(("scale", scale))
        model = Ridge(**RIDGE_PARAMETERS)
    else:
        model = LGBMRegressor(**LIGHTGBM_PARAMETERS)
    model.fit(X, target, sample_weight=train_weights(frame))
    steps.append(("regressor", model))
    detail.update(fitted_feature_cols=active,
                  dropped_constant=[name for name in usable if name not in active], status="fitted", reason=None)
    return Pipeline(steps), detail


def metric_row(frame, method, model, fold=None, game=None):
    row = dict(protocol="leave_game_out", method=method, model=model, fold=fold,
               held_out_game=game, events=len(frame), sessions=int(frame["session_id"].nunique()),
               games=int(frame["game_name"].nunique()), **measures(frame["ms_rating"], frame["prediction"]))
    complete = np.isfinite(frame["prediction"]).all()
    row["complete_oof"] = bool(complete)
    if complete:
        work = frame.assign(error=(frame["prediction"] - frame["ms_rating"]).abs())
        row.update(equal_session_mae=float(work.groupby("session_id")["error"].mean().mean()),
                   equal_game_mae=float(work.groupby("game_name")["error"].mean().mean()),
                   outside_observed_range=int(((frame["prediction"] < frame["ms_rating"].min()) | (frame["prediction"] > frame["ms_rating"].max())).sum()))
    return row


def double_holdout(frames, cohort, inputs, results_dir, log):
    """Preset sensitivity: game out and all known test participants purged."""
    requested = [name for name in DOUBLE_HOLDOUT_METHODS if name in frames]
    plan = dict(protocol="leave_game_and_test_participants_out", requested_methods=DOUBLE_HOLDOUT_METHODS,
                available_methods=requested, model="lightgbm", model_parameters=LIGHTGBM_PARAMETERS,
                selection="fixed in script before comparison model execution",
                training_rule="remove the whole held-out game, then every training event for participants present in that test game",
                outputs_separate_from_primary=True, folds=[])
    if not requested:
        plan.update(status="unavailable", reason="none of the predeclared sensitivity methods available")
        return plan, [], []
    y, games = cohort["ms_rating"].to_numpy(dtype=float), cohort["game_name"].to_numpy()
    outputs, folds = [], []
    for fold, (initial_train, test_idx) in enumerate(LeaveOneGroupOut().split(cohort, y, games), 1):
        test_meta = cohort.iloc[test_idx]
        test_people = set(test_meta["participant_id"].dropna()) - {"unknown", "", "nan"}
        retain = ~cohort.iloc[initial_train]["participant_id"].isin(test_people).to_numpy()
        train_idx = initial_train[retain]
        train_meta = cohort.iloc[train_idx]
        heldout = str(games[test_idx[0]])
        audit = dict(fold=fold, held_out_game=heldout, initial_train_events=len(initial_train),
                     retained_train_events=len(train_idx), removed_train_events=int((~retain).sum()),
                     test_events=len(test_idx), test_participant_count=len(test_people),
                     unknown_test_participant_events=int(test_meta["participant_id"].isna().sum() + test_meta["participant_id"].isin(["unknown", "", "nan"]).sum()),
                     train_games=int(train_meta["game_name"].nunique()),
                     train_target_unique=int(train_meta["ms_rating"].nunique()), models=[])
        if set(train_meta["session_id"]) & set(test_meta["session_id"]):
            raise ValueError("double holdout session leakage")
        if set(train_meta["participant_id"].dropna()) & test_people:
            raise ValueError("double holdout known participant leakage")
        available = len(train_idx) >= 2 and train_meta["ms_rating"].nunique() >= 2
        variants = []
        if available:
            weights = train_weights(train_meta)
            variants += [("baseline", "train_mean", np.full(len(test_idx), np.average(y[train_idx], weights=weights)), {}),
                         ("baseline", "train_median", np.full(len(test_idx), weighted_median(y[train_idx], weights)), {})]
        else:
            reason = "purged training set has fewer than two events or target values"
            audit.update(status="unavailable", reason=reason)
            variants += [("baseline", name, np.full(len(test_idx), np.nan), {"status": "unavailable", "reason": reason}) for name in ["train_mean", "train_median"]]
        for method in requested:
            if available:
                pipe, detail = fit(frames[method].iloc[train_idx], y[train_idx], inputs[method]["requested_feature_columns"], "lightgbm")
                pred = pipe.predict(frames[method].iloc[test_idx][detail["input_feature_cols"]]) if pipe is not None else np.full(len(test_idx), np.nan)
            else:
                detail = dict(status="unavailable", reason=audit["reason"], fitted_feature_cols=[], input_feature_cols=[])
                pred = np.full(len(test_idx), np.nan)
            audit["models"].append(dict(method=method, **detail))
            variants.append((method, "lightgbm", pred, detail))
        if available:
            audit["status"] = "complete" if all(item["status"] == "fitted" for item in audit["models"]) else "partial_feature_unavailable"
        plan["folds"].append(audit)
        for method, kind, pred, detail in variants:
            rows = test_meta.copy()
            rows["method"], rows["model"], rows["fold"], rows["held_out_game"] = method, kind, fold, heldout
            rows["protocol"] = plan["protocol"]
            rows["prediction"] = pred
            rows["absolute_error"] = (rows["prediction"] - rows["ms_rating"]).abs()
            outputs.append(rows)
            metric = metric_row(rows, method, kind, fold, heldout)
            metric.update(protocol=plan["protocol"], participant_overlap=0,
                          train_events_after_purge=len(train_idx), removed_train_events=audit["removed_train_events"],
                          training_available=available, fit_status=detail.get("status", "training_fold_constant"),
                          unavailable_reason=detail.get("reason"), fitted_feature_cols_json=json.dumps(detail.get("fitted_feature_cols", [])))
            folds.append(metric)
        log.info("double holdout %s/%s game=%s train=%s removed=%s test=%s status=%s",
                 fold, len(np.unique(games)), heldout, len(train_idx), audit["removed_train_events"], len(test_idx), audit["status"])
    predictions = pd.concat(outputs, ignore_index=True)
    summaries = []
    for (method, kind), rows in predictions.groupby(["method", "model"], sort=False):
        if len(rows) != len(cohort) or not rows["rating_event_id"].is_unique:
            raise ValueError("double holdout OOF cohort mismatch")
        metric = metric_row(rows, method, kind)
        matching = [row for row in folds if row["method"] == method and row["model"] == kind]
        valid_auc = [row["binary_auc"] for row in matching if row["binary_auc"] is not None]
        metric.update(protocol=plan["protocol"], mean_valid_fold_auc=float(np.mean(valid_auc)) if valid_auc else None,
                      valid_auc_folds=len(valid_auc), total_folds=len(matching))
        summaries.append(metric)
    predictions.to_csv(results_dir / "double_holdout_oof_predictions.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    pd.DataFrame(folds).to_csv(results_dir / "double_holdout_fold_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    pd.DataFrame(summaries).to_csv(results_dir / "double_holdout_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    plan.update(status="complete" if all(item["status"] == "complete" for item in plan["folds"]) else "partial_folds_unavailable", metrics=summaries)
    save_json(results_dir / "double_holdout_manifest.json", plan)
    return plan, summaries, folds


def write_report(path, manifest, metrics, folds, score_groups=None, reference_metrics=None, sensitivity_metrics=None, sensitivity_folds=None):
    cov = manifest["cohort"]
    lines = ["# 晕动评分聚合方式对照：2026-10-10", "",
             f"状态：**{manifest['status']}**。本轮是探索性对照，所有方法在产生本轮结果前固定；不使用测试结果调参或选择标签。", "",
             f"共同 cohort：{cov['events']} 次真实评分，{cov['sessions']} sessions，{cov['games']} 游戏，{cov['participants']} 参与者编号。评分分布：{cov['rating_distribution']}。",
             "先按准备阶段 common_eligible 筛选（每个历史尺度都有至少一个非FOV场景测量及用户测量），再取所有方法具有有限真实评分、至少一个有限注册输入的事件交集；各方法训练和评估使用完全相同的事件及游戏折。被排除事件和方法输入覆盖记录在 manifest。", "",
             "## 方法与输入", "", "| 方法 | 请求特征数 | 描述 | 时间策略 |", "|---|---|---|---|"]
    for method, audit in manifest["inputs"].items():
        lines.append(f"| {method} | {len(audit['requested_feature_columns'])} | {audit['description']} | {audit['temporal_policy']} |")
    lines += ["", "场景集合包含运行中的相机/对象/FOV等候选参数，总相机运动仍可能混合用户头动与导航。用户/头动/控制器集合用于检验交互信息关联；它不构成无人开发期输入。elapsed-only 为预设时间对照。", "",
              "最近窗口聚合若含评分之后的窗口，只能描述附近体验关联；评分前5/30/60秒和多尺度集合依照准备阶段边界提取，历史评分、配对距离与评分时点均不作为输入。", "",
              "## 训练与折外结果", "",
              f"固定 LightGBM L2（objective=regression）：{LIGHTGBM_PARAMETERS}；Ridge：alpha=10。训练中每 session 等权，其内部各真实评分事件等权。中位数补缺/常量筛选以及 Ridge 标准化只拟合训练折；没有归一化测试游戏或调整阈值。",
              "回归输出为连续原始 MS_rating；MAE/RMSE 越低越好。AUC 是预设真实评分>2的二分类补充量，用连续预测排名，不等同原三分类 macro AUC。评分是有序等级，误差采用等距近似。",
              "合并 OOF AUC 与有效游戏折 AUC 均值单列；单类游戏不能计算 AUC。基线每折内恒定，合并相关可能反映游戏标签组成差异。多种方法比较产生的最高分仍需独立验证。", ""]
    columns = ["method", "model", "events", "mae", "rmse", "spearman", "pearson", "binary_auc", "mean_valid_fold_auc", "valid_auc_folds", "mae_improvement_over_mean"]
    titles = ["方法", "模型", "事件数", "MAE", "RMSE", "Spearman", "Pearson", "合并OOF AUC", "有效折均值AUC", "有效折数", "较均值MAE改善"]
    lines += table(metrics, columns, titles)
    lines += ["", "### 改善的不确定性", "",
              "按 session 对固定 OOF 预测重采样；条件于已有游戏，没有模型重训，也没有覆盖未知游戏和共享参与者的全部不确定性。正的MAE改善表示优于均值基线。", ""]
    lines += table([row for row in metrics if row["method"] != "baseline"],
                   ["method", "model", "mae_improvement_ci_low", "mae_improvement_ci_high", "spearman_ci_low", "spearman_ci_high"],
                   ["方法", "模型", "MAE改善CI下界", "上界", "Spearman CI下界", "上界"])
    lines += ["", "与 nearest_mean 使用相同模型种类的配对 MAE 改善；正值为优于 nearest_mean，区间仍条件于本次固定 OOF 和已有游戏。", ""]
    lines += table([row for row in metrics if row["method"] != "baseline"],
                   ["method", "model", "mae_improvement_over_nearest_mean", "nearest_mean_mae_improvement_ci_low", "nearest_mean_mae_improvement_ci_high"],
                   ["方法", "模型", "较nearest_mean MAE改善", "CI下界", "CI上界"])
    lines += ["", "### 高分误估检查", "",
              "下表按真实3/4/5分列出折外平均预测和误差；完整1–5分检查在 score_group_metrics.csv，不能以低分占多数的整体MAE代替高分预测能力。", ""]
    lines += table([row for row in (score_groups or []) if row["actual_ms_rating"] >= 3],
                   ["method", "model", "actual_ms_rating", "events", "mean_prediction", "mean_bias", "mae"],
                   ["方法", "模型", "真实评分", "事件数", "平均预测", "平均偏差", "MAE"])
    if reference_metrics:
        lines += ["", "### 旧窗口模型参考", "",
                  "旧模型先以窗口训练，再按评分事件平均折外预测；本轮先按评分事件聚合特征后训练。即使在相同评分事件上比较，它也不是同训练粒度的聚合方法对照，不用于宣称纯聚合改进。", ""]
        lines += table(reference_metrics, ["method", "events", "mae", "rmse", "spearman", "pearson", "binary_auc"],
                       ["参考", "事件数", "MAE", "RMSE", "Spearman", "Pearson", "合并OOF AUC"])
    if sensitivity_metrics:
        lines += ["", "## 预设敏感性：游戏与测试参与者同时留出", "",
                  "只对 multi_scene/multi_user/multi_joint/multi_scene_head 的固定 LightGBM 运行。每折测试游戏保持不变，从训练再删除该测试游戏所有已知参与者的事件；均值和中位数基线使用同样删除后的训练集。此结果单列，不混入主要对照排名。",
                  "未知参与者标识不能证明身份独立；各折训练可用性、删除事件数和未知标识数记录在 double_holdout_manifest.json。", ""]
        lines += table(sensitivity_metrics, ["method", "model", "events", "mae", "rmse", "spearman", "pearson", "binary_auc", "mean_valid_fold_auc"],
                       ["敏感性方法", "模型", "事件数", "MAE", "RMSE", "Spearman", "Pearson", "合并AUC", "有效折均值AUC"])
        lines += ["", "### 双重留出各游戏折", ""]
        lines += table(sensitivity_folds or [], ["held_out_game", "method", "model", "train_events_after_purge", "removed_train_events", "training_available", "mae", "rmse", "spearman", "binary_auc"],
                       ["测试游戏", "方法", "模型", "保留训练事件", "删除事件", "训练可用", "MAE", "RMSE", "Spearman", "AUC"])
    lines += ["", "### 逐游戏结果", ""]
    lines += table(folds, ["held_out_game", "method", "model", "events", "mae", "rmse", "spearman", "pearson", "binary_auc", "participant_overlap"],
                   ["游戏", "方法", "模型", "事件数", "MAE", "RMSE", "Spearman", "Pearson", "AUC", "参与者交叉"])
    reasons = [row for row in [*metrics, *folds] if row["binary_auc"] is None or row["spearman"] is None]
    if reasons:
        lines += ["", "不可计算指标原因："]
        for row in reasons:
            reason = "; ".join(value for value in [row.get("binary_auc_reason"), row.get("correlation_reason")] if value)
            lines.append(f"- {row.get('held_out_game') or '合并'} / {row['method']} / {row['model']}: {reason}")
    lines += ["", "## 覆盖与限制", "",
              f"游戏事件数：{cov['events_by_game']}。参与者交叉：{manifest.get('participant_overlap_by_fold', [])}。LOGO 不是新参与者留出。",
              "同一session重复评分与各历史窗口可能重叠；传统 Pearson/Spearman p 值为描述量，不能按独立事件显著性解释。原始 p 值保存在 CSV，主要判断应结合效应量、分组区间及游戏差异。",
              "场景运动单位保留源坐标单位/s及/s²，旋转率为度/s；FOV 常量不能提供可学习的改变信息。flow/depth 若未在原CSV中存在，不作为已测输入。",
              "原始头部速度和角速度单位尚未独立核验，控制器幅度为原记录数值；触发器/摇杆统计可能受到评分录入动作影响。multi_scene_head 已预设为去掉控制器的敏感性集合，不用控制器增益证明设计效应。",
              "用户信息带来的增益不能解释为可控设计参数的增益；本次也没有证明修改某参数会降低评分。原三分类门控未在本次回归比较中评估或通过。", "",
              "最终全量拟合模型仅用于复现/推理；报告性能来自 OOF。bundle 包含 input_feature_cols、fitted_feature_cols 和 pipeline，按 input_feature_cols 顺序提供同定义数值。", "",
              f"输入和准备审计：{manifest['preparation_path']}；详细来源哈希、共同 cohort、输入与训练折保留特征见 training_manifest.json。"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / f"configs/{NAME}.json")
    parser.add_argument("--input-dir", type=Path, default=ROOT / f"features/{NAME}")
    parser.add_argument("--preparation", type=Path)
    parser.add_argument("--results-dir", type=Path, default=ROOT / f"results/{NAME}")
    parser.add_argument("--models-dir", type=Path, default=ROOT / f"models/{NAME}")
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    args = parser.parse_args()
    if not (ROOT / "PROTOCOL.md").is_file():
        raise ValueError("frozen PROTOCOL.md must exist before this supplementary experiment")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    kinds = cfg.get("models", ["lightgbm", "ridge"])
    if not kinds or len(kinds) != len(set(kinds)) or set(kinds) - {"lightgbm", "ridge"}:
        raise ValueError("models must be a unique subset of lightgbm/ridge")
    if "model_parameters" in cfg and cfg["model_parameters"] != MODEL_PARAMETERS:
        raise ValueError("registered model_parameters differ from fixed LightGBM parameters")
    if "ridge_parameters" in cfg and cfg["ridge_parameters"] != RIDGE_PARAMETERS:
        raise ValueError("registered ridge_parameters differ from fixed alpha=10")
    if "ridge_alpha" in cfg and cfg["ridge_alpha"] != RIDGE_PARAMETERS["alpha"]:
        raise ValueError("registered ridge_alpha differs from fixed alpha=10")
    frames, cohort, inputs = load_methods(cfg, args.input_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(NAME)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    for handler in [logging.StreamHandler(sys.stdout), logging.FileHandler(args.results_dir / "training_log.txt", mode="w", encoding="utf-8")]:
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        log.addHandler(handler)
    preparation_path = args.preparation or args.input_dir / "preparation_manifest.json"
    preparation = json.loads(preparation_path.read_text(encoding="utf-8")) if preparation_path.is_file() else None
    if not preparation or preparation.get("status") != "complete":
        raise ValueError("complete preparation_manifest.json required before comparison fitting")
    if preparation.get("config_sha256") != digest(args.config):
        raise ValueError("completed preparation configuration hash mismatch")
    for audit in inputs.values():
        if Path(audit["input_path"]).name == "events.csv" and preparation.get("events_sha256") != audit["input_sha256"]:
            raise ValueError("events.csv content hash differs from completed preparation")
    manifest = dict(status="started", config_path=str(args.config.resolve()), config_sha256=digest(args.config),
                    config=cfg, inputs=inputs, models=kinds, model_parameters=LIGHTGBM_PARAMETERS,
                    estimator_full_parameters=dict(lightgbm=LGBMRegressor(**LIGHTGBM_PARAMETERS).get_params(deep=False),
                                                   ridge=Ridge(**RIDGE_PARAMETERS).get_params(deep=False)),
                    ridge_parameters=RIDGE_PARAMETERS, variance_threshold=VARIANCE_THRESHOLD,
                    binary_threshold=BINARY_SCORE_THRESHOLD,
                    weight_policy="one rating event per row; equal total session weight; equal events within session",
                    preparation_path=str(preparation_path.resolve()), preparation_manifest=preparation,
                    preparation_sha256=digest(preparation_path) if preparation_path.is_file() else None,
                    common_cohort_policy="preparation common_eligible AND all-method finite-target/feature intersection; fixed before evaluation",
                    cohort=dict(events=len(cohort), sessions=int(cohort["session_id"].nunique()),
                                games=int(cohort["game_name"].nunique()), participants=int(cohort["participant_id"].nunique()),
                                game_names=sorted(cohort["game_name"].unique()),
                                events_by_game=cohort.groupby("game_name").size().to_dict(),
                                rating_distribution=cohort["ms_rating"].value_counts().sort_index().to_dict()),
                    environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                                     numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, lightgbm=lightgbm.__version__, scipy=scipy.__version__, joblib=joblib.__version__),
                    tuning="none; all registered methods/models reported; no unbiased winner-selection claim",
                    bootstrap_draws=args.bootstrap_draws, original_protocol_gate="not evaluated; raw-score regression and fixed >2 binary AUC are supplementary")
    manifest["double_holdout_predeclared_plan"] = dict(methods=DOUBLE_HOLDOUT_METHODS, model="lightgbm", training_rule="remove test-game participants from training", fixed_in_script_before_execution=True)
    manifest["training_script_sha256"] = digest(Path(__file__))
    manifest["helper_script_sha256"] = {name: digest(ROOT / "scripts" / name) for name in ["train_param_rating_model.py", "train_param_rating_local_all.py"]}
    cohort.to_csv(args.results_dir / "cohort.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    manifest.update(frozen_before_fit_utc=datetime.now(timezone.utc).isoformat(),
                    cohort_sha256=digest(args.results_dir / "cohort.csv"))
    save_json(args.results_dir / "planned_training_manifest.json", manifest)
    if len(cohort) < 2 or cohort["game_name"].nunique() < 2 or cohort["ms_rating"].nunique() < 2:
        manifest.update(status="unavailable", reason="common cohort lacks at least two events/games/target values")
        save_json(args.results_dir / "training_manifest.json", manifest)
        write_report(args.results_dir / "report.md", manifest, [], [])
        log.error(manifest["reason"])
        return 2
    y, groups = cohort["ms_rating"].to_numpy(dtype=float), cohort["game_name"].to_numpy()
    outputs, fold_rows, preprocessing = [], [], []
    overlap_rows = []
    for fold, (train_idx, test_idx) in enumerate(LeaveOneGroupOut().split(cohort, y, groups), 1):
        train_meta, test_meta = cohort.iloc[train_idx], cohort.iloc[test_idx]
        heldout = str(groups[test_idx[0]])
        if set(train_meta["session_id"]) & set(test_meta["session_id"]):
            raise ValueError("train/test session overlap")
        if set(train_meta["rating_event_id"]) & set(test_meta["rating_event_id"]):
            raise ValueError("train/test rating event overlap")
        train_people = set(train_meta["participant_id"].dropna()) - {"unknown", "", "nan"}
        test_people = set(test_meta["participant_id"].dropna()) - {"unknown", "", "nan"}
        shared = sorted(train_people & test_people)
        overlap_rows.append(dict(fold=fold, held_out_game=heldout, count=len(shared), participant_ids=shared))
        weights = train_weights(train_meta)
        baseline_values = [("train_mean", float(np.average(y[train_idx], weights=weights))),
                           ("train_median", weighted_median(y[train_idx], weights))]
        variants = [("baseline", name, np.full(len(test_idx), value), {}) for name, value in baseline_values]
        for method, frame in frames.items():
            for kind in kinds:
                columns = inputs[method]["requested_feature_columns"]
                pipe, detail = fit(frame.iloc[train_idx], y[train_idx], columns, kind)
                prediction = pipe.predict(frame.iloc[test_idx][detail["input_feature_cols"]]) if pipe is not None else np.full(len(test_idx), np.nan)
                preprocessing.append(dict(method=method, model=kind, fold=fold, held_out_game=heldout, **detail))
                variants.append((method, kind, prediction, detail))
        for method, kind, prediction, detail in variants:
            rows = test_meta.copy()
            rows["method"], rows["model"], rows["fold"], rows["held_out_game"] = method, kind, fold, heldout
            rows["protocol"] = "leave_game_out"
            rows["prediction"] = prediction
            rows["absolute_error"] = (rows["prediction"] - rows["ms_rating"]).abs()
            outputs.append(rows)
            measure = metric_row(rows, method, kind, fold, heldout)
            measure.update(participant_overlap=len(shared), fitted_feature_cols_json=json.dumps(detail.get("fitted_feature_cols", [])),
                           fit_status=detail.get("status", "training_fold_constant"))
            fold_rows.append(measure)
        log.info("LOGO %s/%s game=%s events=%s variants=%s participant_overlap=%s", fold, len(np.unique(groups)), heldout, len(test_idx), len(variants), len(shared))
    predictions = pd.concat(outputs, ignore_index=True)
    summary_rows = []
    mean_rows = predictions[(predictions["method"] == "baseline") & (predictions["model"] == "train_mean")]
    for (method, kind), rows in predictions.groupby(["method", "model"], sort=False):
        if len(rows) != len(cohort) or not rows["rating_event_id"].is_unique:
            raise ValueError("OOF event coverage mismatch")
        summary = metric_row(rows, method, kind)
        matching_folds = [row for row in fold_rows if row["method"] == method and row["model"] == kind]
        valid_auc = [row["binary_auc"] for row in matching_folds if row["binary_auc"] is not None]
        summary.update(mean_valid_fold_auc=float(np.mean(valid_auc)) if valid_auc else None,
                       valid_auc_folds=len(valid_auc), total_folds=len(matching_folds))
        if method != "baseline":
            log.info("bootstrap %s/%s: %s session draws", method, kind, args.bootstrap_draws)
            bootstrap = session_bootstrap(rows, mean_rows, "rating_event", args.bootstrap_draws)
            summary.update({key: value for key, value in bootstrap.items() if key in {
                "mae_improvement_over_mean", "mae_improvement_ci_low", "mae_improvement_ci_high", "spearman_ci_low", "spearman_ci_high"}})
            manifest.setdefault("bootstrap", {})[f"{method}/{kind}"] = bootstrap
            if "nearest_mean" in frames:
                nearest_rows = predictions[predictions["method"].eq("nearest_mean") & predictions["model"].eq(kind)]
                nearest_delta = delta_bootstrap(rows, nearest_rows, args.bootstrap_draws)
                summary.update({key: value for key, value in nearest_delta.items() if key in {
                    "mae_improvement_over_nearest_mean", "nearest_mean_mae_improvement_ci_low", "nearest_mean_mae_improvement_ci_high"}})
                manifest.setdefault("nearest_mean_bootstrap", {})[f"{method}/{kind}"] = nearest_delta
        summary_rows.append(summary)
    predictions.to_csv(args.results_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    pd.DataFrame(fold_rows).to_csv(args.results_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    pd.DataFrame(summary_rows).to_csv(args.results_dir / "metrics.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    score_groups = score_group_rows(predictions)
    pd.DataFrame(score_groups).to_csv(args.results_dir / "score_group_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    reference_audit, reference_metrics = legacy_reference(cfg, cohort, args.results_dir)
    manifest["legacy_reference"] = reference_audit
    sensitivity, sensitivity_metrics, sensitivity_folds = double_holdout(frames, cohort, inputs, args.results_dir, log)
    manifest["double_holdout"] = sensitivity
    full_models = {}
    for method, frame in frames.items():
        for kind in kinds:
            pipe, detail = fit(frame, y, inputs[method]["requested_feature_columns"], kind)
            model_path = args.models_dir / f"{method}_{kind}.pkl"
            full_models[f"{method}/{kind}"] = dict(model_path=str(model_path.resolve()), **detail)
            if pipe is not None:
                joblib.dump(dict(pipeline=pipe, input_feature_cols=detail["input_feature_cols"], fitted_feature_cols=detail["fitted_feature_cols"],
                                 requested_feature_columns=inputs[method]["requested_feature_columns"], preprocessing=detail,
                                 method=method, model=kind, target="unchanged MS_rating", observed_score_range=[float(y.min()), float(y.max())],
                                 config_sha256=manifest["config_sha256"], input_sha256=inputs[method]["input_sha256"],
                                 common_cohort_sha256=digest(args.results_dir / "cohort.csv"),
                                 warning="exploratory full-cohort fitted model; performance requires OOF; not causal or certified pure design"), model_path)
    manifest.update(status="complete" if all(item["status"] == "fitted" for item in preprocessing) and all(item["status"] == "fitted" for item in full_models.values()) else "partial_folds_or_models_unavailable",
                    metrics=summary_rows, fold_preprocessing=preprocessing, participant_overlap_by_fold=overlap_rows, full_models=full_models,
                    cohort_sha256=digest(args.results_dir / "cohort.csv"), results_dir=str(args.results_dir.resolve()), models_dir=str(args.models_dir.resolve()))
    save_json(args.results_dir / "training_manifest.json", manifest)
    write_report(args.results_dir / "report.md", manifest, summary_rows, fold_rows, score_groups, reference_metrics, sensitivity_metrics, sensitivity_folds)
    for row in summary_rows:
        log.info("%s/%s MAE=%s RMSE=%s rho=%s AUC=%s", row["method"], row["model"], format_value(row["mae"]), format_value(row["rmse"]), format_value(row["spearman"]), format_value(row["binary_auc"]))
    return 0 if manifest["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
