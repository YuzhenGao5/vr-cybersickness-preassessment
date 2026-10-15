#!/usr/bin/env python3
"""Fixed local VR.net regression, LOGO, and a preset 30-second sensitivity.

This script reuses the audited fold-local preprocessing from
train_param_rating_model.py. It never fetches external data or reads videos.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.model_selection import LeaveOneGroupOut

from train_param_rating_model import (
    BINARY_SCORE_THRESHOLD, FORBIDDEN_FEATURES, MODEL_PARAMETERS, ROOT,
    VARIANCE_THRESHOLD, aggregate_events, clean_json, digest, fit_model,
    format_value, metric_row, sample_weights, save_json, shap_outputs, table,
)

NAME = "param_rating_local_all_2026-10-10"
ANALYSES = [("unlimited_nearest", None), ("nearest_within_30s", 30.0)]
METADATA = ["window_id", "game_name", "session_id", "participant_id", "start_time_sec", "end_time_sec",
            "window_center_sec", "rating_event_id", "ms_rating", "rating_time_sec", "rating_time_delta_sec", "label_is_future"]


def weighted_median(values, weights):
    order = np.argsort(values, kind="stable")
    ordered_y, ordered_w = np.asarray(values)[order], np.asarray(weights)[order]
    index = min(int(np.searchsorted(np.cumsum(ordered_w), ordered_w.sum() / 2)), len(ordered_y) - 1)
    return float(ordered_y[index])


def availability(frame):
    reasons = []
    if frame.empty:
        reasons.append("no eligible labeled windows")
    if frame["game_name"].nunique() < 2:
        reasons.append("fewer than two games; LOGO unavailable")
    if frame["ms_rating"].nunique() < 2:
        reasons.append("fewer than two target values")
    return reasons


def coverage(frame):
    events = frame.drop_duplicates("rating_event_id")
    distance = frame["rating_time_delta_sec"].abs().dropna()
    return dict(windows=len(frame), rating_events=int(frame["rating_event_id"].nunique()),
                sessions=int(frame["session_id"].nunique()), games=int(frame["game_name"].nunique()),
                game_names=sorted(frame["game_name"].dropna().unique().tolist()),
                participants=int(frame["participant_id"].nunique()),
                windows_by_game=frame.groupby("game_name").size().to_dict(),
                sessions_by_game=frame.groupby("game_name")["session_id"].nunique().to_dict(),
                rating_events_by_game=events.groupby("game_name").size().to_dict(),
                window_rating_distribution=frame["ms_rating"].value_counts().sort_index().to_dict(),
                event_rating_distribution=events["ms_rating"].value_counts().sort_index().to_dict(),
                future_labeled_windows=int(frame["label_is_future"].sum()),
                windows_farther_than_30s=int(distance.gt(30).sum()),
                missing_pair_distance=int(frame["rating_time_delta_sec"].isna().sum()),
                absolute_pair_distance=dict(median=float(distance.median()) if len(distance) else None,
                                            p90=float(distance.quantile(.9)) if len(distance) else None,
                                            max=float(distance.max()) if len(distance) else None),
                observed_score_range=[float(frame["ms_rating"].min()), float(frame["ms_rating"].max())] if len(frame) else None)


def event_predictions(windows):
    events = aggregate_events(windows)
    # Both baselines are exactly constant within a held-out fold. Restore that
    # constant after aggregation so rounding cannot manufacture event ranks.
    for name in ["train_mean", "train_median"]:
        constants = windows.loc[windows["model"].eq(name)].groupby("fold")["prediction"].first()
        selected = events["model"].eq(name)
        events.loc[selected, "prediction"] = events.loc[selected, "fold"].map(constants)
    return events


def session_bootstrap(model_rows, mean_rows, resolution, draws=1000, seed=42):
    """Fixed paired OOF draws by session; no model retraining or new games."""
    if draws <= 0:
        return dict(status="disabled", draws=draws)
    key = "window_id" if resolution == "window" else "rating_event_id"
    if not model_rows[key].is_unique or not mean_rows[key].is_unique:
        raise ValueError(f"bootstrap requires unique {key} per model")
    ordered = mean_rows.set_index(key).loc[model_rows[key]]
    y = model_rows["ms_rating"].to_numpy(dtype=float)
    predicted = model_rows["prediction"].to_numpy(dtype=float)
    base = ordered["prediction"].to_numpy(dtype=float)
    if not np.isfinite(predicted).all() or not np.isfinite(base).all():
        return dict(status="unavailable", reason="incomplete OOF predictions", draws=draws)
    groups = list(model_rows.groupby("session_id", sort=True).indices.values())
    if len(groups) < 2:
        return dict(status="unavailable", reason="fewer than two sessions", draws=draws)
    rng = np.random.default_rng(seed)
    delta = np.abs(y - base) - np.abs(y - predicted)
    improvement, correlation = [], []
    for _ in range(draws):
        selected = np.concatenate([groups[index] for index in rng.integers(0, len(groups), len(groups))])
        improvement.append(float(delta[selected].mean()))
        test_y, test_pred = y[selected], predicted[selected]
        if np.ptp(test_y) > 0 and np.ptp(test_pred) > 1e-12 * max(1.0, float(np.max(np.abs(test_pred)))):
            rho = float(spearmanr(test_y, test_pred).statistic)
            if np.isfinite(rho):
                correlation.append(rho)
    return dict(status="complete", draws=draws, seed=seed, session_groups=len(groups),
                mae_improvement_over_mean=float(delta.mean()),
                mae_improvement_ci_low=float(np.percentile(improvement, 2.5)),
                mae_improvement_ci_high=float(np.percentile(improvement, 97.5)),
                spearman_ci_low=float(np.percentile(correlation, 2.5)) if correlation else None,
                spearman_ci_high=float(np.percentile(correlation, 97.5)) if correlation else None,
                valid_spearman_draws=len(correlation),
                method="1000-draw default session-group bootstrap of fixed OOF predictions; conditional on available games; not model-refit uncertainty")


def evaluate(frame, feature_columns, analysis, log, bootstrap_draws):
    audit = dict(analysis=analysis, coverage=coverage(frame), status="started", folds=[])
    reasons = availability(frame)
    if reasons:
        audit.update(status="unavailable", reason="; ".join(reasons))
        log.warning("analysis=%s unavailable=%s", analysis, audit["reason"])
        return audit, pd.DataFrame(), pd.DataFrame(), [], []
    y, groups = frame["ms_rating"].to_numpy(dtype=float), frame["game_name"].to_numpy()
    outputs = []
    for fold, (train_indices, test_indices) in enumerate(LeaveOneGroupOut().split(frame, y, groups), 1):
        train, test = frame.iloc[train_indices], frame.iloc[test_indices]
        if set(train["session_id"]) & set(test["session_id"]):
            raise ValueError("same session occurs on both sides of a game fold")
        if set(train["rating_event_id"]) & set(test["rating_event_id"]):
            raise ValueError("same rating event occurs on both sides of a game fold")
        heldout = str(groups[test_indices[0]])
        weight = sample_weights(train)
        means = float(np.average(y[train_indices], weights=weight))
        medians = weighted_median(y[train_indices], weight)
        pipe, detail = fit_model(train, y[train_indices], feature_columns)
        pred = pipe.predict(test[detail["input_feature_cols"]]) if pipe is not None else np.full(len(test), np.nan)
        # Unknown identifiers do not count as proof of participant overlap.
        train_people = set(train["participant_id"].dropna()) - {"unknown", "", "nan"}
        test_people = set(test["participant_id"].dropna()) - {"unknown", "", "nan"}
        detail.update(fold=fold, held_out_game=heldout, train_windows=len(train), test_windows=len(test),
                      train_sessions=int(train["session_id"].nunique()), test_sessions=int(test["session_id"].nunique()),
                      participant_overlap_count=len(train_people & test_people),
                      participant_overlap_ids=sorted(train_people & test_people),
                      train_target_unique=int(train["ms_rating"].nunique()), weighted_mean=means, weighted_median=medians)
        audit["folds"].append(detail)
        for name, predicted in [("lightgbm", pred), ("train_mean", np.full(len(test), means)), ("train_median", np.full(len(test), medians))]:
            rows = test[METADATA].copy()
            rows["analysis"], rows["model"], rows["fold"], rows["held_out_game"] = analysis, name, fold, heldout
            rows["prediction"] = predicted
            rows["absolute_error"] = (rows["prediction"] - rows["ms_rating"]).abs()
            outputs.append(rows)
        log.info("%s LOGO %s/%s game=%s windows=%s active_features=%s participant_overlap=%s",
                 analysis, fold, len(np.unique(groups)), heldout, len(test), len(detail["fitted_feature_cols"]), detail["participant_overlap_count"])
    windows = pd.concat(outputs, ignore_index=True)
    events = event_predictions(windows)
    events["analysis"] = analysis
    summary_rows, fold_rows = [], []
    for resolution, data in [("window", windows), ("rating_event", events)]:
        for model, rows in data.groupby("model", sort=False):
            summary = metric_row(rows, model, resolution)
            summary["analysis"] = analysis
            if model == "lightgbm":
                log.info("%s %s fixed-OOF session bootstrap: %s draws", analysis, resolution, bootstrap_draws)
                bootstrap = session_bootstrap(rows, data.loc[data["model"].eq("train_mean")], resolution, bootstrap_draws)
                summary.update({name: value for name, value in bootstrap.items() if name in {
                    "mae_improvement_over_mean", "mae_improvement_ci_low", "mae_improvement_ci_high", "spearman_ci_low", "spearman_ci_high"}})
                audit.setdefault("bootstrap", {})[resolution] = bootstrap
            summary_rows.append(summary)
        for (fold, game, model), rows in data.groupby(["fold", "held_out_game", "model"], sort=False):
            detail = next(item for item in audit["folds"] if item["fold"] == fold)
            metric = metric_row(rows, model, resolution, int(fold), game)
            metric.update(analysis=analysis, participant_overlap=detail["participant_overlap_count"],
                          fitted_feature_cols_json=json.dumps(detail["fitted_feature_cols"] if model == "lightgbm" else []),
                          fold_model_status=detail["status"] if model == "lightgbm" else "training_fold_constant")
            fold_rows.append(metric)
    for summary in summary_rows:
        matching = [row for row in fold_rows if row["model"] == summary["model"] and row["resolution"] == summary["resolution"]]
        valid_auc = [row["binary_auc"] for row in matching if row["binary_auc"] is not None]
        summary.update(mean_valid_fold_auc=float(np.mean(valid_auc)) if valid_auc else None,
                       valid_auc_folds=len(valid_auc), total_folds=len(matching))
    audit.update(status="complete" if all(item["status"] == "fitted" for item in audit["folds"]) else "partial_model_folds_unavailable",
                 metrics=summary_rows)
    return audit, windows, events, summary_rows, fold_rows


def write_report(path, manifest, metrics, folds):
    lines = ["# 本地全部 VR.net 日志的参数–评分探索性回归", "",
             "本轮覆盖本地物理文件及已有 ZIP 中按内容去重的可用 data_file.csv；没有下载数据或读取视频。覆盖范围是全部本地可用日志，不代表完整公开的798个记录。模型检验运行参数与附近真实 MS_rating 的关联，尚未认证为纯设计参数模型。", "",
             f"运行状态：**{manifest['status']}**。输入、配置和准备审计来源的哈希已记录。", "",
             "## 数据与输入", "",
             f"配置请求 {len(manifest['feature_columns'])} 项输入：{', '.join(manifest['feature_columns'])}。",
             "主要输入为总相机运动、投影 FOV、对象矩阵项数量的5秒 mean/max/std；若日志实际包含 flow/depth 数值，按准备配置扩展。没有从视频猜测这些字段。身份、时间、配对距离、标签时点和评分列均不进入模型。",
             "速度单位是源坐标单位/s，加速度是源坐标单位/s²，不假定为米；旋转率为度/s，FOV 是度。总相机运动混合导航与用户头动。常量或全缺失列在训练折内删除。", "",
             f"训练资格筛选：{manifest['eligibility_filter']}。", "",
             "## 主分析与预设距离对照", "",
             "主分析保留无限距离的最近评分配对；敏感性分析预先固定为距离绝对值≤30秒，独立重新执行各游戏留出训练。距离只用于筛选，不作输入。两者不按测试结果选择优胜方案。"]
    for audit in manifest["analyses"]:
        cov = audit["coverage"]
        lines += ["", f"### {audit['analysis']}", "",
                  f"- {cov['windows']} 窗口、{cov['rating_events']} 实际评分事件、{cov['sessions']} sessions、{cov['games']} 游戏：{', '.join(cov['game_names']) or '无'}；参与者编号 {cov['participants']} 个。",
                  f"- 事件评分分布：{cov['event_rating_distribution']}；窗口评分分布：{cov['window_rating_distribution']}。",
                  f"- 未来评分配对窗口 {cov['future_labeled_windows']}；超过30秒窗口 {cov['windows_farther_than_30s']}。绝对距离中位数/P90/最大值：{cov['absolute_pair_distance']}。",
                  f"- 每游戏 sessions：{cov['sessions_by_game']}；实际评分事件：{cov['rating_events_by_game']}。",
                  f"- 分析状态：{audit['status']}。"]
        if audit["status"] == "unavailable":
            lines.append(f"- 原因：{audit['reason']}。没有替代拆分或虚构指标。")
        else:
            overlap = [f"{item['held_out_game']}:{item['participant_overlap_count']}" for item in audit["folds"] if item["participant_overlap_count"]]
            lines.append(f"- LOGO {len(audit['folds'])} 折；存在参与者交叉的测试游戏及人数：{', '.join(overlap) or '未发现已知参与者编号交叉'}。本评估没有同时要求参与者留出。")
    lines += ["", "## 固定模型与折外结果", "",
              f"LightGBM 参数固定：{MODEL_PARAMETERS}。无调参；中位数填补和 VarianceThreshold({VARIANCE_THRESHOLD}) 只用训练折。",
              "训练权重：每 session 等权，同 session 每次评分事件等权，同事件多个窗口均分权重。训练均值和训练中位数对照使用相同权重。",
              "window 指每个5秒窗；rating_event 指同一真实评分的折外窗口预测取均值，每次评分只计一次。相邻窗口与重复评分不独立。p 值仅供描述，不据窗口数量宣称独立显著性。",
              "固定二分类 AUC 将真实 MS_rating>2 视为高分，用连续回归预测排名；单类折记为不可计算。基线在单折内恒定，其合并折外相关可能来自折之间标签组成差异，应结合逐游戏结果解读。", ""]
    metric_columns = ["analysis", "resolution", "model", "samples", "mae", "rmse", "spearman", "spearman_p", "pearson", "pearson_p", "binary_auc"]
    metric_titles = ["分析", "计数单位", "模型", "N", "MAE", "RMSE", "Spearman", "p", "Pearson", "p", "合并OOF AUC >2"]
    lines += table(metrics, metric_columns, metric_titles)
    lines += ["", "折 AUC 仅对能计算的游戏折求平均，与合并 OOF AUC 分开列出；单类游戏不补造 AUC。", ""]
    lines += table(metrics, ["analysis", "resolution", "model", "mean_valid_fold_auc", "valid_auc_folds", "total_folds"],
                   ["分析", "单位", "模型", "有效折平均AUC", "有效折数", "总折数"])
    for analysis, _ in ANALYSES:
        selected = {row["model"]: row for row in metrics if row["analysis"] == analysis and row["resolution"] == "rating_event"}
        if "lightgbm" in selected and selected["lightgbm"]["mae"] is not None:
            primary, mean, median = [selected[name] for name in ["lightgbm", "train_mean", "train_median"]]
            lines += ["", f"{analysis} 事件级：LightGBM MAE {primary['mae']:.4f}，均值基线 {mean['mae']:.4f}，中位数基线 {median['mae']:.4f}；较均值改善 {mean['mae']-primary['mae']:.4f} 分，较中位数改善 {median['mae']-primary['mae']:.4f} 分（正值为改善）。"]
    lines += ["", "### 逐游戏结果", ""]
    lines += table(folds, ["held_out_game", *metric_columns, "participant_overlap"], ["测试游戏", *metric_titles, "参与者交叉"])
    unavailable = [row for row in [*metrics, *folds] if row["binary_auc"] is None or row["spearman"] is None]
    if unavailable:
        lines += ["", "不可计算指标原因："]
        for row in unavailable:
            reasons = [row.get("binary_auc_reason"), row.get("correlation_reason")]
            lines.append(f"- {row['analysis']} / {row.get('held_out_game') or '合并'} / {row['resolution']} / {row['model']}: " + "; ".join(value for value in reasons if value))
    lines += ["", "## 不确定性与限制", "",
              "下列区间对固定 OOF 预测按 session 重采样，没有重新训练，条件于已有游戏；不能代表对未知新游戏的完整不确定性。共享参与者跨 session 的依赖也可能影响区间。均值基线 MAE 减模型 MAE 为正代表改善。", ""]
    ci_rows = []
    for audit in manifest["analyses"]:
        for resolution, result in audit.get("bootstrap", {}).items():
            ci_rows.append(dict(analysis=audit["analysis"], resolution=resolution, **result))
    lines += table(ci_rows, ["analysis", "resolution", "mae_improvement_over_mean", "mae_improvement_ci_low", "mae_improvement_ci_high", "spearman_ci_low", "spearman_ci_high"],
                   ["分析", "单位", "MAE改善", "改善CI下界", "改善CI上界", "Spearman CI下界", "CI上界"])
    lines += ["", "- 未来最近评分与较远评分都可能被复用，本任务不是严格历史的实时预测；30秒对照用于观察标签距离的影响。",
              "- 游戏数、评分范围、高分样本数及剔除来源决定本轮覆盖；各游戏内部路径和参与者特征仍可能影响模型。",
              "- 运行参数不能直接解释为作者可控设计参数；坐标跳变、对象可见性和设备 FOV 可能影响特征。",
              "- 折外误差、相关与派生二分类 AUC 都不能证明修改参数会降低评分；本次没有评估或通过原三分类 macro AUC 门控。", ""]
    final = manifest.get("final_model", {})
    lines += ["## 保存模型", "",
              f"全量主分析模型状态：{final.get('status', 'unavailable')}；实际保留输入：{final.get('fitted_feature_cols', [])}。",
              f"删除全缺失：{final.get('dropped_all_missing', [])}；删除常量：{final.get('dropped_constant', [])}。",
              "bundle 包含 pipeline、input_feature_cols、fitted_feature_cols、评分范围和来源哈希。按 input_feature_cols 导出相同定义的数值后使用 pipeline.predict；全量拟合不能用于报告测试性能。",
              f"SHAP 状态：{manifest.get('shap', {}).get('status', 'not_requested')}。若启用，归因来自全量拟合主模型，不是 OOF 或因果证据。", "",
              "## 本地数据准备审计", "",
              f"来源：{manifest['preparation_manifest_path']}；来源库存：{manifest['source_inventory_path']}。以下保留准备 manifest 的覆盖、去重、配对、字段可用性与单位记录。", "", "```json"]
    preparation = manifest.get("preparation_manifest")
    excerpt = {}
    if isinstance(preparation, dict):
        for key, value in preparation.items():
            if isinstance(value, list) and len(value) > 30:
                excerpt[key] = {"entries": len(value), "details": "see preparation manifest"}
            else:
                excerpt[key] = value
    lines += [json.dumps(clean_json(excerpt or preparation), ensure_ascii=False, indent=2), "```", "",
              "详细输出路径、环境版本及 fold 保留特征见 training_manifest.json。"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / f"features/{NAME}.csv")
    parser.add_argument("--preparation", type=Path, default=ROOT / f"features/{NAME}_preparation.json")
    parser.add_argument("--config", type=Path, default=ROOT / f"configs/{NAME}.json")
    parser.add_argument("--results-dir", type=Path, default=ROOT / f"results/{NAME}")
    parser.add_argument("--model", type=Path, default=ROOT / f"models/{NAME}.pkl")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--shap", action="store_true", help="Optional attribution of the final full-data main model")
    args = parser.parse_args()
    if not (ROOT / "PROTOCOL.md").is_file():
        raise SystemExit("PROTOCOL.md missing: experiments forbidden")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    report = args.report or args.results_dir / "report.md"
    log = logging.getLogger(NAME)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    for handler in [logging.StreamHandler(sys.stdout), logging.FileHandler(args.results_dir / "training_log.txt", mode="w", encoding="utf-8")]:
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        log.addHandler(handler)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if cfg["model_parameters"] != MODEL_PARAMETERS:
        raise ValueError("Registered model parameters differ from the reused implementation")
    feature_columns = cfg["feature_columns"]
    if not isinstance(feature_columns, list) or not feature_columns or len(feature_columns) != len(set(feature_columns)):
        raise ValueError("config.feature_columns must be a nonempty unique list")
    if set(feature_columns) & (FORBIDDEN_FEATURES | {"window_has_any_feature", "source_file", "dataset_family"}):
        raise ValueError("identity, time, provenance, and rating columns cannot be model inputs")
    frame = pd.read_csv(args.input, dtype={name: str for name in ["window_id", "game_name", "session_id", "participant_id", "rating_event_id"]}, encoding="utf-8-sig")
    missing = sorted(set(METADATA + feature_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"missing input columns {missing}")
    frame[feature_columns] = frame[feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    frame["ms_rating"] = pd.to_numeric(frame["ms_rating"], errors="coerce")
    frame["rating_time_delta_sec"] = pd.to_numeric(frame["rating_time_delta_sec"], errors="coerce")
    frame["label_is_future"] = frame["label_is_future"].astype(str).str.lower().isin({"true", "1", "1.0", "yes"})
    eligible = frame[feature_columns].notna().any(axis=1) & np.isfinite(frame["ms_rating"])
    if "window_has_any_feature" in frame:
        eligible &= frame["window_has_any_feature"].astype(str).str.lower().isin({"true", "1", "1.0", "yes"})
    filtering = dict(source_windows=len(frame), eligible_windows=int(eligible.sum()), excluded_windows=int((~eligible).sum()),
                     rule="at least one configured finite feature AND declared feature availability if present AND finite MS_rating")
    frame = frame.loc[eligible].reset_index(drop=True)
    if not frame["window_id"].is_unique or frame[["session_id", "game_name", "rating_event_id"]].isna().any().any():
        raise ValueError("duplicate window identifiers or missing game/session/rating identifiers")
    if frame.groupby("session_id")["game_name"].nunique().gt(1).any():
        raise ValueError("one session maps to multiple games")
    if frame.groupby("rating_event_id")["session_id"].nunique().gt(1).any() or frame.groupby("rating_event_id")["ms_rating"].nunique().gt(1).any():
        raise ValueError("rating event identity maps to inconsistent session or score")
    preparation = json.loads(args.preparation.read_text(encoding="utf-8")) if args.preparation.is_file() else None
    if not preparation or preparation.get("status") != "complete":
        raise ValueError("Preparation must be complete before fitting; see extraction log")
    if preparation.get("output_sha256") != digest(args.input):
        raise ValueError("Training input hash differs from completed preparation manifest")
    source_inventory_path = args.results_dir / "source_inventory.json"
    manifest = dict(status="started", feature_columns=feature_columns, input_path=str(args.input.resolve()), input_sha256=digest(args.input),
                    config_path=str(args.config.resolve()), config_sha256=digest(args.config),
                    preparation_manifest_path=str(args.preparation.resolve()), preparation_manifest=preparation,
                    preparation_manifest_sha256=digest(args.preparation) if args.preparation.is_file() else None,
                    source_inventory_path=str(source_inventory_path.resolve()),
                    source_inventory_sha256=digest(source_inventory_path) if source_inventory_path.is_file() else None,
                    model_parameters=MODEL_PARAMETERS, variance_threshold=VARIANCE_THRESHOLD,
                    binary_threshold=BINARY_SCORE_THRESHOLD, eligibility_filter=filtering,
                    weight_policy="equal sessions; equal rating events within session; equal window shares within event",
                    planned_analyses=ANALYSES, bootstrap_draws=args.bootstrap_draws,
                    environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                                     numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, lightgbm=lightgbm.__version__, scipy=scipy.__version__, joblib=joblib.__version__),
                    reused_fold_preprocessing_script=str(ROOT / "scripts/train_param_rating_model.py"),
                    reused_fold_preprocessing_script_sha256=digest(ROOT / "scripts/train_param_rating_model.py"),
                    tuning="none", original_protocol_gate="not evaluated; derived binary AUC does not replace frozen three-class macro AUC")
    audits, window_outputs, event_outputs, metrics, folds = [], [], [], [], []
    for analysis, distance in ANALYSES:
        subset = frame if distance is None else frame.loc[frame["rating_time_delta_sec"].abs().le(distance)].reset_index(drop=True)
        audit, windows, events, summaries, fold_rows = evaluate(subset, feature_columns, analysis, log, args.bootstrap_draws)
        audits.append(audit)
        if not windows.empty:
            window_outputs.append(windows)
            event_outputs.append(events)
        metrics.extend(summaries)
        folds.extend(fold_rows)
    manifest["analyses"] = audits
    pd.DataFrame(metrics).to_csv(args.results_dir / "metrics.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    pd.DataFrame(folds).to_csv(args.results_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    if window_outputs:
        pd.concat(window_outputs, ignore_index=True).to_csv(args.results_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
        pd.concat(event_outputs, ignore_index=True).to_csv(args.results_dir / "event_oof_predictions.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    if audits[0]["status"] != "unavailable":
        pipe, detail = fit_model(frame, frame["ms_rating"].to_numpy(), feature_columns)
        manifest["final_model"] = detail
        if pipe is not None:
            args.model.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(dict(pipeline=pipe, feature_columns=feature_columns,
                             input_feature_cols=detail["input_feature_cols"], fitted_feature_cols=detail["fitted_feature_cols"],
                             preprocessing=detail, target="unaltered MS_rating", observed_score_range=coverage(frame)["observed_score_range"],
                             input_sha256=manifest["input_sha256"], config_sha256=manifest["config_sha256"], model_parameters=MODEL_PARAMETERS,
                             warning="exploratory nearest-rating/runtime-camera association; not pure design or causal intervention benefit"), args.model)
            if args.shap:
                try:
                    manifest["shap"] = shap_outputs(pipe, detail, frame, args.results_dir / "shap_values.csv", args.results_dir / "shap_importance.csv")
                except Exception as exc:
                    log.exception("Optional SHAP unavailable")
                    manifest["shap"] = dict(status="unavailable", reason=f"{type(exc).__name__}: {exc}")
            else:
                manifest["shap"] = dict(status="not_requested")
        else:
            manifest["shap"] = dict(status="unavailable", reason=detail["reason"])
    else:
        manifest["final_model"] = dict(status="unavailable", reason=audits[0]["reason"])
    manifest["status"] = "complete" if all(audit["status"] == "complete" for audit in audits) and manifest["final_model"]["status"] == "fitted" else "partial_or_unavailable"
    manifest["outputs"] = dict(results_dir=str(args.results_dir.resolve()), model=str(args.model.resolve()), report=str(report.resolve()))
    save_json(args.results_dir / "training_manifest.json", manifest)
    write_report(report, manifest, metrics, folds)
    for row in metrics:
        log.info("%s %s %s N=%s MAE=%s RMSE=%s rho=%s AUC=%s", row["analysis"], row["resolution"], row["model"], row["samples"],
                 format_value(row["mae"]), format_value(row["rmse"]), format_value(row["spearman"]), format_value(row["binary_auc"]))
    log.info("completed status=%s results=%s", manifest["status"], args.results_dir)
    return 0 if manifest["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
