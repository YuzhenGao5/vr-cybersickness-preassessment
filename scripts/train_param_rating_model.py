#!/usr/bin/env python3
"""Fixed LightGBM rating regression, LOGO evaluation, and full-fit SHAP.

Run only after preparing param_rating_matrix.csv. Runtime feature windows may
be paired with a future nearby rating; this script preserves and reports that
fact. All preprocessing is fitted within each training fold.
"""
from __future__ import annotations

import argparse
import hashlib
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
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
MODEL_PARAMETERS = dict(n_estimators=100, num_leaves=7, max_depth=3,
                        min_child_samples=20, learning_rate=0.05,
                        reg_lambda=1.0, random_state=42, n_jobs=2, verbosity=-1)
VARIANCE_THRESHOLD = 1e-10
BINARY_SCORE_THRESHOLD = 2.0
FORBIDDEN_FEATURES = {
    "window_id", "game_name", "session_id", "participant_id", "start_time_sec",
    "end_time_sec", "window_center_sec", "rating_event_id", "ms_rating",
    "rating_time_sec", "rating_time_delta_sec", "label_is_future",
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def save_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(content), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def sample_weights(frame: pd.DataFrame) -> np.ndarray:
    """Equal sessions, equal rating events per session, equal windows per event."""
    windows_per_event = frame.groupby(["session_id", "rating_event_id"])["window_id"].transform("size")
    events_per_session = frame.groupby("session_id")["rating_event_id"].transform("nunique")
    weights = 1.0 / (windows_per_event.to_numpy(dtype=float) * events_per_session.to_numpy(dtype=float))
    return weights / weights.mean()


def fit_model(frame, target, feature_columns):
    nonempty = [name for name in feature_columns if frame[name].notna().any()]
    dropped_empty = [name for name in feature_columns if name not in nonempty]
    detail = dict(input_feature_cols=nonempty, fitted_feature_cols=[],
                  dropped_all_missing=dropped_empty, dropped_constant=[], status="unavailable")
    if not nonempty:
        detail["reason"] = "all training feature columns are missing"
        return None, detail
    imputer = SimpleImputer(strategy="median")
    transformed = imputer.fit_transform(frame[nonempty])
    variance = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    try:
        variance.fit(transformed)
    except ValueError:
        detail.update(dropped_constant=nonempty, reason="no nonconstant training feature after median imputation")
        return None, detail
    active = np.array(nonempty)[variance.get_support()].tolist()
    regressor = LGBMRegressor(**MODEL_PARAMETERS)
    regressor.fit(variance.transform(transformed), target, sample_weight=sample_weights(frame))
    pipe = Pipeline([("imputer", imputer), ("variance", variance), ("regressor", regressor)])
    detail.update(fitted_feature_cols=active,
                  dropped_constant=[name for name in nonempty if name not in active], status="fitted", reason=None)
    return pipe, detail


def measures(target, prediction):
    target, prediction = np.asarray(target, dtype=float), np.asarray(prediction, dtype=float)
    result = dict(mae=None, rmse=None, spearman=None, spearman_p=None,
                  pearson=None, pearson_p=None, binary_auc=None, binary_auc_reason=None,
                  correlation_reason=None, positive_events=int((target > BINARY_SCORE_THRESHOLD).sum()),
                  negative_events=int((target <= BINARY_SCORE_THRESHOLD).sum()))
    if len(target) == 0 or not np.isfinite(prediction).all():
        result.update(binary_auc_reason="prediction unavailable for at least one held-out sample",
                      correlation_reason="prediction unavailable for at least one held-out sample")
        return result
    result.update(mae=float(mean_absolute_error(target, prediction)),
                  rmse=float(np.sqrt(mean_squared_error(target, prediction))))
    if len(target) < 2 or np.ptp(target) == 0 or np.ptp(prediction) <= 1e-12 * max(1.0, float(np.max(np.abs(prediction)))):
        result["correlation_reason"] = "fewer than two samples or constant target/prediction"
    else:
        sr, pr = spearmanr(target, prediction), pearsonr(target, prediction)
        result.update(spearman=float(sr.statistic), spearman_p=float(sr.pvalue),
                      pearson=float(pr.statistic), pearson_p=float(pr.pvalue))
    binary = target > BINARY_SCORE_THRESHOLD
    if len(np.unique(binary)) < 2:
        result["binary_auc_reason"] = "held-out labels contain one binary class at fixed MS_rating > 2 threshold"
    else:
        result["binary_auc"] = float(roc_auc_score(binary, prediction))
    return clean_json(result)


def aggregate_events(frame):
    if frame.empty:
        return frame.copy()
    keys = ["model", "fold", "held_out_game", "session_id", "game_name", "rating_event_id"]
    events = frame.groupby(keys, dropna=False, sort=False).agg(
        ms_rating=("ms_rating", "first"), prediction=("prediction", "mean"),
        windows=("window_id", "size"), available_windows=("prediction", "count"),
        participant_id=("participant_id", "first"), label_is_future=("label_is_future", "any"),
    ).reset_index()
    # A fold's mean-score baseline is exactly constant. Repeated averages of
    # that constant must not create floating-point ranks among rating events.
    baseline_by_fold = frame.loc[frame["model"].eq("train_mean")].groupby("fold")["prediction"].first()
    baseline_rows = events["model"].eq("train_mean")
    events.loc[baseline_rows, "prediction"] = events.loc[baseline_rows, "fold"].map(baseline_by_fold)
    # Missing folds must remain unavailable instead of silently averaging a subset.
    events.loc[events["available_windows"].ne(events["windows"]), "prediction"] = np.nan
    return events


def metric_row(frame, model, resolution, fold=None, heldout=None):
    row = dict(protocol="leave_game_out", resolution=resolution, model=model,
               fold=fold, held_out_game=heldout, samples=len(frame),
               sessions=int(frame["session_id"].nunique()), games=int(frame["game_name"].nunique()),
               rating_events=int(frame["rating_event_id"].nunique()),
               **measures(frame["ms_rating"].to_numpy(), frame["prediction"].to_numpy()))
    finite = frame["prediction"].notna().all()
    row["complete_oof"] = bool(finite)
    if finite:
        work = frame.assign(error=(frame["prediction"] - frame["ms_rating"]).abs())
        row["equal_game_mae"] = float(work.groupby("game_name")["error"].mean().mean())
        row["equal_session_mae"] = float(work.groupby("session_id")["error"].mean().mean())
    else:
        row["equal_game_mae"] = row["equal_session_mae"] = None
    return row


def bh_adjust(pvalues):
    result = np.full(len(pvalues), np.nan)
    valid = np.flatnonzero(np.isfinite(pvalues))
    if not len(valid):
        return result
    order = valid[np.argsort(np.asarray(pvalues)[valid])]
    adjusted = np.asarray(pvalues)[order] * len(order) / np.arange(1, len(order) + 1)
    result[order] = np.minimum(1.0, np.minimum.accumulate(adjusted[::-1])[::-1])
    return result


def shap_outputs(pipe, detail, frame, output_path, importance_path):
    import shap
    transformed = pipe.named_steps["variance"].transform(
        pipe.named_steps["imputer"].transform(frame[detail["input_feature_cols"]]))
    active = detail["fitted_feature_cols"]
    explainer = shap.TreeExplainer(pipe.named_steps["regressor"], feature_perturbation="tree_path_dependent")
    values = np.asarray(explainer.shap_values(transformed), dtype=float)
    if values.shape != transformed.shape:
        raise ValueError(f"Unexpected regression SHAP shape {values.shape}, expected {transformed.shape}")
    base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    prediction = pipe.named_steps["regressor"].predict(transformed)
    residual = np.abs(values.sum(axis=1) + base_value - prediction)
    if not np.isfinite(values).all() or not np.allclose(values.sum(axis=1) + base_value, prediction, atol=1e-5, rtol=1e-5):
        raise ValueError("SHAP additive reconstruction did not match full-fit regression prediction")
    output = frame[["window_id", "game_name", "session_id", "participant_id", "rating_event_id", "ms_rating"]].copy()
    output["full_fit_prediction"] = prediction
    output["shap_base_value"] = base_value
    for index, name in enumerate(active):
        output[name] = values[:, index]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig", float_format="%.12g")
    weights = sample_weights(frame)
    rows = []
    for index, name in enumerate(active):
        feature, contribution = transformed[:, index], values[:, index]
        can_correlate = len(feature) >= 2 and np.ptp(feature) > 0 and np.ptp(contribution) > 0
        correlation = spearmanr(feature, contribution) if can_correlate else None
        rho = float(correlation.statistic) if correlation is not None else None
        direction = "更高特征值通常对应更高模型预测" if rho is not None and rho > 0 else (
            "更高特征值通常对应更低模型预测" if rho is not None and rho < 0 else "方向无法确定")
        rows.append(dict(feature=name, mean_abs_shap=float(np.abs(contribution).mean()),
                         event_session_equal_mean_abs_shap=float(np.average(np.abs(contribution), weights=weights)),
                         feature_shap_spearman=rho,
                         feature_shap_p=float(correlation.pvalue) if correlation is not None else None,
                         direction=direction, interpretation="full-fit model attribution; not OOF, independent evidence, or causal effect"))
    importance = pd.DataFrame(rows)
    importance["feature_shap_bh_q"] = bh_adjust(pd.to_numeric(importance["feature_shap_p"], errors="coerce").to_numpy())
    importance = importance.sort_values("event_session_equal_mean_abs_shap", ascending=False).reset_index(drop=True)
    importance["rank"] = np.arange(1, len(importance) + 1)
    importance.to_csv(importance_path, index=False, encoding="utf-8-sig", float_format="%.12g")
    return dict(status="complete", version=shap.__version__, baseline=base_value,
                max_additive_error=float(residual.max()), input_feature_cols=active,
                weighting="ranking by mean absolute SHAP with equal sessions/events/window shares",
                top5=importance.head(5).to_dict("records"),
                source="final model fitted to all available windows; not out-of-fold explanations")


def format_value(value):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "不可计算"
    if isinstance(value, (float, np.floating)):
        return f"{value:.3g}" if 0 < abs(value) < 1e-4 else f"{value:.4f}"
    return str(value)


def table(rows, columns, titles):
    lines = ["| " + " | ".join(titles) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(name)) for name in columns) + " |")
    return lines


def write_report(path, manifest, summary, folds):
    status = manifest["status"]
    lines = ["# 场景运行参数与晕动评分：探索性回归", "",
             "本报告使用真实 MS_rating 的近邻窗口配对，输入包含用户/导航混合相机运动；运行特征尚未认证为纯设计参数。", "",
             f"运行状态：**{status}**。"]
    if status == "unavailable":
        lines += ["", f"无法训练/评估：{manifest.get('reason')}。没有生成可用模型或虚构性能指标。"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    data = manifest["data"]
    lines += ["", "## 数据与配对", "",
              f"- {data['windows']} 个窗口，{data['rating_events']} 次实际评分，{data['sessions']} sessions，**{data['games']} 个游戏**：{', '.join(data['game_names'])}。参与者编号 {data['participants']} 个。窗口数量不等于独立评分数量。",
              f"- 窗口评分分布：{data['window_score_distribution']}；评分事件分布：{data['event_score_distribution']}。",
              f"- 近邻配对中未来标签窗口 {data['future_labeled_windows']} / {data['windows']}；未来与过去标签均按准备阶段既定规则保留。该任务是附近主观评分关联，不能作为已验证的实时历史预测。",
              f"- 配对距离绝对值中位数 {format_value(data['absolute_rating_distance_seconds']['median'])} 秒，P90 {format_value(data['absolute_rating_distance_seconds']['p90'])} 秒，最大 {format_value(data['absolute_rating_distance_seconds']['max'])} 秒。",
              "- 五秒窗口按时间分箱；相邻窗口来自同一连续session，同一实测评分可能监督多个窗口。窗口/事件级相关检验的 p 值仅作探索性描述，不视为独立样本的显著性证明。", "",
              "## 输入、输出与训练", "",
              "- 输入为相机速度、三维速度向量变化得到的加速度大小、旋转率、水平/垂直投影 FOV、记录对象数各自的 mean/max/std。标识符、Unix时间、窗口时间、配对距离、当前或历史评分均不作为输入。",
              f"- 配置请求 {len(manifest['requested_feature_columns'])} 个输入；全量拟合实际保留 {len(manifest['final_model']['fitted_feature_cols'])} 个：{', '.join(manifest['final_model']['fitted_feature_cols']) or '无'}。",
              f"- 全量拟合去除全缺失列：{', '.join(manifest['final_model']['dropped_all_missing']) or '无'}；去除常量/近常量列：{', '.join(manifest['final_model']['dropped_constant']) or '无'}。常量 FOV 不被补进 Top5。",
              f"- 输出为连续 MS_rating 预测，实际标签范围 {data['observed_score_range']}；不截断、不改成规则标签。速度单位为源坐标单位/s，加速度为源坐标单位/s²，不假定为米；角速度为度/s，FOV 为完整非对称投影开角（度），对象数为有效序列化对象模型矩阵项数，不是场景总 GameObject 数。",
              f"- 固定 LightGBM 参数：{MODEL_PARAMETERS}。没有调参；缺失中位数填补、全缺失列移除、VarianceThreshold({VARIANCE_THRESHOLD}) 均只拟合该折训练数据。",
              "- 训练权重预先固定：每 session 等权；同一 session 内每实际评分事件等权；同一评分事件对应的多个窗口均分该事件权重。均分基线使用相同训练权重。", "",
              "## 游戏留出结果", "",
              f"Leave-One-Game-Out 共 {manifest['folds']} 折；每 session 的所有窗口在同一侧。只有 {data['games']} 个游戏，跨游戏证据范围有限。另核对参与者重叠；没有把该协议宣称为参与者留出。",
              "事件级结果将同一 rating_event_id 的折外窗口预测取平均，每次真实评分只计一次。MAE/RMSE 越小越好；AUC 是固定 MS_rating > 2 的二分类补充指标，用原始回归预测排名，没有调阈值，也不是原协议的三分类 macro AUC。", ""]
    columns = ["resolution", "model", "samples", "mae", "rmse", "spearman", "spearman_p", "pearson", "pearson_p", "binary_auc"]
    titles = ["计数单位", "模型", "样本数", "MAE", "RMSE", "Spearman", "p", "Pearson", "p", "AUC (>2)"]
    lines += table(summary, columns, titles)
    primary = next((r for r in summary if r["resolution"] == "rating_event" and r["model"] == "lightgbm"), None)
    baseline = next((r for r in summary if r["resolution"] == "rating_event" and r["model"] == "train_mean"), None)
    if primary and baseline and primary["mae"] is not None:
        change = baseline["mae"] - primary["mae"]
        lines += ["", f"评分事件级 MAE 相对训练均分基线改善 {change:.4f} 分（正值为改善）。该差异未经独立外部验证；不能解释为修改场景参数的收益。"]
    lines += ["", "### 各游戏折", ""]
    lines += table(folds, ["held_out_game", *columns], ["测试游戏", *titles])
    unavailable = [r for r in [*summary, *folds] if r["binary_auc"] is None or r["spearman"] is None]
    if unavailable:
        lines += ["", "不可计算指标的原因："]
        for row in unavailable:
            reasons = [row.get("binary_auc_reason"), row.get("correlation_reason")]
            lines.append(f"- {row.get('held_out_game') or '合并折外'} / {row['resolution']} / {row['model']}: " + "; ".join(r for r in reasons if r))
    lines += ["", "## SHAP 模型归因", "",
              "SHAP 来自最终使用全量数据拟合的模型，解释拟合模型的输出，不是折外证据，也不证明参数变化会造成评分变化。特征值与 SHAP 的 Spearman 方向是总体关联，局部关系可能不同；方向 p/q 仍受重复窗口影响。排序使用训练时相同的 session/评分事件等权方式。", ""]
    shap_info = manifest["shap"]
    if shap_info["status"] == "complete":
        lines += table(shap_info["top5"], ["rank", "feature", "event_session_equal_mean_abs_shap", "feature_shap_spearman", "direction"],
                       ["排名", "特征", "平均绝对SHAP", "特征–SHAP Spearman", "方向"])
        lines += ["", f"SHAP 基值 {shap_info['baseline']:.4f}；加法重建最大误差 {shap_info['max_additive_error']:.3g}。"]
    else:
        lines += [f"SHAP 未完成：{shap_info.get('reason')}。没有生成假的贡献结果。"]
    preparation = manifest.get("preparation_manifest")
    prep_summary = None if preparation is None else {key: preparation.get(key) for key in (
        "source_csv_count", "source_game_count", "processed_session_count", "skipped_session_count", "failed_session_count",
        "output_windows", "usable_windows", "real_rating_events_in_processed_sources", "paired_unique_rating_events", "label_distance_seconds")}
    lines += ["", "## 数据准备审计", "",
              f"完整准备 manifest：{manifest['preparation_manifest_path']}。下列摘要来自实际扫描；逐文件列名、错误和清洗计数保留在该manifest中。", "",
              "```json", json.dumps(clean_json(prep_summary), ensure_ascii=False, indent=2) if prep_summary is not None else "null", "```", ""]
    if preparation is not None:
        source_inventory = preparation.get("session_inventory", [])
        excluded = [r for r in source_inventory if r.get("status") != "processed"]
        excluded_counts = {}
        for record in excluded:
            game = record.get("game_name", "unknown")
            excluded_counts[game] = excluded_counts.get(game, 0) + 1
        lines += [f"- 跳过/失败来源按游戏：{excluded_counts}。缺少矩阵的游戏没有用头动参数代替。",
                  f"- 相机位移 >10 源单位分段次数：{sum(r.get('counters', {}).get('position_jumps', 0) for r in source_inventory)}。这是一条工程质量规则。",
                  f"- 有效视图间隔异常重置次数：{sum(r.get('counters', {}).get('derivative_interval_resets', 0) for r in source_inventory)}。普通未采样相机行不重置；真正坏矩阵、相机身份不符或大间隔才断开差分。",
                  f"- 超过720度/s的旋转观测：{sum(r.get('counters', {}).get('rotation_rate_over_flag', 0) for r in source_inventory)}；仅作为质量标记，未据此裁剪真实人工相机转向。", ""]
        feature_quality = preparation.get("feature_summary", {})
        lines += table([dict(feature=name, **record) for name, record in feature_quality.items()],
                       ["feature", "nonempty", "missing_fraction", "min", "max"],
                       ["输入特征", "非空窗口", "缺失比例", "最小值", "最大值"])
    lines += ["",
              "## 使用与限制", "",
              "- 若全量模型可用，保存的 bundle 包含 pipeline、input_feature_cols、fitted_feature_cols、预处理保留/删除清单、评分范围和来源哈希。推理时按 input_feature_cols 顺序输入数值表；pipeline 内执行相同填补及去常量。",
              "- Unity 场景需先以相同采样、单位、窗口定义导出运行数值；尚不能直接把作者设置填入此模型并宣称预暴露效果有效。",
              "- 总相机运动混合头动与导航；记录对象数量受路径和可见性影响；相机坐标跳变与评分近邻配对会影响归因。",
              "- 折外 CSV 同时保留 window_id、rating_event_id、未来标签标记；训练集预测与全量模型 SHAP 不用于宣称测试性能。",
              "- 原冻结协议的三分类 AUC 继续门控未在本次回归补充分析中评估或宣称通过。", "",
              "产出路径与来源哈希详见训练 manifest。"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "features/param_rating_matrix.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "results/param_rating_report.md")
    parser.add_argument("--model", type=Path, default=ROOT / "models/param_rating_model.pkl")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--shap-output", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/param_rating_analysis_2026-10-10.json")
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": args.results_dir / "param_rating_metrics.csv",
        "fold_metrics": args.results_dir / "param_rating_fold_metrics.csv",
        "oof_predictions": args.results_dir / "param_rating_oof_predictions.csv",
        "event_oof_predictions": args.results_dir / "param_rating_event_oof_predictions.csv",
        "shap_values": args.shap_output or args.results_dir / "param_rating_shap_values.csv",
        "shap_importance": args.results_dir / "param_rating_shap_importance.csv",
        "training_manifest": args.results_dir / "param_rating_training_manifest.json",
        "training_log": args.results_dir / "param_rating_training_log.txt",
        "report": args.report, "model": args.model,
    }
    log = logging.getLogger("param_rating_model")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(message)s")
    for handler in [logging.StreamHandler(sys.stdout), logging.FileHandler(paths["training_log"], mode="w", encoding="utf-8")]:
        handler.setFormatter(formatter)
        log.addHandler(handler)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    feature_columns = cfg["feature_columns"]
    if not isinstance(feature_columns, list) or not feature_columns or len(set(feature_columns)) != len(feature_columns):
        raise ValueError("config.feature_columns must be a nonempty unique list")
    if set(feature_columns) & FORBIDDEN_FEATURES:
        raise ValueError("IDs, timestamps, score, label provenance cannot be input features")
    metadata = ["window_id", "game_name", "session_id", "participant_id", "start_time_sec", "end_time_sec", "window_center_sec",
                "rating_event_id", "ms_rating", "rating_time_sec", "rating_time_delta_sec", "label_is_future"]
    frame = pd.read_csv(args.input, dtype={key: str for key in ["window_id", "game_name", "session_id", "participant_id", "rating_event_id"]}, encoding="utf-8-sig")
    missing = sorted(set(metadata + feature_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"missing input columns: {missing}")
    frame[feature_columns] = frame[feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    frame["ms_rating"] = pd.to_numeric(frame["ms_rating"], errors="coerce")
    frame["rating_time_delta_sec"] = pd.to_numeric(frame["rating_time_delta_sec"], errors="coerce")
    frame["label_is_future"] = frame["label_is_future"].astype(str).str.lower().isin({"true", "1", "1.0", "yes"})
    source_window_count = len(frame)
    feature_any = frame[feature_columns].notna().any(axis=1)
    if "window_has_any_feature" in frame:
        declared_feature_any = frame["window_has_any_feature"].astype(str).str.lower().isin({"true", "1", "1.0", "yes"})
        feature_any &= declared_feature_any
    finite_target = np.isfinite(frame["ms_rating"])
    eligible = feature_any & finite_target
    filter_audit = dict(source_windows=source_window_count, eligible_windows=int(eligible.sum()),
                        excluded_feature_empty_or_declared_empty=int((~feature_any).sum()),
                        excluded_nonfinite_target=int((~finite_target).sum()),
                        excluded_total=int((~eligible).sum()),
                        rule="at least one finite configured input feature AND window_has_any_feature if present AND finite MS_rating")
    frame = frame.loc[eligible].reset_index(drop=True)
    prep_path = ROOT / "results/param_rating_preparation_manifest.json"
    preparation = json.loads(prep_path.read_text(encoding="utf-8")) if prep_path.is_file() else None
    manifest = dict(status="started", requested_feature_columns=feature_columns,
                    model_parameters=MODEL_PARAMETERS, variance_threshold=VARIANCE_THRESHOLD,
                    binary_threshold=BINARY_SCORE_THRESHOLD,
                    binary_label="original MS_rating > 2; raw regression prediction ranking",
                    training_weight_policy="equal sessions, equal rating events within session, equal windows within event",
                    training_eligibility_filter=filter_audit,
                    input_path=str(args.input.resolve()), input_sha256=digest(args.input),
                    config_path=str(args.config.resolve()), config_sha256=digest(args.config),
                    preparation_manifest_path=str(prep_path), preparation_manifest=preparation,
                    preparation_manifest_sha256=digest(prep_path) if prep_path.is_file() else None,
                    outputs={name: str(path.resolve()) for name, path in paths.items()},
                    environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                                     numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
                                     lightgbm=lightgbm.__version__, scipy=scipy.__version__, joblib=joblib.__version__),
                    inference_warning="exploratory runtime and near-rating association; not pure design, clinical diagnosis, or causal intervention benefit",
                    original_protocol_gate="not evaluated; derived binary AUC does not replace the frozen macro AUC")
    reasons = []
    if frame.empty:
        reasons.append("input has no paired windows")
    if not frame["window_id"].is_unique:
        reasons.append("window_id is not unique")
    if frame[["session_id", "game_name", "rating_event_id"]].isna().any().any():
        reasons.append("missing session/game/rating identifiers")
    if not np.isfinite(frame["ms_rating"]).all():
        reasons.append("target contains missing/nonfinite ratings")
    if frame["ms_rating"].nunique() < 2:
        reasons.append("target has fewer than two distinct ratings")
    if frame["game_name"].nunique() < 2:
        reasons.append("fewer than two games; Leave-One-Game-Out unavailable")
    if frame.groupby("session_id")["game_name"].nunique().gt(1).any():
        reasons.append("a session maps to multiple games")
    if frame.groupby("rating_event_id")["ms_rating"].nunique().gt(1).any():
        reasons.append("one rating_event_id maps to different scores")
    if frame.groupby("rating_event_id")["session_id"].nunique().gt(1).any():
        reasons.append("one rating_event_id maps to multiple sessions")
    if reasons:
        manifest.update(status="unavailable", reason="; ".join(reasons))
        save_json(paths["training_manifest"], manifest)
        write_report(args.report, manifest, [], [])
        log.error("unavailable: %s", manifest["reason"])
        return 2
    event_labels = frame.drop_duplicates("rating_event_id")
    distances = frame["rating_time_delta_sec"].abs().dropna()
    manifest["data"] = dict(windows=len(frame), rating_events=int(frame["rating_event_id"].nunique()),
                            sessions=int(frame["session_id"].nunique()), games=int(frame["game_name"].nunique()),
                            game_names=sorted(frame["game_name"].unique()), participants=int(frame["participant_id"].nunique()),
                            future_labeled_windows=int(frame["label_is_future"].sum()),
                            window_score_distribution=frame["ms_rating"].value_counts().sort_index().to_dict(),
                            event_score_distribution=event_labels["ms_rating"].value_counts().sort_index().to_dict(),
                            observed_score_range=[float(frame["ms_rating"].min()), float(frame["ms_rating"].max())],
                            absolute_rating_distance_seconds=dict(median=float(distances.median()) if len(distances) else None,
                                                                p90=float(distances.quantile(.9)) if len(distances) else None,
                                                                max=float(distances.max()) if len(distances) else None))
    y = frame["ms_rating"].to_numpy(dtype=float)
    groups = frame["game_name"].to_numpy()
    prediction_rows, fold_details = [], []
    for fold, (train_idx, test_idx) in enumerate(LeaveOneGroupOut().split(frame, y, groups), 1):
        train, test = frame.iloc[train_idx], frame.iloc[test_idx]
        heldout = str(groups[test_idx[0]])
        if set(train["session_id"]) & set(test["session_id"]):
            raise ValueError("session leakage between train and test")
        if set(train["rating_event_id"]) & set(test["rating_event_id"]):
            raise ValueError("rating event leakage between train and test")
        baseline = float(np.average(y[train_idx], weights=sample_weights(train)))
        pipe, detail = fit_model(train, y[train_idx], feature_columns)
        predicted = pipe.predict(test[detail["input_feature_cols"]]) if pipe is not None else np.full(len(test), np.nan)
        detail.update(fold=fold, held_out_game=heldout, train_windows=len(train), test_windows=len(test),
                      participant_overlap=len(set(train["participant_id"].dropna()) & set(test["participant_id"].dropna())),
                      train_target_unique=int(train["ms_rating"].nunique()))
        fold_details.append(detail)
        for name, values in [("lightgbm", predicted), ("train_mean", np.full(len(test), baseline))]:
            out = test[metadata].copy()
            out["fold"], out["held_out_game"], out["model"] = fold, heldout, name
            out["prediction"] = values
            out["absolute_error"] = (out["prediction"] - out["ms_rating"]).abs()
            prediction_rows.append(out)
        log.info("LOGO %s/%s held_out=%s windows=%s participants_overlap=%s retained=%s status=%s",
                 fold, len(np.unique(groups)), heldout, len(test), detail["participant_overlap"], len(detail["fitted_feature_cols"]), detail["status"])
    predictions = pd.concat(prediction_rows, ignore_index=True)
    events = aggregate_events(predictions)
    summaries, fold_rows = [], []
    for resolution, data in [("window", predictions), ("rating_event", events)]:
        for name, work in data.groupby("model", sort=False):
            summaries.append(metric_row(work, name, resolution))
        for (fold, heldout, name), work in data.groupby(["fold", "held_out_game", "model"], sort=False):
            row = metric_row(work, name, resolution, int(fold), heldout)
            detail = next(d for d in fold_details if d["fold"] == fold)
            row.update(participant_overlap=detail["participant_overlap"],
                       fitted_feature_cols_json=json.dumps(detail["fitted_feature_cols"] if name == "lightgbm" else []),
                       fold_model_status=detail["status"] if name == "lightgbm" else "fitted_train_mean")
            fold_rows.append(row)
    pd.DataFrame(summaries).to_csv(paths["metrics"], index=False, encoding="utf-8-sig", float_format="%.17g")
    pd.DataFrame(fold_rows).to_csv(paths["fold_metrics"], index=False, encoding="utf-8-sig", float_format="%.17g")
    predictions.to_csv(paths["oof_predictions"], index=False, encoding="utf-8-sig", float_format="%.17g")
    events.to_csv(paths["event_oof_predictions"], index=False, encoding="utf-8-sig", float_format="%.17g")
    final_pipe, final_detail = fit_model(frame, y, feature_columns)
    manifest.update(folds=len(fold_details), fold_preprocessing=fold_details,
                    final_model=final_detail, metrics=summaries)
    if final_pipe is None:
        manifest.update(status="partial_no_final_model", shap=dict(status="unavailable", reason=final_detail["reason"]))
    else:
        args.model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(dict(pipeline=final_pipe, requested_feature_columns=feature_columns,
                         input_feature_cols=final_detail["input_feature_cols"],
                         fitted_feature_cols=final_detail["fitted_feature_cols"], preprocessing=final_detail,
                         target="original MS_rating", observed_score_range=manifest["data"]["observed_score_range"],
                         model_parameters=MODEL_PARAMETERS, input_sha256=manifest["input_sha256"],
                         warning=manifest["inference_warning"]), args.model)
        try:
            manifest["shap"] = shap_outputs(final_pipe, final_detail, frame, paths["shap_values"], paths["shap_importance"])
            manifest["status"] = "complete" if all(d["status"] == "fitted" for d in fold_details) else "partial_some_folds_unavailable"
        except Exception as exc:
            log.exception("SHAP unavailable")
            manifest.update(status="partial_shap_unavailable", shap=dict(status="unavailable", reason=f"{type(exc).__name__}: {exc}"))
    save_json(paths["training_manifest"], manifest)
    write_report(args.report, manifest, summaries, fold_rows)
    log.info("completed status=%s windows=%s rating_events=%s games=%s", manifest["status"], len(frame), frame["rating_event_id"].nunique(), frame["game_name"].nunique())
    for row in summaries:
        log.info("%s %s MAE=%s RMSE=%s AUC=%s", row["resolution"], row["model"], format_value(row["mae"]), format_value(row["rmse"]), format_value(row["binary_auc"]))
    return 0 if manifest["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
