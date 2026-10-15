#!/usr/bin/env python3
"""Describe matrix-feature versus rating associations once per real rating event.

Windows assigned to an event are averaged feature by feature without imputation.
Correlation p values and their 18-comparison BH adjustments are descriptive:
ratings repeat within sessions, and only the observed games are represented.
No models are fitted by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_REPETITIONS = 500


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def adjust_bh(pvalues: np.ndarray, family_size: int) -> np.ndarray:
    """Use the configured 18-test family, even for undefined constant features."""
    adjusted = np.full(len(pvalues), np.nan)
    valid = np.flatnonzero(np.isfinite(pvalues))
    if not len(valid):
        return adjusted
    order = valid[np.argsort(pvalues[valid], kind="stable")]
    raw = pvalues[order] * family_size / np.arange(1, len(order) + 1)
    adjusted[order] = np.minimum(1.0, np.minimum.accumulate(raw[::-1])[::-1])
    return adjusted


def correlate(x: np.ndarray, y: np.ndarray) -> dict:
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    result = {"paired_events": int(len(x)), "feature_unique_values": int(len(np.unique(x))),
              "score_unique_values": int(len(np.unique(y))), "spearman_rho": np.nan,
              "spearman_p": np.nan, "pearson_r": np.nan, "pearson_p": np.nan,
              "correlation_status": "unavailable", "unavailable_reason": ""}
    if len(x) < 3:
        result["unavailable_reason"] = "fewer_than_three_finite_event_pairs"
    elif np.ptp(x) == 0:
        result["unavailable_reason"] = "constant_feature"
    elif np.ptp(y) == 0:
        result["unavailable_reason"] = "constant_rating"
    else:
        s, p = spearmanr(x, y), pearsonr(x, y)
        result.update(spearman_rho=float(s.statistic), spearman_p=float(s.pvalue),
                      pearson_r=float(p.statistic), pearson_p=float(p.pvalue),
                      correlation_status="available")
    return result


def aggregate_events(frame: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, dict]:
    required = {"window_id", "rating_event_id", "session_id", "game_name", "ms_rating", *feature_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    if frame["window_id"].duplicated().any():
        raise ValueError("Input window_id values are not unique")
    numeric = frame[feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    score = pd.to_numeric(frame["ms_rating"], errors="coerce")
    valid = numeric.notna().any(axis=1) & np.isfinite(score)
    valid &= frame["rating_event_id"].notna() & frame["rating_event_id"].astype(str).str.strip().ne("")
    valid &= frame["session_id"].notna() & frame["game_name"].notna()
    if "window_has_any_feature" in frame:
        valid &= truthy(frame["window_has_any_feature"])
    work = frame.loc[valid, ["rating_event_id", "session_id", "game_name"]].copy()
    work["ms_rating"] = score.loc[valid]
    work[feature_columns] = numeric.loc[valid]
    if work.empty:
        raise ValueError("No finite labelled feature windows are available")
    check = work.groupby("rating_event_id", sort=False)[["session_id", "game_name", "ms_rating"]].nunique(dropna=False)
    if check.gt(1).any(axis=None):
        raise ValueError("A rating_event_id has inconsistent session, game, or actual score")
    session_games = work[["session_id", "game_name"]].drop_duplicates()
    if session_games["session_id"].duplicated().any():
        raise ValueError("A session_id maps to multiple games")
    aggregation = {"session_id": "first", "game_name": "first", "ms_rating": "first"}
    aggregation.update({name: "mean" for name in feature_columns})
    events = work.groupby("rating_event_id", sort=True).agg(aggregation).reset_index()
    windows = work.groupby("rating_event_id", sort=True).size()
    events["assigned_valid_windows"] = events["rating_event_id"].map(windows).astype(int)
    detail = {"input_windows": int(len(frame)), "valid_windows": int(valid.sum()),
              "excluded_windows": int((~valid).sum()), "unique_actual_rating_events": int(len(events)),
              "sessions": int(events["session_id"].nunique()), "games": int(events["game_name"].nunique()),
              "event_counts_by_game": {str(k): int(v) for k, v in events["game_name"].value_counts().sort_index().items()},
              "label_distribution": {str(k): int(v) for k, v in events["ms_rating"].value_counts().sort_index().items()}}
    return events, detail


def bootstrap_spearman(events: pd.DataFrame, features: list[str], seed: int) -> dict[str, dict]:
    """Resample whole sessions within each observed game; keep their rating events."""
    rng = np.random.default_rng(seed)
    x, y = events[features].to_numpy(dtype=float), events["ms_rating"].to_numpy(dtype=float)
    strata = []
    for _, game in events.groupby("game_name", sort=True):
        groups = [session.index.to_numpy() for _, session in game.groupby("session_id", sort=True)]
        strata.append(groups)
    values = np.full((BOOTSTRAP_REPETITIONS, len(features)), np.nan)
    eligible = np.array([correlate(x[:, i], y)["correlation_status"] == "available" for i in range(len(features))])
    for repetition in range(BOOTSTRAP_REPETITIONS):
        indices = np.concatenate([np.concatenate([groups[i] for i in rng.integers(0, len(groups), size=len(groups))])
                                  for groups in strata])
        target = y[indices]
        for i in np.flatnonzero(eligible):
            feature = x[indices, i]
            finite = np.isfinite(feature) & np.isfinite(target)
            usable_x, usable_y = feature[finite], target[finite]
            if len(usable_x) >= 3 and np.ptp(usable_x) > 0 and np.ptp(usable_y) > 0:
                values[repetition, i] = float(spearmanr(usable_x, usable_y).statistic)
    result = {}
    for i, feature in enumerate(features):
        usable = values[np.isfinite(values[:, i]), i]
        result[feature] = {"spearman_bootstrap_ci95_low": float(np.quantile(usable, 0.025)) if len(usable) else np.nan,
                           "spearman_bootstrap_ci95_high": float(np.quantile(usable, 0.975)) if len(usable) else np.nan,
                           "bootstrap_valid_repetitions": int(len(usable)),
                           "bootstrap_requested_repetitions": BOOTSTRAP_REPETITIONS,
                           "bootstrap_status": "available" if len(usable) else "unavailable_constant_or_insufficient_pairs"}
    return result


def value(number) -> str:
    return "不可计算" if pd.isna(number) else f"{float(number):.4f}"


def report_text(rows: pd.DataFrame, detail: dict, input_path: Path, config_path: Path, seed: int) -> str:
    lines = ["# 场景运行特征与真实评分的相关性", "",
             f"输入：`{input_path.resolve()}`；配置：`{config_path.resolve()}`。", "",
             f"{detail['valid_windows']} 个有效5秒窗口归并为 {detail['unique_actual_rating_events']} 次实测评分，"
             f"覆盖 {detail['sessions']} 个session、{detail['games']} 个已观察游戏。",
             "每个 rating_event_id 只计一次：对它关联的所有有效窗口逐特征取均值，保留该事件真实 MS_rating；"
             "原本为窗口 max/std 的列也取窗口间均值，因此不是整段暴露的全局 max/std。缺失值不补齐，各相关系数仅使用该特征与评分均有限的事件。", "",
             "这是特征与真实评分的关联，不是预测与评分的相关，也不是SHAP归因。最近评分联结可能包含未来评分和较远评分；"
             "窗口平均不能将这种配对转换为前瞻预测或严格评分前历史。", "",
             "Spearman适合查看单调关系，Pearson描述线性关系。全局相关可能包含游戏间差异，必须同时查看每个游戏内结果。"
             "每个scope、每种相关方法分别按配置中18项假设进行Benjamini–Hochberg校正；全常量/无法计算项保留缺失，校正仍使用18作为假设数。", "",
             "p值和BH q值仅作描述：同一session的重复评分不独立，只有这两个已观察游戏，不能把常规相关检验当作独立显著性证据。", "",
             f"全局Spearman的95%区间来自固定 {BOOTSTRAP_REPETITIONS} 次bootstrap（seed={seed}）：在每个game内等数量有放回抽样sessions，"
             "保留被抽session的所有评分事件。该区间条件于已观察游戏，反映这些session的抽样波动；"
             "不包含新游戏泛化不确定性，也不重新训练模型。游戏内相关未另作bootstrap。", "",
             "运行相机运动包含用户头动，线速度仍为源坐标单位/秒；对象计数是渲染记录条目，所有字段尚未认证为作者设计设置。", "",
             "输入SHA256：`" + sha256(input_path) + "`。", "",
             "配置SHA256：`" + sha256(config_path) + "`。", ""]
    for scope in rows["scope"].drop_duplicates():
        part = rows.loc[rows["scope"].eq(scope)]
        lines.extend([f"## {scope}", "", "| 特征 | 评分事件 | Spearman | 95% bootstrap区间 | Spearman p / BH q | Pearson | Pearson p / BH q | 状态 |",
                      "|---|---:|---:|---|---|---:|---|---|"])
        for _, row in part.iterrows():
            ci = (f"[{value(row['spearman_bootstrap_ci95_low'])}, {value(row['spearman_bootstrap_ci95_high'])}]"
                  if scope == "global" and pd.notna(row["spearman_bootstrap_ci95_low"]) else "—")
            state = "可计算" if row["correlation_status"] == "available" else row["unavailable_reason"]
            lines.append(f"| {row['feature']} | {int(row['paired_events'])} | {value(row['spearman_rho'])} | {ci} | "
                         f"{value(row['spearman_p'])} / {value(row['spearman_bh_q'])} | {value(row['pearson_r'])} | "
                         f"{value(row['pearson_p'])} / {value(row['pearson_bh_q'])} | {state} |")
        lines.append("")
    lines.extend(["## 覆盖记录", "", "```json", json.dumps(detail, ensure_ascii=False, indent=2), "```", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "features/param_rating_matrix.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/param_rating_analysis_2026-10-10.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/param_rating_feature_correlations.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "results/param_rating_feature_association.md")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))
    features = cfg["feature_columns"]
    if len(features) != 18 or len(set(features)) != len(features):
        raise ValueError("Expected the fixed list of 18 distinct configured feature columns")
    frame = pd.read_csv(args.input, encoding="utf-8-sig", low_memory=False)
    events, detail = aggregate_events(frame, features)
    events = events.reset_index(drop=True)
    seed = int(cfg.get("random_seed", 42))
    bootstrap = bootstrap_spearman(events, features, seed)
    rows = []
    scopes = [("global", events), *[(str(name), game) for name, game in events.groupby("game_name", sort=True)]]
    for scope, selected in scopes:
        for feature in features:
            row = {"scope": scope, "feature": feature, "total_events_in_scope": int(len(selected)),
                   "sessions_in_scope": int(selected["session_id"].nunique()), "games_in_scope": int(selected["game_name"].nunique()),
                   "event_aggregation": "featurewise_mean_of_valid_assigned_5s_windows_per_rating_event_id",
                   "p_value_interpretation": "descriptive_only_repeated_session_ratings_not_independent",
                   "bh_family_size": len(features),
                   **correlate(selected[feature].to_numpy(dtype=float), selected["ms_rating"].to_numpy(dtype=float))}
            if scope == "global":
                row.update(**bootstrap[feature], bootstrap_scope="sessions_stratified_within_observed_games_not_new_game_generalization")
            else:
                row.update(spearman_bootstrap_ci95_low=np.nan, spearman_bootstrap_ci95_high=np.nan,
                           bootstrap_valid_repetitions=0, bootstrap_requested_repetitions=0,
                           bootstrap_status="not_requested_for_game_scope", bootstrap_scope="not_requested")
            rows.append(row)
    output = pd.DataFrame(rows)
    output["spearman_bh_q"] = np.nan
    output["pearson_bh_q"] = np.nan
    for _, group in output.groupby("scope", sort=False):
        for method in ("spearman", "pearson"):
            output.loc[group.index, method + "_bh_q"] = adjust_bh(group[method + "_p"].to_numpy(dtype=float), len(features))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig", float_format="%.17g")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_text(output, detail, args.input, args.config, seed), encoding="utf-8")
    print(json.dumps({"status": "success", **detail, "correlation_rows": len(output),
                      "bootstrap_repetitions": BOOTSTRAP_REPETITIONS, "output": str(args.output), "report": str(args.report)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
