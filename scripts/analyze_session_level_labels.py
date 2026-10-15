#!/usr/bin/env python3
"""Summarize session-level self-reports for the locally sampled VR.net CSVs."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def session_duration_seconds(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 2:
        return float("nan")
    span = float(values.max() - values.min())
    return span / 1000.0 if float(values.abs().median()) > 1e11 else span


def fmt(value: object, digits: int = 4) -> str:
    return "unknown" if pd.isna(value) else f"{float(value):.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", default="results/vrnet_sampled_sessions_v2.csv")
    parser.add_argument("--data-dir", default="data/raw/vrnet/sampled_csv")
    parser.add_argument("--output-csv", default="results/session_level_labels.csv")
    parser.add_argument("--summary", default="results/session_level_labels_summary.md")
    parser.add_argument("--availability", default="results/session_level_label_availability.md")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    resolve = lambda value: (root / value).resolve() if not Path(value).is_absolute() else Path(value)
    paths = {name: resolve(value) for name, value in vars(args).items()}
    selection = pd.read_csv(paths["selection"], encoding="utf-8-sig")
    records = []
    for row in selection.to_dict("records"):
        session_name = str(row.get("session_name", ""))
        csv_path = paths["data_dir"] / session_name / "data_file.csv"
        result = {
            "session_name": session_name, "game_name": row.get("game_name", ""),
            "max_MS_rating": np.nan, "mean_MS_rating": np.nan,
            "nonempty_MS_rating_count": np.nan, "session_duration_sec": np.nan,
            "source_rows": np.nan, "analysis_status": "missing_data_file",
            "download_status": row.get("download_status", "unknown"),
            "csv_path": str(csv_path.relative_to(root)) if csv_path.is_relative_to(root) else str(csv_path),
        }
        if csv_path.is_file():
            try:
                header = pd.read_csv(csv_path, encoding="utf-8-sig", nrows=0)
                if "MS_rating" not in header.columns:
                    raise ValueError("data_file.csv has no MS_rating column")
                usecols = ["MS_rating"] + (["timestamp"] if "timestamp" in header.columns else [])
                frame = pd.read_csv(csv_path, encoding="utf-8-sig", usecols=usecols)
                ratings = pd.to_numeric(frame["MS_rating"], errors="coerce").dropna()
                result["source_rows"] = len(frame)
                result["nonempty_MS_rating_count"] = int(len(ratings))
                result["max_MS_rating"] = float(ratings.max()) if len(ratings) else np.nan
                result["mean_MS_rating"] = float(ratings.mean()) if len(ratings) else np.nan
                result["session_duration_sec"] = session_duration_seconds(frame["timestamp"]) if "timestamp" in frame else np.nan
                result["analysis_status"] = "analyzed" if len(ratings) else "analyzed_no_nonempty_ratings"
            except Exception as exc:
                result["analysis_status"] = f"parse_error:{type(exc).__name__}:{exc}"
        records.append(result)
    report = pd.DataFrame(records)
    paths["output_csv"].parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(paths["output_csv"], index=False, encoding="utf-8-sig")
    valid = report.loc[report["analysis_status"].isin(["analyzed", "analyzed_no_nonempty_ratings"])]
    has_max = valid["max_MS_rating"].dropna()
    max_counts = Counter(has_max.tolist())
    mean_counts = Counter(valid["mean_MS_rating"].dropna().round(2).tolist())
    count_above2 = int((has_max > 2).sum()) if len(has_max) else None
    count_above5 = int((has_max > 5).sum()) if len(has_max) else None
    log_path = root / "results/sampled_csv_download_log_v2.txt"
    log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else "Download log missing."
    cap_blocked = "SKIPPED: archive is" in log_text and "greater than 500 MiB" in log_text
    common = [
        "**流程验证，非最终结论。**", "",
        f"- 抽样 session：{len(selection)}；有本地 `data_file.csv` 且可计算：{len(valid)}。",
        f"- 非空 MS_rating 事件总数：{int(valid['nonempty_MS_rating_count'].sum()) if len(valid) else '无法计算'}。",
        f"- `max(MS_rating)>2` 的 session：{count_above2 if count_above2 is not None else '无法计算（无已分析 session）'}。",
        f"- `max(MS_rating)>5` 的 session：{count_above5 if count_above5 is not None else '无法计算（无已分析 session）'}。", "",
    ]
    summary = ["# VR.net session 级 MS_rating 汇总", "", *common, "## max(MS_rating) 分布", ""]
    if max_counts:
        summary.extend(["| max 值 | session 数 |", "|---:|---:|"])
        summary.extend(f"| {value:g} | {count} |" for value, count in sorted(max_counts.items()))
    else:
        summary.append("无已下载且可分析的 session CSV；分布不可计算。")
    summary.extend(["", "## mean(MS_rating) 分布", ""])
    if mean_counts:
        summary.extend(["| session 均值（四舍五入至 2 位） | session 数 |", "|---:|---:|"])
        summary.extend(f"| {value:.2f} | {count} |" for value, count in sorted(mean_counts.items()))
    else:
        summary.append("无已下载且可分析的 session CSV；分布不可计算。")
    summary.extend(["", f"ZIP 体积限制拦截：{'是' if cap_blocked else '否/未记录'}。", ""])
    paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary"].write_text("\n".join(summary), encoding="utf-8")

    if len(valid):
        conclusion = "可描述抽样中 session 级评分范围，但单次抽样仍需结合类别覆盖与游戏分布判断；不能仅凭 max/mean 推断风险模型泛化能力。"
    elif cap_blocked:
        conclusion = (
            "无法判断 VR.net session 级 self-report 的实际 max/mean 多样性，因为抽样到的 CSV 均未能下载。"
            "Drive 实际提供的是按游戏打包的 ZIP，而不是可单独选取的 session ZIP；全部超过 500 MiB 限制，因此按规则全部跳过。"
        )
    else:
        conclusion = "没有可分析的 session CSV；当前证据不足以判断标签多样性。"
    availability = [
        "# VR.net session 级标签可用性", "", *common,
        "## max(MS_rating) 分布", "",
    ]
    availability.extend((f"- {value:g}: {count} session" for value, count in sorted(max_counts.items())) if max_counts else ["- 无数据：没有成功分析的 session CSV。"])
    availability.extend(["", "## 判断", "", conclusion,
                         "", "## 下载约束与下一步", "",
                         "- 采样清单和 Drive 元数据列在 `results/vrnet_sampled_sessions_v2.csv` 与 `results/sampled_csv_download_log_v2.txt`。",
                         "- Drive 中的归档按游戏打包，单包超过 500 MiB；本次没有下载 ZIP、没有解压视频，也没有生成虚构的标签统计。",
                         "- 下一步需要数据维护者提供单 session 的 `data_file.csv`/小型 CSV 包，或提供可在 500 MiB 内单独访问的 session 导出；在获得实际 CSV 前，不训练 session 级风险模型。", ""])
    paths["availability"].parent.mkdir(parents=True, exist_ok=True)
    paths["availability"].write_text("\n".join(availability), encoding="utf-8")
    print(f"sampled={len(selection)} analyzed={len(valid)} max_gt2={count_above2} max_gt5={count_above5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
