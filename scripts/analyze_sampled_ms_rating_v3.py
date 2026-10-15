#!/usr/bin/env python3
"""Summarize observed VR.net MS_rating values in retained sampled session CSVs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd


def fmt_number(value) -> str:
    if pd.isna(value):
        return "unknown"
    return f"{float(value):.4f}".rstrip("0").rstrip(".")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sampled_root = root / "data/raw/vrnet/sampled_csv"
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    selection_path = results / "zip_download_selection.csv"
    session_output = results / "sampled_ms_rating_v3.csv"
    summary_output = results / "sampled_ms_rating_summary_v3.md"
    availability_output = results / "sampled_label_availability_v3.md"

    selected = pd.read_csv(selection_path, encoding="utf-8-sig") if selection_path.exists() else pd.DataFrame()
    files = sorted(sampled_root.rglob("data_file.csv")) if sampled_root.exists() else []
    rows: list[dict] = []
    global_counts: Counter = Counter()
    errors: list[str] = []

    for path in files:
        game = path.parent.parent.name
        session = path.parent.name
        try:
            header = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
            label_col = next((column for column in header.columns if str(column).strip().casefold() == "ms_rating"), None)
            if label_col is None:
                raise KeyError(f"MS_rating column absent; columns={list(header.columns)!r}")
            counts: Counter = Counter()
            n_rows = 0
            value_count = 0
            value_sum = 0.0
            value_max = None
            for chunk in pd.read_csv(
                path,
                encoding="utf-8-sig",
                usecols=[label_col],
                chunksize=250_000,
                low_memory=False,
            ):
                n_rows += len(chunk)
                values = pd.to_numeric(chunk[label_col], errors="coerce").dropna()
                if values.empty:
                    continue
                chunk_counts = values.value_counts(dropna=True)
                counts.update({float(value): int(count) for value, count in chunk_counts.items()})
                value_count += int(values.size)
                value_sum += float(values.sum())
                chunk_max = float(values.max())
                value_max = chunk_max if value_max is None else max(value_max, chunk_max)
            global_counts.update(counts)
            value_counts = {fmt_number(value): int(count) for value, count in sorted(counts.items())}
            rows.append({
                "game_name": game,
                "session_name": session,
                "data_file_csv": str(path.relative_to(root)),
                "n_rows": n_rows,
                "nonempty_MS_rating_count": value_count,
                "MS_rating_unique_values": json.dumps([float(value) for value in sorted(counts)], ensure_ascii=False),
                "MS_rating_value_counts": json.dumps(value_counts, ensure_ascii=False),
                "max_MS_rating": value_max if value_count else pd.NA,
                "mean_MS_rating": value_sum / value_count if value_count else pd.NA,
                "status": "ok" if value_count else "no_numeric_MS_rating_values",
            })
        except Exception as exc:
            errors.append(f"{path.relative_to(root)}: {type(exc).__name__}: {exc}")
            rows.append({
                "game_name": game,
                "session_name": session,
                "data_file_csv": str(path.relative_to(root)),
                "n_rows": pd.NA,
                "nonempty_MS_rating_count": pd.NA,
                "MS_rating_unique_values": "[]",
                "MS_rating_value_counts": "{}",
                "max_MS_rating": pd.NA,
                "mean_MS_rating": pd.NA,
                "status": f"error: {type(exc).__name__}: {exc}",
            })

    columns = [
        "game_name", "session_name", "data_file_csv", "n_rows", "nonempty_MS_rating_count",
        "MS_rating_unique_values", "MS_rating_value_counts", "max_MS_rating", "mean_MS_rating", "status",
    ]
    result = pd.DataFrame(rows, columns=columns)
    result.to_csv(session_output, index=False, encoding="utf-8-sig")

    numeric_sessions = result.loc[result["max_MS_rating"].notna()].copy() if not result.empty else result
    gt2 = numeric_sessions.loc[numeric_sessions["max_MS_rating"].gt(2)] if not numeric_sessions.empty else numeric_sessions
    gt5 = numeric_sessions.loc[numeric_sessions["max_MS_rating"].gt(5)] if not numeric_sessions.empty else numeric_sessions
    extracted_games = sorted(result.loc[result["status"].eq("ok"), "game_name"].unique().tolist()) if not result.empty else []
    selected_games = selected["game_name"].astype(str).tolist() if "game_name" in selected else []
    downloaded_games = set(extracted_games)

    global_lines = [f"| {fmt_number(value)} | {count} |" for value, count in sorted(global_counts.items())]
    if not global_lines:
        global_lines = ["| 无可计算数据 | 无 |"]
    game_lines = []
    if not numeric_sessions.empty:
        for game, group in numeric_sessions.groupby("game_name", sort=True):
            max_counts = Counter(group["max_MS_rating"].astype(float).tolist())
            distribution = ", ".join(f"{fmt_number(value)}: {count}" for value, count in sorted(max_counts.items()))
            game_lines.append(f"| {game} | {len(group)} | {distribution} |")
    if not game_lines:
        game_lines = ["| 无可计算数据 | 0 | unknown |"]

    summary = [
        "# 抽样 session 的 MS_rating 汇总",
        "",
        "本报告只汇总实际下载并读取的 `data_file.csv`；未下载或无法解析的值不以 0 代替。",
        "",
        f"- 清单选定游戏数：{len(selected_games)}",
        f"- 实际发现的 session CSV 数：{len(files)}",
        f"- 至少有一个数值 MS_rating 的 session 数：{len(numeric_sessions)}",
        f"- 成功读取的游戏：{', '.join(extracted_games) if extracted_games else '无'}",
        f"- 有 `max(MS_rating) > 2` 的 session：{len(gt2)} / {len(numeric_sessions)} 可计算 session" if len(numeric_sessions) else "- 有 `max(MS_rating) > 2` 的 session：unknown（无可计算 session）",
        f"- 有 `max(MS_rating) > 5` 的 session：{len(gt5)} / {len(numeric_sessions)} 可计算 session" if len(numeric_sessions) else "- 有 `max(MS_rating) > 5` 的 session：unknown（无可计算 session）",
        "",
        "## 全局 MS_rating 原始值频数",
        "",
        "| MS_rating | 频数 |",
        "|---:|---:|",
        *global_lines,
        "",
        "## 各游戏的 session 级 max 分布",
        "",
        "| 游戏 | 可计算 session 数 | max(MS_rating): session 数 |",
        "|---|---:|---|",
        *game_lines,
        "",
        f"数据质量/读取错误：{len(errors)}。",
    ]
    if errors:
        summary.extend(["", "```text", *errors, "```"])
    summary_output.write_text("\n".join(summary) + "\n", encoding="utf-8")

    status_counts = selected["download_status"].value_counts(dropna=False).to_dict() if "download_status" in selected else {}
    mid_high_games = sorted(gt2["game_name"].unique().tolist()) if not gt2.empty else []
    high_games = sorted(gt5["game_name"].unique().tolist()) if not gt5.empty else []
    if not len(numeric_sessions):
        judgement = "目前不能判断 VR.net session 级标签是否具有足够多样性，也不能据此认定其适合训练风险模型：本次没有成功读取任何带数值 MS_rating 的 session CSV。"
        next_steps = "取得单个游戏包内的 CSV 导出/较小归档，或在有足够磁盘与网络配额的环境中按同样顺序逐包下载；先复核标签粒度和分布，再决定训练设计。"
    else:
        distinct_max = int(numeric_sessions["max_MS_rating"].nunique())
        if len(gt5) == 0:
            judgement = (
                f"本次 {len(numeric_sessions)} 个可计算 session 的 max 值包含 {distinct_max} 个不同取值，"
                f"其中 {len(gt2)} 个 session 的 max > 2，但没有 max > 5 的 session。"
                "因此，这批样本可用于标签读取和 session 汇总流程验证，也可谨慎探索 `max > 2` 的二分问题；"
                "它不适合按高风险阈值 `MS_rating >= 6` 训练低/中/高三分类模型，因为高风险类别完全缺失。"
                "只有 3 个游戏，当前结果也不能证明跨游戏泛化能力。"
            )
            next_steps = (
                "继续定向下载经过来源核实的高速飞行、赛车或过山车类游戏，优先寻找出现 `MS_rating >= 6` 的 session。"
                "扩展后按游戏和参与者分组检查类别覆盖，再决定训练设计；划分数据时按游戏/参与者分组，避免同一来源泄漏。"
            )
        else:
            judgement = (
                f"本次 {len(numeric_sessions)} 个可计算 session 的 max 值包含 {distinct_max} 个不同取值，"
                f"其中 {len(gt2)} 个 max > 2、{len(gt5)} 个 max > 5。"
                "这表明抽样中出现了中高标签，但样本仅来自 3 个游戏，只能证明标签管道可用，不能证明模型有效或具有跨游戏泛化能力。"
            )
            next_steps = "继续扩大游戏覆盖，并按游戏和参与者分组评估类别数量；训练/验证划分必须防止同一游戏或参与者泄漏。"

    availability = [
        "# VR.net 抽样标签可用性判断",
        "",
        f"- 选定游戏数：{len(selected_games)}",
        f"- 下载并提取出至少一个 CSV 的游戏数：{len(downloaded_games)}",
        f"- 提取出的 session CSV 数：{len(files)}",
        f"- 有数值 MS_rating 的 session 数：{len(numeric_sessions)}",
        f"- 下载状态：`{json.dumps(status_counts, ensure_ascii=False)}`",
        f"- 全局 MS_rating 原始值分布：`{json.dumps({fmt_number(k): v for k, v in sorted(global_counts.items())}, ensure_ascii=False) if global_counts else 'unknown（无可计算 CSV）'}`",
        f"- max(MS_rating) > 2：{len(gt2)} 个 session" if len(numeric_sessions) else "- max(MS_rating) > 2：unknown（无可计算 session）",
        f"- max(MS_rating) > 5：{len(gt5)} 个 session" if len(numeric_sessions) else "- max(MS_rating) > 5：unknown（无可计算 session）",
        f"- 有 max > 2 的游戏：{', '.join(mid_high_games) if mid_high_games else ('无' if len(numeric_sessions) else 'unknown')}",
        f"- 有 max > 5 的游戏：{', '.join(high_games) if high_games else ('无' if len(numeric_sessions) else 'unknown')}",
        "",
        "## 判断",
        "",
        judgement,
        "",
        "下载状态不等同于标签缺失；未下载的 session 标签应视为未知。即使出现多个 max 值，也只表明观察到取值变化，不代表样本量足以训练或验证模型。",
        "",
        "## 下一步建议",
        "",
        next_steps,
        "",
        "本轮未训练模型、未下载视频，也未执行 Git commit。",
    ]
    availability_output.write_text("\n".join(availability) + "\n", encoding="utf-8")
    print(f"selected_games={len(selected_games)} csv_files={len(files)} labeled_sessions={len(numeric_sessions)} max_gt2={len(gt2) if len(numeric_sessions) else 'unknown'} max_gt5={len(gt5) if len(numeric_sessions) else 'unknown'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
