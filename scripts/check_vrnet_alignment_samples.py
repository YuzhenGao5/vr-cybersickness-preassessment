from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = [
    ("Mars_Miners", "11_2_Mars_Miners", "Mars Miners"),
    ("Army_Men", "28_2_Army_Men", "Army Men"),
    ("Earth_Gym", "5_2_Earth_Gym", "Earth Gym"),
    ("City_Parkour", "146_1_City_Parkour", "City Parkour"),
    ("Creature_Feature", "150_1_Creature_Feature", "Creature Feature"),
    ("Kowloon", "64_1_Kowloon", "Kowloon"),
]


def number_or_none(value: str | None):
    if value is None or not value.strip():
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_time(value: str | None):
    number = number_or_none(value)
    if number is not None:
        return number, "numeric"
    if value:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.timestamp(), "ISO-8601"
        except ValueError:
            pass
    return None, "unknown"


def percentile_pick(items, count):
    if len(items) <= count:
        return items
    positions = sorted(set(round(i * (len(items) - 1) / (count - 1)) for i in range(count)))
    return [items[i] for i in positions]


def inspect_one(archive_name, session, game):
    sample_dir = ROOT / "data" / "raw" / "vrnet" / "samples" / session
    csv_path = sample_dir / "data_file.csv"
    archive_path = ROOT / "data" / "raw" / "vrnet" / "samples" / f"{archive_name}.zip"
    if not csv_path.is_file() or not archive_path.is_file():
        return {"archive": archive_name, "session": session, "game": game,
                "status": "missing csv or archive", "csv_exists": csv_path.is_file(),
                "archive_exists": archive_path.is_file()}

    with zipfile.ZipFile(archive_path) as zf:
        member_names = set(zf.namelist())
    session_prefix = session + "/"
    jpg_members = {name for name in member_names
                   if name.startswith(session_prefix + "video/") and name.lower().endswith(".jpg")}

    csv.field_size_limit(2**31 - 1)
    frames, times, refs = [], [], []
    events, value_counts = [], Counter()
    headers, time_styles = [], Counter()
    missing_frame_rows = missing_time_rows = 0
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        lowered = {h.strip().lower(): h for h in headers}
        frame_col = lowered.get("frame")
        time_col = lowered.get("timestamp")
        video_col = lowered.get("video")
        rating_col = lowered.get("ms_rating")
        if rating_col is None:
            raise ValueError(f"MS_rating column missing: {csv_path}")
        row_count = 0
        for row_count, row in enumerate(reader, start=1):
            frame_raw = (row.get(frame_col) or "").strip() if frame_col else ""
            frame_num = number_or_none(frame_raw)
            frame = int(frame_num) if frame_num is not None else None
            if frame is None:
                missing_frame_rows += 1
            frames.append(frame)

            time_raw = (row.get(time_col) or "").strip() if time_col else ""
            parsed, style = parse_time(time_raw)
            if parsed is None:
                missing_time_rows += 1
            else:
                time_styles[style] += 1
            times.append(parsed)

            ref = (row.get(video_col) or "").strip().replace("\\", "/") if video_col else ""
            refs.append(ref or None)

            rating = (row.get(rating_col) or "").strip()
            if rating:
                value_counts[rating] += 1
                events.append({"row_index": row_count - 1, "frame": frame,
                               "timestamp": parsed, "rating": rating, "video_ref": ref or None})

    valid_times = [t for t in times if t is not None]
    timestamp_numeric = bool(time_styles.get("numeric")) and not time_styles.get("ISO-8601")
    unit = "milliseconds" if timestamp_numeric and valid_times and max(valid_times) > 1e11 else "seconds"
    scale = 1000.0 if unit == "milliseconds" else 1.0
    time_seconds = [(t / scale if t is not None else None) for t in times]
    event_seconds = [(e["timestamp"] / scale if e["timestamp"] is not None else None) for e in events]
    session_times = [t for t in time_seconds if t is not None]
    event_times = [t for t in event_seconds if t is not None]
    gaps = [b - a for a, b in zip(event_times, event_times[1:])]
    median_row_gap = statistics.median([b - a for a, b in zip(session_times, session_times[1:])]) if len(session_times) > 1 else None
    monotonic = all(b >= a for a, b in zip(session_times, session_times[1:]))

    ref_count = frame_matches = 0
    archive_ref_matches = 0
    ref_missing_archive = []
    parsed_ref_examples = []
    for frame, ref in zip(frames, refs):
        if not ref:
            continue
        ref_count += 1
        rel = PurePosixPath(ref)
        if rel.suffix.lower() != ".jpg":
            rel = rel.with_suffix(".jpg")
        stem = rel.stem
        try:
            if frame is not None and int(stem) == frame:
                frame_matches += 1
        except ValueError:
            pass
        member = session_prefix + str(rel)
        if member in member_names:
            archive_ref_matches += 1
        elif len(ref_missing_archive) < 20:
            ref_missing_archive.append(member)
        if len(parsed_ref_examples) < 3:
            parsed_ref_examples.append({"ref": ref, "frame": frame, "member": member,
                                       "frame_matches_stem": frame is not None and stem.isdigit() and int(stem) == frame,
                                       "member_exists": member in member_names})

    # Pick frames around three spread-out FMS events, targeting event time -30s, event, +30s.
    valid_index_time = [(i, t) for i, t in enumerate(time_seconds) if t is not None and refs[i]]
    valid_times_only = [t for _, t in valid_index_time]
    chosen_refs = []
    chosen_frame_details = []
    if valid_index_time and event_times:
        event_positions = percentile_pick(list(range(len(event_times))), min(3, len(event_times)))
        for pos in event_positions:
            et = event_times[pos]
            for offset in (-30.0, 0.0, 30.0):
                target = et + offset
                j = bisect.bisect_left(valid_times_only, target)
                candidates = [k for k in (j - 1, j) if 0 <= k < len(valid_index_time)]
                if not candidates:
                    continue
                best = min(candidates, key=lambda k: abs(valid_times_only[k] - target))
                row_i, row_t = valid_index_time[best]
                ref = refs[row_i]
                rel = PurePosixPath(ref)
                if rel.suffix.lower() != ".jpg":
                    rel = rel.with_suffix(".jpg")
                member = session_prefix + str(rel)
                if member in jpg_members and str(rel) not in chosen_refs:
                    chosen_refs.append(str(rel))
                    chosen_frame_details.append({"target_offset_sec": offset,
                                                 "event_timestamp_sec": et,
                                                 "selected_timestamp_sec": row_t,
                                                 "frame": frames[row_i], "ref": str(rel)})
    # Add spread-out frames until the retained sample contains up to ten images.
    candidates = []
    if valid_index_time:
        candidates = [refs[i] for i, _ in valid_index_time]
    for ref in percentile_pick(candidates, min(10, len(candidates))):
        rel = PurePosixPath(ref)
        if rel.suffix.lower() != ".jpg":
            rel = rel.with_suffix(".jpg")
        if session_prefix + str(rel) in jpg_members and str(rel) not in chosen_refs:
            chosen_refs.append(str(rel))
        if len(chosen_refs) >= 10:
            break
    chosen_refs = chosen_refs[:10]

    bounds = (min(session_times), max(session_times)) if session_times else (None, None)
    full_windows = sum(1 for t in event_times if bounds[0] is not None and t - 30 >= bounds[0] and t + 30 <= bounds[1])
    truncated_windows = len(event_times) - full_windows
    overlapping_pairs = sum(1 for gap in gaps if gap < 60.0)
    expected_fields = [h for h in headers if any(key in h.lower() for key in ("participant", "subject", "session", "game"))]
    timestamp_example = None
    if time_col and times:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f)
            first = next(r, None)
            timestamp_example = first.get(time_col) if first else None
    fmt = "missing"
    if time_col and valid_times:
        if unit == "milliseconds":
            fmt = "numeric integer-like epoch milliseconds" if all(float(t).is_integer() for t in valid_times) else "numeric epoch milliseconds"
        elif time_styles.get("ISO-8601"):
            fmt = "ISO-8601"
        else:
            fmt = "numeric seconds (epoch semantics not confirmed)"

    local_jpgs = list((sample_dir / "video").glob("*.jpg")) if (sample_dir / "video").exists() else []
    return {
        "archive": archive_name, "session": session, "game": game, "status": "analyzed",
        "archive_bytes": archive_path.stat().st_size, "csv_bytes": csv_path.stat().st_size,
        "row_count": row_count, "headers": headers,
        "label_column": "MS_rating", "label_nonempty": sum(value_counts.values()),
        "label_missing": row_count - sum(value_counts.values()), "label_distribution": dict(value_counts),
        "event_times_sec_from_epoch": event_times,
        "event_count_with_timestamp": len(event_times),
        "event_intervals_sec": {"median": statistics.median(gaps) if gaps else None,
                                 "min": min(gaps) if gaps else None, "max": max(gaps) if gaps else None,
                                 "count": len(gaps)},
        "frame_column_present": frame_col is not None,
        "timestamp_column_present": time_col is not None,
        "timestamp_format": fmt, "timestamp_example": timestamp_example,
        "timestamp_valid_rows": len(valid_times), "timestamp_missing_rows": missing_time_rows,
        "timestamp_monotonic": monotonic,
        "timestamp_start_epoch_s": bounds[0], "timestamp_end_epoch_s": bounds[1],
        "duration_sec": bounds[1] - bounds[0] if bounds[0] is not None else None,
        "median_row_interval_sec": median_row_gap if median_row_gap is not None else None,
        "frame_missing_rows": missing_frame_rows,
        "video_column_present": video_col is not None, "video_refs": ref_count,
        "frame_name_match_count": frame_matches,
        "archive_video_match_count": archive_ref_matches,
        "archive_video_missing_count": len(ref_missing_archive),
        "archive_video_missing_examples": ref_missing_archive,
        "video_examples": parsed_ref_examples,
        "zip_jpg_entries": len(jpg_members), "local_jpg_count_before_sampling": len(local_jpgs),
        "selected_jpg_refs": chosen_refs, "selected_jpg_count": len(chosen_refs),
        "selected_jpg_event_window_details": chosen_frame_details,
        "event_window_full_count": full_windows, "event_window_truncated_count": truncated_windows,
        "event_window_overlapping_adjacent_pairs": overlapping_pairs,
        "event_window_possible": bool(event_times and time_col and frame_col and video_col and ref_count),
        "event_window_definition": "nearest timestamped video frame to each target; +/-30 seconds",
        "session_metadata_fields_in_csv": expected_fields,
        "session_name_has_numeric_components": bool(__import__("re").match(r"^\d+_\d+_", session)),
        "participant_id_field_found": any("participant" in h.lower() or "subject" in h.lower() for h in headers),
    }


def show(value, digits=3):
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def create_report(results):
    date = datetime.now().astimezone().strftime("%Y-%m-%d %Z")
    analyzed = [r for r in results if r.get("status") == "analyzed"]
    lines = [
        "# VR.net 多游戏 FMS 事件对齐核验",
        "",
        f"核验日期：{date}。本报告只做样例数据与事件/帧对齐检查；未运行 `preprocess.py`、未训练模型、未做预测实验。",
        "",
        "来源：[VR.net Google Drive game_sessions](https://drive.google.com/drive/folders/1ox_aqi3L_YSodYqMy-9hZOG6QjMYUL4B)；索引 `data/raw/vrnet/main_log.csv`。Drive 按游戏提供 ZIP，本次从五个游戏 ZIP 选择性提取一个 session；另复用此前已下载的 Mars Miners session。根目录约 129 GB 的总 ZIP 未下载。",
        "",
        "`main_log.csv` 中 `MS_rating=1` 是该资源可用性标志，不是 rating 数值。本次实际 rating 均从 session 的 `data_file.csv` 读取。",
        "六个样例 session 的 `MS_rating` 均只在 13 行非空，其余行为空；核验保留这些稀疏事件，没有对空白值作前向填充。",
        "",
        "## 跨 session 概览",
        "",
        "| 游戏 | Session | CSV 行数 | MS_rating 非空事件 | 值分布 | 事件间隔中位数（秒） | 最小–最大（秒） | 时间戳 | 帧名匹配/引用 | ±30 秒窗口 |",
        "|---|---|---:|---:|---|---:|---:|---|---|---|",
    ]
    for r in results:
        if r.get("status") != "analyzed":
            lines.append(f"| {r.get('game')} | `{r.get('session')}` | — | — | — | — | — | — | — | 失败/未分析 |")
            continue
        d = r["event_intervals_sec"]
        vals = ", ".join(f"{k}: {v}" for k, v in sorted(r["label_distribution"].items(), key=lambda x: float(x[0]))) or "无"
        gaps = f"{show(d['min'])}–{show(d['max'])}" if d["min"] is not None else "不可计算"
        frame_match = f"{r['frame_name_match_count']}/{r['video_refs']}"
        win = f"可；完整 {r['event_window_full_count']}/{r['event_count_with_timestamp']}，截断 {r['event_window_truncated_count']}"
        lines.append(f"| {r['game']} | `{r['session']}` | {r['row_count']} | {r['label_nonempty']} | {vals} | {show(d['median'])} | {gaps} | {r['timestamp_format']} | {frame_match} | {win} |")

    lines += ["", "## 各 session 检查明细", ""]
    for r in results:
        if r.get("status") != "analyzed":
            lines += [f"### {r.get('game')} / `{r.get('session')}`", "", "未分析：所需样例文件缺失。", ""]
            continue
        d = r["event_intervals_sec"]
        lines += [
            f"### {r['game']} / `{r['session']}`", "",
            f"- CSV：{r['row_count']:,} 行、{len(r['headers'])} 列；`MS_rating` 非空 {r['label_nonempty']:,}、空值 {r['label_missing']:,}；值分布：{json.dumps(r['label_distribution'], ensure_ascii=False, sort_keys=True)}。",
            f"- 全部列名：`{', '.join(r['headers'])}`。",
            f"- 时间戳：{r['timestamp_format']}；示例 `{r['timestamp_example']}`；有效行 {r['timestamp_valid_rows']:,}/{r['row_count']:,}；单调递增={r['timestamp_monotonic']}；持续 {show(r['duration_sec'])} 秒。",
            f"- 帧号列存在={r['frame_column_present']}；视频引用 {r['video_refs']:,} 条；文件名数字与帧号一致 {r['frame_name_match_count']:,}/{r['video_refs']:,}；引用实际 ZIP 图像成员 {r['archive_video_match_count']:,}/{r['video_refs']:,}；缺失引用 {r['archive_video_missing_count']}。例：`{json.dumps(r['video_examples'], ensure_ascii=False)}`。",
            f"- FMS 事件间隔：中位数 {show(d['median'])} 秒、最小 {show(d['min'])} 秒、最大 {show(d['max'])} 秒；可计算的间隔数 {d['count']}。相邻 ±30 秒窗口重叠对数（事件间隔 <60 秒）={r['event_window_overlapping_adjacent_pairs']}。",
            f"- ±30 秒事件窗口：{('可生成时间窗口并定位最近帧' if r['event_window_possible'] else '不可完整生成')}；完整落在 session 时间范围内 {r['event_window_full_count']} 个，边界截断 {r['event_window_truncated_count']} 个。窗口按事件时间前后各 30 秒取最近帧，不表示窗口内每帧都具有同一个标签。",
            f"- 元数据：CSV 专用 participant/subject/game/session 字段={r['session_metadata_fields_in_csv'] or '未发现'}；独立 participant ID 字段存在={r['participant_id_field_found']}。session 文件夹名包含两个数字段（如 `{r['session']}`），其数字含义没有数据字典支持，故不擅自解释为参与者 ID 或 session 编号。",
            f"- 本地 JPG 保留数：{r['local_jpg_count_before_sampling']}（对齐报告生成时）；计划保留 {r['selected_jpg_count']} 张，引用：`{', '.join(r['selected_jpg_refs'])}`。",
            "",
        ]

    sparse = [r for r in analyzed if r["label_nonempty"] < 5]
    no_gaps = [r for r in analyzed if r["event_intervals_sec"]["count"] == 0]
    lines += [
        "## 事件窗口的可用性与跨游戏差异", "",
        f"- 成功分析 {len(analyzed)} 个 session，覆盖 {len({r['game'] for r in analyzed})} 款游戏。全部样例中每个 session 只取一个 recording；跨游戏标签频数只能描述这些 session，不能当作游戏总体或参与者总体分布。",
        f"- 标签事件少于 5 条的 session：{', '.join(f'{r['session']}（{r['label_nonempty']}）' for r in sparse) if sparse else '本样本无（仍然是单 session 描述统计）'}。没有至少 2 个事件、因此不能计算事件间隔的 session：{', '.join(r['session'] for r in no_gaps) if no_gaps else '无'}。5 条是本报告标识稀疏样例的描述性提醒阈值，不是论文预注册门槛。",
        "- 标签分布需按游戏分别读上表；不同 session 的事件数和取值若有差异，可能同时受 session 时长、报告记录完整性、游戏内容与参与者个体影响，当前样本不能分离这些因素。",
        "- 时间戳、帧号、CSV `video` 引用和 ZIP 中 JPG 条目能建立逐行对应关系，因此时间窗口在工程上可计算。局部窗口须检查是否越过 session 边界；邻近事件间隔少于 60 秒时，±30 秒窗口会重叠。不要把稀疏 `MS_rating` 自动向前/向后填充，也不要无说明地把某次报告复制成整个窗口的逐帧真值。",
        "",
        "## 结论与建议", "",
        "1. **跨场景验证潜力：有，但本次没有完成预测验证。** 样例显示多个游戏 session 都有 `MS_rating` 和视频/场景资源，适合进一步构建跨游戏留一场景的评估数据；仅这 6 个 session、每游戏一个 recording，且没有本次模型训练结果，不能据此宣称模型已具备跨场景准确性。",
        "2. **对齐建议：** 以 rating 事件所在行的 timestamp 为中心，按原始 timestamp 选择 ±30 秒范围内的最近 video/frame 引用；记录边界截断和窗口重叠。先明确标签代表事件时自评还是事件前后窗口结果，避免把事件后帧当成预测事件前风险的输入。",
        "3. **是否下载更多：值得做小批次扩展。** 先依据本报告事件数量、CSV 完整性及申请/许可状态选取多个参与者和游戏，而不是直接下载全量；本次观察到的评分取值也不能用于推断完整量表范围。许可仍需确认。",
        "4. 论文预印本描述口头 FMS 自评与定期提示：[VR.net 论文预印本](https://arxiv.org/html/2306.03381v1)；正式发表记录：[IEEE TVCG 论文记录](https://pubmed.ncbi.nlm.nih.gov/38437109/)，DOI `10.1109/TVCG.2024.3372044`。本报告中的字段/间隔是下载 session 的实际观测；如果与论文提示频率不同，应向数据发布方核对，不从文献描述覆盖原始记录。",
        "",
        "## 复现输入与边界", "",
        "- 选择清单：`results/vrnet_sample_selection.md`；下载/解压日志：`results/vrnet_download_log.txt`；可复核的逐 session CSV 与样例 JPG 位于 `data/raw/vrnet/samples/`。",
        "- 本核验不是统计显著性分析、模型评估或实验；没有运行 `preprocess.py`、没有训练模型、没有提交 Git commit。",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="results/vrnet_alignment_metrics.json")
    parser.add_argument("--report", default="results/vrnet_alignment_check.md")
    args = parser.parse_args()
    results = []
    for sample in SAMPLES:
        print(f"ANALYZING {sample[1]} ({sample[2]})", flush=True)
        result = inspect_one(*sample)
        results.append(result)
        print(json.dumps({"session": result.get("session"), "status": result.get("status"), "rows": result.get("row_count"), "events": result.get("label_nonempty"), "distribution": result.get("label_distribution"), "intervals": result.get("event_intervals_sec"), "images": result.get("selected_jpg_count")}, ensure_ascii=False), flush=True)
    metrics_path = ROOT / args.metrics
    report_path = ROOT / args.report
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(create_report(results), encoding="utf-8")
    print(json.dumps([{"session": r.get("session"), "status": r.get("status"),
                       "rows": r.get("row_count"), "events": r.get("label_nonempty"),
                       "distribution": r.get("label_distribution"),
                       "intervals": r.get("event_intervals_sec"),
                       "images": r.get("selected_jpg_count"),
                       "refs_match": f"{r.get('frame_name_match_count')}/{r.get('video_refs')}"}
                      for r in results], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



