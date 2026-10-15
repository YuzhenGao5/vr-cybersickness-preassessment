"""Audit the original VR.net paired self-report files without editing labels.

This is a descriptive, local-only inventory. It neither trains a ranking model
nor treats filename participant IDs, manual source strings, or game differences
as a verified experimental design. Raw rows and near duplicates are retained.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ALIASES = {"VR Monster Awakens": "Monster Awakens"}
FILE_PATTERN = re.compile(r"^(\d+)\s+(VRLOG-\S+)\s+(.+)$")
PRECISE_TIMESTAMP = re.compile(r"\d{16}")
NATIVE_RGB = re.compile(r"s\d+\.jpg", re.IGNORECASE)
NEAR_DUPLICATE_SEC = 2.0  # Audit flag only; no rating is removed.
OBSERVATIONAL = "Exploratory descriptive audit, not a final conclusion"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def path_label(path: Path, project: Path) -> str:
    try:
        return path.resolve().relative_to(project).as_posix()
    except ValueError:
        return str(path.resolve())


def timestamp_iso(value: int | None) -> str:
    if value is None:
        return "unknown"
    return datetime.fromtimestamp(value / 1_000_000, timezone.utc).isoformat()


def native_inventory(raw_root: Path, project: Path) -> dict[str, list[dict]]:
    inventory = defaultdict(list)
    for log in sorted(raw_root.rglob("gamedata.log")):
        directory = log.parent
        match = re.search(r"VRLOG-\d+", directory.name)
        if not match:
            continue
        rgb_count = sum(
            1 for item in directory.iterdir()
            if item.is_file() and NATIVE_RGB.fullmatch(item.name)
        )
        voice = directory / "voice.csv"
        inventory[match.group()].append({
            "directory": path_label(directory, project),
            "rgb_count": rgb_count,
            "voice_sha256": digest(voice) if voice.is_file() else "unknown",
        })
    return inventory


def near_pairs(rows: list[dict]) -> tuple[int, int]:
    # Rounded scientific-notation timestamps have no usable precision.
    precise = sorted((row for row in rows if row["timestamp_us"] is not None),
                     key=lambda row: (row["timestamp_us"], row["raw_row"]))
    near = same = 0
    for first, second in zip(precise, precise[1:]):
        delta = (second["timestamp_us"] - first["timestamp_us"]) / 1_000_000
        if 0 <= delta <= NEAR_DUPLICATE_SEC:
            near += 1
            same += first["score"] == second["score"]
    return near, same


def inspect_session(path: Path, native: dict, project: Path) -> dict:
    match = FILE_PATTERN.fullmatch(path.stem)
    if match is None:
        raise ValueError(f"Unrecognized self-report filename: {path}")
    participant, vrlog, game = match.groups()
    rows, problems = [], []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, fields in enumerate(csv.reader(stream), 1):
            if not fields or not any(item.strip() for item in fields):
                continue
            if len(fields) != 3:
                problems.append(f"row {line_number}: expected 3 fields, got {len(fields)}")
                continue
            raw_timestamp, raw_score, source = (item.strip() for item in fields)
            try:
                score = float(raw_score)
            except ValueError:
                problems.append(f"row {line_number}: invalid score {raw_score!r}")
                continue
            if not math.isfinite(score):
                problems.append(f"row {line_number}: nonfinite score {raw_score!r}")
                continue
            timestamp = int(raw_timestamp) if PRECISE_TIMESTAMP.fullmatch(raw_timestamp) else None
            rows.append({"raw_row": line_number, "timestamp_us": timestamp,
                         "timestamp_raw": raw_timestamp, "score": score,
                         "source": source})
    manual = [row for row in rows if row["source"] == "manual"]
    manual_scores = [row["score"] for row in manual]
    precise_times = [row["timestamp_us"] for row in rows if row["timestamp_us"] is not None]
    manual_times = [row["timestamp_us"] for row in manual if row["timestamp_us"] is not None]
    near_all, same_all = near_pairs(rows)
    near_manual, same_manual = near_pairs(manual)
    sha = digest(path)
    candidates = native.get(vrlog, [])
    verified = [item for item in candidates if item["voice_sha256"] == sha]
    # Do not resolve an ambiguous match or a voice mismatch by guessing.
    matched = verified[0] if len(verified) == 1 else None
    all_precise = bool(rows) and len(precise_times) == len(rows)
    manual_all_precise = bool(manual) and len(manual_times) == len(manual)
    first = min(precise_times) if all_precise else None
    last = max(precise_times) if all_precise else None
    return {
        "participant_id_from_filename": int(participant),
        "session_id_from_filename": vrlog,
        "game_name_original": game,
        "game_name_normalized": ALIASES.get(game, game),
        "self_report_path": path_label(path, project),
        "self_report_sha256": sha,
        "raw_score_rows": len(rows),
        "raw_manual_rows": len(manual),
        "raw_auto_rows": sum(row["source"] == "auto" for row in rows),
        "unknown_source_rows": sum(row["source"] not in {"auto", "manual"} for row in rows),
        "invalid_rows": len(problems),
        "scores_outside_1_5": sum(not 1 <= row["score"] <= 5 for row in rows),
        "manual_mean_raw": statistics.mean(manual_scores) if manual_scores else "unknown",
        "manual_median_raw": statistics.median(manual_scores) if manual_scores else "unknown",
        "manual_max_raw": max(manual_scores) if manual_scores else "unknown",
        "manual_score_distribution_raw": json.dumps(dict(sorted(Counter(manual_scores).items())), sort_keys=True),
        "precise_timestamp_rows": len(precise_times),
        "imprecise_timestamp_rows": len(rows) - len(precise_times),
        "all_timestamps_precise": all_precise,
        "manual_timestamps_precise": manual_all_precise,
        "first_score_timestamp_us": first if first is not None else "unknown",
        "last_score_timestamp_us": last if last is not None else "unknown",
        "first_score_utc": timestamp_iso(first),
        "last_score_utc": timestamp_iso(last),
        "observed_score_span_sec": (last - first) / 1_000_000 if first is not None else "unknown",
        "all_adjacent_pairs_le_2sec": near_all,
        "all_same_score_adjacent_pairs_le_2sec": same_all,
        "manual_adjacent_pairs_le_2sec": near_manual,
        "manual_same_score_adjacent_pairs_le_2sec": same_manual,
        "manual_exact_duplicate_raw_rows": len(manual) - len({(row["timestamp_raw"], row["score"], row["source"]) for row in manual}),
        "local_native_candidate_dirs": json.dumps([item["directory"] for item in candidates]),
        "local_native_dir": matched["directory"] if matched else "MISSING",
        "local_rgb_count": matched["rgb_count"] if matched else 0,
        "local_match_evidence": "exact VRLOG token and byte-identical voice.csv" if matched else "no verified local recording match",
        "parse_notes": "; ".join(problems) or "none",
    }


def write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def build_pairs(sessions: list[dict]) -> list[dict]:
    by_participant = defaultdict(list)
    for session in sessions:
        by_participant[session["participant_id_from_filename"]].append(session)
    pairs = []
    side_fields = [
        "session_id_from_filename", "game_name_normalized", "self_report_path",
        "self_report_sha256", "raw_score_rows", "raw_manual_rows", "raw_auto_rows",
        "manual_mean_raw", "manual_median_raw", "manual_max_raw",
        "manual_adjacent_pairs_le_2sec", "manual_same_score_adjacent_pairs_le_2sec",
        "manual_exact_duplicate_raw_rows", "all_timestamps_precise",
        "first_score_utc", "last_score_utc", "observed_score_span_sec",
        "local_native_dir", "local_rgb_count", "local_match_evidence",
    ]
    for participant, items in sorted(by_participant.items()):
        if len(items) != 2:
            raise ValueError(f"Participant {participant} has {len(items)} self-report files, expected 2 for a paired inventory")
        a, b = sorted(items, key=lambda item: (item["game_name_normalized"], item["session_id_from_filename"]))
        if not a["raw_manual_rows"] or not b["raw_manual_rows"]:
            raise ValueError(f"Participant {participant} lacks manual-source rows in one session")
        mean_delta = b["manual_mean_raw"] - a["manual_mean_raw"]
        median_delta = b["manual_median_raw"] - a["manual_median_raw"]
        row = {"participant_id_from_filename": participant,
               "pair_game_key": a["game_name_normalized"] + " | " + b["game_name_normalized"]}
        for side, item in (("a", a), ("b", b)):
            row.update({f"{side}_{field}": item[field] for field in side_fields})
        first_a, first_b = a["first_score_timestamp_us"], b["first_score_timestamp_us"]
        precise = isinstance(first_a, int) and isinstance(first_b, int)
        first_side = ("a" if first_a < first_b else "b" if first_b < first_a else "same_first_timestamp") if precise else "unknown"
        second_minus_first = mean_delta if first_side == "a" else -mean_delta if first_side == "b" else "unknown"
        row.update({
            "manual_mean_difference_b_minus_a_raw": mean_delta,
            "manual_median_difference_b_minus_a_raw": median_delta,
            "manual_mean_order_raw": "b_higher" if mean_delta > 0 else "a_higher" if mean_delta < 0 else "tie",
            "first_observed_session_from_score_clock": first_side,
            "manual_mean_second_minus_first_observed_session": second_minus_first,
            "first_score_gap_sec": abs(first_b - first_a) / 1_000_000 if precise else "unknown",
            "first_scores_same_utc_date": a["first_score_utc"][:10] == b["first_score_utc"][:10] if precise else "unknown",
            "both_local_rgb_present": a["local_rgb_count"] > 0 and b["local_rgb_count"] > 0,
            "pair_video_training_readiness": "candidate_only_requires_coverage_protocol_audit" if a["local_rgb_count"] > 0 and b["local_rgb_count"] > 0 else "blocked_missing_paired_video",
            "scope": OBSERVATIONAL,
        })
        pairs.append(row)
    return pairs


def readiness_report(sessions: list[dict], pairs: list[dict], output: Path,
                     project: Path, report_root: Path, raw_root: Path) -> dict:
    manual = sum(item["raw_manual_rows"] for item in sessions)
    auto = sum(item["raw_auto_rows"] for item in sessions)
    matched = [item for item in sessions if item["local_rgb_count"] > 0]
    complete_pairs = sum(row["both_local_rgb_present"] for row in pairs)
    groups = defaultdict(list)
    for pair in pairs:
        groups[pair["pair_game_key"]].append(pair)
    summary = {
        "participants_from_filename": len(pairs), "self_report_sessions": len(sessions),
        "raw_manual_rows": manual, "raw_auto_rows": auto,
        "total_score_rows": manual + auto,
        "manual_adjacent_pairs_le_2sec": sum(item["manual_adjacent_pairs_le_2sec"] for item in sessions),
        "manual_same_score_adjacent_pairs_le_2sec": sum(item["manual_same_score_adjacent_pairs_le_2sec"] for item in sessions),
        "manual_exact_duplicate_raw_rows": sum(item["manual_exact_duplicate_raw_rows"] for item in sessions),
        "matched_local_visual_sessions": len(matched),
        "matched_local_rgb_files": sum(item["local_rgb_count"] for item in matched),
        "complete_local_visual_pairs": complete_pairs,
        "imprecise_timestamp_sessions": sum(not item["all_timestamps_precise"] for item in sessions),
        "imprecise_timestamp_rows": sum(item["imprecise_timestamp_rows"] for item in sessions),
        "different_utc_date_pairs": sum(row["first_scores_same_utc_date"] is False for row in pairs),
        "precise_first_score_chronology_pairs": sum(row["first_observed_session_from_score_clock"] in {"a", "b"} for row in pairs),
        "second_observed_session_manual_mean_higher": sum(isinstance(row["manual_mean_second_minus_first_observed_session"], float) and row["manual_mean_second_minus_first_observed_session"] > 0 for row in pairs),
        "second_observed_session_manual_mean_lower": sum(isinstance(row["manual_mean_second_minus_first_observed_session"], float) and row["manual_mean_second_minus_first_observed_session"] < 0 for row in pairs),
        "second_observed_session_manual_mean_tied": sum(isinstance(row["manual_mean_second_minus_first_observed_session"], float) and row["manual_mean_second_minus_first_observed_session"] == 0 for row in pairs),
    }
    lines = [
        "# 原始 VR.net 同参与者配对评分清单与准备状态", "",
        "**探索性，非最终结论。本报告只做本地文件盘点与描述统计；未训练评分或排序模型。**", "",
        "## 可复现输入与计算约定", "",
        f"- 检查时间（UTC）：{datetime.now(timezone.utc).isoformat()}",
        f"- 原始评分目录：`{path_label(report_root, project)}`。",
        f"- 本地录像查找范围：`{path_label(raw_root, project)}` 下所有 `gamedata.log` 所在目录。",
        "- 排除 `__MACOSX`、`._` 资源文件及 `.ipynb_checkpoints`；未修改原始文件。",
        "- 每个评分 CSV 实际为无表头三列：时间戳、数值评分、`auto/manual` 来源字符串。",
        "- 只将 16 位整数字符串当作可精确使用的 Unix 微秒时间戳；截断科学计数法保留评分但不修复时间。",
        "- `manual` 只表示文件中的来源字段；其与采集方式、人工核验质量的对应关系尚未得到代码本证实。",
        "- 参与者 ID 来自原文件名。唯一名称规范化：`VR Monster Awakens` → `Monster Awakens`；原名称仍在 session inventory 中保留。",
        "- 游戏 A/B 按规范化游戏名排序；差值定义为 B 的原始 manual 均值减 A 的原始 manual 均值，未按结果调整方向。",
        "- 近重复标志：同来源 manual 的相邻精确时间记录间隔 ≤ 2 秒，另报同分值数量；未删除记录。",
        "- 精确重复：时间原文、分值、来源三列完全相同。均值/中位数包含所有原始 manual 行。",
        "- 本地录像匹配同时要求精确 VRLOG token 和 `voice.csv` 与评分文件 SHA256 完全相同。仅同名游戏不建立跨版本录像匹配。", "",
        "## 实际数量", "",
        "| 项目 | 数量 |", "|---|---:|",
    ]
    labels = {
        "participants_from_filename": "参与者文件名 ID / 配对数",
        "self_report_sessions": "实际评分 session 文件",
        "raw_manual_rows": "原始 manual 记录",
        "raw_auto_rows": "原始 auto 记录",
        "total_score_rows": "原始评分记录合计",
        "manual_adjacent_pairs_le_2sec": "manual 相邻 ≤ 2 秒对数",
        "manual_same_score_adjacent_pairs_le_2sec": "其中同分值的近重复对数",
        "manual_exact_duplicate_raw_rows": "manual 精确重复行数",
        "matched_local_visual_sessions": "匹配且有 RGB 的本地 session",
        "matched_local_rgb_files": "匹配本地 RGB JPG 总数",
        "complete_local_visual_pairs": "双方均有本地 RGB 的参与者配对",
        "imprecise_timestamp_sessions": "含不精确时间戳的 session",
        "imprecise_timestamp_rows": "不精确时间戳行数",
        "different_utc_date_pairs": "两次评分开始位于不同 UTC 日期的配对",
        "precise_first_score_chronology_pairs": "双方首评分先后可判定的配对",
        "second_observed_session_manual_mean_higher": "其中后测 session manual 均值较高",
        "second_observed_session_manual_mean_lower": "其中后测 session manual 均值较低",
        "second_observed_session_manual_mean_tied": "其中后测 session manual 均值相同",
    }
    for key, value in summary.items():
        lines.append(f"| {labels[key]} | {value} |")
    lines += ["", "## 游戏配对描述（均值差 B − A）", "",
              "| 游戏 A | 游戏 B | 人数 | A 较高 | B 较高 | 相同 | 人均差值的均值 | 人均差值的中位数 | A首评分较早 / B较早 / 未知 | 双方本地视频配对 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---|---:|"]
    for key, members in sorted(groups.items()):
        a, b = key.split(" | ")
        delta = [row["manual_mean_difference_b_minus_a_raw"] for row in members]
        order_counts = Counter(row["first_observed_session_from_score_clock"] for row in members)
        lines.append(f"| {a} | {b} | {len(members)} | {sum(v < 0 for v in delta)} | {sum(v > 0 for v in delta)} | {sum(v == 0 for v in delta)} | {statistics.mean(delta):.6f} | {statistics.median(delta):.6f} | {order_counts['a']} / {order_counts['b']} / {order_counts['unknown']} | {sum(row['both_local_rgb_present'] for row in members)} |")
    lines += ["", "这些差值是不同游戏、路径、时长和体验顺序共同作用下的观察。没有拟合模型或作因果检验，不能解释为修改某一参数会改善多少不适。", "",
              "## 每个 session 的原始记录与本地视频", "",
              "| ID | Session | 规范化游戏名 | 全部行 | manual | manual 均值 | manual 中位数 | manual 同分近重复对 | 精确时间戳 | RGB JPG |",
              "|---:|---|---|---:|---:|---:|---:|---:|---|---:|"]
    for item in sessions:
        lines.append(f"| {item['participant_id_from_filename']} | {item['session_id_from_filename']} | {item['game_name_normalized']} | {item['raw_score_rows']} | {item['raw_manual_rows']} | {item['manual_mean_raw']:.6f} | {item['manual_median_raw']:.6f} | {item['manual_same_score_adjacent_pairs_le_2sec']} | {'yes' if item['all_timestamps_precise'] else 'no'} | {item['local_rgb_count']} |")
    lines += ["", "## 时间顺序与来源限制", "",
              "- `first_observed_session_from_score_clock` 只比较双方第一条评分的绝对时间，不能证明进入游戏的顺序、洗脱期、随机化或顺序平衡。",
              "- 文件名中的 VRLOG token 不应单独当作实验日期证明；以精确评分时钟为描述依据。",
              "- 评分跨度只是首末评分间隔，不是完整视频时长或完整暴露时长；原始录像覆盖需要逐 session 核对。",
              "- **重要顺序混杂：** 在全部可判定时钟的参与者中，每种游戏配对的首评分顺序固定：Beat Saber→Monster Awakens、Traffic Cop→VR Rome、Cartoon→Voxel、Job Simulator→Epic Roller Coasters、Mini Motor→Pottery。尚未见到同一配对内相反顺序的例子。不能假定这些配对消除了体验累积或顺序效应。",
              f"- 首评分顺序可判定的 {summary['precise_first_score_chronology_pairs']} 对中，后测游戏均值较高 {summary['second_observed_session_manual_mean_higher']} 对、较低 {summary['second_observed_session_manual_mean_lower']} 对、相同 {summary['second_observed_session_manual_mean_tied']} 对。这是描述统计，不能确定是场景本身还是顺序造成。",
              "- P3 `VRLOG-5xxxxxxx VR Monster Awakens.csv` 有 32 条截断科学计数法时间戳，5 条来源为 manual。其平均分可描述，但无法在秒级窗口中对齐到录像。",
              "- 1 条原文完全重复的 manual 行来自 P3 时间已截断的 CSV；它可能由精度丢失形成，不能据此证明两次报告相同或自动删除。",
              "- 评分源 codebook 未核实，不能把 manual 直接称为人审核过的金标准，不能把 auto 当作额外独立被试。",
              "- 如比较未来设计方案，训练/测试必须按参与者整组切分，且区分新参与者与新游戏泛化；重复截图、相邻片段和同一参与者不能散到两侧。", "",
              "### 时间跨度需要单独确认的配对", ""]
    unusual = [row for row in pairs if row["first_scores_same_utc_date"] is False or row["first_scores_same_utc_date"] == "unknown"]
    for row in unusual:
        lines.append(f"- P{row['participant_id_from_filename']}：A {row['a_first_score_utc']}；B {row['b_first_score_utc']}；首评分间距 {row['first_score_gap_sec']} 秒。实验顺序/跨日原因待查。")
    lines += ["", "## 准备状态与下一步", "",
              f"- 当前完整本地视频配对 **{complete_pairs}** 组；只有原始 Epic Roller Coasters 的 5 个 session 有精确匹配本地原生录像。",
              "- **配对评分模型训练门控未通过：缺少每对另一方的视频。** 本脚本没有训练模型、计算 AUC、下载数据、改阈值或修改原始评分。",
              "- 可以先使用这 5 个过山车录像验证矩阵运动监督与清理流程；该任务不会消耗这些配对评分来虚构风险预测能力。",
              "- 配对路线后续需要获得同一原始版本的另一方录像、核查参与者和体验顺序、匹配暴露长度及末端/累计评分定义，再冻结排序协议。固定顺序情况下，排序只能描述该实验条件下的观察差，不能直接作为去除个体差异后的场景固有风险标签。",
              "- 即使获得全部 26 组，这仍只有 26 个参与者级配对；不能通过组合所有片段或镜像配对声称增加独立样本量。", "",
              "## 来源", "",
              "- 原始评分文件和本地录像：逐文件路径及 SHA256 见 `results/coaster_paired_session_inventory.csv` 和 `results/coaster_paired_score_manifest.csv`。",
              "- 原始 SelfReportData 公共档案：[Google Drive](https://drive.google.com/file/d/1VrnacQP-c7iS_pNy7e45QdU3MjNRDtUq/view)；本次使用既有本地解压文件，没有发起下载。",
              "- 发布论文：[VR.net v1](https://arxiv.org/html/2306.03381v1)；来源代码本/实验顺序的缺失不能由论文的一般描述替代。",
              "- 本次可复现命令：`.venv\\Scripts\\python.exe -X utf8 scripts\\build_coaster_pair_manifest.py`。", ""]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--self-reports", type=Path, default=Path("data/raw/vrnet/rollercoaster_raw/self_reports/SelfReportData"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("results/coaster_paired_score_manifest.csv"))
    parser.add_argument("--session-output", type=Path, default=Path("results/coaster_paired_session_inventory.csv"))
    parser.add_argument("--report", type=Path, default=Path("results/coaster_paired_score_readiness.md"))
    args = parser.parse_args()
    project = args.project_root.resolve()

    def absolute(value: Path) -> Path:
        return value.resolve() if value.is_absolute() else project / value

    report_root, raw_root = absolute(args.self_reports), absolute(args.raw_root)
    files = [path for path in sorted(report_root.rglob("*.csv"))
             if not any(part in {"__MACOSX", ".ipynb_checkpoints"} for part in path.parts)
             and not path.name.startswith("._")]
    if not files:
        raise FileNotFoundError(f"No genuine self-report CSV files under {report_root}")
    native = native_inventory(raw_root, project)
    sessions = sorted((inspect_session(path, native, project) for path in files),
                      key=lambda item: (item["participant_id_from_filename"], item["game_name_normalized"]))
    pairs = build_pairs(sessions)
    write_csv(absolute(args.output), pairs)
    write_csv(absolute(args.session_output), sessions)
    summary = readiness_report(sessions, pairs, absolute(args.report), project, report_root, raw_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
