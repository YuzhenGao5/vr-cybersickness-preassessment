#!/usr/bin/env python
"""Small-sample VR.net image feature extraction and FMS-window audit.

Only locally retained JPGs are used. Farneback flow is computed exclusively
between retained images whose CSV frame numbers are truly adjacent at the
recorded sampling stride; sparse JPGs are never treated as consecutive video.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import traceback
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr

STATIC = ["edge_density", "color_entropy", "laplacian_variance", "brightness_mean", "brightness_std"]
FLOW = ["flow_mean_magnitude", "flow_max_magnitude", "flow_direction_change_rate", "flow_direction_entropy"]
BASE = FLOW + STATIC
AGGS = ["mean", "max", "std"]


def parse_num(value):
    try:
        return float(value) if value is not None and str(value).strip() else None
    except (TypeError, ValueError):
        return None


def read_session(folder):
    csv_path = folder / "data_file.csv"
    csv.field_size_limit(2**31 - 1)
    rows, events, headers = [], [], []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        lower = {h.strip().lower(): h for h in headers}
        frame_col, time_col, label_col = lower.get("frame"), lower.get("timestamp"), lower.get("ms_rating")
        if not frame_col or not time_col or not label_col:
            raise ValueError(f"Required frame/timestamp/MS_rating column missing; headers={headers}")
        for i, row in enumerate(reader):
            frame_raw, time_raw = parse_num(row.get(frame_col)), parse_num(row.get(time_col))
            frame = int(frame_raw) if frame_raw is not None else None
            # Observed VR.net timestamps are epoch milliseconds.
            timestamp = time_raw / 1000.0 if time_raw is not None and abs(time_raw) > 1e11 else time_raw
            rating = parse_num(row.get(label_col))
            item = {"frame": frame, "time": timestamp, "rating": rating, "row": i}
            rows.append(item)
            if rating is not None and timestamp is not None:
                events.append({"frame": frame, "time": timestamp, "rating": rating, "row": i})
    return rows, events, headers


def image_features(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    h, w = image.shape[:2]
    scale = min(1.0, 640.0 / max(h, w))
    if scale < 1:
        image = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    quant = (rgb // 32).astype(np.int32)
    bins = quant[:, :, 0] * 64 + quant[:, :, 1] * 8 + quant[:, :, 2]
    counts = np.bincount(bins.ravel(), minlength=512).astype(float)
    probs = counts[counts > 0] / counts.sum()
    entropy = float(-(probs * np.log2(probs)).sum())
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return {
        "edge_density": float(np.count_nonzero(edges) / edges.size),
        "color_entropy": entropy,
        "laplacian_variance": float(lap.var()),
        "brightness_mean": float(gray.mean()),
        "brightness_std": float(gray.std()),
    }, gray


def flow_features(previous, current):
    flow = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
    moving = mag > 0.5
    entropy = None
    if moving.any():
        hist, _ = np.histogram(angle[moving], bins=8, range=(0, 2 * math.pi))
        p = hist[hist > 0] / hist.sum()
        entropy = float(-(p * np.log2(p)).sum())
    return {
        "flow_mean_magnitude": float(mag.mean()),
        "flow_max_magnitude": float(mag.max()),
        "flow_direction_change_rate": None,
        "flow_direction_entropy": entropy,
    }, angle, moving


def safe_mean(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def aggregate(values):
    out = {}
    for feature in BASE:
        vals = [x.get(feature) for x in values if x.get(feature) is not None and np.isfinite(x.get(feature))]
        out[f"{feature}_mean"] = float(np.mean(vals)) if vals else None
        out[f"{feature}_max"] = float(np.max(vals)) if vals else None
        # A one-frame window has no empirical within-window standard deviation.
        out[f"{feature}_std"] = float(np.std(vals, ddof=1)) if len(vals) >= 2 else None
    return out


def fmt(value, digits=4):
    return "NA" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def markdown_table(headers, rows):
    return "| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--window-sec", type=float, default=30)
    ap.add_argument("--log", required=True)
    ap.add_argument("--alignment-report", required=True)
    ap.add_argument("--distribution-report", required=True)
    ap.add_argument("--feasibility-report", required=True)
    args = ap.parse_args()
    paths = {key: Path(getattr(args, key.replace("-", "_"))) for key in [
        "output", "stats", "log", "alignment-report", "distribution-report", "feasibility-report"]}
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    log = ["VR.net retained-JPG feature extraction log", f"Command: {' '.join(sys.argv)}",
           f"Python: {sys.version.split()[0]}", f"OpenCV: {cv2.__version__}", f"NumPy: {np.__version__}",
           f"Window: +/-{args.window_sec:g} seconds", "No archive download/extraction, preprocess.py, or git commit performed.", ""]
    session_rows, all_events = [], []
    for folder in sorted(Path(args.samples_dir).iterdir()):
        if not folder.is_dir() or not (folder / "data_file.csv").is_file():
            continue
        game = folder.name.split("_", 2)[-1].replace("_", " ")
        try:
            rows, events, headers = read_session(folder)
            valid_frames = sorted({r["frame"] for r in rows if r["frame"] is not None})
            stride = statistics.median([b - a for a, b in zip(valid_frames, valid_frames[1:]) if b > a]) if len(valid_frames) > 1 else None
            time_values = [r["time"] for r in rows if r["time"] is not None]
            lo, hi = (min(time_values), max(time_values)) if time_values else (None, None)
            by_frame = {r["frame"]: r for r in rows if r["frame"] is not None}
            jpgs = sorted((folder / "video").glob("*.jpg"), key=lambda p: int(p.stem) if p.stem.isdigit() else 10**30)
            images = {}
            for jpg in jpgs:
                if not jpg.stem.isdigit():
                    log.append(f"{folder.name}: skipped nonnumeric JPG filename {jpg.name}")
                    continue
                frame = int(jpg.stem)
                rec = by_frame.get(frame)
                if rec is None or rec["time"] is None:
                    log.append(f"{folder.name}: skipped {jpg.name}; no matching frame/timestamp in CSV")
                    continue
                computed = image_features(jpg)
                if computed is None:
                    log.append(f"{folder.name}: skipped unreadable JPG {jpg.name}")
                    continue
                images[frame] = {"frame": frame, "time": rec["time"], "path": jpg.name,
                                 "features": computed[0], "gray": computed[1], "flow": {}}
            ordered = sorted(images.values(), key=lambda x: x["frame"])
            valid_pairs = 0
            for left, right in zip(ordered, ordered[1:]):
                gap = right["frame"] - left["frame"]
                dt = right["time"] - left["time"]
                if stride is None or gap != stride or dt <= 0 or dt > 1.0:
                    continue
                flow, angle, moving = flow_features(left["gray"], right["gray"])
                if valid_pairs and previous_angle.shape == angle.shape:
                    changed = (np.abs(np.angle(np.exp(1j * (angle - previous_angle)))) > math.pi / 4)
                    eligible = moving | previous_moving
                    flow["flow_direction_change_rate"] = float(changed[eligible].mean()) if eligible.any() else None
                right["flow"] = flow
                previous_angle, previous_moving = angle, moving
                valid_pairs += 1
            log.append(f"SESSION {folder.name} game={game}: csv_rows={len(rows)} fms_events={len(events)} retained_jpgs={len(jpgs)} matched_images={len(images)} nominal_frame_stride={stride} valid_adjacent_flow_pairs={valid_pairs}")
            if len(ordered) > 1:
                gaps = [b["frame"] - a["frame"] for a, b in zip(ordered, ordered[1:])]
                log.append(f"  retained frame gaps={gaps}; nonadjacent pairs were not passed to Farneback")
            sess_events = sorted(events, key=lambda e: e["time"])
            for ei, event in enumerate(sess_events):
                start, end = event["time"] - args.window_sec, event["time"] + args.window_sec
                selected = [im for im in ordered if start <= im["time"] <= end]
                overlap_with = [j for j, other in enumerate(sess_events) if j != ei and max(start, other["time"] - args.window_sec) < min(end, other["time"] + args.window_sec)]
                feature_items = []
                for im in selected:
                    x = dict(im["features"])
                    x.update(im["flow"])
                    feature_items.append(x)
                row = {"session": folder.name, "game": game, "event_index": ei + 1,
                       "fms": event["rating"], "event_frame": event["frame"], "event_time_sec": event["time"],
                       "window_start_sec": start, "window_end_sec": end,
                       "window_csv_rows": sum(1 for r in rows if r["time"] is not None and start <= r["time"] <= end),
                       "window_jpg_frames": len(selected), "adjacent_flow_pairs": sum(1 for im in selected if im["flow"]),
                       "boundary_truncated": bool(lo is not None and (start < lo or end > hi)),
                       "overlapping_windows": len(overlap_with),
                       "status": "skipped_no_jpg" if not selected else ("partial_static_only" if valid_pairs == 0 else "partial_or_complete")}
                row.update(aggregate(feature_items))
                session_rows.append(row)
                all_events.append(row)
                if not selected:
                    log.append(f"  EVENT {ei+1}: SKIPPED; no retained JPG timestamp falls within +/-{args.window_sec:g}s of the FMS event")
                else:
                    names = [im["path"] for im in selected]
                    log.append(f"  EVENT {ei+1}: partial static features; jpgs={names}; valid adjacent flow pairs in window={row['adjacent_flow_pairs']}")
            matched_events = sum(1 for event in sess_events if any(event["time"]-args.window_sec <= im["time"] <= event["time"]+args.window_sec for im in ordered))
            log.append(f"  event windows={len(sess_events)}; with >=1 matched JPG={matched_events}")
        except Exception:
            log.append(f"SESSION {folder.name}: FAILED\n{traceback.format_exc()}")

    if not session_rows:
        log.append("FATAL: no analyzable session directories found")
        paths["log"].write_text("\n".join(log) + "\n", encoding="utf-8")
        raise SystemExit("No analyzable sessions")

    meta = ["session", "game", "event_index", "fms", "event_frame", "event_time_sec", "window_start_sec", "window_end_sec", "window_csv_rows", "window_jpg_frames", "adjacent_flow_pairs", "boundary_truncated", "overlapping_windows", "status"]
    feature_cols = [f"{f}_{a}" for f in BASE for a in AGGS]
    with paths["output"].open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=meta + feature_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(session_rows)

    # Descriptive statistics at event-window level, both pooled and per game.
    stat_rows = []
    for level, group_name, group in [("overall", "all", session_rows)] + [("game", g, [r for r in session_rows if r["game"] == g]) for g in sorted({r["game"] for r in session_rows})]:
        for col in feature_cols:
            vals = [r.get(col) for r in group if r.get(col) is not None and np.isfinite(r.get(col))]
            n = len(group)
            stat_rows.append({"level": level, "game": group_name if level == "game" else "", "feature_aggregate": col,
                              "n_windows": n, "n_valid": len(vals), "mean": safe_mean(vals),
                              "std": float(np.std(vals, ddof=1)) if len(vals) >= 2 else None,
                              "n_missing": n - len(vals), "missing_rate": (n-len(vals))/n if n else None})
    with paths["stats"].open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stat_rows[0]))
        writer.writeheader(); writer.writerows(stat_rows)

    # Per-event alignment audit.
    align = ["# VR.net JPG 特征与 FMS 对齐检查", "",
             f"窗口：事件时间 ±{args.window_sec:g} 秒。仅使用本地保留的 JPG；光流只在相邻保留帧号差等于 CSV 中位采样步长、且时间差 ≤1 秒时计算。", "",
             "`window_jpg_frames` 是窗口内可匹配本地 JPG 数；`window_csv_rows` 是原始 CSV 时间行数，FMS 仍是稀疏事件，不向每帧扩展标签。", ""]
    for game in sorted({r["game"] for r in session_rows}):
        group = [r for r in session_rows if r["game"] == game]
        align += [f"## {group[0]['session']} — {game}", "",
                  markdown_table(["事件", "FMS", "CSV行数", "JPG数", "光流帧对", "重叠窗口数", "边界截断", "状态"],
                                 [[r["event_index"], fmt(r["fms"], 2), r["window_csv_rows"], r["window_jpg_frames"], r["adjacent_flow_pairs"], r["overlapping_windows"], "是" if r["boundary_truncated"] else "否", r["status"]] for r in group]), ""]
    total = len(session_rows); successful = sum(r["window_jpg_frames"] > 0 for r in session_rows); skipped = total - successful
    align += ["## 汇总", "", f"- session 数：{len({r['session'] for r in session_rows})}", f"- FMS 事件窗口总数：{total}",
              f"- 至少有 1 张匹配 JPG、可计算静态特征的窗口：{successful}", f"- 无 JPG、跳过特征的窗口：{skipped}",
              f"- 窗口内光流帧对总数：{sum(r['adjacent_flow_pairs'] for r in session_rows)}",
              f"- 存在窗口重叠的相邻事件对：{sum(1 for g in {r['session'] for r in session_rows} for a,b in zip([x for x in session_rows if x['session']==g],[x for x in session_rows if x['session']==g][1:]) if b['event_time_sec']-a['event_time_sec'] < 2*args.window_sec)}",
              "", "说明：本地每个 session 仅保留约 10 张时间稀疏 JPG；因此即使静态图像特征可提取，也不代表覆盖完整 ±30 秒视频窗口。所有事件窗口状态均按实际匹配图像填写。", ""]
    paths["alignment-report"].write_text("\n".join(align), encoding="utf-8")

    # Feature distribution report.
    overall = {r["feature_aggregate"]: r for r in stat_rows if r["level"] == "overall"}
    games = sorted({r["game"] for r in session_rows})
    game_stat = {(r["game"], r["feature_aggregate"]): r for r in stat_rows if r["level"] == "game"}
    dist = ["# VR.net 特征分布检查", "", f"分析了 {len(session_rows)} 个稀疏 FMS 事件窗口、{len(games)} 个游戏/session。单位是窗口聚合值；每个游戏仅一个 session，差异仅作描述。", "",
            "## 汇总统计", "", markdown_table(["窗口特征", "均值", "标准差", "缺失数", "缺失率"],
            [[c, fmt(overall[c]["mean"]), fmt(overall[c]["std"]), overall[c]["n_missing"], fmt(overall[c]["missing_rate"])] for c in feature_cols]), "", "## 各游戏的窗口均值", ""]
    base_rows = []
    for feat in BASE:
        cols = [f"{feat}_mean", f"{feat}_max", f"{feat}_std"]
        for aggcol in cols:
            vals = [game_stat.get((g, aggcol), {}).get("mean") for g in games]
            present = [v for v in vals if v is not None and np.isfinite(v)]
            spread = max(present)-min(present) if len(present) >= 2 else None
            base_rows.append([aggcol, *[fmt(game_stat.get((g, aggcol), {}).get("mean")) for g in games], fmt(spread)])
    dist.append(markdown_table(["窗口特征", *games, "游戏均值极差"], base_rows))
    missing50 = [c for c in feature_cols if overall[c]["missing_rate"] is not None and overall[c]["missing_rate"] > 0.5]
    spreads = []
    for feat in BASE:
        vals = [game_stat.get((g, f"{feat}_mean"), {}).get("mean") for g in games]
        vals = [x for x in vals if x is not None and np.isfinite(x)]
        if len(vals) >= 2: spreads.append((max(vals)-min(vals), feat))
    high = sorted(spreads, reverse=True)[:3]; low = sorted(spreads)[:3]
    dist += ["", "## 解释", "",
             "- 特征均值差异仅是 6 个单 session 的描述对照；游戏、session、参与者（无独立参与者 ID 可用于这里的分层）完全混杂，不能归因于游戏本身。",
             f"- 按跨游戏均值极差最大的描述特征：{', '.join(f'{name} ({fmt(spread)})' for spread, name in high) if high else '可比较的特征不足'}。",
             f"- 极差较小的描述特征：{', '.join(f'{name} ({fmt(spread)})' for spread, name in low) if low else '可比较的特征不足'}。此排序不使用显著性阈值。",
             f"- 缺失率 >50% 的窗口特征：{', '.join(missing50) if missing50 else '无'}。需排除或补取帧的候选以此列表为准。",
             "- 光流特征若缺失，是因为稀疏保留 JPG 之间没有满足真实相邻帧条件的帧对；不能用非相邻 JPG 计算或填补。单帧窗口的窗口内标准差按定义留空。",
             "- 完整的总体/分游戏均值、标准差、有效数、缺失数与缺失率见 `features/vrnet_feature_statistics.csv`。", ""]
    paths["distribution-report"].write_text("\n".join(dist), encoding="utf-8")

    # Spearman correlations and session-cluster bootstrap percentile intervals.
    rng = np.random.default_rng(42)
    session_names = sorted({r["session"] for r in session_rows})
    feasibility = ["# VR.net 小规模可行性检查", "", f"FMS 仍按事件记录；关联分析使用事件窗口特征均值与该事件 FMS。有效窗口数 {successful}/{total}。", "",
                   "## Spearman 相关（探索性）", "",
                   "区间由 1,000 次 session-cluster bootstrap（seed=42）得到；p 值为窗口级 Spearman 名义渐近 p 值，未校正多重比较，也未校正窗口重叠。只有 6 个 session，区间和 p 值都不作论文推断。", ""]
    corr_rows = []
    for feat in BASE:
        col = f"{feat}_mean"
        pairs = [(r[col], r["fms"], r["session"]) for r in session_rows if r.get(col) is not None and r["fms"] is not None]
        if len(pairs) < 3 or len({x[0] for x in pairs}) < 2 or len({x[1] for x in pairs}) < 2:
            corr_rows.append([feat, len(pairs), "NA", "NA", "NA", "NA", "有效变异不足" if len(pairs) >= 3 else "有效窗口不足"])
            continue
        xs, ys = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
        result = spearmanr(xs, ys)
        boots = []
        by_session = {s: [p for p in pairs if p[2] == s] for s in session_names}
        for _ in range(1000):
            sampled = rng.choice(session_names, size=len(session_names), replace=True)
            bp = [p for s in sampled for p in by_session[s]]
            bx, by = [p[0] for p in bp], [p[1] for p in bp]
            if len(set(bx)) > 1 and len(set(by)) > 1:
                boots.append(float(spearmanr(bx, by).statistic))
        ci = np.quantile(boots, [0.025, 0.975]) if len(boots) >= 50 else [np.nan, np.nan]
        corr_rows.append([feat, len(pairs), fmt(float(result.statistic)), fmt(float(result.pvalue)), fmt(float(ci[0])), fmt(float(ci[1])), f"bootstrap 有效重复 {len(boots)}/1000"])
    feasibility += [markdown_table(["窗口特征均值", "n", "Spearman ρ", "名义 p", "95% CI 下界", "95% CI 上界", "备注"], corr_rows), "",
                    "## 简单逻辑回归流程演示", ""]
    model_rows = [r for r in session_rows if all(r.get(f"{f}_mean") is not None and np.isfinite(r.get(f"{f}_mean")) for f in STATIC) and r.get("fms") is not None]
    distinct_fms = sorted({r["fms"] for r in model_rows})
    auc_text = "样本量不足，不做模型训练。"
    model_note = ""
    if len(model_rows) > 50 and len(distinct_fms) >= 2:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import LeaveOneGroupOut
        # This is only a pipeline diagnostic. It is NOT a clinical threshold.
        threshold = min(distinct_fms) + 1e-12
        y = np.array([int(r["fms"] >= threshold) for r in model_rows])
        X = np.array([[r[f"{f}_mean"] for f in STATIC] for r in model_rows], dtype=float)
        groups = np.array([r["session"] for r in model_rows])
        fold_aucs = []
        for tr, te in LeaveOneGroupOut().split(X, y, groups):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            clf = make_pipeline(StandardScaler(), LogisticRegression(random_state=42, max_iter=1000))
            clf.fit(X[tr], y[tr])
            from sklearn.metrics import roc_auc_score
            fold_aucs.append(float(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])))
        if fold_aucs:
            auc_text = f"LOSO-session 有效折 AUC 均值 {np.mean(fold_aucs):.4f}（{len(fold_aucs)} 个同时含两类的测试折；各折 AUC：{', '.join(f'{v:.4f}' for v in fold_aucs)}）。"
            model_note = f"用于流程演示的二分类是 FMS ≥ {threshold:g} 对比最低观测值 {min(distinct_fms):g}；该切分只服务于计算 AUC，不是低/高风险临床阈值或项目标签映射。静态特征完整窗口 n={len(model_rows)}；总事件窗口为 {total}。"
        else:
            auc_text = "窗口数 >50，但 LOSO-session 没有同时含两类的测试折，无法得到有效 AUC；不报告虚构值。"
    elif len(model_rows) <= 50:
        auc_text = f"可用于逻辑回归的完整静态特征窗口只有 {len(model_rows)}（≤50），样本量不足，不做模型训练。"
    else:
        auc_text = "FMS 类别不足以形成二分类，不做模型训练。"
    feasibility += [f"- {auc_text}", f"- {model_note}" if model_note else "- 未得到模型 AUC。", "",
                    "## 限制与结论", "",
                    "- 样本量小，当前六个 session 各只有一个游戏样本；事件窗口重叠且同 session 内相关。结果只用于验证解析、图像特征计算和 FMS 时间窗口代码，不作论文结论。",
                    "- 不声称跨游戏预测能力；即使报告 LOSO-session 流程演示 AUC，也不能视为跨游戏泛化证据。该流程不是最终模型，也未触碰 `preprocess.py`。",
                    "- 本地 JPG 很稀疏，静态特征只描述窗口内保留的少量帧；本轮没有可用相邻 JPG 对时，光流相关性为 NA。",
                    "- 建议在获准且可复现的前提下提取事件窗口内连续 JPG 帧，再复跑此脚本；随后增加多个参与者/session 后才评估跨场景和 LOSO。", ""]
    paths["feasibility-report"].write_text("\n".join(feasibility), encoding="utf-8")
    log += ["", f"TOTAL sessions={len(session_names)} event_windows={total} static_feature_windows={successful} skipped_no_jpg={skipped}",
            f"Flow pair count={sum(r['adjacent_flow_pairs'] for r in session_rows)}", f"Feature matrix={paths['output']}",
            f"Statistics={paths['stats']}", "Completed without training final model, running preprocess.py, downloading data, or committing Git."]
    paths["log"].write_text("\n".join(log) + "\n", encoding="utf-8")
    print(f"sessions={len(session_names)} events={total} static_feature_windows={successful} skipped={skipped} flow_pairs={sum(r['adjacent_flow_pairs'] for r in session_rows)}")


if __name__ == "__main__":
    main()
