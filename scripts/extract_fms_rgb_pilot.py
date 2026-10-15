#!/usr/bin/env python3
"""Local-only, historical FMS-event RGB features for an exploratory pilot.

Reads selected JPG members directly from existing ZIPs. No downloads, full
extraction, score filling, future frames, matrix parsing, or video re-encoding.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import logging
import math
from pathlib import Path, PurePosixPath
import platform
import random
import sys
import time
import traceback
import zipfile

import cv2
import numpy as np
import pandas as pd

from extract_lightgbm_features import frame_metrics, resize_frame

GAMES = ["Army_Men", "City_Parkour", "Creature_Feature", "Earth_Gym", "Kowloon", "Mars_Miners"]
WINDOWS = [3, 30]
STATIC = ["edge_density", "color_entropy_bits", "laplacian_variance", "brightness_mean_0_255", "brightness_std_0_255"]
MOTION = ["flow_rate_mean_diag_s", "flow_rate_p95_diag_s", "flow_rate_peripheral_diag_s", "flow_rate_center_diag_s", "flow_direction_concentration", "flow_direction_entropy_bits"]
FEATURES = ["rgb_" + key + "_" + stat for key in STATIC + MOTION for stat in ("mean", "max", "std")]


def entropy(values: np.ndarray) -> float:
    counts, _ = np.histogram(values, bins=16, range=(-math.pi, math.pi))
    total = counts.sum()
    if not total:
        return math.nan
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def summarize(values: list[float]) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return (float(x.mean()), float(x.max()), float(x.std(ddof=0))) if len(x) else (math.nan, math.nan, math.nan)


def load_rows(source) -> pd.DataFrame:
    # The omitted matrix cells can be enormous. The C parser streams the CSV
    # while retaining only these four columns, using bounded-size chunks.
    chunks = []
    for chunk in pd.read_csv(source, usecols=["frame", "timestamp", "video", "MS_rating"],
                             dtype="string", chunksize=10000, encoding="utf-8-sig",
                             low_memory=False, on_bad_lines="error"):
        chunks.append(chunk)
    if not chunks:
        raise ValueError("Empty CSV")
    return pd.concat(chunks, ignore_index=True)


def resolve_member(session: str, value: str, members: set[str]) -> str | None:
    ref = str(value).strip().replace("\\", "/").lstrip("./")
    if not ref:
        return None
    if not PurePosixPath(ref).suffix:
        ref += ".jpg"
    candidates = [ref, session + "/" + ref]
    return next((x for x in candidates if x in members), None)


class FrameCache:
    def __init__(self, archive: zipfile.ZipFile):
        self.archive = archive
        self.items = OrderedDict()

    def read(self, member: str):
        if member in self.items:
            self.items.move_to_end(member)
            return self.items[member]
        payload = self.archive.read(member)
        frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("JPG decode failed: " + member)
        gray, metrics = frame_metrics(resize_frame(frame, 224))
        item = (gray, metrics, hashlib.sha256(payload).hexdigest())
        self.items[member] = item
        if len(self.items) > 512:
            self.items.popitem(last=False)
        return item


def window_features(frame_rows: pd.DataFrame, end: float, window: int, cache: FrameCache) -> dict:
    start = end - window
    empty = {x: math.nan for x in FEATURES}
    result = dict(window_sec=window, start_sec=start, end_sec=end, status="skipped", reason="",
                  source_frame_count=0, sampled_frame_count=0, flow_pair_count=0,
                  frame_members="[]", frame_times_sec="[]", frame_sha256="[]",
                  start_gap_sec=math.nan, end_gap_sec=math.nan, max_gap_sec=math.nan,
                  missing_reference_count=0, **empty)
    if start < 0:
        result["reason"] = "window_before_recording_start"
        return result
    rows = frame_rows.loc[(frame_rows["time_sec"] >= start) & (frame_rows["time_sec"] < end)].copy()
    result["source_frame_count"] = len(rows)
    if len(rows) < 4:
        result["reason"] = "fewer_than_four_source_frames"
        return result
    missing = rows["member"].isna().sum()
    result["missing_reference_count"] = int(missing)
    if missing:
        result["reason"] = "missing_jpg_reference_within_window"
        return result
    # Actual time-based sampling at 2 Hz; duplicate target matches do not
    # create extra frames or independent samples.
    times = rows["time_sec"].to_numpy(dtype=float)
    targets = np.arange(start, end, 0.5)
    indices = []
    for target in targets:
        right = int(np.searchsorted(times, target))
        options = [i for i in (right - 1, right) if 0 <= i < len(times)]
        indices.append(min(options, key=lambda i: abs(times[i] - target)))
    selected = rows.iloc[sorted(set(indices))].drop_duplicates("member", keep="first")
    stamps = selected["time_sec"].to_numpy(dtype=float)
    names = selected["member"].tolist()
    result.update(sampled_frame_count=len(selected), frame_members=json.dumps(names),
                  frame_times_sec=json.dumps(stamps.tolist()),
                  start_gap_sec=float(stamps[0] - start), end_gap_sec=float(end - stamps[-1]),
                  max_gap_sec=float(np.diff(stamps).max()) if len(stamps) > 1 else math.nan)
    if len(selected) < 4:
        result["reason"] = "fewer_than_four_distinct_sampled_frames"
        return result
    if result["start_gap_sec"] > 1.0 or result["end_gap_sec"] > 1.0 or result["max_gap_sec"] > 1.5:
        result["reason"] = "insufficient_temporal_coverage"
        return result
    values = {key: [] for key in STATIC + MOTION}
    hashes = []
    previous = None
    previous_stamp = None
    for name, stamp in zip(names, stamps):
        gray, static, digest = cache.read(name)
        hashes.append(digest)
        for key in STATIC:
            values[key].append(static[key])
        if previous is not None:
            dt = float(stamp - previous_stamp)
            if gray.shape != previous.shape:
                result["reason"] = "image_dimensions_changed_within_window"
                return result
            if 0 < dt <= 1.0:
                flow = cv2.calcOpticalFlowFarneback(previous, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
                rate = magnitude.astype(np.float64) / (dt * math.hypot(*gray.shape))
                h, w = gray.shape
                center = np.zeros(gray.shape, dtype=bool)
                center[h // 4:3 * h // 4, w // 4:3 * w // 4] = True
                values["flow_rate_mean_diag_s"].append(float(rate.mean()))
                values["flow_rate_p95_diag_s"].append(float(np.percentile(rate, 95)))
                values["flow_rate_peripheral_diag_s"].append(float(rate[~center].mean()))
                values["flow_rate_center_diag_s"].append(float(rate[center].mean()))
                valid = magnitude > 1e-3
                theta = angle[valid].astype(float)
                weight = magnitude[valid].astype(float)
                if len(theta) and weight.sum() > 0:
                    concentration = np.hypot(np.sum(weight * np.cos(theta)), np.sum(weight * np.sin(theta))) / weight.sum()
                    values["flow_direction_concentration"].append(float(concentration))
                    values["flow_direction_entropy_bits"].append(entropy((theta + math.pi) % (2 * math.pi) - math.pi))
                result["flow_pair_count"] += 1
        previous, previous_stamp = gray, stamp
    if result["flow_pair_count"] < 2:
        result["reason"] = "fewer_than_two_valid_motion_pairs"
        return result
    for key, data in values.items():
        for stat, value in zip(("mean", "max", "std"), summarize(data)):
            result["rgb_" + key + "_" + stat] = value
    result.update(status="ok", reason="", frame_sha256=json.dumps(hashes))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--sessions-per-game", type=int, default=5)
    args = parser.parse_args()
    project = args.project_dir.resolve()
    if not (project / "PROTOCOL.md").is_file():
        raise FileNotFoundError("PROTOCOL.md missing: experiments prohibited")
    for directory in ("features", "results", "audit"):
        (project / directory).mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(project / "results/fms_rgb_extraction_log.txt", mode="w", encoding="utf-8"), logging.StreamHandler(sys.stdout)])
    cv2.setNumThreads(1)
    cv2.setRNGSeed(42)
    randomizer = random.Random(42)
    raw = project / "data/raw/vrnet"
    local_sources = {}
    for source in sorted(raw.rglob("data_file.csv")):
        local_sources.setdefault(source.parent.name, []).append(source)
    selections = []
    for game in GAMES:
        archive_path = raw / "samples" / (game + ".zip")
        if not archive_path.is_file():
            logging.error("Missing ZIP %s; not downloading", archive_path)
            continue
        with zipfile.ZipFile(archive_path) as archive:
            candidates = sorted(info.filename for info in archive.infolist() if info.filename.endswith("/data_file.csv"))
            chosen = randomizer.sample(candidates, min(args.sessions_per_game, len(candidates)))
            for csv_member in sorted(chosen):
                info = archive.getinfo(csv_member)
                session = PurePosixPath(csv_member).parent.name
                # Prefer an existing byte-size-matching raw CSV; never extract
                # the giant source CSV or trust a different-size cached copy.
                local = next((p for p in local_sources.get(session, []) if p.stat().st_size == info.file_size), None)
                selections.append(dict(game_name=game, session_id=session, zip_path=str(archive_path),
                                       csv_member=csv_member, csv_source=str(local) if local else "zip_member",
                                       csv_uncompressed_bytes=info.file_size, csv_crc=info.CRC,
                                       zip_bytes=archive_path.stat().st_size))
    if not selections:
        raise RuntimeError("No local ZIP session candidates")
    pd.DataFrame(selections).to_csv(project / "results/fms_rgb_sample_selection.csv", index=False, encoding="utf-8-sig")
    plan = dict(created_utc=pd.Timestamp.now(tz="UTC").isoformat(), seed=42,
                selection="random.Random(42), sorted games and CSV members, five sessions per game; no replacements after score/QC inspection",
                selected_sessions=len(selections), windows_sec=WINDOWS, score="original nonempty MS_rating; expected local scale 1-5; no filling/binning",
                input="RGB only; elapsed time separate baseline; no HMD, matrices, score history, future statistics or game identity",
                quality="at least 4 distinct frames and 2 motion pairs; boundary gaps <=1s, sampled max gap <=1.5s; motion dt <=1s; all in-window JPG references present",
                sample_fps=2, resize_max_dimension=224, features=FEATURES,
                models="fixed train-mean/median, time Ridge alpha10, RGB Ridge alpha10, RGB LightGBM depth3/leaves7/150 trees/lr.05/minchild20/lambda5, RGB+time Ridge alpha10",
                evaluation="Leave-One-Game-Out, train-only preprocessing; compare 3s/30s on common valid events; pooled and macro-game MAE/RMSE/Spearman",
                protocol_deviation="User-authorized exploratory raw-FMS regression with game holdout, not original frozen low/medium/high AUC or participant LOSO. No subject mapping; no original gate pass claim.",
                environment=dict(python=platform.python_version(), opencv=cv2.__version__, pandas=pd.__version__, numpy=np.__version__),
                selected=selections)
    # Written before any raw scores or validation metrics are read.
    for path in (project / "audit/fms_rgb_pilot_definition.json", project / "results/fms_rgb_pilot_definition.json"):
        path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    all_events, all_features, session_audit = [], [], []
    started = time.perf_counter()
    for position, selection in enumerate(selections, 1):
        session = selection["session_id"]
        before = time.perf_counter()
        audit = dict(selection, status="failed", error="", raw_rows=0, nonempty_numeric_scores=0, duplicate_reports_removed=0)
        logging.info("[%d/%d] reading %s source=%s", position, len(selections), session, selection["csv_source"])
        try:
            with zipfile.ZipFile(selection["zip_path"]) as archive:
                if selection["csv_source"] == "zip_member":
                    with archive.open(selection["csv_member"]) as stream:
                        rows = load_rows(stream)
                else:
                    rows = load_rows(selection["csv_source"])
                rows["raw_row_index"] = np.arange(len(rows))
                stamps = pd.to_numeric(rows["timestamp"], errors="coerce")
                if stamps.isna().any() or not len(stamps):
                    raise ValueError("Missing/non-numeric timestamp; session quarantined")
                milliseconds = stamps.to_numpy(dtype=np.float64)
                if np.any(np.diff(milliseconds) < 0):
                    raise ValueError("Timestamp moves backwards; session quarantined rather than silently sorted")
                origin = float(milliseconds[0])
                if not 1e12 <= origin < 1e14:
                    raise ValueError("Timestamp epoch-ms assumption unsupported: " + repr(origin))
                rows["time_sec"] = (milliseconds - origin) / 1000.0
                scores = pd.to_numeric(rows["MS_rating"], errors="coerce")
                reports = rows.loc[scores.notna()].copy()
                reports["fms_raw"] = scores.loc[scores.notna()].astype(float)
                audit["raw_rows"] = len(rows)
                audit["nonempty_numeric_scores"] = len(reports)
                before_dedup = len(reports)
                reports = reports.drop_duplicates(["timestamp", "fms_raw"], keep="first")
                audit["duplicate_reports_removed"] = before_dedup - len(reports)
                conflict_stamps = set(reports.groupby("timestamp")["fms_raw"].nunique().loc[lambda x: x > 1].index)
                members = set(archive.namelist())
                frames = rows.loc[rows["video"].notna() & rows["video"].str.strip().ne("")].copy()
                frames["member"] = [resolve_member(session, value, members) for value in frames["video"]]
                frames = frames.drop_duplicates(["timestamp", "video"], keep="first")
                if frames["time_sec"].duplicated().any():
                    raise ValueError("Conflicting video rows at same timestamp; session quarantined")
                cache = FrameCache(archive)
                for _, report in reports.iterrows():
                    event = dict(event_id=session + "__row" + str(int(report["raw_row_index"])),
                                 session_id=session, game_name=selection["game_name"],
                                 raw_row_index=int(report["raw_row_index"]), event_frame=str(report["frame"]),
                                 event_timestamp_ms=float(report["timestamp"]), origin_timestamp_ms=origin,
                                 event_time_sec=float(report["time_sec"]), elapsed_sec=float(report["time_sec"]),
                                 fms_raw=float(report["fms_raw"]), label_status="ok", zip_path=selection["zip_path"],
                                 csv_member=selection["csv_member"], csv_source=selection["csv_source"])
                    if report["timestamp"] in conflict_stamps:
                        event["label_status"] = "conflicting_scores_same_timestamp"
                    elif not math.isfinite(event["fms_raw"]) or not 1 <= event["fms_raw"] <= 5:
                        event["label_status"] = "outside_expected_local_1_5_scale"
                    all_events.append(event)
                    for window in WINDOWS:
                        if event["label_status"] != "ok":
                            feature = dict(window_sec=window, status="skipped", reason=event["label_status"], **{x: math.nan for x in FEATURES})
                        else:
                            try:
                                feature = window_features(frames, event["event_time_sec"], window, cache)
                            except Exception:
                                error = traceback.format_exc()
                                logging.error("%s %ss event %s\n%s", session, window, event["event_id"], error)
                                feature = dict(window_sec=window, status="skipped", reason=error, **{x: math.nan for x in FEATURES})
                        all_features.append(dict(event, **feature))
                audit.update(status="ok", recording_duration_sec=float(rows["time_sec"].iloc[-1]),
                             reports_after_dedup=len(reports), missing_jpg_references=int(frames["member"].isna().sum()),
                             frame_references=len(frames))
        except Exception:
            audit["error"] = traceback.format_exc()
            logging.error("Session %s failed\n%s", session, audit["error"])
        audit["wall_seconds"] = time.perf_counter() - before
        session_audit.append(audit)
        # Checkpoint after every session so progress and failures are durable.
        pd.DataFrame(all_events).to_csv(project / "features/fms_rgb_events.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(all_features).to_csv(project / "features/fms_rgb_event_features.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(session_audit).to_csv(project / "results/fms_rgb_session_audit.csv", index=False, encoding="utf-8-sig")
        logging.info("%s finished %.1fs; cumulative events=%d feature rows=%d", session, audit["wall_seconds"], len(all_events), len(all_features))
    if not all_features:
        raise RuntimeError("No features generated; see full failure log")
    features = pd.DataFrame(all_features)
    coverage = features.groupby(["game_name", "window_sec", "status"]).size().reset_index(name="event_count")
    coverage.to_csv(project / "results/fms_rgb_coverage.csv", index=False, encoding="utf-8-sig")
    feature_stats = features.loc[features["status"] == "ok", FEATURES].agg(["count", "mean", "std", "min", "max"]).T
    feature_stats["missing_rate"] = features.loc[features["status"] == "ok", FEATURES].isna().mean()
    feature_stats.to_csv(project / "features/fms_rgb_feature_statistics.csv", encoding="utf-8-sig", index_label="feature")
    logging.info("DONE %.1fs total; events=%d, valid-window rows=%d, games=%d", time.perf_counter() - started, len(all_events), int((features.status == "ok").sum()), features.game_name.nunique())


if __name__ == "__main__":
    main()
