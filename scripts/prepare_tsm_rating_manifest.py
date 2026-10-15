#!/usr/bin/env python3
"""Build portable, strictly historical video inputs for real VR.net rating events.

No videos are downloaded, no rating is filled, and no missing frame is duplicated.
The fixed event table is the sampling unit; additional image clips are not ratings.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import logging
import math
from pathlib import Path, PurePosixPath
import re
import zipfile

import numpy as np
import pandas as pd

from parse_vrnet_matrices import timestamp_scale

SESSION_RE = re.compile(r"^\d+_\d+_.+$")
REQUIRED_EVENT_COLS = {
    "rating_event_id", "game_name", "participant_id", "session_id", "ms_rating",
    "rating_time_sec", "exposure_elapsed_seconds", "binary_label",
}
BASE_MEASURES = (
    "camera_speed", "camera_acceleration", "camera_rotation_rate", "fov_horizontal",
    "fov_vertical", "object_count", "head_speed", "head_angular_speed",
)
BASELINE_COLUMNS = [
    f"past{window}_{measure}_{stat}"
    for window in (5, 30, 60)
    for measure in BASE_MEASURES
    for stat in ("mean", "max", "std")
] + ["exposure_elapsed_seconds"]
MANIFEST_META_COLUMNS = [
    "event_id", "game_name", "participant_id", "session_id", "binary_true", "fms_raw",
    "rating_time_sec", "exposure_elapsed_seconds", "frame_paths_json", "frame_times_json",
    "frame_ids_json", "window_seconds_json", "source_csv_locator", "source_csv_sha256",
    "timestamp_unit", "session_video_reference_count", "session_jpg_count",
]


def json_dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def integer_frame(value):
    number = finite_number(value)
    return int(number) if number is not None and number >= 0 and number.is_integer() else None


@dataclass(frozen=True)
class CsvSource:
    path: Path
    size_bytes: int
    member: str | None = None

    @contextmanager
    def open(self):
        if self.member is None:
            with self.path.open("rb") as stream:
                yield stream
        else:
            with zipfile.ZipFile(self.path) as archive:
                with archive.open(self.member) as stream:
                    yield stream

    def sha256(self) -> str:
        digest = hashlib.sha256()
        with self.open() as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def relative_locator(self, data_root: Path) -> str:
        path = self.path.relative_to(data_root).as_posix()
        return path if self.member is None else f"{path}::{self.member}"


def owning_session(path: Path, data_root: Path) -> str | None:
    for parent in path.parents:
        if parent == data_root:
            break
        if SESSION_RE.fullmatch(parent.name):
            return parent.name
    return None


def discover(data_root: Path, wanted_sessions: set[str], logger):
    sources = defaultdict(list)
    images = defaultdict(lambda: defaultdict(list))
    excluded_legacy_files = 0
    # ZIP images are intentionally not extracted: only already-extracted JPGs qualify.
    for path in data_root.rglob("*"):
        if not path.is_file():
            continue
        parts = path.relative_to(data_root).parts
        if "rollercoaster_raw" in parts:
            excluded_legacy_files += 1
            continue
        suffix = path.suffix.lower()
        if path.name == "data_file.csv":
            session = owning_session(path, data_root)
            if session in wanted_sessions:
                sources[session].append(CsvSource(path, path.stat().st_size))
        elif suffix in (".jpg", ".jpeg"):
            session = owning_session(path, data_root)
            frame = integer_frame(path.stem)
            if session in wanted_sessions and frame is not None:
                images[session][frame].append(path)
        elif suffix == ".zip":
            try:
                with zipfile.ZipFile(path) as archive:
                    for item in archive.infolist():
                        member = PurePosixPath(item.filename.replace("\\", "/"))
                        if member.name != "data_file.csv":
                            continue
                        session = next((p for p in reversed(member.parts[:-1])
                                        if SESSION_RE.fullmatch(p)), None)
                        if session in wanted_sessions:
                            sources[session].append(CsvSource(path, item.file_size, item.filename))
            except (OSError, zipfile.BadZipFile) as exc:
                logger.warning("ZIP listing failed %s: %s", path, exc)
    return sources, images, excluded_legacy_files


def canonical_source(candidates: list[CsvSource], inventory_record: dict, data_root: Path):
    if not candidates:
        raise ValueError("no modern data_file.csv or existing ZIP member found")
    expected_hash = inventory_record.get("known_sha256")
    expected_size = inventory_record.get("size_bytes")
    if not expected_hash or expected_size is None:
        raise ValueError("canonical source inventory lacks SHA256 or byte size")
    eligible = [source for source in candidates if source.size_bytes == int(expected_size)]
    if not eligible:
        raise ValueError("no discovered CSV matches canonical byte size")
    matches = []
    for source in sorted(eligible, key=lambda x: (x.member is not None, x.relative_locator(data_root))):
        actual_hash = source.sha256()
        if actual_hash == expected_hash:
            matches.append(source)
    if not matches:
        raise ValueError("no discovered CSV matches canonical SHA256; do not infer identity from filename")
    return matches[0], expected_hash, [source.relative_locator(data_root) for source in matches]


def verified_image_index(raw_images: dict[int, list[Path]], data_root: Path):
    resolved = {}
    duplicate_groups = 0
    for frame_id, paths in raw_images.items():
        paths = sorted(paths, key=lambda p: p.relative_to(data_root).as_posix())
        if len(paths) > 1:
            duplicate_groups += 1
            if len({file_sha256(path) for path in paths}) != 1:
                raise ValueError(f"conflicting duplicate JPG frame {frame_id}: "
                                 + ", ".join(p.relative_to(data_root).as_posix() for p in paths))
        resolved[frame_id] = paths[0]
    return resolved, duplicate_groups


def read_frame_clock(source: CsvSource, session: str):
    required = ["frame", "timestamp", "video", "MS_rating"]
    with source.open() as stream:
        header = pd.read_csv(stream, nrows=0, encoding="utf-8-sig").columns.tolist()
    missing = sorted(set(required) - set(header))
    if missing:
        raise ValueError(f"missing clock/video columns: {missing}")
    origin = scale = unit = None
    observations = {}
    reports = {}
    raw_rows = 0
    previous_timestamp = None
    with source.open() as stream:
        chunks = pd.read_csv(stream, usecols=required, dtype=str, chunksize=64,
                             keep_default_na=False, encoding="utf-8-sig")
        for chunk in chunks:
            for row in chunk.to_dict("records"):
                raw_rows += 1
                raw_timestamp = finite_number(row["timestamp"])
                if raw_timestamp is None:
                    if str(row["video"]).strip() or str(row["MS_rating"]).strip():
                        raise ValueError(f"video/report row {raw_rows} has invalid timestamp")
                    continue
                if origin is None:
                    # Exactly the existing parameter preparation convention, not a new clock guess.
                    scale, unit = timestamp_scale(raw_timestamp)
                    origin = raw_timestamp * scale
                timestamp = raw_timestamp * scale - origin
                if timestamp < 0 or (previous_timestamp is not None and timestamp < previous_timestamp):
                    raise ValueError(f"nonmonotonic source clock at record {raw_rows}")
                previous_timestamp = timestamp
                rating = finite_number(row["MS_rating"])
                if rating is not None:
                    reports[f"modern:{session}:record:{raw_rows}"] = (timestamp, rating)
                video = str(row["video"]).strip()
                if not video:
                    continue
                basename = PurePosixPath(video.replace("\\", "/")).name
                frame_id = integer_frame(PurePosixPath(basename).stem)
                source_frame = integer_frame(row["frame"])
                if frame_id is None or source_frame is None or frame_id != source_frame:
                    raise ValueError(f"unsupported or inconsistent video reference at record {raw_rows}: {video!r}")
                if frame_id in observations and abs(observations[frame_id] - timestamp) > 1e-8:
                    raise ValueError(f"JPG frame {frame_id} maps to multiple source timestamps")
                observations[frame_id] = timestamp
    if not observations:
        raise ValueError("CSV has no usable video references")
    return observations, reports, dict(timestamp_unit=unit, timestamp_scale=scale,
                                      timestamp_origin=origin, raw_rows=raw_rows)


def choose_window(frames, rating_time: float, width: int, count: int,
                  min_span_fraction: float, max_frame_gap: float):
    available = [frame for frame in frames if rating_time - width <= frame[0] < rating_time]
    if len(available) < count:
        raise ValueError(f"past{width}: only {len(available)} actual unique frames; need {count}")
    times = np.asarray([frame[0] for frame in available], dtype=float)
    span_fraction = float((times[-1] - times[0]) / width)
    boundary_tolerance = max_frame_gap
    if span_fraction < min_span_fraction:
        raise ValueError(f"past{width}: coverage span {span_fraction:.6f} < {min_span_fraction}")
    if times[0] - (rating_time - width) > boundary_tolerance or rating_time - times[-1] > boundary_tolerance:
        raise ValueError(f"past{width}: video does not cover both historical window boundaries")
    if float(np.diff(times).max()) > max_frame_gap:
        raise ValueError(f"past{width}: actual source frame gap exceeds {max_frame_gap} seconds")
    # One nearest real frame per temporal bin; bins must have distinct observations.
    edges = np.linspace(rating_time - width, rating_time, count + 1)
    chosen = []
    for left, right in zip(edges[:-1], edges[1:]):
        candidates = [frame for frame in available if left <= frame[0] < right]
        if not candidates:
            raise ValueError(f"past{width}: no real JPG in temporal bin [{left:.6f},{right:.6f})")
        chosen.append(min(candidates, key=lambda frame: (abs(frame[0] - (left + right) / 2), frame[1])))
    if len({frame[1] for frame in chosen}) != count:
        raise ValueError(f"past{width}: sampled frame IDs are not unique")
    if not np.all(np.diff([frame[0] for frame in chosen]) > 0):
        raise ValueError(f"past{width}: sampled frame timestamps are not strictly increasing")
    return chosen


def prepare(args):
    project_root = args.project_root.resolve()
    data_root = args.data_root.resolve()
    if not (project_root / "PROTOCOL.md").is_file():
        raise ValueError("PROTOCOL.md missing; project gate forbids experiments")
    if not data_root.is_dir():
        raise ValueError(f"data root does not exist: {data_root}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("tsm_manifest")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(output_dir / "preparation_log.txt", encoding="utf-8"))
    logger.addHandler(logging.StreamHandler())
    source_path = project_root / "results/param_rating_local_all_2026-10-10/source_inventory.json"
    inventory = json.loads(source_path.read_text(encoding="utf-8-sig"))
    canonical = {record["session_id"]: record for record in inventory["sources"]}
    events = pd.read_csv(args.events, encoding="utf-8-sig")
    missing = sorted((REQUIRED_EVENT_COLS | set(BASELINE_COLUMNS)) - set(events.columns))
    if missing:
        raise ValueError(f"fixed event table missing columns: {missing}")
    if events.rating_event_id.isna().any() or not events.rating_event_id.is_unique:
        raise ValueError("real rating event IDs must be present and unique")
    if not events.rating_event_id.astype(str).str.startswith("modern:").all():
        raise ValueError("fixed cohort includes nonmodern events; legacy sources must be analyzed separately")
    if not events.ms_rating.isin([1, 2, 3, 4, 5]).all():
        raise ValueError("fixed modern rating scale is not 1..5")
    if not ((events.ms_rating > 2).astype(int) == events.binary_label).all():
        raise ValueError("binary label differs from frozen MS_rating > 2 mapping")
    if not np.isfinite(events.rating_time_sec).all():
        raise ValueError("fixed event times must be finite")
    sources, raw_images, legacy_count = discover(data_root, set(events.session_id), logger)
    accepted, skipped, sessions = [], [], []
    for session, group in events.groupby("session_id", sort=True):
        try:
            if not raw_images.get(session):
                raise ValueError("no already-extracted modern JPGs found for this session")
            if session not in canonical:
                raise ValueError("session absent from canonical source inventory")
            record = canonical[session]
            if set(group.game_name) != {record["game_name"]} or set(group.participant_id) != {record["participant_id"]}:
                raise ValueError("game/participant identity differs from canonical source inventory")
            logger.info("SESSION %s raw_image_candidates=%s", session,
                        sum(len(paths) for paths in raw_images[session].values()))
            source, source_hash, aliases = canonical_source(sources.get(session, []), record, data_root)
            images, duplicates = verified_image_index(raw_images[session], data_root)
            clock, reports, clock_meta = read_frame_clock(source, session)
            if clock_meta["raw_rows"] != int(record["raw_rows_expected"]):
                raise ValueError("canonical CSV row count differs from source inventory")
            if len(reports) != int(record["rating_events_expected"]):
                raise ValueError("canonical CSV real report count differs from source inventory")
            missing_frames = sorted(set(clock) - set(images))
            if missing_frames:
                raise ValueError(f"incomplete JPG sequence: {len(missing_frames)}/{len(clock)} referenced frames missing; "
                                 f"first missing frame IDs={missing_frames[:10]}")
            frames = sorted((timestamp, frame_id, images[frame_id]) for frame_id, timestamp in clock.items())
            session_meta = dict(session_id=session, game_name=record["game_name"],
                                status="complete_video", referenced_frames=len(clock), actual_jpg_frames=len(images),
                                extra_jpg_frames=len(set(images) - set(clock)), verified_duplicate_frame_groups=duplicates,
                                matching_csv_aliases=aliases, source_sha256=source_hash, **clock_meta)
            accepted_count = 0
            for event in group.to_dict("records"):
                try:
                    actual = reports.get(event["rating_event_id"])
                    if actual is None:
                        raise ValueError("event ID does not identify a real source report")
                    if actual[1] != event["ms_rating"] or abs(actual[0] - event["rating_time_sec"]) > 1e-5:
                        raise ValueError("real source report time/value differs from fixed cohort")
                    windows = [choose_window(frames, actual[0], width, args.frames_per_window,
                                             args.min_span_fraction, args.max_frame_gap_sec)
                               for width in args.windows_sec]
                    row = dict(event_id=event["rating_event_id"], game_name=event["game_name"],
                               participant_id=event["participant_id"], session_id=session,
                               binary_true=int(event["binary_label"]), fms_raw=float(event["ms_rating"]),
                               rating_time_sec=float(actual[0]), exposure_elapsed_seconds=float(event["exposure_elapsed_seconds"]),
                               frame_paths_json=json_dump([[frame[2].relative_to(data_root).as_posix() for frame in window]
                                                          for window in windows]),
                               frame_times_json=json_dump([[frame[0] for frame in window] for window in windows]),
                               frame_ids_json=json_dump([[frame[1] for frame in window] for window in windows]),
                               window_seconds_json=json_dump(args.windows_sec),
                               source_csv_locator=source.relative_locator(data_root), source_csv_sha256=source_hash,
                               timestamp_unit=clock_meta["timestamp_unit"], session_video_reference_count=len(clock),
                               session_jpg_count=len(images))
                    row.update({column: event[column] for column in BASELINE_COLUMNS})
                    accepted.append(row)
                    accepted_count += 1
                except Exception as exc:
                    skipped.append(dict(event_id=event["rating_event_id"], session_id=session,
                                        game_name=event["game_name"], reason=f"{type(exc).__name__}: {exc}"))
            session_meta.update(accepted_events=accepted_count, fixed_cohort_events=len(group))
            sessions.append(session_meta)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            logger.warning("SKIP SESSION %s %s", session, reason)
            skipped.extend(dict(event_id=event_id, session_id=session, game_name=group.game_name.iloc[0], reason=reason)
                           for event_id in group.rating_event_id)
            sessions.append(dict(session_id=session, game_name=group.game_name.iloc[0], status="skipped", reason=reason))
    output_columns = MANIFEST_META_COLUMNS + [col for col in BASELINE_COLUMNS if col not in MANIFEST_META_COLUMNS]
    manifest = pd.DataFrame(accepted, columns=output_columns)
    skipped_frame = pd.DataFrame(skipped, columns=["event_id", "session_id", "game_name", "reason"])
    manifest.to_csv(output_dir / "manifest.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    skipped_frame.to_csv(output_dir / "skipped.csv", index=False, encoding="utf-8-sig")
    if len(manifest) + len(skipped_frame) != len(events):
        raise ValueError("accepted+skipped event accounting does not match fixed cohort")
    labels = {str(k): int(v) for k, v in manifest.binary_true.value_counts().items()}
    game_table = [dict(game_name=game, events=len(group), sessions=group.session_id.nunique(),
                       positives=int(group.binary_true.sum()), negatives=int((group.binary_true == 0).sum()))
                  for game, group in manifest.groupby("game_name", sort=True)]
    summary = dict(status="complete" if len(manifest) else "no_eligible_events",
                   scope="fixed modern real rating events with already-extracted complete JPG sequences",
                   fixed_events=len(events), accepted_events=len(manifest), skipped_events=len(skipped_frame),
                   games=int(manifest.game_name.nunique()), sessions=int(manifest.session_id.nunique()),
                   participants=int(manifest.participant_id.nunique()), binary_distribution=labels,
                   label_rule="0: MS_rating <= 2; 1: MS_rating > 2; actual reports only",
                   windows_sec=args.windows_sec, frames_per_window=args.frames_per_window,
                   historical_interval="[t-W,t); all image timestamps strictly before real report",
                   unique_frames="each window has unique actual frame IDs; windows may share historical images",
                   video_requirement="all CSV video references must exist; sample-only sequences rejected",
                   sampling="one nearest actual frame to each equal-duration temporal-bin center",
                   min_span_fraction=args.min_span_fraction, max_frame_gap_sec=args.max_frame_gap_sec,
                   source_timestamp_policy="reuse parse_vrnet_matrices.timestamp_scale; subtract first finite timestamp; verify against fixed event time",
                   events_sha256=file_sha256(args.events), canonical_inventory_sha256=file_sha256(source_path),
                   manifest_sha256=file_sha256(output_dir / "manifest.csv"),
                   baseline_feature_columns=BASELINE_COLUMNS,
                   baseline_log1p_columns=[col for col in BASELINE_COLUMNS if "fov_" not in col],
                   baseline_warning="73 raw columns; train-fold constants may be removed. Fit only training folds, never reuse the full-fit 0.682 checkpoint for OOF baseline.",
                   excluded_legacy_file_count=legacy_count, game_table=game_table, session_inventory=sessions,
                   skipped_reasons=dict(Counter(skipped_frame.reason)),
                   inference_scope="runtime video classifier, not validated Unity-only pre-exposure risk prediction")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# TSM 真实评分视频样本准备", "", "探索性准备；不训练模型，不填补评分，不复制缺失帧。", "",
             f"- 固定真实评分事件：{len(events)}", f"- 完整视频及历史窗口合格事件：{len(manifest)}",
             f"- 跳过事件：{len(skipped_frame)}", f"- 游戏／session／参与者：{summary['games']}／{summary['sessions']}／{summary['participants']}",
             f"- 二分类：{labels}", f"- 历史窗口：{args.windows_sec} 秒，每窗 {args.frames_per_window} 张不同的实际 JPG。", "",
             "所有帧均使用 CSV 实际 timestamp 对齐，严格早于真实评分。时间单位沿用已验证的现代数据约定；不由帧号或假定 FPS 推算。",
             "现代固定队列与 rollercoaster_raw 的其它版本不混合。完整性以 CSV video 引用全量覆盖为准。", "",
             "## 可用游戏", "", "| 游戏 | 事件 | session | >2 | ≤2 |", "|---|---:|---:|---:|---:|"]
    lines.extend(f"| {r['game_name']} | {r['events']} | {r['sessions']} | {r['positives']} | {r['negatives']} |" for r in game_table)
    lines.extend(["", "## 使用", "", "manifest.csv 的 frame_paths_json 为相对 --data-root 的 POSIX 路径，按 windows_sec 的顺序存放三组图像路径。",
                  "一个事件仍是一条样本；三个窗口不会增加独立评分数量。ID、游戏名、评分时间、标签均不得作为模型特征。",
                  "73 个原始参数特征附在同一行，可重新拟合同事件 LOGO 参数基线；禁止直接以旧 0.682 比较不同队列。",
                  "skipped.csv 保存每个拒绝事件的实际原因，summary.json 保存逐 session 时钟、哈希及完整性信息。",
                  "视频来自实际用户运行；即使输入仅视频，也不能直接声称开发时无用户评估已验证。", ""])
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("FINISH fixed=%s accepted=%s skipped=%s games=%s", len(events), len(manifest), len(skipped_frame), summary["games"])
    print(json.dumps({key: summary[key] for key in ("status", "fixed_events", "accepted_events", "skipped_events", "games", "sessions", "binary_distribution")}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames-per-window", type=int, default=8)
    parser.add_argument("--windows-sec", type=int, nargs="+", default=[5, 30, 60])
    parser.add_argument("--min-span-fraction", type=float, default=0.8)
    parser.add_argument("--max-frame-gap-sec", type=float, default=1.0)
    args = parser.parse_args()
    if args.frames_per_window < 2 or not args.windows_sec or any(width <= 0 for width in args.windows_sec):
        parser.error("positive windows and at least two unique frames per window required")
    if len(set(args.windows_sec)) != len(args.windows_sec):
        parser.error("window widths must be unique")
    if not 0 < args.min_span_fraction <= 1 or args.max_frame_gap_sec <= 0:
        parser.error("coverage fraction must be (0,1], gap threshold must be positive")
    prepare(args)


if __name__ == "__main__":
    main()
