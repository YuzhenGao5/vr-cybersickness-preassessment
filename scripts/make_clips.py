#!/usr/bin/env python3
"""Create timestamp-aligned 1-second clips; never equate 30 JPGs with 1 second.

Source timestamps are explicitly converted to seconds, with origin at the first
CSV timestamp. Irregular captures are resampled with nearest-in-window frames.
Upsampling duplicates source images; it does not recover missing motion. Clips
with gaps, sparse frames or incomplete boundaries are skipped and audited.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
import cv2
import numpy as np
import pandas as pd

SCALES = {'seconds': 1., 'milliseconds': .001, 'microseconds': .000001}


def read_jpg(path):
    # OpenCV imread on Windows cannot consistently open non-ASCII filenames.
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def read_index(csv_path, frames_dir, timestamp_unit):
    df = pd.read_csv(csv_path, usecols=['frame', 'timestamp', 'video'], encoding='utf-8-sig')
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='raise')
    if not np.isfinite(df['timestamp']).all() or not df['timestamp'].is_monotonic_increasing:
        raise ValueError('timestamps must be finite and monotonically increasing')
    origin = float(df.timestamp.iloc[0]) * SCALES[timestamp_unit]
    df['time_sec'] = df.timestamp.astype(float) * SCALES[timestamp_unit] - origin
    df = df[df.video.notna() & df.video.astype(str).str.strip().ne('')].copy()
    paths = []
    for ref in df.video.astype(str):
        rel = PurePosixPath(ref.replace('\\', '/'))
        if '..' in rel.parts or rel.is_absolute() or PureWindowsPath(ref).drive:
            raise ValueError(f'unsafe video reference: {ref}')
        path = Path(frames_dir) / str(rel)
        if not rel.suffix:
            path = path.with_suffix('.jpg')
        path.resolve().relative_to(Path(frames_dir).resolve())
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path.resolve())
    df['jpg_path'] = paths
    df = df.drop_duplicates('time_sec', keep='first').reset_index(drop=True)
    return df, origin


def create_clips(spec, args):
    session, game = spec['session_id'], spec['game_name']
    if session in ('.', '..') or game in ('.', '..') or any(c in session + game for c in '/\\:'):
        raise ValueError('unsafe session or game name')
    index, origin = read_index(spec['csv_path'], spec['frames_dir'], args.timestamp_unit)
    if len(index) < 2:
        raise ValueError(f'{session}: insufficient JPG timestamps')
    times = index.time_sec.to_numpy()
    records, skips, candidates = [], [], []
    last_end = float(times[-1])
    for k in range(int(np.ceil(last_end / args.window_sec))):
        start, end = k * args.window_sec, (k + 1) * args.window_sec
        rows = index[(index.time_sec >= start) & (index.time_sec < end)]
        why = None
        if end > last_end:
            why = 'incomplete session boundary'
        elif len(rows) < 2:
            why = 'fewer than two source frames'
        elif max(np.diff(np.r_[start, rows.time_sec.to_numpy(), end])) > args.max_gap_sec:
            why = 'capture gap exceeds max-gap-sec'
        if why:
            skips.append({'session_id': session, 'start_time': start, 'end_time': end, 'reason': why})
        else:
            candidates.append((k, rows))
    if args.max_clips_per_session and len(candidates) > args.max_clips_per_session:
        selected = set(np.linspace(0, len(candidates)-1, args.max_clips_per_session).round().astype(int))
        candidates = [x for i, x in enumerate(candidates) if i in selected]
    out = Path(args.output_dir).resolve() / game / session
    out.resolve().relative_to(Path(args.output_dir).resolve())
    out.mkdir(parents=True, exist_ok=True)
    for k, rows in candidates:
        if shutil.disk_usage(out).free < args.min_free_bytes:
            raise OSError('clip writing stopped at disk free-space floor')
        start, end = k * args.window_sec, (k+1) * args.window_sec
        count = round(args.fps * args.window_sec)
        targets = start + np.arange(count) / args.fps
        source_times = rows.time_sec.to_numpy()
        picked = np.abs(source_times[:, None] - targets).argmin(axis=0)
        paths = rows.jpg_path.tolist()
        first = read_jpg(paths[picked[0]])
        if first is None:
            raise ValueError(f'cannot decode {paths[picked[0]]}')
        height, width = first.shape[:2]
        clip_id = f'{session}__{k:06d}'
        dest = out / f'{clip_id}.mp4'
        # The system sandbox Temp directory is unavailable to FFmpeg on some
        # Windows hosts. Use a contained sibling output, never a global cwd file.
        tmp_dest = out / (clip_id + '.encoding.mp4')
        try:
            writer = cv2.VideoWriter(str(tmp_dest), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f'video writer cannot open {tmp_dest}')
            cached = {}
            try:
                for i in picked:
                    i = int(i)
                    if i not in cached:
                        img = read_jpg(paths[i])
                        if img is None or img.shape[:2] != (height, width):
                            raise ValueError(f'unreadable or inconsistent frame {paths[i]}')
                        cached[i] = img
                    writer.write(cached[i])
            finally:
                writer.release()
            cap = cv2.VideoCapture(str(tmp_dest))
            decoded = 0
            while cap.read()[0]:
                decoded += 1
            cap.release()
            if decoded != count:
                raise ValueError(f'encoded frame count {decoded} != expected {count}')
            tmp_dest.replace(dest)
        finally:
            if tmp_dest.exists():
                tmp_dest.resolve().relative_to(out.resolve())
                tmp_dest.unlink()
        records.append({'clip_id': clip_id, 'session_id': session, 'game_name': game,
                        'clip_path': str(dest), 'start_time': start, 'end_time': end,
                        'start_frame': int(rows.frame.iloc[0]), 'end_frame': int(rows.frame.iloc[-1]),
                        'source_frame_count': len(rows), 'encoded_frame_count': count,
                        'unique_resampled_frames': len(set(picked.tolist())), 'resampled_fps': args.fps,
                        'resampled_source_frame_ids': json.dumps(rows.frame.to_numpy()[picked].astype(int).tolist()),
                        'resampled_source_times': json.dumps(source_times[picked].tolist()),
                        'resampled_target_times': json.dumps(targets.tolist()),
                        'max_resampling_error_sec': float(np.max(np.abs(source_times[picked]-targets))),
                        'timestamp_unit': args.timestamp_unit, 'origin_seconds': origin,
                        'timestamp_csv': str(Path(spec['csv_path']).resolve()),
                        'clip_sha256': hashlib.sha256(dest.read_bytes()).hexdigest()})
    return records, skips, {'session_id': session, 'source_frames': len(index),
                            'duration_seconds': last_end, 'clips_written': len(records),
                            'windows_skipped': len(skips), 'sampled_limit': args.max_clips_per_session}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sessions-json', type=Path, required=True,
                   help='List of session_id/game_name/csv_path/frames_dir records')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--timestamp-unit', choices=SCALES, required=True)
    p.add_argument('--window-sec', type=float, default=1.)
    p.add_argument('--fps', type=float, default=30.)
    p.add_argument('--max-gap-sec', type=float, default=.5,
                   help='Capture quality guard; NOT a risk threshold')
    p.add_argument('--max-clips-per-session', type=int, default=0,
                   help='0=all; local smoke check can uniformly subsample')
    p.add_argument('--min-free-bytes', type=int, default=50000000000)
    p.add_argument('--log', type=Path, required=True)
    args = p.parse_args()
    args.min_free_bytes = max(args.min_free_bytes, 50000000000)
    if min(args.window_sec, args.fps, args.max_gap_sec) <= 0:
        raise ValueError('time/fps parameters must be positive')
    records, skips, summaries = [], [], []
    try:
        specs = json.loads(args.sessions_json.read_text(encoding='utf-8-sig'))
        for spec in specs:
            r, s, summary = create_clips(spec, args)
            records.extend(r); skips.extend(s); summaries.append(summary)
            print(json.dumps(summary), flush=True)
        if not records:
            raise ValueError('No usable timestamp-aligned clips; stop before feature extraction')
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(args.manifest, index=False, encoding='utf-8-sig')
        pd.DataFrame(skips, columns=['session_id','start_time','end_time','reason']).to_csv(
            args.manifest.with_name(args.manifest.stem + '_skipped.csv'), index=False, encoding='utf-8-sig')
    finally:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(json.dumps({'date_utc': datetime.now(timezone.utc).isoformat(),
                           'resampling_note': 'Duplicates source frames; no recovered 30Hz motion. Windows [start,end).',
                           'sessions': summaries, 'clips': len(records), 'skips': len(skips)},
                           indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
