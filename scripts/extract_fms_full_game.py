#!/usr/bin/env python3
"""Historical RGB features for every session in a verified complete game.

Uses the same fixed 3s/30s, 2 Hz feature recipe as the prior pilot, but reads
the fully extracted CSV and JPG sequence. Does not fill FMS or use future data.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import logging
import math
from pathlib import Path
import platform
import sys
import time
import traceback

import cv2
import numpy as np
import pandas as pd

from extract_fms_rgb_pilot import FEATURES, WINDOWS, load_rows, resolve_member, window_features
from extract_lightgbm_features import frame_metrics, resize_frame


class FileFrameCache:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.items = OrderedDict()

    def read(self, member: str):
        if member in self.items:
            self.items.move_to_end(member)
            return self.items[member]
        path = (self.root / member).resolve()
        path.relative_to(self.root)
        payload = path.read_bytes()
        frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError('JPG decode failed: ' + str(path))
        gray, metrics = frame_metrics(resize_frame(frame, 224))
        item = (gray, metrics, hashlib.sha256(payload).hexdigest())
        self.items[member] = item
        if len(self.items) > 512:
            self.items.popitem(last=False)
        return item


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir', required=True, type=Path)
    parser.add_argument('--game', default='City_Parkour')
    args = parser.parse_args()
    project = args.project_dir.resolve()
    if not (project / 'PROTOCOL.md').is_file():
        raise FileNotFoundError('PROTOCOL.md missing')
    game = args.game
    summary_path = project / 'results/fms_full_game_download_summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    if summary.get('game_name') != game or not summary.get('complete_game'):
        raise RuntimeError('Complete game acquisition must be verified first')
    selected = sorted(summary['sessions'], key=lambda r: r['session_id'])
    if len(selected) != summary['verified_index_session_count']:
        raise RuntimeError('Incomplete manifest')
    root = project / 'data/raw/vrnet/full_game' / game
    members = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if sum(m.lower().endswith(('.jpg', '.jpeg')) for m in members) != summary['retained_jpg_count']:
        raise RuntimeError('Physical JPG count changed since acquisition')
    for r in selected:
        if not Path(r['csv_path']).is_file():
            raise FileNotFoundError(r['csv_path'])
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(project / 'results/fms_full_game_extraction_log.txt', mode='w', encoding='utf-8'), logging.StreamHandler(sys.stdout)])
    logging.info('COMMAND %s', json.dumps(sys.argv, ensure_ascii=False))
    cv2.setNumThreads(1)
    cv2.setRNGSeed(42)
    plan = dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(), game_name=game,
                sessions=[r['session_id'] for r in selected], all_index_sessions=True,
                acquisition=summary, windows_sec=WINDOWS, fps=2, resize_max_dimension=224,
                label='Original nonempty numeric MS_rating, no carry-forward, no binning, expected local recorded scale 1-5',
                input='33 RGB features; elapsed time separately tested; no HMD/matrix/controller/previous-score inputs',
                quality='Same fixed frame and temporal coverage criteria as extract_fms_rgb_pilot.py; historical [t-W,t) only',
                evaluation='Leave-One-Session-Out; conservatively group duplicate numeric prefixes if present (unverified identity)',
                protocol_deviation='User-authorized exploratory raw-score regression, not original AUC/classification/subject gate',
                features=FEATURES, environment=dict(python=platform.python_version(), cv2=cv2.__version__, numpy=np.__version__, pandas=pd.__version__),
                same_game_followup='Known game; prior five sessions inspected. Single-game holdout does not test cross-game/general Unity risk prediction.')
    (project / 'audit/fms_full_game_extraction_definition.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    events, feature_rows, audits = [], [], []
    begin = time.perf_counter()
    for ordinal, source in enumerate(selected, 1):
        sid = source['session_id']
        start = time.perf_counter()
        audit = dict(game_name=game, session_id=sid, csv_path=source['csv_path'], jpg_count=source['jpg_count'], status='failed', error='')
        logging.info('[%d/%d] reading ALL session %s', ordinal, len(selected), sid)
        try:
            rows = load_rows(source['csv_path'])
            rows['raw_row_index'] = np.arange(len(rows))
            timestamps = pd.to_numeric(rows['timestamp'], errors='coerce')
            if timestamps.isna().any():
                raise ValueError('Missing/non-numeric timestamp')
            if np.any(np.diff(timestamps.to_numpy(dtype=float)) < 0):
                raise ValueError('Timestamp moves backwards; not silently sorted')
            origin = float(timestamps.iloc[0])
            if not 1e12 <= origin < 1e14:
                raise ValueError('Epoch-ms timestamp assumption unsupported')
            rows['time_sec'] = (timestamps - origin) / 1000
            numeric_scores = pd.to_numeric(rows['MS_rating'], errors='coerce')
            reports = rows.loc[numeric_scores.notna()].copy()
            reports['fms_raw'] = numeric_scores.loc[numeric_scores.notna()].astype(float)
            initial_reports = len(reports)
            invalid_score_text = rows['MS_rating'].notna() & rows['MS_rating'].str.strip().ne('') & numeric_scores.isna()
            reports = reports.drop_duplicates(['timestamp', 'fms_raw'], keep='first')
            conflicts = set(reports.groupby('timestamp')['fms_raw'].nunique().loc[lambda x:x > 1].index)
            frames = rows.loc[rows['video'].notna() & rows['video'].str.strip().ne('')].copy()
            frames['member'] = [resolve_member(sid, value, members) for value in frames['video']]
            frames = frames.drop_duplicates(['timestamp', 'video'], keep='first')
            if frames['time_sec'].duplicated().any():
                raise ValueError('Conflicting frame references at same timestamp')
            cache = FileFrameCache(root)
            for _, report in reports.iterrows():
                event = dict(event_id=sid + '__row' + str(int(report['raw_row_index'])),
                             session_id=sid, game_name=game, raw_row_index=int(report['raw_row_index']),
                             event_frame=str(report['frame']), event_timestamp_ms=float(report['timestamp']),
                             origin_timestamp_ms=origin, event_time_sec=float(report['time_sec']), elapsed_sec=float(report['time_sec']),
                             fms_raw=float(report['fms_raw']), label_status='ok', csv_source=source['csv_path'])
                if report['timestamp'] in conflicts:
                    event['label_status'] = 'conflicting_scores_same_timestamp'
                elif not math.isfinite(event['fms_raw']) or not 1 <= event['fms_raw'] <= 5:
                    event['label_status'] = 'outside_expected_local_1_5_scale'
                events.append(event)
                for window in WINDOWS:
                    if event['label_status'] != 'ok':
                        values = dict(window_sec=window, status='skipped', reason=event['label_status'], **{k:math.nan for k in FEATURES})
                    else:
                        try:
                            values = window_features(frames, event['event_time_sec'], window, cache)
                        except Exception:
                            failure = traceback.format_exc()
                            logging.error('%s event %s window=%s\n%s', sid, event['event_id'], window, failure)
                            values = dict(window_sec=window, status='skipped', reason=failure, **{k:math.nan for k in FEATURES})
                    feature_rows.append(dict(event, **values))
            audit.update(status='ok', csv_rows=len(rows), recording_duration_sec=float(rows['time_sec'].iloc[-1]),
                         nonempty_numeric_scores=initial_reports, invalid_nonempty_score_text=int(invalid_score_text.sum()),
                         duplicate_reports_removed=initial_reports-len(reports), events=len(reports),
                         missing_jpg_references=int(frames['member'].isna().sum()), frame_references=len(frames))
        except Exception:
            audit['error'] = traceback.format_exc()
            logging.error('SESSION FAILED %s\n%s', sid, audit['error'])
        audit['wall_seconds'] = time.perf_counter() - start
        audits.append(audit)
        pd.DataFrame(events).to_csv(project / 'features/fms_full_game_events.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(feature_rows).to_csv(project / 'features/fms_full_game_event_features.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(audits).to_csv(project / 'results/fms_full_game_session_audit.csv', index=False, encoding='utf-8-sig')
        logging.info('%s finished %.1fs; cumulative scores=%d', sid, audit['wall_seconds'], len(events))
    if any(a['status'] != 'ok' for a in audits):
        raise RuntimeError('One or more full-game sessions failed; do not present partial subset as complete training')
    features = pd.DataFrame(feature_rows)
    if features.empty:
        raise RuntimeError('No feature rows')
    features.groupby(['session_id', 'window_sec', 'status']).size().reset_index(name='events').to_csv(project / 'results/fms_full_game_coverage.csv', index=False, encoding='utf-8-sig')
    valid = features.loc[features.status.eq('ok')]
    stats = valid[FEATURES].agg(['count', 'mean', 'std', 'min', 'max']).T
    stats['missing_rate'] = valid[FEATURES].isna().mean()
    stats.to_csv(project / 'features/fms_full_game_feature_statistics.csv', encoding='utf-8-sig', index_label='feature')
    logging.info('DONE full game: %d sessions, %d scores, %d valid window rows, %.1fs', len(audits), len(events), len(valid), time.perf_counter()-begin)


if __name__ == '__main__':
    main()
