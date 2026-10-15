#!/usr/bin/env python3
"""Generate synchronized engine labels only for enabled, supported thresholds.

Unknown labels are retained; session MS_rating and global quantiles are never
used. Engine supervision stays separate from pixel-only model input features.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from parse_vrnet_matrices import literal_list, numeric_matrix_list, choose_camera, camera_pose
from scipy.spatial.transform import Rotation


def rotation_series(csv_path, unit):
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    required = ['frame', 'timestamp', 'camera_name', 'v_matrix']
    if not set(required).issubset(header):
        return None, 'missing matrix columns'
    scale = {'seconds': 1., 'milliseconds': .001, 'microseconds': .000001}[unit]
    origin, previous = None, None
    rows, errors = [], 0
    for chunk in pd.read_csv(csv_path, usecols=required, chunksize=64, encoding='utf-8-sig'):
        for row in chunk.itertuples(index=False):
            raw = dict(zip(required, (getattr(row, c) for c in required)))
            t = float(raw['timestamp']) * scale
            if origin is None:
                origin = t
            record = {'time_sec': t-origin, 'rotation_rate': np.nan, 'interval_start': np.nan}
            try:
                names = literal_list(raw['camera_name'])
                matrices = numeric_matrix_list(raw['v_matrix'])
                i = choose_camera(names)
                if i is None or i >= len(matrices):
                    raise ValueError('no matched camera matrix')
                _, rotation, _ = camera_pose(matrices[i])
                identity = str(names[i])
                if previous and identity == previous[2] and t > previous[0]:
                    dt = t - previous[0]
                    record['interval_start'] = previous[0] - origin
                    record['rotation_rate'] = float(Rotation.from_matrix(previous[1].T @ rotation).magnitude() * 180 / np.pi / dt)
                previous = (t, rotation, identity)
            except Exception:
                errors += 1
                previous = None
            rows.append(record)
    return pd.DataFrame(rows), f'matrix row errors={errors}; total angular rate includes head movement, not isolated artificial yaw'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clips-manifest', required=True, type=Path)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--report', required=True, type=Path)
    args = p.parse_args()
    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    if not config.get('frozen'):
        raise ValueError('label configuration must be frozen')
    labels = config['labels']
    if isinstance(labels, dict):
        labels = [dict(x, id=k) for k, x in labels.items()]
    clips = pd.read_csv(args.clips_manifest, encoding='utf-8-sig')
    if clips.clip_id.duplicated().any():
        raise ValueError('duplicate clip IDs')
    active = [l for l in labels if l.get('enable', l.get('enabled', False)) and l['status'] == 'usable'
              and isinstance(l.get('threshold_value'), (float, int)) and not isinstance(l.get('threshold_value'), bool)
              and np.isfinite(l['threshold_value'])
              and l.get('evidence', {}).get('transfer_to_vrnet') is True
              and l.get('feature_units_verified') is True
              and l.get('timestamp_alignment_verified') is True
              and l.get('scope_match_verified') is True]
    rows, notes = [], []
    for session, group in clips.groupby('session_id', sort=False):
        series = None
        if active:
            # Current view matrix contains total camera motion. It does not
            # establish artificial locomotion independent of tracked head motion.
            if any(l.get('metric', '').startswith('artificial_camera') for l in active):
                raise ValueError('Artificial camera/head-motion separation is unverified; cannot enable this supervision')
            if any(l['id'] != 'fast_camera_rotation' for l in active):
                raise NotImplementedError('additional validated metrics require explicit engine definitions; refusing to substitute')
            series, note = rotation_series(Path(group.timestamp_csv.iloc[0]), group.timestamp_unit.iloc[0])
            notes.append(f'{session}: {note}')
        for row in group.itertuples():
            out = {'clip_id': row.clip_id, 'session_id': session, 'game_name': row.game_name,
                   'start_time': row.start_time, 'end_time': row.end_time,
                   'label_config_sha256': hashlib.sha256(config_bytes).hexdigest()}
            for label in labels:
                value, reason = 'unknown', f"{label['status']}; no enabled transferable literature threshold"
                if label in active:
                    # Any interval crossing the clip boundary is excluded.
                    intervals = None if series is None else series[(series.time_sec >= row.start_time) &
                              (series.time_sec <= row.end_time) & (series.interval_start >= row.start_time)].dropna()
                    coverage, usable = 0., None
                    clip_config = config.get('clip', {})
                    max_interval = clip_config.get('max_interval_sec', .5)
                    minimum_coverage = clip_config.get('min_time_coverage_fraction', .8)
                    if intervals is not None and len(intervals):
                        dt = intervals.time_sec - intervals.interval_start
                        intervals = intervals[dt.le(max_interval) & dt.gt(0)]
                        coverage = float((intervals.time_sec - intervals.interval_start).sum() / (row.end_time-row.start_time))
                        usable = intervals.rotation_rate
                    if usable is not None and len(usable) and coverage >= minimum_coverage:
                        if label.get('operator') not in ('>', 'greater_than'):
                            raise ValueError('only explicitly defined > operator currently implemented')
                        aggregation = label.get('aggregation', 'max')
                        if aggregation not in ('max', 'mean'):
                            raise ValueError('undefined temporal aggregation')
                        metric = float(usable.max()) if aggregation == 'max' else float(
                            np.average(usable, weights=intervals.time_sec-intervals.interval_start))
                        value = int(metric > label['threshold_value'])
                        reason = 'matched engine intervals; validated configured threshold'
                    else:
                        reason = f'insufficient valid engine time coverage ({coverage:.3f}); requires {minimum_coverage:.3f}'
                out[label['id']] = value
                out[label['id'] + '__reason'] = reason
            rows.append(out)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(args.output, index=False, encoding='utf-8-sig')
    report = ['# 片段级风险标签生成', '', '探索性准备，非最终结论。没有使用 MS_rating 或 P80；unknown 不作为负例。',
              f'窗口：[start_time,end_time)，CSV 时间原点与片段清单一致。共 {len(result)} 个片段，{clips.session_id.nunique()} 个 session。',
              f'冻结配置 SHA256：`{hashlib.sha256(config_bytes).hexdigest()}`。', '', '| 标签 | 状态 | 0 | 1 | unknown |', '|---|---|---:|---:|---:|']
    for l in labels:
        values = result[l['id']].astype(str).value_counts()
        report.append(f"| {l['id']} | {l['status']} | {values.get('0',0)} | {values.get('1',0)} | {values.get('unknown',0)} |")
    report += ['', '启用项数：' + str(len(active)), '没有可用标签时不会训练分类器或生成 SHAP 图。', *notes, '']
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps({'clips': len(result), 'active_labels': len(active), 'status': 'labels_generated' if active else 'blocked_no_valid_thresholds'}))


if __name__ == '__main__':
    main()
