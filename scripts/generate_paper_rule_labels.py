#!/usr/bin/env python3
"""Generate fixed paper-rule clip labels from actual VR.net view matrices.

These are engineering rule labels, not human-validated sickness thresholds.
No FMS values are used. Source coordinates are explicitly ASSUMED metric.
Velocity is assigned to position-interval midpoints; vector acceleration is
assigned to velocity-interval midpoints, using the actual irregular clock.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from parse_vrnet_matrices import camera_pose, choose_camera, literal_list, numeric_matrix_list, timestamp_scale

LABEL_NAME = '按论文规则生成的风险因素标签'
SPEED_THRESHOLD = 3.0
ACCELERATION_THRESHOLD = 0.7


def infer_game(session_id: str) -> str:
    found = re.fullmatch(r'\d+_\d+_(.+)', session_id)
    return found.group(1) if found else 'unknown'


def discover(raw_dir: Path, manifest: pd.DataFrame):
    """Prefer the exact CSV used to encode a clip; deduplicate session copies."""
    preferred = {str(s): Path(g.timestamp_csv.iloc[0]).resolve()
                 for s, g in manifest.groupby('session_id')} if len(manifest) else {}
    paths = sorted(raw_dir.rglob('data_file.csv'))
    by_session = {}
    for path in paths:
        by_session.setdefault(path.parent.name, []).append(path.resolve())
    for session, path in preferred.items():
        if not path.is_file():
            raise FileNotFoundError(f'Exact source CSV for existing video clips is missing: {path}')
        if path not in by_session.setdefault(session, []):
            by_session[session].append(path)
    records = []
    for session, copies in sorted(by_session.items()):
        path = preferred.get(session, copies[0])
        records.append({'session_id': session, 'game_name': infer_game(session),
                        'path': path, 'copies': [str(p) for p in copies]})
    return records


def decode_position(row, stable_camera):
    names = literal_list(row.get('camera_name'))
    matrices = numeric_matrix_list(row.get('v_matrix'))
    if not names or not len(matrices):
        return None, None, 'missing_camera_matrix'
    if len(names) != len(matrices):
        raise ValueError(f'camera_name count {len(names)} != v_matrix count {len(matrices)}')
    if stable_camera is not None and stable_camera not in names:
        return None, None, 'missing_primary_camera'
    if stable_camera is not None and names.count(stable_camera) != 1:
        return None, None, 'ambiguous_primary_camera'
    if stable_camera in names:
        index = names.index(stable_camera)
    else:
        index = choose_camera(names)
    name = str(names[index])
    if names.count(names[index]) != 1:
        return None, None, 'ambiguous_primary_camera'
    position, _, _ = camera_pose(matrices[index])
    if not np.isfinite(position).all():
        raise ValueError('nonfinite inverse-view position')
    return position, name, None


def summarize_session(spec, args, log):
    path, session, game = spec['path'], spec['session_id'], spec['game_name']
    columns = list(pd.read_csv(path, nrows=0, encoding='utf-8-sig').columns)
    if not {'frame', 'timestamp'}.issubset(columns):
        raise ValueError('Missing frame/timestamp columns; cannot invent a timeline')
    has_matrices = {'camera_name', 'v_matrix'}.issubset(columns)
    usecols = ['frame', 'timestamp'] + (['camera_name', 'v_matrix'] if has_matrices else [])
    positions, velocities, accelerations, quality = [], [], [], []
    origin = scale = unit = stable_camera = None
    previous_pose = previous_velocity = None
    last_raw = None
    max_time = 0.0
    count = 0
    counters = {'missing_camera_matrix': 0, 'missing_primary_camera': 0,
                'ambiguous_primary_camera': 0, 'parse_error': 0, 'camera_switch': 0,
                'coordinate_jump': 0, 'capture_gap': 0, 'nonpositive_dt': 0}
    parse_examples = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=args.chunk_size,
                             encoding='utf-8-sig', low_memory=False):
        for row in chunk.to_dict('records'):
            count += 1
            raw = float(row['timestamp'])
            if not math.isfinite(raw):
                raise ValueError(f'Nonfinite timestamp in row {count}')
            if origin is None:
                origin = raw
                scale, unit = timestamp_scale(origin)
                if args.timestamp_unit != 'auto':
                    scale, unit = {'seconds': (1., 'seconds'), 'milliseconds': (.001, 'milliseconds')}[args.timestamp_unit]
            if last_raw is not None and raw < last_raw:
                raise ValueError(f'Nonmonotonic timestamp in row {count}; no silent sorting')
            last_raw = raw
            current_time = (raw - origin) * scale
            max_time = max(max_time, current_time)
            if not has_matrices:
                continue
            try:
                position, camera, error = decode_position(row, stable_camera)
            except Exception:
                position = camera = None
                error = 'parse_error'
                if len(parse_examples) < 5:
                    parse_examples.append({'frame': str(row['frame']), 'error': traceback.format_exc()})
            if error:
                counters[error] += 1
                quality.append((current_time, error))
                previous_pose = previous_velocity = None
                continue
            positions.append((current_time, int(row['frame']), camera, *position.tolist()))
            if stable_camera is None:
                stable_camera = camera
            if previous_pose is not None:
                previous_time, previous_position, previous_camera = previous_pose
                dt = current_time - previous_time
                displacement = float(np.linalg.norm(position - previous_position))
                issue = None
                if camera != previous_camera:
                    issue = 'camera_switch'
                elif dt <= 0:
                    issue = 'nonpositive_dt'
                elif dt > args.max_gap_sec:
                    issue = 'capture_gap'
                elif displacement > args.jump_source_units:
                    issue = 'coordinate_jump'
                if issue:
                    counters[issue] += 1
                    quality.append((current_time, issue))
                    previous_velocity = None
                else:
                    velocity = (position - previous_position) * args.meters_per_source_unit / dt
                    velocity_time = (current_time + previous_time) / 2.
                    velocities.append((velocity_time, float(np.linalg.norm(velocity)), dt))
                    if previous_velocity is not None:
                        old_velocity_time, old_velocity = previous_velocity
                        velocity_dt = velocity_time - old_velocity_time
                        if velocity_dt > 0:
                            acceleration = (velocity - old_velocity) / velocity_dt
                            acceleration_time = (velocity_time + old_velocity_time) / 2.
                            accelerations.append((acceleration_time, float(np.linalg.norm(acceleration))))
                    previous_velocity = (velocity_time, velocity)
            previous_pose = (current_time, position, camera)
    if origin is None or count == 0:
        raise ValueError('Empty source CSV')
    n_windows = max(1, int(math.ceil(max_time)))
    rows = [{'clip_id': f'{session}__{k:06d}', 'session_id': session, 'game_name': game,
             'start_time': float(k), 'end_time': float(k+1),
             'session_duration_sec': max_time, 'partial_session_boundary': bool(k+1 > max_time),
             'raw_timestamp_origin': origin, 'timestamp_unit': unit,
             'meters_per_source_unit_assumed': args.meters_per_source_unit,
             'coordinate_unit_calibrated': False, 'camera_sample_count': 0,
             'camera_speed_sample_count': 0, 'camera_acceleration_sample_count': 0,
             'camera_speed_max_m_s_assumed': np.nan, 'camera_acceleration_max_m_s2_assumed': np.nan,
             'quality_issue_count': 0, 'fast_camera_speed': 'unknown',
             'high_camera_acceleration': 'unknown', 'label_source': LABEL_NAME,
             'source_csv': str(path), 'acceleration_definition': 'vector_velocity_midpoint_difference',
             'fast_camera_speed_reason': 'no_valid_velocity_samples',
             'high_camera_acceleration_reason': 'no_valid_acceleration_samples'} for k in range(n_windows)]
    def window(t):
        k = int(math.floor(t))
        return rows[k] if 0 <= k < n_windows else None
    for t, *_ in positions:
        w = window(t)
        if w is not None:
            w['camera_sample_count'] += 1
    for t, speed, _ in velocities:
        w = window(t)
        if w is not None:
            w['camera_speed_sample_count'] += 1
            old = w['camera_speed_max_m_s_assumed']
            w['camera_speed_max_m_s_assumed'] = speed if np.isnan(old) else max(speed, old)
    for t, acceleration in accelerations:
        w = window(t)
        if w is not None:
            w['camera_acceleration_sample_count'] += 1
            old = w['camera_acceleration_max_m_s2_assumed']
            w['camera_acceleration_max_m_s2_assumed'] = acceleration if np.isnan(old) else max(acceleration, old)
    for t, _ in quality:
        w = window(t)
        if w is not None:
            w['quality_issue_count'] += 1
    for w in rows:
        for label, field, threshold in [('fast_camera_speed', 'camera_speed_max_m_s_assumed', SPEED_THRESHOLD),
                                        ('high_camera_acceleration', 'camera_acceleration_max_m_s2_assumed', ACCELERATION_THRESHOLD)]:
            if math.isfinite(w[field]):
                w[label] = int(w[field] > threshold)
                w[label + '_reason'] = 'fixed_paper_rule_on_observed_valid_samples'
            elif not has_matrices:
                w[label + '_reason'] = 'missing_matrix_columns'
    stats = {'session_id': session, 'game_name': game, 'source_csv': str(path),
             'source_size_bytes': path.stat().st_size, 'csv_rows': count,
             'has_matrix_columns': has_matrices, 'timestamp_unit_inferred': unit,
             'raw_timestamp_origin': origin, 'duration_seconds': max_time,
             'camera_position_samples': len(positions), 'velocity_samples': len(velocities),
             'vector_acceleration_samples': len(accelerations), 'quality_events': counters,
             'parse_error_examples': parse_examples, 'one_second_windows': len(rows),
             'duplicate_source_paths': spec['copies'], 'stable_camera_preference': stable_camera}
    log.write(json.dumps(stats, ensure_ascii=False) + '\n')
    log.flush()
    print(json.dumps({k: stats[k] for k in ['session_id', 'csv_rows', 'has_matrix_columns', 'one_second_windows'] }), flush=True)
    return rows, stats


def distribution(frame, label):
    x = pd.to_numeric(frame[label], errors='coerce')
    return {'positive': int(x.eq(1).sum()), 'negative': int(x.eq(0).sum()), 'unknown': int(x.isna().sum())}


def merge_existing_video(labels, video, manifest):
    keys = ['clip_id', 'session_id', 'game_name']
    for frame, name in [(labels, 'labels'), (video, 'video'), (manifest, 'manifest')]:
        if frame.duplicated(keys).any():
            raise ValueError(f'{name} has duplicate clip keys')
    missing = video[keys].merge(labels[keys], on=keys, how='left', indicator=True)
    if missing['_merge'].ne('both').any():
        raise ValueError('Existing video clips are missing corresponding paper-rule timeline labels')
    timing = video[keys + ['start_time', 'end_time']].merge(
        labels[keys + ['start_time', 'end_time', 'raw_timestamp_origin']], on=keys, validate='one_to_one')
    for field in ['start_time', 'end_time']:
        if not np.allclose(timing[field+'_x'], timing[field+'_y'], rtol=0, atol=1e-6):
            raise ValueError(f'{field} differs between video and labels; cannot claim alignment')
    origins = timing.merge(manifest[keys + ['origin_seconds']], on=keys, validate='one_to_one')
    if len(origins) != len(video) or not np.allclose(origins.raw_timestamp_origin*.001, origins.origin_seconds, rtol=0, atol=1e-4):
        raise ValueError('Clip manifest origin differs from raw CSV origin')
    label_cols = [c for c in labels.columns if c not in video.columns or c in keys]
    return video.merge(labels[label_cols], on=keys, how='left', validate='one_to_one')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-dir', type=Path, default=Path('data/raw/vrnet'))
    p.add_argument('--manifest', type=Path, default=Path('features/video_clips_manifest.csv'))
    p.add_argument('--video-features', type=Path, default=Path('features/lightgbm_features.csv'))
    p.add_argument('--output', type=Path, default=Path('features/paper_rule_labels.csv'))
    p.add_argument('--training-output', type=Path, default=Path('features/paper_rule_training_matrix.csv'))
    p.add_argument('--report', type=Path, default=Path('results/paper_rule_label_distribution.md'))
    p.add_argument('--log', type=Path, default=Path('results/paper_rule_label_generation_log.jsonl'))
    p.add_argument('--definition', type=Path, default=Path('results/paper_rule_definition.json'))
    p.add_argument('--timestamp-unit', choices=['auto', 'milliseconds', 'seconds'], default='auto')
    p.add_argument('--meters-per-source-unit', type=float, default=1.)
    p.add_argument('--max-gap-sec', type=float, default=.5)
    p.add_argument('--jump-source-units', type=float, default=10.)
    p.add_argument('--chunk-size', type=int, default=64)
    a = p.parse_args()
    if min(a.meters_per_source_unit, a.max_gap_sec, a.jump_source_units, a.chunk_size) <= 0:
        raise ValueError('Units and quality parameters must be positive')
    manifest = pd.read_csv(a.manifest, encoding='utf-8-sig')
    video = pd.read_csv(a.video_features, encoding='utf-8-sig')
    if not np.allclose(manifest.end_time-manifest.start_time, 1.):
        raise ValueError('Existing clips must be one second')
    if not manifest.timestamp_unit.eq('milliseconds').all():
        raise ValueError('Existing manifest uses a different clock; do not assume alignment')
    if a.timestamp_unit == 'seconds':
        raise ValueError('The existing video manifest is milliseconds; a seconds override would invalidate its alignment')
    definition = {'label_name': LABEL_NAME, 'created_utc': datetime.now(timezone.utc).isoformat(),
                  'source': 'https://arxiv.org/abs/2306.03381v1', 'doi': '10.48550/arXiv.2306.03381',
                  'paper_location': 'section4.1 PDF p10', 'thresholds': {'fast_camera_speed': {'operator':'>','value':3.,'unit':'m/s'},
                  'high_camera_acceleration':{'operator':'>','value':.7,'unit':'m/s^2'}},
                  'coordinate_unit_assumed': '1 source coordinate unit = 1 meter unless explicit scale supplied',
                  'meters_per_source_unit_assumed': a.meters_per_source_unit, 'coordinate_unit_calibrated': False,
                  'timestamp_method': '(raw timestamp - first CSV timestamp)*inferred scale; epoch magnitude, with an independently audited example duration in unit_and_timestamp_check.md',
                  'window': '[k,k+1) seconds from first source record',
                  'velocity': '(position[i]-position[i-1])/actual_dt; observation time=interval midpoint',
                  'acceleration': 'norm(vector_velocity_difference)/difference_of_velocity_midpoint_times; observation time=midpoint',
                  'window_statistic': 'maximum observed valid motion sample; missing all samples => unknown',
                  'quality_rules': {'max_capture_gap_sec':a.max_gap_sec, 'jump_displacement_source_units':a.jump_source_units,
                  'reset_on_missing_or_invalid_matrix':True,'reset_on_camera_switch':True,
                  'camera_selection':'First valid Main/Center preferred name is held fixed; missing or duplicate identity resets derivatives',
                  'note':'quality/coordinate-boundary guards, not new risk thresholds; quality events are audited'},
                  'paper_does_not_specify_all_local_implementation_details':True,
                  'multi_axis_rotation':'skipped; paper numeric activation threshold unavailable',
                  'human_validated_threshold':False,'fms_used':False,'tsm_training':False,
                  'training_protocol': {'model':'LightGBM per factor','split':'Leave-One-Game-Out','seed':42,
                  'prediction_probability_threshold':.5,'primary_features':'existing 28 RGB feature whitelist; no threshold-defining matrix inputs'},
                  'input_video_features_sha256':hashlib.sha256(a.video_features.read_bytes()).hexdigest(),
                  'input_manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest()}
    a.definition.parent.mkdir(parents=True, exist_ok=True)
    a.definition.write_text(json.dumps(definition, ensure_ascii=False, indent=2), encoding='utf-8')
    specs = discover(a.raw_dir, manifest)
    all_rows, summaries, failures = [], [], []
    a.log.parent.mkdir(parents=True, exist_ok=True)
    with a.log.open('w', encoding='utf-8') as log:
        log.write(json.dumps({'event':'start','definition':definition,'sessions_selected':len(specs)},ensure_ascii=False)+'\n')
        for spec in specs:
            try:
                rows, summary = summarize_session(spec, a, log)
                all_rows.extend(rows)
                summaries.append(summary)
            except Exception:
                error = {'session_id':spec['session_id'],'source_csv':str(spec['path']), 'error':traceback.format_exc()}
                failures.append(error)
                log.write(json.dumps(error,ensure_ascii=False)+'\n');log.flush()
                print(json.dumps({'session_id':spec['session_id'],'failed':True}),flush=True)
        if not all_rows:
            raise ValueError('No actual timestamped sessions could be parsed')
        labels = pd.DataFrame(all_rows)
        training = merge_existing_video(labels, video, manifest)
        for path, frame in [(a.output,labels),(a.training_output,training)]:
            path.parent.mkdir(parents=True,exist_ok=True)
            frame.to_csv(path,index=False,encoding='utf-8-sig')
        overall = {l:distribution(labels,l) for l in ['fast_camera_speed','high_camera_acceleration']}
        paired = {l:distribution(training,l) for l in overall}
        summary = {'created_utc':datetime.now(timezone.utc).isoformat(),'local_sessions_discovered':len(specs),
                   'successful_sessions':len(summaries),'failed_sessions':failures,
                   'matrix_sessions':sum(s['has_matrix_columns'] for s in summaries),
                   'label_windows':len(labels),'video_paired_windows':len(training),
                   'video_paired_sessions':int(training.session_id.nunique()),'video_paired_games':int(training.game_name.nunique()),
                   'all_local_distribution':overall,'video_paired_distribution':paired,'sessions':summaries,
                   'definition_sha256':hashlib.sha256(a.definition.read_bytes()).hexdigest()}
        a.report.parent.mkdir(parents=True,exist_ok=True)
        a.report.with_suffix('.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        lines=['# 按论文规则生成的风险因素标签：分布与对齐','','来源：VR.net arXiv:2306.03381v1，§4.1、PDF p10；https://arxiv.org/abs/2306.03381v1。',
               '**探索性，非最终结论。标签是按论文规则生成的风险因素标签，不是人体验证阈值。**','',
               '- fast_camera_speed：1秒内观察到的有效速度最大值 >3 m/s。',
               '- high_camera_acceleration：1秒内观察到的有效向量加速度最大值 >0.7 m/s²。',
               '- 速度/加速度使用真实非均匀采样间隔及导数中点时间；该聚合和实现细节为本轮固定实现，论文没有给全。',
               '- 接受1源坐标单位=1米的工程假设，没有物理校准；未分离用户头动，目标为总相机运动因素。',
               '- 缺失/非法矩阵、切换、gap>.5秒、位移>10源单位处分段，不跨边界求导；这两项是固定质量门槛，不是新的风险切点。',
               '- 每标签至少1个有效导数样本才给0/1；0只表示观察到的有效样本未超过界限，不保证原生稀疏采样间不存在峰值。',
               '- 多轴旋转跳过；没有改3或0.7以平衡类别；未读取FMS/SSQ作为标签。','',
               f'本地去重session：{len(specs)}；处理成功：{len(summaries)}；有矩阵：{summary["matrix_sessions"]}；失败：{len(failures)}。',
               f'生成片段：{len(labels)}；成功对齐已有视频特征：{len(training)}，{summary["video_paired_sessions"]} session、{summary["video_paired_games"]}游戏。',
               '','| 范围 | 标签 | 正例 | 负例 | unknown |','|---|---|---:|---:|---:|']
        for scope,dist in [('所有本地片段',overall),('实际视频训练片段',paired)]:
            for label,d in dist.items():lines.append(f'| {scope} | {label} | {d["positive"]} | {d["negative"]} | {d["unknown"]} |')
        lines+=['','## 训练片段按游戏分布','','| 游戏 | 标签 | 正例 | 负例 | unknown |','|---|---|---:|---:|---:|']
        for game,group in training.groupby('game_name'):
            for label in overall:
                d=distribution(group,label);lines.append(f'| {game} | {label} | {d["positive"]} | {d["negative"]} | {d["unknown"]} |')
        lines+=['','## 对齐与特征来源','','通过clip_id/session_id/game_name三键one-to-one合并；验证start/end与原CSV时钟origin一致。没有将session最大标签复制到片段。',
                '训练保留全部已有RGB光流/复杂度/亮度/单应性特征；矩阵速度/加速度最大值保留在矩阵中供审计，训练主模型明确排除这些定义标签的直接输入。',
                '现有视频片段仅120个、2游戏、6session，原JPG每片段只有2–8个不同帧并重采样成30帧；不能当30Hz连续捕获。',
                '','## 完整审计','','参数在paper_rule_definition.json于读取标签前写入；逐session记录及错误全文见paper_rule_label_generation_log.jsonl；分布JSON含全部session范围与质量事件。',
                '旧v4冻结配置不覆盖；本轮为用户明确授权的论文操作规则复现。原PROTOCOL中的人体低/中/高风险AUC门控不适用于这个不同标签目标。','']
        if failures:
            lines+=['## 失败session','']+[f'- {f["session_id"]}：详见日志完整错误。' for f in failures]
        a.report.parent.mkdir(parents=True,exist_ok=True)
        a.report.write_text('\n'.join(lines),encoding='utf-8')
        log.write(json.dumps({'event':'complete','label_windows':len(labels),'video_paired_windows':len(training),
                             'all_local_distribution':overall,'video_paired_distribution':paired},ensure_ascii=False)+'\n')
        print(json.dumps({'complete':True,'label_windows':len(labels),'video_windows':len(training),
                          'video_distribution':paired,'failed_sessions':len(failures)},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
