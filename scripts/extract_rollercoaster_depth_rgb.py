"""Actual Epic recordings: strictly historical RGB + native depth-code statistics.

Depth values are JPEG channel codes, not metres. Native .zfp motion is retained
but never guessed/decoded. Farneback below is inferred RGB flow, explicitly named
rgb_flow_ and never described as engine motion vectors.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sys
import time
import traceback
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from extract_fms_rgb_pilot import STATIC, MOTION, entropy, summarize
from extract_lightgbm_features import frame_metrics, resize_frame
from rollercoaster_native_io import load_scores, read_frame_index, resolve_session_dir

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/vrnet/rollercoaster_raw'
PREFIX = 'rollercoaster_depth_rgb'
DEPTH = ['red_code_mean', 'red_code_std', 'red_code_p10', 'red_code_p90',
         'red_code_zero_fraction', 'red_code_full_fraction', 'red_code_gradient_mean', 'red_code_change_per_sec']
FEATURES = [f'rgb_{k}_{s}' for k in STATIC + MOTION for s in ('mean', 'max', 'std')]
FEATURES += [f'native_depth_{k}_{s}' for k in DEPTH for s in ('mean', 'max', 'std')]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


class Cache:
    def __init__(self):
        self.items = OrderedDict()

    def read(self, rgb, depth):
        key = (rgb, depth)
        if key in self.items:
            self.items.move_to_end(key)
            return self.items[key]
        payloads = [Path(p).read_bytes() for p in key]
        images = [cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) for b in payloads]
        if any(im is None for im in images):
            raise ValueError(f'JPEG decode failure: {key}')
        if images[0].shape != images[1].shape:
            raise ValueError(f'RGB/depth dimensions differ: {[im.shape for im in images]}')
        rgb_small = resize_frame(images[0], 224)
        gray, static = frame_metrics(rgb_small)
        # Red-channel raw code; no near/far calibration or metre interpretation.
        code = cv2.resize(images[1][..., 2].astype(np.float32) / 255., (229, 120), interpolation=cv2.INTER_AREA)
        dx = cv2.Sobel(code, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(code, cv2.CV_32F, 0, 1, ksize=3)
        depth_metrics = dict(red_code_mean=float(code.mean()), red_code_std=float(code.std()),
                             red_code_p10=float(np.percentile(code, 10)), red_code_p90=float(np.percentile(code, 90)),
                             red_code_zero_fraction=float((code < 1 / 255).mean()),
                             red_code_full_fraction=float((code > 254 / 255).mean()),
                             red_code_gradient_mean=float(np.hypot(dx, dy).mean()))
        hashes = [hashlib.sha256(b).hexdigest() for b in payloads]
        value = gray, static, code, depth_metrics, hashes
        self.items[key] = value
        if len(self.items) > 256:
            self.items.popitem(last=False)
        return value


def extract_window(index, event_us, cache):
    start_us = event_us - 3_000_000
    result = dict(window_sec=3, status='skipped', reason='', source_frame_count=0,
                  sampled_frame_count=0, start_gap_sec=None, end_gap_sec=None, max_gap_sec=None,
                  frame_ids='[]', frame_timestamps_ms='[]', rgb_depth_sha256='[]',
                  **{f: math.nan for f in FEATURES})
    rows = index.loc[(index.timestamp_ms * 1000 >= start_us) & (index.timestamp_ms * 1000 < event_us)]
    result['source_frame_count'] = len(rows)
    if start_us < int(index.timestamp_ms.iloc[0]) * 1000:
        result['reason'] = 'window_before_recording_start'
        return result
    if len(rows) < 12:
        result['reason'] = 'fewer_than_12_actual_frames'
        return result
    if rows[['rgb_path', 'depth_path']].isna().any(axis=None):
        result['reason'] = 'missing_rgb_or_depth_in_window'
        return result
    # Subtract the window origin before conversion; no float epoch precision loss.
    relative = (rows.timestamp_ms.to_numpy(np.int64) * 1000 - start_us).astype(np.float64)
    quality = dict(start_gap_sec=float(relative[0] / 1e6), end_gap_sec=float((3e6 - relative[-1]) / 1e6),
                   max_gap_sec=float(np.diff(relative).max() / 1e6))
    result.update(quality)
    if quality['start_gap_sec'] > .25 or quality['end_gap_sec'] > .25 or quality['max_gap_sec'] > .75:
        result['reason'] = 'insufficient_temporal_coverage'
        return result
    targets = np.linspace(relative[0], relative[-1], 16)
    chosen = []
    for target in targets:
        right = int(np.searchsorted(relative, target))
        options = [i for i in (right - 1, right) if 0 <= i < len(relative)]
        chosen.append(min(options, key=lambda i: abs(relative[i] - target)))
    selected = rows.iloc[sorted(set(chosen))]
    result['sampled_frame_count'] = len(selected)
    result['frame_ids'] = json.dumps(selected.frame_id.tolist())
    result['frame_timestamps_ms'] = json.dumps(selected.timestamp_ms.tolist())
    if len(selected) < 12:
        result['reason'] = 'fewer_than_12_distinct_sampled_frames'
        return result
    values = {k: [] for k in STATIC + MOTION + DEPTH}
    previous = None
    hashes = []
    for frame in selected.itertuples(index=False):
        gray, static, code, depth, digests = cache.read(frame.rgb_path, frame.depth_path)
        hashes.append(dict(frame_id=int(frame.frame_id), rgb=digests[0], depth=digests[1]))
        for key, value in {**static, **depth}.items():
            if key in values:
                values[key].append(value)
        if previous is not None:
            old_gray, old_code, old_ms = previous
            dt = (int(frame.timestamp_ms) - old_ms) / 1000.
            if not 0 < dt <= .75 or gray.shape != old_gray.shape or code.shape != old_code.shape:
                result['reason'] = 'invalid_sampled_image_pair'
                return result
            values['red_code_change_per_sec'].append(float(np.abs(code - old_code).mean() / dt))
            flow = cv2.calcOpticalFlowFarneback(old_gray, gray, None, .5, 3, 15, 3, 5, 1.2, 0)
            mag, theta = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            rate = mag.astype(np.float64) / (dt * math.hypot(*gray.shape))
            h, w = gray.shape
            center = np.zeros(gray.shape, bool)
            center[h // 4:3 * h // 4, w // 4:3 * w // 4] = True
            for key, value in dict(flow_rate_mean_diag_s=rate.mean(), flow_rate_p95_diag_s=np.percentile(rate, 95),
                                   flow_rate_peripheral_diag_s=rate[~center].mean(), flow_rate_center_diag_s=rate[center].mean()).items():
                values[key].append(float(value))
            valid = mag > 1e-3
            weights, angles = mag[valid].astype(float), theta[valid].astype(float)
            if len(angles) and weights.sum() > 0:
                values['flow_direction_concentration'].append(float(np.hypot((weights * np.cos(angles)).sum(), (weights * np.sin(angles)).sum()) / weights.sum()))
                values['flow_direction_entropy_bits'].append(entropy((angles + math.pi) % (2 * math.pi) - math.pi))
        previous = gray, code, int(frame.timestamp_ms)
    for key, vals in values.items():
        prefix = 'native_depth_' if key in DEPTH else 'rgb_'
        for stat, value in zip(('mean', 'max', 'std'), summarize(vals)):
            result[prefix + key + '_' + stat] = value
    result.update(status='ok', rgb_depth_sha256=json.dumps(hashes))
    return result


def main():
    if not (ROOT / 'PROTOCOL.md').is_file():
        raise FileNotFoundError('PROTOCOL.md absent: experiment prohibited')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(ROOT / f'results/{PREFIX}_extraction_log.txt', mode='a', encoding='utf-8')])
    cv2.setNumThreads(1)
    cv2.setRNGSeed(42)
    manifest = json.loads((ROOT / 'results/rollercoaster_download_manifest.json').read_text(encoding='utf-8'))
    archives = [r for r in manifest if r.get('status') == 'success' and r['name'] != 'SelfReportData.zip']
    if len(archives) != 5:
        raise ValueError(f'Expected all five publicly listed Epic ZIPs; successful manifest count={len(archives)}')
    scores_dir = RAW / 'self_reports/SelfReportData'
    selections = []
    for archive in archives:
        session_dir = resolve_session_dir(archive['destination'])
        token = re.search(r'VRLOG-\d+', session_dir.name).group()
        candidates = list(scores_dir.glob(f'*{token} Epic Roller Coasters.csv'))
        if len(candidates) != 1:
            raise ValueError(f'Expected exactly one score file for {token}, got {len(candidates)}')
        participant = re.match(r'(\d+) ', candidates[0].name).group(1)
        if sha(candidates[0]) != sha(session_dir / 'voice.csv'):
            raise ValueError(f'Independent and in-recording score files differ: {token}')
        selections.append(dict(session_id=token, participant_id=participant, session_dir=str(session_dir),
                               score_path=str(candidates[0]), score_sha256=sha(candidates[0]), archive_sha256=archive['sha256']))
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), client_date='2026-10-10', seed=42,
                selection='all five public Epic data ZIPs; not the unknown .7z, not the whole dataset or paper471 set',
                inputs='RGB plus red-channel native depth JPEG codes, full-image including existing masks; no metre interpretation',
                motion='native ZFP retained but decoder parameters unknown; not used. rgb_flow uses inferred Farneback only',
                window='[report_us-3000000,report_us); integer timestamp alignment, type5 Unixms *1000 vs score Unixus',
                sampling='at most16 nearest distinct actual frames uniformly across available window time',
                quality='>=12 actual and sampled frames, both RGB+depth every source frame, boundary gaps<=.25s, max gap<=.75s',
                labels='all unaltered raw records retained with auto/manual; no filling/binning/dedup/corrections',
                evaluation='manual-only primary inferred from DataCard descriptives, auto-only sensitivity; whole participant leaveout',
                features=FEATURES, selected=selections,
                protocol_deviation='User-authorized exploratory raw-rating regression; original low/medium/high AUC gate not evaluated')
    (ROOT / f'audit/{PREFIX}_extraction_definition.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    all_events, audit = [], []
    begun = time.perf_counter()
    for source in sorted(selections, key=lambda s: int(s['participant_id'])):
        before = time.perf_counter()
        logging.info('READ %s P%s', source['session_id'], source['participant_id'])
        index = read_frame_index(source['session_dir'])
        index_path = ROOT / f'features/{PREFIX}_{source["session_id"]}_frame_index.csv'
        index.to_csv(index_path, index=False, encoding='utf-8-sig')
        reports = load_scores(source['score_path'])
        cache = Cache()
        origin_us = int(index.timestamp_ms.iloc[0]) * 1000
        for number, report in enumerate(reports.itertuples(index=False), 1):
            event = dict(event_id=f'{source["session_id"]}__score_row{number}', session_id=source['session_id'],
                         participant_id=source['participant_id'], game_name='Epic_Roller_Coasters',
                         score_row=number, event_timestamp_us=int(report.timestamp_us),
                         event_time_sec=(int(report.timestamp_us) - origin_us) / 1e6,
                         elapsed_sec=(int(report.timestamp_us) - origin_us) / 1e6,
                         fms_raw=report.fms_raw, rating_source=report.rating_source)
            try:
                result = extract_window(index, int(report.timestamp_us), cache)
            except Exception:
                error = traceback.format_exc()
                logging.error('%s %s\n%s', source['session_id'], number, error)
                result = dict(window_sec=3, status='skipped', reason=error, **{f: math.nan for f in FEATURES})
            all_events.append({**event, **result})
        session_events = pd.DataFrame([r for r in all_events if r['session_id'] == source['session_id']])
        record = dict(**source, raw_scores=len(reports), frame_records=len(index),
                      rgb_files=int(index.rgb_path.notna().sum()), depth_files=int(index.depth_path.notna().sum()),
                      motion_files=int(index.motion_path.notna().sum()),
                      duration_sec=(int(index.timestamp_ms.iloc[-1]) - int(index.timestamp_ms.iloc[0])) / 1000.,
                      frame_interval_median_ms=float(np.median(np.diff(index.timestamp_ms))),
                      source_counts=reports.rating_source.value_counts().to_dict(),
                      window_status_counts=session_events.status.value_counts().to_dict(),
                      skipped_reasons=session_events.loc[session_events.status.ne('ok'), 'reason'].value_counts().to_dict(),
                      eligible_by_source=session_events.loc[session_events.status.eq('ok'), 'rating_source'].value_counts().to_dict(),
                      wall_sec=time.perf_counter() - before)
        audit.append(record)
        pd.DataFrame(all_events).to_csv(ROOT / f'features/{PREFIX}_event_features.csv', index=False, encoding='utf-8-sig')
        (ROOT / f'results/{PREFIX}_session_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
        logging.info('FINISH %s frames=%s scores=%s eligible=%s seconds=%.1f', source['session_id'], len(index), len(reports), record['eligible_by_source'], record['wall_sec'])
    matrix = pd.DataFrame(all_events)
    stats = matrix.loc[matrix.status.eq('ok'), FEATURES].agg(['count', 'mean', 'std', 'min', 'max']).T
    stats['missing_rate'] = matrix.loc[matrix.status.eq('ok'), FEATURES].isna().mean()
    stats.to_csv(ROOT / f'features/{PREFIX}_feature_statistics.csv', encoding='utf-8-sig', index_label='feature')
    lines = ['# 过山车 RGB + 原生深度编码提取', '', '**探索性流程验证，非最终结论。**', '',
             f'- 实际原始记录 {len(matrix)} 条，合格窗口 {int(matrix.status.eq("ok").sum())} 个，{matrix.participant_id.nunique()} 位参与者。',
             '- 根据真实帧时间戳截取评分前3秒。JPG帧号每3帧保存一次，但记录时钟不固定30fps，未按帧号推算时间。',
             '- RGB统计含Farneback推断光流，不是引擎原生motion-flow。',
             '- 深度只使用native JPEG红通道编码统计；遮罩保留、非线性/near-far校准未知，不声称深度单位为米。',
             '- 原生motion.zfp全部保留，因headerless流缺少producer类型、shape、压缩模式配置尚未解码。',
             '- 原始评分及auto/manual来源全部保留，不自动合并紧邻记录；训练按source分别进行。', '',
             '| 参与者 | 录制 | RGB/depth/motion | 时长秒 | 原始评分 | 合格manual/auto | 跳过原因 |',
             '|---|---|---|---:|---:|---|---|']
    for a in audit:
        lines.append(f'| P{a["participant_id"]} | {a["session_id"]} | {a["rgb_files"]}/{a["depth_files"]}/{a["motion_files"]} | {a["duration_sec"]:.3f} | {a["raw_scores"]} | {a["eligible_by_source"]} | {a["skipped_reasons"]} |')
    lines += ['', '## 复现边界', '', '- 公共目录的5个ZIP已全部读取并校验CRC；这不等于官方DataCard所述15sessions或原论文471评分。',
              '- 内部日志有少量尾帧没有图片；缺失窗口按预先固定质量门槛跳过，不补图。',
              '- 所有图片统计来自实际字节，SHA256与实际采样帧号/时间可在事件CSV核对。',
              '- 这些视频受头部方向、驾驶路径和暴露阶段影响；不能直接当独立于路径的Unity静态设计数据。', '']
    (ROOT / f'results/{PREFIX}_extraction_report.md').write_text('\n'.join(lines), encoding='utf-8')
    logging.info('DONE %s records=%s eligible=%s wall_seconds=%.1f', PREFIX, len(matrix), int(matrix.status.eq('ok').sum()), time.perf_counter() - begun)


if __name__ == '__main__':
    main()
