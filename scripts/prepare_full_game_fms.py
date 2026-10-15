#!/usr/bin/env python3
"""Fetch ONE complete verified VR.net game and retain every CSV/JPG.

Fresh download is attempted first. If transport fails, a pre-existing complete
archive can be used only with an explicitly logged provenance and size check.
No original archive is deleted. Only the task-owned temporary ZIP is removed
after successful full extraction and reference checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import traceback
from types import SimpleNamespace
import zipfile

import pandas as pd

from prepare_vrnet_game import download, extract, write_log, sha256, within


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir', type=Path, required=True)
    parser.add_argument('--game', default='City_Parkour')
    args = parser.parse_args()
    project = args.project_dir.resolve()
    if not (project / 'PROTOCOL.md').is_file():
        raise FileNotFoundError('PROTOCOL.md missing')
    game = args.game
    if not game or any(c in game for c in '/\\:') or game in ('.', '..'):
        raise ValueError('Unsafe game name')
    raw = project / 'data/raw/vrnet'
    results = project / 'results'
    audit = project / 'audit'
    results.mkdir(exist_ok=True)
    audit.mkdir(exist_ok=True)
    log = results / 'fms_full_game_download_log.txt'
    log.write_text('', encoding='utf-8')
    write_log(log, 'COMMAND ' + json.dumps(sys.argv, ensure_ascii=False))
    write_log(log, 'Web tool could not open the public Drive root/file view; attempting the direct public archive transport instead. Web result: URL is not accessible via this tool.')
    index = pd.read_csv(raw / 'main_file.csv')
    sessions = pd.read_csv(raw / 'main_log.csv')
    expected = sorted(sessions.loc[sessions['name'].str.endswith('_' + game), 'name'].astype(str))
    count = int(index.loc[index.game_name.eq(game), 'no_of_sessions'].iloc[0])
    if len(expected) != count:
        raise ValueError(f'Index count mismatch: main_file={count}, main_log={len(expected)}')
    sources = pd.read_csv(results / 'vrnet_sampled_sessions_v2.csv')
    metadata = sources.loc[sources.game_name.eq(game), ['archive_name', 'drive_file_id', 'archive_size_bytes']].drop_duplicates()
    if len(metadata) != 1:
        raise ValueError('No unique previously verified official Drive archive metadata')
    meta = metadata.iloc[0].to_dict()
    record = dict(game_name=game, archive_name=str(meta['archive_name']),
                  drive_file_id=str(meta['drive_file_id']), archive_size_bytes=int(meta['archive_size_bytes']), sessions=expected)
    limits = dict(max_zip_bytes=5_000_000_000, min_free_bytes=10_000_000_000)
    work = raw / '.full_game_fms_work'
    destination = raw / 'full_game'
    work.mkdir(exist_ok=True)
    destination.mkdir(exist_ok=True)
    temp_archive = work / record['archive_name']
    # Freeze game choice before reading the previously unused FMS labels.
    plan = dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(), game=record,
                source_url='https://drive.google.com/drive/folders/1LLo4OE83Zeso8MkEXSvW9VrdjiqqlFRR',
                metadata_source='results/vrnet_sampled_sessions_v2.csv; results/vrnet_drive_structure.md (2026-10-07)',
                choice_reason='Complete index-matching 10-session archive under 5GB; prior five-session audit observed scores 1-5. Choice not based on the previous fitted model metrics.',
                selection_caveat='Game and five sessions were inspected in the prior pilot. This is exploratory follow-up, not an independent confirmation set.',
                all_sessions=True, limits=limits,
                extraction_root=str(destination / game),
                modeling='Fixed historical 3s/30s RGB event windows, session/group holdout, unchanged small regressors and guessing baselines; no score filling, no video neural model')
    (audit / 'fms_full_game_acquisition_definition.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    (results / 'fms_full_game_selection.md').write_text(
        '# 完整单游戏选择\n\n'
        f'- 游戏：{game}\n- 官方本地索引：{count} 个 session；本次全部纳入。\n'
        f'- 官方 Drive 文件 ID：{record["drive_file_id"]}\n'
        f'- 已核验目录元数据大小：{record["archive_size_bytes"]:,} bytes。\n'
        '- 来源：https://drive.google.com/drive/folders/1LLo4OE83Zeso8MkEXSvW9VrdjiqqlFRR\n'
        '- 原始元数据：results/vrnet_drive_structure.md；results/vrnet_sampled_sessions_v2.csv。\n'
        '- 选择理由：索引与完整包一致，大小符合限制；已有5次游玩的评分覆盖1–5。未按模型表现选择。\n'
        '- 本游戏曾用于上一轮探索，因此不是独立确认数据集。\n'
        '- 本次先尝试重新下载完整包；如网络失败，明确记录并使用原有完整包，不伪称新下载成功。\n\n'
        '## 全部 session\n\n' + '\n'.join('- ' + s for s in expected) + '\n', encoding='utf-8')
    namespace = SimpleNamespace(work_dir=work, frames_root=destination, csv_root=destination, log=log)
    acquisition = 'fresh_download'
    try:
        archive = download(record, namespace, limits)
    except Exception:
        write_log(log, 'Fresh download unavailable; full failure already recorded. Checking existing local complete archive.\n' + traceback.format_exc())
        original = raw / 'samples' / record['archive_name']
        if not original.is_file() or original.stat().st_size != record['archive_size_bytes']:
            raise RuntimeError('Neither fresh download nor matching-size existing archive available')
        if shutil.disk_usage(work).free - original.stat().st_size < limits['min_free_bytes']:
            raise OSError('Insufficient space for task-owned fallback archive copy')
        original_hash = sha256(original)
        if temp_archive.exists():
            raise RuntimeError('Task temporary ZIP unexpectedly exists after failed fresh download')
        shutil.copyfile(original, temp_archive)
        if sha256(temp_archive) != original_hash:
            raise ValueError('Fallback copy hash differs')
        archive = temp_archive
        acquisition = 'existing_complete_archive_after_failed_fresh_download'
        write_log(log, f'EXISTING ARCHIVE REUSE source={original}; sha256={original_hash}; no new download success claimed')
    try:
        archive_hash = sha256(archive)
        inventory = []
        with zipfile.ZipFile(archive) as z:
            csv_members = [i for i in z.infolist() if i.filename.endswith('/data_file.csv')]
            observed = sorted(PurePosixPath(i.filename).parent.name for i in csv_members)
            if observed != expected:
                raise ValueError(f'ZIP session set differs from index: missing={set(expected)-set(observed)}, extra={set(observed)-set(expected)}')
            unknown_files = [i.filename for i in z.infolist() if not i.is_dir()
                             and not i.filename.endswith('/data_file.csv')
                             and not i.filename.lower().endswith(('.jpg', '.jpeg'))]
            if unknown_files:
                raise ValueError(f'Additional non-CSV/JPG archive files need preservation: {unknown_files[:20]}')
            for i in z.infolist():
                inventory.append(dict(member=i.filename, bytes=i.file_size, compressed_bytes=i.compress_size, crc32=f'{i.CRC:08x}', directory=i.is_dir()))
            write_log(log, f'FULL ARCHIVE VERIFIED sessions={len(observed)}, JPG={sum(i.filename.lower().endswith((".jpg", ".jpeg")) for i in z.infolist())}, all_file_bytes={sum(i.file_size for i in z.infolist())}, sha256={archive_hash}')
        pd.DataFrame(inventory).to_csv(results / 'fms_full_game_archive_inventory.csv', index=False, encoding='utf-8-sig')
        outputs = extract(record, archive, namespace, limits)
        summary = dict(game_name=game, acquisition=acquisition, verified_index_session_count=count,
                       archive_bytes=record['archive_size_bytes'], archive_sha256=archive_hash,
                       archive_deleted_after_verified_extraction=not archive.exists(),
                       complete_game=sorted(r['session_id'] for r in outputs) == expected and all(r['complete_by_csv_references'] for r in outputs),
                       retained_jpg_count=sum(r['jpg_count'] for r in outputs), retained_csv_count=len(outputs), sessions=outputs)
        (results / 'fms_full_game_download_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        write_log(log, 'COMPLETE GAME READY ' + json.dumps({k:v for k,v in summary.items() if k != 'sessions'}, ensure_ascii=False))
        return 0
    except Exception:
        write_log(log, 'FULL GAME PREPARATION FAILED\n' + traceback.format_exc())
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
