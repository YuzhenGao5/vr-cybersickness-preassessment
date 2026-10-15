#!/usr/bin/env python3
"""Sequential Drive download, bounded extraction, CSV/JPG retention and ZIP deletion.

Requires requests and pandas. All IDs/sizes/session names must be supplied in a
verified manifest. Extraction never writes arbitrary archive paths. Partial
downloads are retained only on interruption for an explicitly logged resume.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import pandas as pd
import requests
from download_vrnet_game_csvs_v3 import request_download


def safe_name(s):
    if not s or s in (".", "..") or any(c in s for c in '/\\:'):
        raise ValueError(f"unsafe name: {s!r}")
    return s


def within(path, root):
    p, r = Path(path).resolve(), Path(root).resolve()
    p.relative_to(r)
    return p


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_log(path, message):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(f"[{datetime.now(timezone.utc).isoformat()}] {message}\n")
    print(message, flush=True)


def download(game, args, limits):
    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    dest = within(work / safe_name(game['archive_name']), work)
    partial = within(dest.with_suffix('.zip.part'), work)
    expected = int(game['archive_size_bytes'])
    if expected > limits['max_zip_bytes']:
        raise ValueError(f"ZIP exceeds limit: {expected} > {limits['max_zip_bytes']}")
    if dest.exists():
        if dest.stat().st_size != expected:
            raise ValueError('existing ZIP size differs from verified manifest')
        write_log(args.log, f"verified existing ZIP {dest}, bytes={expected}")
        return dest
    for attempt in range(1, 4):
        offset = partial.stat().st_size if partial.exists() else 0
        free = shutil.disk_usage(work).free
        if free - (expected - offset) < limits['min_free_bytes']:
            raise OSError(f"disk guard: free={free}, remaining ZIP={expected-offset}, floor={limits['min_free_bytes']}")
        write_log(args.log, f"download attempt={attempt}, game={game['game_name']}, id={game['drive_file_id']}, offset={offset}, expected_bytes={expected}, free_bytes={free}")
        started = time.monotonic()
        try:
            with requests.Session() as session:
                response = request_download(session, game['drive_file_id'], offset)
                with response:
                    if 'text/html' in response.headers.get('Content-Type', '').lower():
                        raise ValueError('Drive returned HTML instead of archive: ' + response.text[:2000])
                    if offset and response.status_code != 206:
                        raise ValueError('server did not honor Range resume; partial retained, restart must be explicit')
                    if offset and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise ValueError('Content-Range does not match resume offset')
                    length = response.headers.get('Content-Length')
                    if length and int(length) + offset != expected:
                        raise ValueError(f'content length differs: {length} + {offset} != {expected}')
                    transferred, last = offset, time.monotonic()
                    with partial.open('ab' if offset else 'wb') as out:
                        for block in response.iter_content(1024 * 1024):
                            if not block:
                                continue
                            if transferred + len(block) > min(expected, limits['max_zip_bytes']):
                                raise ValueError('transfer exceeds verified size or 5GB cap')
                            if shutil.disk_usage(work).free - len(block) < limits['min_free_bytes']:
                                raise OSError('disk free space below configured floor; stopping')
                            out.write(block)
                            transferred += len(block)
                            if time.monotonic() - last > 15:
                                write_log(args.log, f"progress game={game['game_name']}, bytes={transferred}/{expected}")
                                last = time.monotonic()
            if partial.stat().st_size != expected:
                raise ValueError(f'incomplete ZIP: {partial.stat().st_size} != {expected}')
            partial.replace(dest)
            write_log(args.log, f"DOWNLOAD SUCCESS game={game['game_name']}, bytes={expected}, seconds={time.monotonic()-started:.3f}, sha256={sha256(dest)}")
            return dest
        except Exception:
            write_log(args.log, 'DOWNLOAD FAILURE full traceback:\n' + traceback.format_exc())
            if attempt == 3 or 'disk' in traceback.format_exc().lower():
                raise
    raise RuntimeError('unreachable')


def extract(game, archive, args, limits):
    if archive.stat().st_size > limits['max_zip_bytes']:
        raise ValueError('existing ZIP exceeds 5GB limit')
    frames_root = Path(args.frames_root).resolve()
    csv_root = Path(args.csv_root).resolve()
    frames_root.mkdir(parents=True, exist_ok=True)
    csv_root.mkdir(parents=True, exist_ok=True)
    if frames_root.drive != csv_root.drive:
        raise ValueError('frames and CSV roots must use same filesystem for combined space guard')
    game_name = safe_name(game['game_name'])
    selected = {safe_name(x) for x in game['sessions']}
    outputs = []
    started = time.monotonic()
    with zipfile.ZipFile(archive) as z:
        targets, counts = [], {s: {'csv': 0, 'jpg': 0} for s in selected}
        for member in z.infolist():
            parts = PurePosixPath(member.filename.replace('\\', '/')).parts
            if member.is_dir():
                continue
            if member.filename.startswith(('/', '\\')) or '..' in parts:
                raise ValueError(f'unsafe ZIP member: {member.filename}')
            matches = selected.intersection(parts)
            if not matches:
                continue
            if len(matches) != 1:
                raise ValueError(f'ambiguous session member {member.filename}')
            s = next(iter(matches))
            filename = safe_name(parts[-1])
            if filename == 'data_file.csv':
                dest = within(csv_root / game_name / s / filename, csv_root)
                kind = 'csv'
            elif filename.lower().endswith(('.jpg', '.jpeg')):
                dest = within(frames_root / game_name / s / 'video' / filename, frames_root)
                kind = 'jpg'
            else:
                continue
            counts[s][kind] += 1
            targets.append((member, dest))
        if len({d for _, d in targets}) != len(targets):
            raise ValueError('duplicate destination ZIP paths')
        for s, count in counts.items():
            if count['csv'] != 1 or count['jpg'] < 2:
                raise ValueError(f'insufficient archive contents {s}: {count}')
        needed = sum(m.file_size for m, d in targets if not d.exists())
        free = shutil.disk_usage(frames_root).free
        write_log(args.log, f"EXTRACT PLAN game={game_name}, selected={counts}, uncompressed_bytes={sum(m.file_size for m,d in targets)}, new_bytes={needed}, free_bytes={free}")
        if free - needed < limits['min_free_bytes']:
            raise OSError(f'extraction disk guard: free={free}, needed={needed}, floor={limits["min_free_bytes"]}')
        for member, dest in targets:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                if dest.stat().st_size != member.file_size:
                    raise ValueError(f'existing destination differs: {dest}')
                # Verify existing bytes against ZIP, not merely existence/size.
                with z.open(member) as src:
                    h = hashlib.sha256()
                    for b in iter(lambda: src.read(4 * 1024 * 1024), b''):
                        h.update(b)
                if sha256(dest) != h.hexdigest():
                    raise ValueError(f'existing destination hash differs: {dest}')
                continue
            temp = within(dest.with_suffix(dest.suffix + '.extracting'), dest.parent)
            try:
                with z.open(member) as src, temp.open('wb') as out:
                    shutil.copyfileobj(src, out, 4 * 1024 * 1024)
                if temp.stat().st_size != member.file_size:
                    raise ValueError('incomplete extracted member')
                temp.replace(dest)
            finally:
                if temp.exists():
                    temp.unlink()
        for s in sorted(selected):
            csv_path = csv_root / game_name / s / 'data_file.csv'
            df = pd.read_csv(csv_path, usecols=['frame', 'timestamp', 'video'], encoding='utf-8-sig')
            refs = df['video'].dropna().astype(str)
            refs = refs[refs.str.strip().ne('')]
            if refs.empty:
                raise ValueError(f'{s}: CSV has no video references; completeness cannot be verified')
            expected = []
            for ref in refs:
                p = PurePosixPath(ref.replace('\\', '/'))
                name = p.name if p.suffix else p.name + '.jpg'
                expected.append(frames_root / game_name / s / 'video' / safe_name(name))
            missing = [str(p) for p in expected if not p.is_file()]
            report = {'game_name': game_name, 'session_id': s, 'csv_path': str(csv_path),
                      'frames_dir': str(frames_root / game_name / s), 'csv_rows': len(df),
                      'jpg_count': counts[s]['jpg'], 'referenced_jpg_count': len(expected),
                      'missing_references': len(missing), 'complete_by_csv_references': not missing,
                      'csv_sha256': sha256(csv_path)}
            outputs.append(report)
            write_log(args.log, f"SESSION VERIFIED {json.dumps(report, ensure_ascii=False)}")
            if missing:
                raise ValueError(f'missing JPG references: {missing[:10]}')
    # Only the explicitly supplied archive is deleted after verified extraction.
    archive = within(archive, Path(args.work_dir))
    archive.unlink()
    write_log(args.log, f"EXTRACT SUCCESS game={game_name}, seconds={time.monotonic()-started:.3f}; ZIP deleted={archive}; free_bytes={shutil.disk_usage(frames_root).free}")
    return outputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--mode', choices=['download', 'extract', 'both'], default='both')
    p.add_argument('--work-dir', required=True, type=Path)
    p.add_argument('--frames-root', required=True, type=Path)
    p.add_argument('--csv-root', required=True, type=Path)
    p.add_argument('--log', required=True, type=Path)
    p.add_argument('--summary', required=True, type=Path)
    args = p.parse_args()
    write_log(args.log, '# VR.net local/server preparation; command=' + json.dumps(sys.argv, ensure_ascii=False))
    manifest = json.loads(args.manifest.read_text(encoding='utf-8-sig'))
    limits = {'max_zip_bytes': min(int(manifest.get('max_zip_bytes', 5000000000)), 5000000000),
              'min_free_bytes': max(int(manifest.get('min_free_bytes', 50000000000)), 50000000000)}
    if args.mode == 'download' and len(manifest['games']) != 1:
        raise ValueError('download-only requires exactly one game; use both for sequential multi-game processing')
    reports = []
    for game in manifest['games']:
        archive = Path(args.work_dir).resolve() / safe_name(game['archive_name'])
        try:
            if args.mode in ('download', 'both'):
                archive = download(game, args, limits)
            if args.mode in ('extract', 'both'):
                reports.extend(extract(game, archive, args, limits))
            elif len(manifest['games']) > 1:
                raise ValueError('download-only mode permits ONE game; use both to process/delete before next')
        except Exception:
            write_log(args.log, 'STOP; full traceback:\n' + traceback.format_exc())
            # Clean only the failed ZIP/partial files belonging to this verified game.
            for path in (archive, archive.with_suffix('.zip.part')):
                path = within(path, args.work_dir)
                if path.exists():
                    path.unlink()
                    write_log(args.log, f'failed archive removed: {path}')
            return 1
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
