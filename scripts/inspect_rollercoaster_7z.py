"""Inspect/extract the small, verified Epic metadata 7z; never run payloads."""
import json
import shutil
import subprocess
from pathlib import Path, PurePosixPath

from prepare_vrnet_game import sha256, within, write_log

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data/raw/vrnet/rollercoaster_raw'
ARCHIVE = BASE / '.download_work/Epic RollerCoaster.7z'
DEST = BASE / 'derived_csv'
LOG = ROOT / 'results/rollercoaster_download_log.txt'


def command(args):
    write_log(LOG, 'COMMAND ' + json.dumps(args, ensure_ascii=False))
    result = subprocess.run(args, capture_output=True, text=True, encoding='utf-8', errors='replace')
    write_log(LOG, f'COMMAND RESULT exit={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}')
    if result.returncode:
        raise RuntimeError(f'tar failed: {result.stderr}')
    return result.stdout


def main():
    tar = shutil.which('tar')
    if not tar:
        raise RuntimeError('tar is unavailable; 7z contents unknown')
    listing = command([tar, '-tf', str(ARCHIVE)])
    members = listing.splitlines()
    files = []
    for member in members:
        p = PurePosixPath(member)
        if p.is_absolute() or '..' in p.parts or any(':' in part for part in p.parts):
            raise ValueError(f'Unsafe archive member {member!r}')
        if not member.endswith('/'):
            if p.suffix != '.csv' or p.name not in {'camera.csv', 'control.csv', 'light.csv', 'object.csv', 'pose.csv'}:
                raise ValueError(f'Unrecognized content, not extracted or executed: {member}')
            files.append(member)
    if len(files) != 25:
        raise ValueError(f'Unexpected metadata file count: {len(files)}')
    verbose = command([tar, '-tvf', str(ARCHIVE)])
    total = sum(int(line.split()[4]) for line in verbose.splitlines() if line.startswith('-'))
    if shutil.disk_usage(BASE).free - total < 10_000_000_000:
        raise OSError('7z metadata extraction disk guard')
    if DEST.exists() and any(DEST.iterdir()):
        raise FileExistsError('Preserved existing metadata; extract destination must be empty')
    DEST.mkdir(parents=True, exist_ok=True)
    command([tar, '-xf', str(ARCHIVE), '-C', str(DEST)])
    info = dict(source='https://drive.google.com/file/d/1k1H9M4IbHrM7iO-yuMJNL1hB8MohQZRK/view',
                archive_bytes=ARCHIVE.stat().st_size, archive_sha256=sha256(ARCHIVE),
                content='25 author-derived metadata CSVs, five per the same five recordings; no extra participants/ratings/video',
                expected_uncompressed_bytes=total, files=[])
    for member in files:
        output = within(DEST.joinpath(*PurePosixPath(member).parts), DEST)
        if not output.is_file():
            raise FileNotFoundError(f'Listed CSV not extracted: {output}')
        info['files'].append(dict(path=str(output), bytes=output.stat().st_size, sha256=sha256(output)))
    if sum(f['bytes'] for f in info['files']) != total:
        raise ValueError('7z extracted byte count mismatch')
    info['status'] = 'success'
    within(ARCHIVE, BASE / '.download_work').unlink()
    info['archive_deleted_after_extraction'] = True
    (ROOT / 'results/rollercoaster_7z_inspection.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    write_log(LOG, f'7Z METADATA SUCCESS files=25 uncompressed_bytes={total}; archive deleted; no executable content')


if __name__ == '__main__':
    main()
