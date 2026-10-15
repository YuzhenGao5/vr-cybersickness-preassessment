#!/usr/bin/env python3
"""Run the user-authorized local check sequentially, log full failures and stop.

This is a bounded video plumbing check, not training permission for TSM. Frozen
unknown labels are respected; model/SHAP gates do not constitute fitted models.
"""
import argparse
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download-summary', type=Path, required=True)
    p.add_argument('--max-clips-per-session', type=int, default=20)
    p.add_argument('--log', type=Path, default=Path('results/local_preparation_execution_log.txt'))
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    log_path = (root / args.log).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', encoding='utf-8') as log:
        def write(text):
            log.write(text + '\n'); log.flush()
            print(text, flush=True)
        write('Started UTC ' + datetime.now(timezone.utc).isoformat())
        write('Python: ' + sys.executable + '; full commands/output logged; no TSM or git commit')
        try:
            specs = json.loads(args.download_summary.read_text(encoding='utf-8-sig'))
            if not all(s['complete_by_csv_references'] for s in specs):
                raise ValueError('a downloaded session is not complete by its CSV references')
            mars = root / 'data/raw/vrnet/samples_full/11_2_Mars_Miners'
            specs.append({'session_id': '11_2_Mars_Miners', 'game_name': 'Mars_Miners',
                          'csv_path': str(mars / 'data_file.csv'), 'frames_dir': str(mars)})
            for s in specs:
                if not Path(s['csv_path']).is_file() or not Path(s['frames_dir']).is_dir():
                    raise FileNotFoundError(f"missing local source: {s['session_id']}")
            sessions_json = root / 'results/local_clip_sessions.json'
            sessions_json.write_text(json.dumps(specs, indent=2, ensure_ascii=False), encoding='utf-8')
            commands = [
                ['scripts/make_clips.py', '--sessions-json', str(sessions_json),
                 '--output-dir', 'data/local_video_clips_v3', '--manifest', 'features/video_clips_manifest.csv',
                 '--timestamp-unit', 'milliseconds', '--window-sec', '1', '--fps', '30',
                 '--max-gap-sec', '.5', '--max-clips-per-session', str(args.max_clips_per_session),
                 '--log', 'results/make_clips_local_log.json'],
                ['scripts/generate_clip_labels.py', '--clips-manifest', 'features/video_clips_manifest.csv',
                 '--config', 'configs/risk_factor_labels_v3.yaml', '--output', 'features/clip_risk_labels_v3.csv',
                 '--report', 'results/clip_label_generation_v3.md'],
                ['scripts/split_data.py', '--manifest', 'features/video_clips_manifest.csv',
                 '--output', 'features/logo_splits_v3.json'],
                ['scripts/extract_lightgbm_features.py', '--clips-manifest', 'features/video_clips_manifest.csv'],
                ['scripts/align_lightgbm_labels.py'],
                ['scripts/train_lightgbm_baseline.py'],
                ['scripts/shap_lightgbm_baseline.py'],
            ]
            for index, command in enumerate(commands, 1):
                cmd = [sys.executable, *command]
                write(f'\nSTEP {index}: ' + json.dumps(cmd, ensure_ascii=False))
                done = subprocess.run(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, encoding='utf-8', errors='replace',
                                      env={**__import__('os').environ, 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8'})
                write(done.stdout)
                write(f'exit_code={done.returncode}')
                if done.returncode:
                    write('STOP: failure; no downstream calls performed')
                    return done.returncode
                if 'extract_lightgbm_features.py' in command[0]:
                    features = pd.read_csv(root / 'features/lightgbm_features.csv')
                    failed = features.extraction_status.eq('failed')
                    if failed.any():
                        raise RuntimeError(f'{failed.sum()} clips failed feature extraction; stopping before alignment/training')
            write('COMPLETE: actual video plumbing and all label/model/SHAP gates executed; metrics depend on supported labels.')
        except Exception:
            write('STOP full traceback:\n' + traceback.format_exc())
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
