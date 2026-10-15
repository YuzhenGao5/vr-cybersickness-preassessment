#!/usr/bin/env python3
"""Frozen outer Leave-One-Game-Out folds, inner whole-session validation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    df = pd.read_csv(a.manifest, encoding='utf-8-sig')
    if df.clip_id.duplicated().any():
        raise ValueError('duplicate clip IDs')
    if df.groupby('session_id').game_name.nunique().gt(1).any():
        raise ValueError('session belongs to multiple games')
    games = sorted(df.game_name.unique())
    folds = []
    for game in games:
        test = df[df.game_name.eq(game)]
        development = df[~df.game_name.eq(game)]
        sessions = sorted(development.session_id.unique())
        if len(sessions) < 2:
            folds.append({'test_game': game, 'status': 'blocked', 'reason': 'fewer than two development sessions',
                          'train': [], 'val': [], 'test': test.clip_id.tolist()})
            continue
        rng = np.random.default_rng(a.seed)
        count = max(1, min(len(sessions)-1, int(np.ceil(.2 * len(sessions)))))
        val_sessions = set(rng.choice(sessions, size=count, replace=False).tolist())
        train = development[~development.session_id.isin(val_sessions)]
        val = development[development.session_id.isin(val_sessions)]
        assert not set(train.session_id) & set(val.session_id)
        assert not set(train.session_id) & set(test.session_id)
        folds.append({'test_game': game, 'status': 'ready', 'train': train.clip_id.tolist(),
                      'val': val.clip_id.tolist(), 'test': test.clip_id.tolist(),
                      'train_sessions': sorted(train.session_id.unique()), 'val_sessions': sorted(val.session_id.unique())})
    report = {'version': '1.0', 'seed': a.seed, 'source_sha256': hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
              'outer': 'leave_one_game_out', 'inner': 'whole_session_holdout', 'n_games': len(games),
              'n_sessions': int(df.session_id.nunique()), 'folds': folds,
              'labels': 'fit no thresholds/normalizers on outer test; class support checked separately'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'games': len(games), 'folds': len(folds), 'ready': sum(f['status']=='ready' for f in folds)}))


if __name__ == '__main__':
    main()
