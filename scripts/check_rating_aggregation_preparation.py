"""Independent provenance and cohort checks for the authorized comparison."""
import json
from pathlib import Path
import pandas as pd
import numpy as np
from extract_param_rating_windows import sha256

ROOT=Path(__file__).resolve().parents[1]
TAG='rating_aggregation_comparison_2026-10-10'
cfg=ROOT/'configs'/f'{TAG}.json'
core=ROOT/'scripts/prepare_rating_aggregation_comparison.py'
base=pd.read_csv(ROOT/'features/param_rating_local_all_2026-10-10.csv',encoding='utf-8-sig')
base=base[base.ms_rating.notna() & base.window_has_any_feature.astype(str).str.lower().isin(['true','1'])]
target=base.drop_duplicates('rating_event_id').set_index('rating_event_id')
inventory=json.loads((ROOT/'results/param_rating_local_all_2026-10-10/source_inventory.json').read_text(encoding='utf-8'))
sources={r['session_id']:r for r in inventory['sources']}
out=ROOT/'features'/TAG
events=pd.read_csv(out/'events.csv',encoding='utf-8-sig').set_index('rating_event_id')
assert events.index.is_unique and set(events.index)==set(target.index)
assert len(events)==1039
assert np.allclose(events.ms_rating,target.loc[events.index].ms_rating,rtol=0,atol=0)
assert np.allclose(events.rating_time_sec,target.loc[events.index].rating_time_sec,rtol=0,atol=1e-5)
metas=[]
for sid in events.session_id.unique():
    p=out/'by_session'/f'{sid}.csv';j=p.with_suffix('.json')
    m=json.loads(j.read_text(encoding='utf-8'));r=sources[sid]
    assert m['status']=='complete' and m['output_sha256']==sha256(p)
    identity=m['identity']
    assert identity['script_sha256']==sha256(core) and identity['config_sha256']==sha256(cfg)
    for key in ['source_locator','known_sha256','size_bytes','game_name','participant_id']:
        assert identity[key]==r[key]
    part=pd.read_csv(p,encoding='utf-8-sig')
    assert set(part.rating_event_id)==set(target[target.session_id==sid].index)
    assert m['rows']==r['raw_rows_expected'] and m['real_reports']==r['rating_events_expected']
    prior=json.loads((ROOT/'features/param_rating_local_all_by_session_2026-10-10'/f'{sid}.json').read_text(encoding='utf-8'))
    for key in ['valid_camera_rows','valid_object_rows','position_jumps','derivative_interval_resets','rotation_rate_over_flag']:
        assert m['counters'].get(key,0)==prior['counters'].get(key,0),(sid,key,m['counters'].get(key),prior['counters'].get(key))
    metas.append(m)
for name,group in base.groupby('rating_event_id',sort=False):
    row=events.loc[name]
    for feature in ['camera_speed','camera_acceleration','camera_rotation_rate','fov_horizontal','fov_vertical','object_count']:
        n=group[feature+'_n'];valid=n>0
        if not valid.any():assert pd.isna(row['nearest_'+feature+'_mean']);continue
        mu=np.average(group.loc[valid,feature+'_mean'],weights=n[valid])
        assert np.isclose(mu,row['nearest_'+feature+'_mean'],rtol=1e-12,atol=1e-12)
        assert row['nearest_'+feature+'_n']==n.sum()
        assert np.isclose(group.loc[valid,feature+'_max'].max(),row['nearest_'+feature+'_max'],rtol=1e-12,atol=1e-12)
eligible=events.common_eligible.astype(str).str.lower().isin(['true','1'])
features=json.loads(cfg.read_text(encoding='utf-8'))['methods']
coverage=[]
for name,spec in features.items():
    frame=events.loc[eligible,spec['feature_columns']]
    coverage.append(dict(method=name,requested_features=len(frame.columns),any_finite_events=int(np.isfinite(frame.to_numpy()).any(axis=1).sum()),
                         all_missing_features=[c for c in frame if frame[c].isna().all()],constant_features=[c for c in frame if frame[c].nunique(dropna=True)<=1]))
summary=dict(status='passed',target_and_prepared_events=len(events),common_events=int(eligible.sum()),excluded_events=int((~eligible).sum()),
             prepared_sessions=int(events.session_id.nunique()),prepared_games=int(events.game_name.nunique()),
             common_sessions=int(events.loc[eligible].session_id.nunique()),common_games=int(events.loc[eligible].game_name.nunique()),
             raw_rows=sum(m['rows'] for m in metas),real_source_reports=sum(m['real_reports'] for m in metas),
             common_label_distribution=events.loc[eligible].ms_rating.value_counts().sort_index().to_dict(),feature_coverage=coverage,
             caveat='non-empty trace does not certify synchronized modalities, physical units, or pure design settings')
result=ROOT/'results'/TAG/'preparation_check.json'
result.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='feature_coverage'},ensure_ascii=False))
