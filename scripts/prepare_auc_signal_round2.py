#!/usr/bin/env python3
"""Frozen, local-only second-round classification preparation. No label expansion."""
from __future__ import annotations
import argparse, json, logging, math, time
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from extract_param_rating_local_all import LocalCsvSource, read_source_csv
from extract_param_rating_windows import MEASURES, MissingCameraObservation, finite_number, full_fov, parse_pose, sha256
from parse_vrnet_matrices import choose_camera, literal_list, numeric_matrix_list, timestamp_scale
from prepare_rating_aggregation_comparison import USER_MEASURES, vector, feature_cols

ROOT = Path(__file__).resolve().parents[1]
TAG = 'auc_signal_round2_2026-10-10'
CFG = ROOT/'configs'/f'{TAG}.json'
OUT = ROOT/'features'/TAG
RES = ROOT/'results'/TAG
OLD = ROOT/'features/rating_aggregation_comparison_2026-10-10/events.csv'
COHORT = ROOT/'results/rating_aggregation_comparison_2026-10-10/cohort.csv'
INVENTORY = ROOT/'results/param_rating_local_all_2026-10-10/source_inventory.json'
SCENE = ('camera_speed','camera_acceleration','camera_rotation_rate','object_count')
HEAD = ('head_speed','head_angular_speed','head_pose_speed','head_pose_turn_rate')
CONTROL = USER_MEASURES[2:]
ROBUST = SCENE+HEAD+CONTROL
DOSE = ('camera_speed','camera_rotation_rate','head_pose_speed','head_pose_turn_rate')

def robust_columns(measures):
    cols=[f'robust{w}_{m}_{s}' for w in (5,30,60) for m in measures for s in ('mean','std','median','p90')]
    cols += [f'{prefix}_{m}' for prefix in ('slope60','trend5vs25') for m in measures]
    for m in DOSE:
        if m in measures: cols += [f'cumulative_{m}_integral',f'cumulative_{m}_mean']
    return cols+['past60_fov_horizontal_mean','past60_fov_vertical_mean','exposure_elapsed_seconds']

def make_config():
    scene=sum([feature_cols(f'past{w}_') for w in (5,30,60)],[])+['exposure_elapsed_seconds']
    head=scene[:-1]+sum([feature_cols(f'past{w}_',USER_MEASURES[:2]) for w in (5,30,60)],[])+scene[-1:]
    groups={'scene_multiscale':scene,'scene_head_multiscale':head,
            'robust_scene_head':robust_columns(SCENE+HEAD),
            'robust_joint':robust_columns(ROBUST),'elapsed_only':['exposure_elapsed_seconds']}
    specs={}
    for name,cols in groups.items():
        logs=[c for c in cols if 'fov_' not in c and not c.startswith(('slope','trend'))]
        specs[name]=dict(input='events.csv',feature_columns=cols,log1p_columns=logs,
            description=name,temporal_policy='strictly before real report; [t-W,t); no rating values as inputs')
    candidates=dict(logistic=[dict(C=c,class_balance=b) for c in (.1,1.,10.) for b in ('none','balanced')],
        lightgbm=[dict(num_leaves=n,n_estimators=t,min_child_samples=m,class_balance='balanced') for n in (3,7) for t in (100,200) for m in (20,40)],
        extra_trees=[dict(min_samples_leaf=5,max_features=.5,class_balance='balanced'),dict(min_samples_leaf=15,max_features=1.,class_balance='balanced')])
    return dict(status='fixed_before_round2_training',seed=42,random_seed=42,
        methods=specs,models=['logistic','lightgbm','extra_trees'],binary_threshold=2.0,
        fixed_candidate_grids=candidates,class_threshold=.5,inner_splits=3,expected_cohort_events=1024,
        protocol_file=str(ROOT/'PROTOCOL.md'),preparation_manifest=str(OUT/'preparation_manifest.json'),
        double_holdout=dict(enabled=True,methods=list(specs),models=['logistic','lightgbm','extra_trees']),
        target='observed MS_rating > 2; fixed original labels, not continuous severity prediction',
        target_auc=0.70,cohort_reference_csv=str(COHORT),cohort_reference_sha256=sha256(COHORT),
        previous_events_csv=str(OLD),previous_events_sha256=sha256(OLD),
        source_inventory_sha256=sha256(INVENTORY),protocol_sha256=sha256(ROOT/'PROTOCOL.md'),
        input_scope='only 86 already-local canonical matrix sessions; unchanged 1024 real events',
        history_windows_seconds=[5,30,60],inner_folds=3,
        camera_rules=dict(maximum_gap_seconds=1.,jump_source_units=10.,rotation_flag_degrees_per_second=720.,fov_rounding_degrees=.001),
        temporal_policy='Every support interval uses endpoints observed strictly before report; window intersections clipped, invalid gaps excluded, missing never treated as zero',
        interval_policy='camera velocity/rotation: exact derivative interval; acceleration: velocity midpoint interval; raw head/controller/object count: mean of two endpoint values on dt<=1s interval',
        head_pose_policy='position difference in unverified source units; normalized four-component quaternion angular distance 2 acos(abs(dot)), independent of internal ordering; norm must be [0.9,1.1]',
        robust_policy='duration-weighted mean/std/median/P90 over valid intervals, 60s weighted OLS slope and last5s minus preceding25s mean; cumulative integral/observed-time mean from session start',
        excluded_inputs=['current/previous/future score','rating frequency/distance','game/session/participant IDs','absolute time','gaze','face','button masks','QC coverage as predictor'],
        preprocessing='predefined log1p nonnegative columns; training-only median imputation, constant removal, logistic scaling',
        fit_weights='equal total base session weights; optional class factors computed from training weighted class totals only',
        evaluation='outer Leave-One-Game-Out; nested 3-fold grouped-game candidate selection; inner objective mean per-game AUC; fixed threshold .5',
        double_holdout_policy='repeat all methods, purge test participants from outer training and validation participants from inner training',
        selection_caveat='previous rounds have seen these games; all results remain exploratory even if target reached',
        probability_caveat='class-weighted scores may be uncalibrated; report Brier/calibration rather than assume probabilities',
        feature_rationale='direct classification matches AUC target; robust summaries reduce extreme leverage; past motion dose/trends represent temporal exposure hypotheses, not established causal effects',
        sources=['https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html',
                 'https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html',
                 'https://arxiv.org/html/2306.03381v1'])

def extract_intervals(record,cfg,logger):
    """One streaming read. Each signal has its own clock; no missing-zero filling."""
    fields=['timestamp','camera_name','p_matrix','v_matrix','m_matrix','MS_rating',
            'head_pos','head_dir','head_vel','head_angvel','IndexTrigger','HandTrigger','Thumbstick']
    fields=[c for c in fields if c in record['columns']]
    intervals={m:[] for m in ROBUST}; reports=[]; counts=Counter(); errors=[]
    origin=scale=selected=prev_camera=prev_velocity=prev_headpos=prev_headquat=prev_thumb=None
    previous={}; number=0; started=time.perf_counter(); last=started
    rules=cfg['camera_rules']; maxgap=rules['maximum_gap_seconds']
    def append(m,a,b,v,observed):
        if 0<b-a<=maxgap and math.isfinite(v): intervals[m].append((a,b,float(v),observed))
    def raw_interval(m,t,v):
        prior=previous.get(m)
        if prior is not None: append(m,prior[0],t,(prior[1]+v)/2,t)
        previous[m]=(t,v)
    for chunk in read_source_csv(LocalCsvSource(record),usecols=fields,dtype=str,chunksize=16,keep_default_na=False,encoding='utf-8-sig'):
        for row in chunk.to_dict('records'):
            number+=1; raw=finite_number(row.get('timestamp'))
            if raw is None:
                counts['invalid_timestamp']+=1;previous.clear()
                prev_camera=prev_velocity=prev_headpos=prev_headquat=prev_thumb=None;continue
            if origin is None: scale,unit=timestamp_scale(raw); origin=raw*scale
            t=raw*scale-origin
            if t<0:
                counts['before_origin']+=1;previous.clear()
                prev_camera=prev_velocity=prev_headpos=prev_headquat=prev_thumb=None;continue
            rating=finite_number(row.get('MS_rating'))
            if rating is not None: reports.append(dict(rating_event_id=f"modern:{record['session_id']}:record:{number}",rating_time_sec=t,ms_rating=rating))
            if row.get('m_matrix','').strip():
                try:
                    raw_interval('object_count',t,float(len(numeric_matrix_list(row['m_matrix']))));counts['valid_object_rows']+=1
                except Exception as e:
                    counts['invalid_object_rows']+=1;previous.pop('object_count',None)
                    if len(errors)<5:errors.append(f'{number}:object:{e}')
            if row.get('v_matrix','').strip():
                try:
                    names=literal_list(row.get('camera_name',''));views=numeric_matrix_list(row['v_matrix'])
                    if not len(views):raise MissingCameraObservation('empty camera observation')
                    if not names or len(names)!=len(views):raise ValueError('camera/view length mismatch')
                    name=selected if selected is not None else str(names[choose_camera(names)])
                    ix=[i for i,n in enumerate(names) if str(n)==name]
                    if len(ix)!=1:raise ValueError('fixed camera missing or repeated')
                    pos,rot=parse_pose(views[ix[0]]);selected=name;counts['valid_camera_rows']+=1
                    if prev_camera is not None:
                        dt=t-prev_camera[0];disp=pos-prev_camera[1]
                        if not 0<dt<=maxgap:counts['derivative_interval_resets']+=1;prev_velocity=None
                        elif np.linalg.norm(disp)>rules['jump_source_units']:counts['position_jumps']+=1;prev_velocity=None
                        else:
                            vel=disp/dt;mid=(t+prev_camera[0])/2
                            append('camera_speed',prev_camera[0],t,np.linalg.norm(vel),t)
                            rate=float(np.degrees(Rotation.from_matrix(prev_camera[2].T@rot).magnitude())/dt)
                            append('camera_rotation_rate',prev_camera[0],t,rate,t)
                            if rate>rules['rotation_flag_degrees_per_second']:counts['rotation_rate_over_flag']+=1
                            if prev_velocity is not None:
                                dmid=mid-prev_velocity[0]
                                if 0<dmid<=maxgap:append('camera_acceleration',prev_velocity[0],mid,np.linalg.norm((vel-prev_velocity[1])/dmid),t)
                            prev_velocity=(mid,vel)
                    prev_camera=(t,pos,rot)
                except MissingCameraObservation:counts['unsampled_camera_rows']+=1
                except Exception as e:
                    counts['invalid_camera_rows']+=1;prev_camera=prev_velocity=None
                    if len(errors)<5:errors.append(f'{number}:camera:{e}')
            for source,m,length in [('head_vel','head_speed',3),('head_angvel','head_angular_speed',3),('IndexTrigger','index_trigger_level',2),('HandTrigger','hand_trigger_level',2)]:
                v=vector(row.get(source,''),length)
                if v is not None:raw_interval(m,t,float(np.linalg.norm(v)) if length==3 else float(v.mean()))
                elif row.get(source,'').strip():previous.pop(m,None);counts['invalid_'+source]+=1
            thumb=vector(row.get('Thumbstick',''),4)
            if thumb is not None:
                raw_interval('thumbstick_magnitude',t,float(np.linalg.norm(thumb.reshape(2,2),axis=1).mean()))
                if prev_thumb is not None:
                    dt=t-prev_thumb[0]
                    if 0<dt<=maxgap:append('thumbstick_change_rate',prev_thumb[0],t,np.linalg.norm((thumb-prev_thumb[1]).reshape(2,2),axis=1).mean()/dt,t)
                prev_thumb=(t,thumb)
            elif row.get('Thumbstick','').strip():prev_thumb=None;previous.pop('thumbstick_magnitude',None)
            hp=vector(row.get('head_pos',''),3)
            if hp is not None:
                if prev_headpos is not None:
                    dt=t-prev_headpos[0];disp=float(np.linalg.norm(hp-prev_headpos[1]))
                    if 0<dt<=maxgap and disp<=rules['jump_source_units']:append('head_pose_speed',prev_headpos[0],t,disp/dt,t)
                    elif disp>rules['jump_source_units']:counts['head_position_jumps']+=1
                prev_headpos=(t,hp)
            elif row.get('head_pos','').strip():prev_headpos=None;counts['invalid_head_pos']+=1
            hq=vector(row.get('head_dir',''),4)
            if hq is not None and .9<=np.linalg.norm(hq)<=1.1:
                hq=hq/np.linalg.norm(hq)
                if prev_headquat is not None:
                    dt=t-prev_headquat[0]
                    if 0<dt<=maxgap:
                        angle=float(np.degrees(2*np.arccos(np.clip(abs(np.dot(hq,prev_headquat[1])),0,1))))
                        append('head_pose_turn_rate',prev_headquat[0],t,angle/dt,t)
                prev_headquat=(t,hq)
            elif row.get('head_dir','').strip():prev_headquat=None;counts['invalid_head_quaternion']+=1
        if time.perf_counter()-last>25:
            logger.info('PROGRESS %s rows=%s seconds=%.1f',record['session_id'],number,time.perf_counter()-started);last=time.perf_counter()
    if number!=record['raw_rows_expected'] or len(reports)!=record['rating_events_expected']:raise ValueError('raw row/report count mismatch')
    arrays={m:np.asarray(a,dtype=float).reshape(-1,4) for m,a in intervals.items()}
    # Reject duplicate/overlapping support intervals rather than double-count time.
    for m,a in arrays.items():
        if len(a)>1 and np.any(a[1:,0]<a[:-1,1]-1e-6):raise ValueError(f'overlapping/nonmonotonic intervals: {m}')
    meta=dict(rows=number,real_reports=len(reports),origin_epoch=origin,timestamp_unit=unit,
              selected_camera=selected,counters=dict(counts),errors=errors,seconds=time.perf_counter()-started,
              valid_intervals={m:len(a) for m,a in arrays.items()})
    return arrays,{r['rating_event_id']:r for r in reports},meta

def select_support(a,left,t):
    if not len(a):return np.empty(0),np.empty(0),np.empty(0)
    # No derivative whose final endpoint was observed on/after report is used.
    keep=(a[:,3]<t)&(a[:,1]>left)&(a[:,0]<t)
    z=a[keep];lo=np.maximum(z[:,0],left);hi=np.minimum(z[:,1],t);w=hi-lo
    valid=w>0
    return z[valid,2],w[valid],(lo[valid]+hi[valid])/2

def weighted_stats(v,w):
    if not len(v):return dict(mean=np.nan,std=np.nan,median=np.nan,p90=np.nan)
    mean=float(np.average(v,weights=w));order=np.argsort(v,kind='stable');cum=np.cumsum(w[order])/w.sum()
    return dict(mean=mean,std=float(np.sqrt(np.average((v-mean)**2,weights=w))),
                median=float(v[order][np.searchsorted(cum,.5,'left')]),p90=float(v[order][np.searchsorted(cum,.9,'left')]))

def summarize_intervals(arrays,t):
    row={};qc={}
    for m,a in arrays.items():
        for width in (5,30,60):
            v,w,mid=select_support(a,max(0,t-width),t)
            row.update({f'robust{width}_{m}_{s}':val for s,val in weighted_stats(v,w).items()})
            qc[f'coverage{width}_{m}']=float(w.sum()/min(t,width)) if t>0 else np.nan
        v,w,mid=select_support(a,max(0,t-60),t)
        if len(v)>1:
            x=mid-float(np.average(mid,weights=w));den=float(np.sum(w*x*x))
            slope=float(np.sum(w*x*(v-np.average(v,weights=w)))/den) if den>1e-12 else np.nan
        else:slope=np.nan
        row[f'slope60_{m}']=slope
        recent=weighted_stats(*select_support(a,max(0,t-5),t)[:2])['mean']
        prior=a[(a[:,3]<t-5)] if len(a) else a
        earlier=weighted_stats(*select_support(prior,max(0,t-30),t-5)[:2])['mean']
        row[f'trend5vs25_{m}']=recent-earlier
        if m in DOSE:
            v,w,mid=select_support(a,0,t);duration=float(w.sum())
            row[f'cumulative_{m}_integral']=float(np.sum(v*w)) if duration else np.nan
            row[f'cumulative_{m}_mean']=float(np.average(v,weights=w)) if duration else np.nan
            qc[f'cumulative_{m}_valid_duration']=duration
            qc[f'cumulative_{m}_coverage']=duration/t if t>0 else np.nan
    return row,qc

def targets():
    if sha256(COHORT)!=json.loads(CFG.read_text(encoding='utf-8'))['cohort_reference_sha256']:raise ValueError('frozen cohort changed')
    reference=pd.read_csv(COHORT,encoding='utf-8-sig');old=pd.read_csv(OLD,encoding='utf-8-sig')
    d=old[old.rating_event_id.isin(reference.rating_event_id)].copy()
    if len(d)!=1024 or set(d.rating_event_id)!=set(reference.rating_event_id):raise ValueError('cohort mismatch')
    return d

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--init',action='store_true');ap.add_argument('--finalize',action='store_true')
    ap.add_argument('--partition-count',type=int,default=1);ap.add_argument('--partition-index',type=int,default=0);args=ap.parse_args()
    if not (ROOT/'PROTOCOL.md').is_file():raise SystemExit('PROTOCOL.md missing')
    for p in (OUT,OUT/'by_session',RES):p.mkdir(parents=True,exist_ok=True)
    if args.init:
        if CFG.exists():raise SystemExit('Frozen round2 config exists; do not overwrite')
        CFG.write_text(json.dumps(make_config(),ensure_ascii=False,indent=2),encoding='utf-8');print(CFG);return
    cfg=json.loads(CFG.read_text(encoding='utf-8'))
    if sha256(OLD)!=cfg['previous_events_sha256'] or sha256(INVENTORY)!=cfg['source_inventory_sha256']:raise ValueError('input hash mismatch')
    data=targets();records=[r for r in json.loads(INVENTORY.read_text(encoding='utf-8'))['sources'] if r['session_id'] in set(data.session_id)]
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',handlers=[logging.FileHandler(RES/f'preparation_part{args.partition_index}.log',encoding='utf-8'),logging.StreamHandler()])
    logger=logging.getLogger('auc_round2');identity_base=dict(script_sha256=sha256(Path(__file__)),config_sha256=sha256(CFG))
    if args.finalize:
        parts=[];qcparts=[];metas=[]
        for rec in records:
            p=OUT/'by_session'/f"{rec['session_id']}.csv";j=p.with_suffix('.json');q=p.with_name(p.stem+'_qc.csv')
            if not all(x.is_file() for x in (p,j,q)):raise ValueError(f'missing output {p}')
            meta=json.loads(j.read_text(encoding='utf-8'))
            if meta['status']!='complete' or meta['identity']['script_sha256']!=identity_base['script_sha256'] or meta['identity']['config_sha256']!=identity_base['config_sha256'] or meta['output_sha256']!=sha256(p) or meta['qc_sha256']!=sha256(q):raise ValueError('cache identity/hash mismatch')
            parts.append(pd.read_csv(p,encoding='utf-8-sig'));qcparts.append(pd.read_csv(q,encoding='utf-8-sig'));metas.append(meta)
        additions=pd.concat(parts,ignore_index=True)
        if not additions.rating_event_id.is_unique or set(additions.rating_event_id)!=set(data.rating_event_id):raise ValueError('prepared event IDs differ from frozen cohort')
        duplicate=set(additions.columns)&set(data.columns)-{'rating_event_id'}
        if duplicate:raise ValueError(f'duplicate output feature names {duplicate}')
        final=data.merge(additions,on='rating_event_id',validate='one_to_one').sort_values(['game_name','session_id','rating_time_sec'])
        final['binary_label']=(final.ms_rating>cfg['binary_threshold']).astype(int)
        for method,spec in cfg['methods'].items():
            missing=set(spec['feature_columns'])-set(final.columns)
            if missing:raise ValueError(f'{method} missing features {missing}')
        final.to_csv(OUT/'events.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
        pd.concat(qcparts,ignore_index=True).to_csv(OUT/'support_quality.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
        manifest=dict(status='complete',events=len(final),sessions=int(final.session_id.nunique()),games=int(final.game_name.nunique()),
                      class_counts=final.binary_label.value_counts().sort_index().to_dict(),config_sha256=sha256(CFG),
                      script_sha256=identity_base['script_sha256'],events_sha256=sha256(OUT/'events.csv'),
                      support_quality_sha256=sha256(OUT/'support_quality.csv'),cohort_sha256=sha256(COHORT),sessions_metadata=metas)
        (OUT/'preparation_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        logger.info('FINALIZED events=%s sessions=%s games=%s',len(final),final.session_id.nunique(),final.game_name.nunique());return
    for i,rec in enumerate(records):
        if i%args.partition_count!=args.partition_index:continue
        sid=rec['session_id'];p=OUT/'by_session'/f'{sid}.csv';j=p.with_suffix('.json');q=p.with_name(p.stem+'_qc.csv')
        identity=dict(identity_base,source_locator=rec['source_locator'],known_sha256=rec['known_sha256'],size_bytes=rec['size_bytes'])
        saved=json.loads(j.read_text(encoding='utf-8')) if j.exists() else {}
        if saved.get('status')=='complete' and saved.get('identity')==identity and p.exists() and q.exists() and saved['output_sha256']==sha256(p) and saved['qc_sha256']==sha256(q):logger.info('RESUME %s',sid);continue
        logger.info('START %s bytes=%s',sid,rec['size_bytes'])
        arrays,reports,meta=extract_intervals(rec,cfg,logger);rows=[];quality=[]
        for event in data[data.session_id==sid].to_dict('records'):
            raw=reports[event['rating_event_id']]
            if raw['ms_rating']!=event['ms_rating'] or abs(raw['rating_time_sec']-event['rating_time_sec'])>1e-5:raise ValueError('raw report value/time differs')
            values,qc=summarize_intervals(arrays,raw['rating_time_sec'])
            rows.append(dict(rating_event_id=event['rating_event_id'],**values));quality.append(dict(rating_event_id=event['rating_event_id'],session_id=sid,**qc))
        pd.DataFrame(rows).to_csv(p,index=False,encoding='utf-8-sig',float_format='%.17g')
        pd.DataFrame(quality).to_csv(q,index=False,encoding='utf-8-sig',float_format='%.17g')
        meta.update(status='complete',identity=identity,session_id=sid,events=len(rows),output_sha256=sha256(p),qc_sha256=sha256(q))
        j.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8');logger.info('DONE %s events=%s seconds=%.1f',sid,len(rows),meta['seconds'])

if __name__=='__main__':main()
