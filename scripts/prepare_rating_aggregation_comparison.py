#!/usr/bin/env python3
"""Local-only, real-report aggregation comparison; no downloads or score filling."""
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

USER_MEASURES = ('head_speed', 'head_angular_speed', 'index_trigger_level',
                 'hand_trigger_level', 'thumbstick_magnitude', 'thumbstick_change_rate')
STATS = ('mean', 'max', 'std')
ROOT = Path(__file__).resolve().parents[1]
TAG = 'rating_aggregation_comparison_2026-10-10'
CFG_PATH = ROOT / 'configs' / (TAG + '.json')
OUT = ROOT / 'features' / TAG
RES = ROOT / 'results' / TAG

def feature_cols(prefix, measures=MEASURES):
    return [f'{prefix}{m}_{s}' for m in measures for s in STATS]

def make_config():
    multi_scene = sum([feature_cols(f'past{w}_') for w in (5, 30, 60)], [])
    multi_head = sum([feature_cols(f'past{w}_', USER_MEASURES[:2]) for w in (5, 30, 60)], [])
    multi_user = sum([feature_cols(f'past{w}_', USER_MEASURES) for w in (5, 30, 60)], [])
    specs = {
        'nearest_mean': [f'nearest_{m}_mean' for m in MEASURES],
        'nearest_pooled': feature_cols('nearest_'),
        'past5_scene': feature_cols('past5_'),
        'past30_scene': feature_cols('past30_'),
        'past60_scene': feature_cols('past60_'),
        'past30_mean_only': [f'past30_{m}_mean' for m in MEASURES],
        'multi_scene': multi_scene + ['exposure_elapsed_seconds'],
        'multi_user': multi_user + ['exposure_elapsed_seconds'],
        'multi_joint': multi_scene + multi_user + ['exposure_elapsed_seconds'],
        'multi_joint_noelapsed': multi_scene + multi_user,
        'multi_scene_head': multi_scene + multi_head + ['exposure_elapsed_seconds'],
        'elapsed_only': ['exposure_elapsed_seconds'],
    }
    return dict(status='fixed_before_comparison_training', seed=42, random_seed=42,
        methods={k:dict(input='events.csv', feature_columns=v) for k,v in specs.items()},
        models=['lightgbm','ridge'], ridge_alpha=10.0,
        model_parameters=dict(n_estimators=100,num_leaves=7,max_depth=3,min_child_samples=20,
          learning_rate=.05,reg_lambda=1.,random_state=42,n_jobs=2,verbosity=-1),
        validation='Leave-One-Game-Out; same common rating events for every method',
        preprocessing='training-fold median imputation and constant removal; Ridge train-fold scaling',
        fit_weights='equal sessions, equal rating events within each session',
        primary_metrics=['MAE','RMSE','Spearman','Pearson'],
        supplementary_auc='actual MS_rating > 2, continuous predicted rating as ranking',
        history_windows_seconds=[5,30,60], historical_interval='[rating_time-W, rating_time)',
        derivative_assignment='later observation timestamp; support may begin before left window boundary, never after the report',
        nearest_aggregation='pool original 5-second moments by real rating_event_id, includes post-report data; diagnostic only',
        common_cohort_rule='existing 1039 real event IDs; require historical non-FOV scene measurement in each width and at least one head/controller measurement in each width',
        report_operation='recording mechanism unconfirmed; no buttons used; scene+head ablation excludes all controller features',
        input_scope='only already-local modern canonical sources; no video',
        user_units='raw head velocity/angular velocity source units unverified; controller magnitude dimensionless',
        time_basis='source timestamp scale inferred as before; exposure elapsed from first finite source timestamp',
        camera_rules=dict(maximum_gap_seconds=1.,jump_source_units=10.,rotation_flag_degrees_per_second=720.,fov_rounding_degrees=.001),
        excluded_inputs=['game/session/participant ID','current/previous/future scores','rating distance','absolute timestamp','gaze','face','button masks'],
        original_protocol='exploratory real-score regression amendment; not the original three-class gate',
        tuning='none; comparing alternatives is exploratory, best observed result is not an independent final test')

def vector(raw, length):
    if not raw or not raw.strip(): return None
    try:
        values=np.asarray(literal_list(raw),dtype=float).reshape(-1)
        return values if len(values)==length and np.isfinite(values).all() else None
    except (ValueError,TypeError,SyntaxError): return None

def summaries(a, prefix, measures):
    out={}
    for m in measures:
        v=a[m].to_numpy(dtype=float); v=v[np.isfinite(v)]
        out[prefix+m+'_mean']=float(v.mean()) if len(v) else np.nan
        out[prefix+m+'_max']=float(v.max()) if len(v) else np.nan
        out[prefix+m+'_std']=float(v.std(ddof=1)) if len(v)>1 else np.nan
        out[prefix+m+'_n']=int(len(v))
    return out

def pool_nearest(windows):
    result={}
    for m in MEASURES:
        ns=windows[m+'_n'].to_numpy(float)
        mus=windows[m+'_mean'].to_numpy(float)
        std=windows[m+'_std'].to_numpy(float)
        mx=windows[m+'_max'].to_numpy(float)
        valid=(ns>0)&np.isfinite(mus)
        n=float(ns[valid].sum())
        mu=float(np.sum(ns[valid]*mus[valid])/n) if n else np.nan
        m2=float(np.sum(np.where(ns[valid]>1,(ns[valid]-1)*np.nan_to_num(std[valid])**2,0)+ns[valid]*(mus[valid]-mu)**2)) if n else np.nan
        result.update({f'nearest_{m}_mean':mu, f'nearest_{m}_max':float(np.nanmax(mx[valid])) if n else np.nan,
                       f'nearest_{m}_std':math.sqrt(max(0.,m2/(n-1))) if n>1 else np.nan,
                       f'nearest_{m}_n':int(n)})
    return result

def extract_trace(record, cfg, logger):
    source=LocalCsvSource(record)
    fields=['frame','timestamp','camera_name','p_matrix','v_matrix','m_matrix','MS_rating',
            'head_vel','head_angvel','IndexTrigger','HandTrigger','Thumbstick']
    fields=[c for c in fields if c in record['columns']]
    rows=[]; reports=[]; counts=Counter(); errors=[]
    origin=scale=selected=None; prev=prev_velocity=prev_thumb=None
    start=time.perf_counter(); last=start; number=0
    rules=cfg['camera_rules']
    for chunk in read_source_csv(source,usecols=fields,dtype=str,chunksize=16,
                                  keep_default_na=False,encoding='utf-8-sig'):
        for row in chunk.to_dict('records'):
            number+=1; raw=finite_number(row.get('timestamp'))
            if raw is None:
                counts['invalid_timestamp']+=1; prev=prev_velocity=prev_thumb=None; continue
            if origin is None:
                scale,unit=timestamp_scale(raw); origin=raw*scale
            t=raw*scale-origin
            if t<0:
                counts['before_origin']+=1; prev=prev_velocity=prev_thumb=None; continue
            rating=finite_number(row.get('MS_rating'))
            if rating is not None:
                reports.append(dict(rating_event_id=f"modern:{record['session_id']}:record:{number}",rating_time_sec=t,ms_rating=rating))
            values={m:np.nan for m in (*MEASURES,*USER_MEASURES)}
            if row.get('m_matrix','').strip():
                try:
                    values['object_count']=float(len(numeric_matrix_list(row['m_matrix'])))
                    counts['valid_object_rows']+=1
                except Exception as e:
                    counts['invalid_object_rows']+=1
                    if len(errors)<5: errors.append(f'{number}:object:{e}')
            if row.get('v_matrix','').strip():
                try:
                    names=literal_list(row.get('camera_name','')); views=numeric_matrix_list(row['v_matrix'])
                    projections=numeric_matrix_list(row.get('p_matrix',''))
                    if not len(views): raise MissingCameraObservation('empty camera observation')
                    if not names or len(names)!=len(views): raise ValueError('camera/view length mismatch')
                    name=selected if selected is not None else str(names[choose_camera(names)])
                    ix=[i for i,n in enumerate(names) if str(n)==name]
                    if len(ix)!=1: raise ValueError('fixed camera missing or repeated')
                    pos,rot=parse_pose(views[ix[0]]); selected=name; counts['valid_camera_rows']+=1
                    if len(projections)==len(names):
                        try: values['fov_horizontal'],values['fov_vertical']=full_fov(projections[ix[0]],rules['fov_rounding_degrees'])
                        except Exception: counts['invalid_fov_rows']+=1
                    if prev is not None:
                        dt=t-prev[0]; displacement=pos-prev[1]
                        if dt<=0 or dt>rules['maximum_gap_seconds']:
                            counts['derivative_interval_resets']+=1; prev_velocity=None
                        elif np.linalg.norm(displacement)>rules['jump_source_units']:
                            counts['position_jumps']+=1; prev_velocity=None
                        else:
                            vel=displacement/dt; midpoint=(t+prev[0])/2
                            values['camera_speed']=float(np.linalg.norm(vel))
                            rate=float(np.degrees(Rotation.from_matrix(prev[2].T@rot).magnitude())/dt)
                            values['camera_rotation_rate']=rate
                            if rate>rules['rotation_flag_degrees_per_second']: counts['rotation_rate_over_flag']+=1
                            if prev_velocity is not None and midpoint>prev_velocity[0]:
                                values['camera_acceleration']=float(np.linalg.norm((vel-prev_velocity[1])/(midpoint-prev_velocity[0])))
                            prev_velocity=(midpoint,vel)
                    prev=(t,pos,rot)
                except MissingCameraObservation:
                    counts['unsampled_camera_rows']+=1
                except Exception as e:
                    counts['invalid_camera_rows']+=1; prev=prev_velocity=None
                    if len(errors)<5: errors.append(f'{number}:camera:{e}')
            # Empty asynchronous camera rows do not reset previous valid camera.
            for raw_name,m in [('head_vel','head_speed'),('head_angvel','head_angular_speed')]:
                v=vector(row.get(raw_name,''),3)
                if v is not None: values[m]=float(np.linalg.norm(v))
                elif row.get(raw_name,'').strip(): counts['invalid_'+raw_name]+=1
            for raw_name,m in [('IndexTrigger','index_trigger_level'),('HandTrigger','hand_trigger_level')]:
                v=vector(row.get(raw_name,''),2)
                if v is not None: values[m]=float(v.mean())
                elif row.get(raw_name,'').strip(): counts['invalid_'+raw_name]+=1
            thumb=vector(row.get('Thumbstick',''),4)
            if thumb is not None:
                values['thumbstick_magnitude']=float(np.linalg.norm(thumb.reshape(2,2),axis=1).mean())
                if prev_thumb is not None:
                    dt=t-prev_thumb[0]
                    if 0<dt<=rules['maximum_gap_seconds']:
                        values['thumbstick_change_rate']=float(np.linalg.norm((thumb-prev_thumb[1]).reshape(2,2),axis=1).mean()/dt)
                prev_thumb=(t,thumb)
            rows.append(dict(timestamp_sec=t,record_number=number,**values))
        if time.perf_counter()-last>25:
            logger.info('PROGRESS %s rows=%s seconds=%.1f',record['session_id'],number,time.perf_counter()-start); last=time.perf_counter()
    if number!=record['raw_rows_expected']: raise ValueError('raw row count mismatch')
    if len(reports)!=record['rating_events_expected']: raise ValueError('actual rating count mismatch')
    trace=pd.DataFrame(rows).sort_values(['timestamp_sec','record_number'],kind='stable')
    return trace,reports,dict(rows=number,real_reports=len(reports),origin_epoch=origin,timestamp_unit=unit,
                              selected_camera=selected,counters=dict(counts),errors=errors,seconds=time.perf_counter()-start)

def prepare_session(record,target,cfg,logger):
    trace,reports,meta=extract_trace(record,cfg,logger)
    real={r['rating_event_id']:r for r in reports}; out=[]
    times=trace.timestamp_sec.to_numpy()
    for event in target.to_dict('records'):
        actual=real[event['rating_event_id']]
        if actual['ms_rating']!=event['ms_rating'] or abs(actual['rating_time_sec']-event['rating_time_sec'])>1e-5:
            raise ValueError('target event differs from raw score/time')
        t=actual['rating_time_sec']; row=dict(game_name=record['game_name'],session_id=record['session_id'],participant_id=record['participant_id'],
                   rating_event_id=event['rating_event_id'],ms_rating=actual['ms_rating'],rating_time_sec=t,exposure_elapsed_seconds=t)
        usable=True
        for width in (5,30,60):
            lo=np.searchsorted(times,max(0.,t-width),'left'); hi=np.searchsorted(times,t,'left')
            hist=trace.iloc[lo:hi]
            row.update(summaries(hist,f'past{width}_',MEASURES+USER_MEASURES))
            nonfov=['camera_speed','camera_acceleration','camera_rotation_rate','object_count']
            scene_ok=bool(hist[nonfov].notna().any().any())
            user_ok=bool(hist[list(USER_MEASURES)].notna().any().any())
            row[f'past{width}_scene_available']=scene_ok;row[f'past{width}_user_available']=user_ok
            row[f'past{width}_raw_rows']=len(hist)
            usable=usable and scene_ok and user_ok
        row['common_eligible']=usable;out.append(row)
    return pd.DataFrame(out),meta

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--init',action='store_true');ap.add_argument('--finalize',action='store_true')
    ap.add_argument('--partition-count',type=int,default=1);ap.add_argument('--partition-index',type=int,default=0)
    args=ap.parse_args()
    if not (ROOT/'PROTOCOL.md').is_file(): raise SystemExit('PROTOCOL.md missing')
    for p in (OUT,RES,OUT/'by_session'):p.mkdir(parents=True,exist_ok=True)
    if args.init:
        if CFG_PATH.exists(): raise SystemExit('Frozen config already exists; do not overwrite')
        CFG_PATH.write_text(json.dumps(make_config(),ensure_ascii=False,indent=2),encoding='utf-8'); print(CFG_PATH);return
    cfg=json.loads(CFG_PATH.read_text(encoding='utf-8'))
    logfile=RES/f'preparation_part{args.partition_index}.log'
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',handlers=[logging.FileHandler(logfile,encoding='utf-8'),logging.StreamHandler()])
    logger=logging.getLogger('rating_aggregation')
    old_path=ROOT/'features/param_rating_local_all_2026-10-10.csv'
    old=pd.read_csv(old_path,encoding='utf-8-sig')
    old=old[old['ms_rating'].notna()&old['window_has_any_feature'].astype(str).str.lower().isin(['true','1'])]
    targets=old[['game_name','session_id','participant_id','rating_event_id','rating_time_sec','ms_rating']].drop_duplicates('rating_event_id')
    if len(targets)!=1039: raise ValueError('Expected frozen 1039 rating events')
    inventory_path=ROOT/'results/param_rating_local_all_2026-10-10/source_inventory.json'
    records=[r for r in json.loads(inventory_path.read_text(encoding='utf-8'))['sources'] if r['session_id'] in set(targets.session_id)]
    if args.finalize:
        parts=[];metas=[];failures=[]
        for record in records:
            sid=record['session_id'];p=OUT/'by_session'/f'{sid}.csv';j=p.with_suffix('.json')
            if not p.is_file() or not j.is_file(): failures.append(sid);continue
            meta=json.loads(j.read_text(encoding='utf-8'))
            if meta.get('status')!='complete' or meta['output_sha256']!=sha256(p): raise ValueError('Cache completion/hash mismatch')
            parts.append(pd.read_csv(p,encoding='utf-8-sig'));metas.append(meta)
        events=pd.concat(parts,ignore_index=True)
        if not events.rating_event_id.is_unique:raise ValueError('Duplicate event IDs')
        nearest=[]
        for event,group in old.groupby('rating_event_id',sort=False):nearest.append(dict(rating_event_id=event,**pool_nearest(group)))
        events=events.merge(pd.DataFrame(nearest),on='rating_event_id',validate='one_to_one')
        events=events.sort_values(['game_name','session_id','rating_time_sec'],kind='stable')
        events.to_csv(OUT/'events.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
        eligible=events['common_eligible'].astype(str).str.lower().isin(['true','1'])
        events.loc[~eligible,['rating_event_id','game_name','session_id','rating_time_sec']+[f'past{w}_{g}_available' for w in (5,30,60) for g in ('scene','user')]].to_csv(OUT/'excluded_events.csv',index=False,encoding='utf-8-sig')
        manifest=dict(status='complete' if not failures else 'partial',target_events=1039,prepared_events=len(events),common_eligible_events=int(eligible.sum()),
                      sessions=int(events.session_id.nunique()),games=int(events.game_name.nunique()),failed_sessions=failures,config_sha256=sha256(CFG_PATH),
                      old_windows_sha256=sha256(old_path),source_inventory_sha256=sha256(inventory_path),events_sha256=sha256(OUT/'events.csv'),
                      sessions_metadata=metas,feature_groups=dict(scene=list(MEASURES),user=list(USER_MEASURES)),
                      unit_caveat=cfg['user_units'],status_caveat='exploratory, not final conclusion')
        (OUT/'preparation_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        logger.info('FINALIZED prepared=%s common=%s failures=%s',len(events),int(eligible.sum()),failures);return
    records=[r for i,r in enumerate(records) if i%args.partition_count==args.partition_index]
    identity_base=dict(script_sha256=sha256(Path(__file__)),config_sha256=sha256(CFG_PATH))
    for i,record in enumerate(records,1):
        sid=record['session_id'];outpath=OUT/'by_session'/f'{sid}.csv';meta_path=outpath.with_suffix('.json')
        identity=dict(identity_base,source_locator=record['source_locator'],known_sha256=record['known_sha256'],
                      size_bytes=record['size_bytes'],game_name=record['game_name'],participant_id=record['participant_id'])
        logger.info('START %s/%s %s bytes=%s',i,len(records),sid,record['size_bytes'])
        try:
            saved=json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
            if saved.get('identity')==identity and saved.get('status')=='complete' and outpath.exists() and saved['output_sha256']==sha256(outpath):
                logger.info('RESUME %s',sid);continue
            part,meta=prepare_session(record,targets[targets.session_id==sid],cfg,logger)
            part.to_csv(outpath,index=False,encoding='utf-8-sig',float_format='%.17g')
            meta.update(status='complete',session_id=sid,game_name=record['game_name'],identity=identity,events=len(part),output_sha256=sha256(outpath))
            meta_path.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
            logger.info('DONE %s events=%s seconds=%.1f',sid,len(part),meta['seconds'])
        except Exception:
            logger.exception('FAILED %s',sid)

if __name__=='__main__': main()
