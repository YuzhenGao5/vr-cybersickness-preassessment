"""Read-only geometric audit of original coaster camera runs.

Candidate groups are conservative leakage guards, not author track IDs.
This script does not infer metres, physical risk cutoffs, or train models.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

S = np.diag([1., 1., -1., 1.])


def markdown(table):
    headers = list(table.columns)
    lines = ['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']
    for row in table.itertuples(index=False, name=None):
        lines.append('| '+' | '.join(str(x).replace('|','\\|') for x in row)+' |')
    return '\n'.join(lines)


def matrices(values, shape):
    arrays = [np.fromstring(str(value), sep=' ') for value in values]
    if any(len(value) != np.prod(shape) for value in arrays):
        raise ValueError('Unexpected matrix length')
    return np.stack(arrays).reshape(-1, *shape)


def load_camera(file):
    df = pd.read_csv(file)
    V = matrices(df['view'], (4, 4))
    C = np.linalg.inv(V) @ S
    # The view has det~-1; a fixed camera-Z reflection yields det~+1.
    xyz = C[:, :3, 3]
    dt = np.r_[np.nan, np.diff(df.timestamp.to_numpy()) / 1000.]
    displacement = np.r_[np.nan, np.linalg.norm(np.diff(xyz, axis=0), axis=1)]
    change = df.name.ne(df.name.shift()).to_numpy()
    run = np.cumsum(change) - 1
    continuous_break = change | (dt > .25) | (displacement > 10) | (dt <= 0)
    seg = np.cumsum(continuous_break) - 1
    df['camera_run_id'] = run
    df['continuous_segment_id'] = seg
    df['displacement'] = displacement
    df['dt'] = dt
    return df, C, xyz


def angular_speed(transform, dt):
    u, _, vt = np.linalg.svd(transform[:, :3, :3])
    r = u @ vt
    relative = np.transpose(r[:-1], (0, 2, 1)) @ r[1:]
    angle = np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2)-1)/2, -1, 1))
    return np.r_[np.nan, np.degrees(angle)/dt[1:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    args = ap.parse_args()
    root = Path(args.root).resolve()
    folders = sorted((root/'data/raw/vrnet/rollercoaster_raw/derived_csv').iterdir())
    summaries, paths, quality = [], {}, []
    for folder in folders:
        if not folder.is_dir():
            continue
        sid = folder.name.split(' ')[-1]
        df, C, xyz = load_camera(folder/'camera.csv')
        p = pd.read_csv(folder/'pose.csv')
        p = p[p.device_id.eq(0)].sort_values('timestamp').drop_duplicates('timestamp')
        H = np.repeat(np.eye(4)[None], len(p), axis=0)
        H[:, :3] = matrices(p.deviceToAbsoluteTracking, (3, 4))
        t = df.timestamp.to_numpy(dtype=np.int64)
        pt = p.timestamp.to_numpy(dtype=np.int64)
        ix = np.searchsorted(pt, t)
        ix = np.clip(ix, 0, len(pt)-1)
        previous = np.maximum(ix-1, 0)
        ix = np.where(np.abs(pt[previous]-t) <= np.abs(pt[ix]-t), previous, ix)
        residual_ms = np.abs(pt[ix] - t)
        rig = C @ np.linalg.inv(S @ H[ix] @ S)
        rig_unflipped = C @ np.linalg.inv(H[ix])
        determinant = np.linalg.det(C[:, :3, :3])
        eye_valid = (df.name.eq('Camera (eye)') & df.name.eq(df.name.shift())
            & df.dt.gt(0) & df.dt.le(.25) & df.displacement.le(10)).to_numpy()
        cam_omega = angular_speed(C, df.dt.to_numpy())
        rig_omega = angular_speed(rig, df.dt.to_numpy())
        wrong_omega = angular_speed(rig_unflipped, df.dt.to_numpy())
        camera_speed = df.displacement.to_numpy()/df.dt.to_numpy()
        rig_speed = np.r_[np.nan,np.linalg.norm(np.diff(rig[:,:3,3],axis=0),axis=1)/df.dt.to_numpy()[1:]]
        quality.append({'session_id':sid, 'camera_rows':len(df),
            'camera_runs':int(df.camera_run_id.max()+1),
            'camera_det_min':float(determinant.min()),'camera_det_max':float(determinant.max()),
            'pose_nearest_median_ms':float(np.median(residual_ms)),
            'pose_nearest_max_ms':int(residual_ms.max()),
            'pose_within_80ms_fraction':float(np.mean(residual_ms<=80)),
            'same_name_jumps_gt10':int(((df.displacement>10)&df.name.eq(df.name.shift())).sum()),
            'eye_camera_speed_mean_source_units_per_sec':float(camera_speed[eye_valid].mean()),
            'eye_candidate_rig_speed_mean_source_units_per_sec':float(rig_speed[eye_valid].mean()),
            'eye_camera_angular_mean_deg_per_sec':float(cam_omega[eye_valid].mean()),
            'eye_candidate_rig_angular_mean_deg_per_sec':float(rig_omega[eye_valid].mean()),
            'eye_unreflected_head_rig_angular_mean_deg_per_sec':float(wrong_omega[eye_valid].mean())})
        for rid, run in df.groupby('camera_run_id', sort=False):
            inds = run.index.to_numpy()
            name = str(run.name.iloc[0])
            mask = ~((df.loc[inds,'displacement'].to_numpy()>10) | (df.loc[inds,'dt'].to_numpy()>.25))
            speed = df.loc[inds,'displacement'].to_numpy() / df.loc[inds,'dt'].to_numpy()
            valid = mask & np.isfinite(speed)
            path = xyz[inds]
            distance_from_start = np.linalg.norm(path-path[0], axis=1)
            span = np.linalg.norm(np.ptp(path, axis=0))
            # Downsample only the geometric audit to max1000 points per run.
            sample = path[np.unique(np.linspace(0,len(path)-1,min(1000,len(path))).astype(int))]
            key = (sid, int(rid))
            valid_path_length = float(df.loc[inds,'displacement'][valid].sum())
            if name == 'Camera' and span >= 50 and len(path)>=100 and valid_path_length >= 100:
                paths[key] = sample
            # Rig candidate variation during all eye-camera stationary menu runs.
            rig_span = np.linalg.norm(np.ptp(rig[inds,:3,3],axis=0))
            wrong_rig_span = np.linalg.norm(np.ptp(rig_unflipped[inds,:3,3],axis=0))
            summaries.append({'session_id':sid,'camera_run_id':int(rid),
                'camera_name':name,'start_timestamp_ms':int(run.timestamp.iloc[0]),
                'end_timestamp_ms':int(run.timestamp.iloc[-1]),
                'start_elapsed_sec':float((run.timestamp.iloc[0]-df.timestamp.iloc[0])/1000.),
                'end_elapsed_sec':float((run.timestamp.iloc[-1]-df.timestamp.iloc[0])/1000.),
                'frame_count':len(run),'start_frame':int(run.framecounter.iloc[0]),
                'end_frame':int(run.framecounter.iloc[-1]),
                'duration_sec':float((run.timestamp.iloc[-1]-run.timestamp.iloc[0])/1000.),
                'continuous_segment_count':int(run.continuous_segment_id.nunique()),
                'trajectory_span_source_units':float(span),
                'valid_path_length_source_units':valid_path_length,
                'p90_speed_source_units_per_sec':float(np.quantile(speed[valid],.9)) if valid.any() else np.nan,
                'candidate_rig_span_source_units':float(rig_span),
                'unreflected_head_candidate_span':float(wrong_rig_span),
                'candidate_track_group':'unknown',
                'group_reason':'Not a long spatially varying Camera run; no track ID inferred'})
    # Freeze grouping before downstream outcomes. World position is preserved:
    # matching paths after free translation/rotation could merge distinct tracks.
    keys=list(paths)
    links=np.eye(len(keys), dtype=int)
    comparisons=[]
    for i,k1 in enumerate(keys):
        for j in range(i+1,len(keys)):
            k2=keys[j]
            a,b=paths[k1],paths[k2]
            da=cKDTree(b).query(a)[0]
            db=cKDTree(a).query(b)[0]
            frac_a=float(np.mean(da<=2.0)); frac_b=float(np.mean(db<=2.0))
            # One-direction coverage merges truncated repeats conservatively.
            # At least 50% overlap in the shorter/longer trajectory, with the
            # other at least20%, to resist a shared loading/station point.
            linked=(max(frac_a,frac_b)>=.5 and min(frac_a,frac_b)>=.2)
            links[i,j]=links[j,i]=int(linked)
            comparisons.append((k1,k2,frac_a,frac_b,float(np.median(da)),float(np.median(db)),linked))
    if keys:
        count, labels=connected_components(csr_matrix(links), directed=False)
        mapping={key:f'geometry_group_{int(labels[i])+1:02d}' for i,key in enumerate(keys)}
        for row in summaries:
            key=(row['session_id'],row['camera_run_id'])
            if key in mapping:
                row['candidate_track_group']=mapping[key]
                row['group_reason']='World-coordinate geometric overlap guard; not a verified named track'
    table=pd.DataFrame(summaries)
    csv=root/'results/coaster_track_group_audit.csv'
    table.to_csv(csv,index=False)
    lines=['# Original coaster track grouping and coordinate audit','',
        'Exploratory engineering audit. No author track IDs, physical metre units, or verified isolated vehicle trajectory are assumed. Raw files were not modified.','',
        '## Fixed grouping algorithm','',
        '- Camera run changes whenever the `name` changes, starting at 0 per session.',
        '- Continuous segments also split for >0.25 s gaps, >10 source-coordinate-unit steps, or nonpositive dt.',
        '- Only `Camera` runs with >=100 frames, world-coordinate bounding-diagonal span >=50 source units, and valid continuous path length >=100 source units are candidate ride trajectories. The path-length condition prevents loading jumps with large bounds but virtually stationary valid frames from masquerading as rides.',
        '- For candidate runs, use up to1000 equally indexed world positions. Two runs link when reciprocal nearest-path coverage at radius2 source units has max>=50% and min>=20%. Connected components are conservative repeated-path leakage guards.',
        '- No centering, spatial rotation, outcome labels, FMS values, or trained-model predictions are used. Unknowns must not be split randomly and called novel tracks.','',
        '## Per-session coordinates and pose timing','',
        markdown(pd.DataFrame(quality)), '',
        '## Camera run audit','',markdown(table),'',
        '## Candidate pair evidence','',
        '| A | B | A fraction within2 | B fraction within2 | A median nearest distance | B median nearest distance | Link |',
        '|---|---|---:|---:|---:|---:|---|']
    for a,b,fa,fb,ma,mb,link in comparisons:
        lines.append(f'| {a[0]} run{a[1]} | {b[0]} run{b[1]} | {fa:.4f} | {fb:.4f} | {ma:.4f} | {mb:.4f} | {link} |')
    lines.extend(['','## Coordinate and head-removal interpretation','',
        '- Observed serialized matrices are row-formatted4x4 with translation in the last column. Direct inverse-view rotation has determinant about-1. The candidate Unity camera transform is `inv(view) @ diag(1,1,-1,1)`; this yields determinant about+1.',
        '- HMD device0 matrices are3x4. Candidate conversion to matching handedness is `S @ H @ S`, and candidate rig is `C @ inv(S @ H @ S)`. The nearest timestamp residuals are reported above; matching timestamps does not validate origins/scale/tracking conventions.',
        '- Rig candidates are diagnostic only. Constant offset after a run can be consistent with a stationary rig, but the Camera vs Camera(eye) channels need separate inspection; direct head subtraction is not producer verified.',
        '- Additional `Camera (eye)` motion metrics compare candidate rig residual against camera motion and deliberately unreflected head pose. Decreased residual supports the handedness candidate in stationary runs; it does not establish physical scale or the full Camera channel convention.',
        '- Absolute physical speed and acceleration thresholds3m/s and0.7m/s² remain unverified because source-coordinate scale is not established.',
        '- The projection scale/offset parameters are constant in this release subset; this audit does not support learning FOV variation.',
        '- First execution computed the CSV but report formatting failed with `ImportError: Missing optional dependency tabulate`. Replaced the optional formatter with a local Markdown table formatter and reran; no data or scientific thresholds were changed.',''])
    (root/'results/coaster_track_group_audit.md').write_text('\n'.join(lines),encoding='utf-8')
    print(table[['session_id','camera_run_id','camera_name','duration_sec','trajectory_span_source_units','candidate_track_group']].to_string(index=False))
    print('Candidate geometric groups:',table.loc[table.candidate_track_group.ne('unknown'),'candidate_track_group'].nunique())
    print(pd.DataFrame(quality).to_string(index=False))


if __name__=='__main__':
    main()
