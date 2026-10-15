"""Exploratory native-coaster RGB -> engineering-motion trial.

No FMS labels or physical risk thresholds are generated. Raw data are read-only.
Candidate rig coordinates are an explicitly uncalibrated sensitivity analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "coaster_motion_trial"
S = np.diag([1., 1., -1., 1.])
FLOW_METRICS = ["magnitude_mean", "magnitude_p90", "magnitude_std", "horizontal_abs", "vertical_abs",
                "horizontal_signed", "vertical_signed", "radial_signed", "tangential_signed", "direction_concentration"]
VISUAL_METRICS = ["brightness_mean", "brightness_std", "edge_density", "laplacian_variance"]
FEATURE_COLS = [f"flow_{m}_{stat}" for m in FLOW_METRICS for stat in ("mean", "std")]
FEATURE_COLS += [f"visual_{m}_{stat}" for m in VISUAL_METRICS for stat in ("mean", "std")]
TARGET_COLS = ["target_camera_speed_mean", "target_camera_acceleration_mean", "target_camera_angular_speed_mean",
               "target_candidate_rig_speed_mean", "target_candidate_rig_acceleration_mean"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_matrix(text, width):
    a = np.fromstring(str(text), sep=" ")
    if len(a) != width * 4 or not np.isfinite(a).all():
        raise ValueError(f"Expected finite {width}x4 matrix, got {text!r}")
    result = np.eye(4)
    result[:width] = a.reshape(width, 4)
    return result


def proper_rotations(matrices):
    r = matrices[:, :3, :3]
    error = np.linalg.norm(r.transpose(0, 2, 1) @ r - np.eye(3), axis=(1, 2))
    determinant = np.linalg.det(r)
    if (error > .005).any() or (np.abs(determinant - 1) > .005).any():
        raise ValueError("Camera-to-world corrected rotation not approximately SO(3)")
    u, _, vh = np.linalg.svd(r)
    out = u @ vh
    if (np.linalg.det(out) < 0).any():
        raise ValueError("Unexpected remaining reflection after Z convention correction")
    return out, float(error.max())


def selected_frames(rows):
    relative = (rows.timestamp_ms.to_numpy(np.int64) - int(rows.timestamp_ms.iloc[0])) / 1000.
    wanted = np.linspace(relative[0], relative[-1], 8)
    chosen = []
    for target in wanted:
        right = int(np.searchsorted(relative, target))
        options = [i for i in (right - 1, right) if 0 <= i < len(relative)]
        chosen.append(min(options, key=lambda i: abs(relative[i] - target)))
    return rows.iloc[sorted(set(chosen))]


def image_gray(path):
    payload = Path(path).read_bytes()
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode JPEG: {path}")
    # Observed upside-down text in native samples; derived image only, raw unchanged.
    image = cv2.flip(image, 0)
    h, w = image.shape[:2]
    image = cv2.resize(image, (160, round(h * 160 / w)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), hashlib.sha256(payload).hexdigest()


def rgb_features(rows):
    values = {k: [] for k in FLOW_METRICS + VISUAL_METRICS}
    previous = None
    mask = None
    hashes = []
    for row in rows.itertuples(index=False):
        gray, digest = image_gray(row.rgb_path)
        hashes.append({"frame_id": int(row.frame_id), "sha256": digest})
        if mask is None:
            h, w = gray.shape
            yy, xx = np.mgrid[:h, :w]
            x = (xx - (w - 1) / 2) / (.43 * w)
            y = (yy - (h - 1) / 2) / (.43 * h)
            mask = x * x + y * y <= 1
            radial_x = xx - (w - 1) / 2
            radial_y = yy - (h - 1) / 2
            radius = np.maximum(np.hypot(radial_x, radial_y), 1)
            radial_x = radial_x / radius
            radial_y = radial_y / radius
        elif gray.shape != previous[0].shape:
            raise ValueError("Inconsistent frame dimensions within a clip")
        values["brightness_mean"].append(float(gray[mask].mean() / 255))
        values["brightness_std"].append(float(gray[mask].std() / 255))
        values["edge_density"].append(float((cv2.Canny(gray, 50, 150)[mask] > 0).mean()))
        values["laplacian_variance"].append(float(cv2.Laplacian(gray, cv2.CV_32F)[mask].var()))
        if previous is not None:
            old, timestamp = previous
            dt = (int(row.timestamp_ms) - timestamp) / 1000
            if not 0 < dt <= .25:
                raise ValueError(f"Invalid selected-frame temporal gap {dt}")
            flow = cv2.calcOpticalFlowFarneback(old, gray, None, .5, 3, 15, 3, 5, 1.2, 0)
            scale = dt * math.hypot(*gray.shape)
            u, v = flow[..., 0] / scale, flow[..., 1] / scale
            mag = np.hypot(u, v)
            total_mag = float(mag[mask].sum())
            record = {
                "magnitude_mean": mag[mask].mean(), "magnitude_p90": np.percentile(mag[mask], 90),
                "magnitude_std": mag[mask].std(), "horizontal_abs": np.abs(u[mask]).mean(),
                "vertical_abs": np.abs(v[mask]).mean(), "horizontal_signed": u[mask].mean(),
                "vertical_signed": v[mask].mean(),
                "radial_signed": (u * radial_x + v * radial_y)[mask].mean(),
                "tangential_signed": (-u * radial_y + v * radial_x)[mask].mean(),
                "direction_concentration": math.hypot(float(u[mask].sum()), float(v[mask].sum())) / max(total_mag, 1e-12),
            }
            for k, value in record.items():
                values[k].append(float(value))
        previous = gray, int(row.timestamp_ms)
    features = {}
    for k, v in values.items():
        prefix = "flow_" if k in FLOW_METRICS else "visual_"
        features[prefix + k + "_mean"] = float(np.mean(v))
        features[prefix + k + "_std"] = float(np.std(v))
    return features, hashes


def freeze_definition(args, inputs):
    definition_path = ROOT / f"audit/{PREFIX}_definition.json"
    definition = {
        "created_at": datetime.now(timezone.utc).isoformat(), "seed": 42,
        "scope": "Exploratory engineering-motion regression; no human sickness grade or validated design-only trajectory",
        "authorization": "User 2026-10-10: 你试试, following proposed motion-learning and paired-score inspection",
        "protocol_deviation": "Separate motion-property task from PROTOCOL v1.0 FMS risk classification; original gate unchanged; user authorized pilot",
        "inputs": inputs, "sample_scope": "Only five locally existing original Epic Roller Coaster recordings",
        "camera_world": "inverse(view) @ diag(1,1,-1,1); Unity negative camera-Z convention; SVD repairs only rounding error<0.005",
        "rig_candidate": "camera_world @ inverse(S @ device0_pose @ S); nearest pose<=50ms; device0 HMD and unit/calibration remain provisional",
        "clip_seconds": 1, "camera_filter": "name exactly Camera, confirmed sampled ride imagery; other names excluded",
        "segmentation": {"name_change": True, "max_gap_seconds": .25, "jump_source_units": 10,
                         "boundary_exclusion_seconds": .5, "min_clip_raw_frames": 6,
                         "max_clip_boundary_gap_seconds": .2, "max_sampled_frames": 8},
        "target_smoothing": "Linear position interpolation on 10Hz actual-time grid within continuous segment; fixed5-point poly2 Savitzky-Golay derivatives (0.5sec); exclude0.5sec segment boundaries; angular central SO(3) difference 0.2sec",
        "target_columns": TARGET_COLS, "units": {"speed": "unverified source-coordinate units/sec", "acceleration": "unverified source-coordinate units/sec^2", "angular_speed": "degrees/sec"},
        "image_processing": "Vertical flip, aspect-preserving resize width160, fixed ellipse(rx=.43width,ry=.43height) to exclude outer viewport; no raw modifications",
        "flow": "Farneback(.5,3,15,3,5,1.2,0); divide flow by actual dt and resized image diagonal; inferred RGB flow, not native engine motion",
        "feature_columns": FEATURE_COLS, "feature_eligibility": "Primary flow only; secondary flow+visual. IDs/time/pose/matrices/targets excluded",
        "models": {"mean": "training-fold target mean", "elapsed_ridge": "Ridge alpha10 using elapsed only", "flow_ridge": "Ridge alpha10 train-only medianimpute/variancefilter/zscore",
                   "flow_lightgbm": {"n_estimators": 150, "num_leaves": 7, "max_depth": 3, "learning_rate": .05, "min_child_samples": 30, "reg_lambda": 5, "colsample_bytree": .9, "random_state": 42, "n_jobs": 2},
                   "full_lightgbm": "same hyperparameters, flow+visual input"},
        "evaluation": "Leave-one-recording-out supplemental; conservative candidate-trajectory-family holdout if>=2known groups, all repeats held together; no random clip split",
        "metrics": "MAE/RMSE/Spearman; compare train-mean and elapsed; no binary AUC or physical risk cutoffs; pergroup and pooled",
        "gate": "No TSM expansion based solely on recording-holdout result; provisional unseen-trajectory test needs positive error reduction against both mean and elapsed and nonnegative ranking across multiple heldoutgroups; small/uncertain groups are insufficient validation",
    }
    if definition_path.exists():
        old = json.loads(definition_path.read_text(encoding="utf-8"))
        for k in definition:
            if k != "created_at" and old.get(k) != definition[k]:
                raise ValueError(f"Frozen motion definition differs at {k}; do not silently change")
        return old
    definition_path.write_text(json.dumps(definition, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Registered definition before extracting training features: %s", definition_path)
    return definition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data/raw/vrnet/rollercoaster_raw")
    args = parser.parse_args()
    if not (ROOT / "PROTOCOL.md").is_file():
        raise FileNotFoundError("PROTOCOL.md missing; experiments prohibited")
    for folder in ("features", "results", "audit"):
        (ROOT / folder).mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(ROOT / f"results/{PREFIX}_extraction_log.txt", encoding="utf-8")])
    cv2.setNumThreads(1)
    cv2.setRNGSeed(42)
    sources = []
    for d in sorted((args.raw_root / "derived_csv").iterdir()):
        camera_path, pose_path = d / "camera.csv", d / "pose.csv"
        token = re.search(r"VRLOG-\d+", d.name).group()
        index_path = ROOT / f"features/rollercoaster_depth_rgb_{token}_frame_index.csv"
        if not all(p.is_file() for p in [camera_path, pose_path, index_path]):
            raise FileNotFoundError(f"Required existing camera/pose/frame index missing: {d}")
        sources.append(dict(session_id=token, camera=str(camera_path.resolve()), pose=str(pose_path.resolve()), index=str(index_path.resolve()),
                            camera_sha256=sha(camera_path), pose_sha256=sha(pose_path), index_sha256=sha(index_path)))
    if len(sources) != 5:
        raise ValueError(f"Expected five actual local recordings; found {len(sources)}")
    definition = freeze_definition(args, sources)
    records, session_audit, segmentation_rows = [], [], []
    started = time.monotonic()
    for source in sources:
        token = source["session_id"]
        camera = pd.read_csv(source["camera"])
        index = pd.read_csv(source["index"])
        if not np.array_equal(camera.framecounter.to_numpy(), index.frame_id.to_numpy()) or not np.array_equal(camera.timestamp.to_numpy(), index.timestamp_ms.to_numpy()):
            raise ValueError(f"Camera/frame index not exactly aligned: {token}")
        camera["camera_run_id"] = camera.name.ne(camera.name.shift()).cumsum() - 1
        t = (camera.timestamp.to_numpy(np.int64) - int(camera.timestamp.iloc[0])) / 1000
        views = np.stack([parse_matrix(x, 4) for x in camera.view])
        cworld = np.linalg.inv(views) @ S
        rot, ortho_error = proper_rotations(cworld)
        positions = cworld[:, :3, 3]
        pose = pd.read_csv(source["pose"])
        pose = pose.loc[pose.device_id.eq(0)].sort_values("timestamp")
        if pose.timestamp.duplicated().any():
            raise ValueError(f"Duplicate device0 timestamps: {token}")
        ht = pose.timestamp.to_numpy(np.int64)
        ct = camera.timestamp.to_numpy(np.int64)
        right = np.searchsorted(ht, ct)
        right = np.clip(right, 0, len(ht) - 1)
        left = np.clip(right - 1, 0, len(ht) - 1)
        closest = np.where(abs(ht[left] - ct) <= abs(ht[right] - ct), left, right)
        delta_ms = abs(ht[closest] - ct)
        hmd = np.stack([parse_matrix(x, 3) for x in pose.deviceToAbsoluteTracking])
        rig = cworld @ np.linalg.inv(S @ hmd[closest] @ S)
        rig_pos = rig[:, :3, 3]
        rig_valid = delta_ms <= 50
        discontinuity = np.r_[True, (np.diff(t) > .25) | (np.diff(t) <= 0) | (np.linalg.norm(np.diff(positions, axis=0), axis=1) > 10) |
                               (camera.name.to_numpy()[1:] != camera.name.to_numpy()[:-1])]
        segment_ids = discontinuity.cumsum() - 1
        participant_files = list((args.raw_root / "self_reports/SelfReportData").glob(f"*{token} Epic Roller Coasters.csv"))
        if len(participant_files) != 1:
            raise ValueError(f"Participant mapping ambiguous: {token}")
        participant_id = re.match(r"(\d+) ", participant_files[0].name).group(1)
        n_start = len(records)
        for segment_id in np.unique(segment_ids):
            idx = np.flatnonzero(segment_ids == segment_id)
            run = int(camera.camera_run_id.iloc[idx[0]])
            name = camera.name.iloc[idx[0]]
            segmentation_rows.append(dict(session_id=token, camera_run_id=run, segment_id=int(segment_id), name=name,
                                          start_timestamp_ms=int(ct[idx[0]]), end_timestamp_ms=int(ct[idx[-1]]), frames=len(idx), duration_sec=float(t[idx[-1]] - t[idx[0]])))
            if name != "Camera" or len(idx) < 8 or t[idx[-1]] - t[idx[0]] < 2:
                continue
            grid = np.arange(t[idx[0]], t[idx[-1]], .1)
            p = np.stack([np.interp(grid, t[idx], positions[idx, k]) for k in range(3)], axis=1)
            v = savgol_filter(p, 5, 2, deriv=1, delta=.1, axis=0)
            a = savgol_filter(p, 5, 2, deriv=2, delta=.1, axis=0)
            gr = Slerp(t[idx], Rotation.from_matrix(rot[idx]))(grid)
            angular = np.full(len(grid), np.nan)
            angular[1:-1] = np.rad2deg((gr[:-2].inv() * gr[2:]).magnitude()) / .2
            rv = np.full(len(grid), np.nan)
            ra = np.full(len(grid), np.nan)
            rig_idx = idx[rig_valid[idx]]
            if len(rig_idx) >= 8:
                rp = np.stack([np.interp(grid, t[rig_idx], rig_pos[rig_idx, k]) for k in range(3)], axis=1)
                rv = np.linalg.norm(savgol_filter(rp, 5, 2, deriv=1, delta=.1, axis=0), axis=1)
                ra = np.linalg.norm(savgol_filter(rp, 5, 2, deriv=2, delta=.1, axis=0), axis=1)
            target_arrays = [np.linalg.norm(v, axis=1), np.linalg.norm(a, axis=1), angular, rv, ra]
            first_k = math.ceil(t[idx[0]] + .5)
            last_k = math.floor(t[idx[-1]] - .5) - 1
            for k in range(first_k, last_k + 1):
                start_ms = int(ct[0]) + k * 1000
                end_ms = start_ms + 1000
                local_idx = idx[(ct[idx] >= start_ms) & (ct[idx] < end_ms)]
                row = dict(session_id=token, participant_id=participant_id, camera_run_id=run, segment_id=int(segment_id),
                           clip_id=f"{token}_run{run}_seg{segment_id}_sec{k}", start_timestamp_ms=start_ms, end_timestamp_ms=end_ms,
                           clip_elapsed_sec=k, raw_frame_count=len(local_idx), selected_frame_count=0, status="skipped", reason="", frame_ids="[]", frame_timestamps_ms="[]", rgb_sha256="[]",
                           **{x: np.nan for x in FEATURE_COLS + TARGET_COLS})
                selected_grid = (grid >= k) & (grid < k + 1)
                if selected_grid.sum() < 8:
                    row["reason"] = "insufficient_target_grid"
                elif len(local_idx) < 6:
                    row["reason"] = "fewer_than_6_actual_frames"
                elif ct[local_idx[0]] - start_ms > 200 or end_ms - ct[local_idx[-1]] > 200:
                    row["reason"] = "insufficient_boundary_frame_coverage"
                elif np.diff(ct[local_idx]).max() > 250:
                    row["reason"] = "actual_frame_gap_over_250ms"
                elif index.rgb_path.iloc[local_idx].isna().any():
                    row["reason"] = "missing_RGB_path"
                else:
                    rows = selected_frames(index.iloc[local_idx])
                    row["selected_frame_count"] = len(rows)
                    row["frame_ids"] = json.dumps(rows.frame_id.tolist())
                    row["frame_timestamps_ms"] = json.dumps(rows.timestamp_ms.tolist())
                    try:
                        features, hashes = rgb_features(rows)
                        row.update(features)
                        row["rgb_sha256"] = json.dumps(hashes)
                        for target, values in zip(TARGET_COLS, target_arrays):
                            row[target] = float(np.nanmean(values[selected_grid]))
                        # Unknown pose match is never filled as a valid candidate-rig target.
                        if not rig_valid[local_idx].all():
                            row[TARGET_COLS[3]] = row[TARGET_COLS[4]] = np.nan
                        row["status"] = "ok"
                    except (ValueError, FileNotFoundError) as exc:
                        row["reason"] = str(exc)
                records.append(row)
                if len(records) % 200 == 0:
                    logging.info("%s processed %s clips, elapsed %.1fs", token, len(records), time.monotonic() - started)
        current = pd.DataFrame(records[n_start:])
        audit = dict(session_id=token, participant_id=participant_id, camera_rows=len(camera), camera_name_counts=camera.name.value_counts().to_dict(),
                     camera_name_runs=int(camera.camera_run_id.max()) + 1, segmentation_count=int(segment_ids.max()) + 1,
                     position_jumps_over10=int((np.linalg.norm(np.diff(positions, axis=0), axis=1) > 10).sum()),
                     corrected_rotation_max_orthogonality_error=ortho_error, nearest_pose_delta_ms_median=float(np.median(delta_ms)),
                     nearest_pose_delta_ms_max=int(delta_ms.max()), clips_considered=len(current),
                     status_counts=current.status.value_counts().to_dict() if len(current) else {},
                     skipped_reasons=current.loc[current.status.eq("skipped"), "reason"].value_counts().to_dict() if len(current) else {})
        # Menu stationary-anchor check is a diagnostic, not physical calibration.
        menu_rows = np.flatnonzero(camera.name.to_numpy() == "Camera (eye)")
        first_run = int(camera.camera_run_id.iloc[menu_rows[0]]) if len(menu_rows) else -1
        menu_idx = np.flatnonzero((camera.camera_run_id.to_numpy() == first_run) & rig_valid)
        if len(menu_idx):
            audit["first_menu_camera_position_range"] = np.ptp(positions[menu_idx], axis=0).tolist()
            audit["first_menu_candidate_rig_position_range"] = np.ptp(rig_pos[menu_idx], axis=0).tolist()
        session_audit.append(audit)
        logging.info("Finished %s: %s", token, audit)
        pd.DataFrame(records).to_csv(ROOT / f"features/{PREFIX}_clips.csv", index=False)
    pd.DataFrame(segmentation_rows).to_csv(ROOT / f"results/{PREFIX}_segments.csv", index=False)
    (ROOT / f"results/{PREFIX}_session_audit.json").write_text(json.dumps(session_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    data = pd.DataFrame(records)
    lines = ["# 本地过山车运动监督提取", "", "**探索性，非最终结论。监督目标是引擎运动量，不是人体晕动症评分。**", "",
             f"- 实际录像：{len(sources)}；片段候选：{len(data)}；成功：{int(data.status.eq('ok').sum())}。",
             "- 使用真实帧时间戳；乘坐相机 Camera；排除菜单相机、跨加载/坐标跳变及时间缺口。",
             "- RGB仅在内存纵向翻转；8帧上限；Farneback是从RGB推断的光流，不是解码原生motion.zfp。",
             "- 相机位置单位未确认，速度/加速度使用源坐标单位。rig为待校准敏感性目标，不声称已完全去除头动。",
             "- 矩阵、目标值、参与者/录像/时间ID不会进入视觉模型；具体预登记见 audit/coaster_motion_trial_definition.json。", "",
             "|录像|候选片段|成功片段|位置跳变>10源单位|", "|---|---:|---:|---:|"]
    for audit in session_audit:
        lines.append(f"|{audit['session_id']}|{audit['clips_considered']}|{audit['status_counts'].get('ok',0)}|{audit['position_jumps_over10']}|")
    lines += ["", "## 目标分布", "", data.loc[data.status.eq("ok"), TARGET_COLS].describe().to_markdown(), "",
              "## 局限", "", "相机名与样图支持乘坐/菜单筛选，但不提供正式track_id。轨道重复须分组；按录像留出不能证明新轨道泛化。候选rig公式与米制尺度未经作者校准。仅一个游戏。"]
    (ROOT / f"results/{PREFIX}_extraction_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info("Extraction complete %.1fs", time.monotonic() - started)


if __name__ == "__main__":
    main()
