"""Small exploratory RGB -> recorded camera-motion pilot on local Epic sessions.

This script intentionally treats all speeds as source-coordinate units/s. The
Unity/OpenVR coordinate scale has not been calibrated, so outputs are not m/s
and are not human-validated sickness labels. It uses the author's exported
camera/pose CSVs and the exact type-5 frame index; it never guesses a frame rate.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from rollercoaster_native_io import read_frame_index, resolve_session_dir

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/vrnet/rollercoaster_raw"
DERIVED = RAW / "derived_csv"
FEATURES_OUT = ROOT / "features/coaster_motion_visual_pilot.csv"
REPORT = ROOT / "results/coaster_motion_visual_pilot.md"
LOG = ROOT / "results/coaster_motion_visual_pilot_log.txt"
S = np.diag([1.0, 1.0, -1.0, 1.0])
IMAGE_WIDTH = 256
CAMERA_MAX_GAP_SEC = 0.20
POSE_MAX_OFFSET_SEC = 0.040
JUMP_CUTOFF_SOURCE_UNITS = 10.0


def matrix16(text: str) -> np.ndarray:
    values = np.fromstring(str(text), sep=" ", dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Matrix contains non-finite values: {text!r}")
    if values.size == 16:
        return values.reshape(4, 4)
    if values.size == 12:
        # OpenVR deviceToAbsoluteTracking is stored as a 3x4 affine transform.
        # Completing its conventional homogeneous row is an explicit format
        # interpretation, so the output remains a candidate HMD correction.
        return np.vstack([values.reshape(3, 4), np.array([0.0, 0.0, 0.0, 1.0])])
    raise ValueError(f"Expected 16 camera or 12 OpenVR pose values, got {values.size}")


def translation(matrix: np.ndarray) -> np.ndarray:
    return matrix[:3, 3].copy()


def nearest_pose(pose_times: np.ndarray, time_ms: int) -> tuple[int, float]:
    right = int(np.searchsorted(pose_times, time_ms))
    choices = [j for j in (right - 1, right) if 0 <= j < len(pose_times)]
    idx = min(choices, key=lambda j: abs(int(pose_times[j]) - int(time_ms)))
    return idx, abs(int(pose_times[idx]) - int(time_ms)) / 1000.0


def read_session(session_folder: Path) -> tuple[pd.DataFrame, dict]:
    token_match = re.search(r"VRLOG-\d+", session_folder.name)
    if not token_match:
        raise ValueError(f"Could not identify VRLOG session in {session_folder}")
    token = token_match.group(0)
    native_candidates = [p.parent for p in RAW.rglob("gamedata.log") if token in str(p.parent)]
    if len(native_candidates) != 1:
        raise ValueError(f"Expected exactly one raw native session for {token}; found {native_candidates}")
    native = resolve_session_dir(native_candidates[0])
    camera_path = session_folder / "camera.csv"
    pose_path = session_folder / "pose.csv"
    if not camera_path.is_file() or not pose_path.is_file():
        raise FileNotFoundError(f"Missing camera.csv or pose.csv in {session_folder}")

    index = read_frame_index(native)
    frame = index.loc[index.rgb_path.notna(), ["frame_id", "timestamp_ms", "rgb_path"]].copy()
    camera = pd.read_csv(camera_path, usecols=["framecounter", "timestamp", "name", "projection", "view"])
    camera = camera.rename(columns={"framecounter": "frame_id", "timestamp": "camera_timestamp_ms"})
    joined = frame.merge(camera, on="frame_id", how="left", validate="one_to_one", indicator=True)
    if (joined._merge != "both").any():
        raise ValueError(f"{token}: camera/frame index mismatch: {int((joined._merge != 'both').sum())}")
    time_delta = (joined.camera_timestamp_ms.astype("int64") - joined.timestamp_ms.astype("int64")).abs()
    if time_delta.max() != 0:
        raise ValueError(f"{token}: camera/frame timestamps do not exactly match (max {time_delta.max()} ms)")
    joined = joined.drop(columns="_merge").sort_values("timestamp_ms").reset_index(drop=True)

    pose = pd.read_csv(pose_path, usecols=["timestamp", "device_id", "deviceToAbsoluteTracking"])
    pose = pose.loc[pose.device_id == 0].sort_values("timestamp").reset_index(drop=True)
    if pose.empty:
        raise ValueError(f"{token}: no device_id 0 rows in pose.csv")
    pose_times = pose.timestamp.to_numpy(np.int64)
    pose_matrices = [matrix16(x) for x in pose.deviceToAbsoluteTracking]

    records = []
    offsets = []
    for row in joined.itertuples(index=False):
        view = matrix16(row.view)
        camera_world = np.linalg.inv(view) @ S
        pidx, offset = nearest_pose(pose_times, int(row.timestamp_ms))
        offsets.append(offset)
        hmd_unity = S @ pose_matrices[pidx] @ S
        rig_position = None
        if offset <= POSE_MAX_OFFSET_SEC:
            candidate_rig = camera_world @ np.linalg.inv(hmd_unity)
            rig_position = translation(candidate_rig)
        records.append({
            "frame_id": int(row.frame_id), "timestamp_ms": int(row.timestamp_ms),
            "elapsed_sec": (int(row.timestamp_ms) - int(joined.timestamp_ms.iloc[0])) / 1000.0,
        "camera_name": str(row.name), "rgb_path": str(row.rgb_path),
            "camera_position": translation(camera_world),
            "rig_candidate_position": rig_position,
        })
    table = pd.DataFrame(records)
    table["pose_offset_sec"] = [nearest_pose(pose_times, t)[1] for t in table.timestamp_ms]
    score_candidates = list((RAW / "self_reports/SelfReportData").glob(f"*{token} Epic Roller Coasters.csv"))
    if len(score_candidates) != 1:
        raise ValueError(f"Expected one independent score metadata file for {token}; found {len(score_candidates)}")
    participant_match = re.match(r"(\d+)\s+", score_candidates[0].name)
    if not participant_match:
        raise ValueError(f"Participant ID missing from source metadata filename: {score_candidates[0].name}")
    diagnostics = {
        "session_id": token,
        "participant_id": participant_match.group(1),
        "frame_rows_with_rgb": int(len(table)),
        "camera_name_counts": {str(k): int(v) for k, v in table.camera_name.value_counts().items()},
        "camera_frame_timestamp_max_abs_diff_ms": int(time_delta.max()),
        "pose_device0_nearest_offset_median_ms": float(np.median(offsets) * 1000),
        "pose_device0_nearest_offset_p95_ms": float(np.percentile(offsets, 95) * 1000),
        "pose_device0_rows_within_40ms": int(np.sum(np.asarray(offsets) <= POSE_MAX_OFFSET_SEC)),
        "pose_device0_rows_total": int(len(offsets)),
        "source_unit": "unknown; derived speed is source-coordinate units/s",
    }
    return table, diagnostics


def add_interval_targets(table: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    camera_switches = 0
    jump_count = 0
    invalid_pose_pairs = 0
    for i in range(1, len(table)):
        a, b = table.iloc[i - 1], table.iloc[i]
        if a.camera_name != b.camera_name:
            camera_switches += 1
            continue
        if a.camera_name != "Camera":
            continue
        dt = (int(b.timestamp_ms) - int(a.timestamp_ms)) / 1000.0
        if dt <= 0 or dt > CAMERA_MAX_GAP_SEC:
            continue
        cam_delta = b.camera_position - a.camera_position
        cam_disp = float(np.linalg.norm(cam_delta))
        if cam_disp > JUMP_CUTOFF_SOURCE_UNITS:
            jump_count += 1
            continue
        rig_speed = math.nan
        ap, bp = a.rig_candidate_position, b.rig_candidate_position
        if isinstance(ap, np.ndarray) and isinstance(bp, np.ndarray):
            rig_disp = float(np.linalg.norm(bp - ap))
            if rig_disp <= JUMP_CUTOFF_SOURCE_UNITS:
                rig_speed = rig_disp / dt
            else:
                invalid_pose_pairs += 1
        else:
            invalid_pose_pairs += 1
        rows.append({
            "session_id": None,
            "frame_id_start": int(a.frame_id), "frame_id_end": int(b.frame_id),
            "t_mid_sec": (float(a.elapsed_sec) + float(b.elapsed_sec)) / 2,
            "camera_speed_source_units_s": cam_disp / dt,
            "rig_candidate_speed_source_units_s": rig_speed,
            "interval_dt_sec": dt,
            "rgb_path_start": a.rgb_path, "rgb_path_end": b.rgb_path,
        })
    return pd.DataFrame(rows), {
        "camera_name_switch_pairs": camera_switches,
        "camera_position_jump_pairs_over_10_source_units": jump_count,
        "rig_candidate_pairs_missing_or_jumping": invalid_pose_pairs,
    }


def image_gray(path: str) -> np.ndarray:
    # np.fromfile/imdecode handles the non-ASCII Windows workspace path.
    payload = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(payload, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot decode RGB image {path}")
    image = cv2.flip(image, 0)  # observed source JPG orientation is vertically inverted
    height = max(1, round(image.shape[0] * IMAGE_WIDTH / image.shape[1]))
    return cv2.resize(image, (IMAGE_WIDTH, height), interpolation=cv2.INTER_AREA)


def flow_features(frames: list[tuple[int, str]]) -> dict | None:
    if len(frames) < 4:
        return None
    targets = np.linspace(0.05, 0.95, 5)
    times = np.array([t for t, _ in frames], dtype=np.float64)
    selected_indices = []
    for target in targets:
        idx = int(np.argmin(np.abs(times - target)))
        if idx not in selected_indices:
            selected_indices.append(idx)
    selected = [frames[i] for i in sorted(selected_indices)]
    if len(selected) < 4:
        return None
    gray = [image_gray(p) for _, p in selected]
    magnitudes, coherences, valid_fracs, pair_rates = [], [], [], []
    for i in range(1, len(selected)):
        dt = selected[i][0] - selected[i - 1][0]
        if dt <= 0 or dt > 0.40:
            continue
        first, second = gray[i - 1], gray[i]
        if first.shape != second.shape:
            continue
        flow = cv2.calcOpticalFlowFarneback(first, second, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, theta = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        valid = (first > 8) & (second > 8)  # avoid the obvious black circular-view surround
        if valid.sum() < 100:
            continue
        values = mag[valid] / (math.hypot(*first.shape) * dt)
        magnitudes.extend(values[::max(1, len(values) // 5000)].astype(float).tolist())
        weights = mag[valid].astype(np.float64)
        angles = theta[valid].astype(np.float64)
        if weights.sum() > 0:
            coherences.append(float(np.hypot(np.sum(weights * np.cos(angles)), np.sum(weights * np.sin(angles))) / weights.sum()))
        valid_fracs.append(float(valid.mean()))
        pair_rates.append(float(np.mean(values)))
    if len(pair_rates) < 3 or not magnitudes:
        return None
    v = np.asarray(magnitudes, dtype=np.float64)
    return {
        "rgb_flow_mean_diag_s": float(v.mean()),
        "rgb_flow_p90_diag_s": float(np.percentile(v, 90)),
        "rgb_flow_p99_diag_s": float(np.percentile(v, 99)),
        "rgb_flow_pair_mean_std": float(np.std(pair_rates)),
        "rgb_flow_direction_concentration": float(np.mean(coherences)),
        "rgb_valid_view_fraction": float(np.mean(valid_fracs)),
        "n_flow_pairs": int(len(pair_rates)),
        "n_selected_frames": int(len(selected)),
        "selected_frame_ids": json.dumps([Path(p).stem for _, p in selected]),
    }


def make_windows(table: pd.DataFrame, intervals: pd.DataFrame, session_id: str, participant: str) -> list[dict]:
    if intervals.empty:
        return []
    table = table.loc[table.camera_name == "Camera"].copy()
    out = []
    last_bin = int(math.floor(float(table.elapsed_sec.max())))
    for bin_id in range(last_bin):
        start, end = float(bin_id), float(bin_id + 1)
        iv = intervals.loc[(intervals.t_mid_sec >= start) & (intervals.t_mid_sec < end)]
        if len(iv) < 8:
            continue
        mids = iv.t_mid_sec.to_numpy(float)
        dts = iv.interval_dt_sec.to_numpy(float)
        if mids.min() > start + 0.16 or mids.max() < end - 0.16 or dts.max() > CAMERA_MAX_GAP_SEC:
            continue
        frame_rows = table.loc[(table.elapsed_sec >= start) & (table.elapsed_sec < end)]
        frames = [(float(r.elapsed_sec - start), str(r.rgb_path)) for r in frame_rows.itertuples(index=False)]
        feats = flow_features(frames)
        if feats is None:
            continue
        rig = iv.rig_candidate_speed_source_units_s.dropna().to_numpy(float)
        out.append({
            "session_id": session_id, "participant_id": participant,
            "window_id": f"{session_id}_{bin_id:05d}",
            "start_sec": start, "end_sec": end,
            "camera_speed_mean_source_units_s": float(iv.camera_speed_source_units_s.mean()),
            "camera_speed_p90_source_units_s": float(iv.camera_speed_source_units_s.quantile(.90)),
            "rig_candidate_speed_mean_source_units_s": float(np.mean(rig)) if len(rig) >= 6 else math.nan,
            "n_camera_intervals": int(len(iv)), "n_rig_candidate_intervals": int(len(rig)),
            **feats,
        })
    return out


def attach_candidate_track_groups(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    audit_path = ROOT / "results/coaster_track_group_audit.csv"
    if not audit_path.is_file():
        data["candidate_geometry_group"] = "unknown"
        return data, {"source": "missing coaster_track_group_audit.csv", "unknown_windows": int(len(data))}
    audit = pd.read_csv(audit_path)
    run_groups = {}
    for row in audit.itertuples(index=False):
        if str(row.candidate_track_group) != "unknown":
            run_groups[(str(row.session_id), int(row.camera_run_id))] = str(row.candidate_track_group)
    assignments = []
    for session_id, rows in data.groupby("session_id", sort=False):
        folders = [p for p in DERIVED.iterdir() if p.is_dir() and session_id in p.name and (p / "camera.csv").is_file()]
        if len(folders) != 1:
            assignments.extend((int(i), "unknown") for i in rows.index)
            continue
        cam = pd.read_csv(folders[0] / "camera.csv", usecols=["framecounter", "name"])
        cam["camera_run_id"] = cam.name.ne(cam.name.shift()).cumsum().astype(int) - 1
        frame_run = dict(zip(cam.framecounter.astype(int), cam.camera_run_id.astype(int)))
        for idx, row in rows.iterrows():
            try:
                stems = json.loads(row.selected_frame_ids)
                frame_ids = [int(str(stem).removeprefix("s")) for stem in stems]
                run_ids = {frame_run[x] for x in frame_ids if x in frame_run}
                groups = {run_groups.get((str(session_id), rid), "unknown") for rid in run_ids}
                known = groups - {"unknown"}
                if len(known) == 1 and len(groups) == 1:
                    group = next(iter(known))
                elif len(groups) > 1:
                    group = "mixed"
                else:
                    group = "unknown"
            except Exception:
                group = "unknown"
            assignments.append((int(idx), group))
    group_map = dict(assignments)
    data["candidate_geometry_group"] = [group_map.get(int(i), "unknown") for i in data.index]
    counts = data.candidate_geometry_group.value_counts().to_dict()
    return data, {"source": str(audit_path), "window_counts": {str(k): int(v) for k, v in counts.items()},
                  "unknown_windows": int(data.candidate_geometry_group.isin(["unknown", "mixed"]).sum())}


def score_grouped_cv(data: pd.DataFrame, target: str, group_col: str) -> tuple[list[dict], dict]:
    feature_cols = [
        "rgb_flow_mean_diag_s", "rgb_flow_p90_diag_s", "rgb_flow_p99_diag_s",
        "rgb_flow_pair_mean_std", "rgb_flow_direction_concentration", "rgb_valid_view_fraction",
    ]
    valid = data.dropna(subset=[target, group_col]).copy()
    if group_col == "candidate_geometry_group":
        valid = valid.loc[~valid[group_col].isin(["unknown", "mixed"])]
    groups = sorted(valid[group_col].astype(str).unique())
    folds = []
    oof = []
    if len(groups) < 3:
        return folds, {"error": f"Need >=3 groups in {group_col}; found {len(groups)}"}
    for group in groups:
        train = valid.loc[valid[group_col].astype(str) != group]
        test = valid.loc[valid[group_col].astype(str) == group]
        if len(train) < 10 or len(test) < 3:
            continue
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))
        model.fit(train[feature_cols], train[target])
        pred = model.predict(test[feature_cols])
        base = np.full(len(test), float(train[target].median()))
        rho = spearmanr(test[target], pred).statistic if test[target].nunique() > 1 and np.std(pred) > 0 else math.nan
        base_rho = spearmanr(test[target], base).statistic if test[target].nunique() > 1 and np.ptp(base) > 1e-12 else math.nan
        fold = {
            "held_out_group": group,
            "n_train_windows": int(len(train)), "n_test_windows": int(len(test)),
            "test_participants": ",".join(sorted(test.participant_id.astype(str).unique())),
            "participant_overlap_with_train": int(len(set(test.participant_id) & set(train.participant_id))),
            "ridge_mae": float(mean_absolute_error(test[target], pred)),
            "ridge_rmse": float(math.sqrt(mean_squared_error(test[target], pred))),
            "train_median_baseline_mae": float(mean_absolute_error(test[target], base)),
            "train_median_baseline_rmse": float(math.sqrt(mean_squared_error(test[target], base))),
            "ridge_spearman": float(rho) if np.isfinite(rho) else None,
            "median_baseline_spearman": float(base_rho) if np.isfinite(base_rho) else None,
        }
        folds.append(fold)
        for row_id, actual, estimated in zip(test.index, test[target].to_numpy(), pred):
            oof.append({"index": int(row_id), "actual": float(actual), "prediction": float(estimated), "group": group})
    if not oof:
        return folds, {"error": "No valid grouped folds"}
    pred_df = pd.DataFrame(oof)
    global_pred = pred_df.prediction.to_numpy(float)
    actual = pred_df.actual.to_numpy(float)
    pooled = {
        "n_oof_windows": int(len(pred_df)),
        "oof_mae": float(mean_absolute_error(actual, global_pred)),
        "oof_rmse": float(math.sqrt(mean_squared_error(actual, global_pred))),
        "oof_spearman": float(spearmanr(actual, global_pred).statistic) if np.std(actual) > 0 and np.std(global_pred) > 0 else None,
        "mean_fold_mae": float(np.mean([f["ridge_mae"] for f in folds])),
        "mean_fold_baseline_mae": float(np.mean([f["train_median_baseline_mae"] for f in folds])),
    }
    return folds, pooled


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-features", action="store_true",
                        help="Reuse already extracted window features and only refresh audit/grouped evaluation")
    args = parser.parse_args()
    ROOT.joinpath("features").mkdir(exist_ok=True)
    ROOT.joinpath("results").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(LOG, mode="w", encoding="utf-8"), logging.StreamHandler(sys.stdout)])
    cv2.setNumThreads(1)
    cv2.setRNGSeed(42)
    all_windows, audits = [], []
    for folder in sorted(p for p in DERIVED.iterdir() if p.is_dir() and (p / "camera.csv").is_file()):
        try:
            table, diag = read_session(folder)
            session_id = diag["session_id"]
            intervals, interval_diag = add_interval_targets(table)
            intervals["session_id"] = session_id
            diag.update(interval_diag)
            diag.update({
                "camera_motion_intervals_after_qc": int(len(intervals)),
                "usable_candidate_rig_intervals": int(intervals.rig_candidate_speed_source_units_s.notna().sum()),
            })
            if args.reuse_features:
                diag["one_second_rgb_windows"] = None
            else:
                windows = make_windows(table, intervals, session_id, diag["participant_id"])
                diag["one_second_rgb_windows"] = int(len(windows))
                all_windows.extend(windows)
            audits.append(diag)
            logging.info("%s: %d aligned frames, %d clean intervals%s",
                         session_id, diag["frame_rows_with_rgb"], len(intervals),
                         " (reuse features)" if args.reuse_features else f", {len(windows)} one-second windows")
        except Exception:
            logging.error("FAILED %s\n%s", folder.name, traceback.format_exc())
            audits.append({"session_folder": folder.name, "error": traceback.format_exc()})
    if args.reuse_features:
        if not FEATURES_OUT.is_file():
            raise FileNotFoundError(f"Cannot reuse missing feature file: {FEATURES_OUT}")
        data = pd.read_csv(FEATURES_OUT)
        participant_map = {a["session_id"]: a["participant_id"] for a in audits if "error" not in a}
        data["participant_id"] = data.session_id.map(participant_map)
        if data.participant_id.isna().any():
            raise ValueError("Some cached windows could not be mapped to verified participants")
        window_counts = data.groupby("session_id").size().to_dict()
        for audit in audits:
            if "error" not in audit:
                audit["one_second_rgb_windows"] = int(window_counts.get(audit["session_id"], 0))
    else:
        data = pd.DataFrame(all_windows)
    if data.empty:
        raise RuntimeError("No complete one-second visual windows passed the predeclared quality checks")
    data, track_group_diagnostics = attach_candidate_track_groups(data)
    data.to_csv(FEATURES_OUT, index=False, encoding="utf-8-sig")
    camera_participant_folds, camera_participant_summary = score_grouped_cv(data, "camera_speed_mean_source_units_s", "participant_id")
    rig_participant_folds, rig_participant_summary = score_grouped_cv(data, "rig_candidate_speed_mean_source_units_s", "participant_id")
    camera_track_folds, camera_track_summary = score_grouped_cv(data, "camera_speed_mean_source_units_s", "candidate_geometry_group")
    rig_track_folds, rig_track_summary = score_grouped_cv(data, "rig_candidate_speed_mean_source_units_s", "candidate_geometry_group")
    report = [
        "# 过山车视觉到相机运动小规模验证",
        "",
        "本报告为一次探索性流程验证，不是晕动症风险模型，也不代表跨游戏/跨轨道泛化。仅使用本地已存在的5份Epic Roller Coasters录像；没有下载或解压其它数据。",
        "",
        "## 口径",
        "",
        "- 输入：纵向翻转校正后的RGB帧，Farneback光流统计（每个1秒片段最多取5帧，按真实记录时间计算相邻图像间隔）。",
        "- 标签1：相机矩阵得到的相机速度；标签2：按候选Unity/OpenVR变换剔除头姿后的rig速度。两者单位均为**未知源坐标单位/秒**，不能称m/s，也不是FMS标签。",
        "- 校验：camera.csv与实际RGB帧索引按frame ID和毫秒时间精确连接；只保留名字为`Camera`的乘车相机；丢弃相机切换、时间间隔超过0.20秒、位移跳变超过10源单位的区间；评分模型没有参与本试验。",
        "- 模型：Ridge(alpha=10)，对比训练集目标中位数基线。分别按参与者和已有候选轨迹几何组留出；候选轨迹组不是作者提供的真实track ID。",
        f"- 轨迹分组来源：`{track_group_diagnostics['source']}`；窗口分布：`{json.dumps(track_group_diagnostics.get('window_counts', {}), ensure_ascii=False)}`；未知或跨组窗口：{track_group_diagnostics['unknown_windows']}。",
        "",
        "## 纳入情况",
        "",
        "| Session | 参与者 | RGB帧 | camera名计数 | pose最近时间差中位数/p95 (ms) | 清洁相机区间 | 一秒视觉窗口 | 切换/跳变剔除 |",
        "|---|---|---:|---|---:|---:|---:|---|",
    ]
    for a in audits:
        if "error" in a:
            report.append(f"| {a.get('session_id', a.get('session_folder'))} | — | — | — | — | — | — | 失败，见日志 |")
        else:
            switches = a.get("camera_name_switch_pairs", 0)
            jumps = a.get("camera_position_jump_pairs_over_10_source_units", 0)
            report.append(
                f"| {a['session_id']} | {a['participant_id']} | {a['frame_rows_with_rgb']} | "
                f"{json.dumps(a['camera_name_counts'], ensure_ascii=False)} | "
                f"{a['pose_device0_nearest_offset_median_ms']:.1f}/{a['pose_device0_nearest_offset_p95_ms']:.1f} | "
                f"{a['camera_motion_intervals_after_qc']} | {a['one_second_rgb_windows']} | 切换{switches}，相机跳变{jumps} |"
            )
    report += ["", "## 按参与者留出", "", "### 相机矩阵运动（含可能的头动）", "", f"汇总：`{json.dumps(camera_participant_summary, ensure_ascii=False)}`", "", "| 留出参与者 | 测试窗口 | Ridge MAE | 中位数基线 MAE | Ridge Spearman |", "|---|---:|---:|---:|---:|"]
    for f in camera_participant_folds:
        report.append(f"| {f['held_out_group']} | {f['n_test_windows']} | {f['ridge_mae']:.4f} | {f['train_median_baseline_mae']:.4f} | {f['ridge_spearman']} |")
    report += ["", "### 去头动候选rig轨迹", "", f"汇总：`{json.dumps(rig_participant_summary, ensure_ascii=False)}`", "", "| 留出参与者 | 测试窗口 | Ridge MAE | 中位数基线 MAE | Ridge Spearman |", "|---|---:|---:|---:|---:|---|"]
    for f in rig_participant_folds:
        report.append(f"| {f['held_out_group']} | {f['n_test_windows']} | {f['ridge_mae']:.4f} | {f['train_median_baseline_mae']:.4f} | {f['ridge_spearman']} |")
    report += ["", "## 按候选轨迹几何组留出", "", "候选几何组来自世界坐标路径重叠审计，不是人工确认的赛道名称。表中同时列出测试参与者与训练参与者重叠数。", "", f"相机矩阵运动汇总：`{json.dumps(camera_track_summary, ensure_ascii=False)}`", "", "| 留出几何组 | 测试窗口 | 测试参与者 | 训练/测试参与者重叠数 | Ridge MAE | 中位数基线 MAE | Spearman |", "|---|---:|---|---:|---:|---:|---:|"]
    for f in camera_track_folds:
        report.append(f"| {f['held_out_group']} | {f['n_test_windows']} | {f['test_participants']} | {f['participant_overlap_with_train']} | {f['ridge_mae']:.4f} | {f['train_median_baseline_mae']:.4f} | {f['ridge_spearman']} |")
    report += ["", f"候选rig运动汇总：`{json.dumps(rig_track_summary, ensure_ascii=False)}`", "", "| 留出几何组 | 测试窗口 | 测试参与者 | 训练/测试参与者重叠数 | Ridge MAE | 中位数基线 MAE | Spearman |", "|---|---:|---|---:|---:|---:|---:|"]
    for f in rig_track_folds:
        report.append(f"| {f['held_out_group']} | {f['n_test_windows']} | {f['test_participants']} | {f['participant_overlap_with_train']} | {f['ridge_mae']:.4f} | {f['train_median_baseline_mae']:.4f} | {f['ridge_spearman']} |")
    report += [
        "",
        "## 解释与限制",
        "",
        "- 这是在同一款过山车、同一小组重复录制中，检验RGB运动能否对应引擎相机记录的流程试验。即使胜过中位数基线，也只支持这一数据上的运动因素识别，不证明晕动症风险预测能力。",
        "- `rig_candidate`依赖未经正式标定的OpenVR设备身份、坐标轴翻转和相机—头姿组合；若它不稳定，不应拿它生成训练标签。报告同时给出原始相机目标作为敏感性对照。",
        "- 10源单位跳变和0.20秒间隔是数据质量筛选值；速度单位/场景尺度未知。菜单与乘车分开后只留下`Camera`段，仍需人工核对轨道、加载和视角切换。",
        "- FOV在这5段记录中没有有效变化；object.csv没有持久实例ID，因此本轮不训练FOV或物体速度检测器。",
        "- 若要做真正的开发时风险分级，下一步需要跨游戏固定路线视频、已校准的设计路径运动量和同场景/同人评分对应关系。",
        "",
        f"输出：`{FEATURES_OUT.relative_to(ROOT).as_posix()}`。执行日志：`{LOG.relative_to(ROOT).as_posix()}`。",
    ]
    REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
    logging.info("wrote %s (%d windows), %s", FEATURES_OUT, len(data), REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
