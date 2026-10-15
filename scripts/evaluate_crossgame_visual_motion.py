"""Evaluate coaster-trained RGB flow regression on all local City Parkour sessions.

The City archive is read in place; no ZIP extraction is performed. The held-out
game's labels are used only for final metrics, never for fitting or tuning.
Camera speeds remain in uncalibrated source-coordinate units per second.
"""
from __future__ import annotations

import json
import logging
import math
import posixpath
import sys
import time
import traceback
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from parse_vrnet_matrices import camera_pose, choose_camera, numeric_matrix_list, literal_list

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "data/raw/vrnet/samples/City_Parkour.zip"
TRAIN_PATH = ROOT / "features/coaster_motion_visual_pilot.csv"
OUTPUT = ROOT / "features/city_parkour_crossgame_predictions.csv"
REPORT = ROOT / "results/city_parkour_crossgame_visual_motion.md"
LOG = ROOT / "results/city_parkour_crossgame_visual_motion_log.txt"
FEATURES = [
    "rgb_flow_mean_diag_s", "rgb_flow_p90_diag_s", "rgb_flow_p99_diag_s",
    "rgb_flow_pair_mean_std", "rgb_flow_direction_concentration", "rgb_valid_view_fraction",
]
USECOLS = ["frame", "timestamp", "camera_name", "v_matrix", "video"]
IMAGE_WIDTH = 256
MAX_CAMERA_DT = 0.50
MAX_STEP_SOURCE_UNITS = 10.0


def image_gray_from_zip(archive: zipfile.ZipFile, member: str) -> np.ndarray | None:
    try:
        content = archive.read(member)
    except KeyError:
        return None
    img = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    height = max(1, round(img.shape[0] * IMAGE_WIDTH / img.shape[1]))
    # City Parkour source images were visually checked upright; do not apply
    # the vertical flip used for the older Epic Roller Coasters recordings.
    return cv2.resize(img, (IMAGE_WIDTH, height), interpolation=cv2.INTER_AREA)


def flow_features(archive: zipfile.ZipFile, prefix: str, frames: list[tuple[float, str]]) -> dict | None:
    if len(frames) < 4:
        return None
    targets = np.linspace(0.05, 0.95, 5)
    times = np.asarray([t for t, _ in frames], dtype=np.float64)
    selected_indices = []
    for target in targets:
        i = int(np.argmin(np.abs(times - target)))
        if i not in selected_indices:
            selected_indices.append(i)
    selected = [frames[i] for i in sorted(selected_indices)]
    decoded = []
    for t, video_path in selected:
        relative = video_path.replace("\\", "/").lstrip("/")
        if relative.startswith("../") or relative != "video/" + relative.removeprefix("video/"):
            return None
        member = posixpath.join(prefix, relative + ".jpg")
        img = image_gray_from_zip(archive, member)
        if img is not None:
            decoded.append((t, img, member))
    if len(decoded) < 4:
        return None
    vals, coherences, valid_fracs, pair_rates = [], [], [], []
    for i in range(1, len(decoded)):
        dt = decoded[i][0] - decoded[i - 1][0]
        first, second = decoded[i - 1][1], decoded[i][1]
        if dt <= 0 or dt > 0.40 or first.shape != second.shape:
            continue
        flow = cv2.calcOpticalFlowFarneback(first, second, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, theta = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        valid = (first > 8) & (second > 8)
        if int(valid.sum()) < 100:
            continue
        values = mag[valid].astype(np.float64) / (math.hypot(*first.shape) * dt)
        stride = max(1, len(values) // 5000)
        vals.extend(values[::stride].tolist())
        weights, angles = mag[valid].astype(np.float64), theta[valid].astype(np.float64)
        if weights.sum() > 0:
            coherences.append(float(np.hypot(np.sum(weights * np.cos(angles)), np.sum(weights * np.sin(angles))) / weights.sum()))
        valid_fracs.append(float(valid.mean()))
        pair_rates.append(float(values.mean()))
    if len(pair_rates) < 3 or not vals:
        return None
    values = np.asarray(vals, dtype=np.float64)
    return {
        "rgb_flow_mean_diag_s": float(values.mean()),
        "rgb_flow_p90_diag_s": float(np.percentile(values, 90)),
        "rgb_flow_p99_diag_s": float(np.percentile(values, 99)),
        "rgb_flow_pair_mean_std": float(np.std(pair_rates)),
        "rgb_flow_direction_concentration": float(np.mean(coherences)),
        "rgb_valid_view_fraction": float(np.mean(valid_fracs)),
        "n_flow_pairs": int(len(pair_rates)),
        "n_selected_frames": int(len(decoded)),
        "selected_image_members": json.dumps([x[2] for x in decoded]),
    }


def read_session(archive: zipfile.ZipFile, csv_info: zipfile.ZipInfo) -> tuple[list[dict], dict]:
    prefix = csv_info.filename.rsplit("/", 1)[0]
    session = prefix.split("/")[-1]
    camera_rows, video_rows = [], []
    row_count = video_count = camera_count = matrix_errors = 0
    first_time = None
    # Chunked CSV parsing limits memory for the larger sessions.
    with archive.open(csv_info) as stream:
        for chunk in pd.read_csv(stream, usecols=USECOLS, chunksize=10000, low_memory=False):
            row_count += len(chunk)
            for row in chunk.itertuples(index=False):
                if pd.isna(row.timestamp):
                    continue
                timestamp = int(row.timestamp)
                if first_time is None:
                    first_time = timestamp
                elapsed = (timestamp - first_time) / 1000.0
                video_value = row.video if isinstance(row.video, str) else None
                if video_value:
                    video_rows.append((elapsed, int(row.frame), video_value))
                    video_count += 1
                if not isinstance(row.camera_name, str) or not isinstance(row.v_matrix, str):
                    continue
                try:
                    names = literal_list(row.camera_name)
                    matrices = numeric_matrix_list(row.v_matrix)
                    cam_i = choose_camera(names)
                    if cam_i is None or cam_i >= len(matrices):
                        raise ValueError("No selected camera matrix")
                    pos, _, _ = camera_pose(matrices[cam_i])
                    camera_rows.append((timestamp, elapsed, int(row.frame), str(names[cam_i]), pos))
                    camera_count += 1
                except Exception:
                    matrix_errors += 1
    if not camera_rows or not video_rows:
        raise ValueError(f"No usable camera or video rows in {session}")
    intervals = []
    segment_id = 0
    for i in range(1, len(camera_rows)):
        a, b = camera_rows[i - 1], camera_rows[i]
        dt = (b[0] - a[0]) / 1000.0
        if a[3] != b[3] or dt <= 0 or dt > MAX_CAMERA_DT:
            segment_id += 1
            continue
        disp = float(np.linalg.norm(b[4] - a[4]))
        if disp > MAX_STEP_SOURCE_UNITS:
            segment_id += 1
            continue
        intervals.append({
            "t0": a[1], "t1": b[1], "t_mid": (a[1] + b[1]) / 2,
            "dt": dt, "speed": disp / dt, "segment": segment_id,
        })
    if not intervals:
        raise ValueError(f"No continuous camera intervals in {session}")
    intervals_df = pd.DataFrame(intervals)
    video_df = pd.DataFrame(video_rows, columns=["elapsed", "frame", "video"])
    windows = []
    duration = max(float(video_df.elapsed.max()), float(intervals_df.t1.max()))
    for bin_id in range(int(math.floor(duration))):
        start, end = float(bin_id), float(bin_id + 1)
        iv = intervals_df.loc[(intervals_df.t_mid >= start) & (intervals_df.t_mid < end)]
        if len(iv) < 4 or iv.segment.nunique() != 1:
            continue
        mids, dts = iv.t_mid.to_numpy(float), iv.dt.to_numpy(float)
        if mids.min() > start + 0.25 or mids.max() < end - 0.25 or dts.max() > MAX_CAMERA_DT:
            continue
        fr = video_df.loc[(video_df.elapsed >= start) & (video_df.elapsed < end)]
        image_samples = [(float(r.elapsed - start), str(r.video)) for r in fr.itertuples(index=False)]
        feats = flow_features(archive, prefix, image_samples)
        if feats is None:
            continue
        windows.append({
            "game": "City_Parkour", "session_id": session,
            "participant_id": session.split("_")[0],
            "window_id": f"{session}_{bin_id:05d}", "start_sec": start,
            "camera_speed_mean_source_units_s": float(iv.speed.mean()),
            "camera_speed_p90_source_units_s": float(iv.speed.quantile(.90)),
            "n_camera_intervals": int(len(iv)),
            **feats,
        })
    audit = {
        "session_id": session, "csv_uncompressed_bytes": int(csv_info.file_size),
        "csv_rows": int(row_count), "camera_matrix_rows_parsed": int(camera_count),
        "camera_matrix_parse_errors": int(matrix_errors), "video_rows": int(video_count),
        "usable_camera_intervals": int(len(intervals)), "one_second_windows": int(len(windows)),
        "camera_dt_median_sec": float(intervals_df.dt.median()),
        "camera_speed_quantiles_source_units_s": {
            str(q): float(intervals_df.speed.quantile(q)) for q in [0, .25, .5, .75, .9, .99, 1]
        },
        "discarded_large_displacement_pairs": int(sum(
            1 for i in range(1, len(camera_rows))
            if camera_rows[i][3] == camera_rows[i - 1][3]
            and (camera_rows[i][0] - camera_rows[i - 1][0]) / 1000 > 0
            and np.linalg.norm(camera_rows[i][4] - camera_rows[i - 1][4]) > MAX_STEP_SOURCE_UNITS
        )),
    }
    return windows, audit


def metric_block(actual: np.ndarray, prediction: np.ndarray) -> dict:
    return {
        "n": int(len(actual)),
        "mae": float(mean_absolute_error(actual, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(actual, prediction))),
        "spearman": float(spearmanr(actual, prediction).statistic) if np.std(actual) > 0 and np.std(prediction) > 0 else None,
    }


def main() -> int:
    if not ZIP_PATH.is_file() or not TRAIN_PATH.is_file():
        raise FileNotFoundError(f"Required local ZIP or coaster training feature file missing: {ZIP_PATH}, {TRAIN_PATH}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(LOG, mode="w", encoding="utf-8"), logging.StreamHandler(sys.stdout)])
    cv2.setNumThreads(1)
    cv2.setRNGSeed(42)
    train = pd.read_csv(TRAIN_PATH)
    train = train.dropna(subset=FEATURES + ["camera_speed_mean_source_units_s"])
    if train.empty:
        raise ValueError("Coaster pilot training table has no complete rows")
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))
    target = "camera_speed_mean_source_units_s"
    model.fit(train[FEATURES], train[target])
    training_median = float(train[target].median())
    logging.info("trained fixed Ridge baseline on %d coaster windows; no City labels used for fitting", len(train))

    all_windows, audits = [], []
    with zipfile.ZipFile(ZIP_PATH) as archive:
        csv_files = sorted((i for i in archive.infolist() if i.filename.endswith("/data_file.csv")), key=lambda x: x.filename)
        if len(csv_files) != 10:
            raise ValueError(f"Expected the 10 local City Parkour session CSVs, found {len(csv_files)}")
        for info in csv_files:
            started = time.perf_counter()
            try:
                windows, audit = read_session(archive, info)
                all_windows.extend(windows)
                audits.append(audit)
                logging.info("%s: %d rows, %d camera rows, %d visual windows in %.1fs",
                             audit["session_id"], audit["csv_rows"], audit["camera_matrix_rows_parsed"],
                             audit["one_second_windows"], time.perf_counter() - started)
            except Exception:
                error = traceback.format_exc()
                logging.error("FAILED %s\n%s", info.filename, error)
                audits.append({"session_id": info.filename.rsplit("/", 1)[0], "error": error})
    data = pd.DataFrame(all_windows)
    if data.empty:
        raise RuntimeError("No City Parkour image/matrix windows passed the predeclared data-quality filters")
    data["prediction_coaster_model"] = model.predict(data[FEATURES])
    data["prediction_training_median"] = training_median
    data.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    actual = data[target].to_numpy(float)
    predictions = data.prediction_coaster_model.to_numpy(float)
    baseline = np.full(len(data), training_median)
    overall_model = metric_block(actual, predictions)
    overall_baseline = metric_block(actual, baseline)
    fold_rows = []
    for session, rows in data.groupby("session_id", sort=True):
        y = rows[target].to_numpy(float)
        p = rows.prediction_coaster_model.to_numpy(float)
        b = np.full(len(rows), training_median)
        fold_rows.append({"session_id": session, "n_windows": len(rows),
                          "model": metric_block(y, p), "training_median": metric_block(y, b)})

    report = [
        "# City Parkour 跨游戏图像到相机运动验证",
        "",
        "这是一次固定模型的跨游戏外部验证：训练只使用Epic Roller Coasters的1秒光流窗口，测试使用本地City Parkour ZIP中的全部10个session。City的目标值只用于最终评估；没有用它调参。ZIP直接读取，没有解压到磁盘。",
        "",
        "## 任务和口径",
        "",
        f"- 训练样本：过山车{len(train)}个窗口；模型为预先固定的标准化Ridge(alpha=10)。",
        "- 测试输入：City Parkour原始视频图像的Farneback光流统计。抽看的City图像方向正立，因此未做过山车旧数据使用的竖直翻转。",
        "- 预测目标：从CenterEyeAnchor相机`v_matrix`推导的一秒平均相机速度；相机运动可能包含用户头部转动。速度是**未校准的源坐标单位/秒**，不是m/s，也不是晕动症评分。",
        "- 时间对齐：只按CSV中的实际timestamp配对矩阵和video路径；未用frame编号假设帧率。相机缺口超过0.50秒、连续位移超过10源单位、相机身份改变的区间被切断；每秒至少4个有效相机区间及4帧用于光流。",
        "- 标签污染检查：`MS_rating`没有作为输入或训练标签。",
        "",
        "## 跨游戏结果",
        "",
        f"模型：`{json.dumps(overall_model, ensure_ascii=False)}`",
        f"过山车训练集固定中位数基线：`{json.dumps(overall_baseline, ensure_ascii=False)}`",
        "",
        "| City session | 窗口数 | 模型MAE | 中位数基线MAE | 模型Spearman |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in fold_rows:
        report.append(f"| {row['session_id']} | {row['n_windows']} | {row['model']['mae']:.4f} | {row['training_median']['mae']:.4f} | {row['model']['spearman']} |")
    report += ["", "## 数据覆盖", "", "| Session | CSV行 | 相机矩阵行 | 视频行 | 矩阵间隔中位数(s) | 有效窗口 | 跳过的坐标大跳变区间 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in audits:
        if "error" in row:
            report.append(f"| {row['session_id']} | — | — | — | — | 0 | 失败，见日志 |")
        else:
            report.append(f"| {row['session_id']} | {row['csv_rows']} | {row['camera_matrix_rows_parsed']} | {row['video_rows']} | {row['camera_dt_median_sec']:.3f} | {row['one_second_windows']} | {row['discarded_large_displacement_pairs']} |")
    report += [
        "",
        "## 结论",
        "",
        "- 这次直接检验了“过山车训练的图像模型，能否预测未参与训练的City Parkour相机运动数据”。MAE与简单训练集基线的关系、Spearman相关和各session差异共同决定是否存在跨游戏运动信号。",
        "- 即使模型胜过基线，目标仍是引擎相机运动，不是晕动症风险；场景尺度未知会影响绝对误差，头动也没有从City目标中分离。",
        "- 若跨游戏结果不佳，优先怀疑运动目标/坐标尺度/图像分布不一致，再决定扩展模型。若有稳定排序关系，再加入真实评分验证风险关联。",
        "",
        f"逐窗口预测：`{OUTPUT.relative_to(ROOT).as_posix()}`。执行日志：`{LOG.relative_to(ROOT).as_posix()}`。",
    ]
    REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
    logging.info("wrote %s (%d City windows), %s", OUTPUT, len(data), REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
