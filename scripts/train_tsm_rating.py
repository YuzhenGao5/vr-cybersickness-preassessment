#!/usr/bin/env python3
"""Frozen official TSM embeddings + fixed logistic heads on genuine rating events.

This is NOT full-network TSM fine-tuning. Each event has three strictly past
RGB clips (5/30/60 seconds, eight frames each) supplied by the audited manifest.
No downloads occur here. All three heads use exactly the same event cohort.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import platform
import random
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                    ToTensor, InterpolationMode)

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_COMMIT = "1f6a834b02120441d99fa706b7a13ccc5293c6fa"
HISTORY_SECONDS = [5, 30, 60]
SEGMENTS = 8
EMBEDDING_DIM = 2048
VARIANCE_THRESHOLD = 1e-10
HEAD_C = 0.1
UNKNOWN_PEOPLE = {"", "unknown", "nan", "none", "null", "<na>"}
PREPROCESS = dict(resize_short_edge=256, center_crop=224, interpolation="bilinear",
                  rgb_mean=[0.485, 0.456, 0.406], rgb_std=[0.229, 0.224, 0.225],
                  history_seconds=HISTORY_SECONDS, segments=SEGMENTS,
                  consensus="mean_segments", pooling="concat_three_histories",
                  augmentation="none", backbone_training="frozen_eval")
ARCHITECTURE = dict(base_model="resnet50", num_class=400, num_segments=8,
                    modality="RGB", dropout=0.8, consensus_type="avg",
                    is_shift=True, shift_div=8, shift_place="blockres",
                    temporal_pool=False, non_local=False, pretrain=None)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(clean_json(value), ensure_ascii=False,
                                    indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_manifest(path, baseline_columns, data_root):
    ids = ["event_id", "game_name", "participant_id", "session_id"]
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={name: str for name in ids})
    required = set(ids + ["binary_true", "fms_raw", "exposure_elapsed_seconds", "rating_time_sec",
                          "frame_paths_json", "frame_times_json", "frame_ids_json",
                          "window_seconds_json"] + baseline_columns)
    if required - set(frame):
        raise ValueError(f"manifest missing columns: {sorted(required - set(frame))}")
    if frame.empty or frame[ids].isna().any().any() or not frame.event_id.is_unique:
        raise ValueError("manifest must contain unique genuine event IDs and complete identities")
    if frame.game_name.nunique() < 2:
        raise ValueError("Leave-One-Game-Out needs at least two games; do not claim multi-game validation")
    if frame.participant_id.str.lower().isin(UNKNOWN_PEOPLE).any():
        raise ValueError("known participant IDs are required for participant-purged sensitivity")
    for col in ["binary_true", "fms_raw", "rating_time_sec", *baseline_columns]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if (not np.isfinite(frame.fms_raw).all() or not frame.binary_true.isin([0, 1]).all()
            or not np.array_equal(frame.binary_true.to_numpy(), (frame.fms_raw > 2).astype(int))):
        raise ValueError("binary_true must equal real fms_raw > 2; labels cannot be missing or regenerated")
    if (frame.groupby("session_id").game_name.nunique().gt(1).any()
            or frame.groupby("session_id").participant_id.nunique().gt(1).any()):
        raise ValueError("session must have exactly one game and one participant")
    if (not np.isfinite(frame.exposure_elapsed_seconds).all() or frame.exposure_elapsed_seconds.lt(0).any()
            or not np.isfinite(frame.rating_time_sec).all()):
        raise ValueError("elapsed time must be finite and nonnegative")
    frame[baseline_columns] = frame[baseline_columns].replace([np.inf, -np.inf], np.nan)
    clips = []
    for row in frame.itertuples(index=False):
        spec = json.loads(row.frame_paths_json)
        times = json.loads(row.frame_times_json)
        numbers = json.loads(row.frame_ids_json)
        windows = json.loads(row.window_seconds_json)
        if (not isinstance(spec, list) or len(spec) != 3
                or any(not isinstance(part, list) or len(part) != SEGMENTS for part in spec)):
            raise ValueError(f"{row.event_id}: frame_paths_json must be three clips of eight frames")
        if (windows != HISTORY_SECONDS or not isinstance(times, list) or len(times) != 3
                or not isinstance(numbers, list) or len(numbers) != 3):
            raise ValueError(f"{row.event_id}: incorrect temporal metadata or history order")
        resolved = []
        for history_index, part in enumerate(spec):
            stamp = np.asarray(times[history_index], dtype=float)
            ids = numbers[history_index]
            if (stamp.shape != (SEGMENTS,) or not np.isfinite(stamp).all()
                    or not np.all(np.diff(stamp) > 0) or not np.all(stamp < row.rating_time_sec)
                    or not np.all(stamp >= row.rating_time_sec - HISTORY_SECONDS[history_index])
                    or not isinstance(ids, list) or len(ids) != SEGMENTS or len(set(ids)) != SEGMENTS):
                raise ValueError(f"{row.event_id}: nonunique/unordered/outside-past-window frame metadata")
            if len(set(part)) != SEGMENTS:
                raise ValueError(f"{row.event_id}: a clip cannot repeat a frame to hide insufficient coverage")
            paths = []
            for value in part:
                if not isinstance(value, str) or Path(value).is_absolute():
                    raise ValueError(f"{row.event_id}: frame paths must be relative to --data-root")
                candidate = (data_root / value).resolve()
                if not candidate.is_relative_to(data_root) or not candidate.is_file():
                    raise FileNotFoundError(f"{row.event_id}: missing/escaping frame: {value}")
                if candidate.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    raise ValueError(f"{row.event_id}: unsupported RGB image: {value}")
                paths.append(candidate)
            resolved.append(paths)
        clips.append(resolved)
    return frame, clips


def validate_config(args):
    config = json.loads(args.config.read_text(encoding="utf-8"))
    expected = {"seed": args.seed, "classification_threshold": 0.5,
                "history_windows_seconds": HISTORY_SECONDS, "frames_per_window": SEGMENTS,
                "tsm_commit": args.expected_tsm_commit, "methods": ["parameters", "video", "fusion"]}
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"frozen config mismatch for {name}: expected {value}, found {config.get(name)}")
    if config.get("history_interval") != "[rating_time - window, rating_time)":
        raise ValueError("frozen config must use strictly past rating windows")
    if config.get("label") != "observed real MS_rating > 2; values <= 2 are class 0":
        raise ValueError("frozen label config mismatch")
    if config.get("head") != "L2 LogisticRegression; C=0.1; equal session total weights then training-only binary class balancing":
        raise ValueError("head must match registered C=0.1 session and class weighting")
    return config


def preparation_summary(manifest):
    sidecar = manifest.parent / "summary.json"
    result = json.loads(sidecar.read_text(encoding="utf-8-sig"))
    if (result.get("status") != "complete" or result.get("manifest_sha256") != sha256(manifest)
            or result.get("windows_sec") != HISTORY_SECONDS or result.get("frames_per_window") != SEGMENTS
            or result.get("accepted_events", 0) < 1):
        raise ValueError("preparation summary missing, incompatible, or manifest hash does not match")
    return result, sidecar


class EventClips(Dataset):
    def __init__(self, clips):
        self.clips = clips
        self.transform = Compose([Resize(256, interpolation=InterpolationMode.BILINEAR),
                                  CenterCrop(224), ToTensor(),
                                  Normalize(PREPROCESS["rgb_mean"], PREPROCESS["rgb_std"])])

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        histories = []
        for paths in self.clips[index]:
            frames = []
            for path in paths:
                with Image.open(path) as image:
                    frames.append(self.transform(image.convert("RGB")))
            histories.append(torch.stack(frames))
        return torch.stack(histories)


def repo_metadata(repo, expected_commit):
    required = ["ops/models.py", "ops/transforms.py", "ops/temporal_shift.py", "ops/basic_ops.py"]
    for value in required:
        if not (repo / value).is_file():
            raise FileNotFoundError(f"official TSM source missing: {repo / value}")
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    if commit != expected_commit:
        raise ValueError(f"TSM commit mismatch: expected {expected_commit}, found {commit}")
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--", *required],
                           check=True, capture_output=True, text=True).stdout.strip()
    if dirty:
        raise ValueError(f"official TSM source has local edits: {dirty}")
    return dict(commit=commit, source_sha256={value: sha256(repo / value) for value in required})


def make_backbone(repo, weights, device):
    # Official TSN constructor with pretrain=None prevents an ImageNet download.
    sys.path.insert(0, str(repo))
    TSN = importlib.import_module("ops.models").TSN
    model = TSN(**ARCHITECTURE, partial_bn=False, print_spec=False)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise ValueError("checkpoint must contain an official state_dict")
    state = {key.removeprefix("module."): value for key, value in state.items()}
    if "base_model.classifier.weight" in state:
        state["new_fc.weight"] = state.pop("base_model.classifier.weight")
        state["new_fc.bias"] = state.pop("base_model.classifier.bias")
    model.load_state_dict(state, strict=True)
    # Keep all Kinetics backbone weights; discard only the action classifier.
    # Official TSN averages the eight segment outputs after this identity head.
    model.new_fc = torch.nn.Identity()
    model.eval().requires_grad_(False)
    return model.to(device)


def extract_embeddings(frame, clips, args, metadata):
    cache = args.output_dir / "embeddings.npz"
    unique = sorted({path for event in clips for history in event for path in history})
    image_digest = hashlib.sha256()
    for index, path in enumerate(unique, start=1):
        image_digest.update(path.relative_to(args.data_root).as_posix().encode("utf-8"))
        image_digest.update(sha256(path).encode("ascii"))
        if index % 500 == 0:
            logging.info("image content hashes: %s/%s", index, len(unique))
    metadata["selected_image_content_sha256"] = image_digest.hexdigest()
    cache_key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8")).hexdigest()
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as saved:
            valid = (str(saved["cache_key"].item()) == cache_key
                     and np.array_equal(saved["event_ids"], frame.event_id.astype(str).to_numpy()))
            embeddings = saved["embeddings"].copy()
        if valid and embeddings.shape == (len(frame), 3 * EMBEDDING_DIM) and np.isfinite(embeddings).all():
            logging.info("reusing verified frozen embeddings: %s", cache)
            return embeddings
        logging.warning("embedding cache key/schema mismatch; re-extracting")
    device = torch.device(args.device)
    backbone = make_backbone(args.tsm_repo, args.weights, device)
    loader = DataLoader(EventClips(clips), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    outputs = []
    with torch.inference_mode():
        for index, batch in enumerate(loader, start=1):
            # Event dimension B, three histories, eight ordered RGB frames.
            count = batch.shape[0]
            values = batch.reshape(count * 3, SEGMENTS, 3, 224, 224).to(device, non_blocking=True)
            result = backbone(values)
            if result.shape != (count * 3, EMBEDDING_DIM):
                raise ValueError(f"unexpected official TSM embedding shape: {tuple(result.shape)}")
            outputs.append(result.reshape(count, 3 * EMBEDDING_DIM).cpu().numpy())
            logging.info("frozen TSM embedding batches: %s/%s (%s events)", index, len(loader),
                         min(index * args.batch_size, len(frame)))
    embeddings = np.concatenate(outputs).astype(np.float32)
    if not np.isfinite(embeddings).all():
        raise ValueError("nonfinite TSM embeddings; refusing training")
    temporary = args.output_dir / "embeddings.partial.npz"
    np.savez_compressed(temporary, cache_key=np.asarray(cache_key),
                        event_ids=frame.event_id.astype(str).to_numpy(dtype=str), embeddings=embeddings)
    temporary.replace(cache)
    save_json(args.output_dir / "embedding_cache_metadata.json", metadata)
    return embeddings


def session_weights(frame):
    values = 1.0 / frame.groupby("session_id")["event_id"].transform("size").to_numpy(dtype=float)
    return values / values.mean()


def fit_head(frame, columns, log_columns, seed):
    y = frame.binary_true.to_numpy(dtype=int)
    base = session_weights(frame)
    prior = float(np.average(y, weights=base))
    usable = [name for name in columns if frame[name].notna().any()]
    detail = dict(input_feature_cols=usable, requested_feature_cols=columns,
                  dropped_all_missing=[name for name in columns if name not in usable],
                  fitted_feature_cols=[], C=HEAD_C, seed=seed, class_balance="balanced",
                  weight_policy="equal total session weight; training-only weighted class balancing",
                  constant_probability=None, training_prior=prior, status="fitted")
    if np.unique(y).size < 2 or not usable:
        detail.update(status="prior_only_single_train_class" if np.unique(y).size < 2 else "prior_only_no_features",
                      constant_probability=prior)
        return None, detail
    logs = [name for name in log_columns if name in usable]
    if logs and frame[logs].lt(0).any().any():
        raise ValueError("registered nonnegative baseline feature is negative; cannot apply log1p")
    ordered = logs + [name for name in usable if name not in logs]
    preprocess = ColumnTransformer([("log1p", FunctionTransformer(np.log1p),
                                     [usable.index(name) for name in logs])], remainder="passthrough")
    imputer = SimpleImputer(strategy="median")
    prepared = preprocess.fit_transform(imputer.fit_transform(frame[usable]))
    variance = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    try:
        variance.fit(prepared)
    except ValueError:
        detail.update(status="prior_only_constant_features", constant_probability=prior)
        return None, detail
    detail["fitted_feature_cols"] = [name for name, keep in zip(ordered, variance.get_support()) if keep]
    weights = base.copy()
    masses = [float(weights[y == label].sum()) for label in [0, 1]]
    factors = [float(weights.sum() / (2.0 * mass)) for mass in masses]
    weights *= np.asarray(factors)[y]
    detail.update(training_class_mass=masses, class_factors=factors,
                  log1p_columns=logs, transformed_feature_order=ordered)
    pipe = Pipeline([("imputer", imputer), ("registered_log1p", preprocess), ("variance", variance),
                     ("scaler", StandardScaler()),
                     ("model", LogisticRegression(C=HEAD_C, solver="lbfgs", penalty="l2",
                                                  max_iter=5000, random_state=seed))])
    pipe.fit(frame[usable], y, model__sample_weight=weights)
    return pipe, detail


def predict_head(pipe, detail, frame):
    if pipe is None:
        return np.full(len(frame), detail["constant_probability"], dtype=float)
    index = list(pipe.named_steps["model"].classes_).index(1)
    return pipe.predict_proba(frame[detail["input_feature_cols"]])[:, index]


def scores(y, probability):
    if len(y) == 0:
        return dict(auc=None, auc_status="NA: no estimable OOF predictions", accuracy=None,
                    f1_macro=None, average_precision=None, brier=None,
                    tn=None, fp=None, fn=None, tp=None)
    predicted = (probability >= 0.5).astype(int)
    cm = confusion_matrix(y, predicted, labels=[0, 1])
    both = np.unique(y).size == 2
    return dict(auc=float(roc_auc_score(y, probability)) if both else None,
                auc_status="computable" if both else "NA: test contains one class",
                accuracy=float(accuracy_score(y, predicted)),
                f1_macro=float(f1_score(y, predicted, labels=[0, 1], average="macro", zero_division=0)),
                average_precision=float(average_precision_score(y, probability)) if both else None,
                brier=float(brier_score_loss(y, probability)),
                tn=int(cm[0, 0]), fp=int(cm[0, 1]), fn=int(cm[1, 0]), tp=int(cm[1, 1]))


def evaluate(frame, methods, seed, output):
    folds, predictions, summaries = [], [], []
    for protocol in ["leave_one_game_out", "leave_one_game_out_participant_purged"]:
        for method, (columns, log_columns) in methods.items():
            method_folds, method_predictions = [], []
            for game in sorted(frame.game_name.unique()):
                test = frame.loc[frame.game_name.eq(game)].copy()
                training = frame.loc[~frame.game_name.eq(game)].copy()
                removed = 0
                if protocol.endswith("participant_purged"):
                    before = len(training)
                    training = training.loc[~training.participant_id.isin(set(test.participant_id))].copy()
                    removed = before - len(training)
                if training.empty:
                    reason = "NA: no training events after participant purge" if removed else "NA: no training events"
                    method_folds.append(dict(protocol=protocol, method=method, test_game=game,
                                             train_events=0, train_sessions=0, train_games=0,
                                             test_events=len(test), test_sessions=test.session_id.nunique(),
                                             test_positives=int(test.binary_true.sum()),
                                             purged_train_events=removed, head_status="unestimable_no_training_events",
                                             effective_features=0, auc=None, auc_status=reason,
                                             accuracy=None, f1_macro=None, average_precision=None,
                                             brier=None, tn=None, fp=None, fn=None, tp=None))
                    logging.warning("%s %s heldout %s: %s; no predictions created", protocol, method, game, reason)
                    continue
                pipe, detail = fit_head(training, columns, log_columns, seed)
                probability = predict_head(pipe, detail, test)
                item = dict(protocol=protocol, method=method, test_game=game,
                            train_events=len(training), train_sessions=training.session_id.nunique(),
                            train_games=training.game_name.nunique(), test_events=len(test),
                            test_sessions=test.session_id.nunique(), test_positives=int(test.binary_true.sum()),
                            purged_train_events=removed, head_status=detail["status"],
                            effective_features=len(detail["fitted_feature_cols"]),
                            **scores(test.binary_true.to_numpy(dtype=int), probability))
                method_folds.append(item)
                for event, score in zip(test.itertuples(index=False), probability):
                    method_predictions.append(dict(protocol=protocol, method=method, event_id=event.event_id,
                                                   game_name=game, session_id=event.session_id,
                                                   participant_id=event.participant_id, fms_raw=event.fms_raw,
                                                   binary_true=int(event.binary_true), score=float(score),
                                                   predicted=int(score >= 0.5)))
                logging.info("%s %s heldout %s AUC=%s macro-F1=%.4f", protocol, method, game,
                             item["auc"], item["f1_macro"])
            prediction_columns = ["protocol", "method", "event_id", "game_name", "session_id",
                                  "participant_id", "fms_raw", "binary_true", "score", "predicted"]
            pred = pd.DataFrame(method_predictions, columns=prediction_columns)
            valid_aucs = [value["auc"] for value in method_folds if value["auc"] is not None]
            valid_f1 = [value["f1_macro"] for value in method_folds if value["f1_macro"] is not None]
            summary = dict(protocol=protocol, method=method, n_events=len(pred),
                           n_sessions=pred.session_id.nunique(), n_games=pred.game_name.nunique(),
                           expected_events=len(frame), missing_oof_events=len(frame) - len(pred),
                           expected_games=frame.game_name.nunique(), oof_complete=len(pred) == len(frame),
                           computable_auc_folds=len(valid_aucs), total_folds=len(method_folds),
                           computable_f1_folds=len(valid_f1), unestimable_folds=len(method_folds) - len(valid_f1),
                           mean_fold_auc=float(np.mean(valid_aucs)) if valid_aucs else None,
                           mean_fold_f1=float(np.mean(valid_f1)) if valid_f1 else None,
                           **{f"pooled_{key}": value for key, value in
                              scores(pred.binary_true.to_numpy(dtype=int), pred.score.to_numpy()).items()})
            folds.extend(method_folds)
            predictions.extend(method_predictions)
            summaries.append(summary)
            # Persist completed method results so long-running jobs retain progress.
            pd.DataFrame(folds).to_csv(output / "fold_metrics.csv", index=False)
            pd.DataFrame(predictions, columns=prediction_columns).to_csv(output / "oof_predictions.csv", index=False)
            pd.DataFrame(summaries).to_csv(output / "metrics.csv", index=False)
    return pd.DataFrame(summaries)


def write_report(output, summary, frame, metadata):
    lines = ["# VR.net 冻结 TSM 骨干与评分分类头对照", "",
             "**探索性，非最终结论。TSM 骨干完全冻结；只训练固定正则化逻辑回归分类头。**", "",
             f"- 实际样本：{len(frame)} 次真实评分事件、{frame.session_id.nunique()} 个 session、"
             f"{frame.game_name.nunique()} 个游戏、{frame.participant_id.nunique()} 个参与者 ID。",
             f"- 标签：MS_rating ≤2 为 0（{int((frame.binary_true == 0).sum())}）；>2 为 1"
             f"（{int((frame.binary_true == 1).sum())}）。无评分复制增加样本。",
             "- 每个评分前 [t−5,t)、[t−30,t)、[t−60,t) 各取 8 帧；每窗 2048 维，拼接为 6144 维。",
             "- 三种输入：parameters（原参数＋头动白名单）、video（TSM）、fusion（两者合并）。",
             "- 三方法在完全相同的有视频事件上评估；原 0.682 所用数据可能更多，不能直接同比。",
             "- 固定 C=0.1、训练会话等权并按训练类别加权；填补、log1p、常量筛除和标准化只拟合训练折。",
             "- 没有外折调参、没有模型择优后再次宣称独立测试。LOGO 与参与者排除敏感性均报告。", "",
             "| 协议 | 输入 | 实际 OOF 事件 | pooled OOF AUC | 平均游戏 AUC | macro F1 | Accuracy | 可算 AUC 折 | 可算 F1 折 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(value):
        return "NA" if value is None or pd.isna(value) else f"{float(value):.4f}"
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.protocol} | {row.method} | {row.n_events}/{row.expected_events} "
                     f"| {fmt(row.pooled_auc)} | {fmt(row.mean_fold_auc)} "
                     f"| {fmt(row.pooled_f1_macro)} | {fmt(row.pooled_accuracy)} "
                     f"| {row.computable_auc_folds}/{row.total_folds} "
                     f"| {row.computable_f1_folds}/{row.total_folds} |")
    lines += ["", "## 如何读取输出", "",
              "- metrics.csv 是已留出游戏的 OOF 汇总；fold_metrics.csv 包含每游戏混淆矩阵与单类 AUC 的 NA 原因。",
              "- 若参与者排除后某折没有训练事件，该折记录不可估计且不生成预测；其它折继续。",
              "- pooled 指标仅包含实际有预测的事件；表中列出实际／预期事件数与可计算 F1 折数。",
              "- 若 OOF 缺折，排除参与者后的 cohort 不完整，不能把差异全归因于协议；若全部不可估计，汇总指标为 NA。",
              "- *_head.pkl 是全样本拟合的部署分类头与预处理流程，不能拿其训练集分数当验证分数。",
              "- 视频或融合推理还需要官方 TSM .pth、对应 Git commit、RGB 预处理与相同历史帧采样方法。",
              "- embeddings.npz 是中间视频特征，单独的 pkl 不包含视频骨干权重。",
              "- 分类输出是未校准分数，不是连续 FMS，也不是经过人体验证的晕动概率。", "",
              "## 限制", "",
              "- 这项实验使用用户体验后的录像；参数和融合模型还含用户头动，未验证无人进入时的 Unity 事前评估。",
              "- 8 帧跨 60 秒采样不能完整表示连续细运动；Kinetics 动作预训练也未保证适合 VR 晕动任务。",
              "- 6144 维视频表示对应有限真实评分，虽有 L2 正则化仍可能过拟合；本轮固定方案没有保证性能提升。",
              "- 中心裁剪沿用官方单裁剪基线，可能丢失 VR 周边画面；结果不应称最佳视频表示。",
              "- LOGO 中同一参与者可在不同游戏出现，故还报告排除测试参与者的敏感性。",
              "- 缺视频事件已由 manifest 阶段排除，缺失机制和 cohort 大小会影响结论。",
              "- 归因与设计参数因果关系尚未验证；本脚本不生成 SHAP 或 Grad-CAM。", "",
              "## 可复现来源", "",
              f"- Official TSM commit: `{metadata['tsm_repo']['commit']}`",
              f"- Weights SHA256: `{metadata['weights_sha256']}`",
              f"- Manifest SHA256: `{metadata['manifest_sha256']}`",
              "- 官方源码：https://github.com/mit-han-lab/temporal-shift-module",
              "- 官方 RGB8 Kinetics checkpoint：https://hanlab18.mit.edu/projects/tsm/models/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tsm-repo", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/tsm_rating_pilot_2026-10-12.json")
    parser.add_argument("--baseline-config", type=Path, default=ROOT / "configs/auc_signal_round2_2026-10-10.json")
    parser.add_argument("--expected-tsm-commit", default=OFFICIAL_COMMIT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name in ["manifest", "data_root", "tsm_repo", "weights", "output_dir", "baseline_config", "config"]:
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.output_dir / "training.log", encoding="utf-8")])
    runtime = dict(status="started", started_utc=utc_now(), task="frozen_tsm_fixed_logistic_heads",
                   seed=args.seed, python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                   sklearn=sklearn.__version__, torch=torch.__version__, torchvision=torchvision.__version__,
                   cuda_runtime=torch.version.cuda, cuda_available=torch.cuda.is_available(),
                   requested_device=args.device, arguments={key: str(value) if isinstance(value, Path) else value
                                                            for key, value in vars(args).items()})
    save_json(args.output_dir / "runtime.json", runtime)
    try:
        if args.batch_size < 1 or args.num_workers < 0:
            raise ValueError("batch-size must be positive and num-workers nonnegative")
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; fix server environment or explicitly choose --device cpu")
        if device.type not in {"cuda", "cpu"}:
            raise ValueError("supported devices: cuda, cuda:N or explicit cpu")
        if device.type == "cuda":
            runtime["gpu_name"] = torch.cuda.get_device_name(device)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        config = validate_config(args)
        save_json(args.output_dir / "frozen_training_config.json", config)
        baseline = json.loads(args.baseline_config.read_text(encoding="utf-8"))["methods"]["scene_head_multiscale"]
        columns, logs = baseline["feature_columns"], baseline["log1p_columns"]
        if len(columns) != 73 or len(set(columns)) != 73 or not set(logs) <= set(columns):
            raise ValueError("baseline registration must contain the original 73 raw columns and log whitelist")
        preparation, sidecar = preparation_summary(args.manifest)
        frame, clips = load_manifest(args.manifest, columns, args.data_root)
        if (preparation["accepted_events"] != len(frame)
                or preparation["games"] != frame.game_name.nunique()
                or preparation["sessions"] != frame.session_id.nunique()):
            raise ValueError("manifest counts differ from verified preparation summary")
        metadata = dict(manifest_sha256=sha256(args.manifest), weights_sha256=sha256(args.weights),
                        preparation_summary_sha256=sha256(sidecar),
                        frozen_config_sha256=sha256(args.config),
                        baseline_config_sha256=sha256(args.baseline_config),
                        encoder_script_sha256=sha256(Path(__file__)),
                        software=dict(torch=torch.__version__, torchvision=torchvision.__version__),
                        tsm_repo=repo_metadata(args.tsm_repo, args.expected_tsm_commit),
                        architecture=ARCHITECTURE, preprocessing=PREPROCESS)
        embeddings = extract_embeddings(frame, clips, args, metadata)
        embedding_columns = [f"tsm_past{window}_d{dimension:04d}" for window in HISTORY_SECONDS
                             for dimension in range(EMBEDDING_DIM)]
        data = pd.concat([frame.reset_index(drop=True), pd.DataFrame(embeddings, columns=embedding_columns)], axis=1)
        methods = dict(parameters=(columns, logs), video=(embedding_columns, []),
                       fusion=(columns + embedding_columns, logs))
        summary = evaluate(data, methods, args.seed, args.output_dir)
        for name, (features, log_features) in methods.items():
            pipe, detail = fit_head(data, features, log_features, args.seed)
            bundle = dict(method=name, pipeline=pipe, **detail, metadata=metadata,
                          task="classify_real_rating_greater_than_2", fullfit_events=len(data),
                          requires_tsm_weights=name in {"video", "fusion"},
                          embedding_columns=embedding_columns if name in {"video", "fusion"} else [],
                          training_note="exploratory fullfit deployment head; OOF evaluation uses separate fold heads")
            joblib.dump(bundle, args.output_dir / f"{name}_head.pkl")
        write_report(args.output_dir, summary, frame, metadata)
        runtime.update(status="completed", completed_utc=utc_now(), events=len(data),
                       sessions=int(data.session_id.nunique()), games=int(data.game_name.nunique()),
                       participants=int(data.participant_id.nunique()), metadata=metadata)
        save_json(args.output_dir / "runtime.json", runtime)
        logging.info("completed: %s", args.output_dir / "report.md")
        return 0
    except Exception as error:
        runtime.update(status="failed", failed_utc=utc_now(), error=str(error), traceback=traceback.format_exc())
        save_json(args.output_dir / "runtime.json", runtime)
        logging.exception("TSM head experiment failed; no success claimed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
