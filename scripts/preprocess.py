#!/usr/bin/env python3
"""
preprocess.py — Mazed and Confused / VRWalking 数据集预处理

数据集：Mazed and Confused / VRWalking
仓库：https://github.com/Jyotinag/VRWalking_Dataset/
路径：路径 A（CYRE 不可用后的降级路径）

重要标注：
- 本数据集使用 POV imagery，特征为【路径依赖特征】
- 不得声称预暴露或独立于用户
- 标签为真实 SSQ / FMS，非代理标签

分析单元：参与者级（一行 = 一个参与者）
评估协议：LOSO（只有一个场景，无法 Leave-One-Scene-Out）

用法：
    python scripts/preprocess.py \
        --csv data/raw/vrwalking/SSQ_scores_NASA_Reaction_Time.csv \
        --video-dir data/raw/vrwalking/videos/ \
        --mapping data/raw/vrwalking/participant_video_mapping.csv \
        --output data/processed/vrwalking/ \
        --config configs/label_mapping.yaml \
        --log results/preprocess_log.txt
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

LABEL_COLS = ["n_p", "o_p", "d_p", "TS_p", "First_FMS", "Second_FMS", "Overall_FMS"]


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("preprocess")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(sh)
    return logger


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_labels(csv_path: Path, logger: logging.Logger) -> pd.DataFrame:
    logger.info(f"读取标签 CSV: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    logger.info(f"原始行数: {len(df)}, 列数: {len(df.columns)}")

    available = [c for c in LABEL_COLS if c in df.columns]
    if not available:
        raise ValueError(f"CSV 中没有任何已知标签列。可用列: {list(df.columns)}")

    df = df.reset_index(drop=True)
    df["participant_id"] = df.index.astype(int)
    out = df[["participant_id"] + available].copy()

    for col in available:
        n_missing = out[col].isna().sum()
        if n_missing > 0:
            logger.warning(f"列 '{col}' 缺失 {n_missing}/{len(out)} 行")

    logger.info(f"保留标签列: {available}")
    logger.info(f"有效参与者数: {len(out)}")
    return out


def load_mapping(mapping_path: Path, logger: logging.Logger) -> pd.DataFrame:
    logger.info(f"读取映射文件: {mapping_path}")
    m = pd.read_csv(mapping_path, encoding="utf-8-sig")

    required = {"participant_id", "video_file"}
    if not required.issubset(set(m.columns)):
        raise ValueError(f"映射文件必须包含列 {required}。实际列: {list(m.columns)}")

    m["participant_id"] = m["participant_id"].astype(int)
    logger.info(f"映射条目数: {len(m)}")
    return m


def verify_video_files(mapping: pd.DataFrame, video_dir: Path, logger: logging.Logger) -> pd.DataFrame:
    logger.info(f"验证视频目录: {video_dir}")
    if not video_dir.exists():
        raise FileNotFoundError(f"视频目录不存在: {video_dir}")

    available = {p.name for p in video_dir.iterdir() if p.is_file()}
    logger.info(f"视频目录中文件数: {len(available)}")

    mapping = mapping.copy()
    mapping["exists"] = mapping["video_file"].apply(lambda f: f in available)

    n_missing = (~mapping["exists"]).sum()
    if n_missing > 0:
        missing_files = mapping.loc[~mapping["exists"], "video_file"].tolist()
        logger.warning(f"缺失 {n_missing} 个视频文件: {missing_files[:10]}...")

    valid = mapping[mapping["exists"]].drop(columns=["exists"]).reset_index(drop=True)
    logger.info(f"有效映射: {len(valid)}")
    return valid


def merge_labels_and_videos(labels: pd.DataFrame, mapping: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    merged = pd.merge(labels, mapping, on="participant_id", how="inner")
    logger.info(f"合并后参与者数: {len(merged)}")

    n_dropped = len(labels) - len(merged)
    if n_dropped > 0:
        logger.warning(f"因缺少视频或标签而丢弃 {n_dropped} 名参与者。")
    return merged


def main():
    parser = argparse.ArgumentParser(description="VRWalking 预处理")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    args = parser.parse_args()

    logger = setup_logger(args.log)
    logger.info("=" * 70)
    logger.info("VRWalking 预处理开始")
    logger.info(f"时间: {datetime.now().isoformat()}")
    logger.info("=" * 70)

    if not args.config.exists():
        raise FileNotFoundError(f"配置文件不存在: {args.config}")
    cfg = load_config(args.config)
    logger.info(f"配置加载成功: {args.config}")

    fs = cfg.get("feature_source", {})
    if fs.get("type") != "path_dependent":
        logger.warning("配置中 feature_source.type 不是 'path_dependent'。")

    labels = load_labels(args.csv, logger)
    mapping = load_mapping(args.mapping, logger)
    mapping = verify_video_files(mapping, args.video_dir, logger)
    merged = merge_labels_and_videos(labels, mapping, logger)

    if len(merged) == 0:
        raise RuntimeError("合并后无有效参与者，无法继续。")

    args.output.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output / "labels.csv", index=False, encoding="utf-8")

    manifest = merged[["participant_id", "video_file"]].copy()
    manifest.to_csv(args.output / "video_manifest.csv", index=False, encoding="utf-8")

    metadata = {
        "dataset": "Mazed and Confused / VRWalking",
        "downgrade_path": "A",
        "feature_source_type": "path_dependent",
        "feature_source_disclaimer": "POV imagery 反映参与者路径、头动方向、移动和暴露历史，不是独立场景表征。不得声称预暴露或独立于用户。",
        "label_source": cfg.get("label_source", "unknown"),
        "primary_label": cfg.get("primary_label", "unknown"),
        "analysis_unit": "participant_level",
        "n_participants": int(len(merged)),
        "n_scenes": 1,
        "evaluation_protocol": "LOSO_only",
        "leave_one_scene_out": False,
        "reason_no_loso_scene": "只有一个场景，无法留出场景",
        "proxy_label": False,
        "proxy_disclaimer": "真实 SSQ/FMS 标签，非代理标签",
        "missing_label_rule": cfg.get("missing_label_rule", "缺失标签不得推断为低风险"),
        "preprocess_time": datetime.now().isoformat(),
        "n_labels_missing_per_column": {
            col: int(merged[col].isna().sum())
            for col in merged.columns
            if col not in ("participant_id", "video_file")
        },
    }

    with open(args.output / "preprocess_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info("=" * 70)
    logger.info("预处理完成")
    logger.info(f"有效参与者: {len(merged)}")
    logger.info("路径依赖特征: 是")
    logger.info("代理标签: 否")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
