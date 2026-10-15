#!/usr/bin/env python3
"""Train and evaluate the small Unity Decision Brain v0 candidate ranker."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset

from unity_decision_policy import (
    DEFAULT_CONFIG,
    UnityDecisionRanker,
    apply_normalization,
    build_unity_flat_export,
    candidate_feature_names,
    evaluate_predictions,
    global_feature_names,
    human_metrics_or_refusal,
    load_config,
    masked_fill_logits,
    normalization_from_training,
    set_reproducible_seed,
    state_dict_to_json,
    write_json_atomic,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "features" / "unity_decision_brain_v0"
DEFAULT_MODEL = REPO_ROOT / "models" / "unity_decision_brain_v0.pth"
DEFAULT_EXPORT = REPO_ROOT / "models" / "unity_decision_brain_v0.json"
DEFAULT_UNITY_EXPORT = REPO_ROOT / "models" / "unity_decision_brain_v0_unity.json"
DEFAULT_METRICS = REPO_ROOT / "results" / "unity_decision_brain_v0_metrics.json"
DEFAULT_HISTORY = REPO_ROOT / "results" / "unity_decision_brain_v0_history.csv"


REQUIRED_ARRAYS = {
    "global_features",
    "candidate_features",
    "candidate_mask",
    "target_mask",
    "target_label_available",
    "skill_mask",
    "target_index",
    "skill_index",
    "teacher_utilities",
    "sample_weight",
    "source_kind",
    "decision_kind",
    "group_id",
    "episode_id",
    "decision_id",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(path: Path, config: dict[str, Any]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Prepared decision split is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        missing = REQUIRED_ARRAYS - set(archive.files)
        if missing:
            raise ValueError(f"{path.name} is missing arrays: {sorted(missing)}")
        payload = {name: archive[name] for name in REQUIRED_ARRAYS}
    count = len(payload["target_index"])
    global_dim = len(global_feature_names(config))
    candidate_dim = len(candidate_feature_names(config))
    skill_count = len(config["skills"])
    if payload["global_features"].shape != (count, global_dim):
        raise ValueError(f"Unexpected global feature shape in {path}")
    if payload["candidate_features"].ndim != 3 or payload["candidate_features"].shape[0] != count:
        raise ValueError(f"Unexpected candidate feature rank in {path}")
    if payload["candidate_features"].shape[2] != candidate_dim:
        raise ValueError(f"Unexpected candidate feature width in {path}")
    candidate_shape = payload["candidate_features"].shape[:2]
    for name in ("candidate_mask", "target_mask", "teacher_utilities"):
        if payload[name].shape != candidate_shape:
            raise ValueError(f"{name} shape mismatch in {path}")
    if payload["skill_mask"].shape != (count, skill_count):
        raise ValueError(f"skill_mask shape mismatch in {path}")
    if payload["target_label_available"].shape != (count,):
        raise ValueError(f"target_label_available shape mismatch in {path}")
    decision_kinds = payload["decision_kind"].astype(str)
    if not set(decision_kinds.tolist()) <= {"target_selection", "skill_transition"}:
        raise ValueError(f"Unsupported decision_kind in {path}")
    expected_target_labels = decision_kinds == "target_selection"
    if not np.array_equal(
        payload["target_label_available"].astype(bool), expected_target_labels
    ):
        raise ValueError(f"decision_kind/target_label_available mismatch in {path}")
    if not set(payload["source_kind"].astype(str).tolist()) <= set(
        config["allowed_source_kinds"]
    ):
        raise ValueError(f"Unsupported source_kind in {path}")
    if not np.isfinite(payload["sample_weight"]).all() or np.any(
        payload["sample_weight"] <= 0
    ):
        raise ValueError(f"sample_weight must be finite and positive in {path}")
    if not np.isfinite(payload["global_features"]).all() or not np.isfinite(
        payload["candidate_features"]
    ).all():
        raise ValueError(f"Non-finite model features in {path}")
    if np.any(~payload["candidate_mask"].astype(bool).any(axis=1)):
        raise ValueError(f"A sample has no present candidate in {path}")
    if np.any(~payload["target_mask"].astype(bool).any(axis=1)):
        raise ValueError(f"A sample has no legal target in {path}")
    if np.any(~payload["skill_mask"].astype(bool).any(axis=1)):
        raise ValueError(f"A sample has no legal skill in {path}")
    for row in range(count):
        target = int(payload["target_index"][row])
        skill = int(payload["skill_index"][row])
        if target < 0 or target >= candidate_shape[1] or not payload["candidate_mask"][row, target]:
            raise ValueError(f"Selected skill-conditioning candidate is absent in {path}, row {row}")
        if payload["target_label_available"][row] and not payload["target_mask"][row, target]:
            raise ValueError(f"Target selection label is illegal in {path}, row {row}")
        if skill < 0 or skill >= skill_count or not payload["skill_mask"][row, skill]:
            raise ValueError(f"Skill label is illegal in {path}, row {row}")
    return payload


def tensors_for_loader(
    split: dict[str, np.ndarray], normalization: dict[str, Any]
) -> TensorDataset:
    global_values, candidate_values = apply_normalization(
        split["global_features"], split["candidate_features"], normalization
    )
    return TensorDataset(
        torch.from_numpy(global_values),
        torch.from_numpy(candidate_values),
        torch.from_numpy(split["candidate_mask"].astype(bool)),
        torch.from_numpy(split["target_mask"].astype(bool)),
        torch.from_numpy(split["target_label_available"].astype(bool)),
        torch.from_numpy(split["skill_mask"].astype(bool)),
        torch.from_numpy(split["target_index"].astype(np.int64)),
        torch.from_numpy(split["skill_index"].astype(np.int64)),
        torch.from_numpy(split["sample_weight"].astype(np.float32)),
    )


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return weights.sum() * 0.0
    selected_weights = weights.to(values.dtype)
    # A fixed per-sample denominator preserves the configured 0.25:1.0
    # bootstrap/human contribution even for batch_size=1 or a homogeneous batch.
    return (values * selected_weights).mean()


def batch_loss(
    model: UnityDecisionRanker,
    batch: tuple[torch.Tensor, ...],
    device: torch.device,
    skill_loss_weight: float,
) -> tuple[torch.Tensor, float, float]:
    (
        global_values,
        candidate_values,
        candidate_mask,
        target_mask,
        target_available,
        skill_mask,
        target_index,
        skill_index,
        sample_weight,
    ) = (value.to(device) for value in batch)
    _, masked_targets, raw_skills = model(
        global_values,
        candidate_values,
        candidate_mask,
        target_mask,
        selected_target_indices=target_index,
    )
    if target_available.any():
        target_losses = functional.cross_entropy(
            masked_targets[target_available], target_index[target_available], reduction="none"
        )
        target_loss = weighted_mean(
            target_losses, sample_weight[target_available]
        )
    else:
        target_loss = masked_targets.sum() * 0.0
    masked_skills = masked_fill_logits(raw_skills, skill_mask)
    skill_losses = functional.cross_entropy(masked_skills, skill_index, reduction="none")
    skill_loss = weighted_mean(skill_losses, sample_weight)
    total = target_loss + float(skill_loss_weight) * skill_loss
    return total, float(target_loss.detach().cpu()), float(skill_loss.detach().cpu())


def evaluate_loss(
    model: UnityDecisionRanker,
    loader: DataLoader,
    device: torch.device,
    skill_loss_weight: float,
) -> dict[str, float]:
    model.eval()
    total_sum = target_sum = skill_sum = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            total, target, skill = batch_loss(model, batch, device, skill_loss_weight)
            total_sum += float(total.cpu())
            target_sum += target
            skill_sum += skill
            batches += 1
    if batches == 0:
        raise ValueError("Cannot evaluate an empty loader")
    return {
        "combined_loss": total_sum / batches,
        "target_loss": target_sum / batches,
        "skill_loss": skill_sum / batches,
    }


def predict_split(
    model: UnityDecisionRanker,
    split: dict[str, np.ndarray],
    normalization: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    global_values, candidate_values = apply_normalization(
        split["global_features"], split["candidate_features"], normalization
    )
    raw_target_parts: list[np.ndarray] = []
    raw_skill_parts: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(global_values), batch_size):
            end = min(len(global_values), start + batch_size)
            global_batch = torch.from_numpy(global_values[start:end]).to(device)
            candidate_batch = torch.from_numpy(candidate_values[start:end]).to(device)
            candidate_mask = torch.from_numpy(
                split["candidate_mask"][start:end].astype(bool)
            ).to(device)
            target_mask = torch.from_numpy(
                split["target_mask"][start:end].astype(bool)
            ).to(device)
            raw_target, masked_target, _ = model(
                global_batch,
                candidate_batch,
                candidate_mask,
                target_mask,
            )
            selected = masked_target.argmax(dim=1)
            target_available = torch.from_numpy(
                split["target_label_available"][start:end].astype(bool)
            ).to(device)
            active_target = torch.from_numpy(
                split["target_index"][start:end].astype(np.int64)
            ).to(device)
            # A target-selection decision is end-to-end: the skill is conditioned on
            # the model-ranked target.  A skill transition already has an active target,
            # so ranking a different target here would evaluate a decision the runtime
            # never asks the model to make.
            selected = torch.where(target_available, selected, active_target)
            _, _, raw_skill = model(
                global_batch,
                candidate_batch,
                candidate_mask,
                target_mask,
                selected_target_indices=selected,
            )
            raw_target_parts.append(raw_target.cpu().numpy())
            raw_skill_parts.append(raw_skill.cpu().numpy())
    return np.concatenate(raw_target_parts), np.concatenate(raw_skill_parts)


def slice_metrics(
    split: dict[str, np.ndarray],
    raw_targets: np.ndarray,
    raw_skills: np.ndarray,
    indices: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    metrics = evaluate_predictions(
        raw_target_logits=raw_targets[indices],
        candidate_mask=split["target_mask"][indices],
        target_index=split["target_index"][indices],
        raw_skill_logits=raw_skills[indices],
        skill_mask=split["skill_mask"][indices],
        skill_index=split["skill_index"][indices],
        teacher_utilities=split["teacher_utilities"][indices],
        skill_names=config["skills"],
        target_label_available=split["target_label_available"][indices],
    )
    # The bootstrap teacher's deterministic score before seeded jitter can be
    # reconstructed entirely from runtime-available model inputs.  Reporting
    # it next to the learned ranker prevents a smoke model from looking useful
    # merely because the labelled scene is easy or candidate order is stable.
    subset_features = split["candidate_features"][indices]
    subset_mask = split["target_mask"][indices].astype(bool)
    subset_labels = split["target_index"][indices].astype(np.int64)
    subset_available = split["target_label_available"][indices].astype(bool)
    feature_order = candidate_feature_names(config)
    priority_index = feature_order.index("priority")
    distance_index = feature_order.index("distance")
    resolved_index = feature_order.index("resolved")
    static_utility = (
        subset_features[:, :, priority_index] * 2.0
        - subset_features[:, :, distance_index] * 0.055
        + np.where(subset_features[:, :, resolved_index] >= 0.5, 0.20, -0.35)
    )
    static_prediction = np.where(subset_mask, static_utility, -1.0e30).argmax(axis=1)
    if subset_available.any() and metrics["target"].get("status") == "reported":
        static_accuracy = float(
            np.mean(static_prediction[subset_available] == subset_labels[subset_available])
        )
        metrics["target"]["baselines"][
            "rule_base_utility_top1_accuracy"
        ] = static_accuracy
        metrics["target"]["model_minus_rule_base_utility_top1"] = float(
            metrics["target"]["top1_accuracy"] - static_accuracy
        )
    return metrics


def skill_label_coverage(split: dict[str, np.ndarray], config: dict[str, Any]) -> dict[str, Any]:
    counts = Counter(config["skills"][int(value)] for value in split["skill_index"])
    observed = [skill for skill in config["skills"] if counts.get(skill, 0) > 0]
    return {
        "counts": {skill: int(counts.get(skill, 0)) for skill in config["skills"]},
        "observed_skills": observed,
        "observed_skill_count": len(observed),
        "all_six_skills_observed": len(observed) == len(config["skills"]),
        "warning": None
        if len(observed) == len(config["skills"])
        else (
            "Skill labels do not cover all six macro skills; macro-F1 must not be "
            "presented as complete six-skill ability."
        ),
    }


def save_history(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(path)


def train(arguments: argparse.Namespace) -> dict[str, Any]:
    config = load_config(arguments.config)
    manifest_path = arguments.data_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepared data manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "unity-decision-prepared-data-1.0":
        raise ValueError("Unsupported prepared data manifest")
    if manifest.get("config_schema_version") != config["schema_version"]:
        raise ValueError("Prepared data config schema does not match the training config")
    if manifest.get("config_sha256") != sha256_file(arguments.config):
        raise ValueError("Prepared data was produced with a different config file")
    if manifest.get("context_schema_version") != config["context_schema_version"]:
        raise ValueError("Prepared context schema does not match the training config")
    if manifest.get("outcome_schema_version") != config["outcome_schema_version"]:
        raise ValueError("Prepared outcome schema does not match the training config")
    if manifest.get("global_feature_names") != global_feature_names(config):
        raise ValueError("Prepared global feature names/order do not match the config")
    if manifest.get("candidate_feature_names") != candidate_feature_names(config):
        raise ValueError("Prepared candidate feature names/order do not match the config")
    if int(manifest.get("split_seed", -1)) != int(config["split"]["seed"]):
        raise ValueError("Prepared split seed is not the frozen config seed")
    if not isinstance(manifest.get("preprocessing"), dict) or manifest[
        "preprocessing"
    ].get("schema_version") != "unity-decision-preprocessing-1.0":
        raise ValueError("Prepared preprocessing contract is missing or unsupported")
    if manifest.get("synthetic_fixture") and not arguments.allow_synthetic_fixture:
        raise ValueError(
            "Synthetic fixture data is test-only; pass --allow-synthetic-fixture explicitly"
        )
    if manifest.get("status") == "smoke_only" and not arguments.allow_smoke:
        raise ValueError(
            "Prepared data is smoke_only; pass --allow-smoke explicitly and do not present it as production"
        )

    split_manifest_path = arguments.data_dir / "split_manifest.json"
    if not split_manifest_path.is_file():
        raise FileNotFoundError(f"Prepared split manifest is missing: {split_manifest_path}")
    if manifest.get("split_manifest_sha256") != sha256_file(split_manifest_path):
        raise ValueError("split_manifest.json hash does not match dataset_manifest.json")
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if split_manifest.get("schema_version") != "unity-decision-group-split-1.0":
        raise ValueError("Unsupported split manifest")
    if int(split_manifest.get("seed", -1)) != int(config["split"]["seed"]):
        raise ValueError("Split manifest does not use the frozen seed 42")

    split_paths = {
        name: arguments.data_dir / f"unity_decision_{name}.npz"
        for name in ("train", "val", "test")
    }
    artifact_hashes = manifest.get("artifact_sha256")
    artifact_sizes = manifest.get("artifact_size_bytes")
    if not isinstance(artifact_hashes, dict) or not isinstance(artifact_sizes, dict):
        raise ValueError("Prepared artifact hash/size metadata is missing")
    for name, path in split_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Prepared decision split is missing: {path}")
        if int(artifact_sizes.get(name, -1)) != path.stat().st_size:
            raise ValueError(f"Prepared {name} split size does not match the manifest")
        if artifact_hashes.get(name) != sha256_file(path):
            raise ValueError(f"Prepared {name} split hash does not match the manifest")
    splits = {name: load_split(path, config) for name, path in split_paths.items()}
    split_groups = {name: set(split["group_id"].tolist()) for name, split in splits.items()}
    if any(
        split_groups[left] & split_groups[right]
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise ValueError("Prepared splits leak a scene/project family")
    if not splits["train"]["target_label_available"].any():
        raise ValueError("Training data has no target_selection labels")
    expected_groups = split_manifest.get("groups")
    if not isinstance(expected_groups, dict) or {
        name: set(expected_groups.get(name, [])) for name in split_groups
    } != split_groups:
        raise ValueError("NPZ group membership does not match split_manifest.json")
    if {
        name: set(manifest.get("split_groups", {}).get(name, []))
        for name in split_groups
    } != split_groups:
        raise ValueError("NPZ group membership does not match dataset_manifest.json")
    actual_split_samples = {name: len(split["target_index"]) for name, split in splits.items()}
    if manifest.get("split_samples") != actual_split_samples:
        raise ValueError("NPZ sample counts do not match dataset_manifest.json")
    if int(manifest.get("samples", -1)) != sum(actual_split_samples.values()):
        raise ValueError("Total NPZ sample count does not match dataset_manifest.json")
    prepared_source_counts = {
        name: int(count)
        for name, count in Counter(
            source for split in splits.values() for source in split["source_kind"].tolist()
        ).items()
    }
    prepared_decision_kind_counts = {
        name: int(count)
        for name, count in Counter(
            kind for split in splits.values() for kind in split["decision_kind"].tolist()
        ).items()
    }
    if manifest.get("source_counts") != prepared_source_counts:
        raise ValueError("NPZ source counts do not match dataset_manifest.json")
    if manifest.get("decision_kind_counts") != prepared_decision_kind_counts:
        raise ValueError("NPZ decision-kind counts do not match dataset_manifest.json")
    if int(manifest.get("target_label_samples", -1)) != int(
        prepared_decision_kind_counts.get("target_selection", 0)
    ):
        raise ValueError("NPZ target label count does not match dataset_manifest.json")

    seed = int(config["training"]["seed"])
    if arguments.seed is not None and arguments.seed != seed:
        raise ValueError(f"Decision Brain v0 freezes seed={seed}; override {arguments.seed} rejected")
    set_reproducible_seed(seed)
    torch.set_num_threads(max(1, int(arguments.threads)))
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable in this Python environment")

    normalization = normalization_from_training(
        splits["train"]["global_features"],
        splits["train"]["candidate_features"],
        splits["train"]["candidate_mask"],
    )
    datasets = {
        name: tensors_for_loader(split, normalization) for name, split in splits.items()
    }
    generator = torch.Generator().manual_seed(seed)
    batch_size = int(arguments.batch_size or config["training"]["batch_size"])
    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=batch_size, shuffle=True,
            generator=generator, num_workers=0,
        ),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=0),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=0),
    }
    model_options = config["model"]
    model = UnityDecisionRanker(
        len(global_feature_names(config)),
        len(candidate_feature_names(config)),
        len(config["skills"]),
        global_hidden_dim=int(model_options["global_hidden_dim"]),
        candidate_hidden_dim=int(model_options["candidate_hidden_dim"]),
        joint_hidden_dim=int(model_options["joint_hidden_dim"]),
        dropout=float(model_options["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    skill_loss_weight = float(config["training"]["skill_loss_weight"])
    epochs = int(arguments.epochs or config["training"]["epochs"])
    patience = int(config["training"]["patience"])
    min_delta = float(config["training"]["min_delta"])
    gradient_clip = float(config["training"]["gradient_clip_norm"])
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_validation = math.inf
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_total = train_target = train_skill = 0.0
        batch_count = 0
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            total, target_loss, skill_loss = batch_loss(
                model, batch, device, skill_loss_weight
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            train_total += float(total.detach().cpu())
            train_target += target_loss
            train_skill += skill_loss
            batch_count += 1
        validation = evaluate_loss(model, loaders["val"], device, skill_loss_weight)
        improved = validation["combined_loss"] < best_validation - min_delta
        history.append(
            {
                "epoch": epoch,
                "train_combined_loss": train_total / max(1, batch_count),
                "train_target_loss": train_target / max(1, batch_count),
                "train_skill_loss": train_skill / max(1, batch_count),
                "val_combined_loss": validation["combined_loss"],
                "val_target_loss": validation["target_loss"],
                "val_skill_loss": validation["skill_loss"],
                "best_so_far": improved,
            }
        )
        if improved:
            best_validation = validation["combined_loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    save_history(arguments.history, history)

    split_metrics: dict[str, Any] = {}
    prediction_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, split in splits.items():
        raw_targets, raw_skills = predict_split(
            model, split, normalization, device, batch_size
        )
        prediction_cache[name] = (raw_targets, raw_skills)
        all_indices = np.arange(len(split["target_index"]), dtype=np.int64)
        metrics = slice_metrics(split, raw_targets, raw_skills, all_indices, config)
        metrics["skill_label_coverage"] = skill_label_coverage(split, config)
        metrics["by_source"] = {}
        for source in sorted(set(split["source_kind"].tolist())):
            indices = np.flatnonzero(split["source_kind"] == source)
            metrics["by_source"][source] = slice_metrics(
                split, raw_targets, raw_skills, indices, config
            )
        metrics["human_metrics"] = human_metrics_or_refusal(
            split["source_kind"],
            config,
            lambda indices, s=split, rt=raw_targets, rs=raw_skills: slice_metrics(
                s, rt, rs, indices, config
            ),
        )
        split_metrics[name] = metrics

    source_counts = prepared_source_counts
    human_decision_samples = sum(
        source_counts.get(source, 0) for source in config["human_source_kinds"]
    )
    decision_kind_counts = prepared_decision_kind_counts
    checkpoint = {
        "schema_version": "unity-decision-brain-checkpoint-1.0",
        "state_dict": best_state,
        "architecture": model.architecture(),
        "normalization": normalization,
        "global_feature_names": global_feature_names(config),
        "candidate_feature_names": candidate_feature_names(config),
        "skills": list(config["skills"]),
        "brain_states": list(config["brain_states"]),
        "planner_modes": list(config["planner_modes"]),
        "semantic_types": list(config["semantic_types"]),
        "preprocessing": manifest["preprocessing"],
        "skill_conditioning_by_decision_kind": manifest["preprocessing"][
            "decision_kind_routing"
        ],
        "data_fingerprint_sha256": manifest["data_fingerprint_sha256"],
        "source_counts": source_counts,
        "decision_kind_counts": decision_kind_counts,
        "target_label_samples": int(decision_kind_counts.get("target_selection", 0)),
        "human_decision_samples": human_decision_samples,
        "human_like_claim_allowed": False,
        "seed": seed,
    }
    arguments.model.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = arguments.model.with_name(arguments.model.name + ".tmp")
    torch.save(checkpoint, temporary_model)
    temporary_model.replace(arguments.model)
    model_sha256 = sha256_file(arguments.model)

    export = {
        "schema_version": "unity-decision-brain-json-1.0",
        "generated_at_utc": utc_now(),
        "model_name": config["claims"]["model_name"],
        "scope": config["claims"]["scope"],
        "data_status": manifest["status"],
        "synthetic_fixture": bool(manifest.get("synthetic_fixture")),
        "architecture": model.architecture(),
        "normalization": normalization,
        "global_feature_names": global_feature_names(config),
        "candidate_feature_names": candidate_feature_names(config),
        "skills": list(config["skills"]),
        "brain_states": list(config["brain_states"]),
        "planner_modes": list(config["planner_modes"]),
        "semantic_types": list(config["semantic_types"]),
        "context_schema_version": config["context_schema_version"],
        "outcome_schema_version": config["outcome_schema_version"],
        "preprocessing": manifest["preprocessing"],
        "skill_conditioning_by_decision_kind": manifest["preprocessing"][
            "decision_kind_routing"
        ],
        "candidate_mask_meaning": "present candidate; padded rows are false",
        "target_mask_meaning": "runtime-legal target after visited/cooldown/reachability gates",
        "weights": state_dict_to_json(best_state),
        "training_provenance": {
            "data_fingerprint_sha256": manifest["data_fingerprint_sha256"],
            "source_counts": source_counts,
            "decision_kind_counts": decision_kind_counts,
            "target_label_samples": int(decision_kind_counts.get("target_selection", 0)),
            "human_decision_samples": human_decision_samples,
            "human_like_claim_allowed": False,
            "seed": seed,
            "best_epoch": best_epoch,
            "skill_label_coverage": {
                split: split_metrics[split]["skill_label_coverage"]
                for split in split_metrics
            },
        },
        "runtime_authority": {
            "model_may_rank_targets": True,
            "model_may_recommend_macro_skill": True,
            "model_may_bypass_navmesh_or_collision_checks": False,
            "model_may_authorize_interaction_methods": False,
            "model_may_bypass_skill_mask": False,
        },
    }
    write_json_atomic(arguments.export, export)
    export_sha256 = sha256_file(arguments.export)

    unity_export_path = arguments.unity_export or arguments.export.with_name(
        arguments.export.stem + "_unity.json"
    )
    golden_index = 0
    golden = splits["test"]
    unity_export = build_unity_flat_export(
        state_dict=best_state,
        architecture=model.architecture(),
        normalization=normalization,
        config=config,
        preprocessing=manifest["preprocessing"],
        generated_at_utc=utc_now(),
        data_status=manifest["status"],
        synthetic_fixture=bool(manifest.get("synthetic_fixture")),
        checkpoint_sha256=model_sha256,
        python_export_sha256=export_sha256,
        data_fingerprint_sha256=manifest["data_fingerprint_sha256"],
        config_sha256=sha256_file(arguments.config),
        source_counts=source_counts,
        decision_kind_counts=decision_kind_counts,
        human_decision_samples=human_decision_samples,
        seed=seed,
        best_epoch=best_epoch,
        golden_global_features=golden["global_features"][golden_index],
        golden_candidate_features=golden["candidate_features"][golden_index],
        golden_candidate_mask=golden["candidate_mask"][golden_index],
        golden_target_mask=golden["target_mask"][golden_index],
        golden_skill_mask=golden["skill_mask"][golden_index],
        golden_target_index=int(golden["target_index"][golden_index]),
        golden_decision_kind=str(golden["decision_kind"][golden_index]),
        golden_decision_id=str(golden["decision_id"][golden_index]),
        golden_episode_id=str(golden["episode_id"][golden_index]),
        golden_split_index=golden_index,
    )
    write_json_atomic(unity_export_path, unity_export)
    unity_export_sha256 = sha256_file(unity_export_path)

    metrics_payload = {
        "schema_version": "unity-decision-brain-metrics-1.0",
        "generated_at_utc": utc_now(),
        "status": "smoke_only" if manifest["status"] == "smoke_only" else "trained",
        "scope": config["claims"]["scope"],
        "human_like_claim_allowed": False,
        "human_decision_samples": human_decision_samples,
        "human_metrics_policy": (
            "Human metrics are refused when no public_human or unity_human labels are present."
        ),
        "source_counts": source_counts,
        "decision_kind_counts": decision_kind_counts,
        "target_label_samples": int(decision_kind_counts.get("target_selection", 0)),
        "data_fingerprint_sha256": manifest["data_fingerprint_sha256"],
        "group_leakage_check": "passed",
        "normalization_fitted_on": "train split only",
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_validation_combined_loss": best_validation,
        "training_elapsed_seconds": elapsed,
        "metrics": split_metrics,
        "interpretation_warning": (
            "Bootstrap accuracy measures imitation of the labelled rule/script policy. "
            "It is not evidence of human-like decisions. Missing skill classes make "
            "six-skill capability untested even when observed-class predictions are accurate. "
            "Always compare target Top-1 with the rule-base-utility baseline and skill "
            "performance with the first-legal-skill baseline."
        ),
        "artifacts": {
            "checkpoint": str(arguments.model.resolve()),
            "checkpoint_sha256": model_sha256,
            "json_export": str(arguments.export.resolve()),
            "json_export_sha256": export_sha256,
            "unity_json_export": str(unity_export_path.resolve()),
            "unity_json_export_sha256": unity_export_sha256,
            "history": str(arguments.history.resolve()),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "threads": int(arguments.threads),
        },
    }
    write_json_atomic(arguments.metrics, metrics_payload)
    return metrics_payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument(
        "--unity-export",
        type=Path,
        help=(
            "Unity JsonUtility-compatible flat export. By default it is written "
            "beside --export with an _unity suffix."
        ),
    )
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, help="Must equal the frozen config seed (42).")
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--allow-synthetic-fixture", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_arg_parser().parse_args(argv)
    try:
        metrics = train(arguments)
    except Exception as exc:
        print(f"Decision Brain training failed: {exc}", file=sys.stderr)
        return 1
    test = metrics["metrics"]["test"]
    print(
        f"Decision Brain v0 {metrics['status']}: "
        f"target_top1={test['target'].get('top1_accuracy')} "
        f"skill_macro_f1={test['skill']['macro_f1']:.4f} "
        f"human_decisions={metrics['human_decision_samples']}"
    )
    print(metrics["artifacts"]["json_export"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
