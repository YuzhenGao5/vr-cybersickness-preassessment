#!/usr/bin/env python3
"""Prepare leakage-controlled Habitat-Web action-prior samples.

Files are processed one scene at a time and samples are retained with a
deterministic reservoir, so the 18M action records are never materialized in
memory together.  Scene grouping happens before samples are extracted.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from habitat_decision_policy import ACTIONS, ACTION_TO_INDEX, FEATURE_NAMES, sha256_file


SPLITS = ("train", "val", "test")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {"schema_version", "seed", "data", "model", "training", "outputs", "claims"}
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"config missing required fields: {missing}")
    return config


def scene_key_from_scene_id(scene_id: str) -> str:
    normalized = str(scene_id).replace("\\", "/").rstrip("/")
    if not normalized:
        raise ValueError("empty scene_id")
    name = normalized.rsplit("/", 1)[-1]
    if "." in name:
        name = name.rsplit(".", 1)[0]
    if not name:
        raise ValueError(f"cannot derive scene key from {scene_id!r}")
    return name


def scene_key_from_content_path(path: str | Path) -> str:
    name = Path(path).name
    if name.endswith(".json.gz"):
        return name[: -len(".json.gz")]
    raise ValueError(f"content path must end in .json.gz: {path}")


def split_scene_ids(
    scene_ids: Sequence[str],
    *,
    seed: int,
    ratios: Mapping[str, float],
) -> dict[str, list[str]]:
    unique = sorted(set(scene_ids))
    if len(unique) < 3:
        raise ValueError("at least three distinct scenes are required for train/val/test")
    values = [float(ratios[name]) for name in SPLITS]
    if any(value <= 0.0 for value in values) or not math.isclose(sum(values), 1.0, abs_tol=1e-9):
        raise ValueError("split ratios must be positive and sum to one")

    ordered = sorted(
        unique,
        key=lambda scene: hashlib.sha256(f"{seed}:{scene}".encode("utf-8")).hexdigest(),
    )
    count = len(ordered)
    train_count = max(1, int(round(count * values[0])))
    val_count = max(1, int(round(count * values[1])))
    if train_count + val_count >= count:
        overflow = train_count + val_count - (count - 1)
        if train_count >= val_count and train_count - overflow >= 1:
            train_count -= overflow
        else:
            val_count -= overflow
    result = {
        "train": sorted(ordered[:train_count]),
        "val": sorted(ordered[train_count : train_count + val_count]),
        "test": sorted(ordered[train_count + val_count :]),
    }
    assert all(result.values())
    assert not (set(result["train"]) & set(result["val"]))
    assert not (set(result["train"]) & set(result["test"]))
    assert not (set(result["val"]) & set(result["test"]))
    return result


def normalize_quaternion_xyzw(values: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(values, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-10:
        raise ValueError("zero quaternion")
    return quaternion / norm


def rotate_vector(quaternion_xyzw: Sequence[float], vector: Sequence[float]) -> np.ndarray:
    """Rotate a vector by an xyzw quaternion."""

    q = normalize_quaternion_xyzw(quaternion_xyzw)
    xyz = q[:3]
    w = q[3]
    value = np.asarray(vector, dtype=np.float64)
    return value + 2.0 * np.cross(xyz, np.cross(xyz, value) + w * value)


def inverse_rotate_vector(
    quaternion_xyzw: Sequence[float], vector: Sequence[float]
) -> np.ndarray:
    q = normalize_quaternion_xyzw(quaternion_xyzw).copy()
    q[:3] *= -1.0
    return rotate_vector(q, vector)


def quaternion_angle(a: Sequence[float], b: Sequence[float]) -> float:
    first = normalize_quaternion_xyzw(a)
    second = normalize_quaternion_xyzw(b)
    cosine = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return float(2.0 * math.acos(cosine))


def valid_agent_state(
    item: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    state = item.get("agent_state")
    if not isinstance(state, Mapping):
        raise ValueError("replay item has no agent_state")
    position = np.asarray(state.get("position"), dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("agent position must contain three finite values")
    rotation = normalize_quaternion_xyzw(state.get("rotation"))
    sensor_missing = False
    try:
        sensor_rotation = normalize_quaternion_xyzw(
            state["sensor_data"]["rgb"]["rotation"]
        )
    except (KeyError, TypeError, ValueError):
        # Identity is an explicit, finite fallback.  The episode audit exposes
        # how often it was needed rather than silently imputing future state.
        sensor_rotation = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        sensor_missing = True
    return position, rotation, sensor_rotation, sensor_missing


def _episode_identifier(episode: Mapping[str, Any]) -> str:
    return str(episode.get("episode_id", ""))


def _episode_hash(identifier: str) -> np.uint64:
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    return np.uint64(int.from_bytes(digest[:8], "little", signed=False))


def goal_waypoints_for_episode(
    episode: Mapping[str, Any],
    goals_by_category: Mapping[str, Any],
) -> tuple[list[np.ndarray], str]:
    """Resolve navigable ObjectNav annotations without consulting replay future.

    Habitat-Web episodes commonly leave ``episode['goals']`` empty.  The source
    file stores goals under ``<scene filename>_<object_category>``.  View points
    are preferred because an object centre is often not navigable.  Goal
    positions are a documented fallback when no valid view point is available.
    """

    scene_id = str(episode.get("scene_id", "")).replace("\\", "/")
    scene_filename = scene_id.rsplit("/", 1)[-1]
    category = str(episode.get("object_category", ""))
    key = f"{scene_filename}_{category}"
    goals = goals_by_category.get(key)
    if not isinstance(goals, list) or not goals:
        raise ValueError(f"missing goals_by_category entry {key!r}")

    view_points: list[np.ndarray] = []
    goal_positions: list[np.ndarray] = []
    for goal in goals:
        if not isinstance(goal, Mapping):
            continue
        try:
            goal_position = np.asarray(goal.get("position"), dtype=np.float64)
            if goal_position.shape == (3,) and np.all(np.isfinite(goal_position)):
                goal_positions.append(goal_position)
        except (TypeError, ValueError):
            pass
        for view_point in goal.get("view_points", []) or []:
            try:
                state = view_point.get("agent_state", {})
                position = np.asarray(state.get("position"), dtype=np.float64)
                if position.shape == (3,) and np.all(np.isfinite(position)):
                    view_points.append(position)
            except (AttributeError, TypeError, ValueError):
                continue
    if view_points:
        return view_points, "view_point"
    if goal_positions:
        return goal_positions, "goal_position_fallback"
    raise ValueError(f"no finite waypoint in goals_by_category entry {key!r}")


def build_features(
    *,
    current_position: np.ndarray,
    current_rotation: np.ndarray,
    current_sensor_rotation: np.ndarray,
    start_position: np.ndarray,
    target_position: np.ndarray,
    previous_position: np.ndarray | None,
    previous_rotation: np.ndarray | None,
    travelled_distance: float,
    prior_actions: Sequence[str],
    history_run_clip: int,
) -> np.ndarray:
    """Build features without reading the current or any future action label."""

    world_to_goal = target_position - current_position
    local_to_goal = inverse_rotate_vector(current_rotation, world_to_goal)
    # Habitat's camera-forward axis is local -Z.  Expose a positive forward value.
    local_goal = np.asarray(
        [local_to_goal[0], local_to_goal[1], -local_to_goal[2]], dtype=np.float64
    )
    horizontal_distance = float(math.hypot(local_goal[0], local_goal[2]))
    goal_distance = float(np.linalg.norm(local_goal))
    goal_direction = local_goal / max(goal_distance, 1e-8)

    # Agent rotation is almost yaw-only in Habitat-Web.  LOOK_UP/LOOK_DOWN is
    # represented by the RGB sensor's local rotation, so these pitch features
    # must come from the strictly historical observation sensor at state[i-1].
    sensor_forward_local = rotate_vector(
        current_sensor_rotation, [0.0, 0.0, -1.0]
    )
    sensor_up_local = rotate_vector(current_sensor_rotation, [0.0, 1.0, 0.0])

    previous_one_hot = np.zeros(len(ACTIONS), dtype=np.float64)
    history_fraction = np.zeros(len(ACTIONS), dtype=np.float64)
    run_length = 0
    if prior_actions:
        previous_one_hot[ACTION_TO_INDEX[prior_actions[-1]]] = 1.0
        counts = Counter(prior_actions)
        history_fraction = np.asarray(
            [counts[action] / len(prior_actions) for action in ACTIONS],
            dtype=np.float64,
        )
        previous_action = prior_actions[-1]
        for action in reversed(prior_actions):
            if action != previous_action:
                break
            run_length += 1

    last_step_local = np.zeros(3, dtype=np.float64)
    last_rotation_angle = 0.0
    if previous_position is not None and previous_rotation is not None:
        world_step = current_position - previous_position
        local_step_raw = inverse_rotate_vector(current_rotation, world_step)
        last_step_local = np.asarray(
            [local_step_raw[0], local_step_raw[1], -local_step_raw[2]],
            dtype=np.float64,
        )
        last_rotation_angle = quaternion_angle(previous_rotation, current_rotation)

    values = np.concatenate(
        (
            local_goal,
            np.asarray(
                [
                    horizontal_distance,
                    goal_distance,
                    *goal_direction,
                    float(sensor_forward_local[1]),
                    float(sensor_up_local[1]),
                    float(current_position[1] - start_position[1]),
                    float(travelled_distance),
                    float(math.log1p(len(prior_actions))),
                ],
                dtype=np.float64,
            ),
            previous_one_hot,
            history_fraction,
            np.asarray(
                [
                    min(run_length, history_run_clip) / float(max(1, history_run_clip)),
                    *last_step_local,
                    last_rotation_angle,
                ],
                dtype=np.float64,
            ),
        )
    )
    if values.shape != (len(FEATURE_NAMES),):
        raise RuntimeError(
            f"feature implementation produced {values.shape}, expected {len(FEATURE_NAMES)}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("non-finite feature")
    return values.astype(np.float32)


@dataclass(frozen=True)
class DecisionSample:
    features: np.ndarray
    label: int
    scene_key: str
    episode_hash: np.uint64


def extract_episode_samples(
    episode: Mapping[str, Any],
    *,
    goals_by_category: Mapping[str, Any],
    history_run_clip: int,
) -> tuple[list[DecisionSample], dict[str, int]]:
    """Extract one episode; the first placeholder STOP is never history."""

    replay = episode.get("reference_replay")
    if not isinstance(replay, list) or len(replay) < 2:
        return [], {"short_or_missing_replay": 1}
    scene_key = scene_key_from_scene_id(str(episode.get("scene_id", "")))
    episode_id = _episode_identifier(episode)
    audit: Counter[str] = Counter()

    try:
        states = [valid_agent_state(item) for item in replay]
    except (TypeError, ValueError):
        return [], {"invalid_agent_state": 1}
    try:
        goal_waypoints, goal_source = goal_waypoints_for_episode(
            episode, goals_by_category
        )
    except ValueError:
        return [], {"missing_or_invalid_goal_annotation": 1}
    audit[f"goal_source_{goal_source}"] += 1
    goal_waypoint_matrix = np.stack(goal_waypoints).astype(np.float64)
    audit["sensor_rotation_fallback_states"] += sum(
        int(state_values[3]) for state_values in states
    )

    actions = [str(item.get("action", "")) for item in replay]
    # Habitat-Web stores the state *after* each listed action.  Row zero is a
    # placeholder STOP whose state is the observation for the first real
    # decision.  Therefore label action[i] is paired with state[i-1].  We can
    # never use state[i] because it already contains the result of the label.
    first_label_index = 1
    if actions[0] == "STOP":
        audit["first_placeholder_stop_removed"] += 1
    else:
        audit["first_action_without_pre_action_state_dropped"] += 1
    if actions[-1] == "STOP":
        audit["final_stop_present"] += 1
    else:
        audit["final_stop_missing"] += 1

    unsupported = [action for action in actions[first_label_index:] if action not in ACTION_TO_INDEX]
    if unsupported:
        return [], {"episode_with_unsupported_action": 1}

    start_position = states[0][0]
    prior_actions: list[str] = []
    samples: list[DecisionSample] = []
    travelled_distance = 0.0
    for index in range(first_label_index, len(replay)):
        observation_index = index - 1
        position, rotation, sensor_rotation, _ = states[observation_index]
        previous_position = states[observation_index - 1][0] if observation_index > 0 else None
        previous_rotation = states[observation_index - 1][1] if observation_index > 0 else None
        if previous_position is not None:
            travelled_distance += float(np.linalg.norm(position - previous_position))
        action = actions[index]
        try:
            # This target is selected solely from the current state and static
            # ObjectNav annotations.  No replay endpoint/future action is read.
            target_position = goal_waypoint_matrix[
                int(np.argmin(np.sum((goal_waypoint_matrix - position) ** 2, axis=1)))
            ]
            features = build_features(
                current_position=position,
                current_rotation=rotation,
                current_sensor_rotation=sensor_rotation,
                start_position=start_position,
                target_position=target_position,
                previous_position=previous_position,
                previous_rotation=previous_rotation,
                travelled_distance=travelled_distance,
                prior_actions=prior_actions,
                history_run_clip=history_run_clip,
            )
        except ValueError:
            audit["invalid_feature"] += 1
            previous_position, previous_rotation = position, rotation
            continue
        samples.append(
            DecisionSample(
                features=features,
                label=ACTION_TO_INDEX[action],
                scene_key=scene_key,
                episode_hash=_episode_hash(episode_id),
            )
        )
        prior_actions.append(action)

    audit["episodes_processed"] += 1
    audit["candidate_samples"] += len(samples)
    return samples, dict(audit)


class Reservoir:
    """Uniform reservoir with deterministic RNG and bounded memory."""

    def __init__(self, capacity: int, seed: int) -> None:
        if capacity < 1:
            raise ValueError("reservoir capacity must be positive")
        self.capacity = int(capacity)
        self.random = random.Random(int(seed))
        self.seen = 0
        self.items: list[DecisionSample] = []

    def add(self, item: DecisionSample) -> None:
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
            return
        replacement = self.random.randrange(self.seen)
        if replacement < self.capacity:
            self.items[replacement] = item

    def arrays(self, scene_to_index: Mapping[str, int]) -> dict[str, np.ndarray]:
        if not self.items:
            raise ValueError("reservoir is empty")
        # Sort provenance after random sampling so the file hash is reproducible.
        ordered = sorted(
            self.items,
            key=lambda item: (item.scene_key, int(item.episode_hash), item.label, item.features.tobytes()),
        )
        return {
            "X": np.stack([item.features for item in ordered]).astype(np.float32),
            "y": np.asarray([item.label for item in ordered], dtype=np.uint8),
            "scene_index": np.asarray(
                [scene_to_index[item.scene_key] for item in ordered], dtype=np.uint16
            ),
            "episode_hash": np.asarray(
                [item.episode_hash for item in ordered], dtype=np.uint64
            ),
        }


def _source_manifest(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], str]:
    entries = [
        {
            "relative_path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return entries, hashlib.sha256(encoded).hexdigest()


def _merge_audit(destination: Counter[str], source: Mapping[str, int]) -> None:
    for key, value in source.items():
        destination[key] += int(value)


def prepare_dataset(
    config: Mapping[str, Any],
    *,
    data_root: Path,
    output_dir: Path,
    max_samples_override: Mapping[str, int] | None = None,
    max_episodes_per_scene: int | None = None,
    max_scenes: int | None = None,
) -> dict[str, Any]:
    data_config = config["data"]
    paths = sorted(data_root.glob(str(data_config["content_glob"])))
    if len(paths) < 3:
        raise FileNotFoundError(
            f"expected at least three Habitat-Web scene files under {data_root}, found {len(paths)}"
        )
    available_scene_count = len(paths)
    if max_scenes is not None:
        if int(max_scenes) < 3:
            raise ValueError("max_scenes must be at least three")
        paths = sorted(
            paths,
            key=lambda path: hashlib.sha256(
                f"{config['seed']}:{scene_key_from_content_path(path)}".encode("utf-8")
            ).hexdigest(),
        )[: min(int(max_scenes), len(paths))]
        paths = sorted(paths)
    path_by_scene = {scene_key_from_content_path(path): path for path in paths}
    if len(path_by_scene) != len(paths):
        raise ValueError("duplicate scene filenames")
    split_scenes = split_scene_ids(
        list(path_by_scene),
        seed=int(config["seed"]),
        ratios=data_config["split_ratios"],
    )
    scene_to_split = {
        scene: split_name
        for split_name, scenes in split_scenes.items()
        for scene in scenes
    }
    max_samples = {
        name: int((max_samples_override or data_config["max_samples"])[name])
        for name in SPLITS
    }
    reservoirs = {
        name: Reservoir(max_samples[name], int(config["seed"]) + 1009 * (index + 1))
        for index, name in enumerate(SPLITS)
    }
    audit_by_split = {name: Counter() for name in SPLITS}
    action_candidates = {name: Counter() for name in SPLITS}
    episode_counts_by_scene: Counter[str] = Counter()

    source_entries, source_fingerprint = _source_manifest(paths)
    for path_index, path in enumerate(paths, start=1):
        expected_scene = scene_key_from_content_path(path)
        split_name = scene_to_split[expected_scene]
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            document = json.load(handle)
        episodes = document.get("episodes")
        if not isinstance(episodes, list):
            raise ValueError(f"{path} has no episodes list")
        if max_episodes_per_scene is None:
            selected_episodes = episodes
        else:
            limit_requested = min(len(episodes), int(max_episodes_per_scene))
            selected_episodes = sorted(
                episodes,
                key=lambda episode: hashlib.sha256(
                    (
                        f"{config['seed']}:{expected_scene}:"
                        f"{_episode_identifier(episode)}"
                    ).encode("utf-8")
                ).hexdigest(),
            )[:limit_requested]
        limit = len(selected_episodes)
        for episode in selected_episodes:
            actual_scene = scene_key_from_scene_id(str(episode.get("scene_id", "")))
            if actual_scene != expected_scene:
                raise ValueError(
                    f"scene-group violation: file {expected_scene} contains {actual_scene}"
                )
            samples, audit = extract_episode_samples(
                episode,
                goals_by_category=document.get("goals_by_category", {}),
                history_run_clip=int(data_config["history_run_clip"]),
            )
            _merge_audit(audit_by_split[split_name], audit)
            episode_counts_by_scene[actual_scene] += 1
            for sample in samples:
                action_candidates[split_name][ACTIONS[sample.label]] += 1
                reservoirs[split_name].add(sample)
        del document, episodes
        gc.collect()
        print(
            f"[{path_index}/{len(paths)}] {expected_scene}: split={split_name}, "
            f"episodes={limit}, reservoir={len(reservoirs[split_name].items)}",
            flush=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    prepared: dict[str, Any] = {}
    for split_name in SPLITS:
        scene_to_index = {
            scene: index for index, scene in enumerate(split_scenes[split_name])
        }
        arrays = reservoirs[split_name].arrays(scene_to_index)
        destination = output_dir / f"{split_name}.npz"
        np.savez_compressed(destination, **arrays)
        selected_counts = Counter(ACTIONS[int(label)] for label in arrays["y"])
        prepared[split_name] = {
            "path": str(destination.resolve()),
            "sha256": sha256_file(destination),
            "samples_seen": reservoirs[split_name].seen,
            "samples_selected": int(len(arrays["y"])),
            "action_candidates": dict(action_candidates[split_name]),
            "action_selected": dict(selected_counts),
            "scene_count": len(split_scenes[split_name]),
            "scenes": split_scenes[split_name],
            "scene_index": scene_to_index,
            "episode_count": int(
                sum(episode_counts_by_scene[scene] for scene in split_scenes[split_name])
            ),
            "audit": dict(audit_by_split[split_name]),
        }

    manifest = {
        "schema_version": "habitat-decision-prepared-1.0",
        "generated_at_utc": utc_now(),
        "status": "success",
        "source": data_config["source"],
        "source_kind": "public_human",
        "license": data_config["license"],
        "data_root": str(data_root.resolve()),
        "content_glob": data_config["content_glob"],
        "source_files": source_entries,
        "source_fingerprint_sha256": source_fingerprint,
        "seed": int(config["seed"]),
        "split_unit": "scene_id",
        "split_ratios": data_config["split_ratios"],
        "scene_overlap_checks": {
            "train_val": sorted(set(split_scenes["train"]) & set(split_scenes["val"])),
            "train_test": sorted(set(split_scenes["train"]) & set(split_scenes["test"])),
            "val_test": sorted(set(split_scenes["val"]) & set(split_scenes["test"])),
        },
        "feature_names": list(FEATURE_NAMES),
        "action_classes": list(ACTIONS),
        "feature_contract": {
            "available_at_inference": [
                "current and previous agent pose",
                "episode start pose",
                "static goals_by_category annotations and their view points",
                "strictly prior actions",
            ],
            "excluded_as_future_leakage": [
                "current action label",
                "future actions",
                "reference-replay endpoint",
                "remaining action count",
                "future path length",
                "success/outcome",
                "scene id and episode id",
                "object category",
            ],
            "goal_proxy": data_config["goal_proxy"],
            "first_placeholder_stop": "removed and not inserted into action history",
            "final_stop": "retained as the natural terminal decision",
        },
        "sampling": data_config["sampling"],
        "max_samples": max_samples,
        "max_episodes_per_scene": max_episodes_per_scene,
        "available_scene_count": available_scene_count,
        "selected_scene_count": len(paths),
        "scene_subset_policy": (
            "all scenes" if max_scenes is None else
            "lowest sha256(seed:scene), before scene-group split"
        ),
        "episode_subset_policy": (
            "all episodes" if max_episodes_per_scene is None else
            "lowest sha256(seed:scene:episode_id), independently within every scene"
        ),
        "splits": prepared,
        "claims": config["claims"],
    }
    if any(manifest["scene_overlap_checks"].values()):
        raise RuntimeError("scene split leakage detected")
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {manifest_path}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/habitat_decision_prior_v1.json",
        help="Project-relative or absolute configuration JSON.",
    )
    parser.add_argument("--data-root", help="Override Habitat-Web dataset root.")
    parser.add_argument("--output-dir", help="Override prepared output directory.")
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Override reservoir capacity independently for every split.",
    )
    parser.add_argument(
        "--max-episodes-per-scene",
        type=int,
        help="Deterministic real-data smoke mode; omitted for the full dataset.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        help="Deterministic whole-scene smoke subset; omitted for all scenes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_root = Path(args.data_root or config["data"]["data_root"])
    output_dir = Path(args.output_dir or config["data"]["prepared_dir"])
    override = None
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be positive")
        override = {name: args.max_samples for name in SPLITS}
    prepare_dataset(
        config,
        data_root=data_root,
        output_dir=output_dir,
        max_samples_override=override,
        max_episodes_per_scene=args.max_episodes_per_scene,
        max_scenes=args.max_scenes,
    )


if __name__ == "__main__":
    main()
