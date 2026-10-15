#!/usr/bin/env python3
"""Visual demo for the trained EyeNavGS 6-DoF motion-imitation model.

This program deliberately makes a narrow claim: the model predicts the next
short-horizon head-motion velocity from recent motion.  It does not consume the
Unity scene, choose goals, avoid risks, or perform autonomous navigation.  The
Unity semantic data is a separately rendered, rescaled display overlay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    import numpy as np
    import torch
except BaseException as import_error:
    # Preserve startup diagnostics even when a frozen windowed app fails before
    # main() is entered (for example, because a native dependency is missing).
    diagnostic_path = os.environ.get("AGENT_DEMO_ERROR_LOG")
    if diagnostic_path:
        try:
            with open(diagnostic_path, "w", encoding="utf-8") as stream:
                stream.write(
                    f"{type(import_error).__name__}: {import_error}\n\n"
                    + traceback.format_exc()
                )
        except OSError:
            pass
    raise


APP_NAME = "AgentExplorationDemo"
MODEL_TITLE = "6-DoF short-horizon motion imitation — NOT scene-aware navigation"
STAGE_HALF_EXTENT_METERS = 1.5
DEFAULT_DISPLAY_RISK_RADIUS = 0.16
EXPECTED_CHECKPOINT_SCHEMA = "1.0.0"
EXPECTED_SCENE_SCHEMA = "1.1.0"


def _resource_root() -> Path:
    """Return the read-only root used by source and PyInstaller one-file runs."""

    if getattr(sys, "frozen", False):
        bundle_root = getattr(sys, "_MEIPASS", None)
        if not bundle_root:
            raise RuntimeError("Frozen application has no PyInstaller resource root")
        return Path(bundle_root).resolve()
    return Path(__file__).resolve().parents[1]


RESOURCE_ROOT = _resource_root()
SCRIPTS_DIR = RESOURCE_ROOT / "scripts"
if SCRIPTS_DIR.is_dir() and str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from exploration_policy import (  # noqa: E402
        FEATURE_NAMES,
        build_model,
        integrate_local_velocity,
        load_checkpoint,
        quaternion_conjugate,
        rotate_vectors,
    )
except ImportError as exc:  # no model substitute is permitted
    error = RuntimeError(
        "The required exploration_policy module is unavailable; "
        "the demo will not fall back to random motion."
    )
    diagnostic_path = os.environ.get("AGENT_DEMO_ERROR_LOG")
    if diagnostic_path:
        try:
            Path(diagnostic_path).write_text(
                f"{type(error).__name__}: {error}\n\n" + traceback.format_exc(),
                encoding="utf-8",
            )
        except OSError:
            pass
    raise error from exc


@dataclass(frozen=True)
class RuntimeResources:
    model: torch.nn.Module
    checkpoint: dict[str, Any]
    metadata: dict[str, Any]
    scene: dict[str, Any]
    model_path: Path
    metadata_path: Path
    scene_path: Path


@dataclass(frozen=True)
class OverlayItem:
    name: str
    category: str
    x: float
    z: float
    radius: float | None = None
    used_default_radius: bool = False


@dataclass(frozen=True)
class SceneOverlay:
    objects: tuple[OverlayItem, ...]
    risks: tuple[OverlayItem, ...]
    synthetic_input: bool
    coordinate_scale: float
    source_object_count: int
    source_risk_count: int


@dataclass(frozen=True)
class RolloutResult:
    positions: np.ndarray
    orientations: np.ndarray
    actions: np.ndarray
    boundary_events_per_step: np.ndarray
    dt_seconds: float

    @property
    def boundary_event_count(self) -> int:
        return int(self.boundary_events_per_step.sum())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required {description} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid {description} JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return payload


def _finite_vector(
    value: Any,
    *,
    length: int,
    name: str,
    positive: bool = False,
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric vector") from exc
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {length} finite values")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} must contain only positive values")
    return array


def load_runtime_resources() -> RuntimeResources:
    """Load and cross-check every bundled resource, with no random fallback."""

    model_path = RESOURCE_ROOT / "models" / "exploration_model.pth"
    metadata_path = RESOURCE_ROOT / "models" / "exploration_model_metadata.json"
    scene_path = RESOURCE_ROOT / "results" / "unity_parse_result.json"

    for path, label in (
        (model_path, "trained model checkpoint"),
        (metadata_path, "model metadata"),
        (scene_path, "Unity semantic parse"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required {label} is missing: {path}")

    metadata = _read_json(metadata_path, "model metadata")
    scene = _read_json(scene_path, "Unity semantic parse")

    expected_hash = metadata.get("model_sha256")
    actual_hash = _sha256(model_path)
    if not isinstance(expected_hash, str) or expected_hash.lower() != actual_hash:
        raise ValueError(
            "The model checkpoint SHA-256 does not match its metadata; refusing to run"
        )

    checkpoint = load_checkpoint(str(model_path))
    if checkpoint.get("schema_version") != EXPECTED_CHECKPOINT_SCHEMA:
        raise ValueError(
            f"Unsupported checkpoint schema: {checkpoint.get('schema_version')!r}"
        )
    if metadata.get("schema_version") != EXPECTED_CHECKPOINT_SCHEMA:
        raise ValueError(f"Unsupported metadata schema: {metadata.get('schema_version')!r}")
    if scene.get("schema_version") != EXPECTED_SCENE_SCHEMA:
        raise ValueError(f"Unsupported Unity parse schema: {scene.get('schema_version')!r}")

    # This requested artifact must demonstrate the genuinely trained EyeNavGS
    # checkpoint.  A synthetic or unknown checkpoint is an error, not a fallback.
    if checkpoint.get("data_source") != "EyeNavGS" or checkpoint.get("synthetic") is not False:
        raise ValueError("Checkpoint is not marked as real EyeNavGS training data")
    if metadata.get("data_source") != "EyeNavGS" or metadata.get("synthetic") is not False:
        raise ValueError("Metadata is not marked as real EyeNavGS training data")
    if metadata.get("dataset_fingerprint_sha256") != checkpoint.get(
        "dataset_fingerprint_sha256"
    ):
        raise ValueError("Checkpoint and metadata dataset fingerprints do not match")

    config = checkpoint.get("model_config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint model_config must be an object")
    required_config = {
        "input_dim",
        "hidden_dim",
        "output_dim",
        "num_layers",
        "dropout",
        "history_length",
    }
    if required_config - set(config):
        raise ValueError(f"Checkpoint model_config is missing {sorted(required_config - set(config))}")
    if int(config["input_dim"]) != 6 or int(config["output_dim"]) != 6:
        raise ValueError("This visualizer requires a 6-input, 6-output checkpoint")
    if metadata.get("model_config") != config:
        raise ValueError("Checkpoint and metadata model configurations do not match")

    feature_names = checkpoint.get("feature_names")
    if list(feature_names or []) != list(FEATURE_NAMES):
        raise ValueError("Checkpoint feature order does not match exploration_policy")
    if metadata.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("Metadata feature order does not match exploration_policy")

    normalization = checkpoint.get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError("Checkpoint normalization must be an object")
    for key in ("x_mean", "y_mean"):
        _finite_vector(normalization.get(key), length=6, name=f"normalization.{key}")
    for key in ("x_std", "y_std"):
        _finite_vector(
            normalization.get(key),
            length=6,
            name=f"normalization.{key}",
            positive=True,
        )
    low = _finite_vector(checkpoint.get("action_clip_low"), length=6, name="action_clip_low")
    high = _finite_vector(
        checkpoint.get("action_clip_high"), length=6, name="action_clip_high"
    )
    if np.any(low >= high):
        raise ValueError("Each action_clip_low value must be below action_clip_high")

    history_length = int(config["history_length"])
    seed = np.asarray(checkpoint.get("demo_seed_velocity"), dtype=np.float64)
    if seed.shape != (history_length, 6) or not np.all(np.isfinite(seed)):
        raise ValueError(
            "demo_seed_velocity must be a finite held-out-test window with shape "
            f"({history_length}, 6)"
        )
    dt_seconds = float(checkpoint.get("median_dt_seconds", float("nan")))
    if not np.isfinite(dt_seconds) or not 0.001 <= dt_seconds <= 0.2:
        raise ValueError("median_dt_seconds is missing or implausible")

    if not isinstance(scene.get("objects"), list) or not isinstance(scene.get("risk_zones"), list):
        raise ValueError("Unity parse must provide objects and risk_zones arrays")
    analysis_scope = scene.get("analysis_scope")
    if not isinstance(analysis_scope, dict) or not isinstance(
        analysis_scope.get("synthetic_input"), bool
    ):
        raise ValueError("Unity parse must declare analysis_scope.synthetic_input")

    model = build_model(config)
    try:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    except (KeyError, RuntimeError) as exc:
        raise ValueError("Checkpoint state_dict is incompatible with model_config") from exc
    model.eval()

    return RuntimeResources(
        model=model,
        checkpoint=checkpoint,
        metadata=metadata,
        scene=scene,
        model_path=model_path,
        metadata_path=metadata_path,
        scene_path=scene_path,
    )


def _world_position(record: dict[str, Any]) -> np.ndarray | None:
    if record.get("position_space") != "world":
        return None
    try:
        position = np.asarray(record.get("position"), dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        return None
    return position


def make_scene_overlay(scene: dict[str, Any]) -> SceneOverlay:
    """Filter to finite world-space data and rescale it for display only."""

    object_rows: list[tuple[dict[str, Any], np.ndarray]] = []
    risk_rows: list[tuple[dict[str, Any], np.ndarray]] = []
    for record in scene["objects"]:
        if isinstance(record, dict):
            position = _world_position(record)
            if position is not None:
                object_rows.append((record, position))
    for record in scene["risk_zones"]:
        if isinstance(record, dict):
            position = _world_position(record)
            if position is not None:
                risk_rows.append((record, position))

    all_positions = [position for _, position in object_rows + risk_rows]
    if all_positions:
        xz = np.asarray([[position[0], position[2]] for position in all_positions])
        minimum = xz.min(axis=0)
        maximum = xz.max(axis=0)
        centre = (minimum + maximum) * 0.5
        largest_span = float(np.max(maximum - minimum))
        coordinate_scale = 2.35 / max(largest_span, 1.0)
    else:
        centre = np.zeros(2, dtype=np.float64)
        coordinate_scale = 1.0

    def display_xz(position: np.ndarray) -> tuple[float, float]:
        transformed = (position[[0, 2]] - centre) * coordinate_scale
        return float(transformed[0]), float(transformed[1])

    objects: list[OverlayItem] = []
    for record, position in object_rows:
        x, z = display_xz(position)
        objects.append(
            OverlayItem(
                name=str(record.get("name") or "unnamed object"),
                category=str(record.get("type") or "unknown"),
                x=x,
                z=z,
            )
        )

    risks: list[OverlayItem] = []
    for record, position in risk_rows:
        x, z = display_xz(position)
        raw_radius = record.get("radius")
        used_default = True
        radius = DEFAULT_DISPLAY_RISK_RADIUS
        if isinstance(raw_radius, (int, float)) and np.isfinite(raw_radius) and raw_radius > 0:
            radius = max(0.025, float(raw_radius) * coordinate_scale)
            used_default = False
        risks.append(
            OverlayItem(
                name=str(record.get("zone_name") or "unnamed risk zone"),
                category=str(record.get("risk_type") or "unknown risk"),
                x=x,
                z=z,
                radius=radius,
                used_default_radius=used_default,
            )
        )

    analysis_scope = scene["analysis_scope"]
    return SceneOverlay(
        objects=tuple(objects),
        risks=tuple(risks),
        synthetic_input=bool(analysis_scope["synthetic_input"]),
        coordinate_scale=coordinate_scale,
        source_object_count=len(scene["objects"]),
        source_risk_count=len(scene["risk_zones"]),
    )


def _reflect_coordinate(value: float, velocity: float) -> tuple[float, float, int]:
    """Reflect an overshoot into the safety cube, returning event count."""

    lower = -STAGE_HALF_EXTENT_METERS
    upper = STAGE_HALF_EXTENT_METERS
    events = 0
    for _ in range(32):
        if value > upper:
            value = 2.0 * upper - value
            velocity = -velocity
            events += 1
        elif value < lower:
            value = 2.0 * lower - value
            velocity = -velocity
            events += 1
        else:
            return value, velocity, events
    # The checkpoint clips make this unreachable in normal use, but an exact
    # clamp gives corrupt inputs no path to escape the safety boundary.
    return float(np.clip(value, lower, upper)), 0.0, events + 1


def run_closed_loop(resources: RuntimeResources, steps: int) -> RolloutResult:
    """Roll out model outputs as the next inputs, starting from test seed history."""

    if steps < 1:
        raise ValueError("steps must be at least 1")
    checkpoint = resources.checkpoint
    normalization = checkpoint["normalization"]
    x_mean = _finite_vector(normalization["x_mean"], length=6, name="x_mean")
    x_std = _finite_vector(
        normalization["x_std"], length=6, name="x_std", positive=True
    )
    y_mean = _finite_vector(normalization["y_mean"], length=6, name="y_mean")
    y_std = _finite_vector(
        normalization["y_std"], length=6, name="y_std", positive=True
    )
    clip_low = _finite_vector(checkpoint["action_clip_low"], length=6, name="clip_low")
    clip_high = _finite_vector(checkpoint["action_clip_high"], length=6, name="clip_high")
    dt_seconds = float(checkpoint["median_dt_seconds"])
    history = np.asarray(checkpoint["demo_seed_velocity"], dtype=np.float64).copy()

    positions = np.zeros((steps + 1, 3), dtype=np.float64)
    orientations = np.zeros((steps + 1, 4), dtype=np.float64)
    orientations[0, 3] = 1.0
    actions = np.zeros((steps, 6), dtype=np.float64)
    boundary_events = np.zeros(steps, dtype=np.int64)

    model = resources.model
    with torch.inference_mode():
        for index in range(steps):
            normalized_history = (history - x_mean) / x_std
            tensor = torch.from_numpy(normalized_history.astype(np.float32))[None, :, :]
            normalized_prediction = model(tensor).cpu().numpy()[0].astype(np.float64)
            predicted_action = normalized_prediction * y_std + y_mean
            predicted_action = np.clip(predicted_action, clip_low, clip_high)
            if not np.all(np.isfinite(predicted_action)):
                raise RuntimeError(f"Model produced a non-finite action at step {index}")

            current_position = positions[index]
            current_orientation = orientations[index]
            next_position, next_orientation = integrate_local_velocity(
                current_position,
                current_orientation,
                predicted_action[:3],
                predicted_action[3:],
                dt_seconds,
            )

            # Reflect the actual world-frame translation at the 3 m safety cube.
            # When reflection occurs, feed the executed (not unbounded) velocity
            # back into the local motion history for the next closed-loop step.
            world_velocity = rotate_vectors(current_orientation, predicted_action[:3])
            reflected_world_velocity = np.asarray(world_velocity, dtype=np.float64).copy()
            step_events = 0
            for axis in range(3):
                next_position[axis], reflected_world_velocity[axis], events = (
                    _reflect_coordinate(next_position[axis], reflected_world_velocity[axis])
                )
                step_events += events
            if step_events:
                predicted_action[:3] = rotate_vectors(
                    quaternion_conjugate(current_orientation), reflected_world_velocity
                )

            positions[index + 1] = next_position
            orientations[index + 1] = next_orientation
            actions[index] = predicted_action
            boundary_events[index] = step_events
            history = np.concatenate((history[1:], predicted_action[None, :]), axis=0)

    if not (
        np.all(np.isfinite(positions))
        and np.all(np.isfinite(orientations))
        and np.all(np.isfinite(actions))
    ):
        raise RuntimeError("Closed-loop rollout contains non-finite values")
    if np.max(np.abs(positions)) > STAGE_HALF_EXTENT_METERS + 1e-9:
        raise RuntimeError("Safety-boundary invariant was violated")

    return RolloutResult(
        positions=positions,
        orientations=orientations,
        actions=actions,
        boundary_events_per_step=boundary_events,
        dt_seconds=dt_seconds,
    )


def _data_limits(values: np.ndarray, minimum_span: float) -> tuple[float, float]:
    lower = float(np.min(values))
    upper = float(np.max(values))
    span = max(upper - lower, minimum_span)
    centre = (upper + lower) * 0.5
    return centre - 0.6 * span, centre + 0.6 * span


def render_demo(
    resources: RuntimeResources,
    overlay: SceneOverlay,
    rollout: RolloutResult,
    *,
    output_dir: Path,
    save_gif: bool,
    save_static: bool,
    show: bool,
    fps: int,
) -> tuple[Path | None, Path | None]:
    """Render the top-down rollout and its executed 6-DoF velocity traces."""

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Rectangle

    if save_gif or save_static:
        output_dir.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(13.5, 7.6), facecolor="#f6f7fb")
    grid = figure.add_gridspec(2, 2, width_ratios=(1.12, 1.0), hspace=0.32, wspace=0.25)
    path_axes = figure.add_subplot(grid[:, 0])
    linear_axes = figure.add_subplot(grid[0, 1])
    angular_axes = figure.add_subplot(grid[1, 1])

    figure.suptitle(
        MODEL_TITLE,
        fontsize=15,
        fontweight="bold",
        color="#17213a",
        y=0.975,
    )

    path_axes.set_facecolor("#ffffff")
    path_axes.add_patch(
        Rectangle(
            (-STAGE_HALF_EXTENT_METERS, -STAGE_HALF_EXTENT_METERS),
            STAGE_HALF_EXTENT_METERS * 2.0,
            STAGE_HALF_EXTENT_METERS * 2.0,
            fill=False,
            edgecolor="#48566f",
            linewidth=1.5,
            linestyle="--",
            zorder=1,
        )
    )

    risk_location_indices: dict[tuple[float, float], int] = {}
    risk_key_entries: list[str] = []
    for risk_number, item in enumerate(overlay.risks, start=1):
        path_axes.add_patch(
            Circle(
                (item.x, item.z),
                float(item.radius),
                facecolor="#ef476f",
                edgecolor="#b62246",
                alpha=0.17,
                linewidth=1.0,
                zorder=2,
            )
        )
        suffix = " [default radius]" if item.used_default_radius else ""
        label_key = (round(item.x, 6), round(item.z, 6))
        label_index = risk_location_indices.get(label_key, 0)
        risk_location_indices[label_key] = label_index + 1
        label_x = item.x - float(item.radius) * 0.55 + 0.11 * label_index
        label_z = item.z + float(item.radius) * 0.55
        path_axes.text(
            label_x,
            label_z,
            f"R{risk_number}",
            fontsize=6.7,
            fontweight="bold",
            color="#8b1834",
            ha="center",
            va="center",
            zorder=3,
        )
        risk_key_entries.append(f"R{risk_number}  {item.category}{suffix}")

    if risk_key_entries:
        path_axes.text(
            0.985,
            0.025,
            "Risk overlay key (static cues):\n" + "\n".join(risk_key_entries),
            transform=path_axes.transAxes,
            fontsize=6.4,
            color="#8b1834",
            ha="right",
            va="bottom",
            linespacing=1.25,
            bbox={
                "facecolor": "white",
                "edgecolor": "#efb5c3",
                "alpha": 0.92,
                "boxstyle": "round,pad=0.35",
            },
            zorder=10,
        )

    if overlay.objects:
        path_axes.scatter(
            [item.x for item in overlay.objects],
            [item.z for item in overlay.objects],
            marker="s",
            s=25,
            c="#457b9d",
            edgecolors="white",
            linewidths=0.5,
            alpha=0.82,
            zorder=4,
        )
        for item in overlay.objects:
            path_axes.annotate(
                item.name,
                (item.x, item.z),
                xytext=(3, -3),
                textcoords="offset points",
                fontsize=6.2,
                color="#244b63",
                zorder=5,
            )

    trajectory_line, = path_axes.plot([], [], color="#ff9f1c", linewidth=2.4, zorder=7)
    current_marker, = path_axes.plot(
        [], [], marker="o", markersize=7.5, color="#e76f51", markeredgecolor="white", zorder=9
    )
    heading_line, = path_axes.plot([], [], color="#17213a", linewidth=2.0, zorder=8)
    boundary_text = path_axes.text(
        0.02,
        0.02,
        "",
        transform=path_axes.transAxes,
        fontsize=8,
        color="#39465e",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#d8dce6", "alpha": 0.9},
        zorder=10,
    )

    path_axes.set_xlim(-1.62, 1.62)
    path_axes.set_ylim(-1.62, 1.62)
    path_axes.set_aspect("equal", adjustable="box")
    path_axes.set_xlabel("display X / model world X (m)")
    path_axes.set_ylabel("display Z / model world Z (m)")
    path_axes.set_title("Top-down closed-loop rollout (3 m × 3 m safety boundary)")
    path_axes.grid(True, color="#e7e9ef", linewidth=0.7, zorder=0)
    path_axes.legend(
        handles=[
            Line2D([0], [0], color="#ff9f1c", lw=2.4, label="model rollout"),
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="none",
                color="#457b9d",
                label="finite world-space object",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                color="#ef476f",
                alpha=0.55,
                label="static risk candidate (display only)",
            ),
        ],
        loc="upper left",
        fontsize=7.5,
        framealpha=0.92,
    )

    time_seconds = (np.arange(rollout.actions.shape[0]) + 1) * rollout.dt_seconds
    linear_colours = ("#0077b6", "#2a9d8f", "#e76f51")
    angular_colours = ("#8338ec", "#ff006e", "#fb8500")
    labels = ("x", "y", "z")
    linear_lines = []
    angular_lines = []
    for index, (colour, label) in enumerate(zip(linear_colours, labels)):
        line, = linear_axes.plot([], [], color=colour, linewidth=1.6, label=f"v{label}")
        linear_lines.append(line)
    angular_degrees = np.rad2deg(rollout.actions[:, 3:])
    for index, (colour, label) in enumerate(zip(angular_colours, labels)):
        line, = angular_axes.plot([], [], color=colour, linewidth=1.6, label=f"ω{label}")
        angular_lines.append(line)

    max_time = max(float(time_seconds[-1]), rollout.dt_seconds)
    linear_axes.set_xlim(0.0, max_time)
    angular_axes.set_xlim(0.0, max_time)
    linear_axes.set_ylim(*_data_limits(rollout.actions[:, :3], minimum_span=0.12))
    angular_axes.set_ylim(*_data_limits(angular_degrees, minimum_span=8.0))
    linear_axes.set_title("Executed local linear velocity")
    angular_axes.set_title("Executed local angular velocity")
    linear_axes.set_ylabel("m/s")
    angular_axes.set_ylabel("deg/s")
    angular_axes.set_xlabel("rollout time (s)")
    for axes in (linear_axes, angular_axes):
        axes.set_facecolor("#ffffff")
        axes.grid(True, color="#e7e9ef", linewidth=0.7)
        axes.legend(loc="upper right", ncol=3, fontsize=8)

    overlay_kind = "SYNTHETIC Unity semantic overlay" if overlay.synthetic_input else "Unity semantic overlay"
    default_radius_count = sum(item.used_default_radius for item in overlay.risks)
    figure.text(
        0.5,
        0.016,
        (
            f"EyeNavGS held-out-test velocity seed • dt={rollout.dt_seconds:.3f}s • "
            f"{overlay_kind}: rescaled for display only, never consumed by the policy • "
            f"null risk radii shown with explicit default ({default_radius_count}/{len(overlay.risks)})"
        ),
        ha="center",
        va="bottom",
        fontsize=8.3,
        color="#4c5870",
    )
    figure.text(
        0.5,
        0.001,
        "No goal selection, collision reasoning, risk avoidance, or cybersickness prediction is demonstrated.",
        ha="center",
        va="bottom",
        fontsize=8.1,
        color="#8b1834",
    )
    figure.subplots_adjust(top=0.90, bottom=0.10)

    def update(frame: int) -> Sequence[Any]:
        pose_index = frame + 1
        positions = rollout.positions[: pose_index + 1]
        trajectory_line.set_data(positions[:, 0], positions[:, 2])
        current = rollout.positions[pose_index]
        current_marker.set_data([current[0]], [current[2]])
        forward = rotate_vectors(
            rollout.orientations[pose_index], np.asarray([0.0, 0.0, 1.0])
        )
        forward_xz = np.asarray([forward[0], forward[2]], dtype=np.float64)
        norm = float(np.linalg.norm(forward_xz))
        if norm > 1e-9:
            forward_xz = forward_xz / norm * 0.16
        heading_line.set_data(
            [current[0], current[0] + forward_xz[0]],
            [current[2], current[2] + forward_xz[1]],
        )
        elapsed = time_seconds[: pose_index]
        for axis, line in enumerate(linear_lines):
            line.set_data(elapsed, rollout.actions[:pose_index, axis])
        for axis, line in enumerate(angular_lines):
            line.set_data(elapsed, angular_degrees[:pose_index, axis])
        events = int(rollout.boundary_events_per_step[:pose_index].sum())
        boundary_text.set_text(
            f"step {pose_index}/{rollout.actions.shape[0]}\n"
            f"safety reflections: {events}"
        )
        return (
            trajectory_line,
            current_marker,
            heading_line,
            boundary_text,
            *linear_lines,
            *angular_lines,
        )

    gif_path: Path | None = None
    static_path: Path | None = None
    update(rollout.actions.shape[0] - 1)
    if save_static:
        static_path = output_dir / "agent_trajectory_static.png"
        figure.savefig(static_path, dpi=155, facecolor=figure.get_facecolor())

    animation: FuncAnimation | None = None
    if save_gif or show:
        animation = FuncAnimation(
            figure,
            update,
            frames=rollout.actions.shape[0],
            interval=1000.0 / fps,
            repeat=show,
            blit=False,
        )
    if save_gif:
        gif_path = output_dir / "agent_exploration.gif"
        assert animation is not None
        animation.save(gif_path, writer=PillowWriter(fps=fps), dpi=95)

    if show:
        plt.show()
    plt.close(figure)
    return gif_path, static_path


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _default_output_dir() -> Path:
    if getattr(sys, "frozen", False):
        local_app_data = os.environ.get("LOCALAPPDATA")
        base = Path(local_app_data) if local_app_data else Path.home()
        return base / APP_NAME
    return Path(__file__).resolve().parent


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize a real EyeNavGS-trained 6-DoF short-horizon motion "
            "imitation model. This is not scene-aware navigation."
        )
    )
    parser.add_argument("--no-show", action="store_true", help="Do not open the live window")
    parser.add_argument("--save-gif", action="store_true", help="Save agent_exploration.gif")
    parser.add_argument(
        "--save-static", action="store_true", help="Save agent_trajectory_static.png"
    )
    parser.add_argument(
        "--steps", type=_positive_int, default=240, help="Closed-loop rollout steps (default: 240)"
    )
    parser.add_argument(
        "--fps", type=_positive_int, default=20, help="Animation frames per second (default: 20)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for requested GIF/PNG outputs",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Load all bundled resources, run a short inference, and exit",
    )
    args = parser.parse_args(argv)
    if args.fps > 60:
        parser.error("--fps must not exceed 60")
    if args.steps > 5000:
        parser.error("--steps must not exceed 5000")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    import matplotlib

    if args.no_show or args.self_test:
        matplotlib.use("Agg")
    else:
        # Pick a real GUI backend explicitly.  PyInstaller's Matplotlib hook
        # otherwise sees the optional Agg call above and may package a
        # double-clickable program whose plt.show() cannot create a window.
        matplotlib.use("TkAgg")

    resources = load_runtime_resources()
    overlay = make_scene_overlay(resources.scene)
    rollout_steps = min(args.steps, 8) if args.self_test else args.steps
    rollout = run_closed_loop(resources, rollout_steps)

    if args.self_test:
        print(
            "SELF_TEST_OK "
            f"source={resources.checkpoint['data_source']} "
            f"synthetic_training={resources.checkpoint['synthetic']} "
            f"steps={rollout_steps} "
            f"world_objects={len(overlay.objects)} "
            f"world_risks={len(overlay.risks)}"
        )
        return 0

    output_dir = (args.output_dir or _default_output_dir()).expanduser().resolve()
    gif_path, static_path = render_demo(
        resources,
        overlay,
        rollout,
        output_dir=output_dir,
        save_gif=args.save_gif,
        save_static=args.save_static,
        show=not args.no_show,
        fps=args.fps,
    )
    print(
        f"Completed {args.steps} closed-loop steps from the held-out-test seed; "
        f"boundary reflections={rollout.boundary_event_count}."
    )
    if gif_path is not None:
        print(f"GIF: {gif_path}")
    if static_path is not None:
        print(f"PNG: {static_path}")
    return 0


def _show_startup_error(exc: BaseException) -> None:
    message = f"{type(exc).__name__}: {exc}"
    print(message, file=sys.stderr)
    # Windowed PyInstaller processes have no stderr.  The build-time smoke
    # harness supplies this private environment variable so a failed packaged
    # launch still leaves actionable diagnostics in its temporary directory.
    diagnostic_path = os.environ.get("AGENT_DEMO_ERROR_LOG")
    if diagnostic_path:
        try:
            Path(diagnostic_path).write_text(
                message + "\n\n" + traceback.format_exc(), encoding="utf-8"
            )
        except OSError:
            pass
    # A double-clicked --windowed EXE has no console.  Show the same fail-closed
    # error there, but never block automated --self-test runs with a dialog.
    if getattr(sys, "frozen", False) and "--self-test" not in sys.argv:
        try:
            from tkinter import messagebox

            messagebox.showerror(f"{APP_NAME} could not start", message)
        except Exception:
            pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:
        _show_startup_error(error)
        raise SystemExit(1)
