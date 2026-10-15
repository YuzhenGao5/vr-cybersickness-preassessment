#!/usr/bin/env python3
"""Independent recurrent branches and holding event controllers for HMD v4.

There are three independent recurrent pathways:

* BodyRoot: continuous 3-D state prediction;
* horizontal: a continuous pose proposal plus onset hazard and one Gaussian
  event-parameter distribution for HeadLocal x/z/yaw/roll;
* vertical: onset hazard plus one Gaussian event-parameter distribution for
  HeadLocal y/pitch.

At runtime an eligible hazard is sampled with a seeded Bernoulli draw.  A
successful draw samples a bounded absolute endpoint and duration exactly once,
then derives and latches the signed amplitude from the current pose.  A
quintic minimum-jerk transition moves to that target and holds it until another event.
No per-frame random value is ever added to a HeadLocal channel.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence

import torch
from torch import nn

from normal_hmd_v4_data import (
    HORIZONTAL_INDICES,
    ROOT_INDICES,
    VERTICAL_INDICES,
)


HORIZONTAL_PARAMETER_DIM = len(HORIZONTAL_INDICES) + 1
VERTICAL_PARAMETER_DIM = len(VERTICAL_INDICES) + 1


class V4Step(NamedTuple):
    root_normalized: torch.Tensor
    horizontal_pose_normalized: torch.Tensor
    horizontal_hazard_logit: torch.Tensor
    horizontal_parameter_mean_normalized: torch.Tensor
    horizontal_parameter_logstd: torch.Tensor
    vertical_hazard_logit: torch.Tensor
    vertical_parameter_mean_normalized: torch.Tensor
    vertical_parameter_logstd: torch.Tensor


class V4Sequence(NamedTuple):
    root_normalized: torch.Tensor
    horizontal_pose_normalized: torch.Tensor
    horizontal_hazard_logit: torch.Tensor
    horizontal_parameter_mean_normalized: torch.Tensor
    horizontal_parameter_logstd: torch.Tensor
    vertical_hazard_logit: torch.Tensor
    vertical_parameter_mean_normalized: torch.Tensor
    vertical_parameter_logstd: torch.Tensor


class V4Hidden(NamedTuple):
    root: tuple[torch.Tensor, torch.Tensor]
    horizontal: tuple[torch.Tensor, torch.Tensor]
    vertical: tuple[torch.Tensor, torch.Tensor]


class EventControllerState(NamedTuple):
    active: torch.Tensor
    cooldown_remaining: torch.Tensor
    elapsed_steps: torch.Tensor
    duration_steps: torch.Tensor
    start_pose_physical: torch.Tensor
    target_pose_physical: torch.Tensor
    amplitude_physical: torch.Tensor


class V4RuntimeState(NamedTuple):
    horizontal: EventControllerState
    vertical: EventControllerState


class RuntimeDiagnostics(NamedTuple):
    horizontal_trigger: torch.Tensor
    vertical_trigger: torch.Tensor
    horizontal_eligible: torch.Tensor
    vertical_eligible: torch.Tensor
    horizontal_probability: torch.Tensor
    vertical_probability: torch.Tensor


def minimum_jerk_torch(phase: torch.Tensor) -> torch.Tensor:
    """Quintic interpolation with zero velocity and acceleration at 0 and 1."""

    u = torch.clamp(phase, 0.0, 1.0)
    return 10.0 * u.pow(3) - 15.0 * u.pow(4) + 6.0 * u.pow(5)


def _as_vector(values: Sequence[float], expected: int, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
    if len(tensor) != expected:
        raise ValueError(f"{name} must have {expected} elements, got {len(tensor)}")
    return tensor


def _cholesky(correlation: Sequence[Sequence[float]], size: int, name: str) -> torch.Tensor:
    matrix = torch.as_tensor(correlation, dtype=torch.float32)
    if matrix.shape != (size, size):
        raise ValueError(f"{name} must have shape {(size, size)}, got {tuple(matrix.shape)}")
    if not torch.allclose(matrix, matrix.T, atol=1e-5, rtol=0.0):
        raise ValueError(f"{name} must be symmetric")
    try:
        return torch.linalg.cholesky(matrix)
    except RuntimeError as exc:
        raise ValueError(f"{name} must be positive definite") from exc


class NormalHMDV4(nn.Module):
    """Three recurrent branches plus independent horizontal/vertical hazards."""

    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 96,
        num_layers: int = 1,
        dropout: float = 0.0,
        *,
        state_mean: Sequence[float],
        state_std: Sequence[float],
        horizontal_parameter_center: Sequence[float],
        horizontal_parameter_scale: Sequence[float],
        vertical_parameter_center: Sequence[float],
        vertical_parameter_scale: Sequence[float],
        horizontal_parameter_low: Sequence[float],
        horizontal_parameter_high: Sequence[float],
        vertical_parameter_low: Sequence[float],
        vertical_parameter_high: Sequence[float],
        horizontal_pose_low: Sequence[float],
        horizontal_pose_high: Sequence[float],
        vertical_pose_low: Sequence[float],
        vertical_pose_high: Sequence[float],
        horizontal_parameter_correlation: Sequence[Sequence[float]] | None = None,
        vertical_parameter_correlation: Sequence[Sequence[float]] | None = None,
        cooldown_steps: int = 4,
        minimum_duration_steps: int = 5,
        maximum_duration_steps: int = 148,
    ) -> None:
        super().__init__()
        if input_dim != 9:
            raise ValueError("NormalHMDV4 requires the frozen 9-D state")
        if hidden_dim < 1 or num_layers < 1:
            raise ValueError("hidden_dim and num_layers must be positive")
        if cooldown_steps < 0 or minimum_duration_steps < 2:
            raise ValueError("invalid cooldown/minimum duration")
        if maximum_duration_steps < minimum_duration_steps:
            raise ValueError("maximum duration precedes minimum duration")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.cooldown_steps = int(cooldown_steps)
        self.minimum_duration_steps = int(minimum_duration_steps)
        self.maximum_duration_steps = int(maximum_duration_steps)

        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.root_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            dropout=recurrent_dropout, batch_first=True,
        )
        self.horizontal_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            dropout=recurrent_dropout, batch_first=True,
        )
        self.vertical_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            dropout=recurrent_dropout, batch_first=True,
        )
        self.root_output = nn.Linear(hidden_dim, len(ROOT_INDICES))
        self.horizontal_pose = nn.Linear(hidden_dim, len(HORIZONTAL_INDICES))
        self.horizontal_hazard = nn.Linear(hidden_dim, 1)
        self.horizontal_parameter_mean = nn.Linear(hidden_dim, HORIZONTAL_PARAMETER_DIM)
        self.horizontal_parameter_logstd = nn.Linear(hidden_dim, HORIZONTAL_PARAMETER_DIM)
        self.vertical_hazard = nn.Linear(hidden_dim, 1)
        self.vertical_parameter_mean = nn.Linear(hidden_dim, VERTICAL_PARAMETER_DIM)
        self.vertical_parameter_logstd = nn.Linear(hidden_dim, VERTICAL_PARAMETER_DIM)

        state_mean_tensor = _as_vector(state_mean, input_dim, "state_mean")
        state_std_tensor = _as_vector(state_std, input_dim, "state_std")
        if torch.any(state_std_tensor <= 0.0):
            raise ValueError("state_std must be positive")
        self.register_buffer("state_mean", state_mean_tensor)
        self.register_buffer("state_std", state_std_tensor)
        self.register_buffer(
            "horizontal_parameter_center",
            _as_vector(
                horizontal_parameter_center,
                HORIZONTAL_PARAMETER_DIM,
                "horizontal_parameter_center",
            ),
        )
        self.register_buffer(
            "horizontal_parameter_scale",
            _as_vector(
                horizontal_parameter_scale,
                HORIZONTAL_PARAMETER_DIM,
                "horizontal_parameter_scale",
            ),
        )
        self.register_buffer(
            "vertical_parameter_center",
            _as_vector(
                vertical_parameter_center,
                VERTICAL_PARAMETER_DIM,
                "vertical_parameter_center",
            ),
        )
        self.register_buffer(
            "vertical_parameter_scale",
            _as_vector(
                vertical_parameter_scale,
                VERTICAL_PARAMETER_DIM,
                "vertical_parameter_scale",
            ),
        )
        if torch.any(self.horizontal_parameter_scale <= 0.0) or torch.any(
            self.vertical_parameter_scale <= 0.0
        ):
            raise ValueError("parameter scales must be positive")

        for name, values, expected in (
            ("horizontal_parameter_low", horizontal_parameter_low, HORIZONTAL_PARAMETER_DIM),
            ("horizontal_parameter_high", horizontal_parameter_high, HORIZONTAL_PARAMETER_DIM),
            ("vertical_parameter_low", vertical_parameter_low, VERTICAL_PARAMETER_DIM),
            ("vertical_parameter_high", vertical_parameter_high, VERTICAL_PARAMETER_DIM),
            ("horizontal_pose_low", horizontal_pose_low, len(HORIZONTAL_INDICES)),
            ("horizontal_pose_high", horizontal_pose_high, len(HORIZONTAL_INDICES)),
            ("vertical_pose_low", vertical_pose_low, len(VERTICAL_INDICES)),
            ("vertical_pose_high", vertical_pose_high, len(VERTICAL_INDICES)),
        ):
            self.register_buffer(name, _as_vector(values, expected, name))
        if torch.any(self.horizontal_parameter_low >= self.horizontal_parameter_high):
            raise ValueError("horizontal parameter bounds are not ordered")
        if torch.any(self.vertical_parameter_low >= self.vertical_parameter_high):
            raise ValueError("vertical parameter bounds are not ordered")
        if torch.any(self.horizontal_pose_low >= self.horizontal_pose_high):
            raise ValueError("horizontal pose bounds are not ordered")
        if torch.any(self.vertical_pose_low >= self.vertical_pose_high):
            raise ValueError("vertical pose bounds are not ordered")

        horizontal_correlation = (
            torch.eye(HORIZONTAL_PARAMETER_DIM).tolist()
            if horizontal_parameter_correlation is None
            else horizontal_parameter_correlation
        )
        vertical_correlation = (
            torch.eye(VERTICAL_PARAMETER_DIM).tolist()
            if vertical_parameter_correlation is None
            else vertical_parameter_correlation
        )
        self.register_buffer(
            "horizontal_parameter_cholesky",
            _cholesky(
                horizontal_correlation,
                HORIZONTAL_PARAMETER_DIM,
                "horizontal_parameter_correlation",
            ),
        )
        self.register_buffer(
            "vertical_parameter_cholesky",
            _cholesky(
                vertical_correlation,
                VERTICAL_PARAMETER_DIM,
                "vertical_parameter_correlation",
            ),
        )
        self.register_buffer("horizontal_hazard_bias", torch.tensor(0.0))
        self.register_buffer("vertical_hazard_bias", torch.tensor(0.0))

    def _decode(
        self,
        root_encoded: torch.Tensor,
        horizontal_encoded: torch.Tensor,
        vertical_encoded: torch.Tensor,
    ) -> V4Sequence:
        return V4Sequence(
            root_normalized=self.root_output(root_encoded),
            horizontal_pose_normalized=self.horizontal_pose(horizontal_encoded),
            horizontal_hazard_logit=self.horizontal_hazard(horizontal_encoded).squeeze(-1),
            horizontal_parameter_mean_normalized=self.horizontal_parameter_mean(
                horizontal_encoded
            ),
            horizontal_parameter_logstd=torch.clamp(
                self.horizontal_parameter_logstd(horizontal_encoded), -5.0, 2.0
            ),
            vertical_hazard_logit=self.vertical_hazard(vertical_encoded).squeeze(-1),
            vertical_parameter_mean_normalized=self.vertical_parameter_mean(
                vertical_encoded
            ),
            vertical_parameter_logstd=torch.clamp(
                self.vertical_parameter_logstd(vertical_encoded), -5.0, 2.0
            ),
        )

    @staticmethod
    def _last(sequence: V4Sequence) -> V4Step:
        return V4Step(*(value[:, -1] for value in sequence))

    def forward(self, state_sequence_normalized: torch.Tensor) -> V4Sequence:
        root, _ = self.root_lstm(state_sequence_normalized)
        horizontal, _ = self.horizontal_lstm(state_sequence_normalized)
        vertical, _ = self.vertical_lstm(state_sequence_normalized)
        return self._decode(root, horizontal, vertical)

    def teacher_forced(
        self, history_normalized: torch.Tensor, target_normalized: torch.Tensor
    ) -> V4Sequence:
        if target_normalized.shape[1] < 1:
            raise ValueError("target sequence is empty")
        inputs = torch.cat(
            (history_normalized, target_normalized[:, :-1]), dim=1
        )
        output = self(inputs)
        first = history_normalized.shape[1] - 1
        stop = first + target_normalized.shape[1]
        return V4Sequence(*(value[:, first:stop] for value in output))

    def warm_history(
        self, history_normalized: torch.Tensor
    ) -> tuple[V4Step, V4Hidden]:
        root, root_hidden = self.root_lstm(history_normalized)
        horizontal, horizontal_hidden = self.horizontal_lstm(history_normalized)
        vertical, vertical_hidden = self.vertical_lstm(history_normalized)
        decoded = self._decode(root, horizontal, vertical)
        return self._last(decoded), V4Hidden(root_hidden, horizontal_hidden, vertical_hidden)

    def advance(
        self, feedback_normalized: torch.Tensor, hidden: V4Hidden
    ) -> tuple[V4Step, V4Hidden]:
        root, root_hidden = self.root_lstm(feedback_normalized[:, None], hidden.root)
        horizontal, horizontal_hidden = self.horizontal_lstm(
            feedback_normalized[:, None], hidden.horizontal
        )
        vertical, vertical_hidden = self.vertical_lstm(
            feedback_normalized[:, None], hidden.vertical
        )
        decoded = self._decode(root, horizontal, vertical)
        return self._last(decoded), V4Hidden(
            root_hidden, horizontal_hidden, vertical_hidden
        )

    def initial_runtime_state(self, previous_normalized: torch.Tensor) -> V4RuntimeState:
        physical = previous_normalized * self.state_std + self.state_mean

        def branch(indices: tuple[int, ...]) -> EventControllerState:
            pose = physical[:, indices]
            batch = len(previous_normalized)
            return EventControllerState(
                active=torch.zeros(batch, dtype=torch.bool, device=physical.device),
                cooldown_remaining=torch.zeros(
                    batch, dtype=torch.long, device=physical.device
                ),
                elapsed_steps=torch.zeros(batch, dtype=torch.long, device=physical.device),
                duration_steps=torch.full(
                    (batch,),
                    self.minimum_duration_steps,
                    dtype=torch.long,
                    device=physical.device,
                ),
                start_pose_physical=pose.clone(),
                target_pose_physical=pose.clone(),
                amplitude_physical=torch.zeros_like(pose),
            )

        return V4RuntimeState(
            horizontal=branch(HORIZONTAL_INDICES),
            vertical=branch(VERTICAL_INDICES),
        )

    @staticmethod
    def seeded_generator(device: torch.device, seed: int) -> torch.Generator:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        return generator

    def _sample_triggers(
        self,
        probability: torch.Tensor,
        eligible: torch.Tensor,
        generator: torch.Generator,
        override: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if override is not None:
            return override.to(dtype=torch.bool, device=eligible.device) & eligible
        trigger = torch.zeros_like(eligible)
        eligible_indices = torch.nonzero(eligible, as_tuple=False).flatten()
        if len(eligible_indices):
            uniform = torch.rand(
                len(eligible_indices),
                dtype=probability.dtype,
                device=probability.device,
                generator=generator,
            )
            trigger[eligible_indices] = uniform < probability[eligible_indices]
        return trigger

    def _sample_parameters(
        self,
        mean_normalized: torch.Tensor,
        logstd: torch.Tensor,
        trigger: torch.Tensor,
        center: torch.Tensor,
        scale: torch.Tensor,
        low: torch.Tensor,
        high: torch.Tensor,
        cholesky: torch.Tensor,
        generator: torch.Generator,
        override_physical: Optional[torch.Tensor],
    ) -> torch.Tensor:
        sampled_normalized = mean_normalized.clone()
        trigger_indices = torch.nonzero(trigger, as_tuple=False).flatten()
        if override_physical is not None:
            physical = override_physical.to(mean_normalized)
            if physical.shape != mean_normalized.shape:
                raise ValueError("parameter override has the wrong shape")
            return torch.clamp(physical, low, high)
        if len(trigger_indices):
            noise = torch.randn(
                (len(trigger_indices), mean_normalized.shape[-1]),
                dtype=mean_normalized.dtype,
                device=mean_normalized.device,
                generator=generator,
            )
            correlated = noise @ cholesky.T
            sampled_normalized[trigger_indices] = (
                mean_normalized[trigger_indices]
                + torch.exp(torch.clamp(logstd[trigger_indices], -5.0, 2.0))
                * correlated
            )
        physical = sampled_normalized * scale + center
        # Smoothly squash Gaussian samples into empirical comfort bounds.  A
        # hard clamp would put artificial point mass exactly on the bounds and
        # make long rollouts dwell there.
        midpoint = 0.5 * (low + high)
        half_range = torch.clamp(0.5 * (high - low), min=1e-6)
        return midpoint + half_range * torch.tanh(
            (physical - midpoint) / half_range
        )

    def _compose_branch(
        self,
        previous_normalized: torch.Tensor,
        indices: tuple[int, ...],
        controller: EventControllerState,
        hazard_logit: torch.Tensor,
        hazard_bias: torch.Tensor,
        parameter_mean_normalized: torch.Tensor,
        parameter_logstd: torch.Tensor,
        parameter_center: torch.Tensor,
        parameter_scale: torch.Tensor,
        parameter_low: torch.Tensor,
        parameter_high: torch.Tensor,
        pose_low: torch.Tensor,
        pose_high: torch.Tensor,
        cholesky: torch.Tensor,
        generator: torch.Generator,
        trigger_override: Optional[torch.Tensor] = None,
        parameter_override_physical: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, EventControllerState, torch.Tensor, torch.Tensor, torch.Tensor]:
        previous_physical = (
            previous_normalized[:, indices] * self.state_std[list(indices)]
            + self.state_mean[list(indices)]
        )
        eligible = (~controller.active) & (controller.cooldown_remaining <= 0)
        probability = torch.sigmoid(hazard_logit + hazard_bias)
        trigger = self._sample_triggers(
            probability, eligible, generator, trigger_override
        )
        sampled = self._sample_parameters(
            parameter_mean_normalized,
            parameter_logstd,
            trigger,
            parameter_center,
            parameter_scale,
            parameter_low,
            parameter_high,
            cholesky,
            generator,
            parameter_override_physical,
        )
        sampled_target = sampled[:, :-1]
        proposed_duration = torch.round(torch.exp(sampled[:, -1])).to(torch.long)
        proposed_duration = torch.clamp(
            proposed_duration,
            self.minimum_duration_steps,
            self.maximum_duration_steps,
        )
        # Event heads model bounded absolute endpoints, not unconditional
        # deltas.  The outer pose clamp is a final safety envelope; the tighter
        # learned P05--P95 parameter bounds already constrain ordinary samples.
        proposed_target = torch.clamp(sampled_target, pose_low, pose_high)
        actual_amplitude = proposed_target - previous_physical

        start = torch.where(
            trigger[:, None], previous_physical, controller.start_pose_physical
        )
        target = torch.where(
            trigger[:, None], proposed_target, controller.target_pose_physical
        )
        amplitude = torch.where(
            trigger[:, None], actual_amplitude, controller.amplitude_physical
        )
        duration = torch.where(trigger, proposed_duration, controller.duration_steps)
        active_now = controller.active | trigger
        elapsed = torch.where(
            trigger,
            torch.zeros_like(controller.elapsed_steps),
            torch.where(
                controller.active,
                controller.elapsed_steps + 1,
                controller.elapsed_steps,
            ),
        )
        denominator = torch.clamp(duration - 1, min=1).to(previous_physical.dtype)
        phase = elapsed.to(previous_physical.dtype) / denominator
        weight = minimum_jerk_torch(phase)
        transition_pose = start + (target - start) * weight[:, None]
        pose = torch.where(active_now[:, None], transition_pose, controller.target_pose_physical)
        finished = active_now & (elapsed >= duration - 1)
        pose = torch.where(finished[:, None], target, pose)

        decremented_cooldown = torch.clamp(controller.cooldown_remaining - 1, min=0)
        next_cooldown = torch.where(
            finished,
            torch.full_like(decremented_cooldown, self.cooldown_steps),
            torch.where(active_now, controller.cooldown_remaining, decremented_cooldown),
        )
        next_controller = EventControllerState(
            active=active_now & ~finished,
            cooldown_remaining=next_cooldown,
            elapsed_steps=torch.where(finished, torch.zeros_like(elapsed), elapsed),
            duration_steps=duration,
            start_pose_physical=start,
            target_pose_physical=target,
            amplitude_physical=amplitude,
        )
        normalized = (
            pose - self.state_mean[list(indices)]
        ) / self.state_std[list(indices)]
        return normalized, next_controller, trigger, eligible, probability

    def runtime_step(
        self,
        previous_normalized: torch.Tensor,
        step: V4Step,
        runtime: V4RuntimeState,
        generator: torch.Generator,
        *,
        horizontal_trigger_override: Optional[torch.Tensor] = None,
        vertical_trigger_override: Optional[torch.Tensor] = None,
        horizontal_parameter_override_physical: Optional[torch.Tensor] = None,
        vertical_parameter_override_physical: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, V4RuntimeState, RuntimeDiagnostics]:
        horizontal, horizontal_controller, h_trigger, h_eligible, h_probability = (
            self._compose_branch(
                previous_normalized,
                HORIZONTAL_INDICES,
                runtime.horizontal,
                step.horizontal_hazard_logit,
                self.horizontal_hazard_bias,
                step.horizontal_parameter_mean_normalized,
                step.horizontal_parameter_logstd,
                self.horizontal_parameter_center,
                self.horizontal_parameter_scale,
                self.horizontal_parameter_low,
                self.horizontal_parameter_high,
                self.horizontal_pose_low,
                self.horizontal_pose_high,
                self.horizontal_parameter_cholesky,
                generator,
                horizontal_trigger_override,
                horizontal_parameter_override_physical,
            )
        )
        vertical, vertical_controller, v_trigger, v_eligible, v_probability = (
            self._compose_branch(
                previous_normalized,
                VERTICAL_INDICES,
                runtime.vertical,
                step.vertical_hazard_logit,
                self.vertical_hazard_bias,
                step.vertical_parameter_mean_normalized,
                step.vertical_parameter_logstd,
                self.vertical_parameter_center,
                self.vertical_parameter_scale,
                self.vertical_parameter_low,
                self.vertical_parameter_high,
                self.vertical_pose_low,
                self.vertical_pose_high,
                self.vertical_parameter_cholesky,
                generator,
                vertical_trigger_override,
                vertical_parameter_override_physical,
            )
        )
        values: list[torch.Tensor] = []
        root_position = {index: offset for offset, index in enumerate(ROOT_INDICES)}
        horizontal_position = {
            index: offset for offset, index in enumerate(HORIZONTAL_INDICES)
        }
        vertical_position = {
            index: offset for offset, index in enumerate(VERTICAL_INDICES)
        }
        for index in range(self.input_dim):
            if index in root_position:
                values.append(step.root_normalized[:, root_position[index]])
            elif index in horizontal_position:
                values.append(horizontal[:, horizontal_position[index]])
            else:
                values.append(vertical[:, vertical_position[index]])
        state = torch.stack(values, dim=-1)
        return (
            state,
            V4RuntimeState(horizontal_controller, vertical_controller),
            RuntimeDiagnostics(
                h_trigger,
                v_trigger,
                h_eligible,
                v_eligible,
                h_probability,
                v_probability,
            ),
        )

    @torch.no_grad()
    def runtime_rollout(
        self,
        history_normalized: torch.Tensor,
        steps: int,
        *,
        seed: int,
        root_noise_std_normalized: float = 0.0,
        root_noise_rho: float = 0.95,
    ) -> tuple[torch.Tensor, list[RuntimeDiagnostics]]:
        if steps < 1:
            raise ValueError("steps must be positive")
        if root_noise_std_normalized < 0.0:
            raise ValueError("root noise standard deviation cannot be negative")
        if not 0.0 <= root_noise_rho < 1.0:
            raise ValueError("root_noise_rho must be in [0, 1)")
        event_generator = self.seeded_generator(history_normalized.device, seed)
        root_generator = self.seeded_generator(history_normalized.device, seed + 1)
        step_output, hidden = self.warm_history(history_normalized)
        previous = history_normalized[:, -1]
        runtime = self.initial_runtime_state(previous)
        root_noise = torch.zeros_like(previous[:, : len(ROOT_INDICES)])
        states: list[torch.Tensor] = []
        diagnostics: list[RuntimeDiagnostics] = []
        for index in range(steps):
            state, runtime, diagnostic = self.runtime_step(
                previous, step_output, runtime, event_generator
            )
            if root_noise_std_normalized > 0.0:
                gain = (1.0 - root_noise_rho**2) ** 0.5
                root_noise = root_noise_rho * root_noise + gain * torch.randn(
                    root_noise.shape,
                    dtype=root_noise.dtype,
                    device=root_noise.device,
                    generator=root_generator,
                ) * root_noise_std_normalized
                state = state.clone()
                state[:, : len(ROOT_INDICES)] += root_noise
            states.append(state)
            diagnostics.append(diagnostic)
            previous = state
            if index + 1 < steps:
                step_output, hidden = self.advance(previous, hidden)
        return torch.stack(states, dim=1), diagnostics


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))
