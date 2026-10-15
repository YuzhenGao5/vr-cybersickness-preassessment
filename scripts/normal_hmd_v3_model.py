#!/usr/bin/env python3
"""Two-branch recurrent model for Normal HMD Agent v3.

The shared LSTM sees the complete BodyRoot/HeadLocal state.  One decoder emits
the seven non-vertical channels; a separate decoder emits one coordinated
vertical activity gate plus HeadLocal-Y and pitch event parameters.  Both axes
use the same closed event phase, so no per-frame random term is required.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
from torch import nn


VERTICAL_INDEX = 4
PITCH_INDEX = 6
MAIN_STATE_INDICES = (0, 1, 2, 3, 5, 7, 8)


class V3Step(NamedTuple):
    main_normalized: torch.Tensor
    vertical_event_logit: torch.Tensor
    vertical_velocity_normalized: torch.Tensor
    vertical_amplitude_normalized: torch.Tensor
    pitch_velocity_normalized: torch.Tensor
    pitch_amplitude_normalized: torch.Tensor
    vertical_log_duration_normalized: torch.Tensor


class V3Rollout(NamedTuple):
    state_normalized: torch.Tensor
    vertical_event_logit: torch.Tensor
    vertical_velocity_normalized: torch.Tensor
    vertical_amplitude_normalized: torch.Tensor
    pitch_velocity_normalized: torch.Tensor
    pitch_amplitude_normalized: torch.Tensor
    vertical_log_duration_normalized: torch.Tensor


class VerticalControllerState(NamedTuple):
    active: torch.Tensor
    previous_gate: torch.Tensor
    elapsed_steps: torch.Tensor
    duration_steps: torch.Tensor
    baseline_y_physical: torch.Tensor
    amplitude_physical: torch.Tensor
    baseline_pitch_physical: torch.Tensor
    pitch_amplitude_physical: torch.Tensor


class NormalHMDV3(nn.Module):
    """Shared temporal encoder with separate main-motion and vertical heads."""

    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        *,
        fixed_dt_seconds: float,
        vertical_state_mean: float,
        vertical_state_std: float,
        vertical_velocity_mean: float,
        vertical_velocity_std: float,
        vertical_amplitude_mean: float,
        vertical_amplitude_std: float,
        vertical_log_duration_mean: float,
        vertical_log_duration_std: float,
        pitch_state_mean: float,
        pitch_state_std: float,
        pitch_velocity_mean: float,
        pitch_velocity_std: float,
        pitch_amplitude_mean: float,
        pitch_amplitude_std: float,
        event_gate_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        if input_dim != 9:
            raise ValueError("Normal HMD v3 requires the frozen 9-D state input")
        if hidden_dim < 1 or num_layers < 1:
            raise ValueError("hidden_dim and num_layers must be positive")
        if (
            fixed_dt_seconds <= 0.0
            or vertical_state_std <= 0.0
            or vertical_velocity_std <= 0.0
            or vertical_amplitude_std <= 0.0
            or vertical_log_duration_std <= 0.0
            or pitch_state_std <= 0.0
            or pitch_velocity_std <= 0.0
            or pitch_amplitude_std <= 0.0
        ):
            raise ValueError("invalid vertical integration normalization")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.fixed_dt_seconds = float(fixed_dt_seconds)
        self.vertical_state_mean = float(vertical_state_mean)
        self.vertical_state_std = float(vertical_state_std)
        self.vertical_velocity_mean = float(vertical_velocity_mean)
        self.vertical_velocity_std = float(vertical_velocity_std)
        self.vertical_amplitude_mean = float(vertical_amplitude_mean)
        self.vertical_amplitude_std = float(vertical_amplitude_std)
        self.vertical_log_duration_mean = float(vertical_log_duration_mean)
        self.vertical_log_duration_std = float(vertical_log_duration_std)
        self.pitch_state_mean = float(pitch_state_mean)
        self.pitch_state_std = float(pitch_state_std)
        self.pitch_velocity_mean = float(pitch_velocity_mean)
        self.pitch_velocity_std = float(pitch_velocity_std)
        self.pitch_amplitude_mean = float(pitch_amplitude_mean)
        self.pitch_amplitude_std = float(pitch_amplitude_std)
        self.event_gate_threshold = float(event_gate_threshold)
        if not 0.0 < self.event_gate_threshold < 1.0:
            raise ValueError("event_gate_threshold must be between zero and one")

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.main_output = nn.Linear(hidden_dim, len(MAIN_STATE_INDICES))
        self.vertical_event = nn.Linear(hidden_dim, 1)
        self.vertical_velocity = nn.Linear(hidden_dim, 1)
        self.vertical_amplitude = nn.Linear(hidden_dim, 1)
        self.pitch_velocity = nn.Linear(hidden_dim, 1)
        self.pitch_amplitude = nn.Linear(hidden_dim, 1)
        self.vertical_log_duration = nn.Linear(hidden_dim, 1)

    def _decode(self, encoded: torch.Tensor) -> V3Step:
        return V3Step(
            main_normalized=self.main_output(encoded),
            vertical_event_logit=self.vertical_event(encoded).squeeze(-1),
            vertical_velocity_normalized=self.vertical_velocity(encoded).squeeze(-1),
            vertical_amplitude_normalized=self.vertical_amplitude(encoded).squeeze(-1),
            pitch_velocity_normalized=self.pitch_velocity(encoded).squeeze(-1),
            pitch_amplitude_normalized=self.pitch_amplitude(encoded).squeeze(-1),
            vertical_log_duration_normalized=self.vertical_log_duration(encoded).squeeze(-1),
        )

    def warm_history(
        self, history: torch.Tensor
    ) -> tuple[V3Step, tuple[torch.Tensor, torch.Tensor]]:
        encoded, hidden = self.lstm(history)
        return self._decode(encoded[:, -1]), hidden

    def advance(
        self,
        feedback_state: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[V3Step, tuple[torch.Tensor, torch.Tensor]]:
        encoded, hidden = self.lstm(feedback_state[:, None, :], hidden)
        return self._decode(encoded[:, 0]), hidden

    def initial_vertical_controller(
        self, previous_state: torch.Tensor
    ) -> VerticalControllerState:
        previous_y_physical = (
            previous_state[:, VERTICAL_INDEX] * self.vertical_state_std
            + self.vertical_state_mean
        )
        previous_pitch_physical = (
            previous_state[:, PITCH_INDEX] * self.pitch_state_std
            + self.pitch_state_mean
        )
        zeros = torch.zeros_like(previous_y_physical)
        false = torch.zeros_like(previous_y_physical, dtype=torch.bool)
        return VerticalControllerState(
            active=false,
            previous_gate=false,
            elapsed_steps=zeros,
            duration_steps=torch.ones_like(previous_y_physical),
            baseline_y_physical=previous_y_physical,
            amplitude_physical=zeros,
            baseline_pitch_physical=previous_pitch_physical,
            pitch_amplitude_physical=zeros,
        )

    def compose_state(
        self,
        previous_state: torch.Tensor,
        step: V3Step,
        controller: VerticalControllerState,
        gate_override: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, VerticalControllerState]:
        """Advance a closed, zero-endpoint-velocity vertical event template.

        A gate rising edge starts one signed bump.  Amplitude and duration are
        sampled once from the neural heads at onset; they are not redrawn every
        frame.  ``16*u^2*(1-u)^2`` is zero with zero derivative at both ends,
        so every event returns exactly to its onset baseline without drift.
        """

        previous_y_physical = (
            previous_state[:, VERTICAL_INDEX] * self.vertical_state_std
            + self.vertical_state_mean
        )
        previous_pitch_physical = (
            previous_state[:, PITCH_INDEX] * self.pitch_state_std
            + self.pitch_state_mean
        )
        gate_value = (
            torch.sigmoid(step.vertical_event_logit)
            if gate_override is None
            else gate_override
        )
        gate = gate_value >= self.event_gate_threshold
        trigger = gate & ~controller.previous_gate & ~controller.active
        proposed_amplitude = (
            step.vertical_amplitude_normalized * self.vertical_amplitude_std
            + self.vertical_amplitude_mean
        )
        proposed_pitch_amplitude = (
            step.pitch_amplitude_normalized * self.pitch_amplitude_std
            + self.pitch_amplitude_mean
        )
        proposed_log_duration = (
            step.vertical_log_duration_normalized * self.vertical_log_duration_std
            + self.vertical_log_duration_mean
        )
        proposed_duration = torch.clamp(torch.exp(proposed_log_duration), 4.0, 74.0)
        baseline = torch.where(trigger, previous_y_physical, controller.baseline_y_physical)
        amplitude = torch.where(trigger, proposed_amplitude, controller.amplitude_physical)
        pitch_baseline = torch.where(
            trigger, previous_pitch_physical, controller.baseline_pitch_physical
        )
        pitch_amplitude = torch.where(
            trigger, proposed_pitch_amplitude, controller.pitch_amplitude_physical
        )
        duration = torch.where(trigger, proposed_duration, controller.duration_steps)
        active_now = controller.active | trigger
        elapsed = torch.where(
            trigger,
            torch.ones_like(controller.elapsed_steps),
            torch.where(
                controller.active,
                controller.elapsed_steps + 1.0,
                torch.zeros_like(controller.elapsed_steps),
            ),
        )
        phase = torch.clamp(elapsed / torch.clamp(duration, min=1.0), 0.0, 1.0)
        bump = 16.0 * phase.square() * (1.0 - phase).square()
        template_y = baseline + amplitude * bump
        template_pitch = pitch_baseline + pitch_amplitude * bump
        finished = active_now & (elapsed >= duration)
        next_y_physical = torch.where(
            finished,
            baseline,
            torch.where(active_now, template_y, previous_y_physical),
        )
        next_pitch_physical = torch.where(
            finished,
            pitch_baseline,
            torch.where(active_now, template_pitch, previous_pitch_physical),
        )
        next_y_normalized = (
            next_y_physical - self.vertical_state_mean
        ) / self.vertical_state_std
        next_pitch_normalized = (
            next_pitch_physical - self.pitch_state_mean
        ) / self.pitch_state_std

        main_by_index = {
            state_index: step.main_normalized[:, main_index]
            for main_index, state_index in enumerate(MAIN_STATE_INDICES)
        }
        features = [
            next_y_normalized
            if index == VERTICAL_INDEX
            else next_pitch_normalized
            if index == PITCH_INDEX
            else main_by_index[index]
            for index in range(self.input_dim)
        ]
        state = torch.stack(features, dim=-1)
        next_controller = VerticalControllerState(
            active=active_now & ~finished,
            previous_gate=gate,
            elapsed_steps=torch.where(finished, torch.zeros_like(elapsed), elapsed),
            duration_steps=duration,
            baseline_y_physical=baseline,
            amplitude_physical=amplitude,
            baseline_pitch_physical=pitch_baseline,
            pitch_amplitude_physical=pitch_amplitude,
        )
        return state, next_controller

    def rollout(
        self,
        history: torch.Tensor,
        steps: int,
        teacher: Optional[torch.Tensor] = None,
        vertical_gate_teacher: Optional[torch.Tensor] = None,
        vertical_onset_teacher: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
        root_feedback_noise_std: float = 0.0,
        root_feedback_noise_rho: float = 0.95,
    ) -> V3Rollout:
        """Closed-loop rollout with optional noisy BodyRoot feedback augmentation.

        The noise augmentation touches only normalized BodyRoot channels 0:3.
        It trains the shared encoder to accept a deployed safe/applied root state,
        while HeadLocal feedback stays the model's noise-free generated state.
        """

        if steps < 1:
            raise ValueError("steps must be positive")
        if teacher is not None and teacher.shape[1] < steps:
            raise ValueError("teacher sequence is shorter than requested rollout")
        if vertical_gate_teacher is not None and vertical_gate_teacher.shape[1] < steps:
            raise ValueError("vertical gate teacher is shorter than requested rollout")
        if vertical_onset_teacher is not None and vertical_onset_teacher.shape[1] < steps:
            raise ValueError("vertical onset teacher is shorter than requested rollout")
        if root_feedback_noise_std < 0.0:
            raise ValueError("root_feedback_noise_std cannot be negative")
        if not 0.0 <= root_feedback_noise_rho < 1.0:
            raise ValueError("root_feedback_noise_rho must be in [0, 1)")

        step_output, hidden = self.warm_history(history)
        previous_state = history[:, -1]
        controller = self.initial_vertical_controller(previous_state)
        if vertical_gate_teacher is not None and vertical_onset_teacher is not None:
            # A target window may begin halfway through an event detected in its
            # history. Mark that gate as already high so step zero cannot create
            # a false new event; it must fall and rise again before rearming.
            continuing = (vertical_gate_teacher[:, 0] >= self.event_gate_threshold) & (
                vertical_onset_teacher[:, 0] < self.event_gate_threshold
            )
            controller = controller._replace(previous_gate=continuing)
        states: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []
        velocities: list[torch.Tensor] = []
        amplitudes: list[torch.Tensor] = []
        pitch_velocities: list[torch.Tensor] = []
        pitch_amplitudes: list[torch.Tensor] = []
        durations: list[torch.Tensor] = []
        root_noise = torch.zeros_like(previous_state[:, :3])
        for step_index in range(steps):
            gate_override = (
                None
                if vertical_gate_teacher is None
                else vertical_gate_teacher[:, step_index]
            )
            predicted_state, controller = self.compose_state(
                previous_state, step_output, controller, gate_override
            )
            states.append(predicted_state)
            gates.append(step_output.vertical_event_logit)
            velocities.append(step_output.vertical_velocity_normalized)
            amplitudes.append(step_output.vertical_amplitude_normalized)
            pitch_velocities.append(step_output.pitch_velocity_normalized)
            pitch_amplitudes.append(step_output.pitch_amplitude_normalized)
            durations.append(step_output.vertical_log_duration_normalized)
            if step_index + 1 == steps:
                break

            feedback = predicted_state
            if teacher is not None and teacher_forcing_ratio > 0.0:
                if teacher_forcing_ratio >= 1.0:
                    feedback = teacher[:, step_index]
                else:
                    choose_teacher = (
                        torch.rand(
                            (len(history), 1),
                            device=history.device,
                            dtype=history.dtype,
                        )
                        < teacher_forcing_ratio
                    )
                    feedback = torch.where(
                        choose_teacher, teacher[:, step_index], predicted_state
                    )
            if root_feedback_noise_std > 0.0:
                innovation_gain = (1.0 - root_feedback_noise_rho**2) ** 0.5
                root_noise = (
                    root_feedback_noise_rho * root_noise
                    + innovation_gain
                    * torch.randn_like(root_noise)
                    * root_feedback_noise_std
                )
                root = feedback[:, :3] + root_noise
                feedback = torch.cat((root, feedback[:, 3:]), dim=-1)
            previous_state = feedback
            step_output, hidden = self.advance(feedback, hidden)

        return V3Rollout(
            state_normalized=torch.stack(states, dim=1),
            vertical_event_logit=torch.stack(gates, dim=1),
            vertical_velocity_normalized=torch.stack(velocities, dim=1),
            vertical_amplitude_normalized=torch.stack(amplitudes, dim=1),
            pitch_velocity_normalized=torch.stack(pitch_velocities, dim=1),
            pitch_amplitude_normalized=torch.stack(pitch_amplitudes, dim=1),
            vertical_log_duration_normalized=torch.stack(durations, dim=1),
        )


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))
