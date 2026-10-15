#!/usr/bin/env python3
"""PyTorch model contract for the independent Normal HMD Agent v2."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class NormalHMDLSTM(nn.Module):
    """Autoregressive state predictor with an explicit stateful rollout API.

    The first call consumes the complete normalized history.  Each subsequent
    decoder call consumes either the ground-truth previous state (teacher
    forcing) or its own previous prediction.  Hidden state is therefore kept
    throughout a rollout and is reset only at an episode boundary.
    """

    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 128,
        output_dim: int = 9,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim != output_dim:
            raise ValueError("autoregressive Normal HMD model requires input_dim == output_dim")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, output_dim)

    def warm_history(
        self, history: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Return the first next-state prediction and persistent (h, c)."""

        encoded, hidden = self.lstm(history)
        return self.output(encoded[:, -1]), hidden

    def advance(
        self,
        feedback_state: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Consume one normalized feedback state and predict the following state."""

        encoded, hidden = self.lstm(feedback_state[:, None, :], hidden)
        return self.output(encoded[:, 0]), hidden

    def rollout(
        self,
        history: torch.Tensor,
        steps: int,
        teacher: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        if steps < 1:
            raise ValueError("steps must be positive")
        if teacher is not None and teacher.shape[1] < steps:
            raise ValueError("teacher sequence is shorter than requested rollout")
        prediction, hidden = self.warm_history(history)
        predictions: list[torch.Tensor] = []
        for step in range(steps):
            predictions.append(prediction)
            if step + 1 == steps:
                break
            feedback = prediction
            if teacher is not None and teacher_forcing_ratio > 0.0:
                if teacher_forcing_ratio >= 1.0:
                    feedback = teacher[:, step]
                else:
                    choose_teacher = (
                        torch.rand(
                            (len(history), 1),
                            device=history.device,
                            dtype=history.dtype,
                        )
                        < teacher_forcing_ratio
                    )
                    feedback = torch.where(choose_teacher, teacher[:, step], prediction)
            prediction, hidden = self.advance(feedback, hidden)
        return torch.stack(predictions, dim=1)


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))
