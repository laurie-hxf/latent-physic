from __future__ import annotations

import torch
from torch import nn


DIRECT_STATE_DIM = 6


class StatefulGRUDirectStatePredictor(nn.Module):
    """Stateful GRU that maps Newton next-state predictions to direct states."""

    is_stateful_direct_state_adapter = True

    def __init__(
        self,
        *,
        input_dim: int,
        state_dim: int = DIRECT_STATE_DIM,
        hidden_size: int = 16,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if int(input_dim) <= 0:
            raise ValueError("input_dim must be positive")
        if int(state_dim) <= 0:
            raise ValueError("state_dim must be positive")
        if int(hidden_size) <= 0:
            raise ValueError("hidden_size must be positive")
        if int(num_layers) <= 0:
            raise ValueError("num_layers must be positive")
        self.input_dim = int(input_dim)
        self.state_dim = int(state_dim)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.output_head = nn.Linear(self.hidden_size, self.state_dim)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        parameter = next(self.parameters())
        return torch.zeros(
            self.num_layers,
            int(batch_size),
            self.hidden_size,
            device=parameter.device if device is None else device,
            dtype=parameter.dtype if dtype is None else dtype,
        )

    def forward_step(
        self,
        inputs: torch.Tensor,
        hidden_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 2:
            raise ValueError(f"inputs must have shape (B, F), got {tuple(inputs.shape)}")
        if int(inputs.shape[-1]) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {int(inputs.shape[-1])}")
        batch_size = int(inputs.shape[0])
        if hidden_state is None:
            hidden_state = self.initial_state(batch_size, device=inputs.device, dtype=inputs.dtype)
        sequence, next_hidden = self.gru(inputs[:, None, :], hidden_state)
        return self.output_head(sequence[:, 0]), next_hidden

    def forward(
        self,
        inputs: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3:
            raise ValueError(f"inputs must have shape (B, T, F), got {tuple(inputs.shape)}")
        if int(inputs.shape[-1]) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {int(inputs.shape[-1])}")
        batch_size = int(inputs.shape[0])
        if hidden_state is None:
            hidden_state = self.initial_state(batch_size, device=inputs.device, dtype=inputs.dtype)
        sequence, next_hidden = self.gru(inputs, hidden_state)
        return self.output_head(sequence), next_hidden
