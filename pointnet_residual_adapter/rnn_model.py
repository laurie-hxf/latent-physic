from __future__ import annotations

import torch
from torch import nn

from .features import normalize_residual_output_mode, residual_output_dim
from .model import ACTION_DIM


class RNNResidualPredictor(nn.Module):
    """Two-layer RNN residual predictor using pooled contact-point features."""

    def __init__(
        self,
        *,
        point_feature_dim: int,
        history_window_steps: int,
        prediction_window_steps: int,
        hidden_size_1: int = 32,
        hidden_size_2: int = 16,
        point_pooling: str = "mean-max",
        linear_output_scale: float = 0.1,
        angular_output_scale: float = 1.0,
        position_output_scale: float = 0.01,
        yaw_output_scale: float = 0.1,
        residual_output_mode: str = "velocity",
    ) -> None:
        super().__init__()
        if int(point_feature_dim) <= 0:
            raise ValueError("point_feature_dim must be positive")
        if int(history_window_steps) <= 0:
            raise ValueError("history_window_steps must be positive")
        if int(prediction_window_steps) <= 0:
            raise ValueError("prediction_window_steps must be positive")
        if int(hidden_size_1) <= 0 or int(hidden_size_2) <= 0:
            raise ValueError("RNN hidden sizes must be positive")
        if point_pooling not in {"mean", "max", "mean-max"}:
            raise ValueError(f"Unsupported point pooling mode: {point_pooling!r}")

        self.point_feature_dim = int(point_feature_dim)
        self.history_window_steps = int(history_window_steps)
        self.prediction_window_steps = int(prediction_window_steps)
        self.hidden_size_1 = int(hidden_size_1)
        self.hidden_size_2 = int(hidden_size_2)
        self.point_pooling = str(point_pooling)
        self.residual_output_mode = normalize_residual_output_mode(residual_output_mode)
        self.residual_dim = residual_output_dim(self.residual_output_mode)

        pooled_dim = self.point_feature_dim * (2 if self.point_pooling == "mean-max" else 1)
        self.rnn1 = nn.RNN(
            input_size=pooled_dim,
            hidden_size=self.hidden_size_1,
            num_layers=1,
            nonlinearity="tanh",
            batch_first=True,
        )
        self.rnn2 = nn.RNN(
            input_size=self.hidden_size_1,
            hidden_size=self.hidden_size_2,
            num_layers=1,
            nonlinearity="tanh",
            batch_first=True,
        )
        action_input_dim = self.prediction_window_steps * ACTION_DIM
        self.output_head = nn.Linear(
            self.hidden_size_2 + action_input_dim,
            self.prediction_window_steps * self.residual_dim,
        )

        velocity_scales = [
            float(linear_output_scale),
            float(linear_output_scale),
            float(angular_output_scale),
        ]
        pose_scales = [
            float(position_output_scale),
            float(position_output_scale),
            float(yaw_output_scale),
        ]
        if self.residual_output_mode == "pose":
            per_step_scales = pose_scales
        elif self.residual_output_mode == "pose_velocity":
            per_step_scales = pose_scales + velocity_scales
        else:
            per_step_scales = velocity_scales
        scales = torch.tensor(per_step_scales, dtype=torch.float32).repeat(self.prediction_window_steps)
        self.register_buffer("output_scales", scales)

    def _pool_points(self, point_features: torch.Tensor, point_mask: torch.Tensor | None) -> torch.Tensor:
        batch_size, _, point_count, _ = point_features.shape
        if point_mask is None:
            mean_pool = point_features.mean(dim=2)
            max_pool = point_features.max(dim=2).values
        else:
            mask_bool = point_mask.to(device=point_features.device)
            if mask_bool.dtype != torch.bool:
                mask_bool = mask_bool > 0
            if mask_bool.shape != (batch_size, point_count):
                raise ValueError(f"point_mask must have shape ({batch_size}, {point_count}), got {tuple(mask_bool.shape)}")
            mask = mask_bool[:, None, :, None]
            mask_float = mask.to(point_features.dtype)
            denom = mask_float.sum(dim=2).clamp_min(1.0)
            mean_pool = (point_features * mask_float).sum(dim=2) / denom
            masked_for_max = point_features.masked_fill(~mask, -torch.finfo(point_features.dtype).max)
            max_pool = masked_for_max.max(dim=2).values
            max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))

        if self.point_pooling == "mean":
            return mean_pool
        if self.point_pooling == "max":
            return max_pool
        return torch.cat((mean_pool, max_pool), dim=-1)

    def forward(
        self,
        point_features: torch.Tensor,
        point_mask: torch.Tensor | None,
        future_actions: torch.Tensor,
    ) -> torch.Tensor:
        if point_features.ndim != 4:
            raise ValueError(f"point_features must have shape (B, H, N, F), got {tuple(point_features.shape)}")
        batch_size, history_steps, _, feature_dim = point_features.shape
        if history_steps != self.history_window_steps:
            raise ValueError(f"Expected H={self.history_window_steps}, got {history_steps}")
        if feature_dim != self.point_feature_dim:
            raise ValueError(f"Expected feature_dim={self.point_feature_dim}, got {feature_dim}")
        expected_action_shape = (batch_size, self.prediction_window_steps, ACTION_DIM)
        if future_actions.shape != expected_action_shape:
            raise ValueError(f"future_actions must have shape {expected_action_shape}, got {tuple(future_actions.shape)}")

        history_sequence = self._pool_points(point_features, point_mask)
        sequence_1, _ = self.rnn1(history_sequence)
        sequence_2, hidden_2 = self.rnn2(sequence_1)
        history_context = hidden_2[-1] if hidden_2.numel() else sequence_2[:, -1]
        action_context = future_actions.reshape(batch_size, -1)
        raw = self.output_head(torch.cat((history_context, action_context), dim=1))
        bounded = torch.tanh(raw) * self.output_scales.to(dtype=raw.dtype, device=raw.device)
        return bounded.reshape(batch_size, self.prediction_window_steps, self.residual_dim)
