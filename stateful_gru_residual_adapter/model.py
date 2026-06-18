from __future__ import annotations

import torch
from torch import nn

from pointnet_residual_adapter.features import normalize_residual_output_mode, residual_output_dim
from pointnet_residual_adapter.model import ACTION_DIM


class StatefulGRUResidualPredictor(nn.Module):
    """Deterministic residual predictor whose hidden state persists across steps."""

    is_stateful_residual_adapter = True

    def __init__(
        self,
        *,
        point_feature_dim: int,
        prediction_window_steps: int = 1,
        hidden_size: int = 16,
        num_layers: int = 2,
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
        if int(prediction_window_steps) != 1:
            raise ValueError("StatefulGRUResidualPredictor currently requires prediction_window_steps=1")
        if int(hidden_size) <= 0:
            raise ValueError("hidden_size must be positive")
        if int(num_layers) <= 0:
            raise ValueError("num_layers must be positive")
        if point_pooling not in {"mean", "max", "mean-max"}:
            raise ValueError(f"Unsupported point pooling mode: {point_pooling!r}")

        output_mode = normalize_residual_output_mode(residual_output_mode)
        if output_mode not in {"velocity", "pose", "pose_velocity"}:
            raise ValueError(
                "StatefulGRUResidualPredictor supports deterministic velocity, pose, or pose_velocity residuals"
            )

        self.point_feature_dim = int(point_feature_dim)
        self.history_window_steps = 1
        self.prediction_window_steps = int(prediction_window_steps)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.point_pooling = str(point_pooling)
        self.residual_output_mode = output_mode
        self.residual_dim = residual_output_dim(output_mode)

        pooled_dim = self.point_feature_dim * (2 if self.point_pooling == "mean-max" else 1)
        self.gru = nn.GRU(
            input_size=pooled_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.output_head = nn.Linear(self.hidden_size + ACTION_DIM, self.residual_dim)
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
            output_scales = pose_scales
        elif self.residual_output_mode == "pose_velocity":
            output_scales = pose_scales + velocity_scales
        else:
            output_scales = velocity_scales
        self.register_buffer(
            "output_scales",
            torch.tensor(output_scales, dtype=torch.float32),
        )

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

    def _pool_points(self, point_features: torch.Tensor, point_mask: torch.Tensor | None) -> torch.Tensor:
        if point_features.ndim != 3:
            raise ValueError(f"point_features must have shape (B, N, F), got {tuple(point_features.shape)}")
        batch_size, point_count, feature_dim = point_features.shape
        if feature_dim != self.point_feature_dim:
            raise ValueError(f"Expected feature_dim={self.point_feature_dim}, got {feature_dim}")

        if point_mask is None:
            mean_pool = point_features.mean(dim=1)
            max_pool = point_features.max(dim=1).values
        else:
            mask_bool = point_mask.to(device=point_features.device)
            if mask_bool.dtype != torch.bool:
                mask_bool = mask_bool > 0
            if mask_bool.shape != (batch_size, point_count):
                raise ValueError(f"point_mask must have shape ({batch_size}, {point_count}), got {tuple(mask_bool.shape)}")
            mask = mask_bool[:, :, None]
            mask_float = mask.to(point_features.dtype)
            mean_pool = (point_features * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)
            masked_for_max = point_features.masked_fill(~mask, -torch.finfo(point_features.dtype).max)
            max_pool = masked_for_max.max(dim=1).values
            max_pool = torch.where(mask_bool.any(dim=1, keepdim=True), max_pool, torch.zeros_like(max_pool))

        if self.point_pooling == "mean":
            return mean_pool
        if self.point_pooling == "max":
            return max_pool
        return torch.cat((mean_pool, max_pool), dim=-1)

    def forward_step(
        self,
        point_features: torch.Tensor,
        point_mask: torch.Tensor | None,
        future_actions: torch.Tensor,
        hidden_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict one residual and return the next persistent GRU state."""
        batch_size = int(point_features.shape[0])
        expected_action_shape = (batch_size, 1, ACTION_DIM)
        if future_actions.shape != expected_action_shape:
            raise ValueError(f"future_actions must have shape {expected_action_shape}, got {tuple(future_actions.shape)}")
        if hidden_state is None:
            hidden_state = self.initial_state(
                batch_size,
                device=point_features.device,
                dtype=point_features.dtype,
            )

        pooled = self._pool_points(point_features, point_mask)
        sequence, next_hidden = self.gru(pooled[:, None, :], hidden_state)
        context = torch.cat((sequence[:, 0], future_actions[:, 0]), dim=-1)
        raw = self.output_head(context)
        bounded = torch.tanh(raw) * self.output_scales.to(dtype=raw.dtype, device=raw.device)
        return bounded[:, None, :], next_hidden

    def forward(
        self,
        point_features: torch.Tensor,
        point_mask: torch.Tensor | None,
        future_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compatibility path that processes a complete history from a zero state."""
        if point_features.ndim != 4:
            raise ValueError(f"point_features must have shape (B, H, N, F), got {tuple(point_features.shape)}")
        batch_size, history_steps, _, _ = point_features.shape
        if future_actions.shape != (batch_size, 1, ACTION_DIM):
            raise ValueError(f"future_actions must have shape ({batch_size}, 1, {ACTION_DIM})")

        hidden = self.initial_state(batch_size, device=point_features.device, dtype=point_features.dtype)
        output: torch.Tensor | None = None
        for history_idx in range(history_steps):
            output, hidden = self.forward_step(
                point_features[:, history_idx],
                point_mask,
                future_actions,
                hidden,
            )
        if output is None:
            raise ValueError("point_features history must contain at least one step")
        return output
