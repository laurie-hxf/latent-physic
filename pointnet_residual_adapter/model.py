from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .features import normalize_residual_output_mode, residual_output_dim


ACTION_DIM = 7


@dataclass(frozen=True)
class ResidualLossWeights:
    linear_velocity: float = 1.0
    angular_velocity_z: float = 0.1
    position_xy: float = 1.0
    yaw: float = 1.0
    horizon_gamma: float = 0.95
    residual_l2: float = 1.0e-4
    residual_smoothness: float = 1.0e-4


class PointNetResidualPredictor(nn.Module):
    """PointNet encoder that predicts planar rigid velocity residuals."""

    def __init__(
        self,
        *,
        point_feature_dim: int,
        history_window_steps: int,
        prediction_window_steps: int,
        pointnet_feature_dim: int = 256,
        action_context_dim: int = 64,
        pooling: str = "mean-max",
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
        if pooling not in {"max", "mean-max"}:
            raise ValueError(f"Unsupported pooling mode: {pooling!r}")

        self.point_feature_dim = int(point_feature_dim)
        self.history_window_steps = int(history_window_steps)
        self.prediction_window_steps = int(prediction_window_steps)
        self.pointnet_feature_dim = int(pointnet_feature_dim)
        self.action_context_dim = int(action_context_dim)
        self.pooling = pooling
        self.residual_output_mode = normalize_residual_output_mode(residual_output_mode)
        self.residual_dim = residual_output_dim(self.residual_output_mode)

        channel_dim = self.history_window_steps * self.point_feature_dim
        self.point_mlp = nn.Sequential(
            nn.Conv1d(channel_dim, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, self.pointnet_feature_dim, kernel_size=1),
            nn.BatchNorm1d(self.pointnet_feature_dim),
            nn.ReLU(inplace=True),
        )

        action_input_dim = self.prediction_window_steps * ACTION_DIM
        if self.action_context_dim > 0:
            self.action_mlp = nn.Sequential(
                nn.Linear(action_input_dim, self.action_context_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.action_context_dim, self.action_context_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.action_mlp = None

        pooled_dim = self.pointnet_feature_dim * (2 if pooling == "mean-max" else 1)
        context_dim = pooled_dim + (self.action_context_dim if self.action_mlp is not None else action_input_dim)
        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )
        self.output_head = nn.Linear(128, self.prediction_window_steps * self.residual_dim)

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

    def forward(
        self,
        point_features: torch.Tensor,
        point_mask: torch.Tensor | None,
        future_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Return residuals with shape ``(batch, P, 3)``."""
        if point_features.ndim != 4:
            raise ValueError(f"point_features must have shape (B, H, N, F), got {tuple(point_features.shape)}")
        batch_size, history_steps, point_count, feature_dim = point_features.shape
        if history_steps != self.history_window_steps:
            raise ValueError(f"Expected H={self.history_window_steps}, got {history_steps}")
        if feature_dim != self.point_feature_dim:
            raise ValueError(f"Expected feature_dim={self.point_feature_dim}, got {feature_dim}")
        if future_actions.shape != (batch_size, self.prediction_window_steps, ACTION_DIM):
            raise ValueError(
                "future_actions must have shape "
                f"({batch_size}, {self.prediction_window_steps}, {ACTION_DIM}), got {tuple(future_actions.shape)}"
            )

        x = point_features.permute(0, 2, 1, 3).reshape(batch_size, point_count, -1)
        x = x.transpose(1, 2).contiguous()
        point_embedding = self.point_mlp(x)

        if point_mask is None:
            max_pool = point_embedding.max(dim=2).values
            if self.pooling == "mean-max":
                mean_pool = point_embedding.mean(dim=2)
                pooled = torch.cat([mean_pool, max_pool], dim=1)
            else:
                pooled = max_pool
        else:
            mask_bool = point_mask.to(device=point_features.device)
            if mask_bool.dtype != torch.bool:
                mask_bool = mask_bool > 0
            if mask_bool.shape != (batch_size, point_count):
                raise ValueError(f"point_mask must have shape ({batch_size}, {point_count}), got {tuple(mask_bool.shape)}")

            mask = mask_bool.unsqueeze(1)
            masked_for_max = point_embedding.masked_fill(~mask, -torch.finfo(point_embedding.dtype).max)
            max_pool = masked_for_max.max(dim=2).values
            max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))

            if self.pooling == "mean-max":
                mask_float = mask.to(point_embedding.dtype)
                denom = mask_float.sum(dim=2).clamp_min(1.0)
                mean_pool = (point_embedding * mask_float).sum(dim=2) / denom
                pooled = torch.cat([mean_pool, max_pool], dim=1)
            else:
                pooled = max_pool

        action_flat = future_actions.reshape(batch_size, -1)
        if self.action_mlp is None:
            action_context = action_flat
        else:
            action_context = self.action_mlp(action_flat)

        context = torch.cat([pooled, action_context], dim=1)
        raw = self.output_head(self.context_mlp(context))
        bounded = torch.tanh(raw) * self.output_scales.to(dtype=raw.dtype, device=raw.device)
        return bounded.reshape(batch_size, self.prediction_window_steps, self.residual_dim)


def residual_velocity_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: ResidualLossWeights,
    residual_output_mode: str = "velocity",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/target shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}")
    if prediction.ndim != 3 or prediction.shape[-1] not in {3, 6}:
        raise ValueError(f"Residual tensors must have shape (B, P, 3 or 6), got {tuple(prediction.shape)}")
    output_mode = normalize_residual_output_mode(residual_output_mode)

    horizon = prediction.shape[1]
    gamma = torch.as_tensor(
        [float(weights.horizon_gamma) ** idx for idx in range(horizon)],
        dtype=prediction.dtype,
        device=prediction.device,
    ).reshape(1, horizon)

    diff = prediction - target
    if output_mode == "pose_velocity":
        pose_diff = diff[..., :3]
        velocity_diff = diff[..., 3:]
    elif output_mode == "pose":
        pose_diff = diff
        velocity_diff = None
    else:
        pose_diff = None
        velocity_diff = diff

    zero_by_horizon = torch.zeros((horizon,), dtype=prediction.dtype, device=prediction.device)
    if velocity_diff is not None:
        linear_mse_by_horizon = (velocity_diff[..., :2] * velocity_diff[..., :2]).mean(dim=(0, 2))
        angular_mse_by_horizon = (velocity_diff[..., 2] * velocity_diff[..., 2]).mean(dim=0)
    else:
        linear_mse_by_horizon = zero_by_horizon
        angular_mse_by_horizon = zero_by_horizon
    if pose_diff is not None:
        position_mse_by_horizon = (pose_diff[..., :2] * pose_diff[..., :2]).mean(dim=(0, 2))
        yaw_mse_by_horizon = (pose_diff[..., 2] * pose_diff[..., 2]).mean(dim=0)
    else:
        position_mse_by_horizon = zero_by_horizon
        yaw_mse_by_horizon = zero_by_horizon

    weighted = (
        float(weights.linear_velocity) * linear_mse_by_horizon
        + float(weights.angular_velocity_z) * angular_mse_by_horizon
        + float(weights.position_xy) * position_mse_by_horizon
        + float(weights.yaw) * yaw_mse_by_horizon
    )
    velocity_loss = (gamma.reshape(-1) * weighted).sum() / gamma.sum().clamp_min(1.0e-8)

    magnitude_loss = prediction.square().mean()
    if horizon > 1:
        smoothness_loss = (prediction[:, 1:] - prediction[:, :-1]).square().mean()
    else:
        smoothness_loss = torch.zeros((), dtype=prediction.dtype, device=prediction.device)

    total = (
        velocity_loss
        + float(weights.residual_l2) * magnitude_loss
        + float(weights.residual_smoothness) * smoothness_loss
    )
    metrics = {
        "loss_velocity": velocity_loss.detach(),
        "loss_linear_xy": linear_mse_by_horizon.mean().detach(),
        "loss_angular_z": angular_mse_by_horizon.mean().detach(),
        "loss_position_xy": position_mse_by_horizon.mean().detach(),
        "loss_yaw": yaw_mse_by_horizon.mean().detach(),
        "loss_residual_l2": magnitude_loss.detach(),
        "loss_residual_smoothness": smoothness_loss.detach(),
        "pred_linear_abs_mean": (velocity_diff[..., :2].abs().mean() if velocity_diff is not None else torch.zeros((), dtype=prediction.dtype, device=prediction.device)).detach(),
        "pred_angular_abs_mean": (velocity_diff[..., 2].abs().mean() if velocity_diff is not None else torch.zeros((), dtype=prediction.dtype, device=prediction.device)).detach(),
        "pred_position_abs_mean": (pose_diff[..., :2].abs().mean() if pose_diff is not None else torch.zeros((), dtype=prediction.dtype, device=prediction.device)).detach(),
        "pred_yaw_abs_mean": (pose_diff[..., 2].abs().mean() if pose_diff is not None else torch.zeros((), dtype=prediction.dtype, device=prediction.device)).detach(),
        "target_linear_abs_mean": (
            target[..., -3:-1].abs().mean()
            if output_mode == "pose_velocity"
            else target[..., :2].abs().mean()
            if output_mode in {"velocity", "acceleration"}
            else torch.zeros((), dtype=prediction.dtype, device=prediction.device)
        ).detach(),
        "target_angular_abs_mean": (
            target[..., -1].abs().mean()
            if output_mode == "pose_velocity"
            else target[..., 2].abs().mean()
            if output_mode in {"velocity", "acceleration"}
            else torch.zeros((), dtype=prediction.dtype, device=prediction.device)
        ).detach(),
        "target_position_abs_mean": (target[..., :2].abs().mean() if pose_diff is not None else torch.zeros((), dtype=prediction.dtype, device=prediction.device)).detach(),
        "target_yaw_abs_mean": (target[..., 2].abs().mean() if pose_diff is not None else torch.zeros((), dtype=prediction.dtype, device=prediction.device)).detach(),
    }
    return total, metrics
