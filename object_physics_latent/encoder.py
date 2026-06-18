from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence


DEFAULT_ENCODER_FEATURE_DIM = 12


def build_mlp(
    *,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    hidden_layers: int,
    activation: type[nn.Module] = nn.ReLU,
    final_activation: type[nn.Module] | None = None,
) -> nn.Sequential:
    if int(input_dim) <= 0 or int(output_dim) <= 0:
        raise ValueError("input_dim and output_dim must be positive")
    if int(hidden_layers) < 0:
        raise ValueError("hidden_layers must be non-negative")

    layers: list[nn.Module] = []
    in_dim = int(input_dim)
    for _ in range(int(hidden_layers)):
        layer = nn.Linear(in_dim, int(hidden_dim))
        nn.init.kaiming_uniform_(layer.weight, a=5.0 ** 0.5)
        nn.init.zeros_(layer.bias)
        layers.append(layer)
        layers.append(activation())
        in_dim = int(hidden_dim)

    final = nn.Linear(in_dim, int(output_dim))
    nn.init.xavier_uniform_(final.weight)
    nn.init.zeros_(final.bias)
    layers.append(final)
    if final_activation is not None:
        layers.append(final_activation())
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class ObjectLatentOutput:
    latent: torch.Tensor
    projection: torch.Tensor
    trajectory_embeddings: torch.Tensor
    pooled_embedding: torch.Tensor
    trajectory_valid_mask: torch.Tensor
    visual_embedding: torch.Tensor | None = None


def lengths_from_valid_mask(valid_mask: torch.Tensor) -> torch.Tensor:
    if valid_mask.dtype != torch.bool:
        valid_mask = valid_mask.to(dtype=torch.bool)
    if valid_mask.ndim != 2:
        raise ValueError(f"valid_mask must have shape (N, T), got {tuple(valid_mask.shape)}")
    lengths = valid_mask.sum(dim=1).to(dtype=torch.long)
    if torch.any(lengths <= 0):
        raise ValueError("all trajectories must contain at least one valid step")
    return lengths


def encoder_batch_to_torch(
    batch: Any,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    add_object_axis: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.as_tensor(batch.features, dtype=dtype, device=device)
    valid_mask = torch.as_tensor(batch.valid_mask, dtype=torch.bool, device=device)
    if add_object_axis:
        if features.ndim != 3 or valid_mask.ndim != 2:
            raise ValueError(
                "EncoderFeatureBatch conversion expects features=(K,T,D) and valid_mask=(K,T); "
                f"got {tuple(features.shape)} and {tuple(valid_mask.shape)}"
            )
        features = features.unsqueeze(0)
        valid_mask = valid_mask.unsqueeze(0)
    return features, valid_mask


class TrajectoryGRUEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = DEFAULT_ENCODER_FEATURE_DIM,
        step_hidden_dim: int = 128,
        step_mlp_layers: int = 1,
        gru_hidden_dim: int = 128,
        gru_layers: int = 1,
        trajectory_embedding_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if int(gru_layers) < 1:
            raise ValueError("gru_layers must be >= 1")
        self.input_dim = int(input_dim)
        self.step_hidden_dim = int(step_hidden_dim)
        self.gru_hidden_dim = int(gru_hidden_dim)
        self.trajectory_embedding_dim = int(trajectory_embedding_dim)
        self.step_mlp = build_mlp(
            input_dim=self.input_dim,
            output_dim=self.step_hidden_dim,
            hidden_dim=self.step_hidden_dim,
            hidden_layers=int(step_mlp_layers),
            final_activation=nn.ReLU,
        )
        self.gru = nn.GRU(
            input_size=self.step_hidden_dim,
            hidden_size=self.gru_hidden_dim,
            num_layers=int(gru_layers),
            batch_first=True,
            dropout=float(dropout) if int(gru_layers) > 1 else 0.0,
        )
        self.output_mlp = build_mlp(
            input_dim=self.gru_hidden_dim,
            output_dim=self.trajectory_embedding_dim,
            hidden_dim=self.trajectory_embedding_dim,
            hidden_layers=1,
        )

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"features must have shape (N, T, D), got {tuple(features.shape)}")
        if features.shape[-1] != self.input_dim:
            raise ValueError(f"feature dim mismatch: got {features.shape[-1]}, expected {self.input_dim}")
        if lengths is None:
            if valid_mask is None:
                lengths = torch.full(
                    (features.shape[0],),
                    int(features.shape[1]),
                    dtype=torch.long,
                    device=features.device,
                )
            else:
                lengths = lengths_from_valid_mask(valid_mask).to(device=features.device)
        else:
            lengths = torch.as_tensor(lengths, dtype=torch.long, device=features.device)
            if lengths.ndim != 1 or lengths.shape[0] != features.shape[0]:
                raise ValueError(f"lengths must have shape ({features.shape[0]},), got {tuple(lengths.shape)}")
            if torch.any(lengths <= 0):
                raise ValueError("all trajectories must contain at least one valid step")

        step_features = self.step_mlp(features)
        packed = pack_padded_sequence(
            step_features,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        last_hidden = hidden[-1]
        return self.output_mlp(last_hidden)


class TrajectorySetEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = DEFAULT_ENCODER_FEATURE_DIM,
        latent_dim: int = 8,
        step_hidden_dim: int = 128,
        step_mlp_layers: int = 1,
        gru_hidden_dim: int = 128,
        gru_layers: int = 1,
        trajectory_embedding_dim: int = 128,
        set_hidden_dim: int = 128,
        trajectory_mlp_layers: int = 1,
        object_mlp_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.trajectory_encoder = TrajectoryGRUEncoder(
            input_dim=int(input_dim),
            step_hidden_dim=int(step_hidden_dim),
            step_mlp_layers=int(step_mlp_layers),
            gru_hidden_dim=int(gru_hidden_dim),
            gru_layers=int(gru_layers),
            trajectory_embedding_dim=int(trajectory_embedding_dim),
            dropout=float(dropout),
        )
        self.trajectory_mlp = build_mlp(
            input_dim=int(trajectory_embedding_dim),
            output_dim=int(set_hidden_dim),
            hidden_dim=int(set_hidden_dim),
            hidden_layers=int(trajectory_mlp_layers),
            final_activation=nn.ReLU,
        )
        self.object_mlp = build_mlp(
            input_dim=int(set_hidden_dim),
            output_dim=self.latent_dim,
            hidden_dim=int(set_hidden_dim),
            hidden_layers=int(object_mlp_layers),
        )

    @staticmethod
    def _as_batched_context(
        features: torch.Tensor,
        valid_mask: torch.Tensor | None,
        trajectory_valid_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.ndim == 3:
            features = features.unsqueeze(0)
            if valid_mask is not None:
                valid_mask = valid_mask.unsqueeze(0)
            if trajectory_valid_mask is not None:
                trajectory_valid_mask = trajectory_valid_mask.unsqueeze(0)
        if features.ndim != 4:
            raise ValueError(f"context features must have shape (B,K,T,D) or (K,T,D), got {tuple(features.shape)}")
        if valid_mask is None:
            valid_mask = torch.ones(features.shape[:3], dtype=torch.bool, device=features.device)
        elif valid_mask.ndim != 3:
            raise ValueError(f"valid_mask must have shape (B,K,T), got {tuple(valid_mask.shape)}")
        else:
            valid_mask = valid_mask.to(device=features.device, dtype=torch.bool)
        if trajectory_valid_mask is None:
            trajectory_valid_mask = valid_mask.any(dim=2)
        elif trajectory_valid_mask.ndim != 2:
            raise ValueError(
                f"trajectory_valid_mask must have shape (B,K), got {tuple(trajectory_valid_mask.shape)}"
            )
        else:
            trajectory_valid_mask = trajectory_valid_mask.to(device=features.device, dtype=torch.bool)
        if torch.any(trajectory_valid_mask.sum(dim=1) <= 0):
            raise ValueError("each object must contain at least one valid context trajectory")
        return features, valid_mask, trajectory_valid_mask

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        trajectory_valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features, valid_mask, trajectory_valid_mask = self._as_batched_context(
            features,
            valid_mask,
            trajectory_valid_mask,
        )
        batch_size, trajectories_per_object, steps, feature_dim = features.shape
        flat_features = features.reshape(batch_size * trajectories_per_object, steps, feature_dim)
        flat_mask = valid_mask.reshape(batch_size * trajectories_per_object, steps)
        flat_traj_valid = trajectory_valid_mask.reshape(-1)

        encoded = torch.zeros(
            (batch_size * trajectories_per_object, self.trajectory_encoder.trajectory_embedding_dim),
            dtype=features.dtype,
            device=features.device,
        )
        if torch.any(flat_traj_valid):
            encoded_valid = self.trajectory_encoder(
                flat_features[flat_traj_valid],
                valid_mask=flat_mask[flat_traj_valid],
            )
            encoded[flat_traj_valid] = encoded_valid

        set_features = self.trajectory_mlp(encoded).reshape(batch_size, trajectories_per_object, -1)
        weights = trajectory_valid_mask.to(dtype=set_features.dtype).unsqueeze(-1)
        pooled = (set_features * weights).sum(dim=1) / torch.clamp(weights.sum(dim=1), min=1.0)
        latent = self.object_mlp(pooled)
        return latent, encoded.reshape(batch_size, trajectories_per_object, -1), pooled, trajectory_valid_mask


class VisualPointSetEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 128,
        embedding_dim: int = 128,
        point_hidden_layers: int = 1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.embedding_dim = int(embedding_dim)
        self.point_mlp = build_mlp(
            input_dim=self.input_dim,
            output_dim=self.embedding_dim,
            hidden_dim=int(hidden_dim),
            hidden_layers=int(point_hidden_layers),
            final_activation=nn.ReLU,
        )
        self.output_mlp = build_mlp(
            input_dim=2 * self.embedding_dim,
            output_dim=self.embedding_dim,
            hidden_dim=int(hidden_dim),
            hidden_layers=1,
            final_activation=nn.ReLU,
        )

    def forward(self, point_features: torch.Tensor) -> torch.Tensor:
        if point_features.ndim == 2:
            point_features = point_features.unsqueeze(0)
        if point_features.ndim != 3:
            raise ValueError(f"visual point features must have shape (B,P,F) or (P,F), got {tuple(point_features.shape)}")
        if point_features.shape[-1] != self.input_dim:
            raise ValueError(
                f"visual point feature dim mismatch: got {point_features.shape[-1]}, expected {self.input_dim}"
            )
        point_embedding = self.point_mlp(point_features)
        mean_embedding = point_embedding.mean(dim=1)
        max_embedding = point_embedding.max(dim=1).values
        return self.output_mlp(torch.cat((mean_embedding, max_embedding), dim=-1))


class ObjectPhysicsEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = DEFAULT_ENCODER_FEATURE_DIM,
        latent_dim: int = 8,
        projection_dim: int = 32,
        step_hidden_dim: int = 128,
        gru_hidden_dim: int = 128,
        gru_layers: int = 1,
        trajectory_embedding_dim: int = 128,
        set_hidden_dim: int = 128,
        visual_input_dim: int = 0,
        visual_hidden_dim: int = 128,
        visual_embedding_dim: int = 128,
        visual_point_hidden_layers: int = 1,
        projection_hidden_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.visual_input_dim = int(visual_input_dim)
        self.set_encoder = TrajectorySetEncoder(
            input_dim=int(input_dim),
            latent_dim=int(latent_dim),
            step_hidden_dim=int(step_hidden_dim),
            gru_hidden_dim=int(gru_hidden_dim),
            gru_layers=int(gru_layers),
            trajectory_embedding_dim=int(trajectory_embedding_dim),
            set_hidden_dim=int(set_hidden_dim),
            dropout=float(dropout),
        )
        if self.visual_input_dim > 0:
            self.visual_encoder = VisualPointSetEncoder(
                input_dim=self.visual_input_dim,
                hidden_dim=int(visual_hidden_dim),
                embedding_dim=int(visual_embedding_dim),
                point_hidden_layers=int(visual_point_hidden_layers),
            )
            self.fusion_mlp = build_mlp(
                input_dim=int(latent_dim) + int(visual_embedding_dim),
                output_dim=int(latent_dim),
                hidden_dim=int(set_hidden_dim),
                hidden_layers=1,
            )
        else:
            self.visual_encoder = None
            self.fusion_mlp = None
        # ``projection_dim`` and ``projection_hidden_dim`` are retained in the
        # constructor for checkpoint/config compatibility. Contrastive learning
        # now operates directly on the unit-norm object latent.
        _ = projection_dim, projection_hidden_dim

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        trajectory_valid_mask: torch.Tensor | None = None,
        visual_features: torch.Tensor | None = None,
    ) -> ObjectLatentOutput:
        trajectory_latent, trajectory_embeddings, pooled, trajectory_mask = self.set_encoder(
            features,
            valid_mask=valid_mask,
            trajectory_valid_mask=trajectory_valid_mask,
        )
        visual_embedding = None
        if self.visual_encoder is not None:
            if visual_features is None:
                raise ValueError("visual_features are required when visual_input_dim > 0")
            visual_embedding = self.visual_encoder(visual_features)
            latent = self.fusion_mlp(torch.cat((trajectory_latent, visual_embedding), dim=-1))
        else:
            latent = trajectory_latent
        latent = F.normalize(latent, dim=-1, eps=1.0e-8)
        return ObjectLatentOutput(
            latent=latent,
            projection=latent,
            trajectory_embeddings=trajectory_embeddings,
            pooled_embedding=pooled,
            trajectory_valid_mask=trajectory_mask,
            visual_embedding=visual_embedding,
        )


def same_object_consistency_loss(latent_a: torch.Tensor, latent_b: torch.Tensor) -> torch.Tensor:
    if latent_a.shape != latent_b.shape:
        raise ValueError(f"latent shapes must match, got {tuple(latent_a.shape)} and {tuple(latent_b.shape)}")
    return torch.mean((latent_a - latent_b) ** 2)


def symmetric_info_nce_loss(
    projection_a: torch.Tensor,
    projection_b: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    if projection_a.shape != projection_b.shape:
        raise ValueError(
            f"projection shapes must match, got {tuple(projection_a.shape)} and {tuple(projection_b.shape)}"
        )
    if projection_a.ndim != 2:
        raise ValueError(f"projections must have shape (B,D), got {tuple(projection_a.shape)}")
    if projection_a.shape[0] == 0:
        raise ValueError("contrastive batch must be non-empty")
    z_a = F.normalize(projection_a, dim=-1)
    z_b = F.normalize(projection_b, dim=-1)
    logits = z_a @ z_b.T / max(float(temperature), 1.0e-8)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


@dataclass(frozen=True)
class LatentRegularizationLosses:
    consistency: torch.Tensor
    contrastive: torch.Tensor
    latent_norm: torch.Tensor
    total: torch.Tensor


def latent_regularization_losses(
    output_a: ObjectLatentOutput,
    output_b: ObjectLatentOutput,
    *,
    consistency_weight: float = 0.1,
    contrastive_weight: float = 0.05,
    temperature: float = 0.1,
    latent_norm_weight: float = 0.0,
    latent_norm_target: float = 8.0,
) -> LatentRegularizationLosses:
    consistency = same_object_consistency_loss(output_a.latent, output_b.latent)
    contrastive = symmetric_info_nce_loss(output_a.latent, output_b.latent, temperature=float(temperature))
    latents = torch.cat((output_a.latent, output_b.latent), dim=0)
    latent_norm = torch.mean((torch.linalg.norm(latents, dim=-1) - 1.0) ** 2)
    # The latent norm is enforced by construction in ObjectPhysicsEncoder. Keep
    # the legacy weight/target arguments accepted, but do not add a soft norm
    # penalty to the objective.
    _ = latent_norm_weight, latent_norm_target
    total = float(consistency_weight) * consistency + float(contrastive_weight) * contrastive
    return LatentRegularizationLosses(
        consistency=consistency,
        contrastive=contrastive,
        latent_norm=latent_norm,
        total=total,
    )
