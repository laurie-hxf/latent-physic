from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from object_physics_latent.encoder import (
    DEFAULT_ENCODER_FEATURE_DIM,
    ObjectLatentOutput,
    ObjectPhysicsEncoder,
)
from object_physics_latent.friction_decoder import LatentConditionedFrictionDecoder


@dataclass(frozen=True)
class TrajectoryConditionedFrictionOutput:
    latent_output: ObjectLatentOutput
    friction: torch.Tensor

    @property
    def latent(self) -> torch.Tensor:
        return self.latent_output.latent

    @property
    def projection(self) -> torch.Tensor:
        return self.latent_output.projection


class _LatentCompatibleModule(nn.Module):
    """Mixin that tolerates legacy projection-head checkpoints."""

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        if strict:
            filtered = {
                key: value
                for key, value in state_dict.items()
                if not key.startswith("encoder.projection_head.")
            }
            return super().load_state_dict(filtered, strict=True)
        return super().load_state_dict(state_dict, strict=False)


class TrajectoryConditionedFrictionModel(_LatentCompatibleModule):
    def __init__(
        self,
        *,
        encoder: ObjectPhysicsEncoder,
        friction_decoder: LatentConditionedFrictionDecoder,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.friction_decoder = friction_decoder

    @classmethod
    def from_dimensions(
        cls,
        *,
        point_feature_dim: int,
        encoder_feature_dim: int = DEFAULT_ENCODER_FEATURE_DIM,
        latent_dim: int = 8,
        projection_dim: int = 32,
        step_hidden_dim: int = 128,
        gru_hidden_dim: int = 128,
        trajectory_embedding_dim: int = 128,
        set_hidden_dim: int = 128,
        visual_feature_dim: int = 0,
        visual_hidden_dim: int = 128,
        visual_embedding_dim: int = 128,
        visual_point_hidden_layers: int = 1,
        decoder_hidden_dim: int = 128,
        decoder_hidden_layers: int = 2,
        decoder_conditioning: str = "film",
        decoder_activation: str = "silu",
        decoder_basis_count: int = 8,
        decoder_basis_base_mode: str = "latent",
        decoder_basis_normalization: str = "zero_mean",
        decoder_basis_activation: str = "tanh",
        decoder_basis_norm_eps: float = 1.0e-4,
        decoder_latent_normalization: str = "none",
        decoder_raw_limit: float | None = None,
        mu_min: float = 0.0,
        mu_max: float = 2.0,
        initial_mu: float = 0.35,
    ) -> "TrajectoryConditionedFrictionModel":
        encoder = ObjectPhysicsEncoder(
            input_dim=int(encoder_feature_dim),
            latent_dim=int(latent_dim),
            projection_dim=int(projection_dim),
            step_hidden_dim=int(step_hidden_dim),
            gru_hidden_dim=int(gru_hidden_dim),
            trajectory_embedding_dim=int(trajectory_embedding_dim),
            set_hidden_dim=int(set_hidden_dim),
            visual_input_dim=int(visual_feature_dim),
            visual_hidden_dim=int(visual_hidden_dim),
            visual_embedding_dim=int(visual_embedding_dim),
            visual_point_hidden_layers=int(visual_point_hidden_layers),
        )
        decoder = LatentConditionedFrictionDecoder(
            point_feature_dim=int(point_feature_dim),
            latent_dim=int(latent_dim),
            hidden_dim=int(decoder_hidden_dim),
            hidden_layers=int(decoder_hidden_layers),
            conditioning=str(decoder_conditioning),
            activation=str(decoder_activation),
            basis_count=int(decoder_basis_count),
            basis_base_mode=str(decoder_basis_base_mode),
            basis_normalization=str(decoder_basis_normalization),
            basis_activation=str(decoder_basis_activation),
            basis_norm_eps=float(decoder_basis_norm_eps),
            latent_normalization=str(decoder_latent_normalization),
            raw_limit=decoder_raw_limit,
            mu_min=float(mu_min),
            mu_max=float(mu_max),
            initial_mu=float(initial_mu),
        )
        return cls(encoder=encoder, friction_decoder=decoder)

    def encode_context(
        self,
        context_features: torch.Tensor,
        context_valid_mask: torch.Tensor | None = None,
        trajectory_valid_mask: torch.Tensor | None = None,
        visual_features: torch.Tensor | None = None,
    ) -> ObjectLatentOutput:
        return self.encoder(
            context_features,
            valid_mask=context_valid_mask,
            trajectory_valid_mask=trajectory_valid_mask,
            visual_features=visual_features,
        )

    def decode_friction(
        self,
        point_features: torch.Tensor,
        latent: torch.Tensor,
        *,
        active_indices: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        return self.friction_decoder(point_features, latent, active_indices=active_indices)

    def forward(
        self,
        *,
        context_features: torch.Tensor,
        point_features: torch.Tensor,
        context_valid_mask: torch.Tensor | None = None,
        trajectory_valid_mask: torch.Tensor | None = None,
        visual_features: torch.Tensor | None = None,
        active_indices: torch.Tensor | np.ndarray | None = None,
    ) -> TrajectoryConditionedFrictionOutput:
        latent_output = self.encode_context(
            context_features,
            context_valid_mask=context_valid_mask,
            trajectory_valid_mask=trajectory_valid_mask,
            visual_features=visual_features,
        )
        friction = self.decode_friction(
            point_features,
            latent_output.latent,
            active_indices=active_indices,
        )
        return TrajectoryConditionedFrictionOutput(latent_output=latent_output, friction=friction)
