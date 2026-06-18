from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
NEWTON_DIR = REPO_ROOT / "newton"
if str(NEWTON_DIR) not in sys.path:
    sys.path.insert(0, str(NEWTON_DIR))

from dino_mlp_warp_friction import (  # noqa: E402
    align_dino_features_to_surface_points,
    build_dino_mlp_input_features,
    positional_encoding_np,
)


def safe_logit(probability: float) -> float:
    clipped = float(np.clip(probability, 1.0e-6, 1.0 - 1.0e-6))
    return float(np.log(clipped / (1.0 - clipped)))


def friction_output_bias(*, initial_mu: float, mu_min: float, mu_max: float) -> float:
    span = float(mu_max) - float(mu_min)
    if span <= 0.0:
        raise ValueError("mu_max must be larger than mu_min")
    probability = (float(initial_mu) - float(mu_min)) / span
    return safe_logit(probability)


def build_decoder_mlp(
    *,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    hidden_layers: int,
    final_bias: float | np.ndarray | None = None,
    final_weight_std: float = 1.0e-3,
    activation: type[nn.Module] = nn.SiLU,
) -> nn.Sequential:
    if int(hidden_layers) < 0:
        raise ValueError("hidden_layers must be non-negative")
    layers: list[nn.Module] = []
    in_dim = int(input_dim)
    for _ in range(int(hidden_layers)):
        layer = nn.Linear(in_dim, int(hidden_dim))
        nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5.0))
        nn.init.zeros_(layer.bias)
        layers.extend([layer, activation()])
        in_dim = int(hidden_dim)

    final = nn.Linear(in_dim, int(output_dim))
    nn.init.normal_(final.weight, mean=0.0, std=float(final_weight_std))
    if final_bias is None:
        nn.init.zeros_(final.bias)
    else:
        bias_array = np.asarray(final_bias, dtype=np.float32).reshape(-1)
        if bias_array.size == 1 and int(output_dim) != 1:
            bias_array = np.full((int(output_dim),), float(bias_array[0]), dtype=np.float32)
        if bias_array.shape != (int(output_dim),):
            raise ValueError(f"final_bias shape {bias_array.shape} does not match output_dim={output_dim}")
        with torch.no_grad():
            final.bias.copy_(torch.as_tensor(bias_array, dtype=torch.float32))
    layers.append(final)
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class PointFeatureMetadata:
    input_dim: int
    encoded_position_dim: int
    dino_dim: int
    position_frequencies: int
    neighbor_radius: float
    neighbor_k: int
    normalize_dino: bool


def build_point_conditioning_features(
    *,
    local_surface_points: np.ndarray,
    half_extents: np.ndarray,
    dino_features: np.ndarray | None = None,
    dino_npz_path: Path | None = None,
    position_frequencies: int = 6,
    neighbor_radius: float = 0.025,
    neighbor_k: int = 16,
    normalize_dino: bool = True,
    max_match_distance: float = 1.0e-5,
) -> tuple[np.ndarray, PointFeatureMetadata, dict[str, np.ndarray]]:
    local_points = np.asarray(local_surface_points, dtype=np.float32).reshape(-1, 3)
    half_extents_np = np.asarray(half_extents, dtype=np.float32).reshape(3)
    if dino_features is None and dino_npz_path is not None:
        dino_features = align_dino_features_to_surface_points(
            dino_npz_path=Path(dino_npz_path),
            local_surface_points=local_points,
            max_match_distance=float(max_match_distance),
        )

    if dino_features is None:
        normalized_points = local_points / np.maximum(half_extents_np.reshape(1, 3), 1.0e-8)
        features = positional_encoding_np(normalized_points, int(position_frequencies)).astype(np.float32)
        stats: dict[str, np.ndarray] = {
            "encoded_position_dim": np.asarray(features.shape[1], dtype=np.int32),
            "neighbor_dino_dim": np.asarray(0, dtype=np.int32),
        }
        metadata = PointFeatureMetadata(
            input_dim=int(features.shape[1]),
            encoded_position_dim=int(features.shape[1]),
            dino_dim=0,
            position_frequencies=int(position_frequencies),
            neighbor_radius=float(neighbor_radius),
            neighbor_k=int(neighbor_k),
            normalize_dino=bool(normalize_dino),
        )
        return features, metadata, stats

    features, stats = build_dino_mlp_input_features(
        local_surface_points=local_points,
        half_extents=half_extents_np,
        dino_features=np.asarray(dino_features, dtype=np.float32),
        position_frequencies=int(position_frequencies),
        neighbor_radius=float(neighbor_radius),
        neighbor_k=int(neighbor_k),
        normalize_dino=bool(normalize_dino),
    )
    encoded_dim = int(np.asarray(stats["encoded_position_dim"]).reshape(()))
    dino_dim = int(np.asarray(stats["neighbor_dino_dim"]).reshape(()))
    metadata = PointFeatureMetadata(
        input_dim=int(features.shape[1]),
        encoded_position_dim=encoded_dim,
        dino_dim=dino_dim,
        position_frequencies=int(position_frequencies),
        neighbor_radius=float(neighbor_radius),
        neighbor_k=int(neighbor_k),
        normalize_dino=bool(normalize_dino),
    )
    return features.astype(np.float32), metadata, stats


class LatentConditionedFrictionDecoder(nn.Module):
    def __init__(
        self,
        *,
        point_feature_dim: int,
        latent_dim: int = 8,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        mu_min: float = 0.0,
        mu_max: float = 2.0,
        initial_mu: float = 0.35,
        conditioning: str = "film",
        activation: str = "silu",
        basis_count: int = 8,
        basis_base_mode: str = "latent",
        basis_normalization: str = "zero_mean",
        basis_activation: str = "tanh",
        basis_norm_eps: float = 1.0e-4,
        latent_normalization: str = "none",
        raw_limit: float | None = None,
    ) -> None:
        super().__init__()
        if float(mu_max) <= float(mu_min):
            raise ValueError("mu_max must be greater than mu_min")
        if int(hidden_layers) < 1:
            raise ValueError("hidden_layers must be >= 1")
        if str(conditioning) not in {"concat", "film", "basis"}:
            raise ValueError(f"Unknown decoder conditioning mode: {conditioning!r}")
        if str(activation) not in {"relu", "silu"}:
            raise ValueError(f"Unknown decoder activation: {activation!r}")
        if int(basis_count) < 1:
            raise ValueError("basis_count must be >= 1")
        if str(basis_base_mode) not in {"latent", "global_shared", "fixed"}:
            raise ValueError(f"Unknown basis base mode: {basis_base_mode!r}")
        if str(basis_normalization) not in {"none", "zero_mean", "unit_std"}:
            raise ValueError(f"Unknown basis normalization mode: {basis_normalization!r}")
        if str(basis_activation) not in {"tanh", "identity"}:
            raise ValueError(f"Unknown basis activation mode: {basis_activation!r}")
        if float(basis_norm_eps) <= 0.0:
            raise ValueError("basis_norm_eps must be positive")
        if str(latent_normalization) not in {"none", "layernorm"}:
            raise ValueError(f"Unknown latent normalization mode: {latent_normalization!r}")
        if raw_limit is not None and float(raw_limit) <= 0.0:
            raise ValueError("raw_limit must be positive when provided")
        self.point_feature_dim = int(point_feature_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = int(hidden_layers)
        self.mu_min = float(mu_min)
        self.mu_max = float(mu_max)
        self.conditioning = str(conditioning)
        self.activation_name = str(activation)
        self.basis_count = int(basis_count)
        self.basis_base_mode = str(basis_base_mode)
        self.basis_normalization = str(basis_normalization)
        self.basis_activation = str(basis_activation)
        self.basis_norm_eps = float(basis_norm_eps)
        self.latent_normalization = str(latent_normalization)
        self.raw_limit = None if raw_limit is None else float(raw_limit)
        # Latent scale is now fixed at the encoder output. Keep the
        # ``layernorm`` option accepted so older configs/checkpoints can still
        # be inspected, but do not apply a decoder-side normalization that can
        # erase object-specific latent differences.
        self.latent_norm_layer = None
        activation_type = nn.ReLU if self.activation_name == "relu" else nn.SiLU
        bias = friction_output_bias(initial_mu=float(initial_mu), mu_min=self.mu_min, mu_max=self.mu_max)
        if self.conditioning == "concat":
            self.mlp = build_decoder_mlp(
                input_dim=self.point_feature_dim + self.latent_dim,
                output_dim=1,
                hidden_dim=self.hidden_dim,
                hidden_layers=self.hidden_layers,
                final_bias=bias,
                activation=activation_type,
            )
        elif self.conditioning == "basis":
            self.basis_net = build_decoder_mlp(
                input_dim=self.point_feature_dim,
                output_dim=self.basis_count,
                hidden_dim=self.hidden_dim,
                hidden_layers=self.hidden_layers,
                final_bias=0.0,
                activation=activation_type,
            )
            self.coef_head = build_decoder_mlp(
                input_dim=self.latent_dim,
                output_dim=self.basis_count,
                hidden_dim=self.hidden_dim,
                hidden_layers=1,
                final_bias=0.0,
                final_weight_std=1.0e-3,
                activation=activation_type,
            )
            if self.basis_base_mode == "latent":
                self.base_head = build_decoder_mlp(
                    input_dim=self.latent_dim,
                    output_dim=1,
                    hidden_dim=self.hidden_dim,
                    hidden_layers=1,
                    final_bias=bias,
                    final_weight_std=1.0e-3,
                    activation=activation_type,
                )
            elif self.basis_base_mode == "global_shared":
                self.basis_base = nn.Parameter(torch.tensor(float(bias), dtype=torch.float32))
            else:
                self.register_buffer("basis_base", torch.tensor(float(bias), dtype=torch.float32))
        else:
            self.point_layers = nn.ModuleList()
            self.film_layers = nn.ModuleList()
            in_dim = self.point_feature_dim
            for _ in range(self.hidden_layers):
                point_layer = nn.Linear(in_dim, self.hidden_dim)
                nn.init.kaiming_uniform_(point_layer.weight, a=math.sqrt(5.0))
                nn.init.zeros_(point_layer.bias)
                self.point_layers.append(point_layer)

                film_layer = nn.Linear(self.latent_dim, 2 * self.hidden_dim)
                nn.init.normal_(film_layer.weight, mean=0.0, std=1.0e-3)
                nn.init.zeros_(film_layer.bias)
                self.film_layers.append(film_layer)
                in_dim = self.hidden_dim

            self.activation = activation_type()
            self.output_layer = nn.Linear(self.hidden_dim, 1)
            nn.init.normal_(self.output_layer.weight, mean=0.0, std=1.0e-3)
            with torch.no_grad():
                self.output_layer.bias.fill_(float(bias))

    def _prepare_inputs(
        self,
        point_features: torch.Tensor,
        latent: torch.Tensor,
        *,
        active_indices: torch.Tensor | np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if latent.ndim == 1:
            latent = latent.unsqueeze(0)
        if latent.ndim != 2:
            raise ValueError(f"latent must have shape (B,L) or (L,), got {tuple(latent.shape)}")
        if latent.shape[-1] != self.latent_dim:
            raise ValueError(f"latent dim mismatch: got {latent.shape[-1]}, expected {self.latent_dim}")
        decoder_latent = self.latent_norm_layer(latent) if self.latent_norm_layer is not None else latent

        features = point_features
        if active_indices is not None:
            index = torch.as_tensor(active_indices, dtype=torch.long, device=features.device)
            features = features.index_select(-2, index)

        if features.ndim == 2:
            if features.shape[-1] != self.point_feature_dim:
                raise ValueError(
                    f"point feature dim mismatch: got {features.shape[-1]}, expected {self.point_feature_dim}"
                )
            features = features.unsqueeze(0).expand(latent.shape[0], -1, -1)
        elif features.ndim == 3:
            if features.shape[0] != latent.shape[0]:
                raise ValueError(f"point feature batch {features.shape[0]} does not match latent batch {latent.shape[0]}")
            if features.shape[-1] != self.point_feature_dim:
                raise ValueError(
                    f"point feature dim mismatch: got {features.shape[-1]}, expected {self.point_feature_dim}"
                )
        else:
            raise ValueError(f"point_features must have shape (P,F) or (B,P,F), got {tuple(features.shape)}")
        return features, decoder_latent

    def _normalize_basis(self, basis: torch.Tensor) -> torch.Tensor:
        if self.basis_normalization == "none":
            return basis
        centered = basis - basis.mean(dim=1, keepdim=True)
        if self.basis_normalization == "zero_mean":
            return centered
        std = centered.std(dim=1, unbiased=False, keepdim=True)
        return centered / std.clamp_min(self.basis_norm_eps)

    def _basis_terms(
        self,
        features: torch.Tensor,
        decoder_latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        basis_raw = self.basis_net(features)
        basis_activated = torch.tanh(basis_raw) if self.basis_activation == "tanh" else basis_raw
        basis = self._normalize_basis(basis_activated)
        coef = self.coef_head(decoder_latent)
        if self.basis_base_mode == "latent":
            base = self.base_head(decoder_latent).squeeze(-1)
        else:
            base = self.basis_base.to(device=decoder_latent.device, dtype=decoder_latent.dtype).expand(
                decoder_latent.shape[0]
            )
        spatial = torch.sum(basis * coef.unsqueeze(1), dim=-1)
        raw = base.unsqueeze(1) + spatial
        return basis_raw, basis, coef, base, spatial, raw

    def forward(
        self,
        point_features: torch.Tensor,
        latent: torch.Tensor,
        *,
        active_indices: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        features, decoder_latent = self._prepare_inputs(
            point_features,
            latent,
            active_indices=active_indices,
        )

        if self.conditioning == "concat":
            latent_features = decoder_latent.unsqueeze(1).expand(-1, features.shape[1], -1)
            raw = self.mlp(torch.cat((features, latent_features), dim=-1)).squeeze(-1)
        elif self.conditioning == "basis":
            _, _, _, _, _, raw = self._basis_terms(features, decoder_latent)
        else:
            hidden = features
            for point_layer, film_layer in zip(self.point_layers, self.film_layers, strict=True):
                hidden = point_layer(hidden)
                gamma, beta = film_layer(decoder_latent).chunk(2, dim=-1)
                hidden = self.activation(
                    hidden * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
                )
            raw = self.output_layer(hidden).squeeze(-1)
        if self.raw_limit is not None:
            raw = self.raw_limit * torch.tanh(raw / self.raw_limit)
        return self.mu_min + (self.mu_max - self.mu_min) * torch.sigmoid(raw)

    @torch.no_grad()
    def basis_diagnostics(
        self,
        point_features: torch.Tensor,
        latent: torch.Tensor,
        *,
        active_indices: torch.Tensor | np.ndarray | None = None,
    ) -> dict[str, float | str]:
        if self.conditioning != "basis":
            return {}
        features, decoder_latent = self._prepare_inputs(
            point_features,
            latent,
            active_indices=active_indices,
        )
        basis_raw, basis, coef, base, spatial, raw = self._basis_terms(features, decoder_latent)

        def scalar_stats(prefix: str, values: torch.Tensor) -> dict[str, float]:
            flat = values.detach().to(dtype=torch.float64).reshape(-1)
            if flat.numel() == 0:
                return {
                    f"{prefix}_mean": float("nan"),
                    f"{prefix}_std": float("nan"),
                    f"{prefix}_min": float("nan"),
                    f"{prefix}_max": float("nan"),
                }
            return {
                f"{prefix}_mean": float(flat.mean().cpu()),
                f"{prefix}_std": float(flat.std(unbiased=False).cpu()),
                f"{prefix}_min": float(flat.min().cpu()),
                f"{prefix}_max": float(flat.max().cpu()),
            }

        result: dict[str, float | str] = {
            "basis_base_mode": self.basis_base_mode,
            "basis_normalization": self.basis_normalization,
            "basis_activation": self.basis_activation,
            "basis_norm_eps": float(self.basis_norm_eps),
        }
        result.update(scalar_stats("basis_raw", basis_raw))
        result.update(scalar_stats("basis", basis))
        result.update(scalar_stats("coef", coef))
        result.update(scalar_stats("base_raw", base))
        result.update(scalar_stats("spatial_raw", spatial))
        result.update(scalar_stats("raw_pre_limit", raw))
        return result

    @torch.no_grad()
    def predict_np(
        self,
        point_features: np.ndarray,
        latent: np.ndarray,
        *,
        active_indices: np.ndarray | None = None,
        device: torch.device | str | None = None,
    ) -> np.ndarray:
        model_device = next(self.parameters()).device
        tensor_device = torch.device(device) if device is not None else model_device
        mu = self.forward(
            torch.as_tensor(point_features, dtype=torch.float32, device=tensor_device),
            torch.as_tensor(latent, dtype=torch.float32, device=tensor_device),
            active_indices=active_indices,
        )
        return mu.detach().cpu().numpy().astype(np.float32)
