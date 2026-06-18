from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .features import FeatureNormalizer, normalizer_from_metadata, normalizer_to_metadata
from .model import PointNetResidualPredictor
from .rnn_model import RNNResidualPredictor
from stateful_gru_residual_adapter.model import StatefulGRUResidualPredictor


@dataclass(frozen=True)
class LoadedAdapterCheckpoint:
    path: Path
    metadata: dict[str, Any]
    model: nn.Module
    normalizer: FeatureNormalizer
    local_surface_points: np.ndarray
    full_point_friction: np.ndarray
    active_contact_mask: np.ndarray
    dino_features: np.ndarray | None
    dino_bottom_feature_copied_from_top: np.ndarray | None


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def save_adapter_checkpoint(
    *,
    checkpoint_path: Path,
    model: nn.Module,
    metadata: dict[str, Any],
    normalizer: FeatureNormalizer,
    local_surface_points: np.ndarray,
    full_point_friction: np.ndarray,
    active_contact_mask: np.ndarray,
    dino_features: np.ndarray | None,
    dino_bottom_feature_copied_from_top: np.ndarray | None,
    training_state: dict[str, Any] | None = None,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_with_norm = dict(metadata)
    metadata_with_norm.update(normalizer_to_metadata(normalizer))
    payload = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata_with_norm,
        "local_surface_points": np.asarray(local_surface_points, dtype=np.float32),
        "full_point_friction": np.asarray(full_point_friction, dtype=np.float32),
        "active_contact_mask": np.asarray(active_contact_mask, dtype=bool),
        "dino_features": None if dino_features is None else np.asarray(dino_features, dtype=np.float32),
        "dino_bottom_feature_copied_from_top": (
            None
            if dino_bottom_feature_copied_from_top is None
            else np.asarray(dino_bottom_feature_copied_from_top, dtype=np.float32)
        ),
    }
    if training_state is not None:
        payload["training_state"] = training_state
    torch.save(payload, checkpoint_path)
    save_json(checkpoint_path.with_name(f"{checkpoint_path.stem}_metadata.json"), metadata_with_norm)


def load_adapter_checkpoint(checkpoint_path: Path, *, map_location: str | torch.device = "cpu") -> LoadedAdapterCheckpoint:
    payload = torch.load(Path(checkpoint_path), map_location=map_location, weights_only=False)
    metadata = dict(payload["metadata"])
    normalizer = normalizer_from_metadata(metadata)
    architecture = str(metadata.get("adapter_architecture", "pointnet_residual_adapter"))
    if architecture == "rnn_residual_adapter":
        model = RNNResidualPredictor(
            point_feature_dim=int(metadata["point_feature_dim"]),
            history_window_steps=int(metadata["history_window_steps"]),
            prediction_window_steps=int(metadata["prediction_window_steps"]),
            hidden_size_1=int(metadata.get("rnn_hidden_size_1", 32)),
            hidden_size_2=int(metadata.get("rnn_hidden_size_2", 16)),
            point_pooling=str(metadata.get("rnn_point_pooling", "mean-max")),
            linear_output_scale=float(metadata["linear_output_scale"]),
            angular_output_scale=float(metadata["angular_output_scale"]),
            position_output_scale=float(metadata.get("position_output_scale", 0.01)),
            yaw_output_scale=float(metadata.get("yaw_output_scale", 0.1)),
            residual_output_mode=str(metadata.get("residual_output_mode", "velocity")),
        )
    elif architecture == "stateful_gru_residual_adapter":
        model = StatefulGRUResidualPredictor(
            point_feature_dim=int(metadata["point_feature_dim"]),
            prediction_window_steps=int(metadata.get("prediction_window_steps", 1)),
            hidden_size=int(metadata.get("gru_hidden_size", 16)),
            num_layers=int(metadata.get("gru_num_layers", 2)),
            point_pooling=str(metadata.get("gru_point_pooling", "mean-max")),
            linear_output_scale=float(metadata["linear_output_scale"]),
            angular_output_scale=float(metadata["angular_output_scale"]),
            position_output_scale=float(metadata.get("position_output_scale", 0.01)),
            yaw_output_scale=float(metadata.get("yaw_output_scale", 0.1)),
            residual_output_mode=str(metadata.get("residual_output_mode", "velocity")),
        )
    elif architecture in {"pointnet_residual_adapter", "pointnet"}:
        model = PointNetResidualPredictor(
            point_feature_dim=int(metadata["point_feature_dim"]),
            history_window_steps=int(metadata["history_window_steps"]),
            prediction_window_steps=int(metadata["prediction_window_steps"]),
            pointnet_feature_dim=int(metadata["pointnet_feature_dim"]),
            action_context_dim=int(metadata["action_context_dim"]),
            pooling=str(metadata["pointnet_pooling"]),
            linear_output_scale=float(metadata["linear_output_scale"]),
            angular_output_scale=float(metadata["angular_output_scale"]),
            position_output_scale=float(metadata.get("position_output_scale", 0.01)),
            yaw_output_scale=float(metadata.get("yaw_output_scale", 0.1)),
            residual_output_mode=str(metadata.get("residual_output_mode", "velocity")),
        )
    else:
        raise ValueError(f"Unsupported adapter architecture in checkpoint: {architecture!r}")
    model.load_state_dict(payload["model_state_dict"])
    dino_features = payload.get("dino_features")
    dino_bottom = payload.get("dino_bottom_feature_copied_from_top")
    return LoadedAdapterCheckpoint(
        path=Path(checkpoint_path),
        metadata=metadata,
        model=model,
        normalizer=normalizer,
        local_surface_points=np.asarray(payload["local_surface_points"], dtype=np.float32),
        full_point_friction=np.asarray(payload["full_point_friction"], dtype=np.float32),
        active_contact_mask=np.asarray(payload["active_contact_mask"], dtype=bool),
        dino_features=None if dino_features is None else np.asarray(dino_features, dtype=np.float32),
        dino_bottom_feature_copied_from_top=None if dino_bottom is None else np.asarray(dino_bottom, dtype=np.float32),
    )
