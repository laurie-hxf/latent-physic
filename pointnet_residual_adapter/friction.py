from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dino_mlp_warp_friction import build_warp_dino_mlp_friction_model
from fit_mujoco_contact_point_friction_params import (
    build_optimizer_param_positions,
    compute_piecewise_side_ids,
    expand_optimizer_params_to_active,
)
from mujoco_contact_friction_fit_utils import compute_active_contact_point_indices
from replay_mujoco_contact_friction_trajectory import (
    build_reference_to_scene_index,
    infer_base_point_friction,
    infer_box_half_extents_and_spacing,
    load_contact_friction_point_cloud,
)


@dataclass(frozen=True)
class FrictionConditioning:
    source_type: str
    source_path: Path | None
    parameterization: str
    full_point_friction: np.ndarray
    active_indices: np.ndarray
    active_contact_mask: np.ndarray
    metadata: dict[str, Any]


def default_checkpoint_point_cloud_path(checkpoint_path: Path | None) -> Path | None:
    if checkpoint_path is None:
        return None
    candidate = Path(checkpoint_path).with_suffix(".ply")
    return candidate if candidate.exists() else None


def maybe_configure_scene_from_point_cloud(args) -> Path | None:
    point_cloud_path = args.friction_point_cloud
    if point_cloud_path is None:
        point_cloud_path = default_checkpoint_point_cloud_path(args.friction_checkpoint)
    if point_cloud_path is None or not Path(point_cloud_path).exists():
        return None

    point_cloud = load_contact_friction_point_cloud(Path(point_cloud_path))
    half_extents, spacing = infer_box_half_extents_and_spacing(point_cloud.local_surface_points)
    args.box_half_extents = half_extents.tolist()
    args.surface_point_spacing = float(spacing)
    args.point_friction = infer_base_point_friction(point_cloud, fallback=float(args.point_friction))
    return Path(point_cloud_path)


def active_indices_from_trajectories(
    *,
    local_surface_points: np.ndarray,
    trajectories: list,
    floor_top_z: float,
    contact_threshold: float,
) -> np.ndarray:
    active_mask = np.zeros(len(local_surface_points), dtype=bool)
    for trajectory in trajectories:
        active_mask[
            compute_active_contact_point_indices(
                local_surface_points=local_surface_points,
                trajectory=trajectory,
                floor_top_z=float(floor_top_z),
                contact_threshold=float(contact_threshold),
            )
        ] = True
    active_indices = np.flatnonzero(active_mask).astype(np.int32)
    if len(active_indices) == 0:
        local_z = np.asarray(local_surface_points, dtype=np.float32)[:, 2]
        active_indices = np.flatnonzero(np.isclose(local_z, float(np.min(local_z)), atol=1.0e-6)).astype(np.int32)
    return active_indices


def _npz_scalar_or_none(data: np.lib.npyio.NpzFile, key: str) -> Any | None:
    if key not in data.files:
        return None
    value = np.asarray(data[key])
    if value.shape == ():
        return value.item()
    return value


def _checkpoint_param_vector(data: np.lib.npyio.NpzFile, *, param_set: str) -> np.ndarray | None:
    key_order = (
        ("best_optimizer_params", "optimizer_params")
        if param_set == "best"
        else ("optimizer_params", "best_optimizer_params")
    )
    for key in key_order:
        if key in data.files:
            return np.asarray(data[key], dtype=np.float32).reshape(-1)
    return None


def _checkpoint_active_params(data: np.lib.npyio.NpzFile, *, param_set: str) -> np.ndarray | None:
    key_order = (
        ("best_active_params", "active_params")
        if param_set == "best"
        else ("active_params", "best_active_params")
    )
    for key in key_order:
        if key in data.files:
            return np.asarray(data[key], dtype=np.float32).reshape(-1)
    return None


def _friction_parameterization(data: np.lib.npyio.NpzFile) -> str:
    if "friction_parameterization" not in data.files:
        return "point"
    return str(np.asarray(data["friction_parameterization"]).item())


def _build_dino_mlp_full_friction(
    *,
    data: np.lib.npyio.NpzFile,
    checkpoint_path: Path,
    local_surface_points: np.ndarray,
    box_half_extents: np.ndarray,
    active_indices: np.ndarray,
    param_set: str,
    point_friction: float,
    min_point_friction: float,
    max_point_friction: float,
    seed: int,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    dino_path_value = _npz_scalar_or_none(data, "dino_feature_npz_path")
    if dino_path_value is None or str(dino_path_value).strip() == "":
        return None

    params = _checkpoint_param_vector(data, param_set=param_set)
    if params is None:
        return None

    hidden_dim = int(_npz_scalar_or_none(data, "dino_mlp_hidden_dim") or 128)
    hidden_layers = int(_npz_scalar_or_none(data, "dino_mlp_hidden_layers") or 2)
    neighbor_radius = float(_npz_scalar_or_none(data, "dino_neighbor_radius") or 0.025)
    neighbor_k = int(_npz_scalar_or_none(data, "dino_neighbor_k") or 16)
    position_frequencies = int(_npz_scalar_or_none(data, "dino_position_frequencies") or 6)
    normalize_dino = bool(_npz_scalar_or_none(data, "dino_feature_normalization") if "dino_feature_normalization" in data.files else True)
    max_match_distance = float(_npz_scalar_or_none(data, "dino_mlp_max_match_distance") or 1.0e-5)

    model, model_metadata = build_warp_dino_mlp_friction_model(
        dino_npz_path=Path(str(dino_path_value)),
        local_surface_points=local_surface_points,
        half_extents=np.asarray(box_half_extents, dtype=np.float32),
        active_capacity=max(int(len(active_indices)), 1),
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        initial_mu=float(point_friction),
        min_mu=float(min_point_friction),
        max_mu=float(max_point_friction),
        seed=int(seed),
        device=str(device),
        position_frequencies=position_frequencies,
        neighbor_radius=neighbor_radius,
        neighbor_k=neighbor_k,
        normalize_dino=normalize_dino,
        max_match_distance=max_match_distance,
    )
    if params.shape != model.params_numpy().shape:
        raise ValueError(
            f"{checkpoint_path} DINO-MLP parameter shape {params.shape} does not match rebuilt model "
            f"shape {model.params_numpy().shape}"
        )
    model.assign_params(params)
    all_indices = np.arange(len(local_surface_points), dtype=np.int32)
    metadata = {
        "dino_feature_npz_path": str(dino_path_value),
        "dino_mlp_hidden_dim": hidden_dim,
        "dino_mlp_hidden_layers": hidden_layers,
        "dino_neighbor_radius": neighbor_radius,
        "dino_neighbor_k": neighbor_k,
        "dino_position_frequencies": position_frequencies,
        "dino_feature_normalization": normalize_dino,
        "dino_mlp_max_match_distance": max_match_distance,
    }
    metadata.update({f"dino_mlp_{key}": np.asarray(value).tolist() for key, value in model_metadata.items()})
    return model.predict_np(all_indices), metadata


def resolve_friction_conditioning(
    *,
    args,
    local_surface_points: np.ndarray,
    box_half_extents: np.ndarray,
    fallback_active_indices: np.ndarray,
    device: str,
) -> FrictionConditioning:
    point_count = len(local_surface_points)
    fallback_active_indices = np.asarray(fallback_active_indices, dtype=np.int32).reshape(-1)
    if len(fallback_active_indices) == 0:
        fallback_active_indices = np.arange(point_count, dtype=np.int32)

    if args.friction_point_cloud is not None:
        point_cloud = load_contact_friction_point_cloud(Path(args.friction_point_cloud))
        reference_to_scene = build_reference_to_scene_index(point_cloud.local_surface_points, local_surface_points)
        full = np.zeros(point_count, dtype=np.float32)
        full[reference_to_scene] = np.asarray(point_cloud.point_friction, dtype=np.float32)
        if point_cloud.active_mask is None:
            active_indices = fallback_active_indices
        else:
            active_indices = reference_to_scene[np.flatnonzero(point_cloud.active_mask).astype(np.int32)]
        active_mask = np.zeros(point_count, dtype=bool)
        active_mask[active_indices] = True
        return FrictionConditioning(
            source_type="point_cloud",
            source_path=Path(args.friction_point_cloud),
            parameterization="point",
            full_point_friction=full,
            active_indices=active_indices.astype(np.int32),
            active_contact_mask=active_mask,
            metadata={},
        )

    if args.friction_checkpoint is not None:
        checkpoint_path = Path(args.friction_checkpoint)
        with np.load(checkpoint_path, allow_pickle=True) as data:
            parameterization = _friction_parameterization(data)
            if "dino_mlp_all_point_friction" in data.files:
                full = np.asarray(data["dino_mlp_all_point_friction"], dtype=np.float32).reshape(-1)
                if full.shape != (point_count,):
                    raise ValueError(
                        f"{checkpoint_path} dino_mlp_all_point_friction shape {full.shape} does not match "
                        f"surface point count {point_count}"
                    )
                active_indices = np.asarray(data["active_indices"], dtype=np.int32).reshape(-1)
                metadata = {"used_dino_mlp_all_point_friction": True}
            else:
                active_indices = np.asarray(data["active_indices"], dtype=np.int32).reshape(-1)
                if np.min(active_indices) < 0 or np.max(active_indices) >= point_count:
                    raise ValueError(f"{checkpoint_path} active_indices do not fit current surface point grid")
                full = np.full(point_count, float(args.point_friction), dtype=np.float32)
                full_is_complete = False
                active_params = _checkpoint_active_params(data, param_set=str(args.checkpoint_param_set))
                if active_params is None or active_params.shape != active_indices.shape:
                    if parameterization == "dino-mlp":
                        rebuilt = _build_dino_mlp_full_friction(
                            data=data,
                            checkpoint_path=checkpoint_path,
                            local_surface_points=local_surface_points,
                            box_half_extents=box_half_extents,
                            active_indices=active_indices,
                            param_set=str(args.checkpoint_param_set),
                            point_friction=float(args.point_friction),
                            min_point_friction=float(args.min_point_friction),
                            max_point_friction=float(args.max_point_friction),
                            seed=int(args.seed),
                            device=device,
                        )
                        if rebuilt is None:
                            raise ValueError(f"{checkpoint_path} does not contain usable active or DINO-MLP friction")
                        full, metadata = rebuilt
                        full_is_complete = True
                    else:
                        params = _checkpoint_param_vector(data, param_set=str(args.checkpoint_param_set))
                        if params is None:
                            raise ValueError(f"{checkpoint_path} does not contain usable friction parameters")
                        side_ids = compute_piecewise_side_ids(local_surface_points, active_indices)
                        positions, _ = build_optimizer_param_positions(
                            parameterization=parameterization,
                            active_side_ids=side_ids,
                            active_count=len(active_indices),
                        )
                        active_params = expand_optimizer_params_to_active(
                            params,
                            positions,
                            parameterization=parameterization,
                        )
                        metadata = {"expanded_optimizer_params": True}
                else:
                    metadata = {}
                if not full_is_complete:
                    full[active_indices] = np.asarray(active_params, dtype=np.float32)
            active_mask = np.zeros(point_count, dtype=bool)
            active_mask[np.asarray(active_indices, dtype=np.int32)] = True
            return FrictionConditioning(
                source_type="checkpoint",
                source_path=checkpoint_path,
                parameterization=parameterization,
                full_point_friction=np.asarray(full, dtype=np.float32),
                active_indices=np.asarray(active_indices, dtype=np.int32),
                active_contact_mask=active_mask,
                metadata=metadata,
            )

    full = np.full(point_count, float(args.point_friction), dtype=np.float32)
    active_mask = np.zeros(point_count, dtype=bool)
    active_mask[fallback_active_indices] = True
    return FrictionConditioning(
        source_type="constant",
        source_path=None,
        parameterization="global",
        full_point_friction=full,
        active_indices=fallback_active_indices,
        active_contact_mask=active_mask,
        metadata={"point_friction": float(args.point_friction)},
    )
