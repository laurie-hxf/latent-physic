from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from mujoco_contact_friction_fit_utils import MujocoTrajectoryCollection


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def init_wandb(
    args: argparse.Namespace,
    trajectory_collection: MujocoTrajectoryCollection,
    active_indices: np.ndarray,
) -> Any | None:
    if not args.wandb:
        return None

    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "wandb logging was requested, but the 'wandb' package is not installed. "
            "Install it with `pip install wandb`."
        ) from exc

    config = {key: to_jsonable(value) for key, value in vars(args).items()}
    representative_trajectory = trajectory_collection.trajectories[0]
    config["trajectory_loaded"] = dict(
        trajectory_npz=str(args.trajectory_npz),
        source_type=trajectory_collection.source_type,
        num_trajectories=trajectory_collection.num_trajectories,
        max_frames=trajectory_collection.max_frames,
        max_steps=trajectory_collection.max_steps,
        representative_num_frames=representative_trajectory.num_frames,
        representative_num_steps=representative_trajectory.num_steps,
        timestep=representative_trajectory.timestep,
        metadata=to_jsonable(trajectory_collection.metadata),
    )
    config["active_contact_point_count"] = int(len(active_indices))

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        dir=str(args.wandb_dir) if args.wandb_dir is not None else None,
        tags=args.wandb_tags,
        group=args.wandb_group,
        mode=args.wandb_mode,
        config=config,
        save_code=True,
    )


def build_wandb_log_payload(
    *,
    loss_value: float,
    position_loss_value: float,
    orientation_loss_value: float,
    linear_velocity_loss_value: float,
    angular_velocity_loss_value: float,
    raw_position_loss_value: float,
    raw_orientation_loss_value: float,
    raw_linear_velocity_loss_value: float,
    raw_angular_velocity_loss_value: float,
    grad_value: np.ndarray | None,
    active_params: np.ndarray | None = None,
    active_indices: np.ndarray,
    grad_norm_value: float | None = None,
    grad_abs_mean_value: float | None = None,
    grad_abs_max_value: float | None = None,
    mu_mean_value: float | None = None,
    mu_std_value: float | None = None,
    mu_min_value: float | None = None,
    mu_max_value: float | None = None,
) -> dict[str, float]:
    if grad_value is None:
        grad_norm = float("nan") if grad_norm_value is None else float(grad_norm_value)
        grad_abs_mean = float("nan") if grad_abs_mean_value is None else float(grad_abs_mean_value)
        grad_abs_max = float("nan") if grad_abs_max_value is None else float(grad_abs_max_value)
    else:
        grad_norm = float(np.linalg.norm(grad_value))
        grad_abs_mean = float(np.mean(np.abs(grad_value)))
        grad_abs_max = float(np.max(np.abs(grad_value)))

    if active_params is None:
        mu_mean = float("nan") if mu_mean_value is None else float(mu_mean_value)
        mu_std = float("nan") if mu_std_value is None else float(mu_std_value)
        mu_min = float("nan") if mu_min_value is None else float(mu_min_value)
        mu_max = float("nan") if mu_max_value is None else float(mu_max_value)
    else:
        mu_mean = float(active_params.mean())
        mu_std = float(active_params.std())
        mu_min = float(active_params.min())
        mu_max = float(active_params.max())

    return {
        "train/loss": float(loss_value),
        "loss/position": float(position_loss_value),
        "loss/orientation": float(orientation_loss_value),
        "loss/linear_velocity": float(linear_velocity_loss_value),
        "loss/angular_velocity": float(angular_velocity_loss_value),
        "loss_raw/position": float(raw_position_loss_value),
        "loss_raw/orientation": float(raw_orientation_loss_value),
        "loss_raw/linear_velocity": float(raw_linear_velocity_loss_value),
        "loss_raw/angular_velocity": float(raw_angular_velocity_loss_value),
        "train/grad_norm": grad_norm,
        "params/contact_point_count": float(len(active_indices)),
        "params/mu_mean": mu_mean,
        "params/mu_std": mu_std,
        "params/mu_min": mu_min,
        "params/mu_max": mu_max,
        "grads/mu_abs_mean": grad_abs_mean,
        "grads/mu_abs_max": grad_abs_max,
    }
