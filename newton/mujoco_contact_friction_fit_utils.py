from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

from pbd_math import transform_points


@dataclass
class MujocoTrajectory:
    time: np.ndarray
    positions: np.ndarray
    quaternions_xyzw: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    step_forces: np.ndarray
    step_application_points: np.ndarray
    timestep: float
    metadata: dict

    @property
    def num_frames(self) -> int:
        return int(self.positions.shape[0])

    @property
    def num_steps(self) -> int:
        return max(self.num_frames - 1, 0)


@dataclass
class MujocoTrajectoryCollection:
    trajectories: list[MujocoTrajectory]
    source_type: str
    source_path: Path
    metadata: dict

    @property
    def num_trajectories(self) -> int:
        return len(self.trajectories)

    @property
    def max_steps(self) -> int:
        if not self.trajectories:
            return 0
        return max(trajectory.num_steps for trajectory in self.trajectories)

    @property
    def max_frames(self) -> int:
        if not self.trajectories:
            return 0
        return max(trajectory.num_frames for trajectory in self.trajectories)


@dataclass
class OptimizationBuffers:
    active_point_friction: wp.array
    active_indices: wp.array
    full_point_friction: wp.array
    contact_weighted_masses: wp.array
    contact_weighted_mass_total: wp.array
    step_forces: wp.array
    step_application_points: wp.array
    target_positions: wp.array
    target_quaternions: wp.array
    target_linear_velocity: wp.array
    target_angular_velocity: wp.array
    loss: wp.array
    position_loss: wp.array
    orientation_loss: wp.array
    linear_velocity_loss: wp.array
    angular_velocity_loss: wp.array
    inactive_point_friction_np: np.ndarray


@dataclass
class BatchedOptimizationBuffers:
    batch_size: int
    max_steps: int
    max_frames: int
    active_point_friction: wp.array
    active_indices: wp.array
    full_point_friction: wp.array
    contact_weighted_masses: wp.array
    contact_weighted_mass_total: wp.array
    step_forces: wp.array
    step_application_points: wp.array
    target_positions: wp.array
    target_quaternions: wp.array
    target_linear_velocity: wp.array
    target_angular_velocity: wp.array
    trajectory_step_counts: wp.array
    frame_scales: wp.array
    loss: wp.array
    position_loss: wp.array
    orientation_loss: wp.array
    linear_velocity_loss: wp.array
    angular_velocity_loss: wp.array
    batch_loss: wp.array
    inactive_point_friction_np: np.ndarray


def quat_wxyz_to_xyzw(quaternions_wxyz: np.ndarray) -> np.ndarray:
    quaternions_xyzw = np.concatenate([quaternions_wxyz[:, 1:4], quaternions_wxyz[:, 0:1]], axis=1)
    norms = np.linalg.norm(quaternions_xyzw, axis=1, keepdims=True)
    safe_norms = np.maximum(norms, 1.0e-8)
    return (quaternions_xyzw / safe_norms).astype(np.float32)


def _truncate_trajectory(
    *,
    time: np.ndarray,
    positions: np.ndarray,
    quaternions_wxyz: np.ndarray,
    linear_velocity: np.ndarray,
    angular_velocity: np.ndarray,
    applied_force: np.ndarray,
    application_point: np.ndarray,
    metadata: dict,
    max_steps: int | None,
) -> MujocoTrajectory:
    if len(time) < 2:
        raise ValueError("trajectory does not contain enough frames")

    timestep = float(metadata.get("timestep", time[1] - time[0]))
    total_steps = len(time) - 1
    if max_steps is None:
        used_steps = total_steps
    else:
        used_steps = min(max(int(max_steps), 1), total_steps)

    used_frames = used_steps + 1
    return MujocoTrajectory(
        time=np.asarray(time[:used_frames], dtype=np.float32),
        positions=np.asarray(positions[:used_frames], dtype=np.float32),
        quaternions_xyzw=quat_wxyz_to_xyzw(np.asarray(quaternions_wxyz[:used_frames], dtype=np.float32)),
        linear_velocity=np.asarray(linear_velocity[:used_frames], dtype=np.float32),
        angular_velocity=np.asarray(angular_velocity[:used_frames], dtype=np.float32),
        step_forces=np.asarray(applied_force[1:used_frames], dtype=np.float32),
        step_application_points=np.asarray(application_point[1:used_frames], dtype=np.float32),
        timestep=timestep,
        metadata=dict(metadata),
    )


def load_mujoco_trajectory(trajectory_npz_path: Path, max_steps: int | None) -> MujocoTrajectory:
    with np.load(trajectory_npz_path, allow_pickle=True) as data:
        time = np.asarray(data["time"], dtype=np.float32)
        positions = np.asarray(data["position"], dtype=np.float32)
        quaternions_wxyz = np.asarray(data["quaternion"], dtype=np.float32)
        linear_velocity = np.asarray(data["linear_velocity"], dtype=np.float32)
        angular_velocity = np.asarray(data["angular_velocity"], dtype=np.float32)
        applied_force = np.asarray(data["applied_force"], dtype=np.float32)
        application_point = np.asarray(data["application_point"], dtype=np.float32)
        metadata_json = data["metadata_json"].item() if "metadata_json" in data.files else "{}"

    metadata = json.loads(metadata_json)
    try:
        return _truncate_trajectory(
            time=time,
            positions=positions,
            quaternions_wxyz=quaternions_wxyz,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            applied_force=applied_force,
            application_point=application_point,
            metadata=metadata,
            max_steps=max_steps,
        )
    except ValueError as exc:
        raise ValueError(f"{trajectory_npz_path} {exc}") from exc


def load_mujoco_trajectory_dataset(
    dataset_npz_path: Path,
    max_steps: int | None,
    max_trajectories: int | None = None,
) -> MujocoTrajectoryCollection:
    with np.load(dataset_npz_path, allow_pickle=True) as data:
        trajectories = np.asarray(data["trajectories"], dtype=np.float32)
        columns = data["columns"].tolist()
        episode_lengths = np.asarray(data["episode_lengths"], dtype=np.int32)
        summary_metadata_json = data["summary_metadata_json"].item() if "summary_metadata_json" in data.files else "{}"
        episode_metadata_json = data["episode_metadata_json"].item() if "episode_metadata_json" in data.files else "[]"

    summary_metadata = json.loads(summary_metadata_json)
    episode_metadata_list = json.loads(episode_metadata_json)
    if trajectories.ndim != 3:
        raise ValueError(f"{dataset_npz_path} expected trajectories with rank 3, got shape {trajectories.shape}")
    if len(columns) != trajectories.shape[2]:
        raise ValueError(
            f"{dataset_npz_path} column count mismatch: {len(columns)} names for width {trajectories.shape[2]}"
        )
    if len(episode_lengths) != trajectories.shape[0]:
        raise ValueError(
            f"{dataset_npz_path} episode length count mismatch: {len(episode_lengths)} lengths for "
            f"{trajectories.shape[0]} trajectories"
        )

    column_to_index = {str(name): idx for idx, name in enumerate(columns)}
    required_columns = [
        "time",
        "pos_x",
        "pos_y",
        "pos_z",
        "quat_w",
        "quat_x",
        "quat_y",
        "quat_z",
        "linvel_x",
        "linvel_y",
        "linvel_z",
        "angvel_x",
        "angvel_y",
        "angvel_z",
        "force_x",
        "force_y",
        "force_z",
        "point_x",
        "point_y",
        "point_z",
    ]
    missing_columns = [name for name in required_columns if name not in column_to_index]
    if missing_columns:
        raise ValueError(f"{dataset_npz_path} missing required columns: {missing_columns}")

    trajectory_list: list[MujocoTrajectory] = []
    trajectory_limit = trajectories.shape[0] if max_trajectories is None else min(
        trajectories.shape[0],
        max(int(max_trajectories), 1),
    )
    for episode_idx in range(trajectory_limit):
        episode_length = int(episode_lengths[episode_idx])
        if episode_length < 2:
            continue

        episode_array = trajectories[episode_idx, :episode_length]
        metadata = {"episode_index": episode_idx}
        if summary_metadata.get("timestep") is not None:
            metadata["timestep"] = summary_metadata["timestep"]
        if isinstance(episode_metadata_list, list) and episode_idx < len(episode_metadata_list):
            episode_meta = episode_metadata_list[episode_idx]
            if isinstance(episode_meta, dict):
                metadata.update(episode_meta)

        trajectory_list.append(
            _truncate_trajectory(
                time=episode_array[:, column_to_index["time"]],
                positions=episode_array[
                    :,
                    [
                        column_to_index["pos_x"],
                        column_to_index["pos_y"],
                        column_to_index["pos_z"],
                    ],
                ],
                quaternions_wxyz=episode_array[
                    :,
                    [
                        column_to_index["quat_w"],
                        column_to_index["quat_x"],
                        column_to_index["quat_y"],
                        column_to_index["quat_z"],
                    ],
                ],
                linear_velocity=episode_array[
                    :,
                    [
                        column_to_index["linvel_x"],
                        column_to_index["linvel_y"],
                        column_to_index["linvel_z"],
                    ],
                ],
                angular_velocity=episode_array[
                    :,
                    [
                        column_to_index["angvel_x"],
                        column_to_index["angvel_y"],
                        column_to_index["angvel_z"],
                    ],
                ],
                applied_force=episode_array[
                    :,
                    [
                        column_to_index["force_x"],
                        column_to_index["force_y"],
                        column_to_index["force_z"],
                    ],
                ],
                application_point=episode_array[
                    :,
                    [
                        column_to_index["point_x"],
                        column_to_index["point_y"],
                        column_to_index["point_z"],
                    ],
                ],
                metadata=metadata,
                max_steps=max_steps,
            )
        )

    if not trajectory_list:
        raise ValueError(f"{dataset_npz_path} does not contain any usable trajectories")

    return MujocoTrajectoryCollection(
        trajectories=trajectory_list,
        source_type="dataset",
        source_path=dataset_npz_path,
        metadata=summary_metadata,
    )


def load_mujoco_trajectories(
    trajectory_npz_path: Path,
    max_steps: int | None,
    max_trajectories: int | None = None,
) -> MujocoTrajectoryCollection:
    with np.load(trajectory_npz_path, allow_pickle=True) as data:
        if "trajectories" in data.files and "columns" in data.files and "episode_lengths" in data.files:
            return load_mujoco_trajectory_dataset(trajectory_npz_path, max_steps, max_trajectories)

    trajectory = load_mujoco_trajectory(trajectory_npz_path, max_steps)
    return MujocoTrajectoryCollection(
        trajectories=[trajectory],
        source_type="single_trajectory",
        source_path=trajectory_npz_path,
        metadata=trajectory.metadata,
    )


def compute_active_contact_point_indices(
    local_surface_points: np.ndarray,
    trajectory: MujocoTrajectory,
    floor_top_z: float,
    contact_threshold: float,
) -> np.ndarray:
    active_mask = np.zeros(len(local_surface_points), dtype=bool)
    for position, quaternion in zip(trajectory.positions, trajectory.quaternions_xyzw, strict=True):
        world_points = np.asarray(transform_points(local_surface_points, position, quaternion), dtype=np.float32)
        active_mask |= world_points[:, 2] <= float(floor_top_z + contact_threshold)
    return np.flatnonzero(active_mask).astype(np.int32)


def run_adam_update(
    params: np.ndarray,
    grads: np.ndarray,
    first_moment: np.ndarray,
    second_moment: np.ndarray,
    step: int,
    learning_rate: float,
    beta1: float,
    beta2: float,
    eps: float,
    min_value: float,
    max_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params64 = np.asarray(params, dtype=np.float64)
    grads64 = np.asarray(grads, dtype=np.float64)
    first_moment64 = np.asarray(first_moment, dtype=np.float64)
    second_moment64 = np.asarray(second_moment, dtype=np.float64)

    first_moment64 = beta1 * first_moment64 + (1.0 - beta1) * grads64
    second_moment64 = beta2 * second_moment64 + (1.0 - beta2) * (grads64 * grads64)
    first_hat = first_moment64 / (1.0 - beta1**step)
    second_hat = second_moment64 / (1.0 - beta2**step)
    params64 = params64 - learning_rate * first_hat / (np.sqrt(second_hat) + eps)
    params64 = np.clip(params64, min_value, max_value)
    return params64.astype(np.float32), first_moment64, second_moment64
