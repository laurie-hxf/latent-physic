from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


BASE_POINT_FEATURE_SCHEMA = [
    "local_point_x_over_half_extent_x",
    "local_point_y_over_half_extent_y",
    "local_point_z_over_half_extent_z",
    "point_velocity_body_x",
    "point_velocity_body_y",
    "point_velocity_body_z",
    "rigid_linear_velocity_body_x",
    "rigid_linear_velocity_body_y",
    "rigid_angular_velocity_z",
    "force_body_x",
    "force_body_y",
    "force_body_z",
    "point_offset_local_x",
    "point_offset_local_y",
    "point_offset_local_z",
    "torque_z",
    "mu_i",
    "is_active_contact_point",
]

ACTION_FEATURE_SCHEMA = [
    "force_body_x",
    "force_body_y",
    "force_body_z",
    "point_offset_local_x",
    "point_offset_local_y",
    "point_offset_local_z",
    "torque_z",
]


def normalize_residual_output_mode(mode: str | None) -> str:
    value = "velocity" if mode is None else str(mode).strip().lower()
    aliases = {
        "velocity": "velocity",
        "vel": "velocity",
        "delta_v": "velocity",
        "deltav": "velocity",
        "acceleration": "acceleration",
        "accel": "acceleration",
        "acc": "acceleration",
        "pose": "pose",
        "position": "pose",
        "pos": "pose",
        "trajectory": "pose",
        "pose_velocity": "pose_velocity",
        "pose+velocity": "pose_velocity",
        "pose-velocity": "pose_velocity",
        "pose_vel": "pose_velocity",
        "pos_vel": "pose_velocity",
        "all": "pose_velocity",
    }
    if value not in aliases:
        raise ValueError(f"Unsupported residual output mode: {mode!r}")
    return aliases[value]


def residual_output_dim(mode: str | None) -> int:
    output_mode = normalize_residual_output_mode(mode)
    return 6 if output_mode == "pose_velocity" else 3


def residual_output_components(mode: str | None) -> tuple[bool, bool]:
    output_mode = normalize_residual_output_mode(mode)
    has_pose = output_mode in {"pose", "pose_velocity"}
    has_velocity = output_mode in {"velocity", "acceleration", "pose_velocity"}
    return has_pose, has_velocity


@dataclass(frozen=True)
class DinoFeatures:
    path: Path
    features: np.ndarray
    bottom_feature_copied_from_top: np.ndarray
    max_match_distance: float

    @property
    def dim(self) -> int:
        return int(self.features.shape[1]) if self.features.ndim == 2 else 0


@dataclass(frozen=True)
class FeatureNormalizer:
    point_feature_mean: np.ndarray
    point_feature_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray


@dataclass(frozen=True)
class TorchFeatureNormalizer:
    point_feature_mean: torch.Tensor
    point_feature_std: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float32)
    single = q.ndim == 1
    q = q.reshape(-1, 4)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.maximum(norm, 1.0e-8)
    x = q[:, 0]
    y = q[:, 1]
    z = q[:, 2]
    w = q[:, 3]

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    matrices = np.empty((q.shape[0], 3, 3), dtype=np.float32)
    matrices[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    matrices[:, 0, 1] = 2.0 * (xy - wz)
    matrices[:, 0, 2] = 2.0 * (xz + wy)
    matrices[:, 1, 0] = 2.0 * (xy + wz)
    matrices[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    matrices[:, 1, 2] = 2.0 * (yz - wx)
    matrices[:, 2, 0] = 2.0 * (xz - wy)
    matrices[:, 2, 1] = 2.0 * (yz + wx)
    matrices[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return matrices[0] if single else matrices


def quaternion_xyzw_to_yaw(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float32)
    single = q.ndim == 1
    q = q.reshape(-1, 4)
    x = q[:, 0]
    y = q[:, 1]
    z = q[:, 2]
    w = q[:, 3]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)).astype(np.float32)
    return yaw[0] if single else yaw


def quaternion_xyzw_to_yaw_torch(quaternion: torch.Tensor) -> torch.Tensor:
    q = quaternion.to(dtype=torch.float32)
    norm = torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1.0e-8)
    q = q / norm
    x = q[..., 0]
    y = q[..., 1]
    z = q[..., 2]
    w = q[..., 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quaternion_xyzw_to_matrix_torch(quaternion: torch.Tensor) -> torch.Tensor:
    q = quaternion.to(dtype=torch.float32)
    norm = torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1.0e-8)
    q = q / norm
    x = q[..., 0]
    y = q[..., 1]
    z = q[..., 2]
    w = q[..., 3]

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    row0 = torch.stack((1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)), dim=-1)
    row1 = torch.stack((2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)), dim=-1)
    row2 = torch.stack((2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def _nearest_indices_chunked(source: np.ndarray, query: np.ndarray, *, chunk_size: int = 1024) -> np.ndarray:
    source_np = np.asarray(source, dtype=np.float32).reshape(-1, 3)
    query_np = np.asarray(query, dtype=np.float32).reshape(-1, 3)
    result = np.empty((len(query_np),), dtype=np.int64)
    for start in range(0, len(query_np), int(chunk_size)):
        end = min(start + int(chunk_size), len(query_np))
        delta = query_np[start:end, None, :] - source_np[None, :, :]
        dist2 = np.sum(delta * delta, axis=-1)
        result[start:end] = np.argmin(dist2, axis=1)
    return result


def load_aligned_dino_features(
    dino_npz_path: Path,
    local_surface_points: np.ndarray,
    *,
    max_match_distance: float,
) -> DinoFeatures:
    path = Path(dino_npz_path)
    with np.load(path, allow_pickle=True) as data:
        if "dino_features" not in data.files:
            raise ValueError(f"{path} does not contain dino_features")
        dino_features = np.asarray(data["dino_features"], dtype=np.float32)
        if "local_points" in data.files:
            dino_points = np.asarray(data["local_points"], dtype=np.float32).reshape(-1, 3)
        elif "points" in data.files:
            dino_points = np.asarray(data["points"], dtype=np.float32).reshape(-1, 3)
        else:
            raise ValueError(f"{path} must contain local_points or points")
        if "bottom_feature_copied_from_top" in data.files:
            bottom_flags = np.asarray(data["bottom_feature_copied_from_top"], dtype=np.float32).reshape(-1)
        else:
            bottom_flags = np.zeros((dino_features.shape[0],), dtype=np.float32)

    target = np.asarray(local_surface_points, dtype=np.float32).reshape(-1, 3)
    if dino_points.shape == target.shape and np.allclose(dino_points, target, atol=float(max_match_distance)):
        aligned_features = dino_features
        aligned_bottom_flags = bottom_flags
    else:
        nearest = _nearest_indices_chunked(dino_points, target)
        matched = dino_points[nearest]
        distances = np.linalg.norm(matched - target, axis=1)
        max_distance = float(np.max(distances)) if len(distances) else 0.0
        if max_distance > float(max_match_distance):
            raise ValueError(
                f"DINO feature points do not align with Newton surface points; "
                f"max nearest local distance={max_distance:.6g} > {float(max_match_distance):.6g}"
            )
        aligned_features = dino_features[nearest]
        aligned_bottom_flags = bottom_flags[nearest]

    return DinoFeatures(
        path=path,
        features=np.asarray(aligned_features, dtype=np.float32).copy(),
        bottom_feature_copied_from_top=np.asarray(aligned_bottom_flags, dtype=np.float32).reshape(-1).copy(),
        max_match_distance=float(max_match_distance),
    )


def point_feature_schema(dino_dim: int) -> list[str]:
    schema = list(BASE_POINT_FEATURE_SCHEMA)
    schema.extend([f"dino_feature_{idx:03d}" for idx in range(int(dino_dim))])
    if int(dino_dim) > 0:
        schema.append("bottom_feature_copied_from_top")
    return schema


def action_features_from_force(
    *,
    quaternion_xyzw: np.ndarray,
    force_world: np.ndarray,
    point_offset_local: np.ndarray,
) -> np.ndarray:
    rotation = quaternion_xyzw_to_matrix(quaternion_xyzw)
    force_body = rotation.T @ np.asarray(force_world, dtype=np.float32).reshape(3)
    offset = np.asarray(point_offset_local, dtype=np.float32).reshape(3)
    torque_z = offset[0] * force_body[1] - offset[1] * force_body[0]
    return np.asarray(
        [
            force_body[0],
            force_body[1],
            force_body[2],
            offset[0],
            offset[1],
            offset[2],
            torque_z,
        ],
        dtype=np.float32,
    )


def build_point_feature_frame(
    *,
    local_surface_points: np.ndarray,
    box_half_extents: np.ndarray,
    quaternion_xyzw: np.ndarray,
    linear_velocity_world: np.ndarray,
    angular_velocity_world: np.ndarray,
    force_world: np.ndarray,
    point_offset_local: np.ndarray,
    point_friction: np.ndarray,
    active_contact_mask: np.ndarray,
    dino: DinoFeatures | None,
) -> np.ndarray:
    local_points = np.asarray(local_surface_points, dtype=np.float32).reshape(-1, 3)
    point_count = local_points.shape[0]
    half_extents = np.maximum(np.asarray(box_half_extents, dtype=np.float32).reshape(1, 3), 1.0e-8)
    rotation = quaternion_xyzw_to_matrix(quaternion_xyzw)
    linear_world = np.asarray(linear_velocity_world, dtype=np.float32).reshape(3)
    angular_world = np.asarray(angular_velocity_world, dtype=np.float32).reshape(3)

    relative_world = local_points @ rotation.T
    point_velocity_world = linear_world.reshape(1, 3) + np.cross(angular_world.reshape(1, 3), relative_world)
    point_velocity_body = point_velocity_world @ rotation
    rigid_linear_velocity_body = rotation.T @ linear_world
    rigid_angular_velocity_body = rotation.T @ angular_world
    action = action_features_from_force(
        quaternion_xyzw=quaternion_xyzw,
        force_world=force_world,
        point_offset_local=point_offset_local,
    )

    repeated = np.column_stack(
        [
            local_points / half_extents,
            point_velocity_body,
            np.full(point_count, rigid_linear_velocity_body[0], dtype=np.float32),
            np.full(point_count, rigid_linear_velocity_body[1], dtype=np.float32),
            np.full(point_count, rigid_angular_velocity_body[2], dtype=np.float32),
            np.full(point_count, action[0], dtype=np.float32),
            np.full(point_count, action[1], dtype=np.float32),
            np.full(point_count, action[2], dtype=np.float32),
            np.full(point_count, action[3], dtype=np.float32),
            np.full(point_count, action[4], dtype=np.float32),
            np.full(point_count, action[5], dtype=np.float32),
            np.full(point_count, action[6], dtype=np.float32),
            np.asarray(point_friction, dtype=np.float32).reshape(point_count),
            np.asarray(active_contact_mask, dtype=np.float32).reshape(point_count),
        ]
    ).astype(np.float32)

    if dino is None or dino.dim == 0:
        return repeated
    if dino.features.shape[0] != point_count:
        raise ValueError(f"DINO point count mismatch: {dino.features.shape[0]} vs {point_count}")
    return np.concatenate(
        [
            repeated,
            np.asarray(dino.features, dtype=np.float32),
            np.asarray(dino.bottom_feature_copied_from_top, dtype=np.float32).reshape(point_count, 1),
        ],
        axis=1,
    ).astype(np.float32)


def build_future_action_features(
    *,
    quaternions_xyzw: np.ndarray,
    step_forces: np.ndarray,
    point_offset_local: np.ndarray,
    start_force_index: int,
    prediction_window_steps: int,
) -> np.ndarray:
    actions = np.zeros((int(prediction_window_steps), len(ACTION_FEATURE_SCHEMA)), dtype=np.float32)
    forces = np.asarray(step_forces, dtype=np.float32).reshape(-1, 3)
    quats = np.asarray(quaternions_xyzw, dtype=np.float32).reshape(-1, 4)
    for horizon_idx in range(int(prediction_window_steps)):
        force_idx = min(max(int(start_force_index) + horizon_idx, 0), max(len(forces) - 1, 0))
        frame_idx = min(force_idx, len(quats) - 1)
        force = forces[force_idx] if len(forces) else np.zeros(3, dtype=np.float32)
        actions[horizon_idx] = action_features_from_force(
            quaternion_xyzw=quats[frame_idx],
            force_world=force,
            point_offset_local=point_offset_local,
        )
    return actions


def build_supervised_batch_tensors(
    *,
    trajectories: list,
    sim_positions: np.ndarray,
    sim_quaternions_xyzw: np.ndarray,
    sim_linear_velocity: np.ndarray,
    sim_angular_velocity: np.ndarray,
    local_surface_points: np.ndarray,
    box_half_extents: np.ndarray,
    point_friction: np.ndarray,
    active_contact_mask: np.ndarray,
    dino: DinoFeatures | None,
    history_window_steps: int,
    prediction_window_steps: int,
    residual_output_mode: str = "velocity",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    output_mode = normalize_residual_output_mode(residual_output_mode)
    batch_size = len(trajectories)
    point_count = len(local_surface_points)
    feature_dim = len(point_feature_schema(0 if dino is None else dino.dim))
    point_features = np.zeros((batch_size, int(history_window_steps), point_count, feature_dim), dtype=np.float32)
    future_actions = np.zeros((batch_size, int(prediction_window_steps), len(ACTION_FEATURE_SCHEMA)), dtype=np.float32)
    targets = np.zeros(
        (batch_size, int(prediction_window_steps), residual_output_dim(output_mode)),
        dtype=np.float32,
    )
    point_mask = np.ones((batch_size, point_count), dtype=bool)

    for batch_idx, trajectory in enumerate(trajectories):
        for history_idx in range(int(history_window_steps)):
            frame_idx = history_idx
            force_idx = min(frame_idx, max(len(trajectory.step_forces) - 1, 0))
            force = trajectory.step_forces[force_idx] if len(trajectory.step_forces) else np.zeros(3, dtype=np.float32)
            point_features[batch_idx, history_idx] = build_point_feature_frame(
                local_surface_points=local_surface_points,
                box_half_extents=box_half_extents,
                quaternion_xyzw=sim_quaternions_xyzw[batch_idx, frame_idx],
                linear_velocity_world=sim_linear_velocity[batch_idx, frame_idx],
                angular_velocity_world=sim_angular_velocity[batch_idx, frame_idx],
                force_world=force,
                point_offset_local=trajectory.force_point_offset_local,
                point_friction=point_friction,
                active_contact_mask=active_contact_mask,
                dino=dino,
            )

        anchor_frame = int(history_window_steps)
        action_start_frame = max(anchor_frame - 1, 0)
        future_actions[batch_idx] = build_future_action_features(
            quaternions_xyzw=sim_quaternions_xyzw[batch_idx],
            step_forces=trajectory.step_forces,
            point_offset_local=trajectory.force_point_offset_local,
            start_force_index=action_start_frame,
            prediction_window_steps=int(prediction_window_steps),
        )

        for horizon_idx in range(int(prediction_window_steps)):
            frame_idx = anchor_frame + horizon_idx
            timestep = float(getattr(trajectory, "timestep", 1.0))
            if timestep <= 0.0:
                raise ValueError(f"Trajectory timestep must be positive, got {timestep}")
            rotation = quaternion_xyzw_to_matrix(sim_quaternions_xyzw[batch_idx, frame_idx])
            pose_delta_world = (
                np.asarray(trajectory.positions[frame_idx], dtype=np.float32)
                - sim_positions[batch_idx, frame_idx]
            )
            pose_delta_body = rotation.T @ pose_delta_world
            yaw_delta = float(
                np.arctan2(
                    np.sin(
                        quaternion_xyzw_to_yaw(trajectory.quaternions_xyzw[frame_idx])
                        - quaternion_xyzw_to_yaw(sim_quaternions_xyzw[batch_idx, frame_idx])
                    ),
                    np.cos(
                        quaternion_xyzw_to_yaw(trajectory.quaternions_xyzw[frame_idx])
                        - quaternion_xyzw_to_yaw(sim_quaternions_xyzw[batch_idx, frame_idx])
                    ),
                )
            )
            linear_delta_world = (
                np.asarray(trajectory.linear_velocity[frame_idx], dtype=np.float32)
                - sim_linear_velocity[batch_idx, frame_idx]
            )
            linear_delta_body = rotation.T @ linear_delta_world
            omega_delta_z = (
                float(trajectory.angular_velocity[frame_idx, 2])
                - float(sim_angular_velocity[batch_idx, frame_idx, 2])
            )
            if output_mode == "acceleration":
                linear_delta_body = linear_delta_body / timestep
                omega_delta_z = omega_delta_z / timestep
            pose_target = [pose_delta_body[0], pose_delta_body[1], yaw_delta]
            velocity_target = [linear_delta_body[0], linear_delta_body[1], omega_delta_z]
            if output_mode == "pose":
                target_values = pose_target
            elif output_mode == "pose_velocity":
                target_values = pose_target + velocity_target
            else:
                target_values = velocity_target
            targets[batch_idx, horizon_idx] = np.asarray(target_values, dtype=np.float32)

    return point_features, point_mask, future_actions, targets


def _stack_trajectory_array_torch(
    trajectories: list,
    attribute: str,
    *,
    frame_count: int,
    device: torch.device | str,
) -> torch.Tensor:
    values = np.stack(
        [np.asarray(getattr(trajectory, attribute)[:frame_count], dtype=np.float32) for trajectory in trajectories],
        axis=0,
    )
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _stack_step_forces_torch(
    trajectories: list,
    *,
    step_count: int,
    device: torch.device | str,
) -> torch.Tensor:
    forces = np.zeros((len(trajectories), int(step_count), 3), dtype=np.float32)
    for batch_idx, trajectory in enumerate(trajectories):
        source = np.asarray(trajectory.step_forces, dtype=np.float32).reshape(-1, 3)
        used = min(len(source), int(step_count))
        if used > 0:
            forces[batch_idx, :used] = source[:used]
            if used < int(step_count):
                forces[batch_idx, used:] = source[used - 1]
    return torch.as_tensor(forces, dtype=torch.float32, device=device)


def _action_features_from_force_torch(
    *,
    rotation: torch.Tensor,
    force_world: torch.Tensor,
    point_offset_local: torch.Tensor,
) -> torch.Tensor:
    force_body = torch.einsum("...i,...ij->...j", force_world, rotation)
    offset = point_offset_local
    torque_z = offset[..., 0] * force_body[..., 1] - offset[..., 1] * force_body[..., 0]
    return torch.cat((force_body, offset, torque_z.unsqueeze(-1)), dim=-1)


def build_supervised_batch_tensors_torch(
    *,
    trajectories: list,
    sim_positions: np.ndarray | torch.Tensor,
    sim_quaternions_xyzw: np.ndarray | torch.Tensor,
    sim_linear_velocity: np.ndarray | torch.Tensor,
    sim_angular_velocity: np.ndarray | torch.Tensor,
    local_surface_points: np.ndarray,
    box_half_extents: np.ndarray,
    point_friction: np.ndarray,
    active_contact_mask: np.ndarray,
    dino: DinoFeatures | None,
    history_window_steps: int,
    prediction_window_steps: int,
    device: torch.device | str,
    residual_output_mode: str = "velocity",
) -> tuple[torch.Tensor, None, torch.Tensor, torch.Tensor]:
    output_mode = normalize_residual_output_mode(residual_output_mode)
    batch_size = len(trajectories)
    history_steps = int(history_window_steps)
    prediction_steps = int(prediction_window_steps)
    frame_count = history_steps + prediction_steps
    step_count = max(history_steps + prediction_steps - 1, 1)

    sim_pos = torch.as_tensor(sim_positions, dtype=torch.float32, device=device)
    sim_quat = torch.as_tensor(sim_quaternions_xyzw, dtype=torch.float32, device=device)
    sim_linear = torch.as_tensor(sim_linear_velocity, dtype=torch.float32, device=device)
    sim_angular = torch.as_tensor(sim_angular_velocity, dtype=torch.float32, device=device)
    gt_pos = _stack_trajectory_array_torch(trajectories, "positions", frame_count=frame_count, device=device)
    gt_quat = _stack_trajectory_array_torch(trajectories, "quaternions_xyzw", frame_count=frame_count, device=device)
    gt_linear = _stack_trajectory_array_torch(trajectories, "linear_velocity", frame_count=frame_count, device=device)
    gt_angular = _stack_trajectory_array_torch(trajectories, "angular_velocity", frame_count=frame_count, device=device)
    timesteps = torch.as_tensor(
        [float(getattr(trajectory, "timestep", 1.0)) for trajectory in trajectories],
        dtype=torch.float32,
        device=device,
    )
    if bool(torch.any(timesteps <= 0.0)):
        raise ValueError("All trajectory timesteps must be positive")
    step_forces = _stack_step_forces_torch(trajectories, step_count=step_count, device=device)
    point_offsets = torch.as_tensor(
        np.stack(
            [np.asarray(trajectory.force_point_offset_local, dtype=np.float32).reshape(3) for trajectory in trajectories],
            axis=0,
        ),
        dtype=torch.float32,
        device=device,
    )

    local_points = torch.as_tensor(local_surface_points, dtype=torch.float32, device=device).reshape(-1, 3)
    point_count = int(local_points.shape[0])
    half_extents = torch.as_tensor(box_half_extents, dtype=torch.float32, device=device).reshape(1, 3).clamp_min(1.0e-8)
    point_friction_t = torch.as_tensor(point_friction, dtype=torch.float32, device=device).reshape(1, 1, point_count, 1)
    active_mask_t = torch.as_tensor(active_contact_mask, dtype=torch.float32, device=device).reshape(1, 1, point_count, 1)

    history_quat = sim_quat[:, :history_steps]
    history_rotation = quaternion_xyzw_to_matrix_torch(history_quat)
    history_linear = sim_linear[:, :history_steps]
    history_angular = sim_angular[:, :history_steps]
    history_forces = step_forces[:, :history_steps]

    relative_world = torch.einsum("bhij,nj->bhni", history_rotation, local_points)
    point_velocity_world = history_linear[:, :, None, :] + torch.cross(
        history_angular[:, :, None, :].expand(-1, -1, point_count, -1),
        relative_world,
        dim=-1,
    )
    point_velocity_body = torch.einsum("bhni,bhij->bhnj", point_velocity_world, history_rotation)
    rigid_linear_body = torch.einsum("bhi,bhij->bhj", history_linear, history_rotation)
    rigid_angular_body = torch.einsum("bhi,bhij->bhj", history_angular, history_rotation)
    history_offsets = point_offsets[:, None, :].expand(batch_size, history_steps, 3)
    history_actions = _action_features_from_force_torch(
        rotation=history_rotation,
        force_world=history_forces,
        point_offset_local=history_offsets,
    )

    local_normalized = (local_points / half_extents).reshape(1, 1, point_count, 3)
    expanded_scalar_features = [
        rigid_linear_body[..., 0].reshape(batch_size, history_steps, 1, 1).expand(-1, -1, point_count, -1),
        rigid_linear_body[..., 1].reshape(batch_size, history_steps, 1, 1).expand(-1, -1, point_count, -1),
        rigid_angular_body[..., 2].reshape(batch_size, history_steps, 1, 1).expand(-1, -1, point_count, -1),
    ]
    for action_idx in range(len(ACTION_FEATURE_SCHEMA)):
        expanded_scalar_features.append(
            history_actions[..., action_idx].reshape(batch_size, history_steps, 1, 1).expand(-1, -1, point_count, -1)
        )

    feature_parts = [
        local_normalized.expand(batch_size, history_steps, -1, -1),
        point_velocity_body,
        *expanded_scalar_features,
        point_friction_t.expand(batch_size, history_steps, -1, -1),
        active_mask_t.expand(batch_size, history_steps, -1, -1),
    ]
    if dino is not None and dino.dim > 0:
        dino_features = torch.as_tensor(dino.features, dtype=torch.float32, device=device).reshape(1, 1, point_count, dino.dim)
        dino_bottom = torch.as_tensor(
            dino.bottom_feature_copied_from_top,
            dtype=torch.float32,
            device=device,
        ).reshape(1, 1, point_count, 1)
        feature_parts.extend(
            [
                dino_features.expand(batch_size, history_steps, -1, -1),
                dino_bottom.expand(batch_size, history_steps, -1, -1),
            ]
        )
    point_features = torch.cat(feature_parts, dim=-1).contiguous()

    future_force_start = max(history_steps - 1, 0)
    future_forces = step_forces[:, future_force_start : future_force_start + prediction_steps]
    future_action_quat = sim_quat[:, future_force_start : future_force_start + prediction_steps]
    future_action_rotation = quaternion_xyzw_to_matrix_torch(future_action_quat)
    future_offsets = point_offsets[:, None, :].expand(batch_size, prediction_steps, 3)
    future_actions = _action_features_from_force_torch(
        rotation=future_action_rotation,
        force_world=future_forces,
        point_offset_local=future_offsets,
    ).contiguous()

    target_quat = sim_quat[:, history_steps : history_steps + prediction_steps]
    target_rotation = quaternion_xyzw_to_matrix_torch(target_quat)
    target_pose_delta_world = (
        gt_pos[:, history_steps : history_steps + prediction_steps]
        - sim_pos[:, history_steps : history_steps + prediction_steps]
    )
    target_pose_delta_body = torch.einsum("bpi,bpij->bpj", target_pose_delta_world, target_rotation)
    target_yaw_delta = torch.atan2(
        torch.sin(
            quaternion_xyzw_to_yaw_torch(gt_quat[:, history_steps : history_steps + prediction_steps])
            - quaternion_xyzw_to_yaw_torch(target_quat)
        ),
        torch.cos(
            quaternion_xyzw_to_yaw_torch(gt_quat[:, history_steps : history_steps + prediction_steps])
            - quaternion_xyzw_to_yaw_torch(target_quat)
        ),
    )
    target_linear_delta_world = (
        gt_linear[:, history_steps : history_steps + prediction_steps]
        - sim_linear[:, history_steps : history_steps + prediction_steps]
    )
    target_linear_delta_body = torch.einsum("bpi,bpij->bpj", target_linear_delta_world, target_rotation)
    target_omega_delta_z = (
        gt_angular[:, history_steps : history_steps + prediction_steps, 2]
        - sim_angular[:, history_steps : history_steps + prediction_steps, 2]
    )
    if output_mode == "acceleration":
        target_linear_delta_body = target_linear_delta_body / timesteps.reshape(batch_size, 1, 1)
        target_omega_delta_z = target_omega_delta_z / timesteps.reshape(batch_size, 1)
    pose_targets = torch.stack(
        (target_pose_delta_body[..., 0], target_pose_delta_body[..., 1], target_yaw_delta),
        dim=-1,
    )
    velocity_targets = torch.stack(
        (target_linear_delta_body[..., 0], target_linear_delta_body[..., 1], target_omega_delta_z),
        dim=-1,
    )
    if output_mode == "pose":
        targets = pose_targets
    elif output_mode == "pose_velocity":
        targets = torch.cat((pose_targets, velocity_targets), dim=-1)
    else:
        targets = velocity_targets
    return point_features, None, future_actions, targets


def compute_feature_normalizer(point_features: np.ndarray, future_actions: np.ndarray) -> FeatureNormalizer:
    feature_mean = np.mean(point_features, axis=(0, 1, 2), dtype=np.float64).astype(np.float32)
    feature_std = np.std(point_features, axis=(0, 1, 2), dtype=np.float64).astype(np.float32)
    action_mean = np.mean(future_actions, axis=(0, 1), dtype=np.float64).astype(np.float32)
    action_std = np.std(future_actions, axis=(0, 1), dtype=np.float64).astype(np.float32)
    return FeatureNormalizer(
        point_feature_mean=feature_mean,
        point_feature_std=np.maximum(feature_std, 1.0e-6).astype(np.float32),
        action_mean=action_mean,
        action_std=np.maximum(action_std, 1.0e-6).astype(np.float32),
    )


def merge_feature_normalizers(features: list[np.ndarray], actions: list[np.ndarray]) -> FeatureNormalizer:
    if not features or not actions:
        raise ValueError("Cannot compute feature normalization without sample tensors")
    return compute_feature_normalizer(np.concatenate(features, axis=0), np.concatenate(actions, axis=0))


def apply_feature_normalizer(
    point_features: np.ndarray,
    future_actions: np.ndarray,
    normalizer: FeatureNormalizer,
) -> tuple[np.ndarray, np.ndarray]:
    normalized_points = (
        (np.asarray(point_features, dtype=np.float32) - normalizer.point_feature_mean.reshape(1, 1, 1, -1))
        / normalizer.point_feature_std.reshape(1, 1, 1, -1)
    )
    normalized_actions = (
        (np.asarray(future_actions, dtype=np.float32) - normalizer.action_mean.reshape(1, 1, -1))
        / normalizer.action_std.reshape(1, 1, -1)
    )
    return normalized_points.astype(np.float32), normalized_actions.astype(np.float32)


def normalizer_to_torch(normalizer: FeatureNormalizer, *, device: torch.device | str) -> TorchFeatureNormalizer:
    return TorchFeatureNormalizer(
        point_feature_mean=torch.as_tensor(normalizer.point_feature_mean, dtype=torch.float32, device=device),
        point_feature_std=torch.as_tensor(normalizer.point_feature_std, dtype=torch.float32, device=device),
        action_mean=torch.as_tensor(normalizer.action_mean, dtype=torch.float32, device=device),
        action_std=torch.as_tensor(normalizer.action_std, dtype=torch.float32, device=device),
    )


def apply_feature_normalizer_torch(
    point_features: torch.Tensor,
    future_actions: torch.Tensor,
    normalizer: FeatureNormalizer | TorchFeatureNormalizer,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(normalizer.point_feature_mean, torch.Tensor):
        point_mean = normalizer.point_feature_mean.to(device=point_features.device, dtype=point_features.dtype)
        point_std = normalizer.point_feature_std.to(device=point_features.device, dtype=point_features.dtype)
        action_mean = normalizer.action_mean.to(device=future_actions.device, dtype=future_actions.dtype)
        action_std = normalizer.action_std.to(device=future_actions.device, dtype=future_actions.dtype)
    else:
        point_mean = torch.as_tensor(normalizer.point_feature_mean, dtype=point_features.dtype, device=point_features.device)
        point_std = torch.as_tensor(normalizer.point_feature_std, dtype=point_features.dtype, device=point_features.device)
        action_mean = torch.as_tensor(normalizer.action_mean, dtype=future_actions.dtype, device=future_actions.device)
        action_std = torch.as_tensor(normalizer.action_std, dtype=future_actions.dtype, device=future_actions.device)

    point_features.sub_(point_mean.reshape(1, 1, 1, -1)).div_(point_std.reshape(1, 1, 1, -1))
    future_actions.sub_(action_mean.reshape(1, 1, -1)).div_(action_std.reshape(1, 1, -1))
    return point_features, future_actions


def normalizer_to_metadata(normalizer: FeatureNormalizer) -> dict[str, list[float]]:
    return {
        "feature_mean": normalizer.point_feature_mean.astype(float).tolist(),
        "feature_std": normalizer.point_feature_std.astype(float).tolist(),
        "action_mean": normalizer.action_mean.astype(float).tolist(),
        "action_std": normalizer.action_std.astype(float).tolist(),
    }


def normalizer_from_metadata(metadata: dict) -> FeatureNormalizer:
    return FeatureNormalizer(
        point_feature_mean=np.asarray(metadata["feature_mean"], dtype=np.float32),
        point_feature_std=np.asarray(metadata["feature_std"], dtype=np.float32),
        action_mean=np.asarray(metadata["action_mean"], dtype=np.float32),
        action_std=np.asarray(metadata["action_std"], dtype=np.float32),
    )
