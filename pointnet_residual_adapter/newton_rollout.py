from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import warp as wp

from fit_mujoco_contact_point_friction_kernels import (
    apply_batched_external_and_surface_point_forces_trajectory_kernel,
    compute_batched_contact_weighted_masses_kernel,
)
from fit_mujoco_contact_point_friction_runtime import (
    reset_scene_states,
    set_batched_box_initial_states_kernel,
)
from newton_surface_points_diff_demo import GRAVITY_MAGNITUDE

from .features import (
    ACTION_FEATURE_SCHEMA,
    DinoFeatures,
    _action_features_from_force_torch,
    normalize_residual_output_mode,
    normalizer_to_torch,
    quaternion_xyzw_to_matrix_torch,
    residual_output_components,
    residual_output_dim,
)
from .kernels import apply_planar_pose_velocity_residual_kernel, apply_planar_velocity_residual_kernel


@dataclass
class RolloutBuffers:
    batch_capacity: int
    step_capacity: int
    full_point_friction: wp.array
    contact_weighted_masses: wp.array
    contact_weighted_mass_total: wp.array
    step_forces: wp.array
    force_point_offsets_local: wp.array
    initial_positions: wp.array
    initial_quaternions: wp.array
    initial_linear_velocity: wp.array
    initial_angular_velocity: wp.array
    trajectory_step_counts: wp.array


@dataclass(frozen=True)
class RigidStateHistory:
    positions: np.ndarray
    quaternions_xyzw: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray


def build_rollout_buffers(
    *,
    device: str,
    batch_capacity: int,
    step_capacity: int,
    point_count: int,
    full_point_friction: np.ndarray,
) -> RolloutBuffers:
    return RolloutBuffers(
        batch_capacity=max(int(batch_capacity), 1),
        step_capacity=max(int(step_capacity), 1),
        full_point_friction=wp.array(np.asarray(full_point_friction, dtype=np.float32), dtype=wp.float32, device=device),
        contact_weighted_masses=wp.zeros(
            max(int(step_capacity), 1) * max(int(batch_capacity), 1) * int(point_count),
            dtype=wp.float32,
            device=device,
        ),
        contact_weighted_mass_total=wp.zeros(
            max(int(step_capacity), 1) * max(int(batch_capacity), 1),
            dtype=wp.float32,
            device=device,
        ),
        step_forces=wp.zeros(max(int(batch_capacity), 1) * max(int(step_capacity), 1), dtype=wp.vec3, device=device),
        force_point_offsets_local=wp.zeros(max(int(batch_capacity), 1), dtype=wp.vec3, device=device),
        initial_positions=wp.zeros(max(int(batch_capacity), 1), dtype=wp.vec3, device=device),
        initial_quaternions=wp.zeros(max(int(batch_capacity), 1), dtype=wp.quat, device=device),
        initial_linear_velocity=wp.zeros(max(int(batch_capacity), 1), dtype=wp.vec3, device=device),
        initial_angular_velocity=wp.zeros(max(int(batch_capacity), 1), dtype=wp.vec3, device=device),
        trajectory_step_counts=wp.zeros(max(int(batch_capacity), 1), dtype=wp.int32, device=device),
    )


def assign_rollout_trajectories(buffers: RolloutBuffers, trajectories: list) -> int:
    batch_size = len(trajectories)
    if batch_size > buffers.batch_capacity:
        raise ValueError(f"batch size {batch_size} exceeds buffer capacity {buffers.batch_capacity}")
    if any(trajectory.num_steps > buffers.step_capacity for trajectory in trajectories):
        max_steps = max(trajectory.num_steps for trajectory in trajectories)
        raise ValueError(f"trajectory steps {max_steps} exceed buffer step capacity {buffers.step_capacity}")

    step_forces = np.zeros((buffers.batch_capacity, buffers.step_capacity, 3), dtype=np.float32)
    point_offsets = np.zeros((buffers.batch_capacity, 3), dtype=np.float32)
    initial_positions = np.zeros((buffers.batch_capacity, 3), dtype=np.float32)
    initial_quaternions = np.zeros((buffers.batch_capacity, 4), dtype=np.float32)
    initial_linear_velocity = np.zeros((buffers.batch_capacity, 3), dtype=np.float32)
    initial_angular_velocity = np.zeros((buffers.batch_capacity, 3), dtype=np.float32)
    step_counts = np.zeros((buffers.batch_capacity,), dtype=np.int32)

    for batch_idx, trajectory in enumerate(trajectories):
        used_steps = min(trajectory.num_steps, buffers.step_capacity)
        if used_steps > 0:
            step_forces[batch_idx, :used_steps] = np.asarray(trajectory.step_forces[:used_steps], dtype=np.float32)
            if used_steps < buffers.step_capacity:
                step_forces[batch_idx, used_steps:] = step_forces[batch_idx, used_steps - 1]
        point_offsets[batch_idx] = np.asarray(trajectory.force_point_offset_local, dtype=np.float32).reshape(3)
        initial_positions[batch_idx] = np.asarray(trajectory.positions[0], dtype=np.float32)
        initial_quaternions[batch_idx] = np.asarray(trajectory.quaternions_xyzw[0], dtype=np.float32)
        initial_linear_velocity[batch_idx] = np.asarray(trajectory.linear_velocity[0], dtype=np.float32)
        initial_angular_velocity[batch_idx] = np.asarray(trajectory.angular_velocity[0], dtype=np.float32)
        step_counts[batch_idx] = int(used_steps)

    buffers.step_forces.assign(step_forces.reshape(-1, 3))
    buffers.force_point_offsets_local.assign(point_offsets)
    buffers.initial_positions.assign(initial_positions)
    buffers.initial_quaternions.assign(initial_quaternions)
    buffers.initial_linear_velocity.assign(initial_linear_velocity)
    buffers.initial_angular_velocity.assign(initial_angular_velocity)
    buffers.trajectory_step_counts.assign(step_counts)
    return batch_size


def extract_state_history(diff_scene, *, batch_size: int, frame_count: int) -> RigidStateHistory:
    device = diff_scene.torch_device
    box_ids = _box_id_tensor(diff_scene, batch_size=batch_size, device=device)
    positions = torch.empty((batch_size, frame_count, 3), dtype=torch.float32, device=device)
    quaternions = torch.empty((batch_size, frame_count, 4), dtype=torch.float32, device=device)
    linear_velocity = torch.empty((batch_size, frame_count, 3), dtype=torch.float32, device=device)
    angular_velocity = torch.empty((batch_size, frame_count, 3), dtype=torch.float32, device=device)
    for frame_idx in range(frame_count):
        pos, quat, lin, ang = _state_frame_tensors(diff_scene, box_ids=box_ids, frame_idx=frame_idx)
        positions[:, frame_idx].copy_(pos)
        quaternions[:, frame_idx].copy_(quat)
        linear_velocity[:, frame_idx].copy_(lin)
        angular_velocity[:, frame_idx].copy_(ang)
    return _history_from_tensors(positions, quaternions, linear_velocity, angular_velocity)


def _box_id_tensor(diff_scene, *, batch_size: int, device: torch.device | str) -> torch.Tensor:
    return torch.as_tensor(diff_scene.box_body_ids_np[:batch_size], dtype=torch.long, device=device)


def _state_frame_tensors(
    diff_scene,
    *,
    box_ids: torch.Tensor,
    frame_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    body_q = wp.to_torch(diff_scene.states[frame_idx].body_q).detach().index_select(0, box_ids)
    body_qd = wp.to_torch(diff_scene.states[frame_idx].body_qd).detach().index_select(0, box_ids)
    return body_q[:, :3], body_q[:, 3:7], body_qd[:, :3], body_qd[:, 3:6]


def _history_from_tensors(
    positions: torch.Tensor,
    quaternions: torch.Tensor,
    linear_velocity: torch.Tensor,
    angular_velocity: torch.Tensor,
) -> RigidStateHistory:
    return RigidStateHistory(
        positions=positions.detach().cpu().numpy().astype(np.float32, copy=False),
        quaternions_xyzw=quaternions.detach().cpu().numpy().astype(np.float32, copy=False),
        linear_velocity=linear_velocity.detach().cpu().numpy().astype(np.float32, copy=False),
        angular_velocity=angular_velocity.detach().cpu().numpy().astype(np.float32, copy=False),
    )


def _build_point_feature_frame_torch(
    *,
    local_surface_points: torch.Tensor,
    box_half_extents: torch.Tensor,
    quaternion_xyzw: torch.Tensor,
    linear_velocity_world: torch.Tensor,
    angular_velocity_world: torch.Tensor,
    force_world: torch.Tensor,
    point_offset_local: torch.Tensor,
    point_friction: torch.Tensor,
    active_contact_mask: torch.Tensor,
    dino_features: torch.Tensor | None,
    dino_bottom_feature_copied_from_top: torch.Tensor | None,
) -> torch.Tensor:
    batch_size = int(quaternion_xyzw.shape[0])
    point_count = int(local_surface_points.shape[0])
    rotation = quaternion_xyzw_to_matrix_torch(quaternion_xyzw)

    relative_world = torch.einsum("bij,nj->bni", rotation, local_surface_points)
    point_velocity_world = linear_velocity_world[:, None, :] + torch.cross(
        angular_velocity_world[:, None, :].expand(-1, point_count, -1),
        relative_world,
        dim=-1,
    )
    point_velocity_body = torch.einsum("bni,bij->bnj", point_velocity_world, rotation)
    rigid_linear_velocity_body = torch.einsum("bi,bij->bj", linear_velocity_world, rotation)
    rigid_angular_velocity_body = torch.einsum("bi,bij->bj", angular_velocity_world, rotation)
    action = _action_features_from_force_torch(
        rotation=rotation,
        force_world=force_world,
        point_offset_local=point_offset_local,
    )

    scalar_features = [
        rigid_linear_velocity_body[:, 0].reshape(batch_size, 1, 1).expand(-1, point_count, -1),
        rigid_linear_velocity_body[:, 1].reshape(batch_size, 1, 1).expand(-1, point_count, -1),
        rigid_angular_velocity_body[:, 2].reshape(batch_size, 1, 1).expand(-1, point_count, -1),
    ]
    for action_idx in range(len(ACTION_FEATURE_SCHEMA)):
        scalar_features.append(action[:, action_idx].reshape(batch_size, 1, 1).expand(-1, point_count, -1))

    feature_parts = [
        (local_surface_points / box_half_extents).reshape(1, point_count, 3).expand(batch_size, -1, -1),
        point_velocity_body,
        *scalar_features,
        point_friction.reshape(1, point_count, 1).expand(batch_size, -1, -1),
        active_contact_mask.reshape(1, point_count, 1).expand(batch_size, -1, -1),
    ]
    if dino_features is not None and dino_bottom_feature_copied_from_top is not None:
        feature_parts.extend(
            [
                dino_features.reshape(1, point_count, -1).expand(batch_size, -1, -1),
                dino_bottom_feature_copied_from_top.reshape(1, point_count, 1).expand(batch_size, -1, -1),
            ]
        )
    return torch.cat(feature_parts, dim=-1).contiguous()


def _future_action_features_torch(
    *,
    quaternion_xyzw: torch.Tensor,
    step_forces: torch.Tensor,
    trajectory_step_counts: torch.Tensor,
    point_offset_local: torch.Tensor,
    step_idx: int,
    prediction_window_steps: int,
) -> torch.Tensor:
    batch_size = int(step_forces.shape[0])
    prediction_steps = int(prediction_window_steps)
    if prediction_steps < 1:
        raise ValueError("prediction_window_steps must be positive")

    horizon = torch.arange(prediction_steps, dtype=torch.long, device=step_forces.device).reshape(1, prediction_steps)
    raw_indices = int(step_idx) + horizon
    max_valid = (trajectory_step_counts - 1).clamp_min(0).reshape(batch_size, 1)
    force_indices = torch.minimum(raw_indices.expand(batch_size, -1), max_valid)
    gathered_forces = torch.gather(step_forces, 1, force_indices.unsqueeze(-1).expand(-1, -1, 3))
    has_future_slice = (int(step_idx) < trajectory_step_counts).reshape(batch_size, 1, 1)
    gathered_forces = torch.where(has_future_slice, gathered_forces, torch.zeros_like(gathered_forces))

    rotation = quaternion_xyzw_to_matrix_torch(quaternion_xyzw).unsqueeze(1).expand(-1, prediction_steps, -1, -1)
    offsets = point_offset_local[:, None, :].expand(batch_size, prediction_steps, 3)
    return _action_features_from_force_torch(
        rotation=rotation,
        force_world=gathered_forces,
        point_offset_local=offsets,
    ).contiguous()


def _normalize_pointnet_inputs(
    *,
    point_features: torch.Tensor,
    future_actions: torch.Tensor,
    point_feature_mean: torch.Tensor,
    point_feature_std: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized_points = (
        point_features - point_feature_mean.reshape(1, 1, 1, -1).to(dtype=point_features.dtype)
    ) / point_feature_std.reshape(1, 1, 1, -1).to(dtype=point_features.dtype)
    normalized_actions = (
        future_actions - action_mean.reshape(1, 1, -1).to(dtype=future_actions.dtype)
    ) / action_std.reshape(1, 1, -1).to(dtype=future_actions.dtype)
    return normalized_points.contiguous(), normalized_actions.contiguous()


def _simulate_step(diff_scene, buffers: RolloutBuffers, args, *, step_idx: int, batch_size: int, point_count: int) -> None:
    state_in = diff_scene.states[step_idx]
    state_out = diff_scene.states[step_idx + 1]
    state_in.clear_forces()
    wp.launch(
        compute_batched_contact_weighted_masses_kernel,
        dim=batch_size * point_count,
        inputs=[
            step_idx,
            diff_scene.box_body_ids_wp,
            state_in.body_q,
            diff_scene.local_surface_points_wp,
            diff_scene.point_masses_wp,
            batch_size,
            point_count,
            float(diff_scene.floor_top_z),
            float(args.friction_contact_threshold),
            buffers.contact_weighted_masses,
            buffers.contact_weighted_mass_total,
        ],
        device=diff_scene.model.device,
    )
    wp.launch(
        apply_batched_external_and_surface_point_forces_trajectory_kernel,
        dim=batch_size * point_count,
        inputs=[
            step_idx,
            diff_scene.box_body_ids_wp,
            state_in.body_q,
            state_in.body_qd,
            diff_scene.model.body_com,
            diff_scene.local_surface_points_wp,
            buffers.contact_weighted_masses,
            buffers.contact_weighted_mass_total,
            buffers.full_point_friction,
            buffers.step_forces,
            buffers.force_point_offsets_local,
            buffers.trajectory_step_counts,
            batch_size,
            point_count,
            buffers.step_capacity,
            float(diff_scene.box_mass),
            float(GRAVITY_MAGNITUDE),
            float(diff_scene.floor_top_z),
            float(args.contact_stiffness),
            float(args.contact_damping),
            float(args.friction_contact_threshold),
            float(args.friction_regularization),
            state_in.body_f,
        ],
        device=diff_scene.model.device,
    )
    diff_scene.collision_pipeline.collide(state_in, diff_scene.contacts)
    diff_scene.solver.step(state_in, state_out, diff_scene.control, diff_scene.contacts, float(args.dt))


def run_open_loop_rollout(
    *,
    diff_scene,
    buffers: RolloutBuffers,
    trajectories: list,
    args,
    initial_body_q: np.ndarray,
    initial_body_qd: np.ndarray,
) -> RigidStateHistory:
    point_count = len(diff_scene.local_surface_points_np)
    batch_size = assign_rollout_trajectories(buffers, trajectories)
    reset_scene_states(diff_scene, initial_body_q, initial_body_qd)
    buffers.contact_weighted_masses.zero_()
    buffers.contact_weighted_mass_total.zero_()
    wp.launch(
        set_batched_box_initial_states_kernel,
        dim=batch_size,
        inputs=[
            diff_scene.box_body_ids_wp,
            buffers.initial_positions,
            buffers.initial_quaternions,
            buffers.initial_linear_velocity,
            buffers.initial_angular_velocity,
            diff_scene.states[0].body_q,
            diff_scene.states[0].body_qd,
        ],
        device=diff_scene.model.device,
    )
    max_steps = max(trajectory.num_steps for trajectory in trajectories)
    for step_idx in range(max_steps):
        _simulate_step(diff_scene, buffers, args, step_idx=step_idx, batch_size=batch_size, point_count=point_count)
    return extract_state_history(diff_scene, batch_size=batch_size, frame_count=max_steps + 1)


def run_closed_loop_pointnet_rollout(
    *,
    diff_scene,
    buffers: RolloutBuffers,
    trajectory,
    args,
    initial_body_q: np.ndarray,
    initial_body_qd: np.ndarray,
    model,
    normalizer,
    local_surface_points: np.ndarray,
    box_half_extents: np.ndarray,
    point_friction: np.ndarray,
    active_contact_mask: np.ndarray,
    dino: DinoFeatures | None,
    torch_device,
) -> tuple[RigidStateHistory, np.ndarray]:
    history, residuals = run_closed_loop_pointnet_rollout_batch(
        diff_scene=diff_scene,
        buffers=buffers,
        trajectories=[trajectory],
        args=args,
        initial_body_q=initial_body_q,
        initial_body_qd=initial_body_qd,
        model=model,
        normalizer=normalizer,
        local_surface_points=local_surface_points,
        box_half_extents=box_half_extents,
        point_friction=point_friction,
        active_contact_mask=active_contact_mask,
        dino=dino,
        torch_device=torch_device,
    )
    return history, residuals[0]


def run_closed_loop_pointnet_rollout_batch(
    *,
    diff_scene,
    buffers: RolloutBuffers,
    trajectories: list,
    args,
    initial_body_q: np.ndarray,
    initial_body_qd: np.ndarray,
    model,
    normalizer,
    local_surface_points: np.ndarray,
    box_half_extents: np.ndarray,
    point_friction: np.ndarray,
    active_contact_mask: np.ndarray,
    dino: DinoFeatures | None,
    torch_device,
) -> tuple[RigidStateHistory, np.ndarray]:
    torch_device = torch.device(torch_device)
    point_count = len(local_surface_points)
    batch_size = assign_rollout_trajectories(buffers, trajectories)
    reset_scene_states(diff_scene, initial_body_q, initial_body_qd)
    buffers.contact_weighted_masses.zero_()
    buffers.contact_weighted_mass_total.zero_()
    wp.launch(
        set_batched_box_initial_states_kernel,
        dim=batch_size,
        inputs=[
            diff_scene.box_body_ids_wp,
            buffers.initial_positions,
            buffers.initial_quaternions,
            buffers.initial_linear_velocity,
            buffers.initial_angular_velocity,
            diff_scene.states[0].body_q,
            diff_scene.states[0].body_qd,
        ],
        device=diff_scene.model.device,
    )

    max_steps = max(trajectory.num_steps for trajectory in trajectories)
    positions = torch.empty((batch_size, max_steps + 1, 3), dtype=torch.float32, device=torch_device)
    quaternions = torch.empty((batch_size, max_steps + 1, 4), dtype=torch.float32, device=torch_device)
    linear_velocity = torch.empty((batch_size, max_steps + 1, 3), dtype=torch.float32, device=torch_device)
    angular_velocity = torch.empty((batch_size, max_steps + 1, 3), dtype=torch.float32, device=torch_device)
    configured_output_mode = normalize_residual_output_mode(
        getattr(args, "pointnet_residual_output_mode", getattr(args, "residual_output_mode", "velocity"))
    )
    residual_dim = residual_output_dim(configured_output_mode)
    applied_residuals = torch.empty((batch_size, max_steps, residual_dim), dtype=torch.float32, device=torch_device)

    box_ids = _box_id_tensor(diff_scene, batch_size=batch_size, device=torch_device)
    (
        positions[:, 0],
        quaternions[:, 0],
        linear_velocity[:, 0],
        angular_velocity[:, 0],
    ) = _state_frame_tensors(diff_scene, box_ids=box_ids, frame_idx=0)

    history_steps = int(args.history_window_steps)
    prediction_steps = int(args.prediction_window_steps)
    if history_steps < 1:
        raise ValueError("history_window_steps must be positive")

    local_points = torch.as_tensor(local_surface_points, dtype=torch.float32, device=torch_device).reshape(-1, 3)
    half_extents = torch.as_tensor(box_half_extents, dtype=torch.float32, device=torch_device).reshape(1, 3).clamp_min(1.0e-8)
    point_friction_t = torch.as_tensor(point_friction, dtype=torch.float32, device=torch_device).reshape(point_count)
    active_mask_t = torch.as_tensor(active_contact_mask, dtype=torch.float32, device=torch_device).reshape(point_count)
    if dino is not None and dino.dim > 0:
        dino_features_t = torch.as_tensor(dino.features, dtype=torch.float32, device=torch_device).reshape(point_count, dino.dim)
        dino_bottom_t = torch.as_tensor(
            dino.bottom_feature_copied_from_top,
            dtype=torch.float32,
            device=torch_device,
        ).reshape(point_count)
    else:
        dino_features_t = None
        dino_bottom_t = None

    step_forces = wp.to_torch(buffers.step_forces).detach().reshape(buffers.batch_capacity, buffers.step_capacity, 3)[
        :batch_size
    ]
    point_offsets = wp.to_torch(buffers.force_point_offsets_local).detach()[:batch_size]
    trajectory_step_counts = wp.to_torch(buffers.trajectory_step_counts).detach().to(dtype=torch.long)[:batch_size]
    normalizer_t = normalizer_to_torch(normalizer, device=torch_device)
    history_buffer: torch.Tensor | None = None
    is_stateful = bool(getattr(model, "is_stateful_residual_adapter", False))
    stateful_hidden: torch.Tensor | None = None
    if is_stateful:
        stateful_hidden = model.initial_state(batch_size, device=torch_device, dtype=torch.float32)
        stateful_hidden_norms = torch.empty((batch_size, max_steps), dtype=torch.float32, device=torch_device)
        stateful_hidden_saturation = torch.empty((batch_size, max_steps), dtype=torch.float32, device=torch_device)
    else:
        stateful_hidden_norms = None
        stateful_hidden_saturation = None
    stateful_reset_interval = int(getattr(args, "stateful_reset_interval", 0) or 0)
    if stateful_reset_interval < 0:
        raise ValueError("stateful_reset_interval must be non-negative")

    model.eval()
    with torch.no_grad():
        residual_gain = float(getattr(args, "pointnet_residual_gain", 1.0))
        residual_output_mode = configured_output_mode
        residual_dt = float(getattr(args, "dt", 0.0))
        if residual_output_mode == "acceleration" and residual_dt <= 0.0:
            raise ValueError("args.dt must be positive when pointnet_residual_output_mode='acceleration'")
        for step_idx in range(max_steps):
            if is_stateful and stateful_reset_interval > 0 and step_idx > 0 and step_idx % stateful_reset_interval == 0:
                stateful_hidden = model.initial_state(batch_size, device=torch_device, dtype=torch.float32)
            frame_idx = step_idx
            pos, quat, lin, ang = _state_frame_tensors(diff_scene, box_ids=box_ids, frame_idx=frame_idx)
            force_idx = min(step_idx, buffers.step_capacity - 1)
            frame_force = step_forces[:, force_idx]

            frame_features = _build_point_feature_frame_torch(
                local_surface_points=local_points,
                box_half_extents=half_extents,
                quaternion_xyzw=quat,
                linear_velocity_world=lin,
                angular_velocity_world=ang,
                force_world=frame_force,
                point_offset_local=point_offsets,
                point_friction=point_friction_t,
                active_contact_mask=active_mask_t,
                dino_features=dino_features_t,
                dino_bottom_feature_copied_from_top=dino_bottom_t,
            )
            future_actions = _future_action_features_torch(
                quaternion_xyzw=quat,
                step_forces=step_forces,
                trajectory_step_counts=trajectory_step_counts,
                point_offset_local=point_offsets,
                step_idx=step_idx,
                prediction_window_steps=prediction_steps,
            )
            if is_stateful:
                model_point_features = frame_features[:, None]
            else:
                if history_buffer is None:
                    history_buffer = frame_features[:, None, :, :].expand(-1, history_steps, -1, -1).clone()
                else:
                    history_buffer[:, :-1].copy_(history_buffer[:, 1:].clone())
                    history_buffer[:, -1].copy_(frame_features)
                model_point_features = history_buffer

            point_features, future_actions = _normalize_pointnet_inputs(
                point_features=model_point_features,
                future_actions=future_actions,
                point_feature_mean=normalizer_t.point_feature_mean,
                point_feature_std=normalizer_t.point_feature_std,
                action_mean=normalizer_t.action_mean,
                action_std=normalizer_t.action_std,
            )
            if is_stateful:
                residual_sequence, stateful_hidden = model.forward_step(
                    point_features[:, 0],
                    None,
                    future_actions,
                    stateful_hidden,
                )
                residual = residual_sequence[:, 0]
                stateful_hidden_norms[:, step_idx].copy_(torch.linalg.vector_norm(stateful_hidden, dim=(0, 2)))
                stateful_hidden_saturation[:, step_idx].copy_(
                    (stateful_hidden.abs() >= 0.95).to(dtype=torch.float32).mean(dim=(0, 2))
                )
            else:
                residual = model(point_features, None, future_actions)[:, 0]
            if residual_output_mode == "acceleration":
                residual = residual * residual_dt
            residual = (residual * residual_gain).contiguous()
            applied_residuals[:, step_idx].copy_(residual)
            _simulate_step(diff_scene, buffers, args, step_idx=step_idx, batch_size=batch_size, point_count=point_count)
            next_frame_idx = step_idx + 1
            if residual_output_mode in {"velocity", "acceleration"}:
                residual_wp = wp.from_torch(residual, dtype=wp.vec3)
                wp.launch(
                    apply_planar_velocity_residual_kernel,
                    dim=batch_size,
                    inputs=[
                        diff_scene.box_body_ids_wp,
                        residual_wp,
                        diff_scene.states[next_frame_idx].body_q,
                        diff_scene.states[next_frame_idx].body_qd,
                    ],
                    device=diff_scene.model.device,
                )
            else:
                has_pose, has_velocity = residual_output_components(residual_output_mode)
                residual_wp = wp.from_torch(residual.reshape(-1), dtype=wp.float32)
                wp.launch(
                    apply_planar_pose_velocity_residual_kernel,
                    dim=batch_size,
                    inputs=[
                        diff_scene.box_body_ids_wp,
                        residual_wp,
                        int(residual.shape[-1]),
                        1 if has_pose else 0,
                        1 if has_velocity else 0,
                        diff_scene.states[next_frame_idx].body_q,
                        diff_scene.states[next_frame_idx].body_qd,
                    ],
                    device=diff_scene.model.device,
                )
            pos, quat, lin, ang = _state_frame_tensors(diff_scene, box_ids=box_ids, frame_idx=next_frame_idx)
            positions[:, next_frame_idx].copy_(pos)
            quaternions[:, next_frame_idx].copy_(quat)
            linear_velocity[:, next_frame_idx].copy_(lin)
            angular_velocity[:, next_frame_idx].copy_(ang)

    if is_stateful:
        model.last_stateful_rollout_diagnostics = {
            "hidden_l2_norm": stateful_hidden_norms.detach().cpu().numpy().astype(np.float32, copy=False),
            "hidden_saturation_fraction": stateful_hidden_saturation.detach().cpu().numpy().astype(np.float32, copy=False),
            "stateful_reset_interval": int(stateful_reset_interval),
        }
    else:
        model.last_stateful_rollout_diagnostics = None
    return _history_from_tensors(positions, quaternions, linear_velocity, angular_velocity), (
        applied_residuals.detach().cpu().numpy().astype(np.float32, copy=False)
    )
