from __future__ import annotations

import argparse

import numpy as np
import warp as wp

from mujoco_contact_friction_fit_utils import BatchedOptimizationBuffers, MujocoTrajectory
from newton_surface_points_diff_demo import (
    DiffScene,
    GRAVITY_MAGNITUDE,
)


def reset_scene_states(diff_scene: DiffScene, initial_body_q: np.ndarray, initial_body_qd: np.ndarray) -> None:
    for state in diff_scene.states:
        state.body_q.assign(initial_body_q)
        if getattr(state, "body_q_prev", None) is not None:
            state.body_q_prev.assign(initial_body_q)
        state.body_qd.assign(initial_body_qd)
        if getattr(state, "body_qdd", None) is not None:
            state.body_qdd.zero_()
        state.body_f.zero_()
        if getattr(state, "body_parent_f", None) is not None:
            state.body_parent_f.zero_()


@wp.kernel
def set_batched_box_initial_states_kernel(
    box_body_ids: wp.array(dtype=wp.int32),
    initial_positions: wp.array(dtype=wp.vec3),
    initial_quaternions: wp.array(dtype=wp.quat),
    initial_linear_velocity: wp.array(dtype=wp.vec3),
    initial_angular_velocity: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    batch_idx = wp.tid()
    body_id = box_body_ids[batch_idx]
    body_q[body_id] = wp.transform(initial_positions[batch_idx], initial_quaternions[batch_idx])
    body_qd[body_id] = wp.spatial_vector(initial_linear_velocity[batch_idx], initial_angular_velocity[batch_idx])


def resolve_point_position_loss_scale(args: argparse.Namespace, point_count: int) -> float:
    reduction = getattr(args, "point_position_loss_reduction", "sum")
    if reduction == "sum":
        return 1.0
    if reduction == "mean":
        return 1.0 / max(int(point_count), 1)
    raise ValueError(f"Unsupported --point-position-loss-reduction: {reduction!r}")


def _pad_vec3_rows(values: np.ndarray, length: int) -> np.ndarray:
    padded = np.zeros((length, 3), dtype=np.float32)
    if len(values) > 0:
        used = min(len(values), length)
        padded[:used] = np.asarray(values[:used], dtype=np.float32)
        if used < length:
            padded[used:] = padded[used - 1]
    return padded


def _pad_vec4_rows(values: np.ndarray, length: int) -> np.ndarray:
    padded = np.zeros((length, 4), dtype=np.float32)
    if len(values) > 0:
        used = min(len(values), length)
        padded[:used] = np.asarray(values[:used], dtype=np.float32)
        if used < length:
            padded[used:] = padded[used - 1]
    return padded


def build_batched_optimization_buffers(
    diff_scene: DiffScene,
    trajectories: list[MujocoTrajectory],
    args: argparse.Namespace,
    active_indices: np.ndarray,
) -> BatchedOptimizationBuffers:
    device = str(diff_scene.torch_device)
    batch_size = len(trajectories)
    point_count = len(diff_scene.local_surface_points_np)
    max_steps = max((trajectory.num_steps for trajectory in trajectories), default=0)
    max_frames = max_steps + 1

    base_point_friction = np.full(point_count, float(args.point_friction), dtype=np.float32)
    active_point_friction = np.full(len(active_indices), float(args.point_friction), dtype=np.float32)
    step_counts = np.asarray([trajectory.num_steps for trajectory in trajectories], dtype=np.int32)
    frame_scales = np.asarray(
        [1.0 / max(trajectory.num_frames, 1) for trajectory in trajectories],
        dtype=np.float32,
    )

    step_forces = np.zeros((batch_size, max(max_steps, 1), 3), dtype=np.float32)
    force_point_offsets_local = np.zeros((batch_size, 3), dtype=np.float32)
    target_positions = np.zeros((batch_size, max(max_frames, 1), 3), dtype=np.float32)
    target_quaternions = np.zeros((batch_size, max(max_frames, 1), 4), dtype=np.float32)
    target_linear_velocity = np.zeros((batch_size, max(max_frames, 1), 3), dtype=np.float32)
    target_angular_velocity = np.zeros((batch_size, max(max_frames, 1), 3), dtype=np.float32)
    initial_positions = np.zeros((batch_size, 3), dtype=np.float32)
    initial_quaternions = np.zeros((batch_size, 4), dtype=np.float32)
    initial_linear_velocity = np.zeros((batch_size, 3), dtype=np.float32)
    initial_angular_velocity = np.zeros((batch_size, 3), dtype=np.float32)

    for batch_idx, trajectory in enumerate(trajectories):
        step_forces[batch_idx] = _pad_vec3_rows(trajectory.step_forces, max(max_steps, 1))
        force_point_offsets_local[batch_idx] = np.asarray(trajectory.force_point_offset_local, dtype=np.float32).reshape(3)
        target_positions[batch_idx] = _pad_vec3_rows(trajectory.positions, max(max_frames, 1))
        target_quaternions[batch_idx] = _pad_vec4_rows(trajectory.quaternions_xyzw, max(max_frames, 1))
        target_linear_velocity[batch_idx] = _pad_vec3_rows(trajectory.linear_velocity, max(max_frames, 1))
        target_angular_velocity[batch_idx] = _pad_vec3_rows(trajectory.angular_velocity, max(max_frames, 1))
        initial_positions[batch_idx] = target_positions[batch_idx, 0]
        initial_quaternions[batch_idx] = target_quaternions[batch_idx, 0]
        initial_linear_velocity[batch_idx] = target_linear_velocity[batch_idx, 0]
        initial_angular_velocity[batch_idx] = target_angular_velocity[batch_idx, 0]

    contact_step_capacity = max(max_steps, 1)

    return BatchedOptimizationBuffers(
        batch_size=batch_size,
        max_steps=max_steps,
        max_frames=max_frames,
        active_point_friction=wp.array(active_point_friction, dtype=wp.float32, device=device, requires_grad=True),
        active_indices=wp.array(active_indices, dtype=wp.int32, device=device),
        full_point_friction=wp.array(base_point_friction, dtype=wp.float32, device=device, requires_grad=True),
        contact_weighted_masses=wp.zeros(contact_step_capacity * batch_size * point_count, dtype=wp.float32, device=device),
        contact_weighted_mass_total=wp.zeros(contact_step_capacity * batch_size, dtype=wp.float32, device=device),
        step_forces=wp.array(step_forces.reshape(-1, 3), dtype=wp.vec3, device=device),
        force_point_offsets_local=wp.array(force_point_offsets_local, dtype=wp.vec3, device=device),
        initial_positions=wp.array(initial_positions, dtype=wp.vec3, device=device),
        initial_quaternions=wp.array(initial_quaternions, dtype=wp.quat, device=device),
        initial_linear_velocity=wp.array(initial_linear_velocity, dtype=wp.vec3, device=device),
        initial_angular_velocity=wp.array(initial_angular_velocity, dtype=wp.vec3, device=device),
        target_positions=wp.array(target_positions.reshape(-1, 3), dtype=wp.vec3, device=device),
        target_quaternions=wp.array(target_quaternions.reshape(-1, 4), dtype=wp.vec4, device=device),
        target_linear_velocity=wp.array(target_linear_velocity.reshape(-1, 3), dtype=wp.vec3, device=device),
        target_angular_velocity=wp.array(target_angular_velocity.reshape(-1, 3), dtype=wp.vec3, device=device),
        trajectory_step_counts=wp.array(step_counts, dtype=wp.int32, device=device),
        frame_scales=wp.array(frame_scales, dtype=wp.float32, device=device),
        loss=wp.zeros(batch_size, dtype=wp.float32, device=device, requires_grad=True),
        position_loss=wp.zeros(batch_size, dtype=wp.float32, device=device, requires_grad=True),
        orientation_loss=wp.zeros(batch_size, dtype=wp.float32, device=device, requires_grad=True),
        linear_velocity_loss=wp.zeros(batch_size, dtype=wp.float32, device=device, requires_grad=True),
        angular_velocity_loss=wp.zeros(batch_size, dtype=wp.float32, device=device, requires_grad=True),
        batch_loss=wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True),
        inactive_point_friction_np=base_point_friction,
    )


def log_message(message: str) -> None:
    print(message, flush=True)


def resolve_batch_size(requested_batch_size: int | None, total_trajectories: int, default_batch_size: int) -> int:
    if total_trajectories <= 0:
        return 0
    if requested_batch_size is None:
        return min(total_trajectories, max(default_batch_size, 1))
    if int(requested_batch_size) <= 0:
        return total_trajectories
    return min(int(requested_batch_size), total_trajectories)


def should_log_trajectory_progress(completed: int, total: int, stride: int) -> bool:
    if total <= 0:
        return False
    if completed == total:
        return True
    if stride <= 0:
        return False
    return completed % stride == 0


def describe_nonfinite_array(name: str, array: np.ndarray) -> str:
    values = np.asarray(array)
    bad_indices = np.argwhere(~np.isfinite(values))
    if bad_indices.size == 0:
        return f"{name} is finite"

    first_bad_index = tuple(int(item) for item in bad_indices[0])
    try:
        first_bad_value = values[first_bad_index]
    except Exception:
        first_bad_value = values.reshape(-1)[0]
    return (
        f"{name} has non-finite values; first_bad_index={first_bad_index} "
        f"first_bad_value={first_bad_value!r} shape={values.shape} dtype={values.dtype}"
    )


def assert_array_finite(name: str, array: np.ndarray, *, context: str) -> None:
    values = np.asarray(array)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError(f"{context}: {describe_nonfinite_array(name, values)}")


def sample_training_batch_indices(
    total_trajectories: int,
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if batch_size >= total_trajectories:
        return np.arange(total_trajectories, dtype=np.int32)
    return np.sort(rng.choice(total_trajectories, size=batch_size, replace=False).astype(np.int32))


def clear_batched_optimization_grads(buffers: BatchedOptimizationBuffers) -> None:
    if buffers.active_point_friction.grad is not None:
        buffers.active_point_friction.grad.zero_()
    if buffers.full_point_friction.grad is not None:
        buffers.full_point_friction.grad.zero_()
    if buffers.loss.grad is not None:
        buffers.loss.grad.zero_()
    if buffers.position_loss.grad is not None:
        buffers.position_loss.grad.zero_()
    if buffers.orientation_loss.grad is not None:
        buffers.orientation_loss.grad.zero_()
    if buffers.linear_velocity_loss.grad is not None:
        buffers.linear_velocity_loss.grad.zero_()
    if buffers.angular_velocity_loss.grad is not None:
        buffers.angular_velocity_loss.grad.zero_()
    if buffers.batch_loss.grad is not None:
        buffers.batch_loss.grad.zero_()


def forward_rollout_with_batched_trajectory_loss(
    diff_scene: DiffScene,
    buffers: BatchedOptimizationBuffers,
    args: argparse.Namespace,
    *,
    scatter_active_point_friction_kernel,
    compute_batched_contact_weighted_masses_kernel,
    apply_batched_external_and_surface_point_forces_trajectory_kernel,
    accumulate_batched_frame_loss_kernel,
    combine_batched_loss_components_kernel,
    sum_batched_losses_kernel,
) -> wp.array:
    point_count = len(diff_scene.local_surface_points_np)
    point_scale = resolve_point_position_loss_scale(args, point_count)

    wp.launch(
        scatter_active_point_friction_kernel,
        dim=int(buffers.active_indices.shape[0]),
        inputs=[buffers.active_indices, buffers.active_point_friction, buffers.full_point_friction],
        device=diff_scene.model.device,
    )

    buffers.loss.zero_()
    buffers.position_loss.zero_()
    buffers.orientation_loss.zero_()
    buffers.linear_velocity_loss.zero_()
    buffers.angular_velocity_loss.zero_()
    buffers.batch_loss.zero_()
    buffers.contact_weighted_masses.zero_()
    buffers.contact_weighted_mass_total.zero_()

    wp.launch(
        set_batched_box_initial_states_kernel,
        dim=buffers.batch_size,
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

    wp.launch(
        accumulate_batched_frame_loss_kernel,
        dim=buffers.batch_size * point_count,
        inputs=[
            0,
            diff_scene.box_body_ids_wp,
            diff_scene.states[0].body_q,
            diff_scene.states[0].body_qd,
            diff_scene.local_surface_points_wp,
            buffers.target_positions,
            buffers.target_quaternions,
            buffers.target_linear_velocity,
            buffers.target_angular_velocity,
            buffers.trajectory_step_counts,
            buffers.frame_scales,
            float(point_scale),
            point_count,
            buffers.max_frames,
            buffers.position_loss,
            buffers.orientation_loss,
            buffers.linear_velocity_loss,
            buffers.angular_velocity_loss,
        ],
        device=diff_scene.model.device,
    )

    for step_idx in range(buffers.max_steps):
        state_in = diff_scene.states[step_idx]
        state_out = diff_scene.states[step_idx + 1]
        state_in.clear_forces()

        wp.launch(
            compute_batched_contact_weighted_masses_kernel,
            dim=buffers.batch_size * point_count,
            inputs=[
                step_idx,
                diff_scene.box_body_ids_wp,
                state_in.body_q,
                diff_scene.local_surface_points_wp,
                diff_scene.point_masses_wp,
                buffers.batch_size,
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
            dim=buffers.batch_size * point_count,
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
                buffers.batch_size,
                point_count,
                buffers.max_steps,
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

        wp.launch(
            accumulate_batched_frame_loss_kernel,
            dim=buffers.batch_size * point_count,
            inputs=[
                step_idx + 1,
                diff_scene.box_body_ids_wp,
                state_out.body_q,
                state_out.body_qd,
                diff_scene.local_surface_points_wp,
                buffers.target_positions,
                buffers.target_quaternions,
                buffers.target_linear_velocity,
                buffers.target_angular_velocity,
                buffers.trajectory_step_counts,
                buffers.frame_scales,
                float(point_scale),
                point_count,
                buffers.max_frames,
                buffers.position_loss,
                buffers.orientation_loss,
                buffers.linear_velocity_loss,
                buffers.angular_velocity_loss,
            ],
            device=diff_scene.model.device,
        )

    wp.launch(
        combine_batched_loss_components_kernel,
        dim=buffers.batch_size,
        inputs=[
            buffers.position_loss,
            buffers.orientation_loss,
            buffers.linear_velocity_loss,
            buffers.angular_velocity_loss,
            float(args.position_loss_weight),
            float(args.orientation_loss_weight),
            float(args.linear_velocity_loss_weight),
            float(args.angular_velocity_loss_weight),
            buffers.loss,
        ],
        device=diff_scene.model.device,
    )
    wp.launch(
        sum_batched_losses_kernel,
        dim=buffers.batch_size,
        inputs=[buffers.loss, float(1.0 / max(buffers.batch_size, 1)), buffers.batch_loss],
        device=diff_scene.model.device,
    )

    return buffers.batch_loss


def evaluate_collection_loss_in_batches(
    diff_scene: DiffScene,
    trajectories: list[MujocoTrajectory],
    args: argparse.Namespace,
    active_indices: np.ndarray,
    active_params: np.ndarray,
    initial_body_q: np.ndarray,
    initial_body_qd: np.ndarray,
    eval_batch_size: int,
    trajectory_progress_every: int,
    *,
    scatter_active_point_friction_kernel,
    compute_batched_contact_weighted_masses_kernel,
    apply_batched_external_and_surface_point_forces_trajectory_kernel,
    accumulate_batched_frame_loss_kernel,
    combine_batched_loss_components_kernel,
    sum_batched_losses_kernel,
) -> tuple[float, float, float, float, float, list[np.ndarray]]:
    total_loss = 0.0
    total_position_loss = 0.0
    total_orientation_loss = 0.0
    total_linear_velocity_loss = 0.0
    total_angular_velocity_loss = 0.0
    representative_body_q_frames: list[np.ndarray] = []
    total_trajectories = len(trajectories)

    for batch_start in range(0, total_trajectories, eval_batch_size):
        batch_end = min(batch_start + eval_batch_size, total_trajectories)
        batch_trajectories = trajectories[batch_start:batch_end]
        buffers = build_batched_optimization_buffers(diff_scene, batch_trajectories, args, active_indices)
        buffers.active_point_friction.assign(active_params)
        buffers.full_point_friction.assign(buffers.inactive_point_friction_np)
        clear_batched_optimization_grads(buffers)
        reset_scene_states(diff_scene, initial_body_q, initial_body_qd)
        forward_rollout_with_batched_trajectory_loss(
            diff_scene=diff_scene,
            buffers=buffers,
            args=args,
            scatter_active_point_friction_kernel=scatter_active_point_friction_kernel,
            compute_batched_contact_weighted_masses_kernel=compute_batched_contact_weighted_masses_kernel,
            apply_batched_external_and_surface_point_forces_trajectory_kernel=apply_batched_external_and_surface_point_forces_trajectory_kernel,
            accumulate_batched_frame_loss_kernel=accumulate_batched_frame_loss_kernel,
            combine_batched_loss_components_kernel=combine_batched_loss_components_kernel,
            sum_batched_losses_kernel=sum_batched_losses_kernel,
        )

        total_loss += float(np.sum(buffers.loss.numpy()))
        total_position_loss += float(np.sum(buffers.position_loss.numpy()))
        total_orientation_loss += float(np.sum(buffers.orientation_loss.numpy()))
        total_linear_velocity_loss += float(np.sum(buffers.linear_velocity_loss.numpy()))
        total_angular_velocity_loss += float(np.sum(buffers.angular_velocity_loss.numpy()))

        if batch_start == 0:
            first_steps = batch_trajectories[0].num_steps
            representative_body_q_frames = [
                state.body_q.numpy().copy() for state in diff_scene.states[: first_steps + 1]
            ]

        completed = batch_end
        if should_log_trajectory_progress(completed, total_trajectories, trajectory_progress_every):
            log_message(f"eval progress {completed}/{total_trajectories} trajectories")

    scale = 1.0 / max(total_trajectories, 1)
    return (
        total_loss * scale,
        total_position_loss * scale,
        total_orientation_loss * scale,
        total_linear_velocity_loss * scale,
        total_angular_velocity_loss * scale,
        representative_body_q_frames,
    )
