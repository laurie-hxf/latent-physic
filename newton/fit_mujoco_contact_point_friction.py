from __future__ import annotations

import argparse
import time

import numpy as np
import warp as wp

from mujoco_contact_friction_fit_utils import (
    MujocoTrajectory,
    MujocoTrajectoryCollection,
    OptimizationBuffers,
    compute_active_contact_point_indices,
    load_mujoco_trajectories,
)
from mujoco_contact_friction_fit_wandb import build_wandb_log_payload, init_wandb
from fit_mujoco_contact_point_friction_io import (
    DEFAULT_TRAIN_BATCH_SIZE,
    parse_args,
)
from fit_mujoco_contact_point_friction_output import export_contact_friction_outputs
from fit_mujoco_contact_point_friction_runtime import (
    assert_array_finite,
    build_batched_optimization_buffers,
    clear_batched_optimization_grads,
    evaluate_collection_loss_in_batches,
    forward_rollout_with_batched_trajectory_loss,
    log_message,
    resolve_batch_size,
    reset_scene_states,
    sample_training_batch_indices,
    should_log_trajectory_progress,
)
from newton_surface_points_diff_demo import (
    _smoothstep01,
    build_diff_scene,
)


@wp.kernel
def scatter_active_point_friction_kernel(
    active_indices: wp.array(dtype=wp.int32),
    active_point_friction: wp.array(dtype=float),
    full_point_friction: wp.array(dtype=float),
):
    tid = wp.tid()
    point_idx = active_indices[tid]
    full_point_friction[point_idx] = active_point_friction[tid]


@wp.kernel
def apply_external_and_surface_point_forces_trajectory_kernel(
    step_idx: int,
    body_id: int,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    local_surface_points: wp.array(dtype=wp.vec3),
    weighted_masses: wp.array(dtype=float),
    total_weighted_mass: wp.array(dtype=float),
    point_friction: wp.array(dtype=float),
    step_forces: wp.array(dtype=wp.vec3),
    step_application_points: wp.array(dtype=wp.vec3),
    total_mass: float,
    gravity_magnitude: float,
    floor_top_z: float,
    contact_stiffness: float,
    contact_damping: float,
    contact_band: float,
    friction_regularization: float,
    body_f: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    pose = body_q[body_id]
    qd = body_qd[body_id]
    world_com = wp.transform_point(pose, body_com[body_id])

    if tid == 0:
        external_force = step_forces[step_idx]
        application_point = step_application_points[step_idx]
        external_moment_arm = application_point - world_com
        external_torque = wp.cross(external_moment_arm, external_force)
        wp.atomic_add(body_f, body_id, wp.spatial_vector(external_force, external_torque))

    total_weight = total_weighted_mass[0]
    if total_weight <= 1.0e-8:
        return

    world_point = wp.transform_point(pose, local_surface_points[tid])
    moment_arm = world_point - world_com

    linear_velocity = wp.spatial_top(qd)
    angular_velocity = wp.spatial_bottom(qd)
    point_velocity = linear_velocity + wp.cross(angular_velocity, moment_arm)

    gap = world_point[2] - floor_top_z
    penetration = wp.max(-gap, 0.0)
    safe_band = wp.max(contact_band, 1.0e-6)
    activation = _smoothstep01((contact_band - gap) / safe_band)
    mass_fraction = weighted_masses[tid] / total_weight

    external_force = step_forces[step_idx]
    normal_load_total = wp.max(0.0, total_mass * gravity_magnitude - external_force[2])
    support_force_z = mass_fraction * normal_load_total
    penalty_force_z = mass_fraction * activation * (
        contact_stiffness * penetration + contact_damping * wp.max(-point_velocity[2], 0.0)
    )
    normal_force = wp.vec3(0.0, 0.0, support_force_z + penalty_force_z)
    tangential_velocity = wp.vec3(point_velocity[0], point_velocity[1], 0.0)
    tangential_speed = wp.sqrt(
        wp.dot(tangential_velocity, tangential_velocity) + friction_regularization * friction_regularization
    )
    normal_load = mass_fraction * normal_load_total
    mu = wp.max(point_friction[tid], 0.0)
    friction_force = -mu * normal_load * (tangential_velocity / tangential_speed)
    total_force = normal_force + friction_force
    total_torque = wp.cross(moment_arm, total_force)
    wp.atomic_add(body_f, body_id, wp.spatial_vector(total_force, total_torque))


@wp.kernel
def accumulate_frame_loss_kernel(
    body_id: int,
    frame_idx: int,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    target_positions: wp.array(dtype=wp.vec3),
    target_quaternions: wp.array(dtype=wp.vec4),
    target_linear_velocity: wp.array(dtype=wp.vec3),
    target_angular_velocity: wp.array(dtype=wp.vec3),
    frame_scale: float,
    position_loss: wp.array(dtype=float),
    orientation_loss: wp.array(dtype=float),
    linear_velocity_loss: wp.array(dtype=float),
    angular_velocity_loss: wp.array(dtype=float),
    accumulate_velocity_loss: int,
):
    pose = body_q[body_id]
    world_position = wp.transform_get_translation(pose)
    target_position = target_positions[frame_idx]
    position_delta = world_position - target_position
    position_loss_value = wp.dot(position_delta, position_delta)

    quat = wp.transform_get_rotation(pose)
    target_quat = target_quaternions[frame_idx]
    dot_q = quat[0] * target_quat[0] + quat[1] * target_quat[1] + quat[2] * target_quat[2] + quat[3] * target_quat[3]
    sign = 1.0
    if dot_q < 0.0:
        sign = -1.0
    quat_dx = sign * quat[0] - target_quat[0]
    quat_dy = sign * quat[1] - target_quat[1]
    quat_dz = sign * quat[2] - target_quat[2]
    quat_dw = sign * quat[3] - target_quat[3]
    orientation_loss_value = quat_dx * quat_dx + quat_dy * quat_dy + quat_dz * quat_dz + quat_dw * quat_dw

    wp.atomic_add(position_loss, 0, frame_scale * position_loss_value)
    wp.atomic_add(orientation_loss, 0, frame_scale * orientation_loss_value)

    if accumulate_velocity_loss != 0:
        spatial_velocity = body_qd[body_id]
        linear_velocity = wp.spatial_top(spatial_velocity)
        angular_velocity = wp.spatial_bottom(spatial_velocity)

        linear_delta = linear_velocity - target_linear_velocity[frame_idx]
        angular_delta = angular_velocity - target_angular_velocity[frame_idx]
        linear_loss_value = wp.dot(linear_delta, linear_delta)
        angular_loss_value = wp.dot(angular_delta, angular_delta)

        wp.atomic_add(linear_velocity_loss, 0, frame_scale * linear_loss_value)
        wp.atomic_add(angular_velocity_loss, 0, frame_scale * angular_loss_value)


@wp.kernel
def compute_batched_contact_weighted_masses_kernel(
    box_body_ids: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    local_surface_points: wp.array(dtype=wp.vec3),
    point_masses: wp.array(dtype=float),
    point_count: int,
    floor_top_z: float,
    contact_band: float,
    weighted_masses: wp.array(dtype=float),
    total_weighted_mass: wp.array(dtype=float),
):
    tid = wp.tid()
    batch_idx = tid // point_count
    point_idx = tid - batch_idx * point_count
    body_id = box_body_ids[batch_idx]
    pose = body_q[body_id]
    world_point = wp.transform_point(pose, local_surface_points[point_idx])
    gap = world_point[2] - floor_top_z
    safe_band = wp.max(contact_band, 1.0e-6)
    activation = _smoothstep01((contact_band - gap) / safe_band)
    weighted_mass = activation * point_masses[point_idx]
    weighted_masses[tid] = weighted_mass
    wp.atomic_add(total_weighted_mass, batch_idx, weighted_mass)


@wp.kernel
def apply_batched_external_and_surface_point_forces_trajectory_kernel(
    step_idx: int,
    box_body_ids: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    local_surface_points: wp.array(dtype=wp.vec3),
    weighted_masses: wp.array(dtype=float),
    total_weighted_mass: wp.array(dtype=float),
    point_friction: wp.array(dtype=float),
    step_forces: wp.array(dtype=wp.vec3),
    step_application_points: wp.array(dtype=wp.vec3),
    trajectory_step_counts: wp.array(dtype=wp.int32),
    point_count: int,
    max_steps: int,
    total_mass: float,
    gravity_magnitude: float,
    floor_top_z: float,
    contact_stiffness: float,
    contact_damping: float,
    contact_band: float,
    friction_regularization: float,
    body_f: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    batch_idx = tid // point_count
    point_idx = tid - batch_idx * point_count
    if step_idx >= trajectory_step_counts[batch_idx]:
        return

    body_id = box_body_ids[batch_idx]
    pose = body_q[body_id]
    qd = body_qd[body_id]
    world_com = wp.transform_point(pose, body_com[body_id])
    step_offset = batch_idx * max_steps + step_idx

    if point_idx == 0:
        external_force = step_forces[step_offset]
        application_point = step_application_points[step_offset]
        external_moment_arm = application_point - world_com
        external_torque = wp.cross(external_moment_arm, external_force)
        wp.atomic_add(body_f, body_id, wp.spatial_vector(external_force, external_torque))

    total_weight = total_weighted_mass[batch_idx]
    if total_weight <= 1.0e-8:
        return

    world_point = wp.transform_point(pose, local_surface_points[point_idx])
    moment_arm = world_point - world_com

    linear_velocity = wp.spatial_top(qd)
    angular_velocity = wp.spatial_bottom(qd)
    point_velocity = linear_velocity + wp.cross(angular_velocity, moment_arm)

    gap = world_point[2] - floor_top_z
    penetration = wp.max(-gap, 0.0)
    safe_band = wp.max(contact_band, 1.0e-6)
    activation = _smoothstep01((contact_band - gap) / safe_band)
    mass_fraction = weighted_masses[tid] / total_weight

    external_force = step_forces[step_offset]
    normal_load_total = wp.max(0.0, total_mass * gravity_magnitude - external_force[2])
    support_force_z = mass_fraction * normal_load_total
    penalty_force_z = mass_fraction * activation * (
        contact_stiffness * penetration + contact_damping * wp.max(-point_velocity[2], 0.0)
    )
    normal_force = wp.vec3(0.0, 0.0, support_force_z + penalty_force_z)
    tangential_velocity = wp.vec3(point_velocity[0], point_velocity[1], 0.0)
    tangential_speed = wp.sqrt(
        wp.dot(tangential_velocity, tangential_velocity) + friction_regularization * friction_regularization
    )
    normal_load = mass_fraction * normal_load_total
    mu = wp.max(point_friction[point_idx], 0.0)
    friction_force = -mu * normal_load * (tangential_velocity / tangential_speed)
    total_force = normal_force + friction_force
    total_torque = wp.cross(moment_arm, total_force)
    wp.atomic_add(body_f, body_id, wp.spatial_vector(total_force, total_torque))


@wp.kernel
def accumulate_batched_frame_loss_kernel(
    frame_idx: int,
    box_body_ids: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    target_positions: wp.array(dtype=wp.vec3),
    target_quaternions: wp.array(dtype=wp.vec4),
    target_linear_velocity: wp.array(dtype=wp.vec3),
    target_angular_velocity: wp.array(dtype=wp.vec3),
    trajectory_step_counts: wp.array(dtype=wp.int32),
    frame_scales: wp.array(dtype=float),
    max_frames: int,
    position_loss: wp.array(dtype=float),
    orientation_loss: wp.array(dtype=float),
    linear_velocity_loss: wp.array(dtype=float),
    angular_velocity_loss: wp.array(dtype=float),
    accumulate_velocity_loss: int,
):
    batch_idx = wp.tid()
    if frame_idx > trajectory_step_counts[batch_idx]:
        return

    body_id = box_body_ids[batch_idx]
    target_offset = batch_idx * max_frames + frame_idx
    frame_scale = frame_scales[batch_idx]

    pose = body_q[body_id]
    world_position = wp.transform_get_translation(pose)
    target_position = target_positions[target_offset]
    position_delta = world_position - target_position
    position_loss_value = wp.dot(position_delta, position_delta)

    quat = wp.transform_get_rotation(pose)
    target_quat = target_quaternions[target_offset]
    dot_q = quat[0] * target_quat[0] + quat[1] * target_quat[1] + quat[2] * target_quat[2] + quat[3] * target_quat[3]
    sign = 1.0
    if dot_q < 0.0:
        sign = -1.0
    quat_dx = sign * quat[0] - target_quat[0]
    quat_dy = sign * quat[1] - target_quat[1]
    quat_dz = sign * quat[2] - target_quat[2]
    quat_dw = sign * quat[3] - target_quat[3]
    orientation_loss_value = quat_dx * quat_dx + quat_dy * quat_dy + quat_dz * quat_dz + quat_dw * quat_dw

    wp.atomic_add(position_loss, batch_idx, frame_scale * position_loss_value)
    wp.atomic_add(orientation_loss, batch_idx, frame_scale * orientation_loss_value)

    if accumulate_velocity_loss != 0:
        spatial_velocity = body_qd[body_id]
        linear_velocity = wp.spatial_top(spatial_velocity)
        angular_velocity = wp.spatial_bottom(spatial_velocity)

        linear_delta = linear_velocity - target_linear_velocity[target_offset]
        angular_delta = angular_velocity - target_angular_velocity[target_offset]
        linear_loss_value = wp.dot(linear_delta, linear_delta)
        angular_loss_value = wp.dot(angular_delta, angular_delta)

        wp.atomic_add(linear_velocity_loss, batch_idx, frame_scale * linear_loss_value)
        wp.atomic_add(angular_velocity_loss, batch_idx, frame_scale * angular_loss_value)


@wp.kernel
def combine_batched_loss_components_kernel(
    position_loss: wp.array(dtype=float),
    orientation_loss: wp.array(dtype=float),
    linear_velocity_loss: wp.array(dtype=float),
    angular_velocity_loss: wp.array(dtype=float),
    position_weight: float,
    orientation_weight: float,
    linear_velocity_weight: float,
    angular_velocity_weight: float,
    loss: wp.array(dtype=float),
):
    tid = wp.tid()
    loss[tid] = (
        position_weight * position_loss[tid]
        + orientation_weight * orientation_loss[tid]
        + linear_velocity_weight * linear_velocity_loss[tid]
        + angular_velocity_weight * angular_velocity_loss[tid]
    )


@wp.kernel
def sum_batched_losses_kernel(
    losses: wp.array(dtype=float),
    scale: float,
    batch_loss: wp.array(dtype=float),
):
    tid = wp.tid()
    wp.atomic_add(batch_loss, 0, scale * losses[tid])


@wp.kernel
def combine_loss_components_kernel(
    position_loss: wp.array(dtype=float),
    orientation_loss: wp.array(dtype=float),
    linear_velocity_loss: wp.array(dtype=float),
    angular_velocity_loss: wp.array(dtype=float),
    position_weight: float,
    orientation_weight: float,
    linear_velocity_weight: float,
    angular_velocity_weight: float,
    loss: wp.array(dtype=float),
):
    loss[0] = (
        position_weight * position_loss[0]
        + orientation_weight * orientation_loss[0]
        + linear_velocity_weight * linear_velocity_loss[0]
        + angular_velocity_weight * angular_velocity_loss[0]
    )


@wp.kernel
def adam_update_kernel(
    params: wp.array(dtype=float),
    grads: wp.array(dtype=wp.float64),
    first_moment: wp.array(dtype=wp.float64),
    second_moment: wp.array(dtype=wp.float64),
    bias_correction1: wp.float64,
    bias_correction2: wp.float64,
    learning_rate: wp.float64,
    beta1: wp.float64,
    beta2: wp.float64,
    eps: wp.float64,
    min_value: wp.float64,
    max_value: wp.float64,
):
    tid = wp.tid()
    one = wp.float64(1.0)
    grad = grads[tid]
    moment_1 = beta1 * first_moment[tid] + (one - beta1) * grad
    moment_2 = beta2 * second_moment[tid] + (one - beta2) * (grad * grad)
    first_hat = moment_1 / bias_correction1
    second_hat = moment_2 / bias_correction2
    updated = wp.float64(params[tid]) - learning_rate * first_hat / (wp.sqrt(second_hat) + eps)
    params[tid] = wp.float32(wp.min(wp.max(updated, min_value), max_value))
    first_moment[tid] = moment_1
    second_moment[tid] = moment_2


@wp.kernel
def add_scaled_scalar_kernel(
    src: wp.array(dtype=float),
    scale: float,
    dst: wp.array(dtype=float),
):
    wp.atomic_add(dst, 0, scale * src[0])


@wp.kernel
def flag_nonfinite_array_at_index_kernel(
    values: wp.array(dtype=float),
    nonfinite_flags: wp.array(dtype=wp.int32),
    flag_index: int,
):
    tid = wp.tid()
    value = values[tid]
    if value != value or value > 1.0e30 or value < -1.0e30:
        wp.atomic_add(nonfinite_flags, flag_index, 1)


@wp.kernel
def add_float32_array_to_float64_if_unflagged_kernel(
    dst: wp.array(dtype=wp.float64),
    src: wp.array(dtype=float),
    nonfinite_flags: wp.array(dtype=wp.int32),
    flag_index: int,
):
    tid = wp.tid()
    if nonfinite_flags[flag_index] == 0:
        dst[tid] = dst[tid] + wp.float64(src[tid])


@wp.kernel
def accumulate_scalar_metrics_if_unflagged_kernel(
    nonfinite_flags: wp.array(dtype=wp.int32),
    flag_index: int,
    loss: wp.array(dtype=float),
    position_loss: wp.array(dtype=float),
    orientation_loss: wp.array(dtype=float),
    linear_velocity_loss: wp.array(dtype=float),
    angular_velocity_loss: wp.array(dtype=float),
    totals: wp.array(dtype=wp.float64),
    good_count: wp.array(dtype=wp.int32),
):
    if nonfinite_flags[flag_index] != 0:
        return

    wp.atomic_add(totals, 0, wp.float64(loss[0]))
    wp.atomic_add(totals, 1, wp.float64(position_loss[0]))
    wp.atomic_add(totals, 2, wp.float64(orientation_loss[0]))
    wp.atomic_add(totals, 3, wp.float64(linear_velocity_loss[0]))
    wp.atomic_add(totals, 4, wp.float64(angular_velocity_loss[0]))
    wp.atomic_add(good_count, 0, 1)


def main() -> None:
    args = parse_args()
    startup_time = time.time()
    log_message(f"loading trajectories from {args.trajectory_npz.resolve()}")
    trajectory_collection = load_mujoco_trajectories(args.trajectory_npz, args.max_steps, args.max_trajectories)
    trajectories = trajectory_collection.trajectories
    representative_trajectory = trajectories[0]
    batch_size = resolve_batch_size(args.batch_size, len(trajectories), DEFAULT_TRAIN_BATCH_SIZE)
    eval_batch_size = resolve_batch_size(args.eval_batch_size, len(trajectories), batch_size)
    args.steps = trajectory_collection.max_steps
    args.dt = representative_trajectory.timestep
    log_message(
        f"loaded {len(trajectories)} trajectories | source={trajectory_collection.source_type} | "
        f"max_steps={trajectory_collection.max_steps} | dt={representative_trajectory.timestep:.6f} | "
        f"train_batch_size={batch_size} | eval_batch_size={eval_batch_size}"
    )

    log_message(f"building diff scene on device={args.device if args.device is not None else 'auto'}")
    args.batch_capacity = max(batch_size, eval_batch_size, 1)
    diff_scene = build_diff_scene(args)
    initial_body_q = diff_scene.states[0].body_q.numpy().copy()
    initial_body_qd = diff_scene.states[0].body_qd.numpy().copy()

    log_message("computing active contact point mask across trajectories")
    active_mask = np.zeros(len(diff_scene.local_surface_points_np), dtype=bool)
    for trajectory_idx, trajectory in enumerate(trajectories, start=1):
        trajectory_active_indices = compute_active_contact_point_indices(
            local_surface_points=diff_scene.local_surface_points_np,
            trajectory=trajectory,
            floor_top_z=diff_scene.floor_top_z,
            contact_threshold=float(args.contact_mask_threshold),
        )
        active_mask[trajectory_active_indices] = True
        if should_log_trajectory_progress(
            trajectory_idx,
            len(trajectories),
            int(args.trajectory_progress_every),
        ):
            log_message(f"active-mask progress {trajectory_idx}/{len(trajectories)} trajectories")
    active_indices = np.flatnonzero(active_mask).astype(np.int32)
    if len(active_indices) == 0:
        raise RuntimeError(
            "No contact points were detected in the target trajectory. "
            "Try increasing --contact-mask-threshold or decreasing --surface-point-spacing."
        )
    log_message(
        f"active contact points={len(active_indices)} / surface points={len(diff_scene.local_surface_points_np)} "
        f"| startup_elapsed={time.time() - startup_time:.2f}s"
    )

    wandb_run = init_wandb(args, trajectory_collection, active_indices)
    if wandb_run is not None:
        log_message(
            f"W&B enabled | project={args.wandb_project} | "
            f"run={wandb_run.name} | mode={args.wandb_mode}"
        )

    device = str(diff_scene.torch_device)
    active_params_np = np.full(len(active_indices), float(args.point_friction), dtype=np.float32)
    active_params = wp.array(active_params_np, dtype=wp.float32, device=device)
    adam_m = wp.zeros(len(active_indices), dtype=wp.float64, device=device)
    adam_v = wp.zeros(len(active_indices), dtype=wp.float64, device=device)
    grad_value_total_wp = wp.zeros(len(active_indices), dtype=wp.float64, device=device)
    nonfinite_flag = wp.zeros(1, dtype=wp.int32, device=device)
    loss_history: list[float] = []
    best_loss = float("inf")
    best_active_params = active_params_np.copy()
    rng = np.random.default_rng(int(args.seed))
    try:
        for iteration in range(1, max(int(args.opt_iters), 0) + 1):
            iteration_start = time.time()
            batch_indices = sample_training_batch_indices(len(trajectories), batch_size, rng)
            batch_trajectories = [trajectories[int(idx)] for idx in batch_indices]
            buffers = build_batched_optimization_buffers(diff_scene, batch_trajectories, args, active_indices)
            buffers.active_point_friction.assign(active_params)
            buffers.full_point_friction.assign(buffers.inactive_point_friction_np)
            clear_batched_optimization_grads(buffers)

            grad_value_total_wp.zero_()
            nonfinite_flag.zero_()
            tape = wp.Tape()
            with tape:
                reset_scene_states(diff_scene, initial_body_q, initial_body_qd)
                forward_rollout_with_batched_trajectory_loss(
                    diff_scene,
                    buffers,
                    args,
                    scatter_active_point_friction_kernel=scatter_active_point_friction_kernel,
                    compute_batched_contact_weighted_masses_kernel=compute_batched_contact_weighted_masses_kernel,
                    apply_batched_external_and_surface_point_forces_trajectory_kernel=apply_batched_external_and_surface_point_forces_trajectory_kernel,
                    accumulate_batched_frame_loss_kernel=accumulate_batched_frame_loss_kernel,
                    combine_batched_loss_components_kernel=combine_batched_loss_components_kernel,
                    sum_batched_losses_kernel=sum_batched_losses_kernel,
                )
            tape.backward(buffers.batch_loss)

            if buffers.active_point_friction.grad is None:
                tape.zero()
                continue

            wp.launch(
                flag_nonfinite_array_at_index_kernel,
                dim=len(active_indices),
                inputs=[buffers.active_point_friction.grad, nonfinite_flag, 0],
                device=diff_scene.model.device,
            )
            if int(nonfinite_flag.numpy()[0]) != 0:
                tape.zero()
                continue

            wp.launch(
                add_float32_array_to_float64_if_unflagged_kernel,
                dim=len(active_indices),
                inputs=[grad_value_total_wp, buffers.active_point_friction.grad, nonfinite_flag, 0],
                device=diff_scene.model.device,
            )

            good_buffer_count = len(batch_trajectories)
            loss_value = float(np.mean(buffers.loss.numpy()))
            position_loss_value = float(np.mean(buffers.position_loss.numpy()))
            orientation_loss_value = float(np.mean(buffers.orientation_loss.numpy()))
            linear_velocity_loss_value = float(np.mean(buffers.linear_velocity_loss.numpy()))
            angular_velocity_loss_value = float(np.mean(buffers.angular_velocity_loss.numpy()))
            grad_value = grad_value_total_wp.numpy()
            assert_array_finite(
                "batch grad_value_total",
                grad_value,
                context=f"iter={iteration:04d} after gradient accumulation",
            )
            beta1 = float(args.adam_beta1)
            beta2 = float(args.adam_beta2)
            bias_correction1 = 1.0 - beta1**iteration
            bias_correction2 = 1.0 - beta2**iteration
            wp.launch(
                adam_update_kernel,
                dim=len(active_indices),
                inputs=[
                    active_params,
                    grad_value_total_wp,
                    adam_m,
                    adam_v,
                    np.float64(bias_correction1),
                    np.float64(bias_correction2),
                    np.float64(args.learning_rate),
                    np.float64(beta1),
                    np.float64(beta2),
                    np.float64(args.adam_eps),
                    np.float64(args.min_point_friction),
                    np.float64(args.max_point_friction),
                ],
                device=diff_scene.model.device,
            )
            active_params_np = active_params.numpy().astype(np.float32)
            adam_m_np = adam_m.numpy()
            adam_v_np = adam_v.numpy()
            assert_array_finite(
                "active_params",
                active_params_np,
                context=f"iter={iteration:04d} after Adam update",
            )
            assert_array_finite(
                "adam_m",
                adam_m_np,
                context=f"iter={iteration:04d} after Adam update",
            )
            assert_array_finite(
                "adam_v",
                adam_v_np,
                context=f"iter={iteration:04d} after Adam update",
            )
            tape.zero()
            loss_history.append(loss_value)

            if loss_value < best_loss:
                best_loss = loss_value
                best_active_params = active_params_np.copy()

            if wandb_run is not None:
                log_payload = build_wandb_log_payload(
                    loss_value=loss_value,
                    position_loss_value=position_loss_value,
                    orientation_loss_value=orientation_loss_value,
                    linear_velocity_loss_value=linear_velocity_loss_value,
                    angular_velocity_loss_value=angular_velocity_loss_value,
                    grad_value=grad_value,
                    active_params=active_params_np,
                    active_indices=active_indices,
                )
                wandb_run.log(log_payload, step=iteration)

            if iteration == 1 or iteration % max(int(args.log_every), 1) == 0 or iteration == int(args.opt_iters):
                log_message(
                    f"iter={iteration:04d} loss={loss_value:.6f} "
                    f"pos={position_loss_value:.6f} "
                    f"ori={orientation_loss_value:.6f} "
                    f"linvel={linear_velocity_loss_value:.6f} "
                    f"angvel={angular_velocity_loss_value:.6f} "
                    f"grad_norm={float(np.linalg.norm(grad_value)):.6f} "
                    f"mu_min={float(active_params_np.min()):.6f} "
                    f"mu_max={float(active_params_np.max()):.6f} "
                    f"batch={good_buffer_count}/{len(batch_trajectories)} "
                    f"elapsed={time.time() - iteration_start:.2f}s"
                )

        log_message("running final evaluation across the configured trajectory set")
        final_loss, final_position_loss, final_orientation_loss, final_linear_velocity_loss, final_angular_velocity_loss, body_q_frames = evaluate_collection_loss_in_batches(
            diff_scene=diff_scene,
            trajectories=trajectories,
            args=args,
            active_indices=active_indices,
            active_params=best_active_params,
            initial_body_q=initial_body_q,
            initial_body_qd=initial_body_qd,
            eval_batch_size=eval_batch_size,
            trajectory_progress_every=int(args.trajectory_progress_every),
            scatter_active_point_friction_kernel=scatter_active_point_friction_kernel,
            compute_batched_contact_weighted_masses_kernel=compute_batched_contact_weighted_masses_kernel,
            apply_batched_external_and_surface_point_forces_trajectory_kernel=apply_batched_external_and_surface_point_forces_trajectory_kernel,
            accumulate_batched_frame_loss_kernel=accumulate_batched_frame_loss_kernel,
            combine_batched_loss_components_kernel=combine_batched_loss_components_kernel,
            sum_batched_losses_kernel=sum_batched_losses_kernel,
        )

        assert_array_finite(
            "best_active_params",
            best_active_params,
            context="final export",
        )
        learned_point_friction = export_contact_friction_outputs(
            args=args,
            trajectory_collection=trajectory_collection,
            representative_trajectory=representative_trajectory,
            trajectories=trajectories,
            diff_scene=diff_scene,
            active_indices=active_indices,
            best_active_params=best_active_params,
            loss_history=loss_history,
            best_loss=best_loss,
            final_loss=final_loss,
            final_position_loss=final_position_loss,
            final_orientation_loss=final_orientation_loss,
            final_linear_velocity_loss=final_linear_velocity_loss,
            final_angular_velocity_loss=final_angular_velocity_loss,
            body_q_frames=body_q_frames,
        )

        if wandb_run is not None:
            wandb_run.summary["surface_points"] = int(len(diff_scene.local_surface_points_np))
            wandb_run.summary["active_contact_points"] = int(len(active_indices))
            wandb_run.summary["final_loss"] = float(final_loss)
            wandb_run.summary["final_position_loss"] = float(final_position_loss)
            wandb_run.summary["final_orientation_loss"] = float(final_orientation_loss)
            wandb_run.summary["final_linear_velocity_loss"] = float(final_linear_velocity_loss)
            wandb_run.summary["final_angular_velocity_loss"] = float(final_angular_velocity_loss)
            wandb_run.summary["mu_mean"] = float(best_active_params.mean())
            wandb_run.summary["mu_std"] = float(best_active_params.std())
            wandb_run.summary["mu_min"] = float(best_active_params.min())
            wandb_run.summary["mu_max"] = float(best_active_params.max())
            wandb_run.summary["results_path"] = str(args.results_path.resolve())
            wandb_run.summary["heatmap_path"] = str(args.heatmap_path.resolve())
            if args.scene_usd_path is not None:
                wandb_run.summary["scene_usd_path"] = str(args.scene_usd_path.resolve())

        log_message(f"trajectory={args.trajectory_npz.resolve()}")
        log_message(f"trajectory_source_type={trajectory_collection.source_type}")
        log_message(f"trajectory_count={len(trajectories)}")
        log_message(f"max_steps={trajectory_collection.max_steps} dt={representative_trajectory.timestep:.6f}")
        log_message(f"surface_points={len(diff_scene.local_surface_points_np)} active_contact_points={len(active_indices)}")
        log_message(f"final_loss={final_loss:.6f}")
        log_message(f"final_position_loss={final_position_loss:.6f}")
        log_message(f"final_orientation_loss={final_orientation_loss:.6f}")
        log_message(f"final_linear_velocity_loss={final_linear_velocity_loss:.6f}")
        log_message(f"final_angular_velocity_loss={final_angular_velocity_loss:.6f}")
        log_message(f"results_written_to={args.results_path.resolve()}")
        log_message(f"heatmap_written_to={args.heatmap_path.resolve()}")
        if args.scene_usd_path is not None:
            log_message(f"scene_usd_written_to={args.scene_usd_path.resolve()}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
