from __future__ import annotations

import warp as wp

from newton_surface_points_diff_demo import _smoothstep01


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
def scatter_indexed_point_friction_kernel(
    active_indices: wp.array(dtype=wp.int32),
    active_param_positions: wp.array(dtype=wp.int32),
    optimizer_params: wp.array(dtype=float),
    full_point_friction: wp.array(dtype=float),
):
    tid = wp.tid()
    point_idx = active_indices[tid]
    param_idx = active_param_positions[tid]
    full_point_friction[point_idx] = optimizer_params[param_idx]


@wp.kernel
def gather_active_point_friction_kernel(
    global_active_point_friction: wp.array(dtype=float),
    active_param_positions: wp.array(dtype=wp.int32),
    batch_active_point_friction: wp.array(dtype=float),
):
    tid = wp.tid()
    batch_active_point_friction[tid] = global_active_point_friction[active_param_positions[tid]]


@wp.func
def rotate_point_by_quat_xyzw(quat: wp.vec4, point: wp.vec3) -> wp.vec3:
    quat_xyz = wp.vec3(quat[0], quat[1], quat[2])
    twice_cross = 2.0 * wp.cross(quat_xyz, point)
    return point + quat[3] * twice_cross + wp.cross(quat_xyz, twice_cross)


@wp.kernel
def compute_batched_contact_weighted_masses_kernel(
    step_idx: int,
    box_body_ids: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    local_surface_points: wp.array(dtype=wp.vec3),
    point_masses: wp.array(dtype=float),
    batch_size: int,
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
    weighted_mass_idx = step_idx * batch_size * point_count + tid
    total_weight_idx = step_idx * batch_size + batch_idx
    weighted_masses[weighted_mass_idx] = weighted_mass
    wp.atomic_add(total_weighted_mass, total_weight_idx, weighted_mass)


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
    force_point_offsets_local: wp.array(dtype=wp.vec3),
    trajectory_step_counts: wp.array(dtype=wp.int32),
    batch_size: int,
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
        application_point = wp.transform_point(pose, force_point_offsets_local[batch_idx])
        external_moment_arm = application_point - world_com
        external_torque = wp.cross(external_moment_arm, external_force)
        wp.atomic_add(body_f, body_id, wp.spatial_vector(external_force, external_torque))

    weighted_mass_idx = step_idx * batch_size * point_count + tid
    total_weight_idx = step_idx * batch_size + batch_idx
    total_weight = total_weighted_mass[total_weight_idx]
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
    mass_fraction = weighted_masses[weighted_mass_idx] / total_weight

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
    local_surface_points: wp.array(dtype=wp.vec3),
    target_positions: wp.array(dtype=wp.vec3),
    target_quaternions: wp.array(dtype=wp.vec4),
    target_linear_velocity: wp.array(dtype=wp.vec3),
    target_angular_velocity: wp.array(dtype=wp.vec3),
    trajectory_step_counts: wp.array(dtype=wp.int32),
    frame_scales: wp.array(dtype=float),
    point_scale: float,
    point_count: int,
    max_frames: int,
    position_loss: wp.array(dtype=float),
    orientation_loss: wp.array(dtype=float),
    linear_velocity_loss: wp.array(dtype=float),
    angular_velocity_loss: wp.array(dtype=float),
):
    tid = wp.tid()
    batch_idx = tid // point_count
    point_idx = tid - batch_idx * point_count
    if frame_idx > trajectory_step_counts[batch_idx]:
        return

    body_id = box_body_ids[batch_idx]
    target_offset = batch_idx * max_frames + frame_idx
    frame_scale = frame_scales[batch_idx]

    pose = body_q[body_id]
    local_point = local_surface_points[point_idx]
    world_position = wp.transform_point(pose, local_point)
    target_position = target_positions[target_offset]
    target_quat = target_quaternions[target_offset]
    target_point_position = rotate_point_by_quat_xyzw(target_quat, local_point) + target_position
    position_delta = world_position - target_point_position
    position_loss_value = wp.dot(position_delta, position_delta)
    wp.atomic_add(position_loss, batch_idx, frame_scale * point_scale * position_loss_value)

    if point_idx != 0:
        return

    quat = wp.transform_get_rotation(pose)
    dot_q = quat[0] * target_quat[0] + quat[1] * target_quat[1] + quat[2] * target_quat[2] + quat[3] * target_quat[3]
    sign = 1.0
    if dot_q < 0.0:
        sign = -1.0
    quat_dx = sign * quat[0] - target_quat[0]
    quat_dy = sign * quat[1] - target_quat[1]
    quat_dz = sign * quat[2] - target_quat[2]
    quat_dw = sign * quat[3] - target_quat[3]
    orientation_loss_value = quat_dx * quat_dx + quat_dy * quat_dy + quat_dz * quat_dz + quat_dw * quat_dw

    wp.atomic_add(orientation_loss, batch_idx, frame_scale * orientation_loss_value)

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
def add_piecewise_regularization_loss_kernel(
    params: wp.array(dtype=float),
    side_ids: wp.array(dtype=wp.int32),
    side_means: wp.array(dtype=float),
    side_inv_counts: wp.array(dtype=float),
    weight: float,
    batch_loss: wp.array(dtype=float),
):
    tid = wp.tid()
    side_id = side_ids[tid]
    if side_id < 0 or side_id > 1:
        return

    delta = params[tid] - side_means[side_id]
    wp.atomic_add(batch_loss, 0, weight * side_inv_counts[side_id] * delta * delta)


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
def accumulate_gradient_norm_sq_kernel(
    grads: wp.array(dtype=float),
    norm_sq: wp.array(dtype=wp.float64),
    nonfinite_count: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    grad = grads[tid]
    if grad != grad or grad > 1.0e38 or grad < -1.0e38:
        wp.atomic_add(nonfinite_count, 0, 1)
        return

    grad64 = wp.float64(grad)
    wp.atomic_add(norm_sq, 0, grad64 * grad64)


@wp.kernel
def accumulate_iteration_scalar_stats_kernel(
    losses: wp.array(dtype=float),
    position_losses: wp.array(dtype=float),
    orientation_losses: wp.array(dtype=float),
    linear_velocity_losses: wp.array(dtype=float),
    angular_velocity_losses: wp.array(dtype=float),
    scale: wp.float64,
    stats: wp.array(dtype=wp.float64),
):
    tid = wp.tid()
    wp.atomic_add(stats, 0, wp.float64(losses[tid]) * scale)
    wp.atomic_add(stats, 1, wp.float64(position_losses[tid]) * scale)
    wp.atomic_add(stats, 2, wp.float64(orientation_losses[tid]) * scale)
    wp.atomic_add(stats, 3, wp.float64(linear_velocity_losses[tid]) * scale)
    wp.atomic_add(stats, 4, wp.float64(angular_velocity_losses[tid]) * scale)


@wp.kernel
def accumulate_gradient_scalar_stats_kernel(
    grads: wp.array(dtype=float),
    stats: wp.array(dtype=wp.float64),
):
    tid = wp.tid()
    grad = grads[tid]
    if not wp.isfinite(grad) or grad > 1.0e38 or grad < -1.0e38:
        wp.atomic_add(stats, 8, wp.float64(1.0))
        return

    grad64 = wp.float64(grad)
    abs_grad = wp.abs(grad64)
    wp.atomic_add(stats, 5, grad64 * grad64)
    wp.atomic_add(stats, 6, abs_grad)
    wp.atomic_max(stats, 7, abs_grad)


@wp.kernel
def accumulate_optimizer_scalar_stats_kernel(
    params: wp.array(dtype=float),
    first_moment: wp.array(dtype=wp.float64),
    second_moment: wp.array(dtype=wp.float64),
    stats: wp.array(dtype=wp.float64),
):
    tid = wp.tid()
    param = params[tid]
    if wp.isfinite(param) and param <= 1.0e38 and param >= -1.0e38:
        param64 = wp.float64(param)
        wp.atomic_add(stats, 0, param64)
        wp.atomic_add(stats, 1, param64 * param64)
        wp.atomic_min(stats, 2, param64)
        wp.atomic_max(stats, 3, param64)
    else:
        wp.atomic_add(stats, 4, wp.float64(1.0))

    moment_1 = first_moment[tid]
    if not wp.isfinite(moment_1):
        wp.atomic_add(stats, 5, wp.float64(1.0))

    moment_2 = second_moment[tid]
    if not wp.isfinite(moment_2):
        wp.atomic_add(stats, 6, wp.float64(1.0))


@wp.kernel
def sparse_adam_update_clipped_kernel(
    global_params: wp.array(dtype=float),
    grads: wp.array(dtype=float),
    active_param_positions: wp.array(dtype=wp.int32),
    first_moment: wp.array(dtype=wp.float64),
    second_moment: wp.array(dtype=wp.float64),
    adam_step: wp.array(dtype=wp.int32),
    beta1_power: wp.array(dtype=wp.float64),
    beta2_power: wp.array(dtype=wp.float64),
    grad_scale: wp.float64,
    learning_rate: wp.float64,
    beta1: wp.float64,
    beta2: wp.float64,
    eps: wp.float64,
    min_value: wp.float64,
    max_value: wp.float64,
):
    tid = wp.tid()
    pos = active_param_positions[tid]
    one = wp.float64(1.0)

    grad = wp.float64(grads[tid]) * grad_scale
    moment_1 = beta1 * first_moment[pos] + (one - beta1) * grad
    moment_2 = beta2 * second_moment[pos] + (one - beta2) * (grad * grad)
    beta1_power_next = beta1_power[pos] * beta1
    beta2_power_next = beta2_power[pos] * beta2

    first_hat = moment_1 / (one - beta1_power_next)
    second_hat = moment_2 / (one - beta2_power_next)
    updated = wp.float64(global_params[pos]) - learning_rate * first_hat / (wp.sqrt(second_hat) + eps)

    global_params[pos] = wp.float32(wp.min(wp.max(updated, min_value), max_value))
    first_moment[pos] = moment_1
    second_moment[pos] = moment_2
    adam_step[pos] = adam_step[pos] + 1
    beta1_power[pos] = beta1_power_next
    beta2_power[pos] = beta2_power_next


