from __future__ import annotations

import warp as wp


INPUT_DIM = 11
HIDDEN0_DIM = 128
HIDDEN1_DIM = 128
HIDDEN2_DIM = 64
OUTPUT_DIM = 3
RESIDUAL_OUTPUT_MODE_ACCELERATION = 0
RESIDUAL_OUTPUT_MODE_VELOCITY = 1


@wp.func
def _quat_mul_xyzw(a: wp.quat, b: wp.quat) -> wp.quat:
    ax = a[0]
    ay = a[1]
    az = a[2]
    aw = a[3]
    bx = b[0]
    by = b[1]
    bz = b[2]
    bw = b[3]
    return wp.quat(
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


@wp.func
def _quat_yaw_xyzw(q: wp.quat) -> float:
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    return wp.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


@wp.func
def _wrap_angle(angle: float) -> float:
    return wp.atan2(wp.sin(angle), wp.cos(angle))


@wp.func
def _rotate_point_by_quat_xyzw(quat: wp.vec4, point: wp.vec3) -> wp.vec3:
    quat_xyz = wp.vec3(quat[0], quat[1], quat[2])
    twice_cross = 2.0 * wp.cross(quat_xyz, point)
    return point + quat[3] * twice_cross + wp.cross(quat_xyz, twice_cross)


@wp.func
def _normalized_feature(value: float, feature_idx: int, mean: wp.array(dtype=float), inv_std: wp.array(dtype=float)) -> float:
    return (value - mean[feature_idx]) * inv_std[feature_idx]


@wp.func
def _feature_value(
    feature_idx: int,
    pose: wp.transform,
    qd: wp.spatial_vector,
    force_world: wp.vec3,
    point_offset_local: wp.vec3,
    mu_features: wp.array(dtype=float),
) -> float:
    quat = wp.transform_get_rotation(pose)
    yaw = _quat_yaw_xyzw(quat)
    c = wp.cos(yaw)
    s = wp.sin(yaw)

    linear_velocity = wp.spatial_top(qd)
    angular_velocity = wp.spatial_bottom(qd)

    v_body_x = c * linear_velocity[0] + s * linear_velocity[1]
    v_body_y = -s * linear_velocity[0] + c * linear_velocity[1]
    force_body_x = c * force_world[0] + s * force_world[1]
    force_body_y = -s * force_world[0] + c * force_world[1]
    torque_z = point_offset_local[0] * force_body_y - point_offset_local[1] * force_body_x

    value = 0.0
    if feature_idx == 0:
        value = v_body_x
    elif feature_idx == 1:
        value = v_body_y
    elif feature_idx == 2:
        value = angular_velocity[2]
    elif feature_idx == 3:
        value = force_body_x
    elif feature_idx == 4:
        value = force_body_y
    elif feature_idx == 5:
        value = point_offset_local[0]
    elif feature_idx == 6:
        value = point_offset_local[1]
    elif feature_idx == 7:
        value = torque_z
    elif feature_idx == 8:
        value = mu_features[0]
    elif feature_idx == 9:
        value = mu_features[1]
    else:
        value = mu_features[2]
    return value


@wp.kernel
def clip_optimizer_params_kernel(
    optimizer_params: wp.array(dtype=float),
    min_value: float,
    max_value: float,
):
    tid = wp.tid()
    value = optimizer_params[tid]
    if value < min_value:
        value = min_value
    if value > max_value:
        value = max_value
    optimizer_params[tid] = value


@wp.kernel
def project_base_delta_optimizer_params_kernel(
    optimizer_params: wp.array(dtype=float),
    min_value: float,
    max_value: float,
    left_right_delta_sum_zero: int,
):
    base = optimizer_params[0]
    delta_left = optimizer_params[1]
    delta_right = optimizer_params[2]
    if base < min_value:
        base = min_value
    if base > max_value:
        base = max_value

    if left_right_delta_sum_zero != 0:
        delta_mean = 0.5 * (delta_left + delta_right)
        delta_left = delta_left - delta_mean
        delta_right = delta_right - delta_mean
        delta = 0.5 * (delta_left - delta_right)
        allowed_delta_abs = base - min_value
        right_allowed = max_value - base
        if right_allowed < allowed_delta_abs:
            allowed_delta_abs = right_allowed
        if allowed_delta_abs < 0.0:
            allowed_delta_abs = 0.0
        if delta < -allowed_delta_abs:
            delta = -allowed_delta_abs
        if delta > allowed_delta_abs:
            delta = allowed_delta_abs
        optimizer_params[0] = base
        optimizer_params[1] = delta
        optimizer_params[2] = -delta
        return

    mu_left = base + delta_left
    mu_right = base + delta_right
    if mu_left < min_value:
        mu_left = min_value
    if mu_left > max_value:
        mu_left = max_value
    if mu_right < min_value:
        mu_right = min_value
    if mu_right > max_value:
        mu_right = max_value

    optimizer_params[0] = base
    optimizer_params[1] = mu_left - base
    optimizer_params[2] = mu_right - base


@wp.kernel
def accumulate_active_mu_features_kernel(
    active_point_friction: wp.array(dtype=float),
    mu_feature_weights: wp.array(dtype=float),
    mu_features: wp.array(dtype=float),
):
    tid = wp.tid()
    mu = active_point_friction[tid]
    weight_base = tid * 3
    wp.atomic_add(mu_features, 0, mu_feature_weights[weight_base + 0] * mu)
    wp.atomic_add(mu_features, 1, mu_feature_weights[weight_base + 1] * mu)
    wp.atomic_add(mu_features, 2, mu_feature_weights[weight_base + 2] * mu)


@wp.kernel
def accumulate_optimizer_mu_features_kernel(
    active_param_positions: wp.array(dtype=wp.int32),
    optimizer_params: wp.array(dtype=float),
    parameterization_id: int,
    mu_feature_weights: wp.array(dtype=float),
    mu_features: wp.array(dtype=float),
):
    tid = wp.tid()
    param_pos = active_param_positions[tid]
    mu = optimizer_params[param_pos]
    if parameterization_id == 3:
        mu = optimizer_params[0] + optimizer_params[1 + param_pos]

    weight_base = tid * 3
    wp.atomic_add(mu_features, 0, mu_feature_weights[weight_base + 0] * mu)
    wp.atomic_add(mu_features, 1, mu_feature_weights[weight_base + 1] * mu)
    wp.atomic_add(mu_features, 2, mu_feature_weights[weight_base + 2] * mu)


@wp.kernel
def scatter_optimizer_point_friction_kernel(
    active_indices: wp.array(dtype=wp.int32),
    active_param_positions: wp.array(dtype=wp.int32),
    optimizer_params: wp.array(dtype=float),
    parameterization_id: int,
    full_point_friction: wp.array(dtype=float),
):
    tid = wp.tid()
    point_idx = active_indices[tid]
    param_pos = active_param_positions[tid]
    mu = optimizer_params[param_pos]
    if parameterization_id == 3:
        mu = optimizer_params[0] + optimizer_params[1 + param_pos]
    full_point_friction[point_idx] = mu


@wp.kernel
def residual_mlp_layer0_kernel(
    step_idx: int,
    box_body_ids: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    step_forces: wp.array(dtype=wp.vec3),
    force_point_offsets_local: wp.array(dtype=wp.vec3),
    trajectory_step_counts: wp.array(dtype=wp.int32),
    batch_size: int,
    max_steps: int,
    feature_mean: wp.array(dtype=float),
    feature_inv_std: wp.array(dtype=float),
    mu_features: wp.array(dtype=float),
    weights: wp.array(dtype=float),
    bias: wp.array(dtype=float),
    hidden0: wp.array(dtype=float),
):
    tid = wp.tid()
    batch_idx = tid // HIDDEN0_DIM
    hidden_idx = tid - batch_idx * HIDDEN0_DIM
    out_idx = (step_idx * batch_size + batch_idx) * HIDDEN0_DIM + hidden_idx

    if step_idx >= trajectory_step_counts[batch_idx]:
        hidden0[out_idx] = 0.0
        return

    body_id = box_body_ids[batch_idx]
    pose = body_q[body_id]
    qd = body_qd[body_id]
    force_world = step_forces[batch_idx * max_steps + step_idx]
    point_offset_local = force_point_offsets_local[batch_idx]

    acc = bias[hidden_idx]
    for feature_idx in range(INPUT_DIM):
        feature = _feature_value(feature_idx, pose, qd, force_world, point_offset_local, mu_features)
        normalized = _normalized_feature(feature, feature_idx, feature_mean, feature_inv_std)
        acc = acc + weights[hidden_idx * INPUT_DIM + feature_idx] * normalized

    hidden0[out_idx] = wp.tanh(acc)


@wp.kernel
def residual_mlp_layer1_kernel(
    step_idx: int,
    trajectory_step_counts: wp.array(dtype=wp.int32),
    batch_size: int,
    weights: wp.array(dtype=float),
    bias: wp.array(dtype=float),
    hidden0: wp.array(dtype=float),
    hidden1: wp.array(dtype=float),
):
    tid = wp.tid()
    batch_idx = tid // HIDDEN1_DIM
    hidden_idx = tid - batch_idx * HIDDEN1_DIM
    out_idx = (step_idx * batch_size + batch_idx) * HIDDEN1_DIM + hidden_idx

    if step_idx >= trajectory_step_counts[batch_idx]:
        hidden1[out_idx] = 0.0
        return

    in_base = (step_idx * batch_size + batch_idx) * HIDDEN0_DIM
    acc = bias[hidden_idx]
    for input_idx in range(HIDDEN0_DIM):
        acc = acc + weights[hidden_idx * HIDDEN0_DIM + input_idx] * hidden0[in_base + input_idx]
    hidden1[out_idx] = wp.tanh(acc)


@wp.kernel
def residual_mlp_layer2_kernel(
    step_idx: int,
    trajectory_step_counts: wp.array(dtype=wp.int32),
    batch_size: int,
    weights: wp.array(dtype=float),
    bias: wp.array(dtype=float),
    hidden1: wp.array(dtype=float),
    hidden2: wp.array(dtype=float),
):
    tid = wp.tid()
    batch_idx = tid // HIDDEN2_DIM
    hidden_idx = tid - batch_idx * HIDDEN2_DIM
    out_idx = (step_idx * batch_size + batch_idx) * HIDDEN2_DIM + hidden_idx

    if step_idx >= trajectory_step_counts[batch_idx]:
        hidden2[out_idx] = 0.0
        return

    in_base = (step_idx * batch_size + batch_idx) * HIDDEN1_DIM
    acc = bias[hidden_idx]
    for input_idx in range(HIDDEN1_DIM):
        acc = acc + weights[hidden_idx * HIDDEN1_DIM + input_idx] * hidden1[in_base + input_idx]
    hidden2[out_idx] = wp.tanh(acc)


@wp.kernel
def residual_mlp_output_kernel(
    step_idx: int,
    trajectory_step_counts: wp.array(dtype=wp.int32),
    batch_size: int,
    weights: wp.array(dtype=float),
    bias: wp.array(dtype=float),
    output_scales: wp.array(dtype=float),
    hidden2: wp.array(dtype=float),
    residuals: wp.array(dtype=float),
):
    tid = wp.tid()
    batch_idx = tid // OUTPUT_DIM
    output_idx = tid - batch_idx * OUTPUT_DIM
    residual_idx = (step_idx * batch_size + batch_idx) * OUTPUT_DIM + output_idx

    if step_idx >= trajectory_step_counts[batch_idx]:
        residuals[residual_idx] = 0.0
        return

    in_base = (step_idx * batch_size + batch_idx) * HIDDEN2_DIM
    acc = bias[output_idx]
    for input_idx in range(HIDDEN2_DIM):
        acc = acc + weights[output_idx * HIDDEN2_DIM + input_idx] * hidden2[in_base + input_idx]
    residuals[residual_idx] = output_scales[output_idx] * wp.tanh(acc)


@wp.kernel
def apply_residual_planar_dynamics_kernel(
    step_idx: int,
    box_body_ids: wp.array(dtype=wp.int32),
    current_body_q: wp.array(dtype=wp.transform),
    sim_body_q: wp.array(dtype=wp.transform),
    sim_body_qd: wp.array(dtype=wp.spatial_vector),
    residuals: wp.array(dtype=float),
    trajectory_step_counts: wp.array(dtype=wp.int32),
    batch_size: int,
    residual_output_mode: int,
    dt: float,
    pred_body_q: wp.array(dtype=wp.transform),
    pred_body_qd: wp.array(dtype=wp.spatial_vector),
):
    batch_idx = wp.tid()
    body_id = box_body_ids[batch_idx]

    sim_pose = sim_body_q[body_id]
    sim_qd = sim_body_qd[body_id]

    if step_idx >= trajectory_step_counts[batch_idx]:
        pred_body_q[body_id] = sim_pose
        pred_body_qd[body_id] = sim_qd
        return

    residual_base = (step_idx * batch_size + batch_idx) * OUTPUT_DIM
    delta_body_x = residuals[residual_base + 0]
    delta_body_y = residuals[residual_base + 1]
    delta_z = residuals[residual_base + 2]

    current_pose = current_body_q[body_id]
    current_yaw = _quat_yaw_xyzw(wp.transform_get_rotation(current_pose))
    c = wp.cos(current_yaw)
    s = wp.sin(current_yaw)
    delta_world_x = c * delta_body_x - s * delta_body_y
    delta_world_y = s * delta_body_x + c * delta_body_y

    sim_pos = wp.transform_get_translation(sim_pose)
    sim_quat = wp.transform_get_rotation(sim_pose)
    sim_linear_velocity = wp.spatial_top(sim_qd)
    sim_angular_velocity = wp.spatial_bottom(sim_qd)

    position_scale = dt
    velocity_scale = 1.0
    if residual_output_mode == RESIDUAL_OUTPUT_MODE_ACCELERATION:
        position_scale = 0.5 * dt * dt
        velocity_scale = dt

    pred_pos = wp.vec3(
        sim_pos[0] + position_scale * delta_world_x,
        sim_pos[1] + position_scale * delta_world_y,
        sim_pos[2],
    )

    yaw_delta = position_scale * delta_z
    half_yaw = 0.5 * yaw_delta
    yaw_delta_quat = wp.quat(0.0, 0.0, wp.sin(half_yaw), wp.cos(half_yaw))
    pred_quat = wp.normalize(_quat_mul_xyzw(yaw_delta_quat, sim_quat))

    pred_linear_velocity = wp.vec3(
        sim_linear_velocity[0] + velocity_scale * delta_world_x,
        sim_linear_velocity[1] + velocity_scale * delta_world_y,
        sim_linear_velocity[2],
    )
    pred_angular_velocity = wp.vec3(
        sim_angular_velocity[0],
        sim_angular_velocity[1],
        sim_angular_velocity[2] + velocity_scale * delta_z,
    )

    pred_body_q[body_id] = wp.transform(pred_pos, pred_quat)
    pred_body_qd[body_id] = wp.spatial_vector(pred_linear_velocity, pred_angular_velocity)


@wp.kernel
def accumulate_residual_frame_loss_kernel(
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
    target_point_position = _rotate_point_by_quat_xyzw(target_quat, local_point) + target_position
    position_delta = world_position - target_point_position
    wp.atomic_add(position_loss, batch_idx, frame_scale * point_scale * wp.dot(position_delta, position_delta))

    if point_idx != 0:
        return

    pred_yaw = _quat_yaw_xyzw(wp.transform_get_rotation(pose))
    target_quat_for_yaw = wp.quat(target_quat[0], target_quat[1], target_quat[2], target_quat[3])
    target_yaw = _quat_yaw_xyzw(target_quat_for_yaw)
    yaw_delta = _wrap_angle(pred_yaw - target_yaw)
    wp.atomic_add(orientation_loss, batch_idx, frame_scale * yaw_delta * yaw_delta)

    pred_qd = body_qd[body_id]
    pred_linear_velocity = wp.spatial_top(pred_qd)
    pred_angular_velocity = wp.spatial_bottom(pred_qd)
    target_linear = target_linear_velocity[target_offset]
    target_angular = target_angular_velocity[target_offset]

    lin_dx = pred_linear_velocity[0] - target_linear[0]
    lin_dy = pred_linear_velocity[1] - target_linear[1]
    ang_dz = pred_angular_velocity[2] - target_angular[2]

    wp.atomic_add(linear_velocity_loss, batch_idx, frame_scale * (lin_dx * lin_dx + lin_dy * lin_dy))
    wp.atomic_add(angular_velocity_loss, batch_idx, frame_scale * ang_dz * ang_dz)


@wp.kernel
def combine_residual_loss_components_kernel(
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
def sum_loss_kernel(
    losses: wp.array(dtype=float),
    scale: float,
    batch_loss: wp.array(dtype=float),
):
    tid = wp.tid()
    wp.atomic_add(batch_loss, 0, scale * losses[tid])


@wp.kernel
def accumulate_residual_regularization_kernel(
    residuals: wp.array(dtype=float),
    trajectory_step_counts: wp.array(dtype=wp.int32),
    batch_size: int,
    max_steps: int,
    residual_weight: float,
    smoothness_weight: float,
    batch_loss_scale: float,
    batch_loss: wp.array(dtype=float),
    residual_norm_mean: wp.array(dtype=float),
    residual_energy_mean: wp.array(dtype=float),
    residual_norm_max: wp.array(dtype=float),
):
    tid = wp.tid()
    batch_idx = tid // max_steps
    step_idx = tid - batch_idx * max_steps
    if step_idx >= trajectory_step_counts[batch_idx]:
        return

    residual_base = (step_idx * batch_size + batch_idx) * OUTPUT_DIM
    rx = residuals[residual_base + 0]
    ry = residuals[residual_base + 1]
    rz = residuals[residual_base + 2]
    residual_sq = rx * rx + ry * ry + rz * rz
    residual_norm = wp.sqrt(residual_sq + 1.0e-12)

    valid_steps = wp.max(trajectory_step_counts[batch_idx], 1)
    stat_scale = 1.0 / float(valid_steps)
    wp.atomic_add(residual_norm_mean, batch_idx, stat_scale * residual_norm)
    wp.atomic_add(residual_energy_mean, batch_idx, stat_scale * residual_sq)
    wp.atomic_max(residual_norm_max, batch_idx, residual_norm)

    wp.atomic_add(batch_loss, 0, batch_loss_scale * residual_weight * residual_sq)

    if step_idx > 0:
        prev_base = ((step_idx - 1) * batch_size + batch_idx) * OUTPUT_DIM
        drx = rx - residuals[prev_base + 0]
        dry = ry - residuals[prev_base + 1]
        drz = rz - residuals[prev_base + 2]
        smooth_sq = drx * drx + dry * dry + drz * drz
        wp.atomic_add(batch_loss, 0, batch_loss_scale * smoothness_weight * smooth_sq)


@wp.kernel
def accumulate_grad_norm_kernel(
    grads: wp.array(dtype=float),
    grad_norm_sq: wp.array(dtype=wp.float64),
    nonfinite_count: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    grad = grads[tid]
    if not wp.isfinite(grad) or grad > 1.0e38 or grad < -1.0e38:
        wp.atomic_add(nonfinite_count, 0, 1)
        return
    grad64 = wp.float64(grad)
    wp.atomic_add(grad_norm_sq, 0, grad64 * grad64)


@wp.kernel
def adam_update_kernel(
    params: wp.array(dtype=float),
    grads: wp.array(dtype=float),
    first_moment: wp.array(dtype=wp.float64),
    second_moment: wp.array(dtype=wp.float64),
    grad_scale: wp.float64,
    learning_rate: wp.float64,
    beta1: wp.float64,
    beta2: wp.float64,
    eps: wp.float64,
    bias_correction1: wp.float64,
    bias_correction2: wp.float64,
):
    tid = wp.tid()
    one = wp.float64(1.0)
    grad = wp.float64(grads[tid]) * grad_scale
    moment_1 = beta1 * first_moment[tid] + (one - beta1) * grad
    moment_2 = beta2 * second_moment[tid] + (one - beta2) * grad * grad
    first_hat = moment_1 / bias_correction1
    second_hat = moment_2 / bias_correction2
    params[tid] = wp.float32(wp.float64(params[tid]) - learning_rate * first_hat / (wp.sqrt(second_hat) + eps))
    first_moment[tid] = moment_1
    second_moment[tid] = moment_2
