from __future__ import annotations

import argparse
import time

import numpy as np
import warp as wp

from mujoco_contact_friction_fit_utils import (
    compute_active_contact_point_indices,
    load_mujoco_trajectories,
)
from mujoco_contact_friction_fit_wandb import build_wandb_log_payload, init_wandb
from fit_mujoco_contact_point_friction_io import (
    DEFAULT_TRAIN_BATCH_SIZE,
    parse_args,
    save_contact_friction_point_cloud,
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


def save_training_checkpoint(
    *,
    checkpoint_path,
    iteration: int,
    active_indices: np.ndarray,
    active_params: np.ndarray,
    adam_m: np.ndarray,
    adam_v: np.ndarray,
    adam_step: np.ndarray,
    best_loss: float,
    best_active_params: np.ndarray,
    loss_history: list[float],
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        checkpoint_path,
        iteration=np.asarray(iteration, dtype=np.int32),
        active_indices=np.asarray(active_indices, dtype=np.int32),
        active_params=np.asarray(active_params, dtype=np.float32),
        adam_m=np.asarray(adam_m, dtype=np.float64),
        adam_v=np.asarray(adam_v, dtype=np.float64),
        adam_step=np.asarray(adam_step, dtype=np.int32),
        best_loss=np.asarray(best_loss, dtype=np.float64),
        best_active_params=np.asarray(best_active_params, dtype=np.float32),
        loss_history=np.asarray(loss_history, dtype=np.float32),
        rng_state=np.asarray(rng.bit_generator.state, dtype=object),
        trajectory_npz_path=np.asarray(str(args.trajectory_npz.resolve())),
        max_steps=np.asarray(-1 if args.max_steps is None else int(args.max_steps), dtype=np.int32),
        max_trajectories=np.asarray(-1 if args.max_trajectories is None else int(args.max_trajectories), dtype=np.int32),
    )


def resolve_checkpoint_point_cloud_path(args: argparse.Namespace, iteration: int):
    if args.checkpoint_point_cloud_dir is None:
        point_cloud_dir = args.checkpoint_path.parent / f"{args.checkpoint_path.stem}_point_clouds"
    else:
        point_cloud_dir = args.checkpoint_point_cloud_dir
    return point_cloud_dir / f"iter_{int(iteration):06d}.ply"


def should_save_iteration_checkpoint(args: argparse.Namespace, iteration: int) -> bool:
    checkpoint_every = int(args.checkpoint_every)
    return checkpoint_every > 0 and (iteration % checkpoint_every == 0 or iteration == int(args.opt_iters))


def save_iteration_checkpoint_and_point_cloud(
    *,
    args: argparse.Namespace,
    iteration: int,
    active_indices: np.ndarray,
    active_params: np.ndarray,
    adam_m: np.ndarray,
    adam_v: np.ndarray,
    adam_step: np.ndarray,
    best_loss: float,
    best_active_params: np.ndarray,
    loss_history: list[float],
    rng: np.random.Generator,
    local_surface_points: np.ndarray,
    point_cloud_color_min: float,
    point_cloud_color_max: float,
) -> None:
    save_training_checkpoint(
        checkpoint_path=args.checkpoint_path,
        iteration=iteration,
        active_indices=active_indices,
        active_params=active_params,
        adam_m=adam_m,
        adam_v=adam_v,
        adam_step=adam_step,
        best_loss=best_loss,
        best_active_params=best_active_params,
        loss_history=loss_history,
        rng=rng,
        args=args,
    )
    checkpoint_point_cloud_path = resolve_checkpoint_point_cloud_path(args, iteration)
    checkpoint_point_friction = np.full(
        len(local_surface_points),
        float(args.point_friction),
        dtype=np.float32,
    )
    checkpoint_point_friction[active_indices] = active_params
    save_contact_friction_point_cloud(
        local_surface_points=local_surface_points,
        point_friction=checkpoint_point_friction,
        output_path=checkpoint_point_cloud_path,
        active_indices=active_indices,
        color_min=point_cloud_color_min,
        color_max=point_cloud_color_max,
    )
    log_message(f"checkpoint_point_cloud_written_to={checkpoint_point_cloud_path.resolve()}")


def load_training_checkpoint(
    *,
    checkpoint_path,
    active_indices: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, list[float]]:
    with np.load(checkpoint_path, allow_pickle=True) as data:
        checkpoint_active_indices = np.asarray(data["active_indices"], dtype=np.int32)
        if checkpoint_active_indices.shape != active_indices.shape or not np.array_equal(checkpoint_active_indices, active_indices):
            raise ValueError(
                f"{checkpoint_path} active point indices do not match the current run. "
                "Use matching trajectory/model/contact-mask settings or start without --resume-checkpoint."
            )

        iteration = int(np.asarray(data["iteration"]).item())
        active_params = np.asarray(data["active_params"], dtype=np.float32)
        adam_m = np.asarray(data["adam_m"], dtype=np.float64)
        adam_v = np.asarray(data["adam_v"], dtype=np.float64)
        if "adam_step" in data.files:
            adam_step = np.asarray(data["adam_step"], dtype=np.int32)
        else:
            adam_step = np.zeros_like(active_indices, dtype=np.int32)
        best_loss = float(np.asarray(data["best_loss"]).item())
        best_active_params = np.asarray(data["best_active_params"], dtype=np.float32)
        loss_history = [float(value) for value in np.asarray(data["loss_history"], dtype=np.float32)]

        expected_shape = active_indices.shape
        for name, values in (
            ("active_params", active_params),
            ("adam_m", adam_m),
            ("adam_v", adam_v),
            ("adam_step", adam_step),
            ("best_active_params", best_active_params),
        ):
            if values.shape != expected_shape:
                raise ValueError(f"{checkpoint_path} {name} has shape {values.shape}, expected {expected_shape}")

        rng_state = data["rng_state"].item()
        rng.bit_generator.state = rng_state

    return iteration, active_params, adam_m, adam_v, adam_step, best_loss, best_active_params, loss_history


def compute_batch_active_point_indices(
    trajectory_active_indices_by_idx: list[np.ndarray],
    batch_indices: np.ndarray,
    point_count: int,
) -> np.ndarray:
    batch_active_mask = np.zeros(int(point_count), dtype=bool)
    for trajectory_idx in batch_indices:
        batch_active_mask[trajectory_active_indices_by_idx[int(trajectory_idx)]] = True
    return np.flatnonzero(batch_active_mask).astype(np.int32)


def resolve_point_cloud_color_bounds(args: argparse.Namespace) -> tuple[float, float]:
    color_min = args.point_cloud_color_min
    color_max = args.point_cloud_color_max
    if color_min is None:
        color_min = float(args.point_friction) - 0.005
    if color_max is None:
        color_max = float(args.point_friction) + 0.005
    if float(color_max) <= float(color_min):
        raise ValueError(
            f"--point-cloud-color-max must be greater than --point-cloud-color-min, got {color_max} <= {color_min}"
        )
    return float(color_min), float(color_max)


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
    accumulate_velocity_loss: int,
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
    trajectory_active_indices_by_idx: list[np.ndarray] = []
    for trajectory_idx, trajectory in enumerate(trajectories, start=1):
        trajectory_active_indices = compute_active_contact_point_indices(
            local_surface_points=diff_scene.local_surface_points_np,
            trajectory=trajectory,
            floor_top_z=diff_scene.floor_top_z,
            contact_threshold=float(args.contact_mask_threshold),
        )
        trajectory_active_indices_by_idx.append(trajectory_active_indices)
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
    point_cloud_color_min, point_cloud_color_max = resolve_point_cloud_color_bounds(args)
    log_message(
        f"point cloud color range=[{point_cloud_color_min:.6f}, {point_cloud_color_max:.6f}]"
    )

    wandb_run = init_wandb(args, trajectory_collection, active_indices)
    if wandb_run is not None:
        log_message(
            f"W&B enabled | project={args.wandb_project} | "
            f"run={wandb_run.name} | mode={args.wandb_mode}"
        )

    active_params_np = np.full(len(active_indices), float(args.point_friction), dtype=np.float32)
    active_param_lookup = np.full(len(diff_scene.local_surface_points_np), -1, dtype=np.int32)
    active_param_lookup[active_indices] = np.arange(len(active_indices), dtype=np.int32)
    adam_m_np = np.zeros(len(active_indices), dtype=np.float64)
    adam_v_np = np.zeros(len(active_indices), dtype=np.float64)
    adam_step_np = np.zeros(len(active_indices), dtype=np.int32)
    loss_history: list[float] = []
    best_loss = float("inf")
    best_active_params = active_params_np.copy()
    rng = np.random.default_rng(int(args.seed))
    start_iteration = 1
    if args.resume_checkpoint is not None:
        (
            resume_iteration,
            active_params_np,
            adam_m_np,
            adam_v_np,
            adam_step_np,
            best_loss,
            best_active_params,
            loss_history,
        ) = load_training_checkpoint(
            checkpoint_path=args.resume_checkpoint,
            active_indices=active_indices,
            rng=rng,
        )
        start_iteration = resume_iteration + 1
        log_message(
            f"resumed checkpoint {args.resume_checkpoint.resolve()} "
            f"at iteration={resume_iteration} best_loss={best_loss:.6f}"
        )
    device = str(diff_scene.torch_device)
    active_indices_wp = wp.array(active_indices, dtype=wp.int32, device=device)
    active_params = wp.array(active_params_np, dtype=wp.float32, device=device)
    adam_m = wp.array(adam_m_np, dtype=wp.float64, device=device)
    adam_v = wp.array(adam_v_np, dtype=wp.float64, device=device)
    adam_step = wp.array(adam_step_np, dtype=wp.int32, device=device)
    beta1_power = wp.array(
        np.power(float(args.adam_beta1), adam_step_np.astype(np.float64)),
        dtype=wp.float64,
        device=device,
    )
    beta2_power = wp.array(
        np.power(float(args.adam_beta2), adam_step_np.astype(np.float64)),
        dtype=wp.float64,
        device=device,
    )

    def save_iteration_checkpoint(iteration: int) -> None:
        save_iteration_checkpoint_and_point_cloud(
            args=args,
            iteration=iteration,
            active_indices=active_indices,
            active_params=active_params_np,
            adam_m=adam_m_np,
            adam_v=adam_v_np,
            adam_step=adam_step_np,
            best_loss=best_loss,
            best_active_params=best_active_params,
            loss_history=loss_history,
            rng=rng,
            local_surface_points=diff_scene.local_surface_points_np,
            point_cloud_color_min=point_cloud_color_min,
            point_cloud_color_max=point_cloud_color_max,
        )

    try:
        for iteration in range(start_iteration, max(int(args.opt_iters), 0) + 1):
            iteration_start = time.time()
            batch_indices = sample_training_batch_indices(len(trajectories), batch_size, rng)
            batch_trajectories = [trajectories[int(idx)] for idx in batch_indices]
            batch_active_indices = compute_batch_active_point_indices(
                trajectory_active_indices_by_idx,
                batch_indices,
                len(diff_scene.local_surface_points_np),
            )
            if len(batch_active_indices) == 0:
                if should_save_iteration_checkpoint(args, iteration):
                    save_iteration_checkpoint(iteration)
                    log_message(f"iter={iteration:04d} skipped=no_batch_active_points checkpoint_saved=1")
                continue
            batch_active_param_positions = active_param_lookup[batch_active_indices]
            if np.any(batch_active_param_positions < 0):
                raise RuntimeError("Batch active points must be a subset of the global active point mask.")

            batch_active_param_positions_wp = wp.array(
                batch_active_param_positions,
                dtype=wp.int32,
                device=device,
            )
            buffers = build_batched_optimization_buffers(diff_scene, batch_trajectories, args, batch_active_indices)
            buffers.full_point_friction.assign(buffers.inactive_point_friction_np)
            wp.launch(
                scatter_active_point_friction_kernel,
                dim=len(active_indices),
                inputs=[active_indices_wp, active_params, buffers.full_point_friction],
                device=diff_scene.model.device,
            )
            wp.launch(
                gather_active_point_friction_kernel,
                dim=len(batch_active_indices),
                inputs=[active_params, batch_active_param_positions_wp, buffers.active_point_friction],
                device=diff_scene.model.device,
            )
            clear_batched_optimization_grads(buffers)

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
                if should_save_iteration_checkpoint(args, iteration):
                    save_iteration_checkpoint(iteration)
                    log_message(f"iter={iteration:04d} skipped=missing_grad checkpoint_saved=1")
                continue

            grad_norm_sq = wp.zeros(1, dtype=wp.float64, device=device)
            nonfinite_grad_count = wp.zeros(1, dtype=wp.int32, device=device)
            wp.launch(
                accumulate_gradient_norm_sq_kernel,
                dim=len(batch_active_indices),
                inputs=[buffers.active_point_friction.grad, grad_norm_sq, nonfinite_grad_count],
                device=diff_scene.model.device,
            )
            if int(nonfinite_grad_count.numpy()[0]) != 0:
                tape.zero()
                if should_save_iteration_checkpoint(args, iteration):
                    save_iteration_checkpoint(iteration)
                    log_message(f"iter={iteration:04d} skipped=nonfinite_grad checkpoint_saved=1")
                continue

            good_buffer_count = len(batch_trajectories)
            loss_value = float(np.mean(buffers.loss.numpy()))
            raw_position_loss_value = float(np.mean(buffers.position_loss.numpy()))
            raw_orientation_loss_value = float(np.mean(buffers.orientation_loss.numpy()))
            raw_linear_velocity_loss_value = float(np.mean(buffers.linear_velocity_loss.numpy()))
            raw_angular_velocity_loss_value = float(np.mean(buffers.angular_velocity_loss.numpy()))
            position_loss_value = float(args.position_loss_weight) * raw_position_loss_value
            orientation_loss_value = float(args.orientation_loss_weight) * raw_orientation_loss_value
            linear_velocity_loss_value = float(args.linear_velocity_loss_weight) * raw_linear_velocity_loss_value
            angular_velocity_loss_value = float(args.angular_velocity_loss_weight) * raw_angular_velocity_loss_value
            raw_grad_norm = float(np.sqrt(max(float(grad_norm_sq.numpy()[0]), 0.0)))
            grad_clip_norm = args.grad_clip_norm
            if grad_clip_norm is None or float(grad_clip_norm) <= 0.0 or raw_grad_norm <= float(grad_clip_norm):
                grad_clip_scale = 1.0
                clipped_grad_norm = raw_grad_norm
            else:
                grad_clip_scale = float(grad_clip_norm) / max(raw_grad_norm, 1.0e-30)
                clipped_grad_norm = float(grad_clip_norm)
            beta1 = float(args.adam_beta1)
            beta2 = float(args.adam_beta2)
            wp.launch(
                sparse_adam_update_clipped_kernel,
                dim=len(batch_active_indices),
                inputs=[
                    active_params,
                    buffers.active_point_friction.grad,
                    batch_active_param_positions_wp,
                    adam_m,
                    adam_v,
                    adam_step,
                    beta1_power,
                    beta2_power,
                    np.float64(grad_clip_scale),
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
            adam_step_np = adam_step.numpy()
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

            if should_save_iteration_checkpoint(args, iteration):
                save_iteration_checkpoint(iteration)

            if wandb_run is not None:
                # Log the clipped batch gradient so the scalar stats match the optimizer update.
                logged_grad_np = np.asarray(buffers.active_point_friction.grad.numpy(), dtype=np.float64)
                if grad_clip_scale != 1.0:
                    logged_grad_np = logged_grad_np * grad_clip_scale
                log_payload = build_wandb_log_payload(
                    loss_value=loss_value,
                    position_loss_value=position_loss_value,
                    orientation_loss_value=orientation_loss_value,
                    linear_velocity_loss_value=linear_velocity_loss_value,
                    angular_velocity_loss_value=angular_velocity_loss_value,
                    raw_position_loss_value=raw_position_loss_value,
                    raw_orientation_loss_value=raw_orientation_loss_value,
                    raw_linear_velocity_loss_value=raw_linear_velocity_loss_value,
                    raw_angular_velocity_loss_value=raw_angular_velocity_loss_value,
                    grad_value=logged_grad_np,
                    active_params=active_params_np,
                    active_indices=active_indices,
                    grad_norm_value=clipped_grad_norm,
                )
                log_payload["params/batch_active_contact_point_count"] = float(len(batch_active_indices))
                log_payload["train/raw_grad_norm"] = float(raw_grad_norm)
                log_payload["train/clipped_grad_norm"] = float(clipped_grad_norm)
                log_payload["train/grad_clip_scale"] = float(grad_clip_scale)
                wandb_run.log(log_payload, step=iteration)

            if iteration == 1 or iteration % max(int(args.log_every), 1) == 0 or iteration == int(args.opt_iters):
                log_message(
                    f"iter={iteration:04d} loss={loss_value:.6f} "
                    f"pos={position_loss_value:.6f} "
                    f"ori={orientation_loss_value:.6f} "
                    f"linvel={linear_velocity_loss_value:.6f} "
                    f"angvel={angular_velocity_loss_value:.6f} "
                    f"grad_norm={clipped_grad_norm:.6f} "
                    f"raw_grad_norm={raw_grad_norm:.6f} "
                    f"clip_scale={grad_clip_scale:.6g} "
                    f"mu_min={float(active_params_np.min()):.6f} "
                    f"mu_max={float(active_params_np.max()):.6f} "
                    f"batch_active_points={len(batch_active_indices)}/{len(active_indices)} "
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
        final_position_loss_contribution = float(args.position_loss_weight) * final_position_loss
        final_orientation_loss_contribution = float(args.orientation_loss_weight) * final_orientation_loss
        final_linear_velocity_loss_contribution = float(args.linear_velocity_loss_weight) * final_linear_velocity_loss
        final_angular_velocity_loss_contribution = float(args.angular_velocity_loss_weight) * final_angular_velocity_loss

        assert_array_finite(
            "best_active_params",
            best_active_params,
            context="final export",
        )
        export_contact_friction_outputs(
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
            wandb_run.summary["final_position_loss"] = float(final_position_loss_contribution)
            wandb_run.summary["final_orientation_loss"] = float(final_orientation_loss_contribution)
            wandb_run.summary["final_linear_velocity_loss"] = float(final_linear_velocity_loss_contribution)
            wandb_run.summary["final_angular_velocity_loss"] = float(final_angular_velocity_loss_contribution)
            wandb_run.summary["final_raw_position_loss"] = float(final_position_loss)
            wandb_run.summary["final_raw_orientation_loss"] = float(final_orientation_loss)
            wandb_run.summary["final_raw_linear_velocity_loss"] = float(final_linear_velocity_loss)
            wandb_run.summary["final_raw_angular_velocity_loss"] = float(final_angular_velocity_loss)
            wandb_run.summary["mu_mean"] = float(best_active_params.mean())
            wandb_run.summary["mu_std"] = float(best_active_params.std())
            wandb_run.summary["mu_min"] = float(best_active_params.min())
            wandb_run.summary["mu_max"] = float(best_active_params.max())
            wandb_run.summary["results_path"] = str(args.results_path.resolve())
            wandb_run.summary["point_cloud_path"] = str(args.point_cloud_path.resolve())
            if args.scene_usd_path is not None:
                wandb_run.summary["scene_usd_path"] = str(args.scene_usd_path.resolve())

        log_message(f"trajectory={args.trajectory_npz.resolve()}")
        log_message(f"trajectory_source_type={trajectory_collection.source_type}")
        log_message(f"trajectory_count={len(trajectories)}")
        log_message(f"max_steps={trajectory_collection.max_steps} dt={representative_trajectory.timestep:.6f}")
        log_message(f"surface_points={len(diff_scene.local_surface_points_np)} active_contact_points={len(active_indices)}")
        log_message(f"final_loss={final_loss:.6f}")
        log_message(f"final_position_loss={final_position_loss_contribution:.6f}")
        log_message(f"final_orientation_loss={final_orientation_loss_contribution:.6f}")
        log_message(f"final_linear_velocity_loss={final_linear_velocity_loss_contribution:.6f}")
        log_message(f"final_angular_velocity_loss={final_angular_velocity_loss_contribution:.6f}")
        log_message(f"results_written_to={args.results_path.resolve()}")
        log_message(f"point_cloud_written_to={args.point_cloud_path.resolve()}")
        if args.scene_usd_path is not None:
            log_message(f"scene_usd_written_to={args.scene_usd_path.resolve()}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
