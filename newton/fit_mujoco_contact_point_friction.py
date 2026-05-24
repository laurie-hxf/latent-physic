from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

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
    optimizer_params: np.ndarray,
    adam_m: np.ndarray,
    adam_v: np.ndarray,
    adam_step: np.ndarray,
    best_loss: float,
    best_active_params: np.ndarray,
    best_optimizer_params: np.ndarray,
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
        optimizer_params=np.asarray(optimizer_params, dtype=np.float32),
        adam_m=np.asarray(adam_m, dtype=np.float64),
        adam_v=np.asarray(adam_v, dtype=np.float64),
        adam_step=np.asarray(adam_step, dtype=np.int32),
        best_loss=np.asarray(best_loss, dtype=np.float64),
        best_active_params=np.asarray(best_active_params, dtype=np.float32),
        best_optimizer_params=np.asarray(best_optimizer_params, dtype=np.float32),
        loss_history=np.asarray(loss_history, dtype=np.float32),
        rng_state=np.asarray(rng.bit_generator.state, dtype=object),
        friction_parameterization=np.asarray(str(args.friction_parameterization)),
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


def run_post_training_eval(args: argparse.Namespace) -> Path:
    eval_script = Path(__file__).resolve().parent.parent / "visualization" / "evaluate_mujoco_contact_friction_experiment.py"
    eval_output_dir = args.eval_output_root / args.experiment_dir.name
    cmd = [
        sys.executable,
        str(eval_script),
        "--experiment-dir",
        str(args.experiment_dir),
        "--eval-dataset",
        str(args.eval_dataset),
        "--output-root",
        str(args.eval_output_root),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--position-loss-weight",
        str(args.position_loss_weight),
        "--orientation-loss-weight",
        str(args.orientation_loss_weight),
        "--linear-velocity-loss-weight",
        str(args.linear_velocity_loss_weight),
        "--angular-velocity-loss-weight",
        str(args.angular_velocity_loss_weight),
        "--point-position-loss-reduction",
        str(args.point_position_loss_reduction),
        "--solver-iterations",
        str(args.solver_iterations),
        "--contact-stiffness",
        str(args.contact_stiffness),
        "--contact-damping",
        str(args.contact_damping),
        "--contact-margin",
        str(args.contact_margin),
        "--friction-contact-threshold",
        str(args.friction_contact_threshold),
        "--contact-mask-threshold",
        str(args.contact_mask_threshold),
        "--friction-regularization",
        str(args.friction_regularization),
    ]
    if args.device is not None:
        cmd.extend(["--device", str(args.device)])
    if args.max_steps is not None:
        cmd.extend(["--max-steps", str(args.max_steps)])
    if args.eval_replay_limit is not None:
        cmd.extend(["--replay-limit", str(args.eval_replay_limit)])
    if args.eval_skip_replay:
        cmd.append("--skip-replay")

    log_message(
        f"running post-training eval dataset={args.eval_dataset.resolve()} "
        f"output_dir={eval_output_dir.resolve()}"
    )
    subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent.parent), check=True)
    return eval_output_dir


def save_iteration_checkpoint_and_point_cloud(
    *,
    args: argparse.Namespace,
    iteration: int,
    active_indices: np.ndarray,
    active_params: np.ndarray,
    optimizer_params: np.ndarray,
    adam_m: np.ndarray,
    adam_v: np.ndarray,
    adam_step: np.ndarray,
    best_loss: float,
    best_active_params: np.ndarray,
    best_optimizer_params: np.ndarray,
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
        optimizer_params=optimizer_params,
        adam_m=adam_m,
        adam_v=adam_v,
        adam_step=adam_step,
        best_loss=best_loss,
        best_active_params=best_active_params,
        best_optimizer_params=best_optimizer_params,
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
    parameterization: str,
    optimizer_param_shape: tuple[int, ...],
    rng: np.random.Generator,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, list[float]]:
    with np.load(checkpoint_path, allow_pickle=True) as data:
        checkpoint_active_indices = np.asarray(data["active_indices"], dtype=np.int32)
        if checkpoint_active_indices.shape != active_indices.shape or not np.array_equal(checkpoint_active_indices, active_indices):
            raise ValueError(
                f"{checkpoint_path} active point indices do not match the current run. "
                "Use matching trajectory/model/contact-mask settings or start without --resume-checkpoint."
            )

        checkpoint_parameterization = (
            str(np.asarray(data["friction_parameterization"]).item())
            if "friction_parameterization" in data.files
            else "point"
        )
        if checkpoint_parameterization != parameterization:
            raise ValueError(
                f"{checkpoint_path} was saved with friction_parameterization={checkpoint_parameterization!r}, "
                f"but the current run uses {parameterization!r}."
            )

        iteration = int(np.asarray(data["iteration"]).item())
        if "optimizer_params" in data.files:
            active_params = np.asarray(data["optimizer_params"], dtype=np.float32)
        else:
            active_params = np.asarray(data["active_params"], dtype=np.float32)
        adam_m = np.asarray(data["adam_m"], dtype=np.float64)
        adam_v = np.asarray(data["adam_v"], dtype=np.float64)
        if "adam_step" in data.files:
            adam_step = np.asarray(data["adam_step"], dtype=np.int32)
        else:
            adam_step = np.zeros(optimizer_param_shape, dtype=np.int32)
        best_loss = float(np.asarray(data["best_loss"]).item())
        if "best_optimizer_params" in data.files:
            best_active_params = np.asarray(data["best_optimizer_params"], dtype=np.float32)
        else:
            best_active_params = np.asarray(data["best_active_params"], dtype=np.float32)
        loss_history = [float(value) for value in np.asarray(data["loss_history"], dtype=np.float32)]

        expected_shape = optimizer_param_shape
        for name, values in (
            ("optimizer_params", active_params),
            ("adam_m", adam_m),
            ("adam_v", adam_v),
            ("adam_step", adam_step),
            ("best_optimizer_params", best_active_params),
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


def compute_piecewise_side_ids(local_surface_points: np.ndarray, active_indices: np.ndarray) -> np.ndarray:
    local_x = np.asarray(local_surface_points, dtype=np.float32)[np.asarray(active_indices, dtype=np.int32), 0]
    side_ids = np.full(len(local_x), -1, dtype=np.int32)
    side_ids[local_x < 0.0] = 0
    side_ids[local_x > 0.0] = 1
    return side_ids


def compute_piecewise_regularization_loss_np(params: np.ndarray, side_ids: np.ndarray) -> float:
    regularization_loss, _, _, _, _ = compute_piecewise_regularization_inputs_np(params, side_ids)
    return regularization_loss


def compute_piecewise_regularization_inputs_np(
    params: np.ndarray,
    side_ids: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    params = np.asarray(params, dtype=np.float64)
    side_ids = np.asarray(side_ids, dtype=np.int32)
    side_means = np.zeros(2, dtype=np.float32)
    side_inv_counts = np.zeros(2, dtype=np.float32)
    side_counts = np.zeros(2, dtype=np.int32)
    side_variances = np.zeros(2, dtype=np.float32)
    regularization_loss = 0.0
    for side_id in (0, 1):
        side_params = params[side_ids == side_id]
        if len(side_params) == 0:
            continue
        side_mean = float(np.mean(side_params))
        side_variance = float(np.mean((side_params - side_mean) ** 2))
        side_means[side_id] = np.float32(side_mean)
        side_inv_counts[side_id] = np.float32(1.0 / len(side_params))
        side_counts[side_id] = len(side_params)
        side_variances[side_id] = np.float32(side_variance)
        regularization_loss += side_variance
    return regularization_loss, side_means, side_inv_counts, side_counts, side_variances


def validate_friction_parameterization(parameterization: str) -> str:
    if parameterization not in {"point", "left-right", "global"}:
        raise ValueError(f"Unsupported friction parameterization: {parameterization!r}")
    return parameterization


def build_optimizer_param_positions(
    *,
    parameterization: str,
    active_side_ids: np.ndarray,
    active_count: int,
) -> tuple[np.ndarray, int]:
    if parameterization == "point":
        return np.arange(int(active_count), dtype=np.int32), int(active_count)
    if parameterization == "global":
        return np.zeros(int(active_count), dtype=np.int32), 1
    if np.any(active_side_ids < 0):
        raise ValueError(
            "left-right friction parameterization requires every active contact point to have local x != 0. "
            "Regenerate surface points with the updated sampler so the split seam has no points."
        )
    return np.asarray(active_side_ids, dtype=np.int32), 2


def expand_optimizer_params_to_active(
    optimizer_params: np.ndarray,
    active_param_positions: np.ndarray,
) -> np.ndarray:
    params = np.asarray(optimizer_params, dtype=np.float32)
    positions = np.asarray(active_param_positions, dtype=np.int32)
    if len(positions) == 0:
        return np.empty(0, dtype=np.float32)
    if np.min(positions) < 0 or np.max(positions) >= len(params):
        raise ValueError("Active parameter positions are outside the optimizer parameter vector.")
    return params[positions].astype(np.float32, copy=True)


def aggregate_optimizer_gradients_np(
    *,
    point_grads: np.ndarray,
    active_param_positions: np.ndarray,
    optimizer_param_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    grads = np.asarray(point_grads, dtype=np.float64)
    positions = np.asarray(active_param_positions, dtype=np.int32)
    if grads.shape != positions.shape:
        raise ValueError(f"Gradient/position shape mismatch: {grads.shape} vs {positions.shape}")
    optimizer_grads = np.zeros(int(optimizer_param_count), dtype=np.float64)
    touched_mask = np.zeros(int(optimizer_param_count), dtype=bool)
    if len(positions) > 0:
        if np.min(positions) < 0 or np.max(positions) >= int(optimizer_param_count):
            raise ValueError("Batch parameter positions are outside the optimizer parameter vector.")
        np.add.at(optimizer_grads, positions, grads)
        touched_mask[np.unique(positions)] = True
    return optimizer_grads, touched_mask


def adam_update_np(
    *,
    params: np.ndarray,
    grads: np.ndarray,
    touched_mask: np.ndarray,
    first_moment: np.ndarray,
    second_moment: np.ndarray,
    adam_step: np.ndarray,
    grad_scale: float,
    learning_rate: float,
    beta1: float,
    beta2: float,
    eps: float,
    min_value: float,
    max_value: float,
) -> None:
    touched_indices = np.flatnonzero(np.asarray(touched_mask, dtype=bool))
    for idx in touched_indices:
        grad = float(grads[idx]) * float(grad_scale)
        step = int(adam_step[idx]) + 1
        first_moment[idx] = beta1 * first_moment[idx] + (1.0 - beta1) * grad
        second_moment[idx] = beta2 * second_moment[idx] + (1.0 - beta2) * (grad * grad)
        first_hat = first_moment[idx] / max(1.0 - beta1**step, 1.0e-30)
        second_hat = second_moment[idx] / max(1.0 - beta2**step, 1.0e-30)
        updated = float(params[idx]) - learning_rate * first_hat / (np.sqrt(second_hat) + eps)
        params[idx] = np.float32(min(max(updated, min_value), max_value))
        adam_step[idx] = step


def compute_parameter_stats_np(params: np.ndarray) -> tuple[float, float, float, float]:
    values = np.asarray(params, dtype=np.float64)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return (
        float(np.mean(values)),
        float(np.std(values)),
        float(np.min(values)),
        float(np.max(values)),
    )


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


def format_nonfinite_gradient_diagnostics(
    *,
    grad_values: np.ndarray,
    loss_values: np.ndarray,
    batch_loss_value: float,
) -> str:
    grads = np.asarray(grad_values)
    losses = np.asarray(loss_values)
    grad_abs = np.abs(grads)
    finite_grad_mask = np.isfinite(grads)
    huge_grad_mask = finite_grad_mask & (grad_abs > 1.0e38)
    bad_grad_mask = (~finite_grad_mask) | huge_grad_mask
    bad_indices = np.flatnonzero(bad_grad_mask)
    if len(bad_indices) > 0:
        first_bad_index = int(bad_indices[0])
        first_bad_value = float(grads.reshape(-1)[first_bad_index])
    else:
        first_bad_index = -1
        first_bad_value = float("nan")

    finite_grad_abs = grad_abs[finite_grad_mask & (~huge_grad_mask)]
    finite_grad_abs_max = float(np.max(finite_grad_abs)) if len(finite_grad_abs) > 0 else float("nan")
    finite_loss_mask = np.isfinite(losses)
    finite_losses = losses[finite_loss_mask]
    loss_min = float(np.min(finite_losses)) if len(finite_losses) > 0 else float("nan")
    loss_max = float(np.max(finite_losses)) if len(finite_losses) > 0 else float("nan")

    return (
        f"grad_nan_count={int(np.count_nonzero(np.isnan(grads)))} "
        f"grad_posinf_count={int(np.count_nonzero(np.isposinf(grads)))} "
        f"grad_neginf_count={int(np.count_nonzero(np.isneginf(grads)))} "
        f"grad_huge_count={int(np.count_nonzero(huge_grad_mask))} "
        f"finite_grad_count={int(np.count_nonzero(finite_grad_mask & (~huge_grad_mask)))} "
        f"finite_grad_abs_max={finite_grad_abs_max:.6g} "
        f"first_bad_grad_index={first_bad_index} "
        f"first_bad_grad_value={first_bad_value:.6g} "
        f"batch_loss={float(batch_loss_value):.6g} "
        f"loss_finite_count={int(np.count_nonzero(finite_loss_mask))}/{losses.size} "
        f"loss_min={loss_min:.6g} "
        f"loss_max={loss_max:.6g}"
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


def main() -> None:
    args = parse_args()
    parameterization = validate_friction_parameterization(str(args.friction_parameterization))
    piecewise_regularization_weight = float(args.piecewise_regularization_weight)
    if piecewise_regularization_weight < 0.0:
        raise ValueError("--piecewise-regularization-weight must be non-negative.")

    startup_time = time.time()
    log_message(f"loading trajectories from {args.trajectory_npz.resolve()}")
    trajectory_collection = load_mujoco_trajectories(args.trajectory_npz, args.max_steps, args.max_trajectories)
    trajectories = trajectory_collection.trajectories
    representative_trajectory = trajectories[0]
    batch_size = resolve_batch_size(args.batch_size, len(trajectories), DEFAULT_TRAIN_BATCH_SIZE)
    args.steps = trajectory_collection.max_steps
    args.dt = representative_trajectory.timestep
    log_message(
        f"loaded {len(trajectories)} trajectories | source={trajectory_collection.source_type} | "
        f"max_steps={trajectory_collection.max_steps} | dt={representative_trajectory.timestep:.6f} | "
        f"train_batch_size={batch_size}"
    )

    log_message(f"building diff scene on device={args.device if args.device is not None else 'auto'}")
    args.batch_capacity = max(batch_size, 1)
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
    active_side_ids = compute_piecewise_side_ids(diff_scene.local_surface_points_np, active_indices)
    active_param_positions, optimizer_param_count = build_optimizer_param_positions(
        parameterization=parameterization,
        active_side_ids=active_side_ids,
        active_count=len(active_indices),
    )
    active_param_lookup = np.full(len(diff_scene.local_surface_points_np), -1, dtype=np.int32)
    active_param_lookup[active_indices] = active_param_positions
    log_message(
        f"friction_parameterization={parameterization} optimizer_parameters={optimizer_param_count}"
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

    optimizer_params_np = np.full(optimizer_param_count, float(args.point_friction), dtype=np.float32)
    active_params_np = expand_optimizer_params_to_active(optimizer_params_np, active_param_positions)
    adam_m_np = np.zeros(optimizer_param_count, dtype=np.float64)
    adam_v_np = np.zeros(optimizer_param_count, dtype=np.float64)
    adam_step_np = np.zeros(optimizer_param_count, dtype=np.int32)
    loss_history: list[float] = []
    best_loss = float("inf")
    best_optimizer_params = optimizer_params_np.copy()
    best_active_params = active_params_np.copy()
    rng = np.random.default_rng(int(args.seed))
    start_iteration = 1
    grad_clip_total_count = 0
    grad_clip_clipped_count = 0
    if args.resume_checkpoint is not None:
        (
            resume_iteration,
            optimizer_params_np,
            adam_m_np,
            adam_v_np,
            adam_step_np,
            best_loss,
            best_optimizer_params,
            loss_history,
        ) = load_training_checkpoint(
            checkpoint_path=args.resume_checkpoint,
            active_indices=active_indices,
            parameterization=parameterization,
            optimizer_param_shape=optimizer_params_np.shape,
            rng=rng,
        )
        active_params_np = expand_optimizer_params_to_active(optimizer_params_np, active_param_positions)
        best_active_params = expand_optimizer_params_to_active(best_optimizer_params, active_param_positions)
        start_iteration = resume_iteration + 1
        log_message(
            f"resumed checkpoint {args.resume_checkpoint.resolve()} "
            f"at iteration={resume_iteration} best_loss={best_loss:.6f}"
        )
    device = str(diff_scene.torch_device)
    active_indices_wp = wp.array(active_indices, dtype=wp.int32, device=device)
    active_param_positions_wp = wp.array(active_param_positions, dtype=wp.int32, device=device)
    optimizer_params = wp.array(optimizer_params_np, dtype=wp.float32, device=device)
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
    best_optimizer_params_device = wp.array(best_optimizer_params, dtype=wp.float32, device=device)

    def sync_best_active_params(*, context: str) -> None:
        nonlocal best_optimizer_params, best_active_params
        best_optimizer_params = best_optimizer_params_device.numpy().astype(np.float32)
        best_active_params = expand_optimizer_params_to_active(best_optimizer_params, active_param_positions)
        assert_array_finite("best_optimizer_params", best_optimizer_params, context=context)
        assert_array_finite("best_active_params", best_active_params, context=context)

    def sync_optimizer_state(*, context: str) -> None:
        nonlocal optimizer_params_np, active_params_np, adam_m_np, adam_v_np, adam_step_np
        optimizer_params_np = optimizer_params.numpy().astype(np.float32)
        active_params_np = expand_optimizer_params_to_active(optimizer_params_np, active_param_positions)
        adam_m_np = adam_m.numpy()
        adam_v_np = adam_v.numpy()
        adam_step_np = adam_step.numpy()
        assert_array_finite("optimizer_params", optimizer_params_np, context=context)
        assert_array_finite("active_params", active_params_np, context=context)
        assert_array_finite("adam_m", adam_m_np, context=context)
        assert_array_finite("adam_v", adam_v_np, context=context)
        sync_best_active_params(context=context)

    def save_iteration_checkpoint(iteration: int) -> None:
        sync_optimizer_state(context=f"iter={iteration:04d} checkpoint")
        save_iteration_checkpoint_and_point_cloud(
            args=args,
            iteration=iteration,
            active_indices=active_indices,
            active_params=active_params_np,
            optimizer_params=optimizer_params_np,
            adam_m=adam_m_np,
            adam_v=adam_v_np,
            adam_step=adam_step_np,
            best_loss=best_loss,
            best_active_params=best_active_params,
            best_optimizer_params=best_optimizer_params,
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
                checkpoint_saved = should_save_iteration_checkpoint(args, iteration)
                if checkpoint_saved:
                    save_iteration_checkpoint(iteration)
                log_message(
                    f"iter={iteration:04d} skipped=no_batch_active_points "
                    f"checkpoint_saved={int(checkpoint_saved)} "
                    f"batch=0/{len(batch_trajectories)} "
                    f"elapsed={time.time() - iteration_start:.2f}s"
                )
                continue
            batch_active_param_positions = active_param_lookup[batch_active_indices]
            if np.any(batch_active_param_positions < 0):
                raise RuntimeError("Batch active points must be a subset of the global active point mask.")

            batch_piecewise_side_ids = compute_piecewise_side_ids(
                diff_scene.local_surface_points_np,
                batch_active_indices,
            )
            batch_active_param_positions_wp = wp.array(
                batch_active_param_positions,
                dtype=wp.int32,
                device=device,
            )
            buffers = build_batched_optimization_buffers(diff_scene, batch_trajectories, args, batch_active_indices)
            buffers.full_point_friction.assign(buffers.inactive_point_friction_np)
            wp.launch(
                scatter_indexed_point_friction_kernel,
                dim=len(active_indices),
                inputs=[active_indices_wp, active_param_positions_wp, optimizer_params, buffers.full_point_friction],
                device=diff_scene.model.device,
            )
            wp.launch(
                gather_active_point_friction_kernel,
                dim=len(batch_active_indices),
                inputs=[optimizer_params, batch_active_param_positions_wp, buffers.active_point_friction],
                device=diff_scene.model.device,
            )
            piecewise_regularization_loss_value = 0.0
            piecewise_regularization_contribution_value = 0.0
            (
                piecewise_regularization_loss_value,
                piecewise_side_means,
                piecewise_side_inv_counts,
                piecewise_side_counts,
                piecewise_side_variances,
            ) = compute_piecewise_regularization_inputs_np(
                buffers.active_point_friction.numpy(),
                batch_piecewise_side_ids,
            )
            piecewise_regularization_contribution_value = (
                piecewise_regularization_weight * piecewise_regularization_loss_value
            )
            batch_piecewise_side_ids_wp = None
            piecewise_side_means_wp = None
            piecewise_side_inv_counts_wp = None
            if piecewise_regularization_weight > 0.0:
                batch_piecewise_side_ids_wp = wp.array(batch_piecewise_side_ids, dtype=wp.int32, device=device)
                piecewise_side_means_wp = wp.array(piecewise_side_means, dtype=wp.float32, device=device)
                piecewise_side_inv_counts_wp = wp.array(piecewise_side_inv_counts, dtype=wp.float32, device=device)
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
                if piecewise_regularization_weight > 0.0:
                    wp.launch(
                        add_piecewise_regularization_loss_kernel,
                        dim=len(batch_active_indices),
                        inputs=[
                            buffers.active_point_friction,
                            batch_piecewise_side_ids_wp,
                            piecewise_side_means_wp,
                            piecewise_side_inv_counts_wp,
                            np.float32(piecewise_regularization_weight),
                            buffers.batch_loss,
                        ],
                        device=diff_scene.model.device,
                    )
            tape.backward(buffers.batch_loss)

            if buffers.active_point_friction.grad is None:
                tape.zero()
                checkpoint_saved = should_save_iteration_checkpoint(args, iteration)
                if checkpoint_saved:
                    save_iteration_checkpoint(iteration)
                log_message(
                    f"iter={iteration:04d} skipped=missing_grad "
                    f"checkpoint_saved={int(checkpoint_saved)} "
                    f"batch_active_points={len(batch_active_indices)}/{len(active_indices)} "
                    f"batch={len(batch_trajectories)}/{len(batch_trajectories)} "
                    f"elapsed={time.time() - iteration_start:.2f}s"
                )
                continue

            scalar_stats = wp.zeros(9, dtype=wp.float64, device=device)
            good_buffer_count = len(batch_trajectories)
            wp.launch(
                accumulate_iteration_scalar_stats_kernel,
                dim=good_buffer_count,
                inputs=[
                    buffers.loss,
                    buffers.position_loss,
                    buffers.orientation_loss,
                    buffers.linear_velocity_loss,
                    buffers.angular_velocity_loss,
                    np.float64(1.0 / max(good_buffer_count, 1)),
                    scalar_stats,
                ],
                device=diff_scene.model.device,
            )
            wp.launch(
                accumulate_gradient_scalar_stats_kernel,
                dim=len(batch_active_indices),
                inputs=[buffers.active_point_friction.grad, scalar_stats],
                device=diff_scene.model.device,
            )
            scalar_stats_np = scalar_stats.numpy()
            nonfinite_grad_count_value = int(round(float(scalar_stats_np[8])))
            if nonfinite_grad_count_value != 0:
                grad_diagnostics = format_nonfinite_gradient_diagnostics(
                    grad_values=buffers.active_point_friction.grad.numpy(),
                    loss_values=buffers.loss.numpy(),
                    batch_loss_value=float(buffers.batch_loss.numpy()[0]),
                )
                tape.zero()
                checkpoint_saved = should_save_iteration_checkpoint(args, iteration)
                if checkpoint_saved:
                    save_iteration_checkpoint(iteration)
                log_message(
                    f"iter={iteration:04d} skipped=nonfinite_grad "
                    f"nonfinite_grad_count={nonfinite_grad_count_value} "
                    f"{grad_diagnostics} "
                    f"checkpoint_saved={int(checkpoint_saved)} "
                    f"batch_active_points={len(batch_active_indices)}/{len(active_indices)} "
                    f"batch={len(batch_trajectories)}/{len(batch_trajectories)} "
                    f"elapsed={time.time() - iteration_start:.2f}s"
                )
                continue

            trajectory_loss_value = float(scalar_stats_np[0])
            loss_value = trajectory_loss_value + piecewise_regularization_contribution_value
            raw_position_loss_value = float(scalar_stats_np[1])
            raw_orientation_loss_value = float(scalar_stats_np[2])
            raw_linear_velocity_loss_value = float(scalar_stats_np[3])
            raw_angular_velocity_loss_value = float(scalar_stats_np[4])
            position_loss_value = float(args.position_loss_weight) * raw_position_loss_value
            orientation_loss_value = float(args.orientation_loss_weight) * raw_orientation_loss_value
            linear_velocity_loss_value = float(args.linear_velocity_loss_weight) * raw_linear_velocity_loss_value
            angular_velocity_loss_value = float(args.angular_velocity_loss_weight) * raw_angular_velocity_loss_value
            if parameterization != "point":
                optimizer_grad_np, optimizer_touched_mask = aggregate_optimizer_gradients_np(
                    point_grads=buffers.active_point_friction.grad.numpy(),
                    active_param_positions=batch_active_param_positions,
                    optimizer_param_count=optimizer_param_count,
                )
                touched_grads = optimizer_grad_np[optimizer_touched_mask]
                if len(touched_grads) == 0:
                    raw_grad_norm = 0.0
                    grad_abs_mean_value = 0.0
                    grad_abs_max_value = 0.0
                else:
                    raw_grad_norm = float(np.linalg.norm(touched_grads))
                    grad_abs_mean_value = float(np.mean(np.abs(touched_grads)))
                    grad_abs_max_value = float(np.max(np.abs(touched_grads)))
            else:
                optimizer_grad_np = None
                optimizer_touched_mask = None
                raw_grad_norm = float(np.sqrt(max(float(scalar_stats_np[5]), 0.0)))
                grad_abs_mean_value = float(scalar_stats_np[6]) / max(len(batch_active_indices), 1)
                grad_abs_max_value = float(scalar_stats_np[7])
            grad_clip_norm = args.grad_clip_norm
            if grad_clip_norm is None or float(grad_clip_norm) <= 0.0 or raw_grad_norm <= float(grad_clip_norm):
                grad_clip_scale = 1.0
                clipped_grad_norm = raw_grad_norm
                grad_was_clipped = False
            else:
                grad_clip_scale = float(grad_clip_norm) / max(raw_grad_norm, 1.0e-30)
                clipped_grad_norm = float(grad_clip_norm)
                grad_abs_mean_value *= grad_clip_scale
                grad_abs_max_value *= grad_clip_scale
                grad_was_clipped = True
            grad_clip_total_count += 1
            if grad_was_clipped:
                grad_clip_clipped_count += 1
            grad_clip_ratio_value = grad_clip_clipped_count / max(grad_clip_total_count, 1)
            beta1 = float(args.adam_beta1)
            beta2 = float(args.adam_beta2)
            if parameterization != "point":
                assert optimizer_grad_np is not None
                assert optimizer_touched_mask is not None
                adam_update_np(
                    params=optimizer_params_np,
                    grads=optimizer_grad_np,
                    touched_mask=optimizer_touched_mask,
                    first_moment=adam_m_np,
                    second_moment=adam_v_np,
                    adam_step=adam_step_np,
                    grad_scale=grad_clip_scale,
                    learning_rate=float(args.learning_rate),
                    beta1=beta1,
                    beta2=beta2,
                    eps=float(args.adam_eps),
                    min_value=float(args.min_point_friction),
                    max_value=float(args.max_point_friction),
                )
                optimizer_params.assign(optimizer_params_np)
                adam_m.assign(adam_m_np)
                adam_v.assign(adam_v_np)
                adam_step.assign(adam_step_np)
                beta1_power.assign(np.power(beta1, adam_step_np.astype(np.float64)))
                beta2_power.assign(np.power(beta2, adam_step_np.astype(np.float64)))

                param_nonfinite_count = int(np.count_nonzero(~np.isfinite(optimizer_params_np)))
                adam_m_nonfinite_count = int(np.count_nonzero(~np.isfinite(adam_m_np)))
                adam_v_nonfinite_count = int(np.count_nonzero(~np.isfinite(adam_v_np)))
                if param_nonfinite_count != 0 or adam_m_nonfinite_count != 0 or adam_v_nonfinite_count != 0:
                    tape.zero()
                    raise FloatingPointError(
                        f"iter={iteration:04d} after Adam update: "
                        f"optimizer_params_nonfinite_count={param_nonfinite_count} "
                        f"adam_m_nonfinite_count={adam_m_nonfinite_count} "
                        f"adam_v_nonfinite_count={adam_v_nonfinite_count}"
                    )
                active_params_np = expand_optimizer_params_to_active(optimizer_params_np, active_param_positions)
                mu_mean_value, mu_std_value, mu_min_value, mu_max_value = compute_parameter_stats_np(active_params_np)
            else:
                wp.launch(
                    sparse_adam_update_clipped_kernel,
                    dim=len(batch_active_indices),
                    inputs=[
                        optimizer_params,
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

                optimizer_stats = wp.array(
                    np.asarray([0.0, 0.0, np.inf, -np.inf, 0.0, 0.0, 0.0], dtype=np.float64),
                    dtype=wp.float64,
                    device=device,
                )
                wp.launch(
                    accumulate_optimizer_scalar_stats_kernel,
                    dim=optimizer_param_count,
                    inputs=[optimizer_params, adam_m, adam_v, optimizer_stats],
                    device=diff_scene.model.device,
                )
                optimizer_stats_np = optimizer_stats.numpy()
                param_nonfinite_count = int(round(float(optimizer_stats_np[4])))
                adam_m_nonfinite_count = int(round(float(optimizer_stats_np[5])))
                adam_v_nonfinite_count = int(round(float(optimizer_stats_np[6])))
                if param_nonfinite_count != 0 or adam_m_nonfinite_count != 0 or adam_v_nonfinite_count != 0:
                    tape.zero()
                    raise FloatingPointError(
                        f"iter={iteration:04d} after Adam update: "
                        f"optimizer_params_nonfinite_count={param_nonfinite_count} "
                        f"adam_m_nonfinite_count={adam_m_nonfinite_count} "
                        f"adam_v_nonfinite_count={adam_v_nonfinite_count}"
                    )

                param_count = max(optimizer_param_count, 1)
                mu_mean_value = float(optimizer_stats_np[0]) / param_count
                mu_squares_mean = float(optimizer_stats_np[1]) / param_count
                mu_std_value = float(np.sqrt(max(mu_squares_mean - mu_mean_value * mu_mean_value, 0.0)))
                mu_min_value = float(optimizer_stats_np[2])
                mu_max_value = float(optimizer_stats_np[3])
            tape.zero()
            loss_history.append(loss_value)

            if loss_value < best_loss:
                best_loss = loss_value
                best_optimizer_params_device.assign(optimizer_params)

            if should_save_iteration_checkpoint(args, iteration):
                save_iteration_checkpoint(iteration)

            if wandb_run is not None:
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
                    grad_value=None,
                    active_params=None,
                    active_indices=active_indices,
                    grad_abs_mean_value=grad_abs_mean_value,
                    grad_abs_max_value=grad_abs_max_value,
                    mu_mean_value=mu_mean_value,
                    mu_std_value=mu_std_value,
                    mu_min_value=mu_min_value,
                    mu_max_value=mu_max_value,
                )
                log_payload["params/batch_active_contact_point_count"] = float(len(batch_active_indices))
                log_payload["params/optimizer_parameter_count"] = float(optimizer_param_count)
                log_payload["params/batch_piecewise_left_count"] = float(piecewise_side_counts[0])
                log_payload["params/batch_piecewise_right_count"] = float(piecewise_side_counts[1])
                log_payload["params/mu_left_mean"] = float(piecewise_side_means[0])
                log_payload["params/mu_right_mean"] = float(piecewise_side_means[1])
                if parameterization == "left-right":
                    log_payload["params/mu_left_param"] = float(optimizer_params_np[0])
                    log_payload["params/mu_right_param"] = float(optimizer_params_np[1])
                if parameterization == "global":
                    log_payload["params/mu_global_param"] = float(optimizer_params_np[0])
                log_payload["train/trajectory_loss"] = float(trajectory_loss_value)
                log_payload["regularization/piecewise"] = float(piecewise_regularization_loss_value)
                log_payload["regularization/var_left"] = float(piecewise_side_variances[0])
                log_payload["regularization/var_right"] = float(piecewise_side_variances[1])
                log_payload["regularization/piecewise_contribution"] = float(
                    piecewise_regularization_contribution_value
                )
                log_payload["regularization/piecewise_weight"] = float(piecewise_regularization_weight)
                log_payload["train/raw_grad_norm"] = float(raw_grad_norm)
                log_payload["train/clipped_grad_norm"] = float(clipped_grad_norm)
                log_payload["train/grad_clip_scale"] = float(grad_clip_scale)
                log_payload["train/clip_ratio"] = float(grad_clip_ratio_value)
                wandb_run.log(log_payload, step=iteration)

            if iteration == 1 or iteration % max(int(args.log_every), 1) == 0 or iteration == int(args.opt_iters):
                log_message(
                    f"iter={iteration:04d} loss={loss_value:.6f} "
                    f"traj_loss={trajectory_loss_value:.6f} "
                    f"piecewise_reg={piecewise_regularization_loss_value:.6g} "
                    f"var_left={piecewise_side_variances[0]:.6g} "
                    f"var_right={piecewise_side_variances[1]:.6g} "
                    f"piecewise_contrib={piecewise_regularization_contribution_value:.6g} "
                    f"mu_left_mean={piecewise_side_means[0]:.6f} "
                    f"mu_right_mean={piecewise_side_means[1]:.6f} "
                    f"pos={position_loss_value:.6f} "
                    f"ori={orientation_loss_value:.6f} "
                    f"linvel={linear_velocity_loss_value:.6f} "
                    f"angvel={angular_velocity_loss_value:.6f} "
                    f"grad_norm={clipped_grad_norm:.6f} "
                    f"raw_grad_norm={raw_grad_norm:.6f} "
                    f"clip_scale={grad_clip_scale:.6g} "
                    f"mu_min={mu_min_value:.6f} "
                    f"mu_max={mu_max_value:.6f} "
                    f"batch_active_points={len(batch_active_indices)}/{len(active_indices)} "
                    f"batch={good_buffer_count}/{len(batch_trajectories)} "
                    f"elapsed={time.time() - iteration_start:.2f}s"
                )

        sync_best_active_params(context="final export")
        final_piecewise_side_ids = compute_piecewise_side_ids(diff_scene.local_surface_points_np, active_indices)
        (
            final_piecewise_regularization_loss,
            final_piecewise_side_means,
            _,
            _,
            final_piecewise_side_variances,
        ) = compute_piecewise_regularization_inputs_np(
            best_active_params,
            final_piecewise_side_ids,
        )
        final_piecewise_regularization_contribution = (
            piecewise_regularization_weight * final_piecewise_regularization_loss
        )

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
            best_optimizer_params=best_optimizer_params,
            loss_history=loss_history,
            best_loss=best_loss,
            body_q_frames=None,
        )

        if wandb_run is not None:
            wandb_run.summary["surface_points"] = int(len(diff_scene.local_surface_points_np))
            wandb_run.summary["active_contact_points"] = int(len(active_indices))
            wandb_run.summary["friction_parameterization"] = parameterization
            wandb_run.summary["optimizer_parameter_count"] = int(optimizer_param_count)
            wandb_run.summary["best_training_loss"] = float(best_loss)
            wandb_run.summary["final_piecewise_regularization"] = float(final_piecewise_regularization_loss)
            wandb_run.summary["final_piecewise_var_left"] = float(final_piecewise_side_variances[0])
            wandb_run.summary["final_piecewise_var_right"] = float(final_piecewise_side_variances[1])
            wandb_run.summary["final_piecewise_regularization_contribution"] = float(
                final_piecewise_regularization_contribution
            )
            wandb_run.summary["final_piecewise_regularization_weight"] = float(piecewise_regularization_weight)
            wandb_run.summary["mu_mean"] = float(best_active_params.mean())
            wandb_run.summary["mu_std"] = float(best_active_params.std())
            wandb_run.summary["mu_min"] = float(best_active_params.min())
            wandb_run.summary["mu_max"] = float(best_active_params.max())
            if parameterization == "left-right":
                wandb_run.summary["mu_left_param"] = float(best_optimizer_params[0])
                wandb_run.summary["mu_right_param"] = float(best_optimizer_params[1])
            if parameterization == "global":
                wandb_run.summary["mu_global_param"] = float(best_optimizer_params[0])
            wandb_run.summary["mu_left_mean"] = float(final_piecewise_side_means[0])
            wandb_run.summary["mu_right_mean"] = float(final_piecewise_side_means[1])
            wandb_run.summary["results_path"] = str(args.results_path.resolve())
            wandb_run.summary["point_cloud_path"] = str(args.point_cloud_path.resolve())
            if args.scene_usd_path is not None:
                wandb_run.summary["scene_usd_path"] = str(args.scene_usd_path.resolve())

        log_message(f"trajectory={args.trajectory_npz.resolve()}")
        log_message(f"trajectory_source_type={trajectory_collection.source_type}")
        log_message(f"trajectory_count={len(trajectories)}")
        log_message(f"max_steps={trajectory_collection.max_steps} dt={representative_trajectory.timestep:.6f}")
        log_message(f"surface_points={len(diff_scene.local_surface_points_np)} active_contact_points={len(active_indices)}")
        log_message(f"friction_parameterization={parameterization} optimizer_parameters={optimizer_param_count}")
        log_message(f"best_training_loss={best_loss:.6f}")
        log_message(f"final_piecewise_regularization={final_piecewise_regularization_loss:.6g}")
        log_message(f"final_piecewise_var_left={final_piecewise_side_variances[0]:.6g}")
        log_message(f"final_piecewise_var_right={final_piecewise_side_variances[1]:.6g}")
        log_message(f"final_piecewise_regularization_contribution={final_piecewise_regularization_contribution:.6g}")
        log_message(f"final_mu_left_mean={final_piecewise_side_means[0]:.6f}")
        log_message(f"final_mu_right_mean={final_piecewise_side_means[1]:.6f}")
        if parameterization == "left-right":
            log_message(f"final_mu_left_param={best_optimizer_params[0]:.6f}")
            log_message(f"final_mu_right_param={best_optimizer_params[1]:.6f}")
        if parameterization == "global":
            log_message(f"final_mu_global_param={best_optimizer_params[0]:.6f}")
        log_message(f"results_written_to={args.results_path.resolve()}")
        log_message(f"point_cloud_written_to={args.point_cloud_path.resolve()}")
        if args.scene_usd_path is not None:
            log_message(f"scene_usd_written_to={args.scene_usd_path.resolve()}")
        if args.eval_dataset is not None:
            eval_output_dir = run_post_training_eval(args)
            log_message(f"post_training_eval_written_to={eval_output_dir.resolve()}")
            if wandb_run is not None:
                wandb_run.summary["post_training_eval_dir"] = str(eval_output_dir.resolve())
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
