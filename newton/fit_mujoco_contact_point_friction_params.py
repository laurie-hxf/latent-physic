from __future__ import annotations

import argparse

import numpy as np

from mujoco_contact_friction_fit_utils import sample_mujoco_trajectory_time_window


def compute_batch_active_point_indices(
    trajectory_active_indices_by_idx: list[np.ndarray],
    batch_indices: np.ndarray,
    point_count: int,
) -> np.ndarray:
    batch_active_mask = np.zeros(int(point_count), dtype=bool)
    for trajectory_idx in batch_indices:
        batch_active_mask[trajectory_active_indices_by_idx[int(trajectory_idx)]] = True
    return np.flatnonzero(batch_active_mask).astype(np.int32)


def resolve_trajectory_load_max_steps(args: argparse.Namespace) -> int | None:
    if not bool(getattr(args, "random_time_windows", False)):
        return args.max_steps
    source_max_steps = getattr(args, "time_window_source_max_steps", None)
    if source_max_steps is None:
        return None
    if int(source_max_steps) < 1:
        raise ValueError("--time-window-source-max-steps must be positive when set.")
    return int(source_max_steps)


def resolve_training_rollout_steps(
    args: argparse.Namespace,
    trajectory_collection,
) -> int:
    if not bool(getattr(args, "random_time_windows", False)):
        return int(trajectory_collection.max_steps)
    if args.window_steps is not None:
        requested_steps = int(args.window_steps)
    elif args.max_steps is not None:
        requested_steps = int(args.max_steps)
    else:
        requested_steps = int(trajectory_collection.max_steps)
    if requested_steps < 1:
        raise ValueError("--window-steps/--max-steps must be positive for random time-window training.")
    return min(requested_steps, int(trajectory_collection.max_steps))


def sample_training_time_windows(
    *,
    trajectories: list,
    window_steps: int,
    rng: np.random.Generator,
    enabled: bool,
) -> tuple[list, np.ndarray]:
    if not enabled:
        return trajectories, np.full(len(trajectories), -1, dtype=np.int32)

    windowed_trajectories = []
    start_steps = np.zeros(len(trajectories), dtype=np.int32)
    for idx, trajectory in enumerate(trajectories):
        windowed_trajectory, start_step = sample_mujoco_trajectory_time_window(
            trajectory,
            window_steps=window_steps,
            rng=rng,
        )
        windowed_trajectories.append(windowed_trajectory)
        start_steps[idx] = int(start_step)
    return windowed_trajectories, start_steps


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
    if parameterization not in {"point", "left-right", "global", "base-delta"}:
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
    if parameterization not in {"left-right", "base-delta"}:
        raise ValueError(f"Unsupported friction parameterization: {parameterization!r}")
    if np.any(active_side_ids < 0):
        raise ValueError(
            f"{parameterization} friction parameterization requires every active contact point to have local x != 0. "
            "Regenerate surface points with the updated sampler so the split seam has no points."
        )
    if parameterization == "left-right":
        return np.asarray(active_side_ids, dtype=np.int32), 2
    return np.asarray(active_side_ids, dtype=np.int32), 3


def initialize_optimizer_params_np(
    *,
    parameterization: str,
    optimizer_param_count: int,
    point_friction: float,
) -> np.ndarray:
    if parameterization == "base-delta":
        if int(optimizer_param_count) != 3:
            raise ValueError(f"base-delta expected 3 optimizer parameters, got {optimizer_param_count}")
        return np.asarray([point_friction, 0.0, 0.0], dtype=np.float32)
    return np.full(int(optimizer_param_count), float(point_friction), dtype=np.float32)


def project_base_delta_optimizer_params_np(
    params: np.ndarray,
    *,
    min_value: float,
    max_value: float,
    left_right_delta_sum_zero: bool,
) -> np.ndarray:
    projected = np.asarray(params, dtype=np.float64).copy()
    if projected.shape != (3,):
        raise ValueError(f"base-delta optimizer parameters must have shape (3,), got {projected.shape}")

    base = float(np.clip(projected[0], min_value, max_value))
    delta_left = float(projected[1])
    delta_right = float(projected[2])

    if left_right_delta_sum_zero:
        delta_mean = 0.5 * (delta_left + delta_right)
        delta_left -= delta_mean
        delta_right -= delta_mean
        delta = 0.5 * (delta_left - delta_right)
        allowed_delta_abs = max(min(base - min_value, max_value - base), 0.0)
        delta = float(np.clip(delta, -allowed_delta_abs, allowed_delta_abs))
        projected[0] = base
        projected[1] = delta
        projected[2] = -delta
        return projected.astype(np.float32)

    mu_left = float(np.clip(base + delta_left, min_value, max_value))
    mu_right = float(np.clip(base + delta_right, min_value, max_value))
    projected[0] = base
    projected[1] = mu_left - base
    projected[2] = mu_right - base
    return projected.astype(np.float32)


def expand_optimizer_params_to_active(
    optimizer_params: np.ndarray,
    active_param_positions: np.ndarray,
    *,
    parameterization: str,
) -> np.ndarray:
    params = np.asarray(optimizer_params, dtype=np.float32)
    positions = np.asarray(active_param_positions, dtype=np.int32)
    if len(positions) == 0:
        return np.empty(0, dtype=np.float32)
    if parameterization == "base-delta":
        if len(params) != 3:
            raise ValueError(f"base-delta expected 3 optimizer parameters, got {len(params)}")
        if np.min(positions) < 0 or np.max(positions) > 1:
            raise ValueError("base-delta active side ids must be 0 for left or 1 for right.")
        return (params[0] + params[1 + positions]).astype(np.float32, copy=True)
    if np.min(positions) < 0 or np.max(positions) >= len(params):
        raise ValueError("Active parameter positions are outside the optimizer parameter vector.")
    return params[positions].astype(np.float32, copy=True)


def aggregate_optimizer_gradients_np(
    *,
    point_grads: np.ndarray,
    active_param_positions: np.ndarray,
    optimizer_param_count: int,
    parameterization: str,
    left_right_delta_sum_zero: bool,
) -> tuple[np.ndarray, np.ndarray]:
    grads = np.asarray(point_grads, dtype=np.float64)
    positions = np.asarray(active_param_positions, dtype=np.int32)
    if grads.shape != positions.shape:
        raise ValueError(f"Gradient/position shape mismatch: {grads.shape} vs {positions.shape}")
    optimizer_grads = np.zeros(int(optimizer_param_count), dtype=np.float64)
    touched_mask = np.zeros(int(optimizer_param_count), dtype=bool)
    if parameterization == "base-delta":
        if int(optimizer_param_count) != 3:
            raise ValueError(f"base-delta expected 3 optimizer parameters, got {optimizer_param_count}")
        if len(positions) > 0:
            if np.min(positions) < 0 or np.max(positions) > 1:
                raise ValueError("base-delta active side ids must be 0 for left or 1 for right.")
            optimizer_grads[0] = float(np.sum(grads))
            touched_mask[0] = True
            for side_id in (0, 1):
                side_mask = positions == side_id
                if np.any(side_mask):
                    optimizer_grads[1 + side_id] = float(np.sum(grads[side_mask]))
                    touched_mask[1 + side_id] = True
            if left_right_delta_sum_zero:
                delta_touched = bool(touched_mask[1] or touched_mask[2])
                delta_grad_mean = 0.5 * (optimizer_grads[1] + optimizer_grads[2])
                optimizer_grads[1] -= delta_grad_mean
                optimizer_grads[2] -= delta_grad_mean
                touched_mask[1] = delta_touched
                touched_mask[2] = delta_touched
        return optimizer_grads, touched_mask

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


