from __future__ import annotations

import time

import numpy as np
import warp as wp

from mujoco_contact_friction_fit_utils import (
    compute_active_contact_point_indices,
    load_mujoco_trajectories,
)
from mujoco_contact_friction_fit_wandb import build_wandb_log_payload, init_wandb
from fit_mujoco_contact_point_friction_checkpoint import (
    load_training_checkpoint,
    run_post_training_eval,
    save_iteration_checkpoint_and_point_cloud,
    should_save_iteration_checkpoint,
)
from fit_mujoco_contact_point_friction_io import DEFAULT_TRAIN_BATCH_SIZE, parse_args
from fit_mujoco_contact_point_friction_kernels import (
    accumulate_gradient_scalar_stats_kernel,
    accumulate_iteration_scalar_stats_kernel,
    accumulate_optimizer_scalar_stats_kernel,
    accumulate_batched_frame_loss_kernel,
    add_piecewise_regularization_loss_kernel,
    apply_batched_external_and_surface_point_forces_trajectory_kernel,
    combine_batched_loss_components_kernel,
    compute_batched_contact_weighted_masses_kernel,
    gather_active_point_friction_kernel,
    scatter_active_point_friction_kernel,
    scatter_indexed_point_friction_kernel,
    sparse_adam_update_clipped_kernel,
    sum_batched_losses_kernel,
)
from fit_mujoco_contact_point_friction_output import export_contact_friction_outputs
from fit_mujoco_contact_point_friction_params import (
    adam_update_np,
    aggregate_optimizer_gradients_np,
    build_optimizer_param_positions,
    compute_batch_active_point_indices,
    compute_parameter_stats_np,
    compute_piecewise_regularization_inputs_np,
    compute_piecewise_side_ids,
    expand_optimizer_params_to_active,
    format_nonfinite_gradient_diagnostics,
    initialize_optimizer_params_np,
    project_base_delta_optimizer_params_np,
    resolve_point_cloud_color_bounds,
    resolve_training_rollout_steps,
    resolve_trajectory_load_max_steps,
    sample_training_time_windows,
    validate_friction_parameterization,
)
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
from newton_surface_points_diff_demo import build_diff_scene


def main() -> None:
    args = parse_args()
    parameterization = validate_friction_parameterization(str(args.friction_parameterization))
    left_right_delta_sum_zero = bool(getattr(args, "left_right_delta_sum_zero", False))
    random_time_windows = bool(getattr(args, "random_time_windows", False))
    if left_right_delta_sum_zero and parameterization != "base-delta":
        raise ValueError("--left-right-delta-sum-zero is only valid with --friction-parameterization base-delta.")
    if not random_time_windows and args.window_steps is not None:
        raise ValueError("--window-steps is only valid with --random-time-windows.")
    if not random_time_windows and args.time_window_source_max_steps is not None:
        raise ValueError("--time-window-source-max-steps is only valid with --random-time-windows.")
    if float(args.max_point_friction) < float(args.min_point_friction):
        raise ValueError("--max-point-friction must be greater than or equal to --min-point-friction.")
    piecewise_regularization_weight = float(args.piecewise_regularization_weight)
    if piecewise_regularization_weight < 0.0:
        raise ValueError("--piecewise-regularization-weight must be non-negative.")

    startup_time = time.time()
    trajectory_load_max_steps = resolve_trajectory_load_max_steps(args)
    log_message(
        f"loading trajectories from {args.trajectory_npz.resolve()} "
        f"load_max_steps={trajectory_load_max_steps if trajectory_load_max_steps is not None else 'full'}"
    )
    trajectory_collection = load_mujoco_trajectories(
        args.trajectory_npz,
        trajectory_load_max_steps,
        args.max_trajectories,
    )
    trajectories = trajectory_collection.trajectories
    representative_trajectory = trajectories[0]
    batch_size = resolve_batch_size(args.batch_size, len(trajectories), DEFAULT_TRAIN_BATCH_SIZE)
    args.steps = resolve_training_rollout_steps(args, trajectory_collection)
    args.dt = representative_trajectory.timestep
    log_message(
        f"loaded {len(trajectories)} trajectories | source={trajectory_collection.source_type} | "
        f"source_max_steps={trajectory_collection.max_steps} | train_rollout_steps={args.steps} | "
        f"random_time_windows={int(random_time_windows)} | dt={representative_trajectory.timestep:.6f} | "
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
        f"friction_parameterization={parameterization} optimizer_parameters={optimizer_param_count} "
        f"left_right_delta_sum_zero={int(left_right_delta_sum_zero)}"
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

    optimizer_params_np = initialize_optimizer_params_np(
        parameterization=parameterization,
        optimizer_param_count=optimizer_param_count,
        point_friction=float(args.point_friction),
    )
    if parameterization == "base-delta":
        optimizer_params_np = project_base_delta_optimizer_params_np(
            optimizer_params_np,
            min_value=float(args.min_point_friction),
            max_value=float(args.max_point_friction),
            left_right_delta_sum_zero=left_right_delta_sum_zero,
        )
    active_params_np = expand_optimizer_params_to_active(
        optimizer_params_np,
        active_param_positions,
        parameterization=parameterization,
    )
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
            left_right_delta_sum_zero=left_right_delta_sum_zero,
            random_time_windows=random_time_windows,
            optimizer_param_shape=optimizer_params_np.shape,
            rng=rng,
        )
        if parameterization == "base-delta":
            optimizer_params_np = project_base_delta_optimizer_params_np(
                optimizer_params_np,
                min_value=float(args.min_point_friction),
                max_value=float(args.max_point_friction),
                left_right_delta_sum_zero=left_right_delta_sum_zero,
            )
            best_optimizer_params = project_base_delta_optimizer_params_np(
                best_optimizer_params,
                min_value=float(args.min_point_friction),
                max_value=float(args.max_point_friction),
                left_right_delta_sum_zero=left_right_delta_sum_zero,
            )
        active_params_np = expand_optimizer_params_to_active(
            optimizer_params_np,
            active_param_positions,
            parameterization=parameterization,
        )
        best_active_params = expand_optimizer_params_to_active(
            best_optimizer_params,
            active_param_positions,
            parameterization=parameterization,
        )
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
        best_active_params = expand_optimizer_params_to_active(
            best_optimizer_params,
            active_param_positions,
            parameterization=parameterization,
        )
        assert_array_finite("best_optimizer_params", best_optimizer_params, context=context)
        assert_array_finite("best_active_params", best_active_params, context=context)

    def sync_optimizer_state(*, context: str) -> None:
        nonlocal optimizer_params_np, active_params_np, adam_m_np, adam_v_np, adam_step_np
        optimizer_params_np = optimizer_params.numpy().astype(np.float32)
        active_params_np = expand_optimizer_params_to_active(
            optimizer_params_np,
            active_param_positions,
            parameterization=parameterization,
        )
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
            batch_source_trajectories = [trajectories[int(idx)] for idx in batch_indices]
            batch_trajectories, batch_window_start_steps = sample_training_time_windows(
                trajectories=batch_source_trajectories,
                window_steps=int(args.steps),
                rng=rng,
                enabled=random_time_windows,
            )
            if random_time_windows and len(batch_window_start_steps) > 0:
                batch_window_start_min = int(np.min(batch_window_start_steps))
                batch_window_start_max = int(np.max(batch_window_start_steps))
                batch_window_start_mean = float(np.mean(batch_window_start_steps))
            else:
                batch_window_start_min = -1
                batch_window_start_max = -1
                batch_window_start_mean = float("nan")
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
                    f"window_start_min={batch_window_start_min} "
                    f"window_start_max={batch_window_start_max} "
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
            if parameterization == "base-delta":
                full_point_friction_np = buffers.inactive_point_friction_np.copy()
                full_point_friction_np[active_indices] = active_params_np
                buffers.full_point_friction.assign(full_point_friction_np)
                batch_active_params = expand_optimizer_params_to_active(
                    optimizer_params_np,
                    batch_active_param_positions,
                    parameterization=parameterization,
                )
                buffers.active_point_friction.assign(batch_active_params)
            else:
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
                    f"window_start_min={batch_window_start_min} "
                    f"window_start_max={batch_window_start_max} "
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
                    f"window_start_min={batch_window_start_min} "
                    f"window_start_max={batch_window_start_max} "
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
                    parameterization=parameterization,
                    left_right_delta_sum_zero=left_right_delta_sum_zero,
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
                    min_value=-float("inf") if parameterization == "base-delta" else float(args.min_point_friction),
                    max_value=float("inf") if parameterization == "base-delta" else float(args.max_point_friction),
                )
                if parameterization == "base-delta":
                    optimizer_params_np = project_base_delta_optimizer_params_np(
                        optimizer_params_np,
                        min_value=float(args.min_point_friction),
                        max_value=float(args.max_point_friction),
                        left_right_delta_sum_zero=left_right_delta_sum_zero,
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
                active_params_np = expand_optimizer_params_to_active(
                    optimizer_params_np,
                    active_param_positions,
                    parameterization=parameterization,
                )
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
                if random_time_windows:
                    log_payload["time_window/start_min"] = float(batch_window_start_min)
                    log_payload["time_window/start_max"] = float(batch_window_start_max)
                    log_payload["time_window/start_mean"] = float(batch_window_start_mean)
                    log_payload["time_window/steps"] = float(args.steps)
                if parameterization == "left-right":
                    log_payload["params/mu_left_param"] = float(optimizer_params_np[0])
                    log_payload["params/mu_right_param"] = float(optimizer_params_np[1])
                if parameterization == "global":
                    log_payload["params/mu_global_param"] = float(optimizer_params_np[0])
                if parameterization == "base-delta":
                    log_payload["params/mu_base_param"] = float(optimizer_params_np[0])
                    log_payload["params/delta_left_param"] = float(optimizer_params_np[1])
                    log_payload["params/delta_right_param"] = float(optimizer_params_np[2])
                    log_payload["params/delta_sum"] = float(optimizer_params_np[1] + optimizer_params_np[2])
                    log_payload["params/mu_left_param"] = float(optimizer_params_np[0] + optimizer_params_np[1])
                    log_payload["params/mu_right_param"] = float(optimizer_params_np[0] + optimizer_params_np[2])
                    log_payload["params/left_right_delta_sum_zero"] = float(left_right_delta_sum_zero)
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
                    f"window_start_min={batch_window_start_min} "
                    f"window_start_max={batch_window_start_max} "
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
            if parameterization == "base-delta":
                wandb_run.summary["mu_base_param"] = float(best_optimizer_params[0])
                wandb_run.summary["delta_left_param"] = float(best_optimizer_params[1])
                wandb_run.summary["delta_right_param"] = float(best_optimizer_params[2])
                wandb_run.summary["delta_sum"] = float(best_optimizer_params[1] + best_optimizer_params[2])
                wandb_run.summary["mu_left_param"] = float(best_optimizer_params[0] + best_optimizer_params[1])
                wandb_run.summary["mu_right_param"] = float(best_optimizer_params[0] + best_optimizer_params[2])
                wandb_run.summary["left_right_delta_sum_zero"] = bool(left_right_delta_sum_zero)
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
        if parameterization == "base-delta":
            log_message(f"final_mu_base_param={best_optimizer_params[0]:.6f}")
            log_message(f"final_delta_left_param={best_optimizer_params[1]:.6f}")
            log_message(f"final_delta_right_param={best_optimizer_params[2]:.6f}")
            log_message(f"final_delta_sum={best_optimizer_params[1] + best_optimizer_params[2]:.6f}")
            log_message(f"final_mu_left_param={best_optimizer_params[0] + best_optimizer_params[1]:.6f}")
            log_message(f"final_mu_right_param={best_optimizer_params[0] + best_optimizer_params[2]:.6f}")
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
