from __future__ import annotations

from pathlib import Path

import numpy as np

from pbd_usd import export_scene_usd

from fit_mujoco_contact_point_friction_io import save_contact_friction_point_cloud


def export_contact_friction_outputs(
    *,
    args,
    trajectory_collection,
    representative_trajectory,
    trajectories,
    diff_scene,
    active_indices: np.ndarray,
    best_active_params: np.ndarray,
    best_optimizer_params: np.ndarray,
    loss_history: list[float],
    best_loss: float,
    body_q_frames: list[np.ndarray] | None = None,
) -> np.ndarray:
    learned_point_friction = np.full(
        len(diff_scene.local_surface_points_np),
        float(args.point_friction),
        dtype=np.float32,
    )
    learned_point_friction[active_indices] = best_active_params

    save_contact_friction_point_cloud(
        local_surface_points=diff_scene.local_surface_points_np,
        point_friction=learned_point_friction,
        output_path=args.point_cloud_path,
        active_indices=active_indices,
        color_min=args.point_cloud_color_min if args.point_cloud_color_min is not None else float(args.point_friction) - 0.005,
        color_max=args.point_cloud_color_max if args.point_cloud_color_max is not None else float(args.point_friction) + 0.005,
    )

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.results_path,
        trajectory_npz_path=np.asarray(str(args.trajectory_npz)),
        trajectory_source_type=np.asarray(trajectory_collection.source_type),
        trajectory_count=np.asarray(len(trajectories), dtype=np.int32),
        trajectory_steps=np.asarray([trajectory.num_steps for trajectory in trajectories], dtype=np.int32),
        trajectory_frames=np.asarray([trajectory.num_frames for trajectory in trajectories], dtype=np.int32),
        representative_time=representative_trajectory.time,
        representative_target_positions=representative_trajectory.positions,
        representative_target_quaternions_xyzw=representative_trajectory.quaternions_xyzw,
        representative_target_linear_velocity=representative_trajectory.linear_velocity,
        representative_target_angular_velocity=representative_trajectory.angular_velocity,
        representative_target_step_forces=representative_trajectory.step_forces,
        representative_force_point_offset_local=representative_trajectory.force_point_offset_local,
        local_surface_points=diff_scene.local_surface_points_np,
        point_masses=diff_scene.point_masses_np,
        active_contact_point_indices=active_indices,
        active_contact_local_points=diff_scene.local_surface_points_np[active_indices],
        learned_point_friction=learned_point_friction,
        learned_active_point_friction=best_active_params,
        friction_parameterization=np.asarray(str(getattr(args, "friction_parameterization", "point"))),
        learned_optimizer_friction=np.asarray(best_optimizer_params, dtype=np.float32),
        loss_history=np.asarray(loss_history, dtype=np.float32),
        best_loss=np.asarray(best_loss, dtype=np.float32),
        point_cloud_path=np.asarray(str(args.point_cloud_path)),
    )

    if args.scene_usd_path is not None:
        args.scene_usd_path.parent.mkdir(parents=True, exist_ok=True)
        export_scene_usd(
            scene=diff_scene.scene,
            output_path=args.scene_usd_path,
            body_q_frames=body_q_frames,
            fps=1.0 / float(args.dt),
        )

    return learned_point_friction
