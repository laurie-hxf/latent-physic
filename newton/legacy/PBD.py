from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pbd_io import build_ee_action_sequence, parse_args, resolve_step_output_dir
from pbd_scene import (
    advance_prescribed_cluster,
    build_scene_from_segmented_ply,
    export_rollout_step,
    find_cluster,
    step_scene,
)
from pbd_usd import export_scene_usd


def main() -> None:
    args = parse_args()
    scene_usd_path = args.scene_usd_path

    ee_actions = build_ee_action_sequence(args)
    has_explicit_actions = args.ee_action is not None or args.ee_actions_json is not None
    step_output_dir = resolve_step_output_dir(
        scene_usd_path=scene_usd_path,
        requested_step_dir=args.save_step_dir,
        has_explicit_actions=has_explicit_actions,
    )

    scene = build_scene_from_segmented_ply(
        ply_path=args.ply_path,
        table_seg_id=args.table_seg_id,
        tee_seg_id=args.tee_seg_id,
        ee_seg_id=args.ee_seg_id,
        table_voxel=args.table_voxel,
        tee_voxel=args.tee_voxel,
        ee_voxel=args.ee_voxel,
        tee_radius_scale=args.tee_radius_scale,
        ee_radius_scale=args.ee_radius_scale,
        tee_mass=args.tee_mass,
        ee_mass=args.ee_mass,
        xpbd_iterations=args.xpbd_iterations,
        table_friction=args.table_friction,
        object_friction=args.object_friction,
        contact_stiffness=args.contact_stiffness,
        contact_damping=args.contact_damping,
        contact_margin=args.contact_margin,
        friction_regularization=args.friction_regularization,
    )
    body_q_frames = [scene.state_0.body_q.detach().cpu().numpy().copy()]

    for cluster in scene.clusters:
        if cluster.control_mode == "prescribed":
            cluster_type = "kinematic"
        elif cluster.is_dynamic:
            cluster_type = "dynamic"
        else:
            cluster_type = "static"
        export_suffix = (
            f", export_points={cluster.shape_count}"
            if cluster.shape_count != cluster.num_collision_shapes
            else ""
        )
        print(
            f"{cluster.name}: seg_id={cluster.segmentation_id}, "
            f"geometry={cluster.collision_geometry}, "
            f"shapes={cluster.num_collision_shapes}{export_suffix}, "
            f"radius={float(cluster.shape_radius.detach().cpu()):.4f}, "
            f"type={cluster_type}, "
            f"control={cluster.control_mode}"
        )

    if step_output_dir is not None:
        export_rollout_step(scene=scene, step_dir=step_output_dir, step_idx=0)

    end_effector_cluster = find_cluster(scene, "end_effector")
    substeps = max(args.substeps, 1)
    sub_dt = args.sim_dt / substeps

    for step_idx, ee_action in enumerate(ee_actions, start=1):
        sub_action = ee_action / substeps
        for _sub in range(substeps):
            advance_prescribed_cluster(
                scene=scene,
                cluster=end_effector_cluster,
                delta_xyz=sub_action,
                dt=sub_dt,
            )
            step_scene(
                scene=scene,
                dt=sub_dt,
                velocity_damping=args.velocity_damping,
                max_velocity=args.max_velocity,
            )
        if step_output_dir is not None:
            export_rollout_step(scene=scene, step_dir=step_output_dir, step_idx=step_idx)
        body_q_frames.append(scene.state_0.body_q.detach().cpu().numpy().copy())

    export_scene_usd(
        scene=scene,
        output_path=scene_usd_path,
        body_q_frames=body_q_frames,
        fps=1.0 / args.sim_dt,
    )
    print(f"Animated USD written to {scene_usd_path.resolve()}")
    if step_output_dir is not None:
        print(f"Per-step scene PLY files written to {step_output_dir.resolve()}")


if __name__ == "__main__":
    main()
