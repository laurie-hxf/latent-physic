from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
NEWTON_DIR = REPO_ROOT / "newton"
for _path in (REPO_ROOT, NEWTON_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from mujoco_contact_friction_fit_utils import load_mujoco_trajectories  # noqa: E402
from newton_surface_points_diff_demo import build_diff_scene  # noqa: E402

from pointnet_residual_adapter.checkpoints import load_adapter_checkpoint  # noqa: E402
from pointnet_residual_adapter.features import DinoFeatures, load_aligned_dino_features, quaternion_xyzw_to_yaw  # noqa: E402
from pointnet_residual_adapter.features import normalize_residual_output_mode  # noqa: E402
from pointnet_residual_adapter.newton_rollout import (  # noqa: E402
    build_rollout_buffers,
    run_closed_loop_pointnet_rollout,
    run_open_loop_rollout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--trajectory-npz", type=Path, required=True)
    parser.add_argument("--trajectory-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-npz", type=Path, default=None)
    parser.add_argument(
        "--pointnet-residual-gain",
        type=float,
        default=None,
        help="Scale the neural-adapter residual. None uses checkpoint training metadata, falling back to 1.0.",
    )
    parser.add_argument(
        "--pointnet-residual-output-mode",
        choices=("checkpoint", "velocity", "acceleration", "pose", "position", "pose_velocity", "all"),
        default="checkpoint",
        help="How to interpret neural-adapter outputs at rollout time. checkpoint uses checkpoint metadata.",
    )
    parser.add_argument(
        "--stateful-reset-interval",
        type=int,
        default=None,
        help="Reset stateful adapter memory every N steps. None uses checkpoint metadata; 0 never resets.",
    )
    parser.add_argument(
        "--dino-feature-npz",
        type=Path,
        default=None,
        help="Override the DINO path stored in the adapter checkpoint. Alignment and values must still match.",
    )
    return parser.parse_args()


def _namespace_from_metadata(metadata: dict, args: argparse.Namespace, trajectory) -> argparse.Namespace:
    contact = dict(metadata["contact_parameters"])
    output_mode = (
        str(metadata.get("residual_output_mode", "velocity"))
        if str(args.pointnet_residual_output_mode) == "checkpoint"
        else str(args.pointnet_residual_output_mode)
    )
    return argparse.Namespace(
        device=args.device,
        steps=int(trajectory.num_steps),
        dt=float(trajectory.timestep),
        batch_capacity=1,
        solver_iterations=int(contact["solver_iterations"]),
        box_mass=float(contact["box_mass"]),
        floor_half_extents=contact["floor_half_extents"],
        box_half_extents=metadata["box_half_extents"],
        box_start_pos=contact.get("box_start_pos"),
        surface_point_spacing=float(metadata["surface_point_spacing"]),
        contact_friction=float(contact["contact_friction"]),
        point_friction=float(metadata.get("point_friction", 0.35)),
        contact_stiffness=float(contact["contact_stiffness"]),
        contact_damping=float(contact["contact_damping"]),
        contact_margin=float(contact["contact_margin"]),
        friction_contact_threshold=float(contact["friction_contact_threshold"]),
        contact_mask_threshold=float(contact["contact_mask_threshold"]),
        friction_regularization=float(contact["friction_regularization"]),
        history_window_steps=int(metadata["history_window_steps"]),
        prediction_window_steps=int(metadata["prediction_window_steps"]),
        pointnet_residual_gain=(
            float(metadata.get("pointnet_residual_gain", 1.0))
            if args.pointnet_residual_gain is None
            else float(args.pointnet_residual_gain)
        ),
        pointnet_residual_output_mode=normalize_residual_output_mode(output_mode),
        stateful_reset_interval=(
            int(metadata.get("stateful_reset_interval", 0))
            if args.stateful_reset_interval is None
            else int(args.stateful_reset_interval)
        ),
    )


def _validate_dino(checkpoint, metadata: dict, override_path: Path | None, local_surface_points: np.ndarray) -> DinoFeatures | None:
    if int(metadata.get("dino_feature_dim", 0)) <= 0:
        return None
    dino_path = override_path if override_path is not None else Path(str(metadata["dino_feature_npz"]))
    dino = load_aligned_dino_features(
        dino_path,
        local_surface_points,
        max_match_distance=float(metadata.get("dino_max_match_distance", 1.0e-5)),
    )
    if checkpoint.dino_features is None or checkpoint.dino_bottom_feature_copied_from_top is None:
        raise ValueError("Adapter checkpoint metadata says DINO is enabled, but the checkpoint does not store aligned DINO tensors")
    if not np.allclose(dino.features, checkpoint.dino_features, atol=1.0e-6):
        raise ValueError("Current DINO feature alignment/values do not match the adapter checkpoint")
    if not np.allclose(
        dino.bottom_feature_copied_from_top,
        checkpoint.dino_bottom_feature_copied_from_top,
        atol=1.0e-6,
    ):
        raise ValueError("Current DINO bottom_feature_copied_from_top flags do not match the adapter checkpoint")
    return dino


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle)).astype(np.float32)


def _rollout_metrics(prefix: str, predicted, target) -> dict[str, float]:
    frames = min(predicted.positions.shape[1], target.positions.shape[0])
    pred_pos = predicted.positions[0, :frames]
    pred_quat = predicted.quaternions_xyzw[0, :frames]
    pred_lin = predicted.linear_velocity[0, :frames]
    pred_ang = predicted.angular_velocity[0, :frames]
    gt_pos = target.positions[:frames]
    gt_quat = target.quaternions_xyzw[:frames]
    gt_lin = target.linear_velocity[:frames]
    gt_ang = target.angular_velocity[:frames]
    yaw_error = _wrap_angle(quaternion_xyzw_to_yaw(pred_quat) - quaternion_xyzw_to_yaw(gt_quat))
    return {
        f"{prefix}_position_mse": float(np.mean((pred_pos - gt_pos) ** 2)),
        f"{prefix}_position_xy_mse": float(np.mean((pred_pos[:, :2] - gt_pos[:, :2]) ** 2)),
        f"{prefix}_yaw_mse": float(np.mean(yaw_error * yaw_error)),
        f"{prefix}_linear_velocity_xy_mse": float(np.mean((pred_lin[:, :2] - gt_lin[:, :2]) ** 2)),
        f"{prefix}_angular_velocity_z_mse": float(np.mean((pred_ang[:, 2] - gt_ang[:, 2]) ** 2)),
        f"{prefix}_final_position_error": float(np.linalg.norm(pred_pos[-1] - gt_pos[-1])),
        f"{prefix}_final_xy_error": float(np.linalg.norm(pred_pos[-1, :2] - gt_pos[-1, :2])),
    }


def main() -> None:
    args = parse_args()
    checkpoint = load_adapter_checkpoint(args.adapter_checkpoint, map_location="cpu")
    metadata = checkpoint.metadata
    collection = load_mujoco_trajectories(
        trajectory_npz_path=args.trajectory_npz,
        max_steps=args.max_steps,
        max_trajectories=None,
    )
    if args.trajectory_index < 0 or args.trajectory_index >= len(collection.trajectories):
        raise IndexError(f"--trajectory-index={args.trajectory_index} is out of range")
    trajectory = collection.trajectories[int(args.trajectory_index)]

    scene_args = _namespace_from_metadata(metadata, args, trajectory)
    diff_scene = build_diff_scene(scene_args)
    if not np.allclose(diff_scene.local_surface_points_np, checkpoint.local_surface_points, atol=1.0e-6):
        raise ValueError("Current surface-point grid does not match adapter checkpoint metadata")
    initial_body_q = diff_scene.states[0].body_q.numpy().copy()
    initial_body_qd = diff_scene.states[0].body_qd.numpy().copy()
    dino = _validate_dino(checkpoint, metadata, args.dino_feature_npz, diff_scene.local_surface_points_np)

    buffers = build_rollout_buffers(
        device=str(diff_scene.torch_device),
        batch_capacity=1,
        step_capacity=int(trajectory.num_steps),
        point_count=len(diff_scene.local_surface_points_np),
        full_point_friction=checkpoint.full_point_friction,
    )

    baseline = run_open_loop_rollout(
        diff_scene=diff_scene,
        buffers=buffers,
        trajectories=[trajectory],
        args=scene_args,
        initial_body_q=initial_body_q,
        initial_body_qd=initial_body_qd,
    )

    torch_device = diff_scene.torch_device
    model = checkpoint.model.to(torch_device)
    corrected, residuals = run_closed_loop_pointnet_rollout(
        diff_scene=diff_scene,
        buffers=buffers,
        trajectory=trajectory,
        args=scene_args,
        initial_body_q=initial_body_q,
        initial_body_qd=initial_body_qd,
        model=model,
        normalizer=checkpoint.normalizer,
        local_surface_points=diff_scene.local_surface_points_np,
        box_half_extents=np.asarray(metadata["box_half_extents"], dtype=np.float32),
        point_friction=checkpoint.full_point_friction,
        active_contact_mask=checkpoint.active_contact_mask,
        dino=dino,
        torch_device=torch_device,
    )
    stateful_diagnostics = getattr(model, "last_stateful_rollout_diagnostics", None)

    metrics = {}
    metrics.update(_rollout_metrics("newton", baseline, trajectory))
    metrics.update(_rollout_metrics("pointnet", corrected, trajectory))

    output_path = args.output_npz
    if output_path is None:
        output_dir = args.adapter_checkpoint.parent / "rollout"
        output_path = output_dir / f"{args.adapter_checkpoint.stem}_traj_{int(args.trajectory_index):04d}.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        adapter_checkpoint=np.asarray(str(args.adapter_checkpoint.resolve())),
        trajectory_npz=np.asarray(str(args.trajectory_npz.resolve())),
        trajectory_index=np.asarray(int(args.trajectory_index), dtype=np.int32),
        metrics=np.asarray(metrics, dtype=object),
        target_time=trajectory.time,
        target_positions=trajectory.positions,
        target_quaternions_xyzw=trajectory.quaternions_xyzw,
        target_linear_velocity=trajectory.linear_velocity,
        target_angular_velocity=trajectory.angular_velocity,
        target_step_forces=trajectory.step_forces,
        newton_positions=baseline.positions[0],
        newton_quaternions_xyzw=baseline.quaternions_xyzw[0],
        newton_linear_velocity=baseline.linear_velocity[0],
        newton_angular_velocity=baseline.angular_velocity[0],
        pointnet_positions=corrected.positions[0],
        pointnet_quaternions_xyzw=corrected.quaternions_xyzw[0],
        pointnet_linear_velocity=corrected.linear_velocity[0],
        pointnet_angular_velocity=corrected.angular_velocity[0],
        pointnet_applied_residuals=residuals,
        pointnet_residual_gain=np.asarray(float(scene_args.pointnet_residual_gain), dtype=np.float32),
        pointnet_residual_output_mode=np.asarray(str(scene_args.pointnet_residual_output_mode)),
        stateful_reset_interval=np.asarray(int(scene_args.stateful_reset_interval), dtype=np.int32),
        stateful_hidden_l2_norm=(
            np.asarray([], dtype=np.float32)
            if not isinstance(stateful_diagnostics, dict)
            else np.asarray(stateful_diagnostics["hidden_l2_norm"], dtype=np.float32)
        ),
        stateful_hidden_saturation_fraction=(
            np.asarray([], dtype=np.float32)
            if not isinstance(stateful_diagnostics, dict)
            else np.asarray(stateful_diagnostics["hidden_saturation_fraction"], dtype=np.float32)
        ),
        full_point_friction=checkpoint.full_point_friction,
        active_contact_mask=checkpoint.active_contact_mask,
    )
    print(
        "rollout complete "
        f"newton_pos_mse={metrics['newton_position_xy_mse']:.6g} "
        f"pointnet_pos_mse={metrics['pointnet_position_xy_mse']:.6g} "
        f"newton_vel_mse={metrics['newton_linear_velocity_xy_mse']:.6g} "
        f"pointnet_vel_mse={metrics['pointnet_linear_velocity_xy_mse']:.6g}",
        flush=True,
    )
    print(f"summary={output_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
