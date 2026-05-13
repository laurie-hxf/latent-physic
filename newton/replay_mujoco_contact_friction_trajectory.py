from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

from fit_mujoco_contact_point_friction import (
    accumulate_batched_frame_loss_kernel,
    apply_batched_external_and_surface_point_forces_trajectory_kernel,
    combine_batched_loss_components_kernel,
    compute_batched_contact_weighted_masses_kernel,
    scatter_active_point_friction_kernel,
    sum_batched_losses_kernel,
)
from fit_mujoco_contact_point_friction_io import DEFAULT_TRAJECTORY_NPZ_PATH
from fit_mujoco_contact_point_friction_runtime import evaluate_collection_loss_in_batches, log_message
from mujoco_contact_friction_fit_utils import MujocoTrajectory, load_mujoco_trajectories
from newton_surface_points_diff_demo import build_diff_scene
from pbd_usd import export_scene_usd
from project_paths import DEFAULT_OUTPUT_DIR


@dataclass
class CheckpointParameters:
    path: Path
    iteration: int
    active_indices: np.ndarray
    active_params: np.ndarray
    best_active_params: np.ndarray
    trajectory_npz_path: Path | None
    max_steps: int | None
    max_trajectories: int | None


@dataclass
class ContactFrictionPointCloud:
    path: Path
    local_surface_points: np.ndarray
    point_friction: np.ndarray
    active_mask: np.ndarray | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument(
        "--trajectory-index",
        type=int,
        required=True,
        help="0-based trajectory index in the dataset NPZ after filtering invalid episodes.",
    )
    parser.add_argument(
        "--param-iteration",
        type=int,
        default=None,
        help=(
            "Load per-point friction from <checkpoint-point-cloud-dir>/iter_XXXXXX.ply for this training iteration. "
            "Use this when you want to replay an intermediate training round."
        ),
    )
    parser.add_argument(
        "--checkpoint-param-set",
        choices=("best", "current"),
        default="best",
        help="Which sparse parameter vector to use from --checkpoint-path when --param-iteration is not set.",
    )
    parser.add_argument(
        "--checkpoint-point-cloud-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing iter_XXXXXX.ply files written during training. "
            "Defaults to <checkpoint-path parent>/<checkpoint stem>_point_clouds."
        ),
    )
    parser.add_argument(
        "--reference-point-cloud",
        type=Path,
        default=None,
        help=(
            "Optional point cloud used only to infer the exact surface-point sampling/order when replaying "
            "sparse checkpoint parameters."
        ),
    )
    parser.add_argument(
        "--trajectory-npz",
        type=Path,
        default=None,
        help="Override the trajectory dataset. Defaults to the path saved in the checkpoint, then the repo default.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override trajectory truncation length during replay. By default this follows the checkpoint metadata.",
    )
    parser.add_argument("--scene-usd-path", type=Path, default=None)
    parser.add_argument("--summary-npz-path", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--position-loss-weight", type=float, default=1.0)
    parser.add_argument("--orientation-loss-weight", type=float, default=0.0)
    parser.add_argument("--linear-velocity-loss-weight", type=float, default=0.0)
    parser.add_argument("--angular-velocity-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--point-position-loss-reduction",
        choices=("sum", "mean"),
        default="mean",
    )
    parser.add_argument("--solver-iterations", type=int, default=10)
    parser.add_argument("--box-mass", type=float, default=1.0)
    parser.add_argument("--floor-half-extents", type=float, nargs=3, default=(2.0, 2.0, 0.05))
    parser.add_argument("--box-half-extents", type=float, nargs=3, default=(0.1, 0.05, 0.025))
    parser.add_argument("--box-start-pos", type=float, nargs=3, default=(0.58, 0.0, 0.025))
    parser.add_argument("--surface-point-spacing", type=float, default=0.02)
    parser.add_argument("--friction-contact-threshold", type=float, default=0.002)
    parser.add_argument("--point-friction", type=float, default=0.1)
    parser.add_argument("--contact-friction", type=float, default=0.0)
    parser.add_argument("--contact-stiffness", type=float, default=2.0e4)
    parser.add_argument("--contact-damping", type=float, default=50.0)
    parser.add_argument("--contact-margin", type=float, default=1.0e-3)
    parser.add_argument("--friction-regularization", type=float, default=1.0e-3)
    return parser.parse_args()


def _maybe_scalar_path(data: np.lib.npyio.NpzFile, key: str) -> Path | None:
    if key not in data.files:
        return None
    value = str(np.asarray(data[key]).item()).strip()
    if not value:
        return None
    return Path(value)


def _maybe_scalar_int(data: np.lib.npyio.NpzFile, key: str) -> int | None:
    if key not in data.files:
        return None
    value = int(np.asarray(data[key]).item())
    return None if value < 0 else value


def load_checkpoint_parameters(checkpoint_path: Path) -> CheckpointParameters:
    with np.load(checkpoint_path, allow_pickle=True) as data:
        return CheckpointParameters(
            path=checkpoint_path,
            iteration=int(np.asarray(data["iteration"]).item()),
            active_indices=np.asarray(data["active_indices"], dtype=np.int32),
            active_params=np.asarray(data["active_params"], dtype=np.float32),
            best_active_params=np.asarray(data["best_active_params"], dtype=np.float32),
            trajectory_npz_path=_maybe_scalar_path(data, "trajectory_npz_path"),
            max_steps=_maybe_scalar_int(data, "max_steps"),
            max_trajectories=_maybe_scalar_int(data, "max_trajectories"),
        )


def default_checkpoint_point_cloud_dir(checkpoint_path: Path) -> Path:
    candidates = [checkpoint_path.parent / f"{checkpoint_path.stem}_point_clouds"]
    if "ckpt" in checkpoint_path.stem:
        candidates.append(checkpoint_path.parent / checkpoint_path.stem.replace("ckpt", "point_clouds"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_iteration_point_cloud_path(
    checkpoint_path: Path,
    checkpoint_point_cloud_dir: Path | None,
    iteration: int,
) -> Path:
    point_cloud_dir = (
        checkpoint_point_cloud_dir
        if checkpoint_point_cloud_dir is not None
        else default_checkpoint_point_cloud_dir(checkpoint_path)
    )
    return point_cloud_dir / f"iter_{int(iteration):06d}.ply"


def load_contact_friction_point_cloud(point_cloud_path: Path) -> ContactFrictionPointCloud:
    vertex_count = None
    rows: list[list[str]] = []
    with point_cloud_path.open("r", encoding="utf-8") as f:
        in_header = True
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if in_header:
                if line.startswith("element vertex "):
                    vertex_count = int(line.split()[2])
                elif line == "end_header":
                    in_header = False
                continue
            rows.append(line.split())

    if vertex_count is None:
        raise ValueError(f"{point_cloud_path} is missing 'element vertex' in the PLY header")
    if len(rows) != vertex_count:
        raise ValueError(f"{point_cloud_path} expected {vertex_count} vertices, found {len(rows)}")

    points = np.zeros((vertex_count, 3), dtype=np.float32)
    point_friction = np.zeros(vertex_count, dtype=np.float32)
    active_mask = np.zeros(vertex_count, dtype=bool)
    point_indices = np.full(vertex_count, -1, dtype=np.int32)

    for row_idx, row in enumerate(rows):
        if len(row) < 9:
            raise ValueError(
                f"{point_cloud_path} row {row_idx} has {len(row)} columns, expected at least 9"
            )
        points[row_idx] = np.asarray(row[:3], dtype=np.float32)
        point_friction[row_idx] = np.float32(row[6])
        active_mask[row_idx] = bool(int(row[7]))
        point_indices[row_idx] = int(row[8])

    if np.any(point_indices < 0):
        raise ValueError(f"{point_cloud_path} contains invalid point_index values")
    if len(np.unique(point_indices)) != vertex_count:
        raise ValueError(f"{point_cloud_path} contains duplicate point_index values")

    reordered_points = np.zeros_like(points)
    reordered_friction = np.zeros_like(point_friction)
    reordered_active_mask = np.zeros_like(active_mask)
    reordered_points[point_indices] = points
    reordered_friction[point_indices] = point_friction
    reordered_active_mask[point_indices] = active_mask

    return ContactFrictionPointCloud(
        path=point_cloud_path,
        local_surface_points=reordered_points,
        point_friction=reordered_friction,
        active_mask=reordered_active_mask,
    )


def infer_box_half_extents_and_spacing(local_surface_points: np.ndarray) -> tuple[np.ndarray, float]:
    points = np.asarray(local_surface_points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"local_surface_points must have shape (N, 3), got {points.shape}")

    half_extents = np.max(np.abs(points), axis=0).astype(np.float32)
    tolerance = max(float(np.max(half_extents)), 1.0) * 1.0e-6

    def _face(axis: int, sign: float) -> np.ndarray:
        mask = np.isclose(points[:, axis], sign * half_extents[axis], atol=tolerance)
        face_points = points[mask]
        if len(face_points) == 0:
            raise ValueError(
                f"Could not infer sampling from face axis={axis} sign={sign:+.0f} for shape {points.shape}"
            )
        return face_points

    top_face = _face(2, 1.0)
    pos_y_face = _face(1, 1.0)
    pos_x_face = _face(0, 1.0)

    count_x_top = len(np.unique(np.round(top_face[:, 0], decimals=8)))
    count_y_top = len(np.unique(np.round(top_face[:, 1], decimals=8)))
    count_x_side = len(np.unique(np.round(pos_y_face[:, 0], decimals=8)))
    count_z_side = len(np.unique(np.round(pos_y_face[:, 2], decimals=8)))
    count_y_front = len(np.unique(np.round(pos_x_face[:, 1], decimals=8)))
    count_z_front = len(np.unique(np.round(pos_x_face[:, 2], decimals=8)))

    if count_x_top != count_x_side:
        raise ValueError(f"Inconsistent inferred x-face counts: top={count_x_top} side={count_x_side}")
    if count_y_top != count_y_front:
        raise ValueError(f"Inconsistent inferred y-face counts: top={count_y_top} front={count_y_front}")
    if count_z_side != count_z_front:
        raise ValueError(f"Inconsistent inferred z-face counts: side={count_z_side} front={count_z_front}")

    count_x = max(count_x_top, 1)
    count_y = max(count_y_top, 1)
    count_z = max(count_z_side, 1)
    spacing = max(
        (2.0 * float(half_extents[0])) / float(count_x),
        (2.0 * float(half_extents[1])) / float(count_y),
        (2.0 * float(half_extents[2])) / float(count_z),
    )
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError(f"Failed to infer a valid surface-point spacing from {points.shape}")

    return half_extents, float(spacing)


def build_reference_to_scene_index(reference_points: np.ndarray, scene_points: np.ndarray) -> np.ndarray:
    if reference_points.shape != scene_points.shape:
        raise ValueError(
            f"Reference points shape {reference_points.shape} does not match scene points shape {scene_points.shape}"
        )

    scale = 1.0e6

    def _key(point: np.ndarray) -> tuple[int, int, int]:
        return tuple(np.rint(np.asarray(point, dtype=np.float64) * scale).astype(np.int64).tolist())

    scene_lookup: dict[tuple[int, int, int], int] = {}
    for scene_idx, point in enumerate(scene_points):
        key = _key(point)
        if key in scene_lookup:
            raise ValueError("Scene surface points contain duplicates; cannot build a stable point mapping")
        scene_lookup[key] = scene_idx

    reference_to_scene = np.full(len(reference_points), -1, dtype=np.int32)
    for reference_idx, point in enumerate(reference_points):
        scene_idx = scene_lookup.get(_key(point))
        if scene_idx is None:
            raise ValueError("Reference point cloud does not match the replay scene surface-point set")
        reference_to_scene[reference_idx] = scene_idx

    if np.any(reference_to_scene < 0):
        raise ValueError("Failed to map all reference points into the replay scene")
    reordered_scene_points = scene_points[reference_to_scene]
    if not np.allclose(reordered_scene_points, reference_points, atol=1.0e-6):
        raise ValueError("Reference point cloud and replay scene points differ after mapping")

    return reference_to_scene


def infer_base_point_friction(point_cloud: ContactFrictionPointCloud, fallback: float) -> float:
    if point_cloud.active_mask is None:
        return float(fallback)
    inactive_mask = ~point_cloud.active_mask
    if not np.any(inactive_mask):
        return float(fallback)
    return float(np.median(point_cloud.point_friction[inactive_mask]))


def resolve_output_stem(args: argparse.Namespace, checkpoint: CheckpointParameters) -> str:
    if args.param_iteration is not None:
        param_tag = f"iter_{int(args.param_iteration):06d}"
    else:
        param_tag = str(args.checkpoint_param_set)
    return f"{checkpoint.path.stem}_{param_tag}_traj_{int(args.trajectory_index):04d}"


def resolve_output_path(maybe_path: Path | None, suffix: str, stem: str) -> Path:
    if maybe_path is not None:
        return maybe_path
    return DEFAULT_OUTPUT_DIR / f"{stem}{suffix}"


def resolve_trajectory_npz_path(args: argparse.Namespace, checkpoint: CheckpointParameters) -> Path:
    if args.trajectory_npz is not None:
        return args.trajectory_npz
    if checkpoint.trajectory_npz_path is not None:
        return checkpoint.trajectory_npz_path
    return DEFAULT_TRAJECTORY_NPZ_PATH


def select_trajectory(trajectory_npz_path: Path, max_steps: int | None, trajectory_index: int) -> MujocoTrajectory:
    trajectory_collection = load_mujoco_trajectories(
        trajectory_npz_path=trajectory_npz_path,
        max_steps=max_steps,
        max_trajectories=None,
    )
    trajectories = trajectory_collection.trajectories
    if trajectory_index < 0 or trajectory_index >= len(trajectories):
        raise IndexError(
            f"--trajectory-index={trajectory_index} is out of range for {trajectory_npz_path} "
            f"(loaded trajectories={len(trajectories)})"
        )
    trajectory = trajectories[trajectory_index]
    episode_index = trajectory.metadata.get("episode_index", trajectory_index)
    log_message(
        f"selected trajectory index={trajectory_index} episode_index={episode_index} "
        f"frames={trajectory.num_frames} steps={trajectory.num_steps} dt={trajectory.timestep:.6f}"
    )
    return trajectory


def main() -> None:
    args = parse_args()

    checkpoint = load_checkpoint_parameters(args.checkpoint_path)
    trajectory_npz_path = resolve_trajectory_npz_path(args, checkpoint)
    replay_max_steps = checkpoint.max_steps if args.max_steps is None else args.max_steps
    output_stem = resolve_output_stem(args, checkpoint)
    scene_usd_path = resolve_output_path(args.scene_usd_path, ".usda", output_stem)
    summary_npz_path = resolve_output_path(args.summary_npz_path, ".npz", output_stem)

    log_message(f"loading checkpoint metadata from {checkpoint.path.resolve()}")
    log_message(f"loading replay trajectory from {trajectory_npz_path.resolve()}")
    trajectory = select_trajectory(trajectory_npz_path, replay_max_steps, int(args.trajectory_index))

    parameter_point_cloud: ContactFrictionPointCloud | None = None
    reference_point_cloud: ContactFrictionPointCloud | None = None
    if args.param_iteration is not None:
        iteration_point_cloud_path = resolve_iteration_point_cloud_path(
            checkpoint_path=checkpoint.path,
            checkpoint_point_cloud_dir=args.checkpoint_point_cloud_dir,
            iteration=int(args.param_iteration),
        )
        if not iteration_point_cloud_path.exists():
            raise FileNotFoundError(
                f"Could not find point cloud for --param-iteration={args.param_iteration}: {iteration_point_cloud_path}"
            )
        parameter_point_cloud = load_contact_friction_point_cloud(iteration_point_cloud_path)
        reference_point_cloud = parameter_point_cloud
        log_message(f"using per-iteration point-cloud parameters from {iteration_point_cloud_path.resolve()}")
    elif args.reference_point_cloud is not None:
        if not args.reference_point_cloud.exists():
            raise FileNotFoundError(f"--reference-point-cloud does not exist: {args.reference_point_cloud}")
        reference_point_cloud = load_contact_friction_point_cloud(args.reference_point_cloud)
        log_message(f"using reference point cloud from {args.reference_point_cloud.resolve()}")
    else:
        auto_reference_path = resolve_iteration_point_cloud_path(
            checkpoint_path=checkpoint.path,
            checkpoint_point_cloud_dir=args.checkpoint_point_cloud_dir,
            iteration=checkpoint.iteration,
        )
        if auto_reference_path.exists():
            reference_point_cloud = load_contact_friction_point_cloud(auto_reference_path)
            log_message(f"using checkpoint iteration point cloud for geometry from {auto_reference_path.resolve()}")

    if reference_point_cloud is not None:
        inferred_half_extents, inferred_spacing = infer_box_half_extents_and_spacing(
            reference_point_cloud.local_surface_points
        )
        args.box_half_extents = inferred_half_extents.tolist()
        args.surface_point_spacing = inferred_spacing
        args.point_friction = infer_base_point_friction(reference_point_cloud, fallback=float(args.point_friction))
        log_message(
            "inferred scene sampling from reference point cloud "
            f"box_half_extents={np.asarray(args.box_half_extents, dtype=np.float32).tolist()} "
            f"surface_point_spacing={float(args.surface_point_spacing):.9g} "
            f"base_point_friction={float(args.point_friction):.9g}"
        )

    args.steps = trajectory.num_steps
    args.dt = trajectory.timestep
    args.batch_capacity = 1

    log_message(f"building replay scene on device={args.device if args.device is not None else 'auto'}")
    diff_scene = build_diff_scene(args)
    initial_body_q = diff_scene.states[0].body_q.numpy().copy()
    initial_body_qd = diff_scene.states[0].body_qd.numpy().copy()

    reference_to_scene = None
    if reference_point_cloud is not None:
        reference_to_scene = build_reference_to_scene_index(
            reference_point_cloud.local_surface_points,
            diff_scene.local_surface_points_np,
        )

    if parameter_point_cloud is not None:
        reference_point_friction = np.asarray(parameter_point_cloud.point_friction, dtype=np.float32)
        scene_point_friction = np.zeros_like(reference_point_friction)
        scene_point_friction[reference_to_scene] = reference_point_friction
        active_indices = np.arange(len(scene_point_friction), dtype=np.int32)
        active_params = scene_point_friction.astype(np.float32)
        param_source = f"point_cloud_iteration_{int(args.param_iteration):06d}"
    else:
        active_params_reference = (
            checkpoint.best_active_params
            if args.checkpoint_param_set == "best"
            else checkpoint.active_params
        )
        active_indices = np.asarray(checkpoint.active_indices, dtype=np.int32).copy()
        if reference_to_scene is not None:
            active_indices = reference_to_scene[active_indices]
        active_params = np.asarray(active_params_reference, dtype=np.float32)
        param_source = f"checkpoint_{args.checkpoint_param_set}"

    if active_indices.ndim != 1 or active_params.ndim != 1:
        raise ValueError("Active contact parameter arrays must be rank-1")
    if active_indices.shape != active_params.shape:
        raise ValueError(
            f"Active parameter shape mismatch: indices={active_indices.shape} params={active_params.shape}"
        )
    if len(active_indices) == 0:
        raise ValueError("No active contact-point parameters were resolved for replay")
    if np.min(active_indices) < 0 or np.max(active_indices) >= len(diff_scene.local_surface_points_np):
        raise ValueError(
            "Resolved active contact-point indices do not fit the replay scene surface-point set. "
            "Pass the training point cloud via --reference-point-cloud or use the matching scene sampling settings."
        )

    (
        final_loss,
        final_position_loss,
        final_orientation_loss,
        final_linear_velocity_loss,
        final_angular_velocity_loss,
        body_q_frames,
    ) = evaluate_collection_loss_in_batches(
        diff_scene=diff_scene,
        trajectories=[trajectory],
        args=args,
        active_indices=active_indices,
        active_params=active_params,
        initial_body_q=initial_body_q,
        initial_body_qd=initial_body_qd,
        eval_batch_size=1,
        trajectory_progress_every=1,
        scatter_active_point_friction_kernel=scatter_active_point_friction_kernel,
        compute_batched_contact_weighted_masses_kernel=compute_batched_contact_weighted_masses_kernel,
        apply_batched_external_and_surface_point_forces_trajectory_kernel=apply_batched_external_and_surface_point_forces_trajectory_kernel,
        accumulate_batched_frame_loss_kernel=accumulate_batched_frame_loss_kernel,
        combine_batched_loss_components_kernel=combine_batched_loss_components_kernel,
        sum_batched_losses_kernel=sum_batched_losses_kernel,
    )

    if not body_q_frames:
        raise RuntimeError("Replay did not produce any body pose frames")

    simulated_body_q = np.asarray(
        [np.asarray(frame[diff_scene.box_body], dtype=np.float32) for frame in body_q_frames],
        dtype=np.float32,
    )
    simulated_positions = simulated_body_q[:, :3]
    simulated_quaternions_xyzw = simulated_body_q[:, 3:]
    final_position_error = float(np.linalg.norm(simulated_positions[-1] - trajectory.positions[-1]))
    final_quaternion_dot = float(
        np.abs(np.sum(simulated_quaternions_xyzw[-1] * trajectory.quaternions_xyzw[-1]))
    )

    scene_usd_path.parent.mkdir(parents=True, exist_ok=True)
    export_scene_usd(
        scene=diff_scene.scene,
        output_path=scene_usd_path,
        body_q_frames=body_q_frames,
        fps=1.0 / float(args.dt),
    )

    summary_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        summary_npz_path,
        checkpoint_path=np.asarray(str(checkpoint.path.resolve())),
        parameter_source=np.asarray(param_source),
        parameter_iteration=np.asarray(-1 if args.param_iteration is None else int(args.param_iteration), dtype=np.int32),
        checkpoint_iteration=np.asarray(checkpoint.iteration, dtype=np.int32),
        trajectory_npz_path=np.asarray(str(trajectory_npz_path.resolve())),
        trajectory_index=np.asarray(int(args.trajectory_index), dtype=np.int32),
        trajectory_episode_index=np.asarray(int(trajectory.metadata.get("episode_index", args.trajectory_index)), dtype=np.int32),
        local_surface_points=diff_scene.local_surface_points_np,
        active_contact_point_indices=active_indices,
        active_point_friction=active_params,
        target_time=trajectory.time,
        target_positions=trajectory.positions,
        target_quaternions_xyzw=trajectory.quaternions_xyzw,
        target_linear_velocity=trajectory.linear_velocity,
        target_angular_velocity=trajectory.angular_velocity,
        target_step_forces=trajectory.step_forces,
        target_step_application_points=trajectory.step_application_points,
        simulated_positions=simulated_positions,
        simulated_quaternions_xyzw=simulated_quaternions_xyzw,
        simulated_body_q=simulated_body_q,
        mean_loss=np.asarray(final_loss, dtype=np.float32),
        mean_position_loss=np.asarray(final_position_loss, dtype=np.float32),
        mean_orientation_loss=np.asarray(final_orientation_loss, dtype=np.float32),
        mean_linear_velocity_loss=np.asarray(final_linear_velocity_loss, dtype=np.float32),
        mean_angular_velocity_loss=np.asarray(final_angular_velocity_loss, dtype=np.float32),
        final_position_error=np.asarray(final_position_error, dtype=np.float32),
        final_quaternion_abs_dot=np.asarray(final_quaternion_dot, dtype=np.float32),
        scene_usd_path=np.asarray(str(scene_usd_path.resolve())),
    )

    log_message(
        f"replay complete | param_source={param_source} | "
        f"mean_loss={final_loss:.6g} | final_position_error={final_position_error:.6g} | "
        f"final_quaternion_abs_dot={final_quaternion_dot:.6g}"
    )
    log_message(f"scene_usd_written_to={scene_usd_path.resolve()}")
    log_message(f"summary_npz_written_to={summary_npz_path.resolve()}")


if __name__ == "__main__":
    main()
