from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Polygon
import numpy as np
import warp as wp

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
NEWTON_DIR = ROOT / "newton"
if str(NEWTON_DIR) not in sys.path:
    sys.path.insert(0, str(NEWTON_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fit_mujoco_contact_point_friction import (  # noqa: E402
    accumulate_batched_frame_loss_kernel,
    apply_batched_external_and_surface_point_forces_trajectory_kernel,
    combine_batched_loss_components_kernel,
    compute_batched_contact_weighted_masses_kernel,
    scatter_active_point_friction_kernel,
    sum_batched_losses_kernel,
)
from fit_mujoco_contact_point_friction_io import DEFAULT_TRAJECTORY_NPZ_PATH  # noqa: E402
from fit_mujoco_contact_point_friction_runtime import evaluate_collection_loss_in_batches, log_message  # noqa: E402
from newton_surface_points_diff_demo import build_diff_scene  # noqa: E402
from replay_mujoco_contact_friction_trajectory import (  # noqa: E402
    ContactFrictionPointCloud,
    build_reference_to_scene_index,
    infer_base_point_friction,
    infer_box_half_extents_and_spacing,
    load_checkpoint_parameters,
    load_contact_friction_point_cloud,
    resolve_iteration_point_cloud_path,
    select_trajectory,
)

OUTPUTS_ROOT = ROOT / "outputs"
PALETTE = (
    "#d95f02",
    "#1b9e77",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#666666",
)


def experiment_checkpoint(experiment_name: str) -> Path:
    return OUTPUTS_ROOT / experiment_name / f"{experiment_name}.npz"


def experiment_point_cloud(experiment_name: str) -> Path:
    return OUTPUTS_ROOT / experiment_name / f"{experiment_name}.ply"


def experiment_point_cloud_dir(experiment_name: str) -> Path | None:
    experiment_dir = OUTPUTS_ROOT / experiment_name
    preferred = experiment_dir / f"{experiment_name}_point_clouds"
    if preferred.exists():
        return preferred
    candidates = sorted(path for path in experiment_dir.glob("*point_cloud*") if path.is_dir())
    return candidates[0] if candidates else None


@dataclass(frozen=True)
class RunSpec:
    label: str
    checkpoint_path: Path
    experiment_name: str | None = None
    reference_point_cloud: Path | None = None
    checkpoint_point_cloud_dir: Path | None = None


@dataclass
class PredictionResult:
    label: str
    checkpoint_path: Path
    parameter_source: str
    parameter_summary: str
    positions: np.ndarray
    quaternions: np.ndarray
    mean_loss: float
    position_loss: float
    orientation_loss: float
    linear_velocity_loss: float
    angular_velocity_loss: float
    final_xy_error: float
    half_extents: np.ndarray

    @property
    def legend_label(self) -> str:
        return f"{self.label} | {self.parameter_summary}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("experiment_name_arg", nargs="*", help="Experiment name(s) under outputs/.")
    parser.add_argument("--experiment-name", dest="experiment_name_options", nargs="+", action="append", default=[])
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--checkpoint-paths", type=Path, nargs="+", default=None)
    parser.add_argument("--labels", type=str, nargs="+", default=None)
    parser.add_argument("--trajectory-index", type=int, default=0)
    parser.add_argument("--trajectory-npz", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--checkpoint-param-set",
        choices=("best", "current"),
        default="best",
        help="Sparse parameter vector to replay when --param-iteration is not set.",
    )
    parser.add_argument(
        "--param-iteration",
        type=int,
        default=None,
        help="Load point friction from <checkpoint-point-cloud-dir>/iter_XXXXXX.ply.",
    )
    parser.add_argument("--checkpoint-point-cloud-dir", type=Path, default=None)
    parser.add_argument(
        "--reference-point-cloud",
        type=Path,
        default=None,
        help="Point cloud used to infer/reorder surface points for older checkpoint geometry.",
    )
    parser.add_argument("--reference-point-clouds", type=Path, nargs="+", default=None)

    parser.add_argument("--solver-iterations", type=int, default=10)
    parser.add_argument("--box-mass", type=float, default=1.0)
    parser.add_argument("--floor-half-extents", type=float, nargs=3, default=(2.0, 2.0, 0.05))
    parser.add_argument("--box-half-extents", type=float, nargs=3, default=(0.1, 0.05, 0.025))
    parser.add_argument("--box-start-pos", type=float, nargs=3, default=(0.58, 0.0, 0.025))
    parser.add_argument("--surface-point-spacing", type=float, default=0.01)
    parser.add_argument("--friction-contact-threshold", type=float, default=0.002)
    parser.add_argument("--point-friction", type=float, default=0.1)
    parser.add_argument("--contact-friction", type=float, default=0.0)
    parser.add_argument("--contact-stiffness", type=float, default=1.0e5)
    parser.add_argument("--contact-damping", type=float, default=50.0)
    parser.add_argument("--contact-margin", type=float, default=1.0e-3)
    parser.add_argument("--friction-regularization", type=float, default=1.0e-3)

    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--trail-frames", type=int, default=90)
    parser.add_argument("--force-arrow-length", type=float, default=0.045)
    parser.add_argument("--bitrate", type=int, default=2400)
    args = parser.parse_args()
    args.run_specs = resolve_run_specs(args, parser)
    return args


def resolve_run_specs(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[RunSpec]:
    experiment_names = list(args.experiment_name_arg)
    for group in args.experiment_name_options:
        experiment_names.extend(group)

    checkpoint_paths: list[Path] = []
    if args.checkpoint_path is not None:
        checkpoint_paths.append(args.checkpoint_path)
    if args.checkpoint_paths is not None:
        checkpoint_paths.extend(args.checkpoint_paths)

    if not experiment_names and not checkpoint_paths:
        parser.error("provide at least one experiment name or checkpoint path")

    total_runs = len(experiment_names) + len(checkpoint_paths)
    if args.labels is not None and len(args.labels) != total_runs:
        parser.error("--labels must have the same count as experiments/checkpoints")
    if args.reference_point_cloud is not None and total_runs != 1 and args.reference_point_clouds is None:
        parser.error("--reference-point-cloud is only unambiguous for one run; use --reference-point-clouds")
    if args.reference_point_clouds is not None and len(args.reference_point_clouds) != total_runs:
        parser.error("--reference-point-clouds must have the same count as experiments/checkpoints")

    labels = args.labels or []
    reference_clouds = args.reference_point_clouds or []
    specs: list[RunSpec] = []

    for run_idx, experiment_name in enumerate(experiment_names):
        reference_point_cloud = None
        if reference_clouds:
            reference_point_cloud = reference_clouds[run_idx]
        else:
            candidate = experiment_point_cloud(experiment_name)
            if candidate.exists():
                reference_point_cloud = candidate
        specs.append(
            RunSpec(
                label=labels[run_idx] if labels else experiment_name,
                checkpoint_path=experiment_checkpoint(experiment_name),
                experiment_name=experiment_name,
                reference_point_cloud=reference_point_cloud,
                checkpoint_point_cloud_dir=experiment_point_cloud_dir(experiment_name),
            )
        )

    checkpoint_offset = len(experiment_names)
    for local_idx, checkpoint_path in enumerate(checkpoint_paths):
        run_idx = checkpoint_offset + local_idx
        specs.append(
            RunSpec(
                label=labels[run_idx] if labels else checkpoint_path.stem,
                checkpoint_path=checkpoint_path,
                reference_point_cloud=(
                    reference_clouds[run_idx]
                    if reference_clouds
                    else args.reference_point_cloud
                    if total_runs == 1
                    else None
                ),
                checkpoint_point_cloud_dir=args.checkpoint_point_cloud_dir if total_runs == 1 else None,
            )
        )
    return specs


def make_eval_args(args: argparse.Namespace, trajectory) -> argparse.Namespace:
    return argparse.Namespace(
        trajectory_npz=args.trajectory_npz,
        max_steps=trajectory.num_steps,
        max_trajectories=None,
        batch_size=1,
        trajectory_progress_every=0,
        device=args.device,
        steps=trajectory.num_steps,
        dt=trajectory.timestep,
        batch_capacity=1,
        solver_iterations=args.solver_iterations,
        box_mass=args.box_mass,
        floor_half_extents=args.floor_half_extents,
        box_half_extents=args.box_half_extents,
        box_start_pos=args.box_start_pos,
        surface_point_spacing=args.surface_point_spacing,
        friction_contact_threshold=args.friction_contact_threshold,
        contact_mask_threshold=args.friction_contact_threshold,
        point_friction=args.point_friction,
        contact_friction=args.contact_friction,
        contact_stiffness=args.contact_stiffness,
        contact_damping=args.contact_damping,
        contact_margin=args.contact_margin,
        friction_regularization=args.friction_regularization,
        initial_force=(0.0, 0.0, 0.0),
        initial_torque=(0.0, 0.0, 0.0),
        force_magnitude=None,
        force_direction=None,
        force_point=None,
        position_loss_weight=1.0,
        orientation_loss_weight=0.0,
        linear_velocity_loss_weight=0.0,
        angular_velocity_loss_weight=0.0,
        point_position_loss_reduction="mean",
        avoid_zero_surface_point_x=getattr(args, "avoid_zero_surface_point_x", True),
    )


def resolve_trajectory_npz_path(args: argparse.Namespace, checkpoint) -> Path:
    if args.trajectory_npz is not None:
        return resolve_existing_trajectory_npz_path(args.trajectory_npz)
    if checkpoint.trajectory_npz_path is not None:
        return resolve_existing_trajectory_npz_path(checkpoint.trajectory_npz_path)
    return resolve_existing_trajectory_npz_path(DEFAULT_TRAJECTORY_NPZ_PATH)


def resolve_existing_trajectory_npz_path(path: Path) -> Path:
    if path.exists():
        return path
    organized_path = path.parent / path.stem / path.name
    if organized_path.exists():
        return organized_path
    return path


def default_output_path(args: argparse.Namespace, checkpoint=None) -> Path:
    if args.param_iteration is not None:
        param_tag = f"iter_{int(args.param_iteration):06d}"
    else:
        param_tag = args.checkpoint_param_set
    specs: list[RunSpec] = args.run_specs
    if len(specs) == 1 and specs[0].experiment_name is not None:
        experiment_name = specs[0].experiment_name
        return OUTPUTS_ROOT / experiment_name / "videos" / (
            f"{experiment_name}_{param_tag}_traj_{int(args.trajectory_index):04d}_comparison.mp4"
        )
    run_tag = "_vs_".join(sanitize_filename(spec.label) for spec in specs[:3])
    if len(specs) > 3:
        run_tag += f"_plus_{len(specs) - 3}"
    return ROOT / "outputs" / "comparison_videos" / f"{run_tag}_{param_tag}_traj_{int(args.trajectory_index):04d}.mp4"


def sanitize_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("_") or "run"


def resolve_reference_point_cloud(
    args: argparse.Namespace,
    checkpoint,
    run_spec: RunSpec,
) -> tuple[ContactFrictionPointCloud | None, ContactFrictionPointCloud | None]:
    parameter_point_cloud = None
    reference_point_cloud = None
    if args.param_iteration is not None:
        iteration_path = resolve_iteration_point_cloud_path(
            checkpoint_path=checkpoint.path,
            checkpoint_point_cloud_dir=run_spec.checkpoint_point_cloud_dir or args.checkpoint_point_cloud_dir,
            iteration=int(args.param_iteration),
        )
        if not iteration_path.exists():
            raise FileNotFoundError(f"Missing --param-iteration point cloud: {iteration_path}")
        parameter_point_cloud = load_contact_friction_point_cloud(iteration_path)
        reference_point_cloud = parameter_point_cloud
        log_message(f"using per-iteration point-cloud parameters from {iteration_path.resolve()}")
    elif run_spec.reference_point_cloud is not None:
        if not run_spec.reference_point_cloud.exists():
            raise FileNotFoundError(f"reference point cloud does not exist: {run_spec.reference_point_cloud}")
        reference_point_cloud = load_contact_friction_point_cloud(run_spec.reference_point_cloud)
        log_message(f"using reference point cloud from {run_spec.reference_point_cloud.resolve()}")
    else:
        auto_reference_path = resolve_iteration_point_cloud_path(
            checkpoint_path=checkpoint.path,
            checkpoint_point_cloud_dir=run_spec.checkpoint_point_cloud_dir or args.checkpoint_point_cloud_dir,
            iteration=checkpoint.iteration,
        )
        if auto_reference_path.exists():
            reference_point_cloud = load_contact_friction_point_cloud(auto_reference_path)
            log_message(f"using checkpoint iteration point cloud for geometry from {auto_reference_path.resolve()}")
    return parameter_point_cloud, reference_point_cloud


def resolve_active_parameters(
    *,
    args: argparse.Namespace,
    checkpoint,
    diff_scene,
    parameter_point_cloud: ContactFrictionPointCloud | None,
    reference_point_cloud: ContactFrictionPointCloud | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    reference_to_scene = None
    if reference_point_cloud is not None:
        reference_to_scene = build_reference_to_scene_index(
            reference_point_cloud.local_surface_points,
            diff_scene.local_surface_points_np,
        )

    if parameter_point_cloud is not None:
        if reference_to_scene is None:
            raise ValueError("internal error: parameter point cloud needs a reference-to-scene map")
        reference_friction = np.asarray(parameter_point_cloud.point_friction, dtype=np.float32)
        scene_friction = np.zeros_like(reference_friction)
        scene_friction[reference_to_scene] = reference_friction
        return (
            np.arange(len(scene_friction), dtype=np.int32),
            scene_friction.astype(np.float32),
            f"point_cloud_iteration_{int(args.param_iteration):06d}",
        )

    active_params_reference = (
        checkpoint.best_active_params if args.checkpoint_param_set == "best" else checkpoint.active_params
    )
    active_indices = np.asarray(checkpoint.active_indices, dtype=np.int32).copy()
    if reference_to_scene is not None:
        active_indices = reference_to_scene[active_indices]
    active_params = np.asarray(active_params_reference, dtype=np.float32)
    return active_indices, active_params, f"checkpoint_{args.checkpoint_param_set}"


def _npz_scalar(data: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in data.files:
        return default
    value = np.asarray(data[key])
    return value.item() if value.shape == () else value.tolist()


def _checkpoint_optimizer_params(checkpoint_path: Path, param_set: str) -> np.ndarray | None:
    key = "best_optimizer_params" if param_set == "best" else "optimizer_params"
    with np.load(checkpoint_path, allow_pickle=True) as data:
        if key in data.files:
            return np.asarray(data[key], dtype=np.float32).reshape(-1)
        return None


def summarize_friction_parameters(
    *,
    checkpoint_path: Path,
    checkpoint_param_set: str,
    active_indices: np.ndarray,
    active_params: np.ndarray,
    local_surface_points: np.ndarray,
) -> str:
    active_params = np.asarray(active_params, dtype=np.float32).reshape(-1)
    active_indices = np.asarray(active_indices, dtype=np.int32).reshape(-1)
    optimizer_params = _checkpoint_optimizer_params(checkpoint_path, checkpoint_param_set)
    with np.load(checkpoint_path, allow_pickle=True) as data:
        parameterization = str(_npz_scalar(data, "friction_parameterization", "point"))

    if parameterization == "global" and optimizer_params is not None and len(optimizer_params) >= 1:
        return f"global mu={float(optimizer_params[0]):.3f}"

    if parameterization == "left-right" and optimizer_params is not None and len(optimizer_params) >= 2:
        return f"left-right muL={float(optimizer_params[0]):.3f} muR={float(optimizer_params[1]):.3f}"

    if parameterization == "base-delta" and optimizer_params is not None and len(optimizer_params) >= 3:
        mu_base = float(optimizer_params[0])
        delta_left = float(optimizer_params[1])
        delta_right = float(optimizer_params[2])
        return f"base-delta base={mu_base:.3f} muL={mu_base + delta_left:.3f} muR={mu_base + delta_right:.3f}"

    side_ids = np.full(len(active_params), -1, dtype=np.int32)
    if len(active_indices) > 0:
        local_x = np.asarray(local_surface_points, dtype=np.float32)[active_indices, 0]
        side_ids[local_x < 0.0] = 0
        side_ids[local_x > 0.0] = 1
    left = active_params[side_ids == 0]
    right = active_params[side_ids == 1]
    left_mean = float(np.mean(left)) if len(left) else float("nan")
    right_mean = float(np.mean(right)) if len(right) else float("nan")
    return (
        f"{parameterization} N={len(active_params)} "
        f"mu={float(np.mean(active_params)):.3f}+/-{float(np.std(active_params)):.3f} "
        f"L/R={left_mean:.3f}/{right_mean:.3f}"
    )


def reference_point_cloud_has_center_x(local_surface_points: np.ndarray) -> bool:
    points = np.asarray(local_surface_points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        return False
    return bool(np.any(np.isclose(points[:, 0], 0.0, atol=1.0e-7)))


def normalize_quaternions_xyzw(quaternions: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternions, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, 1.0e-8)


def rotation_matrix_xyzw(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quaternions_xyzw(np.asarray(quaternion, dtype=np.float32).reshape(1, 4))[0]
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def yaw_from_xyzw(quaternion: np.ndarray) -> float:
    x, y, z, w = normalize_quaternions_xyzw(np.asarray(quaternion, dtype=np.float32).reshape(1, 4))[0]
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def wrap_angle_radians(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def transform_local_point(position: np.ndarray, quaternion: np.ndarray, local_point: np.ndarray) -> np.ndarray:
    return np.asarray(position, dtype=np.float32) + rotation_matrix_xyzw(quaternion) @ np.asarray(local_point, dtype=np.float32)


def block_corners_xy(position: np.ndarray, quaternion: np.ndarray, half_extents: np.ndarray) -> np.ndarray:
    hx, hy = float(half_extents[0]), float(half_extents[1])
    local_corners = np.asarray(
        [
            [-hx, -hy, 0.0],
            [hx, -hy, 0.0],
            [hx, hy, 0.0],
            [-hx, hy, 0.0],
        ],
        dtype=np.float32,
    )
    world = np.asarray(position, dtype=np.float32) + local_corners @ rotation_matrix_xyzw(quaternion).T
    return world[:, :2]


def extract_box_pose_frames(body_q_frames: list[np.ndarray], box_body: int) -> tuple[np.ndarray, np.ndarray]:
    body_q = np.asarray(
        [np.asarray(frame[box_body], dtype=np.float32).reshape(-1) for frame in body_q_frames],
        dtype=np.float32,
    )
    if body_q.shape[1] < 7:
        raise ValueError(f"Unexpected body_q frame shape: {body_q.shape}")
    positions = body_q[:, :3]
    quaternions = normalize_quaternions_xyzw(body_q[:, 3:7])
    return positions, quaternions


def build_frame_indices(frame_count: int, stride: int) -> np.ndarray:
    stride = max(int(stride), 1)
    indices = np.arange(0, frame_count, stride, dtype=np.int32)
    if len(indices) == 0 or int(indices[-1]) != frame_count - 1:
        indices = np.concatenate([indices, np.asarray([frame_count - 1], dtype=np.int32)])
    return indices


def render_video(
    *,
    output_path: Path,
    trajectory,
    target_positions: np.ndarray,
    target_quaternions: np.ndarray,
    predictions: list[PredictionResult],
    half_extents: np.ndarray,
    fps: int,
    dpi: int,
    frame_stride: int,
    trail_frames: int,
    force_arrow_length: float,
    bitrate: int,
) -> None:
    if not predictions:
        raise ValueError("No predictions to render")
    frame_count = min([len(target_positions), *(len(result.positions) for result in predictions)])
    target_positions = target_positions[:frame_count]
    target_quaternions = target_quaternions[:frame_count]
    for result in predictions:
        result.positions = result.positions[:frame_count]
        result.quaternions = result.quaternions[:frame_count]
    frame_indices = build_frame_indices(frame_count, frame_stride)

    target_yaw = np.asarray([yaw_from_xyzw(q) for q in target_quaternions], dtype=np.float32)
    xy_errors = [
        np.linalg.norm(result.positions[:, :2] - target_positions[:, :2], axis=1)
        for result in predictions
    ]
    yaw_errors_deg = [
        np.abs(
            np.rad2deg(
                wrap_angle_radians(
                    np.asarray([yaw_from_xyzw(q) for q in result.quaternions], dtype=np.float32) - target_yaw
                )
            )
        )
        for result in predictions
    ]
    times = np.asarray(trajectory.time[:frame_count], dtype=np.float32)

    local_force_point = np.asarray(trajectory.force_point_offset_local, dtype=np.float32)
    force_points_xy = np.asarray(
        [
            transform_local_point(target_positions[i], target_quaternions[i], local_force_point)[:2]
            for i in range(frame_count)
        ],
        dtype=np.float32,
    )
    forces_xy = np.zeros((frame_count, 2), dtype=np.float32)
    if len(trajectory.step_forces) > 0:
        used = min(frame_count, len(trajectory.step_forces))
        forces_xy[:used] = np.asarray(trajectory.step_forces[:used, :2], dtype=np.float32)
        if used < frame_count:
            forces_xy[used:] = forces_xy[used - 1]
    force_norm_flat = np.linalg.norm(forces_xy, axis=1)
    force_norm = force_norm_flat.reshape(-1, 1)
    force_active = force_norm_flat > 1.0e-6
    force_directions = np.divide(forces_xy, np.maximum(force_norm, 1.0e-8))
    force_vectors = force_directions * float(force_arrow_length)
    held_force_vectors = force_vectors.copy()
    held_force_points_xy = force_points_xy.copy()
    last_active_idx = -1
    for frame_idx in range(frame_count):
        if force_active[frame_idx]:
            last_active_idx = frame_idx
        elif last_active_idx >= 0:
            held_force_vectors[frame_idx] = force_vectors[last_active_idx]
            held_force_points_xy[frame_idx] = force_points_xy[last_active_idx]
    force_on_spans: list[tuple[float, float]] = []
    span_start: int | None = None
    for frame_idx, is_active in enumerate(force_active):
        if is_active and span_start is None:
            span_start = frame_idx
        elif not is_active and span_start is not None:
            force_on_spans.append((float(times[span_start]), float(times[frame_idx - 1])))
            span_start = None
    if span_start is not None:
        force_on_spans.append((float(times[span_start]), float(times[-1])))

    corner_cloud = []
    for i in range(frame_count):
        corner_cloud.append(block_corners_xy(target_positions[i], target_quaternions[i], half_extents))
        for result in predictions:
            corner_cloud.append(block_corners_xy(result.positions[i], result.quaternions[i], half_extents))
    all_xy = np.vstack(
        [
            target_positions[:, :2],
            *(result.positions[:, :2] for result in predictions),
            force_points_xy,
            np.vstack(corner_cloud),
        ]
    )
    xy_min = np.min(all_xy, axis=0)
    xy_max = np.max(all_xy, axis=0)
    center = 0.5 * (xy_min + xy_max)
    radius = 0.5 * max(float(np.max(xy_max - xy_min)), 0.08)
    pad = max(0.02, radius * 0.18)

    fig = plt.figure(figsize=(12.6, 7.2), dpi=dpi)
    fig.subplots_adjust(left=0.22, right=0.72)
    grid = fig.add_gridspec(2, 1, height_ratios=(3.2, 1.0), hspace=0.24)
    ax_xy = fig.add_subplot(grid[0, 0])
    ax_err = fig.add_subplot(grid[1, 0])

    target_label = "MuJoCo ground truth"
    ax_xy.plot(target_positions[:, 0], target_positions[:, 1], color="#222222", alpha=0.18, linewidth=2.0)
    target_trace, = ax_xy.plot([], [], color="#222222", linewidth=2.4, label=target_label)
    target_marker, = ax_xy.plot([], [], marker="o", color="#222222", markersize=5, linestyle="None")
    pred_traces = []
    pred_markers = []
    pred_patches = []
    for run_idx, result in enumerate(predictions):
        color = PALETTE[run_idx % len(PALETTE)]
        ax_xy.plot(result.positions[:, 0], result.positions[:, 1], color=color, alpha=0.14, linewidth=1.8)
        trace, = ax_xy.plot([], [], color=color, linewidth=2.1, linestyle="--", label=result.legend_label)
        marker, = ax_xy.plot([], [], marker="o", color=color, markersize=4.5, linestyle="None")
        patch = Polygon(
            block_corners_xy(result.positions[0], result.quaternions[0], half_extents),
            closed=True,
            facecolor=color,
            edgecolor=color,
            alpha=0.18,
            linewidth=1.6,
        )
        ax_xy.add_patch(patch)
        pred_traces.append(trace)
        pred_markers.append(marker)
        pred_patches.append(patch)
    force_quiver = ax_xy.quiver(
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="#0077bb",
        width=0.010,
        headwidth=5.0,
        headlength=6.5,
        headaxislength=5.5,
        minlength=0.0,
        zorder=12,
        label="applied force direction at contact point",
    )
    force_origin_marker, = ax_xy.plot(
        [],
        [],
        marker="o",
        color="#0077bb",
        markerfacecolor="#0077bb",
        markeredgecolor="white",
        markeredgewidth=1.2,
        markersize=6.5,
        linestyle="None",
        zorder=13,
        label="force application point",
    )
    target_patch = Polygon(
        block_corners_xy(target_positions[0], target_quaternions[0], half_extents),
        closed=True,
        facecolor="none",
        edgecolor="#222222",
        linewidth=2.2,
    )
    ax_xy.add_patch(target_patch)
    info_text = fig.text(
        0.025,
        0.82,
        "",
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#d9d9d9", "alpha": 0.88, "boxstyle": "round,pad=0.35"},
    )
    ax_xy.set_xlim(center[0] - radius - pad, center[0] + radius + pad)
    ax_xy.set_ylim(center[1] - radius - pad, center[1] + radius + pad)
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(alpha=0.22)
    ax_xy.set_xlabel("world x (m)")
    ax_xy.set_ylabel("world y (m)")
    title = f"MuJoCo vs Newton trajectory | {len(predictions)} checkpoint(s)"
    ax_xy.set_title(title, fontsize=11)
    ax_xy.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        fontsize=7,
    )

    err_lines = []
    err_dots = []
    for run_idx, (result, xy_error) in enumerate(zip(predictions, xy_errors, strict=True)):
        color = PALETTE[run_idx % len(PALETTE)]
        line, = ax_err.plot(times, xy_error, color=color, linewidth=1.6, label=f"{result.label} xy error")
        dot, = ax_err.plot([], [], marker="o", color=color, markersize=4.5, linestyle="None")
        err_lines.append(line)
        err_dots.append(dot)
    force_axis = ax_err.twinx()
    force_line, = force_axis.plot(
        times,
        force_norm_flat,
        color="#0077bb",
        linewidth=1.5,
        alpha=0.72,
        label="|force| (N)",
    )
    force_dot, = force_axis.plot([], [], marker="o", color="#0077bb", markersize=4.2, linestyle="None")
    for start_time, end_time in force_on_spans:
        ax_err.axvspan(start_time, end_time, color="#0077bb", alpha=0.08, linewidth=0.0)
    err_time = ax_err.axvline(times[0], color="#555555", linewidth=1.0, alpha=0.7)
    ax_err.set_xlim(float(times[0]), float(times[-1]) if len(times) > 1 else float(times[0] + 1.0e-3))
    max_xy_error = max(float(np.max(xy_error)) for xy_error in xy_errors)
    ax_err.set_ylim(0.0, max(max_xy_error * 1.15, 1.0e-4))
    ax_err.set_xlabel("time (s)")
    ax_err.set_ylabel("xy error (m)")
    ax_err.grid(alpha=0.22)
    force_axis.set_ylabel("|force| (N)", color="#0077bb")
    force_axis.tick_params(axis="y", labelcolor="#0077bb")
    force_axis.set_ylim(0.0, max(float(np.max(force_norm_flat)) * 1.2, 1.0))
    err_handles, err_labels = ax_err.get_legend_handles_labels()
    force_handles, force_labels = force_axis.get_legend_handles_labels()
    ax_err.legend(
        err_handles + force_handles,
        err_labels + force_labels,
        loc="upper left",
        bbox_to_anchor=(1.16, 1.0),
        borderaxespad=0.0,
        fontsize=7,
    )

    def update(frame_number: int):
        i = int(frame_indices[frame_number])
        start = max(0, i - max(int(trail_frames), 1))
        target_trace.set_data(target_positions[start : i + 1, 0], target_positions[start : i + 1, 1])
        target_marker.set_data([target_positions[i, 0]], [target_positions[i, 1]])
        target_patch.set_xy(block_corners_xy(target_positions[i], target_quaternions[i], half_extents))
        for result, trace, marker, patch in zip(predictions, pred_traces, pred_markers, pred_patches, strict=True):
            trace.set_data(result.positions[start : i + 1, 0], result.positions[start : i + 1, 1])
            marker.set_data([result.positions[i, 0]], [result.positions[i, 1]])
            patch.set_xy(block_corners_xy(result.positions[i], result.quaternions[i], half_extents))
        if force_active[i]:
            force_color = "#0077bb"
            force_alpha = 1.0
            force_point = force_points_xy[i : i + 1]
            force_vector = force_vectors[i : i + 1]
        else:
            force_color = "#999999"
            force_alpha = 0.36 if np.any(held_force_vectors[i]) else 0.0
            force_point = held_force_points_xy[i : i + 1]
            force_vector = held_force_vectors[i : i + 1]
        force_quiver.set_offsets(force_point)
        force_quiver.set_UVC(force_vector[:, 0], force_vector[:, 1])
        force_quiver.set_color(force_color)
        force_quiver.set_alpha(force_alpha)
        force_origin_marker.set_data([force_point[0, 0]], [force_point[0, 1]])
        force_origin_marker.set_color(force_color)
        force_origin_marker.set_alpha(max(force_alpha, 0.25))
        for dot, xy_error in zip(err_dots, xy_errors, strict=True):
            dot.set_data([times[i]], [xy_error[i]])
        force_dot.set_data([times[i]], [force_norm_flat[i]])
        err_time.set_xdata([times[i], times[i]])
        best_idx = int(np.argmin([xy_error[i] for xy_error in xy_errors]))
        force_state = "on" if force_active[i] else "off"
        info_text.set_text(
            f"frame {i}/{frame_count - 1}   t={times[i]:.3f}s\n"
            f"force {force_state}: |F|={force_norm_flat[i]:.2f} N   "
            f"Fx={forces_xy[i, 0]:.2f} Fy={forces_xy[i, 1]:.2f}\n"
            f"best xy={predictions[best_idx].label} ({xy_errors[best_idx][i] * 1000.0:.2f} mm)\n"
            f"yaw error={yaw_errors_deg[best_idx][i]:.2f} deg   loss={predictions[best_idx].mean_loss:.6g}"
        )
        return tuple(
            [target_trace, target_marker, target_patch, force_quiver, force_origin_marker, force_dot, err_time, info_text]
            + pred_traces
            + pred_markers
            + pred_patches
            + err_dots
        )

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000.0 / max(int(fps), 1),
        blit=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        writer = animation.PillowWriter(fps=max(int(fps), 1))
    elif suffix in {".mp4", ".m4v", ".mov"}:
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError("Matplotlib cannot find ffmpeg. Use an .gif output path or install ffmpeg for MP4.")
        writer = animation.FFMpegWriter(fps=max(int(fps), 1), bitrate=int(bitrate))
    else:
        raise ValueError(f"Unsupported output suffix {output_path.suffix!r}; use .mp4 or .gif")

    anim.save(str(output_path), writer=writer, dpi=dpi)
    plt.close(fig)


def run_prediction(
    *,
    args: argparse.Namespace,
    run_spec: RunSpec,
    trajectory,
) -> tuple[PredictionResult, Path]:
    run_args = copy.copy(args)
    checkpoint = load_checkpoint_parameters(run_spec.checkpoint_path)
    trajectory_npz_path = resolve_trajectory_npz_path(run_args, checkpoint)
    run_args.trajectory_npz = trajectory_npz_path

    log_message(f"loading checkpoint for {run_spec.label} from {checkpoint.path.resolve()}")

    parameter_point_cloud, reference_point_cloud = resolve_reference_point_cloud(run_args, checkpoint, run_spec)
    if reference_point_cloud is not None:
        inferred_half_extents, inferred_spacing = infer_box_half_extents_and_spacing(
            reference_point_cloud.local_surface_points
        )
        run_args.box_half_extents = inferred_half_extents.tolist()
        run_args.surface_point_spacing = inferred_spacing
        run_args.point_friction = infer_base_point_friction(reference_point_cloud, fallback=float(run_args.point_friction))
        run_args.avoid_zero_surface_point_x = not reference_point_cloud_has_center_x(
            reference_point_cloud.local_surface_points
        )
        log_message(
            "inferred scene sampling from reference point cloud "
            f"box_half_extents={np.asarray(run_args.box_half_extents, dtype=np.float32).tolist()} "
            f"surface_point_spacing={float(run_args.surface_point_spacing):.9g} "
            f"base_point_friction={float(run_args.point_friction):.9g} "
            f"avoid_zero_surface_point_x={bool(run_args.avoid_zero_surface_point_x)}"
        )

    eval_args = make_eval_args(run_args, trajectory)
    log_message(f"building replay scene on device={run_args.device if run_args.device is not None else 'auto'}")
    diff_scene = build_diff_scene(eval_args)
    initial_body_q = diff_scene.states[0].body_q.numpy().copy()
    initial_body_qd = diff_scene.states[0].body_qd.numpy().copy()
    active_indices, active_params, parameter_source = resolve_active_parameters(
        args=run_args,
        checkpoint=checkpoint,
        diff_scene=diff_scene,
        parameter_point_cloud=parameter_point_cloud,
        reference_point_cloud=reference_point_cloud,
    )
    if active_indices.shape != active_params.shape:
        raise ValueError(f"Active parameter shape mismatch: indices={active_indices.shape} params={active_params.shape}")
    if len(active_indices) == 0:
        raise ValueError("No active contact-point parameters were resolved for replay")
    if np.min(active_indices) < 0 or np.max(active_indices) >= len(diff_scene.local_surface_points_np):
        raise ValueError(
            "Resolved active indices do not fit this scene. Pass --reference-point-cloud or matching sampling settings."
        )
    parameter_summary = summarize_friction_parameters(
        checkpoint_path=checkpoint.path,
        checkpoint_param_set=run_args.checkpoint_param_set,
        active_indices=active_indices,
        active_params=active_params,
        local_surface_points=diff_scene.local_surface_points_np,
    )

    log_message(
        f"running Newton replay label={run_spec.label} active_points={len(active_indices)} "
        f"param_source={parameter_source} params={parameter_summary}"
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
        args=eval_args,
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
    predicted_positions, predicted_quaternions = extract_box_pose_frames(body_q_frames, diff_scene.box_body)
    final_xy_error = float(np.linalg.norm(predicted_positions[-1, :2] - trajectory.positions[-1, :2]))

    log_message(
        f"replay complete label={run_spec.label} | "
        f"mean_loss={final_loss:.6g} position_loss={final_position_loss:.6g} "
        f"orientation_loss={final_orientation_loss:.6g} "
        f"linear_velocity_loss={final_linear_velocity_loss:.6g} "
        f"angular_velocity_loss={final_angular_velocity_loss:.6g} "
        f"final_xy_error={final_xy_error:.6g}"
    )
    return (
        PredictionResult(
            label=run_spec.label,
            checkpoint_path=checkpoint.path,
            parameter_source=parameter_source,
            parameter_summary=parameter_summary,
            positions=predicted_positions,
            quaternions=predicted_quaternions,
            mean_loss=float(final_loss),
            position_loss=float(final_position_loss),
            orientation_loss=float(final_orientation_loss),
            linear_velocity_loss=float(final_linear_velocity_loss),
            angular_velocity_loss=float(final_angular_velocity_loss),
            final_xy_error=final_xy_error,
            half_extents=np.asarray(run_args.box_half_extents, dtype=np.float32),
        ),
        trajectory_npz_path,
    )


def main() -> None:
    args = parse_args()
    first_checkpoint = load_checkpoint_parameters(args.run_specs[0].checkpoint_path)
    trajectory_npz_path = resolve_trajectory_npz_path(args, first_checkpoint)
    args.trajectory_npz = trajectory_npz_path
    replay_max_steps = first_checkpoint.max_steps if args.max_steps is None else args.max_steps
    output_path = args.output if args.output is not None else default_output_path(args, first_checkpoint)

    log_message(f"loading trajectory from {trajectory_npz_path.resolve()}")
    trajectory = select_trajectory(trajectory_npz_path, replay_max_steps, int(args.trajectory_index))
    wp.init()

    predictions: list[PredictionResult] = []
    for run_spec in args.run_specs:
        prediction, resolved_trajectory_path = run_prediction(
            args=args,
            run_spec=run_spec,
            trajectory=trajectory,
        )
        if resolved_trajectory_path.resolve() != trajectory_npz_path.resolve():
            log_message(
                f"warning: {run_spec.label} checkpoint metadata points to {resolved_trajectory_path.resolve()}, "
                f"but comparison uses {trajectory_npz_path.resolve()}"
            )
        predictions.append(prediction)

    half_extents = np.asarray(predictions[0].half_extents, dtype=np.float32)
    for prediction in predictions[1:]:
        if not np.allclose(prediction.half_extents, half_extents, atol=1.0e-6):
            log_message(
                f"warning: {prediction.label} inferred box_half_extents={prediction.half_extents.tolist()} "
                f"differs from first run {half_extents.tolist()}; rendering uses first run extents"
            )

    log_message(f"rendering video to {output_path.resolve()}")
    render_video(
        output_path=output_path,
        trajectory=trajectory,
        target_positions=trajectory.positions,
        target_quaternions=trajectory.quaternions_xyzw,
        predictions=predictions,
        half_extents=half_extents,
        fps=args.fps,
        dpi=args.dpi,
        frame_stride=args.frame_stride,
        trail_frames=args.trail_frames,
        force_arrow_length=args.force_arrow_length,
        bitrate=args.bitrate,
    )
    log_message(f"video_written_to={output_path.resolve()}")


if __name__ == "__main__":
    main()
