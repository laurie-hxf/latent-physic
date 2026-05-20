from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from project_paths import DEFAULT_OUTPUT_DIR, REPO_ROOT


DEFAULT_TRAJECTORY_NPZ_PATH = REPO_ROOT / "mujoco" / "outputs" / "block_force_dataset_2000.npz"
DEFAULT_CONTACT_FRICTION_RESULTS_PATH = DEFAULT_OUTPUT_DIR / "mujoco_contact_point_friction_fit.npz"
DEFAULT_CONTACT_FRICTION_CHECKPOINT_PATH = DEFAULT_OUTPUT_DIR / "mujoco_contact_point_friction_fit_checkpoint.npz"
DEFAULT_CONTACT_FRICTION_SCENE_USD_PATH = DEFAULT_OUTPUT_DIR / "mujoco_contact_point_friction_fit.usda"
DEFAULT_CONTACT_FRICTION_POINT_CLOUD_PATH = DEFAULT_OUTPUT_DIR / "mujoco_contact_point_friction_point_cloud.ply"
DEFAULT_TRAIN_BATCH_SIZE = 64
DEFAULT_TRAJECTORY_PROGRESS_EVERY = 256


def _resolve_point_cloud_color_range(
    values: np.ndarray,
    color_min: float | None,
    color_max: float | None,
) -> tuple[float, float]:
    finite_values = values[np.isfinite(values)]
    if color_min is not None and color_max is not None:
        vmin = float(color_min)
        vmax = float(color_max)
    elif len(finite_values) == 0:
        vmin = 0.0 if color_min is None else float(color_min)
        vmax = 1.0 if color_max is None else float(color_max)
    elif color_min is None and color_max is None:
        vmin = float(np.min(finite_values))
        vmax = float(np.max(finite_values))
    elif color_min is None:
        vmax = float(color_max)
        vmin = float(np.min(finite_values))
    else:
        vmin = float(color_min)
        vmax = float(np.max(finite_values))

    if vmax <= vmin:
        pad = max(abs(vmin) * 1.0e-3, 1.0e-6)
        vmin -= pad
        vmax += pad
    return vmin, vmax


def _friction_to_point_cloud_colors(
    values: np.ndarray,
    color_min: float | None = None,
    color_max: float | None = None,
) -> tuple[np.ndarray, float, float]:
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return np.empty((0, 3), dtype=np.uint8), 0.0, 1.0

    vmin, vmax = _resolve_point_cloud_color_range(values, color_min, color_max)
    denom = max(vmax - vmin, 1.0e-8)
    normalized = np.clip((values - vmin) / denom, 0.0, 1.0)
    normalized = np.where(np.isfinite(normalized), normalized, 0.0)

    low_color = np.asarray([255.0, 245.0, 204.0], dtype=np.float32)
    high_color = np.asarray([128.0, 0.0, 38.0], dtype=np.float32)
    colors = low_color[None, :] * (1.0 - normalized[:, None]) + high_color[None, :] * normalized[:, None]
    return np.rint(colors).astype(np.uint8), vmin, vmax


def save_contact_friction_point_cloud(
    *,
    local_surface_points: np.ndarray,
    point_friction: np.ndarray,
    output_path: Path,
    active_indices: np.ndarray | None = None,
    color_min: float | None = None,
    color_max: float | None = None,
) -> None:
    points = np.asarray(local_surface_points, dtype=np.float32)
    friction = np.asarray(point_friction, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"local_surface_points must have shape (N, 3), got {points.shape}")
    if friction.shape != (len(points),):
        raise ValueError(f"point_friction must have shape ({len(points)},), got {friction.shape}")

    active_mask = np.zeros(len(points), dtype=np.int32)
    if active_indices is not None:
        active_mask[np.asarray(active_indices, dtype=np.int32)] = 1

    colors, color_min_used, color_max_used = _friction_to_point_cloud_colors(
        friction,
        color_min=color_min,
        color_max=color_max,
    )
    finite_friction = friction[np.isfinite(friction)]
    friction_min = float(np.min(finite_friction)) if len(finite_friction) > 0 else float("nan")
    friction_max = float(np.max(finite_friction)) if len(finite_friction) > 0 else float("nan")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment generated_by fit_mujoco_contact_point_friction.py\n")
        f.write("comment coordinates local_box_surface_points\n")
        f.write("comment color_map low_friction_rgb 255 245 204 high_friction_rgb 128 0 38\n")
        f.write(f"comment friction_min {friction_min:.9g}\n")
        f.write(f"comment friction_max {friction_max:.9g}\n")
        f.write(f"comment color_friction_min {color_min_used:.9g}\n")
        f.write(f"comment color_friction_max {color_max_used:.9g}\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property float friction\n")
        f.write("property int active_contact\n")
        f.write("property int point_index\n")
        f.write("end_header\n")
        rows = zip(points, colors, friction, active_mask, strict=True)
        for point_idx, (point, color, mu, active) in enumerate(rows):
            f.write(
                f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} "
                f"{float(mu):.9g} {int(active)} {point_idx}\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--trajectory-npz", type=Path, default=DEFAULT_TRAJECTORY_NPZ_PATH)
    parser.add_argument("--max-trajectories", type=int, default=None, help="Use only the first N trajectories when the input NPZ is a dataset.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_TRAIN_BATCH_SIZE,
        help="Trajectories per training iteration. Use <=0 to consume the full dataset each iteration.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Trajectories per evaluation batch. Defaults to --batch-size.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed used for trajectory minibatch sampling.")
    parser.add_argument(
        "--trajectory-progress-every",
        type=int,
        default=DEFAULT_TRAJECTORY_PROGRESS_EVERY,
        help="Print trajectory progress every N trajectories during long train/eval passes. Use <=0 to disable.",
    )
    parser.add_argument("--results-path", type=Path, default=DEFAULT_CONTACT_FRICTION_RESULTS_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CONTACT_FRICTION_CHECKPOINT_PATH)
    parser.add_argument("--resume-checkpoint", type=Path, default=None, help="Resume optimizer state from a checkpoint NPZ.")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Save checkpoint every N successful iterations. Use <=0 to disable periodic checkpointing.",
    )
    parser.add_argument(
        "--checkpoint-point-cloud-dir",
        type=Path,
        default=None,
        help=(
            "Directory for per-checkpoint friction point clouds. Defaults to "
            "<checkpoint-path parent>/<checkpoint stem>_point_clouds."
        ),
    )
    parser.add_argument("--scene-usd-path", type=Path, default=DEFAULT_CONTACT_FRICTION_SCENE_USD_PATH)
    parser.add_argument("--point-cloud-path", type=Path, default=DEFAULT_CONTACT_FRICTION_POINT_CLOUD_PATH)
    parser.add_argument(
        "--point-cloud-color-min",
        type=float,
        default=None,
        help="Friction value mapped to the low end of the point-cloud color ramp. Defaults to --point-friction - 0.005.",
    )
    parser.add_argument(
        "--point-cloud-color-max",
        type=float,
        default=None,
        help="Friction value mapped to the high end of the point-cloud color ramp. Defaults to --point-friction + 0.005.",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default="newton-contact-point-friction-fit")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default="mujoco-contact-friction")
    parser.add_argument("--wandb-mode", type=str, default="online")
    parser.add_argument("--wandb-dir", type=Path, default=None)
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Use only the first N simulation steps from the MuJoCo trajectory.")
    parser.add_argument("--opt-iters", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=2.0e-2)
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=None,
        help="Clip each training batch's contact-point friction gradient to this global L2 norm. Use <=0 to disable.",
    )
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1.0e-8)
    parser.add_argument("--min-point-friction", type=float, default=0.0)
    parser.add_argument("--max-point-friction", type=float, default=2.0)
    parser.add_argument(
        "--friction-parameterization",
        choices=("point", "left-right", "global"),
        default="point",
        help=(
            "Friction parameters to optimize. 'point' keeps one parameter per active surface point; "
            "'left-right' optimizes only two x-split parameters and broadcasts them to all active points; "
            "'global' optimizes one parameter shared by all active points."
        ),
    )
    parser.add_argument("--position-loss-weight", type=float, default=1.0)
    parser.add_argument("--orientation-loss-weight", type=float, default=0.0)
    parser.add_argument("--linear-velocity-loss-weight", type=float, default=0.0)
    parser.add_argument("--angular-velocity-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--piecewise-regularization-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for x-split piecewise-constant friction regularization. "
            "The unweighted term is var(mu[x<0]) + var(mu[x>0]) over each batch's active points."
        ),
    )
    parser.add_argument(
        "--point-position-loss-reduction",
        choices=("sum", "mean"),
        default="mean",
        help=(
            "Reduce surface-point squared-distance loss over points. "
            "'sum' matches per-point loss summation; 'mean' divides by the surface point count."
        ),
    )
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--steps", type=int, default=0, help="Filled automatically from the trajectory after loading.")
    parser.add_argument("--dt", type=float, default=0.0, help="Filled automatically from the trajectory after loading.")
    parser.add_argument("--solver-iterations", type=int, default=10)
    parser.add_argument("--box-mass", type=float, default=1.0)
    parser.add_argument("--floor-half-extents", type=float, nargs=3, default=(2.0, 2.0, 0.05))
    parser.add_argument("--box-half-extents", type=float, nargs=3, default=(0.1, 0.05, 0.025))
    parser.add_argument("--box-start-pos", type=float, nargs=3, default=(0.58, 0.0, 0.025))
    parser.add_argument("--surface-point-spacing", type=float, default=0.02)
    parser.add_argument("--friction-contact-threshold", type=float, default=0.002)
    parser.add_argument("--contact-mask-threshold", type=float, default=0.002)
    parser.add_argument("--point-friction", type=float, default=0.1)
    parser.add_argument("--contact-friction", type=float, default=0.0)
    parser.add_argument("--contact-stiffness", type=float, default=2.0e4)
    parser.add_argument("--contact-damping", type=float, default=50.0)
    parser.add_argument("--contact-margin", type=float, default=1.0e-3)
    parser.add_argument("--friction-regularization", type=float, default=1.0e-3)
    parser.add_argument("--initial-force", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--initial-torque", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--force-magnitude", type=float, default=None)
    parser.add_argument("--force-direction", type=float, nargs=3, default=None)
    parser.add_argument("--force-point", type=float, nargs=3, default=None)
    parser.add_argument("--force-point-local", type=float, nargs=3, default=None)
    parser.add_argument("--force-steps", type=int, default=0)
    parser.add_argument("--loss-target-position", type=float, nargs=3, default=None)
    return parser.parse_args()
