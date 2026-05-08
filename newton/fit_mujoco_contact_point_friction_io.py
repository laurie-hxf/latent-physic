from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from project_paths import DEFAULT_OUTPUT_DIR, REPO_ROOT


DEFAULT_TRAJECTORY_NPZ_PATH = REPO_ROOT / "mujoco" / "outputs" / "block_force_dataset_2000.npz"
DEFAULT_CONTACT_FRICTION_RESULTS_PATH = DEFAULT_OUTPUT_DIR / "mujoco_contact_point_friction_fit.npz"
DEFAULT_CONTACT_FRICTION_CHECKPOINT_PATH = DEFAULT_OUTPUT_DIR / "mujoco_contact_point_friction_fit_checkpoint.npz"
DEFAULT_CONTACT_FRICTION_SCENE_USD_PATH = DEFAULT_OUTPUT_DIR / "mujoco_contact_point_friction_fit.usda"
DEFAULT_CONTACT_FRICTION_HEATMAP_PATH = DEFAULT_OUTPUT_DIR / "mujoco_contact_point_friction_heatmap.png"
DEFAULT_TRAIN_BATCH_SIZE = 64
DEFAULT_TRAJECTORY_PROGRESS_EVERY = 256


def save_contact_friction_heatmap(
    *,
    local_surface_points: np.ndarray,
    active_indices: np.ndarray,
    active_point_friction: np.ndarray,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    active_points = np.asarray(local_surface_points[active_indices], dtype=np.float32)
    if len(active_points) == 0:
        raise ValueError("No active contact points available for heatmap export.")

    z_values = active_points[:, 2]
    bottom_z = float(np.min(z_values))
    bottom_mask = np.isclose(z_values, bottom_z, atol=1.0e-4)
    contact_face_points = active_points[bottom_mask]
    contact_face_friction = np.asarray(active_point_friction[bottom_mask], dtype=np.float32)

    if len(contact_face_points) == 0:
        contact_face_points = active_points
        contact_face_friction = np.asarray(active_point_friction, dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    scatter = ax.scatter(
        contact_face_points[:, 0],
        contact_face_points[:, 1],
        c=contact_face_friction,
        cmap="YlOrRd",
        s=180,
        marker="s",
        edgecolors="black",
        linewidths=0.4,
    )
    ax.set_title("Contact Surface Friction Heatmap")
    ax.set_xlabel("Local X")
    ax.set_ylabel("Local Y")
    ax.set_aspect("equal", adjustable="box")

    x_pad = max(float(np.ptp(contact_face_points[:, 0])) * 0.08, 1.0e-3)
    y_pad = max(float(np.ptp(contact_face_points[:, 1])) * 0.08, 1.0e-3)
    ax.set_xlim(float(contact_face_points[:, 0].min() - x_pad), float(contact_face_points[:, 0].max() + x_pad))
    ax.set_ylim(float(contact_face_points[:, 1].min() - y_pad), float(contact_face_points[:, 1].max() + y_pad))

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Friction Coefficient")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


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
        "--checkpoint-heatmap-dir",
        type=Path,
        default=None,
        help=(
            "Directory for per-checkpoint heatmaps. Defaults to "
            "<checkpoint-path parent>/<checkpoint stem>_heatmaps."
        ),
    )
    parser.add_argument("--scene-usd-path", type=Path, default=DEFAULT_CONTACT_FRICTION_SCENE_USD_PATH)
    parser.add_argument("--heatmap-path", type=Path, default=DEFAULT_CONTACT_FRICTION_HEATMAP_PATH)
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
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1.0e-8)
    parser.add_argument("--min-point-friction", type=float, default=0.0)
    parser.add_argument("--max-point-friction", type=float, default=2.0)
    parser.add_argument("--position-loss-weight", type=float, default=1.0)
    parser.add_argument("--orientation-loss-weight", type=float, default=0.0)
    parser.add_argument("--linear-velocity-loss-weight", type=float, default=0.0)
    parser.add_argument("--angular-velocity-loss-weight", type=float, default=0.0)
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
