from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
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
from fit_mujoco_contact_point_friction_runtime import (  # noqa: E402
    build_batched_optimization_buffers,
    clear_batched_optimization_grads,
    evaluate_collection_loss_in_batches,
    forward_rollout_with_batched_trajectory_loss,
    reset_scene_states,
    resolve_batch_size,
)
from mujoco_contact_friction_fit_utils import load_mujoco_trajectories  # noqa: E402
from newton_surface_points_diff_demo import build_diff_scene  # noqa: E402


DEFAULT_DATASET = ROOT / "mujoco" / "outputs" / "block_force_dataset_fixed_init_20" / "block_force_dataset_fixed_init_20.npz"
DEFAULT_OUTPUT = ROOT / "report_assets" / "topdown_trajectory_overlays_fixed20.png"
OUTPUTS_ROOT = ROOT / "outputs"


def experiment_checkpoint(experiment_name: str, checkpoint_stem: str | None = None) -> Path:
    stem = experiment_name if checkpoint_stem is None else checkpoint_stem
    return OUTPUTS_ROOT / experiment_name / f"{stem}.npz"


@dataclass(frozen=True)
class MethodSpec:
    name: str
    checkpoint: Path
    color: str
    linestyle: str
    linewidth: float = 1.8
    stage: str = ""
    notes: str = ""


@dataclass(frozen=True)
class CheckpointParams:
    active_indices: np.ndarray
    active_params: np.ndarray
    optimizer_params: np.ndarray
    parameterization: str
    iteration: int | None
    train_best_loss: float | None
    train_max_steps: int | None
    train_dataset: str


def parse_optional_max_steps(value: str | None) -> int | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"none", "all", "full", "-1"}:
        return None
    parsed = int(normalized)
    if parsed < 1:
        raise argparse.ArgumentTypeError("--max-steps must be positive, or one of: none, all, full, -1")
    return parsed


DEFAULT_METHODS = [
    MethodSpec(
        name="Pure point",
        checkpoint=ROOT / "debug-stash" / "outputs" / "friction_fit_fixed_init_fisher_stiffness_1e4_4.npz",
        color="#8c564b",
        linestyle=(0, (1.2, 1.2)),
        linewidth=1.5,
    ),
    MethodSpec(
        name="Point+reg v2",
        checkpoint=experiment_checkpoint("fixed_init_0.4_stiffness_1e5_regularization_3000_v2"),
        color="#ff7f0e",
        linestyle=(0, (3.0, 1.6)),
    ),
    MethodSpec(
        name="Global",
        checkpoint=experiment_checkpoint("fixed_init_0.4_stiffness_1e5_regularization_3000_global"),
        color="#1f77b4",
        linestyle=(0, (5.0, 2.0)),
    ),
    MethodSpec(
        name="Left-right",
        checkpoint=experiment_checkpoint("fixed_init_0.35_stiffness_1e5_regularization_3000_left_right"),
        color="#2ca02c",
        linestyle="-",
        linewidth=2.0,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument(
        "--method-source",
        choices=("default", "curated", "auto", "all"),
        default="default",
        help=(
            "default uses the original compact method set; curated uses the historical fixed20 RUNS; "
            "auto scans checkpoint roots; all combines curated RUNS with scanned checkpoints."
        ),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        action="append",
        default=None,
        help="Root directory to scan for checkpoint .npz files when --method-source is auto or all.",
    )
    parser.add_argument("--max-steps", type=parse_optional_max_steps, default=300)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--eval-batch-size", type=int, default=20)
    parser.add_argument("--surface-point-spacing", type=float, default=0.01)
    parser.add_argument("--contact-stiffness", type=float, default=1.0e5)
    parser.add_argument("--contact-damping", type=float, default=50.0)
    parser.add_argument("--friction-contact-threshold", type=float, default=0.002)
    parser.add_argument("--contact-mask-threshold", type=float, default=0.002)
    parser.add_argument("--trajectory-indices", type=int, nargs="*", default=None)
    parser.add_argument(
        "--all-trajectories",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot every trajectory in the dataset unless --trajectory-indices is provided.",
    )
    parser.add_argument("--include-pure-point", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def make_eval_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        trajectory_npz=args.dataset,
        max_steps=args.max_steps,
        max_trajectories=None,
        batch_size=max(int(args.eval_batch_size), 1),
        eval_batch_size=args.eval_batch_size,
        trajectory_progress_every=0,
        device=args.device,
        steps=0,
        dt=0.0,
        batch_capacity=max(int(args.eval_batch_size), 1),
        solver_iterations=10,
        box_mass=1.0,
        floor_half_extents=(2.0, 2.0, 0.05),
        box_half_extents=(0.1, 0.05, 0.025),
        box_start_pos=(0.58, 0.0, 0.025),
        surface_point_spacing=args.surface_point_spacing,
        friction_contact_threshold=args.friction_contact_threshold,
        contact_mask_threshold=args.contact_mask_threshold,
        point_friction=0.1,
        contact_friction=0.0,
        contact_stiffness=args.contact_stiffness,
        contact_damping=args.contact_damping,
        contact_margin=1.0e-3,
        friction_regularization=1.0e-3,
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
    )


def _scalar(data: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in data.files:
        return default
    value = np.asarray(data[key])
    if value.shape == ():
        return value.item()
    return value.tolist()


def checkpoint_has_active_params(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=True) as data:
            return {"active_indices", "best_active_params"}.issubset(set(data.files))
    except Exception:
        return False


def load_checkpoint_params(path: Path) -> CheckpointParams:
    with np.load(path, allow_pickle=True) as data:
        active_indices = np.asarray(data["active_indices"], dtype=np.int32)
        active_params = np.asarray(data["best_active_params"], dtype=np.float32)
        optimizer_params = (
            np.asarray(data["best_optimizer_params"], dtype=np.float32)
            if "best_optimizer_params" in data.files
            else active_params
        )
        parameterization = str(_scalar(data, "friction_parameterization", "point"))
        iteration_value = _scalar(data, "iteration", None)
        best_loss_value = _scalar(data, "best_loss", None)
        max_steps_value = _scalar(data, "max_steps", None)
        train_dataset = str(_scalar(data, "trajectory_npz_path", ""))
    if active_indices.shape[0] != active_params.shape[0]:
        raise ValueError(f"{path} active_indices and best_active_params length mismatch")
    return CheckpointParams(
        active_indices=active_indices,
        active_params=active_params,
        optimizer_params=optimizer_params,
        parameterization=parameterization,
        iteration=None if iteration_value is None else int(iteration_value),
        train_best_loss=None if best_loss_value is None else float(best_loss_value),
        train_max_steps=None if max_steps_value is None else int(max_steps_value),
        train_dataset=train_dataset,
    )


def style_for_index(index: int) -> tuple[str, str]:
    color_maps = ["tab20", "tab20b", "tab20c"]
    color_map = plt.get_cmap(color_maps[(index // 20) % len(color_maps)])
    color = color_map(index % 20)
    linestyles = [
        "-",
        (0, (4.0, 1.7)),
        (0, (1.2, 1.2)),
        (0, (6.0, 2.0, 1.2, 2.0)),
        (0, (2.4, 1.2)),
    ]
    return color, linestyles[index % len(linestyles)]


def unique_method_name(base_name: str, used_names: set[str]) -> str:
    name = base_name
    suffix = 2
    while name in used_names:
        name = f"{base_name}_{suffix}"
        suffix += 1
    used_names.add(name)
    return name


def load_curated_methods(start_index: int = 0) -> list[MethodSpec]:
    try:
        from evaluate_friction_checkpoints_fixed20 import RUNS, resolve_run_path  # noqa: E402
    except Exception as exc:
        print(f"warning: could not import curated checkpoint RUNS: {exc}", flush=True)
        return []

    methods: list[MethodSpec] = []
    used_names: set[str] = set()
    for idx, run in enumerate(RUNS):
        checkpoint = resolve_run_path(run)
        color, linestyle = style_for_index(start_index + idx)
        methods.append(
            MethodSpec(
                name=unique_method_name(str(run["name"]), used_names),
                checkpoint=checkpoint,
                color=color,
                linestyle=linestyle,
                linewidth=1.55,
                stage=str(run.get("stage", "")),
                notes=str(run.get("notes", "")),
            )
        )
    return methods


def discover_checkpoint_methods(
    *,
    roots: list[Path],
    existing_paths: set[Path],
    start_index: int,
) -> list[MethodSpec]:
    methods: list[MethodSpec] = []
    used_names: set[str] = set()
    for root in roots:
        if not root.exists():
            print(f"warning: checkpoint root does not exist: {root}", flush=True)
            continue
        for path in sorted(root.rglob("*.npz")):
            resolved = path.resolve()
            if resolved in existing_paths:
                continue
            if not checkpoint_has_active_params(path):
                continue
            parent = path.parent.name
            base_name = parent if parent == path.stem or parent not in {"outputs", "debug-stash"} else path.stem
            if str(root).endswith("debug-stash/outputs"):
                base_name = f"debug_{path.stem}"
            color, linestyle = style_for_index(start_index + len(methods))
            methods.append(
                MethodSpec(
                    name=unique_method_name(base_name, used_names),
                    checkpoint=path,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.35,
                    stage="Auto-discovered",
                )
            )
    return methods


def select_methods(args: argparse.Namespace) -> list[MethodSpec]:
    if args.method_source == "default":
        return DEFAULT_METHODS if args.include_pure_point else DEFAULT_METHODS[1:]

    methods: list[MethodSpec] = []
    if args.method_source in {"curated", "all"}:
        methods.extend(load_curated_methods())

    if args.method_source in {"auto", "all"}:
        roots = args.checkpoint_root
        if roots is None:
            roots = [OUTPUTS_ROOT, ROOT / "debug-stash" / "outputs"]
        existing_paths = {method.checkpoint.resolve() for method in methods}
        methods.extend(
            discover_checkpoint_methods(
                roots=roots,
                existing_paths=existing_paths,
                start_index=len(methods),
            )
        )

    if not args.include_pure_point:
        methods = [
            method for method in methods
            if "point_random" not in method.name and "debug" not in method.name and "point_only" not in method.name
        ]
    return methods


def select_representative_indices(dataset_path: Path, requested: list[int] | None, all_trajectories: bool) -> list[int]:
    if requested:
        return list(dict.fromkeys(int(i) for i in requested))

    if all_trajectories:
        with np.load(dataset_path, allow_pickle=True) as data:
            return list(range(int(np.asarray(data["trajectories"]).shape[0])))

    with np.load(dataset_path, allow_pickle=True) as data:
        point_offset = np.asarray(data["point_offset_local"], dtype=np.float32)
        direction = np.asarray(data["direction_unit"], dtype=np.float32)
        displacement = np.asarray(data["max_xy_displacement"], dtype=np.float32)
        rotation = np.asarray(data["max_rotation_angle"], dtype=np.float32)

    point_x = point_offset[:, 0]
    xy_direction = direction[:, :2]
    diagonal_score = np.minimum(np.abs(xy_direction[:, 0]), np.abs(xy_direction[:, 1]))
    candidates = [
        int(np.argmin(point_x)),
        int(np.argmax(point_x)),
        int(np.argmin(np.abs(point_x))),
        int(np.argmax(displacement)),
        int(np.argmax(rotation)),
        int(np.argmax(diagonal_score)),
    ]

    unique: list[int] = []
    for idx in candidates:
        if idx not in unique:
            unique.append(idx)
    for idx in np.argsort(-displacement):
        idx = int(idx)
        if idx not in unique:
            unique.append(idx)
        if len(unique) == 6:
            break
    return unique[:6]


def transform_positions_from_body_q_frames(body_q_frames: list[np.ndarray], box_body: int) -> np.ndarray:
    positions = []
    for frame in body_q_frames:
        transform_value = np.asarray(frame[box_body])
        flat = transform_value.reshape(-1)
        if flat.size < 3:
            raise ValueError(f"Unexpected body_q transform shape: {transform_value.shape}")
        positions.append(flat[:3].astype(np.float32))
    return np.asarray(positions, dtype=np.float32)


def transform_batched_positions_from_states(
    body_q_frames: list[np.ndarray],
    box_body_ids: np.ndarray,
    trajectories,
) -> list[np.ndarray]:
    batched_positions: list[np.ndarray] = []
    for batch_idx, trajectory in enumerate(trajectories):
        body_id = int(box_body_ids[batch_idx])
        positions = []
        for frame in body_q_frames[: trajectory.num_frames]:
            transform_value = np.asarray(frame[body_id])
            flat = transform_value.reshape(-1)
            if flat.size < 3:
                raise ValueError(f"Unexpected body_q transform shape: {transform_value.shape}")
            positions.append(flat[:3].astype(np.float32))
        batched_positions.append(np.asarray(positions, dtype=np.float32))
    return batched_positions


def rollout_positions(
    *,
    diff_scene,
    trajectory,
    eval_args: argparse.Namespace,
    active_indices: np.ndarray,
    active_params: np.ndarray,
    initial_body_q: np.ndarray,
    initial_body_qd: np.ndarray,
) -> tuple[np.ndarray, float]:
    loss, _, _, _, _, body_q_frames = evaluate_collection_loss_in_batches(
        diff_scene=diff_scene,
        trajectories=[trajectory],
        args=eval_args,
        active_indices=active_indices,
        active_params=active_params,
        initial_body_q=initial_body_q,
        initial_body_qd=initial_body_qd,
        eval_batch_size=1,
        trajectory_progress_every=0,
        scatter_active_point_friction_kernel=scatter_active_point_friction_kernel,
        compute_batched_contact_weighted_masses_kernel=compute_batched_contact_weighted_masses_kernel,
        apply_batched_external_and_surface_point_forces_trajectory_kernel=apply_batched_external_and_surface_point_forces_trajectory_kernel,
        accumulate_batched_frame_loss_kernel=accumulate_batched_frame_loss_kernel,
        combine_batched_loss_components_kernel=combine_batched_loss_components_kernel,
        sum_batched_losses_kernel=sum_batched_losses_kernel,
    )
    return transform_positions_from_body_q_frames(body_q_frames, diff_scene.box_body), float(loss)


def rollout_positions_for_trajectories(
    *,
    diff_scene,
    trajectories,
    eval_args: argparse.Namespace,
    active_indices: np.ndarray,
    active_params: np.ndarray,
    initial_body_q: np.ndarray,
    initial_body_qd: np.ndarray,
) -> tuple[list[np.ndarray], list[float]]:
    all_positions: list[np.ndarray] = []
    all_losses: list[float] = []
    eval_batch_size = max(int(eval_args.eval_batch_size), 1)
    for batch_start in range(0, len(trajectories), eval_batch_size):
        batch_trajectories = trajectories[batch_start: batch_start + eval_batch_size]
        buffers = build_batched_optimization_buffers(diff_scene, batch_trajectories, eval_args, active_indices)
        buffers.active_point_friction.assign(active_params)
        buffers.full_point_friction.assign(buffers.inactive_point_friction_np)
        clear_batched_optimization_grads(buffers)
        reset_scene_states(diff_scene, initial_body_q, initial_body_qd)
        forward_rollout_with_batched_trajectory_loss(
            diff_scene=diff_scene,
            buffers=buffers,
            args=eval_args,
            scatter_active_point_friction_kernel=scatter_active_point_friction_kernel,
            compute_batched_contact_weighted_masses_kernel=compute_batched_contact_weighted_masses_kernel,
            apply_batched_external_and_surface_point_forces_trajectory_kernel=apply_batched_external_and_surface_point_forces_trajectory_kernel,
            accumulate_batched_frame_loss_kernel=accumulate_batched_frame_loss_kernel,
            combine_batched_loss_components_kernel=combine_batched_loss_components_kernel,
            sum_batched_losses_kernel=sum_batched_losses_kernel,
        )
        body_q_frames = [
            state.body_q.numpy().copy()
            for state in diff_scene.states[: buffers.max_frames]
        ]
        all_positions.extend(
            transform_batched_positions_from_states(
                body_q_frames,
                diff_scene.box_body_ids_np,
                batch_trajectories,
            )
        )
        all_losses.extend(float(value) for value in buffers.loss.numpy()[: len(batch_trajectories)])
    return all_positions, all_losses


def checkpoint_legend_label(method: MethodSpec, checkpoint: CheckpointParams) -> str:
    prefix = method.name
    iteration = f"it={checkpoint.iteration}" if checkpoint.iteration is not None else "it=?"
    params = checkpoint.active_params
    if checkpoint.parameterization == "global" and len(checkpoint.optimizer_params) >= 1:
        return f"{prefix} | global {iteration} mu={float(checkpoint.optimizer_params[0]):.3f}"
    if checkpoint.parameterization == "left-right" and len(checkpoint.optimizer_params) >= 2:
        return (
            f"{prefix} | left-right {iteration} "
            f"L={float(checkpoint.optimizer_params[0]):.3f} R={float(checkpoint.optimizer_params[1]):.3f}"
        )
    return (
        f"{prefix} | {checkpoint.parameterization} {iteration} "
        f"mu={float(np.mean(params)):.3f}+/-{float(np.std(params)):.3f} "
        f"[{float(np.min(params)):.3f},{float(np.max(params)):.3f}]"
    )


def checkpoint_summary(method: MethodSpec, checkpoint: CheckpointParams, losses: list[float]) -> dict:
    params = checkpoint.active_params
    summary = {
        "name": method.name,
        "stage": method.stage,
        "checkpoint": str(method.checkpoint),
        "parameterization": checkpoint.parameterization,
        "iteration": checkpoint.iteration,
        "train_best_loss": checkpoint.train_best_loss,
        "train_max_steps": checkpoint.train_max_steps,
        "train_dataset": checkpoint.train_dataset,
        "active_points": int(len(checkpoint.active_indices)),
        "mu_mean": float(np.mean(params)),
        "mu_std": float(np.std(params)),
        "mu_min": float(np.min(params)),
        "mu_max": float(np.max(params)),
        "overlay_loss_mean": float(np.mean(losses)) if losses else None,
        "overlay_loss_min": float(np.min(losses)) if losses else None,
        "overlay_loss_max": float(np.max(losses)) if losses else None,
    }
    if checkpoint.parameterization == "global" and len(checkpoint.optimizer_params) >= 1:
        summary["mu_global_param"] = float(checkpoint.optimizer_params[0])
    if checkpoint.parameterization == "left-right" and len(checkpoint.optimizer_params) >= 2:
        summary["mu_left_param"] = float(checkpoint.optimizer_params[0])
        summary["mu_right_param"] = float(checkpoint.optimizer_params[1])
    return summary


def equalize_axes(ax) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    radius = 0.5 * max(x1 - x0, y1 - y0, 1.0e-6)
    pad = max(radius * 0.12, 0.002)
    ax.set_xlim(cx - radius - pad, cx + radius + pad)
    ax.set_ylim(cy - radius - pad, cy + radius + pad)
    ax.set_aspect("equal", adjustable="box")


def main() -> None:
    args = parse_args()
    methods = select_methods(args)
    for method in methods:
        if not method.checkpoint.exists():
            raise FileNotFoundError(method.checkpoint)
        if not checkpoint_has_active_params(method.checkpoint):
            raise ValueError(f"{method.checkpoint} is not a training checkpoint with active friction parameters")

    wp.init()
    eval_args = make_eval_args(args)
    collection = load_mujoco_trajectories(eval_args.trajectory_npz, eval_args.max_steps, eval_args.max_trajectories)
    trajectories = collection.trajectories
    eval_args.steps = collection.max_steps
    eval_args.dt = trajectories[0].timestep

    selected_indices = select_representative_indices(args.dataset, args.trajectory_indices, args.all_trajectories)
    selected_indices = [idx for idx in selected_indices if 0 <= idx < len(trajectories)]
    if not selected_indices:
        raise ValueError("No valid trajectory indices selected")
    selected_trajectories = [trajectories[idx] for idx in selected_indices]
    eval_args.eval_batch_size = resolve_batch_size(args.eval_batch_size, len(selected_trajectories), eval_args.batch_size)
    eval_args.batch_size = eval_args.eval_batch_size
    eval_args.batch_capacity = max(eval_args.eval_batch_size, 1)

    diff_scene = build_diff_scene(eval_args)
    initial_body_q = diff_scene.states[0].body_q.numpy().copy()
    initial_body_qd = diff_scene.states[0].body_qd.numpy().copy()
    checkpoint_params = {
        method.name: load_checkpoint_params(method.checkpoint)
        for method in methods
    }
    legend_labels = {
        method.name: checkpoint_legend_label(method, checkpoint_params[method.name])
        for method in methods
    }

    cols = 5 if len(selected_indices) > 12 else 3
    rows = int(np.ceil(len(selected_indices) / cols))
    legend_width = 6.5 if len(methods) > 8 else 2.0
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 4.0 + legend_width, rows * 3.4),
        dpi=180,
        squeeze=False,
    )
    axes_flat = axes.reshape(-1)

    summary_lines = []
    method_positions: dict[str, list[np.ndarray]] = {}
    method_losses: dict[str, list[float]] = {}
    method_summaries: list[dict] = []
    for method_idx, method in enumerate(methods):
        checkpoint = checkpoint_params[method.name]
        print(
            f"rolling out {method_idx + 1}/{len(methods)} {method.name} "
            f"active={len(checkpoint.active_indices)} param={checkpoint.parameterization}",
            flush=True,
        )
        positions, losses = rollout_positions_for_trajectories(
            diff_scene=diff_scene,
            trajectories=selected_trajectories,
            eval_args=eval_args,
            active_indices=checkpoint.active_indices,
            active_params=checkpoint.active_params,
            initial_body_q=initial_body_q,
            initial_body_qd=initial_body_qd,
        )
        method_positions[method.name] = positions
        method_losses[method.name] = losses
        method_summaries.append(checkpoint_summary(method, checkpoint, losses))

    for plot_idx, trajectory_idx in enumerate(selected_indices):
        ax = axes_flat[plot_idx]
        trajectory = selected_trajectories[plot_idx]
        target_xy = trajectory.positions[:, :2]
        ax.plot(target_xy[:, 0], target_xy[:, 1], color="#111111", linewidth=2.4, label="Target")
        ax.scatter(target_xy[0, 0], target_xy[0, 1], color="#111111", s=16, marker="o", zorder=5)
        ax.scatter(target_xy[-1, 0], target_xy[-1, 1], color="#111111", s=28, marker="x", zorder=5)

        losses = {}
        for method in methods:
            pred_positions = method_positions[method.name][plot_idx]
            loss = method_losses[method.name][plot_idx]
            losses[method.name] = loss
            pred_xy = pred_positions[: len(target_xy), :2]
            ax.plot(
                pred_xy[:, 0],
                pred_xy[:, 1],
                color=method.color,
                linestyle=method.linestyle,
                linewidth=method.linewidth,
                label=legend_labels[method.name],
            )
            ax.scatter(pred_xy[-1, 0], pred_xy[-1, 1], color=method.color, s=22, marker="x", zorder=4)

        meta = trajectory.metadata
        point = np.asarray(trajectory.force_point_offset_local, dtype=np.float32)
        force = np.asarray(trajectory.step_forces[0], dtype=np.float32)
        ax.set_title(
            f"traj {trajectory_idx} | local point x={point[0]:.3f}, y={point[1]:.3f}\n"
            f"force xy=({force[0]:.2f}, {force[1]:.2f})",
            fontsize=9,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(alpha=0.22)
        equalize_axes(ax)
        best_method = min(losses, key=losses.get)
        summary_lines.append(
            f"traj={trajectory_idx} episode={meta.get('episode_index', trajectory_idx)} "
            f"best={best_method} " + " ".join(f"{name}={value:.6g}" for name, value in losses.items())
        )

    for ax in axes_flat[len(selected_indices):]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if len(methods) > 8:
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.775, 0.5),
            ncols=1,
            frameon=False,
            fontsize=6.2,
            handlelength=2.6,
            labelspacing=0.48,
        )
        tight_rect = (0, 0.0, 0.77, 0.94)
    else:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncols=min(len(labels), 5),
            frameon=False,
        )
        tight_rect = (0, 0.045, 1, 0.94)
    max_steps_text = "all steps" if args.max_steps is None else f"max_steps={args.max_steps}"
    fig.suptitle(
        f"Top-down trajectory overlays | {Path(args.dataset).stem} | {len(methods)} checkpoints | {max_steps_text}",
        y=0.985,
        fontsize=13,
    )
    fig.tight_layout(rect=tight_rect)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    summary_output = args.summary_output
    if summary_output is None:
        summary_output = args.output.with_name(f"{args.output.stem}_summary.json")
    summary_payload = {
        "dataset": str(args.dataset),
        "output": str(args.output),
        "max_steps": args.max_steps,
        "selected_trajectories": selected_indices,
        "eval_batch_size": eval_args.eval_batch_size,
        "contact_stiffness": args.contact_stiffness,
        "surface_point_spacing": args.surface_point_spacing,
        "methods": method_summaries,
        "trajectory_losses": {
            method.name: {
                str(selected_indices[idx]): float(loss)
                for idx, loss in enumerate(method_losses[method.name])
            }
            for method in methods
        },
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {summary_output}")
    print("selected_trajectories=" + ",".join(str(i) for i in selected_indices))
    for line in summary_lines:
        print(line)


if __name__ == "__main__":
    main()
