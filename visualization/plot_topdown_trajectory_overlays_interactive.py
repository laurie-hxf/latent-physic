from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import warp as wp

from plot_topdown_trajectory_overlays import (
    DEFAULT_DATASET,
    CheckpointParams,
    MethodSpec,
    checkpoint_has_active_params,
    checkpoint_legend_label,
    checkpoint_summary,
    load_checkpoint_params,
    make_eval_args,
    parse_optional_max_steps,
    rollout_positions_for_trajectories,
    select_methods,
    select_representative_indices,
    style_for_index,
    transform_batched_state_histories_from_states,
    unique_method_name,
)
from fit_mujoco_contact_point_friction_runtime import resolve_batch_size
from fit_mujoco_contact_point_friction_runtime import reset_scene_states
from mujoco_contact_friction_fit_utils import load_mujoco_trajectories
from newton_surface_points_diff_demo import build_diff_scene
from residual_dynamics_adapter.train_residual_adapter import (
    assign_rollout_buffer_trajectories,
    build_activation_buffers,
    build_rollout_buffers as build_residual_rollout_buffers,
    clear_gradients as clear_residual_gradients,
    forward_residual_rollout,
    initialize_mlp_parameters,
    load_adapter_checkpoint as load_residual_adapter_checkpoint,
    normalize_residual_output_mode,
    residual_output_mode_from_checkpoint,
)
from pointnet_residual_adapter.checkpoints import (
    LoadedAdapterCheckpoint,
    load_adapter_checkpoint as load_pointnet_adapter_checkpoint,
)
from pointnet_residual_adapter.features import DinoFeatures, normalize_residual_output_mode, quaternion_xyzw_to_yaw
from pointnet_residual_adapter.newton_rollout import (
    build_rollout_buffers as build_pointnet_rollout_buffers,
    run_closed_loop_pointnet_rollout,
    run_closed_loop_pointnet_rollout_batch,
)
from stateful_gru_residual_adapter.conditional_direct_state_eval import (
    LoadedConditionalDirectStateCheckpoint,
    checkpoint_is_conditional_direct_state,
    load_conditional_direct_state_checkpoint,
    run_conditional_direct_state_rollout_batch,
)


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT = ROOT / "report_assets" / "topdown_trajectory_overlays_fixed20_interactive.html"
REFERENCE_COLORS = ["#7a3e9d", "#00798c", "#b7791f", "#c43b5b", "#4b5563"]


@dataclass(frozen=True)
class ResidualCheckpointParams:
    active_indices: np.ndarray
    active_params: np.ndarray
    full_point_friction: np.ndarray
    parameterization: str
    iteration: int | None
    train_max_steps: int | None
    train_dataset: str
    train_friction_end_to_end: bool
    left_right_delta_sum_zero: bool
    mu_features: np.ndarray
    residual_output_mode: str


@dataclass(frozen=True)
class PointNetCheckpointParams:
    metadata: dict
    local_surface_points: np.ndarray
    full_point_friction: np.ndarray
    active_contact_mask: np.ndarray
    dino_features: np.ndarray | None
    dino_bottom_feature_copied_from_top: np.ndarray | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument(
        "--reference-dataset",
        type=Path,
        action="append",
        default=None,
        help="Additional MuJoCo datasets whose ground-truth XY trajectories should be overlaid.",
    )
    parser.add_argument(
        "--reference-label",
        type=str,
        action="append",
        default=None,
        help="Display label for each --reference-dataset. Must be repeated the same number of times when used.",
    )
    parser.add_argument(
        "--reference-color",
        type=str,
        action="append",
        default=None,
        help="CSS color for each --reference-dataset. Must be repeated the same number of times when used.",
    )
    parser.add_argument(
        "--method-source",
        choices=("default", "curated", "auto", "all"),
        default="all",
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
    parser.add_argument("--position-loss-weight", type=float, default=1.0)
    parser.add_argument("--orientation-loss-weight", type=float, default=0.0)
    parser.add_argument("--linear-velocity-loss-weight", type=float, default=0.0)
    parser.add_argument("--angular-velocity-loss-weight", type=float, default=0.0)
    parser.add_argument("--point-position-loss-reduction", choices=("sum", "mean"), default="mean")
    parser.add_argument(
        "--pointnet-residual-gain",
        type=float,
        default=None,
        help="Scale neural-adapter residuals. None uses each checkpoint's training metadata, falling back to 1.0.",
    )
    parser.add_argument(
        "--pointnet-residual-output-mode",
        choices=("checkpoint", "velocity", "acceleration", "pose", "position", "pose_velocity", "all"),
        default="checkpoint",
        help="How to interpret PointNet outputs. checkpoint uses each checkpoint's metadata.",
    )
    parser.add_argument(
        "--stateful-reset-interval",
        type=int,
        default=None,
        help="Reset stateful adapter memory every N steps. None uses checkpoint metadata; 0 never resets.",
    )
    parser.add_argument(
        "--include-pointnet-last-checkpoints",
        action="store_true",
        help="When scanning checkpoint roots, include PointNet *_last.pt checkpoints as separate methods.",
    )
    parser.add_argument("--trajectory-indices", type=int, nargs="*", default=None)
    parser.add_argument(
        "--all-trajectories",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot every trajectory in the dataset unless --trajectory-indices is provided.",
    )
    parser.add_argument("--include-pure-point", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-width", type=int, default=280)
    parser.add_argument("--plot-height", type=int, default=230)
    parser.add_argument("--legend-width", type=int, default=520)
    parser.add_argument("--axis-padding-frac", type=float, default=0.12)
    parser.add_argument(
        "--unified-axis-scale",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use one shared equal-aspect x/y axis range for every panel.",
    )
    parser.add_argument(
        "--reuse-summary",
        type=Path,
        default=None,
        help=(
            "Optional summary JSON from this script. When present, reuse saved XY polylines instead of rerunning Newton."
        ),
    )
    return parser.parse_args()


def default_reference_label(path: Path) -> str:
    stem = path.stem
    marker = "_uniform_mu_"
    if marker in stem:
        mu = stem.rsplit(marker, 1)[1].replace("p", ".")
        return f"Uniform mu={mu}"
    return stem


def collect_reference_payload(args: argparse.Namespace, selected_indices: list[int]) -> list[dict]:
    reference_datasets = args.reference_dataset or []
    reference_labels = args.reference_label or []
    reference_colors = args.reference_color or []
    if reference_labels and len(reference_labels) != len(reference_datasets):
        raise ValueError("--reference-label must be provided once per --reference-dataset")
    if reference_colors and len(reference_colors) != len(reference_datasets):
        raise ValueError("--reference-color must be provided once per --reference-dataset")

    references = []
    used_names: set[str] = set()
    for ref_idx, dataset in enumerate(reference_datasets):
        if not dataset.exists():
            raise FileNotFoundError(dataset)
        collection = load_mujoco_trajectories(dataset, args.max_steps, None)
        tracks = []
        for selected_idx in selected_indices:
            if selected_idx >= len(collection.trajectories):
                raise ValueError(f"{dataset} has no trajectory index {selected_idx}")
            trajectory = collection.trajectories[selected_idx]
            tracks.append(np.asarray(trajectory.positions[:, :2], dtype=np.float32).tolist())

        base_name = f"reference_{dataset.stem}"
        name = base_name
        suffix = 2
        while name in used_names:
            name = f"{base_name}_{suffix}"
            suffix += 1
        used_names.add(name)

        references.append(
            {
                "name": name,
                "label": reference_labels[ref_idx] if reference_labels else default_reference_label(dataset),
                "color": reference_colors[ref_idx] if reference_colors else REFERENCE_COLORS[ref_idx % len(REFERENCE_COLORS)],
                "dataset": str(dataset),
                "tracks": tracks,
            }
        )
    return references


def checkpoint_is_residual_adapter(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=True) as data:
            required = {"w0", "b0", "w1", "b1", "w2", "b2", "w3", "b3", "feature_mean", "feature_std"}
            return required.issubset(set(data.files))
    except Exception:
        return False


def load_residual_checkpoint_params(path: Path) -> ResidualCheckpointParams:
    with np.load(path, allow_pickle=True) as data:
        active_indices = np.asarray(data["active_indices"], dtype=np.int32)
        active_params = np.asarray(data["active_params"], dtype=np.float32)
        full_point_friction = np.asarray(data["full_point_friction"], dtype=np.float32)
        parameterization = str(np.asarray(data["friction_parameterization"]).item())
        iteration = int(np.asarray(data["iteration"]).item()) if "iteration" in data.files else None
        train_max_steps = int(np.asarray(data["max_steps"]).item()) if "max_steps" in data.files else None
        train_dataset = str(np.asarray(data["trajectory_npz_path"]).item()) if "trajectory_npz_path" in data.files else ""
        train_friction_end_to_end = (
            bool(np.asarray(data["train_friction_end_to_end"]).item())
            if "train_friction_end_to_end" in data.files
            else False
        )
        left_right_delta_sum_zero = (
            bool(np.asarray(data["left_right_delta_sum_zero"]).item())
            if "left_right_delta_sum_zero" in data.files
            else False
        )
        mu_features = np.asarray(data["mu_features"], dtype=np.float32) if "mu_features" in data.files else np.zeros(3, dtype=np.float32)
        residual_output_mode = (
            normalize_residual_output_mode(np.asarray(data["residual_output_mode"]).item())
            if "residual_output_mode" in data.files
            else residual_output_mode_from_checkpoint(path)
        )
    if active_indices.shape != active_params.shape:
        raise ValueError(f"{path} active_indices and active_params length mismatch")
    return ResidualCheckpointParams(
        active_indices=active_indices,
        active_params=active_params,
        full_point_friction=full_point_friction,
        parameterization=parameterization,
        iteration=iteration,
        train_max_steps=train_max_steps,
        train_dataset=train_dataset,
        train_friction_end_to_end=train_friction_end_to_end,
        left_right_delta_sum_zero=left_right_delta_sum_zero,
        mu_features=mu_features,
        residual_output_mode=residual_output_mode,
    )


def checkpoint_is_pointnet_adapter(path: Path, *, include_last: bool = False) -> bool:
    if path.suffix != ".pt" or "wandb" in path.parts:
        return False
    if path.name.endswith("_last.pt") and not include_last:
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata", {})
        return (
            "model_state_dict" in payload
            and isinstance(metadata, dict)
            and "point_feature_dim" in metadata
            and "history_window_steps" in metadata
            and "prediction_window_steps" in metadata
            and "full_point_friction" in payload
        )
    except Exception:
        return False


def load_pointnet_checkpoint_params(path: Path) -> PointNetCheckpointParams:
    checkpoint = load_pointnet_adapter_checkpoint(path, map_location="cpu")
    return PointNetCheckpointParams(
        metadata=dict(checkpoint.metadata),
        local_surface_points=np.asarray(checkpoint.local_surface_points, dtype=np.float32),
        full_point_friction=np.asarray(checkpoint.full_point_friction, dtype=np.float32),
        active_contact_mask=np.asarray(checkpoint.active_contact_mask, dtype=bool),
        dino_features=None if checkpoint.dino_features is None else np.asarray(checkpoint.dino_features, dtype=np.float32),
        dino_bottom_feature_copied_from_top=(
            None
            if checkpoint.dino_bottom_feature_copied_from_top is None
            else np.asarray(checkpoint.dino_bottom_feature_copied_from_top, dtype=np.float32)
        ),
    )


def residual_legend_label(method: MethodSpec, checkpoint: ResidualCheckpointParams) -> str:
    iteration = f"it={checkpoint.iteration}" if checkpoint.iteration is not None else "it=?"
    mode = "e2e" if checkpoint.train_friction_end_to_end else "frozen"
    output_mode = "vel" if checkpoint.residual_output_mode == "velocity" else "acc"
    if checkpoint.parameterization == "global":
        return (
            f"{method.name} | residual {mode} {output_mode} global "
            f"{iteration} mu={float(np.mean(checkpoint.active_params)):.3f}"
        )
    if checkpoint.parameterization in {"left-right", "base-delta"}:
        mu = checkpoint.mu_features
        suffix = " sum0" if checkpoint.left_right_delta_sum_zero else ""
        return (
            f"{method.name} | residual {mode} {output_mode} {checkpoint.parameterization}{suffix} {iteration} "
            f"L={float(mu[1]):.3f} R={float(mu[2]):.3f}"
        )
    return (
        f"{method.name} | residual {mode} {output_mode} {checkpoint.parameterization} {iteration} "
        f"mu={float(np.mean(checkpoint.active_params)):.3f}+/-{float(np.std(checkpoint.active_params)):.3f}"
    )


def neural_adapter_architecture_label(metadata: dict) -> tuple[str, str]:
    architecture = str(metadata.get("adapter_architecture", "pointnet_residual_adapter"))
    if architecture == "stateful_gru_residual_adapter":
        return architecture, "Stateful GRU"
    if architecture == "rnn_residual_adapter":
        return architecture, "RNN"
    return "pointnet_residual_adapter", "PointNet"


def residual_output_short_label(output_mode: str) -> str:
    normalized = normalize_residual_output_mode(output_mode)
    if normalized == "acceleration":
        return "acc"
    if normalized == "pose":
        return "pose"
    if normalized == "pose_velocity":
        return "pose+vel"
    return "vel"


def pointnet_legend_label(method: MethodSpec, checkpoint: PointNetCheckpointParams) -> str:
    metadata = checkpoint.metadata
    _, architecture_label = neural_adapter_architecture_label(metadata)
    dino_dim = int(metadata.get("dino_feature_dim", 0))
    feature_label = "with DINO" if dino_dim > 0 else "no DINO"
    output_label = residual_output_short_label(str(metadata.get("residual_output_mode", "velocity")))
    train_count = metadata.get("train_trajectories", "?")
    return (
        f"{method.name} | {architecture_label} {feature_label} {output_label} "
        f"H={int(metadata.get('history_window_steps', 0))} P={int(metadata.get('prediction_window_steps', 0))} "
        f"train={train_count}"
    )


def residual_checkpoint_summary(method: MethodSpec, checkpoint: ResidualCheckpointParams, losses: list[float]) -> dict:
    params = checkpoint.active_params
    summary = {
        "name": method.name,
        "stage": method.stage,
        "checkpoint": str(method.checkpoint),
        "checkpoint_type": "residual_adapter",
        "parameterization": checkpoint.parameterization,
        "residual_output_mode": checkpoint.residual_output_mode,
        "iteration": checkpoint.iteration,
        "train_max_steps": checkpoint.train_max_steps,
        "train_dataset": checkpoint.train_dataset,
        "train_friction_end_to_end": bool(checkpoint.train_friction_end_to_end),
        "left_right_delta_sum_zero": bool(checkpoint.left_right_delta_sum_zero),
        "active_points": int(len(checkpoint.active_indices)),
        "mu_mean": float(np.mean(params)),
        "mu_std": float(np.std(params)),
        "mu_min": float(np.min(params)),
        "mu_max": float(np.max(params)),
        "mu_feature_mean": float(checkpoint.mu_features[0]) if len(checkpoint.mu_features) > 0 else None,
        "mu_left": float(checkpoint.mu_features[1]) if len(checkpoint.mu_features) > 1 else None,
        "mu_right": float(checkpoint.mu_features[2]) if len(checkpoint.mu_features) > 2 else None,
        "overlay_loss_mean": float(np.mean(losses)) if losses else None,
        "overlay_loss_min": float(np.min(losses)) if losses else None,
        "overlay_loss_max": float(np.max(losses)) if losses else None,
    }
    return summary


def pointnet_checkpoint_summary(method: MethodSpec, checkpoint: PointNetCheckpointParams, losses: list[float]) -> dict:
    metadata = checkpoint.metadata
    mu = checkpoint.full_point_friction
    architecture, _ = neural_adapter_architecture_label(metadata)
    return {
        "name": method.name,
        "stage": method.stage,
        "checkpoint": str(method.checkpoint),
        "checkpoint_type": architecture,
        "adapter_architecture": architecture,
        "history_window_steps": int(metadata.get("history_window_steps", 0)),
        "prediction_window_steps": int(metadata.get("prediction_window_steps", 0)),
        "point_feature_dim": int(metadata.get("point_feature_dim", 0)),
        "dino_feature_dim": int(metadata.get("dino_feature_dim", 0)),
        "rnn_hidden_size_1": metadata.get("rnn_hidden_size_1"),
        "rnn_hidden_size_2": metadata.get("rnn_hidden_size_2"),
        "rnn_point_pooling": metadata.get("rnn_point_pooling"),
        "gru_hidden_size": metadata.get("gru_hidden_size"),
        "gru_num_layers": metadata.get("gru_num_layers"),
        "gru_point_pooling": metadata.get("gru_point_pooling"),
        "stateful_rollout": metadata.get("stateful_rollout"),
        "stateful_reset_interval": metadata.get("stateful_reset_interval"),
        "burn_in_steps": metadata.get("burn_in_steps"),
        "tbptt_chunk_steps": metadata.get("tbptt_chunk_steps"),
        "residual_output_mode": normalize_residual_output_mode(metadata.get("residual_output_mode", "velocity")),
        "friction_source_type": metadata.get("friction_source_type"),
        "friction_parameterization": metadata.get("friction_parameterization"),
        "training_dataset": metadata.get("training_dataset"),
        "train_trajectories": metadata.get("train_trajectories"),
        "val_trajectories": metadata.get("val_trajectories"),
        "mu_mean": float(np.mean(mu)),
        "mu_std": float(np.std(mu)),
        "mu_min": float(np.min(mu)),
        "mu_max": float(np.max(mu)),
        "active_points": int(np.count_nonzero(checkpoint.active_contact_mask)),
        "overlay_loss_mean": float(np.mean(losses)) if losses else None,
        "overlay_loss_min": float(np.min(losses)) if losses else None,
        "overlay_loss_max": float(np.max(losses)) if losses else None,
    }


def direct_state_legend_label(method: MethodSpec, checkpoint: LoadedConditionalDirectStateCheckpoint) -> str:
    metadata = checkpoint.metadata
    mode = str(metadata.get("output_mode", "position_velocity")).replace("_", "+")
    return (
        f"{method.name} | Stateful GRU direct {mode} "
        f"input={int(metadata.get('input_dim', 0))} output={int(metadata.get('output_dim', 0))} "
        f"train={metadata.get('train_trajectories', '?')}"
    )


def direct_state_checkpoint_summary(
    method: MethodSpec,
    checkpoint: LoadedConditionalDirectStateCheckpoint,
    losses: list[float],
) -> dict:
    metadata = checkpoint.metadata
    mu = checkpoint.full_point_friction
    return {
        "name": method.name,
        "stage": method.stage,
        "checkpoint": str(method.checkpoint),
        "checkpoint_type": str(metadata.get("adapter_architecture")),
        "adapter_architecture": str(metadata.get("adapter_architecture")),
        "model_semantics": metadata.get("model_semantics"),
        "condition_formula": metadata.get("condition_formula"),
        "rollout_semantics": "open_loop_newton_conditioned_direct_state_sequence",
        "predicted_state_is_fed_back_to_newton": False,
        "output_mode": metadata.get("output_mode"),
        "output_dim": metadata.get("output_dim"),
        "output_schema": metadata.get("output_schema"),
        "unpredicted_state_components_source": (
            "newton_open_loop" if metadata.get("output_mode") == "position" else "not_applicable"
        ),
        "input_dim": metadata.get("input_dim"),
        "gru_hidden_size": metadata.get("gru_hidden_size"),
        "gru_num_layers": metadata.get("gru_num_layers"),
        "training_sequence": metadata.get("training_sequence"),
        "training_dataset": metadata.get("training_dataset"),
        "train_trajectories": metadata.get("train_trajectories"),
        "friction_source_type": metadata.get("friction_source_type"),
        "friction_parameterization": metadata.get("friction_parameterization"),
        "mu_mean": float(np.mean(mu)),
        "mu_std": float(np.std(mu)),
        "mu_min": float(np.min(mu)),
        "mu_max": float(np.max(mu)),
        "active_points": int(np.count_nonzero(checkpoint.active_contact_mask)),
        "overlay_loss_mean": float(np.mean(losses)) if losses else None,
        "overlay_loss_min": float(np.min(losses)) if losses else None,
        "overlay_loss_max": float(np.max(losses)) if losses else None,
    }


def discover_residual_adapter_methods(
    *,
    roots: list[Path],
    existing_paths: set[Path],
    start_index: int,
) -> list[MethodSpec]:
    methods: list[MethodSpec] = []
    used_names: set[str] = set()
    for root in roots:
        if not root.exists():
            print(f"warning: residual checkpoint root does not exist: {root}", flush=True)
            continue
        for path in sorted(root.rglob("*.npz")):
            resolved = path.resolve()
            if resolved in existing_paths:
                continue
            if not checkpoint_is_residual_adapter(path):
                continue
            parent = path.parent.name
            base_name = path.stem if parent in {"checkpoints", "selected_checkpoints"} else parent
            if parent in {"outputs", "debug-stash"}:
                base_name = path.stem
            color, linestyle = style_for_index(start_index + len(methods))
            methods.append(
                MethodSpec(
                    name=unique_method_name(base_name, used_names),
                    checkpoint=path,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.35,
                    stage="Residual adapter",
                )
            )
    return methods


def discover_pointnet_adapter_methods(
    *,
    roots: list[Path],
    existing_paths: set[Path],
    start_index: int,
    include_last: bool,
) -> list[MethodSpec]:
    methods: list[MethodSpec] = []
    used_names: set[str] = set()
    for root in roots:
        if not root.exists():
            print(f"warning: pointnet checkpoint root does not exist: {root}", flush=True)
            continue
        for path in sorted(root.rglob("*.pt")):
            resolved = path.resolve()
            if resolved in existing_paths:
                continue
            if path.stem.endswith(("_best_pretrain", "_best_closed_loop")):
                canonical = path.parent / f"{path.parent.name}.pt"
                if canonical.exists():
                    continue
            if not checkpoint_is_pointnet_adapter(path, include_last=include_last):
                continue
            parent = path.parent.name
            base_name = path.stem if path.name.endswith("_last.pt") or parent in {"checkpoints", "selected_checkpoints"} else parent
            color, linestyle = style_for_index(start_index + len(methods))
            methods.append(
                MethodSpec(
                    name=unique_method_name(base_name, used_names),
                    checkpoint=path,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.55,
                    stage="PointNet residual adapter",
                )
            )
    return methods


def discover_conditional_direct_state_methods(
    *,
    roots: list[Path],
    existing_paths: set[Path],
    start_index: int,
    include_last: bool,
) -> list[MethodSpec]:
    methods: list[MethodSpec] = []
    used_names: set[str] = set()
    for root in roots:
        if not root.exists():
            print(f"warning: direct-state checkpoint root does not exist: {root}", flush=True)
            continue
        for path in sorted(root.rglob("*.pt")):
            resolved = path.resolve()
            if resolved in existing_paths:
                continue
            if not checkpoint_is_conditional_direct_state(path, include_last=include_last):
                continue
            parent = path.parent.name
            base_name = path.stem if path.name.endswith("_last.pt") else parent
            color, linestyle = style_for_index(start_index + len(methods))
            methods.append(
                MethodSpec(
                    name=unique_method_name(base_name, used_names),
                    checkpoint=path,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.55,
                    stage="Stateful GRU direct state",
                )
            )
    return methods


def select_overlay_methods(args: argparse.Namespace) -> list[MethodSpec]:
    explicit_checkpoints = list(getattr(args, "checkpoint", None) or [])
    if explicit_checkpoints:
        used_names: set[str] = set()
        methods: list[MethodSpec] = []
        for idx, checkpoint in enumerate(explicit_checkpoints):
            path = Path(checkpoint)
            color, linestyle = style_for_index(idx)
            methods.append(
                MethodSpec(
                    name=unique_method_name(path.stem, used_names),
                    checkpoint=path,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.55,
                    stage="Explicit checkpoint",
                )
            )
        return methods
    methods = select_methods(args)
    if args.method_source not in {"auto", "all"} or args.checkpoint_root is None:
        return methods
    existing_paths = {method.checkpoint.resolve() for method in methods}
    methods.extend(
        discover_residual_adapter_methods(
            roots=args.checkpoint_root,
            existing_paths=existing_paths,
            start_index=len(methods),
        )
    )
    existing_paths = {method.checkpoint.resolve() for method in methods}
    methods.extend(
        discover_pointnet_adapter_methods(
            roots=args.checkpoint_root,
            existing_paths=existing_paths,
            start_index=len(methods),
            include_last=bool(args.include_pointnet_last_checkpoints),
        )
    )
    existing_paths = {method.checkpoint.resolve() for method in methods}
    methods.extend(
        discover_conditional_direct_state_methods(
            roots=args.checkpoint_root,
            existing_paths=existing_paths,
            start_index=len(methods),
            include_last=bool(args.include_pointnet_last_checkpoints),
        )
    )
    return methods


def residual_rollout_positions_for_trajectories(
    *,
    diff_scene,
    sim_states,
    trajectories,
    eval_args: argparse.Namespace,
    checkpoint_path: Path,
    checkpoint: ResidualCheckpointParams,
    initial_body_q: np.ndarray,
    initial_body_qd: np.ndarray,
) -> tuple[list[np.ndarray], list[float], list[dict[str, np.ndarray]]]:
    device = str(diff_scene.torch_device)
    rng = np.random.default_rng(0)
    eval_args.residual_output_mode = checkpoint.residual_output_mode
    params, adam = initialize_mlp_parameters(eval_args, device, rng)
    _, feature_mean, feature_std, _ = load_residual_adapter_checkpoint(checkpoint_path, params, adam)
    feature_mean = np.asarray(feature_mean, dtype=np.float32)
    feature_std = np.asarray(feature_std, dtype=np.float32)
    feature_mean[8:11] = 0.0
    feature_std[8:11] = 1.0
    feature_inv_std = (1.0 / np.maximum(feature_std, 1.0e-6)).astype(np.float32)
    feature_mean_wp = wp.array(feature_mean, dtype=wp.float32, device=device)
    feature_inv_std_wp = wp.array(feature_inv_std, dtype=wp.float32, device=device)
    mu_features_wp = wp.array(checkpoint.mu_features, dtype=wp.float32, device=device)

    all_positions: list[np.ndarray] = []
    all_losses: list[float] = []
    all_state_histories: list[dict[str, np.ndarray]] = []
    eval_batch_size = max(int(eval_args.eval_batch_size), 1)
    for batch_start in range(0, len(trajectories), eval_batch_size):
        batch_trajectories = trajectories[batch_start: batch_start + eval_batch_size]
        buffers = build_residual_rollout_buffers(
            device=device,
            point_count=len(diff_scene.local_surface_points_np),
            full_point_friction=checkpoint.full_point_friction,
            batch_capacity=min(eval_batch_size, max(len(batch_trajectories), 1)),
            step_capacity=int(eval_args.steps),
        )
        activations = build_activation_buffers(
            device=device,
            batch_capacity=buffers.batch_capacity,
            step_capacity=buffers.step_capacity,
        )
        active_batch_size = assign_rollout_buffer_trajectories(buffers, batch_trajectories)
        clear_residual_gradients(params, activations, buffers, trainable_friction=None)
        reset_scene_states(diff_scene, initial_body_q, initial_body_qd)
        forward_residual_rollout(
            diff_scene=diff_scene,
            sim_states=sim_states,
            buffers=buffers,
            activations=activations,
            batch_size=active_batch_size,
            params=params,
            feature_mean=feature_mean_wp,
            feature_inv_std=feature_inv_std_wp,
            mu_features=mu_features_wp,
            trainable_friction=None,
            args=eval_args,
        )
        body_q_frames = [
            state.body_q.numpy().copy()
            for state in diff_scene.states[: buffers.frame_capacity]
        ]
        body_qd_frames = [
            state.body_qd.numpy().copy()
            for state in diff_scene.states[: buffers.frame_capacity]
        ]
        for batch_idx, trajectory in enumerate(batch_trajectories):
            body_id = int(diff_scene.box_body_ids_np[batch_idx])
            positions = []
            for frame in body_q_frames[: trajectory.num_frames]:
                transform_value = np.asarray(frame[body_id])
                positions.append(transform_value.reshape(-1)[:3].astype(np.float32))
            all_positions.append(np.asarray(positions, dtype=np.float32))
        all_state_histories.extend(
            transform_batched_state_histories_from_states(
                body_q_frames,
                body_qd_frames,
                diff_scene.box_body_ids_np,
                batch_trajectories,
            )
        )
        all_losses.extend(float(value) for value in buffers.loss.numpy()[:active_batch_size])
    return all_positions, all_losses, all_state_histories


def _pointnet_dino_from_checkpoint(checkpoint: LoadedAdapterCheckpoint) -> DinoFeatures | None:
    dino_dim = int(checkpoint.metadata.get("dino_feature_dim", 0))
    if dino_dim <= 0:
        return None
    if checkpoint.dino_features is None or checkpoint.dino_bottom_feature_copied_from_top is None:
        raise ValueError(f"{checkpoint.path} metadata enables DINO but checkpoint tensors are missing")
    dino_path = checkpoint.metadata.get("dino_feature_npz")
    return DinoFeatures(
        path=Path(str(dino_path)) if dino_path else checkpoint.path,
        features=np.asarray(checkpoint.dino_features, dtype=np.float32),
        bottom_feature_copied_from_top=np.asarray(
            checkpoint.dino_bottom_feature_copied_from_top,
            dtype=np.float32,
        ),
        max_match_distance=float(checkpoint.metadata.get("dino_max_match_distance", 1.0e-5)),
    )


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle)).astype(np.float32)


def _pointnet_overlay_loss(predicted, trajectory, args: argparse.Namespace) -> float:
    frames = min(predicted.positions.shape[1], trajectory.num_frames)
    pred_pos = np.asarray(predicted.positions[0, :frames], dtype=np.float32)
    pred_quat = np.asarray(predicted.quaternions_xyzw[0, :frames], dtype=np.float32)
    pred_linear = np.asarray(predicted.linear_velocity[0, :frames], dtype=np.float32)
    pred_angular = np.asarray(predicted.angular_velocity[0, :frames], dtype=np.float32)
    gt_pos = np.asarray(trajectory.positions[:frames], dtype=np.float32)
    gt_quat = np.asarray(trajectory.quaternions_xyzw[:frames], dtype=np.float32)
    gt_linear = np.asarray(trajectory.linear_velocity[:frames], dtype=np.float32)
    gt_angular = np.asarray(trajectory.angular_velocity[:frames], dtype=np.float32)
    yaw_error = _wrap_angle(quaternion_xyzw_to_yaw(pred_quat) - quaternion_xyzw_to_yaw(gt_quat))
    loss = 0.0
    loss += float(args.position_loss_weight) * float(np.mean((pred_pos - gt_pos) ** 2))
    loss += float(args.orientation_loss_weight) * float(np.mean(yaw_error * yaw_error))
    loss += float(args.linear_velocity_loss_weight) * float(np.mean((pred_linear - gt_linear) ** 2))
    loss += float(args.angular_velocity_loss_weight) * float(np.mean((pred_angular - gt_angular) ** 2))
    return float(loss)


def pointnet_rollout_positions_for_trajectories(
    *,
    diff_scene,
    trajectories,
    eval_args: argparse.Namespace,
    checkpoint_path: Path,
    initial_body_q: np.ndarray,
    initial_body_qd: np.ndarray,
) -> tuple[list[np.ndarray], list[float], dict | None, list[dict[str, np.ndarray]]]:
    checkpoint = load_pointnet_adapter_checkpoint(checkpoint_path, map_location="cpu")
    _, architecture_label = neural_adapter_architecture_label(checkpoint.metadata)
    if not np.allclose(diff_scene.local_surface_points_np, checkpoint.local_surface_points, atol=1.0e-6):
        raise ValueError(f"Current surface-point grid does not match {architecture_label} checkpoint: {checkpoint_path}")
    configured_output_mode = str(getattr(eval_args, "pointnet_residual_output_mode", "checkpoint"))
    output_mode = (
        checkpoint.metadata.get("residual_output_mode", "velocity")
        if configured_output_mode == "checkpoint"
        else configured_output_mode
    )
    rollout_args = argparse.Namespace(**vars(eval_args))
    rollout_args.history_window_steps = int(checkpoint.metadata["history_window_steps"])
    rollout_args.prediction_window_steps = int(checkpoint.metadata["prediction_window_steps"])
    rollout_args.pointnet_residual_output_mode = normalize_residual_output_mode(output_mode)
    rollout_args.pointnet_residual_gain = (
        float(checkpoint.metadata.get("pointnet_residual_gain", 1.0))
        if eval_args.pointnet_residual_gain is None
        else float(eval_args.pointnet_residual_gain)
    )
    configured_reset_interval = getattr(eval_args, "stateful_reset_interval", None)
    rollout_args.stateful_reset_interval = (
        int(checkpoint.metadata.get("stateful_reset_interval", 0))
        if configured_reset_interval is None
        else int(configured_reset_interval)
    )
    dino = _pointnet_dino_from_checkpoint(checkpoint)
    torch_device = diff_scene.torch_device
    model = checkpoint.model.to(torch_device)
    model.eval()

    all_positions: list[np.ndarray] = []
    all_losses: list[float] = []
    all_state_histories: list[dict[str, np.ndarray]] = []
    hidden_norms: list[np.ndarray] = []
    hidden_saturation: list[np.ndarray] = []
    eval_batch_size = max(int(rollout_args.eval_batch_size), 1)
    for batch_start in range(0, len(trajectories), eval_batch_size):
        batch_trajectories = trajectories[batch_start : batch_start + eval_batch_size]
        print(
            f"  {architecture_label} trajectories "
            f"{batch_start + 1}-{batch_start + len(batch_trajectories)}/{len(trajectories)}",
            flush=True,
        )
        buffers = build_pointnet_rollout_buffers(
            device=str(torch_device),
            batch_capacity=max(len(batch_trajectories), 1),
            step_capacity=int(rollout_args.steps),
            point_count=len(diff_scene.local_surface_points_np),
            full_point_friction=checkpoint.full_point_friction,
        )
        predicted, _ = run_closed_loop_pointnet_rollout_batch(
            diff_scene=diff_scene,
            buffers=buffers,
            trajectories=batch_trajectories,
            args=rollout_args,
            initial_body_q=initial_body_q,
            initial_body_qd=initial_body_qd,
            model=model,
            normalizer=checkpoint.normalizer,
            local_surface_points=diff_scene.local_surface_points_np,
            box_half_extents=np.asarray(checkpoint.metadata["box_half_extents"], dtype=np.float32),
            point_friction=checkpoint.full_point_friction,
            active_contact_mask=checkpoint.active_contact_mask,
            dino=dino,
            torch_device=torch_device,
        )
        diagnostics = getattr(model, "last_stateful_rollout_diagnostics", None)
        if isinstance(diagnostics, dict):
            hidden_norms.extend(np.asarray(diagnostics["hidden_l2_norm"], dtype=np.float32))
            hidden_saturation.extend(np.asarray(diagnostics["hidden_saturation_fraction"], dtype=np.float32))
        for batch_idx, trajectory in enumerate(batch_trajectories):
            single_predicted = type(predicted)(
                positions=predicted.positions[batch_idx : batch_idx + 1],
                quaternions_xyzw=predicted.quaternions_xyzw[batch_idx : batch_idx + 1],
                linear_velocity=predicted.linear_velocity[batch_idx : batch_idx + 1],
                angular_velocity=predicted.angular_velocity[batch_idx : batch_idx + 1],
            )
            positions = np.asarray(single_predicted.positions[0, : trajectory.num_frames], dtype=np.float32)
            all_positions.append(positions)
            all_state_histories.append(
                {
                    "positions": np.asarray(single_predicted.positions[0, : trajectory.num_frames], dtype=np.float32),
                    "quaternions_xyzw": np.asarray(
                        single_predicted.quaternions_xyzw[0, : trajectory.num_frames],
                        dtype=np.float32,
                    ),
                    "linear_velocity": np.asarray(
                        single_predicted.linear_velocity[0, : trajectory.num_frames],
                        dtype=np.float32,
                    ),
                    "angular_velocity": np.asarray(
                        single_predicted.angular_velocity[0, : trajectory.num_frames],
                        dtype=np.float32,
                    ),
                }
            )
            all_losses.append(_pointnet_overlay_loss(single_predicted, trajectory, rollout_args))
    if hidden_norms:
        rollout_diagnostics = {
            "stateful_reset_interval": int(rollout_args.stateful_reset_interval),
            "hidden_l2_norm_mean": float(np.mean([np.mean(values) for values in hidden_norms])),
            "hidden_l2_norm_max": float(np.max([np.max(values) for values in hidden_norms])),
            "hidden_saturation_fraction_mean": float(np.mean([np.mean(values) for values in hidden_saturation])),
            "hidden_saturation_fraction_max": float(np.max([np.max(values) for values in hidden_saturation])),
        }
    else:
        rollout_diagnostics = None
    return all_positions, all_losses, rollout_diagnostics, all_state_histories


def direct_state_rollout_positions_for_trajectories(
    *,
    diff_scene,
    trajectories,
    eval_args: argparse.Namespace,
    checkpoint_path: Path,
    initial_body_q: np.ndarray,
    initial_body_qd: np.ndarray,
) -> tuple[list[np.ndarray], list[float], dict, list[dict[str, np.ndarray]]]:
    checkpoint = load_conditional_direct_state_checkpoint(checkpoint_path, map_location="cpu")
    if not np.allclose(diff_scene.local_surface_points_np, checkpoint.local_surface_points, atol=1.0e-6):
        raise ValueError(f"Current surface-point grid does not match direct-state checkpoint: {checkpoint_path}")
    torch_device = diff_scene.torch_device
    all_positions: list[np.ndarray] = []
    all_losses: list[float] = []
    all_state_histories: list[dict[str, np.ndarray]] = []
    hidden_norms: list[np.ndarray] = []
    hidden_saturation: list[np.ndarray] = []
    eval_batch_size = max(int(eval_args.eval_batch_size), 1)
    for batch_start in range(0, len(trajectories), eval_batch_size):
        batch_trajectories = trajectories[batch_start : batch_start + eval_batch_size]
        print(
            f"  Stateful GRU direct trajectories "
            f"{batch_start + 1}-{batch_start + len(batch_trajectories)}/{len(trajectories)}",
            flush=True,
        )
        buffers = build_pointnet_rollout_buffers(
            device=str(torch_device),
            batch_capacity=max(len(batch_trajectories), 1),
            step_capacity=int(eval_args.steps),
            point_count=len(diff_scene.local_surface_points_np),
            full_point_friction=checkpoint.full_point_friction,
        )
        predicted, diagnostics = run_conditional_direct_state_rollout_batch(
            diff_scene=diff_scene,
            buffers=buffers,
            trajectories=batch_trajectories,
            args=eval_args,
            initial_body_q=initial_body_q,
            initial_body_qd=initial_body_qd,
            checkpoint=checkpoint,
        )
        hidden_norms.extend(np.asarray(diagnostics["hidden_l2_norm"], dtype=np.float32))
        hidden_saturation.extend(np.asarray(diagnostics["hidden_saturation_fraction"], dtype=np.float32))
        for batch_idx, trajectory in enumerate(batch_trajectories):
            single_predicted = type(predicted)(
                positions=predicted.positions[batch_idx : batch_idx + 1],
                quaternions_xyzw=predicted.quaternions_xyzw[batch_idx : batch_idx + 1],
                linear_velocity=predicted.linear_velocity[batch_idx : batch_idx + 1],
                angular_velocity=predicted.angular_velocity[batch_idx : batch_idx + 1],
            )
            positions = np.asarray(single_predicted.positions[0, : trajectory.num_frames], dtype=np.float32)
            all_positions.append(positions)
            all_state_histories.append(
                {
                    "positions": np.asarray(single_predicted.positions[0, : trajectory.num_frames], dtype=np.float32),
                    "quaternions_xyzw": np.asarray(
                        single_predicted.quaternions_xyzw[0, : trajectory.num_frames],
                        dtype=np.float32,
                    ),
                    "linear_velocity": np.asarray(
                        single_predicted.linear_velocity[0, : trajectory.num_frames],
                        dtype=np.float32,
                    ),
                    "angular_velocity": np.asarray(
                        single_predicted.angular_velocity[0, : trajectory.num_frames],
                        dtype=np.float32,
                    ),
                }
            )
            all_losses.append(_pointnet_overlay_loss(single_predicted, trajectory, eval_args))
    rollout_diagnostics = {
        "stateful_reset_interval": 0,
        "hidden_l2_norm_mean": float(np.mean([np.mean(values) for values in hidden_norms])),
        "hidden_l2_norm_max": float(np.max([np.max(values) for values in hidden_norms])),
        "hidden_saturation_fraction_mean": float(np.mean([np.mean(values) for values in hidden_saturation])),
        "hidden_saturation_fraction_max": float(np.max([np.max(values) for values in hidden_saturation])),
    }
    return all_positions, all_losses, rollout_diagnostics, all_state_histories


def html_escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def css_color(value) -> str:
    if isinstance(value, str):
        return value
    r, g, b = value[:3]
    return f"rgb({int(round(r * 255))}, {int(round(g * 255))}, {int(round(b * 255))})"


def coerce_xy(points: np.ndarray) -> np.ndarray:
    xy = np.asarray(points, dtype=np.float32)
    if xy.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    if xy.ndim == 1:
        if xy.shape[0] < 2:
            return np.empty((0, 2), dtype=np.float32)
        xy = xy.reshape(1, -1)
    if xy.shape[-1] < 2:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(xy[:, :2], dtype=np.float32)


def finite_xy(points: np.ndarray) -> np.ndarray:
    xy = coerce_xy(points)
    if len(xy) == 0:
        return xy
    return xy[np.isfinite(xy).all(axis=1)]


def first_finite_xy(points: np.ndarray) -> np.ndarray | None:
    xy = finite_xy(points)
    if len(xy) == 0:
        return None
    return xy[0]


def last_finite_xy(points: np.ndarray) -> np.ndarray | None:
    xy = finite_xy(points)
    if len(xy) == 0:
        return None
    return xy[-1]


def path_points(points: np.ndarray, x_min: float, y_min: float, scale: float, plot_height: int, pad: int) -> str:
    xy = coerce_xy(points)
    if len(xy) == 0 or not np.isfinite([x_min, y_min, scale]).all():
        return ""
    commands = []
    in_segment = False
    for x, y in xy:
        if not np.isfinite([x, y]).all():
            in_segment = False
            continue
        sx = pad + (float(x) - x_min) * scale
        sy = pad + plot_height - (float(y) - y_min) * scale
        if not np.isfinite([sx, sy]).all():
            in_segment = False
            continue
        command = "L" if in_segment else "M"
        commands.append(f"{command} {sx:.2f},{sy:.2f}")
        in_segment = True
    return " ".join(commands)


def marker_xy(
    point: np.ndarray | None,
    x_min: float,
    y_min: float,
    scale: float,
    plot_height: int,
    pad: int,
) -> tuple[float, float] | None:
    if point is None or not np.isfinite([x_min, y_min, scale]).all():
        return None
    xy = coerce_xy(point)
    if len(xy) == 0 or not np.isfinite(xy[0]).all():
        return None
    sx = pad + (float(xy[0, 0]) - x_min) * scale
    sy = pad + plot_height - (float(xy[0, 1]) - y_min) * scale
    if not np.isfinite([sx, sy]).all():
        return None
    return sx, sy


def axis_bounds(all_xy: list[np.ndarray], padding_frac: float) -> tuple[float, float, float, float]:
    finite_arrays = [finite_xy(xy) for xy in all_xy]
    finite_arrays = [xy for xy in finite_arrays if len(xy) > 0]
    if not finite_arrays:
        return -0.5, 0.5, -0.5, 0.5
    stacked = np.concatenate(finite_arrays, axis=0)
    x_min = float(np.min(stacked[:, 0]))
    x_max = float(np.max(stacked[:, 0]))
    y_min = float(np.min(stacked[:, 1]))
    y_max = float(np.max(stacked[:, 1]))
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    radius = 0.5 * max(x_max - x_min, y_max - y_min, 1.0e-6)
    pad = max(radius * float(padding_frac), 0.002)
    return cx - radius - pad, cx + radius + pad, cy - radius - pad, cy + radius + pad


def axis_tick_values(v_min: float, v_max: float, count: int = 3) -> list[float]:
    return [float(value) for value in np.linspace(float(v_min), float(v_max), int(count))]


def format_axis_tick(value: float, span: float) -> str:
    value = 0.0 if abs(float(value)) < 5.0e-8 else float(value)
    span = abs(float(span))
    if span < 0.02:
        return f"{value:.4f}"
    if span < 0.2:
        return f"{value:.3f}"
    if span < 2.0:
        return f"{value:.2f}"
    return f"{value:.1f}"


def format_loss_value(value: object) -> str:
    try:
        loss = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(loss):
        return "non-finite"
    return f"{loss:.6g}"


def mean_finite_loss(values: list[float]) -> float | None:
    if not values:
        return None
    losses = np.asarray(values, dtype=np.float64)
    losses = losses[np.isfinite(losses)]
    if len(losses) == 0:
        return None
    return float(np.mean(losses))


def load_cached_payload(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "interactive_data" not in payload:
        raise ValueError(f"{path} does not contain interactive_data")
    return payload


def collect_rollout_payload(args: argparse.Namespace) -> dict:
    methods = select_overlay_methods(args)
    for method in methods:
        if not method.checkpoint.exists():
            raise FileNotFoundError(method.checkpoint)
        if (
            not checkpoint_has_active_params(method.checkpoint)
            and not checkpoint_is_residual_adapter(method.checkpoint)
            and not checkpoint_is_pointnet_adapter(method.checkpoint, include_last=True)
            and not checkpoint_is_conditional_direct_state(method.checkpoint, include_last=True)
        ):
            raise ValueError(f"{method.checkpoint} is not a training checkpoint with active friction parameters")

    wp.init()
    eval_args = make_eval_args(args)
    eval_args.residual_l2_weight = 1.0e-4
    eval_args.residual_smoothness_weight = 1.0e-4
    eval_args.residual_output_mode = "acceleration"
    eval_args.velocity_scale = None
    eval_args.angular_velocity_scale = None
    eval_args.acceleration_scale = 2.0
    eval_args.angular_acceleration_scale = 20.0
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
    sim_states = [diff_scene.model.state() for _ in range(max(int(eval_args.steps), 1))]
    initial_body_q = diff_scene.states[0].body_q.numpy().copy()
    initial_body_qd = diff_scene.states[0].body_qd.numpy().copy()
    checkpoint_params: dict[
        str,
        CheckpointParams | ResidualCheckpointParams | PointNetCheckpointParams | LoadedConditionalDirectStateCheckpoint,
    ] = {}
    for method in methods:
        if checkpoint_is_residual_adapter(method.checkpoint):
            checkpoint_params[method.name] = load_residual_checkpoint_params(method.checkpoint)
        elif checkpoint_is_pointnet_adapter(method.checkpoint, include_last=True):
            checkpoint_params[method.name] = load_pointnet_checkpoint_params(method.checkpoint)
        elif checkpoint_is_conditional_direct_state(method.checkpoint, include_last=True):
            checkpoint_params[method.name] = load_conditional_direct_state_checkpoint(method.checkpoint, map_location="cpu")
        else:
            checkpoint_params[method.name] = load_checkpoint_params(method.checkpoint)
    legend_labels = {
        method.name: (
            residual_legend_label(method, checkpoint_params[method.name])
            if isinstance(checkpoint_params[method.name], ResidualCheckpointParams)
            else pointnet_legend_label(method, checkpoint_params[method.name])
            if isinstance(checkpoint_params[method.name], PointNetCheckpointParams)
            else direct_state_legend_label(method, checkpoint_params[method.name])
            if isinstance(checkpoint_params[method.name], LoadedConditionalDirectStateCheckpoint)
            else checkpoint_legend_label(method, checkpoint_params[method.name])
        )
        for method in methods
    }

    target_tracks = []
    target_state_rollouts: list[dict[str, np.ndarray]] = []
    for selected_idx, trajectory in zip(selected_indices, selected_trajectories):
        point = np.asarray(trajectory.force_point_offset_local, dtype=np.float32)
        force = np.asarray(trajectory.step_forces[0], dtype=np.float32)
        target_tracks.append(
            {
                "trajectory_index": int(selected_idx),
                "episode_index": int(trajectory.metadata.get("episode_index", selected_idx)),
                "point": [float(point[0]), float(point[1]), float(point[2])],
                "force": [float(force[0]), float(force[1]), float(force[2])],
                "xy": np.asarray(trajectory.positions[:, :2], dtype=np.float32).tolist(),
            }
        )
        target_state_rollouts.append(
            {
                "timestamps": np.asarray(trajectory.time, dtype=np.float32),
                "positions": np.asarray(trajectory.positions, dtype=np.float32),
                "quaternions_xyzw": np.asarray(trajectory.quaternions_xyzw, dtype=np.float32),
                "linear_velocity": np.asarray(trajectory.linear_velocity, dtype=np.float32),
                "angular_velocity": np.asarray(trajectory.angular_velocity, dtype=np.float32),
            }
        )
    reference_tracks = collect_reference_payload(args, selected_indices)

    method_tracks = {}
    method_losses = {}
    method_state_rollouts: dict[str, list[dict[str, np.ndarray]]] = {}
    method_summaries = []
    for method_idx, method in enumerate(methods):
        checkpoint = checkpoint_params[method.name]
        if isinstance(checkpoint, ResidualCheckpointParams):
            print(
                f"rolling out {method_idx + 1}/{len(methods)} {method.name} residual_adapter "
                f"active={len(checkpoint.active_indices)} param={checkpoint.parameterization}",
                flush=True,
            )
            positions, losses, state_histories = residual_rollout_positions_for_trajectories(
                diff_scene=diff_scene,
                sim_states=sim_states,
                trajectories=selected_trajectories,
                eval_args=eval_args,
                checkpoint_path=method.checkpoint,
                checkpoint=checkpoint,
                initial_body_q=initial_body_q,
                initial_body_qd=initial_body_qd,
            )
            summary = residual_checkpoint_summary(method, checkpoint, losses)
        elif isinstance(checkpoint, PointNetCheckpointParams):
            _, architecture_label = neural_adapter_architecture_label(checkpoint.metadata)
            print(
                f"rolling out {method_idx + 1}/{len(methods)} {method.name} {architecture_label.lower()}_adapter "
                f"active={int(np.count_nonzero(checkpoint.active_contact_mask))} "
                f"dino_dim={int(checkpoint.metadata.get('dino_feature_dim', 0))}",
                flush=True,
            )
            positions, losses, rollout_diagnostics, state_histories = pointnet_rollout_positions_for_trajectories(
                diff_scene=diff_scene,
                trajectories=selected_trajectories,
                eval_args=eval_args,
                checkpoint_path=method.checkpoint,
                initial_body_q=initial_body_q,
                initial_body_qd=initial_body_qd,
            )
            summary = pointnet_checkpoint_summary(method, checkpoint, losses)
            summary["rollout_residual_output_mode"] = str(eval_args.pointnet_residual_output_mode)
            summary["rollout_residual_gain"] = float(
                checkpoint.metadata.get("pointnet_residual_gain", 1.0)
                if eval_args.pointnet_residual_gain is None
                else eval_args.pointnet_residual_gain
            )
            summary["rollout_stateful_diagnostics"] = rollout_diagnostics
        elif isinstance(checkpoint, LoadedConditionalDirectStateCheckpoint):
            print(
                f"rolling out {method_idx + 1}/{len(methods)} {method.name} direct_state_adapter "
                f"output_mode={checkpoint.metadata.get('output_mode')} "
                f"active={int(np.count_nonzero(checkpoint.active_contact_mask))}",
                flush=True,
            )
            positions, losses, rollout_diagnostics, state_histories = direct_state_rollout_positions_for_trajectories(
                diff_scene=diff_scene,
                trajectories=selected_trajectories,
                eval_args=eval_args,
                checkpoint_path=method.checkpoint,
                initial_body_q=initial_body_q,
                initial_body_qd=initial_body_qd,
            )
            summary = direct_state_checkpoint_summary(method, checkpoint, losses)
            summary["rollout_stateful_diagnostics"] = rollout_diagnostics
        else:
            print(
                f"rolling out {method_idx + 1}/{len(methods)} {method.name} "
                f"active={len(checkpoint.active_indices)} param={checkpoint.parameterization}",
                flush=True,
            )
            positions, losses, state_histories = rollout_positions_for_trajectories(
                diff_scene=diff_scene,
                trajectories=selected_trajectories,
                eval_args=eval_args,
                active_indices=checkpoint.active_indices,
                active_params=checkpoint.active_params,
                initial_body_q=initial_body_q,
                initial_body_qd=initial_body_qd,
                return_state_histories=True,
            )
            summary = checkpoint_summary(method, checkpoint, losses)
        method_tracks[method.name] = [
            np.asarray(position[: len(target_tracks[idx]["xy"]), :2], dtype=np.float32).tolist()
            for idx, position in enumerate(positions)
        ]
        method_losses[method.name] = [float(loss) for loss in losses]
        method_state_rollouts[method.name] = state_histories
        summary["legend_label"] = legend_labels[method.name]
        summary["color"] = css_color(method.color)
        method_summaries.append(summary)

    return {
        "dataset": str(args.dataset),
        "max_steps": args.max_steps,
        "selected_trajectories": selected_indices,
        "eval_batch_size": eval_args.eval_batch_size,
        "contact_stiffness": args.contact_stiffness,
        "contact_damping": args.contact_damping,
        "surface_point_spacing": args.surface_point_spacing,
        "friction_contact_threshold": args.friction_contact_threshold,
        "contact_mask_threshold": args.contact_mask_threshold,
        "loss_weights": {
            "position": float(args.position_loss_weight),
            "orientation": float(args.orientation_loss_weight),
            "linear_velocity": float(args.linear_velocity_loss_weight),
            "angular_velocity": float(args.angular_velocity_loss_weight),
        },
        "pointnet_residual_gain": None if args.pointnet_residual_gain is None else float(args.pointnet_residual_gain),
        "pointnet_residual_output_mode": str(args.pointnet_residual_output_mode),
        "stateful_reset_interval": getattr(args, "stateful_reset_interval", None),
        "methods": method_summaries,
        "reference_datasets": [str(path) for path in (args.reference_dataset or [])],
        "trajectory_losses": {
            method.name: {
                str(selected_indices[idx]): float(loss)
                for idx, loss in enumerate(method_losses[method.name])
            }
            for method in methods
        },
        "interactive_data": {
            "targets": target_tracks,
            "references": reference_tracks,
            "methods": [
                {
                    "name": method.name,
                    "label": legend_labels[method.name],
                    "color": css_color(method.color),
                    "stage": method.stage,
                    "checkpoint": str(method.checkpoint),
                    "losses": method_losses[method.name],
                    "tracks": method_tracks[method.name],
                }
                for method in methods
            ],
        },
        "_target_state_rollouts": target_state_rollouts,
        "_method_state_rollouts": method_state_rollouts,
    }


def render_html(payload: dict, args: argparse.Namespace) -> str:
    targets = payload["interactive_data"]["targets"]
    methods = payload["interactive_data"]["methods"]
    references = payload["interactive_data"].get("references", [])
    plot_width = int(args.plot_width)
    plot_height = int(args.plot_height)
    pad = 54
    panel_width = plot_width + pad * 2
    panel_height = plot_height + pad * 2 + 38
    cols = 5 if len(targets) > 12 else 3
    rows = int(np.ceil(len(targets) / cols))
    svg_width = cols * panel_width + int(args.legend_width)
    svg_height = max(rows * panel_height + 34, 820)
    legend_x = cols * panel_width + 24
    legend_y = 52
    legend_row_h = 26

    method_ids = {method["name"]: f"m{idx}" for idx, method in enumerate(methods)}
    reference_ids = {reference["name"]: f"r{idx}" for idx, reference in enumerate(references)}
    overlay_count = len(methods) + len(references)
    overlay_text = f"{len(methods)} checkpoints"
    if references:
        overlay_text += f" + {len(references)} references"
    if args.unified_axis_scale:
        overlay_text += " | shared axes"
    parts: list[str] = []
    title = (
        f"Top-down trajectory overlays | {Path(payload['dataset']).stem} | "
        f"{overlay_text} | {'all steps' if payload['max_steps'] is None else 'max_steps=' + str(payload['max_steps'])}"
    )
    loss_weights = payload.get("loss_weights")
    if loss_weights:
        title += (
            f" | loss w: pos={float(loss_weights.get('position', 0.0)):.3g}, "
            f"rot={float(loss_weights.get('orientation', 0.0)):.3g}"
        )

    parts.append("<!doctype html>")
    parts.append("<html><head><meta charset=\"utf-8\">")
    parts.append(f"<title>{html_escape(title)}</title>")
    parts.append(
        """
<style>
:root { color-scheme: light; }
body {
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: #1f2933;
  background: #f7f8fa;
}
.page { padding: 18px 22px 28px; }
h1 { font-size: 18px; margin: 0 0 4px; font-weight: 650; }
.meta { font-size: 12px; color: #5b6673; margin-bottom: 14px; }
.frame {
  background: white;
  border: 1px solid #d8dee8;
  border-radius: 8px;
  overflow: auto;
  box-shadow: 0 1px 2px rgba(16, 24, 40, 0.06);
}
svg { display: block; background: white; }
.panel-bg { fill: #ffffff; stroke: #dfe5ee; stroke-width: 1; rx: 6; }
.plot-bg { fill: #fbfcfe; stroke: #d8dee8; stroke-width: 1; }
.grid { stroke: #edf1f7; stroke-width: 1; }
.tick { stroke: #98a2b3; stroke-width: 1; }
.tick-label { fill: #667085; font-size: 8px; }
.axis-label, .panel-title { fill: #344054; font-size: 10px; }
.panel-title { font-size: 11px; font-weight: 600; }
.target-line { fill: none; stroke: #101828; stroke-width: 2.4; opacity: 0.9; pointer-events: none; }
.target-marker { fill: #101828; stroke: #101828; pointer-events: none; }
.track-hit {
  fill: none;
  stroke: rgba(0, 0, 0, 0.001);
  stroke-width: 15;
  stroke-linecap: round;
  stroke-linejoin: round;
  pointer-events: stroke;
  cursor: crosshair;
}
.track-line {
  fill: none;
  stroke-width: 1.45;
  opacity: 0.42;
  pointer-events: none;
  transition: opacity 120ms ease, stroke-width 120ms ease;
}
.reference-track.track-line {
  stroke-width: 2.0;
  stroke-dasharray: 5 3;
  opacity: 0.62;
}
.reference-track.track-end {
  opacity: 0.72;
}
.track-end {
  opacity: 0.55;
  pointer-events: none;
  transition: opacity 120ms ease, r 120ms ease;
}
.legend-title { fill: #111827; font-size: 12px; font-weight: 700; }
.legend-item {
  cursor: pointer;
  opacity: 0.76;
  transition: opacity 120ms ease;
  pointer-events: all;
}
.legend-item rect { fill: rgba(255, 255, 255, 0.001); stroke: transparent; }
.legend-label { fill: #344054; font-size: 9.5px; dominant-baseline: middle; }
.legend-swatch { stroke-width: 2.4; }
.legend-icon { stroke: #ffffff; stroke-width: 1.6; }
.legend-icon-text {
  fill: #ffffff;
  font-size: 8px;
  font-weight: 700;
  dominant-baseline: central;
  text-anchor: middle;
  pointer-events: none;
}
.dimmed .track-line { opacity: 0.08; }
.dimmed .track-end { opacity: 0.08; }
.dimmed .legend-item { opacity: 0.26; }
.active-method.track-line { opacity: 0.98; stroke-width: 3.4; }
.active-method.track-end { opacity: 1; r: 3.8; }
.active-method.legend-item { opacity: 1; }
.active-method .legend-label { fill: #111827; font-weight: 700; }
.active-method .legend-box { fill: rgba(37, 99, 235, 0.08); stroke: rgba(37, 99, 235, 0.22); }
.active-method .legend-icon { stroke: #111827; stroke-width: 2.2; }
.active-track.track-line { stroke-width: 4.6; }
.active-track.track-end { r: 4.4; }
.pinned-method.legend-item .legend-box { fill: rgba(16, 24, 40, 0.06); stroke: rgba(16, 24, 40, 0.28); }
.tooltip {
  position: fixed;
  z-index: 10;
  max-width: 460px;
  padding: 8px 10px;
  border: 1px solid #ccd4df;
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.97);
  color: #1f2933;
  font-size: 12px;
  line-height: 1.35;
  box-shadow: 0 8px 24px rgba(16, 24, 40, 0.14);
  display: none;
  pointer-events: none;
}
.toolbar {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 0 0 10px;
  font-size: 12px;
  color: #475467;
}
.toolbar button {
  border: 1px solid #cfd7e3;
  background: #fff;
  border-radius: 6px;
  padding: 5px 9px;
  cursor: pointer;
  font: inherit;
  color: #1f2933;
}
.toolbar button:hover { background: #f4f7fb; }
</style>
"""
    )
    parts.append("</head><body><div class=\"page\">")
    parts.append(f"<h1>{html_escape(title)}</h1>")
    parts.append(
        "<div class=\"meta\">Hover a legend row to isolate a checkpoint. "
        "Hover a trajectory line to isolate its overlay and emphasize that specific trajectory. "
        "Click legend rows or trajectory lines to toggle multiple persistent highlights. "
        "The black curve is the primary dataset ground truth.</div>"
    )
    parts.append("<div class=\"toolbar\"><button type=\"button\" id=\"resetBtn\">Reset highlights</button>")
    parts.append(f"<span>{len(targets)} trajectories, {overlay_count} overlays</span></div>")
    parts.append("<div class=\"frame\">")
    parts.append(f"<svg id=\"overlaySvg\" width=\"{svg_width}\" height=\"{svg_height}\" viewBox=\"0 0 {svg_width} {svg_height}\">")
    parts.append(f"<text x=\"16\" y=\"24\" class=\"legend-title\">{html_escape(title)}</text>")

    global_axis_bounds = None
    if args.unified_axis_scale:
        global_xy = []
        for target_idx, target in enumerate(targets):
            global_xy.append(np.asarray(target["xy"], dtype=np.float32))
            for reference in references:
                global_xy.append(np.asarray(reference["tracks"][target_idx], dtype=np.float32))
            for method in methods:
                global_xy.append(np.asarray(method["tracks"][target_idx], dtype=np.float32))
        global_axis_bounds = axis_bounds(global_xy, args.axis_padding_frac)

    for target_idx, target in enumerate(targets):
        col = target_idx % cols
        row = target_idx // cols
        ox = col * panel_width + 12
        oy = row * panel_height + 38
        all_xy = [np.asarray(target["xy"], dtype=np.float32)]
        for reference in references:
            all_xy.append(np.asarray(reference["tracks"][target_idx], dtype=np.float32))
        for method in methods:
            all_xy.append(np.asarray(method["tracks"][target_idx], dtype=np.float32))
        if global_axis_bounds is None:
            x_min, x_max, y_min, y_max = axis_bounds(all_xy, args.axis_padding_frac)
        else:
            x_min, x_max, y_min, y_max = global_axis_bounds
        scale = min(plot_width / (x_max - x_min), plot_height / (y_max - y_min))
        plot_x = ox + pad
        plot_y = oy + pad
        parts.append(f"<g class=\"panel\" data-traj=\"{target_idx}\">")
        parts.append(f"<rect class=\"panel-bg\" x=\"{ox}\" y=\"{oy}\" width=\"{panel_width - 10}\" height=\"{panel_height - 8}\"/>")
        title_text = (
            f"traj {target['trajectory_index']} | point x={target['point'][0]:.3f}, y={target['point'][1]:.3f} | "
            f"force=({target['force'][0]:.2f},{target['force'][1]:.2f})"
        )
        parts.append(f"<text class=\"panel-title\" x=\"{ox + 12}\" y=\"{oy + 17}\">{html_escape(title_text)}</text>")
        parts.append(f"<rect class=\"plot-bg\" x=\"{plot_x}\" y=\"{plot_y}\" width=\"{plot_width}\" height=\"{plot_height}\"/>")
        for grid_idx in range(1, 4):
            gx = plot_x + grid_idx * plot_width / 4.0
            gy = plot_y + grid_idx * plot_height / 4.0
            parts.append(f"<line class=\"grid\" x1=\"{gx:.2f}\" y1=\"{plot_y}\" x2=\"{gx:.2f}\" y2=\"{plot_y + plot_height}\"/>")
            parts.append(f"<line class=\"grid\" x1=\"{plot_x}\" y1=\"{gy:.2f}\" x2=\"{plot_x + plot_width}\" y2=\"{gy:.2f}\"/>")
        x_span = x_max - x_min
        y_span = y_max - y_min
        for tick_value in axis_tick_values(x_min, x_max):
            tx = plot_x + (tick_value - x_min) * scale
            label = format_axis_tick(tick_value, x_span)
            parts.append(
                f"<line class=\"tick\" x1=\"{tx:.2f}\" y1=\"{plot_y + plot_height:.2f}\" "
                f"x2=\"{tx:.2f}\" y2=\"{plot_y + plot_height + 4:.2f}\"/>"
            )
            parts.append(
                f"<text class=\"tick-label\" x=\"{tx:.2f}\" y=\"{plot_y + plot_height + 13:.2f}\" "
                f"text-anchor=\"middle\">{html_escape(label)}</text>"
            )
        for tick_value in axis_tick_values(y_min, y_max):
            ty = plot_y + plot_height - (tick_value - y_min) * scale
            label = format_axis_tick(tick_value, y_span)
            parts.append(
                f"<line class=\"tick\" x1=\"{plot_x - 4:.2f}\" y1=\"{ty:.2f}\" "
                f"x2=\"{plot_x:.2f}\" y2=\"{ty:.2f}\"/>"
            )
            parts.append(
                f"<text class=\"tick-label\" x=\"{plot_x - 7:.2f}\" y=\"{ty + 3:.2f}\" "
                f"text-anchor=\"end\">{html_escape(label)}</text>"
            )
        for reference_idx, reference in enumerate(references):
            reference_id = reference_ids[reference["name"]]
            track_xy = np.asarray(reference["tracks"][target_idx], dtype=np.float32)
            track_path = path_points(track_xy, x_min, y_min, scale, plot_height, pad)
            finite_count = len(finite_xy(track_xy))
            tooltip = (
                f"{reference['label']}<br>"
                f"trajectory {target['trajectory_index']}<br>"
                f"finite points={finite_count}/{len(track_xy)}<br>"
                f"{reference['dataset']}"
            )
            if track_path:
                parts.append(
                    f"<path class=\"track-line reference-track method-{reference_id}\" data-method=\"{reference_id}\" "
                    f"data-method-name=\"{html_escape(reference['name'])}\" data-traj=\"{target_idx}\" "
                    f"stroke=\"{html_escape(reference['color'])}\" d=\"{track_path}\" transform=\"translate({ox}, {oy})\"/>"
                )
                parts.append(
                    f"<path class=\"track-hit\" data-method=\"{reference_id}\" data-method-name=\"{html_escape(reference['name'])}\" "
                    f"data-traj=\"{target_idx}\" data-tooltip=\"{html_escape(tooltip)}\" "
                    f"d=\"{track_path}\" transform=\"translate({ox}, {oy})\"/>"
                )
            marker = marker_xy(last_finite_xy(track_xy), x_min, y_min, scale, plot_height, pad)
            if marker is not None:
                ex, ey = marker
                parts.append(
                    f"<circle class=\"track-end reference-track method-{reference_id}\" data-method=\"{reference_id}\" "
                    f"data-traj=\"{target_idx}\" cx=\"{ox + ex:.2f}\" cy=\"{oy + ey:.2f}\" r=\"2.5\" "
                    f"fill=\"{html_escape(reference['color'])}\"/>"
                )
        for method_idx, method in enumerate(methods):
            method_id = method_ids[method["name"]]
            track_xy = np.asarray(method["tracks"][target_idx], dtype=np.float32)
            track_path = path_points(track_xy, x_min, y_min, scale, plot_height, pad)
            finite_count = len(finite_xy(track_xy))
            loss_text = format_loss_value(method["losses"][target_idx])
            tooltip = (
                f"{method['label']}<br>"
                f"trajectory {target['trajectory_index']} loss={loss_text}<br>"
                f"finite points={finite_count}/{len(track_xy)}<br>"
                f"{method['checkpoint']}"
            )
            if track_path:
                parts.append(
                    f"<path class=\"track-line method-{method_id}\" data-method=\"{method_id}\" "
                    f"data-method-name=\"{html_escape(method['name'])}\" data-traj=\"{target_idx}\" "
                    f"stroke=\"{html_escape(method['color'])}\" d=\"{track_path}\" transform=\"translate({ox}, {oy})\"/>"
                )
                parts.append(
                    f"<path class=\"track-hit\" data-method=\"{method_id}\" data-method-name=\"{html_escape(method['name'])}\" "
                    f"data-traj=\"{target_idx}\" data-tooltip=\"{html_escape(tooltip)}\" "
                    f"d=\"{track_path}\" transform=\"translate({ox}, {oy})\"/>"
                )
            marker = marker_xy(last_finite_xy(track_xy), x_min, y_min, scale, plot_height, pad)
            if marker is not None:
                ex, ey = marker
                parts.append(
                    f"<circle class=\"track-end method-{method_id}\" data-method=\"{method_id}\" data-traj=\"{target_idx}\" "
                    f"cx=\"{ox + ex:.2f}\" cy=\"{oy + ey:.2f}\" r=\"2.3\" fill=\"{html_escape(method['color'])}\"/>"
                )
        target_xy = np.asarray(target["xy"], dtype=np.float32)
        target_path = path_points(target_xy, x_min, y_min, scale, plot_height, pad)
        if target_path:
            parts.append(f"<path class=\"target-line\" d=\"{target_path}\" transform=\"translate({ox}, {oy})\"/>")
        start_marker = marker_xy(first_finite_xy(target_xy), x_min, y_min, scale, plot_height, pad)
        if start_marker is not None:
            start_x, start_y = start_marker
            parts.append(f"<circle class=\"target-marker\" cx=\"{ox + start_x:.2f}\" cy=\"{oy + start_y:.2f}\" r=\"2.8\"/>")
        end_marker = marker_xy(last_finite_xy(target_xy), x_min, y_min, scale, plot_height, pad)
        if end_marker is not None:
            end_x, end_y = end_marker
            parts.append(
                f"<path class=\"target-marker\" d=\"M {ox + end_x - 4:.2f},{oy + end_y - 4:.2f} "
                f"L {ox + end_x + 4:.2f},{oy + end_y + 4:.2f} "
                f"M {ox + end_x + 4:.2f},{oy + end_y - 4:.2f} "
                f"L {ox + end_x - 4:.2f},{oy + end_y + 4:.2f}\" stroke-width=\"1.8\"/>"
            )
        parts.append(f"<text class=\"axis-label\" x=\"{plot_x + plot_width - 8}\" y=\"{plot_y + plot_height + 29}\" text-anchor=\"end\">x</text>")
        parts.append(f"<text class=\"axis-label\" x=\"{plot_x - 18}\" y=\"{plot_y + 10}\" text-anchor=\"middle\">y</text>")
        parts.append("</g>")

    parts.append(f"<g class=\"legend\" transform=\"translate({legend_x}, {legend_y})\">")
    parts.append("<text class=\"legend-title\" x=\"0\" y=\"0\">Overlays</text>")
    parts.append("<g transform=\"translate(0, 18)\">")
    parts.append("<line class=\"legend-swatch\" x1=\"0\" y1=\"0\" x2=\"26\" y2=\"0\" stroke=\"#101828\"/>")
    parts.append("<text class=\"legend-label\" x=\"34\" y=\"0\">Primary dataset ground truth</text>")
    parts.append("</g>")
    for idx, reference in enumerate(references):
        reference_id = reference_ids[reference["name"]]
        y = 44 + idx * legend_row_h
        tooltip = f"{reference['label']}<br>{reference['dataset']}"
        parts.append(
            f"<g class=\"legend-item method-{reference_id}\" data-method=\"{reference_id}\" "
            f"data-tooltip=\"{html_escape(tooltip)}\" transform=\"translate(0, {y})\">"
        )
        parts.append(f"<rect class=\"legend-box\" x=\"-8\" y=\"-10\" width=\"{args.legend_width - 36}\" height=\"21\" rx=\"4\"/>")
        parts.append(
            f"<line class=\"legend-swatch\" x1=\"0\" y1=\"0\" x2=\"26\" y2=\"0\" "
            f"stroke=\"{html_escape(reference['color'])}\" stroke-dasharray=\"5 3\"/>"
        )
        parts.append(
            f"<circle class=\"legend-icon\" cx=\"39\" cy=\"0\" r=\"7\" fill=\"{html_escape(reference['color'])}\"/>"
        )
        parts.append(f"<text class=\"legend-icon-text\" x=\"39\" y=\"0\">R</text>")
        parts.append(f"<text class=\"legend-label\" x=\"52\" y=\"0\">{html_escape(reference['label'])}</text>")
        parts.append("</g>")
    method_legend_start = 44 + len(references) * legend_row_h
    for idx, method in enumerate(methods):
        method_id = method_ids[method["name"]]
        y = method_legend_start + idx * legend_row_h
        label = method["label"]
        avg_loss = mean_finite_loss(method["losses"])
        avg_loss_text = "non-finite" if avg_loss is None else f"{avg_loss:.6g}"
        tooltip = f"{label}<br>finite mean loss={avg_loss_text}<br>{method['checkpoint']}"
        parts.append(
            f"<g class=\"legend-item method-{method_id}\" data-method=\"{method_id}\" "
            f"data-tooltip=\"{html_escape(tooltip)}\" transform=\"translate(0, {y})\">"
        )
        parts.append(f"<rect class=\"legend-box\" x=\"-8\" y=\"-10\" width=\"{args.legend_width - 36}\" height=\"21\" rx=\"4\"/>")
        parts.append(f"<line class=\"legend-swatch\" x1=\"0\" y1=\"0\" x2=\"26\" y2=\"0\" stroke=\"{html_escape(method['color'])}\"/>")
        parts.append(f"<circle class=\"legend-icon\" cx=\"39\" cy=\"0\" r=\"7\" fill=\"{html_escape(method['color'])}\"/>")
        parts.append(f"<text class=\"legend-icon-text\" x=\"39\" y=\"0\">{idx + 1}</text>")
        parts.append(f"<text class=\"legend-label\" x=\"52\" y=\"0\">{html_escape(label)} | finite mean loss={avg_loss_text}</text>")
        parts.append("</g>")
    parts.append("</g>")
    parts.append("</svg></div><div id=\"tooltip\" class=\"tooltip\"></div>")
    parts.append(
        """
<script>
const svg = document.getElementById('overlaySvg');
const tooltip = document.getElementById('tooltip');
const pinnedMethods = new Set();
const pinnedTracks = new Map();

function showTooltip(evt, html) {
  if (!html) return;
  tooltip.innerHTML = html;
  tooltip.style.display = 'block';
  moveTooltip(evt);
}

function moveTooltip(evt) {
  if (tooltip.style.display !== 'block') return;
  tooltip.style.left = `${evt.clientX + 14}px`;
  tooltip.style.top = `${evt.clientY + 14}px`;
}

function hideTooltip() {
  tooltip.style.display = 'none';
}

function clearHighlightClasses() {
  document.querySelectorAll('.active-method').forEach(el => el.classList.remove('active-method'));
  document.querySelectorAll('.active-track').forEach(el => el.classList.remove('active-track'));
  document.querySelectorAll('.pinned-method').forEach(el => el.classList.remove('pinned-method'));
}

function activateMethod(methodId, pinned = false) {
  document.querySelectorAll(`.method-${methodId}`).forEach(el => el.classList.add('active-method'));
  if (pinned) {
    document.querySelectorAll(`.method-${methodId}`).forEach(el => el.classList.add('pinned-method'));
  }
}

function activateTrack(methodId, trajectoryId, pinned = false) {
  document.querySelectorAll(`.legend-item.method-${methodId}`).forEach(el => {
    el.classList.add('active-method');
    if (pinned) el.classList.add('pinned-method');
  });
  document.querySelectorAll(`.track-line[data-method="${methodId}"][data-traj="${trajectoryId}"], .track-end[data-method="${methodId}"][data-traj="${trajectoryId}"]`).forEach(el => {
    el.classList.add('active-method');
    el.classList.add('active-track');
  });
}

function applyHighlights(hoverHighlight = null) {
  const hasPinnedTracks = pinnedTracks.size > 0;
  const hasPinnedMethods = pinnedMethods.size > 0;
  const hasAnyHighlight = hasPinnedMethods || hasPinnedTracks || Boolean(hoverHighlight);
  svg.classList.toggle('dimmed', hasAnyHighlight);
  clearHighlightClasses();
  pinnedMethods.forEach(methodId => activateMethod(methodId, true));
  pinnedTracks.forEach(track => {
    activateTrack(track.methodId, track.trajectoryId, true);
  });
  if (!hasPinnedMethods && !hasPinnedTracks && hoverHighlight) {
    if (hoverHighlight.trajectoryId === null) {
      activateMethod(hoverHighlight.methodId, false);
    } else {
      activateMethod(hoverHighlight.methodId, false);
      document.querySelectorAll(`.track-line[data-method="${hoverHighlight.methodId}"][data-traj="${hoverHighlight.trajectoryId}"], .track-end[data-method="${hoverHighlight.methodId}"][data-traj="${hoverHighlight.trajectoryId}"]`).forEach(el => {
        el.classList.add('active-track');
      });
    }
  }
}

function togglePinnedTrack(methodId, trajectoryId) {
  const key = `${methodId}:${trajectoryId}`;
  if (pinnedTracks.has(key)) {
    pinnedTracks.delete(key);
  } else {
    pinnedTracks.set(key, { methodId, trajectoryId });
  }
  applyHighlights();
}

function togglePinnedMethod(methodId) {
  if (pinnedMethods.has(methodId)) {
    pinnedMethods.delete(methodId);
  } else {
    pinnedMethods.add(methodId);
  }
  applyHighlights();
}

function clearPinnedHighlights() {
  pinnedMethods.clear();
  pinnedTracks.clear();
  applyHighlights();
}

function hasPinnedHighlight() {
  return pinnedMethods.size > 0 || pinnedTracks.size > 0;
}

function setHighlight(methodId, trajectoryId = null) {
  applyHighlights(methodId ? { methodId, trajectoryId } : null);
}

function clearHighlight() {
  applyHighlights();
  hideTooltip();
}

function resetPinnedHighlight() {
  clearPinnedHighlights();
  hideTooltip();
}

document.querySelectorAll('.legend-item').forEach(item => {
  item.addEventListener('mouseenter', evt => {
    if (!hasPinnedHighlight()) setHighlight(item.dataset.method, null);
    showTooltip(evt, item.dataset.tooltip);
  });
  item.addEventListener('mousemove', moveTooltip);
  item.addEventListener('mouseleave', clearHighlight);
  item.addEventListener('click', evt => {
    evt.stopPropagation();
    togglePinnedMethod(item.dataset.method);
    showTooltip(evt, item.dataset.tooltip);
  });
});

document.querySelectorAll('.track-hit').forEach(path => {
  path.addEventListener('mouseenter', evt => {
    if (!hasPinnedHighlight()) setHighlight(path.dataset.method, path.dataset.traj);
    showTooltip(evt, path.dataset.tooltip);
  });
  path.addEventListener('mousemove', moveTooltip);
  path.addEventListener('mouseleave', clearHighlight);
  path.addEventListener('click', evt => {
    evt.stopPropagation();
    togglePinnedTrack(path.dataset.method, path.dataset.traj);
    showTooltip(evt, path.dataset.tooltip);
  });
});

document.getElementById('resetBtn').addEventListener('click', evt => {
  evt.stopPropagation();
  resetPinnedHighlight();
});
document.addEventListener('click', resetPinnedHighlight);
document.addEventListener('keydown', evt => {
  if (evt.key === 'Escape') resetPinnedHighlight();
});
</script>
"""
    )
    parts.append("</div></body></html>")
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    if args.reuse_summary is not None:
        payload = load_cached_payload(args.reuse_summary)
    else:
        payload = collect_rollout_payload(args)

    html_text = render_html(payload, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_text, encoding="utf-8")

    summary_output = args.summary_output
    if summary_output is None:
        summary_output = args.output.with_name(f"{args.output.stem}_summary.json")
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_payload = dict(payload)
    summary_payload.pop("_target_state_rollouts", None)
    summary_payload.pop("_method_state_rollouts", None)
    summary_payload["output"] = str(args.output)
    summary_payload["unified_axis_scale"] = bool(args.unified_axis_scale)
    summary_output.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {summary_output}")


if __name__ == "__main__":
    main()
