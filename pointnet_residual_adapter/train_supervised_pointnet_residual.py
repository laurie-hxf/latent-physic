from __future__ import annotations

import argparse
import json
import sys
import time
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

from pointnet_residual_adapter.checkpoints import save_adapter_checkpoint, save_json  # noqa: E402
from pointnet_residual_adapter.dataset import sample_window_batch, split_trajectories  # noqa: E402
from pointnet_residual_adapter.features import (  # noqa: E402
    ACTION_FEATURE_SCHEMA,
    DinoFeatures,
    FeatureNormalizer,
    TorchFeatureNormalizer,
    apply_feature_normalizer_torch,
    build_supervised_batch_tensors_torch,
    load_aligned_dino_features,
    normalize_residual_output_mode,
    normalizer_to_torch,
    point_feature_schema,
)
from pointnet_residual_adapter.friction import (  # noqa: E402
    active_indices_from_trajectories,
    maybe_configure_scene_from_point_cloud,
    resolve_friction_conditioning,
)
from pointnet_residual_adapter.model import (  # noqa: E402
    PointNetResidualPredictor,
    ResidualLossWeights,
    residual_velocity_loss,
)
from pointnet_residual_adapter.newton_rollout import (  # noqa: E402
    build_rollout_buffers,
    run_open_loop_rollout,
)


DEFAULT_TRAIN_DATASET = (
    REPO_ROOT
    / "mujoco/outputs/rotation_friction_diagnostics_l0p20_r0p50_2000/"
    / "same_mean_split_left_0p20_right_0p50/same_mean_split_left_0p20_right_0p50.npz"
)
DEFAULT_DINO_FEATURE_NPZ = (
    REPO_ROOT
    / "outputs/mujoco_dino_point_features/block_force_surface_spacing_0p01_dinov2_layers/"
    / "frame_000000/newton_surface_points_dino_features.npz"
)
DEFAULT_FRICTION_CHECKPOINT = (
    REPO_ROOT
    / "outputs/20260531_053758_rotation_l0p20_r0p50_2000_dino_mlp_m300_posonly/"
    / "20260531_053758_rotation_l0p20_r0p50_2000_dino_mlp_m300_posonly.npz"
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--trajectory-npz", type=Path, default=DEFAULT_TRAIN_DATASET)
    parser.add_argument("--friction-checkpoint", type=Path, default=DEFAULT_FRICTION_CHECKPOINT)
    parser.add_argument("--friction-point-cloud", type=Path, default=None)
    parser.add_argument("--checkpoint-param-set", choices=("best", "current"), default="best")
    parser.add_argument("--dino-feature-npz", type=Path, default=DEFAULT_DINO_FEATURE_NPZ)
    parser.add_argument("--without-dino", action="store_true", help="Drop DINO features for the friction-only ablation.")
    parser.add_argument("--dino-max-match-distance", type=float, default=1.0e-5)
    parser.add_argument("--experiment-dir", type=Path, default=REPO_ROOT / "outputs/pointnet_residual/dino_mlp_h4_p1")
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--opt-iters", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--grad-clip-norm", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--random-time-windows", dest="random_time_windows", action="store_true", default=True)
    parser.add_argument("--no-random-time-windows", dest="random_time_windows", action="store_false")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional source trajectory truncation.")
    parser.add_argument("--time-window-source-max-steps", type=int, default=None)
    parser.add_argument("--history-window-steps", type=int, default=4)
    parser.add_argument("--prediction-window-steps", type=int, default=1)
    parser.add_argument("--normalization-batches", type=int, default=4)
    parser.add_argument("--val-every", type=int, default=25)
    parser.add_argument("--val-batches", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default="newton_friction_fitting")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default="pointnet-residual-adapter")
    parser.add_argument("--wandb-mode", type=str, default="online")
    parser.add_argument("--wandb-dir", type=Path, default=None)
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=None)
    parser.add_argument("--pointnet-feature-dim", type=int, default=256)
    parser.add_argument("--pointnet-action-context-dim", type=int, default=64)
    parser.add_argument("--pointnet-pooling", choices=("max", "mean-max"), default="mean-max")
    parser.add_argument("--linear-output-scale", type=float, default=None)
    parser.add_argument("--angular-output-scale", type=float, default=None)
    parser.add_argument("--loss-linear-velocity-weight", type=float, default=1.0)
    parser.add_argument("--loss-angular-velocity-weight", type=float, default=0.1)
    parser.add_argument(
        "--residual-output-mode",
        choices=("velocity", "acceleration"),
        default="velocity",
        help=(
            "Supervised target semantics. velocity predicts delta_v applied directly; "
            "acceleration predicts delta_v / dt and is applied as dt * output in rollout."
        ),
    )
    parser.add_argument("--horizon-gamma", type=float, default=0.95)
    parser.add_argument("--residual-l2-weight", type=float, default=1.0e-4)
    parser.add_argument("--residual-smoothness-weight", type=float, default=1.0e-4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--solver-iterations", type=int, default=10)
    parser.add_argument("--box-mass", type=float, default=1.0)
    parser.add_argument("--floor-half-extents", type=float, nargs=3, default=(2.0, 2.0, 0.05))
    parser.add_argument("--box-half-extents", type=float, nargs=3, default=(0.1, 0.05, 0.025))
    parser.add_argument("--box-start-pos", type=float, nargs=3, default=(0.58, 0.0, 0.025))
    parser.add_argument("--surface-point-spacing", type=float, default=0.01)
    parser.add_argument("--contact-friction", type=float, default=0.0)
    parser.add_argument("--point-friction", type=float, default=0.35)
    parser.add_argument("--min-point-friction", type=float, default=0.0)
    parser.add_argument("--max-point-friction", type=float, default=2.0)
    parser.add_argument("--contact-stiffness", type=float, default=1.0e5)
    parser.add_argument("--contact-damping", type=float, default=50.0)
    parser.add_argument("--contact-margin", type=float, default=1.0e-3)
    parser.add_argument("--friction-contact-threshold", type=float, default=0.002)
    parser.add_argument("--contact-mask-threshold", type=float, default=0.002)
    parser.add_argument("--friction-regularization", type=float, default=1.0e-3)
    parser.add_argument("--steps", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--dt", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--batch-capacity", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()
    return args


def _json_safe_args(args: argparse.Namespace) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, np.ndarray):
            result[key] = value.tolist()
        else:
            result[key] = value
    return result


def _init_wandb(args: argparse.Namespace, *, experiment_dir: Path, metadata: dict) -> object | None:
    if not bool(args.wandb):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("Weights & Biases logging requested with --wandb, but wandb is not installed.") from exc

    wandb_dir = args.wandb_dir if args.wandb_dir is not None else experiment_dir
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.wandb_run_name or experiment_dir.name
    config = {
        **_json_safe_args(args),
        "experiment_dir": str(experiment_dir.resolve()),
        "metadata": metadata,
    }
    init_kwargs = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "name": run_name,
        "group": args.wandb_group,
        "mode": args.wandb_mode,
        "dir": str(wandb_dir),
        "tags": args.wandb_tags,
        "config": config,
    }
    wandb_resume_id = getattr(args, "wandb_resume_id", None)
    if wandb_resume_id is not None:
        init_kwargs.update({"id": str(wandb_resume_id), "resume": "allow"})
    run = wandb.init(
        **init_kwargs,
    )
    _define_wandb_metrics(run)
    return run


def _define_wandb_metrics(wandb_run: object) -> None:
    try:
        wandb_run.define_metric("progress/iteration")
        for prefix in (
            "train",
            "val",
            "train_prediction",
            "val_prediction",
            "train_target",
            "val_target",
            "optim",
        ):
            wandb_run.define_metric(f"{prefix}/*", step_metric="progress/iteration")
    except Exception:
        pass


def _load_max_steps(args: argparse.Namespace) -> int | None:
    if bool(args.random_time_windows):
        return args.time_window_source_max_steps
    return args.max_steps


def _metrics_to_float(metrics: dict[str, torch.Tensor | float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            result[key] = float(value.detach().cpu().item())
        else:
            result[key] = float(value)
    return result


def _build_wandb_log_payload(metrics: dict[str, float]) -> dict[str, float]:
    payload: dict[str, float] = {}
    if "iteration" in metrics:
        payload["progress/iteration"] = float(metrics["iteration"])

    train_key_map = {
        "train_loss": "train/loss_total",
        "loss_velocity": "train/loss_velocity",
        "loss_linear_xy": "train/loss_linear_xy",
        "loss_angular_z": "train/loss_angular_z",
        "loss_residual_l2": "train/loss_residual_l2",
        "loss_residual_smoothness": "train/loss_residual_smoothness",
        "pred_linear_abs_mean": "train_prediction/linear_abs_mean",
        "pred_angular_abs_mean": "train_prediction/angular_abs_mean",
        "target_linear_abs_mean": "train_target/linear_abs_mean",
        "target_angular_abs_mean": "train_target/angular_abs_mean",
        "grad_norm": "optim/grad_norm",
    }
    val_key_map = {
        "val_loss": "val/loss_total",
        "val_loss_velocity": "val/loss_velocity",
        "val_loss_linear_xy": "val/loss_linear_xy",
        "val_loss_angular_z": "val/loss_angular_z",
        "val_loss_residual_l2": "val/loss_residual_l2",
        "val_loss_residual_smoothness": "val/loss_residual_smoothness",
        "val_pred_linear_abs_mean": "val_prediction/linear_abs_mean",
        "val_pred_angular_abs_mean": "val_prediction/angular_abs_mean",
        "val_target_linear_abs_mean": "val_target/linear_abs_mean",
        "val_target_angular_abs_mean": "val_target/angular_abs_mean",
    }
    for source_key, wandb_key in train_key_map.items():
        if source_key in metrics:
            payload[wandb_key] = float(metrics[source_key])
    for source_key, wandb_key in val_key_map.items():
        if source_key in metrics:
            payload[wandb_key] = float(metrics[source_key])
    return payload


def _prepare_batch(
    *,
    trajectories: list,
    rng: np.random.Generator,
    args: argparse.Namespace,
    diff_scene,
    buffers,
    initial_body_q: np.ndarray,
    initial_body_qd: np.ndarray,
    friction,
    dino: DinoFeatures | None,
    normalizer: TorchFeatureNormalizer | FeatureNormalizer | None,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    windows, indices, start_steps = sample_window_batch(
        trajectories,
        batch_size=int(batch_size),
        window_steps=int(args.steps),
        rng=rng,
        random_time_windows=bool(args.random_time_windows),
    )
    sim = run_open_loop_rollout(
        diff_scene=diff_scene,
        buffers=buffers,
        trajectories=windows,
        args=args,
        initial_body_q=initial_body_q,
        initial_body_qd=initial_body_qd,
    )
    point_features, point_mask, future_actions, targets = build_supervised_batch_tensors_torch(
        trajectories=windows,
        sim_positions=sim.positions,
        sim_quaternions_xyzw=sim.quaternions_xyzw,
        sim_linear_velocity=sim.linear_velocity,
        sim_angular_velocity=sim.angular_velocity,
        local_surface_points=diff_scene.local_surface_points_np,
        box_half_extents=np.asarray(args.box_half_extents, dtype=np.float32),
        point_friction=friction.full_point_friction,
        active_contact_mask=friction.active_contact_mask,
        dino=dino,
        history_window_steps=int(args.history_window_steps),
        prediction_window_steps=int(args.prediction_window_steps),
        device=diff_scene.torch_device,
        residual_output_mode=str(args.residual_output_mode),
    )
    if normalizer is not None:
        point_features, future_actions = apply_feature_normalizer_torch(point_features, future_actions, normalizer)
    return point_features, point_mask, future_actions, targets, indices, start_steps


def _collect_normalization(
    *,
    train_trajectories: list,
    rng: np.random.Generator,
    args: argparse.Namespace,
    diff_scene,
    buffers,
    initial_body_q: np.ndarray,
    initial_body_qd: np.ndarray,
    friction,
    dino: DinoFeatures | None,
) -> tuple[FeatureNormalizer, np.ndarray, int]:
    target_samples: list[np.ndarray] = []
    feature_count = 0
    action_count = 0
    feature_sum: torch.Tensor | None = None
    feature_sumsq: torch.Tensor | None = None
    action_sum: torch.Tensor | None = None
    action_sumsq: torch.Tensor | None = None
    point_feature_dim = 0
    sample_batches = max(int(args.normalization_batches), 1)
    for _ in range(sample_batches):
        point_features, _, future_actions, targets, _, _ = _prepare_batch(
            trajectories=train_trajectories,
            rng=rng,
            args=args,
            diff_scene=diff_scene,
            buffers=buffers,
            initial_body_q=initial_body_q,
            initial_body_qd=initial_body_qd,
            friction=friction,
            dino=dino,
            normalizer=None,
            batch_size=min(int(args.batch_size), len(train_trajectories)),
        )
        point_feature_dim = int(point_features.shape[-1])

        batch_feature_count = int(point_features.shape[0] * point_features.shape[1] * point_features.shape[2])
        batch_action_count = int(future_actions.shape[0] * future_actions.shape[1])
        feature_var, feature_mean = torch.var_mean(point_features, dim=(0, 1, 2), correction=0)
        action_var, action_mean = torch.var_mean(future_actions, dim=(0, 1), correction=0)

        feature_mean64 = feature_mean.to(dtype=torch.float64)
        feature_second64 = (feature_var.to(dtype=torch.float64) + feature_mean64.square()) * float(batch_feature_count)
        feature_sum64 = feature_mean64 * float(batch_feature_count)
        action_mean64 = action_mean.to(dtype=torch.float64)
        action_second64 = (action_var.to(dtype=torch.float64) + action_mean64.square()) * float(batch_action_count)
        action_sum64 = action_mean64 * float(batch_action_count)

        feature_sum = feature_sum64 if feature_sum is None else feature_sum + feature_sum64
        feature_sumsq = feature_second64 if feature_sumsq is None else feature_sumsq + feature_second64
        action_sum = action_sum64 if action_sum is None else action_sum + action_sum64
        action_sumsq = action_second64 if action_sumsq is None else action_sumsq + action_second64
        feature_count += batch_feature_count
        action_count += batch_action_count
        target_samples.append(targets.detach().cpu().numpy())
        del point_features, future_actions, targets

    if feature_sum is None or feature_sumsq is None or action_sum is None or action_sumsq is None:
        raise ValueError("Cannot compute feature normalization without sample tensors")
    feature_mean = feature_sum / float(max(feature_count, 1))
    feature_var = (feature_sumsq / float(max(feature_count, 1)) - feature_mean.square()).clamp_min(0.0)
    action_mean = action_sum / float(max(action_count, 1))
    action_var = (action_sumsq / float(max(action_count, 1)) - action_mean.square()).clamp_min(0.0)
    normalizer = FeatureNormalizer(
        point_feature_mean=feature_mean.detach().cpu().numpy().astype(np.float32),
        point_feature_std=torch.sqrt(feature_var).clamp_min(1.0e-6).detach().cpu().numpy().astype(np.float32),
        action_mean=action_mean.detach().cpu().numpy().astype(np.float32),
        action_std=torch.sqrt(action_var).clamp_min(1.0e-6).detach().cpu().numpy().astype(np.float32),
    )
    targets_all = np.concatenate(target_samples, axis=0)
    return normalizer, targets_all, point_feature_dim


def _auto_output_scales(
    targets: np.ndarray,
    *,
    linear_output_scale: float | None,
    angular_output_scale: float | None,
    residual_output_mode: str,
) -> tuple[float, float]:
    output_mode = normalize_residual_output_mode(residual_output_mode)
    abs_targets = np.abs(np.asarray(targets, dtype=np.float32))
    if linear_output_scale is None:
        linear = float(np.percentile(abs_targets[..., :2], 95.0))
        if output_mode == "acceleration":
            linear_output_scale = float(np.clip(max(linear * 1.25, 2.0), 2.0, 20.0))
        else:
            linear_output_scale = float(np.clip(max(linear * 1.25, 0.05), 0.05, 0.2))
    if angular_output_scale is None:
        angular = float(np.percentile(abs_targets[..., 2], 95.0))
        if output_mode == "acceleration":
            angular_output_scale = float(np.clip(max(angular * 1.25, 20.0), 20.0, 200.0))
        else:
            angular_output_scale = float(np.clip(max(angular * 1.25, 0.5), 0.5, 2.0))
    return float(linear_output_scale), float(angular_output_scale)


def _evaluate_supervised(
    *,
    model,
    trajectories: list,
    rng: np.random.Generator,
    args: argparse.Namespace,
    diff_scene,
    buffers,
    initial_body_q: np.ndarray,
    initial_body_qd: np.ndarray,
    friction,
    dino: DinoFeatures | None,
    normalizer: TorchFeatureNormalizer | FeatureNormalizer,
    loss_weights: ResidualLossWeights,
    torch_device: torch.device,
    batch_size: int,
    batches: int,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    accum: dict[str, list[float]] = {}
    with torch.no_grad():
        for _ in range(max(int(batches), 1)):
            point_features, point_mask, future_actions, targets, _, _ = _prepare_batch(
                trajectories=trajectories,
                rng=rng,
                args=args,
                diff_scene=diff_scene,
                buffers=buffers,
                initial_body_q=initial_body_q,
                initial_body_qd=initial_body_qd,
                friction=friction,
                dino=dino,
                normalizer=normalizer,
                batch_size=batch_size,
            )
            prediction = model(
                point_features,
                point_mask,
                future_actions,
            )
            loss, metrics = residual_velocity_loss(prediction, targets, loss_weights)
            losses.append(float(loss.detach().cpu().item()))
            for key, value in _metrics_to_float(metrics).items():
                accum.setdefault(key, []).append(value)
    result = {f"val_{key}": float(np.mean(values)) for key, values in accum.items()}
    result["val_loss"] = float(np.mean(losses))
    model.train()
    return result


def main() -> None:
    args = parse_args()
    start_time = time.time()
    rng = np.random.default_rng(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    if int(args.history_window_steps) < 1 or int(args.prediction_window_steps) < 1:
        raise ValueError("--history-window-steps and --prediction-window-steps must be positive")
    args.residual_output_mode = normalize_residual_output_mode(args.residual_output_mode)
    args.steps = int(args.history_window_steps) + int(args.prediction_window_steps) - 1
    args.eval_batch_size = int(args.eval_batch_size or args.batch_size)
    args.batch_capacity = max(int(args.batch_size), int(args.eval_batch_size), 1)

    configured_point_cloud = maybe_configure_scene_from_point_cloud(args)
    load_max_steps = _load_max_steps(args)
    collection = load_mujoco_trajectories(
        trajectory_npz_path=args.trajectory_npz,
        max_steps=load_max_steps,
        max_trajectories=args.max_trajectories,
    )
    if collection.max_steps < int(args.steps):
        raise ValueError(
            f"Dataset max_steps={collection.max_steps} is shorter than required segment steps={args.steps}"
        )
    args.dt = float(collection.trajectories[0].timestep)
    splits = split_trajectories(
        collection.trajectories,
        train_fraction=float(args.train_fraction),
        seed=int(args.seed),
        min_steps=int(args.steps),
    )

    print(
        f"loaded trajectories train={len(splits.train)} val={len(splits.val)} "
        f"segment_steps={args.steps} dt={args.dt:.6g}",
        flush=True,
    )
    print(f"building Newton scene device={args.device if args.device is not None else 'auto'}", flush=True)
    diff_scene = build_diff_scene(args)
    initial_body_q = diff_scene.states[0].body_q.numpy().copy()
    initial_body_qd = diff_scene.states[0].body_qd.numpy().copy()
    device = str(diff_scene.torch_device)

    fallback_active_indices = active_indices_from_trajectories(
        local_surface_points=diff_scene.local_surface_points_np,
        trajectories=splits.train,
        floor_top_z=float(diff_scene.floor_top_z),
        contact_threshold=float(args.contact_mask_threshold),
    )
    friction = resolve_friction_conditioning(
        args=args,
        local_surface_points=diff_scene.local_surface_points_np,
        box_half_extents=np.asarray(args.box_half_extents, dtype=np.float32),
        fallback_active_indices=fallback_active_indices,
        device=device,
    )
    print(
        f"friction source={friction.source_type} parameterization={friction.parameterization} "
        f"active={len(friction.active_indices)}/{len(diff_scene.local_surface_points_np)} "
        f"mu_mean={float(np.mean(friction.full_point_friction)):.6g}",
        flush=True,
    )

    if bool(args.without_dino):
        dino = None
    else:
        if args.dino_feature_npz is None or not args.dino_feature_npz.exists():
            raise FileNotFoundError(f"--dino-feature-npz does not exist: {args.dino_feature_npz}")
        dino = load_aligned_dino_features(
            args.dino_feature_npz,
            diff_scene.local_surface_points_np,
            max_match_distance=float(args.dino_max_match_distance),
        )
        print(f"loaded DINO features dim={dino.dim} path={dino.path}", flush=True)

    buffers = build_rollout_buffers(
        device=device,
        batch_capacity=int(args.batch_capacity),
        step_capacity=int(args.steps),
        point_count=len(diff_scene.local_surface_points_np),
        full_point_friction=friction.full_point_friction,
    )

    print(f"collecting normalization samples batches={max(int(args.normalization_batches), 1)}", flush=True)
    normalizer, target_samples, point_feature_dim = _collect_normalization(
        train_trajectories=splits.train,
        rng=rng,
        args=args,
        diff_scene=diff_scene,
        buffers=buffers,
        initial_body_q=initial_body_q,
        initial_body_qd=initial_body_qd,
        friction=friction,
        dino=dino,
    )
    linear_scale, angular_scale = _auto_output_scales(
        target_samples,
        linear_output_scale=args.linear_output_scale,
        angular_output_scale=args.angular_output_scale,
        residual_output_mode=str(args.residual_output_mode),
    )
    print(
        f"point_feature_dim={point_feature_dim} output_scales linear={linear_scale:.6g} angular={angular_scale:.6g}",
        flush=True,
    )

    torch_device = diff_scene.torch_device
    normalizer_torch = normalizer_to_torch(normalizer, device=torch_device)
    model = PointNetResidualPredictor(
        point_feature_dim=point_feature_dim,
        history_window_steps=int(args.history_window_steps),
        prediction_window_steps=int(args.prediction_window_steps),
        pointnet_feature_dim=int(args.pointnet_feature_dim),
        action_context_dim=int(args.pointnet_action_context_dim),
        pooling=str(args.pointnet_pooling),
        linear_output_scale=linear_scale,
        angular_output_scale=angular_scale,
    ).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate))
    loss_weights = ResidualLossWeights(
        linear_velocity=float(args.loss_linear_velocity_weight),
        angular_velocity_z=float(args.loss_angular_velocity_weight),
        horizon_gamma=float(args.horizon_gamma),
        residual_l2=float(args.residual_l2_weight),
        residual_smoothness=float(args.residual_smoothness_weight),
    )

    experiment_dir = Path(args.experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    experiment_name = experiment_dir.name
    checkpoint_path = experiment_dir / f"{experiment_name}.pt"
    last_checkpoint_path = experiment_dir / f"{experiment_name}_last.pt"
    metrics_path = experiment_dir / f"{experiment_name}_metrics.json"

    base_metadata = {
        "history_window_steps": int(args.history_window_steps),
        "prediction_window_steps": int(args.prediction_window_steps),
        "point_feature_schema": point_feature_schema(0 if dino is None else dino.dim),
        "point_feature_dim": point_feature_dim,
        "action_feature_schema": ACTION_FEATURE_SCHEMA,
        "dino_feature_npz": None if dino is None else str(dino.path.resolve()),
        "dino_feature_dim": 0 if dino is None else dino.dim,
        "dino_max_match_distance": float(args.dino_max_match_distance),
        "friction_checkpoint": None if args.friction_checkpoint is None else str(args.friction_checkpoint.resolve()),
        "friction_point_cloud": None if args.friction_point_cloud is None else str(args.friction_point_cloud.resolve()),
        "friction_source_type": friction.source_type,
        "friction_parameterization": friction.parameterization,
        "friction_metadata": friction.metadata,
        "configured_point_cloud": None if configured_point_cloud is None else str(configured_point_cloud.resolve()),
        "surface_point_spacing": float(args.surface_point_spacing),
        "box_half_extents": np.asarray(args.box_half_extents, dtype=float).tolist(),
        "point_friction": float(args.point_friction),
        "contact_parameters": {
            "contact_friction": float(args.contact_friction),
            "contact_stiffness": float(args.contact_stiffness),
            "contact_damping": float(args.contact_damping),
            "contact_margin": float(args.contact_margin),
            "friction_contact_threshold": float(args.friction_contact_threshold),
            "contact_mask_threshold": float(args.contact_mask_threshold),
            "friction_regularization": float(args.friction_regularization),
            "solver_iterations": int(args.solver_iterations),
            "box_mass": float(args.box_mass),
            "floor_half_extents": np.asarray(args.floor_half_extents, dtype=float).tolist(),
            "box_start_pos": None if args.box_start_pos is None else np.asarray(args.box_start_pos, dtype=float).tolist(),
        },
        "output_scales": {
            "linear": linear_scale,
            "angular_z": angular_scale,
        },
        "residual_output_mode": str(args.residual_output_mode),
        "linear_output_scale": linear_scale,
        "angular_output_scale": angular_scale,
        "pointnet_feature_dim": int(args.pointnet_feature_dim),
        "action_context_dim": int(args.pointnet_action_context_dim),
        "pointnet_pooling": str(args.pointnet_pooling),
        "training_dataset": str(args.trajectory_npz.resolve()),
        "train_trajectories": len(splits.train),
        "val_trajectories": len(splits.val),
        "random_time_windows": bool(args.random_time_windows),
        "segment_steps": int(args.steps),
        "dt": float(args.dt),
        "seed": int(args.seed),
    }
    wandb_run = _init_wandb(args, experiment_dir=experiment_dir, metadata=base_metadata)

    best_loss = float("inf")
    history: list[dict[str, float]] = []
    for iteration in range(1, int(args.opt_iters) + 1):
        model.train()
        point_features, point_mask, future_actions, targets, _, _ = _prepare_batch(
            trajectories=splits.train,
            rng=rng,
            args=args,
            diff_scene=diff_scene,
            buffers=buffers,
            initial_body_q=initial_body_q,
            initial_body_qd=initial_body_qd,
            friction=friction,
            dino=dino,
            normalizer=normalizer_torch,
            batch_size=min(int(args.batch_size), len(splits.train)),
        )
        prediction = model(point_features, point_mask, future_actions)
        loss, metrics_t = residual_velocity_loss(prediction, targets, loss_weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip_norm))
        optimizer.step()

        metrics = _metrics_to_float(metrics_t)
        metrics["iteration"] = float(iteration)
        metrics["train_loss"] = float(loss.detach().cpu().item())
        metrics["grad_norm"] = float(grad_norm.detach().cpu().item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
        selected_loss = metrics["train_loss"]

        if int(args.val_every) > 0 and (iteration % int(args.val_every) == 0 or iteration == int(args.opt_iters)):
            val_metrics = _evaluate_supervised(
                model=model,
                trajectories=splits.val,
                rng=rng,
                args=args,
                diff_scene=diff_scene,
                buffers=buffers,
                initial_body_q=initial_body_q,
                initial_body_qd=initial_body_qd,
                friction=friction,
                dino=dino,
                normalizer=normalizer_torch,
                loss_weights=loss_weights,
                torch_device=torch_device,
                batch_size=min(int(args.eval_batch_size), len(splits.val)),
                batches=int(args.val_batches),
            )
            metrics.update(val_metrics)
            selected_loss = val_metrics["val_loss"]

        history.append(metrics)
        if wandb_run is not None:
            wandb_run.log(_build_wandb_log_payload(metrics), step=iteration)

        if selected_loss < best_loss:
            best_loss = selected_loss
            metadata = dict(base_metadata)
            metadata.update({"best_iteration": int(iteration), "best_loss": float(best_loss)})
            save_adapter_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                metadata=metadata,
                normalizer=normalizer,
                local_surface_points=diff_scene.local_surface_points_np,
                full_point_friction=friction.full_point_friction,
                active_contact_mask=friction.active_contact_mask,
                dino_features=None if dino is None else dino.features,
                dino_bottom_feature_copied_from_top=None if dino is None else dino.bottom_feature_copied_from_top,
            )

        if int(args.checkpoint_every) > 0 and (
            iteration % int(args.checkpoint_every) == 0 or iteration == int(args.opt_iters)
        ):
            metadata = dict(base_metadata)
            metadata.update({"last_iteration": int(iteration), "best_loss": float(best_loss)})
            save_adapter_checkpoint(
                checkpoint_path=last_checkpoint_path,
                model=model,
                metadata=metadata,
                normalizer=normalizer,
                local_surface_points=diff_scene.local_surface_points_np,
                full_point_friction=friction.full_point_friction,
                active_contact_mask=friction.active_contact_mask,
                dino_features=None if dino is None else dino.features,
                dino_bottom_feature_copied_from_top=None if dino is None else dino.bottom_feature_copied_from_top,
            )

        if int(args.log_every) > 0 and (iteration % int(args.log_every) == 0 or iteration == 1):
            val_part = f" val_loss={metrics['val_loss']:.6g}" if "val_loss" in metrics else ""
            print(
                f"iter={iteration:05d} train_loss={metrics['train_loss']:.6g}{val_part} "
                f"lin_mse={metrics['loss_linear_xy']:.6g} wz_mse={metrics['loss_angular_z']:.6g} "
                f"elapsed={time.time() - start_time:.1f}s",
                flush=True,
            )

    save_json(metrics_path, {"best_loss": best_loss, "history": history})
    if wandb_run is not None:
        wandb_run.summary["best_loss"] = float(best_loss)
        wandb_run.summary["best_checkpoint"] = str(checkpoint_path.resolve())
        wandb_run.summary["last_checkpoint"] = str(last_checkpoint_path.resolve())
        wandb_run.summary["metrics_path"] = str(metrics_path.resolve())
        try:
            wandb_run.save(str(checkpoint_path))
            wandb_run.save(str(metrics_path))
        except Exception:
            pass
        wandb_run.finish()
    print(f"best_checkpoint={checkpoint_path.resolve()}", flush=True)
    print(f"last_checkpoint={last_checkpoint_path.resolve()}", flush=True)
    print(f"metrics={metrics_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
