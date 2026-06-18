from __future__ import annotations

"""PointNet residual curriculum trainer.

The closed-loop phase uses the current adapter to generate visited Newton states,
then trains the same supervised velocity-residual target on those states. Newton
rollout is intentionally outside the autograd graph in this first curriculum.
"""

import argparse
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

from mujoco_contact_friction_fit_utils import load_mujoco_trajectories, slice_mujoco_trajectory_time_window
from newton_surface_points_diff_demo import build_diff_scene

from pointnet_residual_adapter.checkpoints import save_adapter_checkpoint, save_json
from pointnet_residual_adapter.dataset import sample_window_batch
from pointnet_residual_adapter.features import (
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
    quaternion_xyzw_to_matrix_torch,
    residual_output_components,
)
from pointnet_residual_adapter.friction import (
    active_indices_from_trajectories,
    maybe_configure_scene_from_point_cloud,
    resolve_friction_conditioning,
)
from pointnet_residual_adapter.model import PointNetResidualPredictor, ResidualLossWeights, residual_velocity_loss
from pointnet_residual_adapter.newton_rollout import (
    _build_point_feature_frame_torch,
    _future_action_features_torch,
    _normalize_pointnet_inputs,
    build_rollout_buffers,
    run_closed_loop_pointnet_rollout_batch,
    run_open_loop_rollout,
)
from pointnet_residual_adapter.train_supervised_pointnet_residual import (
    DEFAULT_DINO_FEATURE_NPZ,
    DEFAULT_FRICTION_CHECKPOINT,
    DEFAULT_TRAIN_DATASET,
    _build_wandb_log_payload,
    _collect_normalization,
    _init_wandb,
    _json_safe_args,
    _load_max_steps,
    _metrics_to_float,
    _prepare_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--trajectory-npz", type=Path, default=DEFAULT_TRAIN_DATASET)
    parser.add_argument("--friction-checkpoint", type=Path, default=DEFAULT_FRICTION_CHECKPOINT)
    parser.add_argument("--friction-point-cloud", type=Path, default=None)
    parser.add_argument("--checkpoint-param-set", choices=("best", "current"), default="best")
    parser.add_argument("--dino-feature-npz", type=Path, default=DEFAULT_DINO_FEATURE_NPZ)
    parser.add_argument("--without-dino", action="store_true", help="Drop DINO features.")
    parser.add_argument("--dino-max-match-distance", type=float, default=1.0e-5)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("outputs/pointnet_residual/curriculum_h4_p1"),
    )

    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--closed-loop-batch-size", type=int, default=None)
    parser.add_argument("--pretrain-iters", type=int, default=5000)
    parser.add_argument("--closed-loop-iters", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--grad-clip-norm", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-time-windows", dest="random_time_windows", action="store_true", default=True)
    parser.add_argument("--no-random-time-windows", dest="random_time_windows", action="store_false")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional source trajectory truncation.")
    parser.add_argument("--time-window-source-max-steps", type=int, default=None)

    parser.add_argument("--history-window-steps", type=int, default=4)
    parser.add_argument("--prediction-window-steps", type=int, default=1)
    parser.add_argument("--closed-loop-min-horizon-steps", type=int, default=None)
    parser.add_argument("--closed-loop-max-horizon-steps", type=int, default=50)
    parser.add_argument("--horizon-warmup-iters", type=int, default=None)
    parser.add_argument("--closed-loop-start-gain", type=float, default=0.0)
    parser.add_argument("--closed-loop-target-gain", type=float, default=0.02)
    parser.add_argument("--gain-warmup-iters", type=int, default=None)
    parser.add_argument(
        "--closed-loop-loss-mode",
        choices=("supervised", "trajectory"),
        default="supervised",
        help=(
            "Loss used after pretraining. supervised keeps the velocity-residual target on closed-loop "
            "visited states; trajectory uses pose/velocity rollout loss on the corrected trajectory."
        ),
    )
    parser.add_argument(
        "--output-head-init",
        choices=("zero", "small", "default"),
        default="zero",
        help="Initialization for the residual output head before supervised pretraining.",
    )
    parser.add_argument("--output-head-init-std", type=float, default=1.0e-4)

    parser.add_argument("--normalization-batches", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=25)

    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default="newton_friction_fitting")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default="pointnet-residual-curriculum")
    parser.add_argument("--wandb-mode", type=str, default="online")
    parser.add_argument("--wandb-dir", type=Path, default=None)
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=None)

    parser.add_argument("--pointnet-feature-dim", type=int, default=256)
    parser.add_argument("--pointnet-action-context-dim", type=int, default=64)
    parser.add_argument("--pointnet-pooling", choices=("max", "mean-max"), default="mean-max")
    parser.add_argument("--linear-output-scale", type=float, default=None)
    parser.add_argument("--angular-output-scale", type=float, default=None)
    parser.add_argument("--position-output-scale", type=float, default=None)
    parser.add_argument("--yaw-output-scale", type=float, default=None)
    parser.add_argument("--loss-linear-velocity-weight", type=float, default=1.0)
    parser.add_argument("--loss-angular-velocity-weight", type=float, default=0.1)
    parser.add_argument("--loss-position-weight", type=float, default=1.0)
    parser.add_argument("--loss-yaw-weight", type=float, default=1.0)
    parser.add_argument(
        "--residual-output-mode",
        choices=("velocity", "acceleration", "pose", "position", "pose_velocity", "all"),
        default="velocity",
        help=(
            "velocity predicts [dvx_body,dvy_body,domega_z]; acceleration predicts velocity residuals per dt; "
            "pose predicts [dx_body,dy_body,dyaw]; pose_velocity predicts all six."
        ),
    )
    parser.add_argument("--horizon-gamma", type=float, default=0.95)
    parser.add_argument("--residual-l2-weight", type=float, default=1.0e-4)
    parser.add_argument("--residual-smoothness-weight", type=float, default=1.0e-4)
    parser.add_argument("--trajectory-position-loss-weight", type=float, default=1.0)
    parser.add_argument("--trajectory-orientation-loss-weight", type=float, default=1.0)
    parser.add_argument("--trajectory-linear-velocity-loss-weight", type=float, default=0.0)
    parser.add_argument("--trajectory-angular-velocity-loss-weight", type=float, default=0.1)

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


def _args_with(args: argparse.Namespace, **updates: object) -> argparse.Namespace:
    clone = argparse.Namespace(**vars(args))
    for key, value in updates.items():
        setattr(clone, key, value)
    return clone


def _eligible_trajectories(trajectories: list, *, min_steps: int) -> list:
    eligible = [trajectory for trajectory in trajectories if trajectory.num_steps >= int(min_steps)]
    if not eligible:
        raise ValueError(f"No trajectories contain at least {int(min_steps)} steps")
    return eligible


def _auto_curriculum_output_scales(
    targets: np.ndarray,
    *,
    linear_output_scale: float | None,
    angular_output_scale: float | None,
    position_output_scale: float | None,
    yaw_output_scale: float | None,
    residual_output_mode: str,
) -> tuple[float, float, float, float]:
    output_mode = normalize_residual_output_mode(residual_output_mode)
    has_pose, has_velocity = residual_output_components(output_mode)
    abs_targets = np.abs(np.asarray(targets, dtype=np.float32))
    pose_targets = abs_targets[..., :3] if has_pose else None
    velocity_targets = abs_targets[..., -3:] if has_velocity else None

    if position_output_scale is None:
        if pose_targets is None:
            position_output_scale = 0.01
        else:
            position = float(np.percentile(pose_targets[..., :2], 95.0))
            position_output_scale = float(np.clip(max(position * 1.25, 0.005), 0.005, 0.1))
    if yaw_output_scale is None:
        if pose_targets is None:
            yaw_output_scale = 0.1
        else:
            yaw = float(np.percentile(pose_targets[..., 2], 95.0))
            yaw_output_scale = float(np.clip(max(yaw * 1.25, 0.05), 0.05, 1.0))
    if linear_output_scale is None:
        if velocity_targets is None:
            linear_output_scale = 0.05
        else:
            linear = float(np.percentile(velocity_targets[..., :2], 95.0))
            if output_mode == "acceleration":
                linear_output_scale = float(np.clip(max(linear * 1.25, 2.0), 2.0, 20.0))
            else:
                linear_output_scale = float(np.clip(max(linear * 1.25, 0.05), 0.05, 0.2))
    if angular_output_scale is None:
        if velocity_targets is None:
            angular_output_scale = 0.5
        else:
            angular = float(np.percentile(velocity_targets[..., 2], 95.0))
            if output_mode == "acceleration":
                angular_output_scale = float(np.clip(max(angular * 1.25, 20.0), 20.0, 200.0))
            else:
                angular_output_scale = float(np.clip(max(angular * 1.25, 0.5), 0.5, 2.0))
    return (
        float(linear_output_scale),
        float(angular_output_scale),
        float(position_output_scale),
        float(yaw_output_scale),
    )


def _linear_schedule(start: float, end: float, step: int, warmup_steps: int | None) -> float:
    warmup = int(warmup_steps or 0)
    if warmup <= 0:
        return float(end)
    ratio = min(max(float(step) / float(warmup), 0.0), 1.0)
    return float(start) + ratio * (float(end) - float(start))


def _int_linear_schedule(start: int, end: int, step: int, warmup_steps: int | None) -> int:
    value = _linear_schedule(float(start), float(end), step, warmup_steps)
    return int(round(value))


def _init_residual_output_head(model: PointNetResidualPredictor, mode: str, std: float) -> None:
    if mode == "default":
        return
    if mode == "zero":
        torch.nn.init.zeros_(model.output_head.weight)
        torch.nn.init.zeros_(model.output_head.bias)
        return
    if mode == "small":
        torch.nn.init.normal_(model.output_head.weight, mean=0.0, std=float(std))
        torch.nn.init.zeros_(model.output_head.bias)
        return
    raise ValueError(f"Unsupported output-head init mode: {mode!r}")


def _prepare_closed_loop_batch(
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
    normalizer: FeatureNormalizer | TorchFeatureNormalizer,
    normalizer_np: FeatureNormalizer,
    model: PointNetResidualPredictor,
    batch_size: int,
    closed_loop_horizon_steps: int,
    residual_gain: float,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    history_steps = int(args.history_window_steps)
    prediction_steps = int(args.prediction_window_steps)
    if int(closed_loop_horizon_steps) < history_steps:
        raise ValueError("closed_loop_horizon_steps must be >= history_window_steps")

    source_window_steps = int(closed_loop_horizon_steps) + prediction_steps - 1
    windows, indices, start_steps = sample_window_batch(
        trajectories,
        batch_size=int(batch_size),
        window_steps=source_window_steps,
        rng=rng,
        random_time_windows=bool(args.random_time_windows),
    )

    rollout_args = _args_with(
        args,
        steps=source_window_steps,
        pointnet_residual_gain=float(residual_gain),
        pointnet_residual_output_mode=str(args.residual_output_mode),
    )
    sim, _ = run_closed_loop_pointnet_rollout_batch(
        diff_scene=diff_scene,
        buffers=buffers,
        trajectories=windows,
        args=rollout_args,
        initial_body_q=initial_body_q,
        initial_body_qd=initial_body_qd,
        model=model,
        normalizer=normalizer_np,
        local_surface_points=diff_scene.local_surface_points_np,
        box_half_extents=np.asarray(args.box_half_extents, dtype=np.float32),
        point_friction=friction.full_point_friction,
        active_contact_mask=friction.active_contact_mask,
        dino=dino,
        torch_device=diff_scene.torch_device,
    )

    offset = int(closed_loop_horizon_steps) - history_steps
    supervised_window_steps = history_steps + prediction_steps - 1
    frame_end = offset + supervised_window_steps + 1
    sub_windows = [
        slice_mujoco_trajectory_time_window(
            trajectory,
            start_step=offset,
            window_steps=supervised_window_steps,
        )
        for trajectory in windows
    ]

    point_features, point_mask, future_actions, targets = build_supervised_batch_tensors_torch(
        trajectories=sub_windows,
        sim_positions=sim.positions[:, offset:frame_end],
        sim_quaternions_xyzw=sim.quaternions_xyzw[:, offset:frame_end],
        sim_linear_velocity=sim.linear_velocity[:, offset:frame_end],
        sim_angular_velocity=sim.angular_velocity[:, offset:frame_end],
        local_surface_points=diff_scene.local_surface_points_np,
        box_half_extents=np.asarray(args.box_half_extents, dtype=np.float32),
        point_friction=friction.full_point_friction,
        active_contact_mask=friction.active_contact_mask,
        dino=dino,
        history_window_steps=history_steps,
        prediction_window_steps=prediction_steps,
        device=diff_scene.torch_device,
        residual_output_mode=str(args.residual_output_mode),
    )
    point_features, future_actions = apply_feature_normalizer_torch(point_features, future_actions, normalizer)
    return point_features, point_mask, future_actions, targets, indices, start_steps


def _train_step(
    *,
    model: PointNetResidualPredictor,
    optimizer: torch.optim.Optimizer,
    loss_weights: ResidualLossWeights,
    point_features: torch.Tensor,
    point_mask: torch.Tensor | None,
    future_actions: torch.Tensor,
    targets: torch.Tensor,
    grad_clip_norm: float,
    residual_output_mode: str,
) -> dict[str, float]:
    model.train()
    prediction = model(point_features, point_mask, future_actions)
    loss, metrics_t = residual_velocity_loss(
        prediction,
        targets,
        loss_weights,
        residual_output_mode=str(residual_output_mode),
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
    optimizer.step()

    metrics = _metrics_to_float(metrics_t)
    metrics["train_loss"] = float(loss.detach().cpu().item())
    metrics["grad_norm"] = float(grad_norm.detach().cpu().item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
    return metrics


def _yaw_from_quat_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    q = quaternion / torch.linalg.norm(quaternion, dim=-1, keepdim=True).clamp_min(1.0e-8)
    x = q[..., 0]
    y = q[..., 1]
    z = q[..., 2]
    w = q[..., 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _apply_yaw_delta_xyzw(quaternion: torch.Tensor, yaw_delta: torch.Tensor) -> torch.Tensor:
    half = 0.5 * yaw_delta
    zeros = torch.zeros_like(half)
    delta = torch.stack((zeros, zeros, torch.sin(half), torch.cos(half)), dim=-1)

    ax, ay, az, aw = delta.unbind(dim=-1)
    bx, by, bz, bw = quaternion.unbind(dim=-1)
    out = torch.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        dim=-1,
    )
    return out / torch.linalg.norm(out, dim=-1, keepdim=True).clamp_min(1.0e-8)


def _split_residual_output(residual: torch.Tensor, mode: str) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    output_mode = normalize_residual_output_mode(mode)
    if output_mode == "pose":
        return residual[..., :3], None
    if output_mode == "pose_velocity":
        return residual[..., :3], residual[..., 3:6]
    return None, residual[..., :3]


def _body_planar_to_world(delta_body_xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    return torch.stack(
        (
            c * delta_body_xy[..., 0] - s * delta_body_xy[..., 1],
            s * delta_body_xy[..., 0] + c * delta_body_xy[..., 1],
            torch.zeros_like(delta_body_xy[..., 0]),
        ),
        dim=-1,
    )


def _stack_window_array(
    windows: list,
    attribute: str,
    *,
    frame_count: int,
    device: torch.device | str,
) -> torch.Tensor:
    values = np.stack(
        [np.asarray(getattr(window, attribute)[:frame_count], dtype=np.float32) for window in windows],
        axis=0,
    )
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _step_forces_tensor(windows: list, *, step_count: int, device: torch.device | str) -> torch.Tensor:
    forces = np.zeros((len(windows), int(step_count), 3), dtype=np.float32)
    for batch_idx, window in enumerate(windows):
        source = np.asarray(window.step_forces, dtype=np.float32).reshape(-1, 3)
        used = min(len(source), int(step_count))
        if used > 0:
            forces[batch_idx, :used] = source[:used]
            if used < int(step_count):
                forces[batch_idx, used:] = source[used - 1]
    return torch.as_tensor(forces, dtype=torch.float32, device=device)


def _point_offsets_tensor(windows: list, *, device: torch.device | str) -> torch.Tensor:
    offsets = np.stack(
        [np.asarray(window.force_point_offset_local, dtype=np.float32).reshape(3) for window in windows],
        axis=0,
    )
    return torch.as_tensor(offsets, dtype=torch.float32, device=device)


def _trajectory_step_counts_tensor(windows: list, *, device: torch.device | str) -> torch.Tensor:
    counts = np.asarray([int(window.num_steps) for window in windows], dtype=np.int64)
    return torch.as_tensor(counts, dtype=torch.long, device=device)


def _surface_point_position_loss(
    *,
    predicted_positions: torch.Tensor,
    predicted_quaternions: torch.Tensor,
    target_positions: torch.Tensor,
    target_quaternions: torch.Tensor,
    local_surface_points: torch.Tensor,
) -> torch.Tensor:
    pred_rotation = quaternion_xyzw_to_matrix_torch(predicted_quaternions)
    target_rotation = quaternion_xyzw_to_matrix_torch(target_quaternions)
    pred_points = (
        torch.einsum("bfij,nj->bfni", pred_rotation, local_surface_points)
        + predicted_positions[:, :, None, :]
    )
    target_points = (
        torch.einsum("bfij,nj->bfni", target_rotation, local_surface_points)
        + target_positions[:, :, None, :]
    )
    return (pred_points - target_points).square().mean()


def _closed_loop_trajectory_loss(
    *,
    model: PointNetResidualPredictor,
    trajectories: list,
    rng: np.random.Generator,
    args: argparse.Namespace,
    diff_scene,
    buffers,
    initial_body_q: np.ndarray,
    initial_body_qd: np.ndarray,
    friction,
    dino: DinoFeatures | None,
    normalizer: TorchFeatureNormalizer,
    closed_loop_horizon_steps: int,
    residual_gain: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    horizon = int(closed_loop_horizon_steps)
    windows, _, _ = sample_window_batch(
        trajectories,
        batch_size=int(args.closed_loop_batch_size),
        window_steps=horizon,
        rng=rng,
        random_time_windows=bool(args.random_time_windows),
    )
    rollout_args = _args_with(args, steps=horizon)
    base = run_open_loop_rollout(
        diff_scene=diff_scene,
        buffers=buffers,
        trajectories=windows,
        args=rollout_args,
        initial_body_q=initial_body_q,
        initial_body_qd=initial_body_qd,
    )

    device = diff_scene.torch_device
    batch_size = len(windows)
    frame_count = horizon + 1
    dt = float(args.dt)

    base_positions = torch.as_tensor(base.positions[:, :frame_count], dtype=torch.float32, device=device)
    base_quaternions = torch.as_tensor(base.quaternions_xyzw[:, :frame_count], dtype=torch.float32, device=device)
    base_linear = torch.as_tensor(base.linear_velocity[:, :frame_count], dtype=torch.float32, device=device)
    base_angular = torch.as_tensor(base.angular_velocity[:, :frame_count], dtype=torch.float32, device=device)

    target_positions = _stack_window_array(windows, "positions", frame_count=frame_count, device=device)
    target_quaternions = _stack_window_array(windows, "quaternions_xyzw", frame_count=frame_count, device=device)
    target_linear = _stack_window_array(windows, "linear_velocity", frame_count=frame_count, device=device)
    target_angular = _stack_window_array(windows, "angular_velocity", frame_count=frame_count, device=device)

    step_forces = _step_forces_tensor(windows, step_count=horizon, device=device)
    point_offsets = _point_offsets_tensor(windows, device=device)
    trajectory_step_counts = _trajectory_step_counts_tensor(windows, device=device)

    point_count = len(diff_scene.local_surface_points_np)
    local_points = torch.as_tensor(diff_scene.local_surface_points_np, dtype=torch.float32, device=device).reshape(-1, 3)
    half_extents = torch.as_tensor(args.box_half_extents, dtype=torch.float32, device=device).reshape(1, 3).clamp_min(1.0e-8)
    point_friction_t = torch.as_tensor(friction.full_point_friction, dtype=torch.float32, device=device).reshape(point_count)
    active_mask_t = torch.as_tensor(friction.active_contact_mask, dtype=torch.float32, device=device).reshape(point_count)
    if dino is not None and dino.dim > 0:
        dino_features_t = torch.as_tensor(dino.features, dtype=torch.float32, device=device).reshape(point_count, dino.dim)
        dino_bottom_t = torch.as_tensor(
            dino.bottom_feature_copied_from_top,
            dtype=torch.float32,
            device=device,
        ).reshape(point_count)
    else:
        dino_features_t = None
        dino_bottom_t = None

    corrected_positions = [base_positions[:, 0]]
    corrected_quaternions = [base_quaternions[:, 0]]
    corrected_linear = [base_linear[:, 0]]
    corrected_angular = [base_angular[:, 0]]
    applied_residuals: list[torch.Tensor] = []
    history_buffer: torch.Tensor | None = None

    for step_idx in range(horizon):
        quat = corrected_quaternions[-1]
        lin = corrected_linear[-1]
        ang = corrected_angular[-1]
        frame_features = _build_point_feature_frame_torch(
            local_surface_points=local_points,
            box_half_extents=half_extents,
            quaternion_xyzw=quat,
            linear_velocity_world=lin,
            angular_velocity_world=ang,
            force_world=step_forces[:, step_idx],
            point_offset_local=point_offsets,
            point_friction=point_friction_t,
            active_contact_mask=active_mask_t,
            dino_features=dino_features_t,
            dino_bottom_feature_copied_from_top=dino_bottom_t,
        )
        if history_buffer is None:
            history_buffer = frame_features[:, None, :, :].expand(
                -1,
                int(args.history_window_steps),
                -1,
                -1,
            ).clone()
        else:
            history_buffer = torch.cat((history_buffer[:, 1:], frame_features[:, None]), dim=1)

        future_actions = _future_action_features_torch(
            quaternion_xyzw=quat,
            step_forces=step_forces,
            trajectory_step_counts=trajectory_step_counts,
            point_offset_local=point_offsets,
            step_idx=step_idx,
            prediction_window_steps=int(args.prediction_window_steps),
        )
        point_features, future_actions = _normalize_pointnet_inputs(
            point_features=history_buffer,
            future_actions=future_actions,
            point_feature_mean=normalizer.point_feature_mean,
            point_feature_std=normalizer.point_feature_std,
            action_mean=normalizer.action_mean,
            action_std=normalizer.action_std,
        )
        residual = model(point_features, None, future_actions)[:, 0]
        if str(args.residual_output_mode) == "acceleration":
            residual = residual * dt
        residual = residual * float(residual_gain)
        applied_residuals.append(residual)

        yaw = _yaw_from_quat_xyzw(quat)
        pose_residual, velocity_residual = _split_residual_output(residual, str(args.residual_output_mode))
        pose_delta_world = (
            torch.zeros_like(base_positions[:, step_idx + 1])
            if pose_residual is None
            else _body_planar_to_world(pose_residual[:, :2], yaw)
        )
        velocity_delta_world = (
            torch.zeros_like(base_positions[:, step_idx + 1])
            if velocity_residual is None
            else _body_planar_to_world(velocity_residual[:, :2], yaw)
        )
        pose_yaw_delta = torch.zeros_like(yaw) if pose_residual is None else pose_residual[:, 2]
        velocity_yaw_delta = torch.zeros_like(yaw) if velocity_residual is None else dt * velocity_residual[:, 2]

        next_pos = base_positions[:, step_idx + 1] + pose_delta_world + dt * velocity_delta_world
        next_quat = _apply_yaw_delta_xyzw(base_quaternions[:, step_idx + 1], pose_yaw_delta + velocity_yaw_delta)
        next_linear = base_linear[:, step_idx + 1] + velocity_delta_world
        next_angular = base_angular[:, step_idx + 1].clone()
        angular_delta = torch.zeros_like(yaw) if velocity_residual is None else velocity_residual[:, 2]
        next_angular = torch.cat((next_angular[:, :2], (next_angular[:, 2] + angular_delta).reshape(batch_size, 1)), dim=1)

        corrected_positions.append(next_pos)
        corrected_quaternions.append(next_quat)
        corrected_linear.append(next_linear)
        corrected_angular.append(next_angular)

    predicted_positions = torch.stack(corrected_positions, dim=1)
    predicted_quaternions = torch.stack(corrected_quaternions, dim=1)
    predicted_linear = torch.stack(corrected_linear, dim=1)
    predicted_angular = torch.stack(corrected_angular, dim=1)
    residual_stack = torch.stack(applied_residuals, dim=1)

    pred_frames = slice(1, None)
    position_loss = _surface_point_position_loss(
        predicted_positions=predicted_positions[:, pred_frames],
        predicted_quaternions=predicted_quaternions[:, pred_frames],
        target_positions=target_positions[:, pred_frames],
        target_quaternions=target_quaternions[:, pred_frames],
        local_surface_points=local_points,
    )
    yaw_loss = _wrap_angle(
        _yaw_from_quat_xyzw(predicted_quaternions[:, pred_frames])
        - _yaw_from_quat_xyzw(target_quaternions[:, pred_frames])
    ).square().mean()
    linear_velocity_loss = (predicted_linear[:, pred_frames, :2] - target_linear[:, pred_frames, :2]).square().mean()
    angular_velocity_loss = (predicted_angular[:, pred_frames, 2] - target_angular[:, pred_frames, 2]).square().mean()
    residual_l2 = residual_stack.square().mean()
    if horizon > 1:
        residual_smoothness = (residual_stack[:, 1:] - residual_stack[:, :-1]).square().mean()
    else:
        residual_smoothness = torch.zeros((), dtype=residual_stack.dtype, device=device)
    pose_stack, velocity_stack = _split_residual_output(residual_stack, str(args.residual_output_mode))
    zero_metric = torch.zeros((), dtype=residual_stack.dtype, device=device)

    loss = (
        float(args.trajectory_position_loss_weight) * position_loss
        + float(args.trajectory_orientation_loss_weight) * yaw_loss
        + float(args.trajectory_linear_velocity_loss_weight) * linear_velocity_loss
        + float(args.trajectory_angular_velocity_loss_weight) * angular_velocity_loss
        + float(args.residual_l2_weight) * residual_l2
        + float(args.residual_smoothness_weight) * residual_smoothness
    )
    metrics = {
        "train_loss": float(loss.detach().cpu().item()),
        "trajectory_position_loss": float(position_loss.detach().cpu().item()),
        "trajectory_orientation_loss": float(yaw_loss.detach().cpu().item()),
        "trajectory_linear_velocity_loss": float(linear_velocity_loss.detach().cpu().item()),
        "trajectory_angular_velocity_loss": float(angular_velocity_loss.detach().cpu().item()),
        "loss_residual_l2": float(residual_l2.detach().cpu().item()),
        "loss_residual_smoothness": float(residual_smoothness.detach().cpu().item()),
        "pred_position_abs_mean": float(
            (pose_stack[..., :2].abs().mean() if pose_stack is not None else zero_metric).detach().cpu().item()
        ),
        "pred_yaw_abs_mean": float(
            (pose_stack[..., 2].abs().mean() if pose_stack is not None else zero_metric).detach().cpu().item()
        ),
        "pred_linear_abs_mean": float(
            (velocity_stack[..., :2].abs().mean() if velocity_stack is not None else zero_metric).detach().cpu().item()
        ),
        "pred_angular_abs_mean": float(
            (velocity_stack[..., 2].abs().mean() if velocity_stack is not None else zero_metric).detach().cpu().item()
        ),
    }
    return loss, metrics


def _train_closed_loop_trajectory_step(
    *,
    model: PointNetResidualPredictor,
    optimizer: torch.optim.Optimizer,
    trajectories: list,
    rng: np.random.Generator,
    args: argparse.Namespace,
    diff_scene,
    buffers,
    initial_body_q: np.ndarray,
    initial_body_qd: np.ndarray,
    friction,
    dino: DinoFeatures | None,
    normalizer: TorchFeatureNormalizer,
    closed_loop_horizon_steps: int,
    residual_gain: float,
    grad_clip_norm: float,
) -> dict[str, float]:
    model.train()
    loss, metrics = _closed_loop_trajectory_loss(
        model=model,
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
        closed_loop_horizon_steps=int(closed_loop_horizon_steps),
        residual_gain=float(residual_gain),
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
    optimizer.step()
    metrics["grad_norm"] = float(grad_norm.detach().cpu().item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
    return metrics


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
    args.closed_loop_batch_size = int(args.closed_loop_batch_size or args.batch_size)
    supervised_steps = int(args.history_window_steps) + int(args.prediction_window_steps) - 1
    min_closed_loop_horizon = int(args.closed_loop_min_horizon_steps or args.history_window_steps)
    min_closed_loop_horizon = max(min_closed_loop_horizon, int(args.history_window_steps))
    max_closed_loop_horizon = max(int(args.closed_loop_max_horizon_steps), min_closed_loop_horizon)
    max_required_steps = max(supervised_steps, max_closed_loop_horizon + int(args.prediction_window_steps) - 1)
    args.steps = max_required_steps
    args.batch_capacity = max(int(args.batch_size), int(args.closed_loop_batch_size), 1)

    configured_point_cloud = maybe_configure_scene_from_point_cloud(args)
    collection = load_mujoco_trajectories(
        trajectory_npz_path=args.trajectory_npz,
        max_steps=_load_max_steps(args),
        max_trajectories=args.max_trajectories,
    )
    if collection.max_steps < max_required_steps:
        raise ValueError(
            f"Dataset max_steps={collection.max_steps} is shorter than required max curriculum steps={max_required_steps}"
        )
    args.dt = float(collection.trajectories[0].timestep)
    train_trajectories = _eligible_trajectories(collection.trajectories, min_steps=max_required_steps)
    print(
        f"loaded trajectories train={len(train_trajectories)} heldout=0 "
        f"supervised_steps={supervised_steps} closed_loop_horizon=[{min_closed_loop_horizon},{max_closed_loop_horizon}] "
        f"dt={args.dt:.6g}",
        flush=True,
    )

    print(f"building Newton scene device={args.device if args.device is not None else 'auto'}", flush=True)
    diff_scene = build_diff_scene(args)
    initial_body_q = diff_scene.states[0].body_q.numpy().copy()
    initial_body_qd = diff_scene.states[0].body_qd.numpy().copy()
    device = str(diff_scene.torch_device)

    fallback_active_indices = active_indices_from_trajectories(
        local_surface_points=diff_scene.local_surface_points_np,
        trajectories=train_trajectories,
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
        step_capacity=max_required_steps,
        point_count=len(diff_scene.local_surface_points_np),
        full_point_friction=friction.full_point_friction,
    )
    supervised_args = _args_with(args, steps=supervised_steps)

    print(f"collecting normalization samples batches={max(int(args.normalization_batches), 1)}", flush=True)
    normalizer, target_samples, point_feature_dim = _collect_normalization(
        train_trajectories=train_trajectories,
        rng=rng,
        args=supervised_args,
        diff_scene=diff_scene,
        buffers=buffers,
        initial_body_q=initial_body_q,
        initial_body_qd=initial_body_qd,
        friction=friction,
        dino=dino,
    )
    linear_scale, angular_scale, position_scale, yaw_scale = _auto_curriculum_output_scales(
        target_samples,
        linear_output_scale=args.linear_output_scale,
        angular_output_scale=args.angular_output_scale,
        position_output_scale=args.position_output_scale,
        yaw_output_scale=args.yaw_output_scale,
        residual_output_mode=str(args.residual_output_mode),
    )
    print(
        "point_feature_dim="
        f"{point_feature_dim} output_scales "
        f"position={position_scale:.6g} yaw={yaw_scale:.6g} "
        f"linear={linear_scale:.6g} angular={angular_scale:.6g}",
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
        position_output_scale=position_scale,
        yaw_output_scale=yaw_scale,
        residual_output_mode=str(args.residual_output_mode),
    ).to(torch_device)
    _init_residual_output_head(model, str(args.output_head_init), float(args.output_head_init_std))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate))
    loss_weights = ResidualLossWeights(
        linear_velocity=float(args.loss_linear_velocity_weight),
        angular_velocity_z=float(args.loss_angular_velocity_weight),
        position_xy=float(args.loss_position_weight),
        yaw=float(args.loss_yaw_weight),
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
            "position": position_scale,
            "yaw": yaw_scale,
            "linear": linear_scale,
            "angular_z": angular_scale,
        },
        "residual_output_mode": str(args.residual_output_mode),
        "linear_output_scale": linear_scale,
        "angular_output_scale": angular_scale,
        "position_output_scale": position_scale,
        "yaw_output_scale": yaw_scale,
        "pointnet_feature_dim": int(args.pointnet_feature_dim),
        "action_context_dim": int(args.pointnet_action_context_dim),
        "pointnet_pooling": str(args.pointnet_pooling),
        "training_dataset": str(args.trajectory_npz.resolve()),
        "train_trajectories": len(train_trajectories),
        "heldout_trajectories": 0,
        "random_time_windows": bool(args.random_time_windows),
        "supervised_steps": int(supervised_steps),
        "closed_loop_min_horizon_steps": int(min_closed_loop_horizon),
        "closed_loop_max_horizon_steps": int(max_closed_loop_horizon),
        "closed_loop_start_gain": float(args.closed_loop_start_gain),
        "closed_loop_target_gain": float(args.closed_loop_target_gain),
        "gain_warmup_iters": None if args.gain_warmup_iters is None else int(args.gain_warmup_iters),
        "horizon_warmup_iters": None if args.horizon_warmup_iters is None else int(args.horizon_warmup_iters),
        "closed_loop_loss_mode": str(args.closed_loop_loss_mode),
        "trajectory_loss_weights": {
            "position": float(args.trajectory_position_loss_weight),
            "orientation": float(args.trajectory_orientation_loss_weight),
            "linear_velocity": float(args.trajectory_linear_velocity_loss_weight),
            "angular_velocity": float(args.trajectory_angular_velocity_loss_weight),
        },
        "output_head_init": str(args.output_head_init),
        "dt": float(args.dt),
        "seed": int(args.seed),
        "args": _json_safe_args(args),
    }
    wandb_run = _init_wandb(args, experiment_dir=experiment_dir, metadata=base_metadata)

    best_loss = float("inf")
    history: list[dict[str, float | str]] = []
    total_iters = int(args.pretrain_iters) + int(args.closed_loop_iters)
    for iteration in range(1, total_iters + 1):
        trained_in_branch = False
        if iteration <= int(args.pretrain_iters):
            phase = "pretrain"
            phase_step = iteration
            current_gain = 0.0
            current_horizon = supervised_steps
            point_features, point_mask, future_actions, targets, _, _ = _prepare_batch(
                trajectories=train_trajectories,
                rng=rng,
                args=supervised_args,
                diff_scene=diff_scene,
                buffers=buffers,
                initial_body_q=initial_body_q,
                initial_body_qd=initial_body_qd,
                friction=friction,
                dino=dino,
                normalizer=normalizer_torch,
                batch_size=min(int(args.batch_size), len(train_trajectories)),
            )
        else:
            phase = "closed_loop"
            phase_step = iteration - int(args.pretrain_iters)
            gain_warmup = args.gain_warmup_iters
            if gain_warmup is None:
                gain_warmup = max(int(args.closed_loop_iters), 1)
            horizon_warmup = args.horizon_warmup_iters
            if horizon_warmup is None:
                horizon_warmup = max(int(args.closed_loop_iters), 1)
            current_gain = _linear_schedule(
                float(args.closed_loop_start_gain),
                float(args.closed_loop_target_gain),
                phase_step,
                int(gain_warmup),
            )
            current_horizon = _int_linear_schedule(
                int(min_closed_loop_horizon),
                int(max_closed_loop_horizon),
                phase_step,
                int(horizon_warmup),
            )
            current_horizon = max(int(current_horizon), int(args.history_window_steps))
            if str(args.closed_loop_loss_mode) == "trajectory":
                metrics = _train_closed_loop_trajectory_step(
                    model=model,
                    optimizer=optimizer,
                    trajectories=train_trajectories,
                    rng=rng,
                    args=args,
                    diff_scene=diff_scene,
                    buffers=buffers,
                    initial_body_q=initial_body_q,
                    initial_body_qd=initial_body_qd,
                    friction=friction,
                    dino=dino,
                    normalizer=normalizer_torch,
                    closed_loop_horizon_steps=int(current_horizon),
                    residual_gain=float(current_gain),
                    grad_clip_norm=float(args.grad_clip_norm),
                )
                trained_in_branch = True
            else:
                point_features, point_mask, future_actions, targets, _, _ = _prepare_closed_loop_batch(
                    trajectories=train_trajectories,
                    rng=rng,
                    args=args,
                    diff_scene=diff_scene,
                    buffers=buffers,
                    initial_body_q=initial_body_q,
                    initial_body_qd=initial_body_qd,
                    friction=friction,
                    dino=dino,
                    normalizer=normalizer_torch,
                    normalizer_np=normalizer,
                    model=model,
                    batch_size=min(int(args.closed_loop_batch_size), len(train_trajectories)),
                    closed_loop_horizon_steps=int(current_horizon),
                    residual_gain=float(current_gain),
                )

        if not trained_in_branch:
            metrics = _train_step(
                model=model,
                optimizer=optimizer,
                loss_weights=loss_weights,
                point_features=point_features,
                point_mask=point_mask,
                future_actions=future_actions,
                targets=targets,
                grad_clip_norm=float(args.grad_clip_norm),
                residual_output_mode=str(args.residual_output_mode),
            )
        metrics["iteration"] = float(iteration)
        metrics["phase_step"] = float(phase_step)
        metrics["closed_loop_gain"] = float(current_gain)
        metrics["closed_loop_horizon_steps"] = float(current_horizon)
        metrics["phase_is_closed_loop"] = 1.0 if phase == "closed_loop" else 0.0
        metrics["closed_loop_loss_is_trajectory"] = (
            1.0 if phase == "closed_loop" and str(args.closed_loop_loss_mode) == "trajectory" else 0.0
        )
        if phase == "closed_loop":
            metrics["closed_loop_loss"] = metrics["train_loss"]
        else:
            metrics["pretrain_loss"] = metrics["train_loss"]
        selected_loss = metrics["train_loss"]

        history.append({"phase": phase, **metrics})
        if wandb_run is not None:
            payload = _build_wandb_log_payload(metrics)
            payload["curriculum/phase_is_closed_loop"] = float(metrics["phase_is_closed_loop"])
            payload["curriculum/closed_loop_gain"] = float(current_gain)
            payload["curriculum/closed_loop_horizon_steps"] = float(current_horizon)
            payload["curriculum/closed_loop_loss_is_trajectory"] = float(metrics["closed_loop_loss_is_trajectory"])
            for metric_key in (
                "trajectory_position_loss",
                "trajectory_orientation_loss",
                "trajectory_linear_velocity_loss",
                "trajectory_angular_velocity_loss",
                "loss_position_xy",
                "loss_yaw",
                "pred_position_abs_mean",
                "pred_yaw_abs_mean",
                "target_position_abs_mean",
                "target_yaw_abs_mean",
            ):
                if metric_key in metrics:
                    payload[f"train/{metric_key}"] = float(metrics[metric_key])
            if phase == "closed_loop":
                payload["train/closed_loop_loss"] = float(metrics["closed_loop_loss"])
            else:
                payload["train/pretrain_loss"] = float(metrics["pretrain_loss"])
            wandb_run.log(payload, step=iteration)

        if selected_loss < best_loss:
            best_loss = selected_loss
            metadata = dict(base_metadata)
            metadata.update(
                {
                    "best_iteration": int(iteration),
                    "best_phase": phase,
                    "best_loss": float(best_loss),
                    "best_closed_loop_gain": float(current_gain),
                    "best_closed_loop_horizon_steps": int(current_horizon),
                }
            )
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

        if int(args.checkpoint_every) > 0 and (iteration % int(args.checkpoint_every) == 0 or iteration == total_iters):
            metadata = dict(base_metadata)
            metadata.update(
                {
                    "last_iteration": int(iteration),
                    "last_phase": phase,
                    "best_loss": float(best_loss),
                    "last_closed_loop_gain": float(current_gain),
                    "last_closed_loop_horizon_steps": int(current_horizon),
                }
            )
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
            phase_loss_name = "closed_loop_loss" if phase == "closed_loop" else "pretrain_loss"
            print(
                f"iter={iteration:05d}/{total_iters:05d} phase={phase} "
                f"gain={float(current_gain):.4g} horizon={int(current_horizon)} "
                f"{phase_loss_name}={metrics['train_loss']:.6g} "
                f"grad_norm={metrics['grad_norm']:.6g} elapsed={time.time() - start_time:.1f}s",
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
