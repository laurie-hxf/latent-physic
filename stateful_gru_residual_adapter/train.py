from __future__ import annotations

"""Train a deterministic stateful GRU residual adapter with burn-in and TBPTT."""

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

from mujoco_contact_friction_fit_utils import load_mujoco_trajectories
from newton_surface_points_diff_demo import build_diff_scene

from pointnet_residual_adapter.checkpoints import save_adapter_checkpoint, save_json
from pointnet_residual_adapter.dataset import sample_window_batch
from pointnet_residual_adapter.features import (
    ACTION_FEATURE_SCHEMA,
    DinoFeatures,
    FeatureNormalizer,
    TorchFeatureNormalizer,
    normalize_residual_output_mode,
    normalizer_to_torch,
    point_feature_schema,
    quaternion_xyzw_to_matrix_torch,
)
from pointnet_residual_adapter.friction import (
    active_indices_from_trajectories,
    maybe_configure_scene_from_point_cloud,
    resolve_friction_conditioning,
)
from pointnet_residual_adapter.model import ResidualLossWeights, residual_velocity_loss
from pointnet_residual_adapter.newton_rollout import (
    _build_point_feature_frame_torch,
    _future_action_features_torch,
    _normalize_pointnet_inputs,
    build_rollout_buffers,
    run_open_loop_rollout,
)
from pointnet_residual_adapter.train_curriculum_pointnet_residual import (
    _apply_yaw_delta_xyzw,
    _args_with,
    _auto_curriculum_output_scales,
    _body_planar_to_world,
    _eligible_trajectories,
    _point_offsets_tensor,
    _split_residual_output,
    _stack_window_array,
    _step_forces_tensor,
    _surface_point_position_loss,
    _trajectory_step_counts_tensor,
    _wrap_angle,
    _yaw_from_quat_xyzw,
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
)
from stateful_gru_residual_adapter.model import StatefulGRUResidualPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--trajectory-npz", type=Path, default=DEFAULT_TRAIN_DATASET)
    parser.add_argument("--friction-checkpoint", type=Path, default=DEFAULT_FRICTION_CHECKPOINT)
    parser.add_argument("--friction-point-cloud", type=Path, default=None)
    parser.add_argument("--checkpoint-param-set", choices=("best", "current"), default="best")
    parser.add_argument("--dino-feature-npz", type=Path, default=DEFAULT_DINO_FEATURE_NPZ)
    parser.add_argument("--without-dino", dest="without_dino", action="store_true", default=True)
    parser.add_argument("--with-dino", dest="without_dino", action="store_false")
    parser.add_argument("--dino-max-match-distance", type=float, default=1.0e-5)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=REPO_ROOT / "outputs/stateful_gru_residual/gru2_h16_b32_t64",
    )

    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pretrain-iters", type=int, default=5000)
    parser.add_argument("--closed-loop-iters", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-time-windows", dest="random_time_windows", action="store_true", default=True)
    parser.add_argument("--no-random-time-windows", dest="random_time_windows", action="store_false")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional source trajectory truncation.")
    parser.add_argument("--time-window-source-max-steps", type=int, default=None)
    parser.add_argument("--burn-in-steps", type=int, default=32)
    parser.add_argument("--tbptt-chunk-steps", type=int, default=64)
    parser.add_argument("--closed-loop-residual-gain", type=float, default=0.02)
    parser.add_argument("--normalization-batches", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=25)

    parser.add_argument("--gru-hidden-size", type=int, default=16)
    parser.add_argument("--gru-num-layers", type=int, default=2)
    parser.add_argument("--gru-point-pooling", choices=("mean", "max", "mean-max"), default="mean-max")
    parser.add_argument("--linear-output-scale", type=float, default=None)
    parser.add_argument("--angular-output-scale", type=float, default=None)
    parser.add_argument("--position-output-scale", type=float, default=None)
    parser.add_argument("--yaw-output-scale", type=float, default=None)
    parser.add_argument("--output-head-init", choices=("zero", "small", "default"), default="zero")
    parser.add_argument("--output-head-init-std", type=float, default=1.0e-4)

    parser.add_argument("--loss-linear-velocity-weight", type=float, default=1.0)
    parser.add_argument("--loss-angular-velocity-weight", type=float, default=0.1)
    parser.add_argument("--loss-position-weight", type=float, default=1.0)
    parser.add_argument("--loss-yaw-weight", type=float, default=1.0)
    parser.add_argument("--horizon-gamma", type=float, default=0.95)
    parser.add_argument("--trajectory-position-loss-weight", type=float, default=1.0)
    parser.add_argument("--trajectory-orientation-loss-weight", type=float, default=1.0)
    parser.add_argument("--trajectory-linear-velocity-loss-weight", type=float, default=0.1)
    parser.add_argument("--trajectory-angular-velocity-loss-weight", type=float, default=0.1)
    parser.add_argument("--residual-l2-weight", type=float, default=1.0e-4)
    parser.add_argument("--residual-smoothness-weight", type=float, default=1.0e-4)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="newton_friction_fitting")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default="stateful-gru-residual")
    parser.add_argument("--wandb-mode", type=str, default="online")
    parser.add_argument("--wandb-dir", type=Path, default=None)
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=None)

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
    parser.add_argument("--history-window-steps", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--prediction-window-steps", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument(
        "--residual-output-mode",
        choices=("velocity", "pose", "position", "pose_velocity", "all"),
        default="velocity",
        help=(
            "velocity predicts [dvx_body,dvy_body,domega_z]; pose/position predicts "
            "[dx_body,dy_body,dyaw]; pose_velocity/all predicts all six."
        ),
    )
    return parser.parse_args()


def _init_output_head(model: StatefulGRUResidualPredictor, mode: str, std: float) -> None:
    if mode == "default":
        return
    if mode == "zero":
        torch.nn.init.zeros_(model.output_head.weight)
    elif mode == "small":
        torch.nn.init.normal_(model.output_head.weight, mean=0.0, std=float(std))
    else:
        raise ValueError(f"Unsupported output-head init mode: {mode!r}")
    torch.nn.init.zeros_(model.output_head.bias)


def _static_feature_tensors(diff_scene, args, friction, dino: DinoFeatures | None) -> dict[str, torch.Tensor | None]:
    device = diff_scene.torch_device
    point_count = len(diff_scene.local_surface_points_np)
    values: dict[str, torch.Tensor | None] = {
        "local_points": torch.as_tensor(diff_scene.local_surface_points_np, dtype=torch.float32, device=device).reshape(-1, 3),
        "half_extents": torch.as_tensor(args.box_half_extents, dtype=torch.float32, device=device).reshape(1, 3).clamp_min(1.0e-8),
        "point_friction": torch.as_tensor(friction.full_point_friction, dtype=torch.float32, device=device).reshape(point_count),
        "active_mask": torch.as_tensor(friction.active_contact_mask, dtype=torch.float32, device=device).reshape(point_count),
        "dino_features": None,
        "dino_bottom": None,
    }
    if dino is not None and dino.dim > 0:
        values["dino_features"] = torch.as_tensor(dino.features, dtype=torch.float32, device=device).reshape(point_count, dino.dim)
        values["dino_bottom"] = torch.as_tensor(
            dino.bottom_feature_copied_from_top,
            dtype=torch.float32,
            device=device,
        ).reshape(point_count)
    return values


def _sequence_batch(
    *,
    trajectories: list,
    rng: np.random.Generator,
    args: argparse.Namespace,
    diff_scene,
    buffers,
    initial_body_q: np.ndarray,
    initial_body_qd: np.ndarray,
    window_steps: int,
) -> tuple[list, dict[str, torch.Tensor]]:
    windows, _, _ = sample_window_batch(
        trajectories,
        batch_size=min(int(args.batch_size), len(trajectories)),
        window_steps=int(window_steps),
        rng=rng,
        random_time_windows=bool(args.random_time_windows),
    )
    base = run_open_loop_rollout(
        diff_scene=diff_scene,
        buffers=buffers,
        trajectories=windows,
        args=_args_with(args, steps=int(window_steps)),
        initial_body_q=initial_body_q,
        initial_body_qd=initial_body_qd,
    )
    device = diff_scene.torch_device
    frame_count = int(window_steps) + 1
    tensors = {
        "base_positions": torch.as_tensor(base.positions[:, :frame_count], dtype=torch.float32, device=device),
        "base_quaternions": torch.as_tensor(base.quaternions_xyzw[:, :frame_count], dtype=torch.float32, device=device),
        "base_linear": torch.as_tensor(base.linear_velocity[:, :frame_count], dtype=torch.float32, device=device),
        "base_angular": torch.as_tensor(base.angular_velocity[:, :frame_count], dtype=torch.float32, device=device),
        "target_positions": _stack_window_array(windows, "positions", frame_count=frame_count, device=device),
        "target_quaternions": _stack_window_array(windows, "quaternions_xyzw", frame_count=frame_count, device=device),
        "target_linear": _stack_window_array(windows, "linear_velocity", frame_count=frame_count, device=device),
        "target_angular": _stack_window_array(windows, "angular_velocity", frame_count=frame_count, device=device),
        "step_forces": _step_forces_tensor(windows, step_count=int(window_steps), device=device),
        "point_offsets": _point_offsets_tensor(windows, device=device),
        "step_counts": _trajectory_step_counts_tensor(windows, device=device),
    }
    return windows, tensors


def _predict_step(
    *,
    model: StatefulGRUResidualPredictor,
    hidden: torch.Tensor,
    normalizer: TorchFeatureNormalizer,
    static: dict[str, torch.Tensor | None],
    quaternion: torch.Tensor,
    linear_velocity: torch.Tensor,
    angular_velocity: torch.Tensor,
    force: torch.Tensor,
    step_forces: torch.Tensor,
    step_counts: torch.Tensor,
    point_offsets: torch.Tensor,
    step_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    frame_features = _build_point_feature_frame_torch(
        local_surface_points=static["local_points"],
        box_half_extents=static["half_extents"],
        quaternion_xyzw=quaternion,
        linear_velocity_world=linear_velocity,
        angular_velocity_world=angular_velocity,
        force_world=force,
        point_offset_local=point_offsets,
        point_friction=static["point_friction"],
        active_contact_mask=static["active_mask"],
        dino_features=static["dino_features"],
        dino_bottom_feature_copied_from_top=static["dino_bottom"],
    )
    actions = _future_action_features_torch(
        quaternion_xyzw=quaternion,
        step_forces=step_forces,
        trajectory_step_counts=step_counts,
        point_offset_local=point_offsets,
        step_idx=int(step_idx),
        prediction_window_steps=1,
    )
    normalized_points, normalized_actions = _normalize_pointnet_inputs(
        point_features=frame_features[:, None],
        future_actions=actions,
        point_feature_mean=normalizer.point_feature_mean,
        point_feature_std=normalizer.point_feature_std,
        action_mean=normalizer.action_mean,
        action_std=normalizer.action_std,
    )
    prediction, next_hidden = model.forward_step(normalized_points[:, 0], None, normalized_actions, hidden)
    return prediction[:, 0], next_hidden


def _residual_target(
    tensors: dict[str, torch.Tensor],
    step_idx: int,
    residual_output_mode: str,
) -> torch.Tensor:
    next_idx = int(step_idx) + 1
    target_rotation = quaternion_xyzw_to_matrix_torch(tensors["base_quaternions"][:, next_idx])
    pose_delta_world = tensors["target_positions"][:, next_idx] - tensors["base_positions"][:, next_idx]
    pose_delta_body = torch.einsum("bi,bij->bj", pose_delta_world, target_rotation)
    yaw_delta = _wrap_angle(
        _yaw_from_quat_xyzw(tensors["target_quaternions"][:, next_idx])
        - _yaw_from_quat_xyzw(tensors["base_quaternions"][:, next_idx])
    )
    velocity_delta_world = tensors["target_linear"][:, next_idx] - tensors["base_linear"][:, next_idx]
    velocity_delta_body = torch.einsum("bi,bij->bj", velocity_delta_world, target_rotation)
    omega_delta_z = tensors["target_angular"][:, next_idx, 2] - tensors["base_angular"][:, next_idx, 2]
    pose_target = torch.stack((pose_delta_body[:, 0], pose_delta_body[:, 1], yaw_delta), dim=-1)
    velocity_target = torch.stack(
        (velocity_delta_body[:, 0], velocity_delta_body[:, 1], omega_delta_z),
        dim=-1,
    )
    output_mode = normalize_residual_output_mode(residual_output_mode)
    if output_mode == "pose":
        return pose_target
    if output_mode == "pose_velocity":
        return torch.cat((pose_target, velocity_target), dim=-1)
    return velocity_target


def _train_pretrain_step(
    *,
    model: StatefulGRUResidualPredictor,
    optimizer: torch.optim.Optimizer,
    loss_weights: ResidualLossWeights,
    trajectories: list,
    rng: np.random.Generator,
    args: argparse.Namespace,
    diff_scene,
    buffers,
    initial_body_q: np.ndarray,
    initial_body_qd: np.ndarray,
    normalizer: TorchFeatureNormalizer,
    static: dict[str, torch.Tensor | None],
) -> dict[str, float]:
    burn_in = int(args.burn_in_steps)
    chunk = int(args.tbptt_chunk_steps)
    _, tensors = _sequence_batch(
        trajectories=trajectories,
        rng=rng,
        args=args,
        diff_scene=diff_scene,
        buffers=buffers,
        initial_body_q=initial_body_q,
        initial_body_qd=initial_body_qd,
        window_steps=burn_in + chunk,
    )
    batch_size = int(tensors["base_positions"].shape[0])
    hidden = model.initial_state(batch_size, device=diff_scene.torch_device, dtype=torch.float32)
    model.train()
    with torch.no_grad():
        for step_idx in range(burn_in):
            _, hidden = _predict_step(
                model=model,
                hidden=hidden,
                normalizer=normalizer,
                static=static,
                quaternion=tensors["base_quaternions"][:, step_idx],
                linear_velocity=tensors["base_linear"][:, step_idx],
                angular_velocity=tensors["base_angular"][:, step_idx],
                force=tensors["step_forces"][:, step_idx],
                step_forces=tensors["step_forces"],
                step_counts=tensors["step_counts"],
                point_offsets=tensors["point_offsets"],
                step_idx=step_idx,
            )
    hidden = hidden.detach()

    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for step_idx in range(burn_in, burn_in + chunk):
        prediction, hidden = _predict_step(
            model=model,
            hidden=hidden,
            normalizer=normalizer,
            static=static,
            quaternion=tensors["base_quaternions"][:, step_idx],
            linear_velocity=tensors["base_linear"][:, step_idx],
            angular_velocity=tensors["base_angular"][:, step_idx],
            force=tensors["step_forces"][:, step_idx],
            step_forces=tensors["step_forces"],
            step_counts=tensors["step_counts"],
            point_offsets=tensors["point_offsets"],
            step_idx=step_idx,
        )
        predictions.append(prediction)
        targets.append(_residual_target(tensors, step_idx, str(args.residual_output_mode)))

    prediction_stack = torch.stack(predictions, dim=1)
    target_stack = torch.stack(targets, dim=1)
    loss, metrics_t = residual_velocity_loss(
        prediction_stack,
        target_stack,
        loss_weights,
        str(args.residual_output_mode),
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip_norm))
    optimizer.step()
    metrics = _metrics_to_float(metrics_t)
    metrics.update(
        {
            "train_loss": float(loss.detach().cpu().item()),
            "grad_norm": float(grad_norm.detach().cpu().item()),
            "hidden_norm": float(hidden.detach().norm(dim=-1).mean().cpu().item()),
        }
    )
    return metrics


def _closed_loop_next_state(
    *,
    tensors: dict[str, torch.Tensor],
    step_idx: int,
    quaternion: torch.Tensor,
    residual: torch.Tensor,
    dt: float,
    residual_output_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    next_idx = int(step_idx) + 1
    yaw = _yaw_from_quat_xyzw(quaternion)
    pose_residual, velocity_residual = _split_residual_output(residual, residual_output_mode)
    zero_world = torch.zeros_like(tensors["base_positions"][:, next_idx])
    pose_delta_world = (
        zero_world
        if pose_residual is None
        else _body_planar_to_world(pose_residual[:, :2], yaw)
    )
    velocity_delta_world = (
        zero_world
        if velocity_residual is None
        else _body_planar_to_world(velocity_residual[:, :2], yaw)
    )
    pose_yaw_delta = torch.zeros_like(yaw) if pose_residual is None else pose_residual[:, 2]
    velocity_yaw_delta = torch.zeros_like(yaw) if velocity_residual is None else float(dt) * velocity_residual[:, 2]
    next_position = tensors["base_positions"][:, next_idx] + pose_delta_world + float(dt) * velocity_delta_world
    next_quaternion = _apply_yaw_delta_xyzw(
        tensors["base_quaternions"][:, next_idx],
        pose_yaw_delta + velocity_yaw_delta,
    )
    next_linear = tensors["base_linear"][:, next_idx] + velocity_delta_world
    angular_delta = torch.zeros_like(yaw) if velocity_residual is None else velocity_residual[:, 2]
    next_angular = torch.cat(
        (
            tensors["base_angular"][:, next_idx, :2],
            (tensors["base_angular"][:, next_idx, 2] + angular_delta).reshape(-1, 1),
        ),
        dim=-1,
    )
    return next_position, next_quaternion, next_linear, next_angular


def _train_closed_loop_step(
    *,
    model: StatefulGRUResidualPredictor,
    optimizer: torch.optim.Optimizer,
    trajectories: list,
    rng: np.random.Generator,
    args: argparse.Namespace,
    diff_scene,
    buffers,
    initial_body_q: np.ndarray,
    initial_body_qd: np.ndarray,
    normalizer: TorchFeatureNormalizer,
    static: dict[str, torch.Tensor | None],
) -> dict[str, float]:
    burn_in = int(args.burn_in_steps)
    chunk = int(args.tbptt_chunk_steps)
    _, tensors = _sequence_batch(
        trajectories=trajectories,
        rng=rng,
        args=args,
        diff_scene=diff_scene,
        buffers=buffers,
        initial_body_q=initial_body_q,
        initial_body_qd=initial_body_qd,
        window_steps=burn_in + chunk,
    )
    batch_size = int(tensors["base_positions"].shape[0])
    hidden = model.initial_state(batch_size, device=diff_scene.torch_device, dtype=torch.float32)
    position = tensors["base_positions"][:, 0]
    quaternion = tensors["base_quaternions"][:, 0]
    linear = tensors["base_linear"][:, 0]
    angular = tensors["base_angular"][:, 0]
    gain = float(args.closed_loop_residual_gain)

    model.train()
    with torch.no_grad():
        for step_idx in range(burn_in):
            residual, hidden = _predict_step(
                model=model,
                hidden=hidden,
                normalizer=normalizer,
                static=static,
                quaternion=quaternion,
                linear_velocity=linear,
                angular_velocity=angular,
                force=tensors["step_forces"][:, step_idx],
                step_forces=tensors["step_forces"],
                step_counts=tensors["step_counts"],
                point_offsets=tensors["point_offsets"],
                step_idx=step_idx,
            )
            position, quaternion, linear, angular = _closed_loop_next_state(
                tensors=tensors,
                step_idx=step_idx,
                quaternion=quaternion,
                residual=residual * gain,
                dt=float(args.dt),
                residual_output_mode=str(args.residual_output_mode),
            )

    hidden = hidden.detach()
    position = position.detach()
    quaternion = quaternion.detach()
    linear = linear.detach()
    angular = angular.detach()
    predicted_positions: list[torch.Tensor] = []
    predicted_quaternions: list[torch.Tensor] = []
    predicted_linear: list[torch.Tensor] = []
    predicted_angular: list[torch.Tensor] = []
    residuals: list[torch.Tensor] = []

    for step_idx in range(burn_in, burn_in + chunk):
        residual, hidden = _predict_step(
            model=model,
            hidden=hidden,
            normalizer=normalizer,
            static=static,
            quaternion=quaternion,
            linear_velocity=linear,
            angular_velocity=angular,
            force=tensors["step_forces"][:, step_idx],
            step_forces=tensors["step_forces"],
            step_counts=tensors["step_counts"],
            point_offsets=tensors["point_offsets"],
            step_idx=step_idx,
        )
        applied = residual * gain
        position, quaternion, linear, angular = _closed_loop_next_state(
            tensors=tensors,
            step_idx=step_idx,
            quaternion=quaternion,
            residual=applied,
            dt=float(args.dt),
            residual_output_mode=str(args.residual_output_mode),
        )
        predicted_positions.append(position)
        predicted_quaternions.append(quaternion)
        predicted_linear.append(linear)
        predicted_angular.append(angular)
        residuals.append(applied)

    pred_pos = torch.stack(predicted_positions, dim=1)
    pred_quat = torch.stack(predicted_quaternions, dim=1)
    pred_linear = torch.stack(predicted_linear, dim=1)
    pred_angular = torch.stack(predicted_angular, dim=1)
    residual_stack = torch.stack(residuals, dim=1)
    target_slice = slice(burn_in + 1, burn_in + chunk + 1)
    target_pos = tensors["target_positions"][:, target_slice]
    target_quat = tensors["target_quaternions"][:, target_slice]
    target_linear = tensors["target_linear"][:, target_slice]
    target_angular = tensors["target_angular"][:, target_slice]

    position_loss = _surface_point_position_loss(
        predicted_positions=pred_pos,
        predicted_quaternions=pred_quat,
        target_positions=target_pos,
        target_quaternions=target_quat,
        local_surface_points=static["local_points"],
    )
    orientation_loss = _wrap_angle(_yaw_from_quat_xyzw(pred_quat) - _yaw_from_quat_xyzw(target_quat)).square().mean()
    linear_loss = (pred_linear[..., :2] - target_linear[..., :2]).square().mean()
    angular_loss = (pred_angular[..., 2] - target_angular[..., 2]).square().mean()
    residual_l2 = residual_stack.square().mean()
    if chunk > 1:
        residual_smoothness = (residual_stack[:, 1:] - residual_stack[:, :-1]).square().mean()
    else:
        residual_smoothness = torch.zeros((), dtype=residual_stack.dtype, device=residual_stack.device)
    loss = (
        float(args.trajectory_position_loss_weight) * position_loss
        + float(args.trajectory_orientation_loss_weight) * orientation_loss
        + float(args.trajectory_linear_velocity_loss_weight) * linear_loss
        + float(args.trajectory_angular_velocity_loss_weight) * angular_loss
        + float(args.residual_l2_weight) * residual_l2
        + float(args.residual_smoothness_weight) * residual_smoothness
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip_norm))
    optimizer.step()
    return {
        "train_loss": float(loss.detach().cpu().item()),
        "trajectory_position_loss": float(position_loss.detach().cpu().item()),
        "trajectory_orientation_loss": float(orientation_loss.detach().cpu().item()),
        "trajectory_linear_velocity_loss": float(linear_loss.detach().cpu().item()),
        "trajectory_angular_velocity_loss": float(angular_loss.detach().cpu().item()),
        "loss_residual_l2": float(residual_l2.detach().cpu().item()),
        "loss_residual_smoothness": float(residual_smoothness.detach().cpu().item()),
        "pred_residual_abs_mean": float(residual_stack.detach().abs().mean().cpu().item()),
        "grad_norm": float(grad_norm.detach().cpu().item()),
        "hidden_norm": float(hidden.detach().norm(dim=-1).mean().cpu().item()),
    }


def _save_checkpoint(
    *,
    path: Path,
    model: StatefulGRUResidualPredictor,
    base_metadata: dict,
    metadata_updates: dict,
    normalizer: FeatureNormalizer,
    diff_scene,
    friction,
    dino: DinoFeatures | None,
) -> None:
    metadata = dict(base_metadata)
    metadata.update(metadata_updates)
    save_adapter_checkpoint(
        checkpoint_path=path,
        model=model,
        metadata=metadata,
        normalizer=normalizer,
        local_surface_points=diff_scene.local_surface_points_np,
        full_point_friction=friction.full_point_friction,
        active_contact_mask=friction.active_contact_mask,
        dino_features=None if dino is None else dino.features,
        dino_bottom_feature_copied_from_top=None if dino is None else dino.bottom_feature_copied_from_top,
    )


def main() -> None:
    args = parse_args()
    start_time = time.time()
    rng = np.random.default_rng(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    if int(args.burn_in_steps) < 0 or int(args.tbptt_chunk_steps) < 1:
        raise ValueError("--burn-in-steps must be non-negative and --tbptt-chunk-steps must be positive")
    args.history_window_steps = 1
    args.prediction_window_steps = 1
    args.residual_output_mode = normalize_residual_output_mode(args.residual_output_mode)
    if args.residual_output_mode not in {"velocity", "pose", "pose_velocity"}:
        raise ValueError("--residual-output-mode must resolve to velocity, pose, or pose_velocity")
    max_required_steps = int(args.burn_in_steps) + int(args.tbptt_chunk_steps)
    args.steps = max_required_steps
    args.batch_capacity = max(int(args.batch_size), 1)

    configured_point_cloud = maybe_configure_scene_from_point_cloud(args)
    collection = load_mujoco_trajectories(
        trajectory_npz_path=args.trajectory_npz,
        max_steps=_load_max_steps(args),
        max_trajectories=args.max_trajectories,
    )
    if collection.max_steps < max_required_steps:
        raise ValueError(f"Dataset max_steps={collection.max_steps} is shorter than required {max_required_steps}")
    args.dt = float(collection.trajectories[0].timestep)
    train_trajectories = _eligible_trajectories(collection.trajectories, min_steps=max_required_steps)
    print(
        f"loaded trajectories={len(train_trajectories)} burn_in={args.burn_in_steps} "
        f"tbptt_chunk={args.tbptt_chunk_steps} dt={args.dt:.6g}",
        flush=True,
    )

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
    if bool(args.without_dino):
        dino = None
    else:
        from pointnet_residual_adapter.features import load_aligned_dino_features

        if args.dino_feature_npz is None or not args.dino_feature_npz.exists():
            raise FileNotFoundError(f"--dino-feature-npz does not exist: {args.dino_feature_npz}")
        dino = load_aligned_dino_features(
            args.dino_feature_npz,
            diff_scene.local_surface_points_np,
            max_match_distance=float(args.dino_max_match_distance),
        )

    buffers = build_rollout_buffers(
        device=device,
        batch_capacity=int(args.batch_capacity),
        step_capacity=max_required_steps,
        point_count=len(diff_scene.local_surface_points_np),
        full_point_friction=friction.full_point_friction,
    )
    normalization_args = _args_with(args, steps=1)
    normalizer, target_samples, point_feature_dim = _collect_normalization(
        train_trajectories=train_trajectories,
        rng=rng,
        args=normalization_args,
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
    normalizer_torch = normalizer_to_torch(normalizer, device=diff_scene.torch_device)
    model = StatefulGRUResidualPredictor(
        point_feature_dim=point_feature_dim,
        prediction_window_steps=1,
        hidden_size=int(args.gru_hidden_size),
        num_layers=int(args.gru_num_layers),
        point_pooling=str(args.gru_point_pooling),
        linear_output_scale=linear_scale,
        angular_output_scale=angular_scale,
        position_output_scale=position_scale,
        yaw_output_scale=yaw_scale,
        residual_output_mode=str(args.residual_output_mode),
    ).to(diff_scene.torch_device)
    _init_output_head(model, str(args.output_head_init), float(args.output_head_init_std))
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
    static = _static_feature_tensors(diff_scene, args, friction, dino)

    experiment_dir = Path(args.experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    name = experiment_dir.name
    best_pretrain_path = experiment_dir / f"{name}_best_pretrain.pt"
    best_closed_loop_path = experiment_dir / f"{name}_best_closed_loop.pt"
    canonical_path = experiment_dir / f"{name}.pt"
    last_path = experiment_dir / f"{name}_last.pt"
    metrics_path = experiment_dir / f"{name}_metrics.json"

    base_metadata = {
        "adapter_architecture": "stateful_gru_residual_adapter",
        "history_window_steps": 1,
        "prediction_window_steps": 1,
        "point_feature_schema": point_feature_schema(0 if dino is None else dino.dim),
        "point_feature_dim": int(point_feature_dim),
        "action_feature_schema": ACTION_FEATURE_SCHEMA,
        "gru_num_layers": int(args.gru_num_layers),
        "gru_hidden_size": int(args.gru_hidden_size),
        "gru_point_pooling": str(args.gru_point_pooling),
        "stateful_rollout": True,
        "stateful_reset_interval": 0,
        "burn_in_steps": int(args.burn_in_steps),
        "tbptt_chunk_steps": int(args.tbptt_chunk_steps),
        "segment_steps": int(max_required_steps),
        "closed_loop_loss_mode": "trajectory",
        "residual_output_mode": str(args.residual_output_mode),
        "pointnet_residual_gain": float(args.closed_loop_residual_gain),
        "linear_output_scale": float(linear_scale),
        "angular_output_scale": float(angular_scale),
        "position_output_scale": float(position_scale),
        "yaw_output_scale": float(yaw_scale),
        "output_scales": {
            "position": float(position_scale),
            "yaw": float(yaw_scale),
            "linear": float(linear_scale),
            "angular_z": float(angular_scale),
        },
        "dino_feature_npz": None if dino is None else str(dino.path.resolve()),
        "dino_feature_dim": 0 if dino is None else int(dino.dim),
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
        "training_dataset": str(args.trajectory_npz.resolve()),
        "train_trajectories": len(train_trajectories),
        "random_time_windows": bool(args.random_time_windows),
        "trajectory_loss_weights": {
            "position": float(args.trajectory_position_loss_weight),
            "orientation": float(args.trajectory_orientation_loss_weight),
            "linear_velocity": float(args.trajectory_linear_velocity_loss_weight),
            "angular_velocity": float(args.trajectory_angular_velocity_loss_weight),
            "residual_l2": float(args.residual_l2_weight),
            "residual_smoothness": float(args.residual_smoothness_weight),
        },
        "dt": float(args.dt),
        "seed": int(args.seed),
        "args": _json_safe_args(args),
    }
    wandb_run = _init_wandb(args, experiment_dir=experiment_dir, metadata=base_metadata)
    total_iters = int(args.pretrain_iters) + int(args.closed_loop_iters)
    if total_iters < 1:
        raise ValueError("At least one of --pretrain-iters or --closed-loop-iters must be positive")
    best_pretrain = float("inf")
    best_closed_loop = float("inf")
    history: list[dict[str, float | str]] = []

    for iteration in range(1, total_iters + 1):
        if iteration <= int(args.pretrain_iters):
            phase = "pretrain"
            metrics = _train_pretrain_step(
                model=model,
                optimizer=optimizer,
                loss_weights=loss_weights,
                trajectories=train_trajectories,
                rng=rng,
                args=args,
                diff_scene=diff_scene,
                buffers=buffers,
                initial_body_q=initial_body_q,
                initial_body_qd=initial_body_qd,
                normalizer=normalizer_torch,
                static=static,
            )
            if metrics["train_loss"] < best_pretrain:
                best_pretrain = metrics["train_loss"]
                _save_checkpoint(
                    path=best_pretrain_path,
                    model=model,
                    base_metadata=base_metadata,
                    metadata_updates={
                        "canonical_checkpoint_role": "best_pretrain",
                        "best_phase": "pretrain",
                        "best_iteration": iteration,
                        "best_loss": best_pretrain,
                    },
                    normalizer=normalizer,
                    diff_scene=diff_scene,
                    friction=friction,
                    dino=dino,
                )
        else:
            phase = "closed_loop"
            metrics = _train_closed_loop_step(
                model=model,
                optimizer=optimizer,
                trajectories=train_trajectories,
                rng=rng,
                args=args,
                diff_scene=diff_scene,
                buffers=buffers,
                initial_body_q=initial_body_q,
                initial_body_qd=initial_body_qd,
                normalizer=normalizer_torch,
                static=static,
            )
            if metrics["train_loss"] < best_closed_loop:
                best_closed_loop = metrics["train_loss"]
                updates = {
                    "canonical_checkpoint_role": "best_closed_loop",
                    "best_phase": "closed_loop",
                    "best_iteration": iteration,
                    "best_loss": best_closed_loop,
                    "best_closed_loop_gain": float(args.closed_loop_residual_gain),
                    "best_closed_loop_horizon_steps": int(args.tbptt_chunk_steps),
                }
                for path in (best_closed_loop_path, canonical_path):
                    _save_checkpoint(
                        path=path,
                        model=model,
                        base_metadata=base_metadata,
                        metadata_updates=updates,
                        normalizer=normalizer,
                        diff_scene=diff_scene,
                        friction=friction,
                        dino=dino,
                    )

        if not np.isfinite(metrics["train_loss"]):
            raise FloatingPointError(f"Non-finite {phase} loss at iteration {iteration}: {metrics['train_loss']}")
        metrics["iteration"] = float(iteration)
        metrics["phase_is_closed_loop"] = 1.0 if phase == "closed_loop" else 0.0
        history.append({"phase": phase, **metrics})
        if wandb_run is not None:
            payload = _build_wandb_log_payload(metrics)
            payload["curriculum/phase_is_closed_loop"] = metrics["phase_is_closed_loop"]
            payload["stateful/hidden_norm"] = float(metrics["hidden_norm"])
            for metric_name, metric_value in metrics.items():
                if metric_name in {"iteration", "phase_is_closed_loop", "hidden_norm"}:
                    continue
                payload.setdefault(f"train/{metric_name}", float(metric_value))
            wandb_run.log(payload, step=iteration)
        if iteration == total_iters or (
            int(args.checkpoint_every) > 0 and iteration % int(args.checkpoint_every) == 0
        ):
            _save_checkpoint(
                path=last_path,
                model=model,
                base_metadata=base_metadata,
                metadata_updates={
                    "canonical_checkpoint_role": "last",
                    "last_iteration": iteration,
                    "last_phase": phase,
                    "last_closed_loop_gain": float(args.closed_loop_residual_gain),
                    "last_closed_loop_horizon_steps": int(args.tbptt_chunk_steps),
                    "best_pretrain_loss": best_pretrain,
                    "best_closed_loop_loss": best_closed_loop,
                },
                normalizer=normalizer,
                diff_scene=diff_scene,
                friction=friction,
                dino=dino,
            )
        if int(args.log_every) > 0 and (iteration == 1 or iteration % int(args.log_every) == 0):
            print(
                f"iter={iteration:05d}/{total_iters:05d} phase={phase} "
                f"loss={metrics['train_loss']:.6g} grad={metrics['grad_norm']:.6g} "
                f"hidden_norm={metrics['hidden_norm']:.6g} elapsed={time.time() - start_time:.1f}s",
                flush=True,
            )

    if int(args.closed_loop_iters) == 0 and best_pretrain_path.exists():
        payload = torch.load(best_pretrain_path, map_location="cpu", weights_only=False)
        torch.save(payload, canonical_path)
    save_json(
        metrics_path,
        {
            "best_pretrain_loss": best_pretrain,
            "best_closed_loop_loss": best_closed_loop,
            "history": history,
        },
    )
    if wandb_run is not None:
        wandb_run.summary["best_pretrain_loss"] = float(best_pretrain)
        wandb_run.summary["best_closed_loop_loss"] = float(best_closed_loop)
        wandb_run.finish()
    print(f"best_pretrain_checkpoint={best_pretrain_path.resolve()}", flush=True)
    print(
        f"best_closed_loop_checkpoint={best_closed_loop_path.resolve() if best_closed_loop_path.exists() else 'not_created'}",
        flush=True,
    )
    print(f"canonical_checkpoint={canonical_path.resolve()}", flush=True)
    print(f"last_checkpoint={last_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
