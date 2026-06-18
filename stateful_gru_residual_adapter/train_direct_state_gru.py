from __future__ import annotations

"""Train a stateful GRU that directly predicts planar state trajectories."""

import argparse
from dataclasses import dataclass
import sys
import time
from pathlib import Path
from typing import Any

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

from pointnet_residual_adapter.friction import (
    active_indices_from_trajectories,
    maybe_configure_scene_from_point_cloud,
    resolve_friction_conditioning,
)
from pointnet_residual_adapter.newton_rollout import build_rollout_buffers, run_open_loop_rollout
from pointnet_residual_adapter.train_curriculum_pointnet_residual import _args_with, _eligible_trajectories, _wrap_angle, _yaw_from_quat_xyzw
from pointnet_residual_adapter.train_supervised_pointnet_residual import (
    DEFAULT_FRICTION_CHECKPOINT,
    DEFAULT_TRAIN_DATASET,
    _init_wandb,
    _json_safe_args,
    _load_max_steps,
)
from pointnet_residual_adapter.checkpoints import save_json
from stateful_gru_residual_adapter.direct_state_model import DIRECT_STATE_DIM, StatefulGRUDirectStatePredictor


STATE_SCHEMA = (
    "position_world_x",
    "position_world_y",
    "yaw",
    "linear_velocity_world_x",
    "linear_velocity_world_y",
    "angular_velocity_world_z",
)


@dataclass(frozen=True)
class DirectStateNormalizer:
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--trajectory-npz", type=Path, default=DEFAULT_TRAIN_DATASET)
    parser.add_argument("--friction-checkpoint", type=Path, default=DEFAULT_FRICTION_CHECKPOINT)
    parser.add_argument("--friction-point-cloud", type=Path, default=None)
    parser.add_argument("--checkpoint-param-set", choices=("best", "current"), default="best")
    parser.add_argument("--experiment-dir", type=Path, default=REPO_ROOT / "outputs/stateful_gru_direct_state/gru2_h16")
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--opt-iters", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None, help="Optional source trajectory truncation.")
    parser.add_argument("--time-window-source-max-steps", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--random-time-windows", action="store_false", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--normalization-batches", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=25)

    parser.add_argument("--gru-hidden-size", type=int, default=16)
    parser.add_argument("--gru-num-layers", type=int, default=2)
    parser.add_argument("--loss-position-weight", type=float, default=1.0)
    parser.add_argument("--loss-yaw-weight", type=float, default=1.0)
    parser.add_argument("--loss-linear-velocity-weight", type=float, default=0.1)
    parser.add_argument("--loss-angular-velocity-weight", type=float, default=0.1)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="newton_friction_fitting")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default="stateful-gru-direct-state")
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
    return parser.parse_args()


def _state_tensor(
    *,
    positions: torch.Tensor,
    quaternions_xyzw: torch.Tensor,
    linear_velocity: torch.Tensor,
    angular_velocity: torch.Tensor,
) -> torch.Tensor:
    yaw = _yaw_from_quat_xyzw(quaternions_xyzw)
    return torch.stack(
        (
            positions[..., 0],
            positions[..., 1],
            yaw,
            linear_velocity[..., 0],
            linear_velocity[..., 1],
            angular_velocity[..., 2],
        ),
        dim=-1,
    )


def _sample_trajectory_batch(
    trajectories: list,
    *,
    batch_size: int,
    rng: np.random.Generator,
) -> list:
    if not trajectories:
        raise ValueError("Cannot sample from an empty trajectory list")
    count = min(max(int(batch_size), 1), len(trajectories))
    indices = rng.choice(len(trajectories), size=count, replace=False)
    return [trajectories[int(idx)] for idx in indices]


def _padded_target_tensors(trajectories: list, *, max_steps: int, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(trajectories)
    target = torch.zeros((batch_size, int(max_steps), DIRECT_STATE_DIM), dtype=torch.float32, device=device)
    mask = torch.zeros((batch_size, int(max_steps)), dtype=torch.bool, device=device)
    for batch_idx, trajectory in enumerate(trajectories):
        steps = min(int(trajectory.num_steps), int(max_steps))
        if steps <= 0:
            continue
        positions = torch.as_tensor(trajectory.positions[1 : steps + 1], dtype=torch.float32, device=device)
        quaternions = torch.as_tensor(trajectory.quaternions_xyzw[1 : steps + 1], dtype=torch.float32, device=device)
        linear = torch.as_tensor(trajectory.linear_velocity[1 : steps + 1], dtype=torch.float32, device=device)
        angular = torch.as_tensor(trajectory.angular_velocity[1 : steps + 1], dtype=torch.float32, device=device)
        target[batch_idx, :steps] = _state_tensor(
            positions=positions,
            quaternions_xyzw=quaternions,
            linear_velocity=linear,
            angular_velocity=angular,
        )
        mask[batch_idx, :steps] = True
    return target, mask


def _prepare_direct_state_batch(
    *,
    trajectories: list,
    args: argparse.Namespace,
    diff_scene,
    buffers,
    initial_body_q: np.ndarray,
    initial_body_qd: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_steps = max(int(trajectory.num_steps) for trajectory in trajectories)
    base = run_open_loop_rollout(
        diff_scene=diff_scene,
        buffers=buffers,
        trajectories=trajectories,
        args=_args_with(args, steps=max_steps),
        initial_body_q=initial_body_q,
        initial_body_qd=initial_body_qd,
    )
    device = diff_scene.torch_device
    base_positions = torch.as_tensor(base.positions[:, 1 : max_steps + 1], dtype=torch.float32, device=device)
    base_quaternions = torch.as_tensor(base.quaternions_xyzw[:, 1 : max_steps + 1], dtype=torch.float32, device=device)
    base_linear = torch.as_tensor(base.linear_velocity[:, 1 : max_steps + 1], dtype=torch.float32, device=device)
    base_angular = torch.as_tensor(base.angular_velocity[:, 1 : max_steps + 1], dtype=torch.float32, device=device)
    inputs = _state_tensor(
        positions=base_positions,
        quaternions_xyzw=base_quaternions,
        linear_velocity=base_linear,
        angular_velocity=base_angular,
    )
    targets, mask = _padded_target_tensors(trajectories, max_steps=max_steps, device=device)
    return inputs, targets, mask


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(dtype=values.dtype)
    while mask_f.ndim < values.ndim:
        mask_f = mask_f.unsqueeze(-1)
    return (values * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def _direct_state_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    diff = prediction - target
    yaw_error = _wrap_angle(diff[..., 2])
    position_loss = _masked_mean(diff[..., :2].square(), mask)
    yaw_loss = _masked_mean(yaw_error.square(), mask)
    linear_velocity_loss = _masked_mean(diff[..., 3:5].square(), mask)
    angular_velocity_loss = _masked_mean(diff[..., 5].square(), mask)
    loss = (
        float(args.loss_position_weight) * position_loss
        + float(args.loss_yaw_weight) * yaw_loss
        + float(args.loss_linear_velocity_weight) * linear_velocity_loss
        + float(args.loss_angular_velocity_weight) * angular_velocity_loss
    )
    final_prediction, final_target = _final_valid_states(prediction, target, mask)
    final_diff = final_prediction - final_target
    final_loss = (
        final_diff[..., :2].square().mean()
        + _wrap_angle(final_diff[..., 2]).square().mean()
        + final_diff[..., 3:5].square().mean()
        + final_diff[..., 5].square().mean()
    )
    metrics = {
        "loss_position_xy": position_loss,
        "loss_yaw": yaw_loss,
        "loss_linear_velocity_xy": linear_velocity_loss,
        "loss_angular_velocity_z": angular_velocity_loss,
        "final_state_loss": final_loss,
        "pred_position_abs_mean": _masked_mean(prediction[..., :2].abs(), mask),
        "target_position_abs_mean": _masked_mean(target[..., :2].abs(), mask),
    }
    return loss, metrics


def _final_valid_states(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = mask.to(dtype=torch.long).sum(dim=1).clamp_min(1)
    batch_indices = torch.arange(prediction.shape[0], dtype=torch.long, device=prediction.device)
    final_indices = lengths - 1
    return prediction[batch_indices, final_indices], target[batch_indices, final_indices]


def _collect_normalizer(
    *,
    trajectories: list,
    rng: np.random.Generator,
    args: argparse.Namespace,
    diff_scene,
    buffers,
    initial_body_q: np.ndarray,
    initial_body_qd: np.ndarray,
) -> DirectStateNormalizer:
    input_values: list[torch.Tensor] = []
    target_values: list[torch.Tensor] = []
    for _ in range(max(int(args.normalization_batches), 1)):
        batch = _sample_trajectory_batch(trajectories, batch_size=int(args.batch_size), rng=rng)
        inputs, targets, mask = _prepare_direct_state_batch(
            trajectories=batch,
            args=args,
            diff_scene=diff_scene,
            buffers=buffers,
            initial_body_q=initial_body_q,
            initial_body_qd=initial_body_qd,
        )
        input_values.append(inputs[mask].detach())
        target_values.append(targets[mask].detach())
    all_inputs = torch.cat(input_values, dim=0)
    all_targets = torch.cat(target_values, dim=0)
    input_std, input_mean = torch.std_mean(all_inputs, dim=0, correction=0)
    target_std, target_mean = torch.std_mean(all_targets, dim=0, correction=0)
    return DirectStateNormalizer(
        input_mean=input_mean.detach().cpu().numpy().astype(np.float32),
        input_std=input_std.clamp_min(1.0e-6).detach().cpu().numpy().astype(np.float32),
        target_mean=target_mean.detach().cpu().numpy().astype(np.float32),
        target_std=target_std.clamp_min(1.0e-6).detach().cpu().numpy().astype(np.float32),
    )


def _normalizer_tensors(normalizer: DirectStateNormalizer, device: torch.device | str) -> dict[str, torch.Tensor]:
    return {
        "input_mean": torch.as_tensor(normalizer.input_mean, dtype=torch.float32, device=device).reshape(1, 1, -1),
        "input_std": torch.as_tensor(normalizer.input_std, dtype=torch.float32, device=device).reshape(1, 1, -1),
        "target_mean": torch.as_tensor(normalizer.target_mean, dtype=torch.float32, device=device).reshape(1, 1, -1),
        "target_std": torch.as_tensor(normalizer.target_std, dtype=torch.float32, device=device).reshape(1, 1, -1),
    }


def _save_checkpoint(
    *,
    checkpoint_path: Path,
    model: StatefulGRUDirectStatePredictor,
    metadata: dict[str, Any],
    normalizer: DirectStateNormalizer,
    friction,
    local_surface_points: np.ndarray,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata,
        "direct_state_normalizer": {
            "input_mean": normalizer.input_mean,
            "input_std": normalizer.input_std,
            "target_mean": normalizer.target_mean,
            "target_std": normalizer.target_std,
        },
        "local_surface_points": np.asarray(local_surface_points, dtype=np.float32),
        "full_point_friction": np.asarray(friction.full_point_friction, dtype=np.float32),
        "active_contact_mask": np.asarray(friction.active_contact_mask, dtype=bool),
    }
    torch.save(payload, checkpoint_path)
    save_json(checkpoint_path.with_name(f"{checkpoint_path.stem}_metadata.json"), metadata)


def _metrics_to_float(metrics: dict[str, torch.Tensor | float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            result[key] = float(value.detach().cpu().item())
        else:
            result[key] = float(value)
    return result


def main() -> None:
    args = parse_args()
    start_time = time.time()
    rng = np.random.default_rng(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    configured_point_cloud = maybe_configure_scene_from_point_cloud(args)
    collection = load_mujoco_trajectories(
        trajectory_npz_path=args.trajectory_npz,
        max_steps=_load_max_steps(args),
        max_trajectories=args.max_trajectories,
    )
    train_trajectories = _eligible_trajectories(collection.trajectories, min_steps=1)
    args.dt = float(train_trajectories[0].timestep)
    args.steps = collection.max_steps
    args.batch_capacity = max(int(args.batch_size), 1)
    print(
        f"loaded full trajectories={len(train_trajectories)} max_steps={collection.max_steps} dt={args.dt:.6g}",
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
    buffers = build_rollout_buffers(
        device=device,
        batch_capacity=int(args.batch_capacity),
        step_capacity=int(collection.max_steps),
        point_count=len(diff_scene.local_surface_points_np),
        full_point_friction=friction.full_point_friction,
    )
    print(f"collecting direct-state normalization batches={max(int(args.normalization_batches), 1)}", flush=True)
    normalizer = _collect_normalizer(
        trajectories=train_trajectories,
        rng=rng,
        args=args,
        diff_scene=diff_scene,
        buffers=buffers,
        initial_body_q=initial_body_q,
        initial_body_qd=initial_body_qd,
    )
    normalizer_t = _normalizer_tensors(normalizer, diff_scene.torch_device)
    model = StatefulGRUDirectStatePredictor(
        input_dim=DIRECT_STATE_DIM,
        state_dim=DIRECT_STATE_DIM,
        hidden_size=int(args.gru_hidden_size),
        num_layers=int(args.gru_num_layers),
    ).to(diff_scene.torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate))

    experiment_dir = Path(args.experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    name = experiment_dir.name
    best_path = experiment_dir / f"{name}.pt"
    last_path = experiment_dir / f"{name}_last.pt"
    metrics_path = experiment_dir / f"{name}_metrics.json"

    base_metadata = {
        "adapter_architecture": "stateful_gru_direct_state_adapter",
        "state_schema": STATE_SCHEMA,
        "state_dim": DIRECT_STATE_DIM,
        "input_source": "newton_next_state_planar_world",
        "target_source": "mujoco_next_state_planar_world",
        "training_sequence": "full_trajectory",
        "loss_aggregation": "masked_full_trajectory_mean",
        "gru_num_layers": int(args.gru_num_layers),
        "gru_hidden_size": int(args.gru_hidden_size),
        "training_dataset": str(args.trajectory_npz.resolve()),
        "train_trajectories": len(train_trajectories),
        "max_steps": int(collection.max_steps),
        "friction_checkpoint": None if args.friction_checkpoint is None else str(args.friction_checkpoint.resolve()),
        "friction_point_cloud": None if args.friction_point_cloud is None else str(args.friction_point_cloud.resolve()),
        "friction_source_type": friction.source_type,
        "friction_parameterization": friction.parameterization,
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
        "loss_weights": {
            "position": float(args.loss_position_weight),
            "yaw": float(args.loss_yaw_weight),
            "linear_velocity": float(args.loss_linear_velocity_weight),
            "angular_velocity": float(args.loss_angular_velocity_weight),
        },
        "dt": float(args.dt),
        "seed": int(args.seed),
        "args": _json_safe_args(args),
    }
    wandb_run = _init_wandb(args, experiment_dir=experiment_dir, metadata=base_metadata)
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    for iteration in range(1, int(args.opt_iters) + 1):
        model.train()
        batch = _sample_trajectory_batch(train_trajectories, batch_size=int(args.batch_size), rng=rng)
        inputs, targets, mask = _prepare_direct_state_batch(
            trajectories=batch,
            args=args,
            diff_scene=diff_scene,
            buffers=buffers,
            initial_body_q=initial_body_q,
            initial_body_qd=initial_body_qd,
        )
        normalized_inputs = (inputs - normalizer_t["input_mean"]) / normalizer_t["input_std"]
        prediction_normalized, hidden = model(normalized_inputs)
        prediction = prediction_normalized * normalizer_t["target_std"] + normalizer_t["target_mean"]
        loss, metrics_t = _direct_state_loss(prediction, targets, mask, args)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite direct-state loss at iteration {iteration}: {loss.detach().cpu().item()}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip_norm))
        optimizer.step()
        metrics = _metrics_to_float(metrics_t)
        metrics.update(
            {
                "iteration": float(iteration),
                "train_loss": float(loss.detach().cpu().item()),
                "grad_norm": float(grad_norm.detach().cpu().item()),
                "hidden_norm": float(hidden.detach().norm(dim=-1).mean().cpu().item()),
                "valid_steps": float(mask.sum().detach().cpu().item()),
            }
        )
        history.append(metrics)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "progress/iteration": float(iteration),
                    "train/loss_total": metrics["train_loss"],
                    "train/loss_position_xy": metrics["loss_position_xy"],
                    "train/loss_yaw": metrics["loss_yaw"],
                    "train/loss_linear_velocity_xy": metrics["loss_linear_velocity_xy"],
                    "train/loss_angular_velocity_z": metrics["loss_angular_velocity_z"],
                    "train/final_state_loss": metrics["final_state_loss"],
                    "optim/grad_norm": metrics["grad_norm"],
                    "stateful/hidden_norm": metrics["hidden_norm"],
                },
                step=iteration,
            )
        if metrics["train_loss"] < best_loss:
            best_loss = metrics["train_loss"]
            metadata = dict(base_metadata)
            metadata.update({"canonical_checkpoint_role": "best", "best_iteration": iteration, "best_loss": best_loss})
            _save_checkpoint(
                checkpoint_path=best_path,
                model=model,
                metadata=metadata,
                normalizer=normalizer,
                friction=friction,
                local_surface_points=diff_scene.local_surface_points_np,
            )
        if iteration == int(args.opt_iters) or (
            int(args.checkpoint_every) > 0 and iteration % int(args.checkpoint_every) == 0
        ):
            metadata = dict(base_metadata)
            metadata.update(
                {
                    "canonical_checkpoint_role": "last",
                    "last_iteration": iteration,
                    "best_loss": best_loss,
                }
            )
            _save_checkpoint(
                checkpoint_path=last_path,
                model=model,
                metadata=metadata,
                normalizer=normalizer,
                friction=friction,
                local_surface_points=diff_scene.local_surface_points_np,
            )
        if int(args.log_every) > 0 and (iteration == 1 or iteration % int(args.log_every) == 0):
            print(
                f"iter={iteration:05d}/{int(args.opt_iters):05d} whole_traj_loss={metrics['train_loss']:.6g} "
                f"pos={metrics['loss_position_xy']:.6g} yaw={metrics['loss_yaw']:.6g} "
                f"vel={metrics['loss_linear_velocity_xy']:.6g} final={metrics['final_state_loss']:.6g} "
                f"grad={metrics['grad_norm']:.6g} elapsed={time.time() - start_time:.1f}s",
                flush=True,
            )

    save_json(metrics_path, {"best_loss": best_loss, "history": history})
    if wandb_run is not None:
        wandb_run.summary["best_loss"] = float(best_loss)
        wandb_run.summary["best_checkpoint"] = str(best_path.resolve())
        wandb_run.summary["last_checkpoint"] = str(last_path.resolve())
        wandb_run.finish()
    print(f"best_checkpoint={best_path.resolve()}", flush=True)
    print(f"last_checkpoint={last_path.resolve()}", flush=True)
    print(f"metrics={metrics_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
