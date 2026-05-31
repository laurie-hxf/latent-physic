from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import warp as wp


REPO_ROOT = Path(__file__).resolve().parents[1]
NEWTON_DIR = REPO_ROOT / "newton"
for _path in (REPO_ROOT, NEWTON_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from fit_mujoco_contact_point_friction_kernels import (  # noqa: E402
    apply_batched_external_and_surface_point_forces_trajectory_kernel,
    compute_batched_contact_weighted_masses_kernel,
)
from fit_mujoco_contact_point_friction_runtime import (  # noqa: E402
    assert_array_finite,
    log_message,
    reset_scene_states,
    sample_training_batch_indices,
    set_batched_box_initial_states_kernel,
)
from fit_mujoco_contact_point_friction_params import (  # noqa: E402
    compute_piecewise_side_ids,
    expand_optimizer_params_to_active,
    build_optimizer_param_positions,
    initialize_optimizer_params_np,
    project_base_delta_optimizer_params_np,
    sample_training_time_windows,
    validate_friction_parameterization,
)
from mujoco_contact_friction_fit_utils import (  # noqa: E402
    MujocoTrajectory,
    load_mujoco_trajectories,
    slice_mujoco_trajectory_time_window,
)
from mujoco_contact_friction_fit_wandb import build_wandb_log_payload, init_wandb  # noqa: E402
from newton_surface_points_diff_demo import GRAVITY_MAGNITUDE, build_diff_scene  # noqa: E402
from replay_mujoco_contact_friction_trajectory import (  # noqa: E402
    build_reference_to_scene_index,
    infer_base_point_friction,
    infer_box_half_extents_and_spacing,
    load_checkpoint_parameters,
    load_contact_friction_point_cloud,
)

from residual_dynamics_adapter.kernels import (  # noqa: E402
    HIDDEN0_DIM,
    HIDDEN1_DIM,
    HIDDEN2_DIM,
    INPUT_DIM,
    OUTPUT_DIM,
    accumulate_optimizer_mu_features_kernel,
    accumulate_grad_norm_kernel,
    accumulate_residual_frame_loss_kernel,
    accumulate_residual_regularization_kernel,
    adam_update_kernel,
    apply_residual_planar_dynamics_kernel,
    clip_optimizer_params_kernel,
    combine_residual_loss_components_kernel,
    project_base_delta_optimizer_params_kernel,
    residual_mlp_layer0_kernel,
    residual_mlp_layer1_kernel,
    residual_mlp_layer2_kernel,
    residual_mlp_output_kernel,
    scatter_optimizer_point_friction_kernel,
    sum_loss_kernel,
)


DEFAULT_TRAIN_DATASET = (
    REPO_ROOT
    / "mujoco/outputs/rotation_friction_diagnostics_l0p20_r0p50_2000/"
    / "same_mean_split_left_0p20_right_0p50/same_mean_split_left_0p20_right_0p50.npz"
)
EXPERIMENT_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
TIMESTAMPED_EXPERIMENT_NAME_RE = re.compile(r"^\d{8}_\d{6}_.+")


def _current_experiment_timestamp() -> str:
    return datetime.now().astimezone().strftime(EXPERIMENT_TIMESTAMP_FORMAT)


def _apply_experiment_dir_timestamp(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    args.experiment_timestamp = None
    args.requested_experiment_dir = args.experiment_dir
    if not args.timestamp_experiment_dir:
        return

    experiment_dir = args.experiment_dir
    experiment_name = experiment_dir.name
    if not experiment_name:
        parser.error("--experiment-dir must include a directory name.")

    if TIMESTAMPED_EXPERIMENT_NAME_RE.match(experiment_name):
        args.experiment_timestamp = experiment_name[:15]
        return

    timestamp = args.experiment_dir_timestamp or _current_experiment_timestamp()
    if not timestamp or "/" in timestamp or "\\" in timestamp:
        parser.error("--experiment-dir-timestamp must be a non-empty path-safe string.")

    args.experiment_timestamp = timestamp
    args.experiment_dir = experiment_dir.with_name(f"{timestamp}_{experiment_name}")
    if (
        args.wandb_run_name is not None
        and not TIMESTAMPED_EXPERIMENT_NAME_RE.match(args.wandb_run_name)
    ):
        args.wandb_run_name = f"{timestamp}_{args.wandb_run_name}"


@dataclass
class ResidualRolloutBuffers:
    batch_capacity: int
    step_capacity: int
    frame_capacity: int
    full_point_friction: wp.array
    contact_weighted_masses: wp.array
    contact_weighted_mass_total: wp.array
    step_forces: wp.array
    force_point_offsets_local: wp.array
    initial_positions: wp.array
    initial_quaternions: wp.array
    initial_linear_velocity: wp.array
    initial_angular_velocity: wp.array
    target_positions: wp.array
    target_quaternions: wp.array
    target_linear_velocity: wp.array
    target_angular_velocity: wp.array
    trajectory_step_counts: wp.array
    frame_scales: wp.array
    position_loss: wp.array
    orientation_loss: wp.array
    linear_velocity_loss: wp.array
    angular_velocity_loss: wp.array
    loss: wp.array
    batch_loss: wp.array
    residual_norm_mean: wp.array
    residual_energy_mean: wp.array
    residual_norm_max: wp.array


@dataclass
class MLPParameters:
    w0: wp.array
    b0: wp.array
    w1: wp.array
    b1: wp.array
    w2: wp.array
    b2: wp.array
    w3: wp.array
    b3: wp.array
    output_scales: wp.array


@dataclass
class MLPAdamState:
    m_w0: wp.array
    v_w0: wp.array
    m_b0: wp.array
    v_b0: wp.array
    m_w1: wp.array
    v_w1: wp.array
    m_b1: wp.array
    v_b1: wp.array
    m_w2: wp.array
    v_w2: wp.array
    m_b2: wp.array
    v_b2: wp.array
    m_w3: wp.array
    v_w3: wp.array
    m_b3: wp.array
    v_b3: wp.array
    step: int


@dataclass
class MLPActivations:
    batch_capacity: int
    step_capacity: int
    hidden0: wp.array
    hidden1: wp.array
    hidden2: wp.array
    residuals: wp.array


@dataclass
class FrozenFriction:
    checkpoint_path: Path | None
    point_cloud_path: Path | None
    active_indices: np.ndarray
    active_params: np.ndarray
    full_point_friction: np.ndarray
    mu_features: np.ndarray
    parameterization: str
    left_right_delta_sum_zero: bool


@dataclass
class TrainableFriction:
    mode: str
    active_indices_np: np.ndarray
    active_indices_wp: wp.array
    active_param_positions_np: np.ndarray
    active_param_positions_wp: wp.array
    mu_feature_weights_wp: wp.array
    optimizer_params: wp.array
    full_point_friction: wp.array
    adam_m: wp.array
    adam_v: wp.array
    mu_features_np: np.ndarray
    mu_features_wp: wp.array
    step: int
    parameterization: str
    parameterization_id: int
    left_right_delta_sum_zero: bool
    min_value: float
    max_value: float


def parameterization_id(parameterization: str) -> int:
    if parameterization == "point":
        return 0
    if parameterization == "global":
        return 1
    if parameterization == "left-right":
        return 2
    if parameterization == "base-delta":
        return 3
    raise ValueError(f"Unsupported friction parameterization: {parameterization!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--trajectory-npz", type=Path, default=DEFAULT_TRAIN_DATASET)
    parser.add_argument(
        "--friction-checkpoint",
        type=Path,
        default=None,
        help="Frozen or initialization friction checkpoint. Required unless --train-friction-end-to-end is used.",
    )
    parser.add_argument(
        "--train-friction-end-to-end",
        action="store_true",
        help="Train friction active-point parameters jointly with the residual MLP instead of requiring a trained checkpoint.",
    )
    parser.add_argument(
        "--friction-point-cloud",
        type=Path,
        default=None,
        help="Reference point cloud for the frozen checkpoint. Defaults to <checkpoint stem>.ply next to the checkpoint.",
    )
    parser.add_argument("--checkpoint-param-set", choices=("best", "current"), default="best")
    parser.add_argument("--experiment-dir", type=Path, default=REPO_ROOT / "outputs/residual_dynamics_adapter")
    parser.add_argument(
        "--timestamp-experiment-dir",
        dest="timestamp_experiment_dir",
        action="store_true",
        default=True,
        help="Prefix --experiment-dir's final directory name with the current local timestamp.",
    )
    parser.add_argument(
        "--no-timestamp-experiment-dir",
        dest="timestamp_experiment_dir",
        action="store_false",
        help="Keep --experiment-dir exactly as supplied.",
    )
    parser.add_argument(
        "--experiment-dir-timestamp",
        type=str,
        default=None,
        help="Override the timestamp prefix used when --timestamp-experiment-dir is enabled.",
    )
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opt-iters", type=int, default=100)
    parser.add_argument(
        "--resume-adapter",
        type=Path,
        default=None,
        help="Load a saved residual adapter checkpoint before training/evaluation.",
    )
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--friction-learning-rate", type=float, default=None)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1.0e-8)
    parser.add_argument("--grad-clip-norm", type=float, default=10.0)
    parser.add_argument(
        "--skip-nonfinite-grad-batches",
        dest="skip_nonfinite_grad_batches",
        action="store_true",
        default=True,
        help="Skip the parameter update when an iteration produces nonfinite gradients instead of raising.",
    )
    parser.add_argument(
        "--no-skip-nonfinite-grad-batches",
        dest="skip_nonfinite_grad_batches",
        action="store_false",
        help="Raise immediately on nonfinite gradients (legacy fail-fast behavior).",
    )
    parser.add_argument(
        "--max-consecutive-nonfinite-batches",
        type=int,
        default=50,
        help="Abort training if this many consecutive iterations have nonfinite gradients (0 disables the guard).",
    )
    parser.add_argument(
        "--bptt-truncation-steps",
        dest="bptt_truncation_steps",
        type=int,
        default=0,
        help=(
            "Truncated backpropagation-through-time window length (in rollout steps). "
            "0 disables truncation (full BPTT over the whole rollout, default). "
            "When >0, the forward rollout still runs the full window continuously, but "
            "gradients are only backpropagated within contiguous segments of this many "
            "steps, with the carried state detached between segments. This caps the "
            "Jacobian product depth and prevents BPTT gradient explosion."
        ),
    )
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--random-time-windows", dest="random_time_windows", action="store_true", default=True)
    parser.add_argument("--no-random-time-windows", dest="random_time_windows", action="store_false")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--window-steps", type=int, default=None)
    parser.add_argument("--time-window-source-max-steps", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--eval-dataset", type=Path, action="append", default=[])
    parser.add_argument("--eval-after-train", action="store_true")
    parser.add_argument("--eval-trajectory-limit", type=int, default=None)
    parser.add_argument(
        "--eval-heldout-start",
        type=int,
        default=None,
        help="Also evaluate each --eval-dataset subset from this trajectory index onward, e.g. 64 for rotation68.",
    )
    parser.add_argument(
        "--eval-heldout-end",
        type=int,
        default=None,
        help="Exclusive end index for --eval-heldout-start. Defaults to the dataset end.",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default="newton-contact-point-friction-fit")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default="residual-dynamics-adapter")
    parser.add_argument("--wandb-mode", type=str, default="online")
    parser.add_argument("--wandb-dir", type=Path, default=None)
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=None)
    parser.add_argument("--position-loss-weight", type=float, default=1.0)
    parser.add_argument("--orientation-loss-weight", type=float, default=1.0)
    parser.add_argument("--linear-velocity-loss-weight", type=float, default=0.0)
    parser.add_argument("--angular-velocity-loss-weight", type=float, default=0.1)
    parser.add_argument(
        "--point-position-loss-reduction",
        choices=("sum", "mean"),
        default="mean",
        help="Surface-point position loss reduction, matching friction fitting/eval semantics.",
    )
    parser.add_argument("--residual-l2-weight", type=float, default=1.0e-4)
    parser.add_argument("--residual-smoothness-weight", type=float, default=1.0e-4)
    parser.add_argument("--acceleration-scale", type=float, default=2.0)
    parser.add_argument("--angular-acceleration-scale", type=float, default=20.0)
    parser.add_argument("--steps", type=int, default=0, help="Filled after dataset loading.")
    parser.add_argument("--dt", type=float, default=0.0, help="Filled after dataset loading.")
    parser.add_argument("--solver-iterations", type=int, default=10)
    parser.add_argument("--box-mass", type=float, default=1.0)
    parser.add_argument("--floor-half-extents", type=float, nargs=3, default=(2.0, 2.0, 0.05))
    parser.add_argument("--box-half-extents", type=float, nargs=3, default=(0.1, 0.05, 0.025))
    parser.add_argument("--box-start-pos", type=float, nargs=3, default=(0.58, 0.0, 0.025))
    parser.add_argument("--surface-point-spacing", type=float, default=0.01)
    parser.add_argument("--friction-contact-threshold", type=float, default=0.002)
    parser.add_argument("--contact-mask-threshold", type=float, default=0.002)
    parser.add_argument(
        "--friction-parameterization",
        choices=("point", "left-right", "global", "base-delta"),
        default="point",
        help="Only used when --train-friction-end-to-end starts without --friction-checkpoint.",
    )
    parser.add_argument("--min-point-friction", type=float, default=0.0)
    parser.add_argument("--max-point-friction", type=float, default=2.0)
    parser.add_argument("--contact-friction", type=float, default=0.0)
    parser.add_argument("--point-friction", type=float, default=0.35)
    parser.add_argument("--contact-stiffness", type=float, default=1.0e5)
    parser.add_argument("--contact-damping", type=float, default=50.0)
    parser.add_argument("--contact-margin", type=float, default=1.0e-3)
    parser.add_argument("--friction-regularization", type=float, default=1.0e-3)
    args = parser.parse_args()
    _apply_experiment_dir_timestamp(args, parser)
    return args


def _resolve_window_steps(args: argparse.Namespace, dataset_max_steps: int) -> int:
    if not bool(args.random_time_windows):
        return int(dataset_max_steps)
    if args.window_steps is not None:
        requested = int(args.window_steps)
    elif args.max_steps is not None:
        requested = int(args.max_steps)
    else:
        requested = int(dataset_max_steps)
    if requested < 1:
        raise ValueError("--max-steps/--window-steps must be positive.")
    return min(requested, int(dataset_max_steps))


def _resolve_load_max_steps(args: argparse.Namespace) -> int | None:
    if bool(args.random_time_windows):
        return args.time_window_source_max_steps
    return args.max_steps


def resolve_point_position_loss_scale(args: argparse.Namespace, point_count: int) -> float:
    if args.point_position_loss_reduction == "sum":
        return 1.0
    if args.point_position_loss_reduction == "mean":
        return 1.0 / max(int(point_count), 1)
    raise ValueError(f"Unsupported --point-position-loss-reduction: {args.point_position_loss_reduction!r}")


def _default_point_cloud_path(checkpoint_path: Path) -> Path | None:
    candidate = checkpoint_path.with_suffix(".ply")
    return candidate if candidate.exists() else None


def _default_point_cloud_path_or_none(checkpoint_path: Path | None) -> Path | None:
    if checkpoint_path is None:
        return None
    return _default_point_cloud_path(checkpoint_path)


def _pad_vec3_rows(values: np.ndarray, length: int) -> np.ndarray:
    padded = np.zeros((length, 3), dtype=np.float32)
    if len(values) > 0:
        used = min(len(values), length)
        padded[:used] = np.asarray(values[:used], dtype=np.float32)
        if used < length:
            padded[used:] = padded[used - 1]
    return padded


def _pad_vec4_rows(values: np.ndarray, length: int) -> np.ndarray:
    padded = np.zeros((length, 4), dtype=np.float32)
    if len(values) > 0:
        used = min(len(values), length)
        padded[:used] = np.asarray(values[:used], dtype=np.float32)
        if used < length:
            padded[used:] = padded[used - 1]
    return padded


def build_rollout_buffers(
    *,
    device: str,
    point_count: int,
    full_point_friction: np.ndarray,
    batch_capacity: int,
    step_capacity: int,
) -> ResidualRolloutBuffers:
    batch_capacity = max(int(batch_capacity), 1)
    step_capacity = max(int(step_capacity), 1)
    frame_capacity = step_capacity + 1

    return ResidualRolloutBuffers(
        batch_capacity=batch_capacity,
        step_capacity=step_capacity,
        frame_capacity=frame_capacity,
        full_point_friction=wp.array(np.asarray(full_point_friction, dtype=np.float32), dtype=wp.float32, device=device),
        contact_weighted_masses=wp.zeros(step_capacity * batch_capacity * point_count, dtype=wp.float32, device=device),
        contact_weighted_mass_total=wp.zeros(step_capacity * batch_capacity, dtype=wp.float32, device=device),
        step_forces=wp.zeros(batch_capacity * step_capacity, dtype=wp.vec3, device=device),
        force_point_offsets_local=wp.zeros(batch_capacity, dtype=wp.vec3, device=device),
        initial_positions=wp.zeros(batch_capacity, dtype=wp.vec3, device=device),
        initial_quaternions=wp.zeros(batch_capacity, dtype=wp.quat, device=device),
        initial_linear_velocity=wp.zeros(batch_capacity, dtype=wp.vec3, device=device),
        initial_angular_velocity=wp.zeros(batch_capacity, dtype=wp.vec3, device=device),
        target_positions=wp.zeros(batch_capacity * frame_capacity, dtype=wp.vec3, device=device),
        target_quaternions=wp.zeros(batch_capacity * frame_capacity, dtype=wp.vec4, device=device),
        target_linear_velocity=wp.zeros(batch_capacity * frame_capacity, dtype=wp.vec3, device=device),
        target_angular_velocity=wp.zeros(batch_capacity * frame_capacity, dtype=wp.vec3, device=device),
        trajectory_step_counts=wp.zeros(batch_capacity, dtype=wp.int32, device=device),
        frame_scales=wp.zeros(batch_capacity, dtype=wp.float32, device=device),
        position_loss=wp.zeros(batch_capacity, dtype=wp.float32, device=device, requires_grad=True),
        orientation_loss=wp.zeros(batch_capacity, dtype=wp.float32, device=device, requires_grad=True),
        linear_velocity_loss=wp.zeros(batch_capacity, dtype=wp.float32, device=device, requires_grad=True),
        angular_velocity_loss=wp.zeros(batch_capacity, dtype=wp.float32, device=device, requires_grad=True),
        loss=wp.zeros(batch_capacity, dtype=wp.float32, device=device, requires_grad=True),
        batch_loss=wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True),
        residual_norm_mean=wp.zeros(batch_capacity, dtype=wp.float32, device=device),
        residual_energy_mean=wp.zeros(batch_capacity, dtype=wp.float32, device=device),
        residual_norm_max=wp.zeros(batch_capacity, dtype=wp.float32, device=device),
    )


def assign_rollout_buffer_trajectories(
    buffers: ResidualRolloutBuffers,
    trajectories: list[MujocoTrajectory],
) -> int:
    batch_size = len(trajectories)
    if batch_size > buffers.batch_capacity:
        raise ValueError(f"batch size {batch_size} exceeds buffer capacity {buffers.batch_capacity}")
    if any(trajectory.num_steps > buffers.step_capacity for trajectory in trajectories):
        max_steps = max((trajectory.num_steps for trajectory in trajectories), default=0)
        raise ValueError(f"trajectory steps {max_steps} exceed buffer step capacity {buffers.step_capacity}")

    step_forces = np.zeros((buffers.batch_capacity, buffers.step_capacity, 3), dtype=np.float32)
    force_point_offsets_local = np.zeros((buffers.batch_capacity, 3), dtype=np.float32)
    target_positions = np.zeros((buffers.batch_capacity, buffers.frame_capacity, 3), dtype=np.float32)
    target_quaternions = np.zeros((buffers.batch_capacity, buffers.frame_capacity, 4), dtype=np.float32)
    target_linear_velocity = np.zeros((buffers.batch_capacity, buffers.frame_capacity, 3), dtype=np.float32)
    target_angular_velocity = np.zeros((buffers.batch_capacity, buffers.frame_capacity, 3), dtype=np.float32)
    initial_positions = np.zeros((buffers.batch_capacity, 3), dtype=np.float32)
    initial_quaternions = np.zeros((buffers.batch_capacity, 4), dtype=np.float32)
    initial_linear_velocity = np.zeros((buffers.batch_capacity, 3), dtype=np.float32)
    initial_angular_velocity = np.zeros((buffers.batch_capacity, 3), dtype=np.float32)
    step_counts = np.zeros(buffers.batch_capacity, dtype=np.int32)
    frame_scales = np.zeros(buffers.batch_capacity, dtype=np.float32)

    for batch_idx, trajectory in enumerate(trajectories):
        step_forces[batch_idx] = _pad_vec3_rows(trajectory.step_forces, buffers.step_capacity)
        force_point_offsets_local[batch_idx] = np.asarray(trajectory.force_point_offset_local, dtype=np.float32).reshape(3)
        target_positions[batch_idx] = _pad_vec3_rows(trajectory.positions, buffers.frame_capacity)
        target_quaternions[batch_idx] = _pad_vec4_rows(trajectory.quaternions_xyzw, buffers.frame_capacity)
        target_linear_velocity[batch_idx] = _pad_vec3_rows(trajectory.linear_velocity, buffers.frame_capacity)
        target_angular_velocity[batch_idx] = _pad_vec3_rows(trajectory.angular_velocity, buffers.frame_capacity)
        initial_positions[batch_idx] = target_positions[batch_idx, 0]
        initial_quaternions[batch_idx] = target_quaternions[batch_idx, 0]
        initial_linear_velocity[batch_idx] = target_linear_velocity[batch_idx, 0]
        initial_angular_velocity[batch_idx] = target_angular_velocity[batch_idx, 0]
        step_counts[batch_idx] = int(trajectory.num_steps)
        frame_scales[batch_idx] = np.float32(1.0 / max(trajectory.num_frames, 1))

    buffers.step_forces.assign(step_forces.reshape(-1, 3))
    buffers.force_point_offsets_local.assign(force_point_offsets_local)
    buffers.initial_positions.assign(initial_positions)
    buffers.initial_quaternions.assign(initial_quaternions)
    buffers.initial_linear_velocity.assign(initial_linear_velocity)
    buffers.initial_angular_velocity.assign(initial_angular_velocity)
    buffers.target_positions.assign(target_positions.reshape(-1, 3))
    buffers.target_quaternions.assign(target_quaternions.reshape(-1, 4))
    buffers.target_linear_velocity.assign(target_linear_velocity.reshape(-1, 3))
    buffers.target_angular_velocity.assign(target_angular_velocity.reshape(-1, 3))
    buffers.trajectory_step_counts.assign(step_counts)
    buffers.frame_scales.assign(frame_scales)
    return batch_size


def build_activation_buffers(*, device: str, batch_capacity: int, step_capacity: int) -> MLPActivations:
    step_capacity = max(int(step_capacity), 1)
    batch_capacity = max(int(batch_capacity), 1)
    return MLPActivations(
        batch_capacity=batch_capacity,
        step_capacity=step_capacity,
        hidden0=wp.zeros(step_capacity * batch_capacity * HIDDEN0_DIM, dtype=wp.float32, device=device, requires_grad=True),
        hidden1=wp.zeros(step_capacity * batch_capacity * HIDDEN1_DIM, dtype=wp.float32, device=device, requires_grad=True),
        hidden2=wp.zeros(step_capacity * batch_capacity * HIDDEN2_DIM, dtype=wp.float32, device=device, requires_grad=True),
        residuals=wp.zeros(step_capacity * batch_capacity * OUTPUT_DIM, dtype=wp.float32, device=device, requires_grad=True),
    )


def _xavier_uniform(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    limit = math.sqrt(6.0 / float(fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_out, fan_in)).astype(np.float32)


def initialize_mlp_parameters(args: argparse.Namespace, device: str, rng: np.random.Generator) -> tuple[MLPParameters, MLPAdamState]:
    w0 = _xavier_uniform(rng, INPUT_DIM, HIDDEN0_DIM).reshape(-1)
    w1 = _xavier_uniform(rng, HIDDEN0_DIM, HIDDEN1_DIM).reshape(-1)
    w2 = _xavier_uniform(rng, HIDDEN1_DIM, HIDDEN2_DIM).reshape(-1)
    # Start from the frozen physics rollout. The final layer learns first; hidden layers then receive gradients.
    w3 = np.zeros((OUTPUT_DIM, HIDDEN2_DIM), dtype=np.float32).reshape(-1)
    b0 = np.zeros(HIDDEN0_DIM, dtype=np.float32)
    b1 = np.zeros(HIDDEN1_DIM, dtype=np.float32)
    b2 = np.zeros(HIDDEN2_DIM, dtype=np.float32)
    b3 = np.zeros(OUTPUT_DIM, dtype=np.float32)
    output_scales = np.asarray(
        [float(args.acceleration_scale), float(args.acceleration_scale), float(args.angular_acceleration_scale)],
        dtype=np.float32,
    )

    params = MLPParameters(
        w0=wp.array(w0, dtype=wp.float32, device=device, requires_grad=True),
        b0=wp.array(b0, dtype=wp.float32, device=device, requires_grad=True),
        w1=wp.array(w1, dtype=wp.float32, device=device, requires_grad=True),
        b1=wp.array(b1, dtype=wp.float32, device=device, requires_grad=True),
        w2=wp.array(w2, dtype=wp.float32, device=device, requires_grad=True),
        b2=wp.array(b2, dtype=wp.float32, device=device, requires_grad=True),
        w3=wp.array(w3, dtype=wp.float32, device=device, requires_grad=True),
        b3=wp.array(b3, dtype=wp.float32, device=device, requires_grad=True),
        output_scales=wp.array(output_scales, dtype=wp.float32, device=device),
    )
    for param in _parameter_arrays(params):
        param.grad = wp.zeros_like(param)

    def moments_like(param: wp.array) -> tuple[wp.array, wp.array]:
        zeros = np.zeros(param.shape[0], dtype=np.float64)
        return (
            wp.array(zeros, dtype=wp.float64, device=device),
            wp.array(zeros, dtype=wp.float64, device=device),
        )

    m_w0, v_w0 = moments_like(params.w0)
    m_b0, v_b0 = moments_like(params.b0)
    m_w1, v_w1 = moments_like(params.w1)
    m_b1, v_b1 = moments_like(params.b1)
    m_w2, v_w2 = moments_like(params.w2)
    m_b2, v_b2 = moments_like(params.b2)
    m_w3, v_w3 = moments_like(params.w3)
    m_b3, v_b3 = moments_like(params.b3)
    adam = MLPAdamState(
        m_w0=m_w0,
        v_w0=v_w0,
        m_b0=m_b0,
        v_b0=v_b0,
        m_w1=m_w1,
        v_w1=v_w1,
        m_b1=m_b1,
        v_b1=v_b1,
        m_w2=m_w2,
        v_w2=v_w2,
        m_b2=m_b2,
        v_b2=v_b2,
        m_w3=m_w3,
        v_w3=v_w3,
        m_b3=m_b3,
        v_b3=v_b3,
        step=0,
    )
    return params, adam


def _parameter_arrays(params: MLPParameters) -> list[wp.array]:
    return [params.w0, params.b0, params.w1, params.b1, params.w2, params.b2, params.w3, params.b3]


def _adam_array_pairs(adam: MLPAdamState) -> list[tuple[wp.array, wp.array]]:
    return [
        (adam.m_w0, adam.v_w0),
        (adam.m_b0, adam.v_b0),
        (adam.m_w1, adam.v_w1),
        (adam.m_b1, adam.v_b1),
        (adam.m_w2, adam.v_w2),
        (adam.m_b2, adam.v_b2),
        (adam.m_w3, adam.v_w3),
        (adam.m_b3, adam.v_b3),
    ]


def clear_gradients(
    params: MLPParameters,
    activations: MLPActivations,
    buffers: ResidualRolloutBuffers,
    trainable_friction: TrainableFriction | None = None,
) -> None:
    for array in _parameter_arrays(params):
        if array.grad is not None:
            array.grad.zero_()
    if trainable_friction is not None and trainable_friction.optimizer_params.grad is not None:
        trainable_friction.optimizer_params.grad.zero_()
    if trainable_friction is not None and trainable_friction.full_point_friction.grad is not None:
        trainable_friction.full_point_friction.grad.zero_()
    if trainable_friction is not None and trainable_friction.mu_features_wp.grad is not None:
        trainable_friction.mu_features_wp.grad.zero_()
    for array in (activations.hidden0, activations.hidden1, activations.hidden2, activations.residuals):
        if array.grad is not None:
            array.grad.zero_()
    for array in (
        buffers.position_loss,
        buffers.orientation_loss,
        buffers.linear_velocity_loss,
        buffers.angular_velocity_loss,
        buffers.loss,
        buffers.batch_loss,
    ):
        if array.grad is not None:
            array.grad.zero_()


def compute_feature_stats(trajectories: list[MujocoTrajectory], mu_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    feature_sum = np.zeros(INPUT_DIM, dtype=np.float64)
    feature_sq_sum = np.zeros(INPUT_DIM, dtype=np.float64)
    count = 0

    for trajectory in trajectories:
        if trajectory.num_steps <= 0:
            continue
        q = np.asarray(trajectory.quaternions_xyzw[:-1], dtype=np.float64)
        yaw = np.arctan2(2.0 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]), 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2))
        c = np.cos(yaw)
        s = np.sin(yaw)
        linear = np.asarray(trajectory.linear_velocity[:-1], dtype=np.float64)
        angular = np.asarray(trajectory.angular_velocity[:-1], dtype=np.float64)
        forces = np.asarray(trajectory.step_forces, dtype=np.float64)
        offset = np.asarray(trajectory.force_point_offset_local, dtype=np.float64).reshape(3)

        v_body_x = c * linear[:, 0] + s * linear[:, 1]
        v_body_y = -s * linear[:, 0] + c * linear[:, 1]
        force_body_x = c * forces[:, 0] + s * forces[:, 1]
        force_body_y = -s * forces[:, 0] + c * forces[:, 1]
        torque_z = offset[0] * force_body_y - offset[1] * force_body_x

        features = np.stack(
            [
                v_body_x,
                v_body_y,
                angular[:, 2],
                force_body_x,
                force_body_y,
                np.full(trajectory.num_steps, offset[0], dtype=np.float64),
                np.full(trajectory.num_steps, offset[1], dtype=np.float64),
                torque_z,
                np.full(trajectory.num_steps, float(mu_features[0]), dtype=np.float64),
                np.full(trajectory.num_steps, float(mu_features[1]), dtype=np.float64),
                np.full(trajectory.num_steps, float(mu_features[2]), dtype=np.float64),
            ],
            axis=1,
        )
        feature_sum += np.sum(features, axis=0)
        feature_sq_sum += np.sum(features * features, axis=0)
        count += int(features.shape[0])

    if count == 0:
        raise ValueError("Cannot compute feature normalization from an empty training split.")
    mean = feature_sum / float(count)
    variance = np.maximum(feature_sq_sum / float(count) - mean * mean, 1.0e-12)
    std = np.sqrt(variance)
    mean[8:11] = 0.0
    std[8:11] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def split_trajectories(
    trajectories: list[MujocoTrajectory],
    *,
    train_fraction: float,
    val_fraction: float,
    rng: np.random.Generator,
) -> tuple[list[MujocoTrajectory], list[MujocoTrajectory], list[MujocoTrajectory]]:
    if not trajectories:
        raise ValueError("No trajectories loaded.")
    if train_fraction <= 0.0 or train_fraction > 1.0:
        raise ValueError("--train-fraction must be in (0, 1].")
    if val_fraction < 0.0 or val_fraction >= 1.0:
        raise ValueError("--val-fraction must be in [0, 1).")
    if train_fraction + val_fraction > 1.0:
        raise ValueError("--train-fraction + --val-fraction must be <= 1.")

    indices = np.arange(len(trajectories), dtype=np.int32)
    rng.shuffle(indices)
    train_count = max(1, int(round(len(indices) * train_fraction)))
    train_count = min(train_count, len(indices))
    val_count = int(round(len(indices) * val_fraction))
    val_count = min(val_count, max(len(indices) - train_count, 0))

    train_indices = indices[:train_count]
    val_indices = indices[train_count : train_count + val_count]
    test_indices = indices[train_count + val_count :]
    return (
        [trajectories[int(idx)] for idx in train_indices],
        [trajectories[int(idx)] for idx in val_indices],
        [trajectories[int(idx)] for idx in test_indices],
    )


def compute_active_indices_for_trajectories(diff_scene, trajectories: list[MujocoTrajectory], args: argparse.Namespace) -> np.ndarray:
    from mujoco_contact_friction_fit_utils import compute_active_contact_point_indices

    active_mask = np.zeros(len(diff_scene.local_surface_points_np), dtype=bool)
    for trajectory in trajectories:
        active_indices = compute_active_contact_point_indices(
            local_surface_points=diff_scene.local_surface_points_np,
            trajectory=trajectory,
            floor_top_z=diff_scene.floor_top_z,
            contact_threshold=float(args.contact_mask_threshold),
        )
        active_mask[active_indices] = True
    result = np.flatnonzero(active_mask).astype(np.int32)
    if len(result) == 0:
        raise RuntimeError("No active contact points found for end-to-end friction training.")
    return result


def initialize_active_friction_from_parameterization(
    diff_scene,
    active_indices: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    parameterization = validate_friction_parameterization(str(args.friction_parameterization))
    active_side_ids = compute_piecewise_side_ids(diff_scene.local_surface_points_np, active_indices)
    active_param_positions, optimizer_param_count = build_optimizer_param_positions(
        parameterization=parameterization,
        active_side_ids=active_side_ids,
        active_count=len(active_indices),
    )
    optimizer_params = initialize_optimizer_params_np(
        parameterization=parameterization,
        optimizer_param_count=optimizer_param_count,
        point_friction=float(args.point_friction),
    )
    if parameterization == "base-delta":
        optimizer_params = project_base_delta_optimizer_params_np(
            optimizer_params,
            min_value=float(args.min_point_friction),
            max_value=float(args.max_point_friction),
            left_right_delta_sum_zero=False,
        )
    active_params = expand_optimizer_params_to_active(
        optimizer_params,
        active_param_positions,
        parameterization=parameterization,
    )
    return active_params.astype(np.float32), optimizer_params.astype(np.float32), active_param_positions, parameterization


def infer_optimizer_params_from_active(
    diff_scene,
    active_indices: np.ndarray,
    active_params: np.ndarray,
    parameterization: str,
) -> tuple[np.ndarray, np.ndarray]:
    active_side_ids = compute_piecewise_side_ids(diff_scene.local_surface_points_np, active_indices)
    active_param_positions, optimizer_param_count = build_optimizer_param_positions(
        parameterization=parameterization,
        active_side_ids=active_side_ids,
        active_count=len(active_indices),
    )
    if parameterization == "point":
        return np.asarray(active_params, dtype=np.float32).copy(), active_param_positions
    if parameterization == "global":
        return np.asarray([float(np.mean(active_params))], dtype=np.float32), active_param_positions
    if parameterization == "left-right":
        values = np.zeros(2, dtype=np.float32)
        for side_id in (0, 1):
            side_values = np.asarray(active_params, dtype=np.float32)[active_param_positions == side_id]
            values[side_id] = np.float32(np.mean(side_values)) if len(side_values) > 0 else np.float32(np.mean(active_params))
        return values, active_param_positions
    if parameterization == "base-delta":
        active_params = np.asarray(active_params, dtype=np.float32)
        mean_mu = float(np.mean(active_params))
        left_values = active_params[active_param_positions == 0]
        right_values = active_params[active_param_positions == 1]
        left_mu = float(np.mean(left_values)) if len(left_values) > 0 else mean_mu
        right_mu = float(np.mean(right_values)) if len(right_values) > 0 else mean_mu
        return np.asarray([mean_mu, left_mu - mean_mu, right_mu - mean_mu], dtype=np.float32), active_param_positions
    if optimizer_param_count < 0:
        raise ValueError("unreachable")
    raise ValueError(f"Unsupported friction parameterization: {parameterization!r}")


def compute_mu_features_from_active(
    diff_scene,
    active_indices: np.ndarray,
    active_params: np.ndarray,
    parameterization: str,
) -> np.ndarray:
    active_params = np.asarray(active_params, dtype=np.float32)
    active_x = diff_scene.local_surface_points_np[np.asarray(active_indices, dtype=np.int32), 0]
    mean_mu = float(np.mean(active_params)) if len(active_params) > 0 else 0.0
    left_values = active_params[active_x < 0.0]
    right_values = active_params[active_x > 0.0]
    left_mu = float(np.mean(left_values)) if len(left_values) > 0 else mean_mu
    right_mu = float(np.mean(right_values)) if len(right_values) > 0 else mean_mu
    if parameterization == "global":
        left_mu = mean_mu
        right_mu = mean_mu
    return np.asarray([mean_mu, left_mu, right_mu], dtype=np.float32)


def build_mu_feature_weights(diff_scene, active_indices: np.ndarray) -> np.ndarray:
    active_indices = np.asarray(active_indices, dtype=np.int32)
    active_x = diff_scene.local_surface_points_np[active_indices, 0]
    weights = np.zeros((len(active_indices), 3), dtype=np.float32)
    if len(active_indices) == 0:
        return weights.reshape(-1)

    weights[:, 0] = np.float32(1.0 / len(active_indices))
    left_mask = active_x < 0.0
    right_mask = active_x > 0.0
    if np.any(left_mask):
        weights[left_mask, 1] = np.float32(1.0 / int(np.count_nonzero(left_mask)))
    else:
        weights[:, 1] = weights[:, 0]
    if np.any(right_mask):
        weights[right_mask, 2] = np.float32(1.0 / int(np.count_nonzero(right_mask)))
    else:
        weights[:, 2] = weights[:, 0]
    return weights.reshape(-1)


def resolve_frozen_friction(args: argparse.Namespace, diff_scene) -> FrozenFriction:
    if args.friction_checkpoint is None:
        raise ValueError("--friction-checkpoint is required unless --train-friction-end-to-end is used.")
    checkpoint = load_checkpoint_parameters(args.friction_checkpoint)
    parameterization = "point"
    left_right_delta_sum_zero = False
    with np.load(args.friction_checkpoint, allow_pickle=True) as data:
        if "friction_parameterization" in data.files:
            parameterization = str(np.asarray(data["friction_parameterization"]).item())
        if "left_right_delta_sum_zero" in data.files:
            left_right_delta_sum_zero = bool(np.asarray(data["left_right_delta_sum_zero"]).item())

    point_cloud_path = args.friction_point_cloud
    if point_cloud_path is None:
        point_cloud_path = _default_point_cloud_path(args.friction_checkpoint)

    reference_to_scene = None
    if point_cloud_path is not None:
        point_cloud = load_contact_friction_point_cloud(point_cloud_path)
        reference_to_scene = build_reference_to_scene_index(
            point_cloud.local_surface_points,
            diff_scene.local_surface_points_np,
        )
        args.point_friction = infer_base_point_friction(point_cloud, fallback=float(args.point_friction))

    active_indices = np.asarray(checkpoint.active_indices, dtype=np.int32).copy()
    if reference_to_scene is not None:
        active_indices = reference_to_scene[active_indices]

    active_params = (
        np.asarray(checkpoint.best_active_params, dtype=np.float32)
        if args.checkpoint_param_set == "best"
        else np.asarray(checkpoint.active_params, dtype=np.float32)
    )
    if active_indices.shape != active_params.shape:
        raise ValueError(f"Frozen friction shape mismatch: active_indices={active_indices.shape}, params={active_params.shape}")
    if len(active_indices) == 0:
        raise ValueError("Frozen checkpoint has no active friction parameters.")

    full_point_friction = np.full(len(diff_scene.local_surface_points_np), float(args.point_friction), dtype=np.float32)
    full_point_friction[active_indices] = active_params

    mu_features = compute_mu_features_from_active(diff_scene, active_indices, active_params, parameterization)

    return FrozenFriction(
        checkpoint_path=args.friction_checkpoint,
        point_cloud_path=point_cloud_path,
        active_indices=active_indices,
        active_params=active_params,
        full_point_friction=full_point_friction,
        mu_features=mu_features,
        parameterization=parameterization,
        left_right_delta_sum_zero=left_right_delta_sum_zero,
    )


def resolve_initial_friction_state(
    args: argparse.Namespace,
    diff_scene,
    train_trajectories: list[MujocoTrajectory],
) -> FrozenFriction:
    if args.friction_checkpoint is not None:
        return resolve_frozen_friction(args, diff_scene)

    active_indices = compute_active_indices_for_trajectories(diff_scene, train_trajectories, args)
    active_params, _, _, parameterization = initialize_active_friction_from_parameterization(diff_scene, active_indices, args)
    full_point_friction = np.full(len(diff_scene.local_surface_points_np), float(args.point_friction), dtype=np.float32)
    full_point_friction[active_indices] = active_params
    mu_features = compute_mu_features_from_active(diff_scene, active_indices, active_params, parameterization)
    return FrozenFriction(
        checkpoint_path=None,
        point_cloud_path=None,
        active_indices=active_indices,
        active_params=active_params,
        full_point_friction=full_point_friction,
        mu_features=mu_features,
        parameterization=parameterization,
        left_right_delta_sum_zero=False,
    )


def maybe_infer_scene_from_point_cloud(args: argparse.Namespace) -> None:
    point_cloud_path = args.friction_point_cloud
    if point_cloud_path is None:
        point_cloud_path = _default_point_cloud_path_or_none(args.friction_checkpoint)
    if point_cloud_path is None:
        return
    point_cloud = load_contact_friction_point_cloud(point_cloud_path)
    inferred_half_extents, inferred_spacing = infer_box_half_extents_and_spacing(point_cloud.local_surface_points)
    args.box_half_extents = inferred_half_extents.tolist()
    args.surface_point_spacing = inferred_spacing
    args.point_friction = infer_base_point_friction(point_cloud, fallback=float(args.point_friction))
    args.friction_point_cloud = point_cloud_path
    log_message(
        "inferred frozen-checkpoint scene sampling "
        f"box_half_extents={np.asarray(args.box_half_extents, dtype=np.float32).tolist()} "
        f"surface_point_spacing={float(args.surface_point_spacing):.9g} "
        f"base_point_friction={float(args.point_friction):.9g}"
    )


def forward_residual_rollout(
    *,
    diff_scene,
    sim_states,
    buffers: ResidualRolloutBuffers,
    activations: MLPActivations,
    batch_size: int,
    params: MLPParameters,
    feature_mean: wp.array,
    feature_inv_std: wp.array,
    mu_features: wp.array,
    trainable_friction: TrainableFriction | None,
    args: argparse.Namespace,
) -> wp.array:
    batch_size = int(batch_size)
    if batch_size < 1 or batch_size > buffers.batch_capacity:
        raise ValueError(f"active batch size {batch_size} is outside [1, {buffers.batch_capacity}]")
    point_count = len(diff_scene.local_surface_points_np)
    point_scale = resolve_point_position_loss_scale(args, point_count)

    buffers.position_loss.zero_()
    buffers.orientation_loss.zero_()
    buffers.linear_velocity_loss.zero_()
    buffers.angular_velocity_loss.zero_()
    buffers.loss.zero_()
    buffers.batch_loss.zero_()
    buffers.contact_weighted_masses.zero_()
    buffers.contact_weighted_mass_total.zero_()
    buffers.residual_norm_mean.zero_()
    buffers.residual_energy_mean.zero_()
    buffers.residual_norm_max.zero_()
    activations.hidden0.zero_()
    activations.hidden1.zero_()
    activations.hidden2.zero_()
    activations.residuals.zero_()

    if trainable_friction is not None:
        trainable_friction.mu_features_wp.zero_()
        wp.launch(
            accumulate_optimizer_mu_features_kernel,
            dim=int(trainable_friction.active_indices_wp.shape[0]),
            inputs=[
                trainable_friction.active_param_positions_wp,
                trainable_friction.optimizer_params,
                int(trainable_friction.parameterization_id),
                trainable_friction.mu_feature_weights_wp,
                trainable_friction.mu_features_wp,
            ],
            device=diff_scene.model.device,
        )
        mu_features = trainable_friction.mu_features_wp
        wp.launch(
            scatter_optimizer_point_friction_kernel,
            dim=int(trainable_friction.active_indices_wp.shape[0]),
            inputs=[
                trainable_friction.active_indices_wp,
                trainable_friction.active_param_positions_wp,
                trainable_friction.optimizer_params,
                int(trainable_friction.parameterization_id),
                trainable_friction.full_point_friction,
            ],
            device=diff_scene.model.device,
        )
        point_friction = trainable_friction.full_point_friction
    else:
        point_friction = buffers.full_point_friction

    wp.launch(
        set_batched_box_initial_states_kernel,
        dim=batch_size,
        inputs=[
            diff_scene.box_body_ids_wp,
            buffers.initial_positions,
            buffers.initial_quaternions,
            buffers.initial_linear_velocity,
            buffers.initial_angular_velocity,
            diff_scene.states[0].body_q,
            diff_scene.states[0].body_qd,
        ],
        device=diff_scene.model.device,
    )
    wp.launch(
        accumulate_residual_frame_loss_kernel,
        dim=batch_size * point_count,
        inputs=[
            0,
            diff_scene.box_body_ids_wp,
            diff_scene.states[0].body_q,
            diff_scene.states[0].body_qd,
            diff_scene.local_surface_points_wp,
            buffers.target_positions,
            buffers.target_quaternions,
            buffers.target_linear_velocity,
            buffers.target_angular_velocity,
            buffers.trajectory_step_counts,
            buffers.frame_scales,
            float(point_scale),
            point_count,
            buffers.frame_capacity,
            buffers.position_loss,
            buffers.orientation_loss,
            buffers.linear_velocity_loss,
            buffers.angular_velocity_loss,
        ],
        device=diff_scene.model.device,
    )

    for step_idx in range(buffers.step_capacity):
        state_in = diff_scene.states[step_idx]
        state_sim = sim_states[step_idx]
        state_out = diff_scene.states[step_idx + 1]
        state_in.clear_forces()

        wp.launch(
            compute_batched_contact_weighted_masses_kernel,
            dim=batch_size * point_count,
            inputs=[
                step_idx,
                diff_scene.box_body_ids_wp,
                state_in.body_q,
                diff_scene.local_surface_points_wp,
                diff_scene.point_masses_wp,
                batch_size,
                point_count,
                float(diff_scene.floor_top_z),
                float(args.friction_contact_threshold),
                buffers.contact_weighted_masses,
                buffers.contact_weighted_mass_total,
            ],
            device=diff_scene.model.device,
        )
        wp.launch(
            apply_batched_external_and_surface_point_forces_trajectory_kernel,
            dim=batch_size * point_count,
            inputs=[
                step_idx,
                diff_scene.box_body_ids_wp,
                state_in.body_q,
                state_in.body_qd,
                diff_scene.model.body_com,
                diff_scene.local_surface_points_wp,
                buffers.contact_weighted_masses,
                buffers.contact_weighted_mass_total,
                point_friction,
                buffers.step_forces,
                buffers.force_point_offsets_local,
                buffers.trajectory_step_counts,
                batch_size,
                point_count,
                buffers.step_capacity,
                float(diff_scene.box_mass),
                float(GRAVITY_MAGNITUDE),
                float(diff_scene.floor_top_z),
                float(args.contact_stiffness),
                float(args.contact_damping),
                float(args.friction_contact_threshold),
                float(args.friction_regularization),
                state_in.body_f,
            ],
            device=diff_scene.model.device,
        )

        diff_scene.collision_pipeline.collide(state_in, diff_scene.contacts)
        diff_scene.solver.step(state_in, state_sim, diff_scene.control, diff_scene.contacts, float(args.dt))

        wp.launch(
            residual_mlp_layer0_kernel,
            dim=batch_size * HIDDEN0_DIM,
            inputs=[
                step_idx,
                diff_scene.box_body_ids_wp,
                state_in.body_q,
                state_in.body_qd,
                buffers.step_forces,
                buffers.force_point_offsets_local,
                buffers.trajectory_step_counts,
                batch_size,
                buffers.step_capacity,
                feature_mean,
                feature_inv_std,
                mu_features,
                params.w0,
                params.b0,
                activations.hidden0,
            ],
            device=diff_scene.model.device,
        )
        wp.launch(
            residual_mlp_layer1_kernel,
            dim=batch_size * HIDDEN1_DIM,
            inputs=[
                step_idx,
                buffers.trajectory_step_counts,
                batch_size,
                params.w1,
                params.b1,
                activations.hidden0,
                activations.hidden1,
            ],
            device=diff_scene.model.device,
        )
        wp.launch(
            residual_mlp_layer2_kernel,
            dim=batch_size * HIDDEN2_DIM,
            inputs=[
                step_idx,
                buffers.trajectory_step_counts,
                batch_size,
                params.w2,
                params.b2,
                activations.hidden1,
                activations.hidden2,
            ],
            device=diff_scene.model.device,
        )
        wp.launch(
            residual_mlp_output_kernel,
            dim=batch_size * OUTPUT_DIM,
            inputs=[
                step_idx,
                buffers.trajectory_step_counts,
                batch_size,
                params.w3,
                params.b3,
                params.output_scales,
                activations.hidden2,
                activations.residuals,
            ],
            device=diff_scene.model.device,
        )
        wp.launch(
            apply_residual_planar_dynamics_kernel,
            dim=batch_size,
            inputs=[
                step_idx,
                diff_scene.box_body_ids_wp,
                state_in.body_q,
                state_sim.body_q,
                state_sim.body_qd,
                activations.residuals,
                buffers.trajectory_step_counts,
                batch_size,
                float(args.dt),
                state_out.body_q,
                state_out.body_qd,
            ],
            device=diff_scene.model.device,
        )
        wp.launch(
            accumulate_residual_frame_loss_kernel,
            dim=batch_size * point_count,
            inputs=[
                step_idx + 1,
                diff_scene.box_body_ids_wp,
                state_out.body_q,
                state_out.body_qd,
                diff_scene.local_surface_points_wp,
                buffers.target_positions,
                buffers.target_quaternions,
                buffers.target_linear_velocity,
                buffers.target_angular_velocity,
                buffers.trajectory_step_counts,
                buffers.frame_scales,
                float(point_scale),
                point_count,
                buffers.frame_capacity,
                buffers.position_loss,
                buffers.orientation_loss,
                buffers.linear_velocity_loss,
                buffers.angular_velocity_loss,
            ],
            device=diff_scene.model.device,
        )

    wp.launch(
        combine_residual_loss_components_kernel,
        dim=batch_size,
        inputs=[
            buffers.position_loss,
            buffers.orientation_loss,
            buffers.linear_velocity_loss,
            buffers.angular_velocity_loss,
            float(args.position_loss_weight),
            float(args.orientation_loss_weight),
            float(args.linear_velocity_loss_weight),
            float(args.angular_velocity_loss_weight),
            buffers.loss,
        ],
        device=diff_scene.model.device,
    )
    wp.launch(
        sum_loss_kernel,
        dim=batch_size,
        inputs=[buffers.loss, float(1.0 / max(batch_size, 1)), buffers.batch_loss],
        device=diff_scene.model.device,
    )
    wp.launch(
        accumulate_residual_regularization_kernel,
        dim=batch_size * buffers.step_capacity,
        inputs=[
            activations.residuals,
            buffers.trajectory_step_counts,
            batch_size,
            buffers.step_capacity,
            float(args.residual_l2_weight),
            float(args.residual_smoothness_weight),
            float(1.0 / max(batch_size * buffers.step_capacity, 1)),
            buffers.batch_loss,
            buffers.residual_norm_mean,
            buffers.residual_energy_mean,
            buffers.residual_norm_max,
        ],
        device=diff_scene.model.device,
    )
    return buffers.batch_loss


def compute_global_grad_norm(params: MLPParameters, device: str) -> tuple[float, int]:
    norm_sq = wp.zeros(1, dtype=wp.float64, device=device)
    nonfinite_count = wp.zeros(1, dtype=wp.int32, device=device)
    for param in _parameter_arrays(params):
        if param.grad is None:
            continue
        wp.launch(
            accumulate_grad_norm_kernel,
            dim=param.shape[0],
            inputs=[param.grad, norm_sq, nonfinite_count],
            device=device,
        )
    norm_sq_np = norm_sq.numpy()
    nonfinite_np = nonfinite_count.numpy()
    return math.sqrt(max(float(norm_sq_np[0]), 0.0)), int(nonfinite_np[0])


def compute_array_grad_norm(array: wp.array, device: str) -> tuple[float, int]:
    norm_sq = wp.zeros(1, dtype=wp.float64, device=device)
    nonfinite_count = wp.zeros(1, dtype=wp.int32, device=device)
    if array.grad is not None:
        wp.launch(
            accumulate_grad_norm_kernel,
            dim=array.shape[0],
            inputs=[array.grad, norm_sq, nonfinite_count],
            device=device,
        )
    norm_sq_np = norm_sq.numpy()
    nonfinite_np = nonfinite_count.numpy()
    return math.sqrt(max(float(norm_sq_np[0]), 0.0)), int(nonfinite_np[0])


def _read_box_state_np(diff_scene, state) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read the per-batch box pose/velocity from a scene state as detached NumPy arrays.

    body_q is laid out as [px, py, pz, qx, qy, qz, qw] (wp.transform) and body_qd as
    [linear_xyz, angular_xyz] (wp.spatial_vector, matching set_batched_box_initial_states_kernel).
    Returning NumPy copies severs the autodiff graph, which is exactly the detach needed at a
    truncated-BPTT segment boundary.
    """
    body_q = np.asarray(state.body_q.numpy(), dtype=np.float32).reshape(-1, 7)
    body_qd = np.asarray(state.body_qd.numpy(), dtype=np.float32).reshape(-1, 6)
    ids = np.asarray(diff_scene.box_body_ids_np, dtype=np.int64)
    positions = np.ascontiguousarray(body_q[ids, 0:3], dtype=np.float32)
    quaternions = np.ascontiguousarray(body_q[ids, 3:7], dtype=np.float32)
    linear_velocity = np.ascontiguousarray(body_qd[ids, 0:3], dtype=np.float32)
    angular_velocity = np.ascontiguousarray(body_qd[ids, 3:6], dtype=np.float32)
    return positions, quaternions, linear_velocity, angular_velocity


def run_truncated_bptt_segments(
    *,
    diff_scene,
    sim_states,
    seg_buffers: ResidualRolloutBuffers,
    seg_activations: MLPActivations,
    params: MLPParameters,
    feature_mean: wp.array,
    feature_inv_std: wp.array,
    mu_features: wp.array,
    trainable_friction: "TrainableFriction | None",
    args: argparse.Namespace,
    batch_trajectories: list,
    truncation_steps: int,
) -> dict:
    """Run one training iteration with truncated BPTT.

    The full window is rolled out continuously across contiguous segments of
    ``truncation_steps`` steps. Each segment opens its own tape and backpropagates only
    within that segment; the carried state is detached (copied through host memory) between
    segments. Per-segment gradients are accumulated into each parameter's ``.grad`` so the
    caller's existing grad-norm / clip / Adam code can run unchanged.
    """
    device = diff_scene.model.device
    seg_len = max(int(truncation_steps), 1)
    total_steps = min(
        max((int(t.num_steps) for t in batch_trajectories), default=0),
        int(args.steps),
    )

    param_arrays = _parameter_arrays(params)
    grad_accum = [np.zeros(int(p.shape[0]), dtype=np.float64) for p in param_arrays]
    friction_accum = None
    if trainable_friction is not None:
        friction_accum = np.zeros(int(trainable_friction.optimizer_params.shape[0]), dtype=np.float64)

    total_loss = 0.0
    trajectory_loss_sum = 0.0
    residual_mean_sum = 0.0
    residual_max = 0.0
    segment_count = 0
    carried_state = None

    for segment_start in range(0, total_steps, seg_len):
        seg_steps = min(seg_len, total_steps - segment_start)
        if seg_steps <= 0:
            break
        segment_trajectories = [
            slice_mujoco_trajectory_time_window(t, start_step=segment_start, window_steps=seg_steps)
            for t in batch_trajectories
        ]
        active_batch_size = assign_rollout_buffer_trajectories(seg_buffers, segment_trajectories)
        if carried_state is not None:
            positions, quaternions, linear_velocity, angular_velocity = carried_state
            seg_buffers.initial_positions.assign(positions)
            seg_buffers.initial_quaternions.assign(quaternions)
            seg_buffers.initial_linear_velocity.assign(linear_velocity)
            seg_buffers.initial_angular_velocity.assign(angular_velocity)

        tape = wp.Tape()
        with tape:
            forward_residual_rollout(
                diff_scene=diff_scene,
                sim_states=sim_states,
                buffers=seg_buffers,
                activations=seg_activations,
                batch_size=active_batch_size,
                params=params,
                feature_mean=feature_mean,
                feature_inv_std=feature_inv_std,
                mu_features=mu_features,
                trainable_friction=trainable_friction,
                args=args,
            )
        tape.backward(seg_buffers.batch_loss)

        for accum, param in zip(grad_accum, param_arrays):
            if param.grad is not None:
                accum += np.asarray(param.grad.numpy(), dtype=np.float64)
        if friction_accum is not None and trainable_friction.optimizer_params.grad is not None:
            friction_accum += np.asarray(trainable_friction.optimizer_params.grad.numpy(), dtype=np.float64)

        total_loss += float(seg_buffers.batch_loss.numpy()[0])
        trajectory_loss_sum += float(np.mean(seg_buffers.loss.numpy()[:active_batch_size]))
        residual_mean_sum += float(np.mean(seg_buffers.residual_norm_mean.numpy()[:active_batch_size]))
        residual_max = max(residual_max, float(np.max(seg_buffers.residual_norm_max.numpy()[:active_batch_size])))
        segment_count += 1

        carried_state = _read_box_state_np(diff_scene, diff_scene.states[seg_steps])
        tape.zero()

    for accum, param in zip(grad_accum, param_arrays):
        if param.grad is not None:
            param.grad.assign(accum.astype(np.float32))
    if friction_accum is not None and trainable_friction.optimizer_params.grad is not None:
        trainable_friction.optimizer_params.grad.assign(friction_accum.astype(np.float32))

    return {
        "loss": total_loss,
        "trajectory_loss": trajectory_loss_sum,
        "residual_mean": residual_mean_sum / max(segment_count, 1),
        "residual_max": residual_max,
        "segments": segment_count,
    }


def expand_trainable_friction_to_active_np(trainable_friction: TrainableFriction) -> np.ndarray:
    optimizer_params = np.asarray(trainable_friction.optimizer_params.numpy(), dtype=np.float32)
    return expand_optimizer_params_to_active(
        optimizer_params,
        trainable_friction.active_param_positions_np,
        parameterization=trainable_friction.parameterization,
    )


def adam_update(params: MLPParameters, adam: MLPAdamState, args: argparse.Namespace, grad_scale: float, device: str) -> None:
    adam.step += 1
    beta1 = float(args.adam_beta1)
    beta2 = float(args.adam_beta2)
    bias_correction1 = 1.0 - beta1 ** adam.step
    bias_correction2 = 1.0 - beta2 ** adam.step
    for param, (moment_1, moment_2) in zip(_parameter_arrays(params), _adam_array_pairs(adam), strict=True):
        if param.grad is None:
            continue
        wp.launch(
            adam_update_kernel,
            dim=param.shape[0],
            inputs=[
                param,
                param.grad,
                moment_1,
                moment_2,
                np.float64(grad_scale),
                np.float64(args.learning_rate),
                np.float64(beta1),
                np.float64(beta2),
                np.float64(args.adam_eps),
                np.float64(bias_correction1),
                np.float64(bias_correction2),
            ],
            device=device,
        )


def adam_update_array(
    *,
    params: wp.array,
    first_moment: wp.array,
    second_moment: wp.array,
    step: int,
    learning_rate: float,
    beta1: float,
    beta2: float,
    eps: float,
    grad_scale: float,
    device: str,
) -> None:
    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step
    wp.launch(
        adam_update_kernel,
        dim=params.shape[0],
        inputs=[
            params,
            params.grad,
            first_moment,
            second_moment,
            np.float64(grad_scale),
            np.float64(learning_rate),
            np.float64(beta1),
            np.float64(beta2),
            np.float64(eps),
            np.float64(bias_correction1),
            np.float64(bias_correction2),
        ],
        device=device,
    )


def clip_friction_params(trainable_friction: TrainableFriction) -> None:
    if trainable_friction.parameterization == "base-delta":
        wp.launch(
            project_base_delta_optimizer_params_kernel,
            dim=1,
            inputs=[
                trainable_friction.optimizer_params,
                float(trainable_friction.min_value),
                float(trainable_friction.max_value),
                int(trainable_friction.left_right_delta_sum_zero),
            ],
            device=trainable_friction.optimizer_params.device,
        )
    else:
        wp.launch(
            clip_optimizer_params_kernel,
            dim=int(trainable_friction.optimizer_params.shape[0]),
            inputs=[
                trainable_friction.optimizer_params,
                float(trainable_friction.min_value),
                float(trainable_friction.max_value),
            ],
            device=trainable_friction.optimizer_params.device,
        )


def refresh_trainable_friction_features(diff_scene, trainable_friction: TrainableFriction) -> None:
    active_params = expand_trainable_friction_to_active_np(trainable_friction)
    mu_features = compute_mu_features_from_active(
        diff_scene,
        trainable_friction.active_indices_np,
        active_params,
        trainable_friction.parameterization,
    )
    trainable_friction.mu_features_np = mu_features
    trainable_friction.mu_features_wp.assign(mu_features)


def refresh_trainable_friction_device_state(trainable_friction: TrainableFriction) -> None:
    trainable_friction.mu_features_wp.zero_()
    wp.launch(
        accumulate_optimizer_mu_features_kernel,
        dim=int(trainable_friction.active_indices_wp.shape[0]),
        inputs=[
            trainable_friction.active_param_positions_wp,
            trainable_friction.optimizer_params,
            int(trainable_friction.parameterization_id),
            trainable_friction.mu_feature_weights_wp,
            trainable_friction.mu_features_wp,
        ],
        device=trainable_friction.optimizer_params.device,
    )
    wp.launch(
        scatter_optimizer_point_friction_kernel,
        dim=int(trainable_friction.active_indices_wp.shape[0]),
        inputs=[
            trainable_friction.active_indices_wp,
            trainable_friction.active_param_positions_wp,
            trainable_friction.optimizer_params,
            int(trainable_friction.parameterization_id),
            trainable_friction.full_point_friction,
        ],
        device=trainable_friction.optimizer_params.device,
    )


def save_adapter_checkpoint(
    *,
    path: Path,
    iteration: int,
    params: MLPParameters,
    adam: MLPAdamState,
    frozen: FrozenFriction,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    loss_history: list[float],
    args: argparse.Namespace,
    trainable_friction: TrainableFriction | None = None,
    checkpoint_kind: str = "current",
    best_loss: float | None = None,
    best_iteration: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    active_params = (
        expand_trainable_friction_to_active_np(trainable_friction)
        if trainable_friction is not None
        else np.asarray(frozen.active_params, dtype=np.float32)
    )
    full_point_friction = (
        np.asarray(trainable_friction.full_point_friction.numpy(), dtype=np.float32)
        if trainable_friction is not None
        else np.asarray(frozen.full_point_friction, dtype=np.float32)
    )
    mu_features = (
        np.asarray(trainable_friction.mu_features_wp.numpy(), dtype=np.float32)
        if trainable_friction is not None
        else np.asarray(frozen.mu_features, dtype=np.float32)
    )
    arrays = {
        "iteration": np.asarray(iteration, dtype=np.int32),
        "w0": params.w0.numpy(),
        "b0": params.b0.numpy(),
        "w1": params.w1.numpy(),
        "b1": params.b1.numpy(),
        "w2": params.w2.numpy(),
        "b2": params.b2.numpy(),
        "w3": params.w3.numpy(),
        "b3": params.b3.numpy(),
        "output_scales": params.output_scales.numpy(),
        "m_w0": adam.m_w0.numpy(),
        "v_w0": adam.v_w0.numpy(),
        "m_b0": adam.m_b0.numpy(),
        "v_b0": adam.v_b0.numpy(),
        "m_w1": adam.m_w1.numpy(),
        "v_w1": adam.v_w1.numpy(),
        "m_b1": adam.m_b1.numpy(),
        "v_b1": adam.v_b1.numpy(),
        "m_w2": adam.m_w2.numpy(),
        "v_w2": adam.v_w2.numpy(),
        "m_b2": adam.m_b2.numpy(),
        "v_b2": adam.v_b2.numpy(),
        "m_w3": adam.m_w3.numpy(),
        "v_w3": adam.v_w3.numpy(),
        "m_b3": adam.m_b3.numpy(),
        "v_b3": adam.v_b3.numpy(),
        "adam_step": np.asarray(adam.step, dtype=np.int32),
        "feature_mean": np.asarray(feature_mean, dtype=np.float32),
        "feature_std": np.asarray(feature_std, dtype=np.float32),
        "mu_features": mu_features,
        "active_indices": np.asarray(frozen.active_indices, dtype=np.int32),
        "active_params": active_params,
        "full_point_friction": full_point_friction,
        "train_friction_end_to_end": np.asarray(trainable_friction is not None),
        "friction_adam_step": np.asarray(0 if trainable_friction is None else trainable_friction.step, dtype=np.int32),
        "loss_history": np.asarray(loss_history, dtype=np.float32),
        "checkpoint_kind": np.asarray(str(checkpoint_kind)),
        "checkpoint_loss": np.asarray(
            float(loss_history[-1]) if loss_history else float("nan"),
            dtype=np.float64,
        ),
        "best_loss": np.asarray(
            float(best_loss) if best_loss is not None else float("nan"),
            dtype=np.float64,
        ),
        "best_iteration": np.asarray(
            int(best_iteration) if best_iteration is not None else -1,
            dtype=np.int32,
        ),
        "friction_checkpoint_path": np.asarray("" if frozen.checkpoint_path is None else str(frozen.checkpoint_path.resolve())),
        "friction_point_cloud_path": np.asarray("" if frozen.point_cloud_path is None else str(frozen.point_cloud_path.resolve())),
        "friction_parameterization": np.asarray(frozen.parameterization),
        "left_right_delta_sum_zero": np.asarray(bool(frozen.left_right_delta_sum_zero)),
        "trajectory_npz_path": np.asarray(str(args.trajectory_npz.resolve())),
        "max_steps": np.asarray(int(args.steps), dtype=np.int32),
        "args_json": np.asarray(json.dumps(_jsonable_args(args), sort_keys=True)),
    }
    if trainable_friction is not None:
        arrays["friction_adam_m"] = trainable_friction.adam_m.numpy()
        arrays["friction_adam_v"] = trainable_friction.adam_v.numpy()
        arrays["friction_optimizer_params"] = trainable_friction.optimizer_params.numpy()
        arrays["friction_active_param_positions"] = trainable_friction.active_param_positions_np
    np.savez_compressed(path, **arrays)


def load_adapter_checkpoint(path: Path, params: MLPParameters, adam: MLPAdamState) -> tuple[int, np.ndarray, np.ndarray, list[float]]:
    with np.load(path, allow_pickle=True) as data:
        for name, array in [
            ("w0", params.w0),
            ("b0", params.b0),
            ("w1", params.w1),
            ("b1", params.b1),
            ("w2", params.w2),
            ("b2", params.b2),
            ("w3", params.w3),
            ("b3", params.b3),
        ]:
            if name not in data.files:
                raise ValueError(f"{path} is missing residual adapter parameter {name!r}")
            values = np.asarray(data[name], dtype=np.float32)
            expected_shape = array.numpy().shape
            if values.shape != expected_shape:
                raise ValueError(f"{path} parameter {name!r} shape {values.shape} does not match expected {expected_shape}")
            array.assign(values)

        for name, array in [
            ("m_w0", adam.m_w0),
            ("v_w0", adam.v_w0),
            ("m_b0", adam.m_b0),
            ("v_b0", adam.v_b0),
            ("m_w1", adam.m_w1),
            ("v_w1", adam.v_w1),
            ("m_b1", adam.m_b1),
            ("v_b1", adam.v_b1),
            ("m_w2", adam.m_w2),
            ("v_w2", adam.v_w2),
            ("m_b2", adam.m_b2),
            ("v_b2", adam.v_b2),
            ("m_w3", adam.m_w3),
            ("v_w3", adam.v_w3),
            ("m_b3", adam.m_b3),
            ("v_b3", adam.v_b3),
        ]:
            if name not in data.files:
                continue
            values = np.asarray(data[name], dtype=np.float64)
            expected_shape = array.numpy().shape
            if values.shape != expected_shape:
                raise ValueError(f"{path} Adam state {name!r} shape {values.shape} does not match expected {expected_shape}")
            array.assign(values)

        if "output_scales" in data.files:
            output_scales = np.asarray(data["output_scales"], dtype=np.float32)
            expected_shape = params.output_scales.numpy().shape
            if output_scales.shape != expected_shape:
                raise ValueError(f"{path} output_scales shape {output_scales.shape} does not match expected {expected_shape}")
            params.output_scales.assign(output_scales)

        if "feature_mean" not in data.files or "feature_std" not in data.files:
            raise ValueError(f"{path} is missing feature normalization arrays")
        iteration = int(np.asarray(data["iteration"]).item()) if "iteration" in data.files else 0
        adam.step = int(np.asarray(data["adam_step"]).item()) if "adam_step" in data.files else iteration
        feature_mean = np.asarray(data["feature_mean"], dtype=np.float32)
        feature_std = np.asarray(data["feature_std"], dtype=np.float32)
        loss_history = (
            [float(value) for value in np.asarray(data["loss_history"], dtype=np.float32).tolist()]
            if "loss_history" in data.files
            else []
        )
        return iteration, feature_mean, feature_std, loss_history


def load_trainable_friction_checkpoint(path: Path, trainable_friction: TrainableFriction | None, diff_scene) -> None:
    if trainable_friction is None:
        return
    with np.load(path, allow_pickle=True) as data:
        if "friction_optimizer_params" in data.files:
            values = np.asarray(data["friction_optimizer_params"], dtype=np.float32)
            expected_shape = trainable_friction.optimizer_params.numpy().shape
            if values.shape != expected_shape:
                raise ValueError(
                    f"{path} friction_optimizer_params shape {values.shape} does not match expected {expected_shape}"
                )
            trainable_friction.optimizer_params.assign(values)
        elif "active_params" in data.files:
            optimizer_params, _ = infer_optimizer_params_from_active(
                diff_scene=diff_scene,
                active_indices=trainable_friction.active_indices_np,
                active_params=np.asarray(data["active_params"], dtype=np.float32),
                parameterization=trainable_friction.parameterization,
            )
            trainable_friction.optimizer_params.assign(np.asarray(optimizer_params, dtype=np.float32))
        else:
            return

        if "friction_adam_m" in data.files:
            m = np.asarray(data["friction_adam_m"], dtype=np.float64)
            if m.shape == trainable_friction.adam_m.numpy().shape:
                trainable_friction.adam_m.assign(m)
        if "friction_adam_v" in data.files:
            v = np.asarray(data["friction_adam_v"], dtype=np.float64)
            if v.shape == trainable_friction.adam_v.numpy().shape:
                trainable_friction.adam_v.assign(v)
        if "friction_adam_step" in data.files:
            trainable_friction.step = int(np.asarray(data["friction_adam_step"]).item())
        clip_friction_params(trainable_friction)


def build_trainable_friction(frozen: FrozenFriction, diff_scene, args: argparse.Namespace, device: str) -> TrainableFriction | None:
    if not bool(args.train_friction_end_to_end):
        return None
    optimizer_params, active_param_positions = infer_optimizer_params_from_active(
        diff_scene=diff_scene,
        active_indices=np.asarray(frozen.active_indices, dtype=np.int32),
        active_params=np.asarray(frozen.active_params, dtype=np.float32),
        parameterization=frozen.parameterization,
    )
    optimizer_params = np.asarray(optimizer_params, dtype=np.float32)
    optimizer_wp = wp.array(optimizer_params, dtype=wp.float32, device=device, requires_grad=True)
    optimizer_wp.grad = wp.zeros_like(optimizer_wp)
    full_point_friction = wp.array(
        np.asarray(frozen.full_point_friction, dtype=np.float32),
        dtype=wp.float32,
        device=device,
        requires_grad=True,
    )
    full_point_friction.grad = wp.zeros_like(full_point_friction)
    zeros = np.zeros(len(optimizer_params), dtype=np.float64)
    mu_features_wp = wp.array(
        np.asarray(frozen.mu_features, dtype=np.float32),
        dtype=wp.float32,
        device=device,
        requires_grad=True,
    )
    mu_features_wp.grad = wp.zeros_like(mu_features_wp)
    return TrainableFriction(
        mode="end_to_end",
        active_indices_np=np.asarray(frozen.active_indices, dtype=np.int32),
        active_indices_wp=wp.array(np.asarray(frozen.active_indices, dtype=np.int32), dtype=wp.int32, device=device),
        active_param_positions_np=np.asarray(active_param_positions, dtype=np.int32),
        active_param_positions_wp=wp.array(np.asarray(active_param_positions, dtype=np.int32), dtype=wp.int32, device=device),
        mu_feature_weights_wp=wp.array(
            build_mu_feature_weights(diff_scene, frozen.active_indices),
            dtype=wp.float32,
            device=device,
        ),
        optimizer_params=optimizer_wp,
        full_point_friction=full_point_friction,
        adam_m=wp.array(zeros, dtype=wp.float64, device=device),
        adam_v=wp.array(zeros, dtype=wp.float64, device=device),
        mu_features_np=np.asarray(frozen.mu_features, dtype=np.float32),
        mu_features_wp=mu_features_wp,
        step=0,
        parameterization=frozen.parameterization,
        parameterization_id=parameterization_id(frozen.parameterization),
        left_right_delta_sum_zero=bool(frozen.left_right_delta_sum_zero),
        min_value=float(args.min_point_friction),
        max_value=float(args.max_point_friction),
    )


def _jsonable_args(args: argparse.Namespace) -> dict:
    result = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, list):
            result[key] = [str(item) if isinstance(item, Path) else item for item in value]
        elif isinstance(value, tuple):
            result[key] = list(value)
        else:
            result[key] = value
    return result


def evaluate_trajectories(
    *,
    name: str,
    trajectories: list[MujocoTrajectory],
    diff_scene,
    sim_states,
    params: MLPParameters,
    frozen: FrozenFriction,
    feature_mean_wp: wp.array,
    feature_inv_std_wp: wp.array,
    mu_features_wp: wp.array,
    trainable_friction: TrainableFriction | None,
    initial_body_q: np.ndarray,
    initial_body_qd: np.ndarray,
    args: argparse.Namespace,
    batch_size: int,
) -> dict:
    if not trajectories:
        return {
            "name": name,
            "count": 0,
            "mean_loss": float("nan"),
            "median_loss": float("nan"),
        }
    if args.eval_trajectory_limit is not None:
        trajectories = trajectories[: max(int(args.eval_trajectory_limit), 0)]
    trajectories = [
        slice_mujoco_trajectory_time_window(trajectory, start_step=0, window_steps=int(args.steps))
        if trajectory.num_steps > int(args.steps)
        else trajectory
        for trajectory in trajectories
    ]
    device = str(diff_scene.torch_device)
    all_loss: list[np.ndarray] = []
    all_position_loss: list[np.ndarray] = []
    all_orientation_loss: list[np.ndarray] = []
    all_linear_velocity_loss: list[np.ndarray] = []
    all_angular_velocity_loss: list[np.ndarray] = []
    all_residual_norm: list[np.ndarray] = []
    all_residual_energy: list[np.ndarray] = []
    all_residual_max: list[np.ndarray] = []
    eval_buffers = build_rollout_buffers(
        device=device,
        point_count=len(diff_scene.local_surface_points_np),
        full_point_friction=frozen.full_point_friction,
        batch_capacity=min(max(int(batch_size), 1), max(len(trajectories), 1)),
        step_capacity=int(args.steps),
    )
    eval_activations = build_activation_buffers(
        device=device,
        batch_capacity=eval_buffers.batch_capacity,
        step_capacity=eval_buffers.step_capacity,
    )

    for batch_start in range(0, len(trajectories), batch_size):
        batch_trajectories = trajectories[batch_start : batch_start + batch_size]
        active_batch_size = assign_rollout_buffer_trajectories(eval_buffers, batch_trajectories)
        clear_gradients(params, eval_activations, eval_buffers, trainable_friction=trainable_friction)
        reset_scene_states(diff_scene, initial_body_q, initial_body_qd)
        forward_residual_rollout(
            diff_scene=diff_scene,
            sim_states=sim_states,
            buffers=eval_buffers,
            activations=eval_activations,
            batch_size=active_batch_size,
            params=params,
            feature_mean=feature_mean_wp,
            feature_inv_std=feature_inv_std_wp,
            mu_features=mu_features_wp,
            trainable_friction=trainable_friction,
            args=args,
        )
        all_loss.append(eval_buffers.loss.numpy()[:active_batch_size].copy())
        all_position_loss.append(eval_buffers.position_loss.numpy()[:active_batch_size].copy())
        all_orientation_loss.append(eval_buffers.orientation_loss.numpy()[:active_batch_size].copy())
        all_linear_velocity_loss.append(eval_buffers.linear_velocity_loss.numpy()[:active_batch_size].copy())
        all_angular_velocity_loss.append(eval_buffers.angular_velocity_loss.numpy()[:active_batch_size].copy())
        all_residual_norm.append(eval_buffers.residual_norm_mean.numpy()[:active_batch_size].copy())
        all_residual_energy.append(eval_buffers.residual_energy_mean.numpy()[:active_batch_size].copy())
        all_residual_max.append(eval_buffers.residual_norm_max.numpy()[:active_batch_size].copy())

    losses = np.concatenate(all_loss)
    result = {
        "name": name,
        "count": int(len(losses)),
        "mean_loss": float(np.mean(losses)),
        "median_loss": float(np.median(losses)),
        "mean_position_loss": float(np.mean(np.concatenate(all_position_loss))),
        "mean_orientation_loss": float(np.mean(np.concatenate(all_orientation_loss))),
        "mean_linear_velocity_loss": float(np.mean(np.concatenate(all_linear_velocity_loss))),
        "mean_angular_velocity_loss": float(np.mean(np.concatenate(all_angular_velocity_loss))),
        "mean_residual_norm": float(np.mean(np.concatenate(all_residual_norm))),
        "mean_residual_energy": float(np.mean(np.concatenate(all_residual_energy))),
        "max_residual_norm": float(np.max(np.concatenate(all_residual_max))),
    }
    log_message(
        f"eval {name}: count={result['count']} mean_loss={result['mean_loss']:.6g} "
        f"median_loss={result['median_loss']:.6g} mean_residual_norm={result['mean_residual_norm']:.6g} "
        f"max_residual_norm={result['max_residual_norm']:.6g}"
    )
    return result


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive for residual closed-loop training.")
    if args.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive.")
    if args.friction_checkpoint is None and not bool(args.train_friction_end_to_end):
        raise ValueError("--friction-checkpoint is required unless --train-friction-end-to-end is set.")
    if float(args.max_point_friction) < float(args.min_point_friction):
        raise ValueError("--max-point-friction must be >= --min-point-friction.")

    rng = np.random.default_rng(int(args.seed))
    maybe_infer_scene_from_point_cloud(args)

    load_max_steps = _resolve_load_max_steps(args)
    log_message(
        f"loading residual training trajectories from {args.trajectory_npz.resolve()} "
        f"load_max_steps={load_max_steps if load_max_steps is not None else 'full'}"
    )
    collection = load_mujoco_trajectories(args.trajectory_npz, load_max_steps, args.max_trajectories)
    trajectories = collection.trajectories
    train_trajectories, val_trajectories, test_trajectories = split_trajectories(
        trajectories,
        train_fraction=float(args.train_fraction),
        val_fraction=float(args.val_fraction),
        rng=rng,
    )
    args.steps = _resolve_window_steps(args, collection.max_steps)
    args.dt = float(trajectories[0].timestep)
    args.batch_capacity = max(int(args.batch_size), int(args.eval_batch_size), 1)

    log_message(
        f"split trajectories train={len(train_trajectories)} val={len(val_trajectories)} "
        f"test_lite={len(test_trajectories)} source_max_steps={collection.max_steps} "
        f"rollout_steps={args.steps} random_time_windows={int(args.random_time_windows)} dt={args.dt:.6f}"
    )

    log_message(f"building Newton scene on device={args.device if args.device is not None else 'auto'}")
    diff_scene = build_diff_scene(args)
    sim_states = [diff_scene.model.state() for _ in range(max(int(args.steps), 1))]
    initial_body_q = diff_scene.states[0].body_q.numpy().copy()
    initial_body_qd = diff_scene.states[0].body_qd.numpy().copy()
    frozen = resolve_initial_friction_state(args, diff_scene, train_trajectories)
    friction_source = (
        str(frozen.checkpoint_path.resolve())
        if frozen.checkpoint_path is not None
        else "initialized"
    )
    log_message(
        f"loaded friction state source={friction_source} "
        f"parameterization={frozen.parameterization} active_points={len(frozen.active_indices)} "
        f"mu_mean_left_right={frozen.mu_features.tolist()}"
    )
    wandb_run = init_wandb(args, collection, frozen.active_indices)
    if wandb_run is not None:
        log_message(
            f"W&B enabled | project={args.wandb_project} | "
            f"run={wandb_run.name} | mode={args.wandb_mode}"
        )

    device = str(diff_scene.torch_device)
    feature_mean, feature_std = compute_feature_stats(train_trajectories, frozen.mu_features)
    params, adam = initialize_mlp_parameters(args, device, rng)
    trainable_friction = build_trainable_friction(frozen, diff_scene, args, device)
    if trainable_friction is not None:
        log_message(
            f"end_to_end_friction=1 active_points={len(trainable_friction.active_indices_np)} "
            f"friction_lr={float(args.friction_learning_rate if args.friction_learning_rate is not None else args.learning_rate):.3g} "
            f"friction_clip=[{float(args.min_point_friction):.3g}, {float(args.max_point_friction):.3g}]"
        )
    else:
        log_message("end_to_end_friction=0 frozen friction parameters")
    loss_history: list[float] = []
    start_iteration = 1
    if args.resume_adapter is not None:
        resume_iteration, feature_mean, feature_std, loss_history = load_adapter_checkpoint(args.resume_adapter, params, adam)
        load_trainable_friction_checkpoint(args.resume_adapter, trainable_friction, diff_scene)
        if trainable_friction is not None:
            refresh_trainable_friction_device_state(trainable_friction)
            frozen.active_params = expand_trainable_friction_to_active_np(trainable_friction)
            frozen.full_point_friction = np.asarray(trainable_friction.full_point_friction.numpy(), dtype=np.float32)
            frozen.mu_features = np.asarray(trainable_friction.mu_features_wp.numpy(), dtype=np.float32)
        start_iteration = resume_iteration + 1
        log_message(f"resumed residual adapter {args.resume_adapter.resolve()} at iteration={resume_iteration}")

    feature_mean = np.asarray(feature_mean, dtype=np.float32)
    feature_std = np.asarray(feature_std, dtype=np.float32)
    feature_mean[8:11] = 0.0
    feature_std[8:11] = 1.0
    feature_inv_std = (1.0 / np.maximum(feature_std, 1.0e-6)).astype(np.float32)
    assert_array_finite("feature_mean", feature_mean, context="feature normalization")
    assert_array_finite("feature_std", feature_std, context="feature normalization")
    feature_mean_wp = wp.array(feature_mean, dtype=wp.float32, device=device)
    feature_inv_std_wp = wp.array(feature_inv_std, dtype=wp.float32, device=device)
    mu_features_wp = (
        trainable_friction.mu_features_wp
        if trainable_friction is not None
        else wp.array(frozen.mu_features, dtype=wp.float32, device=device)
    )

    checkpoint_path = args.experiment_dir / f"{args.experiment_dir.name}.npz"
    best_checkpoint_path = args.experiment_dir / f"{args.experiment_dir.name}_best.npz"
    metrics_path = args.experiment_dir / f"{args.experiment_dir.name}_metrics.json"

    training_start = time.time()
    log_message(
        "starting residual training "
        f"iters={int(args.opt_iters)} batch_size={int(args.batch_size)} "
        f"lr={float(args.learning_rate):.3g}"
    )
    train_buffers = build_rollout_buffers(
        device=device,
        point_count=len(diff_scene.local_surface_points_np),
        full_point_friction=frozen.full_point_friction,
        batch_capacity=int(args.batch_capacity),
        step_capacity=int(args.steps),
    )
    train_activations = build_activation_buffers(
        device=device,
        batch_capacity=train_buffers.batch_capacity,
        step_capacity=train_buffers.step_capacity,
    )

    truncation_steps = int(getattr(args, "bptt_truncation_steps", 0) or 0)
    truncation_enabled = truncation_steps > 0 and truncation_steps < int(args.steps)
    segment_buffers = None
    segment_activations = None
    if truncation_enabled:
        segment_buffers = build_rollout_buffers(
            device=device,
            point_count=len(diff_scene.local_surface_points_np),
            full_point_friction=frozen.full_point_friction,
            batch_capacity=int(args.batch_capacity),
            step_capacity=truncation_steps,
        )
        segment_activations = build_activation_buffers(
            device=device,
            batch_capacity=segment_buffers.batch_capacity,
            step_capacity=segment_buffers.step_capacity,
        )
        log_message(
            f"truncated BPTT enabled: segment_steps={truncation_steps} "
            f"window_steps={int(args.steps)} segments_per_window~={math.ceil(int(args.steps) / truncation_steps)}"
        )

    if loss_history:
        loss_history_np = np.asarray(loss_history, dtype=np.float64)
        if np.isfinite(loss_history_np).any():
            best_loss = float(np.nanmin(loss_history_np))
            best_iteration = int(np.nanargmin(loss_history_np)) + 1
        else:
            best_loss = float("inf")
            best_iteration = None
    else:
        best_loss = float("inf")
        best_iteration = None
    skipped_nonfinite_batches = 0
    consecutive_nonfinite_batches = 0
    final_iteration = start_iteration + max(int(args.opt_iters), 0) - 1
    for iteration in range(start_iteration, final_iteration + 1):
        iteration_start = time.time()
        batch_indices = sample_training_batch_indices(len(train_trajectories), int(args.batch_size), rng)
        batch_source = [train_trajectories[int(idx)] for idx in batch_indices]
        batch_trajectories, batch_window_starts = sample_training_time_windows(
            trajectories=batch_source,
            window_steps=int(args.steps),
            rng=rng,
            enabled=bool(args.random_time_windows),
        )
        active_batch_size = assign_rollout_buffer_trajectories(train_buffers, batch_trajectories)
        clear_gradients(params, train_activations, train_buffers, trainable_friction)
        if truncation_enabled:
            clear_gradients(params, segment_activations, segment_buffers, trainable_friction)
        reset_scene_states(diff_scene, initial_body_q, initial_body_qd)

        truncated_diag = None
        if truncation_enabled:
            tape = None
            active_buffers = segment_buffers
            truncated_diag = run_truncated_bptt_segments(
                diff_scene=diff_scene,
                sim_states=sim_states,
                seg_buffers=segment_buffers,
                seg_activations=segment_activations,
                params=params,
                feature_mean=feature_mean_wp,
                feature_inv_std=feature_inv_std_wp,
                mu_features=mu_features_wp,
                trainable_friction=trainable_friction,
                args=args,
                batch_trajectories=batch_trajectories,
                truncation_steps=truncation_steps,
            )
        else:
            active_buffers = train_buffers
            tape = wp.Tape()
            with tape:
                forward_residual_rollout(
                    diff_scene=diff_scene,
                    sim_states=sim_states,
                    buffers=train_buffers,
                    activations=train_activations,
                    batch_size=active_batch_size,
                    params=params,
                    feature_mean=feature_mean_wp,
                    feature_inv_std=feature_inv_std_wp,
                    mu_features=mu_features_wp,
                    trainable_friction=trainable_friction,
                    args=args,
                )
            tape.backward(train_buffers.batch_loss)

        grad_norm, nonfinite_grad_count = compute_global_grad_norm(params, device)
        friction_grad_norm = 0.0
        friction_nonfinite_grad_count = 0
        if trainable_friction is not None:
            friction_grad_norm, friction_nonfinite_grad_count = compute_array_grad_norm(
                trainable_friction.optimizer_params,
                device,
            )
        if nonfinite_grad_count != 0 or friction_nonfinite_grad_count != 0:
            if tape is not None:
                tape.zero()
            if not bool(args.skip_nonfinite_grad_batches):
                if nonfinite_grad_count != 0:
                    raise FloatingPointError(
                        f"iter={iteration:04d} nonfinite residual-MLP gradients={nonfinite_grad_count}"
                    )
                raise FloatingPointError(
                    f"iter={iteration:04d} nonfinite friction gradients={friction_nonfinite_grad_count}"
                )
            skipped_nonfinite_batches += 1
            consecutive_nonfinite_batches += 1
            log_message(
                f"iter={iteration:04d} skipping nonfinite-gradient batch "
                f"residual_mlp_nonfinite={nonfinite_grad_count} friction_nonfinite={friction_nonfinite_grad_count} "
                f"skipped_total={skipped_nonfinite_batches} consecutive={consecutive_nonfinite_batches}"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/nonfinite_grad_batch": 1.0,
                        "train/skipped_nonfinite_total": float(skipped_nonfinite_batches),
                        "train/consecutive_nonfinite": float(consecutive_nonfinite_batches),
                        "grads/residual_mlp_nonfinite_count": float(nonfinite_grad_count),
                        "grads/friction_nonfinite_count": float(friction_nonfinite_grad_count),
                    },
                    step=iteration,
                )
            if (
                int(args.max_consecutive_nonfinite_batches) > 0
                and consecutive_nonfinite_batches >= int(args.max_consecutive_nonfinite_batches)
            ):
                raise FloatingPointError(
                    f"iter={iteration:04d} aborting after {consecutive_nonfinite_batches} consecutive "
                    f"nonfinite-gradient batches; lower the learning rate, output scale, or rollout length."
                )
            continue
        consecutive_nonfinite_batches = 0

        if truncated_diag is not None:
            loss_value = float(truncated_diag["loss"])
            trajectory_loss_value = float(truncated_diag["trajectory_loss"])
            residual_norm_value = float(truncated_diag["residual_mean"])
            residual_max_value = float(truncated_diag["residual_max"])
        else:
            loss_value = float(train_buffers.batch_loss.numpy()[0])
            trajectory_loss_value = float(np.mean(train_buffers.loss.numpy()[:active_batch_size]))
            residual_norm_value = float(np.mean(train_buffers.residual_norm_mean.numpy()[:active_batch_size]))
            residual_max_value = float(np.max(train_buffers.residual_norm_max.numpy()[:active_batch_size]))
        loss_history.append(loss_value)

        is_best_loss = np.isfinite(loss_value) and loss_value < best_loss
        if is_best_loss:
            best_loss = loss_value
            best_iteration = int(iteration)
            save_adapter_checkpoint(
                path=best_checkpoint_path,
                iteration=iteration,
                params=params,
                adam=adam,
                frozen=frozen,
                feature_mean=feature_mean,
                feature_std=feature_std,
                loss_history=loss_history,
                args=args,
                trainable_friction=trainable_friction,
                checkpoint_kind="best",
                best_loss=best_loss,
                best_iteration=best_iteration,
            )
            log_message(
                f"best checkpoint written to {best_checkpoint_path.resolve()} "
                f"best_loss={best_loss:.6g} iteration={best_iteration}"
            )

        grad_clip_norm = float(args.grad_clip_norm) if args.grad_clip_norm is not None else 0.0
        if grad_clip_norm > 0.0 and grad_norm > grad_clip_norm:
            grad_scale = grad_clip_norm / max(grad_norm, 1.0e-30)
            clipped_grad_norm = grad_clip_norm
        else:
            grad_scale = 1.0
            clipped_grad_norm = grad_norm
        adam_update(params, adam, args, grad_scale, device)
        friction_clipped_grad_norm = friction_grad_norm
        if trainable_friction is not None:
            trainable_friction.step += 1
            friction_lr = float(args.friction_learning_rate if args.friction_learning_rate is not None else args.learning_rate)
            if grad_clip_norm > 0.0 and friction_grad_norm > grad_clip_norm:
                friction_grad_scale = grad_clip_norm / max(friction_grad_norm, 1.0e-30)
                friction_clipped_grad_norm = grad_clip_norm
            else:
                friction_grad_scale = 1.0
            adam_update_array(
                params=trainable_friction.optimizer_params,
                first_moment=trainable_friction.adam_m,
                second_moment=trainable_friction.adam_v,
                step=trainable_friction.step,
                learning_rate=friction_lr,
                beta1=float(args.adam_beta1),
                beta2=float(args.adam_beta2),
                eps=float(args.adam_eps),
                grad_scale=friction_grad_scale,
                device=device,
            )
            clip_friction_params(trainable_friction)
            refresh_trainable_friction_device_state(trainable_friction)

        if trainable_friction is not None:
            frozen.mu_features = np.asarray(trainable_friction.mu_features_wp.numpy(), dtype=np.float32)
        if tape is not None:
            tape.zero()

        should_log = (
            iteration == start_iteration
            or iteration % max(int(args.log_every), 1) == 0
            or iteration == final_iteration
        )
        if should_log:
            window_start_min = -1
            window_start_max = -1
            window_start_mean = -1.0
            if len(batch_window_starts) > 0 and bool(args.random_time_windows):
                window_start_min = int(np.min(batch_window_starts))
                window_start_max = int(np.max(batch_window_starts))
                window_start_mean = float(np.mean(batch_window_starts))
                window_msg = (
                    f"window_start_min={window_start_min} "
                    f"window_start_max={window_start_max}"
                )
            else:
                window_msg = "window_start_min=-1 window_start_max=-1"
            iteration_elapsed = time.time() - iteration_start
            log_message(
                f"iter={iteration:04d} loss={loss_value:.6g} trajectory_loss={trajectory_loss_value:.6g} "
                f"grad_norm={grad_norm:.6g} clipped_grad_norm={clipped_grad_norm:.6g} "
                f"friction_grad_norm={friction_grad_norm:.6g} friction_clipped_grad_norm={friction_clipped_grad_norm:.6g} "
                f"mu_mean_left_right={frozen.mu_features.tolist()} "
                f"mean_residual_norm={residual_norm_value:.6g} max_residual_norm={residual_max_value:.6g} "
                f"{window_msg} elapsed={iteration_elapsed:.2f}s"
            )
            if wandb_run is not None:
                raw_position_loss_value = float(np.mean(active_buffers.position_loss.numpy()[:active_batch_size]))
                raw_orientation_loss_value = float(np.mean(active_buffers.orientation_loss.numpy()[:active_batch_size]))
                raw_linear_velocity_loss_value = float(np.mean(active_buffers.linear_velocity_loss.numpy()[:active_batch_size]))
                raw_angular_velocity_loss_value = float(np.mean(active_buffers.angular_velocity_loss.numpy()[:active_batch_size]))
                residual_energy_value = float(np.mean(active_buffers.residual_energy_mean.numpy()[:active_batch_size]))
                if trainable_friction is not None:
                    active_params_np = expand_trainable_friction_to_active_np(trainable_friction)
                    frozen.active_params = active_params_np
                else:
                    active_params_np = np.asarray(frozen.active_params, dtype=np.float32)
                log_payload = build_wandb_log_payload(
                    loss_value=loss_value,
                    position_loss_value=float(args.position_loss_weight) * raw_position_loss_value,
                    orientation_loss_value=float(args.orientation_loss_weight) * raw_orientation_loss_value,
                    linear_velocity_loss_value=float(args.linear_velocity_loss_weight) * raw_linear_velocity_loss_value,
                    angular_velocity_loss_value=float(args.angular_velocity_loss_weight) * raw_angular_velocity_loss_value,
                    raw_position_loss_value=raw_position_loss_value,
                    raw_orientation_loss_value=raw_orientation_loss_value,
                    raw_linear_velocity_loss_value=raw_linear_velocity_loss_value,
                    raw_angular_velocity_loss_value=raw_angular_velocity_loss_value,
                    grad_value=None,
                    active_params=active_params_np,
                    active_indices=np.asarray(frozen.active_indices, dtype=np.int32),
                )
                log_payload["train/trajectory_loss"] = float(trajectory_loss_value)
                log_payload["train/best_loss"] = float(best_loss)
                log_payload["train/iteration_elapsed_sec"] = float(iteration_elapsed)
                log_payload["train/end_to_end_friction"] = float(trainable_friction is not None)
                log_payload["train/batch_size"] = float(active_batch_size)
                log_payload["grads/residual_mlp_norm"] = float(grad_norm)
                log_payload["grads/residual_mlp_clipped_norm"] = float(clipped_grad_norm)
                log_payload["grads/residual_mlp_clip_scale"] = float(grad_scale)
                log_payload["grads/friction_norm"] = float(friction_grad_norm)
                log_payload["grads/friction_clipped_norm"] = float(friction_clipped_grad_norm)
                log_payload["params/optimizer_parameter_count"] = float(
                    0 if trainable_friction is None else int(trainable_friction.optimizer_params.shape[0])
                )
                log_payload["params/mu_feature_mean"] = float(frozen.mu_features[0])
                log_payload["params/mu_left_mean"] = float(frozen.mu_features[1])
                log_payload["params/mu_right_mean"] = float(frozen.mu_features[2])
                log_payload["residual/norm_mean"] = float(residual_norm_value)
                log_payload["residual/norm_max"] = float(residual_max_value)
                log_payload["residual/energy_mean"] = float(residual_energy_value)
                log_payload["regularization/residual_l2_weight"] = float(args.residual_l2_weight)
                log_payload["regularization/residual_smoothness_weight"] = float(args.residual_smoothness_weight)
                if bool(args.random_time_windows):
                    log_payload["time_window/start_min"] = float(window_start_min)
                    log_payload["time_window/start_max"] = float(window_start_max)
                    log_payload["time_window/start_mean"] = float(window_start_mean)
                    log_payload["time_window/steps"] = float(args.steps)
                wandb_run.log(log_payload, step=iteration)

        if int(args.checkpoint_every) > 0 and (
            iteration % int(args.checkpoint_every) == 0 or iteration == final_iteration
        ):
            if trainable_friction is not None:
                refresh_trainable_friction_device_state(trainable_friction)
                frozen.active_params = expand_trainable_friction_to_active_np(trainable_friction)
                frozen.full_point_friction = np.asarray(trainable_friction.full_point_friction.numpy(), dtype=np.float32)
                frozen.mu_features = np.asarray(trainable_friction.mu_features_wp.numpy(), dtype=np.float32)
            save_adapter_checkpoint(
                path=checkpoint_path,
                iteration=iteration,
                params=params,
                adam=adam,
                frozen=frozen,
                feature_mean=feature_mean,
                feature_std=feature_std,
                loss_history=loss_history,
                args=args,
                trainable_friction=trainable_friction,
                checkpoint_kind="current",
                best_loss=best_loss if np.isfinite(best_loss) else None,
                best_iteration=best_iteration,
            )
            log_message(f"checkpoint written to {checkpoint_path.resolve()}")

    log_message(
        f"training_elapsed={time.time() - training_start:.2f}s "
        f"skipped_nonfinite_batches={skipped_nonfinite_batches}"
    )

    eval_results = []
    if args.eval_after_train:
        eval_results.append(
            evaluate_trajectories(
                name="residual_train_split",
                trajectories=train_trajectories,
                diff_scene=diff_scene,
                sim_states=sim_states,
                params=params,
                frozen=frozen,
                feature_mean_wp=feature_mean_wp,
                feature_inv_std_wp=feature_inv_std_wp,
                mu_features_wp=mu_features_wp,
                trainable_friction=trainable_friction,
                initial_body_q=initial_body_q,
                initial_body_qd=initial_body_qd,
                args=args,
                batch_size=int(args.eval_batch_size),
            )
        )
        eval_results.append(
            evaluate_trajectories(
                name="residual_val_split",
                trajectories=val_trajectories,
                diff_scene=diff_scene,
                sim_states=sim_states,
                params=params,
                frozen=frozen,
                feature_mean_wp=feature_mean_wp,
                feature_inv_std_wp=feature_inv_std_wp,
                mu_features_wp=mu_features_wp,
                trainable_friction=trainable_friction,
                initial_body_q=initial_body_q,
                initial_body_qd=initial_body_qd,
                args=args,
                batch_size=int(args.eval_batch_size),
            )
        )
        eval_results.append(
            evaluate_trajectories(
                name="residual_test_lite_split",
                trajectories=test_trajectories,
                diff_scene=diff_scene,
                sim_states=sim_states,
                params=params,
                frozen=frozen,
                feature_mean_wp=feature_mean_wp,
                feature_inv_std_wp=feature_inv_std_wp,
                mu_features_wp=mu_features_wp,
                trainable_friction=trainable_friction,
                initial_body_q=initial_body_q,
                initial_body_qd=initial_body_qd,
                args=args,
                batch_size=int(args.eval_batch_size),
            )
        )

    for eval_dataset in args.eval_dataset:
        eval_collection = load_mujoco_trajectories(eval_dataset, args.steps, None)
        eval_results.append(
            evaluate_trajectories(
                name=eval_dataset.stem,
                trajectories=eval_collection.trajectories,
                diff_scene=diff_scene,
                sim_states=sim_states,
                params=params,
                frozen=frozen,
                feature_mean_wp=feature_mean_wp,
                feature_inv_std_wp=feature_inv_std_wp,
                mu_features_wp=mu_features_wp,
                trainable_friction=trainable_friction,
                initial_body_q=initial_body_q,
                initial_body_qd=initial_body_qd,
                args=args,
                batch_size=int(args.eval_batch_size),
            )
        )
        if args.eval_heldout_start is not None:
            heldout_start = max(int(args.eval_heldout_start), 0)
            heldout_end = (
                len(eval_collection.trajectories)
                if args.eval_heldout_end is None
                else min(int(args.eval_heldout_end), len(eval_collection.trajectories))
            )
            if heldout_start < heldout_end:
                eval_results.append(
                    evaluate_trajectories(
                        name=f"{eval_dataset.stem}_heldout_{heldout_start}_{heldout_end}",
                        trajectories=eval_collection.trajectories[heldout_start:heldout_end],
                        diff_scene=diff_scene,
                        sim_states=sim_states,
                        params=params,
                        frozen=frozen,
                        feature_mean_wp=feature_mean_wp,
                        feature_inv_std_wp=feature_inv_std_wp,
                        mu_features_wp=mu_features_wp,
                        trainable_friction=trainable_friction,
                        initial_body_q=initial_body_q,
                        initial_body_qd=initial_body_qd,
                        args=args,
                        batch_size=int(args.eval_batch_size),
                    )
                )

    if eval_results:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump({"eval": eval_results}, f, indent=2, sort_keys=True)
        log_message(f"metrics written to {metrics_path.resolve()}")

    if int(args.opt_iters) == 0 or not checkpoint_path.exists():
        save_adapter_checkpoint(
            path=checkpoint_path,
            iteration=max(final_iteration, 0),
            params=params,
            adam=adam,
            frozen=frozen,
            feature_mean=feature_mean,
            feature_std=feature_std,
            loss_history=loss_history,
            args=args,
            trainable_friction=trainable_friction,
            checkpoint_kind="current",
            best_loss=best_loss if np.isfinite(best_loss) else None,
            best_iteration=best_iteration,
        )
        log_message(f"checkpoint written to {checkpoint_path.resolve()}")

    if wandb_run is not None:
        wandb_run.summary["surface_points"] = int(len(diff_scene.local_surface_points_np))
        wandb_run.summary["active_contact_points"] = int(len(frozen.active_indices))
        wandb_run.summary["friction_parameterization"] = frozen.parameterization
        wandb_run.summary["train_friction_end_to_end"] = bool(trainable_friction is not None)
        wandb_run.summary["best_training_loss"] = float(best_loss)
        wandb_run.summary["skipped_nonfinite_batches"] = int(skipped_nonfinite_batches)
        wandb_run.summary["final_training_loss"] = float(loss_history[-1]) if loss_history else float("nan")
        wandb_run.summary["mu_mean"] = float(np.mean(frozen.active_params))
        wandb_run.summary["mu_std"] = float(np.std(frozen.active_params))
        wandb_run.summary["mu_min"] = float(np.min(frozen.active_params))
        wandb_run.summary["mu_max"] = float(np.max(frozen.active_params))
        wandb_run.summary["mu_feature_mean"] = float(frozen.mu_features[0])
        wandb_run.summary["mu_left_mean"] = float(frozen.mu_features[1])
        wandb_run.summary["mu_right_mean"] = float(frozen.mu_features[2])
        wandb_run.summary["checkpoint_path"] = str(checkpoint_path.resolve())
        if best_checkpoint_path.exists():
            wandb_run.summary["best_checkpoint_path"] = str(best_checkpoint_path.resolve())
        if metrics_path.exists():
            wandb_run.summary["metrics_path"] = str(metrics_path.resolve())
        wandb_run.finish()


if __name__ == "__main__":
    main()
