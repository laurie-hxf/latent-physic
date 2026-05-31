from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from fit_mujoco_contact_point_friction_io import save_contact_friction_point_cloud
from fit_mujoco_contact_point_friction_runtime import log_message


def save_training_checkpoint(
    *,
    checkpoint_path,
    iteration: int,
    active_indices: np.ndarray,
    active_params: np.ndarray,
    optimizer_params: np.ndarray,
    adam_m: np.ndarray,
    adam_v: np.ndarray,
    adam_step: np.ndarray,
    best_loss: float,
    best_active_params: np.ndarray,
    best_optimizer_params: np.ndarray,
    loss_history: list[float],
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        checkpoint_path,
        iteration=np.asarray(iteration, dtype=np.int32),
        active_indices=np.asarray(active_indices, dtype=np.int32),
        active_params=np.asarray(active_params, dtype=np.float32),
        optimizer_params=np.asarray(optimizer_params, dtype=np.float32),
        adam_m=np.asarray(adam_m, dtype=np.float64),
        adam_v=np.asarray(adam_v, dtype=np.float64),
        adam_step=np.asarray(adam_step, dtype=np.int32),
        best_loss=np.asarray(best_loss, dtype=np.float64),
        best_active_params=np.asarray(best_active_params, dtype=np.float32),
        best_optimizer_params=np.asarray(best_optimizer_params, dtype=np.float32),
        loss_history=np.asarray(loss_history, dtype=np.float32),
        rng_state=np.asarray(rng.bit_generator.state, dtype=object),
        friction_parameterization=np.asarray(str(args.friction_parameterization)),
        left_right_delta_sum_zero=np.asarray(bool(getattr(args, "left_right_delta_sum_zero", False))),
        random_time_windows=np.asarray(bool(getattr(args, "random_time_windows", False))),
        window_steps=np.asarray(-1 if getattr(args, "window_steps", None) is None else int(args.window_steps), dtype=np.int32),
        time_window_source_max_steps=np.asarray(
            -1
            if getattr(args, "time_window_source_max_steps", None) is None
            else int(args.time_window_source_max_steps),
            dtype=np.int32,
        ),
        training_rollout_steps=np.asarray(int(getattr(args, "steps", 0)), dtype=np.int32),
        trajectory_npz_path=np.asarray(str(args.trajectory_npz.resolve())),
        max_steps=np.asarray(-1 if args.max_steps is None else int(args.max_steps), dtype=np.int32),
        max_trajectories=np.asarray(-1 if args.max_trajectories is None else int(args.max_trajectories), dtype=np.int32),
        dino_feature_npz_path=np.asarray(
            "" if getattr(args, "dino_feature_npz", None) is None else str(args.dino_feature_npz)
        ),
        dino_neighbor_radius=np.asarray(float(getattr(args, "dino_neighbor_radius", 0.0)), dtype=np.float32),
        dino_neighbor_k=np.asarray(int(getattr(args, "dino_neighbor_k", 0)), dtype=np.int32),
        dino_position_frequencies=np.asarray(int(getattr(args, "dino_position_frequencies", 0)), dtype=np.int32),
        dino_mlp_hidden_dim=np.asarray(int(getattr(args, "dino_mlp_hidden_dim", 0)), dtype=np.int32),
        dino_mlp_hidden_layers=np.asarray(int(getattr(args, "dino_mlp_hidden_layers", 0)), dtype=np.int32),
        dino_mlp_max_match_distance=np.asarray(
            float(getattr(args, "dino_mlp_max_match_distance", 0.0)),
            dtype=np.float32,
        ),
        dino_feature_normalization=np.asarray(bool(getattr(args, "dino_feature_normalization", False))),
    )


def resolve_checkpoint_point_cloud_path(args: argparse.Namespace, iteration: int):
    if args.checkpoint_point_cloud_dir is None:
        point_cloud_dir = args.checkpoint_path.parent / f"{args.checkpoint_path.stem}_point_clouds"
    else:
        point_cloud_dir = args.checkpoint_point_cloud_dir
    return point_cloud_dir / f"iter_{int(iteration):06d}.ply"


def should_save_iteration_checkpoint(args: argparse.Namespace, iteration: int) -> bool:
    checkpoint_every = int(args.checkpoint_every)
    return checkpoint_every > 0 and (iteration % checkpoint_every == 0 or iteration == int(args.opt_iters))


def run_post_training_eval(args: argparse.Namespace) -> Path:
    eval_script = Path(__file__).resolve().parent.parent / "visualization" / "evaluate_mujoco_contact_friction_experiment.py"
    eval_output_dir = args.eval_output_root / args.experiment_dir.name
    cmd = [
        sys.executable,
        str(eval_script),
        "--experiment-dir",
        str(args.experiment_dir),
        "--eval-dataset",
        str(args.eval_dataset),
        "--output-root",
        str(args.eval_output_root),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--position-loss-weight",
        str(args.position_loss_weight),
        "--orientation-loss-weight",
        str(args.orientation_loss_weight),
        "--linear-velocity-loss-weight",
        str(args.linear_velocity_loss_weight),
        "--angular-velocity-loss-weight",
        str(args.angular_velocity_loss_weight),
        "--point-position-loss-reduction",
        str(args.point_position_loss_reduction),
        "--solver-iterations",
        str(args.solver_iterations),
        "--contact-stiffness",
        str(args.contact_stiffness),
        "--contact-damping",
        str(args.contact_damping),
        "--contact-margin",
        str(args.contact_margin),
        "--friction-contact-threshold",
        str(args.friction_contact_threshold),
        "--contact-mask-threshold",
        str(args.contact_mask_threshold),
        "--friction-regularization",
        str(args.friction_regularization),
    ]
    if args.device is not None:
        cmd.extend(["--device", str(args.device)])
    if args.max_steps is not None:
        cmd.extend(["--max-steps", str(args.max_steps)])
    if args.eval_replay_limit is not None:
        cmd.extend(["--replay-limit", str(args.eval_replay_limit)])
    if args.eval_skip_replay:
        cmd.append("--skip-replay")

    log_message(
        f"running post-training eval dataset={args.eval_dataset.resolve()} "
        f"output_dir={eval_output_dir.resolve()}"
    )
    subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent.parent), check=True)
    return eval_output_dir


def save_iteration_checkpoint_and_point_cloud(
    *,
    args: argparse.Namespace,
    iteration: int,
    active_indices: np.ndarray,
    active_params: np.ndarray,
    optimizer_params: np.ndarray,
    adam_m: np.ndarray,
    adam_v: np.ndarray,
    adam_step: np.ndarray,
    best_loss: float,
    best_active_params: np.ndarray,
    best_optimizer_params: np.ndarray,
    loss_history: list[float],
    rng: np.random.Generator,
    local_surface_points: np.ndarray,
    point_cloud_color_min: float,
    point_cloud_color_max: float,
) -> None:
    save_training_checkpoint(
        checkpoint_path=args.checkpoint_path,
        iteration=iteration,
        active_indices=active_indices,
        active_params=active_params,
        optimizer_params=optimizer_params,
        adam_m=adam_m,
        adam_v=adam_v,
        adam_step=adam_step,
        best_loss=best_loss,
        best_active_params=best_active_params,
        best_optimizer_params=best_optimizer_params,
        loss_history=loss_history,
        rng=rng,
        args=args,
    )
    checkpoint_point_cloud_path = resolve_checkpoint_point_cloud_path(args, iteration)
    checkpoint_point_friction = np.full(
        len(local_surface_points),
        float(args.point_friction),
        dtype=np.float32,
    )
    checkpoint_point_friction[active_indices] = active_params
    save_contact_friction_point_cloud(
        local_surface_points=local_surface_points,
        point_friction=checkpoint_point_friction,
        output_path=checkpoint_point_cloud_path,
        active_indices=active_indices,
        color_min=point_cloud_color_min,
        color_max=point_cloud_color_max,
    )
    log_message(f"checkpoint_point_cloud_written_to={checkpoint_point_cloud_path.resolve()}")


def load_training_checkpoint(
    *,
    checkpoint_path,
    active_indices: np.ndarray,
    parameterization: str,
    left_right_delta_sum_zero: bool,
    random_time_windows: bool,
    optimizer_param_shape: tuple[int, ...],
    rng: np.random.Generator,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, list[float]]:
    with np.load(checkpoint_path, allow_pickle=True) as data:
        checkpoint_active_indices = np.asarray(data["active_indices"], dtype=np.int32)
        if checkpoint_active_indices.shape != active_indices.shape or not np.array_equal(checkpoint_active_indices, active_indices):
            raise ValueError(
                f"{checkpoint_path} active point indices do not match the current run. "
                "Use matching trajectory/model/contact-mask settings or start without --resume-checkpoint."
            )

        checkpoint_parameterization = (
            str(np.asarray(data["friction_parameterization"]).item())
            if "friction_parameterization" in data.files
            else "point"
        )
        if checkpoint_parameterization != parameterization:
            raise ValueError(
                f"{checkpoint_path} was saved with friction_parameterization={checkpoint_parameterization!r}, "
                f"but the current run uses {parameterization!r}."
            )
        checkpoint_delta_sum_zero = (
            bool(np.asarray(data["left_right_delta_sum_zero"]).item())
            if "left_right_delta_sum_zero" in data.files
            else False
        )
        if checkpoint_parameterization == "base-delta" and checkpoint_delta_sum_zero != bool(left_right_delta_sum_zero):
            raise ValueError(
                f"{checkpoint_path} was saved with left_right_delta_sum_zero={checkpoint_delta_sum_zero}, "
                f"but the current run uses {bool(left_right_delta_sum_zero)}."
            )
        checkpoint_random_time_windows = (
            bool(np.asarray(data["random_time_windows"]).item())
            if "random_time_windows" in data.files
            else False
        )
        if checkpoint_random_time_windows != bool(random_time_windows):
            raise ValueError(
                f"{checkpoint_path} was saved with random_time_windows={checkpoint_random_time_windows}, "
                f"but the current run uses {bool(random_time_windows)}."
            )

        iteration = int(np.asarray(data["iteration"]).item())
        if "optimizer_params" in data.files:
            active_params = np.asarray(data["optimizer_params"], dtype=np.float32)
        else:
            active_params = np.asarray(data["active_params"], dtype=np.float32)
        adam_m = np.asarray(data["adam_m"], dtype=np.float64)
        adam_v = np.asarray(data["adam_v"], dtype=np.float64)
        if "adam_step" in data.files:
            adam_step = np.asarray(data["adam_step"], dtype=np.int32)
        else:
            adam_step = np.zeros(optimizer_param_shape, dtype=np.int32)
        best_loss = float(np.asarray(data["best_loss"]).item())
        if "best_optimizer_params" in data.files:
            best_active_params = np.asarray(data["best_optimizer_params"], dtype=np.float32)
        else:
            best_active_params = np.asarray(data["best_active_params"], dtype=np.float32)
        loss_history = [float(value) for value in np.asarray(data["loss_history"], dtype=np.float32)]

        expected_shape = optimizer_param_shape
        for name, values in (
            ("optimizer_params", active_params),
            ("adam_m", adam_m),
            ("adam_v", adam_v),
            ("adam_step", adam_step),
            ("best_optimizer_params", best_active_params),
        ):
            if values.shape != expected_shape:
                raise ValueError(f"{checkpoint_path} {name} has shape {values.shape}, expected {expected_shape}")

        rng_state = data["rng_state"].item()
        rng.bit_generator.state = rng_state

    return iteration, active_params, adam_m, adam_v, adam_step, best_loss, best_active_params, loss_history

