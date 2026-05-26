from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

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
    compute_piecewise_regularization_inputs_np,
    compute_piecewise_side_ids,
    scatter_active_point_friction_kernel,
    sum_batched_losses_kernel,
)
from fit_mujoco_contact_point_friction_runtime import (  # noqa: E402
    evaluate_collection_loss_in_batches,
    resolve_batch_size,
)
from mujoco_contact_friction_fit_utils import load_mujoco_trajectories  # noqa: E402
from newton_surface_points_diff_demo import build_diff_scene  # noqa: E402
from replay_mujoco_contact_friction_trajectory import (  # noqa: E402
    build_reference_to_scene_index,
    infer_base_point_friction,
    infer_box_half_extents_and_spacing,
    load_contact_friction_point_cloud,
)


DEFAULT_OUTPUTS_ROOT = ROOT / "outputs"
DEFAULT_EVAL_ROOT = ROOT / "eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name. Defaults to --experiment-dir name when --experiment-dir is supplied.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="Experiment result directory. Defaults to <outputs-root>/<experiment-name>.",
    )
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument(
        "--eval-dataset",
        "--trajectory-npz",
        dest="eval_dataset",
        type=Path,
        required=True,
        help="Evaluation dataset NPZ. All loaded trajectories are evaluated.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Defaults to checkpoint max_steps metadata.")
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=20)
    parser.add_argument("--trajectory-progress-every", type=int, default=20)
    parser.add_argument("--checkpoint-param-set", choices=("best", "current"), default="best")
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument(
        "--replay-limit",
        type=int,
        default=None,
        help="Replay at most N trajectories after full-dataset eval. Defaults to all loaded trajectories.",
    )
    parser.add_argument(
        "--replay-indices",
        type=int,
        nargs="*",
        default=None,
        help="Specific 0-based trajectory indices to replay. Defaults to all loaded trajectories.",
    )
    parser.add_argument("--continue-on-replay-error", action="store_true")
    parser.add_argument("--position-loss-weight", type=float, default=1.0)
    parser.add_argument("--orientation-loss-weight", type=float, default=0.0)
    parser.add_argument("--linear-velocity-loss-weight", type=float, default=0.0)
    parser.add_argument("--angular-velocity-loss-weight", type=float, default=0.0)
    parser.add_argument("--point-position-loss-reduction", choices=("sum", "mean"), default="mean")
    parser.add_argument("--solver-iterations", type=int, default=10)
    parser.add_argument("--box-mass", type=float, default=1.0)
    parser.add_argument("--floor-half-extents", type=float, nargs=3, default=(2.0, 2.0, 0.05))
    parser.add_argument("--box-half-extents", type=float, nargs=3, default=(0.1, 0.05, 0.025))
    parser.add_argument("--box-start-pos", type=float, nargs=3, default=(0.58, 0.0, 0.025))
    parser.add_argument("--surface-point-spacing", type=float, default=0.01)
    parser.add_argument("--friction-contact-threshold", type=float, default=0.002)
    parser.add_argument("--contact-mask-threshold", type=float, default=0.002)
    parser.add_argument("--point-friction", type=float, default=0.1)
    parser.add_argument("--contact-friction", type=float, default=0.0)
    parser.add_argument("--contact-stiffness", type=float, default=1.0e5)
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
    args = parser.parse_args()
    if args.experiment_dir is None and not args.experiment_name:
        parser.error("one of --experiment-name or --experiment-dir is required")
    if args.experiment_dir is None:
        args.experiment_dir = args.outputs_root / str(args.experiment_name)
    if not args.experiment_name:
        args.experiment_name = args.experiment_dir.name
    return args


def scalar(data: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in data.files:
        return default
    value = np.asarray(data[key])
    return value.item() if value.shape == () else value.tolist()


def optional_nonnegative_int(data: np.lib.npyio.NpzFile, key: str) -> int | None:
    value = scalar(data, key, None)
    if value is None:
        return None
    value = int(value)
    return None if value < 0 else value


def load_checkpoint(checkpoint_path: Path, param_set: str) -> dict[str, object]:
    with np.load(checkpoint_path, allow_pickle=True) as data:
        active_indices = np.asarray(data["active_indices"], dtype=np.int32)
        if param_set == "best":
            active_params = np.asarray(data["best_active_params"], dtype=np.float32)
            optimizer_params = (
                np.asarray(data["best_optimizer_params"], dtype=np.float32)
                if "best_optimizer_params" in data.files
                else active_params
            )
        else:
            active_params = np.asarray(data["active_params"], dtype=np.float32)
            optimizer_params = (
                np.asarray(data["optimizer_params"], dtype=np.float32)
                if "optimizer_params" in data.files
                else active_params
            )
        return {
            "active_indices": active_indices,
            "active_params": active_params,
            "optimizer_params": optimizer_params,
            "parameterization": str(scalar(data, "friction_parameterization", "point")),
            "left_right_delta_sum_zero": bool(scalar(data, "left_right_delta_sum_zero", False)),
            "iteration": int(scalar(data, "iteration", -1)),
            "best_loss": float(scalar(data, "best_loss", float("nan"))),
            "train_trajectory_npz": str(scalar(data, "trajectory_npz_path", "")),
            "train_max_steps": optional_nonnegative_int(data, "max_steps"),
        }


def resolve_experiment_paths(args: argparse.Namespace) -> dict[str, Path]:
    experiment_dir = Path(args.experiment_dir)
    experiment_name = str(args.experiment_name)
    return {
        "experiment_dir": experiment_dir,
        "checkpoint": experiment_dir / f"{experiment_name}.npz",
        "point_cloud": experiment_dir / f"{experiment_name}.ply",
        "checkpoint_point_cloud_dir": experiment_dir / f"{experiment_name}_point_clouds",
        "eval_dir": Path(args.output_root) / experiment_name,
    }


def apply_reference_point_cloud_settings(args: argparse.Namespace, point_cloud_path: Path) -> object | None:
    if not point_cloud_path.exists():
        return None
    reference_point_cloud = load_contact_friction_point_cloud(point_cloud_path)
    inferred_half_extents, inferred_spacing = infer_box_half_extents_and_spacing(
        reference_point_cloud.local_surface_points
    )
    args.box_half_extents = inferred_half_extents.tolist()
    args.surface_point_spacing = inferred_spacing
    args.point_friction = infer_base_point_friction(reference_point_cloud, fallback=float(args.point_friction))
    return reference_point_cloud


def select_replay_indices(args: argparse.Namespace, trajectory_count: int) -> list[int]:
    if args.skip_replay:
        return []
    if args.replay_indices is None:
        indices = list(range(trajectory_count))
    else:
        indices = [int(item) for item in args.replay_indices]
    for index in indices:
        if index < 0 or index >= trajectory_count:
            raise IndexError(f"Replay trajectory index {index} out of range for {trajectory_count} trajectories")
    if args.replay_limit is not None:
        if int(args.replay_limit) < 0:
            raise ValueError("--replay-limit must be non-negative")
        indices = indices[: int(args.replay_limit)]
    return indices


def run_replays(
    *,
    args: argparse.Namespace,
    paths: dict[str, Path],
    replay_indices: list[int],
    max_steps: int | None,
) -> list[dict[str, object]]:
    replay_results: list[dict[str, object]] = []
    replay_script = NEWTON_DIR / "replay_mujoco_contact_friction_trajectory.py"
    replay_root = paths["eval_dir"] / "replay"
    for trajectory_index in replay_indices:
        trajectory_dir = replay_root / f"trajectory_{trajectory_index:04d}"
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        scene_usd_path = trajectory_dir / f"{args.experiment_name}_traj_{trajectory_index:04d}.usda"
        summary_npz_path = trajectory_dir / f"{args.experiment_name}_traj_{trajectory_index:04d}.npz"
        log_path = trajectory_dir / "replay.log"

        cmd = [
            sys.executable,
            str(replay_script),
            "--checkpoint-path",
            str(paths["checkpoint"]),
            "--trajectory-index",
            str(trajectory_index),
            "--trajectory-npz",
            str(args.eval_dataset),
            "--checkpoint-param-set",
            str(args.checkpoint_param_set),
            "--scene-usd-path",
            str(scene_usd_path),
            "--summary-npz-path",
            str(summary_npz_path),
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
            "--friction-regularization",
            str(args.friction_regularization),
        ]
        if args.device is not None:
            cmd.extend(["--device", str(args.device)])
        if max_steps is not None:
            cmd.extend(["--max-steps", str(max_steps)])
        if paths["point_cloud"].exists():
            cmd.extend(["--reference-point-cloud", str(paths["point_cloud"])])
        if paths["checkpoint_point_cloud_dir"].exists():
            cmd.extend(["--checkpoint-point-cloud-dir", str(paths["checkpoint_point_cloud_dir"])])

        completed = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_path.write_text(completed.stdout, encoding="utf-8")
        result = {
            "trajectory_index": int(trajectory_index),
            "returncode": int(completed.returncode),
            "scene_usd_path": str(scene_usd_path.resolve()),
            "summary_npz_path": str(summary_npz_path.resolve()),
            "log_path": str(log_path.resolve()),
        }
        replay_results.append(result)
        if completed.returncode != 0 and not args.continue_on_replay_error:
            raise RuntimeError(
                f"Replay failed for trajectory {trajectory_index}; see {log_path.resolve()}"
            )
    return replay_results


def main() -> None:
    args = parse_args()
    paths = resolve_experiment_paths(args)
    checkpoint_path = paths["checkpoint"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    wp.init()
    checkpoint = load_checkpoint(checkpoint_path, str(args.checkpoint_param_set))
    max_steps = int(args.max_steps) if args.max_steps is not None else checkpoint["train_max_steps"]
    reference_point_cloud = apply_reference_point_cloud_settings(args, paths["point_cloud"])

    collection = load_mujoco_trajectories(args.eval_dataset, max_steps, args.max_trajectories)
    trajectories = collection.trajectories
    if not trajectories:
        raise ValueError(f"No trajectories loaded from {args.eval_dataset}")
    args.steps = collection.max_steps
    args.dt = trajectories[0].timestep
    args.eval_batch_size = resolve_batch_size(args.eval_batch_size, len(trajectories), args.eval_batch_size)
    args.batch_capacity = max(args.eval_batch_size, 1)

    print(
        f"[eval] experiment={args.experiment_name} trajectories={len(trajectories)} "
        f"max_steps={max_steps if max_steps is not None else 'all'}",
        flush=True,
    )
    diff_scene = build_diff_scene(args)
    initial_body_q = diff_scene.states[0].body_q.numpy().copy()
    initial_body_qd = diff_scene.states[0].body_qd.numpy().copy()

    active_indices = np.asarray(checkpoint["active_indices"], dtype=np.int32).copy()
    active_params = np.asarray(checkpoint["active_params"], dtype=np.float32)
    if reference_point_cloud is not None:
        reference_to_scene = build_reference_to_scene_index(
            reference_point_cloud.local_surface_points,
            diff_scene.local_surface_points_np,
        )
        active_indices = reference_to_scene[active_indices]

    loss, pos_loss, ori_loss, lin_loss, ang_loss, _ = evaluate_collection_loss_in_batches(
        diff_scene=diff_scene,
        trajectories=trajectories,
        args=args,
        active_indices=active_indices,
        active_params=active_params,
        initial_body_q=initial_body_q,
        initial_body_qd=initial_body_qd,
        eval_batch_size=args.eval_batch_size,
        trajectory_progress_every=int(args.trajectory_progress_every),
        scatter_active_point_friction_kernel=scatter_active_point_friction_kernel,
        compute_batched_contact_weighted_masses_kernel=compute_batched_contact_weighted_masses_kernel,
        apply_batched_external_and_surface_point_forces_trajectory_kernel=(
            apply_batched_external_and_surface_point_forces_trajectory_kernel
        ),
        accumulate_batched_frame_loss_kernel=accumulate_batched_frame_loss_kernel,
        combine_batched_loss_components_kernel=combine_batched_loss_components_kernel,
        sum_batched_losses_kernel=sum_batched_losses_kernel,
    )

    side_ids = compute_piecewise_side_ids(diff_scene.local_surface_points_np, active_indices)
    _, side_means, _, side_counts, side_vars = compute_piecewise_regularization_inputs_np(
        active_params,
        side_ids,
    )

    paths["eval_dir"].mkdir(parents=True, exist_ok=True)
    replay_indices = select_replay_indices(args, len(trajectories))
    replay_results = run_replays(
        args=args,
        paths=paths,
        replay_indices=replay_indices,
        max_steps=max_steps,
    )

    optimizer_params = np.asarray(checkpoint["optimizer_params"], dtype=np.float32)
    metrics = {
        "experiment_name": str(args.experiment_name),
        "experiment_dir": str(paths["experiment_dir"].resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "point_cloud_path": str(paths["point_cloud"].resolve()) if paths["point_cloud"].exists() else None,
        "eval_dataset": str(args.eval_dataset.resolve()),
        "eval_output_dir": str(paths["eval_dir"].resolve()),
        "trajectory_count": int(len(trajectories)),
        "max_steps": None if max_steps is None else int(max_steps),
        "eval_batch_size": int(args.eval_batch_size),
        "checkpoint_iteration": int(checkpoint["iteration"]),
        "checkpoint_param_set": str(args.checkpoint_param_set),
        "friction_parameterization": str(checkpoint["parameterization"]),
        "train_best_loss": float(checkpoint["best_loss"]),
        "eval_loss": float(loss),
        "eval_position_loss": float(pos_loss),
        "eval_orientation_loss": float(ori_loss),
        "eval_linear_velocity_loss": float(lin_loss),
        "eval_angular_velocity_loss": float(ang_loss),
        "active_points": int(len(active_indices)),
        "surface_points": int(len(diff_scene.local_surface_points_np)),
        "mu_mean": float(np.mean(active_params)),
        "mu_std": float(np.std(active_params)),
        "mu_min": float(np.min(active_params)),
        "mu_max": float(np.max(active_params)),
        "mu_left_mean": float(side_means[0]),
        "mu_right_mean": float(side_means[1]),
        "left_count": int(side_counts[0]),
        "right_count": int(side_counts[1]),
        "left_var": float(side_vars[0]),
        "right_var": float(side_vars[1]),
        "replay_count": int(len(replay_results)),
        "replays": replay_results,
    }
    if str(checkpoint["parameterization"]) == "global" and len(optimizer_params) >= 1:
        metrics["mu_global_param"] = float(optimizer_params[0])
    if str(checkpoint["parameterization"]) == "left-right" and len(optimizer_params) >= 2:
        metrics["mu_left_param"] = float(optimizer_params[0])
        metrics["mu_right_param"] = float(optimizer_params[1])
    if str(checkpoint["parameterization"]) == "base-delta" and len(optimizer_params) >= 3:
        metrics["mu_base_param"] = float(optimizer_params[0])
        metrics["delta_left_param"] = float(optimizer_params[1])
        metrics["delta_right_param"] = float(optimizer_params[2])
        metrics["delta_sum"] = float(optimizer_params[1] + optimizer_params[2])
        metrics["mu_left_param"] = float(optimizer_params[0] + optimizer_params[1])
        metrics["mu_right_param"] = float(optimizer_params[0] + optimizer_params[2])
        metrics["left_right_delta_sum_zero"] = bool(checkpoint["left_right_delta_sum_zero"])

    summary_json_path = paths["eval_dir"] / "eval_summary.json"
    summary_npz_path = paths["eval_dir"] / "eval_summary.npz"
    summary_json_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    np.savez_compressed(
        summary_npz_path,
        eval_loss=np.asarray(loss, dtype=np.float32),
        eval_position_loss=np.asarray(pos_loss, dtype=np.float32),
        eval_orientation_loss=np.asarray(ori_loss, dtype=np.float32),
        eval_linear_velocity_loss=np.asarray(lin_loss, dtype=np.float32),
        eval_angular_velocity_loss=np.asarray(ang_loss, dtype=np.float32),
        active_contact_point_indices=active_indices,
        active_point_friction=active_params,
        eval_dataset=np.asarray(str(args.eval_dataset.resolve())),
        checkpoint_path=np.asarray(str(checkpoint_path.resolve())),
    )
    print(f"[eval] summary_json={summary_json_path.resolve()}", flush=True)
    print(f"[eval] summary_npz={summary_npz_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
