from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

import sys

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


DATASET = ROOT / "mujoco" / "outputs" / "block_force_dataset_fixed_init_20" / "block_force_dataset_fixed_init_20.npz"
OUTPUT_JSON = ROOT / "report_assets" / "fixed20_checkpoint_eval_results.json"
OUTPUTS_ROOT = ROOT / "outputs"


def experiment_checkpoint(experiment_name: str, checkpoint_stem: str | None = None) -> Path:
    stem = experiment_name if checkpoint_stem is None else checkpoint_stem
    return OUTPUTS_ROOT / experiment_name / f"{stem}.npz"


def resolve_run_path(run: dict) -> Path:
    if "path" in run:
        return Path(run["path"])
    return experiment_checkpoint(
        str(run["experiment_name"]),
        run.get("checkpoint_stem"),
    )


RUNS = [
    {
        "stage": "Pure pointwise",
        "name": "point_random_init0.30",
        "experiment_name": "friction_fit_random_init_sparse_9",
        "checkpoint_stem": "friction_fit_random_init_sparse_ckpt9",
        "notes": "legacy pointwise, random/sparse initialization",
    },
    {
        "stage": "Pure pointwise",
        "name": "point_random_init0.10",
        "experiment_name": "friction_fit_random_init_sparse_10",
        "checkpoint_stem": "friction_fit_random_init_sparse_ckpt10",
        "notes": "legacy pointwise, random/sparse initialization",
    },
    {
        "stage": "Pure pointwise",
        "name": "point_debug_fisher_init0.30",
        "path": ROOT / "debug-stash" / "outputs" / "friction_fit_random_init_sparse_ckpt_fisher_1.npz",
        "notes": "legacy pointwise, debug-stash Fisher run",
    },
    {
        "stage": "Pure pointwise",
        "name": "point_debug_fisher_init0.10",
        "path": ROOT / "debug-stash" / "outputs" / "friction_fit_random_init_sparse_ckpt_fisher_2.npz",
        "notes": "legacy pointwise, debug-stash Fisher run",
    },
    {
        "stage": "Pure pointwise",
        "name": "point_debug_random_stiff1e4",
        "path": ROOT / "debug-stash" / "outputs" / "friction_fit_random_init_sparse_ckpt_stiffness_1e4_1.npz",
        "notes": "legacy pointwise, trained with contact_stiffness=1e4",
    },
    {
        "stage": "Pure pointwise",
        "name": "point_debug_random_fisher_stiff1e4",
        "path": ROOT / "debug-stash" / "outputs" / "friction_fit_random_init_sparse_ckpt_fisher_stiffness_1e4_2.npz",
        "notes": "legacy pointwise, Fisher run trained with contact_stiffness=1e4",
    },
    {
        "stage": "Pure pointwise",
        "name": "point_debug_fixed_stiff1e4_init0.35",
        "path": ROOT / "debug-stash" / "outputs" / "friction_fit_fixed_init_stiffness_1e4_3.5.npz",
        "notes": "legacy pointwise, fixed-init run trained with contact_stiffness=1e4",
    },
    {
        "stage": "Pure pointwise",
        "name": "point_debug_fixed_fisher_stiff1e4_init0.35",
        "path": ROOT / "debug-stash" / "outputs" / "friction_fit_fixed_init_fisher_stiffness_1e4_3.5.npz",
        "notes": "legacy pointwise, fixed-init Fisher run trained with contact_stiffness=1e4",
    },
    {
        "stage": "Pure pointwise",
        "name": "point_debug_fixed_stiff1e4_init0.40",
        "path": ROOT / "debug-stash" / "outputs" / "friction_fit_fixed_init_stiffness_1e4_4.npz",
        "notes": "legacy pointwise, fixed-init run trained with contact_stiffness=1e4",
    },
    {
        "stage": "Pure pointwise",
        "name": "point_debug_fixed_fisher_stiff1e4_init0.40",
        "path": ROOT / "debug-stash" / "outputs" / "friction_fit_fixed_init_fisher_stiffness_1e4_4.npz",
        "notes": "legacy pointwise, fixed-init Fisher run trained with contact_stiffness=1e4",
    },
    {
        "stage": "Pointwise + piecewise reg v2",
        "name": "pointreg_v2_init0.30",
        "experiment_name": "fixed_init_0.3_stiffness_1e5_regularization_3000_v2",
        "notes": "corrected v2 active set, reg=3000",
    },
    {
        "stage": "Pointwise + piecewise reg v2",
        "name": "pointreg_v2_init0.35",
        "experiment_name": "fixed_init_0.35_stiffness_1e5_regularization_3000_v2",
        "notes": "corrected v2 active set, reg=3000",
    },
    {
        "stage": "Pointwise + piecewise reg v2",
        "name": "pointreg_v2_init0.40",
        "experiment_name": "fixed_init_0.4_stiffness_1e5_regularization_3000_v2",
        "notes": "corrected v2 active set, reg=3000",
    },
    {
        "stage": "Oracle left-right",
        "name": "leftright_init0.30",
        "experiment_name": "fixed_init_0.3_stiffness_1e5_regularization_0_left_right",
        "notes": "two-parameter x-split oracle baseline",
    },
    {
        "stage": "Oracle left-right",
        "name": "leftright_init0.35",
        "experiment_name": "fixed_init_0.35_stiffness_1e5_regularization_3000_left_right",
        "notes": "two-parameter x-split oracle baseline",
    },
    {
        "stage": "Oracle left-right",
        "name": "leftright_init0.40",
        "experiment_name": "fixed_init_0.4_stiffness_1e5_regularization_3000_left_right",
        "notes": "two-parameter x-split oracle baseline",
    },
    {
        "stage": "Global",
        "name": "global_init0.30",
        "experiment_name": "fixed_init_0.3_stiffness_1e5_regularization_3000_global",
        "notes": "single global friction baseline",
    },
    {
        "stage": "Global",
        "name": "global_init0.35",
        "experiment_name": "fixed_init_0.35_stiffness_1e5_regularization_3000_global",
        "notes": "single global friction baseline",
    },
    {
        "stage": "Global",
        "name": "global_init0.40",
        "experiment_name": "fixed_init_0.4_stiffness_1e5_regularization_3000_global",
        "notes": "single global friction baseline",
    },
    {
        "stage": "Global",
        "name": "global_init0.50",
        "experiment_name": "fixed_init_0.5_stiffness_1e5_regularization_3000_global",
        "notes": "single global friction baseline",
    },
]


def make_eval_args() -> argparse.Namespace:
    return argparse.Namespace(
        trajectory_npz=DATASET,
        max_steps=300,
        max_trajectories=None,
        batch_size=20,
        eval_batch_size=20,
        trajectory_progress_every=0,
        device="cuda:0",
        steps=0,
        dt=0.0,
        batch_capacity=20,
        solver_iterations=10,
        box_mass=1.0,
        floor_half_extents=(2.0, 2.0, 0.05),
        box_half_extents=(0.1, 0.05, 0.025),
        box_start_pos=(0.58, 0.0, 0.025),
        surface_point_spacing=0.01,
        friction_contact_threshold=0.002,
        contact_mask_threshold=0.002,
        point_friction=0.1,
        contact_friction=0.0,
        contact_stiffness=1.0e5,
        contact_damping=50.0,
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


def scalar(data: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in data.files:
        return default
    value = np.asarray(data[key])
    return value.item() if value.shape == () else value.tolist()


def load_run_checkpoint(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        active_indices = np.asarray(data["active_indices"], dtype=np.int32)
        active_params = np.asarray(data["best_active_params"], dtype=np.float32)
        optimizer_params = (
            np.asarray(data["best_optimizer_params"], dtype=np.float32)
            if "best_optimizer_params" in data.files
            else active_params
        )
        parameterization = scalar(data, "friction_parameterization", "point")
        return {
            "active_indices": active_indices,
            "active_params": active_params,
            "optimizer_params": optimizer_params,
            "parameterization": str(parameterization),
            "iteration": int(scalar(data, "iteration")),
            "train_best_loss": float(scalar(data, "best_loss")),
            "train_trajectory_npz": str(scalar(data, "trajectory_npz_path", "")),
            "train_max_steps": int(scalar(data, "max_steps", -1)),
        }


def main() -> None:
    wp.init()
    args = make_eval_args()
    collection = load_mujoco_trajectories(args.trajectory_npz, args.max_steps, args.max_trajectories)
    trajectories = collection.trajectories
    args.steps = collection.max_steps
    args.dt = trajectories[0].timestep
    args.eval_batch_size = resolve_batch_size(args.eval_batch_size, len(trajectories), args.batch_size)
    args.batch_capacity = max(args.eval_batch_size, 1)

    diff_scene = build_diff_scene(args)
    initial_body_q = diff_scene.states[0].body_q.numpy().copy()
    initial_body_qd = diff_scene.states[0].body_qd.numpy().copy()

    results = []
    for run in RUNS:
        path = resolve_run_path(run)
        if not path.exists():
            results.append({**run, "path": str(path), "error": "missing checkpoint"})
            continue
        ckpt = load_run_checkpoint(path)
        active_indices = ckpt["active_indices"]
        active_params = ckpt["active_params"]
        if len(active_indices) != len(active_params):
            results.append({**run, "path": str(path), "error": "active_indices/params length mismatch"})
            continue

        print(f"evaluating {run['name']} active={len(active_indices)}", flush=True)
        loss, pos, ori, lin, ang, _ = evaluate_collection_loss_in_batches(
            diff_scene=diff_scene,
            trajectories=trajectories,
            args=args,
            active_indices=active_indices,
            active_params=active_params,
            initial_body_q=initial_body_q,
            initial_body_qd=initial_body_qd,
            eval_batch_size=args.eval_batch_size,
            trajectory_progress_every=0,
            scatter_active_point_friction_kernel=scatter_active_point_friction_kernel,
            compute_batched_contact_weighted_masses_kernel=compute_batched_contact_weighted_masses_kernel,
            apply_batched_external_and_surface_point_forces_trajectory_kernel=apply_batched_external_and_surface_point_forces_trajectory_kernel,
            accumulate_batched_frame_loss_kernel=accumulate_batched_frame_loss_kernel,
            combine_batched_loss_components_kernel=combine_batched_loss_components_kernel,
            sum_batched_losses_kernel=sum_batched_losses_kernel,
        )
        side_ids = compute_piecewise_side_ids(diff_scene.local_surface_points_np, active_indices)
        _, side_means, _, side_counts, side_vars = compute_piecewise_regularization_inputs_np(active_params, side_ids)
        result = {
            **run,
            "path": str(path),
            "parameterization": ckpt["parameterization"],
            "active_points": int(len(active_indices)),
            "iteration": ckpt["iteration"],
            "train_best_loss": ckpt["train_best_loss"],
            "train_max_steps": ckpt["train_max_steps"],
            "eval_loss": float(loss),
            "eval_position_loss": float(pos),
            "eval_orientation_loss": float(ori),
            "eval_linear_velocity_loss": float(lin),
            "eval_angular_velocity_loss": float(ang),
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
        }
        if ckpt["parameterization"] == "global":
            result["mu_global_param"] = float(ckpt["optimizer_params"][0])
        if ckpt["parameterization"] == "left-right":
            result["mu_left_param"] = float(ckpt["optimizer_params"][0])
            result["mu_right_param"] = float(ckpt["optimizer_params"][1])
        results.append(result)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": str(DATASET),
        "evaluation_settings": {
            "max_steps": args.max_steps,
            "eval_trajectories": len(trajectories),
            "contact_stiffness": args.contact_stiffness,
            "surface_point_spacing": args.surface_point_spacing,
            "position_loss_weight": args.position_loss_weight,
            "point_position_loss_reduction": args.point_position_loss_reduction,
        },
        "results": results,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUTPUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
