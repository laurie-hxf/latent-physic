from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from evaluate_experiments import SCHEMA_VERSION, checkpoint_role_from_path, object_array, slugify
from trajectory_metrics import (
    METRIC_VERSION,
    evaluate_state_rollouts,
    evaluation_fingerprint,
    protocol_fingerprint,
)

from single_long_trajectory_friction_experiment.evaluate_contact_field_checkpoints import (
    DEFAULT_ROTATION68,
    DEFAULT_VERY_LONG20,
    DatasetSpec,
    evaluate_checkpoint_on_dataset,
    load_checkpoint_payload,
    summarize_field,
)


ROOT = Path(__file__).resolve().parent.parent


def parse_optional_max_steps(value: str | None) -> int | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"none", "all", "full", "-1"}:
        return None
    parsed = int(normalized)
    if parsed < 1:
        raise argparse.ArgumentTypeError("--max-steps must be positive or full")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--eval-name", choices=("rotation68", "very_long20"), required=True)
    parser.add_argument("--max-steps", type=parse_optional_max_steps, default=300)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--source", choices=("best", "current"), default="best")
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def save_contact_field_eval(
    *,
    checkpoint: Path,
    payload: dict[str, Any],
    dataset: DatasetSpec,
    result: dict[str, Any],
) -> Path:
    target_states = result.pop("_target_state_rollouts")
    predicted_states = result.pop("_predicted_state_rollouts")
    eval_args = result.pop("_eval_args")
    trajectory_indices = list(range(len(target_states)))
    unified_metrics = evaluate_state_rollouts(
        targets=target_states,
        predictions=predicted_states,
        dataset_label=dataset.name,
        trajectory_indices=trajectory_indices,
    )
    checkpoint_role = checkpoint_role_from_path(str(checkpoint))
    protocol = protocol_fingerprint(
        dataset=dataset.path,
        dataset_label=dataset.name,
        selected_trajectories=trajectory_indices,
        max_steps=dataset.max_steps,
        contact_stiffness=float(eval_args.contact_stiffness),
        contact_damping=float(eval_args.contact_damping),
        surface_point_spacing=float(eval_args.surface_point_spacing),
        friction_contact_threshold=float(eval_args.friction_contact_threshold),
        contact_mask_threshold=float(eval_args.contact_mask_threshold),
        residual_gain=None,
        residual_output_mode=None,
        stateful_reset_interval=None,
    )
    eval_fingerprint = evaluation_fingerprint(protocol["id"], str(checkpoint), checkpoint_role)
    method_name = checkpoint.stem
    eval_dir = checkpoint.parent / "eval" / dataset.name
    summary_path = eval_dir / f"{slugify(method_name)}_eval_summary.json"
    rollout_json_path = eval_dir / f"{slugify(method_name)}_trajectory_rollouts.json"
    rollout_npz_path = eval_dir / f"{slugify(method_name)}_trajectory_rollouts.npz"
    method = {
        "name": method_name,
        "checkpoint": str(checkpoint),
        "checkpoint_type": "contact_field",
        "parameterization": payload.get("args", {}).get("parameterization"),
        "iteration": payload.get("iteration"),
        "train_best_loss": payload.get("best_loss"),
        "field": summarize_field(payload),
        "learn_contact_field_names": payload.get("learn_contact_field_names"),
        "overlay_loss_mean": result.get("mean_loss"),
        "overlay_loss_std": result.get("loss_std"),
        "overlay_loss_min": result.get("loss_min"),
        "overlay_loss_max": result.get("loss_max"),
    }
    targets = [
        {
            "trajectory_index": idx,
            "episode_index": idx,
            "point": [0.0, 0.0, 0.0],
            "force": [0.0, 0.0, 0.0],
            "xy": np.asarray(state["positions"], dtype=np.float32)[:, :2].tolist(),
        }
        for idx, state in enumerate(target_states)
    ]
    interactive_method = {
        "name": method_name,
        "label": method_name,
        "color": "rgb(31, 119, 180)",
        "stage": "Contact field",
        "checkpoint": str(checkpoint),
        "losses": [float(row["loss"]) for row in result["trajectories"]],
        "tracks": [
            np.asarray(state["positions"], dtype=np.float32)[:, :2].tolist()
            for state in predicted_states
        ],
    }
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "metric_version": METRIC_VERSION,
        "run_metadata": {
            "created_at_utc": utc_now_iso(),
            "script": str(Path(__file__).resolve()),
        },
        "dataset": str(dataset.path),
        "dataset_label": dataset.name,
        "eval_name": dataset.name,
        "max_steps": dataset.max_steps,
        "selected_trajectories": trajectory_indices,
        "trajectory_count": len(trajectory_indices),
        "eval_batch_size": dataset.eval_batch_size,
        "contact_stiffness": float(eval_args.contact_stiffness),
        "contact_damping": float(eval_args.contact_damping),
        "surface_point_spacing": float(eval_args.surface_point_spacing),
        "loss_weights": {
            "position": float(eval_args.position_loss_weight),
            "orientation": float(eval_args.orientation_loss_weight),
            "linear_velocity": float(eval_args.linear_velocity_loss_weight),
            "angular_velocity": float(eval_args.angular_velocity_loss_weight),
        },
        "point_position_loss_reduction": str(eval_args.point_position_loss_reduction),
        "pointnet_residual_gain": None,
        "pointnet_residual_output_mode": None,
        "stateful_reset_interval": None,
        "evaluation_protocol": {
            "name": dataset.name,
            "dataset_label": dataset.name,
            "max_steps": dataset.max_steps,
            "loss_weights": {
                "position": float(eval_args.position_loss_weight),
                "orientation": float(eval_args.orientation_loss_weight),
                "linear_velocity": float(eval_args.linear_velocity_loss_weight),
                "angular_velocity": float(eval_args.angular_velocity_loss_weight),
            },
            "residual_gain": None,
            "residual_output_mode": None,
            "stateful_reset_interval": None,
            "protocol_fingerprint": protocol,
        },
        "protocol_fingerprint": protocol,
        "evaluation_fingerprint": eval_fingerprint,
        "checkpoint_role": checkpoint_role,
        "method": method,
        "unified_metrics": unified_metrics,
        "legacy_overlay_loss": {
            "mean": result.get("mean_loss"),
            "std": result.get("loss_std"),
            "min": result.get("loss_min"),
            "max": result.get("loss_max"),
        },
        "state_nte_mean": unified_metrics["aggregate"]["metrics"]["state_nte"]["mean"],
        "pose_nte_mean": unified_metrics["aggregate"]["metrics"]["pose_nte"]["mean"],
        "finite_rollout_rate": unified_metrics["aggregate"]["finite_rollout_rate"],
        "complete_rollout_rate": unified_metrics["aggregate"]["complete_rollout_rate"],
        "overlay_loss_mean": result.get("mean_loss"),
        "overlay_loss_std": result.get("loss_std"),
        "overlay_loss_min": result.get("loss_min"),
        "overlay_loss_max": result.get("loss_max"),
        "artifacts": {
            "experiment_dir": str(checkpoint.parent),
            "eval_dir": str(eval_dir),
            "summary_json": str(summary_path),
            "trajectory_rollouts_json": str(rollout_json_path),
            "trajectory_rollouts_npz": str(rollout_npz_path),
            "html_output_dir": str(ROOT / "report_assets" / "eval_html"),
        },
    }
    rollout_payload = dict(summary)
    rollout_payload.pop("trajectory_count", None)
    rollout_payload.update(
        interactive_data={
            "targets": targets,
            "references": [],
            "methods": [interactive_method],
        }
    )
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    rollout_json_path.write_text(json.dumps(rollout_payload, indent=2), encoding="utf-8")
    np.savez_compressed(
        rollout_npz_path,
        schema_version=np.asarray(SCHEMA_VERSION),
        metric_version=np.asarray(METRIC_VERSION),
        dataset=np.asarray(str(dataset.path)),
        eval_name=np.asarray(dataset.name),
        method_name=np.asarray(method_name),
        checkpoint=np.asarray(str(checkpoint)),
        protocol_fingerprint=np.asarray(protocol["id"]),
        evaluation_fingerprint=np.asarray(eval_fingerprint),
        trajectory_indices=np.asarray(trajectory_indices, dtype=np.int32),
        timestamps=object_array([np.asarray(state["timestamps"], dtype=np.float32) for state in target_states]),
        target_positions=object_array([np.asarray(state["positions"], dtype=np.float32) for state in target_states]),
        target_quaternions_xyzw=object_array(
            [np.asarray(state["quaternions_xyzw"], dtype=np.float32) for state in target_states]
        ),
        target_linear_velocity=object_array(
            [np.asarray(state["linear_velocity"], dtype=np.float32) for state in target_states]
        ),
        target_angular_velocity=object_array(
            [np.asarray(state["angular_velocity"], dtype=np.float32) for state in target_states]
        ),
        predicted_positions=object_array(
            [np.asarray(state["positions"], dtype=np.float32) for state in predicted_states]
        ),
        predicted_quaternions_xyzw=object_array(
            [np.asarray(state["quaternions_xyzw"], dtype=np.float32) for state in predicted_states]
        ),
        predicted_linear_velocity=object_array(
            [np.asarray(state["linear_velocity"], dtype=np.float32) for state in predicted_states]
        ),
        predicted_angular_velocity=object_array(
            [np.asarray(state["angular_velocity"], dtype=np.float32) for state in predicted_states]
        ),
        valid_mask=object_array(
            [
                np.isfinite(np.asarray(state["positions"], dtype=np.float32)).all(axis=1)
                & np.isfinite(np.asarray(state["quaternions_xyzw"], dtype=np.float32)).all(axis=1)
                & np.isfinite(np.asarray(state["linear_velocity"], dtype=np.float32)).all(axis=1)
                & np.isfinite(np.asarray(state["angular_velocity"], dtype=np.float32)).all(axis=1)
                for state in predicted_states
            ]
        ),
        target_xy=object_array(
            [np.asarray(state["positions"], dtype=np.float32)[:, :2] for state in target_states]
        ),
        predicted_xy=object_array(
            [np.asarray(state["positions"], dtype=np.float32)[:, :2] for state in predicted_states]
        ),
        losses=np.asarray(interactive_method["losses"], dtype=np.float32),
    )
    return summary_path


def main() -> None:
    args = parse_args()
    wp.init()
    checkpoint = args.checkpoint.resolve()
    dataset = DatasetSpec(
        name=args.eval_name,
        path=args.dataset.resolve(),
        max_steps=args.max_steps,
        eval_batch_size=int(args.eval_batch_size),
    )
    payload = load_checkpoint_payload(checkpoint, args.source)
    result = evaluate_checkpoint_on_dataset(
        payload=payload,
        dataset=dataset,
        device=args.device,
        max_trajectories=None,
        progress_every=0,
        collect_state_rollouts=True,
    )
    summary_path = save_contact_field_eval(
        checkpoint=checkpoint,
        payload=payload,
        dataset=dataset,
        result=result,
    )
    print(json.dumps({"summary_json": str(summary_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
