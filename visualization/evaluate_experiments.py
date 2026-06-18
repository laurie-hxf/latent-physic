from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from plot_topdown_trajectory_overlays_interactive import collect_rollout_payload
from sync_eval_to_notion import sync_eval_summaries_to_notion
from trajectory_metrics import (
    METRIC_VERSION,
    evaluate_state_rollouts,
    evaluation_fingerprint,
    protocol_fingerprint,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HTML_OUTPUT_DIR = ROOT / "report_assets" / "eval_html"
SCHEMA_VERSION = "experiment-eval-v3"


def parse_optional_max_steps(value: str | None) -> int | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"none", "all", "full", "-1"}:
        return None
    parsed = int(normalized)
    if parsed < 1:
        raise argparse.ArgumentTypeError("--max-steps must be positive, or one of: none, all, full, -1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--eval-name",
        type=str,
        default=None,
        help="Name for the eval subdirectory under each experiment. Defaults to an inferred dataset label.",
    )
    parser.add_argument(
        "--method-source",
        choices=("default", "curated", "auto", "all"),
        default="auto",
        help="Uses the same method discovery contract as plot_topdown_trajectory_overlays_interactive.py.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        action="append",
        default=None,
        help="Root directory to scan for checkpoints when --method-source is auto or all.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        default=None,
        help="Evaluate only these explicit checkpoint files. May be repeated.",
    )
    parser.add_argument(
        "--include-pointnet-last-checkpoints",
        action="store_true",
        help="Include PointNet *_last.pt checkpoints as separate methods when scanning.",
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
    parser.add_argument("--orientation-loss-weight", type=float, default=1.0)
    parser.add_argument("--linear-velocity-loss-weight", type=float, default=0.0)
    parser.add_argument("--angular-velocity-loss-weight", type=float, default=0.1)
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
    )
    parser.add_argument(
        "--stateful-reset-interval",
        type=int,
        default=None,
        help="Reset stateful adapter memory every N steps. None uses checkpoint metadata; 0 never resets.",
    )
    parser.add_argument("--trajectory-indices", type=int, nargs="*", default=None)
    parser.add_argument(
        "--all-trajectories",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate every trajectory in the dataset unless --trajectory-indices is provided.",
    )
    parser.add_argument("--include-pure-point", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--reference-dataset",
        type=Path,
        action="append",
        default=None,
        help="Optional reference datasets to include in the saved visualization payload.",
    )
    parser.add_argument("--reference-label", type=str, action="append", default=None)
    parser.add_argument("--reference-color", type=str, action="append", default=None)
    parser.add_argument(
        "--html-output-dir",
        type=Path,
        default=DEFAULT_HTML_OUTPUT_DIR,
        help="Shared directory where render_saved_eval_html.py writes HTML. Stored in eval summaries only.",
    )
    parser.add_argument(
        "--sync-notion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Sync raw per-checkpoint eval rows to a dedicated Notion eval ledger after local files are written. "
            "This never targets the main experiment registry database."
        ),
    )
    parser.add_argument(
        "--notion-fail-on-error",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail the eval command if Notion sync is configured incorrectly or the Notion API returns an error.",
    )
    return parser.parse_args()


def infer_dataset_label(dataset: Path) -> str:
    text = str(dataset)
    if "very_long_rotation_friction_diagnostics_l0p20_r0p50_2000" in text:
        return "very_long2000"
    if "rotation_friction_diagnostics_l0p20_r0p50_2000" in text:
        return "rotation2000"
    if "very_long_rotation_friction_diagnostics_l0p20_r0p50_20" in text:
        return "very_long20"
    if "rotation_friction_diagnostics_l0p20_r0p50_68" in text:
        return "rotation68"
    if "long_rotation_friction_diagnostics" in text:
        return "long20"
    return dataset.stem


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return slug or "eval"


def repo_rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def object_array(values: list[np.ndarray]) -> np.ndarray:
    array = np.empty(len(values), dtype=object)
    for idx, value in enumerate(values):
        array[idx] = value
    return array


def experiment_dir_from_checkpoint(checkpoint: str | Path) -> Path:
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    return checkpoint_path.resolve().parent


def checkpoint_role_from_path(checkpoint: str | Path) -> str:
    name = Path(str(checkpoint)).stem.lower()
    if name.endswith("_last"):
        return "last"
    if name.endswith("_best") or "_best_" in name:
        return "best"
    return "primary"


def method_summary_by_name(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(method.get("name")): method for method in payload.get("methods", [])}


def interactive_method_by_name(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(method.get("name")): method for method in payload.get("interactive_data", {}).get("methods", [])}


def save_method_eval(
    *,
    args: argparse.Namespace,
    payload: dict[str, Any],
    eval_name: str,
    method_summary: dict[str, Any],
    interactive_method: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = interactive_method.get("checkpoint") or method_summary.get("checkpoint")
    if not checkpoint:
        raise ValueError(f"Method {method_summary.get('name')} has no checkpoint path")

    experiment_dir = experiment_dir_from_checkpoint(str(checkpoint))
    eval_dir = experiment_dir / "eval" / eval_name
    method_slug = slugify(str(method_summary.get("name") or Path(str(checkpoint)).stem))
    summary_path = eval_dir / f"{method_slug}_eval_summary.json"
    rollout_json_path = eval_dir / f"{method_slug}_trajectory_rollouts.json"
    rollout_npz_path = eval_dir / f"{method_slug}_trajectory_rollouts.npz"

    targets = payload["interactive_data"]["targets"]
    target_xy = [np.asarray(target["xy"], dtype=np.float32) for target in targets]
    pred_xy = [np.asarray(track, dtype=np.float32) for track in interactive_method.get("tracks", [])]
    losses = np.asarray(interactive_method.get("losses", []), dtype=np.float32)
    trajectory_indices = np.asarray([int(target["trajectory_index"]) for target in targets], dtype=np.int32)
    target_states = payload["_target_state_rollouts"]
    predicted_states = payload["_method_state_rollouts"][str(method_summary.get("name"))]
    dataset_label = infer_dataset_label(Path(str(payload.get("dataset"))))
    actual_residual_gain = method_summary.get("rollout_residual_gain", payload.get("pointnet_residual_gain"))
    actual_residual_output_mode = method_summary.get("residual_output_mode") or method_summary.get("output_mode")
    actual_stateful_reset_interval = (
        (method_summary.get("rollout_stateful_diagnostics") or {}).get("stateful_reset_interval")
        if isinstance(method_summary.get("rollout_stateful_diagnostics"), dict)
        else method_summary.get("stateful_reset_interval", payload.get("stateful_reset_interval"))
    )
    if actual_stateful_reset_interval is None:
        actual_stateful_reset_interval = method_summary.get("stateful_reset_interval", payload.get("stateful_reset_interval"))
    unified_metrics = evaluate_state_rollouts(
        targets=target_states,
        predictions=predicted_states,
        dataset_label=dataset_label,
        trajectory_indices=[int(value) for value in trajectory_indices],
    )
    checkpoint_role = checkpoint_role_from_path(str(checkpoint))
    protocol = protocol_fingerprint(
        dataset=Path(str(payload.get("dataset"))),
        dataset_label=dataset_label,
        selected_trajectories=[int(value) for value in trajectory_indices],
        max_steps=payload.get("max_steps"),
        contact_stiffness=float(payload.get("contact_stiffness")),
        contact_damping=float(payload.get("contact_damping")),
        surface_point_spacing=float(payload.get("surface_point_spacing")),
        friction_contact_threshold=float(payload.get("friction_contact_threshold")),
        contact_mask_threshold=float(payload.get("contact_mask_threshold")),
        residual_gain=None if actual_residual_gain is None else float(actual_residual_gain),
        residual_output_mode=None if actual_residual_output_mode is None else str(actual_residual_output_mode),
        stateful_reset_interval=(
            None if actual_stateful_reset_interval is None else int(actual_stateful_reset_interval)
        ),
    )
    eval_fingerprint = evaluation_fingerprint(protocol["id"], str(checkpoint), checkpoint_role)

    method_payload = {
        "schema_version": SCHEMA_VERSION,
        "metric_version": METRIC_VERSION,
        "run_metadata": payload.get("run_metadata", {}),
        "dataset": payload.get("dataset"),
        "dataset_label": dataset_label,
        "eval_name": eval_name,
        "max_steps": payload.get("max_steps"),
        "selected_trajectories": payload.get("selected_trajectories"),
        "eval_batch_size": payload.get("eval_batch_size"),
        "contact_stiffness": payload.get("contact_stiffness"),
        "surface_point_spacing": payload.get("surface_point_spacing"),
        "loss_weights": payload.get("loss_weights"),
        "point_position_loss_reduction": payload.get("point_position_loss_reduction"),
        "pointnet_residual_gain": payload.get("pointnet_residual_gain"),
        "pointnet_residual_output_mode": payload.get("pointnet_residual_output_mode"),
        "stateful_reset_interval": payload.get("stateful_reset_interval"),
        "evaluation_protocol": {
            "name": eval_name,
            "dataset_label": dataset_label,
            "max_steps": payload.get("max_steps"),
            "loss_weights": payload.get("loss_weights"),
            "point_position_loss_reduction": payload.get("point_position_loss_reduction"),
            "residual_gain": actual_residual_gain,
            "residual_output_mode": actual_residual_output_mode,
            "stateful_reset_interval": actual_stateful_reset_interval,
            "protocol_fingerprint": protocol,
        },
        "protocol_fingerprint": protocol,
        "evaluation_fingerprint": eval_fingerprint,
        "checkpoint_role": checkpoint_role,
        "method": method_summary,
        "unified_metrics": unified_metrics,
        "legacy_overlay_loss": {
            "mean": method_summary.get("overlay_loss_mean"),
            "std": float(np.std(losses)) if losses.size else None,
            "min": method_summary.get("overlay_loss_min"),
            "max": method_summary.get("overlay_loss_max"),
        },
        "interactive_data": {
            "targets": targets,
            "references": payload.get("interactive_data", {}).get("references", []),
            "methods": [interactive_method],
        },
        "artifacts": {
            "experiment_dir": str(experiment_dir),
            "eval_dir": str(eval_dir),
            "summary_json": str(summary_path),
            "trajectory_rollouts_json": str(rollout_json_path),
            "trajectory_rollouts_npz": str(rollout_npz_path),
            "html_output_dir": str(args.html_output_dir),
        },
    }

    summary = dict(method_payload)
    summary.pop("interactive_data", None)
    summary["trajectory_count"] = int(len(trajectory_indices))
    summary["state_nte_mean"] = unified_metrics["aggregate"]["metrics"]["state_nte"]["mean"]
    summary["pose_nte_mean"] = unified_metrics["aggregate"]["metrics"]["pose_nte"]["mean"]
    summary["finite_rollout_rate"] = unified_metrics["aggregate"]["finite_rollout_rate"]
    summary["complete_rollout_rate"] = unified_metrics["aggregate"]["complete_rollout_rate"]
    summary["overlay_loss_mean"] = method_summary.get("overlay_loss_mean")
    summary["overlay_loss_std"] = float(np.std(losses)) if losses.size else None
    summary["overlay_loss_min"] = method_summary.get("overlay_loss_min")
    summary["overlay_loss_max"] = method_summary.get("overlay_loss_max")

    eval_dir.mkdir(parents=True, exist_ok=True)
    rollout_json_path.write_text(json.dumps(method_payload, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    np.savez_compressed(
        rollout_npz_path,
        schema_version=np.asarray(SCHEMA_VERSION),
        metric_version=np.asarray(METRIC_VERSION),
        dataset=np.asarray(str(payload.get("dataset"))),
        eval_name=np.asarray(eval_name),
        method_name=np.asarray(str(method_summary.get("name"))),
        checkpoint=np.asarray(str(checkpoint)),
        protocol_fingerprint=np.asarray(protocol["id"]),
        evaluation_fingerprint=np.asarray(eval_fingerprint),
        trajectory_indices=trajectory_indices,
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
        target_xy=object_array(target_xy),
        predicted_xy=object_array(pred_xy),
        losses=losses,
    )

    return {
        "method": method_summary.get("name"),
        "checkpoint": repo_rel(str(checkpoint)),
        "experiment_dir": repo_rel(experiment_dir),
        "eval_dir": repo_rel(eval_dir),
        "summary_json": repo_rel(summary_path),
        "trajectory_rollouts_json": repo_rel(rollout_json_path),
        "trajectory_rollouts_npz": repo_rel(rollout_npz_path),
    }


def main() -> None:
    args = parse_args()
    if args.checkpoint_root is None:
        args.checkpoint_root = [ROOT / "outputs"]
    eval_name = slugify(args.eval_name or infer_dataset_label(args.dataset))
    # collect_rollout_payload uses the same argparse surface as the interactive plotter.
    payload = collect_rollout_payload(args)
    payload["schema_version"] = SCHEMA_VERSION
    payload["eval_name"] = eval_name
    payload["point_position_loss_reduction"] = str(args.point_position_loss_reduction)
    payload["run_metadata"] = {
        "created_at_utc": utc_now_iso(),
        "cwd": str(Path.cwd()),
        "script": str(Path(__file__).resolve()),
        "argv": [sys.executable, *sys.argv],
    }
    summaries = method_summary_by_name(payload)
    interactive_methods = interactive_method_by_name(payload)

    outputs = []
    for method_name, interactive_method in interactive_methods.items():
        if method_name not in summaries:
            raise ValueError(f"Missing summary for method {method_name}")
        outputs.append(
            save_method_eval(
                args=args,
                payload=payload,
                eval_name=eval_name,
                method_summary=summaries[method_name],
                interactive_method=interactive_method,
            )
        )

    notion_sync = {"status": "disabled"}
    if bool(args.sync_notion):
        summary_paths = [
            (ROOT / output["summary_json"]).resolve()
            if not Path(str(output["summary_json"])).is_absolute()
            else Path(str(output["summary_json"])).resolve()
            for output in outputs
        ]
        notion_sync = sync_eval_summaries_to_notion(
            summary_paths,
            fail_on_error=bool(args.notion_fail_on_error),
        )

    print(json.dumps({"eval_name": eval_name, "outputs": outputs, "notion_sync": notion_sync}, indent=2), flush=True)


if __name__ == "__main__":
    main()
