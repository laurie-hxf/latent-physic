from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
REQUIRED_SUMMARY_FIELDS = (
    "schema_version",
    "dataset",
    "dataset_label",
    "eval_name",
    "loss_weights",
    "method",
    "artifacts",
    "trajectory_count",
    "overlay_loss_mean",
)
FINITE_LOSS_FIELDS = (
    "overlay_loss_mean",
    "overlay_loss_std",
    "overlay_loss_min",
    "overlay_loss_max",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--summary-json", type=Path, action="append", default=None)
    parser.add_argument("--experiment-dir", type=Path, action="append", default=None)
    parser.add_argument("--checkpoint-root", type=Path, action="append", default=None)
    parser.add_argument("--eval-name", type=str, default=None)
    return parser.parse_args()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def discover_summaries(args: argparse.Namespace) -> list[Path]:
    paths = list(args.summary_json or [])
    pattern = (
        f"eval/{args.eval_name}/*_eval_summary.json"
        if args.eval_name
        else "eval/*/*_eval_summary.json"
    )
    for experiment_dir in args.experiment_dir or []:
        paths.extend(experiment_dir.glob(pattern))
    for checkpoint_root in args.checkpoint_root or []:
        paths.extend(checkpoint_root.rglob(pattern))
    return sorted({path.resolve() for path in paths})


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def expected_checkpoint_role(checkpoint: str) -> str:
    stem = Path(checkpoint).stem.lower()
    if stem.endswith("_last"):
        return "last"
    if stem.endswith("_best") or "_best_" in stem:
        return "best"
    return "primary"


def standard_protocol_error(summary: dict[str, Any]) -> str | None:
    dataset_label = str(summary.get("dataset_label") or "")
    if dataset_label == "rotation68":
        expected_count = 68
        expected_max_steps = 300
    elif dataset_label == "very_long20":
        expected_count = 20
        expected_max_steps = None
    else:
        return None
    if summary.get("max_steps") != expected_max_steps:
        return f"{dataset_label} standard protocol requires max_steps={expected_max_steps!r}"
    if summary.get("trajectory_count") != expected_count:
        return f"{dataset_label} standard protocol requires trajectory_count={expected_count}"
    if summary.get("selected_trajectories") != list(range(expected_count)):
        return f"{dataset_label} standard protocol requires all canonical trajectory indices in order"
    return None


def validate_summary(path: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        summary = load_json(path)
    except Exception as exc:
        return {"summary": repo_rel(path), "errors": [f"cannot load JSON: {exc}"], "warnings": []}

    for field in REQUIRED_SUMMARY_FIELDS:
        if field not in summary:
            errors.append(f"missing summary field: {field}")

    method = summary.get("method") if isinstance(summary.get("method"), dict) else {}
    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    checkpoint = str(method.get("checkpoint") or "")
    if not checkpoint:
        errors.append("method.checkpoint is missing")
    elif not resolve_repo_path(checkpoint).exists():
        errors.append(f"checkpoint does not exist: {checkpoint}")

    checkpoint_role = summary.get("checkpoint_role")
    if checkpoint_role is None:
        warnings.append("checkpoint_role is missing; artifact predates experiment-eval-v2 fields")
    elif checkpoint and checkpoint_role != expected_checkpoint_role(checkpoint):
        errors.append(
            f"checkpoint_role={checkpoint_role!r} does not match checkpoint filename role "
            f"{expected_checkpoint_role(checkpoint)!r}"
        )

    protocol = summary.get("evaluation_protocol")
    if not isinstance(protocol, dict):
        warnings.append("evaluation_protocol is missing; protocol must be inferred from top-level fields")
    else:
        for field, top_level in (
            ("name", "eval_name"),
            ("dataset_label", "dataset_label"),
            ("max_steps", "max_steps"),
            ("loss_weights", "loss_weights"),
        ):
            if protocol.get(field) != summary.get(top_level):
                errors.append(f"evaluation_protocol.{field} does not match {top_level}")

    if summary.get("schema_version") != "experiment-eval-v3":
        for field in FINITE_LOSS_FIELDS:
            value = summary.get(field)
            if value is None:
                if field == "overlay_loss_std":
                    warnings.append("overlay_loss_std is missing; artifact predates experiment-eval-v2 fields")
                continue
            try:
                if not math.isfinite(float(value)):
                    errors.append(f"{field} is not finite: {value!r}")
            except (TypeError, ValueError):
                errors.append(f"{field} is not numeric: {value!r}")

    if summary.get("schema_version") == "experiment-eval-v3":
        if summary.get("metric_version") != "trajectory-fit-v1":
            errors.append("experiment-eval-v3 requires metric_version='trajectory-fit-v1'")
        if not isinstance(summary.get("protocol_fingerprint"), dict):
            errors.append("experiment-eval-v3 requires protocol_fingerprint")
        if not isinstance(summary.get("unified_metrics"), dict):
            errors.append("experiment-eval-v3 requires unified_metrics")
        for field in ("state_nte_mean", "pose_nte_mean", "finite_rollout_rate", "complete_rollout_rate"):
            value = summary.get(field)
            if value is None:
                errors.append(f"experiment-eval-v3 requires {field}")
            else:
                try:
                    if not math.isfinite(float(value)):
                        errors.append(f"{field} is not finite: {value!r}")
                except (TypeError, ValueError):
                    errors.append(f"{field} is not numeric: {value!r}")
        for field in ("finite_rollout_rate", "complete_rollout_rate"):
            value = summary.get(field)
            try:
                rate = float(value)
                if not 0.0 <= rate <= 1.0:
                    errors.append(f"{field} must be in [0, 1], got {value!r}")
                elif rate < 1.0:
                    warnings.append(f"{field}={rate:.6g}; model rollout is not fully valid")
            except (TypeError, ValueError):
                pass
        protocol_error = standard_protocol_error(summary)
        if protocol_error and summary.get("eval_name") in {"rotation68", "very_long20"}:
            errors.append(protocol_error)

    rollout_value = artifacts.get("trajectory_rollouts_json")
    if not rollout_value:
        errors.append("artifacts.trajectory_rollouts_json is missing")
    else:
        rollout_path = resolve_repo_path(str(rollout_value))
        if not rollout_path.exists():
            errors.append(f"rollout JSON does not exist: {rollout_value}")
        else:
            try:
                rollout = load_json(rollout_path)
                for field in (
                    "dataset",
                    "dataset_label",
                    "eval_name",
                    "max_steps",
                    "loss_weights",
                    "pointnet_residual_gain",
                    "pointnet_residual_output_mode",
                    "evaluation_protocol",
                    "checkpoint_role",
                ):
                    if rollout.get(field) != summary.get(field):
                        errors.append(f"rollout {field} does not match summary")
                rollout_methods = (rollout.get("interactive_data") or {}).get("methods") or []
                if len(rollout_methods) != 1:
                    errors.append(f"rollout must contain exactly one method, found {len(rollout_methods)}")
                elif checkpoint and str(rollout_methods[0].get("checkpoint") or "") != checkpoint:
                    errors.append("rollout method checkpoint does not match summary method checkpoint")
                targets = (rollout.get("interactive_data") or {}).get("targets") or []
                if summary.get("trajectory_count") is not None and len(targets) != int(summary["trajectory_count"]):
                    errors.append("rollout target count does not match summary trajectory_count")
            except Exception as exc:
                errors.append(f"cannot validate rollout JSON: {exc}")

    rollout_npz_value = artifacts.get("trajectory_rollouts_npz")
    if summary.get("schema_version") == "experiment-eval-v3":
        if not rollout_npz_value:
            errors.append("experiment-eval-v3 requires artifacts.trajectory_rollouts_npz")
        else:
            rollout_npz_path = resolve_repo_path(str(rollout_npz_value))
            if not rollout_npz_path.exists():
                errors.append(f"rollout NPZ does not exist: {rollout_npz_value}")
            else:
                try:
                    import numpy as np

                    with np.load(rollout_npz_path, allow_pickle=True) as data:
                        required = {
                            "timestamps",
                            "target_positions",
                            "target_quaternions_xyzw",
                            "target_linear_velocity",
                            "target_angular_velocity",
                            "predicted_positions",
                            "predicted_quaternions_xyzw",
                            "predicted_linear_velocity",
                            "predicted_angular_velocity",
                            "valid_mask",
                        }
                        missing = sorted(required - set(data.files))
                        if missing:
                            errors.append(f"rollout NPZ missing v3 fields: {missing}")
                except Exception as exc:
                    errors.append(f"cannot validate rollout NPZ: {exc}")

    return {"summary": repo_rel(path), "errors": errors, "warnings": warnings}


def main() -> None:
    args = parse_args()
    summaries = discover_summaries(args)
    if not summaries:
        raise FileNotFoundError("No *_eval_summary.json files found")
    results = [validate_summary(path) for path in summaries]
    error_count = sum(len(result["errors"]) for result in results)
    warning_count = sum(len(result["warnings"]) for result in results)
    print(
        json.dumps(
            {
                "summary_count": len(results),
                "error_count": error_count,
                "warning_count": warning_count,
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if error_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
