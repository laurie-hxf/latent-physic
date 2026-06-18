from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from plot_topdown_trajectory_overlays_interactive import render_html


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HTML_OUTPUT_DIR = ROOT / "report_assets" / "eval_html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        action="append",
        default=None,
        help="Experiment output directory containing eval/<eval-name>/*_trajectory_rollouts.json.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        action="append",
        default=None,
        help="Scan recursively for eval/<eval-name>/*_trajectory_rollouts.json under these roots.",
    )
    parser.add_argument(
        "--rollout-json",
        type=Path,
        action="append",
        default=None,
        help="Explicit per-experiment rollout JSON file written by evaluate_experiments.py.",
    )
    parser.add_argument("--eval-name", type=str, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--html-output-dir", type=Path, default=DEFAULT_HTML_OUTPUT_DIR)
    parser.add_argument("--plot-width", type=int, default=280)
    parser.add_argument("--plot-height", type=int, default=230)
    parser.add_argument("--legend-width", type=int, default=520)
    parser.add_argument("--axis-padding-frac", type=float, default=0.12)
    parser.add_argument("--unified-axis-scale", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def discover_rollout_jsons(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.rollout_json:
        paths.extend(args.rollout_json)
    if args.experiment_dir:
        for experiment_dir in args.experiment_dir:
            paths.extend(sorted((experiment_dir / "eval" / args.eval_name).glob("*_trajectory_rollouts.json")))
    if args.checkpoint_root:
        pattern = f"eval/{args.eval_name}/*_trajectory_rollouts.json"
        for root in args.checkpoint_root:
            paths.extend(sorted(root.rglob(pattern)))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "interactive_data" not in payload:
        raise ValueError(f"{path} has no interactive_data")
    methods = payload["interactive_data"].get("methods", [])
    if len(methods) != 1:
        raise ValueError(f"{path} should contain exactly one saved method, found {len(methods)}")
    return payload


def assert_same_targets(reference: dict[str, Any], candidate: dict[str, Any], path: Path) -> None:
    keys = [
        "dataset",
        "eval_name",
        "max_steps",
        "selected_trajectories",
        "contact_stiffness",
        "surface_point_spacing",
        "loss_weights",
        "point_position_loss_reduction",
        "pointnet_residual_gain",
        "pointnet_residual_output_mode",
        "evaluation_protocol",
    ]
    for key in keys:
        if reference.get(key) != candidate.get(key):
            raise ValueError(f"{path} does not match {key}: {candidate.get(key)!r} != {reference.get(key)!r}")


def build_combined_payload(payloads: list[dict[str, Any]], output_path: Path) -> dict[str, Any]:
    if not payloads:
        raise ValueError("No saved rollout payloads were provided")
    first = payloads[0]
    for payload in payloads[1:]:
        assert_same_targets(first, payload, Path(str(payload.get("artifacts", {}).get("trajectory_rollouts_json", ""))))

    methods = []
    method_summaries = []
    for payload in payloads:
        method = dict(payload["interactive_data"]["methods"][0])
        methods.append(method)
        method_summaries.append(payload.get("method", {}))

    return {
        "schema_version": first.get("schema_version"),
        "dataset": first.get("dataset"),
        "dataset_label": first.get("dataset_label"),
        "eval_name": first.get("eval_name"),
        "max_steps": first.get("max_steps"),
        "selected_trajectories": first.get("selected_trajectories"),
        "eval_batch_size": first.get("eval_batch_size"),
        "contact_stiffness": first.get("contact_stiffness"),
        "surface_point_spacing": first.get("surface_point_spacing"),
        "loss_weights": first.get("loss_weights"),
        "point_position_loss_reduction": first.get("point_position_loss_reduction"),
        "pointnet_residual_gain": first.get("pointnet_residual_gain"),
        "pointnet_residual_output_mode": first.get("pointnet_residual_output_mode"),
        "evaluation_protocol": first.get("evaluation_protocol"),
        "methods": method_summaries,
        "reference_datasets": first.get("reference_datasets", []),
        "trajectory_losses": {
            method["name"]: {
                str(target["trajectory_index"]): float(method["losses"][idx])
                for idx, target in enumerate(first["interactive_data"]["targets"])
            }
            for method in methods
        },
        "interactive_data": {
            "targets": first["interactive_data"]["targets"],
            "references": first["interactive_data"].get("references", []),
            "methods": methods,
        },
        "output": str(output_path),
        "source_rollouts": [
            payload.get("artifacts", {}).get("trajectory_rollouts_json")
            for payload in payloads
        ],
    }


def build_summary_payload(combined: dict[str, Any]) -> dict[str, Any]:
    summary = dict(combined)
    interactive = summary.pop("interactive_data", {})
    summary["trajectory_count"] = len(interactive.get("targets", []))
    summary["method_count"] = len(interactive.get("methods", []))
    summary["reference_count"] = len(interactive.get("references", []))
    return summary


def main() -> None:
    args = parse_args()
    rollout_paths = discover_rollout_jsons(args)
    if not rollout_paths:
        raise FileNotFoundError(
            f"No saved rollout JSON files found for eval_name={args.eval_name!r}. "
            "Run visualization/evaluate_experiments.py first."
        )
    payloads = [load_payload(path) for path in rollout_paths]
    output_path = args.output or (args.html_output_dir / f"{args.eval_name}_saved_eval.html")
    render_args = SimpleNamespace(
        plot_width=int(args.plot_width),
        plot_height=int(args.plot_height),
        legend_width=int(args.legend_width),
        axis_padding_frac=float(args.axis_padding_frac),
        unified_axis_scale=bool(args.unified_axis_scale),
    )
    combined = build_combined_payload(payloads, output_path)
    html_text = render_html(combined, render_args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    summary_path.write_text(json.dumps(build_summary_payload(combined), indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "html": repo_rel(output_path),
                "summary": repo_rel(summary_path),
                "rollout_json_count": len(rollout_paths),
                "rollout_jsons": [repo_rel(path) for path in rollout_paths],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
