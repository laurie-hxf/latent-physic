from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = ROOT / "eval/historical_experiment_registry/registry.json"
DEFAULT_OUTPUT_DIR = ROOT / "eval/unified_trajectory_eval"
DEFAULT_BASELINE = (
    ROOT
    / "outputs/rotation_l0p20_r0p50_2000_global_m300_rotloss_angvel"
    / "rotation_l0p20_r0p50_2000_global_m300_rotloss_angvel.npz"
)
PROTOCOLS = ("rotation68", "very_long20")
CSV_FIELDS = (
    "rank",
    "record_id",
    "experiment_name",
    "family",
    "checkpoint",
    "checkpoint_role",
    "state_nte_mean",
    "pose_nte_mean",
    "xy_rmse_m",
    "yaw_rmse_rad",
    "linear_velocity_rmse_mps",
    "angular_velocity_rmse_radps",
    "finite_rollout_rate",
    "complete_rollout_rate",
    "median_time_to_failure_s",
    "paired_mean_delta_vs_global",
    "paired_bootstrap_95ci_low",
    "paired_bootstrap_95ci_high",
    "trajectory_win_rate_vs_global",
    "summary_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--registry-json", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--outputs-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--top", type=int, default=20)
    return parser.parse_args()


def repo_rel(path: str | Path) -> str:
    value = Path(path)
    try:
        return str(value.resolve().relative_to(ROOT))
    except Exception:
        return str(value)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_key(value: str | Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return str(path.resolve())


def registry_by_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    data = load_json(path)
    result: dict[str, dict[str, Any]] = {}
    for record in data.get("records", []):
        artifact = record.get("artifact_path")
        if artifact:
            result[checkpoint_key(str(artifact))] = record
    return result


def aggregate_mean(summary: dict[str, Any], metric: str) -> float | None:
    value = (
        ((summary.get("unified_metrics") or {}).get("aggregate") or {})
        .get("metrics", {})
        .get(metric, {})
        .get("mean")
    )
    return None if value is None else float(value)


def trajectory_state_nte(summary: dict[str, Any]) -> dict[int, float]:
    result = {}
    for row in (summary.get("unified_metrics") or {}).get("per_trajectory", []):
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            continue
        value = metrics.get("state_nte")
        if value is None or not np.isfinite(float(value)):
            continue
        result[int(row["trajectory_index"])] = float(value)
    return result


def bootstrap_mean_ci(values: np.ndarray, *, seed: int = 0, samples: int = 5000) -> list[float] | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return None
    if len(finite) == 1:
        return [float(finite[0]), float(finite[0])]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(finite), size=(samples, len(finite)))
    means = np.mean(finite[indices], axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def paired_comparison(summary: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any] | None:
    model_values = trajectory_state_nte(summary)
    baseline_values = trajectory_state_nte(baseline)
    indices = sorted(set(model_values) & set(baseline_values))
    if not indices:
        return None
    deltas = np.asarray([model_values[idx] - baseline_values[idx] for idx in indices], dtype=np.float64)
    return {
        "trajectory_count": len(indices),
        "paired_mean_delta": float(np.mean(deltas)),
        "paired_bootstrap_95ci": bootstrap_mean_ci(deltas),
        "trajectory_win_rate": float(np.mean(deltas < 0.0)),
    }


def is_standard_summary(summary: dict[str, Any], protocol: str) -> bool:
    expected_count = 68 if protocol == "rotation68" else 20
    expected_max_steps = 300 if protocol == "rotation68" else None
    return bool(
        summary.get("schema_version") == "experiment-eval-v3"
        and summary.get("metric_version") == "trajectory-fit-v1"
        and summary.get("dataset_label") == protocol
        and summary.get("eval_name") == protocol
        and summary.get("trajectory_count") == expected_count
        and summary.get("selected_trajectories") == list(range(expected_count))
        and summary.get("max_steps") == expected_max_steps
        and isinstance(summary.get("protocol_fingerprint"), dict)
    )


def collect_summaries(outputs_root: Path, registry: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    collected = {protocol: [] for protocol in PROTOCOLS}
    for protocol in PROTOCOLS:
        for path in sorted(outputs_root.rglob(f"eval/{protocol}/*_eval_summary.json")):
            summary = load_json(path)
            if not is_standard_summary(summary, protocol):
                continue
            method = summary.get("method") or {}
            checkpoint = checkpoint_key(str(method.get("checkpoint") or ""))
            record = registry.get(checkpoint, {})
            aggregate = (summary.get("unified_metrics") or {}).get("aggregate") or {}
            row = {
                "record_id": str(record.get("record_id") or repo_rel(path.parent.parent.parent)),
                "experiment_name": str(record.get("experiment_name") or method.get("name") or path.stem),
                "family": str(record.get("family") or method.get("family") or "unknown"),
                "checkpoint": repo_rel(checkpoint),
                "checkpoint_role": str(summary.get("checkpoint_role") or ""),
                "state_nte_mean": aggregate_mean(summary, "state_nte"),
                "pose_nte_mean": aggregate_mean(summary, "pose_nte"),
                "xy_rmse_m": aggregate_mean(summary, "xy_rmse_m"),
                "yaw_rmse_rad": aggregate_mean(summary, "yaw_rmse_rad"),
                "linear_velocity_rmse_mps": aggregate_mean(summary, "linear_velocity_rmse_mps"),
                "angular_velocity_rmse_radps": aggregate_mean(summary, "angular_velocity_rmse_radps"),
                "finite_rollout_rate": float(aggregate.get("finite_rollout_rate", 0.0)),
                "complete_rollout_rate": float(aggregate.get("complete_rollout_rate", 0.0)),
                "median_time_to_failure_s": aggregate.get("median_time_to_failure_s"),
                "summary_path": repo_rel(path),
                "_summary": summary,
            }
            row["rank_eligible"] = bool(
                row["finite_rollout_rate"] == 1.0
                and row["complete_rollout_rate"] == 1.0
                and row["state_nte_mean"] is not None
                and np.isfinite(float(row["state_nte_mean"]))
            )
            collected[protocol].append(row)
    return collected


def attach_paired_stats(rows: list[dict[str, Any]], baseline: dict[str, Any]) -> None:
    for row in rows:
        paired = paired_comparison(row["_summary"], baseline)
        row["paired_comparison_vs_global"] = paired
        row["paired_mean_delta_vs_global"] = None if paired is None else paired["paired_mean_delta"]
        ci = None if paired is None else paired["paired_bootstrap_95ci"]
        row["paired_bootstrap_95ci_low"] = None if ci is None else ci[0]
        row["paired_bootstrap_95ci_high"] = None if ci is None else ci[1]
        row["trajectory_win_rate_vs_global"] = None if paired is None else paired["trajectory_win_rate"]


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})


def format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def write_markdown(
    path: Path,
    *,
    protocol_rows: dict[str, list[dict[str, Any]]],
    comparison_rows: list[dict[str, Any]],
    baseline: dict[str, Any],
    top: int,
) -> None:
    lines = [
        "# Unified Trajectory Evaluation Results",
        "",
        "Metric version: `trajectory-fit-v1`. Lower `state_nte` and `pose_nte` are better.",
        "Only standard-protocol results with finite and complete rollout rates equal to 1 enter rankings.",
        "",
        f"Global baseline checkpoint: `{baseline['checkpoint']}`",
        "",
    ]
    for protocol in PROTOCOLS:
        rows = protocol_rows[protocol]
        ranked = sorted(
            (row for row in rows if row["rank_eligible"]),
            key=lambda row: int(row["rank"]),
        )
        unstable = sorted(
            (row for row in rows if not row["rank_eligible"]),
            key=lambda row: (row["family"], row["experiment_name"], row["checkpoint"]),
        )
        family_best: dict[str, dict[str, Any]] = {}
        for row in ranked:
            family_best.setdefault(row["family"], row)
        lines.extend(
            [
                f"## {protocol}",
                "",
                f"- Standard summaries: {len(rows)}",
                f"- Rank eligible: {len(ranked)}",
                f"- Incomplete or non-finite: {len(unstable)}",
                "",
                markdown_table(
                    ["rank", "state_nte", "pose_nte", "win rate", "family", "experiment"],
                    [
                        [
                            row["rank"],
                            format_number(row["state_nte_mean"]),
                            format_number(row["pose_nte_mean"]),
                            format_number(row["trajectory_win_rate_vs_global"], 3),
                            row["family"],
                            row["experiment_name"],
                        ]
                        for row in ranked[:top]
                    ],
                ),
                "",
                "### Best By Family",
                "",
                markdown_table(
                    ["state_nte", "pose_nte", "win rate", "family", "experiment"],
                    [
                        [
                            format_number(row["state_nte_mean"]),
                            format_number(row["pose_nte_mean"]),
                            format_number(row["trajectory_win_rate_vs_global"], 3),
                            row["family"],
                            row["experiment_name"],
                        ]
                        for row in sorted(family_best.values(), key=lambda value: int(value["rank"]))
                    ],
                ),
                "",
            ]
        )
        if unstable:
            lines.extend(
                [
                    "### Incomplete Or Non-Finite",
                    "",
                    markdown_table(
                        ["finite rate", "complete rate", "family", "experiment"],
                        [
                            [
                                format_number(row["finite_rollout_rate"], 3),
                                format_number(row["complete_rollout_rate"], 3),
                                row["family"],
                                row["experiment_name"],
                            ]
                            for row in unstable
                        ],
                    ),
                    "",
                ]
            )
    lines.extend(
        [
            "## Cross-Protocol Comparison Index",
            "",
            "The index is `100 * (0.7 * rotation68/global + 0.3 * very_long20/global)`.",
            "Only the same checkpoint with complete finite rollouts on both protocols is included; lower is better.",
            "",
            markdown_table(
                ["rank", "index", "rot68", "very_long20", "family", "experiment"],
                [
                    [
                        row["rank"],
                        format_number(row["model_comparison_index"], 2),
                        format_number(row["rotation68_state_nte"]),
                        format_number(row["very_long20_state_nte"]),
                        row["family"],
                        row["experiment_name"],
                    ]
                    for row in comparison_rows[:top]
                ],
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    registry = registry_by_checkpoint(args.registry_json)
    collected = collect_summaries(args.outputs_root, registry)
    baseline_key = checkpoint_key(args.baseline_checkpoint)
    baseline_rows: dict[str, dict[str, Any]] = {}
    for protocol in PROTOCOLS:
        match = next((row for row in collected[protocol] if checkpoint_key(row["checkpoint"]) == baseline_key), None)
        if match is None:
            raise FileNotFoundError(f"Missing standard {protocol} summary for baseline {args.baseline_checkpoint}")
        baseline_rows[protocol] = match
        attach_paired_stats(collected[protocol], match["_summary"])
        ranked = sorted(
            (row for row in collected[protocol] if row["rank_eligible"]),
            key=lambda row: float(row["state_nte_mean"]),
        )
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = rank
        for row in collected[protocol]:
            row.setdefault("rank", None)

    by_protocol_checkpoint = {
        protocol: {checkpoint_key(row["checkpoint"]): row for row in collected[protocol]}
        for protocol in PROTOCOLS
    }
    baseline_rotation = float(baseline_rows["rotation68"]["state_nte_mean"])
    baseline_long = float(baseline_rows["very_long20"]["state_nte_mean"])
    comparison_rows = []
    for checkpoint in sorted(set(by_protocol_checkpoint["rotation68"]) & set(by_protocol_checkpoint["very_long20"])):
        rotation_row = by_protocol_checkpoint["rotation68"][checkpoint]
        long_row = by_protocol_checkpoint["very_long20"][checkpoint]
        if not rotation_row["rank_eligible"] or not long_row["rank_eligible"]:
            continue
        index = 100.0 * (
            0.7 * float(rotation_row["state_nte_mean"]) / baseline_rotation
            + 0.3 * float(long_row["state_nte_mean"]) / baseline_long
        )
        comparison_rows.append(
            {
                "record_id": rotation_row["record_id"],
                "experiment_name": rotation_row["experiment_name"],
                "family": rotation_row["family"],
                "checkpoint": rotation_row["checkpoint"],
                "rotation68_state_nte": rotation_row["state_nte_mean"],
                "very_long20_state_nte": long_row["state_nte_mean"],
                "model_comparison_index": index,
            }
        )
    comparison_rows.sort(key=lambda row: float(row["model_comparison_index"]))
    for rank, row in enumerate(comparison_rows, start=1):
        row["rank"] = rank

    args.output_dir.mkdir(parents=True, exist_ok=True)
    public_protocol_rows = {}
    for protocol in PROTOCOLS:
        rows = sorted(
            collected[protocol],
            key=lambda row: (
                not row["rank_eligible"],
                float(row["state_nte_mean"]) if row["state_nte_mean"] is not None else float("inf"),
            ),
        )
        public_protocol_rows[protocol] = [public_row(row) for row in rows]
        write_csv(args.output_dir / f"{protocol}_ranking.csv", rows)

    with (args.output_dir / "model_comparison_index.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = (
            "rank",
            "record_id",
            "experiment_name",
            "family",
            "checkpoint",
            "rotation68_state_nte",
            "very_long20_state_nte",
            "model_comparison_index",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparison_rows)

    report = {
        "schema_version": "unified-trajectory-eval-summary-v1",
        "metric_version": "trajectory-fit-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": {
            "checkpoint": repo_rel(baseline_key),
            "rotation68_state_nte": baseline_rotation,
            "very_long20_state_nte": baseline_long,
        },
        "counts": {
            protocol: {
                "summaries": len(collected[protocol]),
                "rank_eligible": sum(bool(row["rank_eligible"]) for row in collected[protocol]),
                "incomplete_or_non_finite": sum(not bool(row["rank_eligible"]) for row in collected[protocol]),
                "by_family": dict(Counter(row["family"] for row in collected[protocol])),
            }
            for protocol in PROTOCOLS
        },
        "protocols": public_protocol_rows,
        "model_comparison_index": comparison_rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(
        args.output_dir / "README.md",
        protocol_rows=collected,
        comparison_rows=comparison_rows,
        baseline=report["baseline"],
        top=args.top,
    )
    print(
        json.dumps(
            {
                "output_dir": repo_rel(args.output_dir),
                "counts": report["counts"],
                "comparison_index_count": len(comparison_rows),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
