#!/usr/bin/env python3
"""Export local friction-fitting experiment configs and results to one JSON file."""

from __future__ import annotations

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent


def relpath(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def clean(value: Any) -> Any:
    if isinstance(value, Path):
        return relpath(value)
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return clean(value.item())
        return value.tolist()
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def vector_stats(values: Any) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    out: dict[str, Any] = {
        "count": int(arr.size),
        "finite_count": int(finite.size),
    }
    if finite.size == 0:
        return out
    out.update(
        {
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
        }
    )
    return out


def compact_array(values: Any, max_values: int = 16) -> dict[str, Any]:
    arr = np.asarray(values)
    out: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }
    if arr.dtype.kind in "biufc":
        out["stats"] = vector_stats(arr)
    if arr.size <= max_values:
        out["values"] = clean(arr)
    return out


def summarize_loss_history(values: Any) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite_mask = np.isfinite(arr)
    finite = arr[finite_mask]
    out = vector_stats(arr)
    if finite.size:
        finite_indices = np.flatnonzero(finite_mask)
        argmin = int(finite_indices[np.argmin(finite)])
        argmax = int(finite_indices[np.argmax(finite)])
        out.update(
            {
                "first": float(arr[0]) if np.isfinite(arr[0]) else None,
                "last": float(arr[-1]) if np.isfinite(arr[-1]) else None,
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
                "argmin_index_zero_based": argmin,
                "argmin_iteration_one_based": argmin + 1,
                "argmax_index_zero_based": argmax,
                "argmax_iteration_one_based": argmax + 1,
            }
        )
    return out


def scalar_from_npz(data: np.lib.npyio.NpzFile, key: str) -> Any:
    value = data[key]
    if getattr(value, "shape", None) == ():
        return clean(value.item())
    return None


def summarize_npz(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": relpath(path),
        "size_bytes": path.stat().st_size,
    }
    try:
        with np.load(path, allow_pickle=True) as data:
            keys = list(data.keys())
            summary["keys"] = keys
            scalars: dict[str, Any] = {}
            arrays: dict[str, Any] = {}
            for key in keys:
                value = data[key]
                if value.shape == ():
                    scalar_value = value.item()
                    if key != "rng_state":
                        scalars[key] = clean(scalar_value)
                    continue
                if key == "loss_history":
                    summary["loss_history"] = summarize_loss_history(value)
                elif key in {
                    "active_params",
                    "best_active_params",
                    "optimizer_params",
                    "best_optimizer_params",
                    "learned_active_point_friction",
                    "learned_optimizer_friction",
                    "learned_point_friction",
                    "trajectory_steps",
                    "trajectory_frames",
                    "active_indices",
                    "active_contact_point_indices",
                }:
                    arrays[key] = compact_array(value)

            summary["scalars"] = scalars
            if arrays:
                summary["arrays"] = arrays

            if "active_params" in keys:
                summary["kind"] = "checkpoint"
            elif "learned_point_friction" in keys:
                summary["kind"] = "results"
            else:
                summary["kind"] = "npz"
    except Exception as exc:  # noqa: BLE001 - export should continue on corrupt artifacts.
        summary["error"] = repr(exc)
    return summary


def parse_ply(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    properties: list[str] = []
    vertex_count = None
    data_start = None
    comments: list[str] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("comment "):
            comments.append(stripped[len("comment ") :])
        elif stripped.startswith("element vertex "):
            vertex_count = int(stripped.split()[-1])
        elif stripped.startswith("property "):
            parts = stripped.split()
            properties.append(parts[-1])
        elif stripped == "end_header":
            data_start = index + 1
            break

    if data_start is None or vertex_count is None:
        raise ValueError(f"Invalid PLY header: {path}")

    data_lines = lines[data_start : data_start + vertex_count]
    if not data_lines:
        data = np.empty((0, len(properties)), dtype=float)
    else:
        data = np.loadtxt(data_lines, dtype=float)
        if data.ndim == 1:
            data = data.reshape(1, -1)

    columns = {name: i for i, name in enumerate(properties)}
    required = {"x", "y", "z", "friction"}
    missing = required - set(columns)
    if missing:
        raise ValueError(f"PLY missing properties {sorted(missing)}: {path}")

    x = data[:, columns["x"]]
    z = data[:, columns["z"]]
    friction = data[:, columns["friction"]]
    active = (
        data[:, columns["active_contact"]] > 0.5
        if "active_contact" in columns
        else np.ones_like(friction, dtype=bool)
    )
    bottom = z <= float(np.min(z)) + 1e-6 if z.size else np.zeros_like(friction, dtype=bool)

    def side_stats(mask: np.ndarray) -> dict[str, Any]:
        left = mask & (x < 0.0)
        right = mask & (x > 0.0)
        left_stats = vector_stats(friction[left])
        right_stats = vector_stats(friction[right])
        gap = None
        if left_stats.get("finite_count") and right_stats.get("finite_count"):
            gap = float(right_stats["mean"] - left_stats["mean"])
        return {
            "count": int(np.sum(mask)),
            "left": left_stats,
            "right": right_stats,
            "right_minus_left_mean": gap,
        }

    return {
        "path": relpath(path),
        "vertex_count": int(vertex_count),
        "properties": properties,
        "comments": comments,
        "friction": vector_stats(friction),
        "active": side_stats(active),
        "bottom": side_stats(bottom),
        "active_bottom": side_stats(active & bottom),
    }


def summarize_ply(path: Path) -> dict[str, Any]:
    try:
        return parse_ply(path)
    except Exception as exc:  # noqa: BLE001
        return {"path": relpath(path), "error": repr(exc)}


def parse_iteration_from_name(path: Path) -> int | None:
    match = re.search(r"iter_(\d+)", path.stem)
    return int(match.group(1)) if match else None


def unwrap_wandb_config(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    config: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    for key, value in raw.items():
        if key == "_wandb":
            metadata = value.get("value", value) if isinstance(value, dict) else {}
            continue
        if isinstance(value, dict) and set(value.keys()) == {"value"}:
            config[key] = value["value"]
        elif isinstance(value, dict) and "value" in value:
            config[key] = value["value"]
        else:
            config[key] = value
    return clean(config), clean(metadata)


def extract_wandb_runs(wandb_dir: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if not wandb_dir.exists():
        return runs

    for config_path in sorted(wandb_dir.glob("run-*/files/config.yaml")):
        run_dir = config_path.parents[1]
        run_name = run_dir.name
        run_id = run_name.split("-")[-1]
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8", errors="replace")) or {}
        except Exception as exc:  # noqa: BLE001
            runs.append({"run_dir": relpath(run_dir), "run_id": run_id, "error": repr(exc)})
            continue

        config, metadata = unwrap_wandb_config(raw)
        summary_path = run_dir / "files" / "wandb-summary.json"
        summary = None
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8", errors="replace"))
            except Exception as exc:  # noqa: BLE001
                summary = {"error": repr(exc)}

        command_args: list[str] = []
        git: dict[str, Any] | None = None
        started_at = None
        env_records = metadata.get("e", {}) if isinstance(metadata, dict) else {}
        if isinstance(env_records, dict):
            for record in env_records.values():
                if not isinstance(record, dict):
                    continue
                if not command_args and isinstance(record.get("args"), list):
                    command_args = [str(arg) for arg in record["args"]]
                if git is None and isinstance(record.get("git"), dict):
                    git = record["git"]
                if started_at is None and record.get("startedAt"):
                    started_at = record["startedAt"]

        runs.append(
            {
                "run_dir": relpath(run_dir),
                "run_id": run_id,
                "started_at": started_at,
                "config_path": relpath(config_path),
                "summary_path": relpath(summary_path) if summary_path.exists() else None,
                "command_args": command_args,
                "git": clean(git),
                "config": config,
                "summary": clean(summary),
            }
        )
    return runs


def strings_for_matching(value: Any) -> list[str]:
    strings: list[str] = []
    if isinstance(value, str):
        strings.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            strings.extend(strings_for_matching(item))
    elif isinstance(value, list):
        for item in value:
            strings.extend(strings_for_matching(item))
    return strings


def match_wandb_runs(experiment_name: str, output_dir: Path, runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    artifact_names = {
        f"{experiment_name}.npz",
        f"{experiment_name}.ply",
        f"{experiment_name}_results.npz",
    }
    path_needles = {
        relpath(output_dir) or "",
        str(output_dir.resolve()),
    }
    matched: list[dict[str, Any]] = []
    for run in runs:
        haystack_strings = strings_for_matching(run)
        has_path_match = any(
            needle and needle in value
            for needle in path_needles
            for value in haystack_strings
        )
        has_artifact_match = any(
            artifact in Path(value).name or artifact in value
            for artifact in artifact_names
            for value in haystack_strings
        )
        has_exact_name_match = any(value == experiment_name for value in haystack_strings)
        if has_path_match or has_artifact_match or has_exact_name_match:
            matched.append(run)
    return matched


def parse_name_hints(name: str) -> dict[str, Any]:
    hints: dict[str, Any] = {}
    patterns = {
        "initial_point_friction_from_name": r"fixed_init_([0-9.]+)_",
        "contact_stiffness_from_name": r"stiffness_([^_]+)",
        "piecewise_regularization_weight_from_name": r"regularization_([0-9.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, name)
        if match:
            raw = match.group(1)
            try:
                hints[key] = float(raw) if "." in raw else int(raw)
            except ValueError:
                hints[key] = raw
    if "_left_right" in name:
        hints["parameterization_from_name"] = "left-right"
    elif "_global" in name:
        hints["parameterization_from_name"] = "global"
    elif "_point" in name or re.search(r"regularization_\d+$", name):
        hints["parameterization_from_name"] = "point"
    if name.endswith("_v2"):
        hints["variant_from_name"] = "v2"
    elif name.endswith("_new"):
        hints["variant_from_name"] = "new"
    return hints


def choose_main_checkpoint(summaries: dict[Path, dict[str, Any]], output_dir: Path) -> Path | None:
    candidates = [
        path for path, summary in summaries.items() if summary.get("kind") == "checkpoint"
    ]
    if not candidates:
        return None
    exact = output_dir / f"{output_dir.name}.npz"
    if exact in candidates:
        return exact
    return sorted(candidates, key=lambda p: (p.stat().st_mtime, p.name))[-1]


def build_experiment(output_dir: Path, wandb_runs: list[dict[str, Any]]) -> dict[str, Any]:
    npz_files = sorted(output_dir.glob("*.npz"))
    npz_summaries = {path: summarize_npz(path) for path in npz_files}
    main_checkpoint = choose_main_checkpoint(npz_summaries, output_dir)

    result_files = [
        path
        for path, summary in npz_summaries.items()
        if summary.get("kind") == "results" or path.name.endswith("_results.npz")
    ]
    extra_npz = [
        path
        for path in npz_files
        if path != main_checkpoint and path not in result_files
    ]

    top_level_ply = sorted(output_dir.glob("*.ply"))
    point_cloud = top_level_ply[0] if top_level_ply else None
    recursive_ply = sorted(path for path in output_dir.rglob("*.ply") if path.parent != output_dir)
    latest_recursive = None
    if recursive_ply:
        latest_recursive = sorted(
            recursive_ply,
            key=lambda p: (
                parse_iteration_from_name(p) if parse_iteration_from_name(p) is not None else -1,
                p.stat().st_mtime,
            ),
        )[-1]

    matched_runs = match_wandb_runs(output_dir.name, output_dir, wandb_runs)
    checkpoint_summary = npz_summaries.get(main_checkpoint) if main_checkpoint else None
    result_summaries = [npz_summaries[path] for path in result_files]

    return {
        "experiment_name": output_dir.name,
        "output_dir": relpath(output_dir),
        "name_hints": parse_name_hints(output_dir.name),
        "files": {
            "main_checkpoint": relpath(main_checkpoint),
            "results_npz": [relpath(path) for path in result_files],
            "extra_npz": [relpath(path) for path in extra_npz],
            "point_cloud": relpath(point_cloud),
            "checkpoint_point_cloud_count": len(recursive_ply),
            "latest_checkpoint_point_cloud": relpath(latest_recursive),
        },
        "checkpoint": checkpoint_summary,
        "results": result_summaries,
        "point_cloud": summarize_ply(point_cloud) if point_cloud else None,
        "latest_checkpoint_point_cloud": summarize_ply(latest_recursive) if latest_recursive else None,
        "wandb_runs": matched_runs,
    }


def extract_metric(experiment: dict[str, Any], *path: str) -> Any:
    value: Any = experiment
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def first_result_scalar(experiment: dict[str, Any], key: str) -> Any:
    results = experiment.get("results") or []
    if not results:
        return None
    return (results[0].get("scalars") or {}).get(key)


def build_summary_row(experiment: dict[str, Any]) -> dict[str, Any]:
    checkpoint_scalars = extract_metric(experiment, "checkpoint", "scalars") or {}
    checkpoint_arrays = extract_metric(experiment, "checkpoint", "arrays") or {}
    optimizer = checkpoint_arrays.get("best_optimizer_params") or checkpoint_arrays.get("optimizer_params") or {}
    active = checkpoint_arrays.get("best_active_params") or checkpoint_arrays.get("active_params") or {}
    ply_active = extract_metric(experiment, "point_cloud", "active") or {}
    ply_latest_active = extract_metric(experiment, "latest_checkpoint_point_cloud", "active") or {}

    return {
        "experiment_name": experiment.get("experiment_name"),
        "parameterization": checkpoint_scalars.get("friction_parameterization")
        or first_result_scalar(experiment, "friction_parameterization")
        or extract_metric(experiment, "name_hints", "parameterization_from_name"),
        "initial_point_friction_hint": extract_metric(
            experiment, "name_hints", "initial_point_friction_from_name"
        ),
        "regularization_hint": extract_metric(
            experiment, "name_hints", "piecewise_regularization_weight_from_name"
        ),
        "variant_hint": extract_metric(experiment, "name_hints", "variant_from_name"),
        "iteration": checkpoint_scalars.get("iteration"),
        "best_loss": checkpoint_scalars.get("best_loss") or first_result_scalar(experiment, "best_loss"),
        "loss_argmin_iteration": extract_metric(
            experiment, "checkpoint", "loss_history", "argmin_iteration_one_based"
        ),
        "final_loss_history_value": extract_metric(experiment, "checkpoint", "loss_history", "last"),
        "trajectory_npz_path": checkpoint_scalars.get("trajectory_npz_path")
        or first_result_scalar(experiment, "trajectory_npz_path"),
        "max_steps": checkpoint_scalars.get("max_steps"),
        "max_trajectories": checkpoint_scalars.get("max_trajectories"),
        "active_contact_point_count": extract_metric(
            experiment, "checkpoint", "arrays", "active_indices", "stats", "count"
        )
        or extract_metric(
            experiment, "checkpoint", "arrays", "active_contact_point_indices", "stats", "count"
        ),
        "best_active_mu_mean": extract_metric(active, "stats", "mean"),
        "best_active_mu_std": extract_metric(active, "stats", "std"),
        "best_active_mu_min": extract_metric(active, "stats", "min"),
        "best_active_mu_max": extract_metric(active, "stats", "max"),
        "best_optimizer_values": optimizer.get("values"),
        "point_cloud_active_left_mean": extract_metric(ply_active, "left", "mean"),
        "point_cloud_active_right_mean": extract_metric(ply_active, "right", "mean"),
        "point_cloud_active_right_minus_left": ply_active.get("right_minus_left_mean"),
        "latest_ply_active_left_mean": extract_metric(ply_latest_active, "left", "mean"),
        "latest_ply_active_right_mean": extract_metric(ply_latest_active, "right", "mean"),
        "latest_ply_active_right_minus_left": ply_latest_active.get("right_minus_left_mean"),
        "wandb_run_ids": [run.get("run_id") for run in experiment.get("wandb_runs", [])],
    }


def extract_ground_truth(xml_path: Path) -> dict[str, Any] | None:
    if not xml_path.exists():
        return None
    try:
        root = ET.parse(xml_path).getroot()
    except Exception as exc:  # noqa: BLE001
        return {"xml_path": relpath(xml_path), "error": repr(exc)}

    geoms: dict[str, Any] = {}
    for geom in root.iter("geom"):
        name = geom.attrib.get("name")
        if name in {"push_block_left", "push_block_right", "floor"}:
            friction_raw = geom.attrib.get("friction")
            first_friction = None
            if friction_raw:
                try:
                    first_friction = float(friction_raw.split()[0])
                except ValueError:
                    first_friction = friction_raw
            geoms[name] = {
                "friction": friction_raw,
                "first_friction": first_friction,
            }
    return {"xml_path": relpath(xml_path), "geoms": geoms}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--wandb-dir", type=Path, default=ROOT / "wandb")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "report_assets" / "all_experiment_results.json",
    )
    args = parser.parse_args()

    outputs_dir = args.outputs_dir.resolve()
    wandb_dir = args.wandb_dir.resolve()
    wandb_runs = extract_wandb_runs(wandb_dir)

    experiments: list[dict[str, Any]] = []
    if outputs_dir.exists():
        for output_dir in sorted(path for path in outputs_dir.iterdir() if path.is_dir()):
            if any(output_dir.glob("*.npz")) or any(output_dir.glob("*.ply")):
                experiments.append(build_experiment(output_dir, wandb_runs))

    matched_run_dirs = {
        run.get("run_dir")
        for experiment in experiments
        for run in experiment.get("wandb_runs", [])
    }
    unmatched_wandb_runs = [
        run for run in wandb_runs if run.get("run_dir") not in matched_run_dirs
    ]

    payload = {
        "generated_by": relpath(Path(__file__)),
        "generated_at": np.datetime64("now").astype(str),
        "root": str(ROOT),
        "ground_truth": extract_ground_truth(
            ROOT
            / "mujoco"
            / "third_party"
            / "mujoco_menagerie"
            / "franka_emika_panda"
            / "block_force_scene.xml"
        ),
        "counts": {
            "experiments": len(experiments),
            "wandb_runs": len(wandb_runs),
            "unmatched_wandb_runs": len(unmatched_wandb_runs),
        },
        "summary_table": [build_summary_row(experiment) for experiment in experiments],
        "experiments": experiments,
        "unmatched_wandb_runs": unmatched_wandb_runs,
    }

    output_json = args.output_json.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(clean(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(relpath(output_json))
    print(json.dumps(payload["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
