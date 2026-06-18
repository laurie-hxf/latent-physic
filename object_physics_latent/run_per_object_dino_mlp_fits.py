from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "mujoco" / "outputs" / "object_physics_latent_box_partitions_48x2000_min300" / "manifest.json"
FIT_SCRIPT = REPO_ROOT / "newton" / "fit_mujoco_contact_point_friction.py"


def _tag_value(value: Any) -> str:
    return str(value).replace(" ", "_").replace("/", "_")


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("objects"), list):
        raise ValueError(f"{path} is not a valid object physics latent manifest")
    return manifest


def _parse_object_selection(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    selected: set[str] = set()
    for raw in values:
        for part in str(raw).split(","):
            item = part.strip()
            if item:
                selected.add(item)
    return selected or None


def _matches_selection(record: dict[str, Any], selected: set[str] | None) -> bool:
    if selected is None:
        return True
    object_id = str(record.get("object_id", ""))
    dataset_index = str(record.get("dataset_index", ""))
    return object_id in selected or dataset_index in selected


def _record_family(record: dict[str, Any]) -> str:
    friction_spec = record.get("friction_spec")
    if isinstance(friction_spec, dict):
        return str(friction_spec.get("partition_family", ""))
    return ""


def _filter_objects(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = _parse_object_selection(args.object)
    split_filter = None if args.object_split == "all" else str(args.object_split)
    family_filter = None if args.family == "all" else str(args.family)

    filtered: list[dict[str, Any]] = []
    for record in records:
        if split_filter is not None and str(record.get("object_split")) != split_filter:
            continue
        if family_filter is not None and _record_family(record) != family_filter:
            continue
        if not _matches_selection(record, selected):
            continue
        filtered.append(record)

    if args.limit is not None:
        filtered = filtered[: max(int(args.limit), 0)]
    if int(args.num_shards) < 1:
        raise ValueError("--num-shards must be >= 1")
    if int(args.shard_index) < 0 or int(args.shard_index) >= int(args.num_shards):
        raise ValueError("--shard-index must be in [0, num_shards)")
    if int(args.num_shards) > 1:
        filtered = [
            record
            for record in filtered
            if int(record.get("dataset_index", len(filtered))) % int(args.num_shards) == int(args.shard_index)
        ]
    return filtered


def _object_paths(manifest_path: Path, record: dict[str, Any]) -> tuple[Path, Path, Path]:
    root = manifest_path.parent
    trajectory_rel = record.get("trajectory_npz")
    dino_rel = record.get("dino_feature_npz")
    object_id = str(record.get("object_id", ""))
    if not object_id:
        raise ValueError(f"Manifest record is missing object_id: {record}")
    if not trajectory_rel:
        raise ValueError(f"Manifest record for {object_id} is missing trajectory_npz")
    if not dino_rel:
        raise ValueError(f"Manifest record for {object_id} is missing dino_feature_npz")

    trajectory_npz = root / str(trajectory_rel)
    dino_feature_npz = root / str(dino_rel)
    object_dir = trajectory_npz.parent
    return object_dir, trajectory_npz, dino_feature_npz


def _build_fit_command(
    *,
    args: argparse.Namespace,
    record: dict[str, Any],
    object_dir: Path,
    trajectory_npz: Path,
    dino_feature_npz: Path,
) -> tuple[list[str], Path]:
    object_id = str(record["object_id"])
    object_split = str(record.get("object_split", "unknown"))
    family = _record_family(record) or "unknown"
    experiment_dir = object_dir / str(args.output_subdir)
    command = [
        sys.executable,
        str(FIT_SCRIPT),
        "--trajectory-npz",
        str(trajectory_npz),
        "--experiment-dir",
        str(experiment_dir),
        "--friction-parameterization",
        "dino-mlp",
        "--dino-feature-npz",
        str(dino_feature_npz),
        "--device",
        str(args.device),
        "--batch-size",
        str(args.batch_size),
        "--opt-iters",
        str(args.opt_iters),
        "--learning-rate",
        str(args.learning_rate),
        "--log-every",
        str(args.log_every),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--trajectory-progress-every",
        str(args.trajectory_progress_every),
        "--surface-point-spacing",
        str(args.surface_point_spacing),
        "--contact-stiffness",
        str(args.contact_stiffness),
        "--contact-damping",
        str(args.contact_damping),
        "--contact-mask-threshold",
        str(args.contact_mask_threshold),
        "--friction-contact-threshold",
        str(args.friction_contact_threshold),
        "--max-steps",
        str(args.max_steps),
        "--point-friction",
        str(args.point_friction),
        "--min-point-friction",
        str(args.min_point_friction),
        "--max-point-friction",
        str(args.max_point_friction),
        "--dino-neighbor-radius",
        str(args.dino_neighbor_radius),
        "--dino-neighbor-k",
        str(args.dino_neighbor_k),
        "--dino-position-frequencies",
        str(args.dino_position_frequencies),
        "--dino-mlp-hidden-dim",
        str(args.dino_mlp_hidden_dim),
        "--dino-mlp-hidden-layers",
        str(args.dino_mlp_hidden_layers),
        "--position-loss-weight",
        str(args.position_loss_weight),
        "--orientation-loss-weight",
        str(args.orientation_loss_weight),
        "--linear-velocity-loss-weight",
        str(args.linear_velocity_loss_weight),
        "--angular-velocity-loss-weight",
        str(args.angular_velocity_loss_weight),
        "--point-cloud-color-min",
        str(args.point_cloud_color_min),
        "--point-cloud-color-max",
        str(args.point_cloud_color_max),
        "--seed",
        str(int(args.seed) + int(record.get("dataset_index", 0))),
    ]
    if args.random_time_windows:
        command.append("--random-time-windows")
    if args.window_steps is not None:
        command.extend(["--window-steps", str(args.window_steps)])
    if args.max_trajectories is not None:
        command.extend(["--max-trajectories", str(args.max_trajectories)])
    if args.grad_clip_norm is not None:
        command.extend(["--grad-clip-norm", str(args.grad_clip_norm)])
    if args.no_dino_feature_normalization:
        command.append("--no-dino-feature-normalization")
    if args.wandb:
        command.append("--wandb")
        command.extend(["--wandb-project", str(args.wandb_project)])
        if args.wandb_entity is not None:
            command.extend(["--wandb-entity", str(args.wandb_entity)])
        command.extend(["--wandb-run-name", f"{args.wandb_run_prefix}{object_id}"])
        command.extend(["--wandb-group", str(args.wandb_group)])
        command.extend(["--wandb-mode", str(args.wandb_mode)])
        if args.wandb_dir is not None:
            command.extend(["--wandb-dir", str(args.wandb_dir)])
        wandb_tags = [
            "per-object-dino-mlp",
            f"object:{_tag_value(object_id)}",
            f"split:{_tag_value(object_split)}",
            f"family:{_tag_value(family)}",
        ]
        wandb_tags.extend(args.wandb_tags)
        command.append("--wandb-tags")
        command.extend(wandb_tags)
    for extra in args.extra_fit_arg:
        command.extend(shlex.split(extra))
    return command, experiment_dir


def _write_run_record(
    *,
    path: Path,
    manifest_path: Path,
    record: dict[str, Any],
    command: list[str],
    status: str,
    returncode: int | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": status,
        "returncode": returncode,
        "manifest": str(manifest_path),
        "object_id": record.get("object_id"),
        "dataset_index": record.get("dataset_index"),
        "object_split": record.get("object_split"),
        "friction_spec": record.get("friction_spec"),
        "command": command,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _checkpoint_iteration(checkpoint_path: Path) -> int | None:
    if not checkpoint_path.exists():
        return None
    try:
        with np.load(checkpoint_path, allow_pickle=True) as data:
            if "iteration" in data.files:
                return int(np.asarray(data["iteration"]).item())
            if "loss_history" in data.files:
                return int(len(np.asarray(data["loss_history"])))
    except Exception:
        return None
    return None


def _skip_reason_for_existing_artifact(
    *,
    results_path: Path,
    checkpoint_path: Path,
    min_iteration: int | None,
) -> str | None:
    if results_path.exists():
        return "complete_results"
    if min_iteration is None:
        return None
    iteration = _checkpoint_iteration(checkpoint_path)
    if iteration is not None and iteration >= int(min_iteration):
        return f"checkpoint_iteration_{iteration}_ge_{int(min_iteration)}"
    return None


def _resume_checkpoint_path(
    *,
    results_path: Path,
    checkpoint_path: Path,
    min_iteration: int | None,
) -> Path | None:
    if results_path.exists():
        return None
    if not checkpoint_path.exists():
        return None
    if min_iteration is None:
        return checkpoint_path
    iteration = _checkpoint_iteration(checkpoint_path)
    if iteration is None:
        return checkpoint_path
    if iteration < int(min_iteration):
        return checkpoint_path
    return None


def run(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    manifest = _load_manifest(manifest_path)
    records = _filter_objects(list(manifest["objects"]), args)
    if not records:
        print("No objects matched the requested filters.", file=sys.stderr)
        return 2

    print(f"manifest={manifest_path}")
    print(f"selected_objects={len(records)}")
    print(f"output_subdir={args.output_subdir}")

    for object_number, record in enumerate(records, start=1):
        object_dir, trajectory_npz, dino_feature_npz = _object_paths(manifest_path, record)
        command, experiment_dir = _build_fit_command(
            args=args,
            record=record,
            object_dir=object_dir,
            trajectory_npz=trajectory_npz,
            dino_feature_npz=dino_feature_npz,
        )
        experiment_name = experiment_dir.name
        results_path = experiment_dir / f"{experiment_name}_results.npz"
        checkpoint_path = experiment_dir / f"{experiment_name}.npz"
        run_record_path = experiment_dir / "per_object_dino_mlp_run.json"

        object_id = str(record["object_id"])
        print(f"\n[{object_number}/{len(records)}] {object_id}")
        print(f"experiment_dir={experiment_dir}")

        skip_reason = _skip_reason_for_existing_artifact(
            results_path=results_path,
            checkpoint_path=checkpoint_path,
            min_iteration=args.skip_complete_or_min_iteration,
        )
        if skip_reason is not None:
            print(f"skip_existing_artifact: {skip_reason}")
            if not args.dry_run:
                _write_run_record(
                    path=run_record_path,
                    manifest_path=manifest_path,
                    record=record,
                    command=command,
                    status=f"skipped_{skip_reason}",
                    returncode=0,
                )
            continue

        resume_checkpoint = _resume_checkpoint_path(
            results_path=results_path,
            checkpoint_path=checkpoint_path,
            min_iteration=args.skip_complete_or_min_iteration,
        )
        if resume_checkpoint is not None:
            command.extend(["--resume-checkpoint", str(resume_checkpoint)])

        print("command=" + shlex.join(command))

        if args.dry_run:
            continue
        if args.skip_existing and results_path.exists():
            print(f"skip_existing: {results_path} exists")
            _write_run_record(
                path=run_record_path,
                manifest_path=manifest_path,
                record=record,
                command=command,
                status="skipped_existing",
                returncode=0,
            )
            continue

        _write_run_record(
            path=run_record_path,
            manifest_path=manifest_path,
            record=record,
            command=command,
            status="running",
            returncode=None,
        )
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        status = "complete" if completed.returncode == 0 else "failed"
        _write_run_record(
            path=run_record_path,
            manifest_path=manifest_path,
            record=record,
            command=command,
            status=status,
            returncode=int(completed.returncode),
        )
        if completed.returncode != 0:
            print(f"failed: {object_id} returncode={completed.returncode}", file=sys.stderr)
            if not args.continue_on_error:
                return int(completed.returncode)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit one independent DINO-MLP friction field per object in the "
            "object_physics_latent dataset. Training outputs are written inside each object directory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-subdir", type=str, default="dino_mlp_fit")
    parser.add_argument("--object-split", choices=("train", "validation", "test", "all"), default="all")
    parser.add_argument("--family", choices=("left_right", "front_back", "center_ends", "all"), default="all")
    parser.add_argument(
        "--object",
        action="append",
        default=None,
        help="Object id or dataset_index to run. Can be repeated or comma-separated.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--skip-complete-or-min-iteration",
        type=int,
        default=None,
        help=(
            "Skip objects that already have final results, or whose existing checkpoint iteration is at least this value. "
            "Use this to relaunch only unfinished/short partial object fits."
        ),
    )
    parser.add_argument("--continue-on-error", action="store_true")

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--opt-iters", type=int, default=20000)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--trajectory-progress-every", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--random-time-windows", action="store_true", default=True)
    parser.add_argument("--no-random-time-windows", dest="random_time_windows", action="store_false")
    parser.add_argument("--window-steps", type=int, default=None)

    parser.add_argument("--surface-point-spacing", type=float, default=0.01)
    parser.add_argument("--contact-stiffness", type=float, default=1.0e5)
    parser.add_argument("--contact-damping", type=float, default=50.0)
    parser.add_argument("--contact-mask-threshold", type=float, default=0.002)
    parser.add_argument("--friction-contact-threshold", type=float, default=0.002)
    parser.add_argument("--point-friction", type=float, default=0.35)
    parser.add_argument("--min-point-friction", type=float, default=0.0)
    parser.add_argument("--max-point-friction", type=float, default=2.0)
    parser.add_argument("--point-cloud-color-min", type=float, default=0.0)
    parser.add_argument("--point-cloud-color-max", type=float, default=0.8)

    parser.add_argument("--dino-neighbor-radius", type=float, default=0.025)
    parser.add_argument("--dino-neighbor-k", type=int, default=16)
    parser.add_argument("--dino-position-frequencies", type=int, default=6)
    parser.add_argument("--dino-mlp-hidden-dim", type=int, default=128)
    parser.add_argument("--dino-mlp-hidden-layers", type=int, default=2)
    parser.add_argument("--no-dino-feature-normalization", action="store_true")

    parser.add_argument("--position-loss-weight", type=float, default=1.0)
    parser.add_argument("--orientation-loss-weight", type=float, default=1.0)
    parser.add_argument("--linear-velocity-loss-weight", type=float, default=0.0)
    parser.add_argument("--angular-velocity-loss-weight", type=float, default=0.1)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="newton_friction_fitting")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default="per-object-dino-mlp")
    parser.add_argument("--wandb-run-prefix", type=str, default="per_object_dino_mlp_")
    parser.add_argument("--wandb-mode", type=str, default="online")
    parser.add_argument("--wandb-dir", type=Path, default=None)
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=[])
    parser.add_argument(
        "--extra-fit-arg",
        action="append",
        default=[],
        help="Additional argument string passed through to fit_mujoco_contact_point_friction.py. Repeat as needed.",
    )
    return parser.parse_args()


def main() -> None:
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
