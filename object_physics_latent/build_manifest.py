from __future__ import annotations

import argparse
import json
from pathlib import Path

from object_physics_latent.dataset import build_manifest, discover_npz_datasets, validate_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic multi-object trajectory manifest from MuJoCo dataset NPZ files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Dataset NPZ files or directories recursively containing dataset NPZ files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "mujoco" / "outputs" / "object_physics_latent" / "manifest.json",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context-fraction", type=float, default=0.15)
    parser.add_argument("--query-fraction", type=float, default=0.75)
    parser.add_argument("--eval-fraction", type=float, default=0.10)
    parser.add_argument("--train-object-fraction", type=float, default=0.70)
    parser.add_argument("--validation-object-fraction", type=float, default=0.15)
    parser.add_argument("--test-object-fraction", type=float, default=0.15)
    parser.add_argument("--min-episode-steps", type=int, default=1)
    parser.add_argument("--dino-feature-npz", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = discover_npz_datasets(args.inputs)
    build_manifest(
        datasets,
        output_path=args.output,
        seed=int(args.seed),
        episode_split_fractions=(
            float(args.context_fraction),
            float(args.query_fraction),
            float(args.eval_fraction),
        ),
        object_split_fractions=(
            float(args.train_object_fraction),
            float(args.validation_object_fraction),
            float(args.test_object_fraction),
        ),
        min_episode_steps=int(args.min_episode_steps),
        dino_feature_npz=args.dino_feature_npz,
    )
    summary = validate_manifest(args.output, inspect_datasets=True)
    print(json.dumps({"manifest": str(args.output.resolve()), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
