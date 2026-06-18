from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from object_physics_latent.dataset import ObjectPhysicsDataset, validate_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a multi-object physics manifest and sample one training step.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--objects-per-step", type=int, default=2)
    parser.add_argument("--context-trajectories-per-view", type=int, default=4)
    parser.add_argument("--query-trajectories-per-view", type=int, default=64)
    parser.add_argument("--context-window-steps", type=int, default=300)
    parser.add_argument("--query-window-steps", type=int, default=300)
    parser.add_argument("--cache-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-sample", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validate_manifest(args.manifest, inspect_datasets=True)
    output: dict[str, object] = {"manifest": str(args.manifest.resolve()), **summary}
    if not args.skip_sample:
        dataset = ObjectPhysicsDataset(args.manifest, cache_size=int(args.cache_size))
        samples = dataset.sample_training_step(
            split=str(args.split),
            objects_per_step=int(args.objects_per_step),
            context_trajectories_per_view=int(args.context_trajectories_per_view),
            query_trajectories_per_view=int(args.query_trajectories_per_view),
            context_window_steps=int(args.context_window_steps),
            query_window_steps=int(args.query_window_steps),
            random_context_windows=True,
            random_query_windows=True,
            rng=np.random.default_rng(int(args.seed)),
        )
        output["sampled_objects"] = [
            {
                "object_id": sample.object_spec.object_id,
                "object_split": sample.object_spec.object_split,
                "context_a_shape": list(sample.context_a.features.shape),
                "context_b_shape": list(sample.context_b.features.shape),
                "query_a_count": len(sample.query_a.trajectories),
                "query_b_count": len(sample.query_b.trajectories),
                "context_overlap": bool(
                    set(sample.context_a.episode_indices.tolist())
                    & set(sample.context_b.episode_indices.tolist())
                ),
                "query_overlap": bool(
                    set(sample.query_a.episode_indices.tolist())
                    & set(sample.query_b.episode_indices.tolist())
                ),
            }
            for sample in samples
        ]
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
