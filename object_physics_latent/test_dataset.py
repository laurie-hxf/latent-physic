from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from object_physics_latent.dataset import (
    ENCODER_FEATURE_SCHEMA,
    ObjectPhysicsDataset,
    build_manifest,
    load_manifest,
    validate_manifest,
)


COLUMNS = np.asarray(
    [
        "time",
        "pos_x",
        "pos_y",
        "pos_z",
        "quat_w",
        "quat_x",
        "quat_y",
        "quat_z",
        "linvel_x",
        "linvel_y",
        "linvel_z",
        "angvel_x",
        "angvel_y",
        "angvel_z",
        "force_x",
        "force_y",
        "force_z",
        "point_x",
        "point_y",
        "point_z",
    ]
)


def write_fake_dataset(path: Path, *, left_mu: float, right_mu: float, episodes: int = 24, frames: int = 12) -> None:
    trajectories = np.zeros((episodes, frames, len(COLUMNS)), dtype=np.float32)
    episode_lengths = np.full((episodes,), frames, dtype=np.int32)
    point_offsets = np.zeros((episodes, 3), dtype=np.float32)
    episode_metadata = []
    for episode in range(episodes):
        time = np.arange(frames, dtype=np.float32) * 0.01
        yaw = 0.01 * episode + 0.02 * np.arange(frames, dtype=np.float32)
        trajectories[episode, :, 0] = time
        trajectories[episode, :, 1] = 0.01 * np.arange(frames, dtype=np.float32)
        trajectories[episode, :, 2] = 0.001 * episode
        trajectories[episode, :, 3] = 0.05
        trajectories[episode, :, 4] = np.cos(0.5 * yaw)
        trajectories[episode, :, 7] = np.sin(0.5 * yaw)
        trajectories[episode, :, 8] = 1.0
        trajectories[episode, :, 13] = 2.0
        trajectories[episode, 1:, 14] = 3.0 + episode
        trajectories[episode, 1:, 15] = -1.0
        point_offsets[episode] = [0.05, -0.02, 0.0]
        episode_metadata.append({"episode_index": episode, "point_offset_local": point_offsets[episode].tolist()})
    summary = {
        "timestep": 0.01,
        "scene_path": "fake_box.xml",
        "block_local_bounds_min": [-0.1, -0.05, -0.025],
        "block_local_bounds_max": [0.1, 0.05, 0.025],
        "block_friction": {
            "push_block_left": [left_mu, 0.0, 0.0],
            "push_block_right": [right_mu, 0.0, 0.0],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        trajectories=trajectories,
        columns=COLUMNS,
        episode_lengths=episode_lengths,
        point_offset_local=point_offsets,
        episode_metadata_json=np.asarray(json.dumps(episode_metadata)),
        summary_metadata_json=np.asarray(json.dumps(summary)),
    )


class ObjectPhysicsDatasetTest(unittest.TestCase):
    def test_build_validate_and_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            datasets = []
            for index, (left, right) in enumerate(((0.1, 0.2), (0.2, 0.5), (0.5, 0.2), (0.6, 0.6))):
                path = root / f"object_{index:02d}" / f"object_{index:02d}.npz"
                write_fake_dataset(path, left_mu=left, right_mu=right)
                datasets.append(path)
            manifest_path = root / "manifest.json"
            build_manifest(
                datasets,
                output_path=manifest_path,
                seed=7,
                episode_split_fractions=(0.25, 0.50, 0.25),
                object_split_fractions=(0.50, 0.25, 0.25),
                min_episode_steps=5,
            )
            summary = validate_manifest(manifest_path, inspect_datasets=True)
            self.assertEqual(summary["objects"], 4)
            self.assertEqual(sum(summary["object_split_counts"].values()), 4)
            _, objects = load_manifest(manifest_path)
            for obj in objects:
                self.assertFalse(set(obj.context_episode_indices) & set(obj.query_episode_indices))
                self.assertFalse(set(obj.context_episode_indices) & set(obj.eval_episode_indices))
                self.assertFalse(set(obj.query_episode_indices) & set(obj.eval_episode_indices))

            dataset = ObjectPhysicsDataset(manifest_path, cache_size=2)
            samples = dataset.sample_training_step(
                split="train",
                objects_per_step=2,
                context_trajectories_per_view=2,
                query_trajectories_per_view=3,
                context_window_steps=5,
                query_window_steps=6,
                random_context_windows=True,
                random_query_windows=True,
                rng=np.random.default_rng(123),
            )
            self.assertEqual(len(samples), 2)
            for sample in samples:
                self.assertEqual(sample.context_a.features.shape, (2, 5, len(ENCODER_FEATURE_SCHEMA)))
                self.assertEqual(sample.context_b.features.shape, (2, 5, len(ENCODER_FEATURE_SCHEMA)))
                self.assertTrue(np.all(sample.context_a.valid_mask))
                self.assertEqual(len(sample.query_a.trajectories), 3)
                self.assertEqual(len(sample.query_b.trajectories), 3)
                self.assertFalse(
                    set(sample.context_a.episode_indices.tolist()) & set(sample.context_b.episode_indices.tolist())
                )
                self.assertFalse(set(sample.query_a.episode_indices.tolist()) & set(sample.query_b.episode_indices.tolist()))
                for query in sample.query_a.trajectories + sample.query_b.trajectories:
                    self.assertEqual(query.num_steps, 6)


if __name__ == "__main__":
    unittest.main()
