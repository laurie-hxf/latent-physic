from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fit_mujoco_contact_point_friction_params import sample_training_time_windows
from fit_mujoco_contact_point_friction_runtime import sample_training_batch_indices
from mujoco_contact_friction_fit_utils import MujocoTrajectory, slice_mujoco_trajectory_time_window


@dataclass(frozen=True)
class TrajectorySplits:
    train: list[MujocoTrajectory]
    val: list[MujocoTrajectory]


def split_trajectories(
    trajectories: list[MujocoTrajectory],
    *,
    train_fraction: float,
    seed: int,
    min_steps: int,
) -> TrajectorySplits:
    eligible = [trajectory for trajectory in trajectories if trajectory.num_steps >= int(min_steps)]
    if not eligible:
        raise ValueError(f"No trajectories contain at least {int(min_steps)} steps")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(eligible))
    train_count = int(round(float(train_fraction) * len(eligible)))
    train_count = min(max(train_count, 1), len(eligible))
    if train_count == len(eligible) and len(eligible) > 1:
        train_count -= 1
    train = [eligible[int(idx)] for idx in order[:train_count]]
    val = [eligible[int(idx)] for idx in order[train_count:]]
    if not val:
        val = train[:]
    return TrajectorySplits(train=train, val=val)


def sample_window_batch(
    trajectories: list[MujocoTrajectory],
    *,
    batch_size: int,
    window_steps: int,
    rng: np.random.Generator,
    random_time_windows: bool,
) -> tuple[list[MujocoTrajectory], np.ndarray, np.ndarray]:
    if not trajectories:
        raise ValueError("Cannot sample from an empty trajectory list")
    indices = sample_training_batch_indices(len(trajectories), int(batch_size), rng)
    selected = [trajectories[int(idx)] for idx in indices]
    if bool(random_time_windows):
        windows, start_steps = sample_training_time_windows(
            trajectories=selected,
            window_steps=int(window_steps),
            rng=rng,
            enabled=True,
        )
    else:
        windows = [
            slice_mujoco_trajectory_time_window(trajectory, start_step=0, window_steps=int(window_steps))
            for trajectory in selected
        ]
        start_steps = np.zeros(len(windows), dtype=np.int32)
    return windows, indices.astype(np.int32), start_steps.astype(np.int32)

