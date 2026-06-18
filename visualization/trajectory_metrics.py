from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


METRIC_VERSION = "trajectory-fit-v1"
METRIC_TOLERANCES = {
    "position_m": 0.01,
    "yaw_rad": float(np.deg2rad(5.0)),
    "linear_velocity_mps": 0.10,
    "angular_velocity_radps": 1.0,
}
POSE_WEIGHTS = {"position": 0.5, "yaw": 0.5}
STATE_WEIGHTS = {"position": 0.35, "yaw": 0.35, "linear_velocity": 0.15, "angular_velocity": 0.15}
FAILURE_POSITION_M = 0.05
FAILURE_YAW_RAD = float(np.deg2rad(30.0))
FAILURE_PERSISTENCE_S = 0.1
HORIZONS_BY_DATASET = {
    "rotation68": (0.3,),
    "very_long20": (0.6, 2.0, 4.0, 7.4),
}


def quaternion_xyzw_to_yaw(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.maximum(norm, 1.0e-12)
    x, y, z, w = np.moveaxis(q, -1, 0)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    value = np.asarray(angle, dtype=np.float64)
    return np.arctan2(np.sin(value), np.cos(value))


@lru_cache(maxsize=16)
def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_fingerprint(
    *,
    dataset: Path,
    dataset_label: str,
    selected_trajectories: list[int],
    max_steps: int | None,
    contact_stiffness: float,
    contact_damping: float,
    surface_point_spacing: float,
    friction_contact_threshold: float,
    contact_mask_threshold: float,
    residual_gain: float | None,
    residual_output_mode: str | None,
    stateful_reset_interval: int | None,
) -> dict[str, Any]:
    resolved_dataset = dataset.resolve()
    fields = {
        "metric_version": METRIC_VERSION,
        "dataset": str(resolved_dataset),
        "dataset_sha256": file_sha256(resolved_dataset),
        "dataset_label": str(dataset_label),
        "selected_trajectories": [int(value) for value in selected_trajectories],
        "max_steps": None if max_steps is None else int(max_steps),
        "required_frame_policy": "all_selected_valid_frames",
        "exclude_initial_frame": True,
        "metric_tolerances": METRIC_TOLERANCES,
        "pose_weights": POSE_WEIGHTS,
        "state_weights": STATE_WEIGHTS,
        "contact_stiffness": float(contact_stiffness),
        "contact_damping": float(contact_damping),
        "surface_point_spacing": float(surface_point_spacing),
        "friction_contact_threshold": float(friction_contact_threshold),
        "contact_mask_threshold": float(contact_mask_threshold),
        "residual_gain": None if residual_gain is None else float(residual_gain),
        "residual_output_mode": None if residual_output_mode is None else str(residual_output_mode),
        "stateful_reset_interval": None if stateful_reset_interval is None else int(stateful_reset_interval),
    }
    encoded = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"id": hashlib.sha256(encoded).hexdigest(), "fields": fields}


def evaluation_fingerprint(protocol_id: str, checkpoint: str, checkpoint_role: str) -> str:
    encoded = json.dumps(
        {
            "protocol_fingerprint": str(protocol_id),
            "checkpoint": str(Path(checkpoint).resolve()),
            "checkpoint_role": str(checkpoint_role),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _state_array(state: dict[str, np.ndarray], key: str, width: int) -> np.ndarray:
    value = np.asarray(state[key], dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != width:
        raise ValueError(f"{key} must have shape [frames, {width}], got {value.shape}")
    return value


def _target_timestamps(state: dict[str, np.ndarray], frame_count: int) -> np.ndarray:
    timestamps = np.asarray(state.get("timestamps"), dtype=np.float64)
    if timestamps.shape != (frame_count,):
        raise ValueError(f"timestamps must have shape [{frame_count}], got {timestamps.shape}")
    return timestamps


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2)))


def _persistent_failure_time(
    timestamps: np.ndarray,
    position_error: np.ndarray,
    yaw_error: np.ndarray,
) -> float | None:
    failed = (position_error > FAILURE_POSITION_M) | (yaw_error > FAILURE_YAW_RAD)
    start_idx: int | None = None
    for idx, is_failed in enumerate(failed):
        if is_failed and start_idx is None:
            start_idx = idx
        elif not is_failed:
            start_idx = None
        if start_idx is not None and timestamps[idx] - timestamps[start_idx] >= FAILURE_PERSISTENCE_S:
            return float(timestamps[start_idx])
    return None


def _metric_for_frames(
    *,
    position_error: np.ndarray,
    yaw_error: np.ndarray,
    linear_velocity_error: np.ndarray,
    angular_velocity_error: np.ndarray,
    frame_mask: np.ndarray,
) -> dict[str, float]:
    pos = position_error[frame_mask]
    yaw = yaw_error[frame_mask]
    vel = linear_velocity_error[frame_mask]
    omega = angular_velocity_error[frame_mask]
    if len(pos) == 0:
        raise ValueError("metric frame mask selected no prediction frames")
    p = float(np.mean((pos / METRIC_TOLERANCES["position_m"]) ** 2))
    y = float(np.mean((yaw / METRIC_TOLERANCES["yaw_rad"]) ** 2))
    v = float(np.mean((vel / METRIC_TOLERANCES["linear_velocity_mps"]) ** 2))
    w = float(np.mean((omega / METRIC_TOLERANCES["angular_velocity_radps"]) ** 2))
    return {
        "xy_rmse_m": _rmse(pos),
        "yaw_rmse_rad": _rmse(yaw),
        "linear_velocity_rmse_mps": _rmse(vel),
        "angular_velocity_rmse_radps": _rmse(omega),
        "pose_nte": float(np.sqrt(POSE_WEIGHTS["position"] * p + POSE_WEIGHTS["yaw"] * y)),
        "state_nte": float(
            np.sqrt(
                STATE_WEIGHTS["position"] * p
                + STATE_WEIGHTS["yaw"] * y
                + STATE_WEIGHTS["linear_velocity"] * v
                + STATE_WEIGHTS["angular_velocity"] * w
            )
        ),
    }


def trajectory_metrics(
    *,
    target: dict[str, np.ndarray],
    predicted: dict[str, np.ndarray],
    dataset_label: str,
) -> dict[str, Any]:
    target_positions = _state_array(target, "positions", 3)
    target_quaternions = _state_array(target, "quaternions_xyzw", 4)
    target_linear = _state_array(target, "linear_velocity", 3)
    target_angular = _state_array(target, "angular_velocity", 3)
    target_frames = len(target_positions)
    timestamps = _target_timestamps(target, target_frames)

    predicted_positions = _state_array(predicted, "positions", 3)
    predicted_quaternions = _state_array(predicted, "quaternions_xyzw", 4)
    predicted_linear = _state_array(predicted, "linear_velocity", 3)
    predicted_angular = _state_array(predicted, "angular_velocity", 3)
    predicted_frames = min(
        len(predicted_positions),
        len(predicted_quaternions),
        len(predicted_linear),
        len(predicted_angular),
    )
    compared_frames = min(target_frames, predicted_frames)
    complete = predicted_frames >= target_frames
    if compared_frames <= 1:
        return {
            "complete": False,
            "finite": False,
            "target_frames": int(target_frames),
            "predicted_frames": int(predicted_frames),
            "failure_time_s": 0.0,
            "metrics": None,
            "horizons": {},
        }

    target_positions = target_positions[:compared_frames]
    target_quaternions = target_quaternions[:compared_frames]
    target_linear = target_linear[:compared_frames]
    target_angular = target_angular[:compared_frames]
    predicted_positions = predicted_positions[:compared_frames]
    predicted_quaternions = predicted_quaternions[:compared_frames]
    predicted_linear = predicted_linear[:compared_frames]
    predicted_angular = predicted_angular[:compared_frames]
    timestamps = timestamps[:compared_frames]

    finite_per_frame = (
        np.isfinite(predicted_positions).all(axis=1)
        & np.isfinite(predicted_quaternions).all(axis=1)
        & np.isfinite(predicted_linear).all(axis=1)
        & np.isfinite(predicted_angular).all(axis=1)
    )
    finite = bool(complete and finite_per_frame.all())
    scoring_mask = finite_per_frame.copy()
    scoring_mask[0] = False
    if not scoring_mask.any():
        return {
            "complete": bool(complete),
            "finite": False,
            "target_frames": int(target_frames),
            "predicted_frames": int(predicted_frames),
            "failure_time_s": 0.0,
            "metrics": None,
            "horizons": {},
        }

    position_error = np.linalg.norm(predicted_positions[:, :2] - target_positions[:, :2], axis=1)
    target_yaw = quaternion_xyzw_to_yaw(target_quaternions)
    predicted_yaw = quaternion_xyzw_to_yaw(predicted_quaternions)
    yaw_error = np.abs(wrap_angle(predicted_yaw - target_yaw))
    linear_velocity_error = np.linalg.norm(predicted_linear[:, :2] - target_linear[:, :2], axis=1)
    angular_velocity_error = np.abs(predicted_angular[:, 2] - target_angular[:, 2])

    metrics = _metric_for_frames(
        position_error=position_error,
        yaw_error=yaw_error,
        linear_velocity_error=linear_velocity_error,
        angular_velocity_error=angular_velocity_error,
        frame_mask=scoring_mask,
    )
    last_valid = int(np.flatnonzero(scoring_mask)[-1])
    metrics["final_xy_error_m"] = float(position_error[last_valid])
    metrics["final_yaw_error_rad"] = float(yaw_error[last_valid])

    failure_time = _persistent_failure_time(
        timestamps[scoring_mask],
        position_error[scoring_mask],
        yaw_error[scoring_mask],
    )
    if not complete or not finite:
        first_bad = int(np.flatnonzero(~finite_per_frame)[0]) if not finite_per_frame.all() else compared_frames - 1
        incomplete_time = float(timestamps[min(first_bad, len(timestamps) - 1)])
        failure_time = incomplete_time if failure_time is None else min(failure_time, incomplete_time)

    horizons: dict[str, Any] = {}
    for horizon_s in HORIZONS_BY_DATASET.get(str(dataset_label), ()):
        horizon_mask = scoring_mask & (timestamps <= float(horizon_s) + 1.0e-9)
        if horizon_mask.any():
            horizon_metrics = _metric_for_frames(
                position_error=position_error,
                yaw_error=yaw_error,
                linear_velocity_error=linear_velocity_error,
                angular_velocity_error=angular_velocity_error,
                frame_mask=horizon_mask,
            )
            horizon_metrics["success"] = bool(failure_time is None or failure_time > float(horizon_s))
            horizons[f"{float(horizon_s):g}s"] = horizon_metrics

    return {
        "complete": bool(complete),
        "finite": bool(finite),
        "target_frames": int(target_frames),
        "predicted_frames": int(predicted_frames),
        "duration_s": float(timestamps[-1] - timestamps[0]),
        "failure_time_s": failure_time,
        "metrics": metrics,
        "horizons": horizons,
    }


def _bootstrap_mean_ci(values: np.ndarray, *, seed: int = 0, samples: int = 2000) -> list[float] | None:
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


def _aggregate_values(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        return {"mean": None, "median": None, "p90": None, "bootstrap_95ci_mean": None}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.9)),
        "bootstrap_95ci_mean": _bootstrap_mean_ci(finite),
    }


def aggregate_trajectory_metrics(per_trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "xy_rmse_m",
        "yaw_rmse_rad",
        "linear_velocity_rmse_mps",
        "angular_velocity_rmse_radps",
        "final_xy_error_m",
        "final_yaw_error_rad",
        "pose_nte",
        "state_nte",
    )
    valid_metrics = [row["metrics"] for row in per_trajectory if isinstance(row.get("metrics"), dict)]
    aggregate = {
        name: _aggregate_values([float(metrics[name]) for metrics in valid_metrics])
        for name in metric_names
    }
    trajectory_count = len(per_trajectory)
    finite_count = sum(bool(row.get("finite")) for row in per_trajectory)
    complete_count = sum(bool(row.get("complete")) for row in per_trajectory)
    failure_times = [
        float(row["failure_time_s"])
        for row in per_trajectory
        if row.get("failure_time_s") is not None and np.isfinite(float(row["failure_time_s"]))
    ]
    horizon_names = sorted({name for row in per_trajectory for name in row.get("horizons", {})})
    horizons: dict[str, Any] = {}
    for name in horizon_names:
        rows = [row["horizons"][name] for row in per_trajectory if name in row.get("horizons", {})]
        horizons[name] = {
            "trajectory_count": len(rows),
            "success_rate": float(np.mean([bool(row.get("success")) for row in rows])) if rows else None,
            "state_nte": _aggregate_values([float(row["state_nte"]) for row in rows]),
            "pose_nte": _aggregate_values([float(row["pose_nte"]) for row in rows]),
        }
    return {
        "trajectory_count": trajectory_count,
        "finite_rollout_rate": float(finite_count / trajectory_count) if trajectory_count else 0.0,
        "complete_rollout_rate": float(complete_count / trajectory_count) if trajectory_count else 0.0,
        "median_time_to_failure_s": float(np.median(failure_times)) if failure_times else None,
        "metrics": aggregate,
        "horizons": horizons,
    }


def evaluate_state_rollouts(
    *,
    targets: list[dict[str, np.ndarray]],
    predictions: list[dict[str, np.ndarray]],
    dataset_label: str,
    trajectory_indices: list[int],
) -> dict[str, Any]:
    if len(targets) != len(predictions) or len(targets) != len(trajectory_indices):
        raise ValueError("target, prediction, and trajectory-index counts must match")
    per_trajectory = []
    for trajectory_index, target, predicted in zip(trajectory_indices, targets, predictions, strict=True):
        row = trajectory_metrics(target=target, predicted=predicted, dataset_label=dataset_label)
        row["trajectory_index"] = int(trajectory_index)
        per_trajectory.append(row)
    return {
        "metric_version": METRIC_VERSION,
        "metric_tolerances": METRIC_TOLERANCES,
        "pose_weights": POSE_WEIGHTS,
        "state_weights": STATE_WEIGHTS,
        "failure_definition": {
            "position_m": FAILURE_POSITION_M,
            "yaw_rad": FAILURE_YAW_RAD,
            "persistence_s": FAILURE_PERSISTENCE_S,
        },
        "aggregate": aggregate_trajectory_metrics(per_trajectory),
        "per_trajectory": per_trajectory,
    }
