from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MUJOCO_SCRIPT_DIR = ROOT / "mujoco" / "scripts"
if str(MUJOCO_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(MUJOCO_SCRIPT_DIR))

from run_block_force_demo import (  # noqa: E402
    BLOCK_FRICTION_GEOM_NAMES,
    REST_ANGULAR_THRESHOLD,
    REST_HOLD_TIME,
    REST_LINEAR_THRESHOLD,
    SCENE_PATH,
    apply_dataset_initial_pose,
    block_body_id,
    block_local_bounds,
    first_force_segment,
    reset_scene,
    set_block_freejoint_pose,
    set_split_block_friction,
    set_uniform_block_friction,
    simulate_force,
    single_force_schedule,
    trajectory_motion_metrics,
    trajectory_rows_to_matrix,
    write_batched_dataset_npz,
    write_metadata_json,
    yaw_from_quaternion_wxyz,
    z_axis_rotation_matrix,
    quaternion_wxyz_from_matrix,
)


DEFAULT_OUTPUT_ROOT = ROOT / "mujoco" / "outputs" / "clean_friction_diagnostics"
DEFAULT_EVAL_ROOT = ROOT / "report_assets" / "clean_friction_diagnostics"
EVAL_SCRIPT = ROOT / "visualization" / "evaluate_experiments.py"


@dataclass(frozen=True)
class FrictionSpec:
    name: str
    left_mu: float
    right_mu: float
    diagnostic_family: str
    expected_signal: str


@dataclass(frozen=True)
class ActionSpec:
    name: str
    family: str
    point_offset_local: tuple[float, float, float]
    force_world: tuple[float, float, float]
    duration: float
    initial_xy: tuple[float, float] = (0.58, 0.0)
    initial_yaw: float = 0.0
    segments: tuple[tuple[float, tuple[float, float, float]], ...] = ()


def quat_wxyz_from_yaw(yaw: float) -> np.ndarray:
    return quaternion_wxyz_from_matrix(z_axis_rotation_matrix(float(yaw)))


def rotate_xy(vector: tuple[float, float, float], angle: float) -> tuple[float, float, float]:
    x, y, z = vector
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    return (
        float(cosine * x - sine * y),
        float(sine * x + cosine * y),
        float(z),
    )


def expand_actions(actions: list[ActionSpec], episode_count: int | None, seed: int) -> list[ActionSpec]:
    if episode_count is None:
        return actions
    if episode_count < 0:
        raise ValueError("--episode-count must be non-negative")
    if episode_count <= len(actions):
        return actions[:episode_count]
    if not actions:
        raise ValueError("Cannot expand an empty action suite")

    rng = np.random.default_rng(seed)
    expanded: list[ActionSpec] = []
    for episode_idx in range(episode_count):
        base = actions[episode_idx % len(actions)]
        cycle_idx = episode_idx // len(actions)
        force_scale = float(rng.uniform(0.85, 1.15))
        duration_scale = float(rng.uniform(0.92, 1.04))
        force_rotation = float(rng.uniform(-0.22, 0.22))
        yaw_jitter = float(rng.uniform(-0.30, 0.30))
        xy_jitter = rng.uniform(-0.025, 0.025, size=2)
        initial_xy = (
            float(base.initial_xy[0] + xy_jitter[0]),
            float(base.initial_xy[1] + xy_jitter[1]),
        )
        segments = tuple(
            (
                float(duration * duration_scale),
                rotate_xy(
                    (
                        float(force[0]) * force_scale,
                        float(force[1]) * force_scale,
                        float(force[2]) * force_scale,
                    ),
                    force_rotation,
                ),
            )
            for duration, force in base.segments
        )
        force_world = segments[0][1] if segments else rotate_xy(
            (
                float(base.force_world[0]) * force_scale,
                float(base.force_world[1]) * force_scale,
                float(base.force_world[2]) * force_scale,
            ),
            force_rotation,
        )
        duration = float(sum(duration for duration, _ in segments)) if segments else float(base.duration * duration_scale)
        expanded.append(
            ActionSpec(
                name=f"{base.name}_rep{cycle_idx:04d}",
                family=base.family,
                point_offset_local=base.point_offset_local,
                force_world=force_world,
                duration=duration,
                initial_xy=initial_xy,
                initial_yaw=float(base.initial_yaw + yaw_jitter),
                segments=segments,
            )
        )
    return expanded


def action_suite(scale: str) -> list[ActionSpec]:
    if scale == "smoke":
        return [
            ActionSpec("center_push_x", "global_level", (0.0, 0.0, 0.0), (4.0, 0.0, 0.0), 0.12),
            ActionSpec("right_edge_tangent", "spatial", (0.085, 0.049, 0.0), (0.0, 4.0, 0.0), 0.12),
            ActionSpec("left_edge_tangent", "spatial", (-0.085, 0.049, 0.0), (0.0, -4.0, 0.0), 0.12),
            ActionSpec("right_corner_inward_tangent", "mirrored", (0.095, 0.045, 0.0), (-1.3, 3.8, 0.0), 0.14),
            ActionSpec("left_corner_inward_tangent", "mirrored", (-0.095, 0.045, 0.0), (1.3, -3.8, 0.0), 0.14),
        ]
    if scale == "rotation":
        actions: list[ActionSpec] = []
        lateral_points = [
            ("right_top", 0.085, 0.049, 1.0),
            ("left_top", -0.085, 0.049, -1.0),
            ("right_bottom", 0.085, -0.049, 1.0),
            ("left_bottom", -0.085, -0.049, -1.0),
        ]
        corner_points = [
            ("right_top_corner", 0.095, 0.045, 1.0),
            ("left_top_corner", -0.095, 0.045, -1.0),
            ("right_bottom_corner", 0.095, -0.045, 1.0),
            ("left_bottom_corner", -0.095, -0.045, -1.0),
        ]
        magnitudes = (3.4, 4.2, 5.0)
        durations = (0.10, 0.14)
        yaw_offsets = (0.0, 0.18)

        for yaw in yaw_offsets:
            yaw_tag = f"yaw_{yaw:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p")
            for magnitude in magnitudes:
                for duration in durations:
                    for name, x, y, side_sign in lateral_points:
                        tangent_y = np.sign(y) * side_sign
                        force = (0.0, float(magnitude * tangent_y), 0.0)
                        actions.append(
                            ActionSpec(
                                f"{name}_tangent_f{magnitude:.1f}_d{duration:.2f}_{yaw_tag}".replace(".", "p"),
                                "spatial",
                                (x, y, 0.0),
                                force,
                                duration,
                                initial_yaw=yaw * side_sign,
                            )
                        )

        for magnitude in (3.8, 4.6):
            for duration in (0.12, 0.16):
                for name, x, y, side_sign in corner_points:
                    tangent_y = np.sign(y) * side_sign
                    inward_x = -side_sign
                    force = (float(0.35 * magnitude * inward_x), float(magnitude * tangent_y), 0.0)
                    actions.append(
                        ActionSpec(
                            f"{name}_inward_tangent_f{magnitude:.1f}_d{duration:.2f}".replace(".", "p"),
                            "mirrored",
                            (x, y, 0.0),
                            force,
                            duration,
                        )
                    )

        for name, x, y, side_sign in corner_points:
            tangent_y = np.sign(y) * side_sign
            inward_x = -side_sign
            actions.append(
                ActionSpec(
                    f"{name}_multiseg",
                    "heldout_action",
                    (0.92 * x, 0.9 * y, 0.0),
                    (float(3.0 * inward_x), float(0.7 * tangent_y), 0.0),
                    0.18,
                    initial_yaw=0.22 * side_sign,
                    segments=(
                        (0.09, (float(3.0 * inward_x), float(0.7 * tangent_y), 0.0)),
                        (0.09, (float(-0.6 * inward_x), float(3.4 * tangent_y), 0.0)),
                    ),
                )
            )
        return actions

    if scale == "long-rotation":
        actions: list[ActionSpec] = []
        long_specs = [
            ("right_top_corner", 0.095, 0.045, 1.0, 0.18),
            ("left_top_corner", -0.095, 0.045, -1.0, -0.18),
            ("right_bottom_corner", 0.095, -0.045, 1.0, 0.26),
            ("left_bottom_corner", -0.095, -0.045, -1.0, -0.26),
            ("right_side", 0.088, 0.049, 1.0, 0.12),
            ("left_side", -0.088, 0.049, -1.0, -0.12),
        ]
        for name, x, y, side_sign, yaw in long_specs:
            tangent_y = np.sign(y) * side_sign
            inward_x = -side_sign
            for force_scale in (1.0, 1.25):
                base_tangent = 2.2 * force_scale
                base_inward = 0.7 * force_scale
                segments = (
                    (0.20, (float(base_inward * inward_x), float(base_tangent * tangent_y), 0.0)),
                    (0.18, (float(-0.35 * base_inward * inward_x), float(1.15 * base_tangent * tangent_y), 0.0)),
                    (0.22, (float(0.45 * base_inward * inward_x), float(0.85 * base_tangent * tangent_y), 0.0)),
                    (0.20, (float(-0.25 * base_inward * inward_x), float(0.75 * base_tangent * tangent_y), 0.0)),
                    (0.16, (float(0.20 * base_inward * inward_x), float(0.55 * base_tangent * tangent_y), 0.0)),
                )
                actions.append(
                    ActionSpec(
                        f"{name}_long_multiseg_s{force_scale:.2f}".replace(".", "p"),
                        "heldout_action",
                        (x, y, 0.0),
                        segments[0][1],
                        sum(duration for duration, _ in segments),
                        initial_yaw=yaw,
                        segments=segments,
                    )
            )
        return actions

    if scale == "very-long-rotation":
        actions: list[ActionSpec] = []
        very_long_specs = [
            ("left_top_corner", -0.095, 0.045, -1.0),
            ("left_bottom_corner", -0.095, -0.045, -1.0),
            ("right_top_corner", 0.095, 0.045, 1.0),
            ("right_bottom_corner", 0.095, -0.045, 1.0),
            ("left_side_top", -0.088, 0.049, -1.0),
            ("left_side_bottom", -0.088, -0.049, -1.0),
            ("right_side_top", 0.088, 0.049, 1.0),
            ("right_side_bottom", 0.088, -0.049, 1.0),
        ]
        variants = [
            (0.88, -0.24),
            (1.00, -0.12),
            (1.12, 0.10),
            (1.24, 0.24),
        ]
        for spec_idx, (name, x, y, side_sign) in enumerate(very_long_specs):
            for variant_idx, (force_scale, yaw_abs) in enumerate(variants):
                if len(actions) >= 20:
                    break
                tangent_y = np.sign(y) * side_sign
                if variant_idx % 2 == 1:
                    tangent_y *= -1.0
                inward_x = -side_sign
                yaw = float(yaw_abs * side_sign)
                base_tangent = 4.2 * force_scale
                base_inward = 1.1 * force_scale
                segments = (
                    (0.90, (float(base_inward * inward_x), float(base_tangent * tangent_y), 0.0)),
                    (0.90, (float(-0.35 * base_inward * inward_x), float(-0.82 * base_tangent * tangent_y), 0.0)),
                    (0.90, (float(0.45 * base_inward * inward_x), float(0.92 * base_tangent * tangent_y), 0.0)),
                    (0.90, (float(-0.30 * base_inward * inward_x), float(-0.76 * base_tangent * tangent_y), 0.0)),
                    (0.90, (float(0.25 * base_inward * inward_x), float(0.78 * base_tangent * tangent_y), 0.0)),
                    (0.90, (float(-0.20 * base_inward * inward_x), float(-0.66 * base_tangent * tangent_y), 0.0)),
                    (0.90, (float(0.15 * base_inward * inward_x), float(0.62 * base_tangent * tangent_y), 0.0)),
                    (0.90, (float(-0.10 * base_inward * inward_x), float(-0.54 * base_tangent * tangent_y), 0.0)),
                )
                actions.append(
                    ActionSpec(
                        f"{name}_very_long_s{force_scale:.2f}_v{variant_idx}".replace(".", "p"),
                        "heldout_action",
                        (x, y, 0.0),
                        segments[0][1],
                        sum(duration for duration, _ in segments),
                        initial_yaw=yaw,
                        segments=segments,
                    )
                )
        return actions

    if scale != "full":
        raise ValueError(f"Unknown diagnostic scale: {scale!r}")
    return [
        ActionSpec("center_push_x_low", "global_level", (0.0, 0.0, 0.0), (3.0, 0.0, 0.0), 0.12),
        ActionSpec("center_push_x_high", "global_level", (0.0, 0.0, 0.0), (4.5, 0.0, 0.0), 0.12),
        ActionSpec("center_push_y", "global_level", (0.0, 0.0, 0.0), (0.0, 3.5, 0.0), 0.12),
        ActionSpec("near_center_push_x", "global_level", (0.015, 0.0, 0.0), (4.0, 0.0, 0.0), 0.10),
        ActionSpec("right_side_tangent_a", "spatial", (0.085, 0.049, 0.0), (0.0, 4.2, 0.0), 0.13),
        ActionSpec("left_side_tangent_a", "spatial", (-0.085, 0.049, 0.0), (0.0, -4.2, 0.0), 0.13),
        ActionSpec("right_side_tangent_b", "spatial", (0.085, -0.049, 0.0), (0.0, -4.2, 0.0), 0.13),
        ActionSpec("left_side_tangent_b", "spatial", (-0.085, -0.049, 0.0), (0.0, 4.2, 0.0), 0.13),
        ActionSpec("right_corner_inward_tangent_a", "mirrored", (0.095, 0.045, 0.0), (-1.4, 4.0, 0.0), 0.15),
        ActionSpec("left_corner_inward_tangent_a", "mirrored", (-0.095, 0.045, 0.0), (1.4, -4.0, 0.0), 0.15),
        ActionSpec("right_corner_inward_tangent_b", "mirrored", (0.095, -0.045, 0.0), (-1.4, -4.0, 0.0), 0.15),
        ActionSpec("left_corner_inward_tangent_b", "mirrored", (-0.095, -0.045, 0.0), (1.4, 4.0, 0.0), 0.15),
        ActionSpec(
            "heldout_multiseg_right",
            "heldout_action",
            (0.09, 0.04, 0.0),
            (3.0, 0.7, 0.0),
            0.16,
            initial_yaw=0.20,
            segments=((0.08, (3.0, 0.7, 0.0)), (0.08, (-0.6, 3.2, 0.0))),
        ),
        ActionSpec(
            "heldout_multiseg_left",
            "heldout_action",
            (-0.09, 0.04, 0.0),
            (-3.0, -0.7, 0.0),
            0.16,
            initial_yaw=-0.20,
            segments=((0.08, (-3.0, -0.7, 0.0)), (0.08, (0.6, -3.2, 0.0))),
        ),
    ]


def friction_specs() -> list[FrictionSpec]:
    return [
        FrictionSpec("global_uniform_mu_0p20", 0.20, 0.20, "global_level", "low uniform friction"),
        FrictionSpec("global_uniform_mu_0p35", 0.35, 0.35, "global_level", "mid uniform friction"),
        FrictionSpec("global_uniform_mu_0p50", 0.50, 0.50, "global_level", "high uniform friction"),
        FrictionSpec("same_mean_uniform_0p35", 0.35, 0.35, "same_mean", "same average friction baseline"),
        FrictionSpec("same_mean_split_left_0p20_right_0p50", 0.20, 0.50, "same_mean", "left/right spatial asymmetry"),
        FrictionSpec("same_mean_split_left_0p50_right_0p20", 0.50, 0.20, "same_mean", "reversed left/right spatial asymmetry"),
    ]


def selected_actions(actions: list[ActionSpec], spec: FrictionSpec) -> list[ActionSpec]:
    if spec.diagnostic_family == "global_level":
        return [action for action in actions if action.family == "global_level"]
    if spec.diagnostic_family == "same_mean":
        return [action for action in actions if action.family in {"spatial", "mirrored", "heldout_action"}]
    return actions


def force_schedule_for_action(action: ActionSpec) -> list[dict[str, object]]:
    if not action.segments:
        return single_force_schedule(np.asarray(action.force_world, dtype=np.float64), action.duration)
    schedule = []
    for segment_index, (duration, force_world) in enumerate(action.segments):
        force = np.asarray(force_world, dtype=np.float64)
        magnitude = float(np.linalg.norm(force))
        direction = force / magnitude if magnitude > 1.0e-12 else np.zeros(3, dtype=np.float64)
        schedule.append(
            {
                "segment_index": int(segment_index),
                "duration": float(duration),
                "force_magnitude": magnitude,
                "direction_unit": direction.tolist(),
                "force_world": force.tolist(),
            }
        )
    return schedule


def make_dataset(
    *,
    output_root: Path,
    scene_path: Path,
    spec: FrictionSpec,
    actions: list[ActionSpec],
    total_duration: float,
    stop_on_rest: bool,
) -> dict[str, object]:
    output_dir = output_root / spec.name
    dataset_path = output_dir / f"{spec.name}.npz"
    metadata_path = output_dir / f"{spec.name}.json"

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    if abs(spec.left_mu - spec.right_mu) < 1.0e-12:
        set_uniform_block_friction(model, spec.left_mu)
    else:
        set_split_block_friction(model, spec.left_mu, spec.right_mu)
    data = mujoco.MjData(model)
    bounds_min, bounds_max = block_local_bounds(model)

    trajectories: list[np.ndarray] = []
    episode_metadata: list[dict[str, object]] = []
    body_id = block_body_id(model)
    selected = selected_actions(actions, spec)
    if not selected:
        raise ValueError(f"No actions selected for {spec.name}")

    for episode_id, action in enumerate(selected):
        reset_scene(model, data)
        initial_position = np.array([action.initial_xy[0], action.initial_xy[1], data.xpos[body_id][2]], dtype=np.float64)
        initial_quaternion = quat_wxyz_from_yaw(action.initial_yaw)
        set_block_freejoint_pose(model, data, initial_position, initial_quaternion)

        force_schedule = force_schedule_for_action(action)
        first_segment = first_force_segment(force_schedule)
        force = np.asarray(first_segment["force_world"], dtype=np.float64)
        point_offset = np.asarray(action.point_offset_local, dtype=np.float64)
        trajectory_rows: list[list[float]] = []
        result = simulate_force(
            model,
            data,
            force,
            point_offset,
            action.duration,
            total_duration,
            trajectory_rows=trajectory_rows,
            stop_on_rest=stop_on_rest,
            force_schedule=force_schedule,
        )
        motion_metrics = trajectory_motion_metrics(trajectory_rows)
        trajectories.append(trajectory_rows_to_matrix(trajectory_rows))
        final_yaw = yaw_from_quaternion_wxyz(result["final_block_quaternion_world"])
        episode_metadata.append(
            {
                "episode_id": int(episode_id),
                "action_name": action.name,
                "diagnostic_family": spec.diagnostic_family,
                "action_family": action.family,
                "friction_left_mu": float(spec.left_mu),
                "friction_right_mu": float(spec.right_mu),
                "friction_mean_mu": float(0.5 * (spec.left_mu + spec.right_mu)),
                "friction_delta_right_minus_left": float(spec.right_mu - spec.left_mu),
                "initial_block_position_world": initial_position.tolist(),
                "initial_block_quaternion_world": initial_quaternion.tolist(),
                "initial_contact_face_id": None,
                "initial_contact_face_name": "fixed_upright_scripted_pose",
                "initial_contact_face_normal_local": None,
                "yaw_about_world_z": float(action.initial_yaw),
                "force_magnitude": float(np.linalg.norm(force)),
                "direction_unit": (
                    (force / np.linalg.norm(force)).tolist() if np.linalg.norm(force) > 1.0e-12 else [0.0, 0.0, 0.0]
                ),
                "applied_force_world": force.tolist(),
                "force_segment_count": len(force_schedule),
                "force_schedule": result["force_schedule"],
                "point_offset_local": point_offset.tolist(),
                "force_duration": float(action.duration),
                "total_duration": float(total_duration),
                "recorded_samples": int(result["recorded_samples"]),
                "recorded_end_time": float(result["recorded_end_time"]),
                "rest_reached": bool(result["rest_reached"]),
                "final_block_position_world": result["final_block_position_world"].tolist(),
                "final_block_quaternion_world": result["final_block_quaternion_world"].tolist(),
                "final_yaw_world": float(final_yaw),
                "final_linear_speed": float(result["final_linear_speed"]),
                "final_angular_speed": float(result["final_angular_speed"]),
                "motion_filter_passed": True,
                "motion_filter_attempts": 1,
                "motion_filter_score": float(motion_metrics["max_xy_displacement"] + motion_metrics["max_rotation_angle"]),
                "video_path": None,
                **motion_metrics,
            }
        )

    summary_metadata = {
        "command": " ".join(sys.argv),
        "script_path": str(Path(__file__).resolve()),
        "scene_path": str(scene_path.resolve()),
        "mode": "clean_friction_diagnostic_dataset",
        "diagnostic_family": spec.diagnostic_family,
        "expected_signal": spec.expected_signal,
        "num_episodes": int(len(trajectories)),
        "timestep": float(model.opt.timestep),
        "total_duration": float(total_duration),
        "total_steps": int(total_duration / model.opt.timestep),
        "rest_linear_threshold": REST_LINEAR_THRESHOLD,
        "rest_angular_threshold": REST_ANGULAR_THRESHOLD,
        "rest_hold_time": REST_HOLD_TIME,
        "block_friction_override": {
            "push_block_left": [float(spec.left_mu), 0.0, 0.0],
            "push_block_right": [float(spec.right_mu), 0.0, 0.0],
        },
        "block_friction": {
            geom_name: [float(spec.left_mu if geom_name.endswith("left") else spec.right_mu), 0.0, 0.0]
            for geom_name in BLOCK_FRICTION_GEOM_NAMES
        },
        "block_local_bounds_min": bounds_min.tolist(),
        "block_local_bounds_max": bounds_max.tolist(),
        "dataset_path": str(dataset_path.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "action_names": [action.name for action in selected],
    }
    write_batched_dataset_npz(dataset_path, trajectories, episode_metadata, summary_metadata)
    write_metadata_json(metadata_path, summary_metadata)
    return {
        "name": spec.name,
        "family": spec.diagnostic_family,
        "dataset_path": str(dataset_path.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "episodes": int(len(trajectories)),
    }


def discover_checkpoint_names(outputs_root: Path) -> list[str]:
    names = []
    for child in sorted(outputs_root.iterdir() if outputs_root.exists() else []):
        if child.is_dir() and (child / f"{child.name}.npz").exists():
            names.append(child.name)
    preferred = [
        name
        for name in names
        if any(token in name for token in ("global", "left_right", "point_new"))
        and "results" not in name
    ]
    return preferred or names


def run_eval(
    *,
    experiment_name: str,
    dataset: dict[str, object],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    eval_name = str(dataset["name"])
    experiment_dir = Path(args.outputs_root) / experiment_name
    eval_dir = experiment_dir / "eval" / eval_name
    log_path = eval_dir / "eval_stdout.log"
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--dataset",
        str(dataset["dataset_path"]),
        "--eval-name",
        eval_name,
        "--method-source",
        "auto",
        "--checkpoint-root",
        str(experiment_dir),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--position-loss-weight",
        str(args.position_loss_weight),
        "--orientation-loss-weight",
        str(args.orientation_loss_weight),
        "--linear-velocity-loss-weight",
        str(args.linear_velocity_loss_weight),
        "--angular-velocity-loss-weight",
        str(args.angular_velocity_loss_weight),
        "--point-position-loss-reduction",
        str(args.point_position_loss_reduction),
        "--contact-stiffness",
        str(args.contact_stiffness),
    ]
    if args.device:
        cmd.extend(["--device", str(args.device)])
    if args.max_steps is not None:
        cmd.extend(["--max-steps", str(args.max_steps)])
    if args.max_trajectories is not None:
        cmd.append("--trajectory-indices")
        cmd.extend(str(idx) for idx in range(max(int(args.max_trajectories), 0)))
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "osmesa")
    eval_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env=env,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    base_row = {
        "dataset": str(dataset["name"]),
        "dataset_family": str(dataset["family"]),
        "experiment_name": experiment_name,
        "returncode": int(completed.returncode),
        "eval_dir": str(eval_dir.resolve()),
        "log_path": str(log_path.resolve()),
    }
    if completed.returncode != 0:
        row = dict(base_row)
        row["error"] = "eval failed"
        return [row]

    rows = []
    for summary_path in sorted(eval_dir.glob("*_eval_summary.json")):
        row = dict(base_row)
        row["summary_path"] = str(summary_path.resolve())
        metrics = json.loads(summary_path.read_text(encoding="utf-8"))
        method = metrics.get("method", {})
        row["method"] = method.get("name")
        row["checkpoint"] = method.get("checkpoint")
        for key in (
            "checkpoint_type",
            "friction_parameterization",
            "parameterization",
            "overlay_loss_mean",
            "overlay_loss_min",
            "overlay_loss_max",
            "mu_mean",
            "mu_std",
            "mu_left_mean",
            "mu_right_mean",
            "mu_global_param",
            "mu_left_param",
            "mu_right_param",
            "mu_left",
            "mu_right",
        ):
            if key in metrics:
                row[key] = metrics[key]
            elif key in method:
                row[key] = method[key]
        rows.append(row)
    if not rows:
        row = dict(base_row)
        row["error"] = "eval produced no summary"
        rows.append(row)
    return rows


def write_eval_report(rows: list[dict[str, object]], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "clean_friction_diagnostic_eval_summary.json"
    csv_path = report_dir / "clean_friction_diagnostic_eval_summary.csv"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[report] json={json_path.resolve()}")
    print(f"[report] csv={csv_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--scene", type=Path, default=SCENE_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--outputs-root", type=Path, default=ROOT / "outputs")
    parser.add_argument(
        "--scale",
        choices=("smoke", "full", "rotation", "long-rotation", "very-long-rotation"),
        default="full",
    )
    parser.add_argument(
        "--specs",
        nargs="*",
        default=None,
        help="Optional friction spec names to generate/evaluate. Defaults to all specs.",
    )
    parser.add_argument(
        "--episode-count",
        type=int,
        default=None,
        help="Expand/truncate the selected action suite to this many deterministic action variants.",
    )
    parser.add_argument(
        "--episode-seed",
        type=int,
        default=0,
        help="Seed for deterministic action expansion when --episode-count is used.",
    )
    parser.add_argument("--total-duration", type=float, default=3.0)
    parser.add_argument("--stop-on-rest", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--experiments", nargs="*", default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=20)
    parser.add_argument("--position-loss-weight", type=float, default=1.0)
    parser.add_argument("--orientation-loss-weight", type=float, default=1.0)
    parser.add_argument("--linear-velocity-loss-weight", type=float, default=0.0)
    parser.add_argument("--angular-velocity-loss-weight", type=float, default=0.1)
    parser.add_argument("--point-position-loss-reduction", choices=("sum", "mean"), default="mean")
    parser.add_argument("--contact-stiffness", type=float, default=1.0e5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    actions = expand_actions(action_suite(args.scale), args.episode_count, int(args.episode_seed))
    specs = friction_specs()
    if args.specs is not None:
        requested = set(str(item) for item in args.specs)
        specs = [spec for spec in specs if spec.name in requested]
        missing = sorted(requested - {spec.name for spec in specs})
        if missing:
            raise ValueError(f"Unknown friction spec(s): {missing}")
    datasets = []
    if args.skip_generate:
        for spec in specs:
            dataset_path = Path(args.output_root) / spec.name / f"{spec.name}.npz"
            datasets.append(
                {
                    "name": spec.name,
                    "family": spec.diagnostic_family,
                    "dataset_path": str(dataset_path.resolve()),
                    "metadata_path": str((dataset_path.parent / f"{spec.name}.json").resolve()),
                    "episodes": None,
                }
            )
    else:
        for spec in specs:
            dataset = make_dataset(
                output_root=Path(args.output_root),
                scene_path=Path(args.scene),
                spec=spec,
                actions=actions,
                total_duration=float(args.total_duration),
                stop_on_rest=bool(args.stop_on_rest),
            )
            datasets.append(dataset)
            print(f"[dataset] {dataset['name']} episodes={dataset['episodes']} path={dataset['dataset_path']}")

    manifest_path = Path(args.output_root) / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"datasets": datasets}, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[manifest] {manifest_path.resolve()}")

    if args.skip_eval:
        return
    experiments = args.experiments if args.experiments is not None else discover_checkpoint_names(Path(args.outputs_root))
    if not experiments:
        print("[eval] no checkpoints discovered; skipping")
        return
    rows = []
    for dataset in datasets:
        for experiment_name in experiments:
            print(f"[eval] experiment={experiment_name} dataset={dataset['name']}", flush=True)
            rows.extend(run_eval(experiment_name=experiment_name, dataset=dataset, args=args))
    write_eval_report(rows, Path(args.eval_root))


if __name__ == "__main__":
    main()
