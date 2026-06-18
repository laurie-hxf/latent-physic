from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MUJOCO_SCRIPT_DIR = REPO_ROOT / "mujoco" / "scripts"
for path in (REPO_ROOT, MUJOCO_SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from object_physics_latent.dataset import build_manifest, validate_manifest  # noqa: E402
from run_block_force_demo import (  # noqa: E402
    REST_ANGULAR_THRESHOLD,
    REST_HOLD_TIME,
    REST_LINEAR_THRESHOLD,
    SCENE_PATH,
    block_application_point,
    block_body_id,
    block_local_bounds,
    block_speed_norms,
    first_force_segment,
    normalize_force_schedule_segments,
    record_block_state,
    reset_scene,
    set_block_freejoint_pose,
    simulate_force,
    single_force_schedule,
    trajectory_motion_metrics,
    trajectory_rows_to_matrix,
    write_batched_dataset_npz,
    write_metadata_json,
    yaw_from_quaternion_wxyz,
)
from run_clean_friction_diagnostics import (  # noqa: E402
    ActionSpec,
    action_suite,
    expand_actions,
    force_schedule_for_action,
    quat_wxyz_from_yaw,
)


FRONT_BACK_SCENE_PATH = REPO_ROOT / "mujoco" / "scenes" / "block_force_front_back_scene.xml"
CENTER_ENDS_SCENE_PATH = REPO_ROOT / "mujoco" / "scenes" / "block_force_center_ends_scene.xml"


@dataclass(frozen=True)
class PartitionFamily:
    name: str
    scene_path: str
    region_a_name: str
    region_b_name: str
    region_a_geoms: tuple[str, ...]
    region_b_geoms: tuple[str, ...]
    description: str

    @property
    def all_geoms(self) -> tuple[str, ...]:
        return self.region_a_geoms + self.region_b_geoms


@dataclass(frozen=True)
class ObjectFrictionSpec:
    name: str
    family: PartitionFamily
    region_a_mu: float
    region_b_mu: float

    @property
    def region_values(self) -> dict[str, float]:
        return {
            self.family.region_a_name: float(self.region_a_mu),
            self.family.region_b_name: float(self.region_b_mu),
        }


PARTITION_FAMILIES = (
    PartitionFamily(
        name="left_right",
        scene_path=str(SCENE_PATH.resolve()),
        region_a_name="left",
        region_b_name="right",
        region_a_geoms=("push_block_left",),
        region_b_geoms=("push_block_right",),
        description="split at local x=0 into two 0.10 x 0.10 squares",
    ),
    PartitionFamily(
        name="front_back",
        scene_path=str(FRONT_BACK_SCENE_PATH.resolve()),
        region_a_name="back",
        region_b_name="front",
        region_a_geoms=("push_block_back",),
        region_b_geoms=("push_block_front",),
        description="split at local y=0 into two 0.20 x 0.05 narrow rectangles",
    ),
    PartitionFamily(
        name="center_ends",
        scene_path=str(CENTER_ENDS_SCENE_PATH.resolve()),
        region_a_name="ends",
        region_b_name="center",
        region_a_geoms=("push_block_left_end", "push_block_right_end"),
        region_b_geoms=("push_block_center",),
        description="two 0.05-wide x ends share friction; 0.10-wide center uses another friction",
    ),
)


def friction_tag(value: float) -> str:
    return f"{float(value):.4f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def sample_friction_pairs(
    *,
    count: int,
    seed: int,
    minimum: float,
    maximum: float,
    minimum_difference: float,
) -> list[tuple[float, float]]:
    if count <= 0:
        return []
    rng = np.random.default_rng(int(seed))
    pairs: list[tuple[float, float]] = []
    attempts = 0
    while len(pairs) < int(count):
        attempts += 1
        if attempts > int(count) * 1000:
            raise RuntimeError("Could not sample enough distinct friction pairs")
        first = float(rng.uniform(minimum, maximum))
        second = float(rng.uniform(minimum, maximum))
        if abs(first - second) < float(minimum_difference):
            continue
        rounded = (round(first, 6), round(second, 6))
        if rounded in {(round(a, 6), round(b, 6)) for a, b in pairs}:
            continue
        pairs.append((first, second))
    return pairs


def sample_object_specs(
    *,
    count: int,
    seed: int,
    minimum: float,
    maximum: float,
    minimum_difference: float,
) -> list[ObjectFrictionSpec]:
    if count < len(PARTITION_FAMILIES):
        raise ValueError(f"--num-objects must be at least {len(PARTITION_FAMILIES)}")
    family_counts = [count // len(PARTITION_FAMILIES)] * len(PARTITION_FAMILIES)
    for index in range(count % len(PARTITION_FAMILIES)):
        family_counts[index] += 1

    specs: list[ObjectFrictionSpec] = []
    object_index = 0
    for family_index, (family, family_count) in enumerate(zip(PARTITION_FAMILIES, family_counts, strict=True)):
        pairs = sample_friction_pairs(
            count=family_count,
            seed=int(seed) + 1009 * family_index,
            minimum=float(minimum),
            maximum=float(maximum),
            minimum_difference=float(minimum_difference),
        )
        for region_a_mu, region_b_mu in pairs:
            name = (
                f"object_{object_index:04d}_{family.name}"
                f"_{family.region_a_name}_{friction_tag(region_a_mu)}"
                f"_{family.region_b_name}_{friction_tag(region_b_mu)}"
            )
            specs.append(
                ObjectFrictionSpec(
                    name=name,
                    family=family,
                    region_a_mu=float(region_a_mu),
                    region_b_mu=float(region_b_mu),
                )
            )
            object_index += 1
    return specs


def set_partition_friction(model: mujoco.MjModel, spec: ObjectFrictionSpec) -> dict[str, list[float]]:
    friction_by_geom = {
        **{name: float(spec.region_a_mu) for name in spec.family.region_a_geoms},
        **{name: float(spec.region_b_mu) for name in spec.family.region_b_geoms},
    }
    result: dict[str, list[float]] = {}
    for geom_name, mu in friction_by_geom.items():
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id < 0:
            raise ValueError(f"Scene {spec.family.scene_path} is missing geom {geom_name!r}")
        friction = [float(mu), 0.0, 0.0]
        model.geom_friction[geom_id, :] = np.asarray(friction, dtype=np.float64)
        result[geom_name] = friction
    return result


def simulate_force_with_minimum_recorded_steps(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    force: np.ndarray,
    point_offset: np.ndarray,
    force_duration: float,
    total_duration: float,
    *,
    trajectory_rows: list[list[float]],
    stop_on_rest: bool,
    force_schedule: list[dict[str, object]] | None,
    minimum_recorded_steps: int,
) -> dict[str, object]:
    """Run the shared block-force rollout while delaying rest termination.

    The common simulator stops as soon as the block has remained at rest for
    REST_HOLD_TIME. This dataset-specific variant keeps recording until at
    least minimum_recorded_steps transitions are available, so a fixed-length
    Newton/query window can always be sampled without changing the shared
    MuJoCo generator.
    """

    body_id = block_body_id(model)
    total_steps = int(total_duration / model.opt.timestep)
    required_steps = min(max(int(minimum_recorded_steps), 0), total_steps)
    if force_schedule is None:
        force_schedule = single_force_schedule(force, force_duration)
    normalized_schedule, step_forces, force_steps = normalize_force_schedule_segments(
        force_schedule,
        float(model.opt.timestep),
        total_duration,
    )
    rest_steps = max(1, int(REST_HOLD_TIME / model.opt.timestep))
    rest_counter = 0
    rest_reached = False

    record_block_state(model, data, np.zeros(3, dtype=float), point_offset, trajectory_rows)
    for step in range(total_steps):
        data.ctrl[:] = 0.0
        data.xfrc_applied[body_id] = 0.0
        applied_force = np.zeros(3, dtype=float)
        if step < force_steps:
            applied_force = step_forces[step].copy()
        if np.linalg.norm(applied_force) > 1.0e-12:
            point = block_application_point(model, data, point_offset)
            qfrc = np.zeros(model.nv, dtype=float)
            mujoco.mj_applyFT(
                model,
                data,
                applied_force,
                np.zeros(3, dtype=float),
                point,
                body_id,
                qfrc,
            )
            data.qfrc_applied[:] = qfrc
        else:
            data.qfrc_applied[:] = 0.0

        mujoco.mj_step(model, data)
        record_block_state(model, data, applied_force, point_offset, trajectory_rows)
        if step >= force_steps:
            linear_speed, angular_speed = block_speed_norms(model, data)
            if linear_speed < REST_LINEAR_THRESHOLD and angular_speed < REST_ANGULAR_THRESHOLD:
                rest_counter += 1
                if rest_counter >= rest_steps and step + 1 >= required_steps:
                    rest_reached = True
                    if stop_on_rest:
                        break
            else:
                rest_counter = 0

    data.xfrc_applied[body_id] = 0.0
    data.qfrc_applied[:] = 0.0
    final_linear_speed, final_angular_speed = block_speed_norms(model, data)
    return {
        "rest_reached": rest_reached,
        "recorded_samples": len(trajectory_rows),
        "recorded_end_time": float(trajectory_rows[-1][0]) if trajectory_rows else float(data.time),
        "final_block_position_world": data.xpos[body_id].copy(),
        "final_block_quaternion_world": data.xquat[body_id].copy(),
        "final_linear_speed": final_linear_speed,
        "final_angular_speed": final_angular_speed,
        "force_schedule": normalized_schedule,
        "force_steps": force_steps,
    }


def make_partition_dataset(
    *,
    output_root: Path,
    spec: ObjectFrictionSpec,
    actions: list[ActionSpec],
    total_duration: float,
    stop_on_rest: bool,
    minimum_recorded_steps: int,
) -> dict[str, object]:
    output_dir = Path(output_root) / spec.name
    dataset_path = output_dir / f"{spec.name}.npz"
    metadata_path = output_dir / f"{spec.name}.json"
    scene_path = Path(spec.family.scene_path)

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    block_friction = set_partition_friction(model, spec)
    data = mujoco.MjData(model)
    bounds_min, bounds_max = block_local_bounds(model)
    body_id = block_body_id(model)

    trajectories: list[np.ndarray] = []
    episode_metadata: list[dict[str, object]] = []
    for episode_id, action in enumerate(actions):
        reset_scene(model, data)
        initial_position = np.array([action.initial_xy[0], action.initial_xy[1], data.xpos[body_id][2]], dtype=np.float64)
        initial_quaternion = quat_wxyz_from_yaw(action.initial_yaw)
        set_block_freejoint_pose(model, data, initial_position, initial_quaternion)

        force_schedule = force_schedule_for_action(action)
        first_segment = first_force_segment(force_schedule)
        force = np.asarray(first_segment["force_world"], dtype=np.float64)
        point_offset = np.asarray(action.point_offset_local, dtype=np.float64)
        trajectory_rows: list[list[float]] = []
        if int(minimum_recorded_steps) > 0:
            result = simulate_force_with_minimum_recorded_steps(
                model,
                data,
                force,
                point_offset,
                action.duration,
                total_duration,
                trajectory_rows=trajectory_rows,
                stop_on_rest=stop_on_rest,
                force_schedule=force_schedule,
                minimum_recorded_steps=int(minimum_recorded_steps),
            )
        else:
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
        episode_metadata.append(
            {
                "episode_id": int(episode_id),
                "action_name": action.name,
                "action_family": action.family,
                "friction_partition_family": spec.family.name,
                "friction_region_values": spec.region_values,
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
                "final_yaw_world": float(yaw_from_quaternion_wxyz(result["final_block_quaternion_world"])),
                "final_linear_speed": float(result["final_linear_speed"]),
                "final_angular_speed": float(result["final_angular_speed"]),
                "motion_filter_passed": True,
                "motion_filter_attempts": 1,
                "motion_filter_score": float(motion_metrics["max_xy_displacement"] + motion_metrics["max_rotation_angle"]),
                "video_path": None,
                **motion_metrics,
            }
        )

    body_mass = float(model.body_mass[body_id])
    body_inertia = model.body_inertia[body_id].astype(float).tolist()
    summary_metadata = {
        "script_path": str(Path(__file__).resolve()),
        "scene_path": str(scene_path.resolve()),
        "mode": "object_physics_latent_partition_dataset",
        "friction_partition_family": spec.family.name,
        "friction_partition_description": spec.family.description,
        "friction_region_values": spec.region_values,
        "num_episodes": int(len(trajectories)),
        "timestep": float(model.opt.timestep),
        "total_duration": float(total_duration),
        "total_steps": int(total_duration / model.opt.timestep),
        "minimum_recorded_steps": int(minimum_recorded_steps),
        "minimum_recorded_duration": float(int(minimum_recorded_steps) * model.opt.timestep),
        "rest_linear_threshold": REST_LINEAR_THRESHOLD,
        "rest_angular_threshold": REST_ANGULAR_THRESHOLD,
        "rest_hold_time": REST_HOLD_TIME,
        "block_friction_override": block_friction,
        "block_friction": block_friction,
        "block_local_bounds_min": bounds_min.tolist(),
        "block_local_bounds_max": bounds_max.tolist(),
        "block_mass": body_mass,
        "block_inertia": body_inertia,
        "dataset_path": str(dataset_path.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "action_names": [action.name for action in actions],
    }
    write_batched_dataset_npz(dataset_path, trajectories, episode_metadata, summary_metadata)
    write_metadata_json(metadata_path, summary_metadata)
    return {
        "name": spec.name,
        "family": spec.family.name,
        "dataset_path": str(dataset_path.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "episodes": int(len(trajectories)),
    }


def _generate_one(
    output_root: str,
    spec: ObjectFrictionSpec,
    actions: list[ActionSpec],
    total_duration: float,
    stop_on_rest: bool,
    minimum_recorded_steps: int,
    skip_existing: bool,
) -> dict[str, object]:
    dataset_path = Path(output_root) / spec.name / f"{spec.name}.npz"
    if skip_existing and dataset_path.is_file():
        return {
            "name": spec.name,
            "family": spec.family.name,
            "dataset_path": str(dataset_path.resolve()),
            "metadata_path": str((dataset_path.parent / f"{spec.name}.json").resolve()),
            "episodes": None,
            "reused": True,
        }
    result = make_partition_dataset(
        output_root=Path(output_root),
        spec=spec,
        actions=actions,
        total_duration=float(total_duration),
        stop_on_rest=bool(stop_on_rest),
        minimum_recorded_steps=int(minimum_recorded_steps),
    )
    result["reused"] = False
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate same-shape box objects with left-right, front-back, and center-ends friction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "mujoco" / "outputs" / "object_physics_latent_box_partitions_48x2000_min300",
    )
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--num-objects", type=int, default=48)
    parser.add_argument("--episodes-per-object", type=int, default=2000)
    parser.add_argument("--friction-min", type=float, default=0.10)
    parser.add_argument("--friction-max", type=float, default=0.70)
    parser.add_argument("--minimum-friction-difference", type=float, default=0.08)
    parser.add_argument(
        "--action-scale",
        choices=("smoke", "full", "rotation", "long-rotation", "very-long-rotation"),
        default="rotation",
    )
    parser.add_argument("--total-duration", type=float, default=3.0)
    parser.add_argument(
        "--minimum-recorded-steps",
        type=int,
        default=0,
        help="Delay rest termination until at least this many simulation transitions have been recorded.",
    )
    parser.add_argument("--stop-on-rest", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    manifest_output = (
        Path(args.manifest_output).resolve()
        if args.manifest_output is not None
        else output_root / "manifest.json"
    )
    specs = sample_object_specs(
        count=int(args.num_objects),
        seed=int(args.seed),
        minimum=float(args.friction_min),
        maximum=float(args.friction_max),
        minimum_difference=float(args.minimum_friction_difference),
    )
    actions = expand_actions(action_suite(str(args.action_scale)), int(args.episodes_per_object), int(args.seed))
    plan = {
        "schema_version": 2,
        "output_root": str(output_root),
        "num_objects": len(specs),
        "episodes_per_object": len(actions),
        "total_episodes": len(specs) * len(actions),
        "seed": int(args.seed),
        "workers": int(args.workers),
        "action_scale": str(args.action_scale),
        "total_duration": float(args.total_duration),
        "minimum_recorded_steps": int(args.minimum_recorded_steps),
        "stop_on_rest": bool(args.stop_on_rest),
        "friction_min": float(args.friction_min),
        "friction_max": float(args.friction_max),
        "minimum_friction_difference": float(args.minimum_friction_difference),
        "partition_families": [asdict(family) for family in PARTITION_FAMILIES],
        "object_specs": [
            {
                "name": spec.name,
                "family": spec.family.name,
                "scene_path": spec.family.scene_path,
                "region_values": spec.region_values,
            }
            for spec in specs
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    plan_path = output_root / "generation_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[plan] {plan_path}", flush=True)
    if args.plan_only:
        return

    results: list[dict[str, object]] = []
    worker_count = max(1, min(int(args.workers), len(specs)))
    if worker_count == 1:
        for index, spec in enumerate(specs):
            result = _generate_one(
                str(output_root),
                spec,
                actions,
                float(args.total_duration),
                bool(args.stop_on_rest),
                int(args.minimum_recorded_steps),
                bool(args.skip_existing),
            )
            results.append(result)
            print(f"[{index + 1}/{len(specs)}] {result['family']} {result['dataset_path']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _generate_one,
                    str(output_root),
                    spec,
                    actions,
                    float(args.total_duration),
                    bool(args.stop_on_rest),
                    int(args.minimum_recorded_steps),
                    bool(args.skip_existing),
                ): spec
                for spec in specs
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                print(f"[{completed}/{len(specs)}] {result['family']} {result['dataset_path']}", flush=True)

    datasets = [Path(str(result["dataset_path"])) for result in sorted(results, key=lambda item: str(item["name"]))]
    build_manifest(
        datasets,
        output_path=manifest_output,
        seed=int(args.seed),
        episode_split_fractions=(0.15, 0.75, 0.10),
        object_split_fractions=(0.70, 0.15, 0.15),
        min_episode_steps=1,
    )
    print(json.dumps(validate_manifest(manifest_output, inspect_datasets=True), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
