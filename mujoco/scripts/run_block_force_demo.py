from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

try:
    import imageio.v3 as iio
except ModuleNotFoundError:
    iio = None


ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = ROOT / "third_party" / "mujoco_menagerie" / "franka_emika_panda" / "block_force_scene.xml"

BLOCK_BODY_NAME = "push_block"
FLOOR_GEOM_NAME = "floor"
REST_LINEAR_THRESHOLD = 1e-3
REST_ANGULAR_THRESHOLD = 1e-3
REST_HOLD_TIME = 0.2
SURFACE_EPS = 1e-6
DEFAULT_VIDEO_FRAME_STRIDE = 3
DEFAULT_INIT_X_RANGE = (0.45, 0.70)
DEFAULT_INIT_Y_RANGE = (-0.12, 0.12)
DEFAULT_MIN_SLIDING_DISTANCE = 0.02
DEFAULT_MIN_ROTATION_ANGLE = 0.15
DEFAULT_MAX_RESAMPLE_ATTEMPTS = 50
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "block_force_demo"
BLOCK_FRICTION_GEOM_NAMES = ("push_block_left", "push_block_right")
CONTACT_FACE_NORMALS = (
    ("x_neg", np.array([-1.0, 0.0, 0.0], dtype=np.float64)),
    ("x_pos", np.array([1.0, 0.0, 0.0], dtype=np.float64)),
    ("y_neg", np.array([0.0, -1.0, 0.0], dtype=np.float64)),
    ("y_pos", np.array([0.0, 1.0, 0.0], dtype=np.float64)),
    ("z_neg", np.array([0.0, 0.0, -1.0], dtype=np.float64)),
    ("z_pos", np.array([0.0, 0.0, 1.0], dtype=np.float64)),
)
TRAJECTORY_COLUMNS = [
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


def block_body_id(model: mujoco.MjModel) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BLOCK_BODY_NAME)


def floor_z(model: mujoco.MjModel) -> float:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM_NAME)
    if geom_id < 0:
        return 0.0
    return float(model.geom_pos[geom_id][2])


def block_freejoint_addresses(model: mujoco.MjModel) -> tuple[int, int]:
    body_id = block_body_id(model)
    joint_start = int(model.body_jntadr[body_id])
    joint_count = int(model.body_jntnum[body_id])
    for joint_id in range(joint_start, joint_start + joint_count):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])
    raise ValueError(f"Body '{BLOCK_BODY_NAME}' does not have a freejoint.")


def body_position(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return data.xpos[body_id].copy()


def reset_scene(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("Cannot normalize a near-zero vector.")
    return np.asarray(vector, dtype=np.float64) / norm


def skew_matrix(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )


def rotation_matrix_from_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_unit = normalize_vector(source)
    target_unit = normalize_vector(target)
    dot = float(np.clip(np.dot(source_unit, target_unit), -1.0, 1.0))
    if dot > 1.0 - 1e-10:
        return np.eye(3, dtype=np.float64)
    if dot < -1.0 + 1e-10:
        axis = np.cross(source_unit, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        if np.linalg.norm(axis) < 1e-10:
            axis = np.cross(source_unit, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        axis = normalize_vector(axis)
        axis_cross = skew_matrix(axis)
        return np.eye(3, dtype=np.float64) + 2.0 * (axis_cross @ axis_cross)

    axis_cross = skew_matrix(np.cross(source_unit, target_unit))
    return np.eye(3, dtype=np.float64) + axis_cross + axis_cross @ axis_cross * (1.0 / (1.0 + dot))


def z_axis_rotation_matrix(angle: float) -> np.ndarray:
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    return np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def quaternion_wxyz_from_matrix(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        diagonal_index = int(np.argmax(np.diag(matrix)))
        if diagonal_index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        elif diagonal_index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=np.float64,
            )
    return normalize_vector(quat)


def block_local_corners(bounds_min: np.ndarray, bounds_max: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [x, y, z]
            for x in (bounds_min[0], bounds_max[0])
            for y in (bounds_min[1], bounds_max[1])
            for z in (bounds_min[2], bounds_max[2])
        ],
        dtype=np.float64,
    )


def set_block_freejoint_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    position: np.ndarray,
    quaternion_wxyz: np.ndarray,
) -> None:
    qpos_address, dof_address = block_freejoint_addresses(model)
    data.qpos[qpos_address : qpos_address + 3] = np.asarray(position, dtype=np.float64)
    data.qpos[qpos_address + 3 : qpos_address + 7] = normalize_vector(np.asarray(quaternion_wxyz, dtype=np.float64))
    data.qvel[dof_address : dof_address + 6] = 0.0
    data.qacc[dof_address : dof_address + 6] = 0.0
    data.qfrc_applied[:] = 0.0
    data.xfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)


def sample_contact_face_initial_pose(
    rng: np.random.Generator,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    floor_height: float,
    clearance: float,
) -> dict[str, object]:
    face_id = int(rng.integers(0, len(CONTACT_FACE_NORMALS)))
    face_name, face_normal = CONTACT_FACE_NORMALS[face_id]
    face_to_down = rotation_matrix_from_vectors(face_normal, np.array([0.0, 0.0, -1.0], dtype=np.float64))
    yaw = float(rng.uniform(0.0, 2.0 * np.pi))
    rotation = z_axis_rotation_matrix(yaw) @ face_to_down
    quaternion_wxyz = quaternion_wxyz_from_matrix(rotation)

    corners = block_local_corners(bounds_min, bounds_max)
    min_corner_z = float(np.min(corners @ rotation.T[:, 2]))
    position = np.array(
        [
            float(rng.uniform(x_range[0], x_range[1])),
            float(rng.uniform(y_range[0], y_range[1])),
            floor_height + float(clearance) - min_corner_z,
        ],
        dtype=np.float64,
    )
    return {
        "position": position,
        "quaternion_wxyz": quaternion_wxyz,
        "contact_face_id": face_id,
        "contact_face_name": face_name,
        "contact_face_normal_local": face_normal.copy(),
        "yaw_about_world_z": yaw,
    }


def apply_dataset_initial_pose(
    args: argparse.Namespace,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rng: np.random.Generator,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> dict[str, object]:
    body_id = block_body_id(model)
    if not args.randomize_initial_pose:
        mujoco.mj_forward(model, data)
        return {
            "position": data.xpos[body_id].copy(),
            "quaternion_wxyz": data.xquat[body_id].copy(),
            "contact_face_id": None,
            "contact_face_name": "fixed_xml_pose",
            "contact_face_normal_local": None,
            "yaw_about_world_z": None,
        }

    pose = sample_contact_face_initial_pose(
        rng,
        bounds_min,
        bounds_max,
        (float(args.init_x_min), float(args.init_x_max)),
        (float(args.init_y_min), float(args.init_y_max)),
        floor_z(model),
        float(args.init_clearance),
    )
    set_block_freejoint_pose(model, data, pose["position"], pose["quaternion_wxyz"])
    return pose


def capture_frame(
    data: mujoco.MjData,
    frames: list[np.ndarray] | None,
    renderer: mujoco.Renderer | None,
    frame_index: int,
    frame_stride: int = DEFAULT_VIDEO_FRAME_STRIDE,
) -> None:
    if frames is None or renderer is None:
        return
    if frame_index % max(int(frame_stride), 1) != 0:
        return
    renderer.update_scene(data)
    frames.append(renderer.render().copy())


def write_video(video_path: Path, frames: list[np.ndarray], fps: float) -> None:
    if not frames:
        return
    video_path.parent.mkdir(parents=True, exist_ok=True)
    if iio is not None:
        iio.imwrite(video_path, np.stack(frames), fps=fps)
        return

    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("Video export requires either imageio or opencv-python.") from exc

    first_frame = np.asarray(frames[0])
    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {video_path}.")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def block_application_point(model: mujoco.MjModel, data: mujoco.MjData, offset: np.ndarray) -> np.ndarray:
    body_id = block_body_id(model)
    rotation = data.xmat[body_id].reshape(3, 3)
    return data.xpos[body_id] + rotation @ offset


def record_block_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    force: np.ndarray,
    point_offset: np.ndarray,
    rows: list[list[float]] | None,
) -> None:
    if rows is None:
        return
    body_id = block_body_id(model)
    application_point = block_application_point(model, data, point_offset)
    rows.append(
        [
            data.time,
            data.xpos[body_id][0],
            data.xpos[body_id][1],
            data.xpos[body_id][2],
            data.xquat[body_id][0],
            data.xquat[body_id][1],
            data.xquat[body_id][2],
            data.xquat[body_id][3],
            data.cvel[body_id][3],
            data.cvel[body_id][4],
            data.cvel[body_id][5],
            data.cvel[body_id][0],
            data.cvel[body_id][1],
            data.cvel[body_id][2],
            force[0],
            force[1],
            force[2],
            application_point[0],
            application_point[1],
            application_point[2],
        ]
    )


def block_speed_norms(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, float]:
    body_id = block_body_id(model)
    linear_speed = float(np.linalg.norm(data.cvel[body_id][3:]))
    angular_speed = float(np.linalg.norm(data.cvel[body_id][:3]))
    return linear_speed, angular_speed


def write_trajectory_csv(csv_path: Path, rows: list[list[float]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(TRAJECTORY_COLUMNS)
        writer.writerows(rows)


def trajectory_rows_to_matrix(rows: list[list[float]]) -> np.ndarray:
    if not rows:
        return np.zeros((0, len(TRAJECTORY_COLUMNS)), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64).reshape(-1, len(TRAJECTORY_COLUMNS))


def quaternion_angle_distance(q0: np.ndarray, q1: np.ndarray) -> float:
    q0_norm = normalize_vector(np.asarray(q0, dtype=np.float64))
    q1_norm = normalize_vector(np.asarray(q1, dtype=np.float64))
    dot = abs(float(np.dot(q0_norm, q1_norm)))
    return float(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


def yaw_from_quaternion_wxyz(quaternion: np.ndarray) -> float:
    w, x, y, z = normalize_vector(np.asarray(quaternion, dtype=np.float64))
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def wrapped_angle_delta(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def final_pose_comparison_metrics(
    source_position: np.ndarray,
    source_quaternion_wxyz: np.ndarray,
    compared_position: np.ndarray,
    compared_quaternion_wxyz: np.ndarray,
) -> dict[str, float]:
    source_position = np.asarray(source_position, dtype=np.float64)
    compared_position = np.asarray(compared_position, dtype=np.float64)
    source_yaw = yaw_from_quaternion_wxyz(source_quaternion_wxyz)
    compared_yaw = yaw_from_quaternion_wxyz(compared_quaternion_wxyz)
    yaw_delta = wrapped_angle_delta(compared_yaw - source_yaw)
    position_delta = compared_position - source_position
    return {
        "final_position_delta_norm": float(np.linalg.norm(position_delta)),
        "final_xy_position_delta_norm": float(np.linalg.norm(position_delta[:2])),
        "final_position_delta_x": float(position_delta[0]),
        "final_position_delta_y": float(position_delta[1]),
        "final_position_delta_z": float(position_delta[2]),
        "final_yaw_source": float(source_yaw),
        "final_yaw_compared": float(compared_yaw),
        "final_yaw_delta": float(yaw_delta),
        "final_yaw_delta_abs": float(abs(yaw_delta)),
    }


def trajectory_motion_metrics(rows: list[list[float]]) -> dict[str, float]:
    matrix = trajectory_rows_to_matrix(rows)
    if matrix.shape[0] == 0:
        return {
            "net_xy_displacement": 0.0,
            "max_xy_displacement": 0.0,
            "xy_path_length": 0.0,
            "final_rotation_angle": 0.0,
            "max_rotation_angle": 0.0,
        }

    positions_xy = matrix[:, 1:3]
    xy_offsets = positions_xy - positions_xy[0]
    xy_displacements = np.linalg.norm(xy_offsets, axis=1)
    xy_steps = np.diff(positions_xy, axis=0)
    xy_path_length = float(np.linalg.norm(xy_steps, axis=1).sum()) if xy_steps.size else 0.0

    quaternions = matrix[:, 4:8]
    q0 = quaternions[0]
    rotation_angles = np.asarray([quaternion_angle_distance(q0, quat) for quat in quaternions], dtype=np.float64)

    return {
        "net_xy_displacement": float(np.linalg.norm(positions_xy[-1] - positions_xy[0])),
        "max_xy_displacement": float(xy_displacements.max(initial=0.0)),
        "xy_path_length": xy_path_length,
        "final_rotation_angle": float(rotation_angles[-1]),
        "max_rotation_angle": float(rotation_angles.max(initial=0.0)),
    }


def passes_motion_filter(
    metrics: dict[str, float],
    min_sliding_distance: float,
    min_rotation_angle: float,
) -> bool:
    if min_sliding_distance <= 0.0 and min_rotation_angle <= 0.0:
        return True
    sliding_ok = min_sliding_distance > 0.0 and metrics["max_xy_displacement"] >= min_sliding_distance
    rotation_ok = min_rotation_angle > 0.0 and metrics["max_rotation_angle"] >= min_rotation_angle
    return bool(sliding_ok or rotation_ok)


def motion_filter_score(
    metrics: dict[str, float],
    min_sliding_distance: float,
    min_rotation_angle: float,
) -> float:
    scores: list[float] = []
    if min_sliding_distance > 0.0:
        scores.append(metrics["max_xy_displacement"] / min_sliding_distance)
    if min_rotation_angle > 0.0:
        scores.append(metrics["max_rotation_angle"] / min_rotation_angle)
    return float(max(scores, default=float("inf")))


def write_trajectory_npz(npz_path: Path, rows: list[list[float]], metadata: dict[str, object]) -> None:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = trajectory_rows_to_matrix(rows)
    np.savez_compressed(
        npz_path,
        columns=np.asarray(TRAJECTORY_COLUMNS),
        trajectory=matrix,
        time=matrix[:, 0],
        position=matrix[:, 1:4],
        quaternion=matrix[:, 4:8],
        linear_velocity=matrix[:, 8:11],
        angular_velocity=matrix[:, 11:14],
        applied_force=matrix[:, 14:17],
        application_point=matrix[:, 17:20],
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
    )


def write_metadata_json(metadata_path: Path, metadata: dict[str, object]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def build_run_metadata(
    args: argparse.Namespace,
    model: mujoco.MjModel,
    force: np.ndarray,
    point_offset: np.ndarray,
) -> dict[str, object]:
    direction = np.array([args.dir_x, args.dir_y, args.dir_z], dtype=float)
    direction_unit = direction / np.linalg.norm(direction)
    return {
        "command": " ".join(shlex.quote(arg) for arg in sys.argv),
        "script_path": str(Path(__file__).resolve()),
        "scene_path": str(args.scene.resolve()),
        "headless": bool(args.headless),
        "loop": bool(args.loop),
        "mode": "single",
        "force_magnitude": float(args.force),
        "direction_input": direction.tolist(),
        "direction_unit": direction_unit.tolist(),
        "applied_force_world": force.tolist(),
        "point_offset_local": point_offset.tolist(),
        "force_duration": float(args.force_duration),
        "total_duration": float(args.total_duration),
        "playback_speed": float(args.playback_speed),
        "timestep": float(model.opt.timestep),
        "force_steps": int(args.force_duration / model.opt.timestep),
        "total_steps": int(args.total_duration / model.opt.timestep),
        "rest_linear_threshold": REST_LINEAR_THRESHOLD,
        "rest_angular_threshold": REST_ANGULAR_THRESHOLD,
        "rest_hold_time": REST_HOLD_TIME,
        "video_path": str(args.video_path.resolve()),
        "trajectory_path": str(args.trajectory_path.resolve()) if args.trajectory_path is not None else None,
        "dataset_path": str(args.dataset_path.resolve()),
        "metadata_path": str(args.metadata_path.resolve()),
    }


def simulate_force(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    force: np.ndarray,
    point_offset: np.ndarray,
    force_duration: float,
    total_duration: float,
    frames: list[np.ndarray] | None = None,
    renderer: mujoco.Renderer | None = None,
    trajectory_rows: list[list[float]] | None = None,
    stop_on_rest: bool = False,
    video_frame_stride: int = DEFAULT_VIDEO_FRAME_STRIDE,
    force_schedule: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    body_id = block_body_id(model)
    total_steps = int(total_duration / model.opt.timestep)
    if force_schedule is None:
        force_schedule = single_force_schedule(force, force_duration)
    normalized_schedule, step_forces, force_steps = normalize_force_schedule_segments(
        force_schedule,
        float(model.opt.timestep),
        total_duration,
    )
    rest_steps = max(1, int(REST_HOLD_TIME / model.opt.timestep))
    rest_counter = 0
    trajectory_closed = False

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
        if trajectory_rows is not None and not trajectory_closed:
            record_block_state(model, data, applied_force, point_offset, trajectory_rows)
            if step >= force_steps:
                linear_speed, angular_speed = block_speed_norms(model, data)
                if linear_speed < REST_LINEAR_THRESHOLD and angular_speed < REST_ANGULAR_THRESHOLD:
                    rest_counter += 1
                    if rest_counter >= rest_steps:
                        trajectory_closed = True
                        if stop_on_rest:
                            capture_frame(data, frames, renderer, step, video_frame_stride)
                            break
                else:
                    rest_counter = 0
        capture_frame(data, frames, renderer, step, video_frame_stride)

    data.xfrc_applied[body_id] = 0.0
    data.qfrc_applied[:] = 0.0
    final_linear_speed, final_angular_speed = block_speed_norms(model, data)
    return {
        "rest_reached": trajectory_closed,
        "recorded_samples": len(trajectory_rows) if trajectory_rows is not None else 0,
        "recorded_end_time": float(trajectory_rows[-1][0]) if trajectory_rows else float(data.time),
        "final_block_position_world": data.xpos[body_id].copy(),
        "final_block_quaternion_world": data.xquat[body_id].copy(),
        "final_linear_speed": final_linear_speed,
        "final_angular_speed": final_angular_speed,
        "force_schedule": normalized_schedule,
        "force_steps": force_steps,
    }


def run_headless(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    force: np.ndarray,
    point_offset: np.ndarray,
    force_duration: float,
    total_duration: float,
    video_path: Path | None,
    playback_speed: float,
    trajectory_path: Path | None,
    dataset_path: Path | None,
    metadata_path: Path | None,
    metadata: dict[str, object],
    video_frame_stride: int,
) -> None:
    renderer = mujoco.Renderer(model, height=720, width=960)
    frames: list[np.ndarray] | None = [] if video_path else None
    trajectory_rows: list[list[float]] | None = [] if (trajectory_path or dataset_path) else None
    result = simulate_force(
        model,
        data,
        force,
        point_offset,
        force_duration,
        total_duration,
        frames=frames,
        renderer=renderer,
        trajectory_rows=trajectory_rows,
        video_frame_stride=video_frame_stride,
    )
    if video_path and frames:
        fps = max(1.0, 30.0 * playback_speed)
        write_video(video_path, frames, fps)
        metadata["video_num_frames"] = len(frames)
        metadata["video_fps"] = fps
        metadata["video_frame_stride"] = int(video_frame_stride)
    if trajectory_path and trajectory_rows is not None:
        write_trajectory_csv(trajectory_path, trajectory_rows)
    if dataset_path and trajectory_rows is not None:
        write_trajectory_npz(dataset_path, trajectory_rows, metadata)
    metadata["final_block_position_world"] = result["final_block_position_world"].tolist()
    metadata["final_block_quaternion_world"] = result["final_block_quaternion_world"].tolist()
    metadata["final_linear_speed"] = result["final_linear_speed"]
    metadata["final_angular_speed"] = result["final_angular_speed"]
    metadata["rest_reached"] = result["rest_reached"]
    metadata["recorded_samples"] = result["recorded_samples"]
    metadata["recorded_end_time"] = result["recorded_end_time"]
    if metadata_path:
        write_metadata_json(metadata_path, metadata)
    renderer.close()


def run_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    force: np.ndarray,
    point_offset: np.ndarray,
    force_duration: float,
    total_duration: float,
    playback_speed: float,
    loop: bool,
) -> None:
    import mujoco.viewer

    body_id = block_body_id(model)
    force_steps = int(force_duration / model.opt.timestep)
    total_steps = int(total_duration / model.opt.timestep)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        while viewer.is_running():
            if step >= total_steps:
                if not loop:
                    break
                reset_scene(model, data)
                step = 0

            data.ctrl[:] = 0.0
            data.xfrc_applied[body_id] = 0.0
            if step < force_steps:
                point = block_application_point(model, data, point_offset)
                qfrc = np.zeros(model.nv, dtype=float)
                mujoco.mj_applyFT(
                    model,
                    data,
                    force,
                    np.zeros(3, dtype=float),
                    point,
                    body_id,
                    qfrc,
                )
                data.qfrc_applied[:] = qfrc
            else:
                data.qfrc_applied[:] = 0.0

            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            sim_dt = model.opt.timestep / playback_speed
            remaining = sim_dt - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)
            step += 1

    data.xfrc_applied[body_id] = 0.0
    data.qfrc_applied[:] = 0.0


def block_local_bounds(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    body_id = block_body_id(model)
    geom_start = int(model.body_geomadr[body_id])
    geom_count = int(model.body_geomnum[body_id])
    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []
    for geom_id in range(geom_start, geom_start + geom_count):
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        geom_pos = model.geom_pos[geom_id].copy()
        geom_size = model.geom_size[geom_id].copy()
        mins.append(geom_pos - geom_size)
        maxs.append(geom_pos + geom_size)
    if not mins:
        raise ValueError(f"No box geoms found for body '{BLOCK_BODY_NAME}'.")
    return np.min(np.stack(mins), axis=0), np.max(np.stack(maxs), axis=0)


def sample_direction(rng: np.random.Generator, z_min: float, z_max: float) -> np.ndarray:
    z = float(rng.uniform(z_min, z_max))
    azimuth = float(rng.uniform(0.0, 2.0 * np.pi))
    radial = float(np.sqrt(max(0.0, 1.0 - z * z)))
    return np.array([radial * np.cos(azimuth), radial * np.sin(azimuth), z], dtype=float)


def normalize_force_schedule_segments(
    force_schedule: list[dict[str, object]],
    timestep: float,
    total_duration: float,
) -> tuple[list[dict[str, object]], np.ndarray, int]:
    total_steps = int(total_duration / timestep)
    step_forces = np.zeros((total_steps, 3), dtype=np.float64)
    normalized_segments: list[dict[str, object]] = []
    cursor_step = 0
    for segment_index, segment in enumerate(force_schedule):
        force = np.asarray(segment["force_world"], dtype=np.float64).reshape(3)
        duration = float(segment["duration"])
        if duration <= 0.0:
            raise ValueError("force schedule segment durations must be positive")
        segment_steps = max(1, int(duration / timestep + 1.0e-9))
        start_step = cursor_step
        end_step = min(total_steps, start_step + segment_steps)
        if end_step > start_step:
            step_forces[start_step:end_step] = force
        magnitude = float(np.linalg.norm(force))
        direction = force / magnitude if magnitude > 1.0e-12 else np.zeros(3, dtype=np.float64)
        normalized_segments.append(
            {
                "segment_index": int(segment_index),
                "start_step": int(start_step),
                "end_step": int(end_step),
                "start_time": float(start_step * timestep),
                "end_time": float(end_step * timestep),
                "duration": duration,
                "force_magnitude": magnitude,
                "direction_unit": direction.tolist(),
                "force_world": force.tolist(),
            }
        )
        cursor_step = end_step
        if cursor_step >= total_steps:
            break
    force_steps = int(max((segment["end_step"] for segment in normalized_segments), default=0))
    return normalized_segments, step_forces, force_steps


def single_force_schedule(force: np.ndarray, force_duration: float) -> list[dict[str, object]]:
    force = np.asarray(force, dtype=np.float64).reshape(3)
    magnitude = float(np.linalg.norm(force))
    direction = force / magnitude if magnitude > 1.0e-12 else np.zeros(3, dtype=np.float64)
    return [
        {
            "segment_index": 0,
            "duration": float(force_duration),
            "force_magnitude": magnitude,
            "direction_unit": direction.tolist(),
            "force_world": force.tolist(),
        }
    ]


def sample_force_schedule(
    rng: np.random.Generator,
    args: argparse.Namespace,
    first_magnitude: float,
    first_direction: np.ndarray,
    total_force_duration: float,
) -> list[dict[str, object]]:
    segment_count = max(1, int(args.force_segments))
    segment_duration = float(total_force_duration) / float(segment_count)
    schedule: list[dict[str, object]] = []
    for segment_index in range(segment_count):
        if segment_index == 0:
            magnitude = float(first_magnitude)
            direction = np.asarray(first_direction, dtype=np.float64)
        else:
            magnitude = float(rng.uniform(args.force_min, args.force_max))
            direction = sample_direction(rng, args.dir_z_min, args.dir_z_max)
        force = magnitude * direction
        schedule.append(
            {
                "segment_index": int(segment_index),
                "duration": segment_duration,
                "force_magnitude": magnitude,
                "direction_unit": direction.tolist(),
                "force_world": force.tolist(),
            }
        )
    return schedule


def first_force_segment(force_schedule: list[dict[str, object]]) -> dict[str, object]:
    if not force_schedule:
        raise ValueError("force schedule must contain at least one segment")
    return force_schedule[0]


def set_uniform_block_friction(model: mujoco.MjModel, friction_mu: float) -> None:
    for geom_name in BLOCK_FRICTION_GEOM_NAMES:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id < 0:
            raise ValueError(f"Could not find geom '{geom_name}' for uniform friction override.")
        model.geom_friction[geom_id, :] = np.array([float(friction_mu), 0.0, 0.0], dtype=np.float64)


def set_split_block_friction(model: mujoco.MjModel, left_mu: float, right_mu: float) -> None:
    friction_by_geom = {
        "push_block_left": float(left_mu),
        "push_block_right": float(right_mu),
    }
    for geom_name, friction_mu in friction_by_geom.items():
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id < 0:
            raise ValueError(f"Could not find geom '{geom_name}' for split friction override.")
        model.geom_friction[geom_id, :] = np.array([friction_mu, 0.0, 0.0], dtype=np.float64)


def block_friction_values(model: mujoco.MjModel) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    for geom_name in BLOCK_FRICTION_GEOM_NAMES:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id < 0:
            raise ValueError(f"Could not find geom '{geom_name}'.")
        values[geom_name] = model.geom_friction[geom_id].astype(float).tolist()
    return values


def format_friction_tag(friction_mu: float) -> str:
    return f"{float(friction_mu):.6g}".replace("-", "m").replace(".", "p")


def sample_point_offset(
    rng: np.random.Generator,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    edge_margin_ratio: float,
    mode: str = "surface",
) -> np.ndarray:
    span = bounds_max - bounds_min
    inner_min = bounds_min + edge_margin_ratio * span
    inner_max = bounds_max - edge_margin_ratio * span
    point = rng.uniform(inner_min, inner_max)
    side_faces = (
        (0, bounds_min[0] + SURFACE_EPS),
        (0, bounds_max[0] - SURFACE_EPS),
        (1, bounds_min[1] + SURFACE_EPS),
        (1, bounds_max[1] - SURFACE_EPS),
    )
    all_faces = (
        *side_faces,
        (2, bounds_max[2] - SURFACE_EPS),
    )
    if mode == "surface":
        axis, value = all_faces[int(rng.integers(0, len(all_faces)))]
        point[axis] = value
    elif mode == "side":
        axis, value = side_faces[int(rng.integers(0, len(side_faces)))]
        point[axis] = value
    elif mode == "edge":
        axes = rng.choice(np.array([0, 1, 2]), size=2, replace=False)
        for axis in axes:
            if axis == 2:
                point[axis] = bounds_max[axis] - SURFACE_EPS
            else:
                point[axis] = (bounds_min[axis] + SURFACE_EPS) if rng.random() < 0.5 else (bounds_max[axis] - SURFACE_EPS)
    elif mode == "corner":
        for axis in range(3):
            if axis == 2:
                point[axis] = bounds_max[axis] - SURFACE_EPS
            else:
                point[axis] = (bounds_min[axis] + SURFACE_EPS) if rng.random() < 0.5 else (bounds_max[axis] - SURFACE_EPS)
    else:
        raise ValueError(f"Unknown point sampling mode: {mode}")
    return point.astype(np.float64)


def sample_push_direction_for_point(
    rng: np.random.Generator,
    args: argparse.Namespace,
    point_offset: np.ndarray,
) -> np.ndarray:
    if args.push_direction_mode == "random":
        return sample_direction(rng, args.dir_z_min, args.dir_z_max)

    point_xy = np.asarray(point_offset[:2], dtype=np.float64)
    point_norm = float(np.linalg.norm(point_xy))
    if point_norm < 1.0e-10:
        return sample_direction(rng, args.dir_z_min, args.dir_z_max)

    radial = point_xy / point_norm
    tangent = np.array([-radial[1], radial[0]], dtype=np.float64)
    if rng.random() < 0.5:
        tangent = -tangent

    if args.push_direction_mode == "tangential":
        xy = tangent
    elif args.push_direction_mode == "inward-tangential":
        tangential_weight = float(args.push_tangential_weight)
        inward_weight = float(args.push_inward_weight)
        xy = tangential_weight * tangent - inward_weight * radial
        xy_norm = float(np.linalg.norm(xy))
        if xy_norm < 1.0e-10:
            xy = tangent
        else:
            xy = xy / xy_norm
    else:
        raise ValueError(f"Unknown push direction mode: {args.push_direction_mode}")

    z = float(rng.uniform(args.dir_z_min, args.dir_z_max))
    z = float(np.clip(z, -0.95, 0.95))
    radial_scale = float(np.sqrt(max(0.0, 1.0 - z * z)))
    return np.array([radial_scale * xy[0], radial_scale * xy[1], z], dtype=np.float64)


def write_batched_dataset_npz(
    dataset_path: Path,
    trajectories: list[np.ndarray],
    episode_metadata: list[dict[str, object]],
    summary_metadata: dict[str, object],
) -> None:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    num_episodes = len(trajectories)
    lengths = np.asarray([traj.shape[0] for traj in trajectories], dtype=np.int32)
    max_steps = int(lengths.max(initial=0))
    padded = np.full((num_episodes, max_steps, len(TRAJECTORY_COLUMNS)), np.nan, dtype=np.float32)
    for idx, traj in enumerate(trajectories):
        padded[idx, : traj.shape[0], :] = traj.astype(np.float32, copy=False)

    force_world = np.asarray([episode["applied_force_world"] for episode in episode_metadata], dtype=np.float32)
    direction_unit = np.asarray([episode["direction_unit"] for episode in episode_metadata], dtype=np.float32)
    point_offset_local = np.asarray([episode["point_offset_local"] for episode in episode_metadata], dtype=np.float32)
    initial_position = np.asarray(
        [episode["initial_block_position_world"] for episode in episode_metadata],
        dtype=np.float32,
    )
    initial_quaternion = np.asarray(
        [episode["initial_block_quaternion_world"] for episode in episode_metadata],
        dtype=np.float32,
    )
    initial_contact_face_id = np.asarray(
        [
            -1 if episode["initial_contact_face_id"] is None else int(episode["initial_contact_face_id"])
            for episode in episode_metadata
        ],
        dtype=np.int32,
    )
    force_duration = np.asarray([episode["force_duration"] for episode in episode_metadata], dtype=np.float32)
    final_position = np.asarray([episode["final_block_position_world"] for episode in episode_metadata], dtype=np.float32)
    final_quaternion = np.asarray(
        [episode["final_block_quaternion_world"] for episode in episode_metadata],
        dtype=np.float32,
    )
    rest_reached = np.asarray([episode["rest_reached"] for episode in episode_metadata], dtype=np.bool_)
    recorded_end_time = np.asarray([episode["recorded_end_time"] for episode in episode_metadata], dtype=np.float32)
    final_linear_speed = np.asarray([episode["final_linear_speed"] for episode in episode_metadata], dtype=np.float32)
    final_angular_speed = np.asarray([episode["final_angular_speed"] for episode in episode_metadata], dtype=np.float32)
    net_xy_displacement = np.asarray([episode["net_xy_displacement"] for episode in episode_metadata], dtype=np.float32)
    max_xy_displacement = np.asarray([episode["max_xy_displacement"] for episode in episode_metadata], dtype=np.float32)
    xy_path_length = np.asarray([episode["xy_path_length"] for episode in episode_metadata], dtype=np.float32)
    final_rotation_angle = np.asarray([episode["final_rotation_angle"] for episode in episode_metadata], dtype=np.float32)
    max_rotation_angle = np.asarray([episode["max_rotation_angle"] for episode in episode_metadata], dtype=np.float32)
    motion_filter_passed = np.asarray([episode["motion_filter_passed"] for episode in episode_metadata], dtype=np.bool_)
    motion_filter_attempts = np.asarray([episode["motion_filter_attempts"] for episode in episode_metadata], dtype=np.int32)
    force_schedules = [
        episode.get("force_schedule") or single_force_schedule(episode["applied_force_world"], episode["force_duration"])
        for episode in episode_metadata
    ]
    max_force_segments = max((len(schedule) for schedule in force_schedules), default=0)
    force_segment_counts = np.asarray([len(schedule) for schedule in force_schedules], dtype=np.int32)
    force_schedule_world = np.full((num_episodes, max_force_segments, 3), np.nan, dtype=np.float32)
    force_schedule_direction_unit = np.full((num_episodes, max_force_segments, 3), np.nan, dtype=np.float32)
    force_schedule_magnitude = np.full((num_episodes, max_force_segments), np.nan, dtype=np.float32)
    force_schedule_duration = np.full((num_episodes, max_force_segments), np.nan, dtype=np.float32)
    force_schedule_start_step = np.full((num_episodes, max_force_segments), -1, dtype=np.int32)
    force_schedule_end_step = np.full((num_episodes, max_force_segments), -1, dtype=np.int32)
    for episode_idx, schedule in enumerate(force_schedules):
        for segment_idx, segment in enumerate(schedule):
            force_schedule_world[episode_idx, segment_idx] = np.asarray(segment["force_world"], dtype=np.float32)
            force_schedule_direction_unit[episode_idx, segment_idx] = np.asarray(
                segment["direction_unit"],
                dtype=np.float32,
            )
            force_schedule_magnitude[episode_idx, segment_idx] = float(segment["force_magnitude"])
            force_schedule_duration[episode_idx, segment_idx] = float(segment["duration"])
            if "start_step" in segment:
                force_schedule_start_step[episode_idx, segment_idx] = int(segment["start_step"])
            if "end_step" in segment:
                force_schedule_end_step[episode_idx, segment_idx] = int(segment["end_step"])

    np.savez_compressed(
        dataset_path,
        columns=np.asarray(TRAJECTORY_COLUMNS),
        trajectories=padded,
        episode_lengths=lengths,
        force_world=force_world,
        direction_unit=direction_unit,
        point_offset_local=point_offset_local,
        initial_position=initial_position,
        initial_quaternion=initial_quaternion,
        initial_contact_face_id=initial_contact_face_id,
        force_duration=force_duration,
        final_position=final_position,
        final_quaternion=final_quaternion,
        rest_reached=rest_reached,
        recorded_end_time=recorded_end_time,
        final_linear_speed=final_linear_speed,
        final_angular_speed=final_angular_speed,
        net_xy_displacement=net_xy_displacement,
        max_xy_displacement=max_xy_displacement,
        xy_path_length=xy_path_length,
        final_rotation_angle=final_rotation_angle,
        max_rotation_angle=max_rotation_angle,
        motion_filter_passed=motion_filter_passed,
        motion_filter_attempts=motion_filter_attempts,
        force_segment_counts=force_segment_counts,
        force_schedule_world=force_schedule_world,
        force_schedule_direction_unit=force_schedule_direction_unit,
        force_schedule_magnitude=force_schedule_magnitude,
        force_schedule_duration=force_schedule_duration,
        force_schedule_start_step=force_schedule_start_step,
        force_schedule_end_step=force_schedule_end_step,
        episode_metadata_json=np.asarray(json.dumps(episode_metadata, ensure_ascii=False, sort_keys=True)),
        summary_metadata_json=np.asarray(json.dumps(summary_metadata, ensure_ascii=False, sort_keys=True)),
    )


def generate_uniform_friction_datasets(
    args: argparse.Namespace,
    source_episode_metadata: list[dict[str, object]],
    source_summary_metadata: dict[str, object],
) -> list[tuple[Path, Path]]:
    written_paths: list[tuple[Path, Path]] = []
    for friction_mu in args.uniform_friction_mu:
        tag = format_friction_tag(float(friction_mu))
        output_dir = args.output_dir.with_name(f"{args.output_dir.name}_uniform_mu_{tag}")
        output_name = output_dir.name
        dataset_path = output_dir / f"{output_name}.npz"
        metadata_path = output_dir / f"{output_name}.json"

        model = mujoco.MjModel.from_xml_path(str(args.scene))
        set_uniform_block_friction(model, float(friction_mu))
        data = mujoco.MjData(model)
        bounds_min, bounds_max = block_local_bounds(model)

        trajectories: list[np.ndarray] = []
        episode_metadata: list[dict[str, object]] = []
        for episode_idx, source_episode in enumerate(source_episode_metadata):
            reset_scene(model, data)
            initial_position = np.asarray(source_episode["initial_block_position_world"], dtype=np.float64)
            initial_quaternion = np.asarray(source_episode["initial_block_quaternion_world"], dtype=np.float64)
            set_block_freejoint_pose(model, data, initial_position, initial_quaternion)

            force_schedule = list(source_episode["force_schedule"])
            first_segment = first_force_segment(force_schedule)
            force = np.asarray(first_segment["force_world"], dtype=np.float64)
            point_offset = np.asarray(source_episode["point_offset_local"], dtype=np.float64)
            trajectory_rows: list[list[float]] = []
            result = simulate_force(
                model,
                data,
                force,
                point_offset,
                float(source_episode["force_duration"]),
                float(source_episode["total_duration"]),
                trajectory_rows=trajectory_rows,
                stop_on_rest=True,
                video_frame_stride=int(args.video_frame_stride),
                force_schedule=force_schedule,
            )
            motion_metrics = trajectory_motion_metrics(trajectory_rows)
            motion_filter_passed = passes_motion_filter(
                motion_metrics,
                float(args.min_sliding_distance),
                float(args.min_rotation_angle),
            )
            motion_score = motion_filter_score(
                motion_metrics,
                float(args.min_sliding_distance),
                float(args.min_rotation_angle),
            )

            trajectories.append(trajectory_rows_to_matrix(trajectory_rows))
            metadata = dict(source_episode)
            metadata.update(
                {
                    "source_episode_id": int(source_episode["episode_id"]),
                    "recorded_samples": result["recorded_samples"],
                    "recorded_end_time": result["recorded_end_time"],
                    "rest_reached": bool(result["rest_reached"]),
                    "final_block_position_world": result["final_block_position_world"].tolist(),
                    "final_block_quaternion_world": result["final_block_quaternion_world"].tolist(),
                    "final_linear_speed": float(result["final_linear_speed"]),
                    "final_angular_speed": float(result["final_angular_speed"]),
                    "motion_filter_passed": bool(motion_filter_passed),
                    "source_motion_filter_attempts": int(source_episode["motion_filter_attempts"]),
                    "motion_filter_score": float(motion_score),
                    **motion_metrics,
                    "video_path": None,
                    "source_dataset_path": str(args.dataset_path.resolve()),
                    "friction_override": {
                        geom_name: [float(friction_mu), 0.0, 0.0]
                        for geom_name in BLOCK_FRICTION_GEOM_NAMES
                    },
                }
            )
            episode_metadata.append(metadata)

            if (episode_idx + 1) % args.progress_every == 0 or episode_idx + 1 == len(source_episode_metadata):
                print(
                    f"[uniform-friction mu={float(friction_mu):.6g}] completed "
                    f"{episode_idx + 1}/{len(source_episode_metadata)} episodes"
                )

        total_samples = int(sum(item["recorded_samples"] for item in episode_metadata))
        motion_filter_attempts = [int(item["motion_filter_attempts"]) for item in episode_metadata]
        summary_metadata = {
            **source_summary_metadata,
            "command": " ".join(shlex.quote(arg) for arg in sys.argv),
            "mode": "dataset_uniform_friction_replay",
            "source_dataset_path": str(args.dataset_path.resolve()),
            "source_force_replay": True,
            "source_block_friction": source_summary_metadata.get("block_friction"),
            "friction_override": {
                geom_name: [float(friction_mu), 0.0, 0.0]
                for geom_name in BLOCK_FRICTION_GEOM_NAMES
            },
            "total_samples": total_samples,
            "rejected_motion_attempts": 0,
            "mean_motion_filter_attempts": float(np.mean(motion_filter_attempts)) if motion_filter_attempts else 0.0,
            "max_motion_filter_attempts": int(max(motion_filter_attempts, default=0)),
            "block_local_bounds_min": bounds_min.tolist(),
            "block_local_bounds_max": bounds_max.tolist(),
            "preview_dir": None,
            "save_episode_videos": False,
            "episode_video_dir": None,
            "video_count": 0,
            "dataset_path": str(dataset_path.resolve()),
            "metadata_path": str(metadata_path.resolve()),
        }
        write_batched_dataset_npz(dataset_path, trajectories, episode_metadata, summary_metadata)
        write_metadata_json(metadata_path, summary_metadata)
        written_paths.append((dataset_path, metadata_path))
        print(f"[uniform-friction mu={float(friction_mu):.6g}] dataset_path: {dataset_path}")
        print(f"[uniform-friction mu={float(friction_mu):.6g}] metadata_path: {metadata_path}")
    return written_paths


def simulate_uniform_comparison_rollouts(
    args: argparse.Namespace,
    source_episode_metadata: dict[str, object],
) -> dict[str, dict[str, object]]:
    comparisons: dict[str, dict[str, object]] = {}
    if not args.uniform_friction_mu:
        return comparisons

    source_position = np.asarray(source_episode_metadata["final_block_position_world"], dtype=np.float64)
    source_quaternion = np.asarray(source_episode_metadata["final_block_quaternion_world"], dtype=np.float64)

    for friction_mu in args.uniform_friction_mu:
        model = mujoco.MjModel.from_xml_path(str(args.scene))
        set_uniform_block_friction(model, float(friction_mu))
        data = mujoco.MjData(model)
        reset_scene(model, data)
        set_block_freejoint_pose(
            model,
            data,
            np.asarray(source_episode_metadata["initial_block_position_world"], dtype=np.float64),
            np.asarray(source_episode_metadata["initial_block_quaternion_world"], dtype=np.float64),
        )
        force_schedule = list(source_episode_metadata["force_schedule"])
        first_segment = first_force_segment(force_schedule)
        trajectory_rows: list[list[float]] = []
        result = simulate_force(
            model,
            data,
            np.asarray(first_segment["force_world"], dtype=np.float64),
            np.asarray(source_episode_metadata["point_offset_local"], dtype=np.float64),
            float(source_episode_metadata["force_duration"]),
            float(source_episode_metadata["total_duration"]),
            trajectory_rows=trajectory_rows,
            stop_on_rest=True,
            video_frame_stride=int(args.video_frame_stride),
            force_schedule=force_schedule,
        )
        metrics = final_pose_comparison_metrics(
            source_position,
            source_quaternion,
            result["final_block_position_world"],
            result["final_block_quaternion_world"],
        )
        tag = format_friction_tag(float(friction_mu))
        comparisons[tag] = {
            "uniform_friction_mu": float(friction_mu),
            "final_block_position_world": result["final_block_position_world"].tolist(),
            "final_block_quaternion_world": result["final_block_quaternion_world"].tolist(),
            "recorded_samples": int(result["recorded_samples"]),
            "recorded_end_time": float(result["recorded_end_time"]),
            "rest_reached": bool(result["rest_reached"]),
            **metrics,
        }
    return comparisons


def uniform_difference_score(comparisons: dict[str, dict[str, object]], yaw_weight: float) -> float:
    if not comparisons:
        return 0.0
    scores = [
        float(item["final_xy_position_delta_norm"]) + float(yaw_weight) * float(item["final_yaw_delta_abs"])
        for item in comparisons.values()
    ]
    return float(max(scores, default=0.0))


def generate_dataset(
    args: argparse.Namespace,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> tuple[Path, Path]:
    rng = np.random.default_rng(args.seed)
    bounds_min, bounds_max = block_local_bounds(model)
    preview_dir = args.preview_dir
    video_dir = args.episode_video_dir
    save_any_video = bool(args.save_episode_videos or args.preview_episodes > 0)
    renderer = mujoco.Renderer(model, height=720, width=960) if save_any_video else None
    trajectories: list[np.ndarray] = []
    episode_metadata: list[dict[str, object]] = []
    rejected_motion_attempts = 0

    try:
        for episode_id in range(args.num_episodes):
            save_episode_video = bool(args.save_episode_videos or episode_id < args.preview_episodes)
            max_attempts = int(args.max_resample_attempts) if args.require_motion else 1
            accepted_rollout: dict[str, object] | None = None
            best_rollout: dict[str, object] | None = None

            for attempt_id in range(1, max_attempts + 1):
                reset_scene(model, data)
                initial_pose = apply_dataset_initial_pose(args, model, data, rng, bounds_min, bounds_max)
                magnitude = float(rng.uniform(args.force_min, args.force_max))
                point_offset = sample_point_offset(
                    rng,
                    bounds_min,
                    bounds_max,
                    args.point_edge_margin_ratio,
                    mode=args.point_sampling_mode,
                )
                direction = sample_push_direction_for_point(rng, args, point_offset)
                duration = float(rng.uniform(args.duration_min, args.duration_max))
                force_schedule = sample_force_schedule(rng, args, magnitude, direction, duration)
                force = np.asarray(first_force_segment(force_schedule)["force_world"], dtype=float)

                trajectory_rows: list[list[float]] = []
                result = simulate_force(
                    model,
                    data,
                    force,
                    point_offset,
                    duration,
                    args.total_duration,
                    trajectory_rows=trajectory_rows,
                    stop_on_rest=True,
                    video_frame_stride=int(args.video_frame_stride),
                    force_schedule=force_schedule,
                )
                normalized_force_schedule = result["force_schedule"]
                motion_metrics = trajectory_motion_metrics(trajectory_rows)
                motion_filter_passed = passes_motion_filter(
                    motion_metrics,
                    float(args.min_sliding_distance),
                    float(args.min_rotation_angle),
                )
                motion_score = motion_filter_score(
                    motion_metrics,
                    float(args.min_sliding_distance),
                    float(args.min_rotation_angle),
                )
                rollout = {
                    "attempt_id": attempt_id,
                    "initial_pose": initial_pose,
                    "magnitude": magnitude,
                    "direction": direction,
                    "point_offset": point_offset,
                    "duration": duration,
                    "force": force,
                    "force_schedule": normalized_force_schedule,
                    "trajectory_rows": trajectory_rows,
                    "result": result,
                    "motion_metrics": motion_metrics,
                    "motion_filter_passed": motion_filter_passed,
                    "motion_score": motion_score,
                    "uniform_comparison": None,
                    "uniform_difference_score": 0.0,
                }

                if (
                    args.require_uniform_difference
                    or args.min_uniform_final_xy_delta > 0.0
                    or args.min_uniform_final_yaw_delta > 0.0
                ):
                    source_episode_metadata = {
                        "initial_block_position_world": np.asarray(initial_pose["position"], dtype=float).tolist(),
                        "initial_block_quaternion_world": np.asarray(
                            initial_pose["quaternion_wxyz"],
                            dtype=float,
                        ).tolist(),
                        "final_block_position_world": result["final_block_position_world"].tolist(),
                        "final_block_quaternion_world": result["final_block_quaternion_world"].tolist(),
                        "force_schedule": normalized_force_schedule,
                        "point_offset_local": point_offset.tolist(),
                        "force_duration": duration,
                        "total_duration": float(args.total_duration),
                    }
                    uniform_comparison = simulate_uniform_comparison_rollouts(args, source_episode_metadata)
                    uniform_score = uniform_difference_score(uniform_comparison, float(args.uniform_yaw_score_weight))
                    uniform_filter_passed = (
                        not args.require_uniform_difference
                        or any(
                            float(item["final_xy_position_delta_norm"]) >= float(args.min_uniform_final_xy_delta)
                            and float(item["final_yaw_delta_abs"]) >= float(args.min_uniform_final_yaw_delta)
                            for item in uniform_comparison.values()
                        )
                    )
                    rollout["uniform_comparison"] = uniform_comparison
                    rollout["uniform_difference_score"] = uniform_score
                    rollout["uniform_difference_filter_passed"] = bool(uniform_filter_passed)
                else:
                    rollout["uniform_difference_filter_passed"] = True

                rollout_score = motion_score + float(rollout["uniform_difference_score"])
                best_score = -float("inf") if best_rollout is None else float(best_rollout["motion_score"]) + float(
                    best_rollout["uniform_difference_score"]
                )
                if best_rollout is None or rollout_score > best_score:
                    best_rollout = rollout
                motion_ok = not args.require_motion or motion_filter_passed
                uniform_ok = bool(rollout["uniform_difference_filter_passed"])
                if motion_ok and uniform_ok:
                    accepted_rollout = rollout
                    break

                rejected_motion_attempts += 1

            if accepted_rollout is None:
                assert best_rollout is not None
                best_metrics = best_rollout["motion_metrics"]
                raise RuntimeError(
                    "Could not generate a trajectory satisfying the motion filter "
                    f"for episode {episode_id} after {max_attempts} attempts. "
                    f"Best max_xy_displacement={best_metrics['max_xy_displacement']:.6f} m, "
                    f"best max_rotation_angle={best_metrics['max_rotation_angle']:.6f} rad. "
                    "Lower --min-sliding-distance/--min-rotation-angle, raise force/duration ranges, "
                    "or increase --max-resample-attempts."
                )

            initial_pose = accepted_rollout["initial_pose"]
            magnitude = float(accepted_rollout["magnitude"])
            direction = np.asarray(accepted_rollout["direction"], dtype=float)
            point_offset = np.asarray(accepted_rollout["point_offset"], dtype=float)
            duration = float(accepted_rollout["duration"])
            force = np.asarray(accepted_rollout["force"], dtype=float)
            force_schedule = list(accepted_rollout["force_schedule"])
            trajectory_rows = accepted_rollout["trajectory_rows"]
            result = accepted_rollout["result"]
            motion_metrics = accepted_rollout["motion_metrics"]
            uniform_comparison = accepted_rollout.get("uniform_comparison") or {}
            uniform_difference_score_value = float(accepted_rollout.get("uniform_difference_score", float("nan")))
            uniform_difference_filter_passed = bool(
                accepted_rollout.get("uniform_difference_filter_passed", True)
            )

            episode_video_path: Path | None = None
            if save_episode_video:
                frames: list[np.ndarray] = []
                replay_rows: list[list[float]] = []
                reset_scene(model, data)
                set_block_freejoint_pose(
                    model,
                    data,
                    np.asarray(initial_pose["position"], dtype=float),
                    np.asarray(initial_pose["quaternion_wxyz"], dtype=float),
                )
                simulate_force(
                    model,
                    data,
                    force,
                    point_offset,
                    duration,
                    args.total_duration,
                    frames=frames,
                    renderer=renderer,
                    trajectory_rows=replay_rows,
                    stop_on_rest=True,
                    video_frame_stride=int(args.video_frame_stride),
                    force_schedule=force_schedule,
                )
                if frames:
                    output_dir = video_dir if args.save_episode_videos else preview_dir
                    output_dir.mkdir(parents=True, exist_ok=True)
                    episode_video_path = output_dir / f"episode_{episode_id:05d}.mp4"
                    fps = max(1.0, 30.0 * args.playback_speed)
                    write_video(episode_video_path, frames, fps=fps)

            trajectories.append(trajectory_rows_to_matrix(trajectory_rows))
            episode_metadata.append(
                {
                    "episode_id": episode_id,
                    "seed": int(args.seed),
                    "initial_block_position_world": np.asarray(initial_pose["position"], dtype=float).tolist(),
                    "initial_block_quaternion_world": np.asarray(
                        initial_pose["quaternion_wxyz"],
                        dtype=float,
                    ).tolist(),
                    "initial_contact_face_id": initial_pose["contact_face_id"],
                    "initial_contact_face_name": initial_pose["contact_face_name"],
                    "initial_contact_face_normal_local": (
                        None
                        if initial_pose["contact_face_normal_local"] is None
                        else np.asarray(initial_pose["contact_face_normal_local"], dtype=float).tolist()
                    ),
                    "initial_yaw_about_world_z": initial_pose["yaw_about_world_z"],
                    "force_magnitude": magnitude,
                    "direction_unit": direction.tolist(),
                    "applied_force_world": force.tolist(),
                    "force_segment_count": len(force_schedule),
                    "force_schedule": force_schedule,
                    "point_offset_local": point_offset.tolist(),
                    "force_duration": duration,
                    "total_duration": float(args.total_duration),
                    "recorded_samples": result["recorded_samples"],
                    "recorded_end_time": result["recorded_end_time"],
                    "rest_reached": bool(result["rest_reached"]),
                    "final_block_position_world": result["final_block_position_world"].tolist(),
                    "final_block_quaternion_world": result["final_block_quaternion_world"].tolist(),
                    "final_linear_speed": float(result["final_linear_speed"]),
                    "final_angular_speed": float(result["final_angular_speed"]),
                    "motion_filter_passed": bool(accepted_rollout["motion_filter_passed"]),
                    "motion_filter_attempts": int(accepted_rollout["attempt_id"]),
                    "motion_filter_score": float(accepted_rollout["motion_score"]),
                    "uniform_difference_filter_passed": uniform_difference_filter_passed,
                    "uniform_difference_score": uniform_difference_score_value,
                    "uniform_comparison": uniform_comparison,
                    **motion_metrics,
                    "video_path": str(episode_video_path.resolve()) if episode_video_path is not None else None,
                }
            )

            if (episode_id + 1) % args.progress_every == 0 or episode_id + 1 == args.num_episodes:
                total_samples = sum(item["recorded_samples"] for item in episode_metadata)
                print(
                    f"[dataset] completed {episode_id + 1}/{args.num_episodes} episodes "
                    f"({total_samples} samples, {rejected_motion_attempts} rejected motion attempts)"
                )
    finally:
        if renderer is not None:
            renderer.close()

    total_samples = int(sum(item["recorded_samples"] for item in episode_metadata))
    motion_filter_attempts = [int(item["motion_filter_attempts"]) for item in episode_metadata]
    summary_metadata = {
        "command": " ".join(shlex.quote(arg) for arg in sys.argv),
        "script_path": str(Path(__file__).resolve()),
        "scene_path": str(args.scene.resolve()),
        "mode": "dataset",
        "seed": int(args.seed),
        "num_episodes": int(args.num_episodes),
        "total_samples": total_samples,
        "total_duration": float(args.total_duration),
        "playback_speed": float(args.playback_speed),
        "timestep": float(model.opt.timestep),
        "force_range": [float(args.force_min), float(args.force_max)],
        "duration_range": [float(args.duration_min), float(args.duration_max)],
        "force_segments": int(args.force_segments),
        "force_schedule_mode": "equal-duration-segments",
        "dir_z_range": [float(args.dir_z_min), float(args.dir_z_max)],
        "push_direction_mode": str(args.push_direction_mode),
        "push_tangential_weight": float(args.push_tangential_weight),
        "push_inward_weight": float(args.push_inward_weight),
        "point_sampling_mode": str(args.point_sampling_mode),
        "point_edge_margin_ratio": float(args.point_edge_margin_ratio),
        "block_friction": block_friction_values(model),
        "block_friction_override": (
            None
            if args.block_left_friction is None and args.block_right_friction is None
            else {
                "push_block_left": [float(args.block_left_friction), 0.0, 0.0],
                "push_block_right": [float(args.block_right_friction), 0.0, 0.0],
            }
        ),
        "require_motion": bool(args.require_motion),
        "min_sliding_distance": float(args.min_sliding_distance),
        "min_rotation_angle": float(args.min_rotation_angle),
        "require_uniform_difference": bool(args.require_uniform_difference),
        "min_uniform_final_xy_delta": float(args.min_uniform_final_xy_delta),
        "min_uniform_final_yaw_delta": float(args.min_uniform_final_yaw_delta),
        "uniform_yaw_score_weight": float(args.uniform_yaw_score_weight),
        "max_resample_attempts": int(args.max_resample_attempts),
        "rejected_motion_attempts": int(rejected_motion_attempts),
        "mean_motion_filter_attempts": float(np.mean(motion_filter_attempts)) if motion_filter_attempts else 0.0,
        "max_motion_filter_attempts": int(max(motion_filter_attempts, default=0)),
        "randomize_initial_pose": bool(args.randomize_initial_pose),
        "init_x_range": [float(args.init_x_min), float(args.init_x_max)],
        "init_y_range": [float(args.init_y_min), float(args.init_y_max)],
        "init_clearance": float(args.init_clearance),
        "contact_face_normals": [
            {"id": idx, "name": name, "normal_local": normal.tolist()}
            for idx, (name, normal) in enumerate(CONTACT_FACE_NORMALS)
        ],
        "block_local_bounds_min": bounds_min.tolist(),
        "block_local_bounds_max": bounds_max.tolist(),
        "preview_dir": str(preview_dir.resolve()) if args.preview_episodes > 0 else None,
        "save_episode_videos": bool(args.save_episode_videos),
        "episode_video_dir": str(video_dir.resolve()) if args.save_episode_videos else None,
        "video_frame_stride": int(args.video_frame_stride),
        "video_fps": max(1.0, 30.0 * args.playback_speed),
        "video_count": int(args.num_episodes if args.save_episode_videos else min(args.preview_episodes, args.num_episodes)),
        "dataset_path": str(args.dataset_path.resolve()),
        "metadata_path": str(args.metadata_path.resolve()),
        "uniform_friction_mus": [float(mu) for mu in args.uniform_friction_mu],
    }
    write_batched_dataset_npz(args.dataset_path, trajectories, episode_metadata, summary_metadata)
    write_metadata_json(args.metadata_path, summary_metadata)
    if args.uniform_friction_mu:
        generate_uniform_friction_datasets(args, episode_metadata, summary_metadata)
    return args.dataset_path, args.metadata_path


def validate_args(args: argparse.Namespace) -> None:
    if args.playback_speed <= 0:
        raise ValueError("Playback speed must be positive.")
    if args.total_duration <= 0:
        raise ValueError("Total duration must be positive.")
    if args.num_episodes < 0:
        raise ValueError("num-episodes must be non-negative.")
    if args.progress_every <= 0:
        raise ValueError("progress-every must be positive.")
    if args.preview_episodes < 0:
        raise ValueError("preview-episodes must be non-negative.")
    if args.video_frame_stride <= 0:
        raise ValueError("video-frame-stride must be positive.")
    if args.force_segments <= 0:
        raise ValueError("force-segments must be positive.")
    if any(float(mu) < 0.0 for mu in args.uniform_friction_mu):
        raise ValueError("uniform-friction-mu values must be non-negative.")
    if args.uniform_friction_mu and args.num_episodes <= 0:
        raise ValueError("--uniform-friction-mu is only supported in dataset mode (--num-episodes > 0).")
    if args.require_uniform_difference and not args.uniform_friction_mu:
        raise ValueError("--require-uniform-difference requires at least one --uniform-friction-mu value.")
    if args.min_uniform_final_xy_delta < 0.0:
        raise ValueError("min-uniform-final-xy-delta must be non-negative.")
    if args.min_uniform_final_yaw_delta < 0.0:
        raise ValueError("min-uniform-final-yaw-delta must be non-negative.")
    if args.uniform_yaw_score_weight < 0.0:
        raise ValueError("uniform-yaw-score-weight must be non-negative.")
    if args.push_tangential_weight < 0.0:
        raise ValueError("push-tangential-weight must be non-negative.")
    if args.push_inward_weight < 0.0:
        raise ValueError("push-inward-weight must be non-negative.")
    if (args.block_left_friction is None) != (args.block_right_friction is None):
        raise ValueError("--block-left-friction and --block-right-friction must be supplied together.")
    if args.block_left_friction is not None and args.block_left_friction < 0.0:
        raise ValueError("block-left-friction must be non-negative.")
    if args.block_right_friction is not None and args.block_right_friction < 0.0:
        raise ValueError("block-right-friction must be non-negative.")
    if args.point_edge_margin_ratio < 0.0 or args.point_edge_margin_ratio >= 0.5:
        raise ValueError("point-edge-margin-ratio must be in [0, 0.5).")
    if args.min_sliding_distance < 0.0:
        raise ValueError("min-sliding-distance must be non-negative.")
    if args.min_rotation_angle < 0.0:
        raise ValueError("min-rotation-angle must be non-negative.")
    if args.max_resample_attempts <= 0:
        raise ValueError("max-resample-attempts must be positive.")

    if args.num_episodes > 0:
        if args.init_x_min > args.init_x_max:
            raise ValueError("init-x-min must be <= init-x-max.")
        if args.init_y_min > args.init_y_max:
            raise ValueError("init-y-min must be <= init-y-max.")
        if args.init_clearance < 0.0:
            raise ValueError("init-clearance must be non-negative.")
        if args.force_min <= 0 or args.force_max <= 0 or args.force_min > args.force_max:
            raise ValueError("force-min and force-max must be positive and force-min <= force-max.")
        if args.duration_min <= 0 or args.duration_max <= 0 or args.duration_min > args.duration_max:
            raise ValueError("duration-min and duration-max must be positive and duration-min <= duration-max.")
        if args.dir_z_min < -1.0 or args.dir_z_max > 1.0 or args.dir_z_min > args.dir_z_max:
            raise ValueError("dir-z range must satisfy -1 <= dir-z-min <= dir-z-max <= 1.")
    else:
        if args.force <= 0:
            raise ValueError("Force magnitude must be positive.")
        if args.force_duration <= 0:
            raise ValueError("Force duration must be positive.")


def attach_output_paths(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    output_dir = args.output_dir
    output_name = output_dir.name
    if not output_name:
        parser.error("--output-dir must include a directory name.")

    args.video_path = output_dir / f"{output_name}.mp4"
    args.trajectory_path = output_dir / f"{output_name}.csv" if args.save_trajectory_csv else None
    args.dataset_path = output_dir / f"{output_name}.npz"
    args.metadata_path = output_dir / f"{output_name}.json"
    args.preview_dir = output_dir / f"{output_name}_previews"
    args.episode_video_dir = output_dir / f"{output_name}_videos"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply an external force to the block in the MuJoCo scene.")
    parser.add_argument("--scene", type=Path, default=SCENE_PATH, help="Path to the MuJoCo XML scene.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory for this MuJoCo run. Output file names are derived from the directory name: "
            "<name>.npz dataset, <name>.json metadata, <name>_videos/ episode videos, "
            "<name>_previews/ preview videos, and <name>.mp4 for single headless video."
        ),
    )
    parser.add_argument("--force", type=float, default=2.0, help="Force magnitude in Newtons.")
    parser.add_argument("--dir-x", type=float, default=1.0, help="Direction X component.")
    parser.add_argument("--dir-y", type=float, default=0.0, help="Direction Y component.")
    parser.add_argument("--dir-z", type=float, default=0.0, help="Direction Z component.")
    parser.add_argument("--point-x", type=float, default=0.0, help="Application point offset in the block local X.")
    parser.add_argument("--point-y", type=float, default=0.0, help="Application point offset in the block local Y.")
    parser.add_argument("--point-z", type=float, default=0.0, help="Application point offset in the block local Z.")
    parser.add_argument("--force-duration", type=float, default=0.15, help="How long to apply the external force.")
    parser.add_argument("--total-duration", type=float, default=3.0, help="Total simulation duration.")
    parser.add_argument("--playback-speed", type=float, default=1.0, help="Viewer/video playback speed multiplier.")
    parser.add_argument("--loop", action="store_true", help="Loop the simulation in the interactive viewer.")
    parser.add_argument("--headless", action="store_true", help="Run without the interactive viewer.")
    parser.add_argument(
        "--save-trajectory-csv",
        action="store_true",
        help="Also write <output-dir>/<name>.csv for a single headless rollout.",
    )
    parser.add_argument("--num-episodes", type=int, default=0, help="If > 0, generate a random training dataset.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for dataset generation.")
    parser.add_argument("--force-min", type=float, default=0.5, help="Minimum sampled force magnitude.")
    parser.add_argument("--force-max", type=float, default=8.0, help="Maximum sampled force magnitude.")
    parser.add_argument("--duration-min", type=float, default=0.03, help="Minimum sampled force duration.")
    parser.add_argument("--duration-max", type=float, default=0.35, help="Maximum sampled force duration.")
    parser.add_argument(
        "--force-segments",
        type=int,
        default=1,
        help=(
            "Number of consecutive force segments per dataset episode. Segments share one local application "
            "point, split the sampled force duration equally, and resample magnitude/direction per segment."
        ),
    )
    parser.add_argument("--dir-z-min", type=float, default=-0.35, help="Minimum sampled Z component of force direction.")
    parser.add_argument("--dir-z-max", type=float, default=0.35, help="Maximum sampled Z component of force direction.")
    parser.add_argument(
        "--push-direction-mode",
        choices=("random", "tangential", "inward-tangential"),
        default="random",
        help=(
            "Dataset force direction sampling. The tangential modes choose force directions from the local "
            "application point to increase torque for eccentric and corner pushes."
        ),
    )
    parser.add_argument(
        "--push-tangential-weight",
        type=float,
        default=1.0,
        help="Tangential XY weight for --push-direction-mode inward-tangential.",
    )
    parser.add_argument(
        "--push-inward-weight",
        type=float,
        default=0.35,
        help="Inward radial XY weight for --push-direction-mode inward-tangential.",
    )
    parser.add_argument(
        "--point-sampling-mode",
        choices=("surface", "side", "edge", "corner"),
        default="surface",
        help="Where to sample local force application points in dataset mode.",
    )
    parser.add_argument(
        "--point-edge-margin-ratio",
        type=float,
        default=0.08,
        help="Relative margin when sampling application points on the block surface.",
    )
    parser.add_argument(
        "--require-motion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In dataset mode, reject sampled rollouts that do not slide or rotate enough.",
    )
    parser.add_argument(
        "--min-sliding-distance",
        type=float,
        default=DEFAULT_MIN_SLIDING_DISTANCE,
        help="Minimum max XY displacement in meters for accepting a dataset rollout.",
    )
    parser.add_argument(
        "--min-rotation-angle",
        type=float,
        default=DEFAULT_MIN_ROTATION_ANGLE,
        help="Minimum max orientation change in radians for accepting a dataset rollout.",
    )
    parser.add_argument(
        "--max-resample-attempts",
        type=int,
        default=DEFAULT_MAX_RESAMPLE_ATTEMPTS,
        help="Maximum sampled rollouts to try per accepted dataset episode when --require-motion is enabled.",
    )
    parser.add_argument(
        "--preview-episodes",
        type=int,
        default=0,
        help="Save preview videos for the first N dataset episodes.",
    )
    parser.add_argument(
        "--save-episode-videos",
        action="store_true",
        help="Save a video for every dataset episode.",
    )
    parser.add_argument(
        "--uniform-friction-mu",
        type=float,
        nargs="*",
        default=[],
        help=(
            "Also generate companion datasets with both block geoms set to each supplied sliding friction "
            "coefficient. These companion datasets replay the same accepted initial poses, force schedules, "
            "and local force points as the primary dataset."
        ),
    )
    parser.add_argument(
        "--block-left-friction",
        type=float,
        default=None,
        help="Override sliding friction for push_block_left in the primary dataset.",
    )
    parser.add_argument(
        "--block-right-friction",
        type=float,
        default=None,
        help="Override sliding friction for push_block_right in the primary dataset.",
    )
    parser.add_argument(
        "--require-uniform-difference",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "In dataset mode, reject sampled rollouts unless at least one companion uniform replay differs "
            "from the primary rollout by the requested final XY and yaw thresholds."
        ),
    )
    parser.add_argument(
        "--min-uniform-final-xy-delta",
        type=float,
        default=0.0,
        help="Minimum final XY position delta in meters for --require-uniform-difference.",
    )
    parser.add_argument(
        "--min-uniform-final-yaw-delta",
        type=float,
        default=0.0,
        help="Minimum final yaw delta in radians for --require-uniform-difference.",
    )
    parser.add_argument(
        "--uniform-yaw-score-weight",
        type=float,
        default=0.1,
        help="Meters-per-radian weight used when ranking resampled attempts by uniform replay difference.",
    )
    parser.add_argument(
        "--video-frame-stride",
        type=int,
        default=DEFAULT_VIDEO_FRAME_STRIDE,
        help="Capture every Nth simulation step into exported videos.",
    )
    parser.add_argument(
        "--randomize-initial-pose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In dataset mode, sample a random initial block position and contact face.",
    )
    parser.add_argument(
        "--init-x-min",
        type=float,
        default=DEFAULT_INIT_X_RANGE[0],
        help="Minimum sampled initial block X position in dataset mode.",
    )
    parser.add_argument(
        "--init-x-max",
        type=float,
        default=DEFAULT_INIT_X_RANGE[1],
        help="Maximum sampled initial block X position in dataset mode.",
    )
    parser.add_argument(
        "--init-y-min",
        type=float,
        default=DEFAULT_INIT_Y_RANGE[0],
        help="Minimum sampled initial block Y position in dataset mode.",
    )
    parser.add_argument(
        "--init-y-max",
        type=float,
        default=DEFAULT_INIT_Y_RANGE[1],
        help="Maximum sampled initial block Y position in dataset mode.",
    )
    parser.add_argument(
        "--init-clearance",
        type=float,
        default=0.0,
        help="Extra height above the floor after placing the selected contact face.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N dataset episodes.",
    )
    args = parser.parse_args()
    attach_output_paths(args, parser)
    return args


def main() -> None:
    args = parse_args()
    validate_args(args)

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    if args.block_left_friction is not None and args.block_right_friction is not None:
        set_split_block_friction(model, float(args.block_left_friction), float(args.block_right_friction))
    data = mujoco.MjData(model)
    reset_scene(model, data)

    if args.num_episodes > 0:
        dataset_path, metadata_path = generate_dataset(args, model, data)
        print("dataset_path:", dataset_path)
        print("metadata_path:", metadata_path)
        return

    direction = np.array([args.dir_x, args.dir_y, args.dir_z], dtype=float)
    direction_norm = np.linalg.norm(direction)
    if direction_norm < 1e-8:
        raise ValueError("Force direction must be non-zero.")
    force = args.force * direction / direction_norm
    point_offset = np.array([args.point_x, args.point_y, args.point_z], dtype=float)
    metadata = build_run_metadata(args, model, force, point_offset)

    if args.headless:
        run_headless(
            model,
            data,
            force,
            point_offset,
            args.force_duration,
            args.total_duration,
            args.video_path,
            args.playback_speed,
            args.trajectory_path,
            args.dataset_path,
            args.metadata_path,
            metadata,
            args.video_frame_stride,
        )
    else:
        run_viewer(
            model,
            data,
            force,
            point_offset,
            args.force_duration,
            args.total_duration,
            args.playback_speed,
            args.loop,
        )

    block_pos = body_position(model, data, BLOCK_BODY_NAME)
    print("applied force:", np.array2string(force, precision=4))
    print("application point offset:", np.array2string(point_offset, precision=4))
    print("final block position:", np.array2string(block_pos, precision=4))


if __name__ == "__main__":
    main()
