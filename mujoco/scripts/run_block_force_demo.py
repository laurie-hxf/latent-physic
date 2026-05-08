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
) -> dict[str, object]:
    body_id = block_body_id(model)
    force_steps = int(force_duration / model.opt.timestep)
    total_steps = int(total_duration / model.opt.timestep)
    rest_steps = max(1, int(REST_HOLD_TIME / model.opt.timestep))
    rest_counter = 0
    trajectory_closed = False

    record_block_state(model, data, np.zeros(3, dtype=float), point_offset, trajectory_rows)

    for step in range(total_steps):
        data.ctrl[:] = 0.0
        data.xfrc_applied[body_id] = 0.0
        applied_force = np.zeros(3, dtype=float)
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
            applied_force = force.copy()
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


def sample_point_offset(
    rng: np.random.Generator,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    edge_margin_ratio: float,
) -> np.ndarray:
    span = bounds_max - bounds_min
    inner_min = bounds_min + edge_margin_ratio * span
    inner_max = bounds_max - edge_margin_ratio * span
    point = rng.uniform(inner_min, inner_max)
    faces = (
        (0, bounds_min[0] + SURFACE_EPS),
        (0, bounds_max[0] - SURFACE_EPS),
        (1, bounds_min[1] + SURFACE_EPS),
        (1, bounds_max[1] - SURFACE_EPS),
        (2, bounds_max[2] - SURFACE_EPS),
    )
    axis, value = faces[int(rng.integers(0, len(faces)))]
    point[axis] = value
    return point.astype(np.float64)


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
        episode_metadata_json=np.asarray(json.dumps(episode_metadata, ensure_ascii=False, sort_keys=True)),
        summary_metadata_json=np.asarray(json.dumps(summary_metadata, ensure_ascii=False, sort_keys=True)),
    )


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

    try:
        for episode_id in range(args.num_episodes):
            reset_scene(model, data)
            initial_pose = apply_dataset_initial_pose(args, model, data, rng, bounds_min, bounds_max)
            magnitude = float(rng.uniform(args.force_min, args.force_max))
            direction = sample_direction(rng, args.dir_z_min, args.dir_z_max)
            point_offset = sample_point_offset(rng, bounds_min, bounds_max, args.point_edge_margin_ratio)
            duration = float(rng.uniform(args.duration_min, args.duration_max))
            force = magnitude * direction

            save_episode_video = bool(args.save_episode_videos or episode_id < args.preview_episodes)
            frames: list[np.ndarray] | None = [] if save_episode_video else None
            trajectory_rows: list[list[float]] = []
            result = simulate_force(
                model,
                data,
                force,
                point_offset,
                duration,
                args.total_duration,
                frames=frames,
                renderer=renderer,
                trajectory_rows=trajectory_rows,
                stop_on_rest=True,
                video_frame_stride=int(args.video_frame_stride),
            )
            episode_video_path: Path | None = None
            if frames is not None and frames:
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
                    "video_path": str(episode_video_path.resolve()) if episode_video_path is not None else None,
                }
            )

            if (episode_id + 1) % args.progress_every == 0 or episode_id + 1 == args.num_episodes:
                total_samples = sum(item["recorded_samples"] for item in episode_metadata)
                print(f"[dataset] completed {episode_id + 1}/{args.num_episodes} episodes ({total_samples} samples)")
    finally:
        if renderer is not None:
            renderer.close()

    total_samples = int(sum(item["recorded_samples"] for item in episode_metadata))
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
        "dir_z_range": [float(args.dir_z_min), float(args.dir_z_max)],
        "point_edge_margin_ratio": float(args.point_edge_margin_ratio),
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
    }
    write_batched_dataset_npz(args.dataset_path, trajectories, episode_metadata, summary_metadata)
    write_metadata_json(args.metadata_path, summary_metadata)
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
    if args.point_edge_margin_ratio < 0.0 or args.point_edge_margin_ratio >= 0.5:
        raise ValueError("point-edge-margin-ratio must be in [0, 0.5).")

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply an external force to the block in the MuJoCo scene.")
    parser.add_argument("--scene", type=Path, default=SCENE_PATH, help="Path to the MuJoCo XML scene.")
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
        "--video-path",
        type=Path,
        default=ROOT / "outputs" / "block_force_demo.mp4",
        help="Output video path when running headless.",
    )
    parser.add_argument(
        "--trajectory-path",
        type=Path,
        default=None,
        help="Optional CSV output path for the block trajectory when running a single headless rollout.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=ROOT / "outputs" / "block_force_trajectory.npz",
        help="Compressed NumPy export path. In dataset mode this stores the batched training dataset.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=ROOT / "outputs" / "block_force_metadata.json",
        help="JSON path for recording the run inputs and export metadata when running headless.",
    )
    parser.add_argument("--num-episodes", type=int, default=0, help="If > 0, generate a random training dataset.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for dataset generation.")
    parser.add_argument("--force-min", type=float, default=0.5, help="Minimum sampled force magnitude.")
    parser.add_argument("--force-max", type=float, default=8.0, help="Maximum sampled force magnitude.")
    parser.add_argument("--duration-min", type=float, default=0.03, help="Minimum sampled force duration.")
    parser.add_argument("--duration-max", type=float, default=0.35, help="Maximum sampled force duration.")
    parser.add_argument("--dir-z-min", type=float, default=-0.35, help="Minimum sampled Z component of force direction.")
    parser.add_argument("--dir-z-max", type=float, default=0.35, help="Maximum sampled Z component of force direction.")
    parser.add_argument(
        "--point-edge-margin-ratio",
        type=float,
        default=0.08,
        help="Relative margin when sampling application points on the block surface.",
    )
    parser.add_argument(
        "--preview-episodes",
        type=int,
        default=0,
        help="Save preview videos for the first N dataset episodes.",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=ROOT / "outputs" / "block_force_dataset_previews",
        help="Directory for dataset preview videos.",
    )
    parser.add_argument(
        "--save-episode-videos",
        action="store_true",
        help="Save a video for every dataset episode.",
    )
    parser.add_argument(
        "--episode-video-dir",
        type=Path,
        default=ROOT / "outputs" / "block_force_dataset_videos",
        help="Directory for videos written by --save-episode-videos.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)

    model = mujoco.MjModel.from_xml_path(str(args.scene))
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
