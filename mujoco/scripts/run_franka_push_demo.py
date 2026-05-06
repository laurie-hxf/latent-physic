from __future__ import annotations

import argparse
import time
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = ROOT / "third_party" / "mujoco_menagerie" / "franka_emika_panda" / "franka_table_push_scene.xml"

BLOCK_BODY_NAME = "push_block"
TARGET_BODY_NAME = "push_target"
GRIPPER_SITE_NAME = "gripper"
HOME_KEY_NAME = "home"

APPROACH_QPOS = np.array(
    [0.2897, 0.496673, -0.142836, -2.14746, -0.0295746, 2.52378, -0.492496],
    dtype=float,
)
GRIPPER_OPEN = 0.0
ROT_WEIGHT = 0.2
IK_DAMPING = 1e-4


def reset_to_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, HOME_KEY_NAME)
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)


def body_position(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return data.xpos[body_id].copy()


def get_reference_orientation(model: mujoco.MjModel) -> np.ndarray:
    data = mujoco.MjData(model)
    data.qpos[:7] = APPROACH_QPOS
    data.qpos[7:9] = GRIPPER_OPEN
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, GRIPPER_SITE_NAME)
    return data.site_xmat[site_id].reshape(3, 3).copy()


def orientation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    rotation_error = target @ current.T
    return 0.5 * np.array(
        [
            rotation_error[2, 1] - rotation_error[1, 2],
            rotation_error[0, 2] - rotation_error[2, 0],
            rotation_error[1, 0] - rotation_error[0, 1],
        ],
        dtype=float,
    )


def solve_ik(
    model: mujoco.MjModel,
    target_pos: np.ndarray,
    initial_qpos: np.ndarray,
    reference_orientation: np.ndarray,
    iterations: int = 200,
) -> np.ndarray:
    data = mujoco.MjData(model)
    data.qpos[:7] = initial_qpos
    data.qpos[7:9] = GRIPPER_OPEN
    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, GRIPPER_SITE_NAME)
    joint_ranges = model.jnt_range[:7]

    for _ in range(iterations):
        site_pos = data.site_xpos[site_id].copy()
        site_rot = data.site_xmat[site_id].reshape(3, 3)

        pos_error = target_pos - site_pos
        rot_error = orientation_error(site_rot, reference_orientation)

        if np.linalg.norm(pos_error) < 1e-4 and np.linalg.norm(rot_error) < 1e-3:
            break

        jac_pos = np.zeros((3, model.nv), dtype=float)
        jac_rot = np.zeros((3, model.nv), dtype=float)
        mujoco.mj_jacSite(model, data, jac_pos, jac_rot, site_id)

        jacobian = np.vstack([jac_pos[:, :7], ROT_WEIGHT * jac_rot[:, :7]])
        error = np.concatenate([pos_error, ROT_WEIGHT * rot_error])
        damped = jacobian @ jacobian.T + IK_DAMPING * np.eye(6)
        delta_q = jacobian.T @ np.linalg.solve(damped, error)

        data.qpos[:7] = np.clip(data.qpos[:7] + 0.5 * delta_q, joint_ranges[:, 0], joint_ranges[:, 1])
        mujoco.mj_forward(model, data)

    return data.qpos[:7].copy()


def build_push_waypoints(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[list[np.ndarray], list[float]]:
    block_start = body_position(model, data, BLOCK_BODY_NAME)
    target = body_position(model, data, TARGET_BODY_NAME)

    hover_height = block_start[2] + 0.10
    pre_push_height = block_start[2] + 0.07
    push_height = block_start[2] + 0.045
    retreat_height = block_start[2] + 0.145
    push_end_x = target[0] - 0.02

    waypoints = [
        block_start + np.array([-0.04, 0.0, hover_height - block_start[2]], dtype=float),
        block_start + np.array([-0.015, 0.0, pre_push_height - block_start[2]], dtype=float),
        np.array([push_end_x, target[1], push_height], dtype=float),
        np.array([push_end_x, target[1], retreat_height], dtype=float),
    ]
    durations = [1.5, 1.2, 2.4, 1.0]
    return waypoints, durations


def build_joint_trajectory(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[list[np.ndarray], list[float]]:
    reference_orientation = get_reference_orientation(model)
    waypoints, durations = build_push_waypoints(model, data)

    current_qpos = data.qpos[:7].copy()
    joint_targets: list[np.ndarray] = []
    for waypoint in waypoints:
        current_qpos = solve_ik(model, waypoint, current_qpos, reference_orientation)
        joint_targets.append(current_qpos.copy())
    return joint_targets, durations


def capture_frame(
    data: mujoco.MjData,
    frames: list[np.ndarray] | None,
    renderer: mujoco.Renderer | None,
    frame_index: int,
) -> None:
    if frames is None or renderer is None:
        return
    if frame_index % 3 != 0:
        return
    renderer.update_scene(data)
    frames.append(renderer.render().copy())


def move_joints_linear(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    start_qpos: np.ndarray,
    end_qpos: np.ndarray,
    seconds: float,
    frames: list[np.ndarray] | None = None,
    renderer: mujoco.Renderer | None = None,
    frame_offset: int = 0,
) -> int:
    steps = max(1, int(seconds / model.opt.timestep))
    for step in range(steps):
        alpha = (step + 1) / steps
        data.ctrl[:7] = (1.0 - alpha) * start_qpos + alpha * end_qpos
        data.ctrl[7] = GRIPPER_OPEN
        mujoco.mj_step(model, data)
        capture_frame(data, frames, renderer, frame_offset + step)
    return steps


def run_push_sequence(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    frames: list[np.ndarray] | None = None,
    renderer: mujoco.Renderer | None = None,
) -> None:
    joint_targets, durations = build_joint_trajectory(model, data)

    frame_offset = 0
    current_qpos = data.ctrl[:7].copy()
    for joint_target, seconds in zip(joint_targets, durations):
        frame_offset += move_joints_linear(
            model,
            data,
            current_qpos,
            joint_target,
            seconds,
            frames=frames,
            renderer=renderer,
            frame_offset=frame_offset,
        )
        current_qpos = joint_target

    settle_steps = int(0.8 / model.opt.timestep)
    for settle_idx in range(settle_steps):
        data.ctrl[:7] = current_qpos
        data.ctrl[7] = GRIPPER_OPEN
        mujoco.mj_step(model, data)
        capture_frame(data, frames, renderer, frame_offset + settle_idx)


def run_headless(model: mujoco.MjModel, data: mujoco.MjData, video_path: Path | None) -> None:
    renderer = mujoco.Renderer(model, height=720, width=960)
    frames: list[np.ndarray] | None = [] if video_path else None
    run_push_sequence(model, data, frames=frames, renderer=renderer)
    if video_path and frames:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(video_path, np.stack(frames), fps=30)
    renderer.close()


def run_viewer(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    import mujoco.viewer

    joint_targets, durations = build_joint_trajectory(model, data)
    segment_steps = [max(1, int(seconds / model.opt.timestep)) for seconds in durations]

    current_segment = 0
    current_step = 0
    start_qpos = data.ctrl[:7].copy()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if current_segment < len(joint_targets):
                total_steps = segment_steps[current_segment]
                alpha = min(1.0, (current_step + 1) / total_steps)
                data.ctrl[:7] = (1.0 - alpha) * start_qpos + alpha * joint_targets[current_segment]
                data.ctrl[7] = GRIPPER_OPEN
                current_step += 1
                if current_step >= total_steps:
                    start_qpos = joint_targets[current_segment].copy()
                    current_step = 0
                    current_segment += 1
            else:
                data.ctrl[:7] = start_qpos
                data.ctrl[7] = GRIPPER_OPEN

            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            remaining = model.opt.timestep - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simple MuJoCo Franka pushing demo.")
    parser.add_argument("--scene", type=Path, default=SCENE_PATH, help="Path to the MuJoCo XML scene.")
    parser.add_argument("--headless", action="store_true", help="Run without the interactive viewer.")
    parser.add_argument(
        "--video-path",
        type=Path,
        default=ROOT / "outputs" / "franka_push_demo.mp4",
        help="Output video path when running headless.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    reset_to_home(model, data)

    if args.headless:
        run_headless(model, data, args.video_path)
    else:
        run_viewer(model, data)

    block_pos = body_position(model, data, BLOCK_BODY_NAME)
    final_target = body_position(model, data, TARGET_BODY_NAME)
    print("final block position:", np.array2string(block_pos, precision=4))
    print("target position:", np.array2string(final_target, precision=4))
    print("distance to target:", float(np.linalg.norm(block_pos - final_target)))


if __name__ == "__main__":
    main()
