from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .camera import CameraSpec, add_fixed_cameras_to_xml


REPO_ROOT = Path(__file__).resolve().parents[1]
BLOCK_FORCE_SCENE_PATH = (
    REPO_ROOT
    / "mujoco"
    / "third_party"
    / "mujoco_menagerie"
    / "franka_emika_panda"
    / "block_force_scene.xml"
)


def default_block_force_cameras(
    *,
    target: tuple[float, float, float] = (0.58, 0.0, 0.03),
    fovy: float = 55.0,
) -> list[CameraSpec]:
    return [
        CameraSpec("cam_front", (0.58, -0.45, 0.32), target, fovy=fovy),
        CameraSpec("cam_back", (0.58, 0.45, 0.32), target, fovy=fovy),
        CameraSpec("cam_left", (0.18, 0.0, 0.30), target, fovy=fovy),
        CameraSpec("cam_right", (0.98, 0.0, 0.30), target, fovy=fovy),
        CameraSpec("cam_top", (0.58, 0.0, 0.75), target, fovy=fovy),
    ]


def load_model_with_cameras(
    scene_path: Path,
    cameras: list[CameraSpec],
    *,
    export_xml_path: Path | None = None,
) -> tuple[mujoco.MjModel, str]:
    xml = add_fixed_cameras_to_xml(scene_path, cameras)
    if export_xml_path is not None:
        export_xml_path.parent.mkdir(parents=True, exist_ok=True)
        export_xml_path.write_text(xml, encoding="utf-8")
    model = mujoco.MjModel.from_xml_string(xml)
    return model, xml


def set_body_freejoint_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    position: tuple[float, float, float],
    quaternion_wxyz: tuple[float, float, float, float],
) -> None:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body {body_name!r} does not exist.")

    joint_start = int(model.body_jntadr[body_id])
    joint_count = int(model.body_jntnum[body_id])
    for joint_id in range(joint_start, joint_start + joint_count):
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        qpos_address = int(model.jnt_qposadr[joint_id])
        dof_address = int(model.jnt_dofadr[joint_id])
        quat = np.asarray(quaternion_wxyz, dtype=np.float64)
        quat /= max(float(np.linalg.norm(quat)), 1.0e-12)
        data.qpos[qpos_address : qpos_address + 3] = np.asarray(position, dtype=np.float64)
        data.qpos[qpos_address + 3 : qpos_address + 7] = quat
        data.qvel[dof_address : dof_address + 6] = 0.0
        mujoco.mj_forward(model, data)
        return
    raise ValueError(f"Body {body_name!r} does not have a freejoint.")


def apply_body_point_force(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    body_name: str,
    force_world: tuple[float, float, float],
    point_offset_local: tuple[float, float, float],
) -> None:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body {body_name!r} does not exist.")

    force = np.asarray(force_world, dtype=np.float64)
    if float(np.linalg.norm(force)) <= 1.0e-12:
        data.qfrc_applied[:] = 0.0
        return

    rotation = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
    point = np.asarray(data.xpos[body_id], dtype=np.float64) + rotation @ np.asarray(point_offset_local, dtype=np.float64)
    qfrc = np.zeros(model.nv, dtype=np.float64)
    mujoco.mj_applyFT(
        model,
        data,
        force,
        np.zeros(3, dtype=np.float64),
        point,
        body_id,
        qfrc,
    )
    data.qfrc_applied[:] = qfrc
