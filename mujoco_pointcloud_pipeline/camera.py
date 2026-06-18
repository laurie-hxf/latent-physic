from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


@dataclass(frozen=True)
class CameraSpec:
    name: str
    position: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    fovy: float = 55.0


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass
class CameraFrame:
    camera_name: str
    camera_id: int
    rgb: np.ndarray
    depth: np.ndarray
    position_world: np.ndarray
    rotation_world_from_camera: np.ndarray
    intrinsics: CameraIntrinsics
    segmentation: np.ndarray | None = None


def _normalize(vector: np.ndarray, *, label: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-12:
        raise ValueError(f"{label} is too close to zero.")
    return vector / norm


def _format_vec(values: np.ndarray | tuple[float, ...]) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def look_at_xyaxes(
    position: tuple[float, float, float],
    target: tuple[float, float, float],
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Return MuJoCo camera xyaxes for a camera looking at target.

    MuJoCo cameras look along local -Z. The returned x/y axes are expressed in
    world coordinates and are suitable for a fixed camera's ``xyaxes`` field.
    """

    position_np = np.asarray(position, dtype=np.float64)
    target_np = np.asarray(target, dtype=np.float64)
    forward = _normalize(target_np - position_np, label="camera forward")
    up_np = _normalize(np.asarray(up, dtype=np.float64), label="camera up")

    right = np.cross(forward, up_np)
    if float(np.linalg.norm(right)) < 1.0e-8:
        fallback_up = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
        right = np.cross(forward, fallback_up)
    right = _normalize(right, label="camera right")

    camera_z = -forward
    camera_y = _normalize(np.cross(camera_z, right), label="camera y")
    return right, camera_y


def camera_to_xml_element(spec: CameraSpec) -> ET.Element:
    x_axis, y_axis = look_at_xyaxes(spec.position, spec.target, spec.up)
    attrs = {
        "name": spec.name,
        "mode": "fixed",
        "pos": _format_vec(spec.position),
        "xyaxes": _format_vec(np.concatenate([x_axis, y_axis])),
        "fovy": f"{float(spec.fovy):.12g}",
    }
    return ET.Element("camera", attrs)


def add_fixed_cameras_to_xml(scene_path: Path, cameras: list[CameraSpec]) -> str:
    root = ET.fromstring(scene_path.read_text(encoding="utf-8"))
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{scene_path} does not contain a <worldbody> element.")

    for spec in cameras:
        if worldbody.find(f"camera[@name='{spec.name}']") is not None:
            raise ValueError(f"Scene already contains a camera named {spec.name!r}.")
        worldbody.append(camera_to_xml_element(spec))

    return ET.tostring(root, encoding="unicode")


def camera_intrinsics(model: mujoco.MjModel, camera_id: int, *, width: int, height: int) -> CameraIntrinsics:
    fovy = float(model.cam_fovy[camera_id])
    fy = 0.5 * float(height) / np.tan(0.5 * np.deg2rad(fovy))
    fx = fy
    return CameraIntrinsics(
        fx=float(fx),
        fy=float(fy),
        cx=0.5 * float(width - 1),
        cy=0.5 * float(height - 1),
        width=int(width),
        height=int(height),
    )


def render_camera_frame(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    *,
    width: int,
    height: int,
    include_segmentation: bool = False,
) -> CameraFrame:
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        raise ValueError(f"Camera {camera_name!r} does not exist in the MuJoCo model.")

    renderer.disable_depth_rendering()
    renderer.disable_segmentation_rendering()
    renderer.update_scene(data, camera=camera_name)
    rgb = renderer.render().copy()

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera_name)
    depth = renderer.render().copy()

    segmentation = None
    if include_segmentation:
        renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=camera_name)
        segmentation = renderer.render().copy()
        renderer.disable_segmentation_rendering()

    return CameraFrame(
        camera_name=camera_name,
        camera_id=int(camera_id),
        rgb=np.asarray(rgb, dtype=np.uint8),
        depth=np.asarray(depth, dtype=np.float32),
        position_world=np.asarray(data.cam_xpos[camera_id], dtype=np.float64).copy(),
        rotation_world_from_camera=np.asarray(data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3).copy(),
        intrinsics=camera_intrinsics(model, camera_id, width=width, height=height),
        segmentation=None if segmentation is None else np.asarray(segmentation, dtype=np.int32),
    )


def backproject_depth_to_world(
    depth: np.ndarray,
    mask: np.ndarray,
    position_world: np.ndarray,
    rotation_world_from_camera: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    min_depth: float = 1.0e-6,
    max_depth: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project a masked MuJoCo depth image to world-space points.

    MuJoCo depth is metric distance along the camera ray. The local camera
    forward axis is -Z, so camera points use [x, y, -depth].
    """

    depth_np = np.asarray(depth, dtype=np.float32)
    mask_np = np.asarray(mask, dtype=bool)
    if depth_np.shape != mask_np.shape:
        raise ValueError(f"depth and mask shapes differ: {depth_np.shape} vs {mask_np.shape}")

    valid = mask_np & np.isfinite(depth_np) & (depth_np > float(min_depth))
    if max_depth is not None:
        valid &= depth_np < float(max_depth)
    rows, cols = np.nonzero(valid)
    if len(rows) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.int32)

    z = depth_np[rows, cols].astype(np.float64)
    x = (cols.astype(np.float64) - intrinsics.cx) * z / intrinsics.fx
    y = -(rows.astype(np.float64) - intrinsics.cy) * z / intrinsics.fy
    camera_points = np.column_stack([x, y, -z])

    rotation = np.asarray(rotation_world_from_camera, dtype=np.float64).reshape(3, 3)
    position = np.asarray(position_world, dtype=np.float64).reshape(3)
    world_points = camera_points @ rotation.T + position
    pixels = np.column_stack([rows, cols]).astype(np.int32)
    return world_points.astype(np.float32), pixels
