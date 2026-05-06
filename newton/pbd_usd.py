from __future__ import annotations

from pathlib import Path
import os
import shutil
import tempfile

import numpy as np
import torch
from pxr import Gf, Usd, UsdGeom

from pbd_math import normalize_quaternion
from pbd_types import BuiltScene, RigidBodyCluster


def _to_numpy(values) -> np.ndarray:
    if torch.is_tensor(values):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _make_vec3d(values) -> Gf.Vec3d:
    v = np.asarray(_to_numpy(values), dtype=np.float64)
    return Gf.Vec3d(float(v[0]), float(v[1]), float(v[2]))


def _make_quatf(quaternion_xyzw) -> Gf.Quatf:
    quat = normalize_quaternion(quaternion_xyzw)
    quat = _to_numpy(quat)
    return Gf.Quatf(float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2]))


def _apply_display_color(gprim: UsdGeom.Gprim, color: tuple[float, float, float]) -> None:
    rgb = Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
    gprim.CreateDisplayColorPrimvar("constant").Set([rgb])


def _needs_ascii_path_fallback(path: Path) -> bool:
    try:
        str(path).encode("ascii")
    except UnicodeEncodeError:
        return True
    return False


def _create_stage(output_path: Path) -> tuple[Usd.Stage, Path | None]:
    stage_path = output_path
    temp_path = None
    if _needs_ascii_path_fallback(output_path):
        fd, temp_name = tempfile.mkstemp(
            prefix="pbd_usd_",
            suffix=output_path.suffix or ".usd",
            dir="/tmp",
        )
        os.close(fd)
        temp_path = Path(temp_name)
        temp_path.unlink(missing_ok=True)
        stage_path = temp_path

    stage = Usd.Stage.CreateNew(str(stage_path))
    if stage is None:
        raise RuntimeError(f"Failed to create USD stage at {stage_path}")
    return stage, temp_path


def _finalize_stage(stage: Usd.Stage, output_path: Path, temp_path: Path | None) -> None:
    stage.Save()
    if temp_path is None:
        return
    shutil.copyfile(temp_path, output_path)
    temp_path.unlink(missing_ok=True)


def _define_cluster_xform(
    stage: Usd.Stage,
    cluster: RigidBodyCluster,
    ) -> tuple[UsdGeom.Xform, UsdGeom.XformOp, UsdGeom.XformOp]:
    cluster_xform = UsdGeom.Xform.Define(stage, f"/Scene/{cluster.name}")
    return cluster_xform, cluster_xform.AddTranslateOp(), cluster_xform.AddOrientOp()


def _export_sphere_cluster(
    stage: Usd.Stage,
    cluster: RigidBodyCluster,
) -> tuple[UsdGeom.XformOp, UsdGeom.XformOp]:
    _, translate_op, orient_op = _define_cluster_xform(stage, cluster)
    for idx, local_point in enumerate(cluster.local_shape_positions):
        sphere = UsdGeom.Sphere.Define(stage, f"/Scene/{cluster.name}/shape_{idx:04d}")
        sphere.GetRadiusAttr().Set(float(_to_numpy(cluster.shape_radius)))
        sphere.AddTranslateOp().Set(_make_vec3d(local_point))
        _apply_display_color(sphere, cluster.display_color)
    return translate_op, orient_op


def _export_box(
    stage: Usd.Stage,
    cluster: RigidBodyCluster,
) -> tuple[UsdGeom.XformOp, UsdGeom.XformOp]:
    if cluster.box_half_extents is None:
        raise RuntimeError(f"Cluster '{cluster.name}' is missing box_half_extents for USD export")

    _, translate_op, orient_op = _define_cluster_xform(stage, cluster)
    cube = UsdGeom.Cube.Define(stage, f"/Scene/{cluster.name}/box")
    cube.GetSizeAttr().Set(2.0)
    cube.AddScaleOp().Set(_make_vec3d(cluster.box_half_extents))
    _apply_display_color(cube, cluster.display_color)
    return translate_op, orient_op


def export_scene_usd(
    scene: BuiltScene,
    output_path: Path,
    body_q_frames: list[np.ndarray | torch.Tensor] | None = None,
    fps: float = 24.0,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if body_q_frames is None:
        frames = [scene.state_0.body_q.detach().cpu().numpy().copy()]
    else:
        if len(body_q_frames) == 0:
            raise RuntimeError("body_q_frames must contain at least one frame")
        frames = [np.asarray(_to_numpy(frame), dtype=np.float32) for frame in body_q_frames]

    stage, temp_path = _create_stage(output_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(float(fps))
    stage.SetFramesPerSecond(float(fps))
    stage.SetStartTimeCode(0.0)
    stage.SetEndTimeCode(float(len(frames) - 1))
    root = UsdGeom.Xform.Define(stage, "/Scene")
    stage.SetDefaultPrim(root.GetPrim())

    xform_ops: dict[str, tuple[UsdGeom.XformOp, UsdGeom.XformOp]] = {}
    for cluster in scene.clusters:
        if cluster.collision_geometry == "sphere_cluster":
            xform_ops[cluster.name] = _export_sphere_cluster(stage, cluster)
        elif cluster.collision_geometry == "box":
            xform_ops[cluster.name] = _export_box(stage, cluster)
        else:
            raise RuntimeError(
                f"Unsupported collision geometry '{cluster.collision_geometry}' for cluster '{cluster.name}'"
            )

    for frame_idx, body_q in enumerate(frames):
        for cluster in scene.clusters:
            pose = body_q[cluster.body_id]
            translation_op, orient_op = xform_ops[cluster.name]
            translation_op.Set(_make_vec3d(pose[:3]), time=float(frame_idx))
            orient_op.Set(_make_quatf(pose[3:]), time=float(frame_idx))

    _finalize_stage(stage, output_path, temp_path)
