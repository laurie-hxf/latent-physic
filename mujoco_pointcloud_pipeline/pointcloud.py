from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PointCloud:
    points: np.ndarray
    colors: np.ndarray
    segmentation_ids: np.ndarray
    camera_ids: np.ndarray
    track_ids: np.ndarray

    @classmethod
    def empty(cls) -> "PointCloud":
        return cls(
            points=np.empty((0, 3), dtype=np.float32),
            colors=np.empty((0, 3), dtype=np.uint8),
            segmentation_ids=np.empty((0,), dtype=np.int32),
            camera_ids=np.empty((0,), dtype=np.int32),
            track_ids=np.empty((0,), dtype=np.int32),
        )

    def __len__(self) -> int:
        return int(self.points.shape[0])

    def copy(self) -> "PointCloud":
        return PointCloud(
            points=self.points.copy(),
            colors=self.colors.copy(),
            segmentation_ids=self.segmentation_ids.copy(),
            camera_ids=self.camera_ids.copy(),
            track_ids=self.track_ids.copy(),
        )


def _validate_cloud(cloud: PointCloud) -> None:
    count = len(cloud)
    if cloud.points.shape != (count, 3):
        raise ValueError(f"points must have shape (N, 3), got {cloud.points.shape}")
    if cloud.colors.shape != (count, 3):
        raise ValueError(f"colors must have shape (N, 3), got {cloud.colors.shape}")
    for name, values in (
        ("segmentation_ids", cloud.segmentation_ids),
        ("camera_ids", cloud.camera_ids),
        ("track_ids", cloud.track_ids),
    ):
        if values.shape != (count,):
            raise ValueError(f"{name} must have shape ({count},), got {values.shape}")


def concatenate_point_clouds(clouds: list[PointCloud]) -> PointCloud:
    non_empty = [cloud for cloud in clouds if len(cloud) > 0]
    if not non_empty:
        return PointCloud.empty()
    for cloud in non_empty:
        _validate_cloud(cloud)
    return PointCloud(
        points=np.concatenate([cloud.points for cloud in non_empty], axis=0).astype(np.float32),
        colors=np.concatenate([cloud.colors for cloud in non_empty], axis=0).astype(np.uint8),
        segmentation_ids=np.concatenate([cloud.segmentation_ids for cloud in non_empty], axis=0).astype(np.int32),
        camera_ids=np.concatenate([cloud.camera_ids for cloud in non_empty], axis=0).astype(np.int32),
        track_ids=np.concatenate([cloud.track_ids for cloud in non_empty], axis=0).astype(np.int32),
    )


def _mode(values: np.ndarray, *, default: int) -> int:
    if len(values) == 0:
        return int(default)
    labels, counts = np.unique(values.astype(np.int32), return_counts=True)
    return int(labels[int(np.argmax(counts))])


def voxel_downsample(cloud: PointCloud, voxel_size: float) -> PointCloud:
    if len(cloud) == 0:
        return PointCloud.empty()
    if voxel_size <= 0.0:
        return cloud.copy()
    _validate_cloud(cloud)

    keys = np.floor(cloud.points.astype(np.float64) / float(voxel_size)).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    voxel_count = int(inverse.max()) + 1
    counts = np.bincount(inverse, minlength=voxel_count).astype(np.float64)

    points = np.zeros((voxel_count, 3), dtype=np.float64)
    colors = np.zeros((voxel_count, 3), dtype=np.float64)
    np.add.at(points, inverse, cloud.points.astype(np.float64))
    np.add.at(colors, inverse, cloud.colors.astype(np.float64))
    points /= counts[:, None]
    colors /= counts[:, None]

    segmentation_ids = np.empty((voxel_count,), dtype=np.int32)
    camera_ids = np.empty((voxel_count,), dtype=np.int32)
    track_ids = np.empty((voxel_count,), dtype=np.int32)
    for voxel_idx in range(voxel_count):
        mask = inverse == voxel_idx
        segmentation_ids[voxel_idx] = _mode(cloud.segmentation_ids[mask], default=-1)
        camera_ids[voxel_idx] = _mode(cloud.camera_ids[mask], default=-1)
        track_ids[voxel_idx] = _mode(cloud.track_ids[mask], default=-1)

    return PointCloud(
        points=points.astype(np.float32),
        colors=np.rint(np.clip(colors, 0.0, 255.0)).astype(np.uint8),
        segmentation_ids=segmentation_ids,
        camera_ids=camera_ids,
        track_ids=track_ids,
    )


def write_ascii_ply(path: Path, cloud: PointCloud, *, comments: list[str] | None = None) -> None:
    _validate_cloud(cloud)
    path.parent.mkdir(parents=True, exist_ok=True)
    comments = comments or []

    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment generated_by mujoco_pointcloud_pipeline\n")
        for comment in comments:
            safe_comment = " ".join(str(comment).splitlines())
            f.write(f"comment {safe_comment}\n")
        f.write(f"element vertex {len(cloud)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property int segmentation_id\n")
        f.write("property int camera_id\n")
        f.write("property int track_id\n")
        f.write("end_header\n")
        rows = zip(
            cloud.points,
            cloud.colors,
            cloud.segmentation_ids,
            cloud.camera_ids,
            cloud.track_ids,
            strict=True,
        )
        for point, color, segmentation_id, camera_id, track_id in rows:
            f.write(
                f"{float(point[0]):.8f} {float(point[1]):.8f} {float(point[2]):.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} "
                f"{int(segmentation_id)} {int(camera_id)} {int(track_id)}\n"
            )
