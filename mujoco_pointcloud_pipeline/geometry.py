from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pointcloud import PointCloud


@dataclass(frozen=True)
class PlaneModel:
    normal: np.ndarray
    offset: float
    inlier_count: int

    @property
    def coefficients(self) -> np.ndarray:
        return np.asarray(
            [self.normal[0], self.normal[1], self.normal[2], self.offset],
            dtype=np.float32,
        )


def workspace_mask(points: np.ndarray, bounds: np.ndarray | None) -> np.ndarray:
    points_np = np.asarray(points, dtype=np.float32)
    if bounds is None:
        return np.ones((len(points_np),), dtype=bool)
    bounds_np = np.asarray(bounds, dtype=np.float32).reshape(3, 2)
    return np.all((points_np >= bounds_np[:, 0]) & (points_np <= bounds_np[:, 1]), axis=1)


def filter_cloud_by_mask(cloud: PointCloud, mask: np.ndarray) -> PointCloud:
    mask_np = np.asarray(mask, dtype=bool)
    if mask_np.shape != (len(cloud),):
        raise ValueError(f"mask must have shape ({len(cloud)},), got {mask_np.shape}")
    return PointCloud(
        points=cloud.points[mask_np],
        colors=cloud.colors[mask_np],
        segmentation_ids=cloud.segmentation_ids[mask_np],
        camera_ids=cloud.camera_ids[mask_np],
        track_ids=cloud.track_ids[mask_np],
    )


def fit_plane_ransac(
    points: np.ndarray,
    *,
    distance_threshold: float = 0.01,
    num_iterations: int = 1000,
    max_sample_points: int = 50000,
    seed: int = 0,
) -> PlaneModel | None:
    points_np = np.asarray(points, dtype=np.float32)
    finite = np.isfinite(points_np).all(axis=1)
    points_np = points_np[finite]
    if len(points_np) < 3:
        return None

    rng = np.random.default_rng(seed)
    if len(points_np) > max_sample_points:
        sample_indices = rng.choice(len(points_np), size=max_sample_points, replace=False)
        sample_points = points_np[sample_indices]
    else:
        sample_points = points_np

    best_normal = None
    best_offset = 0.0
    best_count = -1
    for _ in range(max(int(num_iterations), 1)):
        ids = rng.choice(len(sample_points), size=3, replace=False)
        p0, p1, p2 = sample_points[ids]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = float(np.linalg.norm(normal))
        if norm < 1.0e-8:
            continue
        normal = normal / norm
        offset = -float(np.dot(normal, p0))
        distances = np.abs(sample_points @ normal + offset)
        count = int(np.count_nonzero(distances <= float(distance_threshold)))
        if count > best_count:
            best_normal = normal.astype(np.float32)
            best_offset = float(offset)
            best_count = count

    if best_normal is None:
        return None
    return PlaneModel(normal=best_normal, offset=best_offset, inlier_count=best_count)


def remove_plane_from_cloud(
    cloud: PointCloud,
    plane: PlaneModel | None,
    *,
    distance_threshold: float,
) -> PointCloud:
    if plane is None or len(cloud) == 0:
        return cloud.copy()
    distances = np.abs(cloud.points @ plane.normal.astype(np.float32) + np.float32(plane.offset))
    return filter_cloud_by_mask(cloud, distances > float(distance_threshold))


def make_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    segmentation_id: int,
    camera_id: int,
    track_id: int = -1,
) -> PointCloud:
    points_np = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors_np = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if len(colors_np) != len(points_np):
        raise ValueError(f"points and colors have different lengths: {len(points_np)} vs {len(colors_np)}")
    return PointCloud(
        points=points_np,
        colors=colors_np,
        segmentation_ids=np.full((len(points_np),), int(segmentation_id), dtype=np.int32),
        camera_ids=np.full((len(points_np),), int(camera_id), dtype=np.int32),
        track_ids=np.full((len(points_np),), int(track_id), dtype=np.int32),
    )
