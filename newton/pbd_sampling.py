from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from pbd_io import iterate_vertex_rows
from pbd_types import PlyHeader, SegmentConfig, VoxelBucket


def points_to_voxel_keys(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.int32)

    voxel_keys = np.floor(points / voxel_size).astype(np.int32)
    return np.unique(voxel_keys, axis=0)


def voxel_keys_to_centers(voxel_keys: np.ndarray, voxel_size: float) -> np.ndarray:
    if len(voxel_keys) == 0:
        return np.empty((0, 3), dtype=np.float32)
    return ((voxel_keys.astype(np.float32) + 0.5) * np.float32(voxel_size)).astype(np.float32)


def build_occupancy_from_voxel_keys(voxel_keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(voxel_keys) == 0:
        return np.zeros((0, 0, 0), dtype=bool), np.zeros(3, dtype=np.int32)

    min_key = voxel_keys.min(axis=0)
    max_key = voxel_keys.max(axis=0)
    grid_shape = tuple((max_key - min_key + 1).tolist())
    occupied = np.zeros(grid_shape, dtype=bool)
    local_keys = voxel_keys - min_key
    occupied[local_keys[:, 0], local_keys[:, 1], local_keys[:, 2]] = True
    return occupied, min_key


def voxelize_selected_segments(
    ply_path: Path,
    header: PlyHeader,
    configs_by_seg: dict[int, SegmentConfig],
) -> dict[int, np.ndarray]:
    index = header.index
    idx_x = index["x"]
    idx_y = index["y"]
    idx_z = index["z"]
    idx_seg = index["segmentation_id"]

    voxel_buckets: dict[int, dict[tuple[int, int, int], VoxelBucket]] = {
        seg_id: {} for seg_id in configs_by_seg
    }

    for _, cols in iterate_vertex_rows(ply_path, header):
        seg_id = int(cols[idx_seg])
        config = configs_by_seg.get(seg_id)
        if config is None:
            continue

        xyz = np.array(
            [float(cols[idx_x]), float(cols[idx_y]), float(cols[idx_z])],
            dtype=np.float32,
        )

        voxel_key = tuple(int(np.floor(coord / config.voxel_size)) for coord in xyz)
        bucket = voxel_buckets[seg_id].get(voxel_key)
        if bucket is None:
            bucket = VoxelBucket()
            voxel_buckets[seg_id][voxel_key] = bucket
        bucket.update(xyz)

    sampled_positions: dict[int, np.ndarray] = {}
    for seg_id, bucket_map in voxel_buckets.items():
        ordered_keys = sorted(bucket_map.keys())
        sampled_positions[seg_id] = (
            np.array([bucket_map[key].centroid for key in ordered_keys], dtype=np.float32)
            if bucket_map
            else np.empty((0, 3), dtype=np.float32)
        )

    return sampled_positions


def fill_shell_scanlines_3d(occupied: np.ndarray) -> np.ndarray:
    if occupied.size == 0:
        return occupied

    size_x, size_y, size_z = occupied.shape
    filled_x = occupied.copy()
    filled_y = occupied.copy()
    filled_z = occupied.copy()

    for iy in range(size_y):
        for iz in range(size_z):
            xs = np.flatnonzero(occupied[:, iy, iz])
            if xs.size >= 2:
                filled_x[xs.min() : xs.max() + 1, iy, iz] = True

    for ix in range(size_x):
        for iz in range(size_z):
            ys = np.flatnonzero(occupied[ix, :, iz])
            if ys.size >= 2:
                filled_y[ix, ys.min() : ys.max() + 1, iz] = True

    for ix in range(size_x):
        for iy in range(size_y):
            zs = np.flatnonzero(occupied[ix, iy, :])
            if zs.size >= 2:
                filled_z[ix, iy, zs.min() : zs.max() + 1] = True

    support = filled_x.astype(np.int8) + filled_y.astype(np.int8) + filled_z.astype(np.int8)
    return support >= 2


def fill_2d_occupancy_holes(occupied: np.ndarray) -> np.ndarray:
    if occupied.size == 0:
        return occupied

    exterior = np.zeros_like(occupied, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    width, height = occupied.shape

    def enqueue(ix: int, iy: int) -> None:
        if ix < 0 or ix >= width or iy < 0 or iy >= height:
            return
        if occupied[ix, iy] or exterior[ix, iy]:
            return
        exterior[ix, iy] = True
        queue.append((ix, iy))

    for ix in range(width):
        enqueue(ix, 0)
        enqueue(ix, height - 1)
    for iy in range(height):
        enqueue(0, iy)
        enqueue(width - 1, iy)

    while queue:
        ix, iy = queue.popleft()
        enqueue(ix - 1, iy)
        enqueue(ix + 1, iy)
        enqueue(ix, iy - 1)
        enqueue(ix, iy + 1)

    return occupied | ~exterior


def solidify_surface_points(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if len(points) == 0:
        return points

    shell_keys = points_to_voxel_keys(points, voxel_size)
    occupied, min_key = build_occupancy_from_voxel_keys(shell_keys)
    filled = fill_shell_scanlines_3d(occupied) | occupied
    filled_keys = np.argwhere(filled).astype(np.int32) + min_key[None, :]
    return voxel_keys_to_centers(filled_keys, voxel_size)


def flatten_tabletop_points(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if len(points) == 0:
        return points

    xy_cells = np.floor(points[:, :2] / voxel_size).astype(np.int32)
    order = np.lexsort((points[:, 2], xy_cells[:, 1], xy_cells[:, 0]))
    sorted_xy = xy_cells[order]
    sorted_points = points[order]

    keep_last = np.ones(len(sorted_points), dtype=bool)
    keep_last[:-1] = np.any(sorted_xy[:-1] != sorted_xy[1:], axis=1)
    top_xy = sorted_xy[keep_last]
    top_points = sorted_points[keep_last].copy()

    tabletop_z = np.float32(np.percentile(top_points[:, 2], 95.0))
    top_points[:, 2] = tabletop_z

    min_xy = top_xy.min(axis=0)
    max_xy = top_xy.max(axis=0)
    grid_shape = tuple((max_xy - min_xy + 1).tolist())
    occupied = np.zeros(grid_shape, dtype=bool)
    local_xy = top_xy - min_xy
    occupied[local_xy[:, 0], local_xy[:, 1]] = True

    filled = fill_2d_occupancy_holes(occupied)
    hole_cells = np.argwhere(filled & ~occupied)
    if len(hole_cells) == 0:
        return top_points

    hole_xy = hole_cells + min_xy[None, :]
    hole_points = np.empty((len(hole_xy), 3), dtype=np.float32)
    hole_points[:, :2] = (hole_xy.astype(np.float32) + 0.5) * np.float32(voxel_size)
    hole_points[:, 2] = tabletop_z
    return np.concatenate([top_points, hole_points], axis=0).astype(np.float32, copy=False)
