from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .projection import PointDinoFeatures


def save_point_feature_npz(
    path: Path,
    *,
    points: np.ndarray,
    point_features: PointDinoFeatures,
    colors: np.ndarray | None = None,
    segmentation_ids: np.ndarray | None = None,
    camera_ids: np.ndarray | None = None,
    track_ids: np.ndarray | None = None,
    extra_arrays: Mapping[str, np.ndarray] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Write one point cloud plus per-point DINO features as a compressed NPZ."""

    points_np = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    features_np = np.asarray(point_features.features, dtype=np.float32)
    if features_np.shape[0] != points_np.shape[0]:
        raise ValueError(f"features/points length mismatch: {features_np.shape[0]} vs {points_np.shape[0]}")

    payload: dict[str, np.ndarray] = {
        "points": points_np,
        "dino_features": features_np,
        "visibility_counts": np.asarray(point_features.visibility_counts, dtype=np.int32),
        "primary_camera_ids": np.asarray(point_features.primary_camera_ids, dtype=np.int32),
        "depth_fallback_used": np.asarray(point_features.depth_fallback_used, dtype=bool),
        "camera_names": np.asarray(point_features.camera_names),
        "feature_model": np.asarray(point_features.model_name),
        "selected_layers": np.asarray(
            [] if point_features.selected_layers is None else list(point_features.selected_layers),
            dtype=np.int32,
        ),
        "patch_size": np.asarray(point_features.patch_size, dtype=np.int32),
        "depth_threshold": np.asarray(point_features.depth_threshold, dtype=np.float32),
    }
    optional = {
        "colors": None if colors is None else np.asarray(colors, dtype=np.uint8).reshape(-1, 3),
        "segmentation_ids": None if segmentation_ids is None else np.asarray(segmentation_ids, dtype=np.int32),
        "camera_ids": None if camera_ids is None else np.asarray(camera_ids, dtype=np.int32),
        "track_ids": None if track_ids is None else np.asarray(track_ids, dtype=np.int32),
    }
    for key, value in optional.items():
        if value is None:
            continue
        if value.shape[0] != points_np.shape[0]:
            raise ValueError(f"{key}/points length mismatch: {value.shape[0]} vs {points_np.shape[0]}")
        payload[key] = value

    if extra_arrays is not None:
        for key, value in extra_arrays.items():
            if key in payload:
                raise ValueError(f"extra array {key!r} conflicts with a reserved NPZ key")
            value_np = np.asarray(value)
            if value_np.shape[:1] == (points_np.shape[0],):
                payload[key] = value_np
            elif value_np.ndim == 0:
                payload[key] = value_np
            else:
                raise ValueError(
                    f"extra array {key!r} must be scalar or have first dimension {points_np.shape[0]}, "
                    f"got {value_np.shape}"
                )

    if metadata is not None:
        payload["metadata_json"] = np.asarray(json.dumps(dict(metadata), sort_keys=True))

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def save_feature_metadata(path: Path, metadata: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(metadata), indent=2, sort_keys=True), encoding="utf-8")


def read_ascii_ply_vertices(path: Path) -> dict[str, np.ndarray]:
    """Read vertex properties from a simple ASCII PLY file.

    This is intentionally small and supports the PLY files written by this
    repository. Binary PLY is not supported.
    """

    with path.open("r", encoding="utf-8") as f:
        first = f.readline().strip()
        if first != "ply":
            raise ValueError(f"{path} is not a PLY file")
        fmt = f.readline().strip()
        if fmt != "format ascii 1.0":
            raise ValueError(f"{path} is not an ASCII PLY file")

        vertex_count = None
        properties: list[tuple[str, str]] = []
        in_vertex = False
        for line in f:
            stripped = line.strip()
            if stripped == "end_header":
                break
            if stripped.startswith("element "):
                parts = stripped.split()
                in_vertex = len(parts) == 3 and parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
                continue
            if in_vertex and stripped.startswith("property "):
                parts = stripped.split()
                if len(parts) != 3:
                    raise ValueError(f"Unsupported PLY property line: {stripped}")
                properties.append((parts[2], parts[1]))
        else:
            raise ValueError(f"{path} is missing end_header")

        if vertex_count is None:
            raise ValueError(f"{path} does not declare a vertex element")
        rows = []
        for _ in range(vertex_count):
            line = f.readline()
            if not line:
                raise ValueError(f"{path} ended before all vertices were read")
            rows.append(line.split())

    data = np.asarray(rows, dtype=np.float64)
    if data.shape != (vertex_count, len(properties)):
        raise ValueError(f"Unexpected vertex table shape {data.shape}; expected {(vertex_count, len(properties))}")

    result: dict[str, np.ndarray] = {}
    for column, (name, ply_type) in enumerate(properties):
        values = data[:, column]
        if ply_type in {"uchar", "uint8"}:
            result[name] = values.astype(np.uint8)
        elif ply_type in {"int", "int32"}:
            result[name] = values.astype(np.int32)
        else:
            result[name] = values.astype(np.float32)
    return result
