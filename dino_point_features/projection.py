from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .dino_extractor import DinoFeatureExtractor, DinoFeatureMap


@dataclass(frozen=True)
class CameraObservation:
    """RGB-D camera observation in the same convention as mujoco_pointcloud_pipeline."""

    name: str
    rgb: np.ndarray
    depth: np.ndarray
    position_world: np.ndarray
    rotation_world_from_camera: np.ndarray
    intrinsic: np.ndarray


@dataclass(frozen=True)
class PointDinoFeatures:
    features: np.ndarray
    visibility_counts: np.ndarray
    primary_camera_ids: np.ndarray
    depth_fallback_used: np.ndarray
    camera_names: tuple[str, ...]
    model_name: str
    selected_layers: tuple[int, ...] | None
    patch_size: int
    depth_threshold: float


def camera_observation_from_frame(frame) -> CameraObservation:
    """Convert a mujoco_pointcloud_pipeline CameraFrame into a CameraObservation."""

    intrinsic = np.asarray(
        [
            [float(frame.intrinsics.fx), 0.0, float(frame.intrinsics.cx)],
            [0.0, float(frame.intrinsics.fy), float(frame.intrinsics.cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return CameraObservation(
        name=str(frame.camera_name),
        rgb=np.asarray(frame.rgb, dtype=np.uint8),
        depth=np.asarray(frame.depth, dtype=np.float32),
        position_world=np.asarray(frame.position_world, dtype=np.float32).reshape(3),
        rotation_world_from_camera=np.asarray(frame.rotation_world_from_camera, dtype=np.float32).reshape(3, 3),
        intrinsic=intrinsic,
    )


def camera_observations_from_frames(frames: Sequence[object]) -> list[CameraObservation]:
    return [camera_observation_from_frame(frame) for frame in frames]


class DinoFeatureProjector:
    """Project 3-D points into camera views and sample DINO features per point."""

    def __init__(
        self,
        extractor: DinoFeatureExtractor,
        *,
        depth_threshold: float = 0.003,
        front_depth_threshold: float | None = None,
        points_per_chunk: int = 65536,
        l2_normalize: bool = False,
        fallback_to_nearest_depth: bool = True,
    ) -> None:
        self.extractor = extractor
        self.depth_threshold = float(depth_threshold)
        self.front_depth_threshold = (
            0.5 * float(depth_threshold)
            if front_depth_threshold is None
            else float(front_depth_threshold)
        )
        self.points_per_chunk = int(points_per_chunk)
        if self.points_per_chunk <= 0:
            raise ValueError("points_per_chunk must be positive")
        self.l2_normalize = bool(l2_normalize)
        self.fallback_to_nearest_depth = bool(fallback_to_nearest_depth)

    def featurize_points(
        self,
        points_world: np.ndarray,
        observations: Sequence[CameraObservation],
    ) -> PointDinoFeatures:
        feature_map = self.extractor.encode_images([obs.rgb for obs in observations])
        return self.project_feature_map(points_world, observations, feature_map)

    def project_feature_map(
        self,
        points_world: np.ndarray,
        observations: Sequence[CameraObservation],
        feature_map: DinoFeatureMap,
    ) -> PointDinoFeatures:
        if not observations:
            raise ValueError("At least one camera observation is required.")

        points_np = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
        self._validate_observations(observations, feature_map)
        device = self.extractor.device
        feature_dim = int(feature_map.feature_dim)

        if len(points_np) == 0:
            return PointDinoFeatures(
                features=np.empty((0, feature_dim), dtype=np.float32),
                visibility_counts=np.empty((0,), dtype=np.int32),
                primary_camera_ids=np.empty((0,), dtype=np.int32),
                depth_fallback_used=np.empty((0,), dtype=bool),
                camera_names=tuple(obs.name for obs in observations),
                model_name=feature_map.model_name,
                selected_layers=feature_map.selected_layers,
                patch_size=int(feature_map.patch_size),
                depth_threshold=float(self.depth_threshold),
            )

        depths = torch.as_tensor(np.stack([obs.depth for obs in observations]), device=device, dtype=torch.float32)
        positions = torch.as_tensor(
            np.stack([obs.position_world for obs in observations]),
            device=device,
            dtype=torch.float32,
        )
        rotations = torch.as_tensor(
            np.stack([obs.rotation_world_from_camera for obs in observations]),
            device=device,
            dtype=torch.float32,
        )
        intrinsics = torch.as_tensor(
            np.stack([obs.intrinsic for obs in observations]),
            device=device,
            dtype=torch.float32,
        )
        token_grid = feature_map.features.to(device=device, dtype=torch.float32)

        count = len(points_np)
        output = np.zeros((count, feature_dim), dtype=np.float32)
        visibility_counts = np.zeros((count,), dtype=np.int32)
        primary_camera_ids = np.full((count,), -1, dtype=np.int32)
        depth_fallback_used = np.zeros((count,), dtype=bool)

        with torch.no_grad():
            for start in range(0, count, self.points_per_chunk):
                end = min(start + self.points_per_chunk, count)
                chunk = torch.as_tensor(points_np[start:end], device=device, dtype=torch.float32)
                chunk_features, chunk_counts, chunk_primary, chunk_fallback = self._project_chunk(
                    chunk,
                    depths=depths,
                    positions=positions,
                    rotations=rotations,
                    intrinsics=intrinsics,
                    token_grid=token_grid,
                    feature_map=feature_map,
                )
                output[start:end] = chunk_features.cpu().numpy().astype(np.float32, copy=False)
                visibility_counts[start:end] = chunk_counts.cpu().numpy().astype(np.int32, copy=False)
                primary_camera_ids[start:end] = chunk_primary.cpu().numpy().astype(np.int32, copy=False)
                depth_fallback_used[start:end] = chunk_fallback.cpu().numpy().astype(bool, copy=False)

        return PointDinoFeatures(
            features=output,
            visibility_counts=visibility_counts,
            primary_camera_ids=primary_camera_ids,
            depth_fallback_used=depth_fallback_used,
            camera_names=tuple(obs.name for obs in observations),
            model_name=feature_map.model_name,
            selected_layers=feature_map.selected_layers,
            patch_size=int(feature_map.patch_size),
            depth_threshold=float(self.depth_threshold),
        )

    def _project_chunk(
        self,
        points: torch.Tensor,
        *,
        depths: torch.Tensor,
        positions: torch.Tensor,
        rotations: torch.Tensor,
        intrinsics: torch.Tensor,
        token_grid: torch.Tensor,
        feature_map: DinoFeatureMap,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        camera_count, height, width = depths.shape
        _, patch_h, patch_w, feature_dim = token_grid.shape
        point_count = int(points.shape[0])

        rel = points.unsqueeze(0) - positions[:, None, :]
        pts_cam = torch.matmul(rel, rotations)
        proj_depth = -pts_cam[..., 2]
        valid_depth = torch.isfinite(proj_depth) & (proj_depth > 1.0e-6)
        safe_depth = torch.where(valid_depth, proj_depth, torch.ones_like(proj_depth))

        fx = intrinsics[:, 0, 0].unsqueeze(-1)
        fy = intrinsics[:, 1, 1].unsqueeze(-1)
        cx = intrinsics[:, 0, 2].unsqueeze(-1)
        cy = intrinsics[:, 1, 2].unsqueeze(-1)
        col = fx * pts_cam[..., 0] / safe_depth + cx
        row = cy - fy * pts_cam[..., 1] / safe_depth

        finite_pixels = torch.isfinite(col) & torch.isfinite(row)
        in_image = (
            finite_pixels
            & valid_depth
            & (col >= 0.0)
            & (col < float(width))
            & (row >= 0.0)
            & (row < float(height))
        )

        grid_depth = torch.stack(
            [
                2.0 * col / max(width - 1, 1) - 1.0,
                2.0 * row / max(height - 1, 1) - 1.0,
            ],
            dim=-1,
        )
        grid_depth = torch.nan_to_num(grid_depth, nan=2.0, posinf=2.0, neginf=-2.0)
        grid_depth = torch.where(
            in_image[..., None],
            grid_depth,
            torch.full_like(grid_depth, 2.0),
        )
        sampled_depth = F.grid_sample(
            depths[:, None, :, :],
            grid_depth.reshape(camera_count, point_count, 1, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(1).squeeze(-1)

        depth_ok = (
            (proj_depth <= sampled_depth + self.depth_threshold)
            & (proj_depth >= sampled_depth - self.front_depth_threshold)
        )
        visible = in_image & depth_ok

        patch_size = float(feature_map.patch_size)
        effective_width = float(feature_map.effective_width)
        effective_height = float(feature_map.effective_height)
        strict_feature_grid = (
            visible
            & (col >= 0.0)
            & (col < effective_width)
            & (row >= 0.0)
            & (row < effective_height)
        )
        relaxed_feature_grid = (
            in_image
            & (col >= 0.0)
            & (col < effective_width)
            & (row >= 0.0)
            & (row < effective_height)
        )

        col_eff = col.clamp(0.0, max(effective_width - 1.0, 0.0))
        row_eff = row.clamp(0.0, max(effective_height - 1.0, 0.0))
        u = (col_eff / patch_size - 0.5).clamp(0.0, max(float(patch_w - 1), 0.0))
        v = (row_eff / patch_size - 0.5).clamp(0.0, max(float(patch_h - 1), 0.0))
        grid_feat = torch.stack(
            [
                2.0 * u / max(patch_w - 1, 1) - 1.0,
                2.0 * v / max(patch_h - 1, 1) - 1.0,
            ],
            dim=-1,
        )
        grid_feat = torch.nan_to_num(grid_feat, nan=2.0, posinf=2.0, neginf=-2.0)
        grid_feat = torch.where(
            relaxed_feature_grid[..., None],
            grid_feat,
            torch.full_like(grid_feat, 2.0),
        )

        sampled_features = F.grid_sample(
            token_grid.permute(0, 3, 1, 2),
            grid_feat.reshape(camera_count, point_count, 1, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(-1).permute(0, 2, 1)

        valid_float = strict_feature_grid.to(sampled_features.dtype).unsqueeze(-1)
        feature_sum = (sampled_features * valid_float).sum(dim=0)
        counts = strict_feature_grid.sum(dim=0).to(torch.int32)
        denom = counts.clamp(min=1).to(sampled_features.dtype).unsqueeze(-1)
        features = feature_sum / denom

        fallback_used = torch.zeros((point_count,), device=points.device, dtype=torch.bool)
        any_visible = counts > 0
        if self.fallback_to_nearest_depth:
            relaxed_any = relaxed_feature_grid.any(dim=0)
            needs_fallback = (~any_visible) & relaxed_any
            if torch.any(needs_fallback):
                depth_error = torch.abs(proj_depth - sampled_depth)
                depth_error = torch.where(
                    relaxed_feature_grid,
                    depth_error,
                    torch.full_like(depth_error, float("inf")),
                )
                fallback_camera = torch.argmin(depth_error, dim=0)
                fallback_features = sampled_features[
                    fallback_camera,
                    torch.arange(point_count, device=points.device),
                ]
                features = torch.where(needs_fallback.unsqueeze(-1), fallback_features, features)
                fallback_used = needs_fallback

        if self.l2_normalize:
            features = F.normalize(features, p=2.0, dim=-1, eps=1.0e-12)

        primary = torch.argmax(strict_feature_grid.to(torch.int32), dim=0).to(torch.int32)
        if self.fallback_to_nearest_depth:
            depth_error = torch.abs(proj_depth - sampled_depth)
            depth_error = torch.where(
                relaxed_feature_grid,
                depth_error,
                torch.full_like(depth_error, float("inf")),
            )
            fallback_primary = torch.argmin(depth_error, dim=0).to(torch.int32)
            primary = torch.where(fallback_used, fallback_primary, primary)
        primary = torch.where(any_visible | fallback_used, primary, torch.full_like(primary, -1))
        assert features.shape == (point_count, feature_dim)
        return features, counts, primary, fallback_used

    @staticmethod
    def _validate_observations(
        observations: Sequence[CameraObservation],
        feature_map: DinoFeatureMap,
    ) -> None:
        image_shape = None
        for obs in observations:
            rgb = np.asarray(obs.rgb)
            depth = np.asarray(obs.depth)
            if rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError(f"Camera {obs.name} RGB must have shape (H,W,3), got {rgb.shape}")
            if depth.shape != rgb.shape[:2]:
                raise ValueError(f"Camera {obs.name} depth shape {depth.shape} does not match RGB {rgb.shape[:2]}")
            if image_shape is None:
                image_shape = rgb.shape[:2]
            elif image_shape != rgb.shape[:2]:
                raise ValueError("All camera images must have the same H,W for batched DINO extraction.")
            if np.asarray(obs.position_world).shape != (3,):
                raise ValueError(f"Camera {obs.name} position_world must have shape (3,)")
            if np.asarray(obs.rotation_world_from_camera).shape != (3, 3):
                raise ValueError(f"Camera {obs.name} rotation_world_from_camera must have shape (3,3)")
            if np.asarray(obs.intrinsic).shape != (3, 3):
                raise ValueError(f"Camera {obs.name} intrinsic must have shape (3,3)")

        if image_shape is None:
            raise ValueError("No camera observations were supplied.")
        height, width = image_shape
        if int(height) != int(feature_map.image_height) or int(width) != int(feature_map.image_width):
            raise ValueError(
                "Feature map image size does not match observations: "
                f"{feature_map.image_width}x{feature_map.image_height} vs {width}x{height}"
            )
