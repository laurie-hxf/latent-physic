from __future__ import annotations

from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from .camera import CameraFrame, CameraSpec, backproject_depth_to_world, render_camera_frame
from .geometry import (
    PlaneModel,
    filter_cloud_by_mask,
    fit_plane_ransac,
    make_point_cloud,
    remove_plane_from_cloud,
    workspace_mask,
)
from .pointcloud import PointCloud, concatenate_point_clouds, voxel_downsample
from .segmentation import MaskPredictor, combine_instance_masks


@dataclass(frozen=True)
class SegmentSpec:
    name: str
    segmentation_id: int
    text_prompt: str


@dataclass
class SegmentTrack:
    track_id: int
    segment_name: str
    segmentation_id: int
    frame_index: int
    point_count: int
    centroid_world: list[float] | None
    bbox_min_world: list[float] | None
    bbox_max_world: list[float] | None

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class CapturedFrame:
    frame_index: int
    camera_frames: list[CameraFrame]
    segment_clouds: dict[str, PointCloud]
    merged_cloud: PointCloud
    tracks: list[SegmentTrack]
    table_plane: PlaneModel | None = None


class PointCloudTracker:
    def __init__(self) -> None:
        self._next_track_id = 1
        self._track_ids_by_segment: dict[str, int] = {}

    def update(
        self,
        *,
        frame_index: int,
        segments: list[SegmentSpec],
        clouds_by_segment: dict[str, PointCloud],
    ) -> list[SegmentTrack]:
        tracks: list[SegmentTrack] = []
        for segment in segments:
            cloud = clouds_by_segment.get(segment.name, PointCloud.empty())
            if segment.name not in self._track_ids_by_segment:
                self._track_ids_by_segment[segment.name] = self._next_track_id
                self._next_track_id += 1
            track_id = self._track_ids_by_segment[segment.name]

            if len(cloud) > 0:
                centroid = cloud.points.mean(axis=0).astype(float).tolist()
                bbox_min = cloud.points.min(axis=0).astype(float).tolist()
                bbox_max = cloud.points.max(axis=0).astype(float).tolist()
            else:
                centroid = None
                bbox_min = None
                bbox_max = None

            tracks.append(
                SegmentTrack(
                    track_id=track_id,
                    segment_name=segment.name,
                    segmentation_id=int(segment.segmentation_id),
                    frame_index=int(frame_index),
                    point_count=len(cloud),
                    centroid_world=centroid,
                    bbox_min_world=bbox_min,
                    bbox_max_world=bbox_max,
                )
            )
        return tracks


def _all_valid_depth_cloud(
    frame: CameraFrame,
    *,
    camera_index: int,
    min_depth: float,
    max_depth: float | None,
    workspace_bounds: np.ndarray | None,
) -> PointCloud:
    mask = np.isfinite(frame.depth) & (frame.depth > float(min_depth))
    if max_depth is not None:
        mask &= frame.depth < float(max_depth)
    points, pixels = backproject_depth_to_world(
        frame.depth,
        mask,
        frame.position_world,
        frame.rotation_world_from_camera,
        frame.intrinsics,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    if len(points) == 0:
        return PointCloud.empty()
    keep = workspace_mask(points, workspace_bounds)
    points = points[keep]
    pixels = pixels[keep]
    if len(points) == 0:
        return PointCloud.empty()
    colors = frame.rgb[pixels[:, 0], pixels[:, 1]].astype(np.uint8)
    return make_point_cloud(points, colors, segmentation_id=0, camera_id=camera_index)


def _masked_segment_cloud(
    frame: CameraFrame,
    *,
    segment: SegmentSpec,
    mask: np.ndarray,
    camera_index: int,
    min_depth: float,
    max_depth: float | None,
    workspace_bounds: np.ndarray | None,
) -> PointCloud:
    depth_mask = np.asarray(mask, dtype=bool)
    depth_mask &= np.isfinite(frame.depth) & (frame.depth > float(min_depth))
    if max_depth is not None:
        depth_mask &= frame.depth < float(max_depth)
    points, pixels = backproject_depth_to_world(
        frame.depth,
        depth_mask,
        frame.position_world,
        frame.rotation_world_from_camera,
        frame.intrinsics,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    if len(points) == 0:
        return PointCloud.empty()
    keep = workspace_mask(points, workspace_bounds)
    points = points[keep]
    pixels = pixels[keep]
    if len(points) == 0:
        return PointCloud.empty()
    colors = frame.rgb[pixels[:, 0], pixels[:, 1]].astype(np.uint8)
    return make_point_cloud(points, colors, segmentation_id=segment.segmentation_id, camera_id=camera_index)


class RGBDPointCloudPipeline:
    def __init__(
        self,
        *,
        segments: list[SegmentSpec],
        mask_predictor: MaskPredictor,
        voxel_size: float = 0.003,
        min_depth: float = 1.0e-6,
        max_depth: float | None = None,
        workspace_bounds: np.ndarray | None = None,
        remove_table_plane: bool = True,
        plane_distance_threshold: float = 0.01,
        plane_ransac_iterations: int = 1000,
        statistical_outlier_nb_neighbors: int = 0,
        statistical_outlier_std_ratio: float = 2.0,
    ) -> None:
        if not segments:
            raise ValueError("At least one segment is required.")
        self.segments = list(segments)
        self.mask_predictor = mask_predictor
        self.voxel_size = float(voxel_size)
        self.min_depth = float(min_depth)
        self.max_depth = None if max_depth is None else float(max_depth)
        self.workspace_bounds = None if workspace_bounds is None else np.asarray(workspace_bounds, dtype=np.float32).reshape(3, 2)
        self.remove_table_plane = bool(remove_table_plane)
        self.plane_distance_threshold = float(plane_distance_threshold)
        self.plane_ransac_iterations = int(plane_ransac_iterations)
        self.statistical_outlier_nb_neighbors = int(statistical_outlier_nb_neighbors)
        self.statistical_outlier_std_ratio = float(statistical_outlier_std_ratio)
        self.tracker = PointCloudTracker()

    def _remove_statistical_outliers(self, cloud: PointCloud) -> PointCloud:
        if self.statistical_outlier_nb_neighbors <= 0 or len(cloud) == 0:
            return cloud
        try:
            import open3d as o3d
        except ModuleNotFoundError:
            return cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud.points)
        _, inliers = pcd.remove_statistical_outlier(
            nb_neighbors=self.statistical_outlier_nb_neighbors,
            std_ratio=self.statistical_outlier_std_ratio,
        )
        mask = np.zeros((len(cloud),), dtype=bool)
        mask[np.asarray(inliers, dtype=np.int64)] = True
        return filter_cloud_by_mask(cloud, mask)

    def process(self, camera_frames: list[CameraFrame], *, frame_index: int = 0) -> CapturedFrame:
        if not camera_frames:
            raise ValueError("At least one RGB-D frame is required.")

        scene_clouds = [
            _all_valid_depth_cloud(
                frame,
                camera_index=camera_index,
                min_depth=self.min_depth,
                max_depth=self.max_depth,
                workspace_bounds=self.workspace_bounds,
            )
            for camera_index, frame in enumerate(camera_frames)
        ]
        scene_cloud = concatenate_point_clouds(scene_clouds)
        table_plane = None
        if self.remove_table_plane and len(scene_cloud) >= 3:
            table_plane = fit_plane_ransac(
                scene_cloud.points,
                distance_threshold=self.plane_distance_threshold,
                num_iterations=self.plane_ransac_iterations,
            )

        per_camera_segment_clouds: dict[str, list[PointCloud]] = {
            segment.name: []
            for segment in self.segments
        }
        for camera_index, frame in enumerate(camera_frames):
            image_shape = frame.rgb.shape[:2]
            for segment in self.segments:
                instances = self.mask_predictor.predict(
                    frame.rgb,
                    segment.text_prompt,
                    frame_index=frame_index,
                    camera_name=frame.camera_name,
                )
                mask = combine_instance_masks(instances, image_shape)
                cloud = _masked_segment_cloud(
                    frame,
                    segment=segment,
                    mask=mask,
                    camera_index=camera_index,
                    min_depth=self.min_depth,
                    max_depth=self.max_depth,
                    workspace_bounds=self.workspace_bounds,
                )
                per_camera_segment_clouds[segment.name].append(cloud)

        segment_clouds: dict[str, PointCloud] = {}
        for segment in self.segments:
            fused = concatenate_point_clouds(per_camera_segment_clouds[segment.name])
            if self.remove_table_plane and table_plane is not None:
                fused = remove_plane_from_cloud(
                    fused,
                    table_plane,
                    distance_threshold=self.plane_distance_threshold,
                )
            if len(fused) > 0 and self.voxel_size > 0.0:
                fused = voxel_downsample(fused, self.voxel_size)
            fused = self._remove_statistical_outliers(fused)
            segment_clouds[segment.name] = fused

        tracks = self.tracker.update(
            frame_index=frame_index,
            segments=self.segments,
            clouds_by_segment=segment_clouds,
        )
        track_ids_by_segment = {track.segment_name: int(track.track_id) for track in tracks}
        for segment_name, cloud in segment_clouds.items():
            if len(cloud) > 0:
                cloud.track_ids[:] = track_ids_by_segment[segment_name]

        merged = concatenate_point_clouds([segment_clouds[segment.name] for segment in self.segments])
        return CapturedFrame(
            frame_index=int(frame_index),
            camera_frames=camera_frames,
            segment_clouds=segment_clouds,
            merged_cloud=merged,
            tracks=tracks,
            table_plane=table_plane,
        )


class MultiViewPointCloudPipeline:
    """MuJoCo RGB-D source that intentionally avoids MuJoCo segmentation labels."""

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        cameras: list[CameraSpec],
        segments: list[SegmentSpec],
        mask_predictor: MaskPredictor,
        width: int = 320,
        height: int = 240,
        voxel_size: float = 0.003,
        min_depth: float = 1.0e-6,
        max_depth: float | None = None,
        workspace_bounds: np.ndarray | None = None,
        remove_table_plane: bool = True,
        plane_distance_threshold: float = 0.01,
        plane_ransac_iterations: int = 1000,
        statistical_outlier_nb_neighbors: int = 0,
        statistical_outlier_std_ratio: float = 2.0,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive.")
        if not cameras:
            raise ValueError("At least one camera is required.")

        self.model = model
        self.data = data
        self.cameras = list(cameras)
        self.width = int(width)
        self.height = int(height)
        self.renderer = mujoco.Renderer(model, height=self.height, width=self.width)
        self.rgbd_pipeline = RGBDPointCloudPipeline(
            segments=segments,
            mask_predictor=mask_predictor,
            voxel_size=voxel_size,
            min_depth=min_depth,
            max_depth=max_depth,
            workspace_bounds=workspace_bounds,
            remove_table_plane=remove_table_plane,
            plane_distance_threshold=plane_distance_threshold,
            plane_ransac_iterations=plane_ransac_iterations,
            statistical_outlier_nb_neighbors=statistical_outlier_nb_neighbors,
            statistical_outlier_std_ratio=statistical_outlier_std_ratio,
        )

    def close(self) -> None:
        self.renderer.close()

    def capture(self, frame_index: int = 0) -> CapturedFrame:
        mujoco.mj_forward(self.model, self.data)
        camera_frames: list[CameraFrame] = []
        for spec in self.cameras:
            frame = render_camera_frame(
                self.renderer,
                self.model,
                self.data,
                spec.name,
                width=self.width,
                height=self.height,
                include_segmentation=False,
            )
            camera_frames.append(frame)
        return self.rgbd_pipeline.process(camera_frames, frame_index=frame_index)
