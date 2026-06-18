from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mujoco_pointcloud_pipeline.camera import CameraFrame, CameraIntrinsics
from mujoco_pointcloud_pipeline.pipeline import RGBDPointCloudPipeline, SegmentSpec
from mujoco_pointcloud_pipeline.pointcloud import write_ascii_ply
from mujoco_pointcloud_pipeline.segmentation import (
    ColorThresholdMaskPredictor,
    GroundedSam2MaskPredictor,
    MaskPredictor,
    SavedMaskPredictor,
)


DEFAULT_SAM2_CHECKPOINT = Path("/workspace/pgnd/weights/sam2/sam2.1_hiera_large.pt")


def _workspace_bounds(values: list[float] | tuple[float, ...] | None) -> np.ndarray | None:
    if values is None:
        return None
    if len(values) != 6:
        raise ValueError("workspace bounds must be x_min x_max y_min y_max z_min z_max")
    x_min, x_max, y_min, y_max, z_min, z_max = [float(value) for value in values]
    return np.asarray([[x_min, x_max], [y_min, y_max], [z_min, z_max]], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--episode-dir", type=Path, required=True, help="PGND-style episode_xxxx directory.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--camera", type=int, action="append", default=None, help="Camera index. Repeat to choose multiple cameras.")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--object", action="append", default=None, help="Object prompt. Repeat for multiple objects.")
    parser.add_argument(
        "--segmentation-backend",
        choices=("grounded-sam2", "saved-mask", "color-threshold"),
        default="grounded-sam2",
    )
    parser.add_argument("--mask-root", type=Path, default=None, help="Root for saved masks. Defaults to --episode-dir.")
    parser.add_argument("--grounding-model-id", type=str, default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--sam2-config", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2-checkpoint", type=Path, default=DEFAULT_SAM2_CHECKPOINT)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.30)
    parser.add_argument("--depth-scale", type=float, default=0.001, help="Scale raw depth PNG values to meters.")
    parser.add_argument("--min-depth", type=float, default=1.0e-6)
    parser.add_argument("--max-depth", type=float, default=2.0)
    parser.add_argument("--workspace-bounds", type=float, nargs=6, default=None)
    parser.add_argument("--voxel-size", type=float, default=0.003)
    parser.add_argument("--no-remove-table-plane", action="store_true")
    parser.add_argument("--plane-distance-threshold", type=float, default=0.01)
    parser.add_argument("--plane-ransac-iterations", type=int, default=1000)
    parser.add_argument("--statistical-outlier-nb-neighbors", type=int, default=25)
    parser.add_argument("--statistical-outlier-std-ratio", type=float, default=2.0)
    args = parser.parse_args()
    if args.frame_stride <= 0:
        parser.error("--frame-stride must be positive")
    if args.output_dir is None:
        args.output_dir = args.episode_dir / "rgbd_pointcloud_pipeline"
    if args.segmentation_backend == "saved-mask" and args.mask_root is None:
        args.mask_root = args.episode_dir
    return args


def build_segments(args: argparse.Namespace) -> list[SegmentSpec]:
    objects = args.object or ["object"]
    return [
        SegmentSpec(name=str(obj).strip().replace(" ", "_"), segmentation_id=idx + 1, text_prompt=str(obj))
        for idx, obj in enumerate(objects)
    ]


def build_mask_predictor(args: argparse.Namespace) -> MaskPredictor:
    if args.segmentation_backend == "color-threshold":
        return ColorThresholdMaskPredictor()
    if args.segmentation_backend == "saved-mask":
        return SavedMaskPredictor(args.mask_root)
    return GroundedSam2MaskPredictor(
        grounding_model_id=args.grounding_model_id,
        sam2_config=args.sam2_config,
        sam2_checkpoint=args.sam2_checkpoint,
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )


def load_pgnd_calibration(episode_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("opencv-python is required to load PGND rvec/tvec calibration.") from exc

    calibration_dir = episode_dir / "calibration"
    intrinsics = np.load(calibration_dir / "intrinsics.npy").astype(np.float32)
    rvecs = np.load(calibration_dir / "rvecs.npy")
    tvecs = np.load(calibration_dir / "tvecs.npy")
    world_to_camera = np.zeros((len(rvecs), 4, 4), dtype=np.float32)
    for cam_idx in range(len(rvecs)):
        world_to_camera[cam_idx, :3, :3] = cv2.Rodrigues(rvecs[cam_idx])[0]
        world_to_camera[cam_idx, :3, 3] = tvecs[cam_idx, :, 0]
        world_to_camera[cam_idx, 3, 3] = 1.0
    camera_to_world = np.linalg.inv(world_to_camera).astype(np.float32)
    return intrinsics, camera_to_world


def discover_cameras(args: argparse.Namespace) -> list[int]:
    if args.camera is not None:
        return list(args.camera)
    camera_dirs = sorted(args.episode_dir.glob("camera_*"))
    camera_ids = []
    for path in camera_dirs:
        try:
            camera_ids.append(int(path.name.split("_")[-1]))
        except ValueError:
            continue
    if not camera_ids:
        raise FileNotFoundError(f"No camera_* directories found under {args.episode_dir}")
    return camera_ids


def frame_paths(episode_dir: Path, camera_id: int) -> tuple[list[Path], list[Path]]:
    cam_dir = episode_dir / f"camera_{camera_id}"
    rgb_paths = sorted((cam_dir / "rgb").glob("*.jpg"))
    if not rgb_paths:
        rgb_paths = sorted((cam_dir / "rgb").glob("*.png"))
    depth_paths = sorted((cam_dir / "depth").glob("*.png"))
    if not rgb_paths or not depth_paths:
        raise FileNotFoundError(f"Missing rgb/depth frames in {cam_dir}")
    return rgb_paths, depth_paths


def read_rgb_depth(rgb_path: Path, depth_path: Path, depth_scale: float) -> tuple[np.ndarray, np.ndarray]:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("opencv-python is required to read RGB-D image files.") from exc
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read RGB image: {rgb_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise RuntimeError(f"Failed to read depth image: {depth_path}")
    depth = depth_raw.astype(np.float32) * float(depth_scale)
    return rgb, depth


def make_camera_frame(
    *,
    camera_id: int,
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    camera_to_world: np.ndarray,
) -> CameraFrame:
    return CameraFrame(
        camera_name=f"camera_{camera_id}",
        camera_id=int(camera_id),
        rgb=np.asarray(rgb, dtype=np.uint8),
        depth=np.asarray(depth, dtype=np.float32),
        position_world=np.asarray(camera_to_world[:3, 3], dtype=np.float64).copy(),
        rotation_world_from_camera=np.asarray(camera_to_world[:3, :3], dtype=np.float64).copy(),
        intrinsics=CameraIntrinsics(
            fx=float(intrinsic[0, 0]),
            fy=float(intrinsic[1, 1]),
            cx=float(intrinsic[0, 2]),
            cy=float(intrinsic[1, 2]),
            width=int(rgb.shape[1]),
            height=int(rgb.shape[0]),
        ),
    )


def write_capture(args: argparse.Namespace, capture, frame_index: int) -> list[dict[str, object]]:
    frame_dir = args.output_dir / f"frame_{frame_index:06d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    comments = [f"frame_index {frame_index}", "coordinates world"]
    if capture.table_plane is not None:
        coeffs = capture.table_plane.coefficients
        comments.append(
            "table_plane "
            + " ".join(f"{float(value):.9g}" for value in coeffs)
            + f" inliers {capture.table_plane.inlier_count}"
        )
    write_ascii_ply(frame_dir / "merged_segments.ply", capture.merged_cloud, comments=comments)
    for segment_name, cloud in capture.segment_clouds.items():
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in segment_name)
        write_ascii_ply(frame_dir / f"{safe_name}.ply", cloud, comments=comments + [f"segment_name {segment_name}"])
    return [track.to_json_dict() for track in capture.tracks]


def main() -> None:
    args = parse_args()
    cameras = discover_cameras(args)
    intrinsics, camera_to_world = load_pgnd_calibration(args.episode_dir)
    segments = build_segments(args)
    mask_predictor = build_mask_predictor(args)
    workspace_bounds = _workspace_bounds(args.workspace_bounds)
    pipeline = RGBDPointCloudPipeline(
        segments=segments,
        mask_predictor=mask_predictor,
        voxel_size=args.voxel_size,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        workspace_bounds=workspace_bounds,
        remove_table_plane=not bool(args.no_remove_table_plane),
        plane_distance_threshold=args.plane_distance_threshold,
        plane_ransac_iterations=args.plane_ransac_iterations,
        statistical_outlier_nb_neighbors=args.statistical_outlier_nb_neighbors,
        statistical_outlier_std_ratio=args.statistical_outlier_std_ratio,
    )

    rgb_depth_paths = {cam: frame_paths(args.episode_dir, cam) for cam in cameras}
    frame_count = min(len(paths[0]) for paths in rgb_depth_paths.values())
    frame_count = min(frame_count, min(len(paths[1]) for paths in rgb_depth_paths.values()))
    frame_end = frame_count if args.frame_end is None else min(int(args.frame_end), frame_count)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "episode_dir": str(args.episode_dir.resolve()),
        "cameras": cameras,
        "frame_start": int(args.frame_start),
        "frame_end": int(frame_end),
        "frame_stride": int(args.frame_stride),
        "segments": [segment.__dict__ for segment in segments],
        "segmentation_backend": args.segmentation_backend,
        "workspace_bounds": None if workspace_bounds is None else workspace_bounds.tolist(),
        "remove_table_plane": not bool(args.no_remove_table_plane),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    all_tracks: list[dict[str, object]] = []
    for frame_index in range(int(args.frame_start), frame_end, int(args.frame_stride)):
        camera_frames: list[CameraFrame] = []
        for cam in cameras:
            rgb_paths, depth_paths = rgb_depth_paths[cam]
            rgb, depth = read_rgb_depth(rgb_paths[frame_index], depth_paths[frame_index], args.depth_scale)
            camera_frames.append(
                make_camera_frame(
                    camera_id=cam,
                    rgb=rgb,
                    depth=depth,
                    intrinsic=intrinsics[cam],
                    camera_to_world=camera_to_world[cam],
                )
            )
        capture = pipeline.process(camera_frames, frame_index=frame_index)
        all_tracks.extend(write_capture(args, capture, frame_index))
        print(f"processed frame {frame_index}: merged_points={len(capture.merged_cloud)}")

    (args.output_dir / "tracks.json").write_text(json.dumps(all_tracks, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote RGB-D folder point-cloud outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
