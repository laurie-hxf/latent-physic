from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mujoco_pointcloud_pipeline.camera import CameraSpec
from mujoco_pointcloud_pipeline.pipeline import MultiViewPointCloudPipeline, SegmentSpec
from mujoco_pointcloud_pipeline.pointcloud import write_ascii_ply
from mujoco_pointcloud_pipeline.scene import (
    BLOCK_FORCE_SCENE_PATH,
    apply_body_point_force,
    default_block_force_cameras,
    load_model_with_cameras,
)
from mujoco_pointcloud_pipeline.segmentation import (
    ColorThresholdMaskPredictor,
    GroundedSam2MaskPredictor,
    MaskPredictor,
    SavedMaskPredictor,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "mujoco_multiview_pointcloud" / "block_force"
DEFAULT_WORKSPACE_BOUNDS = (0.30, 0.86, -0.30, 0.30, -0.02, 0.35)
DEFAULT_SAM2_CHECKPOINT = Path("/workspace/pgnd/weights/sam2/sam2.1_hiera_large.pt")


def _parse_triplet(text: str) -> tuple[float, float, float]:
    values = [float(part) for part in text.replace(",", " ").split()]
    if len(values) != 3:
        raise argparse.ArgumentTypeError(f"Expected three numeric values, got {text!r}.")
    return float(values[0]), float(values[1]), float(values[2])


def _parse_camera(text: str) -> CameraSpec:
    parts = text.split(":")
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError("Camera spec must be name:px,py,pz:tx,ty,tz[:fovy].")
    name = parts[0]
    position = _parse_triplet(parts[1])
    target = _parse_triplet(parts[2])
    fovy = float(parts[3]) if len(parts) == 4 else 55.0
    return CameraSpec(name=name, position=position, target=target, fovy=fovy)


def _workspace_bounds(values: list[float] | tuple[float, ...] | None) -> np.ndarray | None:
    if values is None:
        return None
    if len(values) != 6:
        raise ValueError("workspace bounds must be x_min x_max y_min y_max z_min z_max")
    x_min, x_max, y_min, y_max, z_min, z_max = [float(value) for value in values]
    return np.asarray([[x_min, x_max], [y_min, y_max], [z_min, z_max]], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--scene", type=Path, default=BLOCK_FORCE_SCENE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--voxel-size", type=float, default=0.003)
    parser.add_argument("--num-steps", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--export-xml", type=Path, default=None)
    parser.add_argument(
        "--camera",
        action="append",
        type=_parse_camera,
        default=None,
        help="Add a camera as name:px,py,pz:tx,ty,tz[:fovy]. Omit to use five default block cameras.",
    )
    parser.add_argument(
        "--object",
        action="append",
        default=None,
        help="Object name/prompt. Repeat for multiple objects. Defaults to 'block'.",
    )
    parser.add_argument(
        "--segmentation-backend",
        choices=("grounded-sam2", "saved-mask", "color-threshold"),
        default="grounded-sam2",
        help="Mask source. color-threshold is only for synthetic smoke tests.",
    )
    parser.add_argument("--mask-root", type=Path, default=None, help="Root directory for --segmentation-backend saved-mask.")
    parser.add_argument("--grounding-model-id", type=str, default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--sam2-config", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2-checkpoint", type=Path, default=DEFAULT_SAM2_CHECKPOINT)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.30)
    parser.add_argument("--min-depth", type=float, default=1.0e-6)
    parser.add_argument("--max-depth", type=float, default=None)
    parser.add_argument("--workspace-bounds", type=float, nargs=6, default=DEFAULT_WORKSPACE_BOUNDS)
    parser.add_argument("--no-remove-table-plane", action="store_true")
    parser.add_argument("--plane-distance-threshold", type=float, default=0.01)
    parser.add_argument("--plane-ransac-iterations", type=int, default=1000)
    parser.add_argument("--statistical-outlier-nb-neighbors", type=int, default=0)
    parser.add_argument("--statistical-outlier-std-ratio", type=float, default=2.0)
    parser.add_argument("--push-force", type=_parse_triplet, default=(0.0, 0.0, 0.0))
    parser.add_argument("--push-point-offset", type=_parse_triplet, default=(0.0, 0.0, 0.0))
    parser.add_argument("--push-body", type=str, default="push_block")
    parser.add_argument("--save-camera-arrays", action="store_true")
    parser.add_argument("--save-camera-rgb", action="store_true")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    if args.num_steps < 0:
        parser.error("--num-steps must be >= 0")
    if args.frame_stride <= 0:
        parser.error("--frame-stride must be positive")
    if args.segmentation_backend == "saved-mask" and args.mask_root is None:
        parser.error("--mask-root is required when --segmentation-backend saved-mask")
    return args


def build_segments(args: argparse.Namespace) -> list[SegmentSpec]:
    objects = args.object or ["block"]
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


def write_camera_debug(frame_dir: Path, capture, *, save_arrays: bool, save_rgb: bool) -> None:
    if not (save_arrays or save_rgb):
        return
    camera_dir = frame_dir / "cameras"
    camera_dir.mkdir(parents=True, exist_ok=True)

    imageio = None
    if save_rgb:
        try:
            import imageio.v3 as iio

            imageio = iio
        except ModuleNotFoundError:
            print("imageio is not installed; skipping camera RGB PNG output.")

    for frame in capture.camera_frames:
        prefix = camera_dir / frame.camera_name
        if save_arrays:
            np.save(prefix.with_suffix(".depth.npy"), frame.depth)
        if save_rgb and imageio is not None:
            imageio.imwrite(prefix.with_suffix(".rgb.png"), frame.rgb)


def capture_and_write(args: argparse.Namespace, pipeline: MultiViewPointCloudPipeline, frame_index: int) -> list[dict[str, object]]:
    capture = pipeline.capture(frame_index=frame_index)
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
        write_ascii_ply(
            frame_dir / f"{safe_name}.ply",
            cloud,
            comments=comments + [f"segment_name {segment_name}"],
        )
    write_camera_debug(
        frame_dir,
        capture,
        save_arrays=bool(args.save_camera_arrays),
        save_rgb=bool(args.save_camera_rgb),
    )
    return [track.to_json_dict() for track in capture.tracks]


def main() -> None:
    args = parse_args()
    cameras = args.camera or default_block_force_cameras()
    segments = build_segments(args)
    mask_predictor = build_mask_predictor(args)
    export_xml_path = args.export_xml or args.output_dir / "block_force_multiview_scene.xml"

    model, _ = load_model_with_cameras(args.scene, cameras, export_xml_path=export_xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    workspace_bounds = _workspace_bounds(args.workspace_bounds)
    metadata = {
        "scene": str(args.scene.resolve()),
        "augmented_scene_xml": str(export_xml_path.resolve()),
        "width": int(args.width),
        "height": int(args.height),
        "voxel_size": float(args.voxel_size),
        "num_steps": int(args.num_steps),
        "frame_stride": int(args.frame_stride),
        "cameras": [camera.__dict__ for camera in cameras],
        "segments": [segment.__dict__ for segment in segments],
        "segmentation_backend": args.segmentation_backend,
        "workspace_bounds": None if workspace_bounds is None else workspace_bounds.tolist(),
        "remove_table_plane": not bool(args.no_remove_table_plane),
        "plane_distance_threshold": float(args.plane_distance_threshold),
        "push_force": list(args.push_force),
        "push_point_offset": list(args.push_point_offset),
        "push_body": args.push_body,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    all_tracks: list[dict[str, object]] = []
    pipeline = MultiViewPointCloudPipeline(
        model=model,
        data=data,
        cameras=cameras,
        segments=segments,
        mask_predictor=mask_predictor,
        width=args.width,
        height=args.height,
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
    try:
        for step in range(args.num_steps + 1):
            if step % args.frame_stride == 0:
                all_tracks.extend(capture_and_write(args, pipeline, step))
            if step == args.num_steps:
                break
            apply_body_point_force(
                model,
                data,
                body_name=args.push_body,
                force_world=args.push_force,
                point_offset_local=args.push_point_offset,
            )
            mujoco.mj_step(model, data)
            data.qfrc_applied[:] = 0.0
    finally:
        pipeline.close()

    (args.output_dir / "tracks.json").write_text(
        json.dumps(all_tracks, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote real-world-style RGB-D point-cloud outputs to {args.output_dir.resolve()}")
    print(f"captured_frames={len({track['frame_index'] for track in all_tracks})} track_records={len(all_tracks)}")


if __name__ == "__main__":
    main()
