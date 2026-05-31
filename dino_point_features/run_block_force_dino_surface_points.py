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
NEWTON_DIR = REPO_ROOT / "newton"
for path in (REPO_ROOT, NEWTON_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dino_point_features.dino_extractor import DinoFeatureExtractor
from dino_point_features.io import save_feature_metadata, save_point_feature_npz
from dino_point_features.projection import DinoFeatureProjector, PointDinoFeatures, camera_observations_from_frames
from mujoco_pointcloud_pipeline.camera import CameraSpec, render_camera_frame
from mujoco_pointcloud_pipeline.pointcloud import PointCloud, write_ascii_ply
from mujoco_pointcloud_pipeline.scene import (
    BLOCK_FORCE_SCENE_PATH,
    apply_body_point_force,
    default_block_force_cameras,
    load_model_with_cameras,
)
from newton_surface_points_demo import sample_box_surface_points


DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "mujoco_dino_point_features" / "block_force_surface_points"


def _parse_triplet(text: str) -> tuple[float, float, float]:
    values = [float(part) for part in text.replace(",", " ").split()]
    if len(values) != 3:
        raise argparse.ArgumentTypeError(f"Expected three numeric values, got {text!r}.")
    return float(values[0]), float(values[1]), float(values[2])


def _parse_camera(text: str) -> CameraSpec:
    parts = text.split(":")
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError("Camera spec must be name:px,py,pz:tx,ty,tz[:fovy].")
    fovy = float(parts[3]) if len(parts) == 4 else 55.0
    return CameraSpec(name=parts[0], position=_parse_triplet(parts[1]), target=_parse_triplet(parts[2]), fovy=fovy)


def _parse_layers(text: str | None) -> tuple[int, ...] | None:
    if text is None:
        return None
    stripped = text.strip()
    if not stripped or stripped.lower() in {"none", "default"}:
        return None
    return tuple(int(part) for part in stripped.replace(",", " ").split())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--scene", type=Path, default=BLOCK_FORCE_SCENE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
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
    parser.add_argument("--box-body", type=str, default="push_block")
    parser.add_argument("--box-half-extents", type=float, nargs=3, default=(0.1, 0.05, 0.025))
    parser.add_argument("--box-mass", type=float, default=1.0)
    parser.add_argument("--surface-point-spacing", type=float, default=0.01)
    parser.add_argument("--allow-zero-split-x", action="store_true")
    parser.add_argument(
        "--bottom-feature-source",
        choices=("top-face", "projected"),
        default="top-face",
        help=(
            "How to assign features to bottom contact-face points. "
            "top-face copies the feature from the top point with the same local x/y; "
            "projected keeps direct projection/fallback results."
        ),
    )
    parser.add_argument("--push-force", type=_parse_triplet, default=(0.0, 0.0, 0.0))
    parser.add_argument("--push-point-offset", type=_parse_triplet, default=(0.0, 0.0, 0.0))

    parser.add_argument("--dino-model", type=str, default="dinov2_vits14")
    parser.add_argument("--selected-layers", type=str, default="2,5,8,11")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dinov3-repo", type=Path, default=REPO_ROOT / "PointWorld" / "third_party" / "dinov3")
    parser.add_argument("--dinov3-weights", type=Path, default=None)
    parser.add_argument("--torchhub-repo", type=str, default="facebookresearch/dinov2")
    parser.add_argument("--dino-use-half", action="store_true")
    parser.add_argument("--depth-threshold", type=float, default=0.003)
    parser.add_argument("--front-depth-threshold", type=float, default=None)
    parser.add_argument("--points-per-chunk", type=int, default=65536)
    parser.add_argument("--l2-normalize-features", action="store_true")
    parser.add_argument(
        "--no-depth-fallback",
        action="store_true",
        help="Leave depth-inconsistent or occluded surface points as zero features instead of nearest-depth fallback.",
    )
    return parser.parse_args()


def _body_points_world(model: mujoco.MjModel, data: mujoco.MjData, body_name: str, local_points: np.ndarray) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body {body_name!r} does not exist.")
    rotation = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
    position = np.asarray(data.xpos[body_id], dtype=np.float64).reshape(3)
    return (np.asarray(local_points, dtype=np.float64) @ rotation.T + position).astype(np.float32)


def _surface_colors(local_points: np.ndarray) -> np.ndarray:
    colors = np.zeros((len(local_points), 3), dtype=np.uint8)
    colors[local_points[:, 0] < 0.0] = np.asarray([224, 71, 56], dtype=np.uint8)
    colors[local_points[:, 0] > 0.0] = np.asarray([46, 115, 242], dtype=np.uint8)
    colors[np.isclose(local_points[:, 0], 0.0)] = np.asarray([170, 170, 170], dtype=np.uint8)
    return colors


def _side_ids(local_points: np.ndarray) -> np.ndarray:
    side_ids = np.zeros((len(local_points),), dtype=np.int32)
    side_ids[local_points[:, 0] < 0.0] = 1
    side_ids[local_points[:, 0] > 0.0] = 2
    return side_ids


def _face_ids(local_points: np.ndarray, half_extents: np.ndarray) -> np.ndarray:
    face_ids = np.full((len(local_points),), -1, dtype=np.int32)
    labels = [
        (0, -float(half_extents[0]), 0),
        (0, float(half_extents[0]), 1),
        (1, -float(half_extents[1]), 2),
        (1, float(half_extents[1]), 3),
        (2, -float(half_extents[2]), 4),
        (2, float(half_extents[2]), 5),
    ]
    for axis, value, label in labels:
        face_ids[np.isclose(local_points[:, axis], value, atol=1.0e-7)] = label
    return face_ids


def _render_camera_frames(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cameras: list[CameraSpec],
    *,
    width: int,
    height: int,
) -> list[object]:
    renderer = mujoco.Renderer(model, height=int(height), width=int(width))
    try:
        mujoco.mj_forward(model, data)
        return [
            render_camera_frame(
                renderer,
                model,
                data,
                camera.name,
                width=int(width),
                height=int(height),
                include_segmentation=False,
            )
            for camera in cameras
        ]
    finally:
        renderer.close()


def _xy_key(point: np.ndarray) -> tuple[float, float]:
    xy = np.round(np.asarray(point[:2], dtype=np.float64), decimals=9)
    return float(xy[0]), float(xy[1])


def _copy_bottom_features_from_top(
    point_features: PointDinoFeatures,
    local_points: np.ndarray,
    half_extents: np.ndarray,
) -> tuple[PointDinoFeatures, np.ndarray]:
    local_np = np.asarray(local_points, dtype=np.float32)
    bottom = np.isclose(local_np[:, 2], -float(half_extents[2]), atol=1.0e-7)
    top = np.isclose(local_np[:, 2], float(half_extents[2]), atol=1.0e-7)
    copied = np.zeros((len(local_np),), dtype=bool)
    if not np.any(bottom):
        return point_features, copied

    top_by_xy = {_xy_key(local_np[index]): int(index) for index in np.nonzero(top)[0]}
    bottom_indices = np.nonzero(bottom)[0]
    top_indices = []
    missing: list[int] = []
    for bottom_index in bottom_indices:
        top_index = top_by_xy.get(_xy_key(local_np[bottom_index]))
        if top_index is None:
            missing.append(int(bottom_index))
        else:
            top_indices.append(top_index)
    if missing:
        raise RuntimeError(f"Could not find matching top-face points for {len(missing)} bottom points.")

    bottom_indices_np = bottom_indices.astype(np.int64)
    top_indices_np = np.asarray(top_indices, dtype=np.int64)
    features = np.asarray(point_features.features, dtype=np.float32).copy()
    visibility_counts = np.asarray(point_features.visibility_counts, dtype=np.int32).copy()
    primary_camera_ids = np.asarray(point_features.primary_camera_ids, dtype=np.int32).copy()
    depth_fallback_used = np.asarray(point_features.depth_fallback_used, dtype=bool).copy()

    features[bottom_indices_np] = features[top_indices_np]
    visibility_counts[bottom_indices_np] = 0
    primary_camera_ids[bottom_indices_np] = primary_camera_ids[top_indices_np]
    depth_fallback_used[bottom_indices_np] = True
    copied[bottom_indices_np] = True

    return (
        PointDinoFeatures(
            features=features,
            visibility_counts=visibility_counts,
            primary_camera_ids=primary_camera_ids,
            depth_fallback_used=depth_fallback_used,
            camera_names=point_features.camera_names,
            model_name=point_features.model_name,
            selected_layers=point_features.selected_layers,
            patch_size=point_features.patch_size,
            depth_threshold=point_features.depth_threshold,
        ),
        copied,
    )


def _write_frame(
    *,
    args: argparse.Namespace,
    frame_index: int,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cameras: list[CameraSpec],
    local_surface_points: np.ndarray,
    point_masses: np.ndarray,
    extractor: DinoFeatureExtractor,
    projector: DinoFeatureProjector,
    feature_metadata: dict[str, object],
) -> dict[str, object]:
    frame_dir = args.output_dir / f"frame_{frame_index:06d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    half_extents = np.asarray(args.box_half_extents, dtype=np.float32)

    points_world = _body_points_world(model, data, args.box_body, local_surface_points)
    colors = _surface_colors(local_surface_points)
    side_ids = _side_ids(local_surface_points)
    face_ids = _face_ids(local_surface_points, half_extents)

    frames = _render_camera_frames(model, data, cameras, width=args.width, height=args.height)
    observations = camera_observations_from_frames(frames)
    feature_map = extractor.encode_images([obs.rgb for obs in observations])
    point_features = projector.project_feature_map(points_world, observations, feature_map)
    bottom_feature_copied_from_top = np.zeros((len(points_world),), dtype=bool)
    if args.bottom_feature_source == "top-face":
        point_features, bottom_feature_copied_from_top = _copy_bottom_features_from_top(
            point_features,
            local_surface_points,
            half_extents,
        )

    ply_cloud = PointCloud(
        points=points_world,
        colors=colors,
        segmentation_ids=side_ids,
        camera_ids=point_features.primary_camera_ids.astype(np.int32),
        track_ids=np.ones((len(points_world),), dtype=np.int32),
    )
    ply_path = frame_dir / "newton_surface_points.ply"
    write_ascii_ply(
        ply_path,
        ply_cloud,
        comments=[
            "coordinates world",
            f"surface_point_spacing {float(args.surface_point_spacing):.9g}",
            "segmentation_id 1 left half, 2 right half",
        ],
    )

    npz_path = frame_dir / "newton_surface_points_dino_features.npz"
    save_point_feature_npz(
        npz_path,
        points=points_world,
        point_features=point_features,
        colors=colors,
        segmentation_ids=side_ids,
        camera_ids=point_features.primary_camera_ids,
        track_ids=np.ones((len(points_world),), dtype=np.int32),
        extra_arrays={
            "local_points": local_surface_points.astype(np.float32),
            "point_masses": point_masses.astype(np.float32),
            "side_ids": side_ids,
            "face_ids": face_ids,
            "is_bottom_contact_face": np.isclose(local_surface_points[:, 2], -half_extents[2], atol=1.0e-7),
            "bottom_feature_copied_from_top": bottom_feature_copied_from_top,
        },
        metadata={
            **feature_metadata,
            "surface_point_spacing": float(args.surface_point_spacing),
            "box_half_extents": [float(v) for v in half_extents],
            "box_body": args.box_body,
            "coordinates": "world",
            "local_points_key": "local_points",
            "bottom_feature_source": args.bottom_feature_source,
            "side_ids": {"left": 1, "right": 2},
            "face_ids": {
                "x_min": 0,
                "x_max": 1,
                "y_min": 2,
                "y_max": 3,
                "z_min_bottom": 4,
                "z_max_top": 5,
            },
        },
    )

    visibility = point_features.visibility_counts
    fallback = point_features.depth_fallback_used
    assigned = (visibility > 0) | fallback
    bottom = np.isclose(local_surface_points[:, 2], -half_extents[2], atol=1.0e-7)
    summary = {
        "frame_index": int(frame_index),
        "ply": str(ply_path),
        "npz": str(npz_path),
        "point_count": int(len(points_world)),
        "feature_dim": int(point_features.features.shape[-1]),
        "visible_feature_count": int(np.count_nonzero(visibility > 0)),
        "depth_fallback_count": int(np.count_nonzero(fallback)),
        "assigned_feature_count": int(np.count_nonzero(assigned)),
        "left_count": int(np.count_nonzero(local_surface_points[:, 0] < 0.0)),
        "right_count": int(np.count_nonzero(local_surface_points[:, 0] > 0.0)),
        "bottom_count": int(np.count_nonzero(bottom)),
        "bottom_visible_feature_count": int(np.count_nonzero((visibility > 0) & bottom)),
        "bottom_depth_fallback_count": int(np.count_nonzero(fallback & bottom)),
        "bottom_assigned_feature_count": int(np.count_nonzero(assigned & bottom)),
        "bottom_feature_copied_from_top_count": int(np.count_nonzero(bottom_feature_copied_from_top)),
    }
    return summary


def main() -> None:
    args = parse_args()
    cameras = args.camera or default_block_force_cameras()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_xml_path = args.export_xml or args.output_dir / "block_force_surface_points_scene.xml"

    local_surface_points, point_masses = sample_box_surface_points(
        np.asarray(args.box_half_extents, dtype=np.float32),
        spacing=float(args.surface_point_spacing),
        total_mass=float(args.box_mass),
        avoid_zero_x=not bool(args.allow_zero_split_x),
    )

    extractor = DinoFeatureExtractor(
        model_name=args.dino_model,
        device=args.device,
        selected_layers=_parse_layers(args.selected_layers),
        dinov3_repo=args.dinov3_repo,
        dinov3_weights=args.dinov3_weights,
        torchhub_repo=args.torchhub_repo,
        use_half=bool(args.dino_use_half),
    )
    projector = DinoFeatureProjector(
        extractor,
        depth_threshold=args.depth_threshold,
        front_depth_threshold=args.front_depth_threshold,
        points_per_chunk=args.points_per_chunk,
        l2_normalize=bool(args.l2_normalize_features),
        fallback_to_nearest_depth=not bool(args.no_depth_fallback),
    )

    model, _ = load_model_with_cameras(args.scene, cameras, export_xml_path=export_xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    run_metadata = {
        "scene": str(args.scene.resolve()),
        "augmented_scene_xml": str(export_xml_path.resolve()),
        "width": int(args.width),
        "height": int(args.height),
        "num_steps": int(args.num_steps),
        "frame_stride": int(args.frame_stride),
        "cameras": [camera.__dict__ for camera in cameras],
        "box_body": args.box_body,
        "box_half_extents": [float(v) for v in args.box_half_extents],
        "box_mass": float(args.box_mass),
        "surface_point_spacing": float(args.surface_point_spacing),
        "surface_point_count": int(len(local_surface_points)),
        "bottom_feature_source": args.bottom_feature_source,
        "push_force": list(args.push_force),
        "push_point_offset": list(args.push_point_offset),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(run_metadata, indent=2, sort_keys=True), encoding="utf-8")

    feature_metadata = {
        "extractor": extractor.metadata(),
        "projection": {
            "depth_threshold": float(args.depth_threshold),
            "front_depth_threshold": (
                0.5 * float(args.depth_threshold)
                if args.front_depth_threshold is None
                else float(args.front_depth_threshold)
            ),
            "points_per_chunk": int(args.points_per_chunk),
            "l2_normalize_features": bool(args.l2_normalize_features),
            "fallback_to_nearest_depth": not bool(args.no_depth_fallback),
        },
        "reference": "PointWorld-style projective DINO feature sampling on Newton surface points",
    }
    save_feature_metadata(args.output_dir / "feature_metadata.json", feature_metadata)

    frame_summaries: list[dict[str, object]] = []
    for step in range(args.num_steps + 1):
        if step % args.frame_stride == 0:
            summary = _write_frame(
                args=args,
                frame_index=step,
                model=model,
                data=data,
                cameras=cameras,
                local_surface_points=local_surface_points,
                point_masses=point_masses,
                extractor=extractor,
                projector=projector,
                feature_metadata=feature_metadata,
            )
            frame_summaries.append(summary)
            print(
                f"frame {step}: surface_points={summary['point_count']} "
                f"features={summary['assigned_feature_count']} "
                f"strict={summary['visible_feature_count']} "
                f"fallback={summary['depth_fallback_count']} "
                f"bottom={summary['bottom_count']} "
                f"bottom_strict={summary['bottom_visible_feature_count']} "
                f"bottom_fallback={summary['bottom_depth_fallback_count']} "
                f"bottom_top_copy={summary['bottom_feature_copied_from_top_count']} "
                f"dim={summary['feature_dim']}"
            )
        if step == args.num_steps:
            break
        apply_body_point_force(
            model,
            data,
            body_name=args.box_body,
            force_world=args.push_force,
            point_offset_local=args.push_point_offset,
        )
        mujoco.mj_step(model, data)
        data.qfrc_applied[:] = 0.0

    (args.output_dir / "feature_summary.json").write_text(
        json.dumps(frame_summaries, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote Newton-surface DINO point-feature outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
