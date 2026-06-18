from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
NEWTON_DIR = REPO_ROOT / "newton"
for path in (REPO_ROOT, NEWTON_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dino_point_features.dino_extractor import DinoFeatureExtractor  # noqa: E402
from dino_point_features.io import save_feature_metadata  # noqa: E402
from dino_point_features.projection import DinoFeatureProjector  # noqa: E402
from dino_point_features.run_block_force_dino_surface_points import (  # noqa: E402
    _parse_layers,
    _write_frame,
)
from mujoco_pointcloud_pipeline.scene import (  # noqa: E402
    default_block_force_cameras,
    load_model_with_cameras,
)
from newton_surface_points_demo import sample_box_surface_points  # noqa: E402
from object_physics_latent.dataset import validate_manifest  # noqa: E402


DEFAULT_DATASET_ROOT = REPO_ROOT / "mujoco" / "outputs" / "object_physics_latent_box_partitions_48x2000_min300"
DEFAULT_MANIFEST = DEFAULT_DATASET_ROOT / "manifest.json"


def _resolve_path(value: str, parent: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (parent / path).resolve()


def _relative_path(path: Path, parent: Path) -> str:
    return os.path.relpath(Path(path).resolve(), start=Path(parent).resolve())


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _format_friction(values: Any) -> str:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("friction value must contain at least one number")
    if array.size == 1:
        array = np.asarray([array[0], 0.0, 0.0], dtype=np.float64)
    if array.size != 3:
        raise ValueError(f"MuJoCo geom friction must have 3 values, got {array.tolist()}")
    return " ".join(f"{float(value):.9g}" for value in array)


def _write_scene_with_object_friction(
    *,
    scene_path: Path,
    block_friction: dict[str, Any],
    output_path: Path,
) -> None:
    tree = ET.parse(scene_path)
    root = tree.getroot()
    found: set[str] = set()
    for geom in root.iter("geom"):
        geom_name = geom.get("name")
        if geom_name is None or geom_name not in block_friction:
            continue
        geom.set("friction", _format_friction(block_friction[geom_name]))
        found.add(geom_name)

    missing = sorted(set(block_friction).difference(found))
    if missing:
        raise ValueError(f"{scene_path} is missing object friction geoms: {missing}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="unicode")


def _build_args(
    *,
    output_dir: Path,
    scene_path: Path,
    export_xml: Path,
    width: int,
    height: int,
    surface_point_spacing: float,
    box_half_extents: tuple[float, float, float],
    box_mass: float,
    bottom_feature_source: str,
    depth_threshold: float,
    front_depth_threshold: float | None,
    points_per_chunk: int,
    l2_normalize_features: bool,
    no_depth_fallback: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        scene=scene_path,
        output_dir=output_dir,
        width=int(width),
        height=int(height),
        num_steps=0,
        frame_stride=1,
        export_xml=export_xml,
        camera=None,
        box_body="push_block",
        box_half_extents=tuple(float(value) for value in box_half_extents),
        box_mass=float(box_mass),
        surface_point_spacing=float(surface_point_spacing),
        allow_zero_split_x=False,
        bottom_feature_source=str(bottom_feature_source),
        push_force=(0.0, 0.0, 0.0),
        push_point_offset=(0.0, 0.0, 0.0),
        depth_threshold=float(depth_threshold),
        front_depth_threshold=front_depth_threshold,
        points_per_chunk=int(points_per_chunk),
        l2_normalize_features=bool(l2_normalize_features),
        no_depth_fallback=bool(no_depth_fallback),
    )


def _feature_npz_path(object_dir: Path, output_subdir: str) -> Path:
    return object_dir / output_subdir / "frame_000000" / "newton_surface_points_dino_features.npz"


def _extract_for_object(
    *,
    record: dict[str, Any],
    manifest_parent: Path,
    extractor: DinoFeatureExtractor,
    projector: DinoFeatureProjector,
    local_surface_points: np.ndarray,
    point_masses: np.ndarray,
    args: argparse.Namespace,
    feature_metadata_base: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    object_id = str(record["object_id"])
    trajectory_npz = _resolve_path(str(record["trajectory_npz"]), manifest_parent)
    object_dir = trajectory_npz.parent
    metadata_path = trajectory_npz.with_suffix(".json")
    metadata = _load_json(metadata_path)

    output_dir = object_dir / str(args.output_subdir)
    npz_path = _feature_npz_path(object_dir, str(args.output_subdir))
    if bool(args.skip_existing) and npz_path.is_file():
        return npz_path, {"object_id": object_id, "npz": str(npz_path), "skipped_existing": True}

    scene_path = Path(str(metadata["scene_path"])).expanduser().resolve()
    object_scene_xml = output_dir / "object_scene_with_friction.xml"
    _write_scene_with_object_friction(
        scene_path=scene_path,
        block_friction=dict(metadata["block_friction"]),
        output_path=object_scene_xml,
    )

    run_args = _build_args(
        output_dir=output_dir,
        scene_path=object_scene_xml,
        export_xml=output_dir / "block_force_surface_points_scene.xml",
        width=int(args.width),
        height=int(args.height),
        surface_point_spacing=float(args.surface_point_spacing),
        box_half_extents=tuple(float(value) for value in args.box_half_extents),
        box_mass=float(args.box_mass),
        bottom_feature_source=str(args.bottom_feature_source),
        depth_threshold=float(args.depth_threshold),
        front_depth_threshold=args.front_depth_threshold,
        points_per_chunk=int(args.points_per_chunk),
        l2_normalize_features=bool(args.l2_normalize_features),
        no_depth_fallback=bool(args.no_depth_fallback),
    )
    cameras = default_block_force_cameras()
    model, _ = load_model_with_cameras(object_scene_xml, cameras, export_xml_path=run_args.export_xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    run_metadata = {
        "object_id": object_id,
        "trajectory_npz": str(trajectory_npz.resolve()),
        "trajectory_metadata": str(metadata_path.resolve()),
        "source_scene": str(scene_path),
        "object_scene_xml": str(object_scene_xml.resolve()),
        "augmented_scene_xml": str(run_args.export_xml.resolve()),
        "friction_partition_family": metadata.get("friction_partition_family"),
        "friction_region_values": metadata.get("friction_region_values"),
        "block_friction": metadata.get("block_friction"),
        "width": int(args.width),
        "height": int(args.height),
        "num_steps": 0,
        "frame_stride": 1,
        "cameras": [camera.__dict__ for camera in cameras],
        "box_body": "push_block",
        "box_half_extents": [float(value) for value in args.box_half_extents],
        "box_mass": float(args.box_mass),
        "surface_point_spacing": float(args.surface_point_spacing),
        "surface_point_count": int(len(local_surface_points)),
        "bottom_feature_source": str(args.bottom_feature_source),
        "push_force": [0.0, 0.0, 0.0],
        "push_point_offset": [0.0, 0.0, 0.0],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(json.dumps(run_metadata, indent=2, sort_keys=True), encoding="utf-8")

    feature_metadata = {
        **feature_metadata_base,
        "object": {
            "object_id": object_id,
            "trajectory_npz": str(trajectory_npz.resolve()),
            "trajectory_metadata": str(metadata_path.resolve()),
            "friction_partition_family": metadata.get("friction_partition_family"),
            "friction_region_values": metadata.get("friction_region_values"),
            "block_friction": metadata.get("block_friction"),
        },
    }
    save_feature_metadata(output_dir / "feature_metadata.json", feature_metadata)

    summary = _write_frame(
        args=run_args,
        frame_index=0,
        model=model,
        data=data,
        cameras=cameras,
        local_surface_points=local_surface_points,
        point_masses=point_masses,
        extractor=extractor,
        projector=projector,
        feature_metadata=feature_metadata,
    )
    summary["object_id"] = object_id
    summary["trajectory_npz"] = str(trajectory_npz.resolve())
    summary["friction_partition_family"] = metadata.get("friction_partition_family")
    (output_dir / "feature_summary.json").write_text(
        json.dumps([summary], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return npz_path, summary


def _update_manifest_dino_paths(
    *,
    manifest_path: Path,
    payload: dict[str, Any],
    feature_paths: dict[str, Path],
    output_path: Path | None,
) -> Path:
    target = manifest_path if output_path is None else Path(output_path).expanduser().resolve()
    target_parent = target.parent
    for record in payload["objects"]:
        object_id = str(record["object_id"])
        record["dino_feature_npz"] = _relative_path(feature_paths[object_id], target_parent)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract PointWorld-style DINO surface-point features for every object in a latent dataset manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--output-subdir", type=str, default="dino_features")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--box-half-extents", type=float, nargs=3, default=(0.1, 0.05, 0.025))
    parser.add_argument("--box-mass", type=float, default=1.0)
    parser.add_argument("--surface-point-spacing", type=float, default=0.01)
    parser.add_argument(
        "--bottom-feature-source",
        choices=("top-face", "projected"),
        default="top-face",
    )
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
    parser.add_argument("--no-depth-fallback", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    payload = _load_json(manifest_path)
    records = list(payload.get("objects", []))
    if args.limit is not None:
        records = records[: int(args.limit)]
    if not records:
        raise ValueError(f"{manifest_path} contains no object records")

    local_surface_points, point_masses = sample_box_surface_points(
        np.asarray(args.box_half_extents, dtype=np.float32),
        spacing=float(args.surface_point_spacing),
        total_mass=float(args.box_mass),
        avoid_zero_x=True,
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
        depth_threshold=float(args.depth_threshold),
        front_depth_threshold=args.front_depth_threshold,
        points_per_chunk=int(args.points_per_chunk),
        l2_normalize=bool(args.l2_normalize_features),
        fallback_to_nearest_depth=not bool(args.no_depth_fallback),
    )
    feature_metadata_base = {
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

    feature_paths: dict[str, Path] = {}
    summaries: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        object_id = str(record["object_id"])
        npz_path, summary = _extract_for_object(
            record=record,
            manifest_parent=manifest_path.parent,
            extractor=extractor,
            projector=projector,
            local_surface_points=local_surface_points,
            point_masses=point_masses,
            args=args,
            feature_metadata_base=feature_metadata_base,
        )
        feature_paths[object_id] = npz_path
        summaries.append(summary)
        if summary.get("skipped_existing"):
            print(f"[{index}/{len(records)}] skipped existing {object_id}: {npz_path}", flush=True)
        else:
            print(
                f"[{index}/{len(records)}] {object_id}: points={summary['point_count']} "
                f"features={summary['assigned_feature_count']} dim={summary['feature_dim']} npz={npz_path}",
                flush=True,
            )

    if args.limit is None:
        updated_manifest = _update_manifest_dino_paths(
            manifest_path=manifest_path,
            payload=payload,
            feature_paths=feature_paths,
            output_path=args.manifest_output,
        )
        validation = validate_manifest(updated_manifest, inspect_datasets=True)
        print(json.dumps({"manifest": str(updated_manifest), **validation}, indent=2, sort_keys=True), flush=True)
    else:
        print(json.dumps({"processed": len(records), "limit": int(args.limit)}, indent=2, sort_keys=True), flush=True)

    assigned = sum(int(summary.get("assigned_feature_count", 0)) for summary in summaries)
    print(json.dumps({"objects": len(records), "assigned_features_total": assigned}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
