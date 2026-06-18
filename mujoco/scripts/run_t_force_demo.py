from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

import run_block_force_demo as block_demo
from t_force_designs import (
    design_box_bounds,
    design_fingerprint,
    load_designs,
    parse_region_friction_overrides,
    prepare_design,
    write_scene_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESIGNS_PATH = ROOT / "configs" / "t_force_designs.json"
DEFAULT_GENERATED_SCENE_DIR = ROOT / "outputs" / "generated_t_force_scenes"
SAMPLE_BOX_POINT_OFFSET = block_demo.sample_point_offset


def make_surface_point_sampler(
    box_bounds: list[tuple[np.ndarray, np.ndarray]],
) -> Callable[[np.random.Generator, np.ndarray, np.ndarray, float, str], np.ndarray]:
    def sample_t_surface_point(
        rng: np.random.Generator,
        _bounds_min: np.ndarray,
        _bounds_max: np.ndarray,
        edge_margin_ratio: float,
        mode: str = "surface",
    ) -> np.ndarray:
        probe_distance = 1.0e-5
        for _ in range(256):
            bounds_min, bounds_max = box_bounds[int(rng.integers(0, len(box_bounds)))]
            point = SAMPLE_BOX_POINT_OFFSET(
                rng,
                bounds_min,
                bounds_max,
                edge_margin_ratio,
                mode=mode,
            )

            distances_to_faces = np.stack((point - bounds_min, bounds_max - point), axis=0)
            selected_faces = np.argwhere(distances_to_faces <= 2.0 * block_demo.SURFACE_EPS)
            outward = np.zeros(3, dtype=np.float64)
            for lower_or_upper, axis in selected_faces:
                outward[axis] += -1.0 if lower_or_upper == 0 else 1.0
            outward_norm = float(np.linalg.norm(outward))
            if outward_norm < 1.0e-12:
                continue

            probe = point + probe_distance * outward / outward_norm
            if not any(
                bool(np.all(probe >= candidate_min) and np.all(probe <= candidate_max))
                for candidate_min, candidate_max in box_bounds
            ):
                return point

        raise RuntimeError(f"Could not sample an exposed {mode} point on the T shape.")

    return sample_t_surface_point


def make_planar_initial_pose_sampler(
    thickness: float,
) -> Callable[..., dict[str, object]]:
    def sample_planar_t_initial_pose(
        rng: np.random.Generator,
        _bounds_min: np.ndarray,
        _bounds_max: np.ndarray,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        floor_height: float,
        clearance: float,
    ) -> dict[str, object]:
        yaw = float(rng.uniform(0.0, 2.0 * np.pi))
        rotation = block_demo.z_axis_rotation_matrix(yaw)
        return {
            "position": np.array(
                [
                    float(rng.uniform(x_range[0], x_range[1])),
                    float(rng.uniform(y_range[0], y_range[1])),
                    float(floor_height + clearance + 0.5 * thickness),
                ],
                dtype=np.float64,
            ),
            "quaternion_wxyz": block_demo.quaternion_wxyz_from_matrix(rotation),
            "contact_face_id": 0,
            "contact_face_name": "t_bottom_z_neg",
            "contact_face_normal_local": np.array([0.0, 0.0, -1.0], dtype=np.float64),
            "yaw_about_world_z": yaw,
        }

    return sample_planar_t_initial_pose


def make_x_split_friction_setter(design: dict[str, object]) -> Callable[[mujoco.MjModel, float, float], None]:
    design_name = str(design["name"])
    centers_by_geom = {
        f"push_block_{part['name']}": float(part["center_xy"][0])
        for part in design["parts"]
    }

    def set_x_split_t_friction(model: mujoco.MjModel, left_mu: float, right_mu: float) -> None:
        if design_name != "left_right":
            raise ValueError(
                "--block-left-friction/--block-right-friction are only valid with --t-design left_right. "
                "Use repeated --region-friction NAME=MU overrides for other T designs."
            )
        for geom_name, center_x in centers_by_geom.items():
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id < 0:
                raise ValueError(f"Could not find T geom '{geom_name}' for X-split friction override.")
            friction_mu = left_mu if center_x < 0.0 else right_mu
            model.geom_friction[geom_id, :] = np.array([float(friction_mu), 0.0, 0.0], dtype=np.float64)

    return set_x_split_t_friction


def configure_t_demo(design: dict[str, object], scene_path: Path) -> None:
    box_bounds = design_box_bounds(design)
    geom_names = tuple(f"push_block_{part['name']}" for part in design["parts"])
    block_demo.SCENE_PATH = scene_path
    block_demo.DEFAULT_OUTPUT_DIR = ROOT / "outputs" / f"t_force_{design['name']}"
    block_demo.BLOCK_FRICTION_GEOM_NAMES = geom_names
    block_demo.set_split_block_friction = make_x_split_friction_setter(design)
    block_demo.sample_point_offset = make_surface_point_sampler(box_bounds)
    block_demo.sample_contact_face_initial_pose = make_planar_initial_pose_sampler(float(design["thickness"]))


def custom_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_argument_group("configurable T shape")
    group.add_argument("--t-design", default="left_right", help="Named T design from --t-designs-file.")
    group.add_argument("--t-designs-file", type=Path, default=DEFAULT_DESIGNS_PATH, help="JSON file containing T designs.")
    group.add_argument("--t-scale", type=float, default=1.0, help="Uniform scale applied to T width and length.")
    group.add_argument("--t-thickness", type=float, default=None, help="Override T thickness in meters.")
    group.add_argument("--friction-scale", type=float, default=1.0, help="Scale every region friction coefficient.")
    group.add_argument(
        "--region-friction",
        action="append",
        default=[],
        metavar="NAME=MU",
        help="Override one named region friction coefficient. Repeat for multiple regions.",
    )
    group.add_argument("--generated-scene-path", type=Path, default=None, help="Path for the generated MuJoCo XML.")
    group.add_argument("--list-t-designs", action="store_true", help="List configured T designs and exit.")
    group.add_argument(
        "--export-all-t-scenes",
        type=Path,
        default=None,
        metavar="DIR",
        help="Generate XML and JSON files for every configured T design, then exit.",
    )
    return parser


def print_designs(designs: dict[str, dict[str, object]]) -> None:
    for name, design in designs.items():
        parts = design.get("parts", [])
        friction_levels = sorted({float(part["friction"]) for part in parts})
        description = str(design.get("description", ""))
        print(f"{name}: {len(parts)} geometry parts, {len(friction_levels)} friction levels; {description}")
        print("  " + ", ".join(f"{part['name']}={float(part['friction']):.4g}" for part in parts))


def export_all_designs(designs: dict[str, dict[str, object]], output_dir: Path) -> None:
    for name, raw_design in designs.items():
        design = prepare_design(name, raw_design)
        scene_path = output_dir / f"{name}.xml"
        _, metadata_path = write_scene_bundle(scene_path, design)
        print(f"{name}: {scene_path} ({metadata_path.name})")


def main() -> None:
    parser = custom_parser()
    custom_args, remaining_args = parser.parse_known_args()
    designs = load_designs(custom_args.t_designs_file)

    if custom_args.list_t_designs:
        print_designs(designs)
        return
    if custom_args.export_all_t_scenes is not None:
        export_all_designs(designs, custom_args.export_all_t_scenes)
        return
    if custom_args.t_design not in designs:
        available = ", ".join(designs)
        raise ValueError(f"Unknown T design '{custom_args.t_design}'. Available designs: {available}")

    overrides = parse_region_friction_overrides(custom_args.region_friction)
    design = prepare_design(
        custom_args.t_design,
        designs[custom_args.t_design],
        xy_scale=float(custom_args.t_scale),
        thickness_override=custom_args.t_thickness,
        friction_scale=float(custom_args.friction_scale),
        region_friction_overrides=overrides,
    )
    fingerprint = design_fingerprint(design)
    scene_path = custom_args.generated_scene_path
    if scene_path is None:
        scene_path = DEFAULT_GENERATED_SCENE_DIR / f"{design['name']}_{fingerprint}.xml"
    write_scene_bundle(scene_path, design)
    configure_t_demo(design, scene_path)

    if "-h" in remaining_args or "--help" in remaining_args:
        print(parser.format_help())
    sys.argv = [sys.argv[0], *remaining_args]
    block_demo.main()


if __name__ == "__main__":
    main()
