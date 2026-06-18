from __future__ import annotations

import copy
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np


SAFE_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def load_designs(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    designs = payload.get("designs")
    if not isinstance(designs, dict) or not designs:
        raise ValueError(f"{path} must contain a non-empty 'designs' object.")
    return designs


def parse_region_friction_overrides(values: list[str]) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Region friction override must be NAME=MU, got: {value}")
        name, friction_text = value.split("=", 1)
        name = name.strip()
        friction = float(friction_text)
        if not name or friction < 0.0:
            raise ValueError(f"Invalid region friction override: {value}")
        overrides[name] = friction
    return overrides


def _validate_part(part: dict[str, Any], index: int) -> None:
    name = part.get("name")
    if not isinstance(name, str) or not SAFE_NAME_RE.fullmatch(name):
        raise ValueError(f"Part {index} name must match {SAFE_NAME_RE.pattern}.")
    center = np.asarray(part.get("center_xy"), dtype=np.float64)
    size = np.asarray(part.get("size_xy"), dtype=np.float64)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError(f"Part '{name}' center_xy must contain two finite numbers.")
    if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0.0):
        raise ValueError(f"Part '{name}' size_xy must contain two positive finite numbers.")
    friction = float(part.get("friction", -1.0))
    if not np.isfinite(friction) or friction < 0.0:
        raise ValueError(f"Part '{name}' friction must be non-negative and finite.")
    if "rgba" in part:
        rgba = np.asarray(part["rgba"], dtype=np.float64)
        if rgba.shape != (4,) or not np.isfinite(rgba).all() or np.any((rgba < 0.0) | (rgba > 1.0)):
            raise ValueError(f"Part '{name}' rgba must contain four values in [0, 1].")


def _validate_non_overlapping(parts: list[dict[str, Any]]) -> None:
    for first_index, first in enumerate(parts):
        first_center = np.asarray(first["center_xy"], dtype=np.float64)
        first_half = 0.5 * np.asarray(first["size_xy"], dtype=np.float64)
        first_min = first_center - first_half
        first_max = first_center + first_half
        for second in parts[first_index + 1 :]:
            second_center = np.asarray(second["center_xy"], dtype=np.float64)
            second_half = 0.5 * np.asarray(second["size_xy"], dtype=np.float64)
            second_min = second_center - second_half
            second_max = second_center + second_half
            overlap = np.minimum(first_max, second_max) - np.maximum(first_min, second_min)
            if np.all(overlap > 1.0e-9):
                raise ValueError(f"Parts '{first['name']}' and '{second['name']}' overlap in XY.")


def _friction_color(friction: float, minimum: float, maximum: float) -> list[float]:
    ratio = 0.5 if maximum <= minimum else (friction - minimum) / (maximum - minimum)
    low = np.array([0.88, 0.28, 0.22], dtype=np.float64)
    high = np.array([0.18, 0.45, 0.95], dtype=np.float64)
    rgb = (1.0 - ratio) * low + ratio * high
    return [float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0]


def prepare_design(
    design_name: str,
    raw_design: dict[str, Any],
    *,
    xy_scale: float = 1.0,
    thickness_override: float | None = None,
    friction_scale: float = 1.0,
    region_friction_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    design = copy.deepcopy(raw_design)
    if xy_scale <= 0.0:
        raise ValueError("xy_scale must be positive.")
    if friction_scale < 0.0:
        raise ValueError("friction_scale must be non-negative.")

    thickness = float(design.get("thickness", 0.05) if thickness_override is None else thickness_override)
    density = float(design.get("density", 1000.0))
    if thickness <= 0.0 or density <= 0.0:
        raise ValueError("T thickness and density must be positive.")

    parts = design.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError(f"Design '{design_name}' must contain a non-empty parts list.")
    for index, part in enumerate(parts):
        _validate_part(part, index)
    names = [str(part["name"]) for part in parts]
    if len(names) != len(set(names)):
        raise ValueError(f"Design '{design_name}' has duplicate part names.")
    _validate_non_overlapping(parts)

    overrides = region_friction_overrides or {}
    unknown_overrides = sorted(set(overrides) - set(names))
    if unknown_overrides:
        raise ValueError(f"Unknown region friction override(s): {', '.join(unknown_overrides)}")

    areas = np.asarray(
        [float(part["size_xy"][0]) * float(part["size_xy"][1]) for part in parts],
        dtype=np.float64,
    )
    centers = np.asarray([part["center_xy"] for part in parts], dtype=np.float64)
    center_of_mass_xy = np.average(centers, axis=0, weights=areas)

    prepared_parts: list[dict[str, Any]] = []
    for part in parts:
        part_name = str(part["name"])
        friction = float(overrides.get(part_name, float(part["friction"]) * friction_scale))
        prepared_parts.append(
            {
                "name": part_name,
                "center_xy": ((np.asarray(part["center_xy"], dtype=np.float64) - center_of_mass_xy) * xy_scale).tolist(),
                "size_xy": (np.asarray(part["size_xy"], dtype=np.float64) * xy_scale).tolist(),
                "friction": friction,
                "rgba": part.get("rgba"),
            }
        )

    frictions = [float(part["friction"]) for part in prepared_parts]
    minimum_friction = min(frictions)
    maximum_friction = max(frictions)
    for part in prepared_parts:
        if part["rgba"] is None:
            part["rgba"] = _friction_color(float(part["friction"]), minimum_friction, maximum_friction)

    return {
        "name": design_name,
        "description": str(design.get("description", "")),
        "thickness": thickness,
        "density": density,
        "xy_scale": float(xy_scale),
        "source_center_of_mass_xy": center_of_mass_xy.tolist(),
        "parts": prepared_parts,
    }


def design_box_bounds(design: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    half_thickness = 0.5 * float(design["thickness"])
    bounds: list[tuple[np.ndarray, np.ndarray]] = []
    for part in design["parts"]:
        center_xy = np.asarray(part["center_xy"], dtype=np.float64)
        half_xy = 0.5 * np.asarray(part["size_xy"], dtype=np.float64)
        bounds.append(
            (
                np.array([center_xy[0] - half_xy[0], center_xy[1] - half_xy[1], -half_thickness]),
                np.array([center_xy[0] + half_xy[0], center_xy[1] + half_xy[1], half_thickness]),
            )
        )
    return bounds


def design_fingerprint(design: dict[str, Any]) -> str:
    encoded = json.dumps(design, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:10]


def _numbers(values: list[float] | tuple[float, ...]) -> str:
    return " ".join(f"{float(value):.10g}" for value in values)


def build_scene_tree(design: dict[str, Any]) -> ET.ElementTree:
    model_name = f"t_force_{design['name']}"
    root = ET.Element("mujoco", {"model": model_name})
    ET.SubElement(root, "statistic", {"center": "0.3 0.0 0.12", "extent": "1.0"})

    option = ET.SubElement(
        root,
        "option",
        {"timestep": "0.002", "iterations": "20", "ls_iterations": "16", "integrator": "implicitfast"},
    )
    ET.SubElement(option, "flag", {"eulerdamp": "disable"})

    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "headlight",
        {"diffuse": "0.6 0.6 0.6", "ambient": "0.3 0.3 0.3", "specular": "0 0 0"},
    )
    ET.SubElement(visual, "rgba", {"haze": "0.15 0.25 0.35 1"})
    ET.SubElement(visual, "global", {"azimuth": "120", "elevation": "-20", "offwidth": "960", "offheight": "720"})

    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0.3 0.5 0.7",
            "rgb2": "0 0 0",
            "width": "512",
            "height": "3072",
        },
    )
    ET.SubElement(
        asset,
        "texture",
        {
            "type": "2d",
            "name": "floor_checker",
            "builtin": "checker",
            "mark": "edge",
            "rgb1": "0.20 0.26 0.32",
            "rgb2": "0.10 0.14 0.18",
            "markrgb": "0.85 0.85 0.85",
            "width": "300",
            "height": "300",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "floor_checker_mat",
            "texture": "floor_checker",
            "texuniform": "true",
            "texrepeat": "6 6",
            "reflectance": "0",
        },
    )

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody, "light", {"pos": "0 0 1.5", "dir": "0 0 -1", "directional": "true"})
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "pos": "0 0 0",
            "size": "0 0 0.05",
            "type": "plane",
            "material": "floor_checker_mat",
            "friction": "0.0 0.0 0.0",
            "contype": "1",
            "conaffinity": "1",
        },
    )

    half_thickness = 0.5 * float(design["thickness"])
    body = ET.SubElement(worldbody, "body", {"name": "push_block", "pos": f"0.58 0.0 {half_thickness:.10g}"})
    ET.SubElement(body, "freejoint")
    for part in design["parts"]:
        center_x, center_y = part["center_xy"]
        size_x, size_y = part["size_xy"]
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"push_block_{part['name']}",
                "type": "box",
                "pos": _numbers([center_x, center_y, 0.0]),
                "size": _numbers([0.5 * size_x, 0.5 * size_y, half_thickness]),
                "density": f"{float(design['density']):.10g}",
                "condim": "3",
                "friction": _numbers([float(part["friction"]), 0.0, 0.0]),
                "rgba": _numbers(part["rgba"]),
                "contype": "2",
                "conaffinity": "3",
                "solref": "0.01 1",
            },
        )

    ET.indent(root, space="  ")
    return ET.ElementTree(root)


def write_scene_bundle(scene_path: Path, design: dict[str, Any]) -> tuple[Path, Path]:
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    tree = build_scene_tree(design)
    tree.write(scene_path, encoding="unicode", xml_declaration=False)
    with scene_path.open("a", encoding="utf-8") as f:
        f.write("\n")

    metadata_path = scene_path.with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(design, f, indent=2, sort_keys=True)
        f.write("\n")
    return scene_path, metadata_path
