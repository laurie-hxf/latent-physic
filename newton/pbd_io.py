from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from project_paths import DEFAULT_PLY_PATH, DEFAULT_SCENE_USD_PATH
from pbd_types import (
    DEFAULT_CONTACT_DAMPING,
    DEFAULT_CONTACT_MARGIN,
    DEFAULT_CONTACT_STIFFNESS,
    DEFAULT_EE_MASS,
    DEFAULT_EE_RADIUS_SCALE,
    DEFAULT_EE_SEG_ID,
    DEFAULT_EE_VOXEL,
    DEFAULT_FRICTION_REGULARIZATION,
    DEFAULT_MAX_VELOCITY,
    DEFAULT_OBJECT_FRICTION,
    DEFAULT_SUBSTEPS,
    DEFAULT_TABLE_FRICTION,
    DEFAULT_TABLE_SEG_ID,
    DEFAULT_TABLE_VOXEL,
    DEFAULT_TEE_MASS,
    DEFAULT_TEE_RADIUS_SCALE,
    DEFAULT_TEE_SEG_ID,
    DEFAULT_TEE_VOXEL,
    DEFAULT_VELOCITY_DAMPING,
    PlyHeader,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Use a segmented ASCII PLY point cloud to build a differentiable rigid-body scene. "
            "Each selected segment becomes one rigid body. "
            "Tee and end-effector are modeled as voxelized sphere clusters with interior filling, "
            "while the table is modeled as a thin static box fitted to the tabletop. "
            "Contacts are resolved with a differentiable hybrid solver: semi-implicit Euler prediction, "
            "XPBD-style compliant contact projection, and regularized compliant contact forces implemented in torch."
        )
    )
    parser.add_argument(
        "--ply-path",
        type=Path,
        default=DEFAULT_PLY_PATH,
        help="Path to the segmented ASCII PLY file.",
    )
    parser.add_argument("--table-seg-id", type=int, default=DEFAULT_TABLE_SEG_ID)
    parser.add_argument("--tee-seg-id", type=int, default=DEFAULT_TEE_SEG_ID)
    parser.add_argument("--ee-seg-id", type=int, default=DEFAULT_EE_SEG_ID)
    parser.add_argument("--table-voxel", type=float, default=DEFAULT_TABLE_VOXEL)
    parser.add_argument("--tee-voxel", type=float, default=DEFAULT_TEE_VOXEL)
    parser.add_argument("--ee-voxel", type=float, default=DEFAULT_EE_VOXEL)
    parser.add_argument(
        "--tee-radius-scale",
        type=float,
        default=DEFAULT_TEE_RADIUS_SCALE,
        help="Collision sphere radius scale for the Tee rigid body, relative to voxel size.",
    )
    parser.add_argument(
        "--ee-radius-scale",
        type=float,
        default=DEFAULT_EE_RADIUS_SCALE,
        help="Collision sphere radius scale for the end-effector rigid body, relative to voxel size.",
    )
    parser.add_argument("--tee-mass", type=float, default=DEFAULT_TEE_MASS)
    parser.add_argument(
        "--ee-mass",
        type=float,
        default=DEFAULT_EE_MASS,
        help="Approximate total mass used to compute end-effector rigid-body inertia.",
    )
    parser.add_argument(
        "--xpbd-iterations",
        type=int,
        default=30,
        help="Number of XPBD-style contact projection iterations per substep.",
    )
    parser.add_argument(
        "--substeps",
        type=int,
        default=DEFAULT_SUBSTEPS,
        help="Number of physics sub-steps per action step to reduce tunneling.",
    )
    parser.add_argument(
        "--velocity-damping",
        type=float,
        default=DEFAULT_VELOCITY_DAMPING,
        help="Per-step velocity damping factor for free dynamic rigid bodies (0..1).",
    )
    parser.add_argument(
        "--max-velocity",
        type=float,
        default=DEFAULT_MAX_VELOCITY,
        help="Maximum linear speed (m/s) clamp for free dynamic rigid bodies.",
    )
    parser.add_argument(
        "--table-friction",
        type=float,
        default=DEFAULT_TABLE_FRICTION,
        help=(
            "Friction coefficient used for table contact and planar support friction."
        ),
    )
    parser.add_argument(
        "--object-friction",
        type=float,
        default=DEFAULT_OBJECT_FRICTION,
        help=(
            "Friction coefficient used for sphere-sphere contact, e.g. tee/end-effector contact."
        ),
    )
    parser.add_argument(
        "--contact-stiffness",
        type=float,
        default=DEFAULT_CONTACT_STIFFNESS,
        help="Normal contact stiffness used by both the compliant contact forces and XPBD compliance.",
    )
    parser.add_argument(
        "--contact-damping",
        type=float,
        default=DEFAULT_CONTACT_DAMPING,
        help="Normal contact damping used to oppose closing velocity in the hybrid contact model.",
    )
    parser.add_argument(
        "--contact-margin",
        type=float,
        default=DEFAULT_CONTACT_MARGIN,
        help="Smooth contact activation length scale in meters for the differentiable contact gate.",
    )
    parser.add_argument(
        "--friction-regularization",
        type=float,
        default=DEFAULT_FRICTION_REGULARIZATION,
        help="Velocity regularization used to keep friction differentiable around zero slip.",
    )
    parser.add_argument("--simulate-steps", type=int, default=0)
    parser.add_argument("--sim-dt", type=float, default=1.0 / 240.0)
    parser.add_argument(
        "--ee-action",
        type=float,
        nargs=3,
        action="append",
        default=None,
        metavar=("DX", "DY", "DZ"),
        help=(
            "Per-step translation applied to the end_effector rigid body. "
            "Can be passed multiple times to define a rollout. "
            "If passed once and --simulate-steps > 1, the same action is repeated."
        ),
    )
    parser.add_argument(
        "--ee-actions-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON file describing the end effector action rollout. "
            "Accepted formats: [[dx, dy, dz], ...], {'actions': [...]}, "
            "or [{'dx': ..., 'dy': ..., 'dz': ...}, ...]."
        ),
    )
    parser.add_argument(
        "--scene-usd-path",
        type=Path,
        default=DEFAULT_SCENE_USD_PATH,
        help=(
            "Animated USD/USDA file used to save the actual collision geometry and body motion."
        ),
    )
    parser.add_argument(
        "--save-step-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory used to save sampled PLYs after every simulated step. "
            "If omitted and end-effector actions are provided, a default step directory is "
            "created next to --scene-usd-path."
        ),
    )
    return parser.parse_args()


def read_ascii_ply_header(ply_path: Path) -> PlyHeader:
    if not ply_path.exists():
        raise FileNotFoundError(ply_path)

    vertex_count = None
    properties: list[str] = []
    data_start_line = 0
    in_vertex_element = False

    with ply_path.open("r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        if first_line != "ply":
            raise RuntimeError(f"{ply_path} is not a PLY file")

        format_line = f.readline().strip()
        if format_line != "format ascii 1.0":
            raise RuntimeError(f"{ply_path} must be an ASCII PLY file, got: {format_line}")

        for line_idx, line in enumerate(f, start=3):
            stripped = line.strip()
            if stripped.startswith("element "):
                _, element_name, count = stripped.split()
                in_vertex_element = element_name == "vertex"
                if in_vertex_element:
                    vertex_count = int(count)
                    properties = []
            elif stripped.startswith("property ") and in_vertex_element:
                properties.append(stripped.split()[-1])
            elif stripped == "end_header":
                data_start_line = line_idx + 1
                break

    if vertex_count is None:
        raise RuntimeError(f"No vertex section found in {ply_path}")

    required = {"x", "y", "z", "segmentation_id"}
    missing = required - set(properties)
    if missing:
        raise RuntimeError(f"{ply_path} is missing required properties: {sorted(missing)}")

    return PlyHeader(vertex_count=vertex_count, properties=properties, data_start_line=data_start_line)


def iterate_vertex_rows(ply_path: Path, header: PlyHeader):
    with ply_path.open("r", encoding="utf-8") as f:
        for _ in range(header.data_start_line - 1):
            next(f)
        for row_idx, line in enumerate(f):
            stripped = line.strip()
            if not stripped:
                continue
            yield row_idx, stripped.split()


def parse_action_triplet(action_like: object, action_idx: int) -> np.ndarray:
    if isinstance(action_like, dict):
        if {"dx", "dy", "dz"}.issubset(action_like):
            values = [action_like["dx"], action_like["dy"], action_like["dz"]]
        elif "delta" in action_like:
            values = action_like["delta"]
        else:
            raise RuntimeError(
                f"Action #{action_idx} must contain dx/dy/dz or delta, got keys: {sorted(action_like)}"
            )
    else:
        values = action_like

    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise RuntimeError(f"Action #{action_idx} must be a length-3 triplet, got: {values!r}")

    return np.asarray(values, dtype=np.float32)


def load_actions_from_json(json_path: Path) -> list[np.ndarray]:
    if not json_path.exists():
        raise FileNotFoundError(json_path)

    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_actions = payload.get("actions") if isinstance(payload, dict) else payload
    if not isinstance(raw_actions, list):
        raise RuntimeError(f"{json_path} must contain a list of actions or a dict with an 'actions' list")

    return [parse_action_triplet(action_like, idx) for idx, action_like in enumerate(raw_actions)]


def build_ee_action_sequence(args: argparse.Namespace) -> list[np.ndarray]:
    if args.simulate_steps < 0:
        raise RuntimeError("--simulate-steps must be >= 0")
    if args.sim_dt <= 0.0:
        raise RuntimeError("--sim-dt must be > 0")
    if args.ee_action is not None and args.ee_actions_json is not None:
        raise RuntimeError("Use either --ee-action or --ee-actions-json, not both")

    if args.ee_actions_json is not None:
        actions = load_actions_from_json(args.ee_actions_json)
        if args.simulate_steps not in (0, len(actions)):
            raise RuntimeError(
                f"--simulate-steps={args.simulate_steps} does not match the number of JSON actions ({len(actions)})"
            )
        return actions

    if args.ee_action:
        base_actions = [np.asarray(action, dtype=np.float32) for action in args.ee_action]
        if len(base_actions) == 1:
            repeat_count = max(int(args.simulate_steps), 1)
            return [base_actions[0].copy() for _ in range(repeat_count)]
        if args.simulate_steps not in (0, len(base_actions)):
            raise RuntimeError(
                f"--simulate-steps={args.simulate_steps} does not match the number of --ee-action entries "
                f"({len(base_actions)})"
            )
        return base_actions

    return [np.zeros(3, dtype=np.float32) for _ in range(args.simulate_steps)]


def resolve_step_output_dir(
    scene_usd_path: Path,
    requested_step_dir: Path | None,
    has_explicit_actions: bool,
) -> Path | None:
    if requested_step_dir is not None:
        return requested_step_dir
    if has_explicit_actions:
        return scene_usd_path.parent / f"{scene_usd_path.stem}_steps"
    return None
