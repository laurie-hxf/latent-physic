from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import newton
from pbd_math import make_transform, transform_points
from pbd_types import (
    CLUSTER_COLORS,
    DEFAULT_CONTACT_DAMPING,
    DEFAULT_CONTACT_MARGIN,
    DEFAULT_CONTACT_STIFFNESS,
    DEFAULT_FRICTION_REGULARIZATION,
    BuiltScene,
    IDENTITY_QUAT,
    RigidBodyCluster,
    SceneState,
)
from pbd_usd import export_scene_usd
from project_paths import DEFAULT_OUTPUT_DIR


DEFAULT_BOX_SCENE_USD_PATH = DEFAULT_OUTPUT_DIR / "newton_box_demo.usda"


def _to_tensor(
    value: float | list[float] | tuple[float, ...] | np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.as_tensor(value, device=device, dtype=dtype)


def _shape_cfg(
    density: float,
    friction: float,
    contact_stiffness: float,
    contact_damping: float,
    contact_margin: float,
) -> newton.ModelBuilder.ShapeConfig:
    return newton.ModelBuilder.ShapeConfig(
        density=density,
        ke=contact_stiffness,
        kd=contact_damping,
        kf=contact_stiffness,
        mu=friction,
        margin=contact_margin,
        mu_torsional=0.0,
        mu_rolling=0.0,
    )


def _box_volume(half_extents: np.ndarray) -> float:
    return float(8.0 * half_extents[0] * half_extents[1] * half_extents[2])


def _normalize_direction(direction: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        raise ValueError("Force direction must be non-zero")
    return (direction / norm).astype(np.float32)


def _compute_point_force_wrench(
    body_pose: torch.Tensor,
    force_magnitude: float,
    force_direction: np.ndarray,
    force_point_world: np.ndarray | None = None,
    force_point_local: np.ndarray | None = None,
) -> np.ndarray:
    force_vec = torch.as_tensor(
        _normalize_direction(np.asarray(force_direction, dtype=np.float32)) * np.float32(force_magnitude),
        device=body_pose.device,
        dtype=body_pose.dtype,
    )

    if force_point_local is not None:
        local_point = torch.as_tensor(force_point_local, device=body_pose.device, dtype=body_pose.dtype)
        world_point = transform_points(local_point.unsqueeze(0), body_pose[:3], body_pose[3:])[0]
    elif force_point_world is not None:
        world_point = torch.as_tensor(force_point_world, device=body_pose.device, dtype=body_pose.dtype)
    else:
        world_point = body_pose[:3]

    lever_arm = world_point - body_pose[:3]
    torque_vec = torch.cross(lever_arm, force_vec, dim=0)
    wrench = torch.cat([force_vec, torque_vec], dim=0)
    return wrench.detach().cpu().numpy().astype(np.float32, copy=False)


def _body_pose_tensor(state: newton.State, body_id: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(state.body_q.numpy()[body_id], device=device, dtype=dtype)


def _make_cluster(
    *,
    name: str,
    body_id: int,
    rest_translation: np.ndarray,
    half_extents: np.ndarray,
    display_color: tuple[float, float, float],
    is_dynamic: bool,
    mass: float,
    torch_device: torch.device,
    dtype: torch.dtype,
) -> RigidBodyCluster:
    return RigidBodyCluster(
        name=name,
        segmentation_id=body_id,
        body_id=body_id,
        local_shape_positions=torch.empty((0, 3), device=torch_device, dtype=dtype),
        shape_radius=_to_tensor(0.0, device=torch_device, dtype=dtype),
        total_mass=_to_tensor(mass, device=torch_device, dtype=dtype),
        rest_translation=_to_tensor(rest_translation, device=torch_device, dtype=dtype),
        fixed_orientation=IDENTITY_QUAT.to(device=torch_device, dtype=dtype).clone(),
        is_dynamic=is_dynamic,
        planar_motion=False,
        display_color=display_color,
        control_mode="free" if is_dynamic else "fixed",
        collision_geometry="box",
        collision_shape_start=body_id,
        collision_shape_count=1,
        box_half_extents=_to_tensor(half_extents, device=torch_device, dtype=dtype),
        inertia_factor_diag=torch.ones(3, device=torch_device, dtype=dtype),
        support_radius=_to_tensor(np.linalg.norm(half_extents[:2]), device=torch_device, dtype=dtype),
    )


def build_box_scene(args: argparse.Namespace) -> tuple[BuiltScene, newton.Model, newton.State, newton.State, newton.Control, newton.CollisionPipeline, newton.Contacts, int]:
    torch_device = torch.device(args.device if args.device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float32

    floor_half_extents = np.asarray(args.floor_half_extents, dtype=np.float32)
    box_half_extents = np.asarray(args.box_half_extents, dtype=np.float32)
    floor_center = np.asarray([0.0, 0.0, -float(floor_half_extents[2])], dtype=np.float32)
    if args.box_start_pos is None:
        box_center = np.asarray(
            [
                0.0,
                0.0,
                float(box_half_extents[2]),
            ],
            dtype=np.float32,
        )
    else:
        box_center = np.asarray(args.box_start_pos, dtype=np.float32)

    builder = newton.ModelBuilder(gravity=-9.81)

    floor_body = builder.add_body(xform=make_transform(floor_center), is_kinematic=True, label="floor")
    builder.add_shape_box(
        body=floor_body,
        hx=float(floor_half_extents[0]),
        hy=float(floor_half_extents[1]),
        hz=float(floor_half_extents[2]),
        cfg=_shape_cfg(
            density=1.0,
            friction=float(args.floor_friction),
            contact_stiffness=float(args.contact_stiffness),
            contact_damping=float(args.contact_damping),
            contact_margin=float(args.contact_margin),
        ),
        label="floor_box",
    )

    box_body = builder.add_body(xform=make_transform(box_center), label="box")
    box_density = float(args.box_mass) / max(_box_volume(box_half_extents), 1e-8)
    builder.add_shape_box(
        body=box_body,
        hx=float(box_half_extents[0]),
        hy=float(box_half_extents[1]),
        hz=float(box_half_extents[2]),
        cfg=_shape_cfg(
            density=box_density,
            friction=float(args.box_friction),
            contact_stiffness=float(args.contact_stiffness),
            contact_damping=float(args.contact_damping),
            contact_margin=float(args.contact_margin),
        ),
        label="box_box",
    )

    model = builder.finalize(device=str(torch_device))
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    collision_pipeline = newton.CollisionPipeline(model, rigid_contact_max=60000)
    contacts = collision_pipeline.contacts()

    box_mass = float(model.body_mass.numpy()[box_body])
    floor_cluster = _make_cluster(
        name="floor",
        body_id=floor_body,
        rest_translation=floor_center,
        half_extents=floor_half_extents,
        display_color=CLUSTER_COLORS["table"],
        is_dynamic=False,
        mass=0.0,
        torch_device=torch_device,
        dtype=dtype,
    )
    box_cluster = _make_cluster(
        name="box",
        body_id=box_body,
        rest_translation=box_center,
        half_extents=box_half_extents,
        display_color=CLUSTER_COLORS["tee"],
        is_dynamic=True,
        mass=box_mass,
        torch_device=torch_device,
        dtype=dtype,
    )

    body_q = torch.stack(
        [
            torch.cat([floor_cluster.rest_translation, floor_cluster.fixed_orientation], dim=0),
            torch.cat([box_cluster.rest_translation, box_cluster.fixed_orientation], dim=0),
        ],
        dim=0,
    )
    body_qd = torch.zeros((2, 6), device=torch_device, dtype=dtype)

    scene = BuiltScene(
        state_0=SceneState(body_q=body_q, body_qd=body_qd),
        state_1=SceneState(body_q=body_q.clone(), body_qd=body_qd.clone()),
        clusters=[floor_cluster, box_cluster],
        cluster_target_translations={},
        cluster_command_velocities={},
        constraint_iterations=1,
        table_friction=_to_tensor(args.floor_friction, device=torch_device, dtype=dtype),
        object_friction=_to_tensor(args.box_friction, device=torch_device, dtype=dtype),
        contact_stiffness=_to_tensor(args.contact_stiffness, device=torch_device, dtype=dtype),
        contact_damping=_to_tensor(args.contact_damping, device=torch_device, dtype=dtype),
        contact_margin=_to_tensor(args.contact_margin, device=torch_device, dtype=dtype),
        friction_regularization=_to_tensor(args.friction_regularization, device=torch_device, dtype=dtype),
        gravity=_to_tensor([0.0, 0.0, -9.81], device=torch_device, dtype=dtype),
        device=torch_device,
        dtype=dtype,
        collision_model=None,
        collision_state=None,
        collision_pipeline=None,
        collision_contacts=None,
    )

    return scene, model, state_0, state_1, control, collision_pipeline, contacts, box_body


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--dt", type=float, default=1.0 / 240.0)
    parser.add_argument("--solver-iterations", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--scene-usd-path", type=Path, default=DEFAULT_BOX_SCENE_USD_PATH)
    parser.add_argument("--box-mass", type=float, default=1.0)
    parser.add_argument("--floor-fraction", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--floor-half-extents",
        type=float,
        nargs=3,
        default=(2.0, 2.0, 0.05),
    )
    parser.add_argument(
        "--box-half-extents",
        type=float,
        nargs=3,
        default=(0.15, 0.30, 0.075),
    )
    parser.add_argument(
        "--box-start-pos",
        type=float,
        nargs=3,
        default=None,
        help="Optional initial box center. If omitted, the box starts in contact with the floor.",
    )
    parser.add_argument(
        "--initial-force",
        type=float,
        nargs=3,
        default=(50.0, 0.0, 0.0),
    )
    parser.add_argument(
        "--initial-torque",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
    )
    parser.add_argument(
        "--force-magnitude",
        type=float,
        default=None,
        help="If set, apply a point force with this magnitude instead of a direct wrench.",
    )
    parser.add_argument(
        "--force-direction",
        type=float,
        nargs=3,
        default=None,
        help="Direction of the point force in world coordinates.",
    )
    parser.add_argument(
        "--force-point",
        type=float,
        nargs=3,
        default=None,
        help="World-space point where the force is applied.",
    )
    parser.add_argument(
        "--force-point-local",
        type=float,
        nargs=3,
        default=None,
        help="Body-local point where the force is applied. Takes precedence over --force-point.",
    )
    parser.add_argument(
        "--force-steps",
        type=int,
        default=6,
        help="How many initial simulation steps the force is applied for.",
    )
    parser.add_argument("--floor-friction", type=float, default=0.6)
    parser.add_argument(
        "--box-friction",
        type=float,
        default=None,
        help="Box friction. Defaults to --floor-friction so both contact materials match.",
    )
    parser.add_argument("--contact-stiffness", type=float, default=DEFAULT_CONTACT_STIFFNESS)
    parser.add_argument("--contact-damping", type=float, default=DEFAULT_CONTACT_DAMPING)
    parser.add_argument("--contact-margin", type=float, default=DEFAULT_CONTACT_MARGIN)
    parser.add_argument(
        "--friction-regularization",
        type=float,
        default=DEFAULT_FRICTION_REGULARIZATION,
    )
    args = parser.parse_args()
    if args.box_friction is None:
        args.box_friction = args.floor_friction
    return args


def main() -> None:
    args = parse_args()
    scene, model, state_0, state_1, control, collision_pipeline, contacts, box_body = build_box_scene(args)

    solver = newton.solvers.SolverXPBD(model=model, iterations=max(int(args.solver_iterations), 1))
    body_q_frames: list[np.ndarray] = [state_0.body_q.numpy().copy()]

    use_point_force = (
        args.force_magnitude is not None
        or args.force_direction is not None
        or args.force_point is not None
        or args.force_point_local is not None
    )
    if use_point_force and (args.force_magnitude is None or args.force_direction is None):
        raise ValueError("When using point-force mode, both --force-magnitude and --force-direction are required")
    if args.force_point is not None and args.force_point_local is not None:
        raise ValueError("Specify only one of --force-point or --force-point-local")

    legacy_wrench = np.zeros(model.body_count * 6, dtype=np.float32)
    legacy_wrench[box_body * 6 : box_body * 6 + 3] = np.asarray(args.initial_force, dtype=np.float32)
    legacy_wrench[box_body * 6 + 3 : box_body * 6 + 6] = np.asarray(args.initial_torque, dtype=np.float32)

    for step_idx in range(max(int(args.steps), 0)):
        state_0.clear_forces()
        state_1.clear_forces()
        if step_idx < max(int(args.force_steps), 0):
            if use_point_force:
                body_wrench = _compute_point_force_wrench(
                    body_pose=_body_pose_tensor(state_0, box_body, scene.device, scene.dtype),
                    force_magnitude=float(args.force_magnitude),
                    force_direction=np.asarray(args.force_direction, dtype=np.float32),
                    force_point_world=None if args.force_point_local is not None else args.force_point,
                    force_point_local=args.force_point_local,
                )
                wrench = np.zeros(model.body_count * 6, dtype=np.float32)
                wrench[box_body * 6 : box_body * 6 + 6] = body_wrench
                state_0.body_f.assign(wrench)
                state_1.body_f.assign(wrench)
            else:
                state_0.body_f.assign(legacy_wrench)
                state_1.body_f.assign(legacy_wrench)
        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, float(args.dt))
        state_0, state_1 = state_1, state_0
        body_q_frames.append(state_0.body_q.numpy().copy())

    export_scene_usd(
        scene=scene,
        output_path=args.scene_usd_path,
        body_q_frames=body_q_frames,
        fps=1.0 / float(args.dt),
    )

    box_pose = state_0.body_q.numpy()[box_body]
    box_vel = state_0.body_qd.numpy()[box_body]
    print(f"USD written to {args.scene_usd_path.resolve()}")
    print(f"final_box_position={box_pose[:3].tolist()} final_box_velocity={box_vel[:3].tolist()}")


if __name__ == "__main__":
    main()
