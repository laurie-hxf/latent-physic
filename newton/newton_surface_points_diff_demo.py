from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import warp as wp

import newton
from newton_surface_points_demo import (
    _make_cluster,
    _make_surface_vis_cluster,
    _normalize_direction,
    _shape_cfg,
    compute_mass_properties,
    sample_box_surface_points,
)
from pbd_math import make_transform
from pbd_types import CLUSTER_COLORS, BuiltScene, SceneState
from pbd_usd import export_scene_usd
from project_paths import DEFAULT_OUTPUT_DIR


DEFAULT_DIFF_SURFACE_POINTS_SCENE_USD_PATH = DEFAULT_OUTPUT_DIR / "newton_surface_points_diff_demo.usda"
GRAVITY_MAGNITUDE = 9.81


@dataclass
class DiffParams:
    force_vector: wp.array
    torque_vector: wp.array | None
    force_point: wp.array | None
    point_friction: wp.array
    contact_weighted_masses: wp.array
    contact_weighted_mass_total: wp.array
    loss: wp.array
    target_position: wp.vec3
    use_point_force: bool
    force_point_is_local: bool


@dataclass
class DiffScene:
    scene: BuiltScene
    model: newton.Model
    states: list[newton.State]
    control: newton.Control
    collision_pipeline: newton.CollisionPipeline
    contacts: newton.Contacts
    solver: newton.solvers.SolverXPBD
    box_body: int
    box_body_ids_np: np.ndarray
    box_body_ids_wp: wp.array
    batch_capacity: int
    box_mass: float
    floor_top_z: float
    box_center: np.ndarray
    local_surface_points_np: np.ndarray
    point_masses_np: np.ndarray
    local_surface_points_wp: wp.array
    point_masses_wp: wp.array
    torch_device: torch.device
    torch_dtype: torch.dtype


def _to_tensor(
    value: float | list[float] | tuple[float, ...] | np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.as_tensor(value, device=device, dtype=dtype)


@wp.func
def _clamp01(value: float) -> float:
    return wp.min(wp.max(value, 0.0), 1.0)


@wp.func
def _smoothstep01(value: float) -> float:
    t = _clamp01(value)
    return t * t * (3.0 - 2.0 * t)


@wp.kernel
def apply_body_wrench_kernel(
    body_id: int,
    force_vector: wp.array(dtype=wp.vec3),
    torque_vector: wp.array(dtype=wp.vec3),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    force = force_vector[0]
    torque = torque_vector[0]
    wp.atomic_add(body_f, body_id, wp.spatial_vector(force, torque))


@wp.kernel
def apply_point_force_kernel(
    body_id: int,
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    force_vector: wp.array(dtype=wp.vec3),
    force_point: wp.array(dtype=wp.vec3),
    force_point_is_local: int,
    body_f: wp.array(dtype=wp.spatial_vector),
):
    force = force_vector[0]
    pose = body_q[body_id]
    point = force_point[0]
    world_point = point
    if force_point_is_local != 0:
        world_point = wp.transform_point(pose, point)

    world_com = wp.transform_point(pose, body_com[body_id])
    moment_arm = world_point - world_com
    torque = wp.cross(moment_arm, force)
    wp.atomic_add(body_f, body_id, wp.spatial_vector(force, torque))


@wp.kernel
def compute_contact_weighted_masses_kernel(
    body_id: int,
    body_q: wp.array(dtype=wp.transform),
    local_surface_points: wp.array(dtype=wp.vec3),
    point_masses: wp.array(dtype=float),
    floor_top_z: float,
    contact_band: float,
    weighted_masses: wp.array(dtype=float),
    total_weighted_mass: wp.array(dtype=float),
):
    tid = wp.tid()
    pose = body_q[body_id]
    world_point = wp.transform_point(pose, local_surface_points[tid])
    gap = world_point[2] - floor_top_z
    safe_band = wp.max(contact_band, 1.0e-6)
    activation = _smoothstep01((contact_band - gap) / safe_band)
    weighted_mass = activation * point_masses[tid]
    weighted_masses[tid] = weighted_mass
    wp.atomic_add(total_weighted_mass, 0, weighted_mass)


@wp.kernel
def apply_surface_point_normal_kernel(
    body_id: int,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    local_surface_points: wp.array(dtype=wp.vec3),
    weighted_masses: wp.array(dtype=float),
    total_weighted_mass: wp.array(dtype=float),
    external_force_vector: wp.array(dtype=wp.vec3),
    force_active: int,
    total_mass: float,
    gravity_magnitude: float,
    floor_top_z: float,
    contact_stiffness: float,
    contact_damping: float,
    contact_band: float,
    body_f: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    total_weight = total_weighted_mass[0]
    if total_weight <= 1.0e-8:
        return

    pose = body_q[body_id]
    qd = body_qd[body_id]
    world_com = wp.transform_point(pose, body_com[body_id])
    world_point = wp.transform_point(pose, local_surface_points[tid])
    moment_arm = world_point - world_com

    linear_velocity = wp.spatial_top(qd)
    angular_velocity = wp.spatial_bottom(qd)
    point_velocity = linear_velocity + wp.cross(angular_velocity, moment_arm)

    gap = world_point[2] - floor_top_z
    penetration = wp.max(-gap, 0.0)
    safe_band = wp.max(contact_band, 1.0e-6)
    activation = _smoothstep01((contact_band - gap) / safe_band)
    mass_fraction = weighted_masses[tid] / total_weight

    external_force = external_force_vector[0] * float(force_active)
    support_force_z = mass_fraction * wp.max(0.0, total_mass * gravity_magnitude - external_force[2])
    penalty_force_z = mass_fraction * activation * (
        contact_stiffness * penetration + contact_damping * wp.max(-point_velocity[2], 0.0)
    )
    normal_force = wp.vec3(0.0, 0.0, support_force_z + penalty_force_z)
    normal_torque = wp.cross(moment_arm, normal_force)
    wp.atomic_add(body_f, body_id, wp.spatial_vector(normal_force, normal_torque))


@wp.kernel
def apply_surface_point_friction_kernel(
    body_id: int,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    local_surface_points: wp.array(dtype=wp.vec3),
    weighted_masses: wp.array(dtype=float),
    total_weighted_mass: wp.array(dtype=float),
    point_friction: wp.array(dtype=float),
    external_force_vector: wp.array(dtype=wp.vec3),
    force_active: int,
    total_mass: float,
    gravity_magnitude: float,
    friction_regularization: float,
    body_f: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    total_weight = total_weighted_mass[0]
    if total_weight <= 1.0e-8:
        return

    pose = body_q[body_id]
    qd = body_qd[body_id]
    world_com = wp.transform_point(pose, body_com[body_id])
    world_point = wp.transform_point(pose, local_surface_points[tid])
    moment_arm = world_point - world_com

    linear_velocity = wp.spatial_top(qd)
    angular_velocity = wp.spatial_bottom(qd)
    point_velocity = linear_velocity + wp.cross(angular_velocity, moment_arm)
    tangential_velocity = wp.vec3(point_velocity[0], point_velocity[1], 0.0)
    tangential_speed = wp.sqrt(
        wp.dot(tangential_velocity, tangential_velocity) + friction_regularization * friction_regularization
    )

    external_force = external_force_vector[0] * float(force_active)
    normal_load_total = wp.max(0.0, total_mass * gravity_magnitude - external_force[2])
    normal_load = (weighted_masses[tid] / total_weight) * normal_load_total
    mu = wp.max(point_friction[0], 0.0)

    friction_force = -mu * normal_load * (tangential_velocity / tangential_speed)
    friction_torque = wp.cross(moment_arm, friction_force)
    wp.atomic_add(body_f, body_id, wp.spatial_vector(friction_force, friction_torque))


@wp.kernel
def final_com_position_loss_kernel(
    body_id: int,
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    target_position: wp.vec3,
    loss: wp.array(dtype=float),
):
    pose = body_q[body_id]
    world_com = wp.transform_point(pose, body_com[body_id])
    delta = world_com - target_position
    loss[0] = wp.dot(delta, delta)


def build_diff_scene(args: argparse.Namespace) -> DiffScene:
    torch_device = torch.device(
        args.device if args.device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    torch_dtype = torch.float32
    warp_device = str(torch_device)

    floor_half_extents = np.asarray(args.floor_half_extents, dtype=np.float32)
    box_half_extents = np.asarray(args.box_half_extents, dtype=np.float32)
    batch_capacity = max(int(getattr(args, "batch_capacity", 1)), 1)
    floor_center = np.asarray([0.0, 0.0, -float(floor_half_extents[2])], dtype=np.float32)
    if args.box_start_pos is None:
        box_center = np.asarray([0.0, 0.0, float(box_half_extents[2])], dtype=np.float32)
    else:
        box_center = np.asarray(args.box_start_pos, dtype=np.float32)

    local_surface_points_np, point_masses_np = sample_box_surface_points(
        half_extents=box_half_extents,
        spacing=float(args.surface_point_spacing),
        total_mass=float(args.box_mass),
    )
    total_mass, local_com, inertia = compute_mass_properties(local_surface_points_np, point_masses_np)

    builder = newton.ModelBuilder(gravity=-GRAVITY_MAGNITUDE)

    floor_body = -1
    box_body_ids: list[int] = []
    for world_idx in range(batch_capacity):
        builder.begin_world(label=f"trajectory_{world_idx}")
        world_floor_body = builder.add_body(xform=make_transform(floor_center), is_kinematic=True, label=f"floor_{world_idx}")
        builder.add_shape_box(
            body=world_floor_body,
            hx=float(floor_half_extents[0]),
            hy=float(floor_half_extents[1]),
            hz=float(floor_half_extents[2]),
            cfg=_shape_cfg(
                density=1.0,
                friction=float(args.contact_friction),
                contact_stiffness=float(args.contact_stiffness),
                contact_damping=float(args.contact_damping),
                contact_margin=float(args.contact_margin),
            ),
            label=f"floor_box_{world_idx}",
        )

        world_box_body = builder.add_body(
            xform=make_transform(box_center),
            mass=total_mass,
            com=wp.vec3(float(local_com[0]), float(local_com[1]), float(local_com[2])),
            inertia=wp.mat33(inertia),
            lock_inertia=True,
            label=f"box_{world_idx}",
        )
        builder.add_shape_box(
            body=world_box_body,
            hx=float(box_half_extents[0]),
            hy=float(box_half_extents[1]),
            hz=float(box_half_extents[2]),
            cfg=_shape_cfg(
                density=1.0,
                friction=float(args.contact_friction),
                contact_stiffness=float(args.contact_stiffness),
                contact_damping=float(args.contact_damping),
                contact_margin=float(args.contact_margin),
            ),
            label=f"box_box_{world_idx}",
        )
        builder.end_world()

        if world_idx == 0:
            floor_body = world_floor_body
        box_body_ids.append(world_box_body)

    box_body_ids_np = np.asarray(box_body_ids, dtype=np.int32)
    box_body = int(box_body_ids_np[0])

    model = builder.finalize(device=warp_device, requires_grad=True)
    states = [model.state() for _ in range(max(int(args.steps), 0) + 1)]
    control = model.control()
    collision_pipeline = newton.CollisionPipeline(
        model,
        rigid_contact_max=60000,
        requires_grad=True,
    )
    contacts = collision_pipeline.contacts()
    solver = newton.solvers.SolverXPBD(model=model, iterations=max(int(args.solver_iterations), 1))

    floor_cluster = _make_cluster(
        name="floor",
        body_id=floor_body,
        rest_translation=floor_center,
        half_extents=floor_half_extents,
        local_surface_points=np.empty((0, 3), dtype=np.float32),
        display_color=CLUSTER_COLORS["table"],
        is_dynamic=False,
        mass=0.0,
        torch_device=torch_device,
        dtype=torch_dtype,
    )
    box_cluster = _make_cluster(
        name="box",
        body_id=box_body,
        rest_translation=box_center,
        half_extents=box_half_extents,
        local_surface_points=local_surface_points_np,
        display_color=CLUSTER_COLORS["tee"],
        is_dynamic=True,
        mass=total_mass,
        torch_device=torch_device,
        dtype=torch_dtype,
    )
    surface_vis_cluster = _make_surface_vis_cluster(
        body_id=box_body,
        rest_translation=box_center,
        local_surface_points=local_surface_points_np,
        point_radius=max(float(args.surface_point_spacing) * 0.18, 1.0e-4),
        torch_device=torch_device,
        dtype=torch_dtype,
    )

    initial_body_q = torch.as_tensor(states[0].body_q.numpy(), device=torch_device, dtype=torch_dtype)
    initial_body_qd = torch.zeros((model.body_count, 6), device=torch_device, dtype=torch_dtype)

    scene = BuiltScene(
        state_0=SceneState(body_q=initial_body_q, body_qd=initial_body_qd),
        state_1=SceneState(body_q=initial_body_q.clone(), body_qd=initial_body_qd.clone()),
        clusters=[floor_cluster, box_cluster, surface_vis_cluster],
        cluster_target_translations={},
        cluster_command_velocities={},
        constraint_iterations=1,
        table_friction=_to_tensor(args.contact_friction, device=torch_device, dtype=torch_dtype),
        object_friction=_to_tensor(args.point_friction, device=torch_device, dtype=torch_dtype),
        contact_stiffness=_to_tensor(args.contact_stiffness, device=torch_device, dtype=torch_dtype),
        contact_damping=_to_tensor(args.contact_damping, device=torch_device, dtype=torch_dtype),
        contact_margin=_to_tensor(args.contact_margin, device=torch_device, dtype=torch_dtype),
        friction_regularization=_to_tensor(args.friction_regularization, device=torch_device, dtype=torch_dtype),
        gravity=_to_tensor([0.0, 0.0, -GRAVITY_MAGNITUDE], device=torch_device, dtype=torch_dtype),
        device=torch_device,
        dtype=torch_dtype,
        collision_model=None,
        collision_state=None,
        collision_pipeline=None,
        collision_contacts=None,
    )

    return DiffScene(
        scene=scene,
        model=model,
        states=states,
        control=control,
        collision_pipeline=collision_pipeline,
        contacts=contacts,
        solver=solver,
        box_body=box_body,
        box_body_ids_np=box_body_ids_np,
        box_body_ids_wp=wp.array(box_body_ids_np, dtype=wp.int32, device=warp_device),
        batch_capacity=batch_capacity,
        box_mass=float(total_mass),
        floor_top_z=float(floor_center[2] + floor_half_extents[2]),
        box_center=box_center,
        local_surface_points_np=local_surface_points_np,
        point_masses_np=point_masses_np,
        local_surface_points_wp=wp.array(local_surface_points_np, dtype=wp.vec3, device=warp_device),
        point_masses_wp=wp.array(point_masses_np, dtype=wp.float32, device=warp_device),
        torch_device=torch_device,
        torch_dtype=torch_dtype,
    )


def build_diff_params(args: argparse.Namespace, diff_scene: DiffScene) -> DiffParams:
    device = str(diff_scene.torch_device)
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

    if use_point_force:
        force_vector_init = _normalize_direction(np.asarray(args.force_direction, dtype=np.float32)) * np.float32(
            args.force_magnitude
        )
        torque_vector = None
        if args.force_point_local is not None:
            force_point_init = np.asarray(args.force_point_local, dtype=np.float32)
            force_point_is_local = True
        elif args.force_point is not None:
            force_point_init = np.asarray(args.force_point, dtype=np.float32)
            force_point_is_local = False
        else:
            force_point_init = np.zeros(3, dtype=np.float32)
            force_point_is_local = True
        force_point = wp.array([force_point_init], dtype=wp.vec3, device=device, requires_grad=True)
    else:
        force_vector_init = np.asarray(args.initial_force, dtype=np.float32)
        torque_vector = wp.array([np.asarray(args.initial_torque, dtype=np.float32)], dtype=wp.vec3, device=device, requires_grad=True)
        force_point = None
        force_point_is_local = False

    if args.loss_target_position is None:
        target_position_np = diff_scene.box_center.copy()
        target_position_np[0] += 0.25
    else:
        target_position_np = np.asarray(args.loss_target_position, dtype=np.float32)

    return DiffParams(
        force_vector=wp.array([force_vector_init], dtype=wp.vec3, device=device, requires_grad=True),
        torque_vector=torque_vector,
        force_point=force_point,
        point_friction=wp.array([np.float32(args.point_friction)], dtype=wp.float32, device=device, requires_grad=True),
        contact_weighted_masses=wp.zeros(
            len(diff_scene.local_surface_points_np),
            dtype=wp.float32,
            device=device,
            requires_grad=True,
        ),
        contact_weighted_mass_total=wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True),
        loss=wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True),
        target_position=wp.vec3(
            float(target_position_np[0]),
            float(target_position_np[1]),
            float(target_position_np[2]),
        ),
        use_point_force=use_point_force,
        force_point_is_local=force_point_is_local,
    )


def forward_rollout(diff_scene: DiffScene, diff_params: DiffParams, args: argparse.Namespace) -> wp.array:
    for step_idx in range(max(int(args.steps), 0)):
        state_in = diff_scene.states[step_idx]
        state_out = diff_scene.states[step_idx + 1]
        state_in.clear_forces()

        force_active = int(step_idx < max(int(args.force_steps), 0))
        if force_active:
            if diff_params.use_point_force:
                wp.launch(
                    apply_point_force_kernel,
                    dim=1,
                    inputs=[
                        diff_scene.box_body,
                        state_in.body_q,
                        diff_scene.model.body_com,
                        diff_params.force_vector,
                        diff_params.force_point,
                        int(diff_params.force_point_is_local),
                        state_in.body_f,
                    ],
                    device=diff_scene.model.device,
                )
            else:
                wp.launch(
                    apply_body_wrench_kernel,
                    dim=1,
                    inputs=[
                        diff_scene.box_body,
                        diff_params.force_vector,
                        diff_params.torque_vector,
                        state_in.body_f,
                    ],
                    device=diff_scene.model.device,
                )

        diff_params.contact_weighted_masses.zero_()
        diff_params.contact_weighted_mass_total.zero_()
        wp.launch(
            compute_contact_weighted_masses_kernel,
            dim=len(diff_scene.local_surface_points_np),
            inputs=[
                diff_scene.box_body,
                state_in.body_q,
                diff_scene.local_surface_points_wp,
                diff_scene.point_masses_wp,
                float(diff_scene.floor_top_z),
                float(args.friction_contact_threshold),
                diff_params.contact_weighted_masses,
                diff_params.contact_weighted_mass_total,
            ],
            device=diff_scene.model.device,
        )
        wp.launch(
            apply_surface_point_normal_kernel,
            dim=len(diff_scene.local_surface_points_np),
            inputs=[
                diff_scene.box_body,
                state_in.body_q,
                state_in.body_qd,
                diff_scene.model.body_com,
                diff_scene.local_surface_points_wp,
                diff_params.contact_weighted_masses,
                diff_params.contact_weighted_mass_total,
                diff_params.force_vector,
                force_active,
                float(diff_scene.box_mass),
                float(GRAVITY_MAGNITUDE),
                float(diff_scene.floor_top_z),
                float(args.contact_stiffness),
                float(args.contact_damping),
                float(args.friction_contact_threshold),
                state_in.body_f,
            ],
            device=diff_scene.model.device,
        )
        wp.launch(
            apply_surface_point_friction_kernel,
            dim=len(diff_scene.local_surface_points_np),
            inputs=[
                diff_scene.box_body,
                state_in.body_q,
                state_in.body_qd,
                diff_scene.model.body_com,
                diff_scene.local_surface_points_wp,
                diff_params.contact_weighted_masses,
                diff_params.contact_weighted_mass_total,
                diff_params.point_friction,
                diff_params.force_vector,
                force_active,
                float(diff_scene.box_mass),
                float(GRAVITY_MAGNITUDE),
                float(args.friction_regularization),
                state_in.body_f,
            ],
            device=diff_scene.model.device,
        )

        diff_scene.collision_pipeline.collide(state_in, diff_scene.contacts)
        diff_scene.solver.step(state_in, state_out, diff_scene.control, diff_scene.contacts, float(args.dt))

    diff_params.loss.zero_()
    wp.launch(
        final_com_position_loss_kernel,
        dim=1,
        inputs=[
            diff_scene.box_body,
            diff_scene.states[max(int(args.steps), 0)].body_q,
            diff_scene.model.body_com,
            diff_params.target_position,
            diff_params.loss,
        ],
        device=diff_scene.model.device,
    )
    return diff_params.loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--dt", type=float, default=1.0 / 240.0)
    parser.add_argument("--solver-iterations", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--scene-usd-path", type=Path, default=DEFAULT_DIFF_SURFACE_POINTS_SCENE_USD_PATH)
    parser.add_argument("--box-mass", type=float, default=1.0)
    parser.add_argument("--floor-half-extents", type=float, nargs=3, default=(2.0, 2.0, 0.05))
    parser.add_argument("--box-half-extents", type=float, nargs=3, default=(0.15, 0.30, 0.075))
    parser.add_argument("--box-start-pos", type=float, nargs=3, default=None)
    parser.add_argument("--surface-point-spacing", type=float, default=0.03)
    parser.add_argument(
        "--friction-contact-threshold",
        type=float,
        default=0.002,
        help="Smooth activation band for surface-point friction near the floor.",
    )
    parser.add_argument(
        "--point-friction",
        type=float,
        default=0.6,
        help="Custom friction coefficient applied through surface points.",
    )
    parser.add_argument(
        "--contact-friction",
        type=float,
        default=0.0,
        help="Newton shape friction. Keep this at 0 to avoid double-counting friction.",
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
    parser.add_argument("--force-steps", type=int, default=6)
    parser.add_argument("--contact-stiffness", type=float, default=2.0e4)
    parser.add_argument("--contact-damping", type=float, default=50.0)
    parser.add_argument("--contact-margin", type=float, default=1.0e-3)
    parser.add_argument("--friction-regularization", type=float, default=1.0e-3)
    parser.add_argument(
        "--loss-target-position",
        type=float,
        nargs=3,
        default=None,
        help="Target COM position used to build the differentiable loss.",
    )
    return parser.parse_args()


def _format_vec3(value: np.ndarray | list[float]) -> str:
    return np.asarray(value, dtype=np.float32).tolist().__repr__()


def main() -> None:
    args = parse_args()
    diff_scene = build_diff_scene(args)
    diff_params = build_diff_params(args, diff_scene)

    tape = wp.Tape()
    with tape:
        loss = forward_rollout(diff_scene, diff_params, args)
    tape.backward(loss)

    body_q_frames = [state.body_q.numpy().copy() for state in diff_scene.states]
    export_scene_usd(
        scene=diff_scene.scene,
        output_path=args.scene_usd_path,
        body_q_frames=body_q_frames,
        fps=1.0 / float(args.dt),
    )

    final_pose = diff_scene.states[max(int(args.steps), 0)].body_q.numpy()[diff_scene.box_body]
    final_velocity = diff_scene.states[max(int(args.steps), 0)].body_qd.numpy()[diff_scene.box_body]
    body_inertia = diff_scene.model.body_inertia.numpy()[diff_scene.box_body]

    print(f"surface_points={len(diff_scene.local_surface_points_np)} total_point_mass={diff_scene.box_mass:.6f}")
    print(f"derived_body_inertia_diag={np.diag(body_inertia).tolist()}")
    print(f"loss_target_position={list(diff_params.target_position)} loss={float(diff_params.loss.numpy()[0]):.6f}")
    print(f"final_box_position={final_pose[:3].tolist()} final_box_velocity={final_velocity[:3].tolist()}")
    print(f"USD written to {args.scene_usd_path.resolve()}")
    print(f"grad_force_vector={_format_vec3(diff_params.force_vector.grad.numpy()[0])}")
    print(f"grad_point_friction={float(diff_params.point_friction.grad.numpy()[0]):.6f}")
    if diff_params.use_point_force:
        point_grad_name = "grad_force_point_local" if diff_params.force_point_is_local else "grad_force_point_world"
        point_value_name = "force_point_local" if diff_params.force_point_is_local else "force_point_world"
        print(f"{point_value_name}={_format_vec3(diff_params.force_point.numpy()[0])}")
        print(f"{point_grad_name}={_format_vec3(diff_params.force_point.grad.numpy()[0])}")
    else:
        print(f"grad_initial_torque={_format_vec3(diff_params.torque_vector.grad.numpy()[0])}")


if __name__ == "__main__":
    main()
