from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import warp as wp

import newton
from pbd_math import make_transform, quaternion_to_matrix, transform_points
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


DEFAULT_SURFACE_POINTS_SCENE_USD_PATH = DEFAULT_OUTPUT_DIR / "newton_surface_points_demo.usda"


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


def _axis_sample_count(half_extent: float, spacing: float, *, avoid_zero: bool = False) -> int:
    raw_count = (2.0 * half_extent) / max(spacing, 1.0e-6)
    nearest_count = int(np.rint(raw_count))
    if nearest_count >= 1 and np.isclose(raw_count, nearest_count, rtol=1.0e-6, atol=1.0e-8):
        count = nearest_count
    else:
        count = max(int(np.ceil(raw_count)), 1)
    if avoid_zero and half_extent > 0.0 and count % 2 == 1:
        count += 1
    return count


def _face_centers(
    half_a: float,
    half_b: float,
    spacing: float,
    *,
    avoid_zero_a: bool = False,
    avoid_zero_b: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    count_a = _axis_sample_count(half_a, spacing, avoid_zero=avoid_zero_a)
    count_b = _axis_sample_count(half_b, spacing, avoid_zero=avoid_zero_b)
    step_a = (2.0 * half_a) / count_a
    step_b = (2.0 * half_b) / count_b
    coords_a = -half_a + (np.arange(count_a, dtype=np.float32) + 0.5) * step_a
    coords_b = -half_b + (np.arange(count_b, dtype=np.float32) + 0.5) * step_b
    patch_area = float(step_a * step_b)
    return coords_a, coords_b, patch_area


def sample_box_surface_points(
    half_extents: np.ndarray,
    spacing: float,
    total_mass: float,
) -> tuple[np.ndarray, np.ndarray]:
    hx, hy, hz = [float(v) for v in half_extents]
    surface_area = 8.0 * (hx * hy + hy * hz + hz * hx)
    surface_density = float(total_mass) / max(surface_area, 1e-8)

    positions: list[list[float]] = []
    masses: list[float] = []

    yz_y, yz_z, yz_area = _face_centers(hy, hz, spacing)
    xz_x, xz_z, xz_area = _face_centers(hx, hz, spacing, avoid_zero_a=True)
    xy_x, xy_y, xy_area = _face_centers(hx, hy, spacing, avoid_zero_a=True)

    for face_x in (-hx, hx):
        point_mass = surface_density * yz_area
        for y in yz_y:
            for z in yz_z:
                positions.append([face_x, float(y), float(z)])
                masses.append(point_mass)

    for face_y in (-hy, hy):
        point_mass = surface_density * xz_area
        for x in xz_x:
            for z in xz_z:
                positions.append([float(x), face_y, float(z)])
                masses.append(point_mass)

    for face_z in (-hz, hz):
        point_mass = surface_density * xy_area
        for x in xy_x:
            for y in xy_y:
                positions.append([float(x), float(y), face_z])
                masses.append(point_mass)

    local_points = np.asarray(positions, dtype=np.float32)
    if hx > 0.0 and np.any(np.isclose(local_points[:, 0], 0.0, atol=1.0e-9)):
        raise RuntimeError("surface-point sampling generated local x=0 points")
    point_masses = np.asarray(masses, dtype=np.float32)
    point_masses *= float(total_mass) / max(float(point_masses.sum()), 1e-8)
    return local_points, point_masses


def compute_mass_properties(local_points: np.ndarray, point_masses: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    total_mass = float(point_masses.sum())
    com = np.sum(local_points * point_masses[:, None], axis=0) / max(total_mass, 1e-8)
    rel = local_points - com[None, :]
    inertia = np.zeros((3, 3), dtype=np.float32)
    eye = np.eye(3, dtype=np.float32)
    for point, mass in zip(rel, point_masses, strict=False):
        inertia += mass * ((float(np.dot(point, point)) * eye) - np.outer(point, point))
    return total_mass, com.astype(np.float32), inertia


def _make_cluster(
    *,
    name: str,
    body_id: int,
    rest_translation: np.ndarray,
    half_extents: np.ndarray,
    local_surface_points: np.ndarray,
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
        local_shape_positions=_to_tensor(local_surface_points, device=torch_device, dtype=dtype),
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


def _make_surface_vis_cluster(
    *,
    body_id: int,
    rest_translation: np.ndarray,
    local_surface_points: np.ndarray,
    point_radius: float,
    torch_device: torch.device,
    dtype: torch.dtype,
) -> RigidBodyCluster:
    return RigidBodyCluster(
        name="surface_points",
        segmentation_id=1000 + body_id,
        body_id=body_id,
        local_shape_positions=_to_tensor(local_surface_points, device=torch_device, dtype=dtype),
        shape_radius=_to_tensor(point_radius, device=torch_device, dtype=dtype),
        total_mass=_to_tensor(0.0, device=torch_device, dtype=dtype),
        rest_translation=_to_tensor(rest_translation, device=torch_device, dtype=dtype),
        fixed_orientation=IDENTITY_QUAT.to(device=torch_device, dtype=dtype).clone(),
        is_dynamic=False,
        planar_motion=False,
        display_color=(0.96, 0.92, 0.18),
        control_mode="fixed",
        collision_geometry="sphere_cluster",
        collision_shape_start=0,
        collision_shape_count=0,
        box_half_extents=None,
        inertia_factor_diag=torch.ones(3, device=torch_device, dtype=dtype),
        support_radius=_to_tensor(0.0, device=torch_device, dtype=dtype),
    )


def compute_surface_point_friction_wrench(
    *,
    body_q: np.ndarray,
    body_qd: np.ndarray,
    local_surface_points: np.ndarray,
    point_masses: np.ndarray,
    floor_top_z: float,
    point_frictions: np.ndarray,
    normal_load_total: float,
    contact_threshold: float,
    regularization: float,
) -> np.ndarray:
    world_points = transform_points(local_surface_points, body_q[:3], body_q[3:])
    active_mask = world_points[:, 2] <= (floor_top_z + contact_threshold)
    if not np.any(active_mask):
        return np.zeros(6, dtype=np.float32)

    active_points = world_points[active_mask]
    active_masses = point_masses[active_mask]
    active_frictions = point_frictions[active_mask]
    active_mass_sum = float(active_masses.sum())
    if active_mass_sum <= 1e-8 or normal_load_total <= 0.0:
        return np.zeros(6, dtype=np.float32)

    linear_velocity = body_qd[:3]
    angular_velocity = body_qd[3:]
    lever_arms = active_points - body_q[:3][None, :]
    point_velocities = linear_velocity[None, :] + np.cross(
        np.broadcast_to(angular_velocity[None, :], lever_arms.shape),
        lever_arms,
    )
    tangential_velocity = point_velocities.copy()
    tangential_velocity[:, 2] = 0.0
    tangential_speed = np.sqrt(
        np.sum(tangential_velocity * tangential_velocity, axis=1) + (regularization**2)
    )

    normal_loads = (active_masses / active_mass_sum) * normal_load_total
    friction_forces = -active_frictions[:, None] * normal_loads[:, None] * (
        tangential_velocity / tangential_speed[:, None]
    )
    total_force = friction_forces.sum(axis=0)
    total_torque = np.cross(lever_arms, friction_forces).sum(axis=0)
    return np.concatenate([total_force, total_torque], axis=0).astype(np.float32, copy=False)


def build_surface_points_scene(
    args: argparse.Namespace,
) -> tuple[
    BuiltScene,
    newton.Model,
    newton.State,
    newton.State,
    newton.Control,
    newton.CollisionPipeline,
    newton.Contacts,
    int,
    np.ndarray,
    np.ndarray,
    float,
]:
    torch_device = torch.device(
        args.device if args.device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    dtype = torch.float32

    floor_half_extents = np.asarray(args.floor_half_extents, dtype=np.float32)
    box_half_extents = np.asarray(args.box_half_extents, dtype=np.float32)
    floor_center = np.asarray([0.0, 0.0, -float(floor_half_extents[2])], dtype=np.float32)
    if args.box_start_pos is None:
        box_center = np.asarray([0.0, 0.0, float(box_half_extents[2])], dtype=np.float32)
    else:
        box_center = np.asarray(args.box_start_pos, dtype=np.float32)

    local_surface_points, point_masses = sample_box_surface_points(
        half_extents=box_half_extents,
        spacing=float(args.surface_point_spacing),
        total_mass=float(args.box_mass),
    )
    total_mass, local_com, inertia = compute_mass_properties(local_surface_points, point_masses)

    builder = newton.ModelBuilder(gravity=-9.81)

    floor_body = builder.add_body(xform=make_transform(floor_center), is_kinematic=True, label="floor")
    builder.add_shape_box(
        body=floor_body,
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
        label="floor_box",
    )

    box_body = builder.add_body(
        xform=make_transform(box_center),
        mass=total_mass,
        com=wp.vec3(float(local_com[0]), float(local_com[1]), float(local_com[2])),
        inertia=wp.mat33(inertia),
        lock_inertia=True,
        label="box",
    )
    builder.add_shape_box(
        body=box_body,
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
        label="box_box",
    )

    model = builder.finalize(device=str(torch_device))
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    collision_pipeline = newton.CollisionPipeline(model, rigid_contact_max=60000)
    contacts = collision_pipeline.contacts()

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
        dtype=dtype,
    )
    box_cluster = _make_cluster(
        name="box",
        body_id=box_body,
        rest_translation=box_center,
        half_extents=box_half_extents,
        local_surface_points=local_surface_points,
        display_color=CLUSTER_COLORS["tee"],
        is_dynamic=True,
        mass=total_mass,
        torch_device=torch_device,
        dtype=dtype,
    )
    surface_vis_cluster = _make_surface_vis_cluster(
        body_id=box_body,
        rest_translation=box_center,
        local_surface_points=local_surface_points,
        point_radius=max(float(args.surface_point_spacing) * 0.18, 1e-4),
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
        clusters=[floor_cluster, box_cluster, surface_vis_cluster],
        cluster_target_translations={},
        cluster_command_velocities={},
        constraint_iterations=1,
        table_friction=_to_tensor(args.contact_friction, device=torch_device, dtype=dtype),
        object_friction=_to_tensor(args.point_friction, device=torch_device, dtype=dtype),
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

    return (
        scene,
        model,
        state_0,
        state_1,
        control,
        collision_pipeline,
        contacts,
        box_body,
        local_surface_points,
        point_masses,
        float(floor_center[2] + floor_half_extents[2]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--dt", type=float, default=1.0 / 240.0)
    parser.add_argument("--solver-iterations", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--scene-usd-path", type=Path, default=DEFAULT_SURFACE_POINTS_SCENE_USD_PATH)
    parser.add_argument("--box-mass", type=float, default=1.0)
    parser.add_argument("--floor-half-extents", type=float, nargs=3, default=(2.0, 2.0, 0.05))
    parser.add_argument("--box-half-extents", type=float, nargs=3,default=(0.15, 0.30, 0.075))
    parser.add_argument("--box-start-pos", type=float, nargs=3, default=None)
    parser.add_argument("--surface-point-spacing", type=float, default=0.03)
    parser.add_argument(
        "--friction-contact-threshold",
        type=float,
        default=0.002,
        help="Surface point is considered friction-active when it is this close to the floor.",
    )
    parser.add_argument(
        "--point-friction",
        type=float,
        default=0.6,
        help="Fallback custom friction coefficient applied through surface points.",
    )
    parser.add_argument(
        "--left-point-friction",
        type=float,
        default=1.0,
        help="Friction coefficient for the left side of the box.",
    )
    parser.add_argument(
        "--right-point-friction",
        type=float,
        default=0.2,
        help="Friction coefficient for the right side of the box.",
    )
    parser.add_argument(
        "--point-friction-split-axis",
        choices=("x", "y"),
        default="y",
        help="Local box axis used to split left/right point friction.",
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
    parser.add_argument("--contact-stiffness", type=float, default=DEFAULT_CONTACT_STIFFNESS)
    parser.add_argument("--contact-damping", type=float, default=DEFAULT_CONTACT_DAMPING)
    parser.add_argument("--contact-margin", type=float, default=DEFAULT_CONTACT_MARGIN)
    parser.add_argument(
        "--friction-regularization",
        type=float,
        default=DEFAULT_FRICTION_REGULARIZATION,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    (
        scene,
        model,
        state_0,
        state_1,
        control,
        collision_pipeline,
        contacts,
        box_body,
        local_surface_points,
        point_masses,
        floor_top_z,
    ) = build_surface_points_scene(args)

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

    box_mass = float(point_masses.sum())
    gravity_load = box_mass * 9.81
    split_axis_idx = 0 if args.point_friction_split_axis == "x" else 1
    point_frictions = np.where(
        local_surface_points[:, split_axis_idx] < 0.0,
        max(float(args.left_point_friction), 0.0),
        max(float(args.right_point_friction), 0.0),
    ).astype(np.float32)

    for step_idx in range(max(int(args.steps), 0)):
        state_0.clear_forces()
        state_1.clear_forces()

        body_external_wrench = np.zeros(6, dtype=np.float32)
        if step_idx < max(int(args.force_steps), 0):
            if use_point_force:
                body_external_wrench = _compute_point_force_wrench(
                    body_pose=_body_pose_tensor(state_0, box_body, scene.device, scene.dtype),
                    force_magnitude=float(args.force_magnitude),
                    force_direction=np.asarray(args.force_direction, dtype=np.float32),
                    force_point_world=None if args.force_point_local is not None else args.force_point,
                    force_point_local=args.force_point_local,
                )
            else:
                body_external_wrench[:3] = np.asarray(args.initial_force, dtype=np.float32)
                body_external_wrench[3:] = np.asarray(args.initial_torque, dtype=np.float32)

        body_q = state_0.body_q.numpy()[box_body]
        body_qd = state_0.body_qd.numpy()[box_body]
        normal_load_total = max(0.0, gravity_load - float(body_external_wrench[2]))
        friction_wrench = compute_surface_point_friction_wrench(
            body_q=body_q,
            body_qd=body_qd,
            local_surface_points=local_surface_points,
            point_masses=point_masses,
            floor_top_z=floor_top_z,
            point_frictions=point_frictions,
            normal_load_total=normal_load_total,
            contact_threshold=float(args.friction_contact_threshold),
            regularization=float(args.friction_regularization),
        )

        total_wrench = np.zeros(model.body_count * 6, dtype=np.float32)
        total_wrench[box_body * 6 : box_body * 6 + 6] = body_external_wrench + friction_wrench
        state_0.body_f.assign(total_wrench)
        state_1.body_f.assign(total_wrench)

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
    body_inertia = model.body_inertia.numpy()[box_body]
    print(f"surface_points={len(local_surface_points)} total_point_mass={box_mass:.6f}")
    print(
        "point_friction_stats="
        f"split_axis={args.point_friction_split_axis} "
        f"left={float(args.left_point_friction):.6f} "
        f"right={float(args.right_point_friction):.6f} "
        f"min={float(point_frictions.min()):.6f} "
        f"mean={float(point_frictions.mean()):.6f} "
        f"max={float(point_frictions.max()):.6f}"
    )
    print(f"derived_body_inertia_diag={np.diag(body_inertia).tolist()}")
    print(f"USD written to {args.scene_usd_path.resolve()}")
    print(f"final_box_position={box_pose[:3].tolist()} final_box_velocity={box_vel[:3].tolist()}")


if __name__ == "__main__":
    main()
