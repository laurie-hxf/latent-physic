from __future__ import annotations

from pathlib import Path

import newton
import numpy as np
import torch
import torch.nn.functional as F

from pbd_io import read_ascii_ply_header
from pbd_math import (
    apply_quaternion_delta,
    compute_shape_density,
    diagonal_inertia_to_world,
    diagonal_inv_inertia_to_world,
    make_transform,
    normalize_quaternion,
    quaternion_delta_to_angular_velocity,
    quaternion_derivative,
    transform_points,
    yaw_only_quaternion,
)
from pbd_sampling import flatten_tabletop_points, solidify_surface_points, voxelize_selected_segments
from pbd_types import (
    CLUSTER_COLORS,
    DEFAULT_CONTACT_DAMPING,
    DEFAULT_CONTACT_MARGIN,
    DEFAULT_CONTACT_STIFFNESS,
    DEFAULT_FRICTION_REGULARIZATION,
    IDENTITY_QUAT,
    BuiltScene,
    RigidBodyCluster,
    SceneState,
    SegmentConfig,
)


def _to_tensor(
    value: float | np.ndarray | torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _to_numpy_f32(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)


def _to_float(value: float | np.ndarray | torch.Tensor) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(np.asarray(value, dtype=np.float32))


def _sync_newton_materials(scene: BuiltScene) -> None:
    if scene.collision_model is None:
        return

    shape_count = scene.collision_model.shape_count
    if shape_count == 0:
        return

    mu = np.full(shape_count, _to_float(scene.object_friction), dtype=np.float32)
    ke = np.full(shape_count, _to_float(scene.contact_stiffness), dtype=np.float32)
    kd = np.full(shape_count, _to_float(scene.contact_damping), dtype=np.float32)
    # Keep the mirror model numerically aligned with the torch-side contact parameters.
    kf = np.full(shape_count, _to_float(scene.contact_stiffness), dtype=np.float32)
    margin = np.full(shape_count, _to_float(scene.contact_margin), dtype=np.float32)

    for cluster in scene.clusters:
        if cluster.collision_shape_count == 0:
            continue
        if cluster.collision_geometry == "box":
            friction_value = _to_float(scene.table_friction)
        else:
            friction_value = _to_float(scene.object_friction)
        start = cluster.collision_shape_start
        stop = start + cluster.collision_shape_count
        mu[start:stop] = friction_value

    scene.collision_model.shape_material_mu.assign(mu)
    scene.collision_model.shape_material_ke.assign(ke)
    scene.collision_model.shape_material_kd.assign(kd)
    scene.collision_model.shape_material_kf.assign(kf)
    scene.collision_model.shape_margin.assign(margin)


def _sync_newton_state(scene: BuiltScene, state: SceneState) -> None:
    if scene.collision_state is None:
        return

    scene.collision_state.body_q.assign(_to_numpy_f32(state.body_q))
    scene.collision_state.body_qd.assign(_to_numpy_f32(state.body_qd))


def _sync_newton_mirror(scene: BuiltScene, state: SceneState) -> None:
    _sync_newton_materials(scene)
    _sync_newton_state(scene, state)


def _safe_norm(values: torch.Tensor, dim: int = -1, eps: float = 1e-9) -> torch.Tensor:
    return torch.sqrt(torch.sum(values * values, dim=dim) + eps)


def _compute_inertia_factor_diag(
    local_shape_positions: torch.Tensor,
    shape_radius: torch.Tensor,
) -> torch.Tensor:
    if local_shape_positions.shape[0] == 0:
        return torch.ones(3, device=shape_radius.device, dtype=shape_radius.dtype)

    x, y, z = local_shape_positions.unbind(dim=-1)
    point_mass_factor = torch.stack(
        [
            torch.mean(y * y + z * z),
            torch.mean(x * x + z * z),
            torch.mean(x * x + y * y),
        ]
    )
    sphere_mass_factor = torch.full_like(point_mass_factor, 0.4) * shape_radius.square()
    return (point_mass_factor + sphere_mass_factor).clamp_min(1e-6)


def _compute_support_radius(
    local_shape_positions: torch.Tensor,
    shape_radius: torch.Tensor,
) -> torch.Tensor:
    if local_shape_positions.shape[0] == 0:
        return shape_radius.clone()
    planar_radius = torch.linalg.vector_norm(local_shape_positions[:, :2], dim=-1)
    return planar_radius.max().clamp_min(0.0) + shape_radius


def _project_configuration_constraints(scene: BuiltScene, state: SceneState) -> SceneState:
    zero3 = torch.zeros(3, device=scene.device, dtype=scene.dtype)
    next_body_q: list[torch.Tensor] = []
    next_body_qd: list[torch.Tensor] = []

    for cluster in scene.clusters:
        body_id = cluster.body_id
        pose = state.body_q[body_id]
        velocity = state.body_qd[body_id]

        if cluster.control_mode == "prescribed":
            translation = scene.cluster_target_translations[cluster.name]
            quaternion = cluster.fixed_orientation
            linear_velocity = scene.cluster_command_velocities[cluster.name]
            angular_velocity = zero3
        elif not cluster.is_dynamic:
            translation = cluster.rest_translation
            quaternion = cluster.fixed_orientation
            linear_velocity = zero3
            angular_velocity = zero3
        else:
            translation = pose[:3]
            quaternion = normalize_quaternion(pose[3:])
            linear_velocity = velocity[:3]
            angular_velocity = velocity[3:]
            if cluster.planar_motion:
                translation = torch.stack(
                    [translation[0], translation[1], cluster.rest_translation[2]]
                )
                quaternion = yaw_only_quaternion(quaternion)
                linear_velocity = torch.stack(
                    [linear_velocity[0], linear_velocity[1], torch.zeros_like(linear_velocity[2])]
                )
                angular_velocity = torch.stack(
                    [
                        torch.zeros_like(angular_velocity[0]),
                        torch.zeros_like(angular_velocity[1]),
                        angular_velocity[2],
                    ]
                )

        next_body_q.append(torch.cat([translation, quaternion], dim=0))
        next_body_qd.append(torch.cat([linear_velocity, angular_velocity], dim=0))

    return SceneState(
        body_q=torch.stack(next_body_q, dim=0),
        body_qd=torch.stack(next_body_qd, dim=0),
    )


def _body_point_velocity(
    pose: torch.Tensor,
    velocity: torch.Tensor,
    lever_arm: torch.Tensor,
) -> torch.Tensor:
    linear = velocity[:3].reshape((1,) * (lever_arm.ndim - 1) + (3,))
    angular = velocity[3:].reshape((1,) * (lever_arm.ndim - 1) + (3,))
    return linear + torch.cross(angular.expand_as(lever_arm), lever_arm, dim=-1)


def _planar_support_wrench(
    scene: BuiltScene,
    state: SceneState,
    cluster: RigidBodyCluster,
    external_force: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    zero3 = torch.zeros(3, device=scene.device, dtype=scene.dtype)
    if not (cluster.is_dynamic and cluster.planar_motion):
        return zero3, zero3

    support_normal = torch.tensor([0.0, 0.0, 1.0], device=scene.device, dtype=scene.dtype)
    support_load = torch.clamp_min(-torch.dot(external_force, support_normal), 0.0)
    if float(support_load.detach().cpu()) <= 0.0:
        return zero3, zero3

    velocity = state.body_qd[cluster.body_id]
    friction_coeff = scene.table_friction.clamp_min(0.0)
    regularization = scene.friction_regularization.clamp_min(1e-6)

    support_force = support_load * support_normal
    tangential_velocity = velocity[:3] - torch.dot(velocity[:3], support_normal) * support_normal
    tangential_speed = _safe_norm(
        tangential_velocity,
        dim=0,
        eps=float(regularization.detach().cpu()) ** 2,
    )
    friction_force = -friction_coeff * support_load * tangential_velocity / tangential_speed

    support_radius = cluster.support_radius.clamp_min(0.0)
    edge_speed = velocity[5] * support_radius
    edge_speed_norm = torch.sqrt(edge_speed.square() + regularization.square())
    torsional_limit = friction_coeff * support_load * support_radius
    support_torque = torch.stack(
        [
            torch.zeros((), device=scene.device, dtype=scene.dtype),
            torch.zeros((), device=scene.device, dtype=scene.dtype),
            -torsional_limit * edge_speed / edge_speed_norm,
        ]
    )
    return support_force + friction_force, support_torque


def _contact_gate(scene: BuiltScene, signed_distance: torch.Tensor) -> torch.Tensor:
    margin = scene.contact_margin.clamp_min(1e-6)
    return torch.sigmoid(-signed_distance / margin)


def _contact_regularization(scene: BuiltScene, dt: torch.Tensor) -> torch.Tensor:
    stiffness = scene.contact_stiffness.clamp_min(1e-6)
    compliance = torch.reciprocal(stiffness)
    return compliance / dt.square().clamp_min(1e-8)


def _constraint_damping_length(scene: BuiltScene, dt: torch.Tensor) -> torch.Tensor:
    stiffness = scene.contact_stiffness.clamp_min(1e-6)
    damping = scene.contact_damping.clamp_min(0.0)
    return dt * damping / stiffness


def _directional_effective_mass(
    inv_mass: torch.Tensor,
    inv_inertia_world: torch.Tensor,
    lever_arm: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    angular_jacobian = torch.cross(lever_arm, direction, dim=-1)
    angular_response = torch.matmul(angular_jacobian, inv_inertia_world.transpose(-1, -2))
    rotational_term = torch.sum(angular_response * angular_jacobian, dim=-1)
    return inv_mass + rotational_term


def _contact_correction(
    direction: torch.Tensor,
    magnitude: torch.Tensor,
    inv_mass_a: torch.Tensor,
    inv_inertia_a_world: torch.Tensor,
    lever_arm_a: torch.Tensor,
    inv_mass_b: torch.Tensor,
    inv_inertia_b_world: torch.Tensor,
    lever_arm_b: torch.Tensor,
    regularization: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    denom = (
        _directional_effective_mass(inv_mass_a, inv_inertia_a_world, lever_arm_a, direction)
        + _directional_effective_mass(inv_mass_b, inv_inertia_b_world, lever_arm_b, -direction)
        + regularization
    ).clamp_min(1e-8)

    lambda_value = magnitude / denom
    impulse = lambda_value.unsqueeze(-1) * direction

    delta_translation_a = inv_mass_a * impulse
    delta_translation_b = -inv_mass_b * impulse
    torque_a = torch.cross(lever_arm_a, impulse, dim=-1)
    torque_b = torch.cross(lever_arm_b, -impulse, dim=-1)
    delta_rotation_a = torch.matmul(torque_a, inv_inertia_a_world.transpose(-1, -2))
    delta_rotation_b = torch.matmul(torque_b, inv_inertia_b_world.transpose(-1, -2))
    return delta_translation_a, delta_rotation_a, delta_translation_b, delta_rotation_b


def _sphere_sphere_contact_wrenches(
    scene: BuiltScene,
    state: SceneState,
    cluster_a: RigidBodyCluster,
    cluster_b: RigidBodyCluster,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    zero3 = torch.zeros(3, device=scene.device, dtype=scene.dtype)
    if cluster_a.shape_count == 0 or cluster_b.shape_count == 0:
        return zero3, zero3, zero3, zero3

    pose_a = state.body_q[cluster_a.body_id]
    pose_b = state.body_q[cluster_b.body_id]
    velocity_a = state.body_qd[cluster_a.body_id]
    velocity_b = state.body_qd[cluster_b.body_id]

    points_a = transform_points(cluster_a.local_shape_positions, pose_a[:3], pose_a[3:])
    points_b = transform_points(cluster_b.local_shape_positions, pose_b[:3], pose_b[3:])

    center_delta = points_a[:, None, :] - points_b[None, :, :]
    distance = _safe_norm(center_delta, dim=-1)
    normal = center_delta / distance.unsqueeze(-1)
    signed_distance = distance - (cluster_a.shape_radius + cluster_b.shape_radius)
    penetration = torch.clamp_min(-signed_distance, 0.0)
    gate = _contact_gate(scene, signed_distance)

    contact_point_a = points_a[:, None, :] - normal * cluster_a.shape_radius
    contact_point_b = points_b[None, :, :] + normal * cluster_b.shape_radius
    lever_arm_a = contact_point_a - pose_a[:3].view(1, 1, 3)
    lever_arm_b = contact_point_b - pose_b[:3].view(1, 1, 3)

    contact_velocity_a = _body_point_velocity(pose_a, velocity_a, lever_arm_a)
    contact_velocity_b = _body_point_velocity(pose_b, velocity_b, lever_arm_b)
    relative_velocity = contact_velocity_a - contact_velocity_b
    normal_speed = torch.sum(relative_velocity * normal, dim=-1)
    approach_speed = torch.clamp_min(-normal_speed, 0.0)

    normal_force_magnitude = gate * (
        scene.contact_stiffness.clamp_min(0.0) * penetration
        + scene.contact_damping.clamp_min(0.0) * approach_speed
    )

    tangential_velocity = relative_velocity - normal_speed.unsqueeze(-1) * normal
    tangential_speed = _safe_norm(
        tangential_velocity,
        dim=-1,
        eps=float(scene.friction_regularization.detach().cpu()) ** 2,
    )
    friction_force = -scene.object_friction.clamp_min(0.0) * normal_force_magnitude.unsqueeze(
        -1
    ) * tangential_velocity / tangential_speed.unsqueeze(-1)
    contact_force = normal_force_magnitude.unsqueeze(-1) * normal + friction_force

    force_a = torch.sum(contact_force, dim=(0, 1))
    torque_a = torch.sum(torch.cross(lever_arm_a, contact_force, dim=-1), dim=(0, 1))
    force_b = -torch.sum(contact_force, dim=(0, 1))
    torque_b = torch.sum(torch.cross(lever_arm_b, -contact_force, dim=-1), dim=(0, 1))
    return force_a, torque_a, force_b, torque_b


def _sphere_box_contact_wrenches(
    scene: BuiltScene,
    state: SceneState,
    sphere_cluster: RigidBodyCluster,
    box_cluster: RigidBodyCluster,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    zero3 = torch.zeros(3, device=scene.device, dtype=scene.dtype)
    if sphere_cluster.shape_count == 0 or box_cluster.box_half_extents is None:
        return zero3, zero3, zero3, zero3

    sphere_pose = state.body_q[sphere_cluster.body_id]
    sphere_velocity = state.body_qd[sphere_cluster.body_id]
    box_pose = state.body_q[box_cluster.body_id]
    box_velocity = state.body_qd[box_cluster.body_id]

    world_points = transform_points(
        sphere_cluster.local_shape_positions,
        sphere_pose[:3],
        sphere_pose[3:],
    )
    closest_points, normal, signed_distance = _box_contact_geometry(
        world_points,
        box_pose[:3],
        box_cluster.box_half_extents,
        scene.dtype,
    )

    surface_distance = signed_distance - sphere_cluster.shape_radius
    penetration = torch.clamp_min(-surface_distance, 0.0)
    gate = _contact_gate(scene, surface_distance)

    contact_point_sphere = world_points - normal * sphere_cluster.shape_radius
    contact_point_box = closest_points
    lever_arm_sphere = contact_point_sphere - sphere_pose[:3].unsqueeze(0)
    lever_arm_box = contact_point_box - box_pose[:3].unsqueeze(0)

    contact_velocity_sphere = _body_point_velocity(sphere_pose, sphere_velocity, lever_arm_sphere)
    contact_velocity_box = _body_point_velocity(box_pose, box_velocity, lever_arm_box)
    relative_velocity = contact_velocity_sphere - contact_velocity_box
    normal_speed = torch.sum(relative_velocity * normal, dim=-1)
    approach_speed = torch.clamp_min(-normal_speed, 0.0)

    normal_force_magnitude = gate * (
        scene.contact_stiffness.clamp_min(0.0) * penetration
        + scene.contact_damping.clamp_min(0.0) * approach_speed
    )

    tangential_velocity = relative_velocity - normal_speed.unsqueeze(-1) * normal
    tangential_speed = _safe_norm(
        tangential_velocity,
        dim=-1,
        eps=float(scene.friction_regularization.detach().cpu()) ** 2,
    )
    friction_force = -scene.table_friction.clamp_min(0.0) * normal_force_magnitude.unsqueeze(
        -1
    ) * tangential_velocity / tangential_speed.unsqueeze(-1)
    contact_force = normal_force_magnitude.unsqueeze(-1) * normal + friction_force

    force_sphere = torch.sum(contact_force, dim=0)
    torque_sphere = torch.sum(torch.cross(lever_arm_sphere, contact_force, dim=-1), dim=0)
    force_box = -torch.sum(contact_force, dim=0)
    torque_box = torch.sum(torch.cross(lever_arm_box, -contact_force, dim=-1), dim=0)
    return force_sphere, torque_sphere, force_box, torque_box


def _compute_contact_wrenches(
    scene: BuiltScene,
    state: SceneState,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    force_sum = [torch.zeros(3, device=scene.device, dtype=scene.dtype) for _ in scene.clusters]
    torque_sum = [torch.zeros(3, device=scene.device, dtype=scene.dtype) for _ in scene.clusters]

    for cluster in scene.clusters:
        if cluster.is_dynamic:
            force_sum[cluster.body_id] = force_sum[cluster.body_id] + cluster.effective_mass * scene.gravity

    for idx_a, cluster_a in enumerate(scene.clusters):
        for idx_b in range(idx_a + 1, len(scene.clusters)):
            cluster_b = scene.clusters[idx_b]
            if not (cluster_a.is_dynamic or cluster_b.is_dynamic):
                continue

            if (
                cluster_a.collision_geometry == "sphere_cluster"
                and cluster_b.collision_geometry == "sphere_cluster"
            ):
                force_a, torque_a, force_b, torque_b = _sphere_sphere_contact_wrenches(
                    scene=scene,
                    state=state,
                    cluster_a=cluster_a,
                    cluster_b=cluster_b,
                )
            elif (
                cluster_a.collision_geometry == "sphere_cluster"
                and cluster_b.collision_geometry == "box"
            ):
                force_a, torque_a, force_b, torque_b = _sphere_box_contact_wrenches(
                    scene=scene,
                    state=state,
                    sphere_cluster=cluster_a,
                    box_cluster=cluster_b,
                )
            elif (
                cluster_b.collision_geometry == "sphere_cluster"
                and cluster_a.collision_geometry == "box"
            ):
                force_b, torque_b, force_a, torque_a = _sphere_box_contact_wrenches(
                    scene=scene,
                    state=state,
                    sphere_cluster=cluster_b,
                    box_cluster=cluster_a,
                )
            else:
                continue

            if cluster_a.is_dynamic:
                force_sum[cluster_a.body_id] = force_sum[cluster_a.body_id] + force_a
                torque_sum[cluster_a.body_id] = torque_sum[cluster_a.body_id] + torque_a
            if cluster_b.is_dynamic:
                force_sum[cluster_b.body_id] = force_sum[cluster_b.body_id] + force_b
                torque_sum[cluster_b.body_id] = torque_sum[cluster_b.body_id] + torque_b

    for cluster in scene.clusters:
        if not (cluster.is_dynamic and cluster.planar_motion):
            continue
        body_id = cluster.body_id
        support_force, support_torque = _planar_support_wrench(
            scene=scene,
            state=state,
            cluster=cluster,
            external_force=force_sum[body_id],
        )
        force_sum[body_id] = force_sum[body_id] + support_force
        torque_sum[body_id] = torque_sum[body_id] + support_torque

    return force_sum, torque_sum


def _integrate_semi_implicit_euler(
    scene: BuiltScene,
    state: SceneState,
    dt: torch.Tensor,
    force_sum: list[torch.Tensor],
    torque_sum: list[torch.Tensor],
) -> SceneState:
    zero3 = torch.zeros(3, device=scene.device, dtype=scene.dtype)

    next_body_q: list[torch.Tensor] = []
    next_body_qd: list[torch.Tensor] = []
    for cluster in scene.clusters:
        body_id = cluster.body_id
        pose = state.body_q[body_id]
        velocity = state.body_qd[body_id]

        if cluster.control_mode == "prescribed":
            translation = scene.cluster_target_translations[cluster.name]
            quaternion = cluster.fixed_orientation
            linear_velocity = scene.cluster_command_velocities[cluster.name]
            angular_velocity = zero3
        elif not cluster.is_dynamic:
            translation = cluster.rest_translation
            quaternion = cluster.fixed_orientation
            linear_velocity = zero3
            angular_velocity = zero3
        else:
            world_inertia = diagonal_inertia_to_world(cluster.inertia_diag, pose[3:])
            world_inv_inertia = diagonal_inv_inertia_to_world(cluster.inv_inertia_diag, pose[3:])
            inertia_times_omega = torch.matmul(world_inertia, velocity[3:].unsqueeze(-1)).squeeze(-1)
            gyroscopic = torch.cross(velocity[3:], inertia_times_omega, dim=-1)

            linear_acceleration = force_sum[body_id] * cluster.inv_mass
            angular_acceleration = torch.matmul(
                world_inv_inertia,
                (torque_sum[body_id] - gyroscopic).unsqueeze(-1),
            ).squeeze(-1)

            linear_velocity = velocity[:3] + dt * linear_acceleration
            angular_velocity = velocity[3:] + dt * angular_acceleration
            translation = pose[:3] + dt * linear_velocity
            quaternion = normalize_quaternion(
                pose[3:] + dt * quaternion_derivative(pose[3:], angular_velocity)
            )

        next_body_q.append(torch.cat([translation, quaternion], dim=0))
        next_body_qd.append(torch.cat([linear_velocity, angular_velocity], dim=0))

    return _project_configuration_constraints(
        scene,
        SceneState(
            body_q=torch.stack(next_body_q, dim=0),
            body_qd=torch.stack(next_body_qd, dim=0),
        ),
    )


def _box_contact_geometry(
    world_points: torch.Tensor,
    box_center: torch.Tensor,
    half_extents: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    local_points = world_points - box_center.unsqueeze(0)
    clamped_points = torch.maximum(torch.minimum(local_points, half_extents), -half_extents)
    closest_points = box_center.unsqueeze(0) + clamped_points
    delta = world_points - closest_points
    outside_distance = _safe_norm(delta, dim=-1)

    outside_mask = torch.any(torch.abs(local_points) > half_extents, dim=-1)
    margin_to_face = half_extents.unsqueeze(0) - torch.abs(local_points)
    face_axis = torch.argmin(margin_to_face, dim=-1)
    face_sign = torch.where(
        local_points.gather(1, face_axis.unsqueeze(-1)).squeeze(-1) >= 0.0,
        torch.ones_like(outside_distance),
        -torch.ones_like(outside_distance),
    )
    inside_normal = F.one_hot(face_axis, num_classes=3).to(dtype=dtype) * face_sign.unsqueeze(-1)
    outside_normal = delta / outside_distance.unsqueeze(-1)
    normal = torch.where(outside_mask.unsqueeze(-1), outside_normal, inside_normal)

    inside_distance = -margin_to_face.gather(1, face_axis.unsqueeze(-1)).squeeze(-1)
    signed_distance = torch.where(outside_mask, outside_distance, inside_distance)
    return closest_points, normal, signed_distance


def _sphere_sphere_constraint_deltas(
    scene: BuiltScene,
    state: SceneState,
    previous_state: SceneState,
    cluster_a: RigidBodyCluster,
    cluster_b: RigidBodyCluster,
    dt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    zero3 = torch.zeros(3, device=scene.device, dtype=scene.dtype)
    if cluster_a.shape_count == 0 or cluster_b.shape_count == 0:
        zero_weight = torch.zeros((), device=scene.device, dtype=scene.dtype)
        return zero3, zero3, zero3, zero3, zero_weight

    pose_a = state.body_q[cluster_a.body_id]
    pose_b = state.body_q[cluster_b.body_id]
    velocity_a = state.body_qd[cluster_a.body_id]
    velocity_b = state.body_qd[cluster_b.body_id]
    prev_pose_a = previous_state.body_q[cluster_a.body_id]
    prev_pose_b = previous_state.body_q[cluster_b.body_id]

    points_a = transform_points(cluster_a.local_shape_positions, pose_a[:3], pose_a[3:])
    points_b = transform_points(cluster_b.local_shape_positions, pose_b[:3], pose_b[3:])
    prev_points_a = transform_points(cluster_a.local_shape_positions, prev_pose_a[:3], prev_pose_a[3:])
    prev_points_b = transform_points(cluster_b.local_shape_positions, prev_pose_b[:3], prev_pose_b[3:])

    center_delta = points_a[:, None, :] - points_b[None, :, :]
    distance = _safe_norm(center_delta, dim=-1)
    normal = center_delta / distance.unsqueeze(-1)
    signed_distance = distance - (cluster_a.shape_radius + cluster_b.shape_radius)
    penetration = torch.clamp_min(-signed_distance, 0.0)
    gate = _contact_gate(scene, signed_distance)

    contact_point_a = points_a[:, None, :] - normal * cluster_a.shape_radius
    contact_point_b = points_b[None, :, :] + normal * cluster_b.shape_radius
    prev_contact_point_a = prev_points_a[:, None, :] - normal * cluster_a.shape_radius
    prev_contact_point_b = prev_points_b[None, :, :] + normal * cluster_b.shape_radius
    lever_arm_a = contact_point_a - pose_a[:3].view(1, 1, 3)
    lever_arm_b = contact_point_b - pose_b[:3].view(1, 1, 3)

    contact_velocity_a = _body_point_velocity(pose_a, velocity_a, lever_arm_a)
    contact_velocity_b = _body_point_velocity(pose_b, velocity_b, lever_arm_b)
    relative_velocity = contact_velocity_a - contact_velocity_b
    normal_speed = torch.sum(relative_velocity * normal, dim=-1)
    damping_bias = _constraint_damping_length(scene, dt) * torch.clamp_min(-normal_speed, 0.0)
    normal_magnitude = gate * (penetration + damping_bias)

    inv_inertia_a_world = diagonal_inv_inertia_to_world(cluster_a.inv_inertia_diag, pose_a[3:])
    inv_inertia_b_world = diagonal_inv_inertia_to_world(cluster_b.inv_inertia_diag, pose_b[3:])
    regularization = _contact_regularization(scene, dt)
    dxa_n, dra_n, dxb_n, drb_n = _contact_correction(
        direction=normal,
        magnitude=normal_magnitude,
        inv_mass_a=cluster_a.inv_mass,
        inv_inertia_a_world=inv_inertia_a_world,
        lever_arm_a=lever_arm_a,
        inv_mass_b=cluster_b.inv_mass,
        inv_inertia_b_world=inv_inertia_b_world,
        lever_arm_b=lever_arm_b,
        regularization=regularization,
    )

    relative_displacement = (contact_point_a - prev_contact_point_a) - (contact_point_b - prev_contact_point_b)
    tangential_displacement = relative_displacement - torch.sum(
        relative_displacement * normal,
        dim=-1,
        keepdim=True,
    ) * normal
    tangential_norm = _safe_norm(
        tangential_displacement,
        dim=-1,
        eps=float(scene.friction_regularization.detach().cpu()) ** 2,
    )
    friction_direction = -tangential_displacement / tangential_norm.unsqueeze(-1)
    friction_limit = scene.object_friction.clamp_min(0.0) * normal_magnitude
    friction_magnitude = torch.minimum(tangential_norm, friction_limit)
    dxa_t, dra_t, dxb_t, drb_t = _contact_correction(
        direction=friction_direction,
        magnitude=friction_magnitude,
        inv_mass_a=cluster_a.inv_mass,
        inv_inertia_a_world=inv_inertia_a_world,
        lever_arm_a=lever_arm_a,
        inv_mass_b=cluster_b.inv_mass,
        inv_inertia_b_world=inv_inertia_b_world,
        lever_arm_b=lever_arm_b,
        regularization=regularization,
    )

    active_weight = gate.sum()
    delta_translation_a = torch.sum(dxa_n + dxa_t, dim=(0, 1))
    delta_rotation_a = torch.sum(dra_n + dra_t, dim=(0, 1))
    delta_translation_b = torch.sum(dxb_n + dxb_t, dim=(0, 1))
    delta_rotation_b = torch.sum(drb_n + drb_t, dim=(0, 1))
    return (
        delta_translation_a,
        delta_rotation_a,
        delta_translation_b,
        delta_rotation_b,
        active_weight,
    )


def _sphere_box_constraint_deltas(
    scene: BuiltScene,
    state: SceneState,
    previous_state: SceneState,
    sphere_cluster: RigidBodyCluster,
    box_cluster: RigidBodyCluster,
    dt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    zero3 = torch.zeros(3, device=scene.device, dtype=scene.dtype)
    if sphere_cluster.shape_count == 0 or box_cluster.box_half_extents is None:
        zero_weight = torch.zeros((), device=scene.device, dtype=scene.dtype)
        return zero3, zero3, zero_weight

    sphere_pose = state.body_q[sphere_cluster.body_id]
    sphere_velocity = state.body_qd[sphere_cluster.body_id]
    prev_sphere_pose = previous_state.body_q[sphere_cluster.body_id]
    box_pose = state.body_q[box_cluster.body_id]
    box_velocity = state.body_qd[box_cluster.body_id]

    world_points = transform_points(
        sphere_cluster.local_shape_positions,
        sphere_pose[:3],
        sphere_pose[3:],
    )
    prev_world_points = transform_points(
        sphere_cluster.local_shape_positions,
        prev_sphere_pose[:3],
        prev_sphere_pose[3:],
    )
    closest_points, normal, signed_distance = _box_contact_geometry(
        world_points,
        box_pose[:3],
        box_cluster.box_half_extents,
        scene.dtype,
    )

    surface_distance = signed_distance - sphere_cluster.shape_radius
    penetration = torch.clamp_min(-surface_distance, 0.0)
    gate = _contact_gate(scene, surface_distance)

    contact_point_sphere = world_points - normal * sphere_cluster.shape_radius
    prev_contact_point_sphere = prev_world_points - normal * sphere_cluster.shape_radius
    contact_point_box = closest_points
    lever_arm_sphere = contact_point_sphere - sphere_pose[:3].unsqueeze(0)
    lever_arm_box = contact_point_box - box_pose[:3].unsqueeze(0)

    contact_velocity_sphere = _body_point_velocity(sphere_pose, sphere_velocity, lever_arm_sphere)
    contact_velocity_box = _body_point_velocity(box_pose, box_velocity, lever_arm_box)
    relative_velocity = contact_velocity_sphere - contact_velocity_box
    normal_speed = torch.sum(relative_velocity * normal, dim=-1)
    damping_bias = _constraint_damping_length(scene, dt) * torch.clamp_min(-normal_speed, 0.0)
    normal_magnitude = gate * (penetration + damping_bias)

    inv_inertia_world = diagonal_inv_inertia_to_world(sphere_cluster.inv_inertia_diag, sphere_pose[3:])
    regularization = _contact_regularization(scene, dt)
    dxt_n, drt_n, _, _ = _contact_correction(
        direction=normal,
        magnitude=normal_magnitude,
        inv_mass_a=sphere_cluster.inv_mass,
        inv_inertia_a_world=inv_inertia_world,
        lever_arm_a=lever_arm_sphere,
        inv_mass_b=torch.zeros_like(sphere_cluster.inv_mass),
        inv_inertia_b_world=torch.zeros_like(inv_inertia_world),
        lever_arm_b=lever_arm_box,
        regularization=regularization,
    )

    relative_displacement = contact_point_sphere - prev_contact_point_sphere
    tangential_displacement = relative_displacement - torch.sum(
        relative_displacement * normal,
        dim=-1,
        keepdim=True,
    ) * normal
    tangential_norm = _safe_norm(
        tangential_displacement,
        dim=-1,
        eps=float(scene.friction_regularization.detach().cpu()) ** 2,
    )
    friction_direction = -tangential_displacement / tangential_norm.unsqueeze(-1)
    friction_limit = scene.table_friction.clamp_min(0.0) * normal_magnitude
    friction_magnitude = torch.minimum(tangential_norm, friction_limit)
    dxt_t, drt_t, _, _ = _contact_correction(
        direction=friction_direction,
        magnitude=friction_magnitude,
        inv_mass_a=sphere_cluster.inv_mass,
        inv_inertia_a_world=inv_inertia_world,
        lever_arm_a=lever_arm_sphere,
        inv_mass_b=torch.zeros_like(sphere_cluster.inv_mass),
        inv_inertia_b_world=torch.zeros_like(inv_inertia_world),
        lever_arm_b=lever_arm_box,
        regularization=regularization,
    )

    active_weight = gate.sum()
    delta_translation = torch.sum(dxt_n + dxt_t, dim=0)
    delta_rotation = torch.sum(drt_n + drt_t, dim=0)
    return delta_translation, delta_rotation, active_weight


def _solve_contact_constraints(
    scene: BuiltScene,
    previous_state: SceneState,
    predicted_state: SceneState,
    dt: torch.Tensor,
) -> SceneState:
    current_state = predicted_state

    for _ in range(scene.constraint_iterations):
        translation_sum = [
            torch.zeros(3, device=scene.device, dtype=scene.dtype) for _ in scene.clusters
        ]
        rotation_sum = [
            torch.zeros(3, device=scene.device, dtype=scene.dtype) for _ in scene.clusters
        ]
        weight_sum = [
            torch.zeros((), device=scene.device, dtype=scene.dtype) for _ in scene.clusters
        ]

        for idx_a, cluster_a in enumerate(scene.clusters):
            for idx_b in range(idx_a + 1, len(scene.clusters)):
                cluster_b = scene.clusters[idx_b]
                if not (cluster_a.is_dynamic or cluster_b.is_dynamic):
                    continue

                if (
                    cluster_a.collision_geometry == "sphere_cluster"
                    and cluster_b.collision_geometry == "sphere_cluster"
                ):
                    dxa, dra, dxb, drb, active_weight = _sphere_sphere_constraint_deltas(
                        scene=scene,
                        state=current_state,
                        previous_state=previous_state,
                        cluster_a=cluster_a,
                        cluster_b=cluster_b,
                        dt=dt,
                    )
                    if cluster_a.is_dynamic:
                        translation_sum[cluster_a.body_id] = translation_sum[cluster_a.body_id] + dxa
                        rotation_sum[cluster_a.body_id] = rotation_sum[cluster_a.body_id] + dra
                        weight_sum[cluster_a.body_id] = weight_sum[cluster_a.body_id] + active_weight
                    if cluster_b.is_dynamic:
                        translation_sum[cluster_b.body_id] = translation_sum[cluster_b.body_id] + dxb
                        rotation_sum[cluster_b.body_id] = rotation_sum[cluster_b.body_id] + drb
                        weight_sum[cluster_b.body_id] = weight_sum[cluster_b.body_id] + active_weight
                    continue

                if (
                    cluster_a.collision_geometry == "sphere_cluster"
                    and cluster_b.collision_geometry == "box"
                    and cluster_a.is_dynamic
                ):
                    dxa, dra, active_weight = _sphere_box_constraint_deltas(
                        scene=scene,
                        state=current_state,
                        previous_state=previous_state,
                        sphere_cluster=cluster_a,
                        box_cluster=cluster_b,
                        dt=dt,
                    )
                    translation_sum[cluster_a.body_id] = translation_sum[cluster_a.body_id] + dxa
                    rotation_sum[cluster_a.body_id] = rotation_sum[cluster_a.body_id] + dra
                    weight_sum[cluster_a.body_id] = weight_sum[cluster_a.body_id] + active_weight
                    continue

                if (
                    cluster_b.collision_geometry == "sphere_cluster"
                    and cluster_a.collision_geometry == "box"
                    and cluster_b.is_dynamic
                ):
                    dxb, drb, active_weight = _sphere_box_constraint_deltas(
                        scene=scene,
                        state=current_state,
                        previous_state=previous_state,
                        sphere_cluster=cluster_b,
                        box_cluster=cluster_a,
                        dt=dt,
                    )
                    translation_sum[cluster_b.body_id] = translation_sum[cluster_b.body_id] + dxb
                    rotation_sum[cluster_b.body_id] = rotation_sum[cluster_b.body_id] + drb
                    weight_sum[cluster_b.body_id] = weight_sum[cluster_b.body_id] + active_weight

        next_body_q_rows: list[torch.Tensor] = []
        for cluster in scene.clusters:
            body_id = cluster.body_id
            pose = current_state.body_q[body_id]
            if not cluster.is_dynamic:
                next_body_q_rows.append(pose)
                continue

            normalizer = weight_sum[body_id].clamp_min(1.0)
            translation_delta = translation_sum[body_id] / normalizer
            rotation_delta = rotation_sum[body_id] / normalizer
            next_translation = pose[:3] + translation_delta
            next_quaternion = apply_quaternion_delta(pose[3:], rotation_delta)
            next_body_q_rows.append(torch.cat([next_translation, next_quaternion], dim=0))

        current_state = _project_configuration_constraints(
            scene,
            SceneState(
                body_q=torch.stack(next_body_q_rows, dim=0),
                body_qd=current_state.body_qd,
            ),
        )

    return current_state


def _resolve_initial_interpenetrations(
    scene: BuiltScene,
    dt: float = 1.0 / 240.0,
    max_passes: int = 4,
) -> None:
    dt_tensor = _to_tensor(dt, device=scene.device, dtype=scene.dtype)
    original_cluster_flags = [
        (cluster.is_dynamic, cluster.control_mode, cluster.planar_motion)
        for cluster in scene.clusters
    ]

    try:
        for cluster in scene.clusters:
            if cluster.control_mode == "fixed":
                continue
            cluster.is_dynamic = True
            cluster.control_mode = "free"
            cluster.planar_motion = False

        current_state = scene.state_0.clone()
        for _ in range(max(int(max_passes), 1)):
            corrected_state = _solve_contact_constraints(
                scene=scene,
                previous_state=current_state,
                predicted_state=current_state,
                dt=dt_tensor,
            )
            max_translation_delta = torch.max(
                torch.abs(corrected_state.body_q[:, :3] - current_state.body_q[:, :3])
            )
            current_state = SceneState(
                body_q=corrected_state.body_q.clone(),
                body_qd=torch.zeros_like(corrected_state.body_qd),
            )
            if float(max_translation_delta.detach().cpu()) < 1e-6:
                break

        body_q = scene.state_0.body_q.clone()
        body_qd = torch.zeros_like(scene.state_0.body_qd)
        for cluster, (_, control_mode, _) in zip(scene.clusters, original_cluster_flags):
            body_id = cluster.body_id
            if control_mode != "fixed":
                cluster.rest_translation = current_state.body_q[body_id, :3].clone()
                body_q[body_id, :3] = cluster.rest_translation
            else:
                body_q[body_id, :3] = cluster.rest_translation
            body_q[body_id, 3:] = cluster.fixed_orientation

        scene.state_0 = SceneState(body_q=body_q, body_qd=body_qd)
        scene.state_1 = scene.state_0.clone()

        for cluster, (_, control_mode, _) in zip(scene.clusters, original_cluster_flags):
            if control_mode == "prescribed":
                scene.cluster_target_translations[cluster.name] = cluster.rest_translation.clone()
                scene.cluster_command_velocities[cluster.name] = torch.zeros(
                    3,
                    device=scene.device,
                    dtype=scene.dtype,
                )
    finally:
        for cluster, (is_dynamic, control_mode, planar_motion) in zip(
            scene.clusters,
            original_cluster_flags,
        ):
            cluster.is_dynamic = is_dynamic
            cluster.control_mode = control_mode
            cluster.planar_motion = planar_motion


def _finalize_velocities(
    scene: BuiltScene,
    previous_state: SceneState,
    resolved_state: SceneState,
    dt: torch.Tensor,
    velocity_damping: float,
    max_velocity: float,
) -> SceneState:
    damping = _to_tensor(velocity_damping, device=scene.device, dtype=scene.dtype)
    max_speed = _to_tensor(max_velocity, device=scene.device, dtype=scene.dtype)
    zero3 = torch.zeros(3, device=scene.device, dtype=scene.dtype)

    next_body_qd: list[torch.Tensor] = []
    for cluster in scene.clusters:
        body_id = cluster.body_id
        new_pose = resolved_state.body_q[body_id]
        old_pose = previous_state.body_q[body_id]

        if cluster.control_mode == "prescribed":
            linear_velocity = scene.cluster_command_velocities[cluster.name]
            angular_velocity = zero3
        elif not cluster.is_dynamic:
            linear_velocity = zero3
            angular_velocity = zero3
        else:
            linear_velocity = (new_pose[:3] - old_pose[:3]) / dt.clamp_min(1e-8)
            angular_velocity = quaternion_delta_to_angular_velocity(old_pose[3:], new_pose[3:], dt)
            linear_velocity = linear_velocity * damping
            angular_velocity = angular_velocity * damping

            if cluster.planar_motion:
                linear_velocity = torch.stack(
                    [linear_velocity[0], linear_velocity[1], torch.zeros_like(linear_velocity[2])]
                )
                angular_velocity = torch.stack(
                    [
                        torch.zeros_like(angular_velocity[0]),
                        torch.zeros_like(angular_velocity[1]),
                        angular_velocity[2],
                    ]
                )

            speed = _safe_norm(linear_velocity, dim=0)
            clip = torch.minimum(
                torch.ones((), device=scene.device, dtype=scene.dtype),
                max_speed / speed.clamp_min(1e-8),
            )
            linear_velocity = linear_velocity * clip

        next_body_qd.append(torch.cat([linear_velocity, angular_velocity], dim=0))

    return _project_configuration_constraints(
        scene,
        SceneState(
            body_q=resolved_state.body_q,
            body_qd=torch.stack(next_body_qd, dim=0),
        ),
    )


def set_body_state(
    state: SceneState,
    body_id: int,
    translation: np.ndarray | torch.Tensor,
    quaternion: np.ndarray | torch.Tensor,
    linear_velocity: np.ndarray | torch.Tensor | None = None,
    angular_velocity: np.ndarray | torch.Tensor | None = None,
) -> None:
    device = state.body_q.device
    dtype = state.body_q.dtype

    next_body_q = state.body_q.clone()
    next_body_qd = state.body_qd.clone()
    next_body_q[body_id, :3] = _to_tensor(translation, device=device, dtype=dtype)
    next_body_q[body_id, 3:] = normalize_quaternion(_to_tensor(quaternion, device=device, dtype=dtype))
    if linear_velocity is not None:
        next_body_qd[body_id, :3] = _to_tensor(linear_velocity, device=device, dtype=dtype)
    if angular_velocity is not None:
        next_body_qd[body_id, 3:] = _to_tensor(angular_velocity, device=device, dtype=dtype)

    state.body_q = next_body_q
    state.body_qd = next_body_qd


def step_scene(
    scene: BuiltScene,
    dt: float,
    velocity_damping: float,
    max_velocity: float,
) -> None:
    _sync_newton_mirror(scene, scene.state_0)
    dt_tensor = _to_tensor(dt, device=scene.device, dtype=scene.dtype)
    force_sum, torque_sum = _compute_contact_wrenches(scene, scene.state_0)
    predicted_state = _integrate_semi_implicit_euler(
        scene=scene,
        state=scene.state_0,
        dt=dt_tensor,
        force_sum=force_sum,
        torque_sum=torque_sum,
    )
    corrected_state = _solve_contact_constraints(
        scene=scene,
        previous_state=scene.state_0,
        predicted_state=predicted_state,
        dt=dt_tensor,
    )
    finalized_state = _finalize_velocities(
        scene=scene,
        previous_state=scene.state_0,
        resolved_state=corrected_state,
        dt=dt_tensor,
        velocity_damping=velocity_damping,
        max_velocity=max_velocity,
    )
    scene.state_1 = finalized_state
    scene.state_0, scene.state_1 = scene.state_1, scene.state_0
    _sync_newton_mirror(scene, scene.state_0)


def write_scene_sample_ply(scene: BuiltScene, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    body_q = scene.state_0.body_q.detach().cpu()
    total_shapes = sum(cluster.shape_count for cluster in scene.clusters)

    with output_path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {total_shapes}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property int segmentation_id\n")
        f.write("property float radius\n")
        f.write("property float mass\n")
        f.write("property uchar is_dynamic\n")
        f.write("end_header\n")

        for cluster in scene.clusters:
            rgb = tuple(int(np.clip(round(channel * 255.0), 0, 255)) for channel in cluster.display_color)
            is_dynamic = 1 if cluster.is_dynamic else 0
            pose = body_q[cluster.body_id]
            world_positions = transform_points(
                cluster.local_shape_positions.detach().cpu(),
                translation=pose[:3],
                quaternion=pose[3:],
            ).detach().cpu().numpy()
            for x, y, z in world_positions:
                f.write(
                    f"{x:.8f} {y:.8f} {z:.8f} "
                    f"{rgb[0]} {rgb[1]} {rgb[2]} "
                    f"{cluster.segmentation_id} {float(cluster.shape_radius.detach().cpu()):.8f} "
                    f"{cluster.shape_mass:.8f} {is_dynamic}\n"
                )


def export_rollout_step(scene: BuiltScene, step_dir: Path, step_idx: int) -> Path:
    step_path = step_dir / f"scene_step_{step_idx:04d}.ply"
    write_scene_sample_ply(scene=scene, output_path=step_path)
    return step_path


def find_cluster(scene: BuiltScene, cluster_name: str) -> RigidBodyCluster:
    for cluster in scene.clusters:
        if cluster.name == cluster_name:
            return cluster
    raise RuntimeError(f"Cluster '{cluster_name}' was not found")


def advance_prescribed_cluster(
    scene: BuiltScene,
    cluster: RigidBodyCluster,
    delta_xyz: np.ndarray | torch.Tensor,
    dt: float,
) -> None:
    if cluster.control_mode != "prescribed":
        raise RuntimeError(f"Cluster '{cluster.name}' is not a prescribed rigid body")

    delta = _to_tensor(delta_xyz, device=scene.device, dtype=scene.dtype)
    next_target = scene.cluster_target_translations[cluster.name] + delta
    command_velocity = delta / _to_tensor(dt, device=scene.device, dtype=scene.dtype).clamp_min(1e-8)
    scene.cluster_target_translations[cluster.name] = next_target
    scene.cluster_command_velocities[cluster.name] = command_velocity
    set_body_state(
        state=scene.state_0,
        body_id=cluster.body_id,
        translation=next_target,
        quaternion=cluster.fixed_orientation,
        linear_velocity=command_velocity,
        angular_velocity=torch.zeros(3, device=scene.device, dtype=scene.dtype),
    )
    _sync_newton_mirror(scene, scene.state_0)


def compute_table_box(points: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        raise RuntimeError("Tabletop point set is empty")

    min_xy = points[:, :2].min(axis=0)
    max_xy = points[:, :2].max(axis=0)
    top_z = float(points[:, 2].max())
    half_extents = np.array(
        [
            0.5 * float(max_xy[0] - min_xy[0]) + 0.5 * voxel_size,
            0.5 * float(max_xy[1] - min_xy[1]) + 0.5 * voxel_size,
            0.5 * voxel_size,
        ],
        dtype=np.float32,
    )
    center = np.array(
        [
            0.5 * float(min_xy[0] + max_xy[0]),
            0.5 * float(min_xy[1] + max_xy[1]),
            top_z - half_extents[2],
        ],
        dtype=np.float32,
    )
    return center, half_extents


def compute_table_contact_lift(
    sphere_centers: np.ndarray,
    sphere_radius: float,
    table_top_z: float,
) -> float:
    if len(sphere_centers) == 0:
        return 0.0

    lowest_sphere_bottom = float(np.min(sphere_centers[:, 2]) - sphere_radius)
    return max(table_top_z - lowest_sphere_bottom, 0.0)


def build_scene_from_segmented_ply(
    ply_path: Path,
    table_seg_id: int,
    tee_seg_id: int,
    ee_seg_id: int,
    table_voxel: float,
    tee_voxel: float,
    ee_voxel: float,
    tee_radius_scale: float,
    ee_radius_scale: float,
    tee_mass: float | torch.Tensor,
    ee_mass: float | torch.Tensor,
    xpbd_iterations: int,
    table_friction: float | torch.Tensor,
    object_friction: float | torch.Tensor,
    contact_stiffness: float | torch.Tensor = DEFAULT_CONTACT_STIFFNESS,
    contact_damping: float | torch.Tensor = DEFAULT_CONTACT_DAMPING,
    contact_margin: float | torch.Tensor = DEFAULT_CONTACT_MARGIN,
    friction_regularization: float | torch.Tensor = DEFAULT_FRICTION_REGULARIZATION,
    device: str | torch.device | None = None,
) -> BuiltScene:
    header = read_ascii_ply_header(ply_path)
    cluster_configs = [
        SegmentConfig(
            name="table",
            segmentation_id=table_seg_id,
            voxel_size=table_voxel,
            total_mass=0.0,
            is_dynamic=False,
            control_mode="fixed",
            planar_motion=False,
            fill_interior=False,
            display_color=CLUSTER_COLORS["table"],
            shape_radius_scale=0.0,
        ),
        SegmentConfig(
            name="tee",
            segmentation_id=tee_seg_id,
            voxel_size=tee_voxel,
            total_mass=tee_mass,
            is_dynamic=True,
            control_mode="free",
            planar_motion=True,
            fill_interior=True,
            display_color=CLUSTER_COLORS["tee"],
            shape_radius_scale=tee_radius_scale,
        ),
        SegmentConfig(
            name="end_effector",
            segmentation_id=ee_seg_id,
            voxel_size=ee_voxel,
            total_mass=ee_mass,
            is_dynamic=False,
            control_mode="prescribed",
            planar_motion=False,
            fill_interior=True,
            display_color=CLUSTER_COLORS["end_effector"],
            shape_radius_scale=ee_radius_scale,
        ),
    ]
    sampled_positions = voxelize_selected_segments(
        ply_path=ply_path,
        header=header,
        configs_by_seg={config.segmentation_id: config for config in cluster_configs},
    )
    sampled_positions[table_seg_id] = flatten_tabletop_points(
        sampled_positions[table_seg_id],
        voxel_size=table_voxel,
    )

    torch_device = torch.device(device) if device is not None else torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    dtype = torch.float32
    identity_quat = IDENTITY_QUAT.to(device=torch_device, dtype=dtype)

    clusters: list[RigidBodyCluster] = []
    cluster_target_translations: dict[str, torch.Tensor] = {}
    cluster_command_velocities: dict[str, torch.Tensor] = {}
    body_q_rows: list[torch.Tensor] = []
    body_qd_rows: list[torch.Tensor] = []
    collision_builder = newton.ModelBuilder(gravity=-9.81)
    next_collision_shape_id = 0
    table_top_z = None

    for config in cluster_configs:
        surface_points = sampled_positions[config.segmentation_id]
        if config.name == "table":
            if len(surface_points) == 0:
                raise RuntimeError(
                    f"Segmentation id {config.segmentation_id} produced 0 tabletop samples after flattening. "
                    f"Try a smaller voxel size."
                )

            rest_translation_np, table_half_extents_np = compute_table_box(
                surface_points,
                voxel_size=config.voxel_size,
            )
            rest_translation = _to_tensor(rest_translation_np, device=torch_device, dtype=dtype)
            table_half_extents = _to_tensor(table_half_extents_np, device=torch_device, dtype=dtype)
            local_shape_positions = _to_tensor(
                surface_points - rest_translation_np,
                device=torch_device,
                dtype=dtype,
            )
            body_id = len(clusters)
            collision_body_id = collision_builder.add_body(
                xform=make_transform(rest_translation_np),
                is_kinematic=True,
                label=config.name,
            )
            if collision_body_id != body_id:
                raise RuntimeError(
                    f"Newton body id mismatch for '{config.name}': expected {body_id}, got {collision_body_id}"
                )
            collision_builder.add_shape_box(
                body=collision_body_id,
                hx=float(table_half_extents_np[0]),
                hy=float(table_half_extents_np[1]),
                hz=float(table_half_extents_np[2]),
                cfg=newton.ModelBuilder.ShapeConfig(
                    density=1.0,
                    ke=_to_float(contact_stiffness),
                    kd=_to_float(contact_damping),
                    kf=_to_float(contact_stiffness),
                    mu=_to_float(table_friction),
                    margin=_to_float(contact_margin),
                    mu_torsional=0.0,
                    mu_rolling=0.0,
                ),
                label=f"{config.name}_box",
            )
            cluster = RigidBodyCluster(
                name=config.name,
                segmentation_id=config.segmentation_id,
                body_id=body_id,
                local_shape_positions=local_shape_positions,
                shape_radius=_to_tensor(0.0, device=torch_device, dtype=dtype),
                total_mass=_to_tensor(0.0, device=torch_device, dtype=dtype),
                rest_translation=rest_translation,
                fixed_orientation=identity_quat.clone(),
                is_dynamic=False,
                planar_motion=False,
                display_color=config.display_color,
                control_mode=config.control_mode,
                collision_geometry="box",
                collision_shape_start=next_collision_shape_id,
                collision_shape_count=1,
                box_half_extents=table_half_extents,
                inertia_factor_diag=torch.ones(3, device=torch_device, dtype=dtype),
                support_radius=torch.linalg.vector_norm(table_half_extents[:2], dim=0),
            )
            next_collision_shape_id += 1
            clusters.append(cluster)
            body_q_rows.append(torch.cat([rest_translation, identity_quat], dim=0))
            body_qd_rows.append(torch.zeros(6, device=torch_device, dtype=dtype))
            table_top_z = float(rest_translation_np[2] + table_half_extents_np[2])
            continue

        points = (
            solidify_surface_points(surface_points, config.voxel_size)
            if config.fill_interior
            else surface_points
        )
        if len(points) == 0:
            raise RuntimeError(
                f"Segmentation id {config.segmentation_id} produced 0 rigid spheres after voxelization. "
                f"Try a smaller voxel size."
            )

        rest_translation_np = points.mean(axis=0, dtype=np.float64).astype(np.float32)
        local_points_np = points - rest_translation_np
        if config.control_mode != "fixed" and table_top_z is not None:
            # Keep every movable sphere cluster from starting below the tabletop proxy box.
            rest_translation_np = rest_translation_np.copy()
            rest_translation_np[2] += compute_table_contact_lift(
                sphere_centers=points,
                sphere_radius=config.shape_radius,
                table_top_z=table_top_z,
            )
        local_shape_positions = _to_tensor(
            local_points_np,
            device=torch_device,
            dtype=dtype,
        )
        shape_radius = _to_tensor(config.shape_radius, device=torch_device, dtype=dtype)
        total_mass = _to_tensor(config.total_mass, device=torch_device, dtype=dtype)
        rest_translation = _to_tensor(rest_translation_np, device=torch_device, dtype=dtype)
        body_id = len(clusters)
        collision_body_id = collision_builder.add_body(
            xform=make_transform(rest_translation_np),
            is_kinematic=not config.is_dynamic,
            label=config.name,
        )
        if collision_body_id != body_id:
            raise RuntimeError(
                f"Newton body id mismatch for '{config.name}': expected {body_id}, got {collision_body_id}"
            )
        collision_shape_start = next_collision_shape_id
        collision_shape_cfg = newton.ModelBuilder.ShapeConfig(
            density=compute_shape_density(
                total_mass=_to_float(config.total_mass),
                shape_radius=config.shape_radius,
                shape_count=len(points),
            ),
            ke=_to_float(contact_stiffness),
            kd=_to_float(contact_damping),
            kf=_to_float(contact_stiffness),
            mu=_to_float(object_friction),
            margin=_to_float(contact_margin),
            mu_torsional=0.0,
            mu_rolling=0.0,
        )
        for idx, local_point in enumerate(local_points_np):
            collision_builder.add_shape_sphere(
                body=collision_body_id,
                xform=make_transform(local_point),
                radius=float(config.shape_radius),
                cfg=collision_shape_cfg,
                label=f"{config.name}_shape_{idx}",
            )
            next_collision_shape_id += 1

        cluster = RigidBodyCluster(
            name=config.name,
            segmentation_id=config.segmentation_id,
            body_id=body_id,
            local_shape_positions=local_shape_positions,
            shape_radius=shape_radius,
            total_mass=total_mass,
            rest_translation=rest_translation,
            fixed_orientation=identity_quat.clone(),
            is_dynamic=bool(config.is_dynamic),
            planar_motion=bool(config.planar_motion),
            display_color=config.display_color,
            control_mode=config.control_mode,
            collision_geometry="sphere_cluster",
            collision_shape_start=collision_shape_start,
            collision_shape_count=int(points.shape[0]),
            inertia_factor_diag=_compute_inertia_factor_diag(local_shape_positions, shape_radius),
            support_radius=_compute_support_radius(local_shape_positions, shape_radius),
        )
        clusters.append(cluster)
        body_q_rows.append(torch.cat([rest_translation, identity_quat], dim=0))
        body_qd_rows.append(torch.zeros(6, device=torch_device, dtype=dtype))

        if cluster.control_mode == "prescribed":
            cluster_target_translations[cluster.name] = rest_translation.clone()
            cluster_command_velocities[cluster.name] = torch.zeros(
                3, device=torch_device, dtype=dtype
            )

    state_0 = SceneState(
        body_q=torch.stack(body_q_rows, dim=0),
        body_qd=torch.stack(body_qd_rows, dim=0),
    )
    state_1 = state_0.clone()
    collision_model = collision_builder.finalize(device="cpu")
    collision_pipeline = newton.CollisionPipeline(collision_model, rigid_contact_max=60000)
    scene = BuiltScene(
        state_0=state_0,
        state_1=state_1,
        clusters=clusters,
        cluster_target_translations=cluster_target_translations,
        cluster_command_velocities=cluster_command_velocities,
        constraint_iterations=max(int(xpbd_iterations), 1),
        table_friction=_to_tensor(table_friction, device=torch_device, dtype=dtype),
        object_friction=_to_tensor(object_friction, device=torch_device, dtype=dtype),
        contact_stiffness=_to_tensor(contact_stiffness, device=torch_device, dtype=dtype),
        contact_damping=_to_tensor(contact_damping, device=torch_device, dtype=dtype),
        contact_margin=_to_tensor(contact_margin, device=torch_device, dtype=dtype),
        friction_regularization=_to_tensor(
            friction_regularization,
            device=torch_device,
            dtype=dtype,
        ),
        gravity=_to_tensor([0.0, 0.0, -9.81], device=torch_device, dtype=dtype),
        device=torch_device,
        dtype=dtype,
        collision_model=collision_model,
        collision_state=collision_model.state(),
        collision_pipeline=collision_pipeline,
        collision_contacts=collision_model.contacts(collision_pipeline),
    )
    _resolve_initial_interpenetrations(scene)
    _sync_newton_mirror(scene, scene.state_0)
    return scene
