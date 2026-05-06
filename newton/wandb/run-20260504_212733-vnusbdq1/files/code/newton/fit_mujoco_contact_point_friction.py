from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import warp as wp

from mujoco_contact_friction_fit_utils import (
    MujocoTrajectory,
    MujocoTrajectoryCollection,
    OptimizationBuffers,
    compute_active_contact_point_indices,
    load_mujoco_trajectories,
    run_adam_update,
)
from mujoco_contact_friction_fit_wandb import build_wandb_log_payload, init_wandb
from newton_surface_points_diff_demo import (
    GRAVITY_MAGNITUDE,
    DiffScene,
    _smoothstep01,
    build_diff_scene,
    compute_contact_weighted_masses_kernel,
)
from pbd_usd import export_scene_usd
from project_paths import DEFAULT_OUTPUT_DIR, REPO_ROOT


DEFAULT_TRAJECTORY_NPZ_PATH = REPO_ROOT / "mujoco" / "outputs" / "block_force_trajectory.npz"
DEFAULT_CONTACT_FRICTION_RESULTS_PATH = DEFAULT_OUTPUT_DIR / "mujoco_contact_point_friction_fit.npz"
DEFAULT_CONTACT_FRICTION_SCENE_USD_PATH = DEFAULT_OUTPUT_DIR / "mujoco_contact_point_friction_fit.usda"
DEFAULT_CONTACT_FRICTION_HEATMAP_PATH = DEFAULT_OUTPUT_DIR / "mujoco_contact_point_friction_heatmap.png"


@wp.kernel
def scatter_active_point_friction_kernel(
    active_indices: wp.array(dtype=wp.int32),
    active_point_friction: wp.array(dtype=float),
    full_point_friction: wp.array(dtype=float),
):
    tid = wp.tid()
    point_idx = active_indices[tid]
    full_point_friction[point_idx] = active_point_friction[tid]


@wp.kernel
def apply_point_force_trajectory_kernel(
    step_idx: int,
    body_id: int,
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    step_forces: wp.array(dtype=wp.vec3),
    step_application_points: wp.array(dtype=wp.vec3),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    force = step_forces[step_idx]
    application_point = step_application_points[step_idx]
    pose = body_q[body_id]
    world_com = wp.transform_point(pose, body_com[body_id])
    moment_arm = application_point - world_com
    torque = wp.cross(moment_arm, force)
    wp.atomic_add(body_f, body_id, wp.spatial_vector(force, torque))


@wp.kernel
def apply_surface_point_normal_trajectory_kernel(
    step_idx: int,
    body_id: int,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    local_surface_points: wp.array(dtype=wp.vec3),
    weighted_masses: wp.array(dtype=float),
    total_weighted_mass: wp.array(dtype=float),
    step_forces: wp.array(dtype=wp.vec3),
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

    external_force = step_forces[step_idx]
    support_force_z = mass_fraction * wp.max(0.0, total_mass * gravity_magnitude - external_force[2])
    penalty_force_z = mass_fraction * activation * (
        contact_stiffness * penetration + contact_damping * wp.max(-point_velocity[2], 0.0)
    )
    normal_force = wp.vec3(0.0, 0.0, support_force_z + penalty_force_z)
    normal_torque = wp.cross(moment_arm, normal_force)
    wp.atomic_add(body_f, body_id, wp.spatial_vector(normal_force, normal_torque))


@wp.kernel
def apply_surface_point_friction_per_point_trajectory_kernel(
    step_idx: int,
    body_id: int,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    local_surface_points: wp.array(dtype=wp.vec3),
    weighted_masses: wp.array(dtype=float),
    total_weighted_mass: wp.array(dtype=float),
    point_friction: wp.array(dtype=float),
    step_forces: wp.array(dtype=wp.vec3),
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

    external_force = step_forces[step_idx]
    normal_load_total = wp.max(0.0, total_mass * gravity_magnitude - external_force[2])
    normal_load = (weighted_masses[tid] / total_weight) * normal_load_total
    mu = wp.max(point_friction[tid], 0.0)

    friction_force = -mu * normal_load * (tangential_velocity / tangential_speed)
    friction_torque = wp.cross(moment_arm, friction_force)
    wp.atomic_add(body_f, body_id, wp.spatial_vector(friction_force, friction_torque))


@wp.kernel
def accumulate_pose_loss_kernel(
    body_id: int,
    frame_idx: int,
    body_q: wp.array(dtype=wp.transform),
    target_positions: wp.array(dtype=wp.vec3),
    target_quaternions: wp.array(dtype=wp.vec4),
    frame_scale: float,
    position_loss: wp.array(dtype=float),
    orientation_loss: wp.array(dtype=float),
):
    pose = body_q[body_id]
    world_position = wp.transform_get_translation(pose)
    target_position = target_positions[frame_idx]
    position_delta = world_position - target_position
    position_loss_value = wp.dot(position_delta, position_delta)

    quat = wp.transform_get_rotation(pose)
    target_quat = target_quaternions[frame_idx]
    dot_q = quat[0] * target_quat[0] + quat[1] * target_quat[1] + quat[2] * target_quat[2] + quat[3] * target_quat[3]
    sign = 1.0
    if dot_q < 0.0:
        sign = -1.0
    quat_dx = sign * quat[0] - target_quat[0]
    quat_dy = sign * quat[1] - target_quat[1]
    quat_dz = sign * quat[2] - target_quat[2]
    quat_dw = sign * quat[3] - target_quat[3]
    orientation_loss_value = quat_dx * quat_dx + quat_dy * quat_dy + quat_dz * quat_dz + quat_dw * quat_dw

    wp.atomic_add(position_loss, 0, frame_scale * position_loss_value)
    wp.atomic_add(orientation_loss, 0, frame_scale * orientation_loss_value)


@wp.kernel
def accumulate_velocity_loss_kernel(
    body_id: int,
    frame_idx: int,
    body_qd: wp.array(dtype=wp.spatial_vector),
    target_linear_velocity: wp.array(dtype=wp.vec3),
    target_angular_velocity: wp.array(dtype=wp.vec3),
    frame_scale: float,
    linear_velocity_loss: wp.array(dtype=float),
    angular_velocity_loss: wp.array(dtype=float),
):
    spatial_velocity = body_qd[body_id]
    linear_velocity = wp.spatial_top(spatial_velocity)
    angular_velocity = wp.spatial_bottom(spatial_velocity)

    linear_delta = linear_velocity - target_linear_velocity[frame_idx]
    angular_delta = angular_velocity - target_angular_velocity[frame_idx]
    linear_loss_value = wp.dot(linear_delta, linear_delta)
    angular_loss_value = wp.dot(angular_delta, angular_delta)

    wp.atomic_add(linear_velocity_loss, 0, frame_scale * linear_loss_value)
    wp.atomic_add(angular_velocity_loss, 0, frame_scale * angular_loss_value)


@wp.kernel
def combine_loss_components_kernel(
    position_loss: wp.array(dtype=float),
    orientation_loss: wp.array(dtype=float),
    linear_velocity_loss: wp.array(dtype=float),
    angular_velocity_loss: wp.array(dtype=float),
    position_weight: float,
    orientation_weight: float,
    linear_velocity_weight: float,
    angular_velocity_weight: float,
    loss: wp.array(dtype=float),
):
    loss[0] = (
        position_weight * position_loss[0]
        + orientation_weight * orientation_loss[0]
        + linear_velocity_weight * linear_velocity_loss[0]
        + angular_velocity_weight * angular_velocity_loss[0]
    )


def reset_scene_states(diff_scene: DiffScene, initial_body_q: np.ndarray, initial_body_qd: np.ndarray) -> None:
    for state in diff_scene.states:
        state.body_q.assign(initial_body_q)
        if getattr(state, "body_q_prev", None) is not None:
            state.body_q_prev.assign(initial_body_q)
        state.body_qd.assign(initial_body_qd)
        if getattr(state, "body_qdd", None) is not None:
            state.body_qdd.zero_()
        state.body_f.zero_()
        if getattr(state, "body_parent_f", None) is not None:
            state.body_parent_f.zero_()


def build_optimization_buffers(
    diff_scene: DiffScene,
    trajectory: MujocoTrajectory,
    args: argparse.Namespace,
    active_indices: np.ndarray,
) -> OptimizationBuffers:
    device = str(diff_scene.torch_device)
    point_count = len(diff_scene.local_surface_points_np)
    base_point_friction = np.full(point_count, float(args.point_friction), dtype=np.float32)
    active_point_friction = np.full(len(active_indices), float(args.point_friction), dtype=np.float32)

    return OptimizationBuffers(
        active_point_friction=wp.array(active_point_friction, dtype=wp.float32, device=device, requires_grad=True),
        active_indices=wp.array(active_indices, dtype=wp.int32, device=device),
        full_point_friction=wp.array(base_point_friction, dtype=wp.float32, device=device, requires_grad=True),
        contact_weighted_masses=wp.zeros(
            point_count,
            dtype=wp.float32,
            device=device,
            requires_grad=True,
        ),
        contact_weighted_mass_total=wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True),
        step_forces=wp.array(trajectory.step_forces, dtype=wp.vec3, device=device),
        step_application_points=wp.array(trajectory.step_application_points, dtype=wp.vec3, device=device),
        target_positions=wp.array(trajectory.positions, dtype=wp.vec3, device=device),
        target_quaternions=wp.array(trajectory.quaternions_xyzw, dtype=wp.vec4, device=device),
        target_linear_velocity=wp.array(trajectory.linear_velocity, dtype=wp.vec3, device=device),
        target_angular_velocity=wp.array(trajectory.angular_velocity, dtype=wp.vec3, device=device),
        loss=wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True),
        position_loss=wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True),
        orientation_loss=wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True),
        linear_velocity_loss=wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True),
        angular_velocity_loss=wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True),
        inactive_point_friction_np=base_point_friction,
    )


def build_optimization_buffers_for_collection(
    diff_scene: DiffScene,
    trajectories: list[MujocoTrajectory],
    args: argparse.Namespace,
    active_indices: np.ndarray,
) -> list[OptimizationBuffers]:
    return [
        build_optimization_buffers(
            diff_scene=diff_scene,
            trajectory=trajectory,
            args=args,
            active_indices=active_indices,
        )
        for trajectory in trajectories
    ]


def clear_optimization_grads(buffers: OptimizationBuffers) -> None:
    if buffers.active_point_friction.grad is not None:
        buffers.active_point_friction.grad.zero_()
    if buffers.full_point_friction.grad is not None:
        buffers.full_point_friction.grad.zero_()
    if buffers.contact_weighted_masses.grad is not None:
        buffers.contact_weighted_masses.grad.zero_()
    if buffers.contact_weighted_mass_total.grad is not None:
        buffers.contact_weighted_mass_total.grad.zero_()
    if buffers.loss.grad is not None:
        buffers.loss.grad.zero_()
    if buffers.position_loss.grad is not None:
        buffers.position_loss.grad.zero_()
    if buffers.orientation_loss.grad is not None:
        buffers.orientation_loss.grad.zero_()
    if buffers.linear_velocity_loss.grad is not None:
        buffers.linear_velocity_loss.grad.zero_()
    if buffers.angular_velocity_loss.grad is not None:
        buffers.angular_velocity_loss.grad.zero_()


def forward_rollout_with_trajectory_loss(
    diff_scene: DiffScene,
    buffers: OptimizationBuffers,
    trajectory: MujocoTrajectory,
    args: argparse.Namespace,
) -> wp.array:
    frame_scale = 1.0 / max(trajectory.num_frames, 1)

    wp.launch(
        scatter_active_point_friction_kernel,
        dim=len(buffers.active_indices.numpy()),
        inputs=[
            buffers.active_indices,
            buffers.active_point_friction,
            buffers.full_point_friction,
        ],
        device=diff_scene.model.device,
    )

    buffers.loss.zero_()
    buffers.position_loss.zero_()
    buffers.orientation_loss.zero_()
    buffers.linear_velocity_loss.zero_()
    buffers.angular_velocity_loss.zero_()
    wp.launch(
        accumulate_pose_loss_kernel,
        dim=1,
        inputs=[
            diff_scene.box_body,
            0,
            diff_scene.states[0].body_q,
            buffers.target_positions,
            buffers.target_quaternions,
            float(frame_scale),
            buffers.position_loss,
            buffers.orientation_loss,
        ],
        device=diff_scene.model.device,
    )
    if args.linear_velocity_loss_weight > 0.0 or args.angular_velocity_loss_weight > 0.0:
        wp.launch(
            accumulate_velocity_loss_kernel,
            dim=1,
            inputs=[
                diff_scene.box_body,
                0,
                diff_scene.states[0].body_qd,
                buffers.target_linear_velocity,
                buffers.target_angular_velocity,
                float(frame_scale),
                buffers.linear_velocity_loss,
                buffers.angular_velocity_loss,
            ],
            device=diff_scene.model.device,
        )

    for step_idx in range(trajectory.num_steps):
        state_in = diff_scene.states[step_idx]
        state_out = diff_scene.states[step_idx + 1]
        state_in.clear_forces()

        wp.launch(
            apply_point_force_trajectory_kernel,
            dim=1,
            inputs=[
                step_idx,
                diff_scene.box_body,
                state_in.body_q,
                diff_scene.model.body_com,
                buffers.step_forces,
                buffers.step_application_points,
                state_in.body_f,
            ],
            device=diff_scene.model.device,
        )

        buffers.contact_weighted_masses.zero_()
        buffers.contact_weighted_mass_total.zero_()
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
                buffers.contact_weighted_masses,
                buffers.contact_weighted_mass_total,
            ],
            device=diff_scene.model.device,
        )
        wp.launch(
            apply_surface_point_normal_trajectory_kernel,
            dim=len(diff_scene.local_surface_points_np),
            inputs=[
                step_idx,
                diff_scene.box_body,
                state_in.body_q,
                state_in.body_qd,
                diff_scene.model.body_com,
                diff_scene.local_surface_points_wp,
                buffers.contact_weighted_masses,
                buffers.contact_weighted_mass_total,
                buffers.step_forces,
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
            apply_surface_point_friction_per_point_trajectory_kernel,
            dim=len(diff_scene.local_surface_points_np),
            inputs=[
                step_idx,
                diff_scene.box_body,
                state_in.body_q,
                state_in.body_qd,
                diff_scene.model.body_com,
                diff_scene.local_surface_points_wp,
                buffers.contact_weighted_masses,
                buffers.contact_weighted_mass_total,
                buffers.full_point_friction,
                buffers.step_forces,
                float(diff_scene.box_mass),
                float(GRAVITY_MAGNITUDE),
                float(args.friction_regularization),
                state_in.body_f,
            ],
            device=diff_scene.model.device,
        )

        diff_scene.collision_pipeline.collide(state_in, diff_scene.contacts)
        diff_scene.solver.step(state_in, state_out, diff_scene.control, diff_scene.contacts, float(args.dt))

        wp.launch(
            accumulate_pose_loss_kernel,
            dim=1,
            inputs=[
                diff_scene.box_body,
                step_idx + 1,
                state_out.body_q,
                buffers.target_positions,
                buffers.target_quaternions,
                float(frame_scale),
                buffers.position_loss,
                buffers.orientation_loss,
            ],
            device=diff_scene.model.device,
        )
        if args.linear_velocity_loss_weight > 0.0 or args.angular_velocity_loss_weight > 0.0:
            wp.launch(
                accumulate_velocity_loss_kernel,
                dim=1,
                inputs=[
                    diff_scene.box_body,
                    step_idx + 1,
                    state_out.body_qd,
                    buffers.target_linear_velocity,
                    buffers.target_angular_velocity,
                    float(frame_scale),
                    buffers.linear_velocity_loss,
                    buffers.angular_velocity_loss,
                ],
                device=diff_scene.model.device,
            )

    wp.launch(
        combine_loss_components_kernel,
        dim=1,
        inputs=[
            buffers.position_loss,
            buffers.orientation_loss,
            buffers.linear_velocity_loss,
            buffers.angular_velocity_loss,
            float(args.position_loss_weight),
            float(args.orientation_loss_weight),
            float(args.linear_velocity_loss_weight),
            float(args.angular_velocity_loss_weight),
            buffers.loss,
        ],
        device=diff_scene.model.device,
    )

    return buffers.loss


def evaluate_loss(
    diff_scene: DiffScene,
    buffers: OptimizationBuffers,
    trajectory: MujocoTrajectory,
    args: argparse.Namespace,
    initial_body_q: np.ndarray,
    initial_body_qd: np.ndarray,
) -> tuple[float, float, float, float, float, list[np.ndarray]]:
    reset_scene_states(diff_scene, initial_body_q, initial_body_qd)
    buffers.full_point_friction.assign(buffers.inactive_point_friction_np)
    clear_optimization_grads(buffers)
    loss = forward_rollout_with_trajectory_loss(diff_scene, buffers, trajectory, args)
    body_q_frames = [state.body_q.numpy().copy() for state in diff_scene.states[: trajectory.num_steps + 1]]
    return (
        float(loss.numpy()[0]),
        float(buffers.position_loss.numpy()[0]),
        float(buffers.orientation_loss.numpy()[0]),
        float(buffers.linear_velocity_loss.numpy()[0]),
        float(buffers.angular_velocity_loss.numpy()[0]),
        body_q_frames,
    )


def evaluate_collection_loss(
    diff_scene: DiffScene,
    buffers_list: list[OptimizationBuffers],
    trajectories: list[MujocoTrajectory],
    args: argparse.Namespace,
    initial_body_q: np.ndarray,
    initial_body_qd: np.ndarray,
) -> tuple[float, float, float, float, float, list[np.ndarray]]:
    if len(buffers_list) != len(trajectories):
        raise ValueError("Buffer count must match trajectory count")

    total_loss = 0.0
    total_position_loss = 0.0
    total_orientation_loss = 0.0
    total_linear_velocity_loss = 0.0
    total_angular_velocity_loss = 0.0
    representative_body_q_frames: list[np.ndarray] = []

    for trajectory_idx, (buffers, trajectory) in enumerate(zip(buffers_list, trajectories, strict=True)):
        (
            loss_value,
            position_loss_value,
            orientation_loss_value,
            linear_velocity_loss_value,
            angular_velocity_loss_value,
            body_q_frames,
        ) = evaluate_loss(
            diff_scene=diff_scene,
            buffers=buffers,
            trajectory=trajectory,
            args=args,
            initial_body_q=initial_body_q,
            initial_body_qd=initial_body_qd,
        )
        total_loss += loss_value
        total_position_loss += position_loss_value
        total_orientation_loss += orientation_loss_value
        total_linear_velocity_loss += linear_velocity_loss_value
        total_angular_velocity_loss += angular_velocity_loss_value
        if trajectory_idx == 0:
            representative_body_q_frames = body_q_frames

    scale = 1.0 / max(len(trajectories), 1)
    return (
        total_loss * scale,
        total_position_loss * scale,
        total_orientation_loss * scale,
        total_linear_velocity_loss * scale,
        total_angular_velocity_loss * scale,
        representative_body_q_frames,
    )


def save_contact_friction_heatmap(
    *,
    local_surface_points: np.ndarray,
    active_indices: np.ndarray,
    active_point_friction: np.ndarray,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    active_points = np.asarray(local_surface_points[active_indices], dtype=np.float32)
    if len(active_points) == 0:
        raise ValueError("No active contact points available for heatmap export.")

    z_values = active_points[:, 2]
    bottom_z = float(np.min(z_values))
    bottom_mask = np.isclose(z_values, bottom_z, atol=1.0e-4)
    contact_face_points = active_points[bottom_mask]
    contact_face_friction = np.asarray(active_point_friction[bottom_mask], dtype=np.float32)

    if len(contact_face_points) == 0:
        contact_face_points = active_points
        contact_face_friction = np.asarray(active_point_friction, dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    scatter = ax.scatter(
        contact_face_points[:, 0],
        contact_face_points[:, 1],
        c=contact_face_friction,
        cmap="YlOrRd",
        s=180,
        marker="s",
        edgecolors="black",
        linewidths=0.4,
    )
    ax.set_title("Contact Surface Friction Heatmap")
    ax.set_xlabel("Local X")
    ax.set_ylabel("Local Y")
    ax.set_aspect("equal", adjustable="box")

    x_pad = max(float(np.ptp(contact_face_points[:, 0])) * 0.08, 1.0e-3)
    y_pad = max(float(np.ptp(contact_face_points[:, 1])) * 0.08, 1.0e-3)
    ax.set_xlim(float(contact_face_points[:, 0].min() - x_pad), float(contact_face_points[:, 0].max() + x_pad))
    ax.set_ylim(float(contact_face_points[:, 1].min() - y_pad), float(contact_face_points[:, 1].max() + y_pad))

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Friction Coefficient")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--trajectory-npz", type=Path, default=DEFAULT_TRAJECTORY_NPZ_PATH)
    parser.add_argument("--max-trajectories", type=int, default=None, help="Use only the first N trajectories when the input NPZ is a dataset.")
    parser.add_argument("--results-path", type=Path, default=DEFAULT_CONTACT_FRICTION_RESULTS_PATH)
    parser.add_argument("--scene-usd-path", type=Path, default=DEFAULT_CONTACT_FRICTION_SCENE_USD_PATH)
    parser.add_argument("--heatmap-path", type=Path, default=DEFAULT_CONTACT_FRICTION_HEATMAP_PATH)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default="newton-contact-point-friction-fit")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default="mujoco-contact-friction")
    parser.add_argument("--wandb-mode", type=str, default="online")
    parser.add_argument("--wandb-dir", type=Path, default=None)
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Use only the first N simulation steps from the MuJoCo trajectory.")
    parser.add_argument("--opt-iters", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=2.0e-2)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1.0e-8)
    parser.add_argument("--min-point-friction", type=float, default=0.0)
    parser.add_argument("--max-point-friction", type=float, default=2.0)
    parser.add_argument("--position-loss-weight", type=float, default=1.0)
    parser.add_argument("--orientation-loss-weight", type=float, default=0.1)
    parser.add_argument("--linear-velocity-loss-weight", type=float, default=0.0)
    parser.add_argument("--angular-velocity-loss-weight", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--steps", type=int, default=0, help="Filled automatically from the trajectory after loading.")
    parser.add_argument("--dt", type=float, default=0.0, help="Filled automatically from the trajectory after loading.")
    parser.add_argument("--solver-iterations", type=int, default=10)
    parser.add_argument("--box-mass", type=float, default=1.0)
    parser.add_argument("--floor-half-extents", type=float, nargs=3, default=(2.0, 2.0, 0.05))
    parser.add_argument("--box-half-extents", type=float, nargs=3, default=(0.1, 0.05, 0.025))
    parser.add_argument("--box-start-pos", type=float, nargs=3, default=(0.58, 0.0, 0.025))
    parser.add_argument("--surface-point-spacing", type=float, default=0.02)
    parser.add_argument("--friction-contact-threshold", type=float, default=0.002)
    parser.add_argument("--contact-mask-threshold", type=float, default=0.002)
    parser.add_argument("--point-friction", type=float, default=0.1)
    parser.add_argument("--contact-friction", type=float, default=0.0)
    parser.add_argument("--contact-stiffness", type=float, default=2.0e4)
    parser.add_argument("--contact-damping", type=float, default=50.0)
    parser.add_argument("--contact-margin", type=float, default=1.0e-3)
    parser.add_argument("--friction-regularization", type=float, default=1.0e-3)
    parser.add_argument("--initial-force", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--initial-torque", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--force-magnitude", type=float, default=None)
    parser.add_argument("--force-direction", type=float, nargs=3, default=None)
    parser.add_argument("--force-point", type=float, nargs=3, default=None)
    parser.add_argument("--force-point-local", type=float, nargs=3, default=None)
    parser.add_argument("--force-steps", type=int, default=0)
    parser.add_argument("--loss-target-position", type=float, nargs=3, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectory_collection = load_mujoco_trajectories(args.trajectory_npz, args.max_steps, args.max_trajectories)
    trajectories = trajectory_collection.trajectories
    representative_trajectory = trajectories[0]
    args.steps = trajectory_collection.max_steps
    args.dt = representative_trajectory.timestep

    diff_scene = build_diff_scene(args)
    initial_body_q = diff_scene.states[0].body_q.numpy().copy()
    initial_body_qd = diff_scene.states[0].body_qd.numpy().copy()

    active_mask = np.zeros(len(diff_scene.local_surface_points_np), dtype=bool)
    for trajectory in trajectories:
        trajectory_active_indices = compute_active_contact_point_indices(
            local_surface_points=diff_scene.local_surface_points_np,
            trajectory=trajectory,
            floor_top_z=diff_scene.floor_top_z,
            contact_threshold=float(args.contact_mask_threshold),
        )
        active_mask[trajectory_active_indices] = True
    active_indices = np.flatnonzero(active_mask).astype(np.int32)
    if len(active_indices) == 0:
        raise RuntimeError(
            "No contact points were detected in the target trajectory. "
            "Try increasing --contact-mask-threshold or decreasing --surface-point-spacing."
        )

    wandb_run = init_wandb(args, trajectory_collection, active_indices)
    if wandb_run is not None:
        print(
            f"W&B enabled | project={args.wandb_project} | "
            f"run={wandb_run.name} | mode={args.wandb_mode}"
        )

    buffers_list = build_optimization_buffers_for_collection(diff_scene, trajectories, args, active_indices)
    active_params = buffers_list[0].active_point_friction.numpy().astype(np.float32)
    adam_m = np.zeros_like(active_params)
    adam_v = np.zeros_like(active_params)
    loss_history: list[float] = []
    best_loss = float("inf")
    best_active_params = active_params.copy()

    try:
        for iteration in range(1, max(int(args.opt_iters), 0) + 1):
            for buffers in buffers_list:
                buffers.active_point_friction.assign(active_params)
                buffers.full_point_friction.assign(buffers.inactive_point_friction_np)
                clear_optimization_grads(buffers)

            loss_value_total = 0.0
            position_loss_value_total = 0.0
            orientation_loss_value_total = 0.0
            linear_velocity_loss_value_total = 0.0
            angular_velocity_loss_value_total = 0.0
            grad_value_total = np.zeros_like(active_params)
            tape = wp.Tape()
            with tape:
                losses: list[wp.array] = []
                for buffers, trajectory in zip(buffers_list, trajectories, strict=True):
                    reset_scene_states(diff_scene, initial_body_q, initial_body_qd)
                    losses.append(forward_rollout_with_trajectory_loss(diff_scene, buffers, trajectory, args))
            for loss in losses:
                tape.backward(loss, grads={loss: np.array([1.0 / len(losses)], dtype=np.float32)})

            for buffers in buffers_list:
                loss_value_total += float(buffers.loss.numpy()[0])
                position_loss_value_total += float(buffers.position_loss.numpy()[0])
                orientation_loss_value_total += float(buffers.orientation_loss.numpy()[0])
                linear_velocity_loss_value_total += float(buffers.linear_velocity_loss.numpy()[0])
                angular_velocity_loss_value_total += float(buffers.angular_velocity_loss.numpy()[0])
                grad_value_total += buffers.active_point_friction.grad.numpy().astype(np.float32)

            scale = 1.0 / max(len(buffers_list), 1)
            loss_value = loss_value_total * scale
            position_loss_value = position_loss_value_total * scale
            orientation_loss_value = orientation_loss_value_total * scale
            linear_velocity_loss_value = linear_velocity_loss_value_total * scale
            angular_velocity_loss_value = angular_velocity_loss_value_total * scale
            grad_value = grad_value_total
            active_params, adam_m, adam_v = run_adam_update(
                params=active_params,
                grads=grad_value,
                first_moment=adam_m,
                second_moment=adam_v,
                step=iteration,
                learning_rate=float(args.learning_rate),
                beta1=float(args.adam_beta1),
                beta2=float(args.adam_beta2),
                eps=float(args.adam_eps),
                min_value=float(args.min_point_friction),
                max_value=float(args.max_point_friction),
            )
            tape.zero()
            loss_history.append(loss_value)

            if loss_value < best_loss:
                best_loss = loss_value
                best_active_params = active_params.copy()

            if wandb_run is not None:
                log_payload = build_wandb_log_payload(
                    loss_value=loss_value,
                    position_loss_value=position_loss_value,
                    orientation_loss_value=orientation_loss_value,
                    linear_velocity_loss_value=linear_velocity_loss_value,
                    angular_velocity_loss_value=angular_velocity_loss_value,
                    grad_value=grad_value,
                    active_params=active_params,
                    active_indices=active_indices,
                )
                wandb_run.log(log_payload, step=iteration)

            if iteration == 1 or iteration % max(int(args.log_every), 1) == 0 or iteration == int(args.opt_iters):
                print(
                    f"iter={iteration:04d} loss={loss_value:.6f} "
                    f"pos={position_loss_value:.6f} "
                    f"ori={orientation_loss_value:.6f} "
                    f"linvel={linear_velocity_loss_value:.6f} "
                    f"angvel={angular_velocity_loss_value:.6f} "
                    f"grad_norm={float(np.linalg.norm(grad_value)):.6f} "
                    f"mu_min={float(active_params.min()):.6f} "
                    f"mu_max={float(active_params.max()):.6f}"
                )

        for buffers in buffers_list:
            buffers.active_point_friction.assign(best_active_params)
        final_loss, final_position_loss, final_orientation_loss, final_linear_velocity_loss, final_angular_velocity_loss, body_q_frames = evaluate_collection_loss(
            diff_scene=diff_scene,
            buffers_list=buffers_list,
            trajectories=trajectories,
            args=args,
            initial_body_q=initial_body_q,
            initial_body_qd=initial_body_qd,
        )

        learned_point_friction = buffers_list[0].inactive_point_friction_np.copy()
        learned_point_friction[active_indices] = best_active_params

        save_contact_friction_heatmap(
            local_surface_points=diff_scene.local_surface_points_np,
            active_indices=active_indices,
            active_point_friction=best_active_params,
            output_path=args.heatmap_path,
        )

        args.results_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.results_path,
            trajectory_npz_path=np.asarray(str(args.trajectory_npz)),
            trajectory_source_type=np.asarray(trajectory_collection.source_type),
            trajectory_count=np.asarray(len(trajectories), dtype=np.int32),
            trajectory_steps=np.asarray([trajectory.num_steps for trajectory in trajectories], dtype=np.int32),
            trajectory_frames=np.asarray([trajectory.num_frames for trajectory in trajectories], dtype=np.int32),
            representative_time=representative_trajectory.time,
            representative_target_positions=representative_trajectory.positions,
            representative_target_quaternions_xyzw=representative_trajectory.quaternions_xyzw,
            representative_target_linear_velocity=representative_trajectory.linear_velocity,
            representative_target_angular_velocity=representative_trajectory.angular_velocity,
            representative_target_step_forces=representative_trajectory.step_forces,
            representative_target_step_application_points=representative_trajectory.step_application_points,
            local_surface_points=diff_scene.local_surface_points_np,
            point_masses=diff_scene.point_masses_np,
            active_contact_point_indices=active_indices,
            active_contact_local_points=diff_scene.local_surface_points_np[active_indices],
            learned_point_friction=learned_point_friction,
            learned_active_point_friction=best_active_params,
            loss_history=np.asarray(loss_history, dtype=np.float32),
            best_loss=np.asarray(best_loss, dtype=np.float32),
            final_loss=np.asarray(final_loss, dtype=np.float32),
            final_position_loss=np.asarray(final_position_loss, dtype=np.float32),
            final_orientation_loss=np.asarray(final_orientation_loss, dtype=np.float32),
            final_linear_velocity_loss=np.asarray(final_linear_velocity_loss, dtype=np.float32),
            final_angular_velocity_loss=np.asarray(final_angular_velocity_loss, dtype=np.float32),
            heatmap_path=np.asarray(str(args.heatmap_path)),
        )

        if args.scene_usd_path is not None:
            args.scene_usd_path.parent.mkdir(parents=True, exist_ok=True)
            export_scene_usd(
                scene=diff_scene.scene,
                output_path=args.scene_usd_path,
                body_q_frames=body_q_frames,
                fps=1.0 / float(args.dt),
            )

        if wandb_run is not None:
            wandb_run.summary["surface_points"] = int(len(diff_scene.local_surface_points_np))
            wandb_run.summary["active_contact_points"] = int(len(active_indices))
            wandb_run.summary["final_loss"] = float(final_loss)
            wandb_run.summary["final_position_loss"] = float(final_position_loss)
            wandb_run.summary["final_orientation_loss"] = float(final_orientation_loss)
            wandb_run.summary["final_linear_velocity_loss"] = float(final_linear_velocity_loss)
            wandb_run.summary["final_angular_velocity_loss"] = float(final_angular_velocity_loss)
            wandb_run.summary["mu_mean"] = float(best_active_params.mean())
            wandb_run.summary["mu_std"] = float(best_active_params.std())
            wandb_run.summary["mu_min"] = float(best_active_params.min())
            wandb_run.summary["mu_max"] = float(best_active_params.max())
            wandb_run.summary["results_path"] = str(args.results_path.resolve())
            wandb_run.summary["heatmap_path"] = str(args.heatmap_path.resolve())
            if args.scene_usd_path is not None:
                wandb_run.summary["scene_usd_path"] = str(args.scene_usd_path.resolve())

        print(f"trajectory={args.trajectory_npz.resolve()}")
        print(f"trajectory_source_type={trajectory_collection.source_type}")
        print(f"trajectory_count={len(trajectories)}")
        print(f"max_steps={trajectory_collection.max_steps} dt={representative_trajectory.timestep:.6f}")
        print(f"surface_points={len(diff_scene.local_surface_points_np)} active_contact_points={len(active_indices)}")
        print(f"final_loss={final_loss:.6f}")
        print(f"final_position_loss={final_position_loss:.6f}")
        print(f"final_orientation_loss={final_orientation_loss:.6f}")
        print(f"final_linear_velocity_loss={final_linear_velocity_loss:.6f}")
        print(f"final_angular_velocity_loss={final_angular_velocity_loss:.6f}")
        print(f"results_written_to={args.results_path.resolve()}")
        print(f"heatmap_written_to={args.heatmap_path.resolve()}")
        if args.scene_usd_path is not None:
            print(f"scene_usd_written_to={args.scene_usd_path.resolve()}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
