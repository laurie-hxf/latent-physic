from __future__ import annotations

import warp as wp


@wp.func
def _rotate_point_by_quat_xyzw(quat: wp.quat, point: wp.vec3) -> wp.vec3:
    quat_xyz = wp.vec3(quat[0], quat[1], quat[2])
    twice_cross = 2.0 * wp.cross(quat_xyz, point)
    return point + quat[3] * twice_cross + wp.cross(quat_xyz, twice_cross)


@wp.func
def _quat_mul_xyzw(a: wp.quat, b: wp.quat) -> wp.quat:
    ax = a[0]
    ay = a[1]
    az = a[2]
    aw = a[3]
    bx = b[0]
    by = b[1]
    bz = b[2]
    bw = b[3]
    return wp.quat(
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


@wp.func
def _apply_yaw_delta_xyzw(quat: wp.quat, yaw_delta: float) -> wp.quat:
    half = 0.5 * yaw_delta
    delta = wp.quat(0.0, 0.0, wp.sin(half), wp.cos(half))
    return wp.normalize(_quat_mul_xyzw(delta, quat))


@wp.kernel
def apply_planar_velocity_residual_kernel(
    box_body_ids: wp.array(dtype=wp.int32),
    residual_body: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    batch_idx = wp.tid()
    body_id = box_body_ids[batch_idx]
    residual = residual_body[batch_idx]
    quat = wp.transform_get_rotation(body_q[body_id])
    delta_world = _rotate_point_by_quat_xyzw(quat, wp.vec3(residual[0], residual[1], 0.0))

    qd = body_qd[body_id]
    linear = wp.spatial_top(qd)
    angular = wp.spatial_bottom(qd)
    body_qd[body_id] = wp.spatial_vector(
        wp.vec3(linear[0] + delta_world[0], linear[1] + delta_world[1], linear[2]),
        wp.vec3(angular[0], angular[1], angular[2] + residual[2]),
    )


@wp.kernel
def apply_planar_pose_velocity_residual_kernel(
    box_body_ids: wp.array(dtype=wp.int32),
    residual_values: wp.array(dtype=wp.float32),
    residual_dim: int,
    has_pose: int,
    has_velocity: int,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    batch_idx = wp.tid()
    body_id = box_body_ids[batch_idx]
    base = batch_idx * residual_dim
    pose_x = float(0.0)
    pose_y = float(0.0)
    pose_yaw = float(0.0)
    vel_x = float(0.0)
    vel_y = float(0.0)
    vel_yaw = float(0.0)
    if has_pose != 0:
        pose_x = residual_values[base + 0]
        pose_y = residual_values[base + 1]
        pose_yaw = residual_values[base + 2]
    if has_velocity != 0:
        velocity_base = base
        if has_pose != 0:
            velocity_base = base + 3
        vel_x = residual_values[velocity_base + 0]
        vel_y = residual_values[velocity_base + 1]
        vel_yaw = residual_values[velocity_base + 2]

    pose = body_q[body_id]
    quat = wp.transform_get_rotation(pose)
    pose_delta_world = _rotate_point_by_quat_xyzw(quat, wp.vec3(pose_x, pose_y, 0.0))
    vel_delta_world = _rotate_point_by_quat_xyzw(quat, wp.vec3(vel_x, vel_y, 0.0))

    pos = wp.transform_get_translation(pose)
    body_q[body_id] = wp.transform(
        wp.vec3(pos[0] + pose_delta_world[0], pos[1] + pose_delta_world[1], pos[2]),
        _apply_yaw_delta_xyzw(quat, pose_yaw),
    )

    qd = body_qd[body_id]
    linear = wp.spatial_top(qd)
    angular = wp.spatial_bottom(qd)
    body_qd[body_id] = wp.spatial_vector(
        wp.vec3(linear[0] + vel_delta_world[0], linear[1] + vel_delta_world[1], linear[2]),
        wp.vec3(angular[0], angular[1], angular[2] + vel_yaw),
    )
