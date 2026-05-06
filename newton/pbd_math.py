from __future__ import annotations

import numpy as np
import torch

from pbd_types import IDENTITY_QUAT


def _torch_identity_quat(reference: torch.Tensor) -> torch.Tensor:
    return IDENTITY_QUAT.to(device=reference.device, dtype=reference.dtype)


def normalize_quaternion(quat_xyzw):
    if torch.is_tensor(quat_xyzw):
        quat = quat_xyzw
        if quat.ndim == 1:
            norm = torch.linalg.vector_norm(quat)
            identity = _torch_identity_quat(quat)
            return torch.where(norm < 1e-8, identity, quat / norm.clamp_min(1e-8))

        norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
        identity = _torch_identity_quat(quat).expand_as(quat)
        return torch.where(norm < 1e-8, identity, quat / norm.clamp_min(1e-8))

    quat = np.asarray(quat_xyzw, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        return IDENTITY_QUAT.detach().cpu().numpy().copy()
    return (quat / norm).astype(np.float32)


def quaternion_multiply(lhs_xyzw: torch.Tensor, rhs_xyzw: torch.Tensor) -> torch.Tensor:
    lx, ly, lz, lw = lhs_xyzw.unbind(dim=-1)
    rx, ry, rz, rw = rhs_xyzw.unbind(dim=-1)
    return torch.stack(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dim=-1,
    )


def quaternion_conjugate(quat_xyzw: torch.Tensor) -> torch.Tensor:
    xyz = -quat_xyzw[..., :3]
    w = quat_xyzw[..., 3:]
    return torch.cat([xyz, w], dim=-1)


def quaternion_derivative(quat_xyzw: torch.Tensor, angular_velocity: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(angular_velocity[..., :1])
    omega = torch.cat([angular_velocity, zeros], dim=-1)
    return 0.5 * quaternion_multiply(omega, quat_xyzw)


def quaternion_to_matrix(quat_xyzw):
    if torch.is_tensor(quat_xyzw):
        quat = normalize_quaternion(quat_xyzw)
        x, y, z, w = quat.unbind(dim=-1)
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        rows = [
            torch.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)], dim=-1),
            torch.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)], dim=-1),
            torch.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], dim=-1),
        ]
        return torch.stack(rows, dim=-2)

    x, y, z, w = normalize_quaternion(quat_xyzw).astype(np.float64)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def rotation_vector_to_quaternion(rotation_vector: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.vector_norm(rotation_vector, dim=-1, keepdim=True)
    half_angle = 0.5 * angle
    safe_angle = angle.clamp_min(1e-8)
    axis = rotation_vector / safe_angle
    xyz = axis * torch.sin(half_angle)
    w = torch.cos(half_angle)
    quat = torch.cat([xyz, w], dim=-1)
    identity = _torch_identity_quat(rotation_vector)
    return torch.where(angle < 1e-8, identity, quat)


def apply_quaternion_delta(quat_xyzw: torch.Tensor, delta_rotation: torch.Tensor) -> torch.Tensor:
    delta_quat = rotation_vector_to_quaternion(delta_rotation)
    return normalize_quaternion(quaternion_multiply(delta_quat, quat_xyzw))


def quaternion_delta_to_angular_velocity(
    quat_prev_xyzw: torch.Tensor,
    quat_next_xyzw: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    delta_quat = quaternion_multiply(quat_next_xyzw, quaternion_conjugate(quat_prev_xyzw))
    delta_quat = normalize_quaternion(delta_quat)
    sign = torch.where(delta_quat[..., 3:] < 0.0, -1.0, 1.0)
    delta_quat = delta_quat * sign

    xyz = delta_quat[..., :3]
    w = delta_quat[..., 3:].clamp(-1.0, 1.0)
    sin_half = torch.linalg.vector_norm(xyz, dim=-1, keepdim=True)
    safe_sin = sin_half.clamp_min(1e-8)
    axis = xyz / safe_sin
    angle = 2.0 * torch.atan2(sin_half, w)
    rotation_vector = axis * angle
    zeros = torch.zeros_like(rotation_vector)
    return torch.where(sin_half < 1e-8, zeros, rotation_vector / dt.clamp_min(1e-8))


def diagonal_inertia_to_world(
    inertia_diag: torch.Tensor,
    quaternion_xyzw: torch.Tensor,
) -> torch.Tensor:
    rotation = quaternion_to_matrix(quaternion_xyzw)
    inertia_local = torch.diag_embed(inertia_diag)
    return rotation @ inertia_local @ rotation.transpose(-1, -2)


def diagonal_inv_inertia_to_world(
    inv_inertia_diag: torch.Tensor,
    quaternion_xyzw: torch.Tensor,
) -> torch.Tensor:
    rotation = quaternion_to_matrix(quaternion_xyzw)
    inv_inertia_local = torch.diag_embed(inv_inertia_diag)
    return rotation @ inv_inertia_local @ rotation.transpose(-1, -2)


def yaw_only_quaternion(quat_xyzw):
    if torch.is_tensor(quat_xyzw):
        quat = normalize_quaternion(quat_xyzw)
        x, y, z, w = quat.unbind(dim=-1)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        half_yaw = 0.5 * yaw
        zeros = torch.zeros_like(half_yaw)
        return torch.stack([zeros, zeros, torch.sin(half_yaw), torch.cos(half_yaw)], dim=-1)

    x, y, z, w = normalize_quaternion(quat_xyzw).astype(np.float64)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    half_yaw = 0.5 * yaw
    return np.array([0.0, 0.0, np.sin(half_yaw), np.cos(half_yaw)], dtype=np.float32)


def make_transform(translation, quaternion=IDENTITY_QUAT):
    t = torch.as_tensor(translation, dtype=torch.float32)
    q = normalize_quaternion(torch.as_tensor(quaternion, dtype=torch.float32))
    return (
        (float(t[0]), float(t[1]), float(t[2])),
        (float(q[0]), float(q[1]), float(q[2]), float(q[3])),
    )


def sphere_volume(radius: float) -> float:
    return (4.0 / 3.0) * np.pi * (radius**3)


def compute_shape_density(total_mass: float, shape_radius: float, shape_count: int) -> float:
    if total_mass <= 0.0 or shape_count == 0:
        return 1.0
    total_volume = float(shape_count) * sphere_volume(shape_radius)
    return float(total_mass / max(total_volume, 1e-8))


def transform_points(local_points, translation, quaternion):
    if torch.is_tensor(local_points) or torch.is_tensor(translation) or torch.is_tensor(quaternion):
        device = None
        if torch.is_tensor(local_points):
            device = local_points.device
        elif torch.is_tensor(translation):
            device = translation.device
        elif torch.is_tensor(quaternion):
            device = quaternion.device

        points = torch.as_tensor(local_points, dtype=torch.float32, device=device)
        t = torch.as_tensor(
            translation,
            dtype=points.dtype,
            device=points.device,
        )
        q = torch.as_tensor(quaternion, dtype=points.dtype, device=points.device)
        if points.shape[0] == 0:
            return points.clone()
        rotation = quaternion_to_matrix(q)
        return points @ rotation.transpose(-1, -2) + t.unsqueeze(-2)

    points = np.asarray(local_points, dtype=np.float32)
    if len(points) == 0:
        return points.copy()
    rotation = quaternion_to_matrix(quaternion)
    return (points @ rotation.T + np.asarray(translation, dtype=np.float32)[None, :]).astype(np.float32)
