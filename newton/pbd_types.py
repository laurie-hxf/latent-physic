from __future__ import annotations

from dataclasses import dataclass, field

import newton
import numpy as np
import torch


DEFAULT_TABLE_SEG_ID = 12
DEFAULT_TEE_SEG_ID = 14
DEFAULT_EE_SEG_ID = 10

DEFAULT_TABLE_VOXEL = 0.03
DEFAULT_TEE_VOXEL = 0.01
DEFAULT_EE_VOXEL = 0.01
DEFAULT_TEE_RADIUS_SCALE = 1.00
DEFAULT_EE_RADIUS_SCALE = 1.00
DEFAULT_TEE_MASS = 0.8
DEFAULT_EE_MASS = 8.0
DEFAULT_SUBSTEPS = 4
DEFAULT_VELOCITY_DAMPING = 0.85
DEFAULT_MAX_VELOCITY = 2.0
DEFAULT_TABLE_FRICTION = 1.0
DEFAULT_OBJECT_FRICTION = 1.0
DEFAULT_CONTACT_STIFFNESS = 2.0e4
DEFAULT_CONTACT_DAMPING = 50.0
DEFAULT_CONTACT_MARGIN = 1.0e-3
DEFAULT_FRICTION_REGULARIZATION = 1.0e-3

CLUSTER_COLORS: dict[str, tuple[float, float, float]] = {
    "table": (0.52, 0.57, 0.66),
    "tee": (0.88, 0.39, 0.18),
    "end_effector": (0.18, 0.67, 0.50),
}

IDENTITY_QUAT = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)


@dataclass
class PlyHeader:
    vertex_count: int
    properties: list[str]
    data_start_line: int

    @property
    def index(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.properties)}


@dataclass
class SegmentConfig:
    name: str
    segmentation_id: int
    voxel_size: float
    total_mass: float | torch.Tensor
    is_dynamic: bool
    control_mode: str
    planar_motion: bool
    fill_interior: bool
    display_color: tuple[float, float, float]
    shape_radius_scale: float

    @property
    def shape_radius(self) -> float:
        return self.voxel_size * self.shape_radius_scale


@dataclass
class VoxelBucket:
    count: int = 0
    sum_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def update(self, xyz) -> None:
        self.count += 1
        self.sum_xyz += np.asarray(xyz, dtype=np.float64)

    @property
    def centroid(self) -> np.ndarray:
        if self.count == 0:
            raise ValueError("VoxelBucket is empty")
        return (self.sum_xyz / self.count).astype(np.float32)


@dataclass
class SceneState:
    body_q: torch.Tensor
    body_qd: torch.Tensor

    def clone(self) -> SceneState:
        return SceneState(body_q=self.body_q.clone(), body_qd=self.body_qd.clone())


@dataclass
class RigidBodyCluster:
    name: str
    segmentation_id: int
    body_id: int
    local_shape_positions: torch.Tensor
    shape_radius: torch.Tensor
    total_mass: torch.Tensor
    rest_translation: torch.Tensor
    fixed_orientation: torch.Tensor
    is_dynamic: bool
    planar_motion: bool
    display_color: tuple[float, float, float]
    control_mode: str = "free"
    collision_geometry: str = "sphere_cluster"
    collision_shape_start: int = 0
    collision_shape_count: int = 0
    box_half_extents: torch.Tensor | None = None
    inertia_factor_diag: torch.Tensor = field(
        default_factory=lambda: torch.ones(3, dtype=torch.float32)
    )
    support_radius: torch.Tensor = field(
        default_factory=lambda: torch.tensor(0.0, dtype=torch.float32)
    )

    @property
    def shape_count(self) -> int:
        return int(self.local_shape_positions.shape[0])

    @property
    def num_collision_shapes(self) -> int:
        return self.collision_shape_count if self.collision_shape_count > 0 else self.shape_count

    @property
    def effective_mass(self) -> torch.Tensor:
        if not self.is_dynamic:
            return torch.zeros((), device=self.total_mass.device, dtype=self.total_mass.dtype)
        return self.total_mass.clamp_min(1e-6)

    @property
    def inv_mass(self) -> torch.Tensor:
        if not self.is_dynamic:
            return torch.zeros((), device=self.total_mass.device, dtype=self.total_mass.dtype)
        return torch.reciprocal(self.effective_mass)

    @property
    def inertia_diag(self) -> torch.Tensor:
        if not self.is_dynamic:
            return torch.ones(3, device=self.total_mass.device, dtype=self.total_mass.dtype)
        return (self.effective_mass * self.inertia_factor_diag).clamp_min(1e-6)

    @property
    def inv_inertia_diag(self) -> torch.Tensor:
        if not self.is_dynamic:
            return torch.zeros(3, device=self.total_mass.device, dtype=self.total_mass.dtype)
        return torch.reciprocal(self.inertia_diag)

    @property
    def shape_mass(self) -> float:
        if self.shape_count == 0:
            return 0.0
        value = self.total_mass / float(self.shape_count)
        return float(value.detach().cpu())


@dataclass
class BuiltScene:
    state_0: SceneState
    state_1: SceneState
    clusters: list[RigidBodyCluster]
    cluster_target_translations: dict[str, torch.Tensor]
    cluster_command_velocities: dict[str, torch.Tensor]
    constraint_iterations: int
    table_friction: torch.Tensor
    object_friction: torch.Tensor
    contact_stiffness: torch.Tensor
    contact_damping: torch.Tensor
    contact_margin: torch.Tensor
    friction_regularization: torch.Tensor
    gravity: torch.Tensor
    device: torch.device
    dtype: torch.dtype
    collision_model: newton.Model | None = None
    collision_state: newton.State | None = None
    collision_pipeline: newton.CollisionPipeline | None = None
    collision_contacts: newton.Contacts | None = None
