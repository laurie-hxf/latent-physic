from __future__ import annotations

"""Checkpoint loading and rollout helpers for conditional direct-state GRUs."""

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
NEWTON_DIR = REPO_ROOT / "newton"
for _path in (REPO_ROOT, NEWTON_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from pointnet_residual_adapter.newton_rollout import RigidStateHistory, run_open_loop_rollout
from stateful_gru_residual_adapter.train_conditional_direct_state_gru import (
    ConditionalStatefulGRUDirectStatePredictor,
    _action_feature_sequence,
    _initial_state_sequence,
    _state_tensor,
    normalize_output_mode,
)


DIRECT_STATE_ARCHITECTURE = "stateful_gru_conditional_direct_state_adapter"


@dataclass(frozen=True)
class LoadedConditionalDirectStateCheckpoint:
    path: Path
    metadata: dict[str, Any]
    model: ConditionalStatefulGRUDirectStatePredictor
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    local_surface_points: np.ndarray
    full_point_friction: np.ndarray
    active_contact_mask: np.ndarray


def checkpoint_is_conditional_direct_state(path: Path, *, include_last: bool = False) -> bool:
    if path.suffix != ".pt" or "wandb" in path.parts:
        return False
    if path.name.endswith("_last.pt") and not include_last:
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata", {})
        return (
            "model_state_dict" in payload
            and isinstance(metadata, dict)
            and metadata.get("adapter_architecture") == DIRECT_STATE_ARCHITECTURE
            and "conditional_direct_state_normalizer" in payload
            and "full_point_friction" in payload
        )
    except Exception:
        return False


def load_conditional_direct_state_checkpoint(
    checkpoint_path: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> LoadedConditionalDirectStateCheckpoint:
    path = Path(checkpoint_path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    metadata = dict(payload["metadata"])
    if metadata.get("adapter_architecture") != DIRECT_STATE_ARCHITECTURE:
        raise ValueError(f"{path} is not a conditional direct-state GRU checkpoint")
    normalizer = payload["conditional_direct_state_normalizer"]
    model = ConditionalStatefulGRUDirectStatePredictor(
        input_dim=int(metadata["input_dim"]),
        state_dim=int(metadata["output_dim"]),
        hidden_size=int(metadata.get("gru_hidden_size", 16)),
        num_layers=int(metadata.get("gru_num_layers", 2)),
    )
    model.load_state_dict(payload["model_state_dict"])
    return LoadedConditionalDirectStateCheckpoint(
        path=path,
        metadata=metadata,
        model=model,
        input_mean=np.asarray(normalizer["input_mean"], dtype=np.float32),
        input_std=np.asarray(normalizer["input_std"], dtype=np.float32),
        target_mean=np.asarray(normalizer["target_mean"], dtype=np.float32),
        target_std=np.asarray(normalizer["target_std"], dtype=np.float32),
        local_surface_points=np.asarray(payload["local_surface_points"], dtype=np.float32),
        full_point_friction=np.asarray(payload["full_point_friction"], dtype=np.float32),
        active_contact_mask=np.asarray(payload["active_contact_mask"], dtype=bool),
    )


def _yaw_quaternion(yaw: torch.Tensor) -> torch.Tensor:
    half = 0.5 * yaw
    zeros = torch.zeros_like(half)
    return torch.stack((zeros, zeros, torch.sin(half), torch.cos(half)), dim=-1)


def run_conditional_direct_state_rollout_batch(
    *,
    diff_scene,
    buffers,
    trajectories: list,
    args,
    initial_body_q: np.ndarray,
    initial_body_qd: np.ndarray,
    checkpoint: LoadedConditionalDirectStateCheckpoint,
) -> tuple[RigidStateHistory, dict[str, Any]]:
    base = run_open_loop_rollout(
        diff_scene=diff_scene,
        buffers=buffers,
        trajectories=trajectories,
        args=args,
        initial_body_q=initial_body_q,
        initial_body_qd=initial_body_qd,
    )
    device = diff_scene.torch_device
    max_steps = max(int(trajectory.num_steps) for trajectory in trajectories)
    batch_size = len(trajectories)

    base_positions = torch.as_tensor(base.positions, dtype=torch.float32, device=device)
    base_quaternions = torch.as_tensor(base.quaternions_xyzw, dtype=torch.float32, device=device)
    base_linear = torch.as_tensor(base.linear_velocity, dtype=torch.float32, device=device)
    base_angular = torch.as_tensor(base.angular_velocity, dtype=torch.float32, device=device)
    base_next_state = _state_tensor(
        positions=base_positions[:, 1 : max_steps + 1],
        quaternions_xyzw=base_quaternions[:, 1 : max_steps + 1],
        linear_velocity=base_linear[:, 1 : max_steps + 1],
        angular_velocity=base_angular[:, 1 : max_steps + 1],
    )
    initial_state = _initial_state_sequence(trajectories, max_steps=max_steps, device=device)
    actions = _action_feature_sequence(
        trajectories=trajectories,
        base_quaternions_xyzw=base_quaternions[:, :max_steps],
        max_steps=max_steps,
        device=device,
    )
    inputs = torch.cat((initial_state, actions, base_next_state), dim=-1).contiguous()

    input_mean = torch.as_tensor(checkpoint.input_mean, dtype=torch.float32, device=device).reshape(1, -1)
    input_std = torch.as_tensor(checkpoint.input_std, dtype=torch.float32, device=device).reshape(1, -1)
    target_mean = torch.as_tensor(checkpoint.target_mean, dtype=torch.float32, device=device).reshape(1, -1)
    target_std = torch.as_tensor(checkpoint.target_std, dtype=torch.float32, device=device).reshape(1, -1)
    model = checkpoint.model.to(device)
    model.eval()

    predicted_steps: list[torch.Tensor] = []
    hidden_norms = torch.empty((batch_size, max_steps), dtype=torch.float32, device=device)
    hidden_saturation = torch.empty((batch_size, max_steps), dtype=torch.float32, device=device)
    hidden = model.initial_state(batch_size, device=device, dtype=torch.float32)
    with torch.no_grad():
        for step_idx in range(max_steps):
            normalized = (inputs[:, step_idx] - input_mean) / input_std
            predicted_normalized, hidden = model.forward_step(normalized, hidden)
            predicted_steps.append(predicted_normalized * target_std + target_mean)
            hidden_norms[:, step_idx].copy_(torch.linalg.vector_norm(hidden, dim=(0, 2)))
            hidden_saturation[:, step_idx].copy_(
                (hidden.abs() >= 0.95).to(dtype=torch.float32).mean(dim=(0, 2))
            )
    predicted = torch.stack(predicted_steps, dim=1)

    positions = base_positions.clone()
    quaternions = base_quaternions.clone()
    linear_velocity = base_linear.clone()
    angular_velocity = base_angular.clone()
    positions[:, 1 : max_steps + 1, :2].copy_(predicted[..., :2])
    quaternions[:, 1 : max_steps + 1].copy_(_yaw_quaternion(predicted[..., 2]))
    output_mode = normalize_output_mode(str(checkpoint.metadata["output_mode"]))
    if output_mode == "position_velocity":
        linear_velocity[:, 1 : max_steps + 1, :2].copy_(predicted[..., 3:5])
        angular_velocity[:, 1 : max_steps + 1, 2].copy_(predicted[..., 5])

    history = RigidStateHistory(
        positions=positions.detach().cpu().numpy().astype(np.float32, copy=False),
        quaternions_xyzw=quaternions.detach().cpu().numpy().astype(np.float32, copy=False),
        linear_velocity=linear_velocity.detach().cpu().numpy().astype(np.float32, copy=False),
        angular_velocity=angular_velocity.detach().cpu().numpy().astype(np.float32, copy=False),
    )
    diagnostics = {
        "hidden_l2_norm": hidden_norms.detach().cpu().numpy().astype(np.float32, copy=False),
        "hidden_saturation_fraction": hidden_saturation.detach().cpu().numpy().astype(np.float32, copy=False),
        "stateful_reset_interval": 0,
    }
    return history, diagnostics
