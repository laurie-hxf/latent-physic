from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp


@wp.kernel
def indexed_dense_relu_kernel(
    input_features: wp.array(dtype=float),
    row_indices: wp.array(dtype=wp.int32),
    params: wp.array(dtype=float),
    weight_offset: int,
    bias_offset: int,
    output: wp.array(dtype=float),
    active_count: int,
    input_dim: int,
    output_dim: int,
):
    tid = wp.tid()
    row = tid // output_dim
    out_col = tid - row * output_dim
    if row >= active_count:
        return

    point_idx = row_indices[row]
    acc = wp.float32(0.0)
    weight_row = weight_offset + out_col * input_dim
    input_row = point_idx * input_dim
    for col in range(input_dim):
        acc = acc + params[weight_row + col] * input_features[input_row + col]
    acc = acc + params[bias_offset + out_col]
    output[tid] = wp.max(acc, wp.float32(0.0))


@wp.kernel
def dense_relu_kernel(
    input_features: wp.array(dtype=float),
    params: wp.array(dtype=float),
    weight_offset: int,
    bias_offset: int,
    output: wp.array(dtype=float),
    active_count: int,
    input_dim: int,
    output_dim: int,
):
    tid = wp.tid()
    row = tid // output_dim
    out_col = tid - row * output_dim
    if row >= active_count:
        return

    acc = wp.float32(0.0)
    weight_row = weight_offset + out_col * input_dim
    input_row = row * input_dim
    for col in range(input_dim):
        acc = acc + params[weight_row + col] * input_features[input_row + col]
    acc = acc + params[bias_offset + out_col]
    output[tid] = wp.max(acc, wp.float32(0.0))


@wp.kernel
def dense_sigmoid_output_kernel(
    input_features: wp.array(dtype=float),
    params: wp.array(dtype=float),
    weight_offset: int,
    bias_offset: int,
    output: wp.array(dtype=float),
    active_count: int,
    input_dim: int,
    min_value: float,
    max_value: float,
):
    row = wp.tid()
    if row >= active_count:
        return

    acc = params[bias_offset]
    input_row = row * input_dim
    for col in range(input_dim):
        acc = acc + params[weight_offset + col] * input_features[input_row + col]
    prob = wp.float32(1.0) / (wp.float32(1.0) + wp.exp(-acc))
    output[row] = min_value + (max_value - min_value) * prob


@wp.kernel
def accumulate_flat_grad_stats_kernel(
    grads: wp.array(dtype=float),
    stats: wp.array(dtype=wp.float64),
):
    tid = wp.tid()
    grad = grads[tid]
    if not wp.isfinite(grad) or grad > 1.0e38 or grad < -1.0e38:
        wp.atomic_add(stats, 3, wp.float64(1.0))
        return

    grad64 = wp.float64(grad)
    abs_grad = wp.abs(grad64)
    wp.atomic_add(stats, 0, grad64 * grad64)
    wp.atomic_add(stats, 1, abs_grad)
    wp.atomic_max(stats, 2, abs_grad)


@wp.kernel
def adam_update_flat_params_kernel(
    params: wp.array(dtype=float),
    grads: wp.array(dtype=float),
    first_moment: wp.array(dtype=wp.float64),
    second_moment: wp.array(dtype=wp.float64),
    adam_step: wp.array(dtype=wp.int32),
    grad_scale: wp.float64,
    learning_rate: wp.float64,
    beta1: wp.float64,
    beta2: wp.float64,
    eps: wp.float64,
):
    tid = wp.tid()
    one = wp.float64(1.0)
    step = adam_step[tid] + 1
    grad = wp.float64(grads[tid]) * grad_scale
    moment_1 = beta1 * first_moment[tid] + (one - beta1) * grad
    moment_2 = beta2 * second_moment[tid] + (one - beta2) * (grad * grad)
    beta1_power = wp.pow(beta1, wp.float64(step))
    beta2_power = wp.pow(beta2, wp.float64(step))
    first_hat = moment_1 / wp.max(one - beta1_power, wp.float64(1.0e-30))
    second_hat = moment_2 / wp.max(one - beta2_power, wp.float64(1.0e-30))
    updated = wp.float64(params[tid]) - learning_rate * first_hat / (wp.sqrt(second_hat) + eps)
    params[tid] = wp.float32(updated)
    first_moment[tid] = moment_1
    second_moment[tid] = moment_2
    adam_step[tid] = step


@dataclass(frozen=True)
class DenseLayerLayout:
    in_dim: int
    out_dim: int
    weight_offset: int
    bias_offset: int

    @property
    def param_count(self) -> int:
        return int(self.in_dim * self.out_dim + self.out_dim)


def positional_encoding_np(points: np.ndarray, num_frequencies: int) -> np.ndarray:
    points_np = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if int(num_frequencies) <= 0:
        return points_np
    parts = [points_np]
    frequencies = (2.0 ** np.arange(int(num_frequencies), dtype=np.float32)).reshape(1, -1, 1)
    values = points_np[:, None, :] * frequencies
    parts.append(np.sin(values).reshape(len(points_np), -1).astype(np.float32))
    parts.append(np.cos(values).reshape(len(points_np), -1).astype(np.float32))
    return np.concatenate(parts, axis=1).astype(np.float32)


def _nearest_indices_chunked(source: np.ndarray, query: np.ndarray, *, chunk_size: int = 1024) -> np.ndarray:
    source_np = np.asarray(source, dtype=np.float32).reshape(-1, 3)
    query_np = np.asarray(query, dtype=np.float32).reshape(-1, 3)
    result = np.empty((len(query_np),), dtype=np.int64)
    for start in range(0, len(query_np), int(chunk_size)):
        end = min(start + int(chunk_size), len(query_np))
        delta = query_np[start:end, None, :] - source_np[None, :, :]
        dist2 = np.sum(delta * delta, axis=-1)
        result[start:end] = np.argmin(dist2, axis=1)
    return result


def align_dino_features_to_surface_points(
    *,
    dino_npz_path: Path,
    local_surface_points: np.ndarray,
    max_match_distance: float = 1.0e-5,
) -> np.ndarray:
    with np.load(dino_npz_path, allow_pickle=True) as data:
        if "dino_features" not in data.files:
            raise ValueError(f"{dino_npz_path} does not contain dino_features.")
        dino_features = np.asarray(data["dino_features"], dtype=np.float32)
        if "local_points" in data.files:
            dino_local_points = np.asarray(data["local_points"], dtype=np.float32).reshape(-1, 3)
        elif "points" in data.files:
            dino_local_points = np.asarray(data["points"], dtype=np.float32).reshape(-1, 3)
        else:
            raise ValueError(f"{dino_npz_path} must contain local_points or points to align features.")

    target = np.asarray(local_surface_points, dtype=np.float32).reshape(-1, 3)
    if dino_local_points.shape == target.shape and np.allclose(dino_local_points, target, atol=max_match_distance):
        return dino_features.astype(np.float32, copy=True)

    nearest = _nearest_indices_chunked(dino_local_points, target)
    matched = dino_local_points[nearest]
    distances = np.linalg.norm(matched - target, axis=1)
    max_distance = float(np.max(distances)) if len(distances) > 0 else 0.0
    if max_distance > float(max_match_distance):
        raise ValueError(
            f"DINO feature points do not align with Newton surface points; "
            f"max nearest local distance={max_distance:.6g} > {float(max_match_distance):.6g}."
        )
    return dino_features[nearest].astype(np.float32, copy=True)


def neighbor_average_features_np(
    points: np.ndarray,
    features: np.ndarray,
    *,
    radius: float,
    k: int,
    chunk_size: int = 512,
) -> np.ndarray:
    points_np = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    features_np = np.asarray(features, dtype=np.float32)
    if features_np.shape[0] != points_np.shape[0]:
        raise ValueError(f"feature/point count mismatch: {features_np.shape[0]} vs {points_np.shape[0]}")
    result = np.empty_like(features_np)
    radius_value = float(radius)
    use_radius = radius_value > 0.0
    radius2 = radius_value * radius_value
    k_value = max(int(k), 1)

    for start in range(0, len(points_np), int(chunk_size)):
        end = min(start + int(chunk_size), len(points_np))
        delta = points_np[start:end, None, :] - points_np[None, :, :]
        dist2 = np.sum(delta * delta, axis=-1)
        if use_radius:
            mask = dist2 <= radius2
            counts = np.sum(mask, axis=1)
            empty = counts == 0
            safe_counts = np.maximum(counts, 1).astype(np.float32)
            averaged = (mask.astype(np.float32) @ features_np) / safe_counts[:, None]
            if np.any(empty):
                nearest = np.argpartition(dist2[empty], kth=min(k_value - 1, len(points_np) - 1), axis=1)[
                    :, :k_value
                ]
                averaged[empty] = np.mean(features_np[nearest], axis=1)
            result[start:end] = averaged.astype(np.float32)
        else:
            nearest = np.argpartition(dist2, kth=min(k_value - 1, len(points_np) - 1), axis=1)[:, :k_value]
            result[start:end] = np.mean(features_np[nearest], axis=1).astype(np.float32)
    return result


def build_dino_mlp_input_features(
    *,
    local_surface_points: np.ndarray,
    half_extents: np.ndarray,
    dino_features: np.ndarray,
    position_frequencies: int,
    neighbor_radius: float,
    neighbor_k: int,
    normalize_dino: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    local_points = np.asarray(local_surface_points, dtype=np.float32).reshape(-1, 3)
    half_extents_np = np.maximum(np.asarray(half_extents, dtype=np.float32).reshape(1, 3), 1.0e-8)
    normalized_points = local_points / half_extents_np
    encoded_position = positional_encoding_np(normalized_points, int(position_frequencies))
    neighbor_dino = neighbor_average_features_np(
        local_points,
        np.asarray(dino_features, dtype=np.float32),
        radius=float(neighbor_radius),
        k=int(neighbor_k),
    )
    stats: dict[str, np.ndarray] = {}
    if normalize_dino:
        mean = neighbor_dino.mean(axis=0, keepdims=True).astype(np.float32)
        std = np.maximum(neighbor_dino.std(axis=0, keepdims=True), 1.0e-6).astype(np.float32)
        neighbor_dino = (neighbor_dino - mean) / std
        stats["dino_mean"] = mean.reshape(-1)
        stats["dino_std"] = std.reshape(-1)
    features = np.concatenate([encoded_position, neighbor_dino], axis=1).astype(np.float32)
    stats["encoded_position_dim"] = np.asarray(encoded_position.shape[1], dtype=np.int32)
    stats["neighbor_dino_dim"] = np.asarray(neighbor_dino.shape[1], dtype=np.int32)
    return features, stats


class WarpDinoMLPFrictionModel:
    def __init__(
        self,
        *,
        input_features: np.ndarray,
        active_capacity: int,
        hidden_dim: int,
        hidden_layers: int,
        initial_mu: float,
        min_mu: float,
        max_mu: float,
        seed: int,
        device: str,
    ) -> None:
        self.input_features_np = np.asarray(input_features, dtype=np.float32)
        if self.input_features_np.ndim != 2:
            raise ValueError(f"input_features must have shape (N,D), got {self.input_features_np.shape}")
        self.point_count = int(self.input_features_np.shape[0])
        self.input_dim = int(self.input_features_np.shape[1])
        self.active_capacity = int(active_capacity)
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = int(hidden_layers)
        if self.active_capacity <= 0:
            raise ValueError("active_capacity must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.hidden_layers <= 0:
            raise ValueError("hidden_layers must be positive")
        self.min_mu = float(min_mu)
        self.max_mu = float(max_mu)
        if self.max_mu <= self.min_mu:
            raise ValueError("max_mu must be greater than min_mu")
        self.device = device
        self.layouts = self._build_layouts()
        params_np = self._initialize_params(float(initial_mu), int(seed))
        self.params = wp.array(params_np, dtype=wp.float32, device=device, requires_grad=True)
        self.first_moment = wp.zeros(len(params_np), dtype=wp.float64, device=device)
        self.second_moment = wp.zeros(len(params_np), dtype=wp.float64, device=device)
        self.adam_step = wp.zeros(len(params_np), dtype=wp.int32, device=device)
        self.input_features = wp.array(self.input_features_np.reshape(-1), dtype=wp.float32, device=device)
        self.hidden_buffers = [
            wp.zeros(self.active_capacity * self.hidden_dim, dtype=wp.float32, device=device, requires_grad=True)
            for _ in range(self.hidden_layers)
        ]

    @property
    def param_count(self) -> int:
        return int(self.params.shape[0])

    def _build_layouts(self) -> list[DenseLayerLayout]:
        layouts: list[DenseLayerLayout] = []
        offset = 0
        in_dim = self.input_dim
        for _ in range(self.hidden_layers):
            weight_offset = offset
            bias_offset = weight_offset + in_dim * self.hidden_dim
            layout = DenseLayerLayout(
                in_dim=in_dim,
                out_dim=self.hidden_dim,
                weight_offset=weight_offset,
                bias_offset=bias_offset,
            )
            layouts.append(layout)
            offset += layout.param_count
            in_dim = self.hidden_dim
        weight_offset = offset
        bias_offset = weight_offset + self.hidden_dim
        layouts.append(
            DenseLayerLayout(
                in_dim=self.hidden_dim,
                out_dim=1,
                weight_offset=weight_offset,
                bias_offset=bias_offset,
            )
        )
        return layouts

    def _initialize_params(self, initial_mu: float, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        params = np.zeros(sum(layout.param_count for layout in self.layouts), dtype=np.float32)
        for layout_idx, layout in enumerate(self.layouts):
            weight_count = layout.in_dim * layout.out_dim
            if layout.out_dim == 1:
                scale = 1.0e-3
            else:
                scale = np.sqrt(2.0 / max(layout.in_dim + layout.out_dim, 1))
            params[layout.weight_offset : layout.weight_offset + weight_count] = rng.normal(
                loc=0.0,
                scale=scale,
                size=weight_count,
            ).astype(np.float32)
            params[layout.bias_offset : layout.bias_offset + layout.out_dim] = 0.0
            if layout_idx == len(self.layouts) - 1:
                probability = (float(initial_mu) - self.min_mu) / max(self.max_mu - self.min_mu, 1.0e-8)
                probability = float(np.clip(probability, 1.0e-4, 1.0 - 1.0e-4))
                params[layout.bias_offset] = np.float32(np.log(probability / (1.0 - probability)))
        return params

    def forward_active(self, active_indices: wp.array, active_count: int, output: wp.array) -> None:
        count = int(active_count)
        if count <= 0:
            return
        if count > self.active_capacity:
            raise ValueError(f"active_count={count} exceeds active_capacity={self.active_capacity}")
        first = self.layouts[0]
        wp.launch(
            indexed_dense_relu_kernel,
            dim=count * self.hidden_dim,
            inputs=[
                self.input_features,
                active_indices,
                self.params,
                int(first.weight_offset),
                int(first.bias_offset),
                self.hidden_buffers[0],
                count,
                int(first.in_dim),
                int(first.out_dim),
            ],
            device=self.device,
        )
        for layer_idx in range(1, self.hidden_layers):
            layout = self.layouts[layer_idx]
            wp.launch(
                dense_relu_kernel,
                dim=count * self.hidden_dim,
                inputs=[
                    self.hidden_buffers[layer_idx - 1],
                    self.params,
                    int(layout.weight_offset),
                    int(layout.bias_offset),
                    self.hidden_buffers[layer_idx],
                    count,
                    int(layout.in_dim),
                    int(layout.out_dim),
                ],
                device=self.device,
            )
        final = self.layouts[-1]
        wp.launch(
            dense_sigmoid_output_kernel,
            dim=count,
            inputs=[
                self.hidden_buffers[self.hidden_layers - 1],
                self.params,
                int(final.weight_offset),
                int(final.bias_offset),
                output,
                count,
                int(final.in_dim),
                float(self.min_mu),
                float(self.max_mu),
            ],
            device=self.device,
        )

    def zero_grad(self) -> None:
        if self.params.grad is not None:
            self.params.grad.zero_()
        for buffer in self.hidden_buffers:
            if buffer.grad is not None:
                buffer.grad.zero_()

    def grad_stats(self) -> tuple[float, float, float, int]:
        if self.params.grad is None:
            return 0.0, 0.0, 0.0, 0
        stats = wp.zeros(4, dtype=wp.float64, device=self.device)
        wp.launch(
            accumulate_flat_grad_stats_kernel,
            dim=self.param_count,
            inputs=[self.params.grad, stats],
            device=self.device,
        )
        values = stats.numpy()
        norm = float(np.sqrt(max(float(values[0]), 0.0)))
        abs_mean = float(values[1]) / max(self.param_count, 1)
        abs_max = float(values[2])
        nonfinite = int(round(float(values[3])))
        return norm, abs_mean, abs_max, nonfinite

    def adam_step_update(
        self,
        *,
        grad_scale: float,
        learning_rate: float,
        beta1: float,
        beta2: float,
        eps: float,
    ) -> None:
        if self.params.grad is None:
            return
        wp.launch(
            adam_update_flat_params_kernel,
            dim=self.param_count,
            inputs=[
                self.params,
                self.params.grad,
                self.first_moment,
                self.second_moment,
                self.adam_step,
                np.float64(grad_scale),
                np.float64(learning_rate),
                np.float64(beta1),
                np.float64(beta2),
                np.float64(eps),
            ],
            device=self.device,
        )

    def params_numpy(self) -> np.ndarray:
        return self.params.numpy().astype(np.float32)

    def assign_params(self, values: np.ndarray) -> None:
        values_np = np.asarray(values, dtype=np.float32)
        if values_np.shape != (self.param_count,):
            raise ValueError(f"Expected params shape {(self.param_count,)}, got {values_np.shape}")
        self.params.assign(values_np)

    def moments_numpy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.first_moment.numpy(), self.second_moment.numpy(), self.adam_step.numpy().astype(np.int32)

    def assign_moments(self, first: np.ndarray, second: np.ndarray, step: np.ndarray) -> None:
        self.first_moment.assign(np.asarray(first, dtype=np.float64))
        self.second_moment.assign(np.asarray(second, dtype=np.float64))
        self.adam_step.assign(np.asarray(step, dtype=np.int32))

    def predict_np(self, indices: np.ndarray, params: np.ndarray | None = None) -> np.ndarray:
        params_np = self.params_numpy() if params is None else np.asarray(params, dtype=np.float32)
        current = self.input_features_np[np.asarray(indices, dtype=np.int64)]
        offset = 0
        for layer_idx in range(self.hidden_layers):
            in_dim = current.shape[1]
            weight_count = in_dim * self.hidden_dim
            weight = params_np[offset : offset + weight_count].reshape(self.hidden_dim, in_dim)
            offset += weight_count
            bias = params_np[offset : offset + self.hidden_dim]
            offset += self.hidden_dim
            current = np.maximum(current @ weight.T + bias[None, :], 0.0)
        weight = params_np[offset : offset + self.hidden_dim].reshape(1, self.hidden_dim)
        offset += self.hidden_dim
        bias = params_np[offset]
        raw = current @ weight.T + bias
        prob = 1.0 / (1.0 + np.exp(-raw))
        return (self.min_mu + (self.max_mu - self.min_mu) * prob.reshape(-1)).astype(np.float32)


def build_warp_dino_mlp_friction_model(
    *,
    dino_npz_path: Path,
    local_surface_points: np.ndarray,
    half_extents: np.ndarray,
    active_capacity: int,
    hidden_dim: int,
    hidden_layers: int,
    initial_mu: float,
    min_mu: float,
    max_mu: float,
    seed: int,
    device: str,
    position_frequencies: int,
    neighbor_radius: float,
    neighbor_k: int,
    normalize_dino: bool,
    max_match_distance: float,
) -> tuple[WarpDinoMLPFrictionModel, dict[str, np.ndarray]]:
    dino_features = align_dino_features_to_surface_points(
        dino_npz_path=Path(dino_npz_path),
        local_surface_points=local_surface_points,
        max_match_distance=float(max_match_distance),
    )
    input_features, input_stats = build_dino_mlp_input_features(
        local_surface_points=local_surface_points,
        half_extents=half_extents,
        dino_features=dino_features,
        position_frequencies=int(position_frequencies),
        neighbor_radius=float(neighbor_radius),
        neighbor_k=int(neighbor_k),
        normalize_dino=bool(normalize_dino),
    )
    model = WarpDinoMLPFrictionModel(
        input_features=input_features,
        active_capacity=int(active_capacity),
        hidden_dim=int(hidden_dim),
        hidden_layers=int(hidden_layers),
        initial_mu=float(initial_mu),
        min_mu=float(min_mu),
        max_mu=float(max_mu),
        seed=int(seed),
        device=str(device),
    )
    metadata: dict[str, np.ndarray] = {
        **input_stats,
        "input_dim": np.asarray(model.input_dim, dtype=np.int32),
        "param_count": np.asarray(model.param_count, dtype=np.int32),
        "hidden_dim": np.asarray(model.hidden_dim, dtype=np.int32),
        "hidden_layers": np.asarray(model.hidden_layers, dtype=np.int32),
        "neighbor_radius": np.asarray(float(neighbor_radius), dtype=np.float32),
        "neighbor_k": np.asarray(int(neighbor_k), dtype=np.int32),
        "position_frequencies": np.asarray(int(position_frequencies), dtype=np.int32),
    }
    return model, metadata
