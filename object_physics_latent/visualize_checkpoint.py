from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colormaps
from matplotlib.colors import Normalize, TwoSlopeNorm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from object_physics_latent.dataset import (  # noqa: E402
    ENCODER_FEATURE_SCHEMA,
    EncoderFeatureBatch,
    ObjectPhysicsDataset,
    ObjectSpec,
)
from object_physics_latent.friction_decoder import build_point_conditioning_features  # noqa: E402
from object_physics_latent.model import TrajectoryConditionedFrictionModel  # noqa: E402


FAMILY_COLORS = {
    "left_right": "#2563eb",
    "front_back": "#dc2626",
    "center_ends": "#16a34a",
}
SPLIT_MARKERS = {
    "train": "o",
    "validation": "^",
    "test": "s",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize per-object physics latents and predicted bottom-surface friction "
            "from an existing object_physics_latent checkpoint. This does not run Newton rollout."
        )
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=("train", "validation", "test"),
    )
    parser.add_argument("--context-trajectories-per-view", type=int, default=None)
    parser.add_argument("--context-window-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-cache-size", type=int, default=1)
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--mu-color-min", type=float, default=0.0)
    parser.add_argument("--mu-color-max", type=float, default=0.8)
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def resolve_path(value: str | Path, *, relative_to: Path = REPO_ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (relative_to / path).resolve()


def stack_encoder_batch(batch: EncoderFeatureBatch, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(batch.features[None], dtype=torch.float32, device=device),
        torch.as_tensor(batch.valid_mask[None], dtype=torch.bool, device=device),
    )


def build_model_from_checkpoint(payload: dict[str, Any], *, device: torch.device) -> TrajectoryConditionedFrictionModel:
    saved_args = dict(payload["args"])
    feature_metadata = dict(payload["feature_metadata"])
    point_feature_dim = int(feature_metadata.get("decoder_input_dim", feature_metadata["input_dim"]))
    visual_feature_dim = int(feature_metadata.get("visual_input_dim", 0))
    model = TrajectoryConditionedFrictionModel.from_dimensions(
        point_feature_dim=point_feature_dim,
        encoder_feature_dim=len(ENCODER_FEATURE_SCHEMA),
        latent_dim=int(saved_args["latent_dim"]),
        projection_dim=int(saved_args["projection_dim"]),
        step_hidden_dim=int(saved_args["step_hidden_dim"]),
        gru_hidden_dim=int(saved_args["gru_hidden_dim"]),
        trajectory_embedding_dim=int(saved_args["trajectory_embedding_dim"]),
        set_hidden_dim=int(saved_args["set_hidden_dim"]),
        visual_feature_dim=visual_feature_dim,
        visual_hidden_dim=int(saved_args.get("visual_hidden_dim", 128)),
        visual_embedding_dim=int(saved_args.get("visual_embedding_dim", 128)),
        visual_point_hidden_layers=int(saved_args.get("visual_point_hidden_layers", 1)),
        decoder_hidden_dim=int(saved_args["decoder_hidden_dim"]),
        decoder_hidden_layers=int(saved_args["decoder_hidden_layers"]),
        decoder_conditioning=str(saved_args.get("decoder_conditioning", "concat")),
        decoder_activation=str(saved_args.get("decoder_activation", "relu")),
        decoder_basis_count=int(saved_args.get("decoder_basis_count", 8)),
        decoder_basis_base_mode=str(saved_args.get("decoder_basis_base_mode", "latent")),
        decoder_basis_normalization=str(saved_args.get("decoder_basis_normalization", "zero_mean")),
        decoder_basis_activation=str(saved_args.get("decoder_basis_activation", "tanh")),
        decoder_basis_norm_eps=float(saved_args.get("decoder_basis_norm_eps", 1.0e-4)),
        decoder_latent_normalization=str(saved_args.get("decoder_latent_normalization", "none")),
        decoder_raw_limit=(
            None
            if saved_args.get("decoder_raw_limit") is None
            else float(saved_args.get("decoder_raw_limit"))
        ),
        mu_min=float(saved_args["min_point_friction"]),
        mu_max=float(saved_args["max_point_friction"]),
        initial_mu=float(saved_args["point_friction"]),
    )
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    return model


def object_model_features(
    obj: ObjectSpec,
    *,
    local_surface_points: np.ndarray,
    saved_args: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    use_dino = not bool(saved_args.get("no_dino", False))
    if use_dino and obj.dino_feature_npz is None:
        raise ValueError(f"{obj.object_id} does not have DINO features in the manifest")
    features, metadata, _ = build_point_conditioning_features(
        local_surface_points=local_surface_points,
        half_extents=np.asarray(saved_args["box_half_extents"], dtype=np.float32),
        dino_npz_path=obj.dino_feature_npz if use_dino else None,
        position_frequencies=int(saved_args["dino_position_frequencies"]),
        neighbor_radius=float(saved_args["dino_neighbor_radius"]),
        neighbor_k=int(saved_args["dino_neighbor_k"]),
        normalize_dino=bool(saved_args["dino_feature_normalization"]),
        max_match_distance=float(saved_args["dino_mlp_max_match_distance"]),
    )
    feature_mode = str(saved_args.get("decoder_point_feature_mode", "full"))
    if feature_mode == "position":
        decoder_features = features[:, : int(metadata.encoded_position_dim)]
    elif feature_mode == "full":
        decoder_features = features
    else:
        raise ValueError(f"Unknown decoder_point_feature_mode: {feature_mode!r}")
    visual_features = (
        features
        if bool(saved_args.get("dino_to_encoder", False)) and int(metadata.dino_dim) > 0
        else None
    )
    return (
        torch.as_tensor(decoder_features, dtype=torch.float32, device=device),
        None
        if visual_features is None
        else torch.as_tensor(visual_features[None], dtype=torch.float32, device=device),
    )


def friction_value(block_friction: dict[str, Any], geom_name: str) -> float:
    values = block_friction.get(geom_name)
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"Ground-truth friction metadata is missing {geom_name!r}")
    return float(values[0])


def ground_truth_bottom_friction(obj: ObjectSpec, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    family = str(obj.friction_spec.get("partition_family"))
    block_friction = dict(obj.friction_spec.get("block_friction", {}))
    x = np.asarray(points[:, 0], dtype=np.float64)
    y = np.asarray(points[:, 1], dtype=np.float64)

    if family == "left_right":
        region_names = ("left", "right")
        labels = np.where(x < 0.0, "left", "right")
        region_mu = {
            "left": friction_value(block_friction, "push_block_left"),
            "right": friction_value(block_friction, "push_block_right"),
        }
    elif family == "front_back":
        region_names = ("back", "front")
        labels = np.where(y < 0.0, "back", "front")
        region_mu = {
            "back": friction_value(block_friction, "push_block_back"),
            "front": friction_value(block_friction, "push_block_front"),
        }
    elif family == "center_ends":
        region_names = ("ends", "center")
        labels = np.where(np.abs(x) < 0.05, "center", "ends")
        region_mu = {
            "ends": friction_value(block_friction, "push_block_left_end"),
            "center": friction_value(block_friction, "push_block_center"),
        }
    else:
        raise ValueError(f"Unsupported partition family for {obj.object_id}: {family!r}")

    target = np.asarray([region_mu[str(label)] for label in labels], dtype=np.float32)
    return target, labels.astype(str), region_names


def safe_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    if first.size < 2 or second.size != first.size:
        return None
    if float(np.std(first)) <= 1.0e-12 or float(np.std(second)) <= 1.0e-12:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def region_means(values: np.ndarray, labels: np.ndarray, names: Sequence[str]) -> dict[str, float]:
    return {
        str(name): float(np.mean(np.asarray(values)[np.asarray(labels) == str(name)]))
        for name in names
    }


def evaluate_object(
    *,
    obj: ObjectSpec,
    model: TrajectoryConditionedFrictionModel,
    dataset: ObjectPhysicsDataset,
    point_features: torch.Tensor,
    visual_features: torch.Tensor | None,
    local_surface_points: np.ndarray,
    active_indices: np.ndarray,
    context_trajectories: int,
    context_window_steps: int,
    rng: np.random.Generator,
    device: torch.device,
) -> dict[str, Any]:
    sample = dataset.sample_object_training_data(
        obj.object_id,
        context_trajectories_per_view=int(context_trajectories),
        query_trajectories_per_view=1,
        context_window_steps=int(context_window_steps),
        query_window_steps=1,
        random_context_windows=False,
        random_query_windows=False,
        rng=rng,
    )

    latents: list[np.ndarray] = []
    projections: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    active_indices_t = torch.as_tensor(active_indices, dtype=torch.long, device=device)
    with torch.no_grad():
        for context_batch in (sample.context_a, sample.context_b):
            context, mask = stack_encoder_batch(context_batch, device=device)
            latent_output = model.encode_context(
                context,
                context_valid_mask=mask,
                visual_features=visual_features,
            )
            predicted = model.decode_friction(
                point_features,
                latent_output.latent,
                active_indices=active_indices_t,
            )
            latents.append(latent_output.latent[0].detach().cpu().numpy().astype(np.float32))
            projections.append(latent_output.projection[0].detach().cpu().numpy().astype(np.float32))
            predictions.append(predicted[0].detach().cpu().numpy().astype(np.float32))

    latent_views = np.stack(latents, axis=0)
    projection_views = np.stack(projections, axis=0)
    prediction_views = np.stack(predictions, axis=0)
    predicted_mu = np.mean(prediction_views, axis=0)
    active_points = local_surface_points[active_indices]
    target_mu, labels, names = ground_truth_bottom_friction(obj, active_points)
    errors = predicted_mu - target_mu
    target_regions = region_means(target_mu, labels, names)
    predicted_regions = region_means(predicted_mu, labels, names)
    first_name, second_name = names
    target_contrast = target_regions[second_name] - target_regions[first_name]
    predicted_contrast = predicted_regions[second_name] - predicted_regions[first_name]

    return {
        "object_id": obj.object_id,
        "object_split": obj.object_split,
        "partition_family": str(obj.friction_spec.get("partition_family")),
        "friction_spec": obj.friction_spec,
        "latent_views": latent_views,
        "latent_mean": np.mean(latent_views, axis=0),
        "projection_views": projection_views,
        "predicted_mu_views": prediction_views,
        "predicted_mu": predicted_mu,
        "target_mu": target_mu,
        "region_labels": labels,
        "region_order": tuple(names),
        "target_region_means": target_regions,
        "predicted_region_means": predicted_regions,
        "metrics": {
            "point_mae": float(np.mean(np.abs(errors))),
            "point_rmse": float(np.sqrt(np.mean(errors**2))),
            "point_bias": float(np.mean(errors)),
            "point_correlation": safe_correlation(target_mu, predicted_mu),
            "predicted_mu_mean": float(np.mean(predicted_mu)),
            "predicted_mu_std": float(np.std(predicted_mu)),
            "target_mu_mean": float(np.mean(target_mu)),
            "target_mu_std": float(np.std(target_mu)),
            "region_mean_mae": float(
                np.mean([abs(predicted_regions[name] - target_regions[name]) for name in names])
            ),
            "target_contrast": float(target_contrast),
            "predicted_contrast": float(predicted_contrast),
            "contrast_error": float(predicted_contrast - target_contrast),
            "contrast_direction_correct": bool(
                np.sign(predicted_contrast) == np.sign(target_contrast) and abs(predicted_contrast) > 1.0e-6
            ),
            "latent_view_distance": float(np.linalg.norm(latent_views[0] - latent_views[1])),
            "prediction_view_mae": float(np.mean(np.abs(prediction_views[0] - prediction_views[1]))),
        },
    }


def short_object_id(object_id: str) -> str:
    parts = str(object_id).split("_")
    return parts[1] if len(parts) > 1 and parts[0] == "object" else str(object_id)


def render_object_comparison(
    record: dict[str, Any],
    *,
    active_points: np.ndarray,
    output_path: Path,
    mu_min: float,
    mu_max: float,
) -> None:
    target = np.asarray(record["target_mu"], dtype=np.float64)
    predicted = np.asarray(record["predicted_mu"], dtype=np.float64)
    error = predicted - target
    error_limit = max(float(np.max(np.abs(error))), 0.05)
    cmap = colormaps["turbo"]
    norm = Normalize(vmin=float(mu_min), vmax=float(mu_max))
    error_norm = TwoSlopeNorm(vmin=-error_limit, vcenter=0.0, vmax=error_limit)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    for ax, values, title in (
        (axes[0, 0], target, "Ground truth bottom friction"),
        (axes[0, 1], predicted, "Predicted bottom friction (mean of 2 context views)"),
    ):
        scatter = ax.scatter(
            active_points[:, 0],
            active_points[:, 1],
            c=values,
            cmap=cmap,
            norm=norm,
            marker="s",
            s=95,
            linewidths=0.0,
        )
        ax.set_title(title)
        ax.set_xlabel("local x (m)")
        ax.set_ylabel("local y (m)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
    fig.colorbar(scatter, ax=(axes[0, 0], axes[0, 1]), label="friction coefficient", shrink=0.92)

    signed_error = axes[1, 0].scatter(
        active_points[:, 0],
        active_points[:, 1],
        c=error,
        cmap=colormaps["coolwarm"],
        norm=error_norm,
        marker="s",
        s=95,
        linewidths=0.0,
    )
    axes[1, 0].set_title("Prediction - ground truth")
    axes[1, 0].set_xlabel("local x (m)")
    axes[1, 0].set_ylabel("local y (m)")
    axes[1, 0].set_aspect("equal")
    axes[1, 0].grid(alpha=0.15)
    fig.colorbar(signed_error, ax=axes[1, 0], label="signed friction error", shrink=0.92)

    ax = axes[1, 1]
    names = list(record["region_order"])
    target_regions = record["target_region_means"]
    predicted_regions = record["predicted_region_means"]
    positions = np.arange(len(names), dtype=np.float64)
    width = 0.36
    ax.bar(positions - width / 2.0, [target_regions[name] for name in names], width, label="ground truth")
    ax.bar(positions + width / 2.0, [predicted_regions[name] for name in names], width, label="predicted")
    ax.set_xticks(positions, names)
    ax.set_ylim(float(mu_min), float(mu_max))
    ax.set_ylabel("region mean friction")
    ax.set_title("Region means")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    metrics = record["metrics"]
    ax.text(
        0.02,
        0.98,
        "\n".join(
            [
                f"point MAE={metrics['point_mae']:.4f}",
                f"point RMSE={metrics['point_rmse']:.4f}",
                f"region MAE={metrics['region_mean_mae']:.4f}",
                f"target contrast={metrics['target_contrast']:+.4f}",
                f"pred contrast={metrics['predicted_contrast']:+.4f}",
                f"direction correct={metrics['contrast_direction_correct']}",
                f"latent view dist={metrics['latent_view_distance']:.4f}",
            ]
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
    )
    fig.suptitle(
        f"{record['object_id']}\n"
        f"split={record['object_split']} | family={record['partition_family']}",
        fontsize=12,
        weight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=145, facecolor="white")
    plt.close(fig)


def pca_project(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    centered = values - np.mean(values, axis=0, keepdims=True)
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    components = right_vectors[:2]
    projected = centered @ components.T
    variances = singular_values**2
    explained = variances[:2] / max(float(np.sum(variances)), 1.0e-12)
    return projected.astype(np.float32), components.astype(np.float32), explained.astype(np.float32)


def render_latent_pca(records: Sequence[dict[str, Any]], output_path: Path) -> dict[str, Any]:
    view_latents = np.stack([record["latent_views"] for record in records], axis=0)
    flat_views = view_latents.reshape(-1, view_latents.shape[-1])
    flat_projected, components, explained = pca_project(flat_views)
    projected = flat_projected.reshape(len(records), 2, 2)
    means = np.mean(projected, axis=1)

    fig, ax = plt.subplots(figsize=(11.5, 8.5), constrained_layout=True)
    for index, record in enumerate(records):
        color = FAMILY_COLORS.get(record["partition_family"], "#6b7280")
        marker = SPLIT_MARKERS.get(record["object_split"], "o")
        ax.plot(projected[index, :, 0], projected[index, :, 1], color=color, alpha=0.28, linewidth=0.9)
        ax.scatter(
            means[index, 0],
            means[index, 1],
            color=color,
            marker=marker,
            s=55,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        ax.text(means[index, 0], means[index, 1], short_object_id(record["object_id"]), fontsize=6.5)

    family_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color, label=family, markersize=7)
        for family, color in FAMILY_COLORS.items()
    ]
    split_handles = [
        plt.Line2D([0], [0], marker=marker, color="#374151", linestyle="none", label=split, markersize=7)
        for split, marker in SPLIT_MARKERS.items()
    ]
    first_legend = ax.legend(handles=family_handles, title="partition family", loc="upper left")
    ax.add_artist(first_legend)
    ax.legend(handles=split_handles, title="object split", loc="upper right")
    ax.set_xlabel(f"PC1 ({100.0 * explained[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({100.0 * explained[1]:.1f}% variance)")
    ax.set_title("Per-object physics latent PCA\nlabels show object index; line joins two disjoint context views")
    ax.grid(alpha=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=165, facecolor="white")
    plt.close(fig)
    return {
        "components": components,
        "explained_variance_ratio": explained,
        "view_coordinates": projected,
        "mean_coordinates": means,
    }


def render_latent_heatmap(records: Sequence[dict[str, Any]], output_path: Path) -> None:
    values = np.stack([record["latent_mean"] for record in records], axis=0)
    limit = max(float(np.max(np.abs(values))), 1.0e-3)
    fig_height = max(8.0, 0.27 * len(records))
    fig, ax = plt.subplots(figsize=(9.0, fig_height), constrained_layout=True)
    image = ax.imshow(values, aspect="auto", cmap="coolwarm", norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit))
    ax.set_xticks(np.arange(values.shape[1]), [f"z{i}" for i in range(values.shape[1])])
    ax.set_yticks(
        np.arange(len(records)),
        [
            f"{short_object_id(record['object_id'])} {record['partition_family']} {record['object_split']}"
            for record in records
        ],
        fontsize=6.5,
    )
    ax.set_title("Mean object latent vectors")
    fig.colorbar(image, ax=ax, label="latent value", shrink=0.75)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=165, facecolor="white")
    plt.close(fig)


def pairwise_distances(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    differences = values[:, None, :] - values[None, :, :]
    return np.sqrt(np.sum(differences**2, axis=-1))


def render_latent_distance_matrix(records: Sequence[dict[str, Any]], output_path: Path) -> np.ndarray:
    values = np.stack([record["latent_mean"] for record in records], axis=0)
    distances = pairwise_distances(values)
    labels = [short_object_id(record["object_id"]) for record in records]
    fig, ax = plt.subplots(figsize=(10.5, 9.5), constrained_layout=True)
    image = ax.imshow(distances, cmap="viridis", aspect="equal")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=90, fontsize=5.5)
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=5.5)
    ax.set_title("Pairwise Euclidean distance between mean object latents")
    fig.colorbar(image, ax=ax, label="latent distance", shrink=0.82)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=165, facecolor="white")
    plt.close(fig)
    return distances


def aggregate_metrics(records: Sequence[dict[str, Any]], *, latent_distances: np.ndarray) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups["all"].append(record)
        groups[f"split/{record['object_split']}"].append(record)
        groups[f"family/{record['partition_family']}"].append(record)

    summaries: dict[str, Any] = {}
    for name, items in groups.items():
        summaries[name] = {
            "object_count": len(items),
            "point_mae_mean": float(np.mean([item["metrics"]["point_mae"] for item in items])),
            "point_rmse_mean": float(np.mean([item["metrics"]["point_rmse"] for item in items])),
            "region_mean_mae_mean": float(np.mean([item["metrics"]["region_mean_mae"] for item in items])),
            "contrast_direction_accuracy": float(
                np.mean([item["metrics"]["contrast_direction_correct"] for item in items])
            ),
            "latent_view_distance_mean": float(
                np.mean([item["metrics"]["latent_view_distance"] for item in items])
            ),
            "prediction_view_mae_mean": float(
                np.mean([item["metrics"]["prediction_view_mae"] for item in items])
            ),
        }

    target_maps = np.stack([record["target_mu"] for record in records], axis=0)
    target_distances = pairwise_distances(target_maps)
    upper = np.triu_indices(len(records), k=1)
    nearest_indices = np.argsort(latent_distances, axis=1)[:, 1]
    nearest_same_family = [
        records[index]["partition_family"] == records[int(neighbor)]["partition_family"]
        for index, neighbor in enumerate(nearest_indices)
    ]
    summaries["latent_structure"] = {
        "different_object_distance_mean": float(np.mean(latent_distances[upper])),
        "different_object_distance_min": float(np.min(latent_distances[upper])),
        "target_map_distance_correlation": safe_correlation(latent_distances[upper], target_distances[upper]),
        "nearest_neighbor_same_family_rate": float(np.mean(nearest_same_family)),
    }
    return summaries


def write_csv(records: Sequence[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "object_id",
        "object_split",
        "partition_family",
        "point_mae",
        "point_rmse",
        "point_bias",
        "point_correlation",
        "region_mean_mae",
        "target_contrast",
        "predicted_contrast",
        "contrast_error",
        "contrast_direction_correct",
        "predicted_mu_mean",
        "predicted_mu_std",
        "target_mu_mean",
        "target_mu_std",
        "latent_view_distance",
        "prediction_view_mae",
        "latent_mean",
        "target_region_means",
        "predicted_region_means",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                "object_id": record["object_id"],
                "object_split": record["object_split"],
                "partition_family": record["partition_family"],
                **record["metrics"],
                "latent_mean": json.dumps(jsonable(record["latent_mean"])),
                "target_region_means": json.dumps(record["target_region_means"], sort_keys=True),
                "predicted_region_means": json.dumps(record["predicted_region_means"], sort_keys=True),
            }
            writer.writerow({name: row.get(name) for name in fieldnames})


def create_contact_sheet(image_paths: Sequence[Path], output_path: Path, *, columns: int = 4) -> None:
    from PIL import Image, ImageDraw

    if not image_paths:
        return
    images = [Image.open(path).convert("RGB") for path in image_paths]
    thumb_width = 420
    thumb_height = int(images[0].height * thumb_width / images[0].width)
    rows = int(np.ceil(len(images) / int(columns)))
    header_height = 55
    sheet = Image.new("RGB", (int(columns) * thumb_width, header_height + rows * thumb_height), "#f6f7f9")
    draw = ImageDraw.Draw(sheet)
    draw.text((18, 18), "Object physics latent: predicted friction vs ground truth", fill="#111827")
    for index, image in enumerate(images):
        image.thumbnail((thumb_width, thumb_height))
        x = (index % int(columns)) * thumb_width
        y = header_height + (index // int(columns)) * thumb_height
        sheet.paste(image, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    for image in images:
        image.close()


def write_gallery(
    records: Sequence[dict[str, Any]],
    *,
    output_path: Path,
    checkpoint: Path,
    aggregate: dict[str, Any],
) -> None:
    cards = []
    for record in records:
        image_name = f"per_object/{record['object_id']}.png"
        metrics = record["metrics"]
        cards.append(
            f"""
            <article class="card">
              <a href="{html.escape(image_name)}"><img src="{html.escape(image_name)}" loading="lazy"></a>
              <h3>{html.escape(record['object_id'])}</h3>
              <p>split=<b>{html.escape(record['object_split'])}</b> |
                 family=<b>{html.escape(record['partition_family'])}</b></p>
              <p>point MAE={metrics['point_mae']:.4f} |
                 region MAE={metrics['region_mean_mae']:.4f} |
                 direction={metrics['contrast_direction_correct']}</p>
            </article>
            """
        )
    all_metrics = aggregate["all"]
    latent_metrics = aggregate["latent_structure"]
    document = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Object physics latent checkpoint visualization</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #f3f4f6; color: #111827; }}
    .links a {{ margin-right: 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 18px; }}
    .card {{ background: white; padding: 12px; border-radius: 10px; box-shadow: 0 1px 5px #0002; }}
    img {{ width: 100%; height: auto; }}
    code {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <h1>Object physics latent checkpoint visualization</h1>
  <p>Checkpoint: <code>{html.escape(str(checkpoint))}</code></p>
  <p>Objects={len(records)} | point MAE={all_metrics['point_mae_mean']:.4f} |
     region MAE={all_metrics['region_mean_mae_mean']:.4f} |
     contrast direction accuracy={all_metrics['contrast_direction_accuracy']:.1%} |
     latent-to-target-map distance correlation={latent_metrics['target_map_distance_correlation']}</p>
  <p class="links">
    <a href="latent_pca.png">latent PCA</a>
    <a href="latent_vectors_heatmap.png">latent vectors</a>
    <a href="latent_distance_matrix.png">latent distance matrix</a>
    <a href="friction_contact_sheet.png">friction contact sheet</a>
    <a href="object_metrics.csv">metrics CSV</a>
    <a href="summary.json">summary JSON</a>
  </p>
  <div class="grid">{''.join(cards)}</div>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is unavailable")

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    saved_args = dict(payload["args"])
    manifest = (
        resolve_path(args.manifest)
        if args.manifest is not None
        else resolve_path(saved_args["manifest"])
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else checkpoint.parent / "visualization" / checkpoint.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model_from_checkpoint(payload, device=device)
    local_surface_points = np.asarray(payload["local_surface_points"], dtype=np.float32)
    active_indices = np.asarray(payload["active_indices"], dtype=np.int32)
    active_points = local_surface_points[active_indices]
    dataset = ObjectPhysicsDataset(manifest, cache_size=int(args.dataset_cache_size))
    selected_ids = [
        object_id
        for object_id in dataset.object_ids()
        if dataset.get_object(object_id).object_split in set(args.splits)
    ]
    if args.max_objects is not None:
        selected_ids = selected_ids[: int(args.max_objects)]
    context_trajectories = (
        int(args.context_trajectories_per_view)
        if args.context_trajectories_per_view is not None
        else int(saved_args["context_trajectories_per_view"])
    )
    context_window_steps = (
        int(args.context_window_steps)
        if args.context_window_steps is not None
        else int(saved_args["context_window_steps"])
    )

    records: list[dict[str, Any]] = []
    image_paths: list[Path] = []
    print(
        f"checkpoint={checkpoint} iteration={payload.get('iteration')} "
        f"objects={len(selected_ids)} device={device}"
    )
    for index, object_id in enumerate(selected_ids):
        obj = dataset.get_object(object_id)
        features, visual_features = object_model_features(
            obj,
            local_surface_points=local_surface_points,
            saved_args=saved_args,
            device=device,
        )
        record = evaluate_object(
            obj=obj,
            model=model,
            dataset=dataset,
            point_features=features,
            visual_features=visual_features,
            local_surface_points=local_surface_points,
            active_indices=active_indices,
            context_trajectories=context_trajectories,
            context_window_steps=context_window_steps,
            rng=np.random.default_rng(int(args.seed) + 1009 * index),
            device=device,
        )
        records.append(record)
        image_path = output_dir / "per_object" / f"{object_id}.png"
        render_object_comparison(
            record,
            active_points=active_points,
            output_path=image_path,
            mu_min=float(args.mu_color_min),
            mu_max=float(args.mu_color_max),
        )
        image_paths.append(image_path)
        print(
            f"[{index + 1:02d}/{len(selected_ids):02d}] {object_id} "
            f"mae={record['metrics']['point_mae']:.4f} "
            f"region_mae={record['metrics']['region_mean_mae']:.4f} "
            f"direction={record['metrics']['contrast_direction_correct']}"
        )

    pca = render_latent_pca(records, output_dir / "latent_pca.png")
    render_latent_heatmap(records, output_dir / "latent_vectors_heatmap.png")
    latent_distances = render_latent_distance_matrix(records, output_dir / "latent_distance_matrix.png")
    aggregate = aggregate_metrics(records, latent_distances=latent_distances)
    create_contact_sheet(image_paths, output_dir / "friction_contact_sheet.png")
    write_csv(records, output_dir / "object_metrics.csv")

    latent_views = np.stack([record["latent_views"] for record in records], axis=0)
    projection_views = np.stack([record["projection_views"] for record in records], axis=0)
    predicted_mu_views = np.stack([record["predicted_mu_views"] for record in records], axis=0)
    target_mu = np.stack([record["target_mu"] for record in records], axis=0)
    np.savez_compressed(
        output_dir / "object_latent_friction_data.npz",
        object_ids=np.asarray([record["object_id"] for record in records], dtype=object),
        object_splits=np.asarray([record["object_split"] for record in records], dtype=object),
        partition_families=np.asarray([record["partition_family"] for record in records], dtype=object),
        active_indices=active_indices,
        active_points=active_points,
        latent_views=latent_views,
        latent_means=np.mean(latent_views, axis=1),
        projection_views=projection_views,
        predicted_mu_views=predicted_mu_views,
        predicted_mu=np.mean(predicted_mu_views, axis=1),
        target_mu=target_mu,
        pca_components=pca["components"],
        pca_explained_variance_ratio=pca["explained_variance_ratio"],
        pca_view_coordinates=pca["view_coordinates"],
        latent_distance_matrix=latent_distances,
    )

    summary_records = [
        {
            "object_id": record["object_id"],
            "object_split": record["object_split"],
            "partition_family": record["partition_family"],
            "friction_spec": record["friction_spec"],
            "latent_views": record["latent_views"],
            "latent_mean": record["latent_mean"],
            "target_region_means": record["target_region_means"],
            "predicted_region_means": record["predicted_region_means"],
            "metrics": record["metrics"],
            "visualization": str((output_dir / "per_object" / f"{record['object_id']}.png").resolve()),
        }
        for record in records
    ]
    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": payload.get("iteration"),
        "checkpoint_best_loss": payload.get("best_loss"),
        "training_target_iterations": saved_args.get("opt_iters"),
        "checkpoint_is_training_complete": int(payload.get("iteration", -1)) >= int(saved_args.get("opt_iters", 0)),
        "manifest": str(manifest),
        "output_dir": str(output_dir.resolve()),
        "device": str(device),
        "context_trajectories_per_view": context_trajectories,
        "context_window_steps": context_window_steps,
        "active_point_count": int(len(active_indices)),
        "pca_explained_variance_ratio": pca["explained_variance_ratio"],
        "aggregate": aggregate,
        "objects": summary_records,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    write_gallery(records, output_path=output_dir / "gallery.html", checkpoint=checkpoint, aggregate=aggregate)
    print(json.dumps(jsonable(aggregate), indent=2, sort_keys=True))
    print(f"gallery={output_dir / 'gallery.html'}")


if __name__ == "__main__":
    main()
