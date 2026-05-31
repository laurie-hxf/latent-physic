from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_npz", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-points", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--point-size", type=float, default=4.0)
    return parser.parse_args()


def _sample_indices(count: int, max_points: int, seed: int) -> np.ndarray:
    if count <= max_points:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, size=max_points, replace=False))


def _rgb_colors(data: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    if "colors" not in data:
        return np.full((len(indices), 3), 0.25, dtype=np.float32)
    return np.asarray(data["colors"], dtype=np.float32)[indices] / 255.0


def _status_colors(data: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    visibility = np.asarray(data["visibility_counts"], dtype=np.int32)[indices]
    fallback = np.asarray(data.get("depth_fallback_used", np.zeros_like(visibility, dtype=bool)), dtype=bool)[indices]
    colors = np.full((len(indices), 3), [0.55, 0.55, 0.55], dtype=np.float32)
    colors[visibility > 0] = np.asarray([0.10, 0.36, 0.82], dtype=np.float32)
    colors[fallback] = np.asarray([0.95, 0.45, 0.08], dtype=np.float32)
    return colors


def _pca_feature_colors(data: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    features = np.asarray(data["dino_features"], dtype=np.float32)[indices]
    visibility = np.asarray(data["visibility_counts"], dtype=np.int32)[indices]
    fallback = np.asarray(data.get("depth_fallback_used", np.zeros_like(visibility, dtype=bool)), dtype=bool)[indices]
    assigned = (visibility > 0) | fallback
    colors = np.full((len(indices), 3), 0.7, dtype=np.float32)
    if np.count_nonzero(assigned) < 3:
        return colors

    feat = features[assigned]
    feat = feat - feat.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(feat, full_matrices=False)
    components = feat @ vt[: min(3, vt.shape[0])].T
    if components.shape[1] < 3:
        components = np.pad(components, ((0, 0), (0, 3 - components.shape[1])))
    lo = np.percentile(components, 1.0, axis=0)
    hi = np.percentile(components, 99.0, axis=0)
    scale = np.maximum(hi - lo, 1.0e-6)
    colors[assigned] = np.clip((components - lo) / scale, 0.0, 1.0)
    return colors


def _scatter_2d(ax, points: np.ndarray, x_idx: int, y_idx: int, colors: np.ndarray, title: str, point_size: float) -> None:
    ax.scatter(points[:, x_idx], points[:, y_idx], c=colors, s=point_size, linewidths=0)
    labels = ["x", "y", "z"]
    ax.set_xlabel(labels[x_idx])
    ax.set_ylabel(labels[y_idx])
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)


def main() -> None:
    args = parse_args()
    output = args.output or args.input_npz.with_name(args.input_npz.stem + "_viz.png")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(args.input_npz) as loaded:
        data = {name: loaded[name] for name in loaded.files}

    points = np.asarray(data["points"], dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N,3), got {points.shape}")
    indices = _sample_indices(len(points), int(args.max_points), int(args.seed))
    points_view = points[indices]

    rgb = _rgb_colors(data, indices)
    status = _status_colors(data, indices)
    pca = _pca_feature_colors(data, indices)

    visibility = np.asarray(data["visibility_counts"], dtype=np.int32)
    fallback = np.asarray(data.get("depth_fallback_used", np.zeros_like(visibility, dtype=bool)), dtype=bool)
    assigned = int(np.count_nonzero((visibility > 0) | fallback))
    strict = int(np.count_nonzero(visibility > 0))
    fallback_count = int(np.count_nonzero(fallback))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    title = (
        f"{args.input_npz.name}: points={len(points)}, assigned={assigned}, "
        f"strict={strict}, fallback={fallback_count}"
    )
    fig.suptitle(title)
    _scatter_2d(axes[0, 0], points_view, 0, 1, rgb, "top-down xy, RGB", args.point_size)
    _scatter_2d(axes[0, 1], points_view, 0, 2, rgb, "side xz, RGB", args.point_size)
    _scatter_2d(axes[1, 0], points_view, 0, 1, status, "feature assignment: blue strict, orange fallback", args.point_size)
    _scatter_2d(axes[1, 1], points_view, 0, 1, pca, "DINO feature PCA colors", args.point_size)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
