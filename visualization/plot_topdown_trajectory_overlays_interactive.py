from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import warp as wp

from plot_topdown_trajectory_overlays import (
    DEFAULT_DATASET,
    CheckpointParams,
    checkpoint_has_active_params,
    checkpoint_legend_label,
    checkpoint_summary,
    load_checkpoint_params,
    make_eval_args,
    parse_optional_max_steps,
    rollout_positions_for_trajectories,
    select_methods,
    select_representative_indices,
)
from fit_mujoco_contact_point_friction_runtime import resolve_batch_size
from mujoco_contact_friction_fit_utils import load_mujoco_trajectories
from newton_surface_points_diff_demo import build_diff_scene


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT = ROOT / "report_assets" / "topdown_trajectory_overlays_fixed20_interactive.html"
REFERENCE_COLORS = ["#7a3e9d", "#00798c", "#b7791f", "#c43b5b", "#4b5563"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument(
        "--reference-dataset",
        type=Path,
        action="append",
        default=None,
        help="Additional MuJoCo datasets whose ground-truth XY trajectories should be overlaid.",
    )
    parser.add_argument(
        "--reference-label",
        type=str,
        action="append",
        default=None,
        help="Display label for each --reference-dataset. Must be repeated the same number of times when used.",
    )
    parser.add_argument(
        "--reference-color",
        type=str,
        action="append",
        default=None,
        help="CSS color for each --reference-dataset. Must be repeated the same number of times when used.",
    )
    parser.add_argument(
        "--method-source",
        choices=("default", "curated", "auto", "all"),
        default="all",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        action="append",
        default=None,
        help="Root directory to scan for checkpoint .npz files when --method-source is auto or all.",
    )
    parser.add_argument("--max-steps", type=parse_optional_max_steps, default=300)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--eval-batch-size", type=int, default=20)
    parser.add_argument("--surface-point-spacing", type=float, default=0.01)
    parser.add_argument("--contact-stiffness", type=float, default=1.0e5)
    parser.add_argument("--contact-damping", type=float, default=50.0)
    parser.add_argument("--friction-contact-threshold", type=float, default=0.002)
    parser.add_argument("--contact-mask-threshold", type=float, default=0.002)
    parser.add_argument("--position-loss-weight", type=float, default=1.0)
    parser.add_argument("--orientation-loss-weight", type=float, default=0.0)
    parser.add_argument("--linear-velocity-loss-weight", type=float, default=0.0)
    parser.add_argument("--angular-velocity-loss-weight", type=float, default=0.0)
    parser.add_argument("--trajectory-indices", type=int, nargs="*", default=None)
    parser.add_argument(
        "--all-trajectories",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot every trajectory in the dataset unless --trajectory-indices is provided.",
    )
    parser.add_argument("--include-pure-point", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-width", type=int, default=280)
    parser.add_argument("--plot-height", type=int, default=230)
    parser.add_argument("--legend-width", type=int, default=520)
    parser.add_argument("--axis-padding-frac", type=float, default=0.12)
    parser.add_argument(
        "--unified-axis-scale",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use one shared equal-aspect x/y axis range for every panel.",
    )
    parser.add_argument(
        "--reuse-summary",
        type=Path,
        default=None,
        help=(
            "Optional summary JSON from this script. When present, reuse saved XY polylines instead of rerunning Newton."
        ),
    )
    return parser.parse_args()


def default_reference_label(path: Path) -> str:
    stem = path.stem
    marker = "_uniform_mu_"
    if marker in stem:
        mu = stem.rsplit(marker, 1)[1].replace("p", ".")
        return f"Uniform mu={mu}"
    return stem


def collect_reference_payload(args: argparse.Namespace, selected_indices: list[int]) -> list[dict]:
    reference_datasets = args.reference_dataset or []
    reference_labels = args.reference_label or []
    reference_colors = args.reference_color or []
    if reference_labels and len(reference_labels) != len(reference_datasets):
        raise ValueError("--reference-label must be provided once per --reference-dataset")
    if reference_colors and len(reference_colors) != len(reference_datasets):
        raise ValueError("--reference-color must be provided once per --reference-dataset")

    references = []
    used_names: set[str] = set()
    for ref_idx, dataset in enumerate(reference_datasets):
        if not dataset.exists():
            raise FileNotFoundError(dataset)
        collection = load_mujoco_trajectories(dataset, args.max_steps, None)
        tracks = []
        for selected_idx in selected_indices:
            if selected_idx >= len(collection.trajectories):
                raise ValueError(f"{dataset} has no trajectory index {selected_idx}")
            trajectory = collection.trajectories[selected_idx]
            tracks.append(np.asarray(trajectory.positions[:, :2], dtype=np.float32).tolist())

        base_name = f"reference_{dataset.stem}"
        name = base_name
        suffix = 2
        while name in used_names:
            name = f"{base_name}_{suffix}"
            suffix += 1
        used_names.add(name)

        references.append(
            {
                "name": name,
                "label": reference_labels[ref_idx] if reference_labels else default_reference_label(dataset),
                "color": reference_colors[ref_idx] if reference_colors else REFERENCE_COLORS[ref_idx % len(REFERENCE_COLORS)],
                "dataset": str(dataset),
                "tracks": tracks,
            }
        )
    return references


def html_escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def css_color(value) -> str:
    if isinstance(value, str):
        return value
    r, g, b = value[:3]
    return f"rgb({int(round(r * 255))}, {int(round(g * 255))}, {int(round(b * 255))})"


def path_points(points: np.ndarray, x_min: float, y_min: float, scale: float, plot_height: int, pad: int) -> str:
    if len(points) == 0:
        return ""
    coords = []
    for x, y in np.asarray(points, dtype=np.float32):
        sx = pad + (float(x) - x_min) * scale
        sy = pad + plot_height - (float(y) - y_min) * scale
        coords.append(f"{sx:.2f},{sy:.2f}")
    return "M " + " L ".join(coords)


def marker_xy(point: np.ndarray, x_min: float, y_min: float, scale: float, plot_height: int, pad: int) -> tuple[float, float]:
    sx = pad + (float(point[0]) - x_min) * scale
    sy = pad + plot_height - (float(point[1]) - y_min) * scale
    return sx, sy


def axis_bounds(all_xy: list[np.ndarray], padding_frac: float) -> tuple[float, float, float, float]:
    stacked = np.concatenate([xy for xy in all_xy if len(xy) > 0], axis=0)
    x_min = float(np.min(stacked[:, 0]))
    x_max = float(np.max(stacked[:, 0]))
    y_min = float(np.min(stacked[:, 1]))
    y_max = float(np.max(stacked[:, 1]))
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    radius = 0.5 * max(x_max - x_min, y_max - y_min, 1.0e-6)
    pad = max(radius * float(padding_frac), 0.002)
    return cx - radius - pad, cx + radius + pad, cy - radius - pad, cy + radius + pad


def axis_tick_values(v_min: float, v_max: float, count: int = 3) -> list[float]:
    return [float(value) for value in np.linspace(float(v_min), float(v_max), int(count))]


def format_axis_tick(value: float, span: float) -> str:
    value = 0.0 if abs(float(value)) < 5.0e-8 else float(value)
    span = abs(float(span))
    if span < 0.02:
        return f"{value:.4f}"
    if span < 0.2:
        return f"{value:.3f}"
    if span < 2.0:
        return f"{value:.2f}"
    return f"{value:.1f}"


def load_cached_payload(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "interactive_data" not in payload:
        raise ValueError(f"{path} does not contain interactive_data")
    return payload


def collect_rollout_payload(args: argparse.Namespace) -> dict:
    methods = select_methods(args)
    for method in methods:
        if not method.checkpoint.exists():
            raise FileNotFoundError(method.checkpoint)
        if not checkpoint_has_active_params(method.checkpoint):
            raise ValueError(f"{method.checkpoint} is not a training checkpoint with active friction parameters")

    wp.init()
    eval_args = make_eval_args(args)
    collection = load_mujoco_trajectories(eval_args.trajectory_npz, eval_args.max_steps, eval_args.max_trajectories)
    trajectories = collection.trajectories
    eval_args.steps = collection.max_steps
    eval_args.dt = trajectories[0].timestep

    selected_indices = select_representative_indices(args.dataset, args.trajectory_indices, args.all_trajectories)
    selected_indices = [idx for idx in selected_indices if 0 <= idx < len(trajectories)]
    if not selected_indices:
        raise ValueError("No valid trajectory indices selected")
    selected_trajectories = [trajectories[idx] for idx in selected_indices]
    eval_args.eval_batch_size = resolve_batch_size(args.eval_batch_size, len(selected_trajectories), eval_args.batch_size)
    eval_args.batch_size = eval_args.eval_batch_size
    eval_args.batch_capacity = max(eval_args.eval_batch_size, 1)

    diff_scene = build_diff_scene(eval_args)
    initial_body_q = diff_scene.states[0].body_q.numpy().copy()
    initial_body_qd = diff_scene.states[0].body_qd.numpy().copy()
    checkpoint_params: dict[str, CheckpointParams] = {
        method.name: load_checkpoint_params(method.checkpoint)
        for method in methods
    }
    legend_labels = {
        method.name: checkpoint_legend_label(method, checkpoint_params[method.name])
        for method in methods
    }

    target_tracks = []
    for selected_idx, trajectory in zip(selected_indices, selected_trajectories):
        point = np.asarray(trajectory.force_point_offset_local, dtype=np.float32)
        force = np.asarray(trajectory.step_forces[0], dtype=np.float32)
        target_tracks.append(
            {
                "trajectory_index": int(selected_idx),
                "episode_index": int(trajectory.metadata.get("episode_index", selected_idx)),
                "point": [float(point[0]), float(point[1]), float(point[2])],
                "force": [float(force[0]), float(force[1]), float(force[2])],
                "xy": np.asarray(trajectory.positions[:, :2], dtype=np.float32).tolist(),
            }
        )
    reference_tracks = collect_reference_payload(args, selected_indices)

    method_tracks = {}
    method_losses = {}
    method_summaries = []
    for method_idx, method in enumerate(methods):
        checkpoint = checkpoint_params[method.name]
        print(
            f"rolling out {method_idx + 1}/{len(methods)} {method.name} "
            f"active={len(checkpoint.active_indices)} param={checkpoint.parameterization}",
            flush=True,
        )
        positions, losses = rollout_positions_for_trajectories(
            diff_scene=diff_scene,
            trajectories=selected_trajectories,
            eval_args=eval_args,
            active_indices=checkpoint.active_indices,
            active_params=checkpoint.active_params,
            initial_body_q=initial_body_q,
            initial_body_qd=initial_body_qd,
        )
        method_tracks[method.name] = [
            np.asarray(position[: len(target_tracks[idx]["xy"]), :2], dtype=np.float32).tolist()
            for idx, position in enumerate(positions)
        ]
        method_losses[method.name] = [float(loss) for loss in losses]
        summary = checkpoint_summary(method, checkpoint, losses)
        summary["legend_label"] = legend_labels[method.name]
        summary["color"] = css_color(method.color)
        method_summaries.append(summary)

    return {
        "dataset": str(args.dataset),
        "max_steps": args.max_steps,
        "selected_trajectories": selected_indices,
        "eval_batch_size": eval_args.eval_batch_size,
        "contact_stiffness": args.contact_stiffness,
        "surface_point_spacing": args.surface_point_spacing,
        "loss_weights": {
            "position": float(args.position_loss_weight),
            "orientation": float(args.orientation_loss_weight),
            "linear_velocity": float(args.linear_velocity_loss_weight),
            "angular_velocity": float(args.angular_velocity_loss_weight),
        },
        "methods": method_summaries,
        "reference_datasets": [str(path) for path in (args.reference_dataset or [])],
        "trajectory_losses": {
            method.name: {
                str(selected_indices[idx]): float(loss)
                for idx, loss in enumerate(method_losses[method.name])
            }
            for method in methods
        },
        "interactive_data": {
            "targets": target_tracks,
            "references": reference_tracks,
            "methods": [
                {
                    "name": method.name,
                    "label": legend_labels[method.name],
                    "color": css_color(method.color),
                    "stage": method.stage,
                    "checkpoint": str(method.checkpoint),
                    "losses": method_losses[method.name],
                    "tracks": method_tracks[method.name],
                }
                for method in methods
            ],
        },
    }


def render_html(payload: dict, args: argparse.Namespace) -> str:
    targets = payload["interactive_data"]["targets"]
    methods = payload["interactive_data"]["methods"]
    references = payload["interactive_data"].get("references", [])
    plot_width = int(args.plot_width)
    plot_height = int(args.plot_height)
    pad = 54
    panel_width = plot_width + pad * 2
    panel_height = plot_height + pad * 2 + 38
    cols = 5 if len(targets) > 12 else 3
    rows = int(np.ceil(len(targets) / cols))
    svg_width = cols * panel_width + int(args.legend_width)
    svg_height = max(rows * panel_height + 34, 820)
    legend_x = cols * panel_width + 24
    legend_y = 52
    legend_row_h = 26

    method_ids = {method["name"]: f"m{idx}" for idx, method in enumerate(methods)}
    reference_ids = {reference["name"]: f"r{idx}" for idx, reference in enumerate(references)}
    overlay_count = len(methods) + len(references)
    overlay_text = f"{len(methods)} checkpoints"
    if references:
        overlay_text += f" + {len(references)} references"
    if args.unified_axis_scale:
        overlay_text += " | shared axes"
    parts: list[str] = []
    title = (
        f"Top-down trajectory overlays | {Path(payload['dataset']).stem} | "
        f"{overlay_text} | {'all steps' if payload['max_steps'] is None else 'max_steps=' + str(payload['max_steps'])}"
    )
    loss_weights = payload.get("loss_weights")
    if loss_weights:
        title += (
            f" | loss w: pos={float(loss_weights.get('position', 0.0)):.3g}, "
            f"rot={float(loss_weights.get('orientation', 0.0)):.3g}"
        )

    parts.append("<!doctype html>")
    parts.append("<html><head><meta charset=\"utf-8\">")
    parts.append(f"<title>{html_escape(title)}</title>")
    parts.append(
        """
<style>
:root { color-scheme: light; }
body {
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: #1f2933;
  background: #f7f8fa;
}
.page { padding: 18px 22px 28px; }
h1 { font-size: 18px; margin: 0 0 4px; font-weight: 650; }
.meta { font-size: 12px; color: #5b6673; margin-bottom: 14px; }
.frame {
  background: white;
  border: 1px solid #d8dee8;
  border-radius: 8px;
  overflow: auto;
  box-shadow: 0 1px 2px rgba(16, 24, 40, 0.06);
}
svg { display: block; background: white; }
.panel-bg { fill: #ffffff; stroke: #dfe5ee; stroke-width: 1; rx: 6; }
.plot-bg { fill: #fbfcfe; stroke: #d8dee8; stroke-width: 1; }
.grid { stroke: #edf1f7; stroke-width: 1; }
.tick { stroke: #98a2b3; stroke-width: 1; }
.tick-label { fill: #667085; font-size: 8px; }
.axis-label, .panel-title { fill: #344054; font-size: 10px; }
.panel-title { font-size: 11px; font-weight: 600; }
.target-line { fill: none; stroke: #101828; stroke-width: 2.4; opacity: 0.9; pointer-events: none; }
.target-marker { fill: #101828; stroke: #101828; pointer-events: none; }
.track-hit {
  fill: none;
  stroke: rgba(0, 0, 0, 0.001);
  stroke-width: 15;
  stroke-linecap: round;
  stroke-linejoin: round;
  pointer-events: stroke;
  cursor: crosshair;
}
.track-line {
  fill: none;
  stroke-width: 1.45;
  opacity: 0.42;
  pointer-events: none;
  transition: opacity 120ms ease, stroke-width 120ms ease;
}
.reference-track.track-line {
  stroke-width: 2.0;
  stroke-dasharray: 5 3;
  opacity: 0.62;
}
.reference-track.track-end {
  opacity: 0.72;
}
.track-end {
  opacity: 0.55;
  pointer-events: none;
  transition: opacity 120ms ease, r 120ms ease;
}
.legend-title { fill: #111827; font-size: 12px; font-weight: 700; }
.legend-item {
  cursor: default;
  opacity: 0.76;
  transition: opacity 120ms ease;
  pointer-events: all;
}
.legend-item rect { fill: rgba(255, 255, 255, 0.001); stroke: transparent; }
.legend-label { fill: #344054; font-size: 9.5px; dominant-baseline: middle; }
.legend-swatch { stroke-width: 2.4; }
.legend-icon { stroke: #ffffff; stroke-width: 1.6; }
.legend-icon-text {
  fill: #ffffff;
  font-size: 8px;
  font-weight: 700;
  dominant-baseline: central;
  text-anchor: middle;
  pointer-events: none;
}
.dimmed .track-line { opacity: 0.08; }
.dimmed .track-end { opacity: 0.08; }
.dimmed .legend-item { opacity: 0.26; }
.active-method.track-line { opacity: 0.98; stroke-width: 3.4; }
.active-method.track-end { opacity: 1; r: 3.8; }
.active-method.legend-item { opacity: 1; }
.active-method .legend-label { fill: #111827; font-weight: 700; }
.active-method .legend-box { fill: rgba(37, 99, 235, 0.08); stroke: rgba(37, 99, 235, 0.22); }
.active-method .legend-icon { stroke: #111827; stroke-width: 2.2; }
.active-track.track-line { stroke-width: 4.6; }
.active-track.track-end { r: 4.4; }
.pinned-method.legend-item .legend-box { fill: rgba(16, 24, 40, 0.06); stroke: rgba(16, 24, 40, 0.28); }
.tooltip {
  position: fixed;
  z-index: 10;
  max-width: 460px;
  padding: 8px 10px;
  border: 1px solid #ccd4df;
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.97);
  color: #1f2933;
  font-size: 12px;
  line-height: 1.35;
  box-shadow: 0 8px 24px rgba(16, 24, 40, 0.14);
  display: none;
  pointer-events: none;
}
.toolbar {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 0 0 10px;
  font-size: 12px;
  color: #475467;
}
.toolbar button {
  border: 1px solid #cfd7e3;
  background: #fff;
  border-radius: 6px;
  padding: 5px 9px;
  cursor: pointer;
  font: inherit;
  color: #1f2933;
}
.toolbar button:hover { background: #f4f7fb; }
</style>
"""
    )
    parts.append("</head><body><div class=\"page\">")
    parts.append(f"<h1>{html_escape(title)}</h1>")
    parts.append(
        "<div class=\"meta\">Hover a legend row to isolate a checkpoint. "
        "Hover a trajectory line to isolate its overlay and emphasize that specific trajectory. "
        "The black curve is the primary dataset ground truth.</div>"
    )
    parts.append("<div class=\"toolbar\"><button type=\"button\" id=\"resetBtn\">Reset highlight</button>")
    parts.append(f"<span>{len(targets)} trajectories, {overlay_count} overlays</span></div>")
    parts.append("<div class=\"frame\">")
    parts.append(f"<svg id=\"overlaySvg\" width=\"{svg_width}\" height=\"{svg_height}\" viewBox=\"0 0 {svg_width} {svg_height}\">")
    parts.append(f"<text x=\"16\" y=\"24\" class=\"legend-title\">{html_escape(title)}</text>")

    global_axis_bounds = None
    if args.unified_axis_scale:
        global_xy = []
        for target_idx, target in enumerate(targets):
            global_xy.append(np.asarray(target["xy"], dtype=np.float32))
            for reference in references:
                global_xy.append(np.asarray(reference["tracks"][target_idx], dtype=np.float32))
            for method in methods:
                global_xy.append(np.asarray(method["tracks"][target_idx], dtype=np.float32))
        global_axis_bounds = axis_bounds(global_xy, args.axis_padding_frac)

    for target_idx, target in enumerate(targets):
        col = target_idx % cols
        row = target_idx // cols
        ox = col * panel_width + 12
        oy = row * panel_height + 38
        all_xy = [np.asarray(target["xy"], dtype=np.float32)]
        for reference in references:
            all_xy.append(np.asarray(reference["tracks"][target_idx], dtype=np.float32))
        for method in methods:
            all_xy.append(np.asarray(method["tracks"][target_idx], dtype=np.float32))
        if global_axis_bounds is None:
            x_min, x_max, y_min, y_max = axis_bounds(all_xy, args.axis_padding_frac)
        else:
            x_min, x_max, y_min, y_max = global_axis_bounds
        scale = min(plot_width / (x_max - x_min), plot_height / (y_max - y_min))
        plot_x = ox + pad
        plot_y = oy + pad
        parts.append(f"<g class=\"panel\" data-traj=\"{target_idx}\">")
        parts.append(f"<rect class=\"panel-bg\" x=\"{ox}\" y=\"{oy}\" width=\"{panel_width - 10}\" height=\"{panel_height - 8}\"/>")
        title_text = (
            f"traj {target['trajectory_index']} | point x={target['point'][0]:.3f}, y={target['point'][1]:.3f} | "
            f"force=({target['force'][0]:.2f},{target['force'][1]:.2f})"
        )
        parts.append(f"<text class=\"panel-title\" x=\"{ox + 12}\" y=\"{oy + 17}\">{html_escape(title_text)}</text>")
        parts.append(f"<rect class=\"plot-bg\" x=\"{plot_x}\" y=\"{plot_y}\" width=\"{plot_width}\" height=\"{plot_height}\"/>")
        for grid_idx in range(1, 4):
            gx = plot_x + grid_idx * plot_width / 4.0
            gy = plot_y + grid_idx * plot_height / 4.0
            parts.append(f"<line class=\"grid\" x1=\"{gx:.2f}\" y1=\"{plot_y}\" x2=\"{gx:.2f}\" y2=\"{plot_y + plot_height}\"/>")
            parts.append(f"<line class=\"grid\" x1=\"{plot_x}\" y1=\"{gy:.2f}\" x2=\"{plot_x + plot_width}\" y2=\"{gy:.2f}\"/>")
        x_span = x_max - x_min
        y_span = y_max - y_min
        for tick_value in axis_tick_values(x_min, x_max):
            tx = plot_x + (tick_value - x_min) * scale
            label = format_axis_tick(tick_value, x_span)
            parts.append(
                f"<line class=\"tick\" x1=\"{tx:.2f}\" y1=\"{plot_y + plot_height:.2f}\" "
                f"x2=\"{tx:.2f}\" y2=\"{plot_y + plot_height + 4:.2f}\"/>"
            )
            parts.append(
                f"<text class=\"tick-label\" x=\"{tx:.2f}\" y=\"{plot_y + plot_height + 13:.2f}\" "
                f"text-anchor=\"middle\">{html_escape(label)}</text>"
            )
        for tick_value in axis_tick_values(y_min, y_max):
            ty = plot_y + plot_height - (tick_value - y_min) * scale
            label = format_axis_tick(tick_value, y_span)
            parts.append(
                f"<line class=\"tick\" x1=\"{plot_x - 4:.2f}\" y1=\"{ty:.2f}\" "
                f"x2=\"{plot_x:.2f}\" y2=\"{ty:.2f}\"/>"
            )
            parts.append(
                f"<text class=\"tick-label\" x=\"{plot_x - 7:.2f}\" y=\"{ty + 3:.2f}\" "
                f"text-anchor=\"end\">{html_escape(label)}</text>"
            )
        for reference_idx, reference in enumerate(references):
            reference_id = reference_ids[reference["name"]]
            track_xy = np.asarray(reference["tracks"][target_idx], dtype=np.float32)
            track_path = path_points(track_xy, x_min, y_min, scale, plot_height, pad)
            tooltip = (
                f"{reference['label']}<br>"
                f"trajectory {target['trajectory_index']}<br>"
                f"{reference['dataset']}"
            )
            parts.append(
                f"<path class=\"track-line reference-track method-{reference_id}\" data-method=\"{reference_id}\" "
                f"data-method-name=\"{html_escape(reference['name'])}\" data-traj=\"{target_idx}\" "
                f"stroke=\"{html_escape(reference['color'])}\" d=\"{track_path}\" transform=\"translate({ox}, {oy})\"/>"
            )
            parts.append(
                f"<path class=\"track-hit\" data-method=\"{reference_id}\" data-method-name=\"{html_escape(reference['name'])}\" "
                f"data-traj=\"{target_idx}\" data-tooltip=\"{html_escape(tooltip)}\" "
                f"d=\"{track_path}\" transform=\"translate({ox}, {oy})\"/>"
            )
            ex, ey = marker_xy(track_xy[-1], x_min, y_min, scale, plot_height, pad)
            parts.append(
                f"<circle class=\"track-end reference-track method-{reference_id}\" data-method=\"{reference_id}\" "
                f"data-traj=\"{target_idx}\" cx=\"{ox + ex:.2f}\" cy=\"{oy + ey:.2f}\" r=\"2.5\" "
                f"fill=\"{html_escape(reference['color'])}\"/>"
            )
        for method_idx, method in enumerate(methods):
            method_id = method_ids[method["name"]]
            track_xy = np.asarray(method["tracks"][target_idx], dtype=np.float32)
            track_path = path_points(track_xy, x_min, y_min, scale, plot_height, pad)
            loss = float(method["losses"][target_idx])
            tooltip = (
                f"{method['label']}<br>"
                f"trajectory {target['trajectory_index']} loss={loss:.6g}<br>"
                f"{method['checkpoint']}"
            )
            parts.append(
                f"<path class=\"track-line method-{method_id}\" data-method=\"{method_id}\" "
                f"data-method-name=\"{html_escape(method['name'])}\" data-traj=\"{target_idx}\" "
                f"stroke=\"{html_escape(method['color'])}\" d=\"{track_path}\" transform=\"translate({ox}, {oy})\"/>"
            )
            parts.append(
                f"<path class=\"track-hit\" data-method=\"{method_id}\" data-method-name=\"{html_escape(method['name'])}\" "
                f"data-traj=\"{target_idx}\" data-tooltip=\"{html_escape(tooltip)}\" "
                f"d=\"{track_path}\" transform=\"translate({ox}, {oy})\"/>"
            )
            ex, ey = marker_xy(track_xy[-1], x_min, y_min, scale, plot_height, pad)
            parts.append(
                f"<circle class=\"track-end method-{method_id}\" data-method=\"{method_id}\" data-traj=\"{target_idx}\" "
                f"cx=\"{ox + ex:.2f}\" cy=\"{oy + ey:.2f}\" r=\"2.3\" fill=\"{html_escape(method['color'])}\"/>"
            )
        target_xy = np.asarray(target["xy"], dtype=np.float32)
        target_path = path_points(target_xy, x_min, y_min, scale, plot_height, pad)
        parts.append(f"<path class=\"target-line\" d=\"{target_path}\" transform=\"translate({ox}, {oy})\"/>")
        start_x, start_y = marker_xy(target_xy[0], x_min, y_min, scale, plot_height, pad)
        end_x, end_y = marker_xy(target_xy[-1], x_min, y_min, scale, plot_height, pad)
        parts.append(f"<circle class=\"target-marker\" cx=\"{ox + start_x:.2f}\" cy=\"{oy + start_y:.2f}\" r=\"2.8\"/>")
        parts.append(
            f"<path class=\"target-marker\" d=\"M {ox + end_x - 4:.2f},{oy + end_y - 4:.2f} "
            f"L {ox + end_x + 4:.2f},{oy + end_y + 4:.2f} "
            f"M {ox + end_x + 4:.2f},{oy + end_y - 4:.2f} "
            f"L {ox + end_x - 4:.2f},{oy + end_y + 4:.2f}\" stroke-width=\"1.8\"/>"
        )
        parts.append(f"<text class=\"axis-label\" x=\"{plot_x + plot_width - 8}\" y=\"{plot_y + plot_height + 29}\" text-anchor=\"end\">x</text>")
        parts.append(f"<text class=\"axis-label\" x=\"{plot_x - 18}\" y=\"{plot_y + 10}\" text-anchor=\"middle\">y</text>")
        parts.append("</g>")

    parts.append(f"<g class=\"legend\" transform=\"translate({legend_x}, {legend_y})\">")
    parts.append("<text class=\"legend-title\" x=\"0\" y=\"0\">Overlays</text>")
    parts.append("<g transform=\"translate(0, 18)\">")
    parts.append("<line class=\"legend-swatch\" x1=\"0\" y1=\"0\" x2=\"26\" y2=\"0\" stroke=\"#101828\"/>")
    parts.append("<text class=\"legend-label\" x=\"34\" y=\"0\">Primary dataset ground truth</text>")
    parts.append("</g>")
    for idx, reference in enumerate(references):
        reference_id = reference_ids[reference["name"]]
        y = 44 + idx * legend_row_h
        tooltip = f"{reference['label']}<br>{reference['dataset']}"
        parts.append(
            f"<g class=\"legend-item method-{reference_id}\" data-method=\"{reference_id}\" "
            f"data-tooltip=\"{html_escape(tooltip)}\" transform=\"translate(0, {y})\">"
        )
        parts.append(f"<rect class=\"legend-box\" x=\"-8\" y=\"-10\" width=\"{args.legend_width - 36}\" height=\"21\" rx=\"4\"/>")
        parts.append(
            f"<line class=\"legend-swatch\" x1=\"0\" y1=\"0\" x2=\"26\" y2=\"0\" "
            f"stroke=\"{html_escape(reference['color'])}\" stroke-dasharray=\"5 3\"/>"
        )
        parts.append(
            f"<circle class=\"legend-icon\" cx=\"39\" cy=\"0\" r=\"7\" fill=\"{html_escape(reference['color'])}\"/>"
        )
        parts.append(f"<text class=\"legend-icon-text\" x=\"39\" y=\"0\">R</text>")
        parts.append(f"<text class=\"legend-label\" x=\"52\" y=\"0\">{html_escape(reference['label'])}</text>")
        parts.append("</g>")
    method_legend_start = 44 + len(references) * legend_row_h
    for idx, method in enumerate(methods):
        method_id = method_ids[method["name"]]
        y = method_legend_start + idx * legend_row_h
        label = method["label"]
        avg_loss = float(np.mean(method["losses"])) if method["losses"] else float("nan")
        tooltip = f"{label}<br>mean loss={avg_loss:.6g}<br>{method['checkpoint']}"
        parts.append(
            f"<g class=\"legend-item method-{method_id}\" data-method=\"{method_id}\" "
            f"data-tooltip=\"{html_escape(tooltip)}\" transform=\"translate(0, {y})\">"
        )
        parts.append(f"<rect class=\"legend-box\" x=\"-8\" y=\"-10\" width=\"{args.legend_width - 36}\" height=\"21\" rx=\"4\"/>")
        parts.append(f"<line class=\"legend-swatch\" x1=\"0\" y1=\"0\" x2=\"26\" y2=\"0\" stroke=\"{html_escape(method['color'])}\"/>")
        parts.append(f"<circle class=\"legend-icon\" cx=\"39\" cy=\"0\" r=\"7\" fill=\"{html_escape(method['color'])}\"/>")
        parts.append(f"<text class=\"legend-icon-text\" x=\"39\" y=\"0\">{idx + 1}</text>")
        parts.append(f"<text class=\"legend-label\" x=\"52\" y=\"0\">{html_escape(label)} | mean loss={avg_loss:.4g}</text>")
        parts.append("</g>")
    parts.append("</g>")
    parts.append("</svg></div><div id=\"tooltip\" class=\"tooltip\"></div>")
    parts.append(
        """
<script>
const svg = document.getElementById('overlaySvg');
const tooltip = document.getElementById('tooltip');
let activeMethod = null;
let activeTrajectory = null;
let pinnedMethod = null;
let pinnedTrajectory = null;

function showTooltip(evt, html) {
  if (!html) return;
  tooltip.innerHTML = html;
  tooltip.style.display = 'block';
  moveTooltip(evt);
}

function moveTooltip(evt) {
  if (tooltip.style.display !== 'block') return;
  tooltip.style.left = `${evt.clientX + 14}px`;
  tooltip.style.top = `${evt.clientY + 14}px`;
}

function hideTooltip() {
  tooltip.style.display = 'none';
}

function setHighlight(methodId, trajectoryId = null) {
  activeMethod = methodId;
  activeTrajectory = trajectoryId;
  svg.classList.toggle('dimmed', Boolean(methodId));
  document.querySelectorAll('.active-method').forEach(el => el.classList.remove('active-method'));
  document.querySelectorAll('.active-track').forEach(el => el.classList.remove('active-track'));
  document.querySelectorAll('.pinned-method').forEach(el => el.classList.remove('pinned-method'));
  if (!methodId) return;
  document.querySelectorAll(`.method-${methodId}`).forEach(el => el.classList.add('active-method'));
  if (pinnedMethod === methodId) {
    document.querySelectorAll(`.method-${methodId}`).forEach(el => el.classList.add('pinned-method'));
  }
  if (trajectoryId !== null) {
    document.querySelectorAll(`.track-line[data-method="${methodId}"][data-traj="${trajectoryId}"]`).forEach(el => {
      el.classList.add('active-track');
    });
    document.querySelectorAll(`.track-end[data-method="${methodId}"][data-traj="${trajectoryId}"]`).forEach(el => {
      el.classList.add('active-track');
    });
  }
}

function clearHighlight() {
  if (pinnedMethod) {
    setHighlight(pinnedMethod, pinnedTrajectory);
  } else {
    setHighlight(null, null);
  }
  hideTooltip();
}

document.querySelectorAll('.legend-item').forEach(item => {
  item.addEventListener('mouseenter', evt => {
    if (!pinnedMethod) setHighlight(item.dataset.method, null);
    showTooltip(evt, item.dataset.tooltip);
  });
  item.addEventListener('mousemove', moveTooltip);
  item.addEventListener('mouseleave', clearHighlight);
  item.addEventListener('click', evt => {
    evt.stopPropagation();
    if (pinnedMethod === item.dataset.method && pinnedTrajectory === null) {
      pinnedMethod = null;
      pinnedTrajectory = null;
      clearHighlight();
      return;
    }
    pinnedMethod = item.dataset.method;
    pinnedTrajectory = null;
    setHighlight(pinnedMethod, pinnedTrajectory);
    showTooltip(evt, item.dataset.tooltip);
  });
});

document.querySelectorAll('.track-hit').forEach(path => {
  path.addEventListener('mouseenter', evt => {
    if (!pinnedMethod) setHighlight(path.dataset.method, path.dataset.traj);
    showTooltip(evt, path.dataset.tooltip);
  });
  path.addEventListener('mousemove', moveTooltip);
  path.addEventListener('mouseleave', clearHighlight);
  path.addEventListener('click', evt => {
    evt.stopPropagation();
    pinnedMethod = path.dataset.method;
    pinnedTrajectory = path.dataset.traj;
    setHighlight(pinnedMethod, pinnedTrajectory);
    showTooltip(evt, path.dataset.tooltip);
  });
});

function resetPinnedHighlight() {
  pinnedMethod = null;
  pinnedTrajectory = null;
  setHighlight(null, null);
  hideTooltip();
}

document.getElementById('resetBtn').addEventListener('click', evt => {
  evt.stopPropagation();
  resetPinnedHighlight();
});
document.addEventListener('click', resetPinnedHighlight);
document.addEventListener('keydown', evt => {
  if (evt.key === 'Escape') resetPinnedHighlight();
});
</script>
"""
    )
    parts.append("</div></body></html>")
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    if args.reuse_summary is not None:
        payload = load_cached_payload(args.reuse_summary)
    else:
        payload = collect_rollout_payload(args)

    html_text = render_html(payload, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_text, encoding="utf-8")

    summary_output = args.summary_output
    if summary_output is None:
        summary_output = args.output.with_name(f"{args.output.stem}_summary.json")
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_payload = dict(payload)
    summary_payload["output"] = str(args.output)
    summary_payload["unified_axis_scale"] = bool(args.unified_axis_scale)
    summary_output.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {summary_output}")


if __name__ == "__main__":
    main()
