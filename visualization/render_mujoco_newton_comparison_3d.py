from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warp as wp

from render_mujoco_newton_comparison_video import (
    OUTPUTS_ROOT,
    PALETTE,
    PredictionResult,
    build_frame_indices,
    default_output_path as default_video_output_path,
    load_checkpoint_parameters,
    normalize_quaternions_xyzw,
    resolve_run_specs,
    resolve_trajectory_npz_path,
    rotation_matrix_xyzw,
    run_prediction,
    select_trajectory,
    transform_local_point,
    wrap_angle_radians,
    yaw_from_xyzw,
)


BOX_FACE_I = [0, 0, 4, 4, 0, 0, 2, 2, 0, 0, 1, 1]
BOX_FACE_J = [1, 2, 5, 6, 4, 5, 6, 7, 3, 7, 5, 6]
BOX_FACE_K = [2, 3, 6, 7, 5, 1, 7, 3, 7, 4, 6, 2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("experiment_name_arg", nargs="*", help="Experiment name(s) under outputs/.")
    parser.add_argument("--experiment-name", dest="experiment_name_options", nargs="+", action="append", default=[])
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--checkpoint-paths", type=Path, nargs="+", default=None)
    parser.add_argument("--labels", type=str, nargs="+", default=None)
    parser.add_argument("--trajectory-index", type=int, default=0)
    parser.add_argument("--trajectory-npz", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--checkpoint-param-set",
        choices=("best", "current"),
        default="best",
        help="Sparse parameter vector to replay when --param-iteration is not set.",
    )
    parser.add_argument(
        "--param-iteration",
        type=int,
        default=None,
        help="Load point friction from <checkpoint-point-cloud-dir>/iter_XXXXXX.ply.",
    )
    parser.add_argument("--checkpoint-point-cloud-dir", type=Path, default=None)
    parser.add_argument("--reference-point-cloud", type=Path, default=None)
    parser.add_argument("--reference-point-clouds", type=Path, nargs="+", default=None)

    parser.add_argument("--solver-iterations", type=int, default=10)
    parser.add_argument("--box-mass", type=float, default=1.0)
    parser.add_argument("--floor-half-extents", type=float, nargs=3, default=(2.0, 2.0, 0.05))
    parser.add_argument("--box-half-extents", type=float, nargs=3, default=(0.1, 0.05, 0.025))
    parser.add_argument("--box-start-pos", type=float, nargs=3, default=(0.58, 0.0, 0.025))
    parser.add_argument("--surface-point-spacing", type=float, default=0.01)
    parser.add_argument("--friction-contact-threshold", type=float, default=0.002)
    parser.add_argument("--point-friction", type=float, default=0.1)
    parser.add_argument("--contact-friction", type=float, default=0.0)
    parser.add_argument("--contact-stiffness", type=float, default=1.0e5)
    parser.add_argument("--contact-damping", type=float, default=50.0)
    parser.add_argument("--contact-margin", type=float, default=1.0e-3)
    parser.add_argument("--friction-regularization", type=float, default=1.0e-3)

    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--trail-frames", type=int, default=60)
    parser.add_argument("--force-arrow-length", type=float, default=0.08)
    parser.add_argument("--include-plotlyjs", choices=("cdn", "inline"), default="cdn")
    args = parser.parse_args()
    args.run_specs = resolve_run_specs(args, parser)
    return args


def default_output_path(args: argparse.Namespace) -> Path:
    video_path = default_video_output_path(args)
    if len(args.run_specs) == 1 and args.run_specs[0].experiment_name is not None:
        return video_path.parent.parent / "visualizations" / f"{video_path.stem}_3d.html"
    return OUTPUTS_ROOT / "comparison_3d" / f"{video_path.stem}_3d.html"


def box_vertices(position: np.ndarray, quaternion: np.ndarray, half_extents: np.ndarray) -> np.ndarray:
    hx, hy, hz = np.asarray(half_extents, dtype=np.float32)
    local_vertices = np.asarray(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float32,
    )
    return np.asarray(position, dtype=np.float32) + local_vertices @ rotation_matrix_xyzw(quaternion).T


def make_box_mesh(
    *,
    name: str,
    position: np.ndarray,
    quaternion: np.ndarray,
    half_extents: np.ndarray,
    color: str,
    opacity: float,
    legendgroup: str,
    showlegend: bool,
) -> go.Mesh3d:
    vertices = box_vertices(position, quaternion, half_extents)
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=BOX_FACE_I,
        j=BOX_FACE_J,
        k=BOX_FACE_K,
        name=name,
        legendgroup=legendgroup,
        showlegend=showlegend,
        color=color,
        opacity=opacity,
        flatshading=True,
        hoverinfo="name",
    )


def make_floor_mesh(x_range: tuple[float, float], y_range: tuple[float, float]) -> go.Mesh3d:
    x0, x1 = x_range
    y0, y1 = y_range
    z = -1.0e-4
    return go.Mesh3d(
        x=[x0, x1, x1, x0],
        y=[y0, y0, y1, y1],
        z=[z, z, z, z],
        i=[0, 0],
        j=[1, 2],
        k=[2, 3],
        name="floor z=0",
        color="#d9d9d9",
        opacity=0.26,
        hoverinfo="skip",
        showlegend=False,
    )


def trim_to_common_frames(
    trajectory,
    predictions: list[PredictionResult],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[PredictionResult]]:
    frame_count = min([len(trajectory.positions), *(len(result.positions) for result in predictions)])
    target_positions = np.asarray(trajectory.positions[:frame_count], dtype=np.float32)
    target_quaternions = normalize_quaternions_xyzw(trajectory.quaternions_xyzw[:frame_count])
    for result in predictions:
        result.positions = np.asarray(result.positions[:frame_count], dtype=np.float32)
        result.quaternions = normalize_quaternions_xyzw(result.quaternions[:frame_count])
    return target_positions, target_quaternions, np.asarray(trajectory.time[:frame_count], dtype=np.float32), predictions


def sampled_frame_indices(frame_count: int, frame_stride: int, max_frames: int) -> np.ndarray:
    indices = build_frame_indices(frame_count, max(int(frame_stride), 1))
    max_frames = max(int(max_frames), 2)
    if len(indices) > max_frames:
        positions = np.linspace(0, len(indices) - 1, max_frames)
        indices = indices[np.unique(np.rint(positions).astype(np.int32))]
        if int(indices[-1]) != frame_count - 1:
            indices = np.concatenate([indices, np.asarray([frame_count - 1], dtype=np.int32)])
    return indices.astype(np.int32)


def compute_force_series(trajectory, target_positions: np.ndarray, target_quaternions: np.ndarray):
    frame_count = len(target_positions)
    local_force_point = np.asarray(trajectory.force_point_offset_local, dtype=np.float32)
    force_points = np.asarray(
        [
            transform_local_point(target_positions[i], target_quaternions[i], local_force_point)
            for i in range(frame_count)
        ],
        dtype=np.float32,
    )
    forces = np.zeros((frame_count, 3), dtype=np.float32)
    if len(trajectory.step_forces) > 0:
        used = min(frame_count, len(trajectory.step_forces))
        forces[:used] = np.asarray(trajectory.step_forces[:used, :3], dtype=np.float32)
        if used < frame_count:
            forces[used:] = forces[used - 1]
    force_norm = np.linalg.norm(forces, axis=1)
    force_dirs = np.divide(forces, np.maximum(force_norm.reshape(-1, 1), 1.0e-8))
    return force_points, forces, force_norm, force_dirs


def compute_errors(
    target_positions: np.ndarray,
    target_quaternions: np.ndarray,
    predictions: list[PredictionResult],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    target_yaw = np.asarray([yaw_from_xyzw(q) for q in target_quaternions], dtype=np.float32)
    xy_errors = [
        np.linalg.norm(result.positions[:, :2] - target_positions[:, :2], axis=1)
        for result in predictions
    ]
    yaw_errors = [
        np.abs(
            np.rad2deg(
                wrap_angle_radians(
                    np.asarray([yaw_from_xyzw(q) for q in result.quaternions], dtype=np.float32) - target_yaw
                )
            )
        )
        for result in predictions
    ]
    return xy_errors, yaw_errors


def line3d(
    *,
    name: str,
    points: np.ndarray,
    color: str,
    width: float,
    legendgroup: str,
    showlegend: bool,
    dash: str | None = None,
    opacity: float = 1.0,
    hovertext: list[str] | None = None,
) -> go.Scatter3d:
    line = {"color": color, "width": width}
    if dash is not None:
        line["dash"] = dash
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="lines",
        line=line,
        opacity=opacity,
        name=name,
        legendgroup=legendgroup,
        showlegend=showlegend,
        hovertext=hovertext,
        hoverinfo="text" if hovertext is not None else "name",
    )


def marker3d(
    *,
    name: str,
    point: np.ndarray,
    color: str,
    size: float,
    legendgroup: str,
    showlegend: bool,
    symbol: str = "circle",
    hovertext: str | None = None,
) -> go.Scatter3d:
    return go.Scatter3d(
        x=[float(point[0])],
        y=[float(point[1])],
        z=[float(point[2])],
        mode="markers",
        marker={"color": color, "size": size, "symbol": symbol},
        name=name,
        legendgroup=legendgroup,
        showlegend=showlegend,
        hovertext=hovertext,
        hoverinfo="text" if hovertext is not None else "name",
    )


def force_arrow_trace(
    *,
    point: np.ndarray,
    direction: np.ndarray,
    force_norm: float,
    arrow_length: float,
    color: str = "#0077bb",
) -> go.Scatter3d:
    if not np.isfinite(force_norm) or force_norm <= 1.0e-6:
        coords = np.empty((0, 3), dtype=np.float32)
    else:
        start = np.asarray(point, dtype=np.float32)
        end = start + np.asarray(direction, dtype=np.float32) * float(arrow_length)
        coords = np.vstack([start, end])
    return go.Scatter3d(
        x=coords[:, 0] if len(coords) else [],
        y=coords[:, 1] if len(coords) else [],
        z=coords[:, 2] if len(coords) else [],
        mode="lines",
        line={"color": color, "width": 8},
        name="force direction",
        legendgroup="force",
        showlegend=True,
        hoverinfo="name",
    )


def make_error_cursor(times: np.ndarray, x_time: float, y_max: float) -> go.Scatter:
    return go.Scatter(
        x=[x_time, x_time],
        y=[0.0, y_max],
        mode="lines",
        line={"color": "#555555", "width": 1.2},
        name="current time",
        showlegend=False,
        hoverinfo="skip",
    )


def make_trajectory_hover(name: str, times: np.ndarray, positions: np.ndarray) -> list[str]:
    return [
        f"{name}<br>t={times[i]:.3f}s<br>x={positions[i, 0]:.4f}<br>y={positions[i, 1]:.4f}<br>z={positions[i, 2]:.4f}"
        for i in range(len(times))
    ]


def scene_ranges(
    target_positions: np.ndarray,
    predictions: list[PredictionResult],
    half_extents: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    xy_arrays = [target_positions[:, :2], *(result.positions[:, :2] for result in predictions)]
    xy = np.vstack(xy_arrays)
    xy_min = np.min(xy, axis=0)
    xy_max = np.max(xy, axis=0)
    xy_margin = max(float(half_extents[0]), float(half_extents[1]), 0.02) * 1.8
    xy_min -= xy_margin
    xy_max += xy_margin
    xy_center = 0.5 * (xy_min + xy_max)
    xy_radius = max(0.08, 0.5 * float(np.max(xy_max - xy_min)))
    pad = max(0.03, xy_radius * 0.22)
    z_top = max(
        float(np.max(target_positions[:, 2])),
        *(float(np.max(result.positions[:, 2])) for result in predictions),
    )
    z_high = max(z_top + float(half_extents[2]) * 4.0, 0.18)
    return (
        (float(xy_center[0] - xy_radius - pad), float(xy_center[0] + xy_radius + pad)),
        (float(xy_center[1] - xy_radius - pad), float(xy_center[1] + xy_radius + pad)),
        (-0.01, z_high),
    )


def build_figure(
    *,
    trajectory,
    predictions: list[PredictionResult],
    half_extents: np.ndarray,
    frame_indices: np.ndarray,
    trail_frames: int,
    force_arrow_length: float,
) -> go.Figure:
    target_positions, target_quaternions, times, predictions = trim_to_common_frames(trajectory, predictions)
    xy_errors, yaw_errors = compute_errors(target_positions, target_quaternions, predictions)
    force_points, _forces, force_norm, force_dirs = compute_force_series(trajectory, target_positions, target_quaternions)

    x_range, y_range, z_range = scene_ranges(target_positions, predictions, half_extents)
    max_xy_error = max(float(np.max(error)) for error in xy_errors) if xy_errors else 1.0e-4
    error_y_max = max(max_xy_error * 1.15, 1.0e-4)

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.74, 0.26],
        vertical_spacing=0.06,
        specs=[[{"type": "scene"}], [{"type": "xy"}]],
    )

    fig.add_trace(make_floor_mesh(x_range, y_range), row=1, col=1)
    fig.add_trace(
        line3d(
            name="MuJoCo trajectory",
            points=target_positions,
            color="#222222",
            width=6.0,
            legendgroup="target",
            showlegend=True,
            opacity=0.45,
            hovertext=make_trajectory_hover("MuJoCo", times, target_positions),
        ),
        row=1,
        col=1,
    )
    for run_idx, result in enumerate(predictions):
        color = PALETTE[run_idx % len(PALETTE)]
        fig.add_trace(
            line3d(
                name=f"{result.legend_label} trajectory",
                points=result.positions,
                color=color,
                width=4.0,
                dash="dash",
                legendgroup=result.label,
                showlegend=True,
                opacity=0.72,
                hovertext=make_trajectory_hover(result.label, times, result.positions),
            ),
            row=1,
            col=1,
        )

    first_idx = int(frame_indices[0])
    dynamic_trace_indices: list[int] = []
    first_trail_start = max(0, first_idx - max(int(trail_frames), 1))

    fig.add_trace(
        line3d(
            name="MuJoCo current trail",
            points=target_positions[first_trail_start : first_idx + 1],
            color="#222222",
            width=8.0,
            legendgroup="target",
            showlegend=False,
            opacity=0.88,
        ),
        row=1,
        col=1,
    )
    dynamic_trace_indices.append(len(fig.data) - 1)
    for run_idx, result in enumerate(predictions):
        color = PALETTE[run_idx % len(PALETTE)]
        fig.add_trace(
            line3d(
                name=f"{result.label} current trail",
                points=result.positions[first_trail_start : first_idx + 1],
                color=color,
                width=6.0,
                legendgroup=result.label,
                showlegend=False,
                opacity=0.95,
            ),
            row=1,
            col=1,
        )
        dynamic_trace_indices.append(len(fig.data) - 1)

    fig.add_trace(
        make_box_mesh(
            name="MuJoCo box",
            position=target_positions[first_idx],
            quaternion=target_quaternions[first_idx],
            half_extents=half_extents,
            color="#222222",
            opacity=0.23,
            legendgroup="target",
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    dynamic_trace_indices.append(len(fig.data) - 1)
    fig.add_trace(
        marker3d(
            name="MuJoCo current center",
            point=target_positions[first_idx],
            color="#222222",
            size=4.8,
            legendgroup="target",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    dynamic_trace_indices.append(len(fig.data) - 1)

    for run_idx, result in enumerate(predictions):
        color = PALETTE[run_idx % len(PALETTE)]
        fig.add_trace(
            make_box_mesh(
                name=f"{result.legend_label} box",
                position=result.positions[first_idx],
                quaternion=result.quaternions[first_idx],
                half_extents=half_extents,
                color=color,
                opacity=0.35,
                legendgroup=result.label,
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        dynamic_trace_indices.append(len(fig.data) - 1)
        fig.add_trace(
            marker3d(
                name=f"{result.label} current center",
                point=result.positions[first_idx],
                color=color,
                size=4.2,
                legendgroup=result.label,
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        dynamic_trace_indices.append(len(fig.data) - 1)

    fig.add_trace(
        marker3d(
            name="force application point",
            point=force_points[first_idx],
            color="#0077bb",
            size=5.2,
            symbol="x",
            legendgroup="force",
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    dynamic_trace_indices.append(len(fig.data) - 1)
    fig.add_trace(
        force_arrow_trace(
            point=force_points[first_idx],
            direction=force_dirs[first_idx],
            force_norm=float(force_norm[first_idx]),
            arrow_length=force_arrow_length,
        ),
        row=1,
        col=1,
    )
    dynamic_trace_indices.append(len(fig.data) - 1)

    for run_idx, (result, xy_error, yaw_error) in enumerate(zip(predictions, xy_errors, yaw_errors, strict=True)):
        color = PALETTE[run_idx % len(PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=times,
                y=xy_error,
                mode="lines",
                line={"color": color, "width": 2.0},
                name=f"{result.label} xy error",
                legendgroup=result.label,
                showlegend=True,
                customdata=np.stack([yaw_error], axis=1),
                hovertemplate="t=%{x:.3f}s<br>xy=%{y:.5f}m<br>yaw=%{customdata[0]:.3f}deg<extra></extra>",
            ),
            row=2,
            col=1,
        )
    fig.add_trace(make_error_cursor(times, float(times[first_idx]), error_y_max), row=2, col=1)
    dynamic_trace_indices.append(len(fig.data) - 1)

    frames = []
    for frame_idx in frame_indices:
        i = int(frame_idx)
        trail_start = max(0, i - max(int(trail_frames), 1))
        frame_data = [
            line3d(
                name="MuJoCo current trail",
                points=target_positions[trail_start : i + 1],
                color="#222222",
                width=8.0,
                legendgroup="target",
                showlegend=False,
                opacity=0.88,
            ),
        ]
        for run_idx, result in enumerate(predictions):
            color = PALETTE[run_idx % len(PALETTE)]
            frame_data.append(
                line3d(
                    name=f"{result.label} current trail",
                    points=result.positions[trail_start : i + 1],
                    color=color,
                    width=6.0,
                    legendgroup=result.label,
                    showlegend=False,
                    opacity=0.95,
                )
            )
        frame_data.extend(
            [
            make_box_mesh(
                name="MuJoCo box",
                position=target_positions[i],
                quaternion=target_quaternions[i],
                half_extents=half_extents,
                color="#222222",
                opacity=0.23,
                legendgroup="target",
                showlegend=True,
            ),
            marker3d(
                name="MuJoCo current center",
                point=target_positions[i],
                color="#222222",
                size=4.8,
                legendgroup="target",
                showlegend=False,
                hovertext=f"MuJoCo<br>t={times[i]:.3f}s",
            ),
            ]
        )
        for run_idx, result in enumerate(predictions):
            color = PALETTE[run_idx % len(PALETTE)]
            frame_data.append(
                make_box_mesh(
                    name=f"{result.legend_label} box",
                    position=result.positions[i],
                    quaternion=result.quaternions[i],
                    half_extents=half_extents,
                    color=color,
                    opacity=0.35,
                    legendgroup=result.label,
                    showlegend=True,
                )
            )
            frame_data.append(
                marker3d(
                    name=f"{result.label} current center",
                    point=result.positions[i],
                    color=color,
                    size=4.2,
                    legendgroup=result.label,
                    showlegend=False,
                    hovertext=(
                        f"{result.label}<br>t={times[i]:.3f}s<br>"
                        f"xy error={xy_errors[run_idx][i]:.5f}m<br>"
                        f"yaw error={yaw_errors[run_idx][i]:.3f}deg"
                    ),
                )
            )
        frame_data.append(
            marker3d(
                name="force application point",
                point=force_points[i],
                color="#0077bb",
                size=5.2,
                symbol="x",
                legendgroup="force",
                showlegend=True,
                hovertext=f"force point<br>t={times[i]:.3f}s<br>|F|={force_norm[i]:.3f}N",
            )
        )
        frame_data.append(
            force_arrow_trace(
                point=force_points[i],
                direction=force_dirs[i],
                force_norm=float(force_norm[i]),
                arrow_length=force_arrow_length,
            )
        )
        frame_data.append(make_error_cursor(times, float(times[i]), error_y_max))
        frames.append(go.Frame(data=frame_data, traces=dynamic_trace_indices, name=str(i)))
    fig.frames = frames

    slider_steps = [
        {
            "method": "animate",
            "label": str(int(frame_idx)),
            "args": [
                [str(int(frame_idx))],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
        }
        for frame_idx in frame_indices
    ]
    play_button = {
        "label": "Play",
        "method": "animate",
        "args": [
            None,
            {
                "frame": {"duration": 80, "redraw": True},
                "fromcurrent": True,
                "transition": {"duration": 0},
            },
        ],
    }
    pause_button = {
        "label": "Pause",
        "method": "animate",
        "args": [
            [None],
            {
                "mode": "immediate",
                "frame": {"duration": 0, "redraw": False},
                "transition": {"duration": 0},
            },
        ],
    }

    title_parts = [
        f"MuJoCo vs Newton 3D replay | trajectory {len(times)} frames",
        *(f"{result.label}: final xy={result.final_xy_error * 1000.0:.2f} mm" for result in predictions),
    ]
    fig.update_layout(
        title="<br>".join(title_parts),
        height=980,
        margin={"l": 30, "r": 30, "t": 86, "b": 30},
        legend={"groupclick": "toggleitem", "itemsizing": "constant"},
        scene={
            "xaxis": {"title": "world x (m)", "range": x_range},
            "yaxis": {"title": "world y (m)", "range": y_range},
            "zaxis": {"title": "world z (m)", "range": z_range},
            "aspectmode": "manual",
            "aspectratio": {
                "x": max(x_range[1] - x_range[0], 1.0e-6),
                "y": max(y_range[1] - y_range[0], 1.0e-6),
                "z": max(z_range[1] - z_range[0], 1.0e-6),
            },
            "camera": {"eye": {"x": 1.25, "y": -1.65, "z": 0.9}},
        },
        xaxis={"title": "time (s)", "range": [float(times[0]), float(times[-1])]},
        yaxis={"title": "xy error (m)", "range": [0.0, error_y_max]},
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "frame "},
                "steps": slider_steps,
                "x": 0.08,
                "y": 0.0,
                "len": 0.86,
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "buttons": [play_button, pause_button],
                "x": 0.08,
                "y": 0.08,
                "xanchor": "left",
                "yanchor": "bottom",
            }
        ],
    )
    return fig


def load_predictions(args: argparse.Namespace):
    first_checkpoint = load_checkpoint_parameters(args.run_specs[0].checkpoint_path)
    trajectory_npz_path = resolve_trajectory_npz_path(args, first_checkpoint)
    args.trajectory_npz = trajectory_npz_path
    replay_max_steps = first_checkpoint.max_steps if args.max_steps is None else args.max_steps

    trajectory = select_trajectory(trajectory_npz_path, replay_max_steps, int(args.trajectory_index))
    wp.init()

    predictions: list[PredictionResult] = []
    for run_spec in args.run_specs:
        prediction, resolved_trajectory_path = run_prediction(
            args=args,
            run_spec=run_spec,
            trajectory=trajectory,
        )
        if resolved_trajectory_path.resolve() != trajectory_npz_path.resolve():
            print(
                f"warning: {run_spec.label} checkpoint metadata points to {resolved_trajectory_path.resolve()}, "
                f"but comparison uses {trajectory_npz_path.resolve()}"
            )
        predictions.append(prediction)
    return trajectory, predictions


def main() -> None:
    args = parse_args()
    trajectory, predictions = load_predictions(args)
    if not predictions:
        raise ValueError("No predictions were produced")

    half_extents = np.asarray(predictions[0].half_extents, dtype=np.float32)
    for prediction in predictions[1:]:
        if not np.allclose(prediction.half_extents, half_extents, atol=1.0e-6):
            print(
                f"warning: {prediction.label} box_half_extents={prediction.half_extents.tolist()} "
                f"differs from first run {half_extents.tolist()}; visualization uses first run extents"
            )

    frame_count = min([len(trajectory.positions), *(len(result.positions) for result in predictions)])
    frame_indices = sampled_frame_indices(frame_count, args.frame_stride, args.max_frames)
    fig = build_figure(
        trajectory=trajectory,
        predictions=predictions,
        half_extents=half_extents,
        frame_indices=frame_indices,
        trail_frames=int(args.trail_frames),
        force_arrow_length=float(args.force_arrow_length),
    )

    output_path = args.output if args.output is not None else default_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    include_plotlyjs = True if args.include_plotlyjs == "inline" else "cdn"
    fig.write_html(str(output_path), include_plotlyjs=include_plotlyjs, full_html=True)
    print(f"html_written_to={output_path.resolve()}")


if __name__ == "__main__":
    main()
