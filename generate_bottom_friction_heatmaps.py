#!/usr/bin/env python3
"""Generate 2D heatmaps from surface friction values in ASCII PLY files."""

from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


AXES = ("x", "y", "z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read ASCII PLY point clouds, extract a surface, and generate "
            "friction heatmaps."
        )
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=["iter_*.ply"],
        help=(
            "Input PLY file path(s), glob(s), or directory path(s). "
            "Directories are scanned non-recursively for *.ply / *.PLY. "
            "Default: iter_*.ply"
        ),
    )
    parser.add_argument(
        "--output",
        default="bottom_friction_heatmaps",
        help="Output directory. Default: bottom_friction_heatmaps",
    )
    parser.add_argument(
        "--axis",
        choices=AXES,
        default="z",
        help="Axis used to identify the bottom surface. Default: z",
    )
    parser.add_argument(
        "--side",
        choices=("min", "max"),
        default="min",
        help="Use min or max coordinate on --axis as the surface. Default: min",
    )
    parser.add_argument(
        "--all-surfaces",
        action="store_true",
        help="Generate heatmaps for all six axis-aligned surfaces.",
    )
    parser.add_argument(
        "--scale",
        choices=("global", "per-file"),
        default="global",
        help="Color scale mode. Default: global. Fixed --vmin/--vmax override this.",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Fixed lower bound for the color scale.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Fixed upper bound for the color scale.",
    )
    parser.add_argument(
        "--cmap",
        default="inferno",
        help="Matplotlib colormap name. Default: inferno",
    )
    parser.add_argument("--dpi", type=int, default=220, help="PNG DPI. Default: 220")
    parser.add_argument(
        "--no-overview",
        action="store_true",
        help="Do not generate the combined overview PNG.",
    )
    parser.add_argument(
        "--individual",
        action="store_true",
        help="Also generate one heatmap PNG per input file.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also write CSV tables for each surface and a summary report.",
    )
    parser.add_argument(
        "--no-csv",
        dest="csv",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def surface_label(axis: str, side: str) -> str:
    return f"{axis}_{side}_surface"


def make_norm(
    args: argparse.Namespace,
    global_min: float,
    global_max: float,
    grid: np.ndarray | None = None,
) -> tuple[Normalize, float, float, str]:
    if args.vmin is not None or args.vmax is not None:
        vmin = global_min if args.vmin is None else args.vmin
        vmax = global_max if args.vmax is None else args.vmax
        if vmin >= vmax:
            raise ValueError("--vmin must be smaller than --vmax")
        return Normalize(vmin=vmin, vmax=vmax, clip=True), vmin, vmax, "fixed"

    if args.scale == "per-file":
        if grid is None:
            raise ValueError("per-file scale requires a grid")
        vmin = float(grid.min())
        vmax = float(grid.max())
        return Normalize(vmin=vmin, vmax=vmax), vmin, vmax, "per-file"

    return Normalize(vmin=global_min, vmax=global_max), global_min, global_max, "global"


def colorbar_extend(data_min: float, data_max: float, vmin: float, vmax: float) -> str:
    lower = data_min < vmin
    upper = data_max > vmax
    if lower and upper:
        return "both"
    if lower:
        return "min"
    if upper:
        return "max"
    return "neither"


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        candidate = Path(pattern)
        if candidate.is_dir():
            paths.extend(
                sorted(
                    p
                    for p in candidate.iterdir()
                    if p.is_file() and p.suffix.lower() == ".ply"
                )
            )
            continue

        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(candidate)

    seen = set()
    unique_paths = []
    for path in sorted(paths):
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(path)

    missing = [str(path) for path in unique_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Input file(s) not found: " + ", ".join(missing))
    if not unique_paths:
        raise FileNotFoundError("No input PLY files matched.")
    return unique_paths


def read_ascii_ply_vertices(path: Path) -> tuple[list[str], list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8") as handle:
        header = []
        for line in handle:
            header.append(line.rstrip("\n"))
            if line.strip() == "end_header":
                break

        if not header or header[-1].strip() != "end_header":
            raise ValueError(f"{path}: missing end_header")
        if not any(line.strip() == "format ascii 1.0" for line in header):
            raise ValueError(f"{path}: only ASCII PLY format is supported")

        vertex_count = None
        properties = []
        in_vertex = False
        for line in header:
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif in_vertex and len(parts) >= 3 and parts[0] == "property":
                properties.append(parts[-1])

        if vertex_count is None:
            raise ValueError(f"{path}: missing vertex element")

        vertices = []
        for _ in range(vertex_count):
            line = handle.readline()
            if not line:
                raise ValueError(f"{path}: unexpected EOF in vertex data")
            vertices.append(line.split())

    return header, properties, vertices


def extract_surface_grid(path: Path, axis: str, side: str) -> dict[str, object]:
    _, properties, vertices = read_ascii_ply_vertices(path)
    required = {"x", "y", "z", "friction"}
    missing = required - set(properties)
    if missing:
        raise ValueError(f"{path}: missing properties: {', '.join(sorted(missing))}")

    prop_idx = {name: index for index, name in enumerate(properties)}
    axis_idx = prop_idx[axis]
    axis_values = np.array([float(row[axis_idx]) for row in vertices], dtype=float)
    surface_value = (
        float(axis_values.min()) if side == "min" else float(axis_values.max())
    )
    tolerance = max(1e-8, float(axis_values.max() - axis_values.min()) * 1e-5)
    surface_rows = [
        row
        for row in vertices
        if abs(float(row[axis_idx]) - surface_value) <= tolerance
    ]
    if not surface_rows:
        raise ValueError(f"{path}: no surface vertices detected")

    grid_axes = [name for name in AXES if name != axis]
    horizontal_axis, vertical_axis = grid_axes
    h_idx = prop_idx[horizontal_axis]
    v_idx = prop_idx[vertical_axis]
    friction_idx = prop_idx["friction"]

    h_values = np.array(sorted({float(row[h_idx]) for row in surface_rows}))
    v_values = np.array(sorted({float(row[v_idx]) for row in surface_rows}))
    grid = np.full((len(v_values), len(h_values)), np.nan, dtype=float)
    h_lookup = {value: index for index, value in enumerate(h_values)}
    v_lookup = {value: index for index, value in enumerate(v_values)}

    for row in surface_rows:
        h_value = float(row[h_idx])
        v_value = float(row[v_idx])
        friction = float(row[friction_idx])
        grid[v_lookup[v_value], h_lookup[h_value]] = friction

    if np.isnan(grid).any():
        raise ValueError(f"{path}: surface points do not form a complete 2D grid")

    return {
        "path": path,
        "surface_label": surface_label(axis, side),
        "horizontal_axis": horizontal_axis,
        "vertical_axis": vertical_axis,
        "horizontal_values": h_values,
        "vertical_values": v_values,
        "grid": grid,
        "surface_axis": axis,
        "surface_side": side,
        "surface_value": surface_value,
        "surface_count": len(surface_rows),
    }


def save_individual_heatmap(
    item: dict[str, object],
    out_dir: Path,
    norm: Normalize,
    cmap: str,
    dpi: int,
    extend: str,
) -> None:
    path = item["path"]
    label = item["surface_label"]
    assert isinstance(path, Path)
    assert isinstance(label, str)
    h_values = item["horizontal_values"]
    v_values = item["vertical_values"]
    grid = item["grid"]
    horizontal_axis = item["horizontal_axis"]
    vertical_axis = item["vertical_axis"]
    assert isinstance(h_values, np.ndarray)
    assert isinstance(v_values, np.ndarray)
    assert isinstance(grid, np.ndarray)
    assert isinstance(horizontal_axis, str)
    assert isinstance(vertical_axis, str)

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=dpi)
    image = ax.imshow(
        grid,
        origin="lower",
        extent=[h_values.min(), h_values.max(), v_values.min(), v_values.max()],
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_title(f"{path.stem} {label} friction")
    ax.set_xlabel(horizontal_axis)
    ax.set_ylabel(vertical_axis)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.extend = extend
    colorbar.set_label("friction")
    ax.text(
        0.01,
        0.99,
        f"min {grid.min():.6f}\nmax {grid.max():.6f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="white",
        bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=3),
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"{path.stem}_{label}_friction_heatmap.png")
    plt.close(fig)


def save_csv(item: dict[str, object], out_dir: Path) -> None:
    path = item["path"]
    label = item["surface_label"]
    h_values = item["horizontal_values"]
    v_values = item["vertical_values"]
    grid = item["grid"]
    horizontal_axis = item["horizontal_axis"]
    vertical_axis = item["vertical_axis"]
    assert isinstance(path, Path)
    assert isinstance(label, str)
    assert isinstance(h_values, np.ndarray)
    assert isinstance(v_values, np.ndarray)
    assert isinstance(grid, np.ndarray)
    assert isinstance(horizontal_axis, str)
    assert isinstance(vertical_axis, str)

    data = np.column_stack(
        [
            np.repeat(v_values, len(h_values)),
            np.tile(h_values, len(v_values)),
            grid.reshape(-1),
        ]
    )
    csv_path = out_dir / f"{path.stem}_{label}_friction_grid.csv"
    header = f"{vertical_axis},{horizontal_axis},friction"
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="", fmt="%.12g")


def save_overview(
    items: list[dict[str, object]],
    out_dir: Path,
    norm: Normalize,
    cmap: str,
    dpi: int,
    scale_text: str,
    extend: str,
) -> None:
    surface = items[0]["surface_label"]
    assert isinstance(surface, str)
    cols = 3
    rows = int(math.ceil(len(items) / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(14, 3.9 * rows),
        dpi=dpi,
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes).ravel()
    last_image = None

    for ax, item in zip(axes_array, items):
        path = item["path"]
        h_values = item["horizontal_values"]
        v_values = item["vertical_values"]
        grid = item["grid"]
        horizontal_axis = item["horizontal_axis"]
        vertical_axis = item["vertical_axis"]
        assert isinstance(path, Path)
        assert isinstance(h_values, np.ndarray)
        assert isinstance(v_values, np.ndarray)
        assert isinstance(grid, np.ndarray)
        assert isinstance(horizontal_axis, str)
        assert isinstance(vertical_axis, str)

        last_image = ax.imshow(
            grid,
            origin="lower",
            extent=[h_values.min(), h_values.max(), v_values.min(), v_values.max()],
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_title(f"{path.stem}\n{grid.min():.6f} - {grid.max():.6f}", fontsize=10)
        ax.set_xlabel(horizontal_axis)
        ax.set_ylabel(vertical_axis)

    for ax in axes_array[len(items) :]:
        ax.axis("off")

    if last_image is not None:
        fig.colorbar(
            last_image,
            ax=axes_array[: len(items)],
            shrink=0.82,
            label="friction",
            extend=extend,
        )
    fig.suptitle(f"{items[0]['surface_label']} friction heatmaps, {scale_text}", fontsize=13)
    fig.savefig(out_dir / f"all_{surface}_friction_heatmaps.png")
    plt.close(fig)


def save_report(
    items: list[dict[str, object]],
    out_dir: Path,
    global_min: float,
    global_max: float,
) -> None:
    report_path = out_dir / f"{items[0]['surface_label']}_friction_report.csv"
    with report_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            "file,surface,horizontal_axis,vertical_axis,grid_width,grid_height,"
            "surface_points,min_friction,max_friction,global_min,global_max\n"
        )
        for item in items:
            path = item["path"]
            grid = item["grid"]
            h_values = item["horizontal_values"]
            v_values = item["vertical_values"]
            assert isinstance(path, Path)
            assert isinstance(grid, np.ndarray)
            assert isinstance(h_values, np.ndarray)
            assert isinstance(v_values, np.ndarray)
            handle.write(
                f"{path.name},{item['surface_label']},{item['horizontal_axis']},"
                f"{item['vertical_axis']},{len(h_values)},{len(v_values)},"
                f"{item['surface_count']},{grid.min():.12g},{grid.max():.12g},"
                f"{global_min:.12g},{global_max:.12g}\n"
            )


def process_surface(
    input_paths: list[Path],
    out_dir: Path,
    axis: str,
    side: str,
    args: argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    items = [extract_surface_grid(path, axis, side) for path in input_paths]
    global_min = min(float(item["grid"].min()) for item in items)  # type: ignore[union-attr]
    global_max = max(float(item["grid"].max()) for item in items)  # type: ignore[union-attr]
    full_data_min = global_min
    full_data_max = global_max
    for item in items:
        grid = item["grid"]
        assert isinstance(grid, np.ndarray)
        full_data_min = min(full_data_min, float(grid.min()))
        full_data_max = max(full_data_max, float(grid.max()))

    reference_grid = items[0]["grid"]
    assert isinstance(reference_grid, np.ndarray)
    overview_norm, scale_vmin, scale_vmax, scale_mode = make_norm(
        args,
        global_min,
        global_max,
        reference_grid,
    )
    scale_text = f"color scale {scale_vmin:.6f} - {scale_vmax:.6f} ({scale_mode})"
    extend = colorbar_extend(full_data_min, full_data_max, scale_vmin, scale_vmax)

    for item in items:
        grid = item["grid"]
        assert isinstance(grid, np.ndarray)
        norm, _, _, _ = make_norm(args, global_min, global_max, grid)
        if args.individual:
            save_individual_heatmap(item, out_dir, norm, args.cmap, args.dpi, extend)
        if args.csv:
            save_csv(item, out_dir)

    if not args.no_overview:
        save_overview(
            items,
            out_dir,
            overview_norm,
            args.cmap,
            args.dpi,
            scale_text,
            extend,
        )

    if args.csv:
        save_report(items, out_dir, global_min, global_max)

    print(f"Wrote {surface_label(axis, side)} heatmaps to: {out_dir}")
    print("Individual heatmaps: enabled" if args.individual else "Individual heatmaps: disabled")
    print("CSV output: disabled" if not args.csv else "CSV output: enabled")
    print(scale_text)
    if args.vmin is not None or args.vmax is not None:
        print(f"Requested clamp: {args.vmin if args.vmin is not None else global_min:.12g} to {args.vmax if args.vmax is not None else global_max:.12g}")
    print(f"Global {surface_label(axis, side)} friction range: {global_min:.12g} to {global_max:.12g}")
    for item in items:
        path = item["path"]
        grid = item["grid"]
        h_values = item["horizontal_values"]
        v_values = item["vertical_values"]
        assert isinstance(path, Path)
        assert isinstance(grid, np.ndarray)
        assert isinstance(h_values, np.ndarray)
        assert isinstance(v_values, np.ndarray)
        print(
            f"{path.name}: grid {len(h_values)} x {len(v_values)}, "
            f"range {grid.min():.12g} to {grid.max():.12g}"
        )


def main() -> None:
    args = parse_args()
    input_paths = expand_inputs(args.input)
    out_dir = Path(args.output)

    if args.all_surfaces:
        for axis in AXES:
            for side in ("min", "max"):
                process_surface(
                    input_paths,
                    out_dir / surface_label(axis, side),
                    axis,
                    side,
                    args,
                )
    else:
        process_surface(input_paths, out_dir, args.axis, args.side, args)


if __name__ == "__main__":
    main()
