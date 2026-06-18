from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "mujoco" / "outputs" / "object_physics_latent_box_partitions_48x2000_min300"
FAMILY_DESCRIPTIONS = {
    "left_right": "local x=0 split: two 0.10 x 0.10 squares",
    "front_back": "local y=0 split: two 0.20 x 0.05 narrow rectangles",
    "center_ends": "left/right ends share friction; center uses another friction",
}


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_manifest_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def geom_topdown(scene_path: Path, friction_by_geom: dict[str, list[float]]) -> list[dict[str, object]]:
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "push_block")
    if body_id < 0:
        raise ValueError(f"{scene_path} does not contain body 'push_block'")
    geom_start = int(model.body_geomadr[body_id])
    geom_count = int(model.body_geomnum[body_id])
    geoms: list[dict[str, object]] = []
    for geom_id in range(geom_start, geom_start + geom_count):
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        mu_values = friction_by_geom.get(str(name), model.geom_friction[geom_id].astype(float).tolist())
        geoms.append(
            {
                "name": str(name),
                "center_xy": model.geom_pos[geom_id, :2].astype(float),
                "half_size_xy": model.geom_size[geom_id, :2].astype(float),
                "mu": float(mu_values[0]),
            }
        )
    return geoms


def short_geom_label(name: str) -> str:
    return str(name).replace("push_block_", "").replace("_", " ")


def quaternion_wxyz_to_yaw(quaternion: np.ndarray) -> float:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64).reshape(4)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def episode_preview(dataset_path: Path, episode_index: int = 0) -> dict[str, np.ndarray | float]:
    with np.load(dataset_path, allow_pickle=True) as data:
        columns = [str(value) for value in data["columns"].tolist()]
        column = {name: index for index, name in enumerate(columns)}
        length = int(np.asarray(data["episode_lengths"], dtype=np.int32)[episode_index])
        values = np.asarray(data["trajectories"][episode_index, :length], dtype=np.float32)
        point_offset = np.asarray(data["point_offset_local"][episode_index], dtype=np.float32)
        force_world = np.asarray(data["force_world"][episode_index], dtype=np.float32)
        initial_quaternion = np.asarray(data["initial_quaternion"][episode_index], dtype=np.float32)
    positions = values[:, [column["pos_x"], column["pos_y"]]]
    yaw = quaternion_wxyz_to_yaw(initial_quaternion)
    rotation = np.asarray(
        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
        dtype=np.float32,
    )
    force_point_world = positions[0] + rotation @ point_offset[:2]
    return {
        "positions": positions,
        "force_world": force_world[:2],
        "force_point_world": force_point_world,
        "initial_yaw": yaw,
    }


def render_object_preview(
    *,
    record: dict,
    manifest_path: Path,
    output_path: Path,
    mu_min: float,
    mu_max: float,
    trajectory_episode_index: int,
) -> dict[str, object]:
    dataset_path = resolve_manifest_path(str(record["trajectory_npz"]), manifest_path)
    metadata_path = dataset_path.with_suffix(".json")
    metadata = load_json(metadata_path)
    family = str(metadata["friction_partition_family"])
    friction_regions = dict(metadata["friction_region_values"])
    friction_by_geom = dict(metadata["block_friction"])
    geoms = geom_topdown(Path(metadata["scene_path"]), friction_by_geom)
    trajectory = episode_preview(dataset_path, episode_index=int(trajectory_episode_index))

    cmap = colormaps["turbo"]
    norm = Normalize(vmin=float(mu_min), vmax=float(mu_max))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4), constrained_layout=True)
    fig.patch.set_facecolor("#f6f7f9")

    ax = axes[0]
    ax.set_facecolor("#eef1f5")
    for geom in geoms:
        center = np.asarray(geom["center_xy"], dtype=np.float64)
        half = np.asarray(geom["half_size_xy"], dtype=np.float64)
        mu = float(geom["mu"])
        rect = Rectangle(
            center - half,
            2.0 * half[0],
            2.0 * half[1],
            facecolor=cmap(norm(mu)),
            edgecolor="#111827",
            linewidth=2.2,
        )
        ax.add_patch(rect)
        ax.text(
            center[0],
            center[1],
            f"{short_geom_label(str(geom['name']))}\nmu={mu:.3f}",
            ha="center",
            va="center",
            fontsize=9,
            color="white" if norm(mu) > 0.52 else "#111827",
            weight="bold",
        )
    ax.annotate("", xy=(0.115, -0.065), xytext=(0.055, -0.065), arrowprops={"arrowstyle": "->", "lw": 1.8})
    ax.annotate("", xy=(-0.115, 0.065), xytext=(-0.115, 0.005), arrowprops={"arrowstyle": "->", "lw": 1.8})
    ax.text(0.118, -0.065, "+x", va="center", fontsize=9)
    ax.text(-0.115, 0.068, "+y", ha="center", fontsize=9)
    ax.set_xlim(-0.14, 0.14)
    ax.set_ylim(-0.085, 0.085)
    ax.set_aspect("equal")
    ax.set_xlabel("local x (m)")
    ax.set_ylabel("local y (m)")
    ax.set_title("Top-view friction partition")
    ax.grid(alpha=0.16)

    ax = axes[1]
    positions = np.asarray(trajectory["positions"], dtype=np.float32)
    force_world = np.asarray(trajectory["force_world"], dtype=np.float32)
    force_point_world = np.asarray(trajectory["force_point_world"], dtype=np.float32)
    ax.plot(positions[:, 0], positions[:, 1], color="#2563eb", linewidth=2.2, label="block center trajectory")
    ax.scatter(positions[0, 0], positions[0, 1], color="#16a34a", s=55, zorder=3, label="start")
    ax.scatter(positions[-1, 0], positions[-1, 1], color="#dc2626", s=55, zorder=3, label="end")
    force_norm = float(np.linalg.norm(force_world))
    arrow = force_world / max(force_norm, 1.0e-8) * 0.045
    ax.quiver(
        [force_point_world[0]],
        [force_point_world[1]],
        [arrow[0]],
        [arrow[1]],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="#f59e0b",
        width=0.012,
        label="initial force",
    )
    ax.scatter(force_point_world[0], force_point_world[1], marker="x", color="#111827", s=55, zorder=4)
    margin = 0.04
    min_xy = np.minimum(np.min(positions, axis=0), force_point_world) - margin
    max_xy = np.maximum(np.max(positions, axis=0), force_point_world) + margin
    span = np.maximum(max_xy - min_xy, 0.12)
    center = 0.5 * (min_xy + max_xy)
    radius = 0.55 * float(np.max(span))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_aspect("equal")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_title(f"Example trajectory #{int(trajectory_episode_index)}")
    ax.grid(alpha=0.22)
    ax.legend(loc="best", fontsize=8)

    title = (
        f"{record['object_id']}\n"
        f"family={family} | object split={record['object_split']} | "
        + ", ".join(f"{name} mu={float(value):.3f}" for name, value in sorted(friction_regions.items()))
    )
    fig.suptitle(title, fontsize=12, weight="bold")
    scalar_mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(scalar_mappable, ax=axes, orientation="horizontal", fraction=0.055, pad=0.08)
    colorbar.set_label("MuJoCo sliding friction coefficient")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {
        "object_id": record["object_id"],
        "family": family,
        "object_split": record["object_split"],
        "friction_regions": friction_regions,
        "preview": str(output_path.resolve()),
        "dataset": str(dataset_path.resolve()),
    }


def create_contact_sheet(
    preview_paths: list[Path],
    *,
    title: str,
    output_path: Path,
    columns: int = 4,
) -> None:
    if not preview_paths:
        return
    from PIL import Image, ImageDraw

    images = [Image.open(path).convert("RGB") for path in preview_paths]
    thumb_width = 480
    thumb_height = int(images[0].height * thumb_width / images[0].width)
    rows = int(np.ceil(len(images) / int(columns)))
    header_height = 70
    sheet = Image.new("RGB", (columns * thumb_width, header_height + rows * thumb_height), "#f6f7f9")
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 20), title, fill="#111827")
    for index, image in enumerate(images):
        image.thumbnail((thumb_width, thumb_height))
        x = (index % columns) * thumb_width
        y = header_height + (index // columns) * thumb_height
        sheet.paste(image, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    for image in images:
        image.close()


def write_gallery_html(records: list[dict[str, object]], *, output_path: Path) -> None:
    cards = []
    for record in records:
        preview = Path(str(record["preview"]))
        relative_preview = preview.relative_to(output_path.parent)
        cards.append(
            f"""
            <article class="card">
              <a href="{relative_preview.as_posix()}"><img src="{relative_preview.as_posix()}" loading="lazy"></a>
              <h3>{record['object_id']}</h3>
              <p>family: <b>{record['family']}</b> | split: <b>{record['object_split']}</b></p>
              <p>{json.dumps(record['friction_regions'], sort_keys=True)}</p>
            </article>
            """
        )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Object Physics Latent Dataset Gallery</title>
<style>
body {{ font-family: sans-serif; margin: 24px; background: #f6f7f9; color: #111827; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 18px; }}
.card {{ background: white; padding: 12px; border-radius: 10px; box-shadow: 0 1px 5px #0002; }}
.card img {{ width: 100%; border-radius: 6px; }}
h3 {{ font-size: 14px; overflow-wrap: anywhere; }}
p {{ font-size: 13px; }}
</style>
</head>
<body>
<h1>Object Physics Latent Dataset Gallery</h1>
<p>48 objects, each with 2000 trajectories. Color indicates the actual MuJoCo sliding friction coefficient.</p>
<section class="grid">{''.join(cards)}</section>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mu-min", type=float, default=0.10)
    parser.add_argument("--mu-max", type=float, default=0.70)
    parser.add_argument("--trajectory-episode-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    manifest_path = dataset_root / "manifest.json"
    manifest = load_json(manifest_path)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else dataset_root / "previews"
    )
    object_dir = output_dir / "objects"
    records = []
    for index, record in enumerate(manifest["objects"], start=1):
        output_path = object_dir / f"{record['object_id']}.png"
        rendered = render_object_preview(
            record=record,
            manifest_path=manifest_path,
            output_path=output_path,
            mu_min=float(args.mu_min),
            mu_max=float(args.mu_max),
            trajectory_episode_index=int(args.trajectory_episode_index),
        )
        records.append(rendered)
        print(f"[{index}/{len(manifest['objects'])}] {output_path}", flush=True)

    by_family: dict[str, list[Path]] = {}
    for record in records:
        by_family.setdefault(str(record["family"]), []).append(Path(str(record["preview"])))
    for family, preview_paths in sorted(by_family.items()):
        create_contact_sheet(
            preview_paths,
            title=f"{family}: {FAMILY_DESCRIPTIONS.get(family, '')}",
            output_path=output_dir / f"{family}_contact_sheet.png",
            columns=4,
        )
    create_contact_sheet(
        [Path(str(record["preview"])) for record in records],
        title="All 48 object friction configurations",
        output_path=output_dir / "all_objects_contact_sheet.png",
        columns=4,
    )
    write_gallery_html(records, output_path=output_dir / "gallery.html")
    summary = {
        "dataset_root": str(dataset_root),
        "objects": len(records),
        "families": dict(Counter(str(record["family"]) for record in records)),
        "preview_dir": str(output_dir),
        "gallery": str((output_dir / "gallery.html").resolve()),
    }
    (output_dir / "preview_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
