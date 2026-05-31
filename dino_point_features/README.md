# DINO Point Features

This folder assigns a DINO feature vector to each 3-D point produced by the
MuJoCo multi-view point-cloud pipeline.

The implementation follows the useful part of PointWorld's `scene_featurizer.py`:

1. render or load synchronized RGB-D camera views;
2. extract ViT patch tokens from each RGB image with DINO;
3. project each world-space 3-D point into every camera;
4. keep only projections that pass image bounds and depth consistency checks;
5. bilinearly sample the DINO patch-token grid at the projected pixel;
6. average valid camera features into one feature vector per point.

The output is a sidecar NPZ, not a PLY, because DINO features are high
dimensional.

## Newton surface-point example

To assign DINO features to the same box surface-point grid used by the Newton
friction fitting path, run:

```bash
env MUJOCO_GL=osmesa python dino_point_features/run_block_force_dino_surface_points.py \
    --output-dir outputs/mujoco_dino_point_features/block_force_surface_spacing_0p01_dinov2_layers \
    --dino-model dinov2_vits14 \
    --selected-layers 2,5,8,11 \
    --surface-point-spacing 0.01 \
    --width 320 \
    --height 240 \
    --num-steps 0
```

This writes `newton_surface_points.ply` plus
`newton_surface_points_dino_features.npz`. The NPZ includes both world-space
`points` and `local_points`, so these features can be matched back to the
Newton friction/contact point indices. By default, bottom contact-face points
copy the DINO feature from the top-face point with the same local `x/y`; use
`--bottom-feature-source projected` to keep the direct projection/fallback
result instead.

## MuJoCo block-force example

Use the public DINOv2 backbone for an accessible run:

```bash
env MUJOCO_GL=osmesa python dino_point_features/run_block_force_dino_pointcloud.py \
    --output-dir outputs/mujoco_dino_point_features/block_force_dinov2 \
    --segmentation-backend color-threshold \
    --dino-model dinov2_vits14 \
    --width 320 \
    --height 240 \
    --num-steps 0
```

For closer parity with PointWorld's released encoder, install the local DINOv3
submodule and checkpoint, then run:

```bash
env MUJOCO_GL=osmesa python dino_point_features/run_block_force_dino_pointcloud.py \
    --output-dir outputs/mujoco_dino_point_features/block_force_dinov3 \
    --segmentation-backend color-threshold \
    --dino-model dinov3_vitl16 \
    --dinov3-repo PointWorld/third_party/dinov3 \
    --dinov3-weights PointWorld/third_party/dinov3/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
    --width 320 \
    --height 240 \
    --num-steps 0
```

`PointWorld/third_party/dinov3` in this workspace currently needs the actual
DINOv3 checkout plus weights before the DINOv3 command can run.

## Outputs

Each frame directory contains the original point clouds and DINO sidecars:

```text
frame_000000/
  merged_segments.ply
  merged_segments_dino_features.npz
  block.ply
  block_dino_features.npz
```

Important NPZ arrays:

- `points`: `(N, 3)` world-space point coordinates.
- `dino_features`: `(N, D)` per-point DINO feature vectors.
- `visibility_counts`: number of cameras that contributed to each point.
- `primary_camera_ids`: first contributing camera id, or `-1` if none passed.
- `depth_fallback_used`: true when a point missed the strict depth threshold
  in all cameras and used the nearest-depth projected camera instead.
- `colors`, `segmentation_ids`, `camera_ids`, `track_ids`: original point-cloud metadata.

By default, nearest-depth fallback is enabled so fused or voxel-averaged points
still receive a DINO vector. Pass `--no-depth-fallback` to keep only the strict
PointWorld-style depth-visible features; in that mode points with
`visibility_counts[i] == 0` receive an all-zero feature vector.

## Visualization

Create a quick PNG summary from any sidecar NPZ:

```bash
python dino_point_features/visualize_point_features.py \
    outputs/mujoco_dino_point_features/block_force_dinov2/frame_000000/block_dino_features.npz
```

The figure shows RGB top/side projections, strict-vs-fallback feature assignment,
and a PCA color projection of the DINO vectors.
