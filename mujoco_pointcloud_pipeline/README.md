# Real-World-Style Multi-View Point-Cloud Pipeline

This folder treats MuJoCo as an RGB-D camera source, not as an oracle. The
object mask comes from a vision backend, then depth is back-projected, fused
across cameras, cropped, and cleaned with RANSAC table-plane removal.

The code mirrors the real-world path used by `/workspace/pgnd`:

1. Multi-view RGB-D frames.
2. GroundingDINO object detection from text prompts.
3. SAM2 instance masks from the detected boxes.
4. Depth-mask back-projection using camera intrinsics/extrinsics.
5. Multi-camera fusion and voxel downsampling.
6. RANSAC table-plane estimation from the full workspace depth cloud.
7. Object cloud cleanup, PLY export, and simple centroid tracking.

## MuJoCo RGB-D Source

Use this when validating the perception geometry in simulation. It does not use
MuJoCo geom/body segmentation.

```bash
env MUJOCO_GL=osmesa python mujoco_pointcloud_pipeline/run_block_force_multiview_pointcloud.py \
    --output-dir outputs/mujoco_multiview_pointcloud/block_force_grounded_sam2 \
    --object "block" \
    --width 320 \
    --height 240 \
    --num-steps 0 \
    --segmentation-backend grounded-sam2 \
    --sam2-checkpoint /workspace/pgnd/weights/sam2/sam2.1_hiera_large.pt
```

If the model weights are not installed, the synthetic smoke-test backend can
validate the RGB-D and RANSAC path without using MuJoCo segmentation:

```bash
env MUJOCO_GL=osmesa python mujoco_pointcloud_pipeline/run_block_force_multiview_pointcloud.py \
    --output-dir outputs/mujoco_multiview_pointcloud/block_force_color_smoke \
    --object "block" \
    --width 96 \
    --height 72 \
    --num-steps 0 \
    --segmentation-backend color-threshold
```

## PGND-Style Real RGB-D Folder

For a real recording laid out like PGND:

```text
episode_0000/
  calibration/intrinsics.npy
  calibration/rvecs.npy
  calibration/tvecs.npy
  camera_0/rgb/000000.jpg
  camera_0/depth/000000.png
  camera_1/rgb/000000.jpg
  camera_1/depth/000000.png
```

run:

```bash
python mujoco_pointcloud_pipeline/run_rgbd_folder_pointcloud.py \
    --episode-dir /path/to/episode_0000 \
    --output-dir /path/to/episode_0000/object_pointclouds \
    --camera 0 \
    --camera 1 \
    --object "white cotton rope" \
    --segmentation-backend grounded-sam2 \
    --sam2-checkpoint /workspace/pgnd/weights/sam2/sam2.1_hiera_large.pt \
    --workspace-bounds -0.5 0.8 -0.5 0.5 -0.02 0.5
```

If masks were already produced by PGND/SAM2 under `camera_i/mask/`, reuse them:

```bash
python mujoco_pointcloud_pipeline/run_rgbd_folder_pointcloud.py \
    --episode-dir /path/to/episode_0000 \
    --object "white cotton rope" \
    --segmentation-backend saved-mask \
    --mask-root /path/to/episode_0000
```

## Outputs

Each processed frame writes:

- `frame_XXXXXX/merged_segments.ply`
- `frame_XXXXXX/<object_name>.ply`
- `tracks.json`
- `metadata.json`

PLY fields:

```text
x y z red green blue segmentation_id camera_id track_id
```

Coordinates are in the shared world/calibration frame. `segmentation_id` is the
pipeline object label, not a MuJoCo geom id.

## Dependencies

For the GroundingDINO/SAM2 backend, follow PGND's dependency setup:

```bash
pip install iopath
pip install segment-anything
pip install --no-deps git+https://github.com/IDEA-Research/GroundingDINO
pip install --no-deps git+https://github.com/facebookresearch/sam2
```

The PGND README expects weights under `/workspace/pgnd/weights/`.
