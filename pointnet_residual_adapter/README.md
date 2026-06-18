# Supervised PointNet Residual Adapter

This module trains a supervised PointNet residual model separate from
`residual_dynamics_adapter/`.

The default training target is explicit:

```text
target = MuJoCo rigid velocity - Newton rigid velocity
```

The model consumes Newton surface-point history features and predicts body-frame
planar residuals:

```text
[delta_v_body_x, delta_v_body_y, delta_omega_z]
```

`--residual-output-mode velocity` keeps this target as a velocity delta and
rollout applies:

```text
v_next = Newton(v_current) + pointnet_residual_gain * delta_v
```

`--residual-output-mode acceleration` trains on `delta_v / dt` and rollout
applies:

```text
v_next = Newton(v_current) + pointnet_residual_gain * dt * acceleration_residual
```

Older checkpoints without `residual_output_mode` metadata are interpreted as
`velocity`.

With `--history-window-steps H`, the model sees the latest `H` feature frames
ending at the current frame and predicts the one-step residual for the next
frame. For example, `H=4,P=1` trains
`feature_{t-3}, feature_{t-2}, feature_{t-1}, feature_t -> delta_v_{t+1}`.

Newton rollout remains Warp/Newton. PointNet training and inference use PyTorch.
There is no gradient through Newton in this first implementation.

## Train

```bash
python pointnet_residual_adapter/train_supervised_pointnet_residual.py \
    --trajectory-npz mujoco/outputs/rotation_friction_diagnostics_l0p20_r0p50_2000/same_mean_split_left_0p20_right_0p50/same_mean_split_left_0p20_right_0p50.npz \
    --friction-checkpoint /workspace/outputs/20260531_053758_rotation_l0p20_r0p50_2000_dino_mlp_m300_posonly/20260531_053758_rotation_l0p20_r0p50_2000_dino_mlp_m300_posonly.npz \
    --dino-feature-npz /workspace/outputs/mujoco_dino_point_features/block_force_surface_spacing_0p01_dinov2_layers/frame_000000/newton_surface_points_dino_features.npz \
    --experiment-dir outputs/pointnet_residual/dino_mlp_h4_p1 \
    --history-window-steps 4 \
    --prediction-window-steps 1 \
    --residual-output-mode velocity \
    --batch-size 64 \
    --opt-iters 10000 \
    --device cuda:0
```

For the friction-only ablation, add:

```bash
--without-dino
```

## Rollout

```bash
python pointnet_residual_adapter/rollout_pointnet_residual.py \
    --adapter-checkpoint outputs/pointnet_residual/dino_mlp_h4_p1/dino_mlp_h4_p1.pt \
    --trajectory-npz mujoco/outputs/rotation_friction_diagnostics_l0p20_r0p50_68/same_mean_split_left_0p20_right_0p50/same_mean_split_left_0p20_right_0p50.npz \
    --trajectory-index 0 \
    --device cuda:0
```

The rollout script compares the frozen Newton prediction against the closed-loop
PointNet residual rollout and writes an `.npz` summary under
`<checkpoint dir>/rollout/` by default.

By default rollout uses the checkpoint's `residual_output_mode` metadata. Use
`--pointnet-residual-output-mode velocity` or `acceleration` only for explicit
diagnostics.

## Implemented Milestone

- Online Newton supervised data generation.
- Frozen friction conditioning from checkpoint, PLY, or constant fallback.
- Optional aligned DINO sidecar features with fail-fast grid checks.
- PointNet shared point MLP with masked mean+max pooling.
- Supervised residual loss with horizon discounting, magnitude regularization,
  and smoothness regularization.
- Receding-horizon inference using velocity residuals by default, with an
  acceleration-output diagnostic mode.

Cached supervised datasets and Warp-only PointNet are intentionally left for a
later milestone.
