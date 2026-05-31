# Residual Dynamics Adapter

This folder contains the first closed-loop residual dynamics experiment described in
`docs/residual_dynamics_adapter_design.md`.

The adapter freezes a learned contact-friction checkpoint and trains only a small
Warp MLP:

```text
11 -> 128 -> 128 -> 64 -> 3
```

The output is bounded residual planar dynamics:

```text
[delta_a_body_x, delta_a_body_y, delta_alpha_z]
```

The position term uses the same surface-point squared-distance loss as the
friction fitting/evaluation path. Use `--point-position-loss-reduction mean`
or `sum` to match the desired friction-eval reduction.

The main entry point is:

```bash
python residual_dynamics_adapter/train_residual_adapter.py \
  --friction-checkpoint outputs/rotation_l0p20_r0p50_2000_global_m300_rotloss_angvel/rotation_l0p20_r0p50_2000_global_m300_rotloss_angvel.npz \
  --trajectory-npz mujoco/outputs/rotation_friction_diagnostics_l0p20_r0p50_2000/same_mean_split_left_0p20_right_0p50/same_mean_split_left_0p20_right_0p50.npz \
  --experiment-dir outputs/residual_global_m50 \
  --device cuda:0 \
  --max-steps 50
```

To train friction and the residual adapter end-to-end from scratch, omit
`--friction-checkpoint` and enable joint friction training:

```bash
python residual_dynamics_adapter/train_residual_adapter.py \
  --train-friction-end-to-end \
  --friction-parameterization global \
  --point-friction 0.35 \
  --trajectory-npz mujoco/outputs/rotation_friction_diagnostics_l0p20_r0p50_2000/same_mean_split_left_0p20_right_0p50/same_mean_split_left_0p20_right_0p50.npz \
  --experiment-dir outputs/residual_e2e_global_m50 \
  --device cuda:0 \
  --max-steps 50
```

You can also pass `--friction-checkpoint` together with
`--train-friction-end-to-end` to initialize friction from an existing checkpoint
and continue optimizing it jointly with the residual MLP.

End-to-end friction supports the same optimizer parameterizations as the friction
fitting path: `point`, `global`, `left-right`, and `base-delta`. Checkpoint
resume expects the same active contact-point set, so keep scene sampling,
trajectory source, and contact-mask settings consistent.

For the main comparison, train separate adapters on top of the frozen global and
spatial friction checkpoints, then evaluate them with the same rotation-aware loss
weights:

```text
position=1.0, yaw=1.0, linear velocity=0.0, angular velocity z=0.1
```
