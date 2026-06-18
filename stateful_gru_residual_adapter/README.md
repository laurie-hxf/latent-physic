# Stateful GRU Residual Adapter

This package contains the deterministic, stateful GRU residual model and its
trainer. It does not use a PointNet encoder. It reuses adapter-agnostic feature,
friction, Newton rollout, checkpoint, and evaluation utilities from
`pointnet_residual_adapter`.

The default architecture is:

```text
raw surface-point features -> mean-max pooling
-> 2-layer GRU, hidden size 16
-> linear deterministic residual head
```

The deterministic output head supports three experiment modes:

```text
velocity:      [delta_v_body_x, delta_v_body_y, delta_omega_z]
pose:          [delta_x_body, delta_y_body, delta_yaw]
pose_velocity: pose and velocity outputs concatenated into 6 dimensions
```

The hidden state persists across rollout steps. Training uses a burn-in segment
followed by a truncated-BPTT chunk. The trainer saves:

```text
<name>_best_pretrain.pt
<name>_best_closed_loop.pt
<name>.pt                 # canonical best closed-loop checkpoint
<name>_last.pt
```

Training never launches evaluation automatically.

The default closed-loop residual gain is `0.02`, matching the current
conservative PointNet-residual comparison setting. Checkpoints store this gain;
the unified evaluation interface uses the stored value unless
`--pointnet-residual-gain` explicitly overrides it.

Run the default training configuration:

```bash
./run_stateful_gru_residual_adapter.sh
```

The launcher follows `run_train_eval_template.sh`: it creates a timestamped
experiment root, sequentially trains the three output modes as separate W&B
runs, evaluates their canonical checkpoints, renders saved rollouts, validates
the eval artifacts, and refreshes the historical registry.

Evaluate the canonical best closed-loop checkpoint explicitly:

```bash
python visualization/evaluate_experiments.py \
  --dataset mujoco/outputs/rotation_friction_diagnostics_l0p20_r0p50_68/same_mean_split_left_0p20_right_0p50/same_mean_split_left_0p20_right_0p50.npz \
  --eval-name rotation68_stateful \
  --method-source auto \
  --checkpoint-root outputs/stateful_gru_residual/stateful_gru2_h16_b32_t64 \
  --stateful-reset-interval 0
```

Repeat the same evaluation with `--stateful-reset-interval 4` to test whether
performance depends on memory longer than the old explicit `H=4` window.
