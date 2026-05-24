#!/usr/bin/env bash

python visualization/generate_bottom_friction_heatmaps.py \
    fixed_init_0.35_stiffness_1e5_regularization_3000_left_right \
    --vmin 0.3 \
    --vmax 0.42

python /workspace/newton/replay_mujoco_contact_friction_trajectory.py \
    --checkpoint-path /workspace/outputs/fixed_init_0.35_stiffness_1e5_regularization_3000_left_right/fixed_init_0.35_stiffness_1e5_regularization_3000_left_right.npz \
    --trajectory-index 6 \
    --trajectory-npz /workspace/mujoco/outputs/block_force_dataset_fixed_init_20_long/block_force_dataset_fixed_init_20_long.npz \
    --reference-point-cloud /workspace/outputs/fixed_init_0.35_stiffness_1e5_regularization_3000_left_right/fixed_init_0.35_sparse_point_clouds_stiffness_1e5_regularization_3000_left_right/iter_000100.ply

python visualization/evaluate_mujoco_contact_friction_experiment.py \
      --experiment-name fixed_init_0.35_stiffness_1e5_regularization_3000_global \
      --eval-dataset /workspace/mujoco/outputs/block_force_dataset_fixed_init_20_long/block_force_dataset_fixed_init_20_long.npz \
      --device cuda:0



# 单个实验评估
# 只算 eval，不生成每条轨迹 replay：

python visualization/evaluate_mujoco_contact_friction_experiment.py \
    --experiment-name fixed_init_0.35_stiffness_1e5_regularization_3000_left_right \
    --eval-dataset mujoco/outputs/block_force_dataset_fixed_init_20_long/block_force_dataset_fixed_init_20_long.npz \
    --device cuda:0 \
    --eval-batch-size 20 \
    --skip-replay

# 算 eval 并 replay 指定轨迹：

python visualization/evaluate_mujoco_contact_friction_experiment.py \
    --experiment-name fixed_init_0.35_stiffness_1e5_regularization_3000_left_right \
    --eval-dataset mujoco/outputs/block_force_dataset_fixed_init_20_long/block_force_dataset_fixed_init_20_long.npz \
    --device cuda:0 \
    --eval-batch-size 20 \
    --replay-indices 0 1 2

# 单条轨迹 replay

python newton/replay_mujoco_contact_friction_trajectory.py \
    fixed_init_0.35_stiffness_1e5_regularization_3000_left_right \
    --trajectory-npz mujoco/outputs/block_force_dataset_fixed_init_20_long/block_force_dataset_fixed_init_20_long.npz \
    --trajectory-index 0 \
    --device cuda:0

# 摩擦热力图
# 对某个实验最终/中间 PLY 生成底面热力图：

python visualization/generate_bottom_friction_heatmaps.py \
    fixed_init_0.35_stiffness_1e5_regularization_3000_left_right \
    --axis z \
    --side min \
    --csv \
    --individual

# 对指定 PLY 或目录：

python visualization/generate_bottom_friction_heatmaps.py \
    --input report_assets/group_ply_inputs/global/*.ply \
    --output report_assets/group_heatmaps/global \
    --axis z \
    --side min \
    --csv

# 静态 top-down PNG
# 全部可发现 ckpt + long 全轨迹：

python visualization/plot_topdown_trajectory_overlays.py \
    --method-source all \
    --dataset mujoco/outputs/block_force_dataset_fixed_init_20_long/block_force_dataset_fixed_init_20_long.npz \
    --output report_assets/topdown_trajectory_overlays_fixed20_long_all_discovered_ckpts.png \
    --summary-output report_assets/topdown_trajectory_overlays_fixed20_long_all_discovered_ckpts_summary.json \
    --all-trajectories \
    --max-steps none \
    --eval-batch-size 20 \
    --device cuda:0

# 只画固定 curated 20 个：

python visualization/plot_topdown_trajectory_overlays.py \
    --method-source curated \
    --dataset mujoco/outputs/block_force_dataset_fixed_init_20_long/block_force_dataset_fixed_init_20_long.npz \
    --output report_assets/topdown_curated_long.png \
    --all-trajectories \
    --max-steps none \
    --device cuda:0

# 交互式 top-down HTML

python visualization/plot_topdown_trajectory_overlays_interactive.py \
    --method-source all \
    --dataset mujoco/outputs/block_force_dataset_fixed_init_20_long/block_force_dataset_fixed_init_20_long.npz \
    --output report_assets/topdown_trajectory_overlays_fixed20_long_all_discovered_ckpts_interactive.html \
    --summary-output report_assets/topdown_trajectory_overlays_fixed20_long_all_discovered_ckpts_interactive_summary.json \
    --all-trajectories \
    --max-steps none \
    --eval-batch-size 20 \
    --device cuda:0

# 如果已经有 summary，不想重跑 Newton：

python visualization/plot_topdown_trajectory_overlays_interactive.py \
    --reuse-summary report_assets/topdown_trajectory_overlays_fixed20_long_all_discovered_ckpts_interactive_summary.json \
    --output report_assets/topdown_cached.html

# 2D 对比视频
# 单 checkpoint：

python visualization/render_mujoco_newton_comparison_video.py \
    fixed_init_0.35_stiffness_1e5_regularization_3000_left_right \
    --trajectory-npz mujoco/outputs/block_force_dataset_fixed_init_20_long/block_force_dataset_fixed_init_20_long.npz \
    --trajectory-index 0 \
    --output outputs/comparison_videos/left_right_traj0.mp4 \
    --device cuda:0

# 多个 checkpoint overlay：

python visualization/render_mujoco_newton_comparison_video.py \
    fixed_init_0.3_stiffness_1e5_regularization_3000_global \
    fixed_init_0.35_stiffness_1e5_regularization_3000_left_right \
    --labels global left-right \
    --trajectory-npz mujoco/outputs/block_force_dataset_fixed_init_20_long/block_force_dataset_fixed_init_20_long.npz \
    --trajectory-index 0 \
    --output outputs/comparison_videos/global_vs_lr_traj0.mp4 \
    --device cuda:0

# 3D Plotly HTML

python visualization/render_mujoco_newton_comparison_3d.py \
    fixed_init_0.35_stiffness_1e5_regularization_3000_left_right \
    --trajectory-npz mujoco/outputs/block_force_dataset_fixed_init_20_long/block_force_dataset_fixed_init_20_long.npz \
    --trajectory-index 0 \
    --output outputs/comparison_3d/left_right_traj0_3d.html \
    --device cuda:0
