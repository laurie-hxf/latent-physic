python /workspace/newton/fit_mujoco_contact_point_friction.py \
    --trajectory-npz /workspace/mujoco/outputs/block_force_dataset_fixed_init_2000.npz \
    --checkpoint-path outputs/fixed_init_0.35_stiffness_1e5_regularization_3000_global.npz \
    --point-cloud-path outputs/fixed_init_0.35_stiffness_1e5_regularization_3000_global.ply \
    --checkpoint-point-cloud-dir outputs/fixed_init_0.35_sparse_point_clouds_stiffness_1e5_regularization_3000_global \
    --device cuda:0 \
    --batch-size 256 \
    --eval-batch-size 32 \
    --opt-iters 20000 \
    --learning-rate 1e-4 \
    --log-every 1 \
    --wandb \
    --wandb-project newton-contact-point-friction-fit \
    --wandb-run-name fixed_init_0.35_stiffness_1e5_regularization_3000_global \
    --checkpoint-every 100 \
    --grad-clip-norm 1.0 \
    --surface-point-spacing 0.01 \
    --contact-stiffness 1e5 \
    --piecewise-regularization-weight 3000 \
    --max-steps 300 \
    --friction-parameterization global \
    --resume-checkpoint outputs/fixed_init_0.35_stiffness_1e5_regularization_3000_global.npz \
    --point-friction 0.35 &


python /workspace/newton/fit_mujoco_contact_point_friction.py \
    --trajectory-npz /workspace/mujoco/outputs/block_force_dataset_fixed_init_2000.npz \
    --checkpoint-path outputs/fixed_init_0.5_stiffness_1e5_regularization_3000_global.npz \
    --point-cloud-path outputs/fixed_init_0.5_stiffness_1e5_regularization_3000_global.ply \
    --checkpoint-point-cloud-dir outputs/fixed_init_0.5_sparse_point_clouds_stiffness_1e5_regularization_3000_global \
    --device cuda:0 \
    --batch-size 256 \
    --eval-batch-size 32 \
    --opt-iters 20000 \
    --learning-rate 1e-4 \
    --log-every 1 \
    --wandb \
    --wandb-project newton-contact-point-friction-fit \
    --wandb-run-name fixed_init_0.5_stiffness_1e5_regularization_3000_global \
    --checkpoint-every 100 \
    --grad-clip-norm 1.0 \
    --surface-point-spacing 0.01 \
    --contact-stiffness 1e5 \
    --piecewise-regularization-weight 3000 \
    --max-steps 300 \
    --friction-parameterization global \
    --point-friction 0.5 &

# env MUJOCO_GL=osmesa python mujoco/scripts/run_block_force_demo.py \
#     --num-episodes 2000 \
#     --no-randomize-initial-pose \
#     --dataset-path mujoco/outputs/block_force_dataset_fixed_init_2000.npz \
#     --metadata-path mujoco/outputs/block_force_dataset_fixed_init_2000.json \
#     --save-episode-videos \
#     --episode-video-dir mujoco/outputs/block_force_dataset_fixed_init_2000_videos \
#     --progress-every 100 &

wait

# python generate_bottom_friction_heatmaps.py \
#     --input /workspace/outputs/fixed_init_0.35_sparse_point_clouds_stiffness_1e5_regularization_3000_left_right \
#     --output /workspace/outputs/fixed_init_0.35_sparse_point_clouds_stiffness_1e5_regularization_3000_left_right/heatmap \
#     --vmin 0.3 \
#     --vmax 0.42

# python /workspace/newton/replay_mujoco_contact_friction_trajectory.py \
#     --checkpoint-path /workspace/outputs/fixed_init_0.35_stiffness_1e5_regularization_3000.npz \
#     --trajectory-index 1 \
#     --trajectory-npz /workspace/mujoco/outputs/block_force_dataset_fixed_init_20.npz \
#     --reference-point-cloud /workspace/outputs/fixed_init_0.35_sparse_point_clouds_stiffness_1e5_regularization_3000/iter_000100.ply