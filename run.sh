python /workspace/newton/fit_mujoco_contact_point_friction.py \
    --trajectory-npz /workspace/mujoco/outputs/block_force_dataset_2000_random_init.npz \
    --checkpoint-path outputs/friction_fit_random_init_sparse_ckpt9.npz \
    --point-cloud-path outputs/friction_fit_random_init_sparse_point_cloud9.ply \
    --checkpoint-point-cloud-dir outputs/friction_fit_random_init_sparse_point_clouds9 \
    --device cuda:0 \
    --batch-size 256 \
    --eval-batch-size 32 \
    --opt-iters 10000 \
    --learning-rate 1e-4 \
    --log-every 1 \
    --wandb \
    --wandb-project newton-contact-point-friction-fit \
    --wandb-run-name mujoco-fit-random-init-sparse-002 \
    --checkpoint-every 100 \
    --grad-clip-norm 1.0 \
    --surface-point-spacing 0.01 \
    --point-friction 0.3

python /workspace/newton/replay_mujoco_contact_friction_trajectory.py \
    --checkpoint-path /workspace/outputs/friction_fit_random_init_sparse_ckpt8.npz \
    --trajectory-index 225 \
    --trajectory-npz /workspace/mujoco/outputs/block_force_dataset_2000_random_init.npz