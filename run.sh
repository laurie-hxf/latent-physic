python newton/fit_mujoco_contact_point_friction.py \
  --trajectory-npz /workspace/mujoco/outputs/block_force_dataset_2000.npz \
  --checkpoint-path outputs/friction_fit_ckpt.npz \
  --device cuda:0 \
  --batch-size 512 \
  --eval-batch-size 32 \
  --opt-iters 10000 \
  --learning-rate 1e-4 \
  --log-every 1 \
  --wandb \
  --wandb-project newton-contact-point-friction-fit \
  --wandb-run-name mujoco-fit-006 \
  --checkpoint-every 100

python /workspace/newton/fit_mujoco_contact_point_friction.py \
    --trajectory-npz /workspace/mujoco/outputs/block_force_dataset_2000_random_init.npz \
    --checkpoint-path outputs/friction_fit_random_init_sparse_ckpt.npz \
    --point-cloud-path outputs/friction_fit_random_init_sparse_point_cloud.ply \
    --checkpoint-point-cloud-dir outputs/friction_fit_random_init_sparse_point_clouds \
    --device cuda:0 \
    --max-steps 500 \
    --batch-size 128 \
    --eval-batch-size 32 \
    --opt-iters 10000 \
    --learning-rate 1e-4 \
    --log-every 1 \
    --wandb \
    --wandb-project newton-contact-point-friction-fit \
    --wandb-run-name mujoco-fit-random-init-sparse-002 \
    --checkpoint-every 100