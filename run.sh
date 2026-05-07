python newton/fit_mujoco_contact_point_friction.py \
    --trajectory-npz /workspace/mujoco/outputs/block_force_dataset_2000.npz \
    --device cuda:0 \
    --max-steps 200 \
    --max-trajectories 256 \
    --batch-size 16 \
    --eval-batch-size 32 \
    --opt-iters 20 \
    --learning-rate 1e-4 \
    --log-every 1 \
    --trajectory-progress-every 4 \
    --wandb \
    --wandb-run-name mujoco-fit-003

python newton/fit_mujoco_contact_point_friction.py \
  --trajectory-npz /workspace/mujoco/outputs/block_force_dataset_2000.npz \
  --device cuda:0 \
  --batch-size 32 \
  --eval-batch-size 32 \
  --opt-iters 20 \
  --learning-rate 1e-4 \
  --log-every 1 \
  --wandb \
  --wandb-project newton-contact-point-friction-fit \
  --wandb-run-name mujoco-fit-004