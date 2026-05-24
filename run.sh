DATASET=/workspace/mujoco/outputs/block_force_dataset_fixed_init_2000_new/block_force_dataset_fixed_init_2000_new.npz

  # python newton/fit_mujoco_contact_point_friction.py \
  #   --trajectory-npz "$DATASET" \
  #   --experiment-dir outputs/fixed_init_0.4_stiffness_1e5_regularization_0_global_new \
  #   --device cuda:0 \
  #   --batch-size 256 \
  #   --opt-iters 20000 \
  #   --learning-rate 1e-4 \
  #   --log-every 10 \
  #   --checkpoint-every 100 \
  #   --grad-clip-norm 1.0 \
  #   --surface-point-spacing 0.01 \
  #   --contact-stiffness 1e5 \
  #   --max-steps 300 \
  #   --friction-parameterization global \
  #   --wandb \
  #   --wandb-project newton_friction_fitting \
  #   --wandb-run-name fixed_init_0.4_stiffness_1e5_regularization_0_global_new \
  #   --point-friction 0.4 &


  # python newton/fit_mujoco_contact_point_friction.py \
  #   --trajectory-npz "$DATASET" \
  #   --experiment-dir outputs/fixed_init_0.35_stiffness_1e5_regularization_0_left_right_new \
  #   --device cuda:0 \
  #   --batch-size 256 \
  #   --opt-iters 20000 \
  #   --learning-rate 1e-4 \
  #   --log-every 10 \
  #   --checkpoint-every 100 \
  #   --grad-clip-norm 1.0 \
  #   --surface-point-spacing 0.01 \
  #   --contact-stiffness 1e5 \
  #   --max-steps 300 \
  #   --friction-parameterization left-right \
  #   --wandb \
  #   --wandb-project newton_friction_fitting \
  #   --wandb-run-name fixed_init_0.35_stiffness_1e5_regularization_0_left_right_new \
  #   --point-friction 0.3 &

  python newton/fit_mujoco_contact_point_friction.py \
    --trajectory-npz "$DATASET" \
    --experiment-dir outputs/fixed_init_0.4_stiffness_1e5_regularization_3000_point_new \
    --device cuda:0 \
    --batch-size 256 \
    --opt-iters 20000 \
    --learning-rate 1e-4 \
    --log-every 10 \
    --checkpoint-every 100 \
    --grad-clip-norm 1.0 \
    --surface-point-spacing 0.01 \
    --contact-stiffness 1e5 \
    --piecewise-regularization-weight 3000 \
    --max-steps 300 \
    --friction-parameterization point \
    --wandb \
    --wandb-project newton_friction_fitting \
    --wandb-run-name fixed_init_0.4_stiffness_1e5_regularization_3000_point_new \
    --point-friction 0.4 &

  python newton/fit_mujoco_contact_point_friction.py \
    --trajectory-npz "$DATASET" \
    --experiment-dir outputs/fixed_init_0.4_stiffness_1e5_regularization_0_point_new \
    --device cuda:0 \
    --batch-size 256 \
    --opt-iters 20000 \
    --learning-rate 1e-4 \
    --log-every 10 \
    --checkpoint-every 100 \
    --grad-clip-norm 1.0 \
    --surface-point-spacing 0.01 \
    --contact-stiffness 1e5 \
    --piecewise-regularization-weight 0 \
    --max-steps 300 \
    --friction-parameterization point \
    --wandb \
    --wandb-project newton_friction_fitting \
    --wandb-run-name fixed_init_0.4_stiffness_1e5_regularization_0_point_new \
    --point-friction 0.4 &


  env MUJOCO_GL=osmesa python mujoco/scripts/run_block_force_demo.py \
    --num-episodes 2000 \
    --no-randomize-initial-pose \
    --output-dir mujoco/outputs/block_force_dataset_multiseg_2000_long_videos \
    --force-segments 2 \
    --duration-min 0.2 \
    --duration-max 1.0 \
    --total-duration 5.0 \
    --force-min 0.5 \
    --force-max 8.0 \
    --save-episode-videos \
    --progress-every 50 &

wait
