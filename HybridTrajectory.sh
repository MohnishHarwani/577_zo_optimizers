#!/bin/bash

rm -r out_fast

python HybridTrajectory.py \
  --epochs 1 \
  --steps_per_epoch 200 \
  --seq_len 512 \
  --batch_size 8 \
  --plane traj_pca \
  --center_mode trained \
  --field_grid 11 \
  --field_batches 1 \
  --viz_batch_size 4 \
  --eval_seq_len 128 \
  --mag_scale log \
  --traj_every 10 \
  --zo_dirs 8 \
  --R_threshold 0.45 \
  --save_dir out_fast
