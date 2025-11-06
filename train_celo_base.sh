#!/bin/bash

python train_celo_base.py --save_ckpt celo_base_longer.pt --meta_epochs 10 --meta_steps 200 --unroll_steps 20 --directions 16 --sigma 0.02 --outer_lr 1e-3
