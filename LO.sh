#!/bin/bash

python LO.py --models "celo adamw " --steps_per_epoch=1000 --save_dir "out_celo" \
 --celo_meta_first_order --meta_train_celo --celo_meta_steps 10
