#!/bin/bash

rm -r out_fast

python HybridTrajectory.py \
  --field_grid 41 \
  --save_dir out_fast
