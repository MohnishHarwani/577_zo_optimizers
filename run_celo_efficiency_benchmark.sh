#!/bin/bash

python run_celo_efficiency_benchmark.py \
  --use_celo_hf \
  --target_val_loss 2.30 \
  --max_steps 3000 \
  --eval_every 50

# #VeLO vs AdamW
# echo "VeLO vs AdamW"
# python run_celo_efficiency_benchmark.py \
#   --velo_ckpt velo_pretrained.pt \
#   --run_adamw  --run_velo \
#   --target_val_loss 2.8 --max_steps 10000 --log_every 10
#
# #base CeLO vs AdamW
# echo "base CeLO vs AdamW"
# python run_celo_efficiency_benchmark.py --celo_ckpt celo_base.pt \
#   --run_celo_base --run_adamw --target_val_loss 2.30 --max_steps 2000
#
# # fp16 quantization with autocast
# echo  "fp16 quantization with autocast"
# python run_celo_efficiency_benchmark.py --celo_ckpt celo_base.pt --run_celo_quant_fp16
#
# # dynamic int8 on Linear layers (CPU-friendly)
# echo "dynamic int8 on linear layers"
# python run_celo_efficiency_benchmark.py --celo_ckpt celo_base.pt --run_celo_quant_int8_linear
#
# # pruning 40%
# echo "pruning by 40%"
# python run_celo_efficiency_benchmark.py --celo_ckpt celo_base.pt --run_celo_prune --prune_amount 0.4
#
# # distillation
# echo "distillation"
# python run_celo_efficiency_benchmark.py --celo_ckpt celo_base.pt \
#   --run_celo_distill --distill_steps 200 --student_hidden_sched 16 --student_hidden_rule 16
