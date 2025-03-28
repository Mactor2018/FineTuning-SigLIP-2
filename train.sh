#!/bin/bash

# 设置CUDA可见设备
export CUDA_VISIBLE_DEVICES=0

# 运行训练脚本
python train.py \
  --data_path "" \
  --base_dir "/work/home/yinshb/yinshb/zjx/model_and_data/data" \
  --output_dir "output/siglip2-birds" \
  --model_name "google/siglip2-so400m-patch14-384" \
  --batch_size 16 \
  --eval_batch_size 8 \
  --learning_rate 2e-6 \
  --num_epochs 10 \
  --weight_decay 0.01 \
  --seed 42 \
  --test_size 0.3 