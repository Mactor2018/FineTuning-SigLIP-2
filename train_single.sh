#!/bin/bash

# 设置CUDA可见设备
export CUDA_VISIBLE_DEVICES=0  # 只使用一张GPU
export HF_ENDPOINT=https://hf-mirror.com
# 禁用内核版本警告
export ACCELERATE_DISABLE_KERNEL_WARNING=1

# 运行单GPU训练
python train.py \
  --data_path "/work/home/yinshb/yinshb/zjx2/FishNet/json/bird_train_pure.json" \
  --base_dir "/work/home/yinshb/yinshb/zjx/model_and_data/data" \
  --output_dir "output/siglip2-birds" \
  --model_name "google/siglip2-so400m-patch14-384" \
  --batch_size 32 \
  --eval_batch_size 64 \
  --learning_rate 2e-6 \
  --num_epochs 10 \
  --weight_decay 0.01 \
  --seed 42 \
  --test_size 0.3 \
  --num_workers 2 \
  --mixed_precision "bf16" \
  --resize_size 384 