#!/bin/bash

# 设置CUDA可见设备
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用4张GPU，根据实际可用数量调整
export OMP_NUM_THREADS=8  # 可以设置为总核心数的1/4
export HF_ENDPOINT=https://hf-mirror.com
# 禁用内核版本警告
export ACCELERATE_DISABLE_KERNEL_WARNING=1

# 运行分布式训练 - 不使用DeepSpeed，仅使用原生PyTorch DDP
torchrun --nproc_per_node=4 train.py \
  --data_path "/work/home/yinshb/yinshb/zjx2/FishNet/json/bird_train_pure.json" \
  --base_dir "/work/home/yinshb/yinshb/zjx/model_and_data/data" \
  --output_dir "output/siglip2-birds" \
  --model_name "google/siglip2-so400m-patch14-384" \
  --batch_size 16 \
  --eval_batch_size 32 \
  --learning_rate 2e-6 \
  --num_epochs 10 \
  --weight_decay 0.01 \
  --seed 42 \
  --test_size 0.3 \
  --num_workers 2 \
  --mixed_precision "bf16" \
  --resize_size 384 