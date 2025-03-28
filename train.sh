#!/bin/bash

# 设置CUDA可见设备
export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com

# 运行训练脚本
python train.py \
  --local_model_path "/work/home/yinshb/yinshb/zjx2/Ablation_study/VisionEncoder/siglip2-so400m-patch14-384"\
  --data_path "/work/home/yinshb/yinshb/zjx2/FishNet/json/bird_train_pure.json" \
  --base_dir "/work/home/yinshb/yinshb/zjx/model_and_data/data" \
  --output_dir "output/siglip2-birds" \
  --model_name "google/siglip2-so400m-patch14-384" \
  --batch_size 256 \
  --eval_batch_size 256 \
  --learning_rate 2e-6 \
  --num_epochs 10 \
  --weight_decay 0.01 \
  --seed 42 \
  --test_size 0.3 \
  --num_workers 12 \
  --mixed_precision "bf16" \
  --resize_size 384 