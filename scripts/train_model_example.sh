#!/bin/bash

MODEL_NAME="Qwen/Qwen3-4B-Base"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py \
    --teacher_model_name_or_path $MODEL_NAME \
    --save_directory "/path/to/save/directory" \
    --dataset "EleutherAI/the_pile_deduplicated" \
    --fsdp_config $MODEL_NAME \