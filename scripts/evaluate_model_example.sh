#!/bin/bash

MODEL_NAME="Qwen/Qwen3-4B-Base"

python3 evaluate.py \
    --teacher_model_name_or_path $MODEL_NAME \
    --student_model_name_or_path "/path/to/student/model" \
    --save_directory "/path/to/save/directory" \
    --num_layers_to_patch 4 \