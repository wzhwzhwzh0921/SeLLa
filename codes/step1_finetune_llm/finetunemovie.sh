#!/bin/bash
# export NCCL_P2P_DISABLE=1
# export NCCL_P2P_DISABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=5 # Use GPU 0 and 1
# Guide:
# This script supports distributed training on multi-gpu workers (as well as single-worker training).
# Please set the options below according to the comments.
# For multi-gpu workers training, these options should be manually set for each worker.
# After setting the options, please run the script on each worker.


# Number of GPUs per GPU worker
GPUS_PER_NODE=1


# Number of GPU workers, for single-worker training, please set to 1
NNODES=${NNODES:-1}

# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
NODE_RANK=${NODE_RANK:-0}

# The ip address of the rank-0 worker, for single-worker training, please set to localhost
MASTER_ADDR=${MASTER_ADDR:-localhost}

# The port for communication
MASTER_PORT=${MASTER_PORT:-7009}

MODEL="/datas/wuxi/Projects/datas/models/Qwen2-7B" 
# Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See https://qwen.readthedocs.io/en/latest/training/SFT/example.html#data-preparation for more information.

DS_CONFIG_PATH="./ds_config_zero3.json"

# TRAIN_DATA="../data/book/train_data.jsonl"
TRAIN_DATA="../data/movie/train_data.jsonl"

# EVAL_DATA="../data/book/test_data.jsonl"
# EVAL_DATA="../data/book/test_data_test.jsonl"

EVAL_DATA="../data/movie/test_data.jsonl"
# EVAL_DATA="../data/movie/test_data_test.jsonl"

# LOG_FILE="../log/book/finetune_book_log.out"
LOG_FILE="../log/movie/finetune_movie_log2.out"

# OUTPUT_DIR="../output/book"
OUTPUT_DIR="../output/movie"

USE_LORA=True
Q_LORA=False


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

nohup torchrun $DISTRIBUTED_ARGS finetune.py \
    --model_name_or_path $MODEL \
    --bf16 True \
    --data_path $TRAIN_DATA \
    --eval_data_path $EVAL_DATA \
    --output_dir output_qwen \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 3 \
    --learning_rate 8e-5 \
    --weight_decay 0.01 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length 768 \
    --lazy_preprocess True \
    --use_lora ${USE_LORA} \
    --q_lora ${Q_LORA} \
    --gradient_checkpointing \
    --deepspeed ${DS_CONFIG_PATH}\
    --output_dir ${OUTPUT_DIR} \
>${LOG_FILE} 2>&1 &
