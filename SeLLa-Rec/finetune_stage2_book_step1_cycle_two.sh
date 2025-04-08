#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=7 # Use GPU 0 and 1
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
MASTER_PORT=${MASTER_PORT:-6002}

MODEL="/workspace/wzh/self_reasearch/Qwen2-7B" # Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See https://qwen.readthedocs.io/en/latest/training/SFT/example.html#data-preparation for more information.
TRAIN_DATA="/workspace/wzh/self_reasearch/Chat_Qwen/examples/sft/data/data_llm/book-collm-original-two/train_data.jsonl"
DS_CONFIG_PATH="/workspace/wzh/self_reasearch/Chat_Qwen/examples/sft/ds_config_zero3.json"
EVAL_DATA="/workspace/wzh/self_reasearch/Chat_Qwen/examples/sft/data/data_llm/book-collm-original-two/test_data.jsonl"
OUTPUT_DIR="/workspace/wzh/self_reasearch/Chat_Qwen/examples/sft/data/data_llm/book-collm-original-two/model/cycle/ab/888"
USE_LORA=True
Q_LORA=False

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS /workspace/wzh/self_reasearch/Chat_Qwen/examples/sft/finetune_stage2_book_step1_cycle_two.py \
    --model_name_or_path $MODEL \
    --data_path $TRAIN_DATA \
    --eval_data_path $EVAL_DATA \
    --bf16 True \
    --output_dir output_qwen \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 150 \
    --eval_steps 150 \
    --save_total_limit 10 \
    --learning_rate 8e-4 \
    --weight_decay 0.01 \
    --adam_beta2 0.95 \
    --warmup_ratio 0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length 768 \
    --lazy_preprocess True \
    --use_lora ${USE_LORA} \
    --q_lora ${Q_LORA} \
    --gradient_checkpointing \
    --deepspeed ${DS_CONFIG_PATH} \
    --output_dir ${OUTPUT_DIR}

##best
# torchrun $DISTRIBUTED_ARGS /workspace/wzh/self_reasearch/Chat_Qwen/examples/sft/finetune_stage2_book_step1_cycle_two.py \
#     --model_name_or_path $MODEL \
#     --data_path $TRAIN_DATA \
#     --eval_data_path $EVAL_DATA \
#     --bf16 True \
#     --output_dir output_qwen \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 32 \
#     --gradient_accumulation_steps 8 \
#     --evaluation_strategy "no" \
#     --save_strategy "epoch" \
#     --save_steps 150 \
#     --eval_steps 150 \
#     --save_total_limit 10 \
#     --learning_rate 8e-4 \
#     --weight_decay 0.01 \
#     --adam_beta2 0.95 \
#     --warmup_ratio 0 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --report_to "none" \
#     --model_max_length 768 \
#     --lazy_preprocess True \
#     --use_lora ${USE_LORA} \
#     --q_lora ${Q_LORA} \
#     --gradient_checkpointing \
#     --deepspeed ${DS_CONFIG_PATH} \
#     --output_dir ${OUTPUT_DIR}


