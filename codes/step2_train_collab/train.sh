
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0 # Use GPU 0 and 1

GPUS_PER_NODE=1

NNODES=${NNODES:-1}

NODE_RANK=${NODE_RANK:-0}

MASTER_ADDR=${MASTER_ADDR:-localhost}

MASTER_PORT=${MASTER_PORT:-7005}

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
# LOG_DIR="../log/book/train_mf_movie_log.out"
LOG_DIR="../log/book/train_mf_book_log.out"
nohup torchrun $DISTRIBUTED_ARGS train_mf.py > ${LOG_DIR}
