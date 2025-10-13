#!/bin/bash

###################################
# Cluster Training Script for Diffusion Planner
###################################

# Default values
LOGGER='qualcomm'
NUM_OBSERVED=64
NUM_PREDICTED=10
BATCH_SIZE=2048
MAX_FILES=500000  # 5e5
GPUS="0,1,2,3,4,5,6,7"
NNODES=1
NPROC_PER_NODE=8
SAVE_DIR="/data/temp/Expire180Days/users/luobwang/diffusion_planner_factorized_train_log"
TRAIN_SET_PATH="/data/temp/Expire180Days/automlops-fraprd/long-retention/diffusion_planner/train"
TRAIN_SET_MAPPING_PATH="/data/temp/Expire180Days/automlops-fraprd/long-retention/diffusion_planner/train/file_mapping_20250731_071801.pkl"
RUN_PYTHON_PATH="/.cache/pypoetry/virtualenvs/unitraj-YTOfUZVn-py3.9/bin/python"
USE_BATCH_AWARE=true
TRAIN_EPOCHS=1000
LEARNING_RATE=3e-4
SEED=3407
NOTES=""
RESUME_MODEL_PATH=""
USE_EMA=true
PORT="22323"
NUM_WORKERS=32
CHUNK_SIZE=2
DECODER_AGENT_ATTN=false
USE_CHUNKING=false
IF_FACTORIZED=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --logger)
            LOGGER="$2"
            shift 2
            ;;
        --num-observed)
            NUM_OBSERVED="$2"
            shift 2
            ;;
        --num-predicted)
            NUM_PREDICTED="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-files)
            MAX_FILES="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --nproc-per-node)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        --save-dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --train-set)
            TRAIN_SET_PATH="$2"
            shift 2
            ;;
        --mapping-pkl)
            TRAIN_SET_MAPPING_PATH="$2"
            shift 2
            ;;
        --python-path)
            RUN_PYTHON_PATH="$2"
            shift 2
            ;;
        --job-name)
            JOB_NAME="$2"
            shift 2
            ;;
        --use-batch-aware)
            USE_BATCH_AWARE="$2"
            shift 2
            ;;
        --train-epochs)
            TRAIN_EPOCHS="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --notes)
            NOTES="$2"
            shift 2
            ;;
        --resume-model-path)
            RESUME_MODEL_PATH="$2"
            shift 2
            ;;
        --use-ema)
            USE_EMA="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --chunk-size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --decoder-agent-attn-mask)
            DECODER_AGENT_ATTN="$2"
            shift 2
            ;;
        --use-chunking)
            USE_CHUNKING="$2"
            shift 2
            ;;
        --if-factorized)
            IF_FACTORIZED="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --logger <type>         Logger type (qualcomm|wandb|tensorboard|none) [default: qualcomm]"
            echo "  --num-observed <n>      Number of observed agents [default: 32]"
            echo "  --num-predicted <n>     Number of predicted neighbors [default: 10]"
            echo "  --batch-size <n>        Batch size [default: 512]"
            echo "  --max-files <n>         Maximum number of training files [default: 500000]"
            echo "  --gpus <list>           GPU IDs to use (comma-separated) [default: 0,1]"
            echo "  --nnodes <n>            Number of nodes [default: 1]"
            echo "  --nproc-per-node <n>    Processes per node [default: 2]"
            echo "  --save-dir <path>       Save directory [default: /data/temp/Expire180Days/users/luobwang/diffusion_planner_train_log]"
            echo "  --train-set <path>      Training data path"
            echo "  --mapping-pkl <path>    Mapping pickle file path"
            echo "  --python-path <path>    Python executable path"
            echo "  --job-name <name>       Job name (auto-generated if not provided)"
            echo "  --use-batch-aware <bool> Use batch-aware sampling [default: true]"
            echo "  --train-epochs <n>      Training epochs [default: 500]"
            echo "  --learning-rate <lr>    Learning rate [default: 5e-4]"
            echo "  --seed <n>              Random seed [default: 3407]"
            echo "  --notes <text>          Training notes"
            echo "  --resume-model-path <path> Path to resume model from"
            echo "  --use-ema <bool>        Use EMA [default: true]"
            echo "  --port <port>           DDP port [default: 22323]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set job name if not provided
if [ -z "$JOB_NAME" ]; then
    JOB_NAME="factorized_diffusion_planner_${MAX_FILES}_samples-${NUM_OBSERVED}_observed-${NUM_PREDICTED}_predicted-chunk_size_${CHUNK_SIZE}-future_mask_${DECODER_AGENT_ATTN}-use_chunking_${USE_CHUNKING}-if_factorized_${IF_FACTORIZED}"
fi

# Export CUDA devices
export CUDA_VISIBLE_DEVICES=$GPUS

# Count actual GPUs
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
ACTUAL_NPROC=${#GPU_ARRAY[@]}

# Use actual GPU count if nproc-per-node wasn't explicitly set
if [ "$NPROC_PER_NODE" -eq 2 ] && [ "$ACTUAL_NPROC" -ne 2 ]; then
    NPROC_PER_NODE=$ACTUAL_NPROC
fi

###################################
# Print configuration
###################################
echo "===== Diffusion Planner Training Configuration ====="
echo "Job Name: $JOB_NAME"
echo "Logger: $LOGGER"
echo "Dataset Configuration:"
echo "  Training Set: $TRAIN_SET_PATH"
echo "  Mapping File: $TRAIN_SET_MAPPING_PATH"
echo "  Max Files: $MAX_FILES"
echo "  Num Observed: $NUM_OBSERVED"
echo "  Num Predicted: $NUM_PREDICTED"
echo "  Use Batch-Aware: $USE_BATCH_AWARE"
echo "Training Configuration:"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Training Epochs: $TRAIN_EPOCHS"
echo "  Seed: $SEED"
echo "  Use EMA: $USE_EMA"
echo "  Use chunking: $USE_CHUNKING"
echo "  Chunk_szie: $CHUNK_SIZE"
echo "  If factorized: $IF_FACTORIZED"
echo "  Use future agent mask in decoder attention: $DECODER_AGENT_ATTN"
if [ -n "$RESUME_MODEL_PATH" ]; then
    echo "  Resume from: $RESUME_MODEL_PATH"
fi
echo "Hardware Configuration:"
echo "  GPUs: $GPUS (Total: $ACTUAL_NPROC)"
echo "  Nodes: $NNODES"
echo "  Processes per node: $NPROC_PER_NODE"
echo "  DDP Port: $PORT"
echo "Paths:"
echo "  Save Directory: $SAVE_DIR"
echo "  Python Path: $RUN_PYTHON_PATH"
if [ -n "$NOTES" ]; then
    echo "Notes: $NOTES"
fi
echo "===================================================="

# Verify paths exist
if [ ! -d "$TRAIN_SET_PATH" ]; then
    echo "Error: Training set path does not exist: $TRAIN_SET_PATH"
    exit 1
fi

if [ ! -f "$TRAIN_SET_MAPPING_PATH" ]; then
    echo "Error: Mapping file does not exist: $TRAIN_SET_MAPPING_PATH"
    exit 1
fi

if [ ! -f "$RUN_PYTHON_PATH" ]; then
    echo "Error: Python executable not found: $RUN_PYTHON_PATH"
    exit 1
fi

# Create save directory if it doesn't exist
mkdir -p "$SAVE_DIR"

###################################
# Build training command
###################################
TRAIN_CMD="$RUN_PYTHON_PATH -m torch.distributed.run \
    --nnodes $NNODES \
    --nproc-per-node $NPROC_PER_NODE \
    --standalone \
    train_factorized_predictor.py \
    --name $JOB_NAME \
    --save_dir $SAVE_DIR \
    --train_set $TRAIN_SET_PATH \
    --mapping_pkl $TRAIN_SET_MAPPING_PATH \
    --agent_num $NUM_OBSERVED \
    --predicted_neighbor_num $NUM_PREDICTED \
    --max_files $MAX_FILES \
    --logger $LOGGER \
    --batch_size $BATCH_SIZE \
    --train_epochs $TRAIN_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --seed $SEED \
    --use_ema $USE_EMA \
    --num_workers $NUM_WORKERS \
    --port $PORT\
    --chunk_size $CHUNK_SIZE \
    --decoder_agent_attn_mask $DECODER_AGENT_ATTN \
    --use_chunking $USE_CHUNKING \
    --if_factorized $IF_FACTORIZED"

# Add optional arguments
if [ "$USE_BATCH_AWARE" = "true" ]; then
    TRAIN_CMD="$TRAIN_CMD --use_batch_aware"
fi

if [ -n "$NOTES" ]; then
    TRAIN_CMD="$TRAIN_CMD --notes \"$NOTES\""
fi

if [ -n "$RESUME_MODEL_PATH" ]; then
    TRAIN_CMD="$TRAIN_CMD --resume_model_path \"$RESUME_MODEL_PATH\""
fi

###################################
# Run training
###################################
echo ""
echo "Starting training..."
echo "Command: $TRAIN_CMD"
echo ""

# Use sudo if available, otherwise run directly
if command -v sudo &> /dev/null; then
    sudo -E $TRAIN_CMD
else
    eval $TRAIN_CMD
fi

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "Training completed successfully!"
else
    echo ""
    echo "Training failed with exit code: $?"
    exit 1
fi