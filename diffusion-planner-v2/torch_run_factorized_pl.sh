#!/bin/bash

###################################
# PyTorch Lightning Training Script for Diffusion Planner
###################################

# Default values
LOGGER='qualcomm'
NUM_OBSERVED=64
NUM_PREDICTED=10
BATCH_SIZE=256
MAX_FILES=1000000  # should not be effective
GPUS="0,1,2,3,4,5,6,7"
NNODES=1
SAVE_DIR="/data/out/users/luobwang/factorized_planner_train_log_5"
TRAIN_SET_PATH="/mnt/machine-learning-storage/ML2/automlops-fraprd/long-retention/diffusion_planner/trainval_batch"
TRAIN_SET_MAPPING_PATH="/data/out/users/luobwang/dataset_splits/dataset_1m_path_mapping.pkl"
VAL_SET_PATH="/mnt/machine-learning-storage/ML2/automlops-fraprd/long-retention/diffusion_planner/trainval_batch"
VAL_SET_MAPPING_PATH="/data/out/users/luobwang/dataset_splits/validation_path_mapping.pkl"
USE_BATCH_AWARE=true
TRAIN_EPOCHS=1000
LEARNING_RATE=3e-4
SEED=42
NOTES=""
RESUME_MODEL_PATH=""
USE_EMA=true
NUM_WORKERS=32
CHUNK_SIZE=2
CHUNK_OVERLAP=1
DECODER_DEPTH=3
ENCODER_DEPTH=3
HIDDEN_DIM=192
DECODER_AGENT_ATTN=true
USE_CHUNKING=true
IF_FACTORIZED=true
USE_CAUSAL=true
USE_AGENT_VALIDITY_IN_TEMPORAL=false
USE_CHUNK_T_EMBED=false
EGO_SEPARATE=false
KEY_PADDING=false
PAD_LEFT=false
PAD_HISTORY=false
V2=false
RESIDUAL_EMB=false

# PyTorch Lightning specific defaults
ACCELERATOR="gpu"
STRATEGY="ddp"
GRADIENT_CLIP_VAL=5.0
LOG_EVERY_N_STEPS=50

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
        --val-set)
            VAL_SET_PATH="$2"
            shift 2
            ;;
        --val-mapping-pkl)
            VAL_SET_MAPPING_PATH="$2"
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
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --chunk-size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --chunk-overlap)
            CHUNK_OVERLAP="$2"
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
        --use-causal-attn)
            USE_CAUSAL="$2"
            shift 2
            ;;
        --use-agent-validity-in-temporal)
            USE_AGENT_VALIDITY_IN_TEMPORAL="$2"
            shift 2
            ;;
        --use-chunk-t-embed)
            USE_CHUNK_T_EMBED="$2"
            shift 2
            ;;
        --ego-separate)
            EGO_SEPARATE="$2"
            shift 2
            ;;
        --key-padding)
            KEY_PADDING="$2"
            shift 2
            ;;
        --pad-left)
            PAD_LEFT="$2"
            shift 2
            ;;
        --pad-history)
            PAD_HISTORY="$2"
            shift 2
            ;;
        --v2)
            V2="$2"
            shift 2
            ;;
        --residual-emb)
            RESIDUAL_EMB="$2"
            shift 2
            ;;
        # PyTorch Lightning specific arguments
        --accelerator)
            ACCELERATOR="$2"
            shift 2
            ;;
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --gradient-clip-val)
            GRADIENT_CLIP_VAL="$2"
            shift 2
            ;;
        --log-every-n-steps)
            LOG_EVERY_N_STEPS="$2"
            shift 2
            ;;
        --decoder-depth)
            DECODER_DEPTH="$2"
            shift 2
            ;;
        --encoder-depth)
            ENCODER_DEPTH="$2"
            shift 2
            ;;
        --hidden-dim)
            HIDDEN_DIM="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --logger <type>         Logger type (wandb|tensorboard|none) [default: wandb]"
            echo "  --num-observed <n>      Number of observed agents [default: 64]"
            echo "  --num-predicted <n>     Number of predicted neighbors [default: 10]"
            echo "  --batch-size <n>        Batch size [default: 2048]"
            echo "  --max-files <n>         Maximum number of training files [default: 500000]"
            echo "  --gpus <list>           GPU IDs to use (comma-separated) [default: 0,1,2,3,4,5,6,7]"
            echo "  --nnodes <n>            Number of nodes [default: 1]"
            echo "  --save-dir <path>       Save directory"
            echo "  --train-set <path>      Training data path"
            echo "  --mapping-pkl <path>    Mapping pickle file path"
            echo "  --python-path <path>    Python executable path"
            echo "  --job-name <name>       Job name (auto-generated if not provided)"
            echo "  --use-batch-aware <bool> Use batch-aware sampling [default: true]"
            echo "  --train-epochs <n>      Training epochs [default: 1000]"
            echo "  --learning-rate <lr>    Learning rate [default: 3e-4]"
            echo "  --seed <n>              Random seed [default: 3407]"
            echo "  --notes <text>          Training notes"
            echo "  --resume-model-path <path> Path to resume model from"
            echo "  --use-ema <bool>        Use EMA [default: true]"
            echo "  --num-workers <n>       Number of data loading workers [default: 32]"
            echo ""
            echo "PyTorch Lightning Options:"
            echo "  --accelerator <type>    Accelerator type (gpu|cpu|tpu) [default: gpu]"
            echo "  --strategy <type>       Training strategy (ddp|fsdp|deepspeed) [default: ddp]"
            echo "  --precision <type>      Training precision (16-mixed|bf16-mixed|32) [default: 16-mixed]"
            echo "  --gradient-clip-val <val> Gradient clipping value [default: 5.0]"
            echo "  --log-every-n-steps <n> Log every N steps [default: 50]"
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
    JOB_NAME="ml-${MAX_FILES}-${NUM_OBSERVED}_ob-${NUM_PREDICTED}_pre-chunk_size_${CHUNK_SIZE}-chunk_overlap_${CHUNK_OVERLAP}-future_mask_${DECODER_AGENT_ATTN}-use_chunking_${USE_CHUNKING}-if_fac_${IF_FACTORIZED}-use_causal_${USE_CAUSAL}-temp_valid_${USE_AGENT_VALIDITY_IN_TEMPORAL}-dec_depth_${DECODER_DEPTH}-lr_${LEARNING_RATE}-hd_${HIDDEN_DIM}-use_t-${USE_CHUNK_T_EMBED}-ego_sep_${EGO_SEPARATE}-key_pad_${KEY_PADDING}-left_${PAD_LEFT}-history_${PAD_HISTORY}-v2_${V2}-res_emb_${RESIDUAL_EMB}"
fi

# Parse GPU configuration for Lightning
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

# Set devices for Lightning
if [ "$NUM_GPUS" -eq 1 ]; then
    DEVICES=1
else
    DEVICES=$NUM_GPUS
fi

###################################
# Print configuration
###################################
echo "===== PyTorch Lightning Diffusion Planner Training Configuration ====="
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
echo "  Chunk_size: $CHUNK_SIZE"
echo "  If factorized: $IF_FACTORIZED"
echo "  Encoder depth: $ENCODER_DEPTH"
echo "  Decoder depth: $DECODER_DEPTH"
echo "  Hidden dim: $HIDDEN_DIM"
echo "  Use future agent mask in decoder attention: $DECODER_AGENT_ATTN"
if [ -n "$RESUME_MODEL_PATH" ]; then
    echo "  Resume from: $RESUME_MODEL_PATH"
fi
echo "Hardware Configuration:"
echo "  GPUs: $GPUS (Total: $NUM_GPUS)"
echo "  Nodes: $NNODES"
echo "  Accelerator: $ACCELERATOR"
echo "  Strategy: $STRATEGY"
echo "  Precision: $PRECISION"
echo "  Devices per node: $DEVICES"
echo "Paths:"
echo "  Save Directory: $SAVE_DIR"
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

# Create save directory if it doesn't exist
mkdir -p "$SAVE_DIR"

# Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$GPUS

###################################
# Build training command
###################################

# For PyTorch Lightning, we don't need torch.distributed.run
# Lightning handles distributed training internally
TRAIN_CMD="torchrun --nnodes=$PET_NNODES --nproc_per_node=$PET_NPROC_PER_NODE train_pl.py \
    --name $JOB_NAME \
    --save_dir $SAVE_DIR \
    --train_set $TRAIN_SET_PATH \
    --train_mapping_pkl $TRAIN_SET_MAPPING_PATH \
    --val_set $VAL_SET_PATH \
    --val_mapping_pkl $VAL_SET_MAPPING_PATH \
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
    --encoder_depth $ENCODER_DEPTH \
    --decoder_depth $DECODER_DEPTH \
    --hidden_dim $HIDDEN_DIM \
    --chunk_size $CHUNK_SIZE \
    --chunk_overlap $CHUNK_OVERLAP \
    --decoder_agent_attn_mask $DECODER_AGENT_ATTN \
    --use_chunking $USE_CHUNKING \
    --if_factorized $IF_FACTORIZED \
    --use_causal_attn $USE_CAUSAL
    --use_agent_validity_in_temporal $USE_AGENT_VALIDITY_IN_TEMPORAL \
    --use_chunk_t_embed $USE_CHUNK_T_EMBED \
    --ego_separate $EGO_SEPARATE \
    --key_padding $KEY_PADDING \
    --pad_left $PAD_LEFT \
    --pad_history $PAD_HISTORY \
    --v2 $V2 \
    --residual_emb $RESIDUAL_EMB \
    --accelerator $ACCELERATOR \
    --devices $DEVICES \
    --strategy $STRATEGY"

# Add conditional arguments
if [ "$USE_BATCH_AWARE" = "true" ]; then
    TRAIN_CMD="$TRAIN_CMD --use_batch_aware"
fi

if [ -n "$NOTES" ]; then
    TRAIN_CMD="$TRAIN_CMD --notes \"$NOTES\""
fi

if [ -n "$RESUME_MODEL_PATH" ]; then
    TRAIN_CMD="$TRAIN_CMD --resume_model_path \"$RESUME_MODEL_PATH\""
fi

# Add multi-node support if needed
if [ "$NNODES" -gt 1 ]; then
    echo "Multi-node training detected. Please ensure you set the following environment variables:"
    echo "  MASTER_ADDR: IP address of the master node"
    echo "  MASTER_PORT: Port on the master node"
    echo "  NODE_RANK: Rank of this node (0 for master)"
    echo ""
    TRAIN_CMD="$TRAIN_CMD --num_nodes $NNODES"
fi

###################################
# Run training
###################################

echo ""
echo "Starting PyTorch Lightning training..."
echo "Command: $TRAIN_CMD"
echo ""

# Execute the command
eval $TRAIN_CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "Training completed successfully!"
    echo "Checkpoints saved in: $SAVE_DIR/checkpoints/$JOB_NAME/"
    echo "Logs available in the configured logger (WandB/TensorBoard)"
else
    echo ""
    echo "Training failed with exit code: $?"
    exit 1
fi