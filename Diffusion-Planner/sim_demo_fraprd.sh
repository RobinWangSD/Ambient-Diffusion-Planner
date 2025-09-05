#!/bin/bash

# Environment setup
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HYDRA_FULL_ERROR=1
export NUPLAN_DEVKIT_ROOT="/root/planning-intern/nuplan-devkit"
export NUPLAN_DATA_ROOT="/data/ml4/preprocessed/nuplan-v1.1_raw/data/cache"
export NUPLAN_MAPS_ROOT="/data/ml4/preprocessed/nuplan-v1.1_raw/maps"
export NUPLAN_EXP_ROOT="/data/out/users/luobwang/nuplan_exp"

# Configuration
ARGS_FILE="/data/out/users/luobwang/diffusion_planner_factorized_train_log/training_log/pl_factorized_diffusion_planner_1000000_samples-32_observed-15_predicted-chunk_size_1-future_mask_true-use_chunking_true-if_factorized_true/2025-08-05-23:02:07/args.json"
CKPT_FILE="/data/out/users/luobwang/diffusion_planner_factorized_train_log/checkpoints/pl_factorized_diffusion_planner_1000000_samples-32_observed-15_predicted-chunk_size_1-future_mask_true-use_chunking_true-if_factorized_true/epoch\=39-train_loss\=0.0755.ckpt"
SPLIT="test14-random"
VIDEO_SAVE_DIR="/data/out/users/luobwang/nuplan_exp/exp/video"

# Run simulation
python /root/planning-intern/nuplan-devkit/nuplan/planning/script/run_simulation.py \
    +simulation=closed_loop_nonreactive_agents \
    planner=factorized_diffusion_planner \
    planner.diffusion_planner.config.args_file=$ARGS_FILE \
    planner.diffusion_planner.ckpt_path=$CKPT_FILE \
    scenario_builder=nuplan_challenge \
    scenario_filter=$SPLIT \
    experiment_uid=factorized_diffusion_planner/${SPLIT}_no_guidance \
    verbose=true \
    worker=ray_distributed \
    worker.threads_per_node=10 \
    distributed_mode='SINGLE_NODE' \
    number_of_gpus_allocated_per_simulation=0.15 \
    enable_simulation_progress_bar=true \
    +planner.diffusion_planner.render=true \
    +planner.pluto_planner.save_dir=$VIDEO_SAVE_DIR \
    "hydra.searchpath=['pkg://diffusion_planner.config.scenario_filter','pkg://diffusion_planner.config','pkg://nuplan.planning.script.config.common','pkg://nuplan.planning.script.experiments']"