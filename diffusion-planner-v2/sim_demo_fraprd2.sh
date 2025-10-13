#!/bin/bash

# Environment setup
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HYDRA_FULL_ERROR=1
export NUPLAN_DEVKIT_ROOT="/root/planning-intern/nuplan-devkit"
export NUPLAN_DATA_ROOT="/data/temp/Expire180Days/users/luobwang/nuplan_raw/dataset"
export NUPLAN_MAPS_ROOT="/data/temp/Expire180Days/users/luobwang/nuplan_raw/dataset/maps"
export NUPLAN_EXP_ROOT="/data/temp/Expire180Days/users/luobwang/nuplan_raw/exp"

# Configuration
ARGS_FILE="/data/temp/Expire180Days/users/luobwang/diffusion_planner_train_log/2025-07-31-17:57:00/args.json"
CKPT_FILE="/data/temp/Expire180Days/users/luobwang/diffusion_planner_train_log/2025-07-31-17:57:00/model_epoch_100_trainloss_0.0781.pth"
SPLIT="test14-random"

# Run simulation
python /root/planning-intern/nuplan-devkit/nuplan/planning/script/run_simulation.py \
    +simulation=closed_loop_nonreactive_agents \
    planner=diffusion_planner \
    planner.diffusion_planner.config.args_file=$ARGS_FILE \
    planner.diffusion_planner.ckpt_path=$CKPT_FILE \
    scenario_builder=nuplan \
    scenario_filter=$SPLIT \
    experiment_uid=diffusion_planner/${SPLIT}_no_guidance \
    verbose=true \
    worker=ray_distributed \
    worker.threads_per_node=10 \
    distributed_mode='SINGLE_NODE' \
    number_of_gpus_allocated_per_simulation=0.15 \
    enable_simulation_progress_bar=true \
    "hydra.searchpath=['pkg://diffusion_planner.config.scenario_filter','pkg://diffusion_planner.config','pkg://nuplan.planning.script.config.common','pkg://nuplan.planning.script.experiments']"