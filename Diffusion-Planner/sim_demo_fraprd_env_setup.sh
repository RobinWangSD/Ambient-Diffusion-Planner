#!/bin/bash

# Environment setup
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HYDRA_FULL_ERROR=1
export NUPLAN_DEVKIT_ROOT="/root/planning-intern/nuplan-devkit"
export NUPLAN_DATA_ROOT="/data/ml4/preprocessed/nuplan-v1.1_raw/data/cache"
export NUPLAN_MAPS_ROOT="/data/ml4/preprocessed/nuplan-v1.1_raw/maps"
export NUPLAN_EXP_ROOT="/data/out/users/luobwang/nuplan_exp"