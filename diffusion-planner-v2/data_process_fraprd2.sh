###################################
# User Configuration Section
###################################
NUPLAN_DATA_PATH="/data/temp/Expire180Days/users/luobwang/nuplan_raw/dataset/nuplan-v1.1/splits/train" # nuplan training data path (e.g., "/data/nuplan-v1.1/trainval")
NUPLAN_MAP_PATH="/data/temp/Expire180Days/users/luobwang/nuplan_raw/dataset/maps" # nuplan map path (e.g., "/data/nuplan-v1.1/maps")

TRAIN_SET_PATH="/data/temp/Expire180Days/automlops-fraprd/long-retention/diffusion_planner/train" # preprocess training data
###################################

/.cache/pypoetry/virtualenvs/unitraj-YTOfUZVn-py3.9/bin/python data_process.py \
--data_path $NUPLAN_DATA_PATH \
--map_path $NUPLAN_MAP_PATH \
--save_path $TRAIN_SET_PATH \
--total_scenarios 1000000 \
--agent_num 128 \

