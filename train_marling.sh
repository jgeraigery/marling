#!/bin/bash
# Copyright (c) 2026 Electronic Arts Inc. All rights reserved. See LICENSE for details.


# Exit immediately if a command exits with a non-zero status
set -e
 
OUR_REPO_DIR="/workflow/"
SESSION_OUTS_DIR="$HOME/logs/session_outs"

# Create SESSION_OUTS_DIR if it does not exist
mkdir -p $SESSION_OUTS_DIR

# Save our passed in argument
CONFIG_FILE=$1

# Clear our arguments (so that "conda activate" does not take it)
set --

# Source Conda activation script to ensure conda's env is intialized
source "$HOME/miniconda3/bin/activate"

# Initialize Conda
conda init --all

# Activate our environment
conda activate marling_env
 
# Move to our repo
cd "$OUR_REPO_DIR"

# Export our repo's directory to PYTHONPATH for finding the light_malib package needed to run training
export PYTHONPATH=$PYTHONPATH:$OUR_REPO_DIR

# Our seeds
array_global_seeds=(42 392 533 5727 9369) # Global, general seeds
array_rollout_manager_seeds=(319 3653 7955 8870 9958) # Rollout manager seeds

# Iterate over indices
for i in "${!array_global_seeds[@]}"; do
    SESSION_OUT_FILENAME=${CONFIG_FILE%.yaml}
    SESSION_OUT_FILENAME=${SESSION_OUT_FILENAME#.\/expr_configs\/tizero\/tizero_full_training_10v10_}
    SESSION_OUT_FILENAME="${SESSION_OUTS_DIR}/${SESSION_OUT_FILENAME}_${array_global_seeds[$i]}_${array_rollout_manager_seeds[$i]}_date_$(date +%Y-%m-%d-%H-%M-%S).out"
    echo "Running training with global seed ${array_global_seeds[$i]} and rollout manager seed ${array_rollout_manager_seeds[$i]}"
    echo "Output will be saved to ${SESSION_OUT_FILENAME}"
    python light_malib/main_tizero.py --config "$CONFIG_FILE" --seed ${array_global_seeds[$i]} --rollout-seed ${array_rollout_manager_seeds[$i]} 2>&1 | tee $SESSION_OUT_FILENAME 
done