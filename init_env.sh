#!/bin/bash
# Copyright (c) 2026 Electronic Arts Inc. All rights reserved. See LICENSE for details.


# Exit immediately if a command exits with a non-zero status
set -e

ORIGINAL_DIR=$(pwd) # Saving original dir to come back to after the script

OUR_REPO_DIR="/workflow/" # Our repo's directory

cd "$OUR_REPO_DIR"

ENV_NAME="marling_env"    # Name of the Conda environment
YML_FILE="environment.yml"  # Path to the environment.yml file

create_env() {
    echo "Creating Conda environment: $ENV_NAME"
    conda env create -f "$YML_FILE"
    echo "Environment '$ENV_NAME' created successfully."
}

update_env() {
    echo "Updating Conda environment: $ENV_NAME"
    conda env update -f "$YML_FILE" --prune
    echo "Environment '$ENV_NAME' updated successfully."
}

# Install Conda if needed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed or not in PATH. Setting up Conda"

    # Download the installer, install Conda, and then remove Conda
    mkdir -p "$HOME/miniconda3"
    wget https://repo.anaconda.com/miniconda/Miniconda3-py310_25.1.1-0-Linux-x86_64.sh -O $HOME/miniconda3/miniconda.sh
    chmod +x "$HOME/miniconda3/miniconda.sh"
    ls "$HOME/miniconda3"
    "$HOME/miniconda3/miniconda.sh" -b -u -p "$HOME/miniconda3/"
    rm "$HOME/miniconda3/miniconda.sh"

    # Source Conda
    source "$HOME/miniconda3/bin/activate"
fi


## Creating/Updating the Conda environment if needed
if conda env list | grep -q "^$ENV_NAME\s"; then
    echo "Environment '$ENV_NAME' already exists. Updating it."
    update_env
else
    create_env
fi

# Source Conda
source "$HOME/miniconda3/bin/activate"

# Init Conda
conda init --all

# Activate the environment
conda activate "$ENV_NAME"
echo "Environment '$ENV_NAME' is now active."

## Installing Google Research Football

# Installing Dependencies
DEBIAN_FRONTEND=noninteractive apt-get install -qy git cmake build-essential libgl1-mesa-dev libsdl2-dev \
libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
libdirectfb-dev libst-dev mesa-utils xvfb x11vnc python3-pip

# Installing the engine itself
REPO_DIR="football"

# Ensure the repository directory exists
if [ ! -d "$HOME/$REPO_DIR" ]; then
    mkdir -p "$HOME/$REPO_DIR"
fi

# Check if the repo is already initialized as a git repository
if ! git -C "$HOME/$REPO_DIR" rev-parse --git-dir > /dev/null 2>&1; then
    echo "Cloning repository into directory $REPO_DIR..."
    git clone https://github.com/google-research/football.git "$HOME/$REPO_DIR"
else
    echo "Valid git repository found in $REPO_DIR. Skipping clone."
fi

# Change into the repository directory
cd "$HOME/$REPO_DIR"

python -m pip install --upgrade-strategy only-if-needed .

# Change back into the original directory
cd "$ORIGINAL_DIR"
