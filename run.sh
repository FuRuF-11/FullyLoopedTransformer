#!/bin/bash

set -ex


export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
mkdir -p $NANOCHAT_BASE_DIR
uv sync --extra gpu
# -----------------------------------------------------------------------------
# wandb setup
# export WANDB_API_KEY="<YOUR_WANDB_API_KEY>"
uv run wandb login

# -----------------------------------------------------------------------------
# dataset download

uv run -m nanochat.dataset -n 250 
uv run -m nanochat.dataset -t 1

# ------------------------------------------------
# Exp setup

CONFIGS=(
    "config/FLT.yaml"
    "config/LT_ia.yaml"
    "config/LT_i.yaml"
    "config/LT.yaml"
    # "config/FLT_res.yaml"
)

# Number of processes/GPUs to use
NPROC_PER_NODE=8
# batch_size
DBS=8
# k loops
Ks=(
    "3"
)

# layer
Layers=(
    "3"
    "6"
    "12"
    "18"
)


# ------------------------------------------------
# exp execution

for CONFIG in "${CONFIGS[@]}"; do
    uv run torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
        --config $CONFIG \
        --device_batch_size 8 \
        --loss_type END \
        -k 12 \
        -l 6
done

rm -rf ${LOCAL}
