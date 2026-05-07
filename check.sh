#!/bin/bash

# -----------------------------------------------------------------------------
# Diagnostic training runs: 2000 steps, per-layer grad norms + hidden norms
# Results logged to wandb project "FLT_check"
# No model checkpoints saved

set -ex

export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
# -----------------------------------------------------------------------------
# wandb setup
# export WANDB_API_KEY="<YOUR_WANDB_API_KEY>"
export WANDB_INIT_TIMEOUT=300
export WANDB_HTTP_TIMEOUT=120
uv run wandb login

# -----------------------------------------------------------------------------
# Experiment settings

NPROC_PER_NODE=4
DBS=8

CONFIGS=(
    "config/LT_ia.yaml"
    "config/LT_i.yaml"
    "config/FLT_res.yaml"
)

Ks=(
    "6"
    "9"
    "12"
)

Layers=(
    "6"
    "12"
)

# -----------------------------------------------------------------------------
# Run diagnostic checks: all (config, K, depth) combinations

for CONFIG in "${CONFIGS[@]}"; do
    for K in "${Ks[@]}"; do
        uv run torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_check -- \
            --config $CONFIG \
            --device_batch_size $DBS \
            --loss_type END \
            --num_iterations 2000 \
            -k $K \
            -l 6
    done
done

# -----------------------------------------------------------------------------

rm -rf ${LOCAL}
