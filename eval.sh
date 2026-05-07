#!/bin/bash

set -ex

# -----------------------------------------------------------------------------
# Environment setup
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"

# -----------------------------------------------------------------------------
# wandb setup
# export WANDB_API_KEY="<YOUR_WANDB_API_KEY>"
export WANDB_INIT_TIMEOUT=300
export WANDB_HTTP_TIMEOUT=120
uv run wandb login

# -----------------------------------------------------------------------------
# Eval settings

# Number of GPUs
NPROC_PER_NODE=4

# Per-device batch size for evaluation
DBS=4

# Evaluation modes: comma-separated from {core, bpb, sample}
EVAL_MODES="core,bpb,sample"

# wandb project name
PROJECT="FullyLoopedTransformer"

# Max examples per CORE task (-1 = all)
MAX_PER_TASK=-1

# -----------------------------------------------------------------------------
# Models to evaluate
# Each entry is the absolute path to a checkpoint directory.

CHECKPOINT_DIRS=(
    "$NANOCHAT_BASE_DIR/base_checkpoints/FLT_L12_D12_END_attnfull_kv3"
    "$NANOCHAT_BASE_DIR/base_checkpoints/FLT_L12_D12_END_attnmla_r128"
    "$NANOCHAT_BASE_DIR/base_checkpoints/FLT_L12_D12_END_attnfull_winSSSL"
)

# -----------------------------------------------------------------------------
# Budget evaluation loop

for CKPT_DIR in "${CHECKPOINT_DIRS[@]}"; do
    echo "========================================"
    echo "Evaluating: $CKPT_DIR"
    echo "========================================"
    uv run torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_budget \
        --checkpoint-dir "$CKPT_DIR" \
        --device_batch_size $DBS \
        --eval "$EVAL_MODES" \
        --max-per-task $MAX_PER_TASK \
        --project "$PROJECT" \
        --no-budget
done

# -----------------------------------------------------------------------------

rm -rf ${LOCAL}
