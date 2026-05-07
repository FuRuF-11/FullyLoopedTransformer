# Fully Looped Transformer

A PyTorch implementation of looped transformer architectures for language modeling. The project trains and evaluates several model variants — **FLT** (Fully Looped Transformer), **LT** (Looped Transformer).

## Open-Source Foundation

This project is built on top of [nanochat](https://github.com/karpathy/nanochat), an open-source language model framework. The core infrastructure — including the training loop, tokenizer, dataloader, optimizer implementations (Muon, AdamW), evaluation harness, and web UI — originates from that codebase. The novel contributions of this project are the looped transformer architectures (FLT, LT, LT_i, LT_ia), the multi-step loop training objectives, and the associated experiments.

External resources inherited from the base project include:

- **Pretraining dataset**: `karpathy/fineweb-edu-100b-shuffle` hosted on HuggingFace
- **Evaluation bundle**: hosted at `karpathy-public.s3.us-west-2.amazonaws.com`
- **Optimizer code**: Muon optimizer attributed to Keller et al. / modded-nanogpt contributors

These references appear throughout the codebase and are part of the upstream open-source project, not specific to this work.

---

## Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- CUDA-capable GPU(s) (the default scripts target 8× A100 for training and 4× A100 for evaluation)
- A [Weights & Biases](https://wandb.ai) account for experiment tracking

Install all dependencies with:

```bash
uv sync --extra gpu
```

---

## Setup: Weights & Biases API Key

All three scripts log metrics to Weights & Biases. Before running any script, open the file and uncomment the `WANDB_API_KEY` line, then fill in your own key:

```bash
# In run.sh / eval.sh / check.sh, find this line and uncomment it:
export WANDB_API_KEY="<YOUR_WANDB_API_KEY>"
```

You can find your API key at <https://wandb.ai/authorize>.

---

## Step 1 — Pretraining (`run.sh`)

`run.sh` downloads the pretraining dataset, then trains all four model variants (FLT, LT_ia, LT_i, LT) sequentially using distributed training across 8 GPUs.

```bash
bash run.sh
```

**What it does:**

1. Creates the artifact cache directory at `$HOME/.cache/nanochat`.
2. Downloads ~250 shards of the pretraining text corpus (≈25 GB).
3. Launches `scripts.base_train` via `torchrun` for each config in `CONFIGS`:
   - `config/FLT.yaml`
   - `config/LT_ia.yaml`
   - `config/LT_i.yaml`
   - `config/LT.yaml`
4. Each run uses `--loss_type END`, `K=12` loop iterations, and `L=6` layers.
5. Checkpoints are saved to `$HOME/.cache/nanochat/base_checkpoints/<run_name>/`.

**Key parameters (editable at the top of `run.sh`):**

| Variable | Default | Description |
|---|---|---|
| `NPROC_PER_NODE` | `8` | Number of GPUs per node |
| `DBS` | `8` | Per-device batch size |
| `CONFIGS` | four variants | Model configs to train |

---

## Step 2 — Evaluation (`eval.sh`)

`eval.sh` evaluates one or more trained checkpoints on the CORE benchmark, bits-per-byte (BPB), and perplexity metrics.

```bash
bash eval.sh
```

**What it does:**

1. Iterates over the checkpoint directories listed in `CHECKPOINT_DIRS`.
2. For each checkpoint, runs `scripts.base_budget` with evaluation modes `core,bpb,sample`.
3. Results are logged to the wandb project `FullyLoopedTransformer`.

**Before running**, update `CHECKPOINT_DIRS` in `eval.sh` to point to the checkpoints produced by `run.sh`:

```bash
CHECKPOINT_DIRS=(
    "$NANOCHAT_BASE_DIR/base_checkpoints/<your_run_name>"
    ...
)
```

**Key parameters (editable in `eval.sh`):**

| Variable | Default | Description |
|---|---|---|
| `NPROC_PER_NODE` | `4` | Number of GPUs |
| `DBS` | `4` | Per-device batch size |
| `EVAL_MODES` | `core,bpb,sample` | Evaluation modes to run |
| `MAX_PER_TASK` | `-1` | Max examples per task (`-1` = all) |

---

## Step 3 — Diagnostic Checks (`check.sh`)

`check.sh` runs short diagnostic training runs (2000 steps each) and logs per-layer gradient norms and hidden-state norms to wandb. No checkpoints are saved. This is useful for inspecting the internal training dynamics of the looped models.

```bash
bash check.sh
```

**What it does:**

1. Iterates over all combinations of configs (`LT_ia`, `LT_i`, `FLT_res`) and loop counts `K ∈ {6, 9, 12}`.
2. Each run trains for 2000 steps with `--loss_type END` and `L=6` layers.
3. Per-layer gradient norms and hidden-state norms are logged every step to the wandb project `FLT_check`.

**Key parameters (editable in `check.sh`):**

| Variable | Default | Description |
|---|---|---|
| `NPROC_PER_NODE` | `4` | Number of GPUs |
| `DBS` | `8` | Per-device batch size |
| `CONFIGS` | three variants | Model configs to diagnose |
| `Ks` | `6, 9, 12` | Loop counts to sweep |

---

## Project Structure

```
.
├── config/             # Model configuration YAML files
│   ├── FLT.yaml
│   ├── FLT_res.yaml
│   ├── LT.yaml
│   ├── LT_i.yaml
│   └── LT_ia.yaml
├── nanochat/           # Core library (model, tokenizer, dataloader, engine, ...)
├── scripts/            # Training and evaluation entry points
│   ├── base_train.py   # Pretraining
│   ├── base_eval.py    # CORE / BPB evaluation
│   ├── base_budget.py  # Budget-controlled evaluation
│   ├── base_check.py   # Diagnostic gradient/norm logging
│   └── base_rl.py      # Reinforcement learning fine-tuning
├── tasks/              # Benchmark task implementations (ARC, GSM8K, MMLU, ...)
├── tests/              # Unit tests
├── run.sh              # Step 1: pretraining
├── eval.sh             # Step 2: evaluation
├── check.sh            # Step 3: diagnostic checks
└── pyproject.toml
```

---

## Typical Workflow

```
1. Fill in WANDB_API_KEY in run.sh / eval.sh / check.sh
2. bash run.sh          # pretrain all model variants
3. bash eval.sh         # evaluate checkpoints on benchmarks
4. bash check.sh        # inspect gradient / hidden-state dynamics
```
