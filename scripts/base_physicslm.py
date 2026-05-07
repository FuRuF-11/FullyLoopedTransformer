"""
scripts/base_physicslm.py

Pretraining on PhysicsLM4-style synthetic tasks (Brevo, Depo, Mano, Lano).

Token IDs are used directly — no BPE tokenizer.  All synthetic tasks share
a small integer vocabulary that fits within VOCAB_SIZE = 10000.
Training objective: standard next-token prediction (GPT-style) on all tokens.
max_seq_len = 4096.  Evaluation: val CE loss + per-task greedy accuracy.

Usage:
    # single GPU
    python -m scripts.base_physicslm

    # distributed, 8 GPUs
    torchrun --standalone --nproc_per_node=8 -m scripts.base_physicslm

    # CPU debug
    python -m scripts.base_physicslm --depth=4 --device_batch_size=1 \\
        --total_batch_size=4096 --num_iterations=100 --eval_every=20 \\
        --acc_eval_every=50 --acc_eval_samples=10 --no_compile
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import random
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
import wandb

from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint
from nanochat.common import (
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    print0,
)
from nanochat.gpt import GPT, GPTConfig

from synthetic.brevo import evaluate as eval_brevo
from synthetic.brevo import generate as gen_brevo
from synthetic.depo import evaluate as eval_depo
from synthetic.depo import generate as gen_depo
from synthetic.lano import VALID_CONFIGS
from synthetic.lano import evaluate as eval_lano
from synthetic.lano import generate as gen_lano
from synthetic.mano import evaluate as eval_mano
from synthetic.mano import generate as gen_mano

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
# All synthetic token IDs fit in [0, 9999]:
#   Brevo  : content 1-125, special 9997-9999
#   Depo   : content 1-200, special 9000-9500, 9700, 9999
#   Mano   : ops 1-8, values 5000-5022, special 9998-9999
#   Lano   : content 1-9
VOCAB_SIZE = 10000

# ---------------------------------------------------------------------------
# Task generators  (each returns a flat list[int] of token IDs)
# ---------------------------------------------------------------------------
TASK_GENERATORS: dict = {
    # Brevo1: single-token nodes, N in [30, 110]
    "brevo1": lambda rng: gen_brevo(N=rng.randint(30, 110), rng=rng)[0],
    # Brevo2: multi-token nodes (words), N in [30, 50]
    "brevo2": lambda rng: gen_brevo(N=rng.randint(30, 50), multi=True, rng=rng)[0],
    # Depo1: large mini_vocab (single/double-token words), N in [225, 375]
    "depo1": lambda rng: gen_depo(
        N=rng.randint(225, 375), K=8,
        mini_vocab=50, min_tlen=1, max_tlen=2, rng=rng
    )[0],
    # Depo2: default mini_vocab (5-7 token words), N in [75, 125]
    "depo2": lambda rng: gen_depo(N=rng.randint(75, 125), K=16, rng=rng)[0],
    # Mano: arithmetic expressions, depth L in [5, 16]
    "mano": lambda rng: gen_mano(L=rng.randint(5, 16), ops="asm", rng=rng),
    # Lano: CFG sequences, random config from all 9 variants
    "lano": lambda rng: gen_lano(rng.choice(VALID_CONFIGS), rng=rng)[0],
}

DEFAULT_TASK_WEIGHTS: dict = {
    "brevo1": 1.0,
    "brevo2": 0.5,
    "depo1":  1.0,
    "depo2":  0.5,
    "mano":   1.0,
    "lano":   1.0,
}

# ---------------------------------------------------------------------------
# On-the-fly synthetic data loader
# ---------------------------------------------------------------------------
def synthetic_data_loader(B: int, T: int, device: str, seed: int,
                           task_weights: dict | None = None):
    """Infinite generator of (x, y) training batches from synthetic tasks.

    Samples are concatenated end-to-end (data packing) and sliced into
    fixed-length B×T chunks.  Each DDP rank should pass a distinct seed.

    Yields:
        x: LongTensor [B, T]  — input token IDs
        y: LongTensor [B, T]  — next-token targets (x shifted by 1)
    """
    if task_weights is None:
        task_weights = DEFAULT_TASK_WEIGHTS

    tasks   = list(task_weights.keys())
    weights = [task_weights[t] for t in tasks]
    gens    = {t: TASK_GENERATORS[t] for t in tasks}

    rng      = random.Random(seed)
    use_cuda = (device == "cuda")
    needed   = B * (T + 1)   # +1 so we can create both x and y
    buffer: list[int] = []

    while True:
        while len(buffer) < needed:
            task = rng.choices(tasks, weights=weights)[0]
            buffer.extend(gens[task](rng))

        chunk  = buffer[:needed]
        buffer = buffer[needed:]

        arr = torch.tensor(chunk, dtype=torch.long, pin_memory=use_cuda)
        arr = arr.view(B, T + 1)
        x = arr[:, :-1].to(device=device, non_blocking=use_cuda)
        y = arr[:, 1:].to(device=device, non_blocking=use_cuda)
        yield x, y


# ---------------------------------------------------------------------------
# Per-task accuracy  (greedy auto-regressive decoding)
# ---------------------------------------------------------------------------
@torch.no_grad()
def _greedy_decode(model, prompt: list[int], max_new: int,
                   stop_token: int | None, device: str) -> list[int]:
    """Single-sequence greedy decode (no KV cache, for short eval sequences)."""
    tokens = list(prompt)
    for _ in range(max_new):
        x      = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = model(x)          # (1, T, vocab)  — targets=None → logits
        nt     = int(logits[0, -1].argmax())
        tokens.append(nt)
        if stop_token is not None and nt == stop_token:
            break
    return tokens[len(prompt):]


@torch.no_grad()
def evaluate_task_accuracy(model, device: str, n_samples: int = 50,
                            seed: int = 7) -> dict[str, float]:
    """Evaluate per-task greedy accuracy on freshly generated test samples.

    Tasks and prompt/answer splits:
      brevo : prompt = [BOS, edges, 9997, query, 9998]  answer = topo + EOS
      depo  : prompt = [BOS, preamble, 9000+k, query, 9500]  answer = word
      mano  : prompt = tokens[:-1]   answer = 1 token (5000+ans)
      lano  : prompt = first 20% tokens   answer = remaining tokens

    Returns:
        dict {task_name: accuracy in [0, 1]}
    """
    rng = random.Random(seed)
    results: dict[str, float] = {}

    # ---- Brevo (single-token nodes) ----
    correct = 0
    for _ in range(n_samples):
        s    = gen_brevo(N=rng.randint(30, 110), rng=rng)
        toks = s[0]
        # First 9998 in the sequence is BOS-1 (answer-start marker)
        split  = toks.index(9998) + 1          # inclusive
        prompt = toks[:split]
        gen    = _greedy_decode(model, prompt, max_new=128,
                                stop_token=9998, device=device)
        ok, _, _ = eval_brevo(prompt + gen)
        if ok:
            correct += 1
    results["brevo"] = correct / n_samples

    # ---- Depo (single query, mini_vocab=50) ----
    correct = valid_total = 0
    for _ in range(n_samples):
        s    = gen_depo(N=rng.randint(50, 150), K=4,
                        mini_vocab=50, min_tlen=1, max_tlen=2, M=1, rng=rng)
        toks = s[0]
        try:
            sep = toks.index(9500)
        except ValueError:
            continue
        prompt  = toks[:sep + 1]    # inclusive of 9500 separator
        # Generate one answer word token-by-token; stop at end-of-word (tok > 50)
        gen: list[int] = []
        cur = list(prompt)
        for _ in range(30):
            x      = torch.tensor([cur], dtype=torch.long, device=device)
            logits = model(x)
            nt     = int(logits[0, -1].argmax())
            if nt >= 9000:              # hit a special token unexpectedly
                break
            gen.append(nt)
            cur.append(nt)
            if nt > 50:                 # end-of-word marker for mini_vocab=50
                break
        ok, nc, nt_count = eval_depo({0: toks[:sep + 1] + gen}, mini_vocab=50)
        valid_total += 1
        if ok and nt_count > 0:
            correct += 1
    results["depo"] = correct / max(1, valid_total)

    # ---- Mano (1-token answer: 5000 + answer_value) ----
    correct = 0
    for _ in range(n_samples):
        toks   = gen_mano(L=10, ops="asm", rng=rng)
        prompt = toks[:-1]
        gen    = _greedy_decode(model, prompt, max_new=1,
                                stop_token=None, device=device)
        if gen and gen[0] == toks[-1]:
            correct += 1
    results["mano"] = correct / n_samples

    # ---- Lano (give first 20 % as prefix, generate rest, check CFG) ----
    correct = 0
    for _ in range(n_samples):
        cfg  = rng.choice(VALID_CONFIGS)
        s    = gen_lano(cfg, rng=rng)
        toks = s[0]
        k       = max(1, len(toks) // 5)
        prompt  = toks[:k]
        gen     = _greedy_decode(model, prompt, max_new=len(toks) - k + 32,
                                 stop_token=None, device=device)
        full    = (prompt + gen)[: len(toks)]   # trim to reference length
        if len(full) == len(toks) and eval_lano(full, cfg):
            correct += 1
    results["lano"] = correct / n_samples

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Synthetic pretraining on PhysicsLM4 tasks"
)
# Model
parser.add_argument("--model_type",       type=str,   default="FLT")
parser.add_argument("-k",                 type=int,   default=1,
                    help="Loop depth K for recurrent model variants")
parser.add_argument("--depth", "-l",      type=int,   default=12,
                    help="Number of transformer layers (model depth)")
parser.add_argument("--loss_type",        type=str,   default="END",
                    help="Which loop steps contribute to loss: END | FEW | STEP")
# Optimization
parser.add_argument("--device_batch_size",type=int,   default=4)
parser.add_argument("--total_batch_size", type=int,   default=131072,
                    help="Total tokens per optimizer step (default 128 K)")
parser.add_argument("--num_iterations",   type=int,   default=5000)
parser.add_argument("--embedding_lr",     type=float, default=0.2)
parser.add_argument("--unembedding_lr",   type=float, default=0.004)
parser.add_argument("--matrix_lr",        type=float, default=0.02)
parser.add_argument("--weight_decay",     type=float, default=0.0)
parser.add_argument("--grad_clip",        type=float, default=1.0)
parser.add_argument("--warmup_ratio",     type=float, default=0.05)
parser.add_argument("--warmdown_ratio",   type=float, default=0.2)
parser.add_argument("--final_lr_frac",    type=float, default=0.0)
# Eval / save
parser.add_argument("--eval_every",       type=int,   default=200,
                    help="Steps between val-loss evaluations")
parser.add_argument("--eval_batches",     type=int,   default=20,
                    help="Number of batches used for val-loss estimation")
parser.add_argument("--acc_eval_every",   type=int,   default=1000,
                    help="Steps between per-task accuracy evaluations (-1 disables)")
parser.add_argument("--acc_eval_samples", type=int,   default=50,
                    help="Samples per task for accuracy evaluation")
parser.add_argument("--save_every",       type=int,   default=-1,
                    help="Checkpoint every N steps (-1 = only at end)")
parser.add_argument("--resume_from_step", type=int,   default=-1)
# Runtime
parser.add_argument("--project",          type=str,   default="FullyLoopedTransformer-synthetic")
parser.add_argument("--no_compile",       action="store_true", default=False)
parser.add_argument("--activation_offload", action="store_true", default=False)

args = parser.parse_args()
run  = f"{args.model_type}_K{args.k}_D{args.depth}_{args.loss_type}_synth"

# ---------------------------------------------------------------------------
# Runtime / DDP init
# ---------------------------------------------------------------------------
device_type  = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = (ddp_rank == 0)
autocast_ctx   = (
    torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
    if device_type == "cuda" else nullcontext()
)
synchronize     = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory  = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

wandb_run = (
    DummyWandb()
    if not master_process
    else wandb.init(project=args.project, name=run, config=vars(args))
)

# ---------------------------------------------------------------------------
# Model  (vocab_size = 10000, seq_len = 4096)
# ---------------------------------------------------------------------------
MAX_SEQ_LEN  = 4096
depth        = args.depth
num_layers   = depth
model_dim    = depth * 64
num_heads    = max(1, (model_dim + 127) // 128)
num_kv_heads = num_heads

print0(f"VOCAB_SIZE={VOCAB_SIZE}  num_layers={num_layers}  "
       f"model_dim={model_dim}  num_heads={num_heads}")

model_config_kwargs = dict(
    model_type        = args.model_type,
    loss_type         = args.loss_type,
    K                 = args.k,
    sequence_len      = MAX_SEQ_LEN,
    vocab_size        = VOCAB_SIZE,
    n_layer           = num_layers,
    n_head            = num_heads,
    n_kv_head         = num_kv_heads,
    n_embd            = model_dim,
    activation_offload= args.activation_offload,
)
with torch.device("meta"):
    model_config = GPTConfig(**model_config_kwargs)
    model        = GPT(model_config)
model.to_empty(device=device)
model.init_weights()

base_dir       = get_base_dir()
checkpoint_dir = os.path.join(base_dir, "synthetic_checkpoints", run)
resuming       = (args.resume_from_step != -1)
if resuming:
    print0(f"Resuming from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(
        checkpoint_dir, args.resume_from_step, device,
        load_optimizer=True, rank=ddp_rank,
    )
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data

orig_model = model
if not args.no_compile:
    model = torch.compile(model, dynamic=False)
    print0("torch.compile enabled")
else:
    print0("torch.compile disabled (eager mode)")

num_params          = sum(p.numel() for p in model.parameters())
num_flops_per_token = model.estimate_flops()
print0(f"Parameters: {num_params:,}   FLOPs/token: {num_flops_per_token:e}")

# ---------------------------------------------------------------------------
# Gradient accumulation
# ---------------------------------------------------------------------------
total_batch_size   = args.total_batch_size
device_batch_size  = args.device_batch_size
tokens_per_iter    = device_batch_size * MAX_SEQ_LEN
world_tokens_per_iter = tokens_per_iter * ddp_world_size
assert total_batch_size % world_tokens_per_iter == 0, (
    f"total_batch_size={total_batch_size} must be divisible by "
    f"world×device×seq = {world_tokens_per_iter}"
)
grad_accum_steps = total_batch_size // world_tokens_per_iter
print0(f"Tokens/micro-batch/rank: {tokens_per_iter:,}   "
       f"grad_accum_steps: {grad_accum_steps}")

# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------
optimizers = model.setup_optimizers(
    unembedding_lr = args.unembedding_lr,
    embedding_lr   = args.embedding_lr,
    matrix_lr      = args.matrix_lr,
    weight_decay   = args.weight_decay,
)
adamw_optimizer, muon_optimizer = optimizers

if resuming:
    for opt, dat in zip(optimizers, optimizer_data):
        opt.load_state_dict(dat)
    del optimizer_data

# ---------------------------------------------------------------------------
# Data loaders
# Each DDP rank gets a distinct seed for training data diversity.
# Validation uses a fixed seed so the benchmark is reproducible.
# ---------------------------------------------------------------------------
train_loader = synthetic_data_loader(
    device_batch_size, MAX_SEQ_LEN, device,
    seed=1337 + ddp_rank * 1000,
)
# Val seed is deliberately distant from all training seeds (1337 + rank*1000) so
# there is negligible probability of generating the same sample.  This is a
# *statistical* separation: because data is synthesised on-the-fly there is no
# hard guarantee, but the probability of a collision is negligible given the
# enormous sample space of each task.
VAL_SEED = 0   # constant — val samples never change between runs

# ---------------------------------------------------------------------------
# LR / momentum schedulers
# ---------------------------------------------------------------------------
num_iterations = args.num_iterations

def get_lr_multiplier(it: int) -> float:
    warmup   = round(args.warmup_ratio   * num_iterations)
    warmdown = round(args.warmdown_ratio * num_iterations)
    if it < warmup:
        return (it + 1) / warmup
    if it <= num_iterations - warmdown:
        return 1.0
    progress = (num_iterations - it) / warmdown
    return progress + (1.0 - progress) * args.final_lr_frac

def get_muon_momentum(it: int) -> float:
    frac = min(it / 300, 1.0)
    return (1.0 - frac) * 0.85 + frac * 0.95

# ---------------------------------------------------------------------------
# Loop state
# ---------------------------------------------------------------------------
if not resuming:
    step               = 0
    val_loss           = float("inf")
    min_val_loss       = float("inf")
    smooth_train_loss  = 0.0
    total_training_time = 0.0
else:
    step               = meta_data["step"]
    val_loss           = meta_data.get("val_loss", float("inf"))
    loop_state         = meta_data["loop_state"]
    min_val_loss       = loop_state["min_val_loss"]
    smooth_train_loss  = loop_state["smooth_train_loss"]
    total_training_time= loop_state["total_training_time"]

# prefetch first training batch
x, y = next(train_loader)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
while True:
    last_step        = (step == num_iterations)
    flops_so_far     = num_flops_per_token * total_batch_size * step

    # ---- Validation loss ------------------------------------------------
    if last_step or step % args.eval_every == 0:
        orig_model.eval()
        val_loader = synthetic_data_loader(
            device_batch_size, MAX_SEQ_LEN, device, seed=VAL_SEED
        )
        total_val = 0.0
        with torch.no_grad(), autocast_ctx:
            for _ in range(args.eval_batches):
                xv, yv    = next(val_loader)
                loss_list = orig_model(xv, yv)
                total_val += torch.mean(torch.stack(loss_list)).item()
        val_loss = total_val / args.eval_batches
        if val_loss < min_val_loss:
            min_val_loss = val_loss
        print0(f"Step {step:05d} | val loss: {val_loss:.4f}  (min: {min_val_loss:.4f})")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/loss": val_loss,
        })
        orig_model.train()

    # ---- Per-task accuracy -----------------------------------------------
    acc_results: dict = {}
    if args.acc_eval_every > 0 and (
        last_step or (step > 0 and step % args.acc_eval_every == 0)
    ):
        orig_model.eval()
        with autocast_ctx:
            acc_results = evaluate_task_accuracy(
                orig_model, device,
                n_samples=args.acc_eval_samples,
                # Offset by a large prime so this seed never coincides with any
                # training seed (1337 + rank*1000).  Still varies per step so
                # each evaluation tests a fresh set of samples.
                seed=step + 999983,
            )
        acc_str = "  ".join(f"{k}: {v:.3f}" for k, v in acc_results.items())
        print0(f"Step {step:05d} | accuracy  {acc_str}")
        wandb_run.log({
            "step": step,
            **{f"acc/{k}": v for k, v in acc_results.items()},
        })
        orig_model.train()

    # ---- Checkpoint ------------------------------------------------------
    if last_step or (
        step > 0
        and step != args.resume_from_step
        and args.save_every > 0
        and step % args.save_every == 0
    ):
        save_checkpoint(
            checkpoint_dir, step,
            orig_model.state_dict(),
            [opt.state_dict() for opt in optimizers],
            {
                "step":         step,
                "val_loss":     val_loss,
                "model_config": model_config_kwargs,
                "args":         vars(args),
                "loop_state": {
                    "min_val_loss":         min_val_loss,
                    "smooth_train_loss":    smooth_train_loss,
                    "total_training_time":  total_training_time,
                },
            },
            rank=ddp_rank,
        )

    if last_step:
        break

    # ---- Single optimisation step ----------------------------------------
    synchronize()
    t0 = time.time()

    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss_list  = model(x, y)
            loss       = torch.mean(torch.stack(loss_list))
        train_loss = loss_list[-1].detach()   # last loop step loss, for logging
        loss       = loss / grad_accum_steps
        loss.backward()
        x, y = next(train_loader)             # prefetch while GPU is busy

    grad_norm = -1.0
    if args.grad_clip > 0.0:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            orig_model.parameters(), args.grad_clip
        ).item()

    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    for group in muon_optimizer.param_groups:
        group["momentum"] = get_muon_momentum(step)
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)

    synchronize()
    dt = time.time() - t0
    if step > 10:
        total_training_time += dt

    # ---- Logging ---------------------------------------------------------
    ema_beta          = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased          = smooth_train_loss / (1 - ema_beta ** (step + 1))
    tok_per_sec       = int(total_batch_size / dt)
    flops_per_sec     = num_flops_per_token * total_batch_size / dt
    mfu               = 100 * flops_per_sec / (989e12 * ddp_world_size)

    print0(
        f"step {step:05d}/{num_iterations:05d} ({100*step/num_iterations:.1f}%) "
        f"| loss: {debiased:.4f} | grad: {grad_norm:.3f} "
        f"| lrm: {lrm:.3f} | {dt*1000:.0f}ms | {tok_per_sec:,} tok/s "
        f"| mfu: {mfu:.1f}% | {total_training_time/60:.1f}m"
    )

    if step % 50 == 0:
        wandb_run.log({
            "step":                    step,
            "total_training_flops":    flops_so_far,
            "total_training_time":     total_training_time,
            "train/loss":              debiased,
            "train/loss_unsmooth":     train_loss.item(),
            "train/lrm":               lrm,
            "train/grad_norm":         grad_norm,
            "train/tok_per_sec":       tok_per_sec,
            "train/mfu":               mfu,
        })

    step += 1

# ---------------------------------------------------------------------------
# End-of-run summary
# ---------------------------------------------------------------------------
print0(f"Peak memory : {get_max_memory() / 1024 / 1024:.1f} MiB")
print0(f"Training time: {total_training_time / 60:.1f} m")
print0(f"Min val loss : {min_val_loss:.4f}")
if acc_results:
    print0("Final accuracy: " + "  ".join(
        f"{k}: {v:.3f}" for k, v in acc_results.items()
    ))

wandb_run.finish()
compute_cleanup()
