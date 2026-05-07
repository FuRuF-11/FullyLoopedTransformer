"""
Diagnostic training script for analyzing looped transformer training instability.

Runs for a fixed 2000 steps, logs per-layer gradient norms and per-loop hidden state
norms every step to wandb project "FLT_check". No model checkpoints are saved.

Single GPU / CPU:
python -m scripts.base_check -- --depth=4 --max_seq_len=512 \
    --device_batch_size=1 --total_batch_size=512 --num_iterations=20

Multi-GPU:
torchrun --standalone --nproc_per_node=4 -m scripts.base_check -- \
    --config config/FLT.yaml --depth=12 -k 6
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
from contextlib import nullcontext

import wandb
import torch
import numpy as np

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader, tokenizing_distributed_data_loader_with_state
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.utils import Config

print_banner()

# -----------------------------------------------------------------------------
# User settings

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config/FLT.yaml")
parser.add_argument("--project", type=str, default="FullyLoopedTransformer")
parser.add_argument("-k", type=int)
parser.add_argument("-l", type=int)
parser.add_argument("--device_batch_size", type=int)
parser.add_argument("--loss_type", type=str, default="END")
parser.add_argument("--activation_offload", action="store_true", default=False)
parser.add_argument("--activation_offload_keep_last", type=int, default=1)
parser.add_argument("--attn_type", type=str, default=None)
parser.add_argument("--kv_lora_rank", type=int, default=None)
parser.add_argument("--window_pattern", type=str, default=None)
parser.add_argument("--n_kv_head", type=int, default=None)
parser.add_argument("--num_iterations", type=int, default=-1,
                    help="Number of training steps (-1 = auto-scale from target_param_data_ratio).")
# depth is a shortcut for -l
parser.add_argument("--depth", type=int, default=None)

args = parser.parse_args()
config = Config(args.config)
config.loss_type = args.loss_type
config.project = args.project
config.k = args.k if args.k is not None else config.k
# --depth and -l are aliases
if args.depth is not None:
    config.num_hidden_layers = args.depth
elif args.l is not None:
    config.num_hidden_layers = args.l
config.device_batch_size = args.device_batch_size if args.device_batch_size is not None else config.device_batch_size
config.activation_offload = args.activation_offload
config.activation_offload_keep_last = args.activation_offload_keep_last
config.attn_type = args.attn_type if args.attn_type is not None else getattr(config, 'attn_type', 'full')
config.kv_lora_rank = args.kv_lora_rank if args.kv_lora_rank is not None else getattr(config, 'kv_lora_rank', 128)
config.window_pattern = args.window_pattern if args.window_pattern is not None else getattr(config, 'window_pattern', 'L')
config.n_kv_head = args.n_kv_head

# Diagnostic run: no torch.compile (allows collect_norms kwarg to work cleanly)
use_compile = False

# Build run name with check_ prefix
_attn_suffix = f"_attn{config.attn_type}"
if config.attn_type == "mla":
    _attn_suffix += f"_r{config.kv_lora_rank}"
if config.n_kv_head is not None:
    _attn_suffix += f"_kv{config.n_kv_head}"
if config.window_pattern != "L":
    _attn_suffix += f"_win{config.window_pattern}"
run = f"check_{config.model_type}_L{config.k}_D{config.num_hidden_layers}_{config.loss_type}{_attn_suffix}"

# Runtime
device_type = ""
# Model architecture
depth = config.num_hidden_layers
max_seq_len = 1024
# Training horizon. Only one of these 3 will be used, in this order of precedence.
num_iterations = args.num_iterations  # explicit number of steps (-1 = disable)
target_flops = -1.0                   # calculate num_iterations to reach target_flops (-1 = disable)
target_param_data_ratio = 20          # calculate num_iterations to maintain fixed data:param ratio (Chinchilla=20) (-1 = disable)
device_batch_size = config.device_batch_size
total_batch_size = 524288
embedding_lr = 0.2
unembedding_lr = 0.004
weight_decay = 0.0
matrix_lr = 0.02
grad_clip = 1.0
warmup_ratio = 0.0
warmdown_ratio = 0.2
final_lr_frac = 0.0
# Evaluation (more frequent since only 2k steps; CORE disabled)
eval_every = 200
eval_tokens = 20 * 524288
core_metric_every = -1  # disabled
# No checkpointing
save_every = -1

config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
user_config = {k: globals()[k] for k in config_keys}
user_config["attn_type"] = config.attn_type
user_config["kv_lora_rank"] = config.kv_lora_rank
user_config["window_pattern"] = config.window_pattern
user_config["n_kv_head_override"] = config.n_kv_head

# -----------------------------------------------------------------------------
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project=config.project, name=run, config=user_config)

tokenizer = get_tokenizer()
token_bytes = get_token_bytes(tokenizer, device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# Model dimensions
num_layers = depth
model_dim = depth * 64
num_heads = max(1, (model_dim + 127) // 128)
if config.n_kv_head is not None:
    assert num_heads % config.n_kv_head == 0
    num_kv_heads = config.n_kv_head
else:
    num_kv_heads = num_heads
print0(f"num_layers: {num_layers}")
print0(f"model_dim: {model_dim}")
print0(f"num_heads: {num_heads}")
print0(f"num_kv_heads: {num_kv_heads} (GQA ratio: {num_heads // num_kv_heads}:1)")

tokens_per_fwdbwd = device_batch_size * max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# -----------------------------------------------------------------------------
model_config_kwargs = dict(
    model_type=config.model_type,
    loss_type=config.loss_type,
    K=config.k,
    sequence_len=max_seq_len,
    vocab_size=vocab_size,
    n_layer=num_layers,
    n_head=num_heads,
    n_kv_head=num_kv_heads,
    n_embd=model_dim,
    activation_offload=config.activation_offload,
    attn_type=config.attn_type,
    kv_lora_rank=config.kv_lora_rank,
    window_pattern=config.window_pattern,
)
with torch.device("meta"):
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)
model.to_empty(device=device)
model.init_weights()

orig_model = model
# No torch.compile for diagnostic script
print0("torch.compile disabled (diagnostic mode)")

# ---- Per-loop gradient norm tracking via backward hooks ----
_model_type_has_per_loop_grads = True
# Independent countdown counters for block / attn / mlp — each fires K times
# per backward pass (once per loop iteration), counting down from K to 0.
_block_bwd_counts = [0] * num_layers
_attn_bwd_counts  = [0] * num_layers
_mlp_bwd_counts   = [0] * num_layers

def _reset_bwd_counts():
    for i in range(num_layers):
        _block_bwd_counts[i] = config.k
        _attn_bwd_counts[i]  = config.k
        _mlp_bwd_counts[i]   = config.k

def _register_loop_hooks(norms_dict):
    """Register full_backward_hook on block, block.attn, and block.mlp.

    norms_dict keys:
        ('block', k_idx, i) — gradient of the block's output per loop
        ('attn',  k_idx, i) — gradient of attn sub-module's output per loop
        ('mlp',   k_idx, i) — gradient of mlp sub-module's output per loop
    """
    handles = []
    for i_idx, block in enumerate(orig_model.transformer.h):
        def _make_block_hook(i=i_idx):
            def hook(module, grad_input, grad_output):
                _block_bwd_counts[i] -= 1
                k_idx = _block_bwd_counts[i]
                gn = grad_output[0].detach().float().norm().item() \
                     if grad_output[0] is not None else 0.0
                norms_dict[('block', k_idx, i)] = gn
            return hook

        def _make_attn_hook(i=i_idx):
            def hook(module, grad_input, grad_output):
                _attn_bwd_counts[i] -= 1
                k_idx = _attn_bwd_counts[i]
                gn = grad_output[0].detach().float().norm().item() \
                     if grad_output[0] is not None else 0.0
                norms_dict[('attn', k_idx, i)] = gn
            return hook

        def _make_mlp_hook(i=i_idx):
            def hook(module, grad_input, grad_output):
                _mlp_bwd_counts[i] -= 1
                k_idx = _mlp_bwd_counts[i]
                gn = grad_output[0].detach().float().norm().item() \
                     if grad_output[0] is not None else 0.0
                norms_dict[('mlp', k_idx, i)] = gn
            return hook

        handles.append(block.register_full_backward_hook(_make_block_hook()))
        handles.append(block.attn.register_full_backward_hook(_make_attn_hook()))
        handles.append(block.mlp.register_full_backward_hook(_make_mlp_hook()))
    return handles

num_params = sum(p.numel() for p in model.parameters())
print0(f"Number of parameters: {num_params:,}")
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# Calculate number of iterations. Either it is given, or from target flops, or from target data:param ratio (in that order)
assert num_iterations > 0 or target_param_data_ratio > 0 or target_flops > 0
if num_iterations > 0:
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif target_flops > 0:
    num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif target_param_data_ratio > 0:
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
total_tokens = total_batch_size * num_iterations
print0(f"Total diagnostic tokens: {total_tokens:,}")
print0(f"Tokens : Params ratio: {total_batch_size * num_iterations / num_params:.2f}")

# -----------------------------------------------------------------------------
optimizers = model.setup_optimizers(unembedding_lr=unembedding_lr, embedding_lr=embedding_lr, matrix_lr=matrix_lr, weight_decay=weight_decay)
adamw_optimizer, muon_optimizer = optimizers

# -----------------------------------------------------------------------------
base_dir = get_base_dir()
tokens_dir = os.path.join(base_dir, "tokenized_data")
train_loader = tokenizing_distributed_data_loader_with_state(device_batch_size, max_seq_len, split="train", device=device, resume_state_dict=None)
build_val_loader = lambda: tokenizing_distributed_data_loader(device_batch_size, max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader)

# -----------------------------------------------------------------------------
def get_lr_multiplier(it):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac

def get_muon_momentum(it):
    frac = min(it / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

# -----------------------------------------------------------------------------
step = 0
min_val_bpb = float("inf")
smooth_train_loss = 0
total_training_time = 0

all_loss = []
all_grad_norm = []

# -----------------------------------------------------------------------------
while True:
    last_step = step == num_iterations

    # Validation BPB
    if last_step or step % eval_every == 0:
        model.eval()
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": num_flops_per_token * total_batch_size * step,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()
        if device_type == "cuda":
            torch.cuda.empty_cache()

    if last_step:
        break

    # -------------------------------------------------------------------------
    # Training step
    synchronize()
    t0 = time.time()
    hidden_norms = None
    loop_grad_norms = {}
    _loop_hook_handles = []
    for micro_step in range(grad_accum_steps):
        is_last_micro = (micro_step == grad_accum_steps - 1)
        with autocast_ctx:
            result = model(x, y, collect_norms=True)
            loss_list, hidden_norms = result
            loss = torch.mean(torch.stack(loss_list))
        train_loss = loss_list[-1].detach()
        loss = loss / grad_accum_steps

        # Register per-loop backward hooks only on the last micro-step
        if is_last_micro and _model_type_has_per_loop_grads:
            loop_grad_norms = {}
            _reset_bwd_counts()
            _loop_hook_handles = _register_loop_hooks(loop_grad_norms)

        loss.backward()

        if _loop_hook_handles:
            for h in _loop_hook_handles:
                h.remove()
            _loop_hook_handles = []

        x, y, dataloader_state_dict = next(train_loader)

    # Gradient clipping
    grad_clip_enabled = grad_clip > 0.0
    if grad_clip_enabled:
        grad_norm = torch.nn.utils.clip_grad_norm_(orig_model.parameters(), grad_clip).item()

    # Per-layer gradient norms (computed after clipping, before zero_grad)
    layer_grad_norms = {}
    if grad_clip_enabled:
        for i, block in enumerate(orig_model.transformer.h):
            attn_gs = [p.grad.norm() for p in block.attn.parameters() if p.grad is not None]
            mlp_gs  = [p.grad.norm() for p in block.mlp.parameters()  if p.grad is not None]
            if attn_gs:
                layer_grad_norms[f"grad_norm/block_{i:02d}/attn"] = torch.stack(attn_gs).norm().item()
            if mlp_gs:
                layer_grad_norms[f"grad_norm/block_{i:02d}/mlp"]  = torch.stack(mlp_gs).norm().item()
        embed_g  = orig_model.transformer.wte.weight.grad
        lmhead_g = orig_model.lm_head.weight.grad
        if embed_g  is not None:
            layer_grad_norms["grad_norm/embed"]   = embed_g.norm().item()
        if lmhead_g is not None:
            layer_grad_norms["grad_norm/lm_head"] = lmhead_g.norm().item()

    # Optimizer step
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    muon_momentum = get_muon_momentum(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0

    # -------------------------------------------------------------------------
    # Logging (every step)
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_so_far = num_flops_per_token * total_batch_size * step
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100
    if step > 10:
        total_training_time += dt

    print0(
        f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | "
        f"loss: {debiased_smooth_loss:.6f} | "
        + (f"grad norm: {grad_norm:.4f} | " if grad_clip_enabled else "")
        + f"lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f}"
    )

    log_data = {
        "step": step,
        "total_training_flops": flops_so_far,
        "total_training_time": total_training_time,
        "train/loss": debiased_smooth_loss,
        "train/loss_unsmooth": train_loss.item(),
        "train/lrm": lrm,
        "train/dt": dt,
        "train/tok_per_sec": tok_per_sec,
        "train/mfu": mfu,
    }
    if grad_clip_enabled:
        log_data["train/grad_norm"] = grad_norm
        log_data.update(layer_grad_norms)
    if hidden_norms is not None:
        for k, hn in enumerate(hidden_norms):
            log_data[f"hidden_norm/loop_{k:02d}"] = hn.item()
    if _model_type_has_per_loop_grads and loop_grad_norms:
        for k_idx in range(config.k):
            for i_idx in range(num_layers):
                # Block-level output gradient (all sub-modules combined)
                val = loop_grad_norms.get(('block', k_idx, i_idx))
                if val is not None:
                    log_data[f"grad_norm/block_{i_idx:02d}/loop_k{k_idx:02d}"] = val
                # Attn sub-module output gradient — all K loops on one chart per block
                val = loop_grad_norms.get(('attn', k_idx, i_idx))
                if val is not None:
                    log_data[f"grad_norm/block_{i_idx:02d}/attn/loop_k{k_idx:02d}"] = val
                # MLP sub-module output gradient
                val = loop_grad_norms.get(('mlp', k_idx, i_idx))
                if val is not None:
                    log_data[f"grad_norm/block_{i_idx:02d}/mlp/loop_k{k_idx:02d}"] = val
    wandb_run.log(log_data)

    # Milestone loss/grad std logging (end of warmup, start of warmdown, final)
    all_loss.append(train_loss.item())
    if grad_clip_enabled:
        all_grad_norm.append(grad_norm)
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if step == warmup_iters or step == num_iterations - warmdown_iters or last_step:
        all_loss_arr = np.array(all_loss)
        milestone_data = {"step": step, "train/loss_std": all_loss_arr.std()}
        if grad_clip_enabled:
            milestone_data["train/grad_norm_std"] = np.array(all_grad_norm).std()
        all_loss = []
        all_grad_norm = []
        wandb_run.log(milestone_data)

    step += 1

# -----------------------------------------------------------------------------
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time / 60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

wandb_run.finish()
compute_cleanup()
