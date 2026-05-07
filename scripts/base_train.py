"""
Train model. Run as:

python base_train.py

or distributed as:

torchrun --nproc_per_node=8 base_train.py

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max_seq_len=512 --device_batch_size=1 --eval_tokens=512 --core_metric_every=-1 --total_batch_size=512 --num_iterations=20
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
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.utils import Config
from scripts.base_eval import evaluate_model

print_banner()

# -----------------------------------------------------------------------------
# User settings

import argparse
parser = argparse.ArgumentParser()
# config file
parser.add_argument(
        "--config",
        type=str,
        default="config/FLT.yaml",
    )
# wandb run project name
parser.add_argument(
        "--project",
        type=str,
        default="FullyLoopedTransformer",
    )
# k loops
parser.add_argument(
        "-k",
        type=int
    )
# layer
parser.add_argument(
        "-l",
        type=int
    )
# batch_size
parser.add_argument(
        "--device_batch_size",
        type=int
    )
# loss_type
parser.add_argument(
        "--loss_type",
        type=str,
        default="END"
    )
# activation offload
parser.add_argument(
        "--activation_offload",
        action="store_true",
        default=False,
        help="Use gradient checkpointing on loop bodies to reduce activation memory usage"
    )
parser.add_argument(
        "--activation_offload_keep_last",
        type=int,
        default=1,
        help="Keep activations for the last N loop iterations without checkpointing (default: 1)"
    )
# attention variant
parser.add_argument(
        "--attn_type",
        type=str,
        default=None,
    )
parser.add_argument(
        "--kv_lora_rank",
        type=int,
        default=None,
    )
parser.add_argument(
        "--window_pattern",
        type=str,
        default=None,
    )
parser.add_argument(
        "--n_kv_head",
        type=int,
        default=None,
        help="Number of KV heads for GQA. Must divide n_head evenly. Default: same as n_head (GQA disabled)."
    )
# torch.compile
parser.add_argument(
        "--no_compile",
        action="store_true",
        default=False,
        help="Disable torch.compile (eager mode)"
    )
# num_iterations
parser.add_argument(
        "--num_iterations",
        type=int,
        default=-1,
    )


args = parser.parse_args()
config=Config(args.config)
config.loss_type = args.loss_type
config.project = args.project if args.project is not None else config.name
config.k = args.k if args.k is not None else config.k
config.num_hidden_layers = args.l if args.l is not None else config.num_hidden_layers
config.device_batch_size = args.device_batch_size if args.device_batch_size is not None else config.device_batch_size
config.activation_offload = args.activation_offload
config.activation_offload_keep_last = args.activation_offload_keep_last
config.num_iterations=args.num_iterations
config.attn_type = args.attn_type if args.attn_type is not None else getattr(config, 'attn_type', 'full')
config.kv_lora_rank = args.kv_lora_rank if args.kv_lora_rank is not None else getattr(config, 'kv_lora_rank', 128)
config.window_pattern = args.window_pattern if args.window_pattern is not None else getattr(config, 'window_pattern', 'L')
config.n_kv_head = args.n_kv_head  # None means "use n_head" (resolved later after depth is known)
use_compile = not args.no_compile

# run = "dummy" # wandb run name default ("dummy" is special - we won't log to wandb)
# Build attention suffix: attn type, GQA ratio (if not 1:1), window pattern (if not full)
_attn_suffix = f"_attn{config.attn_type}"
if config.attn_type == "mla":
    _attn_suffix += f"_r{config.kv_lora_rank}"
if config.n_kv_head is not None:
    _attn_suffix += f"_kv{config.n_kv_head}"
if config.window_pattern != "L":
    _attn_suffix += f"_win{config.window_pattern}"
if config.num_iterations>0:
    run = f"{config.model_type}_L{config.k}_D{config.num_hidden_layers}_{config.loss_type}{_attn_suffix}_{config.num_iterations}"
else:
    run = f"{config.model_type}_L{config.k}_D{config.num_hidden_layers}_{config.loss_type}{_attn_suffix}"

# Runtime
device_type = "" # cuda|cpu|mps (empty => autodetect good device type default, in order: CUDA > MPS > CPU)
# Model architecture
depth = config.num_hidden_layers # the depth of the Transformer model to train, rest of the kwargs are derived
max_seq_len = 1024 # max context length
# Training horizon. Only one of these 3 will be used, in this order of precedence.
num_iterations = config.num_iterations # explicit number of steps of the optimization (-1 = disable)
target_flops = -1.0 # calculate num_iterations to reach target_flops. Useful for scaling laws experiments (-1 = disable)
target_param_data_ratio = 20 # calculate num_iterations to maintain fixed data:param ratio (Chinchilla=20) (-1 = disable)
# Optimization
device_batch_size = config.device_batch_size # per-device batch size (set to not OOM)
total_batch_size = 524288 # total desired batch size, in #tokens
embedding_lr = 0.2 # learning rate for the embedding parameters (Adam)
unembedding_lr = 0.004 # learning rate for the unembedding parameters (Adam)
weight_decay = 0.0 # weight decay for the embedding/unembedding parameters (Adam)
matrix_lr = 0.02 # learning rate for the matrix parameters (Muon)
grad_clip = 1.0 # gradient clipping value (0.0 = disabled)
warmup_ratio = 0.0 # ratio of iterations for LR warmup
warmdown_ratio = 0.2 # ratio of iterations for LR warmdown
final_lr_frac = 0.0 # final LR is this fraction of the initial LR
resume_from_step = -1 # resume training from this step of the optimization (-1 = disable)
# Evaluation
eval_every = 250 # every how many steps to evaluate the model for val bpb
eval_tokens = 20*524288 # number of tokens to evaluate val loss on
core_metric_every = 2000 # every how many steps to evaluate the core metric (-1 = disable)
core_metric_max_per_task = 500 # examples per task in estimating the core metric
sample_every = 1 # every how many steps to sample from the model
save_every = -1 # every how many steps to save model checkpoints (-1 = disable, and save only at the end of the run)
# Output
model_tag = run # optionally override the model tag for the output checkpoint directory name
# now allow CLI to override the settings via the configurator lol
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# exec(open(os.path.join('nanochat', 'configurator.py')).read()) # overrides from command line or config file
user_config = {k: globals()[k] for k in config_keys} # will be useful for logging
# config.* attributes are not globals, so add them explicitly for wandb
user_config["attn_type"] = config.attn_type
user_config["kv_lora_rank"] = config.kv_lora_rank
user_config["window_pattern"] = config.window_pattern
user_config["n_kv_head_override"] = config.n_kv_head  # None if not set by user
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project=config.project, name=run, config=user_config)

# Tokenizer will be useful for evaluation, also we need the vocab size
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(tokenizer,device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# Model kwargs are derived from the desired depth of the model
num_layers = depth
model_dim = depth * 64 # aspect ratio 64 (usually this is varied from 64 -> 128 as model size increases)
num_heads = max(1, (model_dim + 127) // 128) # head dim 128 (the division here is ceil div)
if config.n_kv_head is not None:
    assert num_heads % config.n_kv_head == 0, \
        f"--n_kv_head={config.n_kv_head} must evenly divide n_head={num_heads}"
    num_kv_heads = config.n_kv_head
else:
    num_kv_heads = num_heads # default is 1:1 GQA (Group Query Attention) ratio (i.e. GQA is disabled)
print0(f"num_layers: {num_layers}")
print0(f"model_dim: {model_dim}")
print0(f"num_heads: {num_heads}")
print0(f"num_kv_heads: {num_kv_heads} (GQA ratio: {num_heads // num_kv_heads}:1)")

# Optimizer / data / training length related hyperparameters
# figure out the needed gradient accumulation to reach the desired total batch size
tokens_per_fwdbwd = device_batch_size * max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# -----------------------------------------------------------------------------
# Initialize the Model

# Create a new model with random weights
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
    # All tensors are created as meta tensors (they have shape/dtype but no data)
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)
model.to_empty(device=device) # All tensors get storage on target device but with uninitialized (garbage) data
model.init_weights() # All tensors get initialized

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
output_dirname = model_tag if model_tag else f"d{depth}" # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data # free up this memory after the copy

orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
if use_compile:
    model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe
    print0("torch.compile enabled")
else:
    print0("torch.compile disabled (eager mode)")
num_params = sum(p.numel() for p in model.parameters())
print0(f"Number of parameters: {num_params:,}")
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# Calculate number of iterations. Either it is given, or from target flops, or from target data:param ratio (in that order)
assert num_iterations > 0 or target_param_data_ratio > 0 or target_flops > 0
if num_iterations > 0:
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif target_flops > 0:
    # calculate the number of iterations from the target flops
    num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif target_param_data_ratio > 0:
    # calculate the number of iterations from the target param data ratio
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Params ratio: {total_batch_size * num_iterations / num_params:.2f}") # Chinchilla is ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (Muon for Linear layers, AdamW for embedding and lm_head)
optimizers = model.setup_optimizers(unembedding_lr=unembedding_lr, embedding_lr=embedding_lr, matrix_lr=matrix_lr, weight_decay=weight_decay)
adamw_optimizer, muon_optimizer = optimizers

if resuming:
    for opt, dat in zip(optimizers, optimizer_data):
        opt.load_state_dict(dat)
    del optimizer_data # free up the memory

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
tokens_dir = os.path.join(base_dir, "tokenized_data")
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state(device_batch_size, max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict)
build_val_loader = lambda: tokenizing_distributed_data_loader(device_batch_size, max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Set up hyperparameter schedulers

# Learning rate scheduler
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

# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# -----------------------------------------------------------------------------
# Loop state (variables updated by the training loop)

if not resuming:
    step = 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# -----------------------------------------------------------------------------
# Training loop
all_loss=[]
all_grad_norm=[]

while True:
    last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end
    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
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
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()
        if device_type == "cuda":
            torch.cuda.empty_cache()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    results = {}
    if core_metric_every > 0 and (last_step or (step > 0 and step % core_metric_every == 0)):
        model.eval()
        with autocast_ctx:
            results = evaluate_model(orig_model, tokenizer, device, max_per_task=core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()
        if device_type == "cuda":
            torch.cuda.empty_cache()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    # if master_process and (last_step or (step > 0 and step % sample_every == 0)):
    #     model.eval()
    #     prompts = [
    #         "The capital of France is",
    #         "The chemical symbol of gold is",
    #         "If yesterday was Friday, then tomorrow will be",
    #         "The opposite of hot is",
    #         "The planets of the solar system are:",
    #         "My favorite color is",
    #         "If 5*x + 3 = 13, then x is",
    #     ]
    #     engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
    #     for prompt in prompts:
    #         tokens = tokenizer(prompt, prepend="<|im_start|>")
    #         with autocast_ctx:
    #             sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
    #         print0(tokenizer.decode(sample[0]))
    #     model.train()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    if last_step or (step > 0 and step != resume_from_step and save_every > 0 and step % save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(), # model parameters
            [opt.state_dict() for opt in optimizers], # optimizer states
            { # metadata saved as json
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": model_config_kwargs,
                "user_config": user_config, # inputs to the training script
                "device_batch_size": device_batch_size,
                "max_seq_len": max_seq_len,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": { # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss_list = model(x, y)
            loss = torch.mean(torch.stack(loss_list))
        train_loss = loss_list[-1].detach() # last loop step loss, for logging
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        loss.backward()
        x, y, dataloader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
    # gradient clipping
    grad_clip_enabled = grad_clip > 0.0
    if grad_clip_enabled:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(orig_model.parameters(), grad_clip)
        grad_norm = grad_norm_tensor.item() # GPU tensor -> CPU float (note: cpu-gpu sync point)
    # step the optimizers
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
    # logging
    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item() # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size # bfloat16 H100 SXM and without 2:4 sparsity
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100 # in %
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    print_grad_norm = f" grad norm: {grad_norm:.4f} |" if grad_clip_enabled else ""
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} |{print_grad_norm} lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | total time: {total_training_time/60:.2f}m")
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
        }
        if grad_clip_enabled:
            log_data["train/grad_norm"] = grad_norm
        wandb_run.log(log_data)
    # loss std: log at end of warmup, start of warmdown, and final step
    all_loss.append(train_loss.item())
    if grad_clip_enabled:
        all_grad_norm.append(grad_norm)
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if step == warmup_iters or step == num_iterations - warmdown_iters or last_step:
        all_loss_arr = np.array(all_loss)
        log_data = {
            "step": step,
            "train/loss_std": all_loss_arr.std(),
        }
        if grad_clip_enabled:
            log_data["train/grad_norm_std"] = np.array(all_grad_norm).std()
        all_loss = []
        all_grad_norm = []
        wandb_run.log(log_data)
    log_data = {
        "step": step,
        "train/loss_unsmooth": train_loss.item(),
    }
    wandb_run.log(log_data)
    # state update
    step += 1



#    ██      ███████████                      ███             ██████████                █████   
#   ████    ░█░░░███░░░█                     ░░░             ░░███░░░░░█               ░░███    
#  ██░░██   ░   ░███  ░  ████████   ██████   ████  ████████   ░███  █ ░  ████████    ███████    
# ░░  ░░        ░███    ░░███░░███ ░░░░░███ ░░███ ░░███░░███  ░██████   ░░███░░███  ███░░███    
#               ░███     ░███ ░░░   ███████  ░███  ░███ ░███  ░███░░█    ░███ ░███ ░███ ░███    
#               ░███     ░███      ███░░███  ░███  ░███ ░███  ░███ ░   █ ░███ ░███ ░███ ░███    
#               █████    █████    ░░████████ █████ ████ █████ ██████████ ████ █████░░████████   
#              ░░░░░    ░░░░░      ░░░░░░░░ ░░░░░ ░░░░ ░░░░░ ░░░░░░░░░░ ░░░░ ░░░░░  ░░░░░░░░    
                                                                                              

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Base model training", data=[
    user_config, # CLI args
    { # stats about the training setup
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Params ratio": total_batch_size * num_iterations / num_params,
        "DDP world size": ddp_world_size,
        "warmup_ratio": warmup_ratio,
        "warmdown_ratio": warmdown_ratio,
        "final_lr_frac": final_lr_frac,
    },
    { # stats about training outcomes
        "Minimum validation bpb": min_val_bpb,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
