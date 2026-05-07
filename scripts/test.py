"""
Train model. Run as:

python base_train.py

or distributed as:

torchrun --nproc_per_node=8 base_train.py

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max_seq_len=512 --device_batch_size=1 --eval_tokens=512 --core_metric_every=-1 --total_batch_size=512 --num_iterations=20
"""

# import os
# os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# import time
# from contextlib import nullcontext

# import wandb
# import torch

# from nanochat.gpt import GPT, GPTConfig
# from nanochat.dataloader import tokenizing_distributed_data_loader, tokenizing_distributed_data_loader_with_state
# from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type
# from nanochat.tokenizer import get_tokenizer, get_token_bytes
# from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
# from nanochat.loss_eval import evaluate_bpb
# from nanochat.engine import Engine
# from nanochat.utils import Config
# from scripts.base_eval import evaluate_model

# print_banner()

# # -----------------------------------------------------------------------------
# # User settings

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument(
#         "--config",
#         type=str,
#         default="config/loopedGPT.yaml",
#     )
# args = parser.parse_args()
# config=Config(args.config)

# # run = "dummy" # wandb run name default ("dummy" is special - we won't log to wandb)
# run = f"{config.model_type}_L{config.k}_D{config.num_hidden_layers}"
# # Runtime
# device_type = "" # cuda|cpu|mps (empty => autodetect good device type default, in order: CUDA > MPS > CPU)
# # Model architecture
# depth = config.num_hidden_layers # the depth of the Transformer model to train, rest of the kwargs are derived
# max_seq_len = 1024 # max context length
# # Training horizon. Only one of these 3 will be used, in this order of precedence.
# num_iterations = -1 # explicit number of steps of the optimization (-1 = disable)
# target_flops = -1.0 # calculate num_iterations to reach target_flops. Useful for scaling laws experiments (-1 = disable)
# target_param_data_ratio = 20 # calculate num_iterations to maintain fixed data:param ratio (Chinchilla=20) (-1 = disable)
# # Optimization
# device_batch_size = 16 # per-device batch size (set to not OOM)
# total_batch_size = 524288 # total desired batch size, in #tokens
# embedding_lr = 0.2 # learning rate for the embedding parameters (Adam)
# unembedding_lr = 0.004 # learning rate for the unembedding parameters (Adam)
# weight_decay = 0.0 # weight decay for the embedding/unembedding parameters (Adam)
# matrix_lr = 0.02 # learning rate for the matrix parameters (Muon)
# grad_clip = 1.0 # gradient clipping value (0.0 = disabled)
# warmup_ratio = 0.0 # ratio of iterations for LR warmup
# warmdown_ratio = 0.2 # ratio of iterations for LR warmdown
# final_lr_frac = 0.0 # final LR is this fraction of the initial LR
# resume_from_step = -1 # resume training from this step of the optimization (-1 = disable)
# # Evaluation
# eval_every = 250 # every how many steps to evaluate the model for val bpb
# eval_tokens = 20*524288 # number of tokens to evaluate val loss on
# core_metric_every = 2000 # every how many steps to evaluate the core metric (-1 = disable)
# core_metric_max_per_task = 500 # examples per task in estimating the core metric
# sample_every = 2000 # every how many steps to sample from the model
# save_every = -1 # every how many steps to save model checkpoints (-1 = disable, and save only at the end of the run)
# # Output
# model_tag = "" # optionally override the model tag for the output checkpoint directory name
# # now allow CLI to override the settings via the configurator lol
# config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# # exec(open(os.path.join('nanochat', 'configurator.py')).read()) # overrides from command line or config file
# user_config = {k: globals()[k] for k in config_keys} # will be useful for logging
# # -----------------------------------------------------------------------------

# # Compute init
# device_type = autodetect_device_type() if device_type == "" else device_type
# ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
# master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
# autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
# synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
# get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# # wandb logging init
# # use_dummy_wandb = run == "dummy" or not master_process
# # wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="LoopedGPT", name=run, config=user_config)

# # Tokenizer will be useful for evaluation, also we need the vocab size
# tokenizer = get_tokenizer()
# token_bytes = get_token_bytes(tokenizer,device=device)
# vocab_size = tokenizer.get_vocab_size()
# print0(f"Vocab size: {vocab_size:,}")

# # Model kwargs are derived from the desired depth of the model
# num_layers = depth
# model_dim = depth * 64 # aspect ratio 64 (usually this is varied from 64 -> 128 as model size increases)
# num_heads = max(1, (model_dim + 127) // 128) # head dim 128 (the division here is ceil div)
# num_kv_heads = num_heads # default is 1:1 GQA (Group Query Attention) ratio (i.e. GQA is disabled)
# print0(f"num_layers: {num_layers}")
# print0(f"model_dim: {model_dim}")
# print0(f"num_heads: {num_heads}")
# print0(f"num_kv_heads: {num_kv_heads}")

# # Optimizer / data / training length related hyperparameters
# # figure out the needed gradient accumulation to reach the desired total batch size
# tokens_per_fwdbwd = device_batch_size * max_seq_len # tokens per iteration for a single rank
# world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
# assert total_batch_size % world_tokens_per_fwdbwd == 0
# grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
# print0(f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
# print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
# print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# # -----------------------------------------------------------------------------
# # Initialize the Model

# # Create a new model with random weights
# model_config_kwargs = dict(
#     K=config.k,
#     sequence_len=max_seq_len, 
#     vocab_size=vocab_size, 
#     n_layer=num_layers, 
#     n_head=num_heads, 
#     n_kv_head=num_kv_heads, 
#     n_embd=model_dim
#     )
# with torch.device("meta"):
#     # All tensors are created as meta tensors (they have shape/dtype but no data)
#     model_config = GPTConfig(**model_config_kwargs)
#     model = GPT(model_config)
# model.to_empty(device=device) # All tensors get storage on target device but with uninitialized (garbage) data
# model.init_weights() # All tensors get initialized

# # If we are resuming, overwrite the model parameters with those of the checkpoint
# base_dir = get_base_dir()
# output_dirname = model_tag if model_tag else f"d{depth}" # e.g. d12
# checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
# resuming = resume_from_step != -1
# if resuming:
#     print0(f"Resuming optimization from step {resume_from_step}")
#     model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, resume_from_step, device, load_optimizer=True, rank=ddp_rank)
#     model.load_state_dict(model_data, strict=True, assign=True)
#     del model_data # free up this memory after the copy

# orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
# model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe
# num_params = sum(p.numel() for p in model.parameters())
# print0(f"Number of parameters: {num_params:,}")
# num_flops_per_token = model.estimate_flops()
# print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# # Calculate number of iterations. Either it is given, or from target flops, or from target data:param ratio (in that order)
# assert num_iterations > 0 or target_param_data_ratio > 0 or target_flops > 0
# if num_iterations > 0:
#     print0(f"Using user-provided number of iterations: {num_iterations:,}")
# elif target_flops > 0:
#     # calculate the number of iterations from the target flops
#     num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))
#     print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
# elif target_param_data_ratio > 0:
#     # calculate the number of iterations from the target param data ratio
#     target_tokens = target_param_data_ratio * num_params
#     num_iterations = target_tokens // total_batch_size
#     print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
# else:
#     raise ValueError("No training horizon specified")
# total_tokens = total_batch_size * num_iterations
# print0(f"Total number of training tokens: {total_tokens:,}")
# print0(f"Tokens : Params ratio: {total_batch_size * num_iterations / num_params:.2f}") # Chinchilla is ~20
# print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# # -----------------------------------------------------------------------------
# # Initialize the Optimizer (Muon for Linear layers, AdamW for embedding and lm_head)
# optimizers = model.setup_optimizers(unembedding_lr=unembedding_lr, embedding_lr=embedding_lr, matrix_lr=matrix_lr, weight_decay=weight_decay)
# adamw_optimizer, muon_optimizer = optimizers

# if resuming:
#     for opt, dat in zip(optimizers, optimizer_data):
#         opt.load_state_dict(dat)
#     del optimizer_data # free up the memory

# # -----------------------------------------------------------------------------
# # Initialize the DataLoaders for train/val
# tokens_dir = os.path.join(base_dir, "tokenized_data")
# dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
# train_loader = tokenizing_distributed_data_loader_with_state(device_batch_size, max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict)
# build_val_loader = lambda: tokenizing_distributed_data_loader(device_batch_size, max_seq_len, split="val", device=device)
# x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data

# # -----------------------------------------------------------------------------
# # Set up hyperparameter schedulers

# # Learning rate scheduler
# def get_lr_multiplier(it):
#     warmup_iters = round(warmup_ratio * num_iterations)
#     warmdown_iters = round(warmdown_ratio * num_iterations)
#     if it < warmup_iters:
#         return (it + 1) / warmup_iters
#     elif it <= num_iterations - warmdown_iters:
#         return 1.0
#     else:
#         progress = (num_iterations - it) / warmdown_iters
#         return progress * 1.0 + (1 - progress) * final_lr_frac

# # Momentum scheduler for Muon optimizer
# def get_muon_momentum(it):
#     frac = min(it / 300, 1)
#     momentum = (1 - frac) * 0.85 + frac * 0.95
#     return momentum

# # -----------------------------------------------------------------------------
# # Loop state (variables updated by the training loop)

# if not resuming:
#     step = 0
#     min_val_bpb = float("inf")
#     smooth_train_loss = 0 # EMA of training loss
#     total_training_time = 0 # total wall-clock time of training
# else:
#     step = meta_data["step"]
#     loop_state = meta_data["loop_state"]
#     val_bpb = meta_data["val_bpb"]
#     min_val_bpb = loop_state["min_val_bpb"]
#     smooth_train_loss = loop_state["smooth_train_loss"]
#     total_training_time = loop_state["total_training_time"]



# # once in a while: sample from the model (only on master process)
# # use the original uncompiled model because the inputs keep changing shape
# if master_process:
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




# #    ██      ███████████                      ███             ██████████                █████   
# #   ████    ░█░░░███░░░█                     ░░░             ░░███░░░░░█               ░░███    
# #  ██░░██   ░   ░███  ░  ████████   ██████   ████  ████████   ░███  █ ░  ████████    ███████    
# # ░░  ░░        ░███    ░░███░░███ ░░░░░███ ░░███ ░░███░░███  ░██████   ░░███░░███  ███░░███    
# #               ░███     ░███ ░░░   ███████  ░███  ░███ ░███  ░███░░█    ░███ ░███ ░███ ░███    
# #               ░███     ░███      ███░░███  ░███  ░███ ░███  ░███ ░   █ ░███ ░███ ░███ ░███    
# #               █████    █████    ░░████████ █████ ████ █████ ██████████ ████ █████░░████████   
# #              ░░░░░    ░░░░░      ░░░░░░░░ ░░░░░ ░░░░ ░░░░░ ░░░░░░░░░░ ░░░░ ░░░░░  ░░░░░░░░    
                                                                                              

# compute_cleanup()