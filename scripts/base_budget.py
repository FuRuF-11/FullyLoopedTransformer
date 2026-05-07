"""
Evaluate the CORE metric for a given model.

Run on a single GPU:
python -m scripts.base_eval

Run with torchrun on e.g. 8 GPUs:
torchrun --nproc_per_node=8 -m scripts.base_eval

The script will print the CORE metric to the console.
"""
import io
import os
import csv
import time
import json
import yaml
import shutil
import random
import zipfile
import tempfile
import hashlib
from contextlib import nullcontext

import torch
import wandb
import argparse

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type, download_file_with_lock
from nanochat.tokenizer import HuggingFaceTokenizer, get_token_bytes
from nanochat.checkpoint_manager import load_model, build_model, find_last_step
from nanochat.core_eval import evaluate_task, evaluate_lambada_ppl
from nanochat.dataloader import tokenizing_distributed_data_loader, tokenizing_distributed_wiki_data_loader
from nanochat.loss_eval import evaluate_bpb, evaluate_ppl
from nanochat.engine import Engine
from nanochat.utils import Config


# -----------------------------------------------------------------------------
# File I/O with retry (guards against transient NFS errno-5 errors)

def _read_with_retry(path, max_retries=5, backoff=2.0):
    """Read an entire text file with exponential-backoff retry on OSError.

    Errno 5 (Input/output error) appears transiently on NFS-backed storage;
    retrying after a short wait is sufficient to recover in most cases.
    """
    for attempt in range(max_retries):
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                return fh.read()
        except OSError as exc:
            if attempt < max_retries - 1:
                wait = backoff * (2 ** attempt)
                print(f"[I/O retry {attempt + 1}/{max_retries}] {exc} — retrying '{path}' in {wait:.1f}s")
                time.sleep(wait)
            else:
                raise


# -----------------------------------------------------------------------------
# HuggingFace loading utilities

class ModelWrapper:
    """Lightweight wrapper to give HuggingFace models a nanochat-compatible interface."""
    def __init__(self, model, max_seq_len=None):
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids, targets=None, loss_reduction='mean'):
        logits = self.model(input_ids).logits
        if targets is None:
            return logits
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction=loss_reduction
        )
        return loss

    def get_device(self):
        return next(self.model.parameters()).device


def load_hf_model(hf_path: str, device):
    """Load a HuggingFace model and tokenizer."""
    print0(f"Loading HuggingFace model from: {hf_path}")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(hf_path)
    model.to(device)
    model.eval()
    max_seq_len = 1024 if "gpt2" in hf_path else None
    model = ModelWrapper(model, max_seq_len=max_seq_len)
    tokenizer = HuggingFaceTokenizer.from_pretrained(hf_path)
    return model, tokenizer


def get_hf_token_bytes(tokenizer, device="cpu"):
    """Compute token_bytes tensor for a HuggingFace tokenizer."""
    vocab_size = tokenizer.tokenizer.get_vocab_size()
    token_bytes = torch.zeros(vocab_size, dtype=torch.int64, device=device)
    for token_id in range(vocab_size):
        token_str = tokenizer.tokenizer.decode([token_id])
        token_bytes[token_id] = len(token_str.encode('utf-8'))
    return token_bytes

# -----------------------------------------------------------------------------
# CORE evaluation

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

def place_eval_bundle(file_path):
    """Unzip eval_bundle.zip and place it in the base directory."""
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print0(f"Placed eval_bundle directory at {eval_bundle_dir}")


def evaluate_model(model, tokenizer, device, max_per_task=-1):
    """
    Evaluate a base model on the CORE benchmark.
    Returns dict with results, centered_results, and core_metric.
    """
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    # Download the eval bundle if needed
    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)

    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")

    config = yaml.safe_load(_read_with_retry(config_path))
    tasks = config['icl_tasks']

    # Load random baseline values
    random_baselines = {}
    reader = csv.DictReader(io.StringIO(_read_with_retry(eval_meta_data)))
    for row in reader:
        task_name = row['Eval Task']
        random_baseline = row['Random baseline']
        random_baselines[task_name] = float(random_baseline)

    # Evaluate each task
    results = {}
    centered_results = {}
    for task in tasks:
        start_time = time.time()
        label = task['label']
        task_meta = {
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' ')
        }
        print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ", end='')

        data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
        data = [json.loads(line) for line in _read_with_retry(data_path).splitlines() if line.strip()]

        # Shuffle for consistent subsampling when using max_per_task
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)
        results[label] = accuracy
        random_baseline = random_baselines[label]
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result
        elapsed = time.time() - start_time
        print0(f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {elapsed:.2f}s")

    core_metric = sum(centered_results.values()) / len(centered_results)
    out = {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric
    }
    return out

def evaluate_model_with_lambada_ppl(model, tokenizer, device, max_per_task=-1):
    out = {}
    
    # 加载 LAMBADA 数据
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    
    config = yaml.safe_load(_read_with_retry(config_path))

    # 查找 LAMBADA 任务
    lambada_task = None
    for task in config['icl_tasks']:
        if 'lambada' in task['label'].lower():
            lambada_task = task
            break
    
    if lambada_task is None:
        print0("Warning: LAMBADA task not found in config")
        return out
    
    # 加载 LAMBADA 数据
    print0(f"Computing LAMBADA PPL... ", end='')
    start_time = time.time()
    
    data_path = os.path.join(data_base_path, lambada_task['dataset_uri'])
    data = [json.loads(line) for line in _read_with_retry(data_path).splitlines() if line.strip()]
    
    # 与原始评测保持一致的 shuffle
    shuffle_rng = random.Random(1337)
    shuffle_rng.shuffle(data)
    if max_per_task > 0:
        data = data[:max_per_task]
    
    # 计算 PPL
    ppl_results = evaluate_lambada_ppl(model, tokenizer, data, device)
    elapsed = time.time() - start_time
    print0(f"PPL: {ppl_results['perplexity']:.2f} | time: {elapsed:.2f}s")
    # 添加到输出结果
    out['lambada_openai_ppl'] = ppl_results['perplexity']
    out['lambada_openai_avg_loss'] = ppl_results['average_loss']
    
    return out


# -----------------------------------------------------------------------------
# Main

def main():
    parser = argparse.ArgumentParser(description="Base model evaluation")
    parser.add_argument('--eval', type=str, default='core,bpb,sample', help='Comma-separated evaluations to run: core,bpb,sample (default: all)')
    parser.add_argument('--hf-path', type=str, default=None, help='HuggingFace model path (e.g. openai-community/gpt2-xl)')
    parser.add_argument('--model-tag', type=str, default=None, help='nanochat model tag (subdirectory name under base_checkpoints/). Required unless --hf-path or --checkpoint-dir is given.')
    parser.add_argument('--checkpoint-dir', type=str, default=None, help='Absolute path to checkpoint directory, bypasses base_checkpoints/ convention entirely.')
    parser.add_argument('--step', type=int, default=None, help='Model step to load (default = last)')
    parser.add_argument('--max-per-task', type=int, default=-1, help='Max examples per CORE task (-1 = all)')
    parser.add_argument('--split-tokens', type=int, default=40*524288, help='Number of tokens to evaluate per split for BPB')
    parser.add_argument('--device-type', type=str, default='', help='cuda|cpu|mps (empty = autodetect)')
    
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
        )
    # num_iterations (used in run name when training was bounded by iteration count)
    parser.add_argument(
            "--num_iterations",
            type=int,
            default=-1,
        )

    # no-budget mode: evaluate once at the model's native K, run name = eval_{model_tag}
    parser.add_argument(
            "--no-budget",
            action="store_true",
            default=False,
            help="Skip the budget loop; evaluate at the model's native K once. "
                 "wandb run name becomes eval_{model_tag}.",
        )

    # wandb run project name
    parser.add_argument(
        "--project",
        type=str,
        default="FullyLoopedTransformer",
    )
    # wandb run project name
    parser.add_argument(
        "--config",
        type=str,
        default="config/FLT.yaml",
    )

    args = parser.parse_args()
    config=Config(args.config)
    config.project = args.project
    config.loss_type = args.loss_type
    config.k = args.k if args.k is not None else config.k
    config.num_hidden_layers = args.l if args.l is not None else config.num_hidden_layers
    config.device_batch_size = args.device_batch_size if args.device_batch_size is not None else config.device_batch_size
    config.attn_type = args.attn_type if args.attn_type is not None else getattr(config, 'attn_type', 'full')
    config.kv_lora_rank = args.kv_lora_rank if args.kv_lora_rank is not None else getattr(config, 'kv_lora_rank', 128)
    config.window_pattern = args.window_pattern if args.window_pattern is not None else getattr(config, 'window_pattern', 'L')
    config.n_kv_head = args.n_kv_head
    config.num_iterations = args.num_iterations

    eval_mode = "eval" if args.no_budget else "budget_eval"

    user_config={
        "eval_mode": eval_mode,
        "model_type": args.config,
        "split_tokens": args.split_tokens,
        "layers": args.l,
        "K": args.k,
        "loss_type": args.loss_type,
        "attn_type": config.attn_type,
        "kv_lora_rank": config.kv_lora_rank,
        "window_pattern": config.window_pattern,
        "n_kv_head_override": config.n_kv_head,
    }
    
    # Parse evaluation modes
    eval_modes = set(mode.strip() for mode in args.eval.split(','))
    valid_modes = {'core', 'bpb', 'sample'}
    invalid = eval_modes - valid_modes
    if invalid:
        parser.error(f"Invalid eval modes: {invalid}. Valid: {valid_modes}")
    
    # Distributed / precision setup
    device_type = autodetect_device_type() if args.device_type == '' else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    budgets = [1, 3, 6, 9, 12]

    # Load model and tokenizer once, outside the budget loop
    is_hf_model = args.hf_path is not None
    if is_hf_model:
        model, tokenizer = load_hf_model(args.hf_path, device)
        sequence_len = model.max_seq_len or 1024
        token_bytes = get_hf_token_bytes(tokenizer, device=device)
        model_tag = args.hf_path.replace("/", "-")
    else:
        if args.checkpoint_dir is not None:
            # Direct path mode: bypass base_checkpoints/ convention entirely
            checkpoint_dir = args.checkpoint_dir
            if not os.path.isdir(checkpoint_dir):
                raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
            step = args.step if args.step is not None else find_last_step(checkpoint_dir)
            model, tokenizer, meta = build_model(checkpoint_dir, step, device, phase="eval")
            model_tag = os.path.basename(checkpoint_dir.rstrip("/"))
        elif args.model_tag is not None:
            base_dir = get_base_dir()
            checkpoint_dir = os.path.join(base_dir, "base_checkpoints", args.model_tag)
            if not os.path.isdir(checkpoint_dir):
                raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
            model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
            model_tag = args.model_tag
        else:
            parser.error("One of --hf-path, --model-tag, or --checkpoint-dir is required.")
        sequence_len = meta["model_config"]["sequence_len"]
        token_bytes = get_token_bytes(tokenizer, device=device)

    # Stable 8-char run ID derived from model_tag so that re-running the same
    # checkpoint updates the existing wandb run instead of creating a duplicate.
    # Only rank 0 initialises wandb; other DDP ranks set wandb_run=None.
    run_name = f"{eval_mode}_{model_tag}"
    if ddp_rank == 0:
        _run_id = hashlib.md5(run_name.encode()).hexdigest()[:8]
        wandb_run = wandb.init(
            project=config.project,
            name=run_name,
            id=_run_id,
            resume="allow",
            config=user_config,
        )
    else:
        wandb_run = None

    def _run_evals(log_dict):
        """Run all requested evaluation modes; results merged into log_dict in-place."""
        # --- BPB evaluation ---
        if 'bpb' in eval_modes:
            print0("\n" + "="*80)
            print0("BPB Evaluation")
            print0("="*80)
            tokens_per_step = config.device_batch_size * sequence_len * ddp_world_size
            if args.split_tokens % tokens_per_step != 0:
                args.split_tokens = (args.split_tokens // tokens_per_step) * tokens_per_step
                print0(f"Adjusted split_tokens to {args.split_tokens} (must be divisible by {tokens_per_step})")
            steps = args.split_tokens // tokens_per_step

            for split_name in ["train", "val"]:
                loader = tokenizing_distributed_data_loader(config.device_batch_size, sequence_len, split_name, device=device)
                with autocast_ctx:
                    bpb = evaluate_bpb(model, loader, steps, token_bytes)
                print0(f"{split_name} bpb: {bpb:.6f}")
                log_dict[f"{split_name}/bpb"] = bpb

        # --- CORE evaluation ---
        if 'core' in eval_modes:
            print0("\n" + "="*80)
            print0("CORE Evaluation")
            print0("="*80)
            with autocast_ctx:
                core_results = evaluate_model(model, tokenizer, device, max_per_task=args.max_per_task)
            log_dict["core/metric"] = core_results["core_metric"]
            for label, acc in core_results["results"].items():
                centered = core_results["centered_results"][label]
                log_dict[f"core/{label}_acc"] = acc
                log_dict[f"core/{label}_centered"] = centered

        # --- Perplexity Evaluation (always run) ---
        print0("\n" + "="*80)
        print0("Perplexity Evaluation")
        print0("="*80)

        loader = tokenizing_distributed_wiki_data_loader(config.device_batch_size, sequence_len, "val", device=device)
        with autocast_ctx:
            ppl = evaluate_ppl(model, loader)
        print0(f"val ppl: {ppl:.6f}")
        log_dict["core/wiki_test2_ppl"] = ppl

        with autocast_ctx:
            ppl_out = evaluate_model_with_lambada_ppl(model, tokenizer, device, max_per_task=args.max_per_task)
        log_dict["core/lambada_openai_ppl"] = ppl_out["lambada_openai_ppl"]

    if args.no_budget:
        # ------------------------------------------------------------------
        # No-budget mode: evaluate once at the model's native K.
        # ------------------------------------------------------------------
        print0(f"Evaluating model: {model_tag}")
        print0(f"Eval modes: {', '.join(sorted(eval_modes))}")

        log_dict = {"model_tag": model_tag}
        _run_evals(log_dict)

        if ddp_rank == 0:
            # Single-point eval: write to summary so metrics appear in the run
            # table for cross-run comparison without generating spurious charts.
            wandb_run.summary.update(log_dict)
    else:
        # ------------------------------------------------------------------
        # Budget loop: sweep K ∈ [1, 3, 6, 9, 12] and log per-budget metrics.
        # ------------------------------------------------------------------
        all_results: dict[int, dict] = {}

        for budget in budgets:
            if not is_hf_model:
                model.K = budget

            print0(f"Evaluating model: {model_tag}")
            print0(f"Eval modes: {', '.join(sorted(eval_modes))}")

            log_dict = {"model_tag": model_tag}
            _run_evals(log_dict)

            if ddp_rank == 0:
                wandb_run.log(log_dict, step=budget)

            all_results[budget] = {
                k: v for k, v in log_dict.items()
                if isinstance(v, (int, float)) and v is not None
            }

        # ------------------------------------------------------------------
        # Bar charts — one chart per metric, K (budget) on the x-axis.
        # Only rank 0 builds and logs the charts.
        # ------------------------------------------------------------------
        if ddp_rank == 0:
            all_metric_keys: set[str] = set()
            for d in all_results.values():
                all_metric_keys.update(d.keys())

            bar_log: dict = {}
            for metric_key in sorted(all_metric_keys):
                rows = [
                    [f"K={b}", all_results[b][metric_key]]
                    for b in budgets
                    if metric_key in all_results[b]
                ]
                if rows:
                    table = wandb.Table(columns=["K", metric_key], data=rows)
                    safe_key = metric_key.replace("/", "_")
                    bar_log[f"bar/{safe_key}"] = wandb.plot.bar(
                        table, "K", metric_key, title=f"{metric_key} vs K"
                    )
            if bar_log:
                wandb_run.log(bar_log)

    if ddp_rank == 0:
        wandb_run.finish()
    
    compute_cleanup()


if __name__ == "__main__":
    main()

