"""
Evaluate the CORE metric for a given model.

Run on a single GPU:
python -m scripts.base_eval

Run with torchrun on e.g. 8 GPUs:
torchrun --nproc_per_node=8 -m scripts.base_eval

The script will print the CORE metric to the console.
"""
import os
import csv
import time
import json
import yaml
import shutil
import random
import zipfile
import tempfile
from contextlib import nullcontext

import torch
import wandb
import argparse
from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type, download_file_with_lock
from nanochat.tokenizer import HuggingFaceTokenizer, get_token_bytes
from nanochat.checkpoint_manager import load_model
from nanochat.core_eval import evaluate_task, evaluate_lambada_ppl
from nanochat.dataloader import tokenizing_distributed_data_loader, tokenizing_distributed_wiki_data_loader
from nanochat.loss_eval import evaluate_bpb, evaluate_ppl
from nanochat.engine import Engine
from nanochat.utils import Config


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

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    tasks = config['icl_tasks']

    # Load random baseline values
    random_baselines = {}
    with open(eval_meta_data, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
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
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]

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
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
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
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]
    
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
    parser.add_argument('--model-tag', type=str, default=None, help='nanochat model tag to identify the checkpoint directory')
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
    config=args.config
    user_config={
        "eval_mode": "base_eval",
        "model_type": config,
        "split_tokens": args.split_tokens,
        "layers": args.l,
        "K": args.k,
        "loss_type": args.loss_type
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

    
    config=Config(config)
    config.loss_type = args.loss_type
    config.k = args.k if args.k is not None else config.k
    config.num_hidden_layers = args.l if args.l is not None else config.num_hidden_layers
    config.device_batch_size = args.device_batch_size if args.device_batch_size is not None else config.device_batch_size
    
    if "rl" in config.model_type:
        model_tag=f"{config.model_type}_L{config.k}_D{config.num_hidden_layers}_{config.loss_type}_mdl_sen"
    else:
        model_tag=f"{config.model_type}_L{config.k}_D{config.num_hidden_layers}_{config.loss_type}"
    wandb_run = wandb.init(project=config.project, name="base_eval_"+model_tag, config=user_config)
    # Load model and tokenizer
    is_hf_model = args.hf_path is not None
    if is_hf_model:
        model, tokenizer = load_hf_model(args.hf_path, device)
        sequence_len = model.max_seq_len or 1024
        token_bytes = get_hf_token_bytes(tokenizer, device=device)
        model_name = args.hf_path
        model_slug = args.hf_path.replace("/", "-")
    else:
        model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
        sequence_len = meta["model_config"]["sequence_len"]
        token_bytes = get_token_bytes(tokenizer, device=device)
        model_name = f"{model_tag} (step {meta['step']})"
        model_slug = f"base_model_{meta['step']:06d}"

    print0(f"Evaluating model: {model_name}")
    print0(f"Eval modes: {', '.join(sorted(eval_modes))}")

    # Results to log
    core_results = None
    bpb_results = {}
    samples = []
    unconditioned_samples = []
    # --- BPB evaluation ---
    if 'bpb' in eval_modes:
        print0("\n" + "="*80)
        print0("BPB Evaluation")
        print0("="*80)
        tokens_per_step = args.device_batch_size * sequence_len * ddp_world_size
        if args.split_tokens % tokens_per_step != 0:
            # Adjust to nearest multiple
            args.split_tokens = (args.split_tokens // tokens_per_step) * tokens_per_step
            print0(f"Adjusted split_tokens to {args.split_tokens} (must be divisible by {tokens_per_step})")
        steps = args.split_tokens // tokens_per_step

        for split_name in ["train", "val"]:
            # loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, sequence_len, split_name, device=device)
            loader = tokenizing_distributed_data_loader(args.device_batch_size, sequence_len, split_name, device=device)
            with autocast_ctx:
                bpb = evaluate_bpb(model, loader, steps, token_bytes)
            bpb_results[split_name] = bpb
            print0(f"{split_name} bpb: {bpb:.6f}")
            if split_name == "train":
                wandb_run.log({
                    "model_tag": model_tag,
                    "train/bpb": bpb
                })
            else:
                wandb_run.log({
                    "model_tag": model_tag,
                    "val/bpb": bpb
                })

    # --- CORE evaluation ---
    if 'core' in eval_modes:
        print0("\n" + "="*80)
        print0("CORE Evaluation")
        print0("="*80)
        with autocast_ctx:
            core_results = evaluate_model(model, tokenizer, device, max_per_task=args.max_per_task)
        wandb.log({
            "model_tag": model_tag,
            "core/metric": core_results["core_metric"],
        })
        for label, acc in core_results["results"].items():
            centered = core_results["centered_results"][label]
            wandb.log({
                "model_tag": model_tag,
                f"core/{label}_acc": acc,
                f"core/{label}_centered": centered,
            })
    
    # --- Perplexity Evaluation ---
    # 2 PPL
    print0("\n" + "="*80)
    print0("Perplexity Evaluation")
    print0("="*80)
    
    # wikitext2 perplexity
    loader = tokenizing_distributed_wiki_data_loader(args.device_batch_size, sequence_len, split_name, device=device)
    with autocast_ctx:
        ppl = evaluate_ppl(model, loader)
    print0(f"{split_name} ppl: {bpb:.6f}")
    wandb_run.log({
        "model_tag": model_tag,
        "core/wiki_test2_ppl": ppl
    })
    
    # lambada_openai perplexity
    with autocast_ctx:
        ppl_out = evaluate_model_with_lambada_ppl(model, tokenizer, device, max_per_task=args.max_per_task)
    wandb_run.log({
        "model_tag": model_tag,
        "core/lambada_openai_ppl": ppl_out["lambada_openai_ppl"],
    })
    wandb_run.finish()
    compute_cleanup()


if __name__ == "__main__":
    main()


# # -----------------------------------------------------------------------------
# # nanochat specific function dealing with I/O etc.

# # ~162MB of data needed to evaluate the CORE metric
# EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

# def place_eval_bundle(file_path):
#     # here file_path is the path to the eval_bundle.zip file
#     # we need to unzip it and place it in the base directory
#     base_dir = get_base_dir()
#     eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
#     with tempfile.TemporaryDirectory() as tmpdir:
#         with zipfile.ZipFile(file_path, 'r') as zip_ref:
#             zip_ref.extractall(tmpdir)
#         extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
#         shutil.move(extracted_bundle_dir, eval_bundle_dir)
#     print0(f"Placed eval_bundle directory at {eval_bundle_dir}")

# def evaluate_model(model, tokenizer, device, max_per_task=-1):
#     """
#     Evaluate a base model on the CORE benchmark.
#     - max_per_task: crop the data to this many examples per task for testing (-1 = disable)
#     """
#     # Load config and task metadata
#     base_dir = get_base_dir()
#     eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
#     # Download the eval bundle to disk (and unzip if needed)
#     if not os.path.exists(eval_bundle_dir):
#         download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)
#     config_path = os.path.join(eval_bundle_dir, "core.yaml")
#     data_base_path = os.path.join(eval_bundle_dir, "eval_data")
#     eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")
#     with open(config_path, 'r', encoding='utf-8') as f:
#         config = yaml.safe_load(f)
#     tasks = config['icl_tasks']

#     # Load random baseline values from eval metadata
#     random_baselines = {}
#     with open(eval_meta_data, 'r', encoding='utf-8') as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             task_name = row['Eval Task']
#             random_baseline = row['Random baseline']
#             random_baselines[task_name] = float(random_baseline)

#     # Evaluate each task
#     results = {}
#     centered_results = {}
#     for task in tasks:
#         start_time = time.time()
#         label = task['label']
#         task_meta = {
#             'task_type': task['icl_task_type'],
#             'dataset_uri': task['dataset_uri'],
#             'num_fewshot': task['num_fewshot'][0],
#             'continuation_delimiter': task.get('continuation_delimiter', ' ')
#         }
#         print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ", end='')

#         # Load data for this task
#         data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
#         with open(data_path, 'r', encoding='utf-8') as f:
#             data = [json.loads(line.strip()) for line in f]

#         # shuffle the data because in many cases it appears ordered but we want
#         # the ability to only run a subset of the data for debugging purposes etc.
#         shuffle_rng = random.Random(1337)
#         shuffle_rng.shuffle(data)
#         if max_per_task > 0:
#             data = data[:max_per_task]

#         # run the evaluation for this task
#         accuracy = evaluate_task(model, tokenizer, data, device, task_meta)

#         results[label] = accuracy
#         random_baseline = random_baselines[label]
#         centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
#         centered_results[label] = centered_result
#         end_time = time.time()
#         print0(f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {end_time - start_time:.2f}s")

#     core_metric = sum(centered_results.values()) / len(centered_results)
#     out = {
#         "results": results,
#         "centered_results": centered_results,
#         "core_metric": core_metric
#     }
#     return out

# # -----------------------------------------------------------------------------
# # HuggingFace loading utilities and light wrappers for a model

# class ModelWrapper:
#     """Lightweight wrapper for a HuggingFace model"""
#     def __init__(self, model, max_seq_len=None):
#         self.model = model
#         self.max_seq_len = max_seq_len

#     def __call__(self, input_ids):
#         outputs = self.model(input_ids)
#         logits = outputs.logits
#         return logits

# def load_hf_model(hf_path: str, device):
#     print0(f"Loading model from: {hf_path}")
#     # Load the model
#     from transformers import AutoModelForCausalLM
#     model = AutoModelForCausalLM.from_pretrained(hf_path)
#     model.to(device)
#     model.eval()
#     max_seq_len = 1024 if "openai-community/gpt2" in hf_path else None
#     model = ModelWrapper(model, max_seq_len=max_seq_len)
#     # Load the tokenizer
#     tokenizer = HuggingFaceTokenizer.from_pretrained(hf_path)
#     return model, tokenizer

# # -----------------------------------------------------------------------------
# def main():
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--hf-path', type=str, default=None, help='HuggingFace model path to evaluate')
#     parser.add_argument('--max-per-task', type=int, default=-1, help='Max examples per task to evaluate (-1 = disable)')
#     parser.add_argument('--model-tag', type=str, default=None, help='optional model tag for the output directory name')
#     parser.add_argument('--step', type=str, default=None, help='optional model step for the output directory name')
    

#     # k loops
#     parser.add_argument(
#             "-k",
#             type=int
#         )
#     # layer
#     parser.add_argument(
#             "-l",
#             type=int
#         )
#     # batch_size
#     parser.add_argument(
#             "--device_batch_size",
#             type=int
#         )
#     # loss_type
#     parser.add_argument(
#             "--loss_type",
#             type=str,
#             default="NTP"
#         )
    
#     args = parser.parse_args()
#     config=Config(args.config)
#     config.loss_type = args.loss_type
#     config.k = args.k if args.k is not None else config.k
#     config.num_hidden_layers = args.l if args.l is not None else config.num_hidden_layers
#     config.device_batch_size = args.device_batch_size if args.device_batch_size is not None else config.device_batch_size
        
    
    
    
#     if "rl" in config.model_type:
#         model_tag=f"{config.model_type}_L{config.k}_D{config.num_hidden_layers}_{config.loss_type}_mdl_sen"
#     else:
#         model_tag=f"{config.model_type}_L{config.k}_D{config.num_hidden_layers}_{config.loss_type}"

#     # distributed / precision setup
#     device_type = autodetect_device_type()
#     ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
#     autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

#     # Load model and tokenizer from command line or from file system
#     if args.hf_path is not None:
#         # atm assume that if a path is given, it's a huggingface model path
#         hf_path = args.hf_path
#         print0(f"Loading huggingface model from: {hf_path}")
#         model, tokenizer = load_hf_model(hf_path, device)
#         model_name = hf_path # just for logging
#         model_slug = hf_path.replace("/", "-") # for the output csv file
#     else:
#         # load a local model from the file system
#         model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=model_tag, step=args.step)
#         model_name = f"base_model (step {meta['step']})" # just for logging
#         model_slug = f"base_model_{meta['step']:06d}" # for the output csv file

#     # Evaluate the model
#     with autocast_ctx:
#         out = evaluate_model(model, tokenizer, device, max_per_task=args.max_per_task)

#     # Write out the results to a csv file
#     core_metric = None
#     centered_results = {}
#     if ddp_rank == 0:
#         base_dir = get_base_dir()
#         output_csv_path = os.path.join(base_dir, "base_eval", f"{model_slug}.csv")
#         os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
#         results = out["results"]
#         centered_results = out["centered_results"]
#         core_metric = out["core_metric"]
#         with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
#             f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
#             for label in results:
#                 f.write(f"{label:<35}, {results[label]:<10.6f}, {centered_results[label]:<10.6f}\n")
#             f.write(f"{'CORE':<35}, {'':<10}, {core_metric:<10.6f}\n")
#         # Print the content of the csv file to console too
#         print0("="*80)
#         print0(f"Model: {model_name}")
#         print0("="*80)
#         with open(output_csv_path, 'r', encoding='utf-8') as f:
#             print0(f.read())

#     # Log to report
#     from nanochat.report import get_report
#     get_report().log(section="Base model evaluation", data=[
#         {
#             "Model": model_name,
#             "CORE metric": core_metric,
#         },
#         centered_results, # the full table
#     ])

#     compute_cleanup()

# if __name__ == "__main__":
#     main()


