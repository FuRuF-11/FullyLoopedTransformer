"""
Quick WikiText-2 PPL evaluation for a single checkpoint.

Single GPU:
    python -m scripts.wiki_ppl --checkpoint-dir /path/to/ckpt

Multi-GPU:
    torchrun --standalone --nproc_per_node=4 -m scripts.wiki_ppl --checkpoint-dir /path/to/ckpt

Options:
    --checkpoint-dir PATH   Absolute path to checkpoint directory
    --model-tag TAG         Subdirectory under ~/.cache/nanochat/base_checkpoints/
    --hf-path PATH          HuggingFace model path
    --step INT              Specific training step to load (default: last)
    -k INT                  Override loop count K at inference
    --device-batch-size INT Batch size per device (default: 4)
    --split val|test        WikiText-2 split to evaluate (default: val)
    --check-norms           Report per-loop hidden-state L2 norms on WikiText-2
    --norm-batches INT      Number of batches used for norm/logit checking (default: 8)
    --check-logits          Report output logit entropy and top-1 probability on WikiText-2
"""
import argparse
import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import torch.distributed as dist

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type
from nanochat.tokenizer import get_token_bytes
from nanochat.checkpoint_manager import load_model, build_model, find_last_step
from nanochat.dataloader import tokenizing_distributed_wiki_data_loader
from nanochat.loss_eval import evaluate_ppl


@torch.no_grad()
def evaluate_hidden_norms(model, loader, num_batches, autocast_ctx, ddp_world_size, device):
    """
    Run up to `num_batches` wiki batches with collect_norms=True and report
    per-loop hidden-state L2 norms averaged across batches and DDP ranks.
    """
    norm_sums = None  # list[Tensor], one scalar tensor per loop
    batch_count = torch.tensor(0, dtype=torch.long, device=device)

    for i, (x, y) in enumerate(loader):
        if i >= num_batches:
            break
        with autocast_ctx:
            result = model(x, targets=y, loss_reduction='mean', collect_norms=True)
        _, hidden_norms = result  # list of K detached scalar tensors

        if norm_sums is None:
            norm_sums = [torch.zeros(1, device=device) for _ in hidden_norms]
        for k, hn in enumerate(hidden_norms):
            norm_sums[k] += hn.detach().to(device)
        batch_count += 1

    if norm_sums is None:
        print0("  [check-norms] No batches available.")
        return

    if ddp_world_size > 1:
        dist.all_reduce(batch_count, op=dist.ReduceOp.SUM)
        for k in range(len(norm_sums)):
            dist.all_reduce(norm_sums[k], op=dist.ReduceOp.SUM)

    total = batch_count.item()
    print0(f"\n{'='*52}")
    print0(f"Hidden-state L2 norms on WikiText-2")
    print0(f"({total} batches x {ddp_world_size} ranks, K={len(norm_sums)})")
    print0(f"{'Loop':>6}  {'Avg norm':>12}")
    print0(f"{'-'*22}")
    for k, ns in enumerate(norm_sums):
        avg = ns.item() / total
        print0(f"{k:>6d}  {avg:>12.4f}")
    print0(f"{'='*52}")


@torch.no_grad()
def evaluate_logit_stats(model, loader, num_batches, autocast_ctx, ddp_world_size, device,
                         vocab_size):
    """
    Compute average per-token output entropy and top-1 probability on WikiText-2.

    - High entropy (close to log(vocab_size)) → near-uniform output → poor calibration
    - Low top-1 probability → model is not confident in any single token
    Both together mean the model cannot assign meaningful per-token probabilities,
    which directly causes high PPL even if relative rankings (CORE) are correct.
    """
    entropy_sum   = torch.tensor(0.0, device=device)
    top1_prob_sum = torch.tensor(0.0, device=device)
    batch_count   = torch.tensor(0,   dtype=torch.long, device=device)

    for i, (x, _) in enumerate(loader):
        if i >= num_batches:
            break
        with autocast_ctx:
            logits = model(x)           # inference mode: no targets → returns logits
        if isinstance(logits, list):    # shouldn't happen, but guard anyway
            logits = logits[-1]
        logits = logits.float()         # (B, T, vocab_size)

        probs = F.softmax(logits, dim=-1)                        # (B, T, vocab_size)
        entropy   = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()  # scalar
        top1_prob = probs.max(dim=-1).values.mean()              # scalar

        entropy_sum   += entropy
        top1_prob_sum += top1_prob
        batch_count   += 1

    if batch_count.item() == 0:
        print0("  [check-logits] No batches available.")
        return

    if ddp_world_size > 1:
        dist.all_reduce(entropy_sum,   op=dist.ReduceOp.SUM)
        dist.all_reduce(top1_prob_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(batch_count,   op=dist.ReduceOp.SUM)

    total        = batch_count.item()
    avg_entropy  = entropy_sum.item()   / total
    avg_top1     = top1_prob_sum.item() / total
    rand_entropy = math.log(vocab_size)

    print0(f"\n{'='*52}")
    print0(f"Output logit statistics on WikiText-2")
    print0(f"({total} batches x {ddp_world_size} ranks)")
    print0(f"  Avg per-token entropy : {avg_entropy:.4f} nats")
    print0(f"  Random baseline       : {rand_entropy:.4f} nats")
    print0(f"  Entropy ratio         : {avg_entropy / rand_entropy:.4f}  (1.0 = fully random)")
    print0(f"  Avg top-1 probability : {avg_top1:.6f}")
    print0(f"  PPL implied by entropy: {math.exp(avg_entropy):.2f}")
    print0(f"{'='*52}")


def main():
    parser = argparse.ArgumentParser(description="Quick WikiText-2 PPL evaluation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint-dir", type=str, help="Absolute path to checkpoint directory")
    group.add_argument("--model-tag", type=str, help="Subdirectory under base_checkpoints/")
    group.add_argument("--hf-path", type=str, help="HuggingFace model path")
    parser.add_argument("--step", type=int, default=None, help="Training step to load (default: last)")
    parser.add_argument("-k", type=int, default=None, help="Override loop count K at inference")
    parser.add_argument("--device-batch-size", type=int, default=4)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"],
                        help="WikiText-2 split (default: val)")
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty=autodetect)")
    parser.add_argument("--check-norms", action="store_true",
                        help="Report per-loop hidden-state L2 norms on WikiText-2")
    parser.add_argument("--check-logits", action="store_true",
                        help="Report output entropy and top-1 probability on WikiText-2")
    parser.add_argument("--norm-batches", type=int, default=8,
                        help="Number of batches used for norm/logit checking (default: 8)")
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        if device_type == "cuda"
        else nullcontext()
    )

    # --- Load model ---
    is_hf_model = args.hf_path is not None
    if is_hf_model:
        from transformers import AutoModelForCausalLM
        from nanochat.tokenizer import HuggingFaceTokenizer
        from scripts.base_budget import ModelWrapper, get_hf_token_bytes
        print0(f"Loading HuggingFace model: {args.hf_path}")
        _hf = AutoModelForCausalLM.from_pretrained(args.hf_path).to(device).eval()
        model = ModelWrapper(_hf, max_seq_len=1024)
        sequence_len = model.max_seq_len
        model_tag = args.hf_path.replace("/", "-")
        vocab_size = model.config.vocab_size if hasattr(model, 'config') else 50257
    elif args.checkpoint_dir is not None:
        import os
        if not os.path.isdir(args.checkpoint_dir):
            raise FileNotFoundError(f"Not found: {args.checkpoint_dir}")
        step = args.step if args.step is not None else find_last_step(args.checkpoint_dir)
        model, tokenizer, meta = build_model(args.checkpoint_dir, step, device, phase="eval")
        sequence_len = meta["model_config"]["sequence_len"]
        model_tag = os.path.basename(args.checkpoint_dir.rstrip("/"))
        vocab_size = meta["model_config"]["vocab_size"]
    else:  # model_tag
        import os
        base_dir = get_base_dir()
        checkpoint_dir = os.path.join(base_dir, "base_checkpoints", args.model_tag)
        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(f"Not found: {checkpoint_dir}")
        model, tokenizer, meta = load_model("base", device, phase="eval",
                                            model_tag=args.model_tag, step=args.step)
        sequence_len = meta["model_config"]["sequence_len"]
        model_tag = args.model_tag
        vocab_size = meta["model_config"]["vocab_size"]

    # Override K if requested
    if args.k is not None and not is_hf_model:
        model.K = args.k
        print0(f"K overridden to {args.k}")

    k_str = str(model.K) if not is_hf_model else "N/A"
    print0(f"Model: {model_tag}  |  seq_len={sequence_len}  |  batch={args.device_batch_size}  |  K={k_str}")

    # --- PPL Evaluation ---
    loader = tokenizing_distributed_wiki_data_loader(
        args.device_batch_size, sequence_len, args.split, device=device
    )
    with autocast_ctx:
        ppl = evaluate_ppl(model, loader)

    print0(f"\nWikiText-2 {args.split} PPL = {ppl:.4f}")

    # --- Hidden-norm check (native GPT models only) ---
    if args.check_norms:
        if is_hf_model:
            print0("\n[check-norms] Skipped: not supported for HuggingFace models.")
        else:
            norm_loader = tokenizing_distributed_wiki_data_loader(
                args.device_batch_size, sequence_len, args.split, device=device
            )
            evaluate_hidden_norms(
                model, norm_loader, args.norm_batches,
                autocast_ctx, ddp_world_size, device,
            )

    # --- Logit statistics check ---
    if args.check_logits:
        logit_loader = tokenizing_distributed_wiki_data_loader(
            args.device_batch_size, sequence_len, args.split, device=device
        )
        evaluate_logit_stats(
            model, logit_loader, args.norm_batches,
            autocast_ctx, ddp_world_size, device, vocab_size,
        )

    compute_cleanup()


if __name__ == "__main__":
    main()
