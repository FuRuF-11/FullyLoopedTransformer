import os
import time
import logging
from collections import deque

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info, get_base_dir
from nanochat.dataset import list_parquet_files
from nanochat.tokenizer import get_tokenizer

logger = logging.getLogger(__name__)

def _retry(fn, max_retries=5, base_delay=2.0):
    """Call fn(), retrying on OSError with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return fn()
        except OSError as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(f"I/O error (attempt {attempt + 1}/{max_retries}), retrying in {delay:.0f}s: {e}")
            time.sleep(delay)

def tokenizing_distributed_data_loader_with_state(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128, device="cuda", resume_state_dict=None):
    """
    Stream pretraining text from parquet files, tokenize, yield training batches.

    This implementation became a bit more complex because we wish to support approximate resume training.
    Instead of turning this into a Class, we opt to return the state_dict with every batch,
    and then the caller can pass in a state_dict to resume training from a desired point.
    Note that this resumption is atm only *approximate* for simplicity.
    We won't repeat the same documents but we might skip a few.
    The state_dict that is returned can be later passed into this function via `resume_state_dict` to approximately resume.

    Perfect state resumption is possible but would be a lot more bloated, probably not worth it atm.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    # infinite iterator over document batches (list of text strings)
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    def document_batches():
        parquet_paths = list_parquet_files()
        assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
        parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
        resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
        resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
        first_pass = True
        pq_idx = resume_pq_idx # we kick off parquet files at the resume index (or by default just 0)
        while True: # iterate infinitely (multi-epoch)
            pq_idx = resume_pq_idx if first_pass else 0
            while pq_idx < len(parquet_paths): # iterate over all parquet files
                filepath = parquet_paths[pq_idx]
                pf = _retry(lambda: pq.ParquetFile(filepath))
                # Start from resume point if resuming on same file, otherwise from DDP rank
                # I know this state resumption is a little bit tricky and a little bit hacky... sigh.
                if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                    base_idx = resume_rg_idx // ddp_world_size # in units of ddp_world_size
                    base_idx += 1 # advance by 1 so that we definitely don't repeat data after resuming
                    rg_idx = base_idx * ddp_world_size + ddp_rank
                    if rg_idx >= pf.num_row_groups:
                        pq_idx += 1
                        continue
                    resume_rg_idx = None # set to None as we only want to do this a single time
                else:
                    rg_idx = ddp_rank
                while rg_idx < pf.num_row_groups:
                    rg = _retry(lambda: pf.read_row_group(rg_idx))
                    batch = rg.column('text').to_pylist() # each batch is a parquet group, e.g. 1024 rows
                    # the tokenizer encode might want to go in even smaller batches, e.g. 128 rows
                    for i in range(0, len(batch), tokenizer_batch_size):
                        yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx)
                    rg_idx += ddp_world_size # advance to the next row group (in DDP)
                pq_idx += 1 # advance to the next parquet file
            first_pass = False
    batches = document_batches()

    # Now emit batches of tokens.
    needed_tokens = B * T + 1 # +1 is because we also need the target at the last token
    # get the tokenizer and the bos token
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    # scratch buffer holds the tokens for one iteration
    token_buffer = deque() # we stream tokens on the right and pop from the left
    while True:
        # Accumulate enough tokens for one iteration before yielding.
        while len(token_buffer) < needed_tokens:
            doc_batch, (pq_idx, rg_idx) = next(batches)
            token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
            for tokens in token_lists:
                token_buffer.extend(tokens)
        # Move tokens from the deque into the scratch buffer
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        # CUDA supports memory pinning for asynchronous transfers between CPU and GPU
        use_cuda_optimizations = device == "cuda"
        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda_optimizations) # in PyTorch, long=int64
        # Create the inputs/targets as 1D tensors
        inputs_cpu = scratch[:-1]
        targets_cpu = scratch[1:]
        # Reshape to 2D and move to GPU async
        inputs = inputs_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        targets = targets_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx} # we need this in case we wish to approximately resume training
        yield inputs, targets, state_dict

def tokenizing_distributed_data_loader(*args, **kwargs):
    # helper function that only emits the inputs/targets and not the state_dict
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state(*args, **kwargs):
        yield inputs, targets


def load_and_tokenize_all(parquet_paths, tokenizer, bos_token, tokenizer_batch_size=128):
    all_tokens = []    
    for filepath in parquet_paths:
        pf = _retry(lambda: pq.ParquetFile(filepath))
        for rg_idx in range(pf.num_row_groups):
            rg = _retry(lambda: pf.read_row_group(rg_idx))
            docs = rg.column('text').to_pylist()
            for i in range(0, len(docs), tokenizer_batch_size):
                doc_batch = docs[i:i + tokenizer_batch_size]
                token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
                for tokens in token_lists:
                    all_tokens.extend(tokens)
    
    return all_tokens

def tokenizing_distributed_wiki_data_loader_with_state(B, T, split="test", tokenizer_batch_size=128, device="cuda", resume_state_dict=None):
    """
    WikiText-2 test data load
    """
    base_dir = get_base_dir()
    WIKI_DATA_DIR = os.path.join(base_dir, "wiki_data")
    parquet_paths = list_parquet_files(WIKI_DATA_DIR)
    assert len(parquet_paths) != 0, "No dataset parquet files found"
    
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    use_cuda = device == "cuda"
    
    # 一次性加载（小数据集，无内存压力）
    all_tokens = load_and_tokenize_all(
        parquet_paths, tokenizer, bos_token, tokenizer_batch_size
    )
    
    total_batches = (len(all_tokens) - 1) // (B * T)
    
    if total_batches == 0:
        return
    
    tokens_tensor = torch.tensor(all_tokens, dtype=torch.long)
    if use_cuda:
        tokens_tensor = tokens_tensor.pin_memory()
    
    # DDP: 交错分配
    for batch_idx in range(ddp_rank, total_batches, ddp_world_size):
        start = batch_idx * B * T
        chunk = tokens_tensor[start : start + B * T + 1]
        
        inputs = chunk[:-1].view(B, T).to(device=device, non_blocking=use_cuda)
        targets = chunk[1:].view(B, T).to(device=device, non_blocking=use_cuda)
        
        yield inputs, targets


def tokenizing_distributed_wiki_data_loader(*args, **kwargs):
    # helper function that only emits the inputs/targets and not the state_dict
    for inputs, targets in tokenizing_distributed_wiki_data_loader_with_state(*args, **kwargs):
        yield inputs, targets


#  reasoning data loader
def tokenizing_distributed_reasoning_data_loader_with_state(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128, device="cuda", resume_state_dict=None):
    """
    Stream pretraining text from parquet files, tokenize, yield training batches.

    This implementation became a bit more complex because we wish to support approximate resume training.
    Instead of turning this into a Class, we opt to return the state_dict with every batch,
    and then the caller can pass in a state_dict to resume training from a desired point.
    Note that this resumption is atm only *approximate* for simplicity.
    We won't repeat the same documents but we might skip a few.
    The state_dict that is returned can be later passed into this function via `resume_state_dict` to approximately resume.

    Perfect state resumption is possible but would be a lot more bloated, probably not worth it atm.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    def document_batches(batch_size: int):
        # 1. 收集所有 parquet 文件路径（每个文件 = 一条数据）
        if split == "train":
            parquet_paths = list_parquet_files("data/arc-agi/train")
        elif split == "val":
            parquet_paths = list_parquet_files("data/arc-agi/val")
        
        assert len(parquet_paths) != 0, "No dataset parquet files found"
        
        # 断点恢复：只需要文件索引
        resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
        first_pass = True
        while True:  # 无限多 epoch
            # --------- 决定本轮从哪个 pq_idx 开始 ----------
            if first_pass:
                start_pq_idx = resume_pq_idx
                first_pass = False
            else:
                start_pq_idx = 0

            # --------- DDP：交错分配文件索引 ----------
            if ddp:
                import math
                k = max(0, math.ceil((start_pq_idx - ddp_rank) / ddp_world_size))
                pq_idx = ddp_rank + k * ddp_world_size
            else:
                pq_idx = start_pq_idx

            # --------- 主循环：按 batch_size 聚合 sample ----------
            while pq_idx < len(parquet_paths):
                doc_batch = []
                first_pq_idx_in_batch = pq_idx  # 用于 state_dict

                # 收集最多 batch_size 个 sample
                while pq_idx < len(parquet_paths) and len(doc_batch) < batch_size:
                    filepath = parquet_paths[pq_idx]

                    pf = _retry(lambda: pq.ParquetFile(filepath))
                    table = _retry(lambda: pf.read())
                    data = table.to_pydict()

                    # === 按你的 schema 构造 sample ===
                    if "text" in data and len(data["text"]) == 1:
                        sample = data["text"][0]
                    else:
                        sample = data     # 自己按需要改

                    doc_batch.append(sample)

                    # 下一个属于本 rank 的文件
                    if ddp:
                        pq_idx += ddp_world_size
                    else:
                        pq_idx += 1

                # 如果这一轮没收集到任何 sample（理论上不会），就 break
                if len(doc_batch) == 0:
                    break

                # 返回一个 batch：List[sample]，以及 state（只记第一个 pq_idx）
                yield doc_batch, (first_pq_idx_in_batch, 0)

    batches = document_batches()

    # # Now emit batches of tokens.
    # needed_tokens = B * T + 1 # +1 is because we also need the target at the last token
    # # get the tokenizer and the bos token
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()

    while True:
        # 从 document_batches 里拿一个 batch（每个元素是一个完整 sample）
        doc_batch, (pq_idx, rg_idx) = next(batches)   # doc_batch: List[str], len = B

        token_lists = []
        for text in doc_batch:
            # encode 成 token 序列，可以加 BOS/EOS
            tokens = tokenizer.encode(text, prepend=bos_token)

            # 截断到 T+1（因为后面要 inputs[:-1], targets[1:]）
            if len(tokens) > T + 1:
                tokens = tokens[:T + 1]

            token_lists.append(tokens)

        # 现在 token_lists 里有 B 个序列，每个长度 <= T+1
        # 按“每条序列内部 pad 到 T+1”的方式构造 batch
        B_actual = len(token_lists)
        max_len = max(len(t) for t in token_lists)
        seq_len = min(max_len, T + 1)  # 不超过 T+1

        batch_tokens = torch.full(
            (B_actual, seq_len),
            bos_token,
            dtype=torch.long,
            pin_memory=(device == "cuda"),
        )

        for i, tokens in enumerate(token_lists):
            l = min(len(tokens), seq_len)
            batch_tokens[i, :l] = torch.tensor(tokens[:l], dtype=torch.long)

        # 像原来一样构造 inputs / targets
        inputs_cpu  = batch_tokens[:, :-1]   # (B, T)
        targets_cpu = batch_tokens[:, 1:]    # (B, T)

        inputs  = inputs_cpu.to(device, non_blocking=True)
        targets = targets_cpu.to(device, non_blocking=True)

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx}
        yield inputs, targets, state_dict


def tokenizing_distributed_reasoning_data_loader(*args, **kwargs):
    # helper function that only emits the inputs/targets and not the state_dict
    for inputs, targets, state_dict in tokenizing_distributed_reasoning_data_loader_with_state(*args, **kwargs):
        yield inputs, targets


