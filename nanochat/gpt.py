"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
"""

import math
from contextlib import nullcontext
from functools import partial
from torch.utils.checkpoint import checkpoint
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW

@dataclass
class GPTConfig:
    model_type: str="FLT"
    dropout: float=0.01
    K: int=1
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    activation_offload: bool = False  # use gradient checkpointing to reduce activation memory during training
    activation_offload_keep_last: int = 1  # keep activations for the last N loop iterations (no recomputation)
    loss_type: str = "END"  # which loop steps contribute to loss: STEP (all), FEW (last 1/3), END (last only)
    attn_type: str = "full"  # attention variant: "full" (standard GQA/MHA), "mla" (Multi-head Latent Attention)
    kv_lora_rank: int = 128  # MLA: rank of the KV compression bottleneck
    window_pattern: str = "L"  # SWA: per-layer pattern tiled across layers. L=full context, S=quarter context. e.g. "SSSL"


def _cpu_offload_ctx():
    """Context manager: offloads autograd-saved tensors to CPU pinned memory during
    forward, fetches them back to the originating GPU on demand during backward.

    pack returns (pinned_cpu_tensor, src_device) so unpack always targets the correct
    GPU in multi-device (DDP) setups.  The GPU→CPU copy is a true async DMA because
    the destination is pre-allocated pinned memory; CUDA stream ordering guarantees the
    CPU→GPU copy in unpack completes before any downstream op on the same stream.
    """
    def pack(t):
        src_device = t.device
        if not t.is_cuda:
            # Non-CUDA tensor (e.g. CPU debug run): just pass through
            return (t, src_device)
        # Pre-allocate pinned CPU memory, then kick off an async DMA (GPU → CPU pinned).
        # This is genuinely non-blocking: the DMA engine transfers data while the GPU
        # continues with the next kernel.  t.to("cpu") without a pre-pinned target would
        # fall back to a synchronous copy into pageable memory.
        # cpu_pinned = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
        # cpu_pinned.copy_(t, non_blocking=True)
        # return (cpu_pinned, src_device)
        return (t.cpu(), src_device)

    def unpack(packed):
        cpu_tensor, src_device = packed
        # Non-blocking DMA back to the originating GPU (pinned → GPU is always async).
        # CUDA stream ordering ensures the result is ready before the next op on that stream.
        return cpu_tensor.to(device=src_device, non_blocking=True)

    return torch.autograd.graph.saved_tensors_hooks(pack, unpack)


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def _lm_loss(h, lm_head, vocab_size, softcap, targets, loss_reduction):
    """Compute cross-entropy loss for one loop step.

    Intended to be called via torch.utils.checkpoint so that the large
    (B, T, vocab_size) logit tensor is NOT saved during forward.
    Only `h` (B, T, n_embd) is kept as a checkpoint input; logits are
    recomputed on demand during backward.  For STEP/FEW with K steps this
    avoids keeping K × (B × T × vocab_size) fp32 tensors alive simultaneously.
    """
    logits = lm_head(h)
    logits = logits[..., :vocab_size].float()
    logits = softcap * torch.tanh(logits / softcap)
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=-1,
        reduction=loss_reduction,
    )


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, q, k, v, cos_sin, kv_cache, if_cache, window_size=(-1, 0)):
        B, T, C = q.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(q).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(k).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(v).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # QK rotary embedding
        q, k = norm(q), norm(k) # QK norm
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)

        # Apply KV cache: insert current k,v into cache, get the full view so far
        if kv_cache is not None and if_cache == True:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2) # number of queries in this forward pass
        Tk = k.size(2) # number of keys/values in total (in the cache + current forward pass)

        # Build attention mask. window_left == -1 means full context (no sliding window).
        window_left = window_size[0]
        use_swa = window_left != -1
        enable_gqa = self.n_head != self.n_kv_head

        if not use_swa and (kv_cache is None or Tq == Tk):
            # Full context, training or simple cache case: use fast causal path
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        elif not use_swa and Tq == 1:
            # Full context, single-token decode
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            # Sliding window or chunked inference: build explicit mask
            # i = query position (0-indexed within this chunk), j = key position (global)
            prefix_len = Tk - Tq  # number of tokens already in cache before this chunk
            q_pos = torch.arange(Tq, device=q.device) + prefix_len  # absolute query positions
            k_pos = torch.arange(Tk, device=q.device)               # absolute key positions
            # Causal: key must not be in the future
            causal_mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)  # (Tq, Tk)
            # Sliding window: key must be within window_left tokens of the query
            if use_swa:
                window_mask = (q_pos.unsqueeze(1) - k_pos.unsqueeze(0)) < window_left  # (Tq, Tk)
                attn_mask = causal_mask & window_mask
            else:
                attn_mask = causal_mask
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLACausalSelfAttention(nn.Module):
    """Multi-head Latent Attention (MLA) from DeepSeek-V2.
    KV is compressed to a low-rank latent c_kv before being expanded back to K and V.
    This reduces KV cache size during inference by factor (n_kv_head * head_dim) / kv_lora_rank.
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.kv_lora_rank = config.kv_lora_rank
        assert config.n_embd % config.n_head == 0
        assert config.n_kv_head <= config.n_head and config.n_head % config.n_kv_head == 0
        # Q: standard projection (no compression)
        self.c_q = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        # KV: compress x → c_kv, then expand to K and V separately
        self.c_kv_down = nn.Linear(config.n_embd, config.kv_lora_rank, bias=False)
        self.c_k_up   = nn.Linear(config.kv_lora_rank, self.n_kv_head * self.head_dim, bias=False)
        self.c_v_up   = nn.Linear(config.kv_lora_rank, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_head * self.head_dim, config.n_embd, bias=False)

    def forward(self, q, k, v, cos_sin, kv_cache, if_cache, window_size=(-1, 0)):
        B, T, C = q.size()
        q = self.c_q(q).view(B, T, self.n_head, self.head_dim)
        # KV compression: both k and v share the same down-projection input
        c_kv = norm(self.c_kv_down(k))  # (B, T, kv_lora_rank)  latent norm stabilises down/up gradient coupling
        k = self.c_k_up(c_kv).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v_up(c_kv).view(B, T, self.n_kv_head, self.head_dim)
        # RoPE + QK norm
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # KV cache (stores expanded K, V — same interface as CausalSelfAttention)
        if kv_cache is not None and if_cache:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq, Tk = q.size(2), k.size(2)
        # Attention (identical logic to CausalSelfAttention)
        enable_gqa = self.n_head != self.n_kv_head
        if kv_cache is None or Tq == Tk:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        elif Tq == 1:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)
            prefix_len = Tk - Tq
            attn_mask[:, :prefix_len] = True
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


def _build_attn(config, layer_idx):
    if config.attn_type == "mla":
        return MLACausalSelfAttention(config, layer_idx)
    return CausalSelfAttention(config, layer_idx)


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = _build_attn(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, h=None, cos_sin=None, kv_cache=None, if_cache=True, window_size=(-1, 0)):
        if h is None:
            x = x + self.attn(norm(x), norm(x), norm(x), cos_sin, kv_cache, if_cache, window_size)
        else:
            x = x + self.attn(norm(h), norm(x), norm(x), cos_sin, kv_cache, if_cache, window_size)
        x = x + self.mlp(norm(x))
        return x

class Block_res(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = _build_attn(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, h=None, cos_sin=None, kv_cache=None, if_cache=True, window_size=(-1, 0)):
        if h is None:
            x = x + self.attn(norm(x), norm(x), norm(x), cos_sin, kv_cache, if_cache, window_size)
        else:
            x = norm(x+h)
            x = x + self.attn(norm(x), norm(x), norm(x), cos_sin, kv_cache, if_cache, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        self.K=config.K
        self.loss_type=config.loss_type
        self.window_sizes = self._compute_window_sizes(config)
        # For DDP, we want vocab_size divisible by world_size. Also, there are potential performance benefits, see:
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} to be divisible by {pad_vocab_size_to}")
        if "res" in self.config.model_type:
            self.transformer = nn.ModuleDict({
                "wte": nn.Embedding(padded_vocab_size, config.n_embd),
                "h": nn.ModuleList([Block_res(config, layer_idx) for layer_idx in range(config.n_layer)]),
            })
        else:
            self.transformer = nn.ModuleDict({
                "wte": nn.Embedding(padded_vocab_size, config.n_embd),
                "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
            })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            if isinstance(block.attn, MLACausalSelfAttention):
                s_up = 3**0.5 * self.config.kv_lora_rank**-0.5 # c_k_up/c_v_up input dim is kv_lora_rank, not n_embd
                torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_kv_down.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_k_up.weight, -s_up, s_up)
                torch.nn.init.uniform_(block.attn.c_v_up.weight, -s_up, s_up)
            else:
                torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast token embeddings to bf16: optimizer can tolerate it and it saves memory
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _compute_window_sizes(self, config):
        """
        Compute per-layer (left, right) window size tuples from window_pattern string.
        L = full context (-1, 0), S = quarter context (seq_len//4, 0).
        Pattern is tiled across layers. Final layer is always forced to L (full context).
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern '{pattern}': only S and L are allowed."
        short_window = max(config.sequence_len // 4, 1)
        char_to_size = {"L": (-1, 0), "S": (short_window, 0)}
        sizes = [char_to_size[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        sizes[-1] = (-1, 0)  # final layer always attends to full context
        return sizes

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """ Return the estimated FLOPs per token for the model. Ref: https://arxiv.org/abs/2204.02311 """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h, q, t = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate out all parameters into 3 groups (matrix, embedding, lm_head)
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params)
        # Create the AdamW optimizer for the embedding and lm_head
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        # Combine them the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def let_make_some_noise(self, x, alpha=0.01):
        if (not self.training) or alpha <= 0:
            return x
        noise = torch.randn_like(x, dtype=torch.float32) * alpha
        noise = noise.to(x.dtype)
        return x + noise


    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', loss_type=None, collect_norms=False):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        
        few_steps = list(range(2*self.K//3, self.K))

        if self.config.model_type in ("FLT", "FLT_res"):
            h_list = []
            hidden_norms = [] if collect_norms else None
            x_0 = self.transformer.wte(idx)
            h=None
            for i in range(self.K):
                x_i=norm(x_0)
                # only the kv in final loop will be cached
                if_cache = True if i==self.K-1 else False
                if self.training and self.config.activation_offload and i < self.K - self.config.activation_offload_keep_last:
                    def _run_blocks(x_i):
                        for layer_idx, block in enumerate(self.transformer.h):
                            x_i = block(x_i, h=h, cos_sin=cos_sin, kv_cache=kv_cache, if_cache=if_cache, window_size=self.window_sizes[layer_idx])
                        return x_i
                    x_i = checkpoint(_run_blocks, x_i, use_reentrant=False)
                else:
                    for layer_idx, block in enumerate(self.transformer.h):
                        x_i = block(x_i, h=h, cos_sin=cos_sin, kv_cache=kv_cache, if_cache=if_cache, window_size=self.window_sizes[layer_idx])
                if collect_norms:
                    hidden_norms.append(x_i.detach().float().norm(dim=-1).mean())
                h = norm(x_i)
                if self.loss_type == "STEP":
                    h_list.append(h)
                elif self.loss_type == "FEW":
                    if i in few_steps:
                        h_list.append(h)
                elif self.loss_type == "END":
                    if i == self.K-1:
                        h_list.append(h)

        elif self.config.model_type=="LT_ia":
            # LT with input injection via first-block attention (h injected into block 0 only)
            h_list = []
            hidden_norms = [] if collect_norms else None
            x_0 = self.transformer.wte(idx)
            x_i = norm(x_0)
            if self.training and self.config.activation_offload and self.config.activation_offload_keep_last < self.K:
                def _run_blocks(x_i):
                    for layer_idx, block in enumerate(self.transformer.h):
                        x_i = block(x_i, cos_sin=cos_sin, kv_cache=kv_cache, if_cache=False, window_size=self.window_sizes[layer_idx])
                    return x_i
                x_i = checkpoint(_run_blocks, x_i, use_reentrant=False)
            else:
                for layer_idx, block in enumerate(self.transformer.h):
                    x_i = block(x_i, cos_sin=cos_sin, kv_cache=kv_cache, if_cache=False, window_size=self.window_sizes[layer_idx])
            if collect_norms:
                hidden_norms.append(x_i.detach().float().norm(dim=-1).mean())
            h = norm(x_i)
            # K-1 Forward
            for i in range(self.K-1):
                x_i=norm(x_0)
                if_cache = True if i==self.K-2 else False
                if self.training and self.config.activation_offload and i < self.K - self.config.activation_offload_keep_last - 1:
                    def _run_blocks(x_i):
                        for layer_idx, block in enumerate(self.transformer.h):
                            if layer_idx == 0:
                                x_i = block(x_i, h=h, cos_sin=cos_sin, kv_cache=kv_cache, if_cache=if_cache, window_size=self.window_sizes[layer_idx])
                            else:
                                x_i = block(x_i, cos_sin=cos_sin, kv_cache=kv_cache, if_cache=if_cache, window_size=self.window_sizes[layer_idx])
                        return x_i
                    x_i = checkpoint(_run_blocks, x_i, use_reentrant=False)
                else:
                    for layer_idx, block in enumerate(self.transformer.h):
                        if layer_idx == 0:
                            x_i = block(x_i, h=h, cos_sin=cos_sin, kv_cache=kv_cache, if_cache=if_cache, window_size=self.window_sizes[layer_idx])
                        else:
                            x_i = block(x_i, cos_sin=cos_sin, kv_cache=kv_cache, if_cache=if_cache, window_size=self.window_sizes[layer_idx])
                if collect_norms:
                    hidden_norms.append(x_i.detach().float().norm(dim=-1).mean())
                h = norm(x_i)
            h_list.append(h)

        elif self.config.model_type=="LT_i":
            # LT with additive input injection: x_i = norm(x_0 + h)
            h_list = []
            hidden_norms = [] if collect_norms else None
            x_0 = self.transformer.wte(idx)
            x_i = norm(x_0)
            if self.training and self.config.activation_offload and self.config.activation_offload_keep_last < self.K:
                def _run_blocks(x_i):
                    for layer_idx, block in enumerate(self.transformer.h):
                        x_i = block(x_i, cos_sin=cos_sin, kv_cache=kv_cache, if_cache=False, window_size=self.window_sizes[layer_idx])
                    return x_i
                x_i = checkpoint(_run_blocks, x_i, use_reentrant=False)
            else:
                for layer_idx, block in enumerate(self.transformer.h):
                    x_i = block(x_i, cos_sin=cos_sin, kv_cache=kv_cache, if_cache=False, window_size=self.window_sizes[layer_idx])
            if collect_norms:
                hidden_norms.append(x_i.detach().float().norm(dim=-1).mean())
            h = norm(x_i)
            # K-1 Forward
            for i in range(self.K-1):
                x_i=norm(x_0+h)
                if_cache = True if i==self.K-2 else False
                if self.training and self.config.activation_offload and i < self.K - self.config.activation_offload_keep_last - 1:
                    def _run_blocks(x_i):
                        for layer_idx, block in enumerate(self.transformer.h):
                            x_i = block(x_i, cos_sin=cos_sin, kv_cache=kv_cache, if_cache=if_cache, window_size=self.window_sizes[layer_idx])
                        return x_i
                    x_i = checkpoint(_run_blocks, x_i, use_reentrant=False)
                else:
                    for layer_idx, block in enumerate(self.transformer.h):
                        x_i = block(x_i, cos_sin=cos_sin, kv_cache=kv_cache, if_cache=if_cache, window_size=self.window_sizes[layer_idx])
                if collect_norms:
                    hidden_norms.append(x_i.detach().float().norm(dim=-1).mean())
                h = norm(x_i)
            h_list.append(h)
            
        elif self.config.model_type=="LT":
            # original LT without input injection
            # First Forward (global step 0)
            h_list = []
            hidden_norms = [] if collect_norms else None
            x_0 = self.transformer.wte(idx)
            x_i = norm(x_0)
            if_cache = (self.K == 1)  # only cache on the final loop step
            if self.training and self.config.activation_offload and self.config.activation_offload_keep_last < self.K:
                def _run_blocks(x_i):
                    for layer_idx, block in enumerate(self.transformer.h):
                        x_i = block(x_i, cos_sin=cos_sin, kv_cache=kv_cache, if_cache=False, window_size=self.window_sizes[layer_idx])
                    return x_i
                x_i = checkpoint(_run_blocks, x_i, use_reentrant=False)
            else:
                for layer_idx, block in enumerate(self.transformer.h):
                    x_i = block(x_i, cos_sin=cos_sin, kv_cache=kv_cache, if_cache=if_cache, window_size=self.window_sizes[layer_idx])
            if collect_norms:
                hidden_norms.append(x_i.detach().float().norm(dim=-1).mean())
            h = norm(x_i)
            # global step 0: collect h for STEP always; for FEW/END only when K==1
            if self.loss_type == "STEP":
                h_list.append(h)
            elif self.loss_type == "FEW" and 0 in few_steps:
                h_list.append(h)
            elif self.loss_type == "END" and self.K == 1:
                h_list.append(h)

            # K-1 more forwards (global steps 1..K-1)
            for i in range(self.K-1):
                global_step = i + 1  # i is 0-indexed within this sub-loop; global step is i+1
                x_i = h
                if_cache = (global_step == self.K - 1)
                if self.training and self.config.activation_offload and i < self.K - self.config.activation_offload_keep_last - 1:
                    def _run_blocks(x_i):
                        for layer_idx, block in enumerate(self.transformer.h):
                            x_i = block(x_i, cos_sin=cos_sin, kv_cache=kv_cache, if_cache=if_cache, window_size=self.window_sizes[layer_idx])
                        return x_i
                    x_i = checkpoint(_run_blocks, x_i, use_reentrant=False)
                else:
                    for layer_idx, block in enumerate(self.transformer.h):
                        x_i = block(x_i, cos_sin=cos_sin, kv_cache=kv_cache, if_cache=if_cache, window_size=self.window_sizes[layer_idx])
                if collect_norms:
                    hidden_norms.append(x_i.detach().float().norm(dim=-1).mean())
                h = norm(x_i)
                if self.loss_type == "STEP":
                    h_list.append(h)
                elif self.loss_type == "FEW" and global_step in few_steps:
                    h_list.append(h)
                elif self.loss_type == "END" and global_step == self.K - 1:
                    h_list.append(h)

        elif self.config.model_type=="FLT_rl":
            # third version
            x_0 = self.transformer.wte(idx)
            h_list = []
            hidden_norms = [] if collect_norms else None
            h = None
            for i in range(self.K):
                x_i=norm(x_0)
                # only the kv in final loop will be cache
                if_cache = True if i==self.K-1 else False
                if self.training and self.config.activation_offload and i < self.K - self.config.activation_offload_keep_last:
                    def _run_blocks(x_i):
                        for layer_idx, block in enumerate(self.transformer.h):
                            x_i = block(x_i, h=h, cos_sin=cos_sin, kv_cache=kv_cache, if_cache=if_cache, window_size=self.window_sizes[layer_idx])
                        return x_i
                    x_i = checkpoint(_run_blocks, x_i, use_reentrant=False)
                else:
                    for layer_idx, block in enumerate(self.transformer.h):
                        x_i = block(x_i, h=h, cos_sin=cos_sin, kv_cache=kv_cache, if_cache=if_cache, window_size=self.window_sizes[layer_idx])
                if collect_norms:
                    hidden_norms.append(x_i.detach().float().norm(dim=-1).mean())
                h = norm(x_i)
                h_list.append(h)
            
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        if targets is not None:
            # training: compute loss for each h in h_list (supports STEP/FEW/END loss_type)
            # For STEP/FEW, h_list contains K (or K/3) entries. Each lm_head call would
            # produce a (B, T, vocab_size) fp32 logits tensor and cross_entropy would save
            # softmax(logits) for its backward — totalling ~1.6 GB per step. With K steps
            # all backward nodes live simultaneously, easily exceeding 10 GB.
            #
            # Fix: wrap each step's lm_head+loss in checkpoint so that logits are NOT
            # saved during forward; only h_ (B,T,n_embd, ~6 MB) is kept. During backward
            # PyTorch recomputes lm_head one step at a time — true streaming over K steps.
            loss_list = []
            use_loss_ckpt = self.training and len(h_list) > 1
            for h_ in h_list:
                if use_loss_ckpt:
                    loss = checkpoint(
                        _lm_loss, h_, self.lm_head, self.config.vocab_size,
                        softcap, targets, loss_reduction,
                        use_reentrant=False,
                    )
                else:
                    loss = _lm_loss(h_, self.lm_head, self.config.vocab_size,
                                    softcap, targets, loss_reduction)
                loss_list.append(loss)
            if collect_norms:
                return loss_list, hidden_norms
            return loss_list
        else:
            # inference: only need logits from the final loop step
            logits = self.lm_head(h_list[-1])
            logits = logits[..., :self.config.vocab_size]
            logits = logits.float()
            logits = softcap * torch.tanh(logits / softcap)
            return logits

        # # Forward the lm_head (compute logits)
        # softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        # logits = self.lm_head(h) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        # logits = logits[..., :self.config.vocab_size] # slice to remove padding
        # logits = logits.float() # switch to fp32 for logit softcap and loss computation
        # logits = softcap * torch.tanh(logits / softcap) # squash the logits

        # if targets is not None:
        #     # training: given the targets, compute and return the loss
        #     # TODO experiment with chunked cross-entropy?
        #     loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
        #     return loss
        # else:
        #     # inference: just return the logits directly
        #     return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token

            # FLT+RL
            
            # first version
            # x_0 = self.transformer.wte(idx)
            # h=self.let_make_some_noise(norm(x_0), 0.1) if self.training else None
            # for i in range(self.K):
            #     x_i=norm(x_0)
            #     # only the kv in final loop will be cache 
            #     if_cache = True if i==self.K-1 else False 
            #     for block in self.transformer.h:
            #         x_i = block(x_i, h=h, cos_sin=cos_sin, kv_cache=kv_cache, if_cache=if_cache)
            #     h = norm(x_i)
            
            # second version
            # x_0 = self.transformer.wte(idx)
            # x_i=norm(x_0)
            # # only the kv in final loop will be cache 
            # for i, block in enumerate(self.transformer.h):
            #     x_i = block(x_i, cos_sin=cos_sin, kv_cache=kv_cache, if_cache=False)
            # h = self.let_make_some_noise(norm(x_i), 0.01)
            
            # for i in range(self.K-1):
            #     x_i=norm(x_0)
            #     # only the kv in final loop will be cache 
            #     if_cache = True if i==self.K-2 else False 
            #     for block in self.transformer.h:
            #         x_i = block(x_i, h=h, cos_sin=cos_sin, kv_cache=kv_cache, if_cache=if_cache)
            #     h = norm(x_i)
