"""
Analytically count the parameters of the GPT model (nanochat/gpt.py).

架构参数推导方式与 base_train.py 完全一致:
    num_layers  = depth
    model_dim   = depth * 64                          # aspect ratio 64
    num_heads   = max(1, ceil(model_dim / 128))       # head_dim ≈ 128
    num_kv_heads = num_heads                          # 默认 1:1，即禁用 GQA

参数统计覆盖所有可学习参数（无 bias，RMSNorm 无参数）：

  1. wte (token embedding)       : padded_vocab_size × n_embd
  2. lm_head (output projection) : n_embd × padded_vocab_size
  3. 每个 Transformer Block（共 n_layer 个）：
       attn.c_q   : n_embd × (n_head   × head_dim)  ← Q 投影
       attn.c_k   : n_embd × (n_kv_head × head_dim) ← K 投影（GQA 时比 Q 小）
       attn.c_v   : n_embd × (n_kv_head × head_dim) ← V 投影（GQA 时比 Q 小）
       attn.c_proj: (n_head × head_dim) × n_embd    ← 注意力输出投影
       mlp.c_fc   : n_embd × (4 × n_embd)           ← FFN 升维
       mlp.c_proj : (4 × n_embd) × n_embd           ← FFN 降维

用法:
    python -m scripts.count_params                   # 打印 depth 1..32 的汇总表
    python -m scripts.count_params --depth 12        # 单个 depth 的详细报告
    python -m scripts.count_params --depth 12 --vocab_size 50304
    python -m scripts.count_params --depth 12 --n_kv_head 3   # 启用 GQA
"""

import argparse
import math


# ---------------------------------------------------------------------------
# 核心计算
# ---------------------------------------------------------------------------

def padded_vocab_size(vocab_size: int, pad_to: int = 64) -> int:
    """将 vocab_size 向上对齐到 pad_to 的整数倍（与 GPT.__init__ 中逻辑一致）。"""
    return ((vocab_size + pad_to - 1) // pad_to) * pad_to


def derive_arch(depth: int, vocab_size: int = 50304, n_kv_head: int | None = None) -> dict:
    """
    根据 depth 推导所有架构超参数（与 base_train.py 中推导方式完全一致）。

    参数
    ----
    depth       : 模型深度，同时决定层数和隐层维度 (n_embd = depth * 64)
    vocab_size  : 词表大小（训练脚本中由 tokenizer 提供，默认 50304）
    n_kv_head   : KV head 数量；None 表示与 n_head 相同（即不使用 GQA，
                  与 base_train.py 默认行为一致）
    """
    n_layer = depth
    n_embd  = depth * 64                            # aspect ratio = 64
    n_head  = max(1, (n_embd + 127) // 128)         # ceil(n_embd / 128)，head_dim ≈ 128
    if n_kv_head is None:
        n_kv_head = n_head                          # base_train.py 默认：num_kv_heads = num_heads
    head_dim = n_embd // n_head                     # 每个 attention head 的维度
    pvocab   = padded_vocab_size(vocab_size)
    return dict(
        n_layer=n_layer, n_embd=n_embd, n_head=n_head,
        n_kv_head=n_kv_head, head_dim=head_dim,
        vocab_size=vocab_size, pvocab=pvocab,
    )


def count_params(depth: int, vocab_size: int = 50304, n_kv_head: int | None = None) -> dict:
    """返回各组件参数量的详细分解。"""
    a = derive_arch(depth, vocab_size, n_kv_head)
    n_embd, n_head, n_kv_head_, head_dim, n_layer, pvocab = (
        a["n_embd"], a["n_head"], a["n_kv_head"], a["head_dim"], a["n_layer"], a["pvocab"]
    )

    # ---------- 每个 Block 内的分项 ----------
    attn_q    = n_embd * (n_head    * head_dim)  # c_q：n_head × head_dim = n_embd（无 GQA 时）
    attn_k    = n_embd * (n_kv_head_ * head_dim) # c_k：GQA 时 n_kv_head < n_head → 更小
    attn_v    = n_embd * (n_kv_head_ * head_dim) # c_v：同 c_k
    attn_proj = (n_head * head_dim) * n_embd     # c_proj：注意力输出 → 残差流
    mlp_fc    = n_embd * (4 * n_embd)            # c_fc：FFN 升维 4×
    mlp_proj  = (4 * n_embd) * n_embd            # c_proj：FFN 降维

    per_block  = attn_q + attn_k + attn_v + attn_proj + mlp_fc + mlp_proj
    all_blocks = n_layer * per_block

    # ---------- 嵌入 & 输出层 ----------
    embedding = pvocab * n_embd    # wte
    lm_head   = n_embd * pvocab    # lm_head（权重不与 wte 共享）

    total = embedding + lm_head + all_blocks

    return dict(
        **a,
        attn_q=attn_q, attn_k=attn_k, attn_v=attn_v,
        attn_proj=attn_proj, mlp_fc=mlp_fc, mlp_proj=mlp_proj,
        per_block=per_block, all_blocks=all_blocks,
        embedding=embedding, lm_head=lm_head,
        total=total,
    )


# ---------------------------------------------------------------------------
# 格式化输出
# ---------------------------------------------------------------------------

def fmt_num(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.4f} B"
    if n >= 1_000_000:
        return f"{n / 1e6:.4f} M"
    return f"{n:,}"


def pct(part: int, total: int) -> str:
    return f"{100 * part / total:.1f}%"


def print_detail(r: dict) -> None:
    SEP = "=" * 66
    sep = "-" * 66
    print(f"\n{SEP}")
    print(f"  模型架构参数（由 depth={r['n_layer']} 推导）")
    print(SEP)
    print(f"  {'n_layer  (层数)':30s}: {r['n_layer']}")
    print(f"  {'n_embd   (隐层维度  = depth×64)':30s}: {r['n_embd']}")
    print(f"  {'n_head   (Q head 数, head_dim≈128)':30s}: {r['n_head']}")
    print(f"  {'n_kv_head(KV head 数, GQA 比值)':30s}: {r['n_kv_head']}  "
          f"({'与 Q head 相同，GQA 关闭' if r['n_kv_head'] == r['n_head'] else 'GQA 启用'})")
    print(f"  {'head_dim (= n_embd / n_head)':30s}: {r['head_dim']}")
    print(f"  {'vocab_size (原始词表大小)':30s}: {r['vocab_size']:,}")
    print(f"  {'padded_vocab (对齐到 64 的整数倍)':30s}: {r['pvocab']:,}")
    total = r["total"]

    print(f"\n{sep}")
    print(f"  参数量明细")
    print(sep)
    print(f"  {'wte (token embedding)':40s}: {fmt_num(r['embedding']):>14s}  ({pct(r['embedding'], total)})")
    print(f"  {'lm_head (输出投影，权重不共享)':40s}: {fmt_num(r['lm_head']):>14s}  ({pct(r['lm_head'], total)})")

    print(f"\n  每个 Transformer Block 内部（共 {r['n_layer']} 层）：")
    print(f"  {'  attn.c_q  (Q投影  n_embd→n_head×head_dim)':40s}: {fmt_num(r['attn_q']):>14s}")
    print(f"  {'  attn.c_k  (K投影  n_embd→n_kv_head×head_dim)':40s}: {fmt_num(r['attn_k']):>14s}")
    print(f"  {'  attn.c_v  (V投影  n_embd→n_kv_head×head_dim)':40s}: {fmt_num(r['attn_v']):>14s}")
    print(f"  {'  attn.c_proj (注意力输出→残差流)':40s}: {fmt_num(r['attn_proj']):>14s}")
    print(f"  {'  mlp.c_fc  (FFN 升维  n_embd→4×n_embd)':40s}: {fmt_num(r['mlp_fc']):>14s}")
    print(f"  {'  mlp.c_proj (FFN 降维  4×n_embd→n_embd)':40s}: {fmt_num(r['mlp_proj']):>14s}")
    print(f"  {'  单个 Block 合计':40s}: {fmt_num(r['per_block']):>14s}")
    print(f"  {'  × %d 层 = 全部 Block 合计' % r['n_layer']:40s}: {fmt_num(r['all_blocks']):>14s}  ({pct(r['all_blocks'], total)})")

    print(f"\n{sep}")
    print(f"  {'总参数量':40s}: {fmt_num(total):>14s}  ({r['total']:,})")
    print(SEP)


def print_table(depths: list[int], vocab_size: int = 50304) -> None:
    cols = f"{'depth':>6}  {'n_embd':>7}  {'n_head':>6}  {'embedding':>13}  {'all_blocks':>13}  {'total':>13}"
    print(cols)
    print("─" * len(cols))
    for d in depths:
        r = count_params(d, vocab_size=vocab_size)
        print(f"{d:>6}  {r['n_embd']:>7}  {r['n_head']:>6}  "
              f"{r['embedding']:>13,}  {r['all_blocks']:>13,}  {r['total']:>13,}  "
              f"({fmt_num(r['total'])})")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="按 base_train.py 的推导规则计算 GPT 模型参数量。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m scripts.count_params                    # 打印 depth 1..32 汇总表
  python -m scripts.count_params --depth 12         # depth=12 的详细报告
  python -m scripts.count_params --depth 24 --n_kv_head 4   # 开启 GQA
        """,
    )
    parser.add_argument(
        "--depth", type=int, default=None,
        help="模型深度（= 层数 n_layer）。若不指定则打印 1..sweep_max 的汇总表。",
    )
    parser.add_argument(
        "--vocab_size", type=int, default=50304,
        help="词表大小（默认 50304，与 GPTConfig 默认值一致）。",
    )
    parser.add_argument(
        "--n_kv_head", type=int, default=None,
        help="KV head 数量。不指定时等于 n_head（与 base_train.py 默认行为一致，即禁用 GQA）。",
    )
    parser.add_argument(
        "--sweep_max", type=int, default=32,
        help="汇总表模式下打印 depth 1..sweep_max（默认 32）。",
    )
    args = parser.parse_args()

    if args.depth is not None:
        r = count_params(args.depth, vocab_size=args.vocab_size, n_kv_head=args.n_kv_head)
        print_detail(r)
    else:
        print(f"\n参数量汇总表（depth 1..{args.sweep_max}，vocab_size={args.vocab_size:,}）\n")
        print_table(list(range(1, args.sweep_max + 1)), vocab_size=args.vocab_size)


if __name__ == "__main__":
    main()
