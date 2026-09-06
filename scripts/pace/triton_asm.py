"""Counts the tensor-core and copy instructions in the PTX Triton generates for the attention
kernel at one configuration, to see which hardware paths the compiler used.

    python scripts/pace/triton_asm.py --seqlen 4096 --head-dim 128 [--causal] > summary.md

Uses the autotuned configuration for that shape (running autotuning first if needed).
"""

import argparse
import re

import torch

from flash_lab import triton_attention as ta

PATTERNS = {
    "wgmma.mma_async (Hopper warpgroup MMA)": r"\bwgmma\.mma_async",
    "mma.sync (Ampere-style MMA)": r"\bmma\.sync",
    "cp.async.bulk.tensor (TMA)": r"\bcp\.async\.bulk\.tensor",
    "cp.async (Ampere async copy)": r"\bcp\.async\.(?:cg|ca)",
    "ldmatrix": r"\bldmatrix",
    "stmatrix": r"\bstmatrix",
    "ex2.approx": r"\bex2\.approx",
    "mbarrier": r"\bmbarrier\.",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--causal", action="store_true")
    args = parser.parse_args()

    q = torch.randn(1, args.seqlen, 8, args.head_dim, device="cuda", dtype=torch.bfloat16)
    torch.ops.flash_lab.attention_triton(q, q, q, args.causal, args.head_dim**-0.5)
    config = ta.autotune_choice(args.seqlen, args.head_dim, args.causal)
    if config is None:  # autotuning was off; it is the fixed configuration
        config = dict(ta._FIXED_CONFIG)

    kernel = ta._attention_fwd_kernel.warmup(
        q, q, q, torch.empty_like(q), torch.empty(1, 8, args.seqlen, device="cuda"),
        *q.stride()[:3], *q.stride()[:3], *q.stride()[:3], *q.stride()[:3],
        8, 1, args.seqlen, args.seqlen, 1.0, ta._seq_bucket(args.seqlen),
        HEAD_DIM=args.head_dim, CAUSAL=args.causal, grid=(1,), **config,
    )  # fmt: skip
    ptx = kernel.asm["ptx"]
    target = re.search(r"\.target\s+(\S+)", ptx)
    print(f"# Triton PTX: seqlen {args.seqlen}, head_dim {args.head_dim}, causal {args.causal}")
    print()
    print(f"- triton {ta.triton.__version__}, target {target.group(1) if target else '?'}")
    print(f"- configuration: {config}")
    print()
    print("| instruction | count in PTX |")
    print("|---|---|")
    for label, pattern in PATTERNS.items():
        print(f"| {label} | {len(re.findall(pattern, ptx))} |")


if __name__ == "__main__":
    main()
