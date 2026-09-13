"""Runs decode steps for several implementations under NVTX ranges, for Nsight Systems.

    nsys profile -t cuda,nvtx python -m bench.decode_trace --impls decode_copy,splitkv

Each implementation gets a warmup phase and then `--steps` steps, each inside an NVTX range
named `<impl>:step` that ends after a device synchronize, so the range length is the step's wall
time and `nsys stats` can attribute every kernel (attention and copies alike) to its step.
"""

import argparse
import math

import torch

import flash_lab


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--impls", default="decode_copy,decode_inplace,splitkv")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--heads-kv", type=int, default=8)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    gen = torch.Generator(device="cuda").manual_seed(0)
    shape = (args.batch, args.context, args.heads_kv, args.head_dim)
    k_cache = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=gen)
    v_cache = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=gen)
    q = torch.randn(
        args.batch, 1, args.heads, args.head_dim, device="cuda", dtype=torch.bfloat16, generator=gen
    )
    seq_lens = torch.full((args.batch,), args.context, device="cuda", dtype=torch.int32)
    scale = 1.0 / math.sqrt(args.head_dim)

    for impl in args.impls.split(","):
        for _ in range(3):
            flash_lab.decode(q, k_cache, v_cache, seq_lens, softmax_scale=scale, impl=impl)
        torch.cuda.synchronize()
        for _ in range(args.steps):
            with torch.cuda.nvtx.range(f"{impl}:step"):
                flash_lab.decode(q, k_cache, v_cache, seq_lens, softmax_scale=scale, impl=impl)
                torch.cuda.synchronize()


if __name__ == "__main__":
    main()
