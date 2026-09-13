"""Gives every implementation the same calling convention for the benchmark harness.

prefill_runner(impl, q, k, v, causal, scale) and decode_runner(impl, q, k_cache, v_cache,
seq_lens, scale) return (run, context): run() computes the output with no other work, in the
[B, S, H, D] layout, and context is entered once around the timing loop (SDPA backend selection).
Input preparation, such as layout views, happens before run() is created and is not timed.
"""

import contextlib

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import flash_lab

SDPA_BACKENDS = {
    "sdpa_math": SDPBackend.MATH,
    "sdpa_efficient": SDPBackend.EFFICIENT_ATTENTION,
    "sdpa_flash": SDPBackend.FLASH_ATTENTION,
    "sdpa_cudnn": SDPBackend.CUDNN_ATTENTION,
}


class Unsupported(Exception):
    """The implementation cannot run this input (missing package, dtype, or shape)."""


def _flash_attn():
    try:
        import flash_attn
    except ImportError as exc:
        raise Unsupported(f"flash-attn is not installed ({exc})") from exc
    return flash_attn


def _sdpa(backend, q, k, v, causal, scale):
    # SDPA wants [B, H, S, D]; transposed views keep the [B, S, H, D] storage, as flash-attn does.
    qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    gqa = q.shape[2] != k.shape[2]

    def run():
        out = F.scaled_dot_product_attention(
            qt, kt, vt, is_causal=causal, scale=scale, enable_gqa=gqa
        )
        return out.transpose(1, 2)

    return run, sdpa_kernel(backend)


def prefill_runner(impl, q, k, v, causal, scale):
    if impl in SDPA_BACKENDS:
        if causal and q.shape[1] != k.shape[1]:
            raise Unsupported("SDPA's is_causal is top-left aligned")
        return _sdpa(SDPA_BACKENDS[impl], q, k, v, causal, scale)
    if impl == "flash_attn":
        flash_attn = _flash_attn()
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise Unsupported("flash-attn runs fp16 and bf16 only")
        return (
            lambda: flash_attn.flash_attn_func(q, k, v, softmax_scale=scale, causal=causal),
            contextlib.nullcontext(),
        )
    if impl in flash_lab.ops.PREFILL_IMPLS:
        if impl not in flash_lab.available_impls()["attention"]:
            raise Unsupported(f"{impl} is not compiled into this build")
        return (
            lambda: flash_lab.attention(q, k, v, causal=causal, softmax_scale=scale, impl=impl),
            contextlib.nullcontext(),
        )
    raise Unsupported(f"unknown prefill impl {impl!r}")


def decode_runner(impl, q, k_cache, v_cache, seq_lens, scale):
    """q is [B, 1, H, D]; every sequence uses all S_max rows of its contiguous cache.

    `splitkv@N` runs the split-KV kernel with a fixed N splits instead of its heuristic.
    """
    num_splits = None
    if "@" in impl:
        impl, splits = impl.split("@")
        num_splits = int(splits)
    if impl in SDPA_BACKENDS:
        return _sdpa(SDPA_BACKENDS[impl], q, k_cache, v_cache, False, scale)
    if impl == "flash_attn":
        flash_attn = _flash_attn()
        return (
            lambda: flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache, cache_seqlens=seq_lens, softmax_scale=scale
            ),
            contextlib.nullcontext(),
        )
    if impl in flash_lab.ops.DECODE_IMPLS:
        if impl not in flash_lab.available_impls()["decode"]:
            raise Unsupported(f"{impl} is not compiled into this build")
        return (
            lambda: flash_lab.decode(
                q, k_cache, v_cache, seq_lens, softmax_scale=scale, impl=impl, num_splits=num_splits
            ),
            contextlib.nullcontext(),
        )
    raise Unsupported(f"unknown decode impl {impl!r}")
