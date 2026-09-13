"""Public entry points.

Each kernel is registered by the compiled extension as torch.ops.flash_lab.<op>. This module
validates inputs, picks an implementation, and calls the op. Kernels are listed in the tables
below in order of preference; impl="auto" takes the first one that supports the input.
"""

import math
from dataclasses import dataclass

import torch

from flash_lab import (
    triton_attention,  # noqa: F401  (registers attention_triton if Triton is present)
)
from flash_lab.layouts import AttnShape, DecodeShape, check_attention_inputs, check_decode_inputs

try:
    from flash_lab import _C  # noqa: F401  (importing the extension registers the CUDA ops)
except ImportError as exc:
    HAS_EXTENSION = False
    EXTENSION_ERROR = str(exc)
else:
    HAS_EXTENSION = True
    EXTENSION_ERROR = ""


@dataclass(frozen=True)
class PrefillImpl:
    op: str
    dtypes: tuple[torch.dtype, ...]
    head_dims: tuple[int, ...]
    gqa: bool = True
    causal_unequal_lengths: bool = True  # causal with S_q != S_k (bottom-right aligned)

    def unsupported_reason(self, shape: AttnShape, dtype: torch.dtype, causal: bool) -> str | None:
        if dtype not in self.dtypes:
            return f"dtype {dtype} is not one of {self.dtypes}"
        if shape.head_dim not in self.head_dims:
            return f"head_dim {shape.head_dim} is not one of {self.head_dims}"
        if shape.group > 1 and not self.gqa:
            return "grouped-query attention (H_kv < H) is not supported"
        if causal and shape.seqlen_q != shape.seqlen_k and not self.causal_unequal_lengths:
            return "causal attention requires S_q == S_k"
        return None


@dataclass(frozen=True)
class DecodeImpl:
    op: str
    dtypes: tuple[torch.dtype, ...]
    head_dims: tuple[int, ...]
    paged: bool = False
    in_auto: bool = True  # False for baselines kept for comparison only

    def unsupported_reason(self, shape: DecodeShape, dtype: torch.dtype) -> str | None:
        if dtype not in self.dtypes:
            return f"dtype {dtype} is not one of {self.dtypes}"
        if shape.head_dim not in self.head_dims:
            return f"head_dim {shape.head_dim} is not one of {self.head_dims}"
        if shape.paged and not self.paged:
            return "paged KV caches are not supported"
        return None


PREFILL_IMPLS: dict[str, PrefillImpl] = {
    "mma_pipelined": PrefillImpl(
        op="attention_mma_pipelined", dtypes=(torch.bfloat16, torch.float16), head_dims=(64, 128)
    ),
    "triton": PrefillImpl(
        op="attention_triton", dtypes=(torch.bfloat16, torch.float16), head_dims=(64, 128)
    ),
    "mma": PrefillImpl(
        op="attention_mma", dtypes=(torch.bfloat16, torch.float16), head_dims=(64, 128)
    ),
    "fp32_fused": PrefillImpl(
        op="attention_fp32_fused", dtypes=(torch.float32,), head_dims=(64, 128)
    ),
}
_DECODE_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
DECODE_IMPLS: dict[str, DecodeImpl] = {
    "splitkv": DecodeImpl(op="decode_splitkv", dtypes=_DECODE_DTYPES, head_dims=(64, 128)),
    "decode_inplace": DecodeImpl(op="decode_inplace", dtypes=_DECODE_DTYPES, head_dims=(64, 128)),
    "decode_copy": DecodeImpl(
        op="decode_copy", dtypes=_DECODE_DTYPES, head_dims=(64, 128), in_auto=False
    ),
}


def _op_exists(op: str) -> bool:
    return hasattr(torch.ops.flash_lab, op)


def available_impls() -> dict[str, list[str]]:
    """Implementations compiled into this build, by entry point."""
    return {
        "attention": [name for name, spec in PREFILL_IMPLS.items() if _op_exists(spec.op)],
        "decode": [name for name, spec in DECODE_IMPLS.items() if _op_exists(spec.op)],
    }


def _select(table: dict, impl: str, device: torch.device, describe: str, reason_of):
    if impl != "auto" and impl not in table:
        raise ValueError(f"unknown impl {impl!r}; choose from {['auto', *table]}")
    if device.type != "cuda":
        raise ValueError(f"flash_lab kernels take CUDA tensors, got device {device}")
    missing = "" if HAS_EXTENSION else f" (the CUDA extension did not load: {EXTENSION_ERROR})"

    if impl != "auto":
        spec = table[impl]
        reason = reason_of(spec)
        if reason is not None:
            raise ValueError(f"impl {impl!r} cannot run {describe}: {reason}")
        if not _op_exists(spec.op):
            raise RuntimeError(f"impl {impl!r} is not available in this build{missing}")
        return spec
    for spec in table.values():
        if getattr(spec, "in_auto", True) and reason_of(spec) is None and _op_exists(spec.op):
            return spec
    raise ValueError(f"no available implementation supports {describe}{missing}")


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    softmax_scale: float | None = None,
    impl: str = "auto",
    return_lse: bool = False,
):
    """Attention forward. q is [B, S_q, H, D]; k and v are [B, S_k, H_kv, D] with H % H_kv == 0.

    Returns o [B, S_q, H, D] in the input dtype and, with return_lse=True, the natural-log
    log-sum-exp of each score row as fp32 [B, H, S_q]. Causal masking with S_q != S_k is
    aligned to the bottom-right corner.
    """
    shape = check_attention_inputs(q, k, v)
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(shape.head_dim)
    describe = (
        f"dtype={q.dtype}, head_dim={shape.head_dim}, heads={shape.heads}/{shape.heads_kv}, "
        f"S_q={shape.seqlen_q}, S_k={shape.seqlen_k}, causal={causal}"
    )
    spec = _select(
        PREFILL_IMPLS,
        impl,
        q.device,
        describe,
        lambda s: s.unsupported_reason(shape, q.dtype, causal),
    )
    out, lse = getattr(torch.ops.flash_lab, spec.op)(q, k, v, causal, scale)
    return (out, lse) if return_lse else out


def decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    block_tables: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    impl: str = "auto",
    num_splits: int | None = None,
    return_lse: bool = False,
):
    """One decode step: each sequence's single query token attends to its first seq_lens[b] keys.

    The cache is [B, S_max, H_kv, D], or [num_blocks, page_size, H_kv, D] with block_tables.
    seq_lens values are not checked on the host (that would force a device sync).
    num_splits=None lets the kernel choose how many blocks share one sequence's keys.
    """
    q3, shape = check_decode_inputs(q, k_cache, v_cache, seq_lens, block_tables)
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(shape.head_dim)
    if num_splits is not None and num_splits < 1:
        raise ValueError(f"num_splits must be positive, got {num_splits}")
    describe = (
        f"dtype={q.dtype}, head_dim={shape.head_dim}, heads={shape.heads}/{shape.heads_kv}, "
        f"paged={shape.paged}"
    )
    spec = _select(
        DECODE_IMPLS, impl, q.device, describe, lambda s: s.unsupported_reason(shape, q.dtype)
    )
    out, lse = getattr(torch.ops.flash_lab, spec.op)(
        q3, k_cache, v_cache, seq_lens, scale, num_splits or 0
    )
    if q.dim() == 4:
        out = out.unsqueeze(1)
    return (out, lse) if return_lse else out
