"""Shape, dtype, and stride checks shared by the public API and the reference implementation.

Prefill tensors use the [B, S, H, D] layout (batch, sequence, heads, head_dim). K and V may have
fewer heads than Q (grouped-query attention); query head h reads KV head h // (H // H_kv).
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AttnShape:
    batch: int
    seqlen_q: int
    seqlen_k: int
    heads: int
    heads_kv: int
    head_dim: int

    @property
    def group(self) -> int:
        return self.heads // self.heads_kv


@dataclass(frozen=True)
class DecodeShape:
    batch: int
    heads: int
    heads_kv: int
    head_dim: int
    max_seqlen: int  # S_max of a contiguous cache, max_blocks * page_size of a paged one
    page_size: int  # 0 for a contiguous cache

    @property
    def group(self) -> int:
        return self.heads // self.heads_kv

    @property
    def paged(self) -> bool:
        return self.page_size > 0


def _require_tensor(name: str, x: object) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x).__name__}")


def _check_heads(heads: int, heads_kv: int) -> None:
    if heads_kv == 0 or heads % heads_kv != 0:
        raise ValueError(
            f"the number of query heads ({heads}) must be a multiple of KV heads ({heads_kv})"
        )


def _check_unit_stride(named: dict[str, torch.Tensor]) -> None:
    for name, x in named.items():
        if x.stride(-1) != 1:
            raise ValueError(
                f"{name} must be contiguous in its last dimension, got strides {tuple(x.stride())}"
            )


def check_attention_inputs(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, require_unit_stride: bool = True
) -> AttnShape:
    named = {"q": q, "k": k, "v": v}
    for name, x in named.items():
        _require_tensor(name, x)
        if x.dim() != 4:
            raise ValueError(f"{name} must be 4-D [B, S, H, D], got shape {tuple(x.shape)}")
    if not (q.dtype == k.dtype == v.dtype):
        raise TypeError(f"q, k, v must share a dtype, got {q.dtype}, {k.dtype}, {v.dtype}")
    if not (q.device == k.device == v.device):
        raise ValueError(f"q, k, v must be on one device, got {q.device}, {k.device}, {v.device}")
    if k.shape != v.shape:
        raise ValueError(
            f"k and v must have the same shape, got {tuple(k.shape)} and {tuple(v.shape)}"
        )

    batch, seqlen_q, heads, head_dim = q.shape
    batch_k, seqlen_k, heads_kv, head_dim_k = k.shape
    if batch_k != batch:
        raise ValueError(f"batch size mismatch: q has {batch}, k and v have {batch_k}")
    if head_dim_k != head_dim:
        raise ValueError(f"head_dim mismatch: q has {head_dim}, k and v have {head_dim_k}")
    if min(batch, seqlen_q, seqlen_k, heads, head_dim) == 0:
        raise ValueError(f"empty input: q {tuple(q.shape)}, k {tuple(k.shape)}")
    _check_heads(heads, heads_kv)
    if require_unit_stride:
        _check_unit_stride(named)
    return AttnShape(batch, seqlen_q, seqlen_k, heads, heads_kv, head_dim)


def check_decode_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    block_tables: torch.Tensor | None = None,
    *,
    require_unit_stride: bool = True,
) -> tuple[torch.Tensor, DecodeShape]:
    """Validates decode inputs and returns q as a [B, H, D] view plus the problem shape.

    q is [B, 1, H, D] or [B, H, D]. A contiguous cache is [B, S_max, H_kv, D]. A paged cache is
    [num_blocks, page_size, H_kv, D] with block_tables [B, max_blocks] int32. seq_lens [B] int32
    counts the valid keys of each sequence, including the token being decoded.
    """
    named = {"q": q, "k_cache": k_cache, "v_cache": v_cache, "seq_lens": seq_lens}
    if block_tables is not None:
        named["block_tables"] = block_tables
    for name, x in named.items():
        _require_tensor(name, x)
    if len({x.device for x in named.values()}) != 1:
        devices = {name: str(x.device) for name, x in named.items()}
        raise ValueError(f"decode inputs must be on one device, got {devices}")

    if q.dim() == 4:
        if q.shape[1] != 1:
            raise ValueError(f"decode takes one query token per sequence, got q {tuple(q.shape)}")
        q = q[:, 0]
    elif q.dim() != 3:
        raise ValueError(f"q must be [B, 1, H, D] or [B, H, D], got shape {tuple(q.shape)}")
    if k_cache.dim() != 4:
        raise ValueError(f"k_cache must be 4-D, got shape {tuple(k_cache.shape)}")
    if k_cache.shape != v_cache.shape:
        raise ValueError(
            f"k_cache and v_cache must have the same shape, "
            f"got {tuple(k_cache.shape)} and {tuple(v_cache.shape)}"
        )
    if not (q.dtype == k_cache.dtype == v_cache.dtype):
        raise TypeError(
            f"q and the caches must share a dtype, got {q.dtype}, {k_cache.dtype}, {v_cache.dtype}"
        )

    batch, heads, head_dim = q.shape
    if seq_lens.dtype != torch.int32:
        raise TypeError(f"seq_lens must be int32, got {seq_lens.dtype}")
    if seq_lens.shape != (batch,):
        raise ValueError(f"seq_lens must have shape ({batch},), got {tuple(seq_lens.shape)}")

    if block_tables is None:
        batch_c, max_seqlen, heads_kv, head_dim_c = k_cache.shape
        page_size = 0
        if batch_c != batch:
            raise ValueError(f"batch size mismatch: q has {batch}, the cache has {batch_c}")
    else:
        if block_tables.dtype != torch.int32:
            raise TypeError(f"block_tables must be int32, got {block_tables.dtype}")
        if block_tables.dim() != 2 or block_tables.shape[0] != batch:
            raise ValueError(
                f"block_tables must be [{batch}, max_blocks], got {tuple(block_tables.shape)}"
            )
        _, page_size, heads_kv, head_dim_c = k_cache.shape
        max_seqlen = block_tables.shape[1] * page_size

    if head_dim_c != head_dim:
        raise ValueError(f"head_dim mismatch: q has {head_dim}, the cache has {head_dim_c}")
    if min(batch, heads, head_dim, max_seqlen) == 0:
        raise ValueError(f"empty input: q {tuple(q.shape)}, k_cache {tuple(k_cache.shape)}")
    _check_heads(heads, heads_kv)
    if require_unit_stride:
        _check_unit_stride({"q": q, "k_cache": k_cache, "v_cache": v_cache})
    return q, DecodeShape(batch, heads, heads_kv, head_dim, max_seqlen, page_size)
