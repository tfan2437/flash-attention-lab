"""Plain PyTorch attention used as ground truth by the tests.

Everything is computed in float64 unless another dtype is requested. With causal=True and
S_q != S_k the mask is aligned to the bottom-right corner, as in FlashAttention-2: query i sees
key j iff j <= i + (S_k - S_q). A row that sees no keys returns zeros and an LSE of -inf.
LSE values are natural-log log-sum-exp of the scaled scores, shape [B, H, S_q].
"""

import math

import torch

from flash_lab.layouts import check_attention_inputs, check_decode_inputs


def causal_mask(seqlen_q: int, seqlen_k: int, device: torch.device | None = None) -> torch.Tensor:
    """[S_q, S_k] boolean mask, True where the key is hidden from the query."""
    rows = torch.arange(seqlen_q, device=device)[:, None]
    cols = torch.arange(seqlen_k, device=device)[None, :]
    return cols > rows + (seqlen_k - seqlen_q)


def _expand_kv(x: torch.Tensor, group: int) -> torch.Tensor:
    # [B, S, H_kv, D] -> [B, S, H, D], so query head h lines up with KV head h // group.
    return x.repeat_interleave(group, dim=2) if group > 1 else x


def _default_scale(head_dim: int, softmax_scale: float | None) -> float:
    return softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    softmax_scale: float | None = None,
    return_lse: bool = False,
    dtype: torch.dtype = torch.float64,
):
    """Materializes the full score matrix. Returns o [B, S_q, H, D] in `dtype` (and lse)."""
    shape = check_attention_inputs(q, k, v, require_unit_stride=False)
    scale = _default_scale(shape.head_dim, softmax_scale)
    q_ = q.to(dtype)
    k_ = _expand_kv(k.to(dtype), shape.group)
    v_ = _expand_kv(v.to(dtype), shape.group)

    scores = torch.einsum("bqhd,bkhd->bhqk", q_, k_) * scale
    if causal:
        hidden = causal_mask(shape.seqlen_q, shape.seqlen_k, q.device)
        scores = scores.masked_fill(hidden, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    # Fully masked rows have lse = -inf; shifting them by 0 keeps exp() at exactly 0.
    shift = lse.masked_fill(torch.isneginf(lse), 0.0)
    probs = torch.exp(scores - shift.unsqueeze(-1))
    out = torch.einsum("bhqk,bkhd->bqhd", probs, v_)
    return (out, lse) if return_lse else out


def attention_tiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    softmax_scale: float | None = None,
    return_lse: bool = False,
    block_q: int = 64,
    block_k: int = 64,
    dtype: torch.dtype = torch.float64,
):
    """FlashAttention-2 forward written with tensor ops.

    Same tiling, online softmax, and causal tile skipping as the kernels, vectorized over batch
    and heads. It never forms the [S_q, S_k] matrix, so agreement with attention() checks the
    algorithm independently of any kernel.
    """
    shape = check_attention_inputs(q, k, v, require_unit_stride=False)
    scale = _default_scale(shape.head_dim, softmax_scale)
    seqlen_q, seqlen_k = shape.seqlen_q, shape.seqlen_k
    offset = seqlen_k - seqlen_q
    neg_inf = float("-inf")

    q_ = q.to(dtype).transpose(1, 2)  # [B, H, S_q, D]
    k_ = _expand_kv(k.to(dtype), shape.group).transpose(1, 2)  # [B, H, S_k, D]
    v_ = _expand_kv(v.to(dtype), shape.group).transpose(1, 2)
    out = torch.zeros_like(q_)
    lse = torch.full(q_.shape[:-1], neg_inf, dtype=dtype, device=q.device)

    for q0 in range(0, seqlen_q, block_q):
        q1 = min(q0 + block_q, seqlen_q)
        stats_shape = (shape.batch, shape.heads, q1 - q0)
        row_max = torch.full(stats_shape, neg_inf, dtype=dtype, device=q.device)
        row_sum = torch.zeros(stats_shape, dtype=dtype, device=q.device)
        acc = torch.zeros((*stats_shape, shape.head_dim), dtype=dtype, device=q.device)

        # The last row of this block sees keys up to q1 - 1 + offset; later tiles are skipped.
        k_end = min(seqlen_k, q1 + offset) if causal else seqlen_k
        for k0 in range(0, max(k_end, 0), block_k):
            k1 = min(k0 + block_k, seqlen_k)
            s = q_[:, :, q0:q1] @ k_[:, :, k0:k1].transpose(-1, -2) * scale
            if causal and k1 - 1 > q0 + offset:  # tile crosses the diagonal
                r = torch.arange(q0, q1, device=q.device)[:, None]
                c = torch.arange(k0, k1, device=q.device)[None, :]
                s = s.masked_fill(c > r + offset, neg_inf)

            new_max = torch.maximum(row_max, s.amax(dim=-1))
            # A row that has only seen masked keys keeps a max of -inf; use 0 as its reference
            # point so the exponentials below are exactly 0 instead of nan.
            ref = new_max.masked_fill(torch.isneginf(new_max), 0.0)
            alpha = torch.exp(row_max - ref)
            p = torch.exp(s - ref.unsqueeze(-1))
            row_sum = alpha * row_sum + p.sum(dim=-1)
            acc = alpha.unsqueeze(-1) * acc + p @ v_[:, :, k0:k1]
            row_max = new_max

        seen = row_sum > 0
        safe_sum = row_sum.masked_fill(~seen, 1.0)
        out[:, :, q0:q1] = acc / safe_sum.unsqueeze(-1)
        lse[:, :, q0:q1] = torch.where(seen, row_max + torch.log(safe_sum), neg_inf)

    out = out.transpose(1, 2)
    return (out, lse) if return_lse else out


def gather_cache(
    cache: torch.Tensor, block_tables: torch.Tensor | None, batch_idx: int, length: int
) -> torch.Tensor:
    """Rows 0..length-1 of one sequence as [length, H_kv, D], from a contiguous or paged cache."""
    if block_tables is None:
        return cache[batch_idx, :length]
    page_size = cache.shape[1]
    num_pages = -(-length // page_size)
    pages = block_tables[batch_idx, :num_pages].long()
    return cache[pages].reshape(-1, *cache.shape[2:])[:length]


def decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    block_tables: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    return_lse: bool = False,
    dtype: torch.dtype = torch.float64,
):
    """One query token per sequence against the first seq_lens[b] cached keys.

    Returns o with the same rank as q ([B, 1, H, D] or [B, H, D]) and lse [B, H].
    """
    q3, shape = check_decode_inputs(
        q, k_cache, v_cache, seq_lens, block_tables, require_unit_stride=False
    )
    scale = _default_scale(shape.head_dim, softmax_scale)
    outs, lses = [], []
    for b, length in enumerate(seq_lens.tolist()):
        if not 1 <= length <= shape.max_seqlen:
            raise ValueError(f"seq_lens[{b}] = {length} is outside [1, {shape.max_seqlen}]")
        keys = gather_cache(k_cache, block_tables, b, length)
        values = gather_cache(v_cache, block_tables, b, length)
        o_b, lse_b = attention(
            q3[b][None, None],
            keys[None],
            values[None],
            softmax_scale=scale,
            return_lse=True,
            dtype=dtype,
        )
        outs.append(o_b[0, 0])
        lses.append(lse_b[0, :, 0])
    out = torch.stack(outs)
    if q.dim() == 4:
        out = out.unsqueeze(1)
    lse = torch.stack(lses)
    return (out, lse) if return_lse else out
