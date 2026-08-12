"""Checks of the fp64 reference itself: against torch SDPA, closed-form cases, and the tiled
FlashAttention-2 formulation."""

import pytest
import torch
import torch.nn.functional as F

from flash_lab import reference

EXACT = {"rtol": 0.0, "atol": 1e-12}


def make_qkv(batch, seqlen_q, seqlen_k, heads, heads_kv, head_dim, scale=1.0):
    q = torch.randn(batch, seqlen_q, heads, head_dim, dtype=torch.float64) * scale
    k = torch.randn(batch, seqlen_k, heads_kv, head_dim, dtype=torch.float64) * scale
    v = torch.randn(batch, seqlen_k, heads_kv, head_dim, dtype=torch.float64)
    return q, k, v


def sdpa_fp64(q, k, v, causal):
    """torch SDPA on [B, S, H, D] inputs. Its causal mask is top-left aligned, so this is only
    comparable to the reference when S_q == S_k."""
    group = q.shape[2] // k.shape[2]
    k = k.repeat_interleave(group, dim=2)
    v = v.repeat_interleave(group, dim=2)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal
    )
    return out.transpose(1, 2)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "batch,seqlen,heads,heads_kv,head_dim", [(2, 7, 4, 4, 16), (1, 33, 8, 2, 32), (3, 1, 2, 1, 8)]
)
def test_matches_torch_sdpa(batch, seqlen, heads, heads_kv, head_dim, causal):
    q, k, v = make_qkv(batch, seqlen, seqlen, heads, heads_kv, head_dim)
    out = reference.attention(q, k, v, causal=causal)
    torch.testing.assert_close(out, sdpa_fp64(q, k, v, causal), **EXACT)


def test_single_key_returns_its_value():
    q, k, v = make_qkv(2, 5, 1, 4, 2, 8)
    out, lse = reference.attention(q, k, v, return_lse=True)
    expected = v.repeat_interleave(2, dim=2).expand_as(out)
    torch.testing.assert_close(out, expected, **EXACT)
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k.repeat_interleave(2, dim=2)) / 8**0.5
    torch.testing.assert_close(lse, scores[..., 0], **EXACT)


def test_identical_keys_average_the_values():
    q, k, v = make_qkv(1, 4, 6, 2, 2, 8)
    k = k[:, :1].expand_as(k)
    out = reference.attention(q, k, v)
    torch.testing.assert_close(out, v.mean(dim=1, keepdim=True).expand_as(out), **EXACT)


def test_gqa_equals_repeated_kv_heads():
    q, k, v = make_qkv(2, 9, 11, 8, 2, 16)
    grouped = reference.attention(q, k, v, causal=True)
    repeated = reference.attention(
        q, k.repeat_interleave(4, dim=2), v.repeat_interleave(4, dim=2), causal=True
    )
    torch.testing.assert_close(grouped, repeated, **EXACT)


def test_causal_mask_is_bottom_right_aligned():
    seqlen_q, seqlen_k = 3, 7
    q, k, v = make_qkv(1, seqlen_q, seqlen_k, 2, 2, 8)
    out = reference.attention(q, k, v, causal=True)
    for i in range(seqlen_q):
        visible = seqlen_k - seqlen_q + i + 1
        row = reference.attention(q[:, i : i + 1], k[:, :visible], v[:, :visible])
        torch.testing.assert_close(out[:, i : i + 1], row, **EXACT)


def test_rows_without_visible_keys_are_zero():
    # With S_q > S_k and bottom-right alignment, the first S_q - S_k rows see no keys.
    q, k, v = make_qkv(2, 5, 2, 2, 1, 8)
    out, lse = reference.attention(q, k, v, causal=True, return_lse=True)
    assert torch.all(out[:, :3] == 0)
    assert torch.all(torch.isneginf(lse[:, :, :3]))
    assert torch.isfinite(out).all() and torch.isfinite(lse[:, :, 3:]).all()


def test_large_scores_stay_finite_in_fp32():
    # Scores around +-100 overflow exp() in fp32 unless the row max is subtracted first.
    q, k, v = make_qkv(1, 64, 64, 2, 2, 64, scale=10.0)
    out32 = reference.attention(q.float(), k.float(), v.float(), causal=True, dtype=torch.float32)
    out64 = reference.attention(q, k, v, causal=True)
    assert torch.isfinite(out32).all()
    torch.testing.assert_close(out32.double(), out64, rtol=0.0, atol=1e-2)


TILED_CASES = [
    # batch, seqlen_q, seqlen_k, heads, heads_kv, head_dim, block_q, block_k
    (2, 16, 16, 2, 2, 8, 16, 16),
    (1, 37, 37, 4, 2, 16, 16, 8),
    (2, 1, 29, 3, 1, 8, 16, 8),
    (1, 29, 13, 2, 2, 8, 8, 8),
    (1, 13, 50, 2, 1, 8, 4, 16),
    (1, 64, 64, 1, 1, 32, 64, 64),
]


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("case", TILED_CASES)
def test_tiled_matches_naive(case, causal):
    batch, seqlen_q, seqlen_k, heads, heads_kv, head_dim, block_q, block_k = case
    q, k, v = make_qkv(batch, seqlen_q, seqlen_k, heads, heads_kv, head_dim)
    out, lse = reference.attention(q, k, v, causal=causal, return_lse=True)
    out_t, lse_t = reference.attention_tiled(
        q, k, v, causal=causal, return_lse=True, block_q=block_q, block_k=block_k
    )
    torch.testing.assert_close(out_t, out, **EXACT)
    torch.testing.assert_close(lse_t, lse, **EXACT)


def make_cache(batch, max_seqlen, heads_kv, head_dim):
    k = torch.randn(batch, max_seqlen, heads_kv, head_dim, dtype=torch.float64)
    v = torch.randn(batch, max_seqlen, heads_kv, head_dim, dtype=torch.float64)
    return k, v


def to_paged(cache, page_size, generator):
    """Scatters a contiguous [B, S_max, H_kv, D] cache into shuffled pages."""
    batch, max_seqlen = cache.shape[:2]
    pages_per_seq = max_seqlen // page_size
    num_blocks = batch * pages_per_seq + 3  # a few unused blocks
    order = torch.randperm(num_blocks, generator=generator)[: batch * pages_per_seq]
    block_tables = order.view(batch, pages_per_seq).to(torch.int32)
    pool = torch.zeros(num_blocks, page_size, *cache.shape[2:], dtype=cache.dtype)
    pool[order.long()] = cache.reshape(batch * pages_per_seq, page_size, *cache.shape[2:])
    return pool, block_tables


def test_decode_matches_prefill_rows():
    batch, max_seqlen, heads, heads_kv, head_dim = 3, 20, 4, 2, 16
    k_cache, v_cache = make_cache(batch, max_seqlen, heads_kv, head_dim)
    q = torch.randn(batch, 1, heads, head_dim, dtype=torch.float64)
    seq_lens = torch.tensor([1, 7, 20], dtype=torch.int32)
    out, lse = reference.decode(q, k_cache, v_cache, seq_lens, return_lse=True)
    assert out.shape == q.shape and lse.shape == (batch, heads)
    for b, n in enumerate(seq_lens.tolist()):
        row, row_lse = reference.attention(
            q[b : b + 1], k_cache[b : b + 1, :n], v_cache[b : b + 1, :n], return_lse=True
        )
        torch.testing.assert_close(out[b : b + 1], row, **EXACT)
        torch.testing.assert_close(lse[b], row_lse[0, :, 0], **EXACT)


def test_paged_decode_matches_contiguous():
    batch, max_seqlen, heads, heads_kv, head_dim, page_size = 3, 16, 4, 1, 8, 4
    k_cache, v_cache = make_cache(batch, max_seqlen, heads_kv, head_dim)
    gen = torch.Generator().manual_seed(1)
    k_pool, block_tables = to_paged(k_cache, page_size, gen)
    v_pool, _ = to_paged(v_cache, page_size, torch.Generator().manual_seed(1))
    q = torch.randn(batch, heads, head_dim, dtype=torch.float64)
    seq_lens = torch.tensor([5, 16, 9], dtype=torch.int32)

    contiguous = reference.decode(q, k_cache, v_cache, seq_lens)
    paged = reference.decode(q, k_pool, v_pool, seq_lens, block_tables=block_tables)
    assert paged.shape == q.shape
    torch.testing.assert_close(paged, contiguous, **EXACT)


def test_decode_rejects_lengths_outside_the_cache():
    k_cache, v_cache = make_cache(2, 8, 1, 8)
    q = torch.randn(2, 1, 1, 8, dtype=torch.float64)
    with pytest.raises(ValueError, match="outside"):
        reference.decode(q, k_cache, v_cache, torch.tensor([3, 9], dtype=torch.int32))
    with pytest.raises(ValueError, match="outside"):
        reference.decode(q, k_cache, v_cache, torch.tensor([0, 1], dtype=torch.int32))
