"""Input validation of the public API. Runs on CPU: errors are raised before any kernel."""

import pytest
import torch

import flash_lab


def qkv(batch=2, seqlen_q=8, seqlen_k=8, heads=4, heads_kv=4, head_dim=16, dtype=torch.float32):
    q = torch.randn(batch, seqlen_q, heads, head_dim, dtype=dtype)
    k = torch.randn(batch, seqlen_k, heads_kv, head_dim, dtype=dtype)
    v = torch.randn(batch, seqlen_k, heads_kv, head_dim, dtype=dtype)
    return q, k, v


def test_rejects_wrong_rank():
    q, k, v = qkv()
    with pytest.raises(ValueError, match="4-D"):
        flash_lab.attention(q[0], k, v)


def test_rejects_non_tensors():
    _, k, v = qkv()
    with pytest.raises(TypeError, match="torch.Tensor"):
        flash_lab.attention([[1.0]], k, v)


def test_rejects_mixed_dtypes():
    q, k, v = qkv()
    with pytest.raises(TypeError, match="dtype"):
        flash_lab.attention(q, k.double(), v)


def test_rejects_mixed_devices():
    q, k, v = qkv()
    with pytest.raises(ValueError, match="one device"):
        flash_lab.attention(q, k.to("meta"), v)


def test_rejects_heads_not_divisible_by_kv_heads():
    q, k, v = qkv(heads=6, heads_kv=4)
    with pytest.raises(ValueError, match="multiple of KV heads"):
        flash_lab.attention(q, k, v)


def test_rejects_mismatched_batch_and_head_dim():
    q, k, v = qkv()
    with pytest.raises(ValueError, match="batch"):
        flash_lab.attention(q[:1], k, v)
    with pytest.raises(ValueError, match="head_dim"):
        flash_lab.attention(q[..., :8], k, v)
    with pytest.raises(ValueError, match="same shape"):
        flash_lab.attention(q, k, v[:, :4])


def test_requires_unit_stride_in_head_dim():
    q, k, v = qkv()
    strided = torch.randn(2, 8, 4, 32)[..., ::2]
    with pytest.raises(ValueError, match="last dimension"):
        flash_lab.attention(strided, k, v)


def test_rejects_unknown_impl():
    q, k, v = qkv()
    with pytest.raises(ValueError, match="unknown impl"):
        flash_lab.attention(q, k, v, impl="does-not-exist")


def test_rejects_cpu_tensors():
    q, k, v = qkv()
    with pytest.raises(ValueError, match="CUDA"):
        flash_lab.attention(q, k, v)


def decode_inputs(batch=2, max_seqlen=16, heads=4, heads_kv=2, head_dim=16):
    q = torch.randn(batch, 1, heads, head_dim)
    k_cache = torch.randn(batch, max_seqlen, heads_kv, head_dim)
    v_cache = torch.randn(batch, max_seqlen, heads_kv, head_dim)
    seq_lens = torch.full((batch,), max_seqlen, dtype=torch.int32)
    return q, k_cache, v_cache, seq_lens


def test_decode_validation():
    q, k_cache, v_cache, seq_lens = decode_inputs()
    with pytest.raises(TypeError, match="int32"):
        flash_lab.decode(q, k_cache, v_cache, seq_lens.long())
    with pytest.raises(ValueError, match="seq_lens must have shape"):
        flash_lab.decode(q, k_cache, v_cache, seq_lens[:1])
    with pytest.raises(ValueError, match="seq_lens must be contiguous"):
        flash_lab.decode(q, k_cache, v_cache, seq_lens[:1].expand(2))
    with pytest.raises(ValueError, match="one query token"):
        flash_lab.decode(q.expand(2, 3, 4, 16), k_cache, v_cache, seq_lens)
    with pytest.raises(ValueError, match="same shape"):
        flash_lab.decode(q, k_cache, v_cache[:, :8], seq_lens)
    with pytest.raises(ValueError, match="num_splits"):
        flash_lab.decode(q, k_cache, v_cache, seq_lens, num_splits=0)


def test_decode_paged_validation():
    q, _, _, seq_lens = decode_inputs()
    pool = torch.randn(10, 4, 2, 16)
    tables = torch.zeros(2, 4, dtype=torch.int64)
    with pytest.raises(TypeError, match="block_tables must be int32"):
        flash_lab.decode(q, pool, pool, seq_lens, block_tables=tables)
    with pytest.raises(ValueError, match="block_tables must be"):
        flash_lab.decode(q, pool, pool, seq_lens, block_tables=tables[:1].int())


def test_available_impls_lists_both_entry_points():
    impls = flash_lab.available_impls()
    assert set(impls) == {"attention", "decode"}
