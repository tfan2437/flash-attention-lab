"""Small, fast cases for every CUDA kernel, run under compute-sanitizer by
scripts/pace/sanitize.sh (`pytest -m sanitize`). Correctness is checked elsewhere; these only
have to exercise ragged edges, masking, GQA, and splits with few threads."""

import pytest
import torch

import flash_lab

pytestmark = [pytest.mark.gpu, pytest.mark.sanitize]

PREFILL = [
    ("naive", torch.float32),
    ("fp32_fused", torch.float32),
    ("fp32_regtile", torch.float32),
    ("mma", torch.bfloat16),
    ("mma_pipelined", torch.bfloat16),
]


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("shape", [(2, 70, 70, 4, 2, 64), (1, 33, 65, 2, 1, 128)], ids=str)
@pytest.mark.parametrize("impl,dtype", PREFILL, ids=lambda x: str(x).replace("torch.", ""))
def test_prefill(impl, dtype, shape, causal):
    batch, seqlen_q, seqlen_k, heads, heads_kv, head_dim = shape
    q = torch.randn(batch, seqlen_q, heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, seqlen_k, heads_kv, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, seqlen_k, heads_kv, head_dim, device="cuda", dtype=dtype)
    out = flash_lab.attention(q, k, v, causal=causal, impl=impl)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("num_splits", [None, 3])
@pytest.mark.parametrize("impl", ["decode_copy", "decode_inplace", "splitkv"])
def test_decode(impl, num_splits):
    q = torch.randn(3, 1, 8, 64, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.randn(3, 150, 2, 64, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn(3, 150, 2, 64, device="cuda", dtype=torch.bfloat16)
    seq_lens = torch.tensor([150, 1, 77], device="cuda", dtype=torch.int32)
    out = flash_lab.decode(q, k_cache, v_cache, seq_lens, impl=impl, num_splits=num_splits)
    assert torch.isfinite(out).all()


def test_mma_tile(flash_ops):
    a = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(32, 16, device="cuda", dtype=torch.bfloat16)
    assert torch.isfinite(flash_ops.mma_tile_test(a, b, True)).all()
