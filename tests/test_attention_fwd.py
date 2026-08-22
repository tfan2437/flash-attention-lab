"""Prefill kernels against the fp64 reference.

Tolerance rule, as in the flash-attn test suite: the kernel's max error against fp64 must be at
most twice the error of PyTorch's math backend at the same dtype, plus a small floor. An absolute
guard on top catches a kernel and a baseline that are both badly wrong.
"""

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import flash_lab
from flash_lab import reference

pytestmark = pytest.mark.gpu

FLOOR = {torch.float32: 1e-6, torch.bfloat16: 1e-3, torch.float16: 1e-3}
GUARD = {torch.float32: 2e-5, torch.bfloat16: 2e-2, torch.float16: 1e-2}
LSE_FLOOR = {torch.float32: 1e-5, torch.bfloat16: 1e-3, torch.float16: 1e-3}

# (impl, dtype) pairs under test. Each kernel adds its rows here.
IMPLS = [
    ("fp32_fused", torch.float32),
    ("mma", torch.bfloat16),
    ("mma", torch.float16),
]

SHAPES = [
    # batch, seqlen_q, seqlen_k, heads, heads_kv, head_dim
    (1, 1, 1, 1, 1, 64),
    (3, 7, 7, 8, 8, 64),
    (1, 16, 16, 1, 1, 128),
    (3, 63, 63, 8, 2, 128),
    (1, 128, 128, 8, 1, 64),
    (3, 257, 257, 8, 8, 128),
    (1, 1024, 1024, 8, 2, 64),
    (2, 1024, 1024, 4, 4, 128),
    (2, 37, 101, 4, 2, 64),  # S_q < S_k
    (1, 101, 37, 4, 4, 128),  # S_q > S_k: causal rows without keys
    (2, 1, 300, 8, 2, 128),
]
SLOW_SHAPES = [(1, 4096, 4096, 4, 4, 128), (1, 8192, 8192, 2, 1, 64)]


def make_inputs(shape, dtype, scale=1.0):
    batch, seqlen_q, seqlen_k, heads, heads_kv, head_dim = shape
    q = torch.randn(batch, seqlen_q, heads, head_dim, device="cuda", dtype=dtype) * scale
    k = torch.randn(batch, seqlen_k, heads_kv, head_dim, device="cuda", dtype=dtype) * scale
    v = torch.randn(batch, seqlen_k, heads_kv, head_dim, device="cuda", dtype=dtype)
    return q, k, v


def sdpa_math(q, k, v, causal):
    """PyTorch's math backend at the input dtype, with the bottom-right causal mask."""
    group = q.shape[2] // k.shape[2]
    k = k.repeat_interleave(group, dim=2)
    v = v.repeat_interleave(group, dim=2)
    mask = None
    if causal:
        mask = ~reference.causal_mask(q.shape[1], k.shape[1], q.device)
    with sdpa_kernel(SDPBackend.MATH):
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask
        )
    return out.transpose(1, 2)


def check_output(out, lse, q, k, v, causal):
    dtype = q.dtype
    ref, ref_lse = reference.attention(q, k, v, causal=causal, return_lse=True)
    assert out.shape == q.shape and out.dtype == dtype
    assert lse.shape == ref_lse.shape and lse.dtype == torch.float32
    assert torch.isfinite(out).all()

    err = (out.double() - ref).abs().max().item()
    # The math backend returns nan for rows without visible keys; those rows are defined as 0.
    err_torch = (sdpa_math(q, k, v, causal).double() - ref).nan_to_num(0.0).abs().max().item()
    assert err <= 2 * err_torch + FLOOR[dtype], f"max error {err:.3g}, torch math {err_torch:.3g}"
    assert err <= GUARD[dtype] * (1 + v.abs().max().item())

    # LSE error grows with the magnitude of the scores, so it gets the same relative rule:
    # at most twice the error of plain PyTorch ops at the input dtype.
    visible = torch.isfinite(ref_lse)
    assert torch.equal(torch.isfinite(lse), visible)
    assert torch.isneginf(lse[~visible]).all()
    if visible.any():
        _, torch_lse = reference.attention(q, k, v, causal=causal, return_lse=True, dtype=dtype)
        lse_err = (lse.double() - ref_lse)[visible].abs().max().item()
        lse_err_torch = (torch_lse.double() - ref_lse)[visible].abs().max().item()
        assert lse_err <= 2 * lse_err_torch + LSE_FLOOR[dtype], (
            f"lse error {lse_err:.3g}, torch {lse_err_torch:.3g}"
        )


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("shape", SHAPES, ids=str)
@pytest.mark.parametrize("impl,dtype", IMPLS, ids=lambda x: str(x).replace("torch.", ""))
def test_matches_reference(impl, dtype, shape, causal):
    q, k, v = make_inputs(shape, dtype)
    out, lse = flash_lab.attention(q, k, v, causal=causal, impl=impl, return_lse=True)
    check_output(out, lse, q, k, v, causal)


@pytest.mark.slow
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("shape", SLOW_SHAPES, ids=str)
@pytest.mark.parametrize("impl,dtype", IMPLS, ids=lambda x: str(x).replace("torch.", ""))
def test_long_sequences(impl, dtype, shape, causal):
    q, k, v = make_inputs(shape, dtype)
    out, lse = flash_lab.attention(q, k, v, causal=causal, impl=impl, return_lse=True)
    check_output(out, lse, q, k, v, causal)


@pytest.mark.parametrize("impl,dtype", IMPLS, ids=lambda x: str(x).replace("torch.", ""))
def test_large_scores(impl, dtype):
    # Scores of order 100 overflow exp() unless the running max is subtracted.
    q, k, v = make_inputs((2, 129, 129, 4, 2, 64), dtype, scale=10.0)
    out, lse = flash_lab.attention(q, k, v, causal=True, impl=impl, return_lse=True)
    check_output(out, lse, q, k, v, causal=True)


@pytest.mark.parametrize("impl,dtype", IMPLS, ids=lambda x: str(x).replace("torch.", ""))
def test_strided_inputs(impl, dtype):
    # q, k, v sliced out of one packed [B, S, 3, H, D] projection: no copies, unusual strides.
    batch, seqlen, heads, head_dim = 2, 77, 4, 64
    qkv = torch.randn(batch, seqlen, 3, heads, head_dim, device="cuda", dtype=dtype)
    q, k, v = qkv.unbind(dim=2)
    assert not q.is_contiguous()
    out, lse = flash_lab.attention(q, k, v, causal=True, impl=impl, return_lse=True)
    check_output(out, lse, q, k, v, causal=True)


@pytest.mark.parametrize("impl,dtype", IMPLS, ids=lambda x: str(x).replace("torch.", ""))
def test_deterministic(impl, dtype):
    q, k, v = make_inputs((2, 300, 300, 8, 2, 128), dtype)
    first = flash_lab.attention(q, k, v, causal=True, impl=impl)
    second = flash_lab.attention(q, k, v, causal=True, impl=impl)
    assert torch.equal(first, second)


@pytest.mark.parametrize("impl,dtype", IMPLS, ids=lambda x: str(x).replace("torch.", ""))
def test_rejects_unsupported_inputs(impl, dtype):
    q, k, v = make_inputs((1, 8, 8, 2, 2, 96), dtype)
    with pytest.raises(ValueError, match="head_dim"):
        flash_lab.attention(q, k, v, impl=impl)
    q, k, v = make_inputs((1, 8, 8, 2, 2, 64), torch.float64)
    with pytest.raises(ValueError, match="dtype"):
        flash_lab.attention(q, k, v, impl=impl)


@pytest.mark.parametrize("tile", ["32x8", "16x16", "8x32"])
def test_fp32_tile_shapes(tile, monkeypatch):
    monkeypatch.setenv("FLASH_LAB_FP32_TILE", tile)
    q, k, v = make_inputs((2, 75, 75, 4, 2, 128), torch.float32)
    out, lse = flash_lab.attention(q, k, v, causal=True, impl="fp32_fused", return_lse=True)
    check_output(out, lse, q, k, v, causal=True)


def test_auto_picks_a_kernel_for_fp32():
    q, k, v = make_inputs((1, 32, 32, 2, 2, 64), torch.float32)
    torch.testing.assert_close(
        flash_lab.attention(q, k, v), flash_lab.attention(q, k, v, impl="fp32_fused")
    )
