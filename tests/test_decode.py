"""Decode kernels against the fp64 reference, with the same tolerance rule as prefill."""

import pytest
import torch

import flash_lab
from flash_lab import reference

pytestmark = pytest.mark.gpu

FLOOR = {torch.float32: 1e-6, torch.bfloat16: 1e-3, torch.float16: 1e-3}
LSE_FLOOR = {torch.float32: 1e-5, torch.bfloat16: 1e-3, torch.float16: 1e-3}
IMPLS = ["decode_copy", "decode_inplace", "splitkv"]
DTYPES = [torch.bfloat16, torch.float16, torch.float32]

CASES = [
    # batch, heads, heads_kv, max_seqlen, head_dim, seq_lens
    (1, 8, 8, 64, 64, [64]),
    (4, 8, 1, 1100, 128, [1, 5, 1023, 1100]),
    (3, 32, 4, 4097, 128, [4097, 64, 1024]),
    (2, 6, 2, 300, 64, [300, 17]),
    (2, 32, 8, 700, 128, [700, 333]),
    (16, 8, 2, 512, 64, "random"),
]


def make_case(case, dtype, gen):
    batch, heads, heads_kv, max_seqlen, head_dim, lens = case
    q = torch.randn(batch, 1, heads, head_dim, device="cuda", dtype=dtype, generator=gen)
    shape = (batch, max_seqlen, heads_kv, head_dim)
    k_cache = torch.randn(shape, device="cuda", dtype=dtype, generator=gen)
    v_cache = torch.randn(shape, device="cuda", dtype=dtype, generator=gen)
    if lens == "random":
        seq_lens = torch.randint(1, max_seqlen + 1, (batch,), device="cuda", generator=gen)
    else:
        seq_lens = torch.tensor(lens, device="cuda")
    return q, k_cache, v_cache, seq_lens.to(torch.int32)


def check(out, lse, q, k_cache, v_cache, seq_lens):
    dtype = q.dtype
    ref, ref_lse = reference.decode(q, k_cache, v_cache, seq_lens, return_lse=True)
    plain, plain_lse = reference.decode(q, k_cache, v_cache, seq_lens, return_lse=True, dtype=dtype)
    assert out.shape == q.shape and out.dtype == dtype and torch.isfinite(out).all()
    err = (out.double() - ref).abs().max().item()
    err_torch = (plain.double() - ref).abs().max().item()
    assert err <= 2 * err_torch + FLOOR[dtype], f"max error {err:.3g}, plain torch {err_torch:.3g}"
    lse_err = (lse.double() - ref_lse).abs().max().item()
    lse_err_torch = (plain_lse.double() - ref_lse).abs().max().item()
    assert lse_err <= 2 * lse_err_torch + LSE_FLOOR[dtype], f"lse error {lse_err:.3g}"


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"B{c[0]}-H{c[1]}-Hkv{c[2]}-S{c[3]}-D{c[4]}")
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).removeprefix("torch."))
@pytest.mark.parametrize("impl", IMPLS)
def test_matches_reference(impl, dtype, case):
    gen = torch.Generator(device="cuda").manual_seed(0)
    q, k_cache, v_cache, seq_lens = make_case(case, dtype, gen)
    out, lse = flash_lab.decode(q, k_cache, v_cache, seq_lens, impl=impl, return_lse=True)
    check(out, lse, q, k_cache, v_cache, seq_lens)


@pytest.mark.parametrize("num_splits", [1, 2, 8, 64, None])
def test_splitkv_split_counts(num_splits):
    # More splits than chunks leaves some splits empty; they must not disturb the merge.
    gen = torch.Generator(device="cuda").manual_seed(1)
    q, k_cache, v_cache, seq_lens = make_case(
        (3, 16, 4, 2000, 128, [2000, 40, 777]), torch.bfloat16, gen
    )
    out, lse = flash_lab.decode(
        q, k_cache, v_cache, seq_lens, impl="splitkv", num_splits=num_splits, return_lse=True
    )
    check(out, lse, q, k_cache, v_cache, seq_lens)


@pytest.mark.parametrize("impl", IMPLS)
def test_strided_cache_and_rank3_query(impl):
    # Caches carved out of a larger allocation (every other KV head), q given as [B, H, D].
    gen = torch.Generator(device="cuda").manual_seed(2)
    big = torch.randn(2, 600, 8, 128, device="cuda", dtype=torch.bfloat16, generator=gen)
    k_cache, v_cache = big[:, :, 0::2], big[:, :, 1::2]
    q = torch.randn(2, 16, 128, device="cuda", dtype=torch.bfloat16, generator=gen)
    seq_lens = torch.tensor([600, 123], device="cuda", dtype=torch.int32)
    out, lse = flash_lab.decode(q, k_cache, v_cache, seq_lens, impl=impl, return_lse=True)
    assert out.shape == q.shape
    check(out, lse, q, k_cache, v_cache, seq_lens)


@pytest.mark.parametrize("impl", IMPLS)
def test_deterministic(impl):
    gen = torch.Generator(device="cuda").manual_seed(3)
    q, k_cache, v_cache, seq_lens = make_case(
        (4, 32, 8, 3000, 128, [3000, 1, 2048, 999]), torch.bfloat16, gen
    )
    first = flash_lab.decode(q, k_cache, v_cache, seq_lens, impl=impl)
    second = flash_lab.decode(q, k_cache, v_cache, seq_lens, impl=impl)
    assert torch.equal(first, second)


def test_auto_skips_the_copy_baseline():
    gen = torch.Generator(device="cuda").manual_seed(4)
    q, k_cache, v_cache, seq_lens = make_case((2, 8, 8, 64, 64, [64, 3]), torch.bfloat16, gen)
    auto = flash_lab.decode(q, k_cache, v_cache, seq_lens)
    torch.testing.assert_close(
        auto, flash_lab.decode(q, k_cache, v_cache, seq_lens, impl="splitkv"), rtol=0, atol=0
    )


@pytest.mark.parametrize("impl", IMPLS)
def test_rejects_unsupported(impl):
    q = torch.randn(1, 1, 4, 96, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(1, 8, 4, 96, device="cuda", dtype=torch.bfloat16)
    seq_lens = torch.tensor([8], device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="head_dim"):
        flash_lab.decode(q, cache, cache, seq_lens, impl=impl)
    q, cache = q[..., :64], torch.randn(4, 4, 4, 64, device="cuda", dtype=torch.bfloat16)
    tables = torch.zeros(1, 2, device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="paged"):
        flash_lab.decode(q.contiguous(), cache, cache, seq_lens, block_tables=tables, impl=impl)
