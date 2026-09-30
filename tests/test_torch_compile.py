"""The ops trace under torch.compile with fullgraph=True, i.e. without graph breaks."""

import pytest
import torch

import flash_lab

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize(
    "impl,dtype",
    [
        ("naive", torch.float32),
        ("fp32_fused", torch.float32),
        ("fp32_regtile", torch.float32),
        ("mma", torch.bfloat16),
        ("mma_pipelined", torch.bfloat16),
        ("triton", torch.bfloat16),
    ],
    ids=lambda x: str(x).replace("torch.", ""),
)
def test_attention_compiles_without_graph_breaks(impl, dtype):
    torch._dynamo.reset()

    def block(q, k, v):
        out, lse = flash_lab.attention(q, k, v, causal=True, impl=impl, return_lse=True)
        return out * 2 + 1, lse.exp()

    q = torch.randn(2, 96, 4, 64, device="cuda", dtype=dtype)
    k = torch.randn(2, 96, 2, 64, device="cuda", dtype=dtype)
    v = torch.randn(2, 96, 2, 64, device="cuda", dtype=dtype)
    compiled = torch.compile(block, fullgraph=True)
    for got, want in zip(compiled(q, k, v), block(q, k, v), strict=True):
        torch.testing.assert_close(got, want)


@pytest.mark.parametrize("impl", ["decode_copy", "decode_inplace", "splitkv"])
def test_decode_compiles_without_graph_breaks(impl):
    torch._dynamo.reset()

    def step(q, k_cache, v_cache, seq_lens):
        return flash_lab.decode(q, k_cache, v_cache, seq_lens, impl=impl) * 2

    q = torch.randn(3, 1, 8, 128, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.randn(3, 500, 2, 128, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn(3, 500, 2, 128, device="cuda", dtype=torch.bfloat16)
    seq_lens = torch.tensor([500, 7, 260], device="cuda", dtype=torch.int32)
    compiled = torch.compile(step, fullgraph=True)
    torch.testing.assert_close(
        compiled(q, k_cache, v_cache, seq_lens), step(q, k_cache, v_cache, seq_lens)
    )
