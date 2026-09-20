"""The fp32 kernels inside torch.nn.MultiheadAttention's own projections, compared with the
module at the course's tolerance (atol 1e-6, rtol 1e-5): rebuilding the course-era kernels
outside the course must not change their results. Head dims are 64 and 128, the sizes these
kernels support, instead of the course's 16."""

import pytest
import torch
import torch.nn.functional as F

import flash_lab

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("batch,seqlen,hidden", [(4, 10, 512), (16, 100, 1024)])
@pytest.mark.parametrize("impl", ["naive", "fp32_fused", "fp32_regtile"])
def test_matches_multihead_attention(impl, batch, seqlen, hidden, causal):
    heads = 8
    mha = torch.nn.MultiheadAttention(hidden, heads, batch_first=True, device="cuda").eval()
    x = torch.randn(batch, seqlen, hidden, device="cuda")
    mask = None
    if causal:
        mask = torch.nn.Transformer.generate_square_subsequent_mask(seqlen, device="cuda")
    with torch.no_grad():
        expected, _ = mha(x, x, x, attn_mask=mask, need_weights=False)
        qkv = F.linear(x, mha.in_proj_weight, mha.in_proj_bias)
        q, k, v = qkv.view(batch, seqlen, 3, heads, hidden // heads).unbind(dim=2)
        attn = flash_lab.attention(q, k, v, causal=causal, impl=impl)
        out = F.linear(attn.reshape(batch, seqlen, hidden), mha.out_proj.weight, mha.out_proj.bias)
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-5)
