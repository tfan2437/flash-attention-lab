"""flash_lab.hf.attention_forward, the transformers attention function, on synthetic tensors in
transformers' [B, H, S, D] layout. The model-level tests are in test_e2e_llama.py."""

import pytest
import torch

import flash_lab
from flash_lab.hf import attention_forward


class Module(torch.nn.Module):
    is_causal = True


def hf_inputs(batch, seqlen_q, cache_len, heads, heads_kv, head_dim, device="cpu"):
    """query [B, H, S_q, D] and a [B, H_kv, cache_len, D] key/value buffer whose unused rows hold
    large values, so reading past the valid rows would show up in the output."""
    q = torch.randn(batch, heads, seqlen_q, head_dim, device=device, dtype=torch.bfloat16)
    k = 50 * torch.ones(batch, heads_kv, cache_len, head_dim, device=device, dtype=torch.bfloat16)
    v = 50 * torch.ones(batch, heads_kv, cache_len, head_dim, device=device, dtype=torch.bfloat16)
    return q, k, v


def test_rejects_what_it_cannot_run():
    q, k, v = hf_inputs(2, 4, 8, 4, 2, 64)
    positions = torch.arange(4).expand(2, 4)
    with pytest.raises(ValueError, match="dropout"):
        attention_forward(Module(), q, k, v, None, dropout=0.1, position_ids=positions)
    with pytest.raises(ValueError, match="position_ids"):
        attention_forward(Module(), q, k, v, None)
    with pytest.raises(NotImplementedError, match="sliding"):
        attention_forward(Module(), q, k, v, None, position_ids=positions, sliding_window=4)
    padded = torch.stack([torch.arange(4), torch.tensor([1, 1, 0, 1])])
    with pytest.raises(ValueError, match="unpadded"):
        attention_forward(Module(), q, k, v, None, position_ids=padded)
    with pytest.raises(ValueError, match="does not fit"):
        attention_forward(Module(), q, k, v, None, position_ids=positions + 5)


@pytest.mark.gpu
@pytest.mark.parametrize("prefill_impl", ["triton", "mma_pipelined"])
def test_prefill_reads_only_the_valid_cache_rows(prefill_impl):
    q, k, v = hf_inputs(2, 77, 200, 8, 2, 128, device="cuda")
    k[:, :, :77].normal_()
    v[:, :, :77].normal_()
    positions = torch.arange(77, device="cuda").expand(2, 77)
    out, weights = attention_forward(
        Module(), q, k, v, None, scaling=0.1, position_ids=positions, prefill_impl=prefill_impl
    )
    assert weights is None
    ref = flash_lab.reference.attention(
        q.transpose(1, 2),
        k[:, :, :77].transpose(1, 2),
        v[:, :, :77].transpose(1, 2),
        causal=True,
        softmax_scale=0.1,
    )
    assert out.shape == (2, 77, 8, 128)
    torch.testing.assert_close(out.double(), ref, atol=2e-2, rtol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("decode_impl", ["splitkv", "decode_inplace"])
def test_decode_step_uses_position_ids_as_lengths(decode_impl):
    q, k, v = hf_inputs(3, 1, 300, 8, 2, 128, device="cuda")
    k[:, :, :251].normal_()
    v[:, :, :251].normal_()
    # One position per sequence (generate's layout), or one shared row (a plain forward call).
    for positions in (torch.full((3, 1), 250, device="cuda"), torch.tensor([[250]], device="cuda")):
        out, _ = attention_forward(
            Module(), q, k, v, None, scaling=0.1, position_ids=positions, decode_impl=decode_impl
        )
        ref = flash_lab.reference.attention(
            q.transpose(1, 2),
            k[:, :, :251].transpose(1, 2),
            v[:, :, :251].transpose(1, 2),
            causal=True,
            softmax_scale=0.1,
        )
        assert out.shape == (3, 1, 8, 128)
        torch.testing.assert_close(out.double(), ref, atol=2e-2, rtol=0)
