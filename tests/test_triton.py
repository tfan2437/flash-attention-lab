"""The autotuned path of the Triton kernel (the other tests pin one configuration)."""

import pytest
import torch

import flash_lab
from flash_lab import triton_attention
from tests.test_attention_fwd import check_output, make_inputs

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


@pytest.mark.parametrize(
    "shape,causal",
    [((2, 1024, 1024, 8, 2, 128), True), ((2, 2048, 2048, 8, 8, 64), False)],
    ids=["d128-causal", "d64-full"],
)
def test_autotuned_kernel(shape, causal, monkeypatch):
    monkeypatch.setenv("FLASH_LAB_TRITON_AUTOTUNE", "1")
    q, k, v = make_inputs(shape, torch.bfloat16)
    out, lse = flash_lab.attention(q, k, v, causal=causal, impl="triton", return_lse=True)
    check_output(out, lse, q, k, v, causal)
    choice = triton_attention.autotune_choice(shape[2], shape[5], causal)
    assert choice is not None and {"BLOCK_M", "BLOCK_N", "num_warps", "num_stages"} <= set(choice)
