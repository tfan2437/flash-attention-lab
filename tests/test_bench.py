"""CPU checks of the benchmark harness: FLOP models, statistics, naming, and suite definitions."""

import pytest

from bench import baselines, flops, suites
from bench.run import gpu_slug, summarize


def test_attention_flops_counts_two_matmuls_and_halves_causal():
    full = flops.attention_flops(2, 4, 128, 256, 64, causal=False)
    assert full == 4 * 2 * 4 * 128 * 256 * 64
    assert flops.attention_flops(2, 4, 128, 256, 64, causal=True) == full / 2


def test_decode_bytes_reads_the_kv_cache_once():
    # bf16, one sequence, 8 KV heads, 1000 keys, D=128: 2 * 8 * 1000 * 128 * 2 bytes of K and V
    nbytes = flops.decode_bytes(1, 32, 8, 1000, 128, elem=2)
    assert nbytes == 2 * 8 * 1000 * 128 * 2 + 2 * 32 * 128 * 2


def test_summarize():
    stats = summarize([5.0, 1.0, 3.0, 2.0, 4.0])
    assert stats == {"median": 3.0, "p10": 1.0, "p90": 5.0, "min": 1.0}


def test_gpu_slug():
    assert gpu_slug("NVIDIA H100 80GB HBM3") == "h100-80gb-hbm3"
    assert gpu_slug("NVIDIA A100-SXM4-80GB") == "a100-sxm4-80gb"


@pytest.mark.parametrize("name,suite", sorted(suites.SUITES.items()))
def test_suites_are_well_formed(name, suite):
    assert suite.kind in ("prefill", "decode")
    assert suite.configs and suite.impls and suite.dtypes
    for cfg in suite.configs:
        assert cfg.heads % cfg.heads_kv == 0
        assert cfg.head_dim in (64, 128)
    known = set(baselines.SDPA_BACKENDS) | {"flash_attn", "fp32_fused", "mma", "mma_pipelined"}
    known |= {"triton", "naive", "fp32_regtile", "decode_copy", "decode_inplace"}
    known |= {"splitkv"}
    assert set(suite.impls) <= known
