"""Named benchmark configurations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PrefillConfig:
    batch: int
    heads: int
    heads_kv: int
    seqlen: int  # S_q == S_k
    head_dim: int


@dataclass(frozen=True)
class DecodeConfig:
    batch: int
    heads: int
    heads_kv: int
    context: int  # keys per sequence, all sequences the same length
    head_dim: int


@dataclass(frozen=True)
class Suite:
    kind: str  # "prefill" or "decode"
    configs: tuple
    dtypes: tuple[str, ...]
    impls: tuple[str, ...]
    causal: tuple[bool, ...] = (False, True)
    flush_l2: bool = False  # decode is memory bound and cache sensitive, prefill is not
    n_iters: int = 50


P = PrefillConfig
D = DecodeConfig

PREFILL_BF16_IMPLS = (
    "mma",
    "mma_pipelined",
    "triton",
    "sdpa_flash",
    "sdpa_cudnn",
    "sdpa_efficient",
    "flash_attn",
)
DECODE_IMPLS = (
    "decode_copy",
    "decode_inplace",
    "splitkv",
    "flash_attn",
    "sdpa_flash",
    "sdpa_efficient",
)

SUITES = {
    "prefill_fp32": Suite(
        kind="prefill",
        configs=(P(4, 32, 32, 1024, 128), P(4, 32, 32, 4096, 128), P(2, 16, 16, 2048, 64)),
        dtypes=("fp32",),
        impls=("fp32_fused", "sdpa_math", "sdpa_efficient"),
    ),
    "prefill_bf16": Suite(
        kind="prefill",
        configs=(
            P(4, 32, 32, 1024, 128),
            P(4, 32, 32, 4096, 128),
            P(2, 32, 32, 8192, 128),
            P(1, 32, 32, 16384, 128),
            P(8, 16, 16, 2048, 64),
            P(4, 32, 8, 4096, 128),  # GQA
        ),
        dtypes=("bf16",),
        impls=PREFILL_BF16_IMPLS,
    ),
    "decode_ctx": Suite(
        kind="decode",
        configs=tuple(
            D(batch, 32, heads_kv, context, 128)
            for batch in (1, 8)
            for heads_kv in (32, 8)
            for context in (512, 2048, 8192, 32768)
        ),
        dtypes=("bf16",),
        impls=DECODE_IMPLS,
        causal=(False,),
        flush_l2=True,
        n_iters=200,
    ),
    "decode_batch": Suite(
        kind="decode",
        configs=tuple(D(batch, 32, 8, 2048, 128) for batch in (1, 4, 16, 64)),
        dtypes=("bf16",),
        impls=DECODE_IMPLS,
        causal=(False,),
        flush_l2=True,
        n_iters=200,
    ),
}
