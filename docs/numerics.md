# Numerics

How the kernels round, what the tests accept as correct, and the errors measured on an H100.
Every number comes from a committed result file, named next to its table, or from a constant in
the code.

## Definitions

### The fp64 reference

[flash_lab/reference.py](../flash_lab/reference.py) upcasts Q, K, and V to fp64, materializes the
scaled score matrix, and returns the output and the natural-log LSE of every score row. Causal
masks are aligned to the bottom-right corner, as in FlashAttention-2: query `i` sees key `j` iff
`j <= i + (S_k - S_q)`. A row that sees no key is defined as zeros with an LSE of `-inf`.
`decode()` applies the same computation to each sequence's first `seq_lens[b]` cached keys.
[tests/test_reference.py](../tests/test_reference.py) checks the reference at `atol=1e-12`
against PyTorch SDPA in fp64, against closed-form cases, and against `attention_tiled`, an fp64
version of the FlashAttention-2 tiling and online softmax.

### Error metric

The error of an output is its largest absolute difference from the reference, taken in fp64:
`(out.double() - ref).abs().max()`. The LSE error is the same quantity over the rows whose
reference LSE is finite. [bench/run.py](../bench/run.py) checks prefill outputs on a sample of
query rows: all rows when `S <= 256`, otherwise 256 rows that always include the first and the
last (`sample_rows`). Decode outputs are checked in full.

Absolute error fits what a correct output is. For a row with at least one visible key,
`o_i = sum_j p_ij v_j` with `p_ij >= 0` and `sum_j p_ij = 1`: a convex combination of V rows, so
`|o_i[d]| <= max|V|` whatever the scores are. Outputs can also be arbitrarily close to zero, which
rules out elementwise relative error. The prefill tests' absolute guard scales with `max|V|`.

### Tolerance rule

The prefill and decode tests follow the flash-attn test suite: an implementation's error may be
at most twice the error of PyTorch's own computation at the same dtype on the same inputs, plus a
small floor (`check_output` in [tests/test_attention_fwd.py](../tests/test_attention_fwd.py),
`check` in [tests/test_decode.py](../tests/test_decode.py)).

```
err     = max |o - o_ref64|        err_torch     = max |o_torch - o_ref64|
lse_err = max |lse - lse_ref64|    lse_err_torch = max |lse_torch - lse_ref64|    (finite rows)

pass if   err <= 2 * err_torch + FLOOR[dtype], and every output is finite
          err <= GUARD[dtype] * (1 + max|v|)                              (prefill tests only)
          lse_err <= 2 * lse_err_torch + LSE_FLOOR[dtype]
          lse is -inf exactly where lse_ref64 is                          (prefill tests only)
```

| dtype | `FLOOR` | `GUARD` | `LSE_FLOOR` |
|---|---|---|---|
| fp32 | 1e-6 | 2e-5 | 1e-5 |
| bf16 | 1e-3 | 2e-2 | 1e-3 |
| fp16 | 1e-3 | 1e-2 | 1e-3 |

| Check | `o_torch` | `lse_torch` |
|---|---|---|
| Prefill tests, `check_output` | SDPA math backend, bottom-right mask, KV heads repeated | `reference.attention` at the input dtype |
| Decode tests, `check` | `reference.decode` at the input dtype | the same call |
| Benchmark prefill, `check_prefill` | SDPA math backend on the sampled rows | not checked |
| Benchmark decode, `check_decode` | `reference.decode` at the input dtype | not checked |

At the input dtype, the reference keeps every intermediate tensor (scores, LSE, probabilities,
output) in that dtype. SDPA's math backend returns NaN for rows without visible keys; the prefill
checks replace those with 0, the defined value. The benchmark applies the output rule with the
same `FLOOR` and records a failing implementation as `"incorrect"` without timing it. The rule is
relative because the achievable error depends on the inputs (score magnitudes, how concentrated
each row's weights are, how many keys a row sees), and the baseline sees the same inputs: the
factor of 2 compares implementations, not test cases. The floor covers inputs on which the
baseline is nearly exact; the guard catches a kernel and a baseline that are both badly wrong.
The LSE gets the same relative form because its error grows with the size of the scores.

## fp32 kernels

`naive` ([csrc/prefill/naive.cu](../csrc/prefill/naive.cu)), `fp32_fused`
([csrc/prefill/fp32_fused.cu](../csrc/prefill/fp32_fused.cu)), and `fp32_regtile`
([csrc/prefill/fp32_regtile.cu](../csrc/prefill/fp32_regtile.cu)) compute in fp32 on CUDA cores,
so TF32 never applies to them; [tests/conftest.py](../tests/conftest.py) and the benchmarks turn
TF32 off for PyTorch's fp32 baselines. The build has no `--use_fast_math` ([setup.py](../setup.py)),
so `expf`, `exp2f`, and `logf` are the standard CUDA functions, not the approximate intrinsics.

### Accumulation order

| Kernel | Scores | Row sum `l` | `P V` | Normalization |
|---|---|---|---|---|
| `naive` | one accumulator per score, `d` in order, then `* scale` | whole row: strided per-thread sums, a warp butterfly, the 8 warp sums in order | one accumulator per output, keys in order | `P = exp(s - m) * (1 / l)` before `P V` |
| `fp32_fused` | one accumulator per score, `d` in order, then `* scale` | one thread per row, keys in order (`expf`) | one thread per output: rescale, then the tile's keys in order | `o / l` at the end |
| `fp32_regtile` | one accumulator per score, `d` in order, then `* scale_log2` | per lane over its 8 columns of each tile (`exp2f`), 8 lanes combined by a butterfly at the end | per lane: rescale once per tile, then the tile's keys in order | `o * (1 / l)` at the end |

`fp32_fused` takes 8 keys per tile by default (16 or 32 with `FLASH_LAB_FP32_TILE`),
`fp32_regtile` 64. Each score and each output element is accumulated by one thread; the only
cross-thread steps are fixed shuffle butterflies and the shared-memory step of the `naive` softmax.

### Error growth with S and D

A score is a length-D dot product; the worst-case rounding bound of a recursive sum grows linearly
with the number of terms, so the score error bound grows with D. Score errors do not pile up
across keys: to first order, errors `e_j` in a row's scaled scores change its output by
`sum_j p_j e_j (v_j - o)`, at most `2 * max_j |e_j| * max|V|` however many keys the row sees. S
enters through the sums over keys, `l` and `sum_j exp(s_j - m) v_j`, whose worst-case bounds grow
linearly with the number of visible keys. Measured errors stay far below these bounds: each fp32
kernel's max error at S = 4096 is lower than at S = 1024 (see the table below).

### Why fp64 and not another fp32 implementation

fp32 addition is not associative: the same terms summed in a different order round differently.
Two correct fp32 implementations with different reduction orders (tile sizes, thread mappings, a
tree instead of a running sum) disagree in the last bits, so a near-bitwise comparison between
them measures whether they sum in the same order, not how accurate either is. A more accurate
summation can even land further from a sequential result. Comparing with fp64 measures each
implementation's own error, and PyTorch's error at the same dtype gives the scale.

### Port fidelity

[tests/test_port_fidelity.py](../tests/test_port_fidelity.py) runs the fp32 kernels between
`torch.nn.MultiheadAttention`'s own projections (`in_proj_weight` and bias, then `out_proj`) and
requires the result to match the module's forward at `atol=1e-6, rtol=1e-5`, the tolerance the
course-era versions of these kernels were held to. Shapes are (batch, seqlen, hidden) =
(4, 10, 512) and (16, 100, 1024) with 8 heads (head dims 64 and 128), causal and not. This check
compares two fp32 computations, so it uses a fixed tolerance instead of an error ratio.

## bf16 and fp16 kernels

### Matmuls

`mma` and `mma_pipelined` use `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` (`f16` for
fp16): 16-bit operands, fp32 accumulators ([csrc/common/ptx.cuh](../csrc/common/ptx.cuh)). A
product of two 16-bit values is exact in fp32, barring overflow and underflow (8 or 11
significant bits each, against 24), so rounding comes from the accumulation, whose order inside
one instruction is up to the hardware; `tile_scores` adds the `D / 16` k-steps in order. Triton's
`tl.dot` also returns fp32, and its `P V` accumulates into an fp32 `acc`. The decode kernels use
no tensor cores: they convert loaded elements to fp32 and do all arithmetic in fp32, `P V`
included, so their only 16-bit rounding is the final store.

### Softmax in fp32, P in 16 bits

The online softmax runs on the fp32 accumulators, once per 64-key tile (`online_softmax` in
[csrc/prefill/mma_tile.cuh](../csrc/prefill/mma_tile.cuh)):

1. The tile max of the raw scores updates the running max `m`, and `ref = m * scale_log2`, where
   `scale_log2 = softmax_scale * log2(e)` is computed on the host in double and rounded to fp32
   (`make_attn_params` in [csrc/common/attn_host.h](../csrc/common/attn_host.h)).
2. `p = exp2f(fmaf(s, scale_log2, -ref))`: one FFMA, rounded once, and one exp2. Mathematically
   `p = exp(softmax_scale * (s - m))`.
3. `l = rescale * l + sum(p)` on the fp32 `p`, and the output accumulator is multiplied by the
   same `rescale`, which is `exp2f(fmaf(m_old, scale_log2, -ref))`.
4. `accumulate_pv` rounds `p` to 16 bits (`pack2`: `__floats2bfloat162_rn` or
   `__floats2half2_rn`). The packed probabilities of two adjacent 8-key score tiles are exactly
   one A fragment of the `P V` mma, so P never leaves registers.

So `l` sums unrounded probabilities while `P V` uses rounded ones. Triton does the same:
`l_i * alpha + tl.sum(p, 1)` on the fp32 `p`, then `tl.dot(p.to(v.dtype), v, acc)`. Rounding
moves each probability by at most 2^-8 of its value in bf16 and 2^-11 in fp16, so the weights
applied to V sum to 1 only within that relative amount. In fp16, probabilities below 2^-14 are
subnormal and those below 2^-25 round to zero; each multiplies one V row, so the loss is at most
the weight times `max|V|`.

Subtracting the running max is what keeps these values in range. fp16's largest finite value is
65504, so `exp(x)` overflows fp16 for `x` above about 11.1 and fp32 above about 88.7;
`test_large_scores` multiplies Q and K by 10, giving scores of order 100. With the max
subtracted, every exponent argument is at most 0 (up to the rounding in `fmaf`), so every
probability lies in [0, 1] before it is packed and `l` is at most the number of keys seen. A
later, larger max only shrinks the earlier terms, since `rescale <= 1`.

### Output and LSE

In `mma` and `mma_pipelined` the four lanes of a quad combine their partial row sums once
(`finish_rows`), and `o_acc * (1 / l)` is rounded once to the output dtype; Triton computes
`acc / l` and casts. Storing the output in bf16 costs up to 2^-8 of `|o|` on its own, for every
implementation that returns bf16, the baseline included.

`store_lse` writes `lse = m * softmax_scale + log(l)`: `m` is the raw maximum, so the product is
the maximum scaled score, and `l` sums `exp(softmax_scale * (s - m))`. Kernels that keep the max
in the log2 domain convert with ln 2 instead: Triton (`m_i * 0.6931471805599453 + tl.log(l_i)`),
`fp32_regtile`, `decode_inplace`, and `splitkv` (`m * M_LN2 + logf(l)`). `naive`, `fp32_fused`,
and `decode_copy` work in natural units (`m + logf(l)`). Every kernel returns an fp32
natural-log LSE.

### `softmax_scale` must be positive

The tensor-core kernels take the max on raw scores, which relies on
`max(scale * s) = scale * max(s)` for `scale > 0`. With a negative scale the max would pick the
wrong key, exponent arguments would be positive, and a masked `-inf` would become `+inf`.
`flash_lab.attention` rejects a scale that is not positive (`if not scale > 0` in
[flash_lab/ops.py](../flash_lab/ops.py), which also rejects NaN) for every prefill
implementation. All other kernels scale before taking the max; `flash_lab.decode` has no check.

## Masking

### Masked scores and rows without keys

Masked keys get a score of `-inf`: keys at or past `S_k` in a partial tile and, with
`causal=True`, keys `j > i + (S_k - S_q)`; `exp2(-inf) = 0` removes them from `l` and `P V`. With
`S_q > S_k`, bottom-right alignment leaves the first `S_q - S_k` rows without a visible key. Their
output is 0 and their LSE `-inf`, and the tests require `-inf` in exactly the reference's
positions. The test shape (B, S_q, S_k, H, H_kv, D) = (1, 101, 37, 4, 4, 128) covers this for
every prefill implementation.

### The `ref = 0` rule

A row whose running max is still `-inf` would compute `exp2(-inf - (-inf))`, which is NaN, as is
`fmaf(-inf, scale_log2, +inf)`. Every online-softmax loop uses 0 as the reference point while the
max is `-inf` (`fp32_fused.cu`, `fp32_regtile.cu`, `mma_tile.cuh`, Triton's `_tile_loop`,
`decode.cu`, and the fp64 `attention_tiled`). A masked score then gives `exp2(-inf - 0) = 0`
exactly, the rescale factor is 0, and `l` and the accumulator stay exactly 0. At the end `l = 0`
marks the row: the kernels write 0 without dividing by `l` and write `-inf` as the LSE. `naive`
checks the row max directly; `decode_copy` needs no guard, since every decode row has a key.

### Zero-filled tails

`P V` multiplies every V row of a tile by its probability, masked rows included. Since `0 * NaN`
and `0 * inf` are NaN, a stale or uninitialized shared-memory row with such a bit pattern would
turn the output row into NaN. The kernels never read past the end of Q, K, or V and store zeros
there instead: directly in `naive`'s GEMM tiles, `fp32_fused`, `fp32_regtile`, `mma`, and
`decode_copy`; in `mma_pipelined` through `cp.async` with a source size of 0, which reads nothing
and zero-fills (`cp_async_16` in `ptx.cuh`); in Triton through `other=0.0` on the masked loads,
while the unmasked loop covers only whole tiles inside the sequence. `decode.cu` zeroes the K
registers of a key past the block's range, scores it `-inf`, and skips its V row. With the
`ref = 0` rule, a masked key then adds an exact 0 to every accumulator. compute-sanitizer's
initcheck (uninitialized global memory) and racecheck (shared-memory hazards) report no errors on
the 27 cases of [tests/test_sanitize.py](../tests/test_sanitize.py), which cover ragged shapes and
causal masks for every CUDA kernel
([profiling/sanitize/97dbb68.txt](../profiling/sanitize/97dbb68.txt)); Triton is not in that run.

## Split-KV decode

`decode_inplace` and `splitkv` are one kernel, [csrc/decode/decode.cu](../csrc/decode/decode.cu),
with one split or several. With `num_splits = n`, a sequence's keys are cut into at most `n`
contiguous ranges at multiples of 32 keys; ranges that start past `seq_lens[b]` are empty. Q is
multiplied by `scale_log2` as it is loaded, so scores are in the log2 domain. A block's 4 warps
take its range's chunks round-robin, each with its own fp32 max, sum, and output, and the block
combines them in warp order 0 to 3, skipping warps that saw no key:

```
m = max_w m_w        l = sum_w l_w * 2^(m_w - m)        o = sum_w o_w * 2^(m_w - m)
```

With one split the block writes `o / l` in the output dtype and `lse = m * ln 2 + log(l)`. With
more, it writes the unnormalized `o` and `(m, l)` to fp32 workspaces, and `merge_kernel` applies
the same formulas across splits in order 0 to n - 1, one thread per output column, skipping empty
splits. The skip matters in the warp combine: in an empty split all four warps have `m_w = -inf`,
so `2^(m_w - m)` would be NaN. The output is rounded once, after the merge.

With no atomics and fixed orders, the result is bitwise reproducible for a given `n`. A different
`n` moves the range boundaries, changes which keys share a running max, and reorders the final
sums, so it changes the fp32 rounding. With `num_splits=None`, `choose_num_splits` doubles `n`,
starting from 1, while `n < 64`, the grid has fewer than 16 blocks per SM, and the doubled `n`
leaves at least 256 keys of the cache length `S_max` per split. The grid size follows B, H, and
the heads per block (set by the GQA group and D), so the default `n` depends on the GPU's SM
count, B, H, H_kv, D, and `S_max`, not on the sequence lengths.

`splitkv` with `num_splits=1` takes exactly the `decode_inplace` path. In the decode benchmark
files below, `splitkv` ran with its default count; the files do not record it, but by the
heuristic's rule, with the 132 SMs recorded in the JSON, it is 2 to 64 splits across those
configurations. There `splitkv` reported the same max error as `decode_inplace`, to all digits,
in every configuration. `test_splitkv_split_counts` checks `n` in {1, 2, 8, 64, default}; with
64 splits its 40-key sequence leaves most splits empty.

## Measured errors

The benchmark checks each implementation before timing it and records `max_err_vs_ref64` and
`max_err_torch_vs_ref64`. Inputs are `randn` with seed 0 and `softmax_scale = 1 / sqrt(D)`; the
baseline is the SDPA math backend for prefill and `reference.decode` for decode, both at the
input dtype. Prefill errors are over the 256 sampled rows, decode errors over all outputs with
every sequence at the full context. Ratio is `max_err_vs_ref64 / max_err_torch_vs_ref64`; the
check allows 2 plus the floor divided by the baseline error. External rows are baselines run by
the same harness. The tables come from the final measurement run at commit cc64b18: three runs
per suite, tagged `final1` to `final3`. The check runs once per run, before timing, on
identically seeded inputs, and the three runs report identical error fields for every result of
all four kernel suites. Each table takes one configuration and links the `final1` file.

### fp32 prefill

Source: [20261003-070824-cc64b18-prefill_fp32-final1.json](../bench/results/h100-80gb-hbm3/20261003-070824-cc64b18-prefill_fp32-final1.json);
the `final2` and `final3` files report the same errors.

| Implementation | dtype | Config (B, H, H_kv, S, D) | Causal | `max_err_vs_ref64` | `max_err_torch_vs_ref64` | Ratio |
|---|---|---|---|---|---|---|
| `naive` | fp32 | (4, 32, 32, 4096, 128) | no | 4.49e-7 | 4.87e-7 | 0.922 |
| `fp32_fused` | fp32 | (4, 32, 32, 4096, 128) | no | 4.97e-7 | 4.87e-7 | 1.02 |
| `fp32_regtile` | fp32 | (4, 32, 32, 4096, 128) | no | 5.11e-7 | 4.87e-7 | 1.05 |
| `naive` | fp32 | (4, 32, 32, 4096, 128) | yes | 1.08e-6 | 1.36e-6 | 0.792 |
| `fp32_fused` | fp32 | (4, 32, 32, 4096, 128) | yes | 1.25e-6 | 1.36e-6 | 0.917 |
| `fp32_regtile` | fp32 | (4, 32, 32, 4096, 128) | yes | 1.31e-6 | 1.36e-6 | 0.961 |

Across the file's six (configuration, mask) cases, the ratio of the three fp32 kernels ranges
from 0.531 to 1.38. Each kernel's max error at S = 4096 is lower than at S = 1024, with and
without the mask (`fp32_fused` without it: 6.88e-7 at S = 1024). In every configuration of the
file the causal maxima are larger, for each implementation and the baseline alike; the relative
rule absorbs that.

### bf16 prefill

Source: [20261003-070947-cc64b18-prefill_bf16-final1.json](../bench/results/h100-80gb-hbm3/20261003-070947-cc64b18-prefill_bf16-final1.json);
the `final2` and `final3` files report the same errors.

| Implementation | dtype | Config (B, H, H_kv, S, D) | Causal | `max_err_vs_ref64` | `max_err_torch_vs_ref64` | Ratio |
|---|---|---|---|---|---|---|
| `mma` | bf16 | (4, 32, 32, 4096, 128) | no | 5.28e-4 | 4.81e-4 | 1.10 |
| `mma_pipelined` | bf16 | (4, 32, 32, 4096, 128) | no | 5.28e-4 | 4.81e-4 | 1.10 |
| `triton` | bf16 | (4, 32, 32, 4096, 128) | no | 5.28e-4 | 4.81e-4 | 1.10 |
| `flash_attn` (external) | bf16 | (4, 32, 32, 4096, 128) | no | 5.53e-4 | 4.81e-4 | 1.15 |
| `mma` | bf16 | (4, 32, 32, 4096, 128) | yes | 3.21e-3 | 3.21e-3 | 1.00 |
| `mma_pipelined` | bf16 | (4, 32, 32, 4096, 128) | yes | 3.21e-3 | 3.21e-3 | 1.00 |
| `triton` | bf16 | (4, 32, 32, 4096, 128) | yes | 3.21e-3 | 3.21e-3 | 1.00 |
| `flash_attn` (external) | bf16 | (4, 32, 32, 4096, 128) | yes | 3.21e-3 | 3.21e-3 | 1.00 |

Across the file's 12 cases, the ratio of the three bf16 kernels ranges from 1.00 to 1.50, the
largest at (8, 16, 16, 2048, 64) without the mask. `mma`, `mma_pipelined`, and `triton` report
the same max error, to all digits, in all 12 cases, and with the causal mask above every
implementation in the file reports the baseline's value. Equal errors mean the implementations
returned the same bf16 value at the worst element: the differences between their internal results
were smaller than the bf16 rounding there, so they do not show.

Triton was autotuned in each run, and the `triton_config` field of every Triton result records
the choice as (BLOCK_M, BLOCK_N, num_warps, num_stages). For (4, 32, 32, 4096, 128) it was
(128, 128, 8, 3) in all three runs, with and without the mask. At (2, 32, 32, 8192, 128) and
(1, 32, 32, 16384, 128), with and without the mask, two runs picked (128, 128, 8, 3) and one
picked (128, 64, 8, 3); at (8, 16, 16, 2048, 64) with the mask, two picked (128, 64, 8, 3) and
one (64, 64, 4, 3). The other cases kept one choice, and the max errors are the same in all
three runs, including the cases where the choice changed.

### Decode

Source: [20261003-071018-cc64b18-decode_ctx-final1.json](../bench/results/h100-80gb-hbm3/20261003-071018-cc64b18-decode_ctx-final1.json);
the `final2` and `final3` files report the same errors.

| Implementation | dtype | Config (B, H, H_kv, context, D) | `max_err_vs_ref64` | `max_err_torch_vs_ref64` | Ratio |
|---|---|---|---|---|---|
| `decode_copy` | bf16 | (1, 32, 32, 32768, 128) | 1.07e-4 | 1.09e-3 | 0.0978 |
| `decode_inplace` | bf16 | (1, 32, 32, 32768, 128) | 1.07e-4 | 1.09e-3 | 0.0978 |
| `splitkv` | bf16 | (1, 32, 32, 32768, 128) | 1.07e-4 | 1.09e-3 | 0.0978 |
| `flash_attn` (external) | bf16 | (1, 32, 32, 32768, 128) | 1.37e-4 | 1.09e-3 | 0.126 |
| `decode_copy` | bf16 | (8, 32, 8, 8192, 128) | 2.41e-4 | 3.31e-3 | 0.0727 |
| `decode_inplace` | bf16 | (8, 32, 8, 8192, 128) | 2.41e-4 | 3.31e-3 | 0.0727 |
| `splitkv` | bf16 | (8, 32, 8, 8192, 128) | 2.41e-4 | 3.31e-3 | 0.0727 |
| `flash_attn` (external) | bf16 | (8, 32, 8, 8192, 128) | 2.69e-4 | 3.31e-3 | 0.0812 |

The JSON does not record split counts; by the heuristic's rule, with the 132 SMs recorded in the
JSON, `splitkv` uses 64 splits on the first configuration and 32 on the second. The decode
baseline holds every intermediate tensor in bf16, while the decode kernels compute in fp32 and
round once. Across this file and
[20261003-071028-cc64b18-decode_batch-final1.json](../bench/results/h100-80gb-hbm3/20261003-071028-cc64b18-decode_batch-final1.json)
(whose `final2` and `final3` runs also agree), the ratio of the three decode kernels ranges from
0.0580 to 0.147, so the factor of 2 is far from binding for decode, and the three report the same
max error in every configuration of both files.

## End to end

### Llama-3.1-8B benchmark

[bench/e2e.py](../bench/e2e.py) checks each attention implementation inside Llama-3.1-8B-Instruct
(bf16) before timing it: its logits at the last prompt position, computed through a static cache
as `generate()` does, against an fp32 copy of the model on SDPA with TF32 off. Prompts are random
token ids below 128000 (seed 0), 512 and 8192 tokens long, batch 1. The rule is the kernel rule
with the bf16 floor of 1e-3; the baseline is the bf16 model on SDPA without a cache. These logits
come from the prompt pass, so the check covers the prefill kernels (Triton for `flash_lab`,
`mma_pipelined` for `flash_lab_cuda`); the decode kernels show up in the comparison of 256 greedy
tokens with those of `sdpa`, the first implementation in the run.

Source: [20261003-072354-cc64b18-e2e_llama-final1.json](../bench/results/h100-80gb-hbm3/20261003-072354-cc64b18-e2e_llama-final1.json)
(model revision 0e9e39f; torch 2.8.0+cu126, transformers 4.57.6, flash-attn 2.8.3.post1). Each
value is the maximum over the three runs, `final1` to `final3`, which agree exactly on every value
in the table.

| Prompt | Implementation | `max_logit_err_vs_fp32` | `max_logit_err_torch_vs_fp32` | Ratio | Status | First differing token |
|---|---|---|---|---|---|---|
| 512 | `sdpa` | 0.0645 | 0.0579 | 1.11 | ok | reference |
| 512 | `flash_attention_2` | 13.2 | 0.0579 | 229 | incorrect | not run |
| 512 | `flash_lab` | 0.0644 | 0.0579 | 1.11 | ok | none |
| 512 | `flash_lab_cuda` | 0.0672 | 0.0579 | 1.16 | ok | none |
| 8192 | `sdpa` | 0.0943 | 0.103 | 0.913 | ok | reference |
| 8192 | `flash_attention_2` | 2.62 | 0.103 | 25.3 | incorrect | not run |
| 8192 | `flash_lab` | 0.0738 | 0.103 | 0.715 | ok | none |
| 8192 | `flash_lab_cuda` | 0.0991 | 0.103 | 0.959 | ok | 0 |

The `sdpa` rows use the static-cache path like the others, hence their difference from the
no-cache baseline. `flash_lab` produces SDPA's 256 tokens on both prompts, and `flash_lab_cuda`
on the 512-token prompt. On the 8192-token prompt `flash_lab_cuda` differs from SDPA at the first
new token in all three runs; that token is chosen from the logits at the last prompt position.
There the fp32 model's top two logits differ by 0.0446 (`fp32_top2_margin`; 0.253 on the
512-token prompt), less than the max logit error of SDPA (0.0943 with the cache, 0.103 without)
and of both `flash_lab` variants (0.0738, 0.0991), so any of them may rank either token first.
The logit check, not token equality, is the correctness criterion.

transformers' `flash_attention_2` implementation, run through the same static-cache path, failed
the check in all three runs, with max logit errors of 13.2 and 2.62 against limits of 0.117 and
0.208, and was not timed. The flash-attn kernels that the harness calls directly pass the kernel
checks above.

The longer prompts of the context sweep
([20261003-081403-13a5521-e2e_llama-context.json](../bench/results/h100-80gb-hbm3/20261003-081403-13a5521-e2e_llama-context.json),
2K to 32K tokens) are logit-checked only up to 8192 tokens (`--check-max-len`). In fp32 with
grouped KV heads, transformers' SDPA path runs PyTorch's math backend, whose score matrix for one
sequence is 32 GiB at 16K tokens and does not fit next to the 32 GB of fp32 weights. At those
lengths the kernels' own checks against fp64 cover correctness: `prefill_seqlen` up to 16K keys
and `decode_long` up to 128K, all passing. The batch-8 run
([20261003-081816-13a5521-e2e_llama-batch8.json](../bench/results/h100-80gb-hbm3/20261003-081816-13a5521-e2e_llama-batch8.json))
computes the fp32 reference one sequence at a time and checks all eight; every implementation
passes, with max logit errors from 0.096 to 0.164 against SDPA's 0.158 to 0.164 without a cache.

### Llama-3.2-1B tests

[tests/test_e2e_llama.py](../tests/test_e2e_llama.py) runs Llama-3.2-1B-Instruct (head dim 64) in
bf16 with `attn_implementation="flash_lab"`. On two random 300-token sequences, the max logit
error of `flash_lab` against an fp32 copy of the model must be at most twice SDPA's, with no
floor. On five chat prompts with 64 greedy tokens each, the tokens must equal SDPA's up to the
first step at which SDPA's own top two logits are within two bf16 steps,
`2 * eps_bf16 * 2^floor(log2 |top logit|)`; later steps are not compared. The same check runs with
a static cache and the decode step compiled into one graph. The test's docstring gives the reason:
at such a near-tie any change in rounding can pick the other token, and on these prompts PyTorch's
eager attention leaves SDPA at the same near-ties.

## Determinism

No kernel uses atomic operations; there are none in `csrc/` or in the Triton kernel. Every
reduction has a fixed order (in-thread loops, shuffle butterflies with fixed partners,
shared-memory trees, the decode warp combine, the split merge), and which thread computes an
output element depends only on the shapes and launch parameters. Launch order does not enter:
`mma_pipelined` starts the heaviest causal query blocks first, which changes when a block runs,
not what it computes. A kernel run twice on the same inputs, GPU, and build therefore returns the
same bits. `test_deterministic` checks this with `torch.equal` for every implementation: prefill
at (B, S_q, S_k, H, H_kv, D) = (2, 300, 300, 8, 2, 128), causal, and decode at
(B, H, H_kv, S_max, D) = (4, 32, 8, 3000, 128) with sequence lengths 3000, 1, 2048, and 999. The
tests do not compare results across GPU models, CUDA versions, or Triton versions.

| Kernel | Bitwise reproducible | What changes the result |
|---|---|---|
| `naive`, `fp32_regtile` | yes | nothing |
| `fp32_fused` | yes | `FLASH_LAB_FP32_TILE`: 8, 16, or 32 keys per tile |
| `mma`, `mma_pipelined` | yes | nothing |
| `triton` | yes, for a fixed configuration | the configuration, autotuned per process unless pinned |
| `decode_copy`, `decode_inplace` | yes | nothing |
| `splitkv` | yes, for a fixed `num_splits` | `num_splits`; the default depends on the SM count, B, H, H_kv, D, `S_max` |

- Keys per tile set where the online softmax rescales and how key sums are grouped: for
  `fp32_fused` that is `FLASH_LAB_FP32_TILE`, for Triton `BLOCK_N`. Triton's autotuner times its
  8 candidate configurations the first time a key (length bucket, head dim, causal flag) is seen
  in a process and keeps the fastest, so another process can pick a different one when two time
  closely. `FLASH_LAB_TRITON_AUTOTUNE=0` (set by [tests/conftest.py](../tests/conftest.py)) or
  `FLASH_LAB_TRITON_CONFIG` pins one.
- `impl="auto"` takes the first implementation that is built and supports the input
  ([flash_lab/ops.py](../flash_lab/ops.py)): `triton` for bf16 and fp16 prefill, `fp32_regtile`
  for fp32, `splitkv` for decode. On a machine without Triton the same call runs a different
  kernel and returns different bits.
