# Performance report

Measured results on one H100, how they were taken, and what explains them. The numbers come from
three runs of every suite at commit cc64b18, summarized in
[summary-cc64b18.md](../bench/results/h100-80gb-hbm3/summary-cc64b18.md), and from single-run
sweeps at commits ef88a7a and 13a5521, whose kernels are those of cc64b18. Profiler evidence is in
[profiling/metrics.md](../profiling/metrics.md); how each kernel works is in [design.md](design.md);
errors are in [numerics.md](numerics.md).

## Method

- **Hardware and software.** NVIDIA H100 80GB HBM3 (SXM5, 132 SMs), driver 595.71.05, torch
  2.8.0+cu126, Triton 3.4.0, flash-attn 2.8.3.post1, CUDA 12.6; details in
  [environment.md](environment.md).
- **Harness.** [bench/run.py](../bench/run.py) checks each implementation's output against the
  fp64 reference first (at most twice PyTorch's error at the same dtype; a failure is recorded as
  `incorrect` and not timed), then runs 10 warmup calls and 50 timed calls (prefill) or 200
  (decode), each between two CUDA events, and reports the median. Decode zeroes a 256 MB buffer
  between calls so the KV cache does not stay in the 50 MiB L2.
- **What is timed.** The Python-level op call: argument checks, dispatch, the kernel launches, and
  any copies an implementation makes (`decode_copy`'s copies are its point). For large prefill
  shapes this is the kernel time; for short decode calls the host work is a visible share.
- **Units.** Prefill TFLOP/s count `4 * B * H * S_q * S_k * D` operations, half of that for causal
  attention (the FlashAttention-2 convention). Decode GB/s count K and V read once plus `q` and
  `o`. Percent of peak uses NVIDIA's H100 SXM datasheet: 989.4 TFLOP/s dense bf16, 67 TFLOP/s
  fp32, 3,350 GB/s.
- **Repetition.** [scripts/pace/final.sh](../scripts/pace/final.sh) runs the four suites three
  times, interleaved, from one clean commit. Tables give the median of the three per-run medians;
  ratios to flash-attn are taken within each run before the median, because flash-attn's timing
  varies more between runs than within one (see [History](#history)). Our kernels repeat within
  about 1% across runs.
- **Clocks.** PACE does not allow locking clocks. Every result records the SM clock sampled
  during its timed loop. The GPU stayed at its 1980 MHz maximum for the decode, fp32, and
  end-to-end runs (median), but the bf16 prefill runs drew close to the 700 W limit and the clock
  dropped while some implementations ran: at the headline shape the per-run medians were
  1,785-1,838 MHz for `mma_pipelined`, 1,845-1,875 for `triton`, and 1,695-1,762 for flash-attn,
  which runs last in each configuration. Throughput comparisons at equal clocks come from Nsight
  Compute, where every kernel ran at 1.81-1.82 GHz; they are given next to the benchmark ratios.

## Prefill, bf16

At `(B, H, H_kv, S, D) = (4, 32, 32, 4096, 128)`:

| implementation | non-causal TFLOP/s | x flash-attn | causal TFLOP/s | x flash-attn |
|---|---|---|---|---|
| `mma` (CUDA, `mma.sync`) | 104.5 | 0.28 | 95.7 | 0.29 |
| `mma_pipelined` (CUDA, + `cp.async`) | 229.2 | 0.62 | 216.7 | 0.67 |
| `triton` | 442.3 | 1.20 | 391.4 | 1.21 |
| flash-attn 2.8.3 | 369.3 | 1.00 | 325.4 | 1.00 |
| SDPA flash backend | 328.7 | 0.89 | 300.3 | 0.92 |
| SDPA cuDNN backend | 610.5 | 1.65 | 543.8 | 1.67 |
| SDPA memory-efficient backend | 181.4 | 0.49 | 167.2 | 0.51 |

At equal clocks under Nsight Compute the non-causal kernels take 10.61 ms (`mma`), 4.68 ms
(`mma_pipelined`), 2.36 ms (`triton`), and 2.76 ms (flash-attn): 0.26x, 0.59x, and 1.17x
flash-attn, against 0.28x, 0.62x, and 1.20x in the benchmark.

Every configuration of the suite, non-causal / causal TFLOP/s:

| (B, H, H_kv, S, D) | mma | mma_pipelined | triton | flash_attn | sdpa_flash | sdpa_cudnn |
|---|---|---|---|---|---|---|
| (4, 32, 32, 1024, 128) | 79.7 / 69.1 | 219.3 / 177.4 | 365.6 / 265.1 | 349.1 / 235.0 | 326.9 / 216.0 | 575.0 / 402.8 |
| (4, 32, 32, 4096, 128) | 104.5 / 95.7 | 229.2 / 216.7 | 442.3 / 391.4 | 369.3 / 325.4 | 328.7 / 300.3 | 610.5 / 543.8 |
| (2, 32, 32, 8192, 128) | 116.1 / 104.3 | 230.8 / 224.2 | 438.7 / 423.5 | 356.5 / 354.2 | 343.4 / 296.9 | 590.6 / 562.6 |
| (1, 32, 32, 16384, 128) | 116.5 / 118.1 | 228.1 / 231.6 | 443.1 / 440.2 | 368.1 / 346.0 | 332.8 / 318.6 | 614.9 / 590.3 |
| (8, 16, 16, 2048, 64) | 121.8 / 107.4 | 200.2 / 177.2 | 382.2 / 303.9 | 313.9 / 247.1 | 296.7 / 244.9 | 498.8 / 379.5 |
| (4, 32, 8, 4096, 128) | 124.4 / 117.5 | 227.4 / 212.5 | 444.3 / 400.9 | 369.2 / 335.1 | 340.7 / 306.6 | 581.1 / 606.7 |

Against sequence length at 16K tokens per call (`prefill_seqlen`, one run at ef88a7a), non-causal /
causal TFLOP/s:

| (B, S) | mma | mma_pipelined | triton | flash_attn | sdpa_flash | sdpa_cudnn |
|---|---|---|---|---|---|---|
| (32, 512) | 87.9 / 70.2 | 214.6 / 170.3 | 315.9 / 217.9 | 335.6 / 224.1 | 320.8 / 199.1 | 538.4 / 358.6 |
| (16, 1024) | 91.4 / 81.5 | 207.6 / 199.6 | 385.8 / 299.3 | 333.3 / 284.3 | 332.7 / 265.0 | 612.3 / 469.3 |
| (8, 2048) | 97.2 / 89.1 | 223.2 / 211.5 | 404.2 / 346.3 | 355.9 / 302.1 | 326.5 / 280.2 | 625.1 / 548.8 |
| (4, 4096) | 104.6 / 95.9 | 229.1 / 215.6 | 438.8 / 392.6 | 362.8 / 327.7 | 313.9 / 301.4 | 576.4 / 599.4 |
| (2, 8192) | 116.3 / 104.3 | 230.8 / 224.3 | 433.8 / 421.6 | 372.9 / 347.4 | 337.5 / 297.9 | 597.2 / 556.7 |
| (1, 16384) | 116.9 / 118.2 | 227.9 / 229.8 | 435.4 / 433.5 | 361.5 / 357.7 | 332.6 / 316.0 | 614.8 / 579.7 |

What the numbers say:

- **`mma` to `mma_pipelined`, 2.2x.** Moving K and V through a two-stage `cp.async` ring hides
  the global loads: `long_scoreboard`, 7.6 cycles per instruction in `mma`, leaves the top stall
  reasons and the tensor pipe goes from 15.9% to 37.1% active
  ([metrics](../profiling/metrics.md#mma_pipelined-loads-hidden-instruction-overhead-left)).
  `mma_pipelined` is nearly flat from S = 512 to 16384 (208 to 231 TFLOP/s): once the loads are
  hidden it is bound by instruction issue, not memory.
- **`mma_pipelined` against flash-attn, 0.62x to 0.67x.** Both use `mma.sync`; flash-attn keeps
  the tensor pipe 62.4% active while issuing fewer instructions per cycle. Its blocks have 128
  query rows over 4 warps, 32 per warp, so every K and V fragment feeds two MMAs, where
  `mma_pipelined` gives each warp 16 rows and pays one `ldmatrix` per MMA.
- **`triton` against flash-attn, 1.20x.** On sm_90a Triton 3.4 compiles the same algorithm to
  Hopper's warpgroup MMA (`wgmma`), which flash-attn 2 does not use; the benchmark ratio is 1.20x
  and the equal-clock ratio 1.17x. In the length sweep it is 0.94x flash-attn at S = 512 and 1.14x
  to 1.21x from S = 1024 on, as its larger tiles fill up.
- **cuDNN leads.** PyTorch's cuDNN backend reaches 610.5 TFLOP/s (61.7% of peak) at the headline
  shape, 1.65x flash-attn and 1.38x `triton`; FlashAttention-3 and cuDNN build on Hopper features
  (`wgmma`, TMA, warp specialization) that none of the kernels here use.
- **GQA, D = 64, fp16.** With 8 KV heads the prefill kernels run as fast as with 32: the K and V
  tiles are a small share of the work. At `D = 64` everything slows (fewer FLOPs per byte loaded),
  `mma_pipelined` to 200 TFLOP/s, 0.64x flash-attn. fp16 runs within 5% of bf16
  (`*-prefill_bf16-fp16.json`: `mma_pipelined` 221.5, `triton` 435.4, flash-attn 353.0 at the
  headline shape).

## Prefill, fp32

TFLOP/s against the 67 TFLOP/s fp32 peak, which counts FMAs on CUDA cores (TF32 is off):

| implementation | (4, 32, 32, 4096, 128) | causal | (2, 16, 16, 2048, 64) |
|---|---|---|---|
| `naive` | 3.8 | 1.9 | 3.5 |
| `fp32_fused` | 7.7 | 7.6 | 7.3 |
| `fp32_regtile` | 17.6 | 16.1 | 32.6 |
| SDPA math backend | 25.6 | 10.4 | 16.2 |
| SDPA memory-efficient backend | 46.1 | 45.4 | 32.5 |

- `naive` to `fp32_fused` (2.0x) removes the 8 GiB score matrix: one call moved 53.7 GB through
  DRAM, an intensity of 20.5 FLOP per byte; the fused kernel moves about the 1.07 GB minimum.
  `naive` also cannot skip masked tiles, so its causal rate halves.
- `fp32_fused` to `fp32_regtile` (2.3x at `D = 128`, 4.5x at `D = 64`): register micro-tiles cut
  shared-memory wavefronts 6.3x. At `D = 128` the kernel needs 254 registers and 117 KiB of shared
  memory, one block per SM, and global-load latency is left exposed; at `D = 64` it fits 3 blocks
  per SM and matches PyTorch's memory-efficient backend (32.6 against 32.5).

## Decode

Decode is a memory-bound matrix-vector product: one FLOP per byte of K and V with MHA, four with
4 query heads per KV head. GB/s at H = 32, D = 128, bf16 (`decode_ctx`, three runs):

| B | H_kv | context | decode_copy | decode_inplace | splitkv | flash_attn | sdpa_flash | sdpa_efficient |
|---|---|---|---|---|---|---|---|---|
| 1 | 32 | 512 | 75.7 | 103.0 | 174.4 | 501.3 | 492.8 | 340.2 |
| 1 | 32 | 2048 | 74.7 | 109.8 | 597.1 | 1140.3 | 1128.1 | 424.4 |
| 1 | 32 | 8192 | 79.3 | 113.4 | 1679.9 | 1986.2 | 1975.9 | 474.6 |
| 1 | 32 | 32768 | 82.2 | 114.5 | 2171.0 | 2779.6 | 2776.2 | 506.4 |
| 1 | 8 | 512 | 20.7 | 25.3 | 43.0 | 149.3 | 145.5 | - |
| 1 | 8 | 2048 | 29.2 | 27.0 | 161.1 | 434.1 | 432.7 | - |
| 1 | 8 | 8192 | 22.1 | 27.5 | 521.9 | 981.4 | 975.0 | - |
| 1 | 8 | 32768 | 22.3 | 27.6 | 674.2 | 1874.4 | 1867.7 | - |
| 8 | 32 | 512 | 288.2 | 734.7 | 1102.4 | 1663.7 | 1655.8 | 1745.2 |
| 8 | 32 | 2048 | 341.0 | 854.5 | 2100.0 | 2448.3 | 2333.3 | 2319.7 |
| 8 | 32 | 8192 | 350.2 | 891.1 | 2483.4 | 2718.6 | 2536.0 | 2524.5 |
| 8 | 32 | 32768 | 354.5 | 901.9 | 2691.7 | 2632.1 | 2358.1 | 2498.0 |
| 8 | 8 | 512 | 118.8 | 194.1 | 322.2 | 795.8 | 786.3 | - |
| 8 | 8 | 2048 | 132.5 | 207.7 | 652.5 | 1568.1 | 1556.5 | - |
| 8 | 8 | 8192 | 143.7 | 216.2 | 948.8 | 2486.4 | 2478.6 | - |
| 8 | 8 | 32768 | 141.9 | 218.0 | 1135.0 | 3004.3 | 2998.2 | - |

SDPA's memory-efficient backend does not take grouped KV heads (`-`).

- **Copies, then parallelism.** `decode_copy` copies the live cache twice per call (in the Nsight
  Systems trace 48.8% of its GPU time is copy kernels); `decode_inplace` reads the cache in place
  but runs one block per head, 32 blocks on 132 SMs, and stays near 115 GB/s for one sequence.
  `splitkv` splits each sequence's keys across blocks and merges the partial results, reaching
  2,171 GB/s (64.8% of peak) for one sequence at 32K and 2,692 GB/s (80.3%) for eight, where it
  is 1.02x flash-attn and ahead of both SDPA backends.
- **Long context.** One sequence of 32 heads reaches 2,422 GB/s at 64K and 2,594 GB/s (77.4%) at
  128K keys (`decode_long`), against flash-attn's 2,966 and 3,072.
- **Short context.** At 512 keys a call moves 8 MB, and fixed costs dominate: two kernel launches
  (decode and merge) and the Python dispatch for `splitkv`, against one fused launch for
  flash-attn; `splitkv` is 0.35x flash-attn there.
- **Grouped-query attention.** With 8 KV heads `splitkv` reaches 674 to 1,135 GB/s at 32K, 0.36x
  to 0.38x flash-attn. Sharing each K and V row among a group's query heads cuts the bytes 4x but
  not the work: `decode_kernel` computes every head's dot products and `P V` updates on CUDA cores
  (0.2% tensor-pipe activity), while flash-attn's split-KV kernel puts the group's heads into the
  rows of a tensor-core MMA and reaches 93.7% of DRAM bandwidth. A tensor-core GQA path is the
  largest decode improvement left (see [design.md](design.md#not-done-and-why)).
- **D = 64** (`decode_d64`, Llama-3.2-1B's shape): 603 GB/s for one sequence and 1,049 for eight at
  32K, 0.44x and 0.39x flash-attn; the same GQA limit with half the bytes per row.

Batch at 2,048 keys, H_kv = 8 (`decode_batch`, GB/s):

| B | decode_copy | decode_inplace | splitkv | flash_attn | sdpa_flash |
|---|---|---|---|---|---|
| 1 | 28.2 | 27.0 | 160.8 | 434.1 | 433.4 |
| 4 | 74.2 | 104.8 | 566.2 | 1142.0 | 1132.1 |
| 16 | 207.4 | 414.4 | 904.5 | 2030.7 | 2024.3 |
| 64 | 231.9 | 836.2 | 1096.4 | 2615.9 | 2472.8 |

### Split count and head grouping

`splitkv` at a fixed number of splits, 32K keys, GB/s (`*-decode_ctx-splits.json`, one run):

| splits | B=1, MHA | B=1, GQA | B=8, MHA | B=8, GQA |
|---|---|---|---|---|
| 1 | 115 | 28 | 904 | 218 |
| 2 | 229 | 55 | 1696 | 435 |
| 4 | 454 | 108 | 2746 | 845 |
| 8 | 872 | 212 | 2739 | 847 |
| 16 | 1576 | 406 | 2696 | 1099 |
| 32 | 2338 | 724 | 2875 | 1087 |
| 64 | 2177 | 682 | 2823 | 1138 |
| 128 | 1913 | 719 | 2709 | 1110 |

The heuristic aims at 16 blocks per SM, capped at 64 splits and at least 256 keys per split, and
picks 64, 64, 16, and 64 splits for the four columns. That is the best count for eight GQA
sequences and 6% to 7% below the best, 32 splits, for the other three columns. A rule based on
keys per split (1,024 at those optima) would fit these four points better; it was not changed,
because four points are not enough to choose it.

Query heads per block with 8 KV heads (`*-decode_ctx-heads?.json`, `splitkv` GB/s, one run):

| heads per block | B=1, 2K | B=1, 32K | B=8, 2K | B=8, 32K |
|---|---|---|---|---|
| 1 | 171 | 692 | 679 | 901 |
| 2 | 166 | 678 | 669 | 1004 |
| 4 | 162 | 676 | 655 | 1138 |

Grouping helps only where the kernel is near its bandwidth limit (eight sequences at 32K, 1.26x
from 1 to 4 heads); elsewhere the extra registers per lane cost a few percent.

## End to end

Llama-3.1-8B-Instruct in bf16 with transformers 4.57.6, batch 1, random-token prompts, 256 greedy
tokens, a static KV cache, and the decode step compiled by `generate()` (CUDA graphs).
[bench/e2e.py](../bench/e2e.py) checks each implementation's logits against an fp32 copy of the
model before timing it. Three runs (`*-e2e_llama-final?.json`):

| prompt | implementation | time to first token (ms) | decode step (ms) | decode tokens/s |
|---|---|---|---|---|
| 512 | `sdpa` | 24.8 | 10.40 | 96.2 |
| 512 | `flash_lab` (Triton prefill) | 28.7 | 8.21 | 121.8 |
| 512 | `flash_lab_cuda` (`mma_pipelined` prefill) | 23.1 | 8.22 | 121.7 |
| 8192 | `sdpa` | 523.3 | 22.68 | 44.1 |
| 8192 | `flash_lab` | 274.8 | 9.58 | 104.3 |
| 8192 | `flash_lab_cuda` | 311.7 | 9.58 | 104.4 |

Longer prompts (one run, `*-e2e_llama-context.json`; logits checked up to 8K tokens, see
[numerics.md](numerics.md#llama-31-8b-benchmark)):

| prompt | `sdpa` TTFT / step (ms) | `flash_lab` TTFT / step (ms) | step speedup |
|---|---|---|---|
| 2048 | 82.5 / 12.81 | 68.8 / 8.38 | 1.53x |
| 4096 | 194.2 / 16.14 | 131.6 / 8.43 | 1.91x |
| 16384 | 1617.4 / 35.35 | 634.4 / 11.69 | 3.02x |
| 32768 | 5483.0 / 60.75 | 1587.2 / 14.31 | 4.25x |

At batch 8 (`*-e2e_llama-batch8.json`), decode reaches 898 tokens/s against 645 with 512-token
prompts and 590 against 311 with 4096-token prompts.

What the baseline is: `sdpa` is transformers' SDPA path with a static cache. It passes an explicit
mask over the whole cache, which keeps PyTorch from its flash kernels and sends prefill and
decode to the memory-efficient kernel, with K and V repeated for the query groups and no split
of the keys. The eager Nsight Systems trace at 8K
([llama8b-steps-ctx8k.md](../profiling/reports/h100-80gb-hbm3/llama8b-steps-ctx8k.md)) shows the
memory-efficient attention kernel taking 47.5% of an `sdpa` step's GPU time, plus 19.8% in an
elementwise kernel that `flash_lab` steps do not run, against 23.3% for `flash_lab`'s decode and
merge kernels.
The comparison measures this integration against transformers' own path, not against
flash-attn's kernels, which beat `splitkv` at GQA decode. transformers' `flash_attention_2` path
with a static cache failed the logit check (errors of 13.2 and 2.6 against limits of 0.12 and
0.21) and is not timed. The step time of `flash_lab` grows by 5.9 ms from 2K to 32K tokens: the
time to read the extra 4.0 GB of cache at about 680 GB/s, which is what `splitkv` sustains for one
sequence with grouped KV heads (674 GB/s in `decode_ctx`).

## Profiles

[profiling/metrics.md](../profiling/metrics.md) tabulates the Nsight Compute metrics of every
kernel and checks each expectation set before profiling. The roofline places each kernel at its
measured DRAM intensity:

![H100 roofline](figures/roofline_h100.svg)

The fused kernels sit at their minimum-traffic intensity, far right of the ridge, so their
distance below the roof is instruction efficiency, not memory; `naive` moves 50x the minimum
bytes; the decode kernels sit on the bandwidth slope, where `splitkv` with MHA is close to the
roof and with GQA is not.

## Tuning record

| change | measured effect | source |
|---|---|---|
| `fp32_fused` tile 32 x 8 (default), 16 x 16, 8 x 32 | 7.7, 7.6, 6.2 TFLOP/s | `*-prefill_fp32-final1.json`, `*-tile-*.json` |
| Two-stage `cp.async` ring (`mma` to `mma_pipelined`) | 102.5 to 210.7 TFLOP/s | 20260830-072645-80ade99 |
| Softmax scale folded into one FFMA before `exp2` | 210.7 to 224.2 TFLOP/s (`mma_pipelined`) | 20260920-073601-430b6e7 |
| `__launch_bounds__(128, 3)` on `mma` | 172 back to 168 registers, 3 blocks per SM | commit 430b6e7, ptxas |
| Split heuristic from 2 to 16 blocks per SM | 1,571 to 2,169 GB/s (B=1, MHA, 32K) | 20260913-065721-03fa2ce, -030321-445a612 |
| Triton autotune (8 candidates) | `(128, 128, 8, 3)` at the headline shape in all runs | `triton_config` in each result |

Tried and not kept (commit 430b6e7): bounding `mma_pipelined` to 3 blocks per SM, which spills
and changed nothing at `D = 128`, and the maximum shared-memory carveout, slower at `D = 64`.

## History

The headline prefill shape across commits (non-causal TFLOP/s, single runs before cc64b18):

| commit | `mma` | `mma_pipelined` | `triton` | flash-attn |
|---|---|---|---|---|
| 3b23a63 | 102.0 | - | - | 362.9 |
| 80ade99 | 102.5 | 210.7 | - | 372.4 |
| 4a6a799 | 102.5 | 210.7 | 446.5 | 346.5 |
| 430b6e7 | 104.3 | 224.2 | 441.9 | 314.0 |
| cc64b18 (final1) | 104.6 | 229.2 | 443.1 | 369.4 |

Between code changes our kernels repeat closely (within 1% over the three final runs), while
flash-attn's rate moves between 314 and 372 TFLOP/s from run to run, so a single-run ratio such as
`mma_pipelined` at 0.71x flash-attn (430b6e7) says more about flash-attn's run than about the
kernel. That is why the final numbers take three
interleaved runs and compute ratios within each run.

## Where the gaps are

- **`mma_pipelined` to flash-attn (0.62x).** Work decomposition: 16 query rows per warp against
  32, so twice the `ldmatrix` traffic per MMA and more instructions per FLOP. Moving to two m16
  tiles per warp raises the output accumulators to 128 registers per lane at `D = 128`.
- **Everything to cuDNN (1.38x over `triton`).** Hopper's asynchronous warpgroup MMA, TMA loads,
  and warp-specialized pipelines; the CUDA kernels here target the sm_80 instruction set on
  purpose.
- **GQA decode (0.36x to 0.38x flash-attn).** Tensor-core MMAs over the query heads of a group.
- **Short-context decode.** Two launches and Python dispatch per call; a fused merge, or capture
  in a CUDA graph as the end-to-end path does, would remove most of it.
- **Split count.** About 7% at MHA, from a rule based on keys per split.
