# Profiler metrics

One table across the Nsight Compute digests in
[reports/h100-80gb-hbm3/](reports/h100-80gb-hbm3/), then what each kernel's profile says, checked
against what was expected before profiling.

The digests were taken by [scripts/pace/profiles.sh](../scripts/pace/profiles.sh) at commit
ef88a7a on the H100 described in [docs/environment.md](../docs/environment.md); the kernel sources
are those of cc64b18, which produced the benchmark numbers in
[bench/results/h100-80gb-hbm3/summary-cc64b18.md](../bench/results/h100-80gb-hbm3/summary-cc64b18.md).
Each digest is one warm launch under `ncu --set full` (after 5 calls of the implementation), with
caches flushed before every replay pass. Prefill is bf16 or fp32 at (B, H, H_kv, S, D) =
(4, 32, 32, 4096, 128), non-causal unless marked; decode is bf16 at a 32K context, one sequence
with 32 KV heads (MHA) or eight sequences with 8 (GQA, 4 query heads per KV head). Registers and
spills for every instantiation are in [ptxas/cc64b18.txt](ptxas/cc64b18.txt).

Durations are under the profiler, one serialized launch with cold caches, so they differ from the
benchmarked times. DRAM bytes do not count writes still held in the 50 MiB L2 when a kernel
ends, which is why a fused kernel can show slightly fewer bytes than it must write.

## Table

Regenerate with `python -m bench.ncu_table`. Tensor pipe is
`sm__pipe_tensor_cycles_active`, FMA pipe `sm__inst_executed_pipe_fma` (both % of peak over
active cycles), DRAM bandwidth the read and write shares of peak, achieved occupancy
`sm__warps_active` (% of 64 warps per SM), and stalls are cycles per issued instruction.

| implementation | kernel | duration (ms) | SM clock (GHz) | SM throughput % | tensor pipe % | FMA pipe % | DRAM bandwidth % | DRAM MB | L2 hit % | achieved occupancy % | registers | top stalls (cycles per issued instruction) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| naive | gemm_kernel | 137.386 | 1.98 | 80.8 | 1.7 | 27.9 | 2.0 | 9114 | 99.0 | 99.2 | 30 | not_selected 6.85, wait 2.61, math_pipe_throttle 2.13 |
| naive | scale_mask_kernel | 8.854 | 1.98 | 58.7 | 1.5 | 17.4 | 57.8 | 17158 | 50.0 | 78.5 | 20 | long_scoreboard 11.98, wait 3.67, short_scoreboard 1.50 |
| naive | softmax_kernel | 6.829 | 1.98 | 36.3 | 0.0 | 19.4 | 79.8 | 18273 | 53.5 | 91.8 | 31 | long_scoreboard 19.52, barrier 7.59, mio_throttle 2.85 |
| naive | gemm_kernel | 133.756 | 1.98 | 80.8 | 1.5 | 28.9 | 2.0 | 9136 | 86.5 | 99.8 | 30 | not_selected 5.67, long_scoreboard 3.66, wait 2.43 |
| fp32_fused | fp32_fused_kernel | 142.897 | 1.98 | 86.9 | 0.2 | 20.3 | 0.2 | 1076 | 98.1 | 37.3 | 78 | mio_throttle 3.18, not_selected 2.07, wait 1.89 |
| fp32_regtile | fp32_regtile_kernel | 62.457 | 1.98 | 35.0 | 0.2 | 28.4 | 0.5 | 1072 | 89.4 | 6.2 | 254 | long_scoreboard 1.12, selected 1.00, wait 0.26 |
| mma | mma_attention_kernel | 10.613 | 1.81 | 23.9 | 15.9 | 9.2 | 1.5 | 533 | 86.4 | 18.6 | 168 | long_scoreboard 7.63, wait 1.55, short_scoreboard 1.11 |
| mma_pipelined | mma_pipelined_kernel | 4.678 | 1.82 | 51.7 | 37.1 | 21.2 | 3.4 | 534 | 97.6 | 12.5 | 214 | wait 1.13, selected 1.00, not_selected 0.38 |
| triton | _attention_fwd_kernel | 2.363 | 1.81 | 47.6 | 49.1 | 14.8 | 6.7 | 532 | 89.5 | 12.5 | 244 | wait 1.33, barrier 1.32, selected 1.00 |
| flash-attn (baseline) | flash_fwd_kernel | 2.757 | 1.82 | 60.7 | 62.4 | 11.4 | 6.5 | 600 | 92.2 | 12.3 | 255 | wait 1.86, selected 1.00, math_pipe_throttle 0.83 |
| decode_inplace | decode_kernel | 4.686 | 1.98 | 1.0 | 0.0 | 1.8 | 3.4 | 540 | 0.3 | 6.2 | 56 | long_scoreboard 19.91, wait 1.68, selected 1.00 |
| splitkv | decode_kernel | 0.211 | 1.99 | 23.9 | 0.2 | 10.2 | 76.3 | 540 | 1.3 | 48.9 | 56 | long_scoreboard 27.23, wait 1.72, selected 1.00 |
| splitkv | merge_kernel | 0.032 | 1.99 | 0.7 | 0.0 | 1.5 | 1.0 | 1 | 48.2 | 6.2 | 32 | long_scoreboard 25.02, wait 2.64, selected 1.00 |
| flash-attn (baseline) | flash_fwd_splitkv_kernel | 0.177 | 1.79 | 30.4 | 31.9 | 4.3 | 90.9 | 540 | 0.9 | 10.5 | 254 | long_scoreboard 4.07, wait 1.74, selected 1.00 |
| flash-attn (baseline) | flash_fwd_splitkv_combine_kernel | 0.008 | 1.99 | 0.3 | 0.1 | 2.3 | 0.6 | 0 | 87.9 | 6.1 | 52 | long_scoreboard 8.16, wait 2.11, short_scoreboard 1.23 |
| splitkv, GQA | decode_kernel | 0.904 | 1.99 | 19.8 | 0.2 | 9.6 | 35.8 | 1085 | 1.6 | 18.4 | 135 | long_scoreboard 11.03, wait 1.23, selected 1.00 |
| splitkv, GQA | merge_kernel | 0.032 | 1.99 | 5.9 | 0.0 | 3.0 | 8.1 | 9 | 11.6 | 11.9 | 32 | long_scoreboard 24.19, wait 2.66, selected 1.00 |
| flash-attn, GQA (baseline) | flash_fwd_splitkv_kernel | 0.343 | 1.80 | 31.2 | 31.9 | 4.2 | 93.7 | 1076 | 1.5 | 12.1 | 254 | long_scoreboard 5.12, wait 1.75, barrier 1.40 |
| flash-attn, GQA (baseline) | flash_fwd_splitkv_combine_kernel | 0.006 | 1.98 | 2.3 | 0.1 | 2.7 | 2.8 | 1 | 67.4 | 6.1 | 56 | long_scoreboard 5.20, wait 2.12, short_scoreboard 1.18 |
| mma_pipelined, causal | mma_pipelined_kernel | 2.415 | 1.83 | 51.6 | 36.2 | 21.0 | 6.6 | 533 | 95.4 | 12.4 | 214 | wait 1.12, selected 1.00, not_selected 0.38 |
| triton, causal | _attention_fwd_kernel | 1.312 | 1.82 | 43.9 | 45.8 | 14.7 | 12.1 | 533 | 85.5 | 12.5 | 244 | wait 1.35, barrier 1.26, selected 1.00 |
| flash-attn, causal (baseline) | flash_fwd_kernel | 1.505 | 1.82 | 58.1 | 60.8 | 11.5 | 14.3 | 723 | 82.3 | 12.3 | 255 | wait 1.82, selected 1.00, math_pipe_throttle 0.79 |

## Prefill

### naive: the score matrix dominates the traffic, the GEMMs the time

Expected: DRAM-bound on the materialized score matrix, with measured bytes far above the minimum.

Measured: the bytes, yes; DRAM-bound, only in the two small kernels. One call moves 53.7 GB
(9.1 + 17.2 + 18.3 + 9.1 GB in the four launches) where the fp32 inputs and output need 1.07 GB,
an intensity of 20.5 FLOP per byte against 1024 for the fused kernels (see the
[roofline](../docs/figures/roofline_h100.svg)). But 95% of the time is in the two GEMMs, at 81%
SM throughput, 28% FMA-pipe utilization, and 2% of DRAM bandwidth: they are bound by issuing
loads and FMAs from 16 x 16 shared-memory tiles with one output per thread (`not_selected` is the
top stall, so warps are ready and wait for issue slots). Only `scale_mask_kernel` (58% of DRAM
bandwidth) and `softmax_kernel` (80%), 5% of the time together, are memory-bound. Partly confirmed.

### fp32_fused: shared-memory issue

Expected: high `mio_throttle` or `short_scoreboard`, high `barrier`, low FMA utilization, bank
conflicts near zero.

Measured: `mio_throttle` leads (3.18 cycles per instruction), `barrier` 1.01, FMA pipe 20%, and
0.2 billion bank conflicts in 34.4 billion shared-memory wavefronts (0.6%): the `D + 1` padding
works, and the kernel is limited by the rate of shared-memory instructions, two per FMA.
DRAM traffic is already near the minimum (1.08 GB). Confirmed.

### fp32_regtile: fewer shared-memory operations per FMA

Expected: FMA utilization up several-fold, `barrier` down, `long_scoreboard` visible.

Measured: shared-memory wavefronts fall 6.3x (34.4 to 5.5 billion), bank conflicts to 0.5 million,
`barrier` from 1.01 to 0.02, and `long_scoreboard` becomes the top stall (1.12). But the FMA pipe
only goes from 20% to 28% (1.4x, not several-fold): 254 registers and 117 KiB of shared memory
leave one 4-warp block per SM (6.2% achieved occupancy), so global-load latency has nothing to
hide behind. Time still drops 2.3x (142.9 to 62.4 ms benchmarked) because each FMA costs fewer
instructions. Partly confirmed.

### mma: tensor cores waiting on synchronous loads

Expected: tensor pipe active but far from saturated, `long_scoreboard` and `barrier` dominant,
168 registers limiting residency to 3 blocks per SM.

Measured: tensor pipe 15.9%, `long_scoreboard` 7.63 cycles per instruction (the next stall is
1.55), 168 registers and 18.6% achieved occupancy, which is 3 blocks of 4 warps out of 64 warp
slots. Every tile waits for its own K and V loads before any math. Confirmed (`barrier` is not
among the top three).

### mma_pipelined: loads hidden, instruction overhead left

Expected: `long_scoreboard` reduced by the cp.async overlap and tensor-pipe utilization up; the
remaining gap to flash-attn and cuDNN explained by instruction mix and missing Hopper features.

Measured: `long_scoreboard` leaves the top stalls (now `wait`, `selected`, `not_selected`),
the tensor pipe goes from 15.9% to 37.1%, and the benchmarked time drops 2.2x. The gap to
flash-attn is not Hopper features: flash-attn 2 uses the same `mma.sync` instructions (its
digest shows HMMA and no GMMA) and reaches 62.4% tensor-pipe activity while issuing fewer
instructions per cycle (38.7% of issue slots against 53.1%). The difference is instructions per
MMA: flash-attn's kernel (`Flash_fwd_kernel_traits<128, 128, 64, 4>`) spreads 128 query rows over
4 warps, 32 per warp, so every K and V fragment loaded feeds two MMAs, where `mma_pipelined`
gives each warp 16 (see
[docs/design.md](../docs/design.md#comparison-with-flash-attn-2)). The gap to cuDNN (and to the
Triton kernel) is where Hopper's `wgmma` comes in. Causal profiles look the same (36.2% tensor
pipe against flash-attn's 60.8%). Confirmed, with the flash-attn gap attributed to work
decomposition rather than instruction set.

### triton and flash-attn

Not a hypothesis of the original plan, but the comparison the README leans on. The Triton kernel
compiles to `wgmma` on sm_90a (its digest counts GMMA instructions; flash-attn's counts none),
keeps the tensor pipe 49.1% active at 34.9% issue-slot use and 244 registers, and runs the shape
in 2.36 ms under the profiler against flash-attn's 2.76 ms. Its leading stalls are `wait` and
`barrier`, the latter from the warpgroup MMA synchronization.

## Decode

Expected: Nsight Systems shows copy kernels dominating each `decode_copy` step; `decode_inplace`
removes them; `splitkv` raises DRAM throughput with more splits at small batch.

- Copies: in the Nsight Systems trace
  ([decode-steps-b8-ctx8k.md](reports/h100-80gb-hbm3/decode-steps-b8-ctx8k.md), B = 8, MHA, 8K
  context), 48.8% of `decode_copy`'s GPU time is PyTorch copy kernels, and a step takes 3,147 us
  against 1,228 us for `decode_inplace` and 462 us for `splitkv`. Confirmed.
- Splits: one sequence at 32K keys with one block per head uses 32 blocks on 132 SMs;
  `decode_inplace` reaches 3.4% of DRAM bandwidth with `long_scoreboard` at 19.9 cycles per
  instruction. With the default 64 splits, `splitkv`'s decode kernel reaches 76.3% at 48.9%
  achieved occupancy, and the merge kernel adds 0.03 ms. The split sweep
  (`*-decode_ctx-splits.json`) shows bandwidth doubling with each doubling of splits up to 16, a
  peak at 32 (2,338 GB/s), and a decline past it; the default 64 is 7% below that peak. Confirmed.
- GQA: with 4 query heads per KV head, `splitkv` reaches 35.8% of DRAM bandwidth against
  flash-attn's 93.7%. flash-attn's split-KV kernel runs on tensor cores (31.9% tensor pipe, HMMA),
  putting a group's query heads into the rows of an MMA, while `decode_kernel` computes each head
  separately on CUDA cores (0.2%), so its instruction count grows with the group size while its
  bytes do not. This is the main decode gap (see
  [docs/design.md](../docs/design.md#not-done-and-why)).
