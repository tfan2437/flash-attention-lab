# Design

This document describes how the kernels work: the algorithm they share, how each version divides
the work among threads and lays out memory, and what changed from one version to the next. Each
resource figure (registers, shared memory, resident blocks) names its source: the code,
`ptxas -v` for the sm_90 build ([profiling/ptxas/cc64b18.txt](../profiling/ptxas/cc64b18.txt), the
commit that produced the final numbers), or an Nsight Compute digest in
[profiling/reports/h100-80gb-hbm3/](../profiling/reports/h100-80gb-hbm3/). Measured times are the
medians of three runs in [summary-cc64b18.md](../bench/results/h100-80gb-hbm3/summary-cc64b18.md); TFLOP/s count causal work as half of
the full matrix, the FlashAttention-2 convention. The GPU is the H100
described in [environment.md](environment.md); benchmark results are in
[bench/results/h100-80gb-hbm3/](../bench/results/h100-80gb-hbm3/).

## Notation and conventions

| Symbol | Meaning |
|---|---|
| `B` | batch size |
| `S_q`, `S_k` | query and key lengths of a prefill call |
| `H`, `H_kv` | query heads and key/value heads, with `H % H_kv == 0` |
| `group` | `H / H_kv`; query head `h` reads KV head `h / group` |
| `D` | head dimension; every kernel supports 64 and 128 |
| `B_r`, `B_c` | query rows per block and keys per tile |
| `scale` | `softmax_scale`, `1 / sqrt(D)` by default |
| `g`, `t` | `lane / 4` and `lane % 4` in the tensor-core fragment layouts |

- Prefill: `q` is `[B, S_q, H, D]`; `k` and `v` are `[B, S_k, H_kv, D]`; `o` is `[B, S_q, H, D]`
  in the input dtype. Any strides are accepted as long as the last dimension is contiguous; the
  vectorized kernels also need 16-byte-aligned rows ([Layout and strides](#layout-and-strides)).
- `lse` is fp32 `[B, H, S_q]`: for row `i`, the natural log of `sum_j exp(scale * q_i . k_j)` over
  the keys the row can see.
- Causal masking is aligned to the bottom-right corner: key `j` is visible to query `i` iff
  `j <= i + (S_k - S_q)`. `AttnParams` carries `S_k - S_q` as `causal_offset`
  ([csrc/common/params.h](../csrc/common/params.h)). With `S_q > S_k`, rows `i < S_q - S_k` see
  no key.
- `flash_lab.attention` raises unless `softmax_scale > 0` ([flash_lab/ops.py](../flash_lab/ops.py));
  the tensor-core kernels depend on it (see [The exp2 domain](#the-exp2-domain)).
- Decode: `q` is `[B, 1, H, D]` or `[B, H, D]`; `k_cache` and `v_cache` are `[B, S_max, H_kv, D]`;
  `seq_lens` is int32 `[B]`, the valid cache rows of each sequence including the token being
  decoded. `o` has the shape of `q`; `lse` is `[B, H]`.
- Every kernel accumulates in fp32. Benchmark shapes are written `(B, H, H_kv, S, D)`, as in
  [bench/suites.py](../bench/suites.py).

| impl | Version | Source | dtypes |
|---|---|---|---|
| `naive` | v0 | [csrc/prefill/naive.cu](../csrc/prefill/naive.cu) | fp32 |
| `fp32_fused` | v1 | [csrc/prefill/fp32_fused.cu](../csrc/prefill/fp32_fused.cu) | fp32 |
| `fp32_regtile` | v2 | [csrc/prefill/fp32_regtile.cu](../csrc/prefill/fp32_regtile.cu) | fp32 |
| `mma` | v3 | [csrc/prefill/mma.cu](../csrc/prefill/mma.cu), [mma_tile.cuh](../csrc/prefill/mma_tile.cuh) | bf16, fp16 |
| `mma_pipelined` | v4 | [csrc/prefill/mma_pipelined.cu](../csrc/prefill/mma_pipelined.cu), [mma_tile.cuh](../csrc/prefill/mma_tile.cuh) | bf16, fp16 |
| `triton` | - | [flash_lab/triton_attention.py](../flash_lab/triton_attention.py) | bf16, fp16 |
| `decode_copy` | dec v0 | [csrc/decode/decode_copy.cu](../csrc/decode/decode_copy.cu) | bf16, fp16, fp32 |
| `decode_inplace` | dec v1 | [csrc/decode/decode.cu](../csrc/decode/decode.cu), one split | bf16, fp16, fp32 |
| `splitkv` | dec v2 | [csrc/decode/decode.cu](../csrc/decode/decode.cu) | bf16, fp16, fp32 |

## The algorithm

### FlashAttention-2 forward

The forward pass never forms the `S_q x S_k` score matrix. A block owns `B_r` query rows of one
(batch, head) and walks the key tiles those rows can see, keeping three running values per row:
the max `m` of the scores so far, the sum `l` of their exponentials relative to `m`, and the
unnormalized output `O`.

```
for each block of B_r query rows of one (batch, head), in parallel:
    m = -inf, l = 0, O = 0                   per row; O is B_r x D, fp32
    for each B_c-key tile (K_j, V_j) that some row of the block can see:
        S     = scale * Q K_j^T              B_r x B_c; hidden keys set to -inf
        m_new = max(m, rowmax(S))
        alpha = exp(m - m_new)               rescales what was accumulated so far
        P     = exp(S - m_new)
        l     = alpha * l + rowsum(P)
        O     = alpha * O + P V_j
        m     = m_new
    o   = O / l
    lse = m + ln(l)
```

As in FlashAttention-2:

- `O` is divided by `l` once per row, after the last tile.
- In the fused kernels, grid x is the query block and grid y is `b * H + h`. Blocks never
  communicate and nothing is accumulated with atomics, so outputs are bitwise reproducible
  (`test_deterministic` in [tests/test_attention_fwd.py](../tests/test_attention_fwd.py)).
- In `fp32_regtile`, `mma`, and `mma_pipelined` each warp owns 16 whole query rows, so a row's `m`
  and `l` stay in one warp and are reduced with shuffles.
- With causal masking the key loop stops after the last key the block's last valid row can see,
  `kv_end = min(S_k, min(q0 + B_r, S_q) + S_k - S_q)` in the CUDA kernels; later tiles are never
  loaded. The Triton bound is in [its section](#triton).

`reference.attention_tiled` ([flash_lab/reference.py](../flash_lab/reference.py)) is the same
loop in fp64 tensor ops, causal tile skipping included; `test_tiled_matches_naive` checks it
against the dense reference, which tests the algorithm apart from any kernel.

### The exp2 domain

All kernels except `naive`, `fp32_fused`, and `decode_copy` compute `exp(x)` as
`2^(x * log2(e))` with `exp2f` or `tl.math.exp2`. The host computes
`scale_log2 = scale * log2(e)` in double precision and passes it as a float
(`make_attn_params`, `make_decode_params`). No fast-math option is used ([setup.py](../setup.py)).

The tensor-core kernels fold the scale into the exponent (`online_softmax` in
[csrc/prefill/mma_tile.cuh](../csrc/prefill/mma_tile.cuh)):

```
s      raw score q . k from the mma accumulator, not multiplied by scale
m      running max of the raw scores of the row
ref    = m_new * scale_log2, or 0 while m_new is -inf
p      = exp2f(fmaf(s, scale_log2, -ref))        = exp(scale * (s - m_new))
alpha  = exp2f(fmaf(m_old, scale_log2, -ref))    = exp(scale * (m_old - m_new))
lse    = m * scale + logf(l)                     (store_lse)
```

Each probability costs one FFMA and one `exp2f`; scaling the tile first would add an FMUL per
score. The max of unscaled scores is the right reference only for `scale > 0`: then
`max(scale * s) = scale * max(s)`, and a masked `-inf` stays `-inf` through the `fmaf`. A negative
scale would turn the max into a min and masked scores into `+inf`, which is why
`flash_lab.attention` rejects it. The other kernels scale before taking the max:

| Kernel | Where the scale is applied | Exponential | LSE from `m` and `l` |
|---|---|---|---|
| `naive` | `scale_mask_kernel` multiplies `S` | `expf` | `m + ln l` |
| `fp32_fused` | each score times `scale` | `expf` | `m + ln l` |
| `fp32_regtile` | each score times `scale_log2` | `exp2f` | `m ln 2 + ln l` |
| `mma`, `mma_pipelined` | inside the `fmaf`; the max is of raw scores | `exp2f` | `m * scale + ln l` |
| `triton` | the score tile times `scale_log2` | `tl.math.exp2` | `m ln 2 + ln l` |
| `decode_inplace`, `splitkv` | `q` times `scale_log2`, once, as it is loaded | `exp2f` | `m ln 2 + ln l` |
| `decode_copy` | each score times `scale` | `expf` | `m + ln l` |

### Masks, ragged edges, and rows without keys

- Rows past the end are zero-filled as they are loaded (Q rows at or past `S_q`, K and V rows at
  or past `S_k`): the fp32 kernels and `mma` branch around the load, `mma_pipelined` issues
  `cp.async` with a source size of 0, and `triton` uses masked loads with `other=0.0`. Nothing past
  the tensor is read, and a V row past the end holds zeros, so its product with a zero
  probability cannot be NaN from stale memory.
- Hidden keys score `-inf`. While all of a row's scores are `-inf`, `m` stays `-inf`, and the
  online-softmax kernels that can meet such a row (`fp32_fused`, `fp32_regtile`, `mma_tile.cuh`,
  `triton`, `decode_kernel`) use 0 as the reference point (`ref = new_max == -INFINITY ? 0 : ...`),
  so the exponentials are `exp2(-inf) = 0` rather than `exp2(-inf - (-inf)) = NaN`.
- A row with `l = 0` after the last tile saw no key (causal, `S_q > S_k`). It is written as zeros
  with `lse = -inf`, the definition the fp64 reference uses.
- `fp32_regtile`, `mma`, and `mma_pipelined` decide once per tile and warp whether element masks
  are needed (`needs_mask`): only for a partial last tile, or a causal tile whose last key is past
  what the warp's first row can see. The test is warp-uniform, so it does not diverge. `triton`
  splits the key loop instead; `naive` and `fp32_fused` mask every score.

## Prefill kernels

| Version | impl | Threads | `B_r x B_c` | Q held in | K and V loaded by | Change |
|---|---|---|---|---|---|---|
| v0 | `naive` | 256 | 16 x 16 GEMM tiles | - | - | baseline: scores through HBM |
| v1 | `fp32_fused` | 256 | 32 x 8, 16 x 16, 8 x 32 | shared memory | scalar loads | fused, online softmax |
| v2 | `fp32_regtile` | 128 | 64 x 64 | shared memory | float4 loads | register micro-tiles |
| v3 | `mma` | 128 | 64 x 64 | registers | 16-byte loads | tensor cores, bf16 and fp16 |
| v4 | `mma_pipelined` | 128 | 64 x 64 | registers | `cp.async`, two stages | loads overlap the math |
| - | `triton` | 128 or 256 | autotuned | registers | the compiler | same algorithm in Triton |

### naive (v0)

[csrc/prefill/naive.cu](../csrc/prefill/naive.cu) runs four fp32 launches and keeps the score
matrix in global memory:

1. `gemm_kernel<true>`: `S = Q K^T` into a `[B * H, S_q, S_k]` buffer.
2. `scale_mask_kernel`: `S *= scale`; entries above the bottom-right diagonal become `-inf`.
3. `softmax_kernel`: one 256-thread block per row reads the row three times (max, sum of
   `exp(S - max)`, then `P = exp(S - max) / sum` written in place) and writes the row's LSE.
4. `gemm_kernel<false>`: `O = P V`, stored directly into the `[B, S_q, H, D]` output.

The matrix products are the same as in the fused kernels; the difference is the round trip of
`S` through HBM. At `(4, 32, 32, 4096, 128)` the buffer is 128 slabs of 4096 x 4096 floats, 8 GiB,
and every launch reads or writes all of it. The later versions exist to remove this traffic.

- The GEMM grid is `(ceil(N / 16), ceil(M / 16), B * H)` with 16 x 16 threads, one output per
  thread; each 16-wide step of the inner dimension stages a 16 x 16 tile of both operands in
  shared memory. `block_reduce` in the softmax uses warp shuffles, then one shared-memory step
  across the 8 warps.
- An `Operand` describes each matrix by base pointer, batch, head, and row strides, and a head
  divisor, so Q, K, V, and O are used in place and K and V are read at head `h / group`.
- The two shared-memory tiles are `[16][17]` floats, 2,176 bytes per block (ptxas). The extra
  column puts the 16 elements of a tile column in 16 different banks, which the transposed store
  of K in the `Q K^T` GEMM (`b_s[tx][ty]`) needs.
- Registers (ptxas, sm_90): `gemm_kernel` 30, `softmax_kernel` 31, `scale_mask_kernel` 20.

Measured (fp32, `(4, 32, 32, 4096, 128)`): 286.8 ms, 3.8 TFLOP/s. With the causal mask it takes
the same time, 283.3 ms, since it computes and stores the whole matrix and masks it afterwards, so
by the half-work convention it reaches 1.9 TFLOP/s.

### fp32_fused (v1)

[csrc/prefill/fp32_fused.cu](../csrc/prefill/fp32_fused.cu) is the first fused kernel and the
fp32 baseline for the later ones. Scores exist one tile at a time in shared memory, and the online
softmax replaces the three passes over each row; nothing of size `S_q x S_k` is allocated.

- Grid `(ceil(S_q / B_r), B * H)`, 256 threads. `B_r * B_c == 256` (a `static_assert`), so every
  thread computes one score per tile. The tile is 32 x 8 by default; `FLASH_LAB_FP32_TILE=16x16`
  or `8x32` selects another compiled shape at launch.
- Each key tile runs four phases separated by `__syncthreads()`:
  1. all threads load the tile's K and V rows, zero-filled past `S_k`;
  2. thread `t` computes the score of row `t / B_c` and key `t % B_c`, a `D`-long dot product
     from shared memory accumulated in order of `d` in one register;
  3. threads `0 .. B_r - 1`, one per row, update `m` and `l` and turn scores into probabilities;
  4. all threads rescale the output accumulator and add `P V`, `B_r * D / 256` elements each.

```
static shared memory; rows padded to D + 1 floats
  q_tile  [B_r][D + 1]   the block's query rows, loaded once
  k_tile  [B_c][D + 1]   current K tile
  v_tile  [B_c][D + 1]   current V tile
  o_acc   [B_r][D + 1]   unnormalized output
  scores  [B_r][B_c]     scores, then probabilities
  row_max, row_sum, row_rescale   [B_r] each
```

In phase 2 a warp reads the same `d` from 8, 16, or 32 K rows (tiles 32 x 8, 16 x 16, 8 x 32).
A row of `D + 1` floats is one word past a multiple of 32 words, so row `c` starts in bank
`c mod 32` and the reads hit different banks; rows of exactly `D` floats would all start in bank 0.

What limits it: both arithmetic phases take two shared-memory operands per FMA, only `B_r` of 256
threads work in phase 3, and each tile passes four block-wide barriers. In
[fp32-fused-s4096-full.md](../profiling/reports/h100-80gb-hbm3/fp32-fused-s4096-full.md) the
leading stall reason is `mio_throttle` (the shared-memory instruction queue is full), and
`barrier` is among the larger ones. The same digest reports, for `D = 128` and 32 x 8, 78
registers and 42,688 bytes of static shared memory, which allow 3 blocks (24 warps) per SM by
either limit. The 16 x 16 and 8 x 32 shapes use 64 and 80 registers at `D = 128` (ptxas).

Measured: 142.9 ms and 7.7 TFLOP/s at `(4, 32, 32, 4096, 128)`, 2.0x `naive`; causal 72.5 ms,
7.6 TFLOP/s, since masked tiles are now skipped. At `(2, 16, 16, 2048, 64)` 7.3 TFLOP/s against
3.5. The tile shapes are compared in `*-prefill_fp32-tile-*.json`.

### fp32_regtile (v2)

[csrc/prefill/fp32_regtile.cu](../csrc/prefill/fp32_regtile.cu) moves the arithmetic into
registers. Each lane computes a 4 x 8 micro-tile of scores, keeps its rows' softmax state in
registers, and owns a 4 x `D / 8` block of the output. Per 4-wide step along `d` a lane loads 4
float4 of Q and 8 float4 of K (12 shared-memory loads) for 128 FMAs: about 2.7 FMAs per load,
against 0.5 in v1.

- Grid `(ceil(S_q / 64), B * H)`, 4 warps, `B_r x B_c = 64 x 64`; warp `w` owns query rows
  `16w .. 16w + 15` of the block.
- Lane `l` has `ri = l / 8`, `ci = l % 8`. It holds the scores of rows `ri + 4i` (`i < 4`) and
  keys `ci + 8j` (`j < 8`), and the outputs of the same rows at columns `4 ci + 32 c + (0..3)`
  (`c < D / 32`).
- The 8 lanes with the same `ri` share their rows, so the tile max is three `__shfl_xor_sync`
  steps (1, 2, 4). Each lane keeps a share of `l`, combined the same way after the last tile.
- In the score layout a lane holds 8 of a row's keys; `P V` needs all 64. P therefore goes
  through a per-warp `[16][72]` buffer in shared memory, ordered by `__syncwarp()` rather than a
  block barrier.

```
one warp: 16 query rows x 64 keys; lane = 8 ri + ci

               key  ci        ci + 8    ci + 16   ...   ci + 56
  row ri            s[0][0]   s[0][1]   s[0][2]   ...   s[0][7]
  row ri + 4        s[1][0]   s[1][1]   ...
  row ri + 8        s[2][0]   ...
  row ri + 12       s[3][0]   ...                       s[3][7]

  lanes  0 ..  7 (ri = 0): rows 0, 4, 8, 12     lanes  8 .. 15 (ri = 1): rows 1, 5, 9, 13
  lanes 16 .. 23 (ri = 2): rows 2, 6, 10, 14    lanes 24 .. 31 (ri = 3): rows 3, 7, 11, 15

dynamic shared memory (kSmemBytes): 119,808 bytes at D = 128, 70,656 at D = 64
  q_s [64][D + 4] floats    k_s [64][D + 4]    v_s [64][D + 4]    p_s 4 warps x [16][72]
```

Bank conflicts: keys are interleaved (`ci + 8j`) so that one float4 read of K by the warp touches
8 consecutive rows. A row of `D + 4` floats is 16 bytes past a multiple of 128 bytes, so those 8
rows start in 8 different 16-byte bank groups and cover the 32 banks once; the 4 Q rows of the
same step are broadcasts to 8 lanes each. P element `(r, k)` sits at `72 r + k`, and
`72 = 8 (mod 32)`, so lane `(ri, ci)` stores to bank `8 ri + ci + const`, all different. In `P V`
the P reads are broadcasts and the V reads are 8 consecutive float4 of one row.

Resources: at `D = 128`, 254 registers (ptxas, sm_90, no spills) and 119,808 bytes of shared
memory, which leave one 4-warp block per SM: Nsight Compute reports 2 blocks allowed by registers,
1 by shared memory, theoretical occupancy 6.25%
([fp32-regtile-s4096-full.md](../profiling/reports/h100-80gb-hbm3/fp32-regtile-s4096-full.md)).
At `D = 64`, 168 registers with a 12-byte spill, and 3 blocks per SM. The version trades occupancy
for fewer instructions per FMA. Against v1 the digest shows a fraction of the shared-memory
wavefronts, no significant barrier stalls, and a busier FMA pipe; the leading stall becomes
`long_scoreboard`, since K and V are loaded synchronously and no other block is resident to run
during the wait.

Measured: 62.4 ms and 17.6 TFLOP/s at `(4, 32, 32, 4096, 128)`, 2.3x `fp32_fused` (causal
16.1 TFLOP/s); at `(2, 16, 16, 2048, 64)`, where the smaller tiles allow 3 blocks per SM,
32.6 TFLOP/s, 4.5x `fp32_fused` and level with PyTorch's memory-efficient SDPA backend (32.5).
At `D = 128` that backend reaches 46.1.

### Tensor-core building blocks

`mma` and `mma_pipelined` share their per-tile math in
[csrc/prefill/mma_tile.cuh](../csrc/prefill/mma_tile.cuh) and differ only in how tiles reach
shared memory and how `O` is written. The wrappers in [csrc/common/ptx.cuh](../csrc/common/ptx.cuh)
emit `mma.sync.aligned.m16n8k16.row.col.f32.{bf16,f16}.{bf16,f16}.f32` and
`ldmatrix.sync.aligned.m8n8.x4[.trans].shared.b16`.

```
one block: 64 query rows of one (b, h), 4 warps, one 64-key tile at a time
  warp w: rows q0 + 16w .. q0 + 16w + 15, one m16 tile
    S (16 x 64)  = Q_w (16 x D) K^T          8 n8 tiles, D / 16 k-steps
    O (16 x D)  += P (16 x 64) V (64 x D)    D / 8 n8 tiles, 4 k-steps
  K and V tiles: [64][D + 8] elements in shared memory, read by all 4 warps
```

**Fragment layout.** The register layout of `mma.sync.m16n8k16` as the kernels use it:

```
A: 16 x 16 (rows m, columns k), bf16/fp16; 4 registers per lane, 2 elements each
                    k = 2t, 2t + 1       k = 2t + 8, 2t + 9
     m = g              a0                   a2
     m = g + 8          a1                   a3

B: 16 x 8 (rows k, columns n); 2 registers per lane
     b0 = B[2t][g], B[2t + 1][g]          b1 = B[2t + 8][g], B[2t + 9][g]

C, D: 16 x 8, fp32; 4 registers per lane
     c0, c1 = C[g][2t], C[g][2t + 1]      c2, c3 = C[g + 8][2t], C[g + 8][2t + 1]

lane that holds each element of C:
               n = 0,1   2,3   4,5   6,7
     m = 0         0     1     2     3        c0, c1
     m = 1         4     5     6     7
       ...
     m = 7        28    29    30    31
     m = 8         0     1     2     3        c2, c3
       ...
     m = 15       28    29    30    31

The lower-indexed element of an A or B pair sits in the low 16 bits of its register.
The four lanes of a quad, 4g .. 4g + 3, hold rows g and g + 8 completely.
```

A warp keeps `s[8][4]`, the C fragments of 8 n8 tiles of scores (64 keys), and
`o_acc[D / 8][4]`, the C fragments of its 16 x `D` output.

**Operand loads.** `ldmatrix.x4` loads four 8 x 8 matrices of 16-bit elements: lanes
`8i .. 8i + 7` supply the row addresses of matrix `i`, and lane `l` receives row `l / 4`,
columns `2 (l % 4)` and `2 (l % 4) + 1` of each, which is the fragment shape above. With
`.trans`, lane `l` receives rows `2 (l % 4)` and `2 (l % 4) + 1` of column `l / 4`.

| Operand | Stored as | Instruction | Lane `l` points at | Fills |
|---|---|---|---|---|
| Q, A of `Q K^T` | `[row][d]` | `ldmatrix.x4` | row `16 w + l % 16`, column `16 ks + 8 (l / 16)` | `a0 .. a3` of k-step `ks` |
| K, B of `Q K^T` | `[key][d]` | `ldmatrix.x4` | key `16 p + l % 8 + 8 (l / 16)`, column `16 ks + 8 ((l / 8) % 2)` | `b0, b1` of n8 tiles `2p`, `2p + 1` |
| V, B of `P V` | `[key][d]` | `ldmatrix.x4.trans` | key `16 ks + l % 16`, column `16 p + 8 (l / 16)` | `b0, b1` of n8 tiles `2p`, `2p + 1` |

- Q is loaded once per block (`load_q_fragments`); its `D / 16` A fragments, 32 registers at
  `D = 128`, stay in registers for every key tile.
- K needs no transpose. The B operand of `Q K^T` is `K^T` (k is `d`, n is the key), and in a
  `.row.col` mma a column of B is contiguous along k. A key's row of the `[key][d]` tile is such a
  column, so plain `ldmatrix` gives lane `l` (key `g`, `d = 2t, 2t + 1`), which is `b0`.
- V needs one. The B operand of `P V` is V (k is the key, n is `d`), whose columns are strided in
  `[key][d]`; `ldmatrix.trans` transposes each 8 x 8 matrix in flight, so lane `l` receives
  (keys `2t, 2t + 1`, column `g`).
- Per warp and 64-key tile at `D = 128`: 32 `ldmatrix.x4` and 64 `mma` for the scores, 32
  `ldmatrix.x4.trans` and 64 `mma` for `P V`.

**P stays in registers.** The C fragments of two adjacent score tiles are, element for element,
the A fragment of one `P V` k-step. `online_softmax` overwrites `s` with fp32 probabilities, and
`accumulate_pv` packs pairs to 16 bits (`pack2`, round to nearest) and issues the mma, with no
shared memory and no shuffles in between:

```
P V k-step ks covers keys 16 ks .. 16 ks + 15 of the tile

  s[2ks][0],   s[2ks][1]     row g,     keys 16 ks + 2t, + 1       ->  a0 = pack2(...)
  s[2ks][2],   s[2ks][3]     row g + 8, keys 16 ks + 2t, + 1       ->  a1
  s[2ks+1][0], s[2ks+1][1]   row g,     keys 16 ks + 8 + 2t, + 1   ->  a2
  s[2ks+1][2], s[2ks+1][3]   row g + 8, keys 16 ks + 8 + 2t, + 1   ->  a3
```

`l` is summed from the fp32 probabilities; `P V` uses them rounded to bf16 or fp16.

**Softmax on fragments.** A row's tile max is the max of the lane's 16 values for that row plus
two `__shfl_xor_sync` steps (1, 2) across the quad. Each lane keeps a partial `l` for its two
rows, and `finish_rows` combines the quad's partials once at the end. At `D = 128` a lane holds 32
registers of Q fragments, 64 output accumulators, and 32 scores before any addresses or K and V
fragments; the builds use 168 to 214 registers.

**Shared-memory padding.** Tiles are `[64][D + 8]` elements, so a row is 272 bytes at `D = 128`
and 144 at `D = 64`, 16 bytes past a multiple of 128 in both cases. One 8 x 8 matrix of an
`ldmatrix` is 8 consecutive rows of 16 bytes; with this stride the 8 rows start in 8 different
16-byte bank groups and cover the 32 banks once, so every `ldmatrix` is conflict-free. Unpadded
rows (256 or 128 bytes) would put all 8 in the same 4 banks.

**Testing the layout.** `mma_tile_test`
([csrc/prefill/mma_tile_test.cu](../csrc/prefill/mma_tile_test.cu)) runs one 16 x K by K x 16
product with the same calls, B read as K and as V.
[tests/test_mma_primitives.py](../tests/test_mma_primitives.py) checks it against a matmul and
with an identity matrix, which turns a lane or register mix-up into a visible permutation.

### mma (v3)

[csrc/prefill/mma.cu](../csrc/prefill/mma.cu) is the first tensor-core kernel: bf16 or fp16
inputs, fp32 accumulation. Against v2, the matrix products run on `mma.sync`, Q and P stay in
registers, and the softmax works on accumulator fragments. Per block:

1. The 128 threads copy the block's 64 Q rows into the K buffer with 16-byte loads (Q has no
   buffer of its own, `static_assert(kBlockM == kBlockN)`); after a barrier each warp loads its Q
   fragments.
2. For each key tile: a barrier, then all threads load K and V with 16-byte loads (global memory
   to registers to shared memory), another barrier, then `tile_scores`, `mask_scores`,
   `online_softmax`, `accumulate_pv`.
3. `finish_rows`; each lane stores its output fragments with 4-byte stores, so a warp store
   instruction gives each of 8 rows 16 bytes. The first lane of each quad stores the LSE.

Shared memory is two static `[64][D + 8]` buffers, 34,816 bytes at `D = 128` and 18,432 at
`D = 64`. `__launch_bounds__(128, 3)` holds registers at or under 168 so that three blocks
(12 warps) stay resident per SM; with synchronous loads, other resident warps are the only latency
hiding. Folding the scale into the FFMA had raised the unbounded `D = 128` build to 172 registers,
one block per SM fewer (commit 430b6e7; comment in `mma.cu`). ptxas (sm_90): 168 registers at
`D = 128` and 145 at `D = 64`, no spills.

[mma-s4096-full.md](../profiling/reports/h100-80gb-hbm3/mma-s4096-full.md) (bf16,
`(4, 32, 32, 4096, 128)`, non-causal) reports 168 registers, 34.8 KB of static shared memory, and
3 blocks per SM allowed by registers. `long_scoreboard`, waiting on global loads, is by far the
largest stall reason, and the tensor pipe is idle most of the time: every tile waits for its own
loads before any math starts, and 12 warps per SM do not cover the wait.

Measured (bf16, `(4, 32, 32, 4096, 128)`): 104.5 TFLOP/s non-causal and 95.7 causal, 0.28x and
0.29x flash-attn in the same runs.

### mma_pipelined (v4)

[csrc/prefill/mma_pipelined.cu](../csrc/prefill/mma_pipelined.cu) keeps the per-tile math of v3
and changes three things: K and V go through a two-stage ring filled with `cp.async`, so the copies
for tile `j + 1` land while tile `j` is computed; causal query blocks are launched heaviest first;
and `O` leaves through shared memory with 16-byte stores.

```
dynamic shared memory (kSmemBytes): four slots of [64][D + 8] bf16/fp16 elements,
17,408 bytes each at D = 128 (69,632 in all) and 9,216 at D = 64 (36,864 in all)

  slot 0   stage 0, K   [64 keys][D + 8]   after the loop: O staging, warp w in rows 16w .. 16w+15
  slot 1   stage 0, V   [64 keys][D + 8]
  slot 2   stage 1, K   [64 keys][D + 8]   before the loop: the Q tile, [64 query rows][D + 8]
  slot 3   stage 1, V   [64 keys][D + 8]

  one row at D = 128: | d 0..7 | d 8..15 | ... | d 120..127 | 8 pad elements, never touched |
                        16 B     16 B            16 B         16 B
```

At `D = 128` the buffer exceeds the 48 KB static limit, so it is dynamic shared memory, with the
limit raised by `cudaFuncSetAttribute`. The padding is the one described above.

**The cp.async ring.** `cp_async_16` issues `cp.async.cg.shared.global` 16-byte copies: global to
shared memory without passing through registers, cached in L2 only. With `valid == false` the
source size is 0, so the destination is zero-filled and nothing is read. Each thread commits its
copies in groups, and `cp.async.wait_group 1` waits until at most the newest group is in flight.

```
              cp.async (per thread)                        after one barrier
prologue      issue Q -> slot 2, commit                    (group G_Q)
              issue tile 0 -> stage 0, commit              (group G_0)
              wait_group 1: G_Q has landed                 ldmatrix Q fragments; barrier
tile 0        issue tile 1 -> stage 1, commit
              wait_group 1: G_0 has landed                 scores, softmax, P V on stage 0; barrier
tile 1        issue tile 2 -> stage 0, commit
              wait_group 1: tile 1 has landed              scores, softmax, P V on stage 1; barrier
tile j        issue tile j + 1 -> stage (j + 1) % 2, commit (empty on the last tile)
              wait_group 1: tile j has landed              compute on stage j % 2; barrier
```

The commit is unconditional, an empty group on the last tile, so "all but the newest group"
always means "this tile's copies". Of the two barriers per tile, the first makes every thread's
copies visible before any warp reads the stage, and the second keeps the next iteration from
refilling a stage that a warp still reads. Q is its own group, committed before tile 0's, so the
prologue waits for Q alone while tile 0's copies continue.

**Heaviest-first causal order.** With causal masking and `S_q = S_k`, query block `q` needs `q + 1`
key tiles, so the last block of a (batch, head) does the most work. When `causal` is set the
kernel takes its query block as `gridDim.x - 1 - blockIdx.x`, so within each (batch, head) row of
the grid the block indices run from the heaviest query block to the lightest, and the grid ends
with short blocks instead of long ones that would run alone in the last wave. CUDA does not
specify block dispatch order; the reordering assumes it roughly follows the linear block index.

**Output through shared memory.** After the loop's final barrier, stage 0's K slot is free and no
copy targets it. Each warp writes its 16 x `D` output into rows `16w .. 16w + 15` of that slot with
4-byte stores, synchronizes with `__syncwarp()` (the rows are private to the warp), and copies the
rows to global memory with 16-byte loads and stores: one warp store instruction writes 2 complete
rows at `D = 128` (4 at `D = 64`), against 16 bytes in each of 8 rows in v3. The same row stride
keeps the fragment stores conflict-free: a quad writes 16 contiguous bytes, and rows `g = 0 .. 7`
start 16 bytes apart modulo 128.

**Resources.** ptxas (sm_90): 214 registers at `D = 128` and 145 at `D = 64`, no spills, which
give 2 blocks (8 warps) and 3 blocks per SM, limited by registers. There is no minimum-blocks
bound: a bound of 3 blocks per SM (168 registers, with spills) and the maximum shared-memory
carveout were measured and not kept, with no gain at `D = 128` and a loss at `D = 64`
(commit 430b6e7). The committed digest
[mma-pipelined-s4096-full.md](../profiling/reports/h100-80gb-hbm3/mma-pipelined-s4096-full.md)
reports 214 registers and 2 blocks per SM. In it, `long_scoreboard` leaves the top stall reasons
(`wait`, `selected`, and `not_selected` lead) and the tensor pipe is active 37.1% of the time,
against 15.9% for `mma`
([profiling/metrics.md](../profiling/metrics.md#mma_pipelined-loads-hidden-instruction-overhead-left)).

Measured: 229.2 TFLOP/s non-causal and 216.7 causal at `(4, 32, 32, 4096, 128)`, 2.2x and 2.3x
`mma`, and 0.62x and 0.67x flash-attn in the same runs.

### triton

[flash_lab/triton_attention.py](../flash_lab/triton_attention.py) is the same algorithm in
Triton, with tile shapes from autotuning and instruction selection left to the compiler.

- Program grid `(cdiv(S_q, BLOCK_M), B * H)`. A program loads the Q tile of its `BLOCK_M` rows
  once (rows past `S_q` masked to 0) and keeps `acc` (`BLOCK_M x D`, fp32), `m_i`, and `l_i`.
- Per key tile (`_tile_loop`): `s = tl.dot(q, tl.trans(k)) * scale_log2`, the exp2 online
  softmax with the same `-inf` guard, and `acc = tl.dot(p.to(v.dtype), v, acc)`, which rounds P
  to the input dtype as the CUDA kernels do. Strides are explicit and the KV head is `h // group`.

**Two key loops.** The key range is split at `full_end`, a multiple of `BLOCK_N`:

```
causal:      end      = max(min(S_k, row0 + BLOCK_M + S_k - S_q), 0)
             full_end = max(min(row0 + S_k - S_q + 1, end), 0), rounded down to BLOCK_N
non-causal:  end      = S_k
             full_end = S_k rounded down to BLOCK_N

loop 1  [0, full_end)     MASKED=False: no bounds on the loads, no causal comparison
loop 2  [full_end, end)   MASKED=True: masked loads past S_k and the causal test on the scores
```

Every tile of loop 1 ends at or before the first row's last visible key, so all rows of the
program see all of its keys. `MASKED` and `CAUSAL` are `tl.constexpr`, so each loop compiles
separately and the unmasked one carries no mask code.

**Autotuning.** The key is `(SEQ_BUCKET, HEAD_DIM, CAUSAL)`; `SEQ_BUCKET` is `S_k` rounded up to a
power of two, capped at 16384 (`_seq_bucket`), and is used only as the key. Candidates, as
`(BLOCK_M, BLOCK_N, num_warps, num_stages)`:

```
(64, 32, 4, 4)    (64, 64, 4, 3)    (64, 64, 4, 4)    (64, 128, 4, 3)
(128, 32, 4, 4)   (128, 64, 4, 3)   (128, 64, 8, 3)   (128, 128, 8, 3)
```

`FLASH_LAB_TRITON_AUTOTUNE=0` pins `(64, 64, 4, 3)` so that tests do not compile every candidate;
`FLASH_LAB_TRITON_CONFIG=BLOCK_M,BLOCK_N,num_warps,num_stages` pins any configuration, so a
profiler sees no trial launches.

**What it compiles to.** Triton 3.4 targets sm_90a on the H100. Two digests count instructions in
the PTX of the autotuned configuration for one shape each:
[triton-ptx-s4096-d128.md](../profiling/reports/h100-80gb-hbm3/triton-ptx-s4096-d128.md)
(non-causal, `D = 128`; `BLOCK_M` 128, `BLOCK_N` 64, 8 warps, 3 stages) and
[triton-ptx-s2048-d64-causal.md](../profiling/reports/h100-80gb-hbm3/triton-ptx-s2048-d64-causal.md)
(causal, `D = 64`; 64, 128, 4 warps, 3 stages). Both have `wgmma.mma_async` and no `mma.sync`,
`cp.async` and no TMA (`cp.async.bulk.tensor`), no `ldmatrix`, and a few `stmatrix`. The CUDA
kernels are built for sm_80 and sm_90 and use `mma.sync`, so on the H100 the Triton kernel runs on
Hopper's warpgroup MMA and `mma_pipelined` does not. A comparison of the two measures that
difference as well as the code.

Measured: 442.3 TFLOP/s non-causal and 391.4 causal at `(4, 32, 32, 4096, 128)`, 1.20x and
1.21x flash-attn and 1.9x `mma_pipelined`; at `(1, 32, 32, 16384, 128)` 443.1 and 440.2, 1.20x and
1.28x flash-attn. Autotuning chose `(128, 128, 8, 3)` for the first shape in all three runs; for
some other shapes it alternated between `BLOCK_N` 64 and 128 from run to run, with throughput
within a few percent (the `triton_config` field of each result).

### Prefill resource summary

Registers are ptxas output for sm_90. Shared memory follows from the `constexpr` tile sizes and
matches ptxas and Nsight Compute. Blocks per SM is the smaller of what the register file allows
(65,536 registers per SM, allocated per warp in units of 256) and what shared memory allows
(228 KiB per SM, 1 KiB of it reserved per block); where a Nsight Compute digest exists, it reports
the same block count.

| Kernel | Threads | Registers, `D` = 128 / 64 | Shared memory, `D` = 128 / 64 | Blocks per SM, `D` = 128 / 64 |
|---|---|---|---|---|
| `naive`, GEMM | 256 | 30 | 2,176 B static | 8 |
| `fp32_fused`, 32 x 8 | 256 | 78 / 40 | 42,688 / 22,208 B static | 3 / 6 |
| `fp32_regtile` | 128 | 254 / 168 (12 B spill) | 119,808 / 70,656 B dynamic | 1 / 3 |
| `mma` | 128 | 168 / 145 | 34,816 / 18,432 B static | 3 / 3 |
| `mma_pipelined` | 128 | 214 / 145 | 69,632 / 36,864 B dynamic | 2 / 3 |

## Decode kernels

Decode is attention for one new token per sequence: `q` is `[B, H, D]`, and sequence `b` attends
to the first `seq_lens[b]` rows of its cache. Each (query head, key) pair costs `4 D` FLOPs, the
dot product and the `P V` update, against `4 D` bytes of bf16 K and V, so with MHA there is about
one FLOP per byte read. Decode is bound by memory bandwidth: the kernels use CUDA cores, convert K
and V to fp32 as they load them, and are judged by achieved bandwidth against the HBM peak. The
three ops share their arguments and `DecodeParams`
([csrc/common/params.h](../csrc/common/params.h)).

### decode_copy (dec v0)

[csrc/decode/decode_copy.cu](../csrc/decode/decode_copy.cu) is the baseline the other two are
measured against, kept on purpose; `impl="auto"` never selects it (`in_auto=False` in
[flash_lab/ops.py](../flash_lab/ops.py)). On every call the host

1. computes `live = seq_lens.max().item()`, a device-to-host copy that blocks the CPU until the
   GPU reaches it and rules out capturing the step in a CUDA graph;
2. copies the live rows of each cache, `k_cache.narrow(1, 0, live).contiguous()`;
3. copies them again into a per-head `[B, H_kv, L, D]` layout, `.permute(0, 2, 1, 3).contiguous()`.

(PyTorch skips a copy only when the view is already contiguous.) The kernel runs one block of 128
threads per (sequence, query head), grid `B * H`, over 32-key tiles: all threads stage the tile's
K and V rows as fp32 in shared memory, rows padded to `D + 1`; threads 0 to 31 compute one score
each while the other 96 wait; the tile max and sum are shared-memory tree reductions with a
barrier per step, 15 `__syncthreads()` per tile in all; and one thread per output column updates
the accumulator. Each query head of a GQA group reads the shared KV head's rows on its own.

Shared memory (`q_s[D]`, `k_s` and `v_s` `[32][D + 1]`, `o_s[D]`, two 32-float arrays, two
scalars) is 34,312 bytes at `D = 128` and 17,416 at `D = 64`; ptxas reports 94 and 80 registers
(sm_90). The Nsight Systems digest
[decode-steps-b8-ctx8k.md](../profiling/reports/h100-80gb-hbm3/decode-steps-b8-ctx8k.md) shows
elementwise copy kernels and a reduction kernel (the max) in every `decode_copy` step, next to the
attention kernel.

Measured in the same trace (B = 8, MHA, 8K context, bf16): a `decode_copy` step takes 3,147 us,
48.8% of its GPU time in PyTorch's copy kernels, against 1,228 us for `decode_inplace` and 462 us
for `splitkv`. In the benchmark it reaches 82 GB/s with one sequence at 32K context, 2.5% of
the HBM peak.

### decode_inplace (dec v1)

`decode_inplace` is the kernel in [csrc/decode/decode.cu](../csrc/decode/decode.cu) run with one
split. Against `decode_copy`: the host only allocates `o` and `lse` and launches, with no sync and
no copies; the kernel reads `k_cache[b, key, h_kv, :]` through the cache's strides; a block serves
`kHeads` query heads that share a KV head, so every K and V row it loads is used for all of them;
and 4 warps take the block's 32-key chunks in turn (warp `w` takes chunks `w, w + 4, ...`), each
with its own `m`, `l`, and `O`, merged once at the end.

Grid `(B * H / kHeads, num_splits)`, 128 threads. Block `x` serves sequence `x / (H / kHeads)` and
query heads `h0 .. h0 + kHeads - 1` with `h0 = (x % (H / kHeads)) * kHeads`, which all read KV
head `h0 / group`. The two matrix-vector products use different lane mappings:

```
decode_kernel: one warp on one 32-key chunk; E = D / 8, C = D / 32 (D = 128: E = 16, C = 4)

Q K^T, steps s = 0 .. 7, four keys per step
  lanes  0 ..  7   key 4s       each lane holds elements [E (l % 8), E (l % 8) + E) of every
  lanes  8 .. 15   key 4s + 1   query head of the block, pre-multiplied by scale * log2(e),
  lanes 16 .. 23   key 4s + 2   loads the same elements of its key row (16-byte loads), and the
  lanes 24 .. 31   key 4s + 3   8 lanes add their partial dot products with shfl_xor 1, 2, 4;
                                lane 8i stores the score of key 4s + i in probs[warp][head]

softmax: lane l takes key l of the chunk; the chunk max is a 5-step warp reduction

P V, keys j = 0 .. 31
  lane l owns output columns [C l, C l + C) of every head of the block: it loads those C
  elements of V row j (the warp reads the whole row) and adds probs[warp][head][j] times them

static shared memory, floats
  probs   [4 warps][kHeads][32]   scores, then probabilities, of the current chunk
  warp_o  [4 warps][kHeads][D]    per-warp unnormalized outputs, for the final merge
  warp_m, warp_l   [4 warps][kHeads]
```

`Q K^T` reduces over `d`, so eight lanes split a key row: the row read is coalesced, three
shuffles finish the dot product, and keys past the range are not loaded and score `-inf`. `P V`
reduces over keys, so a lane owns output columns and accumulates in registers with no cross-lane
sum; owning keys instead would need a cross-lane sum for every output column. The softmax keeps
`m` warp-uniform and a per-lane share of `l`.

Warp merge: each warp sums its `l` across lanes, lane 0 writes `(m, l)` per head, and every lane
writes its output columns to `warp_o`. After a barrier, threads combine the 4 warps per (head,
column): `m = max m_w`, `num = sum O_w 2^(m_w - m)`, `den = sum l_w 2^(m_w - m)`, skipping warps
that saw no key. With one split the block writes `o = num / den` (0 if `den = 0`) and
`lse = m ln 2 + ln den`.

Heads per block (`heads_per_block`) is the largest power of two that divides `group`, capped at 4
for `D = 128` and 8 for `D = 64` so that the per-lane copy of the queries (`kHeads * D / 8`
floats) stays at 64 registers. MHA gets 1; Llama-3.1-8B (32 query heads, 8 KV heads, `D = 128`)
gets 4, the `decode_kernel<__nv_bfloat16, 128, 4>` in
[llama8b-steps-ctx8k.md](../profiling/reports/h100-80gb-hbm3/llama8b-steps-ctx8k.md).
`FLASH_LAB_DECODE_HEADS_PER_BLOCK` overrides it for experiments.

With one split the grid has `B * H / kHeads` blocks: 32 for one sequence with 32 query heads, 8
when those share 8 KV heads, against 132 SMs. With a small batch most SMs are idle; `splitkv`
addresses that.

Measured (bf16, 32K context, MHA): 114.5 GB/s for one sequence (3.4% of the HBM peak) and
901.9 GB/s for eight (26.9%). Nsight Compute shows the one-sequence case at 3.4% of DRAM bandwidth
with `long_scoreboard` at 19.9 cycles per instruction: 32 blocks cannot keep enough loads in
flight.

### splitkv (dec v2)

`splitkv` divides each sequence's keys among `num_splits` blocks, the flash-decoding scheme, and
merges their results in a second kernel.

```
one sequence b and one head tile (query heads h0 .. h0 + kHeads - 1, KV head h_kv)
keys [0, len), len = seq_lens[b]; ps = ceil(len / num_splits), rounded up to a multiple of 32

  split 0: [0, ps)         split 1: [ps, 2 ps)      ...   split n - 1: [(n - 1) ps, len)
  block (x, 0)             block (x, 1)             ...   block (x, n - 1)
     x = b * H / kHeads + tile; 4 warps take 32-key chunks in turn, each keeping m (log2
     domain), l, and O[D] per head; the warps are merged through shared memory
        |                        |                              |
        v                        v                              v
  fp32, per head h of the tile:  partial_o[b, h, s, :] = O_s (not divided by l_s)
                                 partial_ml[b, h, s, :] = (m_s, l_s)
        |                        |                              |
        +------------------------+------------------------------+
                                 v
  merge_kernel: one block per (b, h), one thread per column d, splits in order s = 0, 1, ...
     m    = max over s of m_s                (splits with m_s = -inf are skipped)
     l    = sum over s of l_s * 2^(m_s - m)
     o[d] = sum over s of O_s[d] * 2^(m_s - m), divided by l
     lse  = m ln 2 + ln l
```

Each block computes its range from `seq_lens[b]`, so sequences of different lengths split
differently, and a split that starts past the end is empty. `m_s` is in the log2 domain because
`q` was pre-scaled; `partial_o` (`[B, H, num_splits, D]`) and `partial_ml`
(`[B, H, num_splits, 2]`) are allocated on every call. The merge runs in a fixed order with no
atomics, so the result is bitwise reproducible; `test_splitkv_split_counts` in
[tests/test_decode.py](../tests/test_decode.py) runs 1, 2, 8, and 64 splits on ragged lengths,
which leaves some splits empty. With one split the merge kernel is not launched and
`decode_kernel` writes `o` and `lse` itself; that run is `decode_inplace`, which ignores its
`num_splits` argument.

**Choosing num_splits.** `choose_num_splits` runs on the host from shapes only: the cache capacity
`S_max`, not `seq_lens`, which stays on the device.

```
blocks = B * H / kHeads
splits = 1
while splits < 64 and blocks * splits < 16 * SM_count and S_max / (2 * splits) >= 256:
    splits *= 2
```

That is the smallest power of two giving at least 16 blocks per SM, capped at 64 splits and at
256 keys of `S_max` per split; a positive `num_splits` argument overrides it, up to 128. For one
sequence with 32 query heads, `D = 128`, and `S_max = 32768` on 132 SMs, the 32 blocks become
2,048: 64 splits of 512 keys.

The 16 blocks per SM describe the grid, not residency: at `D = 128`, at most 9 blocks of the bf16
one-head instantiation, and 3 of the four-head one, fit on an SM at once (table below). The first
heuristic aimed at 2 blocks per SM, which left too few warps resident to keep loads in flight
(commit 445a612); with a grid several times larger than what fits, every SM stays at its
occupancy limit until the last wave.

Measured (bf16, 32K context, MHA): 2,171 GB/s for one sequence (64.8% of the HBM peak, 64
splits, 0.78x flash-attn) and 2,692 GB/s for eight (80.3%, 16 splits, 1.02x flash-attn). The
split sweep (`*-decode_ctx-splits.json`) finds 32 splits best in both cases, 2,338 and 2,875 GB/s,
about 7% above the heuristic's choice: the rule counts blocks per SM, while what matters here is
closer to the number of keys per split (1,024 at the best point for both).

GQA: a block shares each K and V row among `kHeads` query heads, so bytes per (query head, key)
pair fall by a factor of `kHeads`, while the instructions per pair do not: each head still costs
its own dot product, shuffles, and `P V` update on CUDA cores. See
[Not done, and why](#not-done-and-why).

Measured (bf16, eight sequences at 32K): 1,135 GB/s with `H_kv = 8` against 2,692 with
`H_kv = 32`, while flash-attn reaches 3,004 GB/s with `H_kv = 8`. Grouping helps: 1, 2, and 4
query heads per block give 901, 1,004, and 1,138 GB/s in the heads sweep
(`*-decode_ctx-heads?.json`).

### Decode resource summary

Registers are ptxas output for sm_90; shared memory follows from the array sizes and matches
ptxas. Blocks per SM are computed as for prefill, for bf16; shared memory never limits them.

| `D` | Heads per block | Shared memory | Registers, bf16 / fp16 / fp32 | Blocks per SM, bf16 |
|---|---|---|---|---|
| 128 | 1 | 2,592 B | 56 / 56 / 68 | 9 |
| 128 | 2 | 5,184 B | 90 / 88 / 106 | 5 |
| 128 | 4 | 10,368 B | 135 / 135 / 195 | 3 |
| 64 | 1 | 1,568 B | 40 / 40 / 48 | 12 |
| 64 | 2 | 3,136 B | 61 / 62 / 56 | 8 |
| 64 | 4 | 6,272 B | 80 / 80 / 88 | 6 |
| 64 | 8 | 12,544 B | 190 / 186 / 196 | 2 |

The fp32 instantiation with 2 heads and the fp16 one with 4 heads, both at `D = 64`, spill 4
bytes. `merge_kernel` uses 32 registers, no shared memory, and `D` threads per block.
`decode_copy_kernel` (94 and 80 registers, 34,312 and 17,416 bytes) fits 5 blocks per SM at
`D = 128` and 6 at `D = 64`.

## API and layout

### Layout and strides

The tensors are `[B, S, H, D]` because that is how they arrive:

- A fused QKV projection produces `[B, S, 3, H, D]`, and `unbind(dim=2)` gives `q`, `k`, and `v`
  as strided views (`test_strided_inputs` in
  [tests/test_attention_fwd.py](../tests/test_attention_fwd.py),
  [tests/test_port_fidelity.py](../tests/test_port_fidelity.py)).
- flash-attn's `flash_attn_func` and `flash_attn_with_kvcache` take the same layouts, so the
  benchmark passes the same tensors to both libraries ([bench/baselines.py](../bench/baselines.py)).
- transformers keeps `[B, H, S, D]`; `transpose(1, 2)` makes it a `[B, S, H, D]` view, and a
  static cache `[B, H_kv, max_len, D]` becomes a `[B, max_len, H_kv, D]` view that the decode
  kernels read in place ([flash_lab/hf.py](../flash_lab/hf.py)).

So the kernels take batch, sequence, and head strides as arguments (in elements, in `AttnParams`
and `DecodeParams`) and require only a unit stride in `D`, which keeps each row contiguous for
vector loads and `ldmatrix`. The kernels that move 16 bytes at a time (`fp32_regtile`, `mma`,
`mma_pipelined`, `decode_inplace`, `splitkv`) also require every row to start on a 16-byte
boundary (`check_16_byte_rows`, `check_16_byte_cache_rows`). Outputs are allocated contiguous:
`o` as `[B, S_q, H, D]` in the input dtype and `lse` as fp32 `[B, H, S_q]`, so the LSE values of
one block's rows are one contiguous run. The LSE is returned because combining attention over
disjoint key ranges needs it (`splitkv` does this internally), and a backward pass would read it.

### GQA by index mapping

Query head `h` reads KV head `h / group` in every kernel. K and V are never expanded with
`repeat_interleave`, which would allocate and write `group` copies and read `group` times the
bytes on every call; only the untimed fp64 reference expands them. In decode the mapping goes one
step further: a block serves `kHeads` heads of a group and reads each cache row once for all.

### Bottom-right causal alignment

With `S_q < S_k`, as in a prefill against a cached prefix, the new queries are the last `S_q`
positions: query `i` sits at position `i + S_k - S_q` and sees keys up to there. This is the
FlashAttention-2 convention, which [flash_lab/reference.py](../flash_lab/reference.py) also
defines. SDPA's `is_causal` is top-left aligned, so the benchmark does not run SDPA on causal
inputs with `S_q != S_k` ([bench/baselines.py](../bench/baselines.py)).

### Op registration and torch.compile

- [csrc/bindings.cpp](../csrc/bindings.cpp) declares the schemas with `TORCH_LIBRARY(flash_lab, m)`.
  Every attention and decode op is functional: it returns new `(o, lse)` tensors and mutates
  nothing.
- Each `.cu` file registers its CUDA implementation with `TORCH_LIBRARY_IMPL(flash_lab, CUDA, m)`;
  importing `flash_lab._C`, an otherwise empty extension module, runs every registration. The ops
  are `torch.ops.flash_lab.<op>` and launch on PyTorch's current stream under a device guard.
- The Triton kernel joins the same namespace through
  `torch.library.custom_op("flash_lab::attention_triton", mutates_args=())` and dispatches the
  same way; without Triton (macOS) it is not registered.
- `torch.compile` traces with fake tensors, so each op needs its output shapes without running.
  `_register_fakes` in [flash_lab/ops.py](../flash_lab/ops.py) registers a shape-only
  implementation (`torch.library.register_fake`) for every compiled op: prefill returns an empty
  tensor like `q` and an fp32 `[B, H, S_q]`, decode an empty tensor like `q` and an fp32
  `[B, H]`. The Triton op registers its own.
  [tests/test_torch_compile.py](../tests/test_torch_compile.py) compiles every attention and
  decode path with `fullgraph=True`; with a static cache, the transformers `generate()` path
  compiles the decode step.

### impl="auto"

`PREFILL_IMPLS` and `DECODE_IMPLS` in [flash_lab/ops.py](../flash_lab/ops.py) list the
implementations in order of measured speed on the H100, and `impl="auto"` takes the first that
supports the input (dtype, head dim) and is compiled in: `triton`, `mma_pipelined`, `mma` for
bf16 and fp16 prefill; `fp32_regtile`, `fp32_fused`, `naive` for fp32; `splitkv`, then
`decode_inplace` for decode, with `decode_copy` excluded. The set of compiled ops is computed once
at import, so the choice costs a few comparisons per call, and error messages are formatted only
when a call fails. An explicit `impl` that cannot run the input raises `ValueError` with the
reason.

### Input checks

Python ([flash_lab/layouts.py](../flash_lab/layouts.py)) raises `ValueError` or `TypeError` with
the shapes in the message. The C++ ops repeat the checks with `TORCH_CHECK` and add alignment and
grid limits (`B * H <= 65535` for prefill), so a direct `torch.ops` call cannot launch on bad
input ([csrc/common/attn_host.h](../csrc/common/attn_host.h),
[csrc/decode/decode_host.h](../csrc/decode/decode_host.h)). `seq_lens` values are not checked on
the host, which would need a sync. `FLASH_LAB_DEBUG=1` builds synchronize after every launch, so
a fault is reported at the kernel that caused it ([csrc/common/checks.h](../csrc/common/checks.h)).

## Comparison with flash-attn 2

The external baseline is flash-attn 2.8.3 ([environment.md](environment.md)).
[flash-attn-s4096-full.md](../profiling/reports/h100-80gb-hbm3/flash-attn-s4096-full.md)
profiles its forward kernel at the shape of the `mma` digests (bf16, `(4, 32, 32, 4096, 128)`,
non-causal). The kernel is `flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, ...>>`:
head dim 128, 128-row query blocks, 64-key tiles, and 4 warps, on a (32, 4, 32) grid of query
blocks, batch, and heads. It uses 255 registers per thread and 64 KiB of dynamic shared memory and
runs 2 blocks (8 warps) per SM. Its tensor-core instructions are HMMA (`mma.sync`) and the GMMA
pipe, which runs `wgmma`, is idle, so on the H100 it uses the same instruction family as `mma` and
`mma_pipelined`. Its `long_scoreboard` stalls are small; the largest stall reasons besides
`selected` are `wait` and `math_pipe_throttle`. What differs:

- **Rows per warp.** 128 rows over 4 warps is 32 query rows per warp if the rows are split evenly,
  as the FlashAttention-2 paper describes; the kernels here give each warp 16. A warp with two m16
  tiles uses every K and V fragment it loads twice, so it issues half as many `ldmatrix` per
  `mma`, and each K and V tile in shared memory serves twice as many query rows. The cost is
  registers: 128 fp32 output accumulators per lane at `D = 128` instead of 64. This is the main
  structural change not made here.
- **Similar budgets otherwise.** Both use 64-key tiles at `D = 128` and run 2 blocks of 4 warps
  per SM (`mma_pipelined` at `D = 128`).
- **Where the gap is.** Both digests show few global-load stalls, so memory latency is no longer
  the difference. flash-attn keeps the tensor pipe busier while issuing fewer instructions per
  cycle than `mma_pipelined`, which points at instruction overhead per `mma`.

Measured: `mma_pipelined` runs at 0.62x flash-attn non-causal (range 0.62 to 0.63 over three
runs) and 0.67x causal (0.62 to 0.68) at `(4, 32, 32, 4096, 128)`. In the digests flash-attn's
tensor pipe is active 62.4% of the time at 38.7% issue-slot use, `mma_pipelined`'s 37.1% at
53.1%.

Decode: flash-attn's `flash_fwd_splitkv_kernel` (`Flash_fwd_kernel_traits<128, 64, 128, 4>`,
254 registers) runs its scores and `P V` on tensor cores even for one query token (31.9%
tensor-pipe activity, against 0.2% for `decode_kernel`), and it splits far less: 7 splits for
one sequence of 32 heads at 32K, a (1, 7, 32) grid, where `splitkv` launches 64 splits of 32
blocks. It reaches 90.9% of DRAM bandwidth there, and 93.7% with eight sequences of 8 KV heads,
where `decode_kernel` reaches 35.8% (digests `flash-attn-decode-*` and `splitkv-*`).

## Not done, and why

- **Backward pass.** The kernels are forward only, and no autograd formula is registered; the
  project covers inference, prefill and decode. The LSE output is what a FlashAttention-2 backward
  pass would read to recompute P without storing it.
- **Variable-length and paged inputs.** `attention` takes dense tensors with one `S_q` and one
  `S_k` per call and has no packed layout with cumulative sequence offsets, so prompts of
  different lengths are padded or run separately (the transformers integration accepts unpadded
  batches only, [flash_lab/hf.py](../flash_lab/hf.py)). The decode API accepts a paged cache
  (`[num_blocks, page_size, H_kv, D]` with `block_tables`) and the fp64 reference implements it,
  but every decode kernel rejects it ("paged KV caches are not supported"). The dense cache covers
  the benchmarks and the transformers static cache.
- **Hopper features in the CUDA kernels.** No `wgmma`, TMA, warp specialization, or thread-block
  clusters. The extension is built for sm_80 and sm_90 from one source
  (`TORCH_CUDA_ARCH_LIST="8.0;9.0"` in [setup.py](../setup.py)) with instructions both support:
  `mma.sync`, `ldmatrix`, `cp.async`. `wgmma` needs the sm_90a target and changes the work
  decomposition (a warpgroup of 4 warps issues one MMA of 64 rows, with B in shared memory), so it
  would be a different kernel rather than the next step of this progression. The Triton kernel
  shows what the instruction path alone changes for the same algorithm; FlashAttention-3 is built
  on these features.
- **Tensor-core GQA decode.** `decode_kernel` runs on CUDA cores and, with GQA, reuses each K and
  V row for `kHeads` heads but computes each head's dot products and `P V` updates separately. A
  tensor-core path would put a group's query heads in the M dimension of an mma tile, so one K or
  V fragment feeds all of them. Not done.
- **FP8.** Inputs and caches are bf16, fp16, or fp32. FP8 inputs or an FP8 KV cache would need
  scale factors and a tolerance policy of their own against the fp64 reference; not attempted.
- **Other options.** No dropout, attention bias, or sliding window; the transformers integration
  raises for dropout and `sliding_window`.

## References

- T. Dao, D. Y. Fu, S. Ermon, A. Rudra, C. Ré. FlashAttention: Fast and Memory-Efficient Exact
  Attention with IO-Awareness. NeurIPS 2022.
- T. Dao. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. 2023.
- J. Shah, G. Bikshandi, Y. Zhang, V. Thakkar, P. Ramani, T. Dao. FlashAttention-3: Fast and
  Accurate Attention with Asynchrony and Low-precision. 2024.
- T. Dao, D. Haziza, F. Massa, G. Sizov. Flash-Decoding for long-context inference. 2023.
- M. Milakov, N. Gimelshein. Online normalizer calculation for softmax. 2018.
- NVIDIA. PTX ISA: `mma.sync` m16n8k16 fragment layouts, `ldmatrix`, `cp.async`, `wgmma`.
- NVIDIA. CUDA C++ Programming Guide: shared-memory banks, occupancy, dynamic shared memory.
- Triton tutorials: Fused Attention. Read for reference; the kernel here does not copy it.
