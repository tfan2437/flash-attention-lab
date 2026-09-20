#pragma once

// Per-warp math on one 64-key tile, shared by the tensor-core prefill kernels (mma.cu and
// mma_pipelined.cu). They differ only in how tiles reach shared memory and how O is written.
//
// A block has 4 warps and each warp owns 16 query rows, one m16 tile of mma.sync.m16n8k16.
// Lane l holds fragment rows g = l / 4 and g + 8 and columns 2t, 2t + 1 with t = l % 4.
// Shared-memory tiles are [64][D + 8]: the 16-byte pad puts the eight row addresses of every
// ldmatrix in different banks.

#include <cmath>

#include "common/params.h"
#include "common/ptx.cuh"

namespace flash_lab::mma_tile {

constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
constexpr int kBlockM = 16 * kWarps;  // query rows per block
constexpr int kBlockN = 64;           // keys per tile

template <int D>
constexpr int kStride = D + 8;

// The warp's 16 rows of Q as A fragments, one per 16-wide k-step, kept for the whole kernel.
template <typename T, int D>
__device__ __forceinline__ void load_q_fragments(uint32_t (&q_frag)[D / 16][4], const T* q_smem,
                                                 int warp, int lane) {
#pragma unroll
  for (int ks = 0; ks < D / 16; ++ks) {
    ldmatrix_x4(q_frag[ks],
                q_smem + (warp * 16 + lane % 16) * kStride<D> + ks * 16 + (lane / 16) * 8);
  }
}

// s = Q K^T for 16 rows x 64 keys (8 n8 tiles). K stored [key][d] is already the column-major
// B operand, and one ldmatrix.x4 covers two n8 tiles of one k-step.
template <typename T, int D>
__device__ __forceinline__ void tile_scores(float (&s)[8][4], const uint32_t (&q_frag)[D / 16][4],
                                            const T* k_smem, int lane) {
#pragma unroll
  for (int j = 0; j < 8; ++j) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
#pragma unroll
  for (int ks = 0; ks < D / 16; ++ks) {
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      uint32_t kb[4];
      const int key = pair * 16 + lane % 8 + 8 * (lane / 16);
      ldmatrix_x4(kb, k_smem + key * kStride<D> + ks * 16 + 8 * ((lane / 8) % 2));
      mma_16816<T>(s[2 * pair], q_frag[ks], kb[0], kb[1]);
      mma_16816<T>(s[2 * pair + 1], q_frag[ks], kb[2], kb[3]);
    }
  }
}

// Hides keys past the end or above the causal diagonal. Scores stay unscaled: the softmax scale
// is folded into the exponent below. `needs_mask` is warp-uniform, so tiles away from the edges
// skip the index math.
__device__ __forceinline__ void mask_scores(float (&s)[8][4], const AttnParams& p, bool needs_mask,
                                            int warp_row0, int k0, int lane) {
  if (!needs_mask) return;
  const int g = lane / 4;
  const int t = lane % 4;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int row = warp_row0 + g + (e >= 2 ? 8 : 0);
      const int col = k0 + 8 * j + 2 * t + (e & 1);
      if (col >= p.seqlen_k || (p.causal && col > row + p.causal_offset)) s[j][e] = -INFINITY;
    }
  }
}

// Online softmax on the accumulator fragments: updates the running max (of unscaled scores) and
// this lane's partial sums for rows g and g + 8, turns s into probabilities, and rescales the
// output accumulator. With a positive scale, max(scale * s) = scale * max(s), so the max is taken
// on raw scores and each probability costs one FFMA plus exp2: 2^(s * scale_log2 - ref).
template <int D>
__device__ __forceinline__ void online_softmax(float (&s)[8][4], float (&o_acc)[D / 8][4],
                                               float (&row_max)[2], float (&row_sum)[2],
                                               float scale_log2) {
  float tile_max[2] = {-INFINITY, -INFINITY};
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    tile_max[0] = fmaxf(tile_max[0], fmaxf(s[j][0], s[j][1]));
    tile_max[1] = fmaxf(tile_max[1], fmaxf(s[j][2], s[j][3]));
  }
  float ref[2];
  float rescale[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    // The four lanes of a quad hold the same two rows.
    tile_max[i] = fmaxf(tile_max[i], __shfl_xor_sync(0xffffffff, tile_max[i], 1));
    tile_max[i] = fmaxf(tile_max[i], __shfl_xor_sync(0xffffffff, tile_max[i], 2));
    const float new_max = fmaxf(row_max[i], tile_max[i]);
    // A row that has only seen masked keys keeps a max of -inf; 0 as the reference point makes
    // its exponentials exactly 0 instead of nan.
    ref[i] = new_max == -INFINITY ? 0.f : new_max * scale_log2;
    rescale[i] = exp2f(fmaf(row_max[i], scale_log2, -ref[i]));
    row_max[i] = new_max;
  }
  float tile_sum[2] = {0.f, 0.f};
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    s[j][0] = exp2f(fmaf(s[j][0], scale_log2, -ref[0]));
    s[j][1] = exp2f(fmaf(s[j][1], scale_log2, -ref[0]));
    s[j][2] = exp2f(fmaf(s[j][2], scale_log2, -ref[1]));
    s[j][3] = exp2f(fmaf(s[j][3], scale_log2, -ref[1]));
    tile_sum[0] += s[j][0] + s[j][1];
    tile_sum[1] += s[j][2] + s[j][3];
  }
  row_sum[0] = rescale[0] * row_sum[0] + tile_sum[0];
  row_sum[1] = rescale[1] * row_sum[1] + tile_sum[1];
#pragma unroll
  for (int n = 0; n < D / 8; ++n) {
    o_acc[n][0] *= rescale[0];
    o_acc[n][1] *= rescale[0];
    o_acc[n][2] *= rescale[1];
    o_acc[n][3] *= rescale[1];
  }
}

// o_acc += P V. The fp32 probabilities of score tiles 2ks and 2ks + 1, packed to 16 bits, are
// exactly the A fragment of k-step ks, so P never leaves registers. V stored [key][d] must be
// transposed into the B operand, which ldmatrix.trans does on the way.
template <typename T, int D>
__device__ __forceinline__ void accumulate_pv(float (&o_acc)[D / 8][4], const float (&s)[8][4],
                                              const T* v_smem, int lane) {
#pragma unroll
  for (int ks = 0; ks < kBlockN / 16; ++ks) {
    const uint32_t a[4] = {
        pack2<T>(s[2 * ks][0], s[2 * ks][1]),
        pack2<T>(s[2 * ks][2], s[2 * ks][3]),
        pack2<T>(s[2 * ks + 1][0], s[2 * ks + 1][1]),
        pack2<T>(s[2 * ks + 1][2], s[2 * ks + 1][3]),
    };
#pragma unroll
    for (int pair = 0; pair < D / 16; ++pair) {
      uint32_t vb[4];
      ldmatrix_x4_trans(vb,
                        v_smem + (ks * 16 + lane % 16) * kStride<D> + pair * 16 + 8 * (lane / 16));
      mma_16816<T>(o_acc[2 * pair], a, vb[0], vb[1]);
      mma_16816<T>(o_acc[2 * pair + 1], a, vb[2], vb[3]);
    }
  }
}

// Combines the quad's partial sums and returns 1 / l for rows g and g + 8 (0 for rows that saw
// no key, which then produce zeros).
__device__ __forceinline__ void finish_rows(float (&row_sum)[2], float (&inv)[2]) {
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    row_sum[i] += __shfl_xor_sync(0xffffffff, row_sum[i], 1);
    row_sum[i] += __shfl_xor_sync(0xffffffff, row_sum[i], 2);
    inv[i] = row_sum[i] > 0.f ? 1.f / row_sum[i] : 0.f;
  }
}

// Natural-log LSE of rows g and g + 8, written by the first lane of each quad. row_max holds
// unscaled scores, so the scale is applied here.
__device__ __forceinline__ void store_lse(const AttnParams& p, int b, int h, int warp_row0,
                                          int lane, const float (&row_max)[2],
                                          const float (&row_sum)[2]) {
  if (lane % 4 != 0) return;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int row = warp_row0 + lane / 4 + 8 * i;
    if (row >= p.seqlen_q) continue;
    const int64_t index = (static_cast<int64_t>(b) * p.heads + h) * p.seqlen_q + row;
    p.lse[index] = row_sum[i] > 0.f ? row_max[i] * p.scale + logf(row_sum[i]) : -INFINITY;
  }
}

}  // namespace flash_lab::mma_tile
