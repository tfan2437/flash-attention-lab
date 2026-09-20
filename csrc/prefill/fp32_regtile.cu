// FlashAttention-2 forward in fp32 with register blocking: the v2 step between the scalar fused
// kernel (fp32_fused.cu, v1) and the tensor-core kernels.
//
// v1 needs a fresh shared-memory operand for every FMA. Here each lane computes a 4 x 8
// micro-tile of scores from registers: per 4-wide step along d it loads 4 Q values for each of
// its 4 rows and 4 K values for each of its 8 columns (12 float4 loads) and does 128 FMAs, about
// 2.7 FMAs per shared-memory load instead of 0.5. Row reductions stay within the 8 lanes that
// share a row group (three shuffles), the online softmax runs in registers, and P goes through a
// small per-warp buffer for P V, where each lane owns a 4 x (D / 8) block of the output.
//
// Block: 4 warps, 64 query rows (16 per warp), keys in tiles of 64. Lane l works on row group
// ri = l / 8 and column group ci = l % 8, with rows ri + 4i and columns ci + 8j interleaved so
// that the float4 reads of a warp land on distinct banks given rows padded to D + 4 floats.
// Shared memory is about 117 KB at D = 128 (one block per SM): the point of v2 is instruction
// efficiency, not occupancy.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cmath>

#include "common/attn_host.h"

namespace flash_lab {
namespace {

constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
constexpr int kBlockM = 64;
constexpr int kBlockN = 64;
constexpr int kPStride = kBlockN + 8;  // P rows: 8 * ri + ci covers all 32 banks

template <int D>
constexpr int kStride = D + 4;

template <int D>
constexpr int kSmemBytes =
    (3 * 64 * kStride<D> + kWarps * 16 * kPStride) * static_cast<int>(sizeof(float));

// Rows [row0, row0 + 64) of a [S, D] slice into shared memory with float4 loads; rows past
// `rows_valid` are zero-filled.
template <int D>
__device__ __forceinline__ void load_tile(float* dst, const float* src, int64_t row_stride,
                                          int row0, int rows_valid, int tid) {
  constexpr int kVecs = D / 4;
#pragma unroll
  for (int i = tid; i < 64 * kVecs; i += kThreads) {
    const int r = i / kVecs;
    const int c = (i % kVecs) * 4;
    float4 val = make_float4(0.f, 0.f, 0.f, 0.f);
    if (r < rows_valid) {
      val = *reinterpret_cast<const float4*>(src + static_cast<int64_t>(row0 + r) * row_stride + c);
    }
    *reinterpret_cast<float4*>(dst + r * kStride<D> + c) = val;
  }
}

template <int D>
__global__ void __launch_bounds__(kThreads) fp32_regtile_kernel(const AttnParams p) {
  extern __shared__ __align__(16) float smem[];
  float* q_s = smem;
  float* k_s = q_s + kBlockM * kStride<D>;
  float* v_s = k_s + kBlockN * kStride<D>;

  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int ri = lane / 8;
  const int ci = lane % 8;
  float* p_s = v_s + kBlockN * kStride<D> + warp * 16 * kPStride;

  const int q0 = blockIdx.x * kBlockM;
  const int b = blockIdx.y / p.heads;
  const int h = blockIdx.y % p.heads;
  const int h_kv = h / p.group;
  const float* q = static_cast<const float*>(p.q) + b * p.q_batch_stride + h * p.q_head_stride;
  const float* k = static_cast<const float*>(p.k) + b * p.k_batch_stride + h_kv * p.k_head_stride;
  const float* v = static_cast<const float*>(p.v) + b * p.v_batch_stride + h_kv * p.v_head_stride;
  float* o = static_cast<float*>(p.o) + b * p.o_batch_stride + h * p.o_head_stride;

  load_tile<D>(q_s, q, p.q_seq_stride, q0, min(kBlockM, p.seqlen_q - q0), tid);

  float o_acc[4][D / 8] = {};
  float row_max[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
  float row_sum[4] = {0.f, 0.f, 0.f, 0.f};  // this lane's share, combined at the end

  const int warp_row0 = q0 + warp * 16;
  const float* q_rows = q_s + warp * 16 * kStride<D>;
  int kv_end = p.seqlen_k;
  if (p.causal) kv_end = min(kv_end, min(q0 + kBlockM, p.seqlen_q) + p.causal_offset);
  const int num_tiles = kv_end > 0 ? (kv_end + kBlockN - 1) / kBlockN : 0;

  for (int tile = 0; tile < num_tiles; ++tile) {
    const int k0 = tile * kBlockN;
    const int keys_valid = min(kBlockN, p.seqlen_k - k0);
    __syncthreads();  // the previous tile's readers are done (and Q has landed)
    load_tile<D>(k_s, k, p.k_seq_stride, k0, keys_valid, tid);
    load_tile<D>(v_s, v, p.v_seq_stride, k0, keys_valid, tid);
    __syncthreads();

    // 4 x 8 scores per lane: rows ri + 4i, columns ci + 8j.
    float s[4][8] = {};
#pragma unroll 4
    for (int d = 0; d < D; d += 4) {
      float4 qv[4];
      float4 kv[8];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        qv[i] = *reinterpret_cast<const float4*>(q_rows + (ri + 4 * i) * kStride<D> + d);
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        kv[j] = *reinterpret_cast<const float4*>(k_s + (ci + 8 * j) * kStride<D> + d);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          s[i][j] += qv[i].x * kv[j].x;
          s[i][j] += qv[i].y * kv[j].y;
          s[i][j] += qv[i].z * kv[j].z;
          s[i][j] += qv[i].w * kv[j].w;
        }
      }
    }

    const bool needs_mask =
        keys_valid < kBlockN || (p.causal && k0 + kBlockN - 1 > warp_row0 + p.causal_offset);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int row = warp_row0 + ri + 4 * i;
      float tile_max = -INFINITY;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        s[i][j] *= p.scale_log2;
        if (needs_mask) {
          const int col = k0 + ci + 8 * j;
          if (col >= p.seqlen_k || (p.causal && col > row + p.causal_offset)) s[i][j] = -INFINITY;
        }
        tile_max = fmaxf(tile_max, s[i][j]);
      }
      // The 8 lanes of a row group differ only in their low three lane bits.
      tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 1));
      tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 2));
      tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 4));
      const float new_max = fmaxf(row_max[i], tile_max);
      const float ref = new_max == -INFINITY ? 0.f : new_max;
      const float rescale = exp2f(row_max[i] - ref);
      float tile_sum = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float prob = exp2f(s[i][j] - ref);
        tile_sum += prob;
        p_s[(ri + 4 * i) * kPStride + ci + 8 * j] = prob;
      }
      row_sum[i] = rescale * row_sum[i] + tile_sum;
      row_max[i] = new_max;
#pragma unroll
      for (int n = 0; n < D / 8; ++n) o_acc[i][n] *= rescale;
    }
    __syncwarp();

    // O += P V: this lane owns rows ri + 4i and columns 4 ci + 32 c4 + (0..3).
#pragma unroll 4
    for (int key = 0; key < kBlockN; ++key) {
      float prob[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) prob[i] = p_s[(ri + 4 * i) * kPStride + key];
#pragma unroll
      for (int c4 = 0; c4 < D / 32; ++c4) {
        const float4 vv =
            *reinterpret_cast<const float4*>(v_s + key * kStride<D> + 4 * ci + 32 * c4);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          o_acc[i][4 * c4 + 0] += prob[i] * vv.x;
          o_acc[i][4 * c4 + 1] += prob[i] * vv.y;
          o_acc[i][4 * c4 + 2] += prob[i] * vv.z;
          o_acc[i][4 * c4 + 3] += prob[i] * vv.w;
        }
      }
    }
  }

  // Rows that saw no key (causal with S_q > S_k) produce zeros and an LSE of -inf.
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float total = row_sum[i];
    total += __shfl_xor_sync(0xffffffff, total, 1);
    total += __shfl_xor_sync(0xffffffff, total, 2);
    total += __shfl_xor_sync(0xffffffff, total, 4);
    const int row = warp_row0 + ri + 4 * i;
    if (row >= p.seqlen_q) continue;
    const float inv = total > 0.f ? 1.f / total : 0.f;
    float* out_row = o + static_cast<int64_t>(row) * p.o_seq_stride;
#pragma unroll
    for (int c4 = 0; c4 < D / 32; ++c4) {
      *reinterpret_cast<float4*>(out_row + 4 * ci + 32 * c4) =
          make_float4(o_acc[i][4 * c4] * inv, o_acc[i][4 * c4 + 1] * inv,
                      o_acc[i][4 * c4 + 2] * inv, o_acc[i][4 * c4 + 3] * inv);
    }
    if (ci == 0) {
      const int64_t index = (static_cast<int64_t>(b) * p.heads + h) * p.seqlen_q + row;
      p.lse[index] = total > 0.f ? row_max[i] * static_cast<float>(M_LN2) + logf(total) : -INFINITY;
    }
  }
}

template <int D>
void launch(const AttnParams& p, cudaStream_t stream) {
  auto* kernel = fp32_regtile_kernel<D>;
  C10_CUDA_CHECK(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes<D>));
  const dim3 grid((p.seqlen_q + kBlockM - 1) / kBlockM, p.batch * p.heads);
  kernel<<<grid, kThreads, kSmemBytes<D>, stream>>>(p);
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> attention_fp32_regtile(const at::Tensor& q, const at::Tensor& k,
                                                          const at::Tensor& v, bool causal,
                                                          double softmax_scale) {
  check_attention_inputs(q, k, v, at::kFloat, {64, 128});
  check_16_byte_rows(q, "q");
  check_16_byte_rows(k, "k");
  check_16_byte_rows(v, "v");
  const at::cuda::OptionalCUDAGuard guard(q.device());
  auto [o, lse] = alloc_attention_outputs(q);
  const AttnParams p = make_attn_params(q, k, v, o, lse, causal, softmax_scale);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (p.head_dim == 64) {
    launch<64>(p, stream);
  } else {
    launch<128>(p, stream);
  }
  FLASH_LAB_LAUNCH_CHECK();
  return {o, lse};
}

}  // namespace flash_lab

TORCH_LIBRARY_IMPL(flash_lab, CUDA, m) {
  m.impl("attention_fp32_regtile", &flash_lab::attention_fp32_regtile);
}
