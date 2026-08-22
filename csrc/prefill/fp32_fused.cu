// FlashAttention-2 forward on CUDA cores in fp32: the straightforward fused design, kept as the
// baseline that the tensor-core kernel is measured against.
//
// A block of 256 threads owns BR query rows of one (batch, head) and walks the keys in tiles of
// BC. Q, the current K/V tile, and the output accumulator live in shared memory, with rows padded
// to D + 1 floats so that threads reading the same column of different rows hit different banks.
// Each tile goes through four phases separated by barriers:
//   1. all threads load the K and V tile (keys past the end are zero-filled),
//   2. each thread computes one score (BR * BC == 256),
//   3. one thread per row updates the running max and sum (online softmax),
//   4. all threads rescale O and add P @ V.
// Dot products accumulate in order of d into a single fp32 register, so the result is
// deterministic. The tile shape can be switched with FLASH_LAB_FP32_TILE=32x8|16x16|8x32.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "common/attn_host.h"

namespace flash_lab {
namespace {

constexpr int kThreads = 256;

template <int D, int BR, int BC>
__global__ void __launch_bounds__(kThreads) fp32_fused_kernel(const AttnParams p) {
  static_assert(BR * BC == kThreads, "each thread computes exactly one score per tile");

  __shared__ float q_tile[BR][D + 1];
  __shared__ float k_tile[BC][D + 1];
  __shared__ float v_tile[BC][D + 1];
  __shared__ float o_acc[BR][D + 1];
  __shared__ float scores[BR][BC];  // scores, then probabilities after the softmax phase
  __shared__ float row_max[BR];
  __shared__ float row_sum[BR];
  __shared__ float row_rescale[BR];

  const int tid = threadIdx.x;
  const int q0 = blockIdx.x * BR;
  const int b = blockIdx.y / p.heads;
  const int h = blockIdx.y % p.heads;
  const int h_kv = h / p.group;

  const float* q = static_cast<const float*>(p.q) + b * p.q_batch_stride + h * p.q_head_stride;
  const float* k = static_cast<const float*>(p.k) + b * p.k_batch_stride + h_kv * p.k_head_stride;
  const float* v = static_cast<const float*>(p.v) + b * p.v_batch_stride + h_kv * p.v_head_stride;
  float* o = static_cast<float*>(p.o) + b * p.o_batch_stride + h * p.o_head_stride;

  for (int i = tid; i < BR * D; i += kThreads) {
    const int r = i / D;
    const int d = i % D;
    q_tile[r][d] = q0 + r < p.seqlen_q ? q[static_cast<int64_t>(q0 + r) * p.q_seq_stride + d] : 0.f;
    o_acc[r][d] = 0.f;
  }
  if (tid < BR) {
    row_max[tid] = -INFINITY;
    row_sum[tid] = 0.f;
  }
  __syncthreads();

  // With causal masking the last valid row of this block sees keys up to row + causal_offset;
  // tiles past that are skipped entirely.
  int kv_end = p.seqlen_k;
  if (p.causal) kv_end = min(kv_end, min(q0 + BR, p.seqlen_q) + p.causal_offset);
  const int num_tiles = kv_end > 0 ? (kv_end + BC - 1) / BC : 0;

  const int score_row = tid / BC;
  const int score_col = tid % BC;

  for (int t = 0; t < num_tiles; ++t) {
    const int k0 = t * BC;
    for (int i = tid; i < BC * D; i += kThreads) {
      const int c = i / D;
      const int d = i % D;
      const bool valid = k0 + c < p.seqlen_k;
      k_tile[c][d] = valid ? k[static_cast<int64_t>(k0 + c) * p.k_seq_stride + d] : 0.f;
      v_tile[c][d] = valid ? v[static_cast<int64_t>(k0 + c) * p.v_seq_stride + d] : 0.f;
    }
    __syncthreads();

    float s = 0.f;
    for (int d = 0; d < D; ++d) s += q_tile[score_row][d] * k_tile[score_col][d];
    const int key = k0 + score_col;
    const bool hidden = key >= p.seqlen_k || (p.causal && key > q0 + score_row + p.causal_offset);
    scores[score_row][score_col] = hidden ? -INFINITY : s * p.scale;
    __syncthreads();

    if (tid < BR) {
      const float old_max = row_max[tid];
      float new_max = old_max;
      for (int c = 0; c < BC; ++c) new_max = fmaxf(new_max, scores[tid][c]);
      // A row that has only seen hidden keys keeps a max of -inf. Using 0 as its reference
      // point makes every exponential below exactly 0 instead of nan.
      const float ref = new_max == -INFINITY ? 0.f : new_max;
      float tile_sum = 0.f;
      for (int c = 0; c < BC; ++c) {
        const float prob = expf(scores[tid][c] - ref);
        scores[tid][c] = prob;
        tile_sum += prob;
      }
      const float rescale = expf(old_max - ref);
      row_rescale[tid] = rescale;
      row_sum[tid] = rescale * row_sum[tid] + tile_sum;
      row_max[tid] = new_max;
    }
    __syncthreads();

    for (int i = tid; i < BR * D; i += kThreads) {
      const int r = i / D;
      const int d = i % D;
      float acc = row_rescale[r] * o_acc[r][d];
      for (int c = 0; c < BC; ++c) acc += scores[r][c] * v_tile[c][d];
      o_acc[r][d] = acc;
    }
    __syncthreads();
  }

  // Rows that saw no key (causal with S_q > S_k) produce zeros and an LSE of -inf.
  for (int i = tid; i < BR * D; i += kThreads) {
    const int r = i / D;
    const int d = i % D;
    if (q0 + r < p.seqlen_q) {
      const float sum = row_sum[r];
      o[static_cast<int64_t>(q0 + r) * p.o_seq_stride + d] = sum > 0.f ? o_acc[r][d] / sum : 0.f;
    }
  }
  if (tid < BR && q0 + tid < p.seqlen_q) {
    const float sum = row_sum[tid];
    const int64_t row = (static_cast<int64_t>(b) * p.heads + h) * p.seqlen_q + q0 + tid;
    p.lse[row] = sum > 0.f ? row_max[tid] + logf(sum) : -INFINITY;
  }
}

template <int D, int BR, int BC>
void launch(const AttnParams& p, cudaStream_t stream) {
  const dim3 grid((p.seqlen_q + BR - 1) / BR, p.batch * p.heads);
  fp32_fused_kernel<D, BR, BC><<<grid, kThreads, 0, stream>>>(p);
}

template <int D>
void launch_with_tile(const AttnParams& p, cudaStream_t stream) {
  const char* tile = std::getenv("FLASH_LAB_FP32_TILE");
  if (tile == nullptr || std::strcmp(tile, "32x8") == 0) {
    launch<D, 32, 8>(p, stream);
  } else if (std::strcmp(tile, "16x16") == 0) {
    launch<D, 16, 16>(p, stream);
  } else if (std::strcmp(tile, "8x32") == 0) {
    launch<D, 8, 32>(p, stream);
  } else {
    TORCH_CHECK(false, "FLASH_LAB_FP32_TILE must be 32x8, 16x16, or 8x32, got ", tile);
  }
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> attention_fp32_fused(const at::Tensor& q, const at::Tensor& k,
                                                        const at::Tensor& v, bool causal,
                                                        double softmax_scale) {
  check_attention_inputs(q, k, v, at::kFloat, {64, 128});
  const at::cuda::OptionalCUDAGuard guard(q.device());
  auto [o, lse] = alloc_attention_outputs(q);
  const AttnParams p = make_attn_params(q, k, v, o, lse, causal, softmax_scale);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (p.head_dim == 64) {
    launch_with_tile<64>(p, stream);
  } else {
    launch_with_tile<128>(p, stream);
  }
  FLASH_LAB_LAUNCH_CHECK();
  return {o, lse};
}

}  // namespace flash_lab

TORCH_LIBRARY_IMPL(flash_lab, CUDA, m) {
  m.impl("attention_fp32_fused", &flash_lab::attention_fp32_fused);
}
