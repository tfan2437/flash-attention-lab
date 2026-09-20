// Unfused attention in fp32, the v0 baseline: the same FLOPs as the fused kernels, but the
// [S_q, S_k] score matrix of every (batch, head) goes through HBM.
//
// Four launches per call:
//   1. gemm (NT):  S = Q K^T, 16 x 16 shared-memory tiles, one output per thread.
//   2. scale_mask: S *= softmax_scale, -inf above the (bottom-right) causal diagonal.
//   3. softmax:    one block per row; max and sum with warp shuffles and a shared-memory step,
//                  then P = exp(S - max) / sum written in place, and the row's LSE.
//   4. gemm (NN):  O = P V, written straight into the [B, S_q, H, D] output.
// The GEMMs read Q, K, V, and O through their strides, so no layout copies are made. Tiles are
// padded to [16][17] so that threads reading one column of a tile hit 16 different banks.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cmath>

#include "common/attn_host.h"

namespace flash_lab {
namespace {

constexpr int kTile = 16;

// One batched matrix operand: element (z, row, col) of a [batch * heads] stack of matrices.
// `heads_div` maps the stack index's head to the operand's head (the GQA group for K and V).
struct Operand {
  float* base;
  int64_t batch_stride, head_stride, row_stride;  // the column stride is 1
  int heads, heads_div;

  __device__ float* at(int z, int64_t row, int col) const {
    const int b = z / heads;
    const int h = (z % heads) / heads_div;
    return base + b * batch_stride + h * head_stride + row * row_stride + col;
  }
};

// C (M x N) = A (M x K) * B, with B given as N x K rows (kTransB, as K for Q K^T) or as K x N
// rows (as V for P V). Out-of-range elements load as 0 and are not stored.
template <bool kTransB>
__global__ void __launch_bounds__(kTile* kTile)
    gemm_kernel(Operand a, Operand b, Operand c, int m, int n, int k) {
  __shared__ float a_s[kTile][kTile + 1];
  __shared__ float b_s[kTile][kTile + 1];
  const int z = blockIdx.z;
  const int ty = threadIdx.y;
  const int tx = threadIdx.x;
  const int row = blockIdx.y * kTile + ty;
  const int col = blockIdx.x * kTile + tx;

  float acc = 0.f;
  for (int k0 = 0; k0 < k; k0 += kTile) {
    a_s[ty][tx] = row < m && k0 + tx < k ? *a.at(z, row, k0 + tx) : 0.f;
    if (kTransB) {
      // b_s[kk][j] = B^T element (k0 + kk, col block + j) = B row (col block + j), column k0 + kk
      const int brow = blockIdx.x * kTile + ty;
      b_s[tx][ty] = brow < n && k0 + tx < k ? *b.at(z, brow, k0 + tx) : 0.f;
    } else {
      b_s[ty][tx] = k0 + ty < k && col < n ? *b.at(z, k0 + ty, col) : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < kTile; ++kk) acc += a_s[ty][kk] * b_s[kk][tx];
    __syncthreads();
  }
  if (row < m && col < n) *c.at(z, row, col) = acc;
}

__global__ void scale_mask_kernel(float* s, int64_t total, int seqlen_q, int seqlen_k, float scale,
                                  bool causal, int causal_offset) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int col = static_cast<int>(i % seqlen_k);
  const int row = static_cast<int>((i / seqlen_k) % seqlen_q);
  s[i] = causal && col > row + causal_offset ? -INFINITY : s[i] * scale;
}

constexpr int kSoftmaxThreads = 256;

// Reduces `value` over the block with warp shuffles and one shared-memory step.
template <bool kMax>
__device__ float block_reduce(float value, float* scratch) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    const float other = __shfl_xor_sync(0xffffffff, value, offset);
    value = kMax ? fmaxf(value, other) : value + other;
  }
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  value = kMax ? -INFINITY : 0.f;
  for (int w = 0; w < kSoftmaxThreads / 32; ++w)
    value = kMax ? fmaxf(value, scratch[w]) : value + scratch[w];
  __syncthreads();  // scratch is reused by the next reduction
  return value;
}

// One block per score row: P = exp(S - max) / sum in place, and LSE = max + log(sum).
// Rows with no visible key become zeros with an LSE of -inf.
__global__ void __launch_bounds__(kSoftmaxThreads)
    softmax_kernel(float* s, float* lse, int seqlen_k) {
  __shared__ float scratch[kSoftmaxThreads / 32];
  float* row = s + static_cast<int64_t>(blockIdx.x) * seqlen_k;
  float m = -INFINITY;
  for (int j = threadIdx.x; j < seqlen_k; j += kSoftmaxThreads) m = fmaxf(m, row[j]);
  m = block_reduce<true>(m, scratch);
  if (m == -INFINITY) {
    for (int j = threadIdx.x; j < seqlen_k; j += kSoftmaxThreads) row[j] = 0.f;
    if (threadIdx.x == 0) lse[blockIdx.x] = -INFINITY;
    return;
  }
  float sum = 0.f;
  for (int j = threadIdx.x; j < seqlen_k; j += kSoftmaxThreads) sum += expf(row[j] - m);
  sum = block_reduce<false>(sum, scratch);
  const float inv = 1.f / sum;
  for (int j = threadIdx.x; j < seqlen_k; j += kSoftmaxThreads) row[j] = expf(row[j] - m) * inv;
  if (threadIdx.x == 0) lse[blockIdx.x] = m + logf(sum);
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> attention_naive(const at::Tensor& q, const at::Tensor& k,
                                                   const at::Tensor& v, bool causal,
                                                   double softmax_scale) {
  check_attention_inputs(q, k, v, at::kFloat, {64, 128});
  const at::cuda::OptionalCUDAGuard guard(q.device());
  auto [o, lse] = alloc_attention_outputs(q);
  const int batch = static_cast<int>(q.size(0));
  const int seqlen_q = static_cast<int>(q.size(1));
  const int heads = static_cast<int>(q.size(2));
  const int head_dim = static_cast<int>(q.size(3));
  const int seqlen_k = static_cast<int>(k.size(1));
  const int group = heads / static_cast<int>(k.size(2));
  const int stacks = batch * heads;

  // The materialized score matrix, one [S_q, S_k] slab per (batch, head).
  at::Tensor s = at::empty({stacks, seqlen_q, seqlen_k}, q.options());
  const int64_t slab = static_cast<int64_t>(seqlen_q) * seqlen_k;
  const Operand q_op{q.data_ptr<float>(), q.stride(0), q.stride(2), q.stride(1), heads, 1};
  const Operand k_op{k.data_ptr<float>(), k.stride(0), k.stride(2), k.stride(1), heads, group};
  const Operand v_op{v.data_ptr<float>(), v.stride(0), v.stride(2), v.stride(1), heads, group};
  const Operand o_op{o.data_ptr<float>(), o.stride(0), o.stride(2), o.stride(1), heads, 1};
  // The score slabs are indexed by z directly: treat them as heads of a single batch entry.
  const Operand s_op{s.data_ptr<float>(), 0, slab, seqlen_k, stacks, 1};

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 block(kTile, kTile);
  gemm_kernel<true><<<dim3((seqlen_k + kTile - 1) / kTile, (seqlen_q + kTile - 1) / kTile, stacks),
                      block, 0, stream>>>(q_op, k_op, s_op, seqlen_q, seqlen_k, head_dim);
  FLASH_LAB_LAUNCH_CHECK();
  const int64_t total = static_cast<int64_t>(stacks) * slab;
  scale_mask_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(
      s.data_ptr<float>(), total, seqlen_q, seqlen_k, static_cast<float>(softmax_scale), causal,
      seqlen_k - seqlen_q);
  FLASH_LAB_LAUNCH_CHECK();
  softmax_kernel<<<stacks * seqlen_q, kSoftmaxThreads, 0, stream>>>(
      s.data_ptr<float>(), lse.data_ptr<float>(), seqlen_k);
  FLASH_LAB_LAUNCH_CHECK();
  gemm_kernel<false><<<dim3((head_dim + kTile - 1) / kTile, (seqlen_q + kTile - 1) / kTile, stacks),
                       block, 0, stream>>>(s_op, v_op, o_op, seqlen_q, head_dim, seqlen_k);
  FLASH_LAB_LAUNCH_CHECK();
  return {o, lse};
}

}  // namespace flash_lab

TORCH_LIBRARY_IMPL(flash_lab, CUDA, m) { m.impl("attention_naive", &flash_lab::attention_naive); }
