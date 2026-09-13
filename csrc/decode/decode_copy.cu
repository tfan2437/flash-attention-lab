// Decode baseline in the style of the course-era kernel, kept on purpose as the "before".
//
// The host does what the original wrapper did every step: it finds the longest live sequence
// (a device-to-host sync), copies the live part of the K and V caches, and copies them again into
// a per-head [B, H_kv, L, D] layout. The kernel then runs one block per (sequence, head) over
// 32-key tiles staged in shared memory, with one thread per key for the scores and
// shared-memory tree reductions for the max and the sum.
//
// Nsight Systems shows the copies dominating the step; decode.cu reads the cache in place.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cmath>

#include "decode/decode_host.h"

namespace flash_lab {
namespace {

constexpr int kThreads = 128;
constexpr int kTile = 32;

template <typename T, int D>
__global__ void __launch_bounds__(kThreads) decode_copy_kernel(const DecodeParams p) {
  __shared__ float q_s[D];
  __shared__ float k_s[kTile][D + 1];
  __shared__ float v_s[kTile][D + 1];
  __shared__ float o_s[D];
  __shared__ float scores[kTile];
  __shared__ float reduce[kTile];
  __shared__ float run_max;
  __shared__ float run_sum;

  const int tid = threadIdx.x;
  const int b = blockIdx.x / p.heads;
  const int h = blockIdx.x % p.heads;
  const int h_kv = h / p.group;
  const int len = p.seq_lens[b];
  const T* q = static_cast<const T*>(p.q) + b * p.q_batch_stride + h * p.q_head_stride;
  const T* k = static_cast<const T*>(p.k) + b * p.k_batch_stride + h_kv * p.k_head_stride;
  const T* v = static_cast<const T*>(p.v) + b * p.v_batch_stride + h_kv * p.v_head_stride;

  for (int d = tid; d < D; d += kThreads) {
    q_s[d] = to_float(q[d]);
    o_s[d] = 0.f;
  }
  if (tid == 0) {
    run_max = -INFINITY;
    run_sum = 0.f;
  }
  __syncthreads();

  for (int k0 = 0; k0 < len; k0 += kTile) {
    for (int i = tid; i < kTile * D; i += kThreads) {
      const int c = i / D;
      const int d = i % D;
      const bool valid = k0 + c < len;
      k_s[c][d] = valid ? to_float(k[static_cast<int64_t>(k0 + c) * p.k_seq_stride + d]) : 0.f;
      v_s[c][d] = valid ? to_float(v[static_cast<int64_t>(k0 + c) * p.v_seq_stride + d]) : 0.f;
    }
    __syncthreads();

    if (tid < kTile) {
      float dot = 0.f;
      for (int d = 0; d < D; ++d) dot += q_s[d] * k_s[tid][d];
      scores[tid] = k0 + tid < len ? dot * p.scale : -INFINITY;
      reduce[tid] = scores[tid];
    }
    __syncthreads();
    for (int stride = kTile / 2; stride > 0; stride /= 2) {
      if (tid < stride) reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
      __syncthreads();
    }
    const float new_max = fmaxf(run_max, reduce[0]);  // the tile has at least one valid key
    const float rescale = expf(run_max - new_max);
    __syncthreads();

    if (tid < kTile) {
      scores[tid] = expf(scores[tid] - new_max);
      reduce[tid] = scores[tid];
    }
    __syncthreads();
    for (int stride = kTile / 2; stride > 0; stride /= 2) {
      if (tid < stride) reduce[tid] += reduce[tid + stride];
      __syncthreads();
    }
    if (tid == 0) {
      run_sum = rescale * run_sum + reduce[0];
      run_max = new_max;
    }
    for (int d = tid; d < D; d += kThreads) {
      float acc = rescale * o_s[d];
      for (int c = 0; c < kTile; ++c) acc += scores[c] * v_s[c][d];
      o_s[d] = acc;
    }
    __syncthreads();
  }

  T* o = static_cast<T*>(p.o) + b * p.o_batch_stride + h * p.o_head_stride;
  for (int d = tid; d < D; d += kThreads) o[d] = from_float<T>(o_s[d] / run_sum);
  if (tid == 0) p.lse[b * p.heads + h] = run_max + logf(run_sum);
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> decode_copy(const at::Tensor& q, const at::Tensor& k_cache,
                                               const at::Tensor& v_cache,
                                               const at::Tensor& seq_lens, double softmax_scale,
                                               int64_t /*num_splits*/) {
  check_decode_inputs(q, k_cache, v_cache, seq_lens);
  const at::cuda::OptionalCUDAGuard guard(q.device());

  // The copies this baseline is about.
  const int64_t live = seq_lens.max().item<int>();
  TORCH_CHECK(live >= 1 && live <= k_cache.size(1), "seq_lens must be in [1, S_max]");
  const at::Tensor k_live = k_cache.narrow(1, 0, live).contiguous();
  const at::Tensor v_live = v_cache.narrow(1, 0, live).contiguous();
  const at::Tensor k_heads = k_live.permute({0, 2, 1, 3}).contiguous();  // [B, H_kv, L, D]
  const at::Tensor v_heads = v_live.permute({0, 2, 1, 3}).contiguous();

  at::Tensor o = at::empty(q.sizes(), q.options());
  at::Tensor lse = at::empty({q.size(0), q.size(1)}, q.options().dtype(at::kFloat));
  DecodeParams p = make_decode_params(q, k_cache, v_cache, seq_lens, o, lse, softmax_scale);
  p.k = k_heads.data_ptr();
  p.v = v_heads.data_ptr();
  p.k_batch_stride = k_heads.stride(0);
  p.k_head_stride = k_heads.stride(1);
  p.k_seq_stride = k_heads.stride(2);
  p.v_batch_stride = v_heads.stride(0);
  p.v_head_stride = v_heads.stride(1);
  p.v_seq_stride = v_heads.stride(2);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dispatch_decode_dtype(q.scalar_type(), [&](auto tag) {
    using T = decltype(tag);
    if (p.head_dim == 64) {
      decode_copy_kernel<T, 64><<<p.batch * p.heads, kThreads, 0, stream>>>(p);
    } else {
      decode_copy_kernel<T, 128><<<p.batch * p.heads, kThreads, 0, stream>>>(p);
    }
  });
  FLASH_LAB_LAUNCH_CHECK();
  return {o, lse};
}

}  // namespace flash_lab

TORCH_LIBRARY_IMPL(flash_lab, CUDA, m) { m.impl("decode_copy", &flash_lab::decode_copy); }
