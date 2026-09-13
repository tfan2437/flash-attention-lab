#pragma once

// Host-side helpers shared by the decode ops: checks at the op boundary, output allocation,
// DecodeParams setup, and element conversions used by the kernels.

#include <ATen/ATen.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <tuple>

#include "common/checks.h"
#include "common/params.h"

namespace flash_lab {

// q [B, H, D]; caches [B, S_max, H_kv, D]; seq_lens [B] int32.
inline void check_decode_inputs(const at::Tensor& q, const at::Tensor& k_cache,
                                const at::Tensor& v_cache, const at::Tensor& seq_lens) {
  FLASH_LAB_CHECK_CUDA(q);
  FLASH_LAB_CHECK_CUDA(k_cache);
  FLASH_LAB_CHECK_CUDA(v_cache);
  FLASH_LAB_CHECK_CUDA(seq_lens);
  const at::ScalarType dtype = q.scalar_type();
  TORCH_CHECK(dtype == at::kBFloat16 || dtype == at::kHalf || dtype == at::kFloat,
              "decode takes bf16, fp16, or fp32");
  FLASH_LAB_CHECK_DTYPE(k_cache, dtype);
  FLASH_LAB_CHECK_DTYPE(v_cache, dtype);
  FLASH_LAB_CHECK_DTYPE(seq_lens, at::kInt);
  TORCH_CHECK(q.dim() == 3 && k_cache.dim() == 4,
              "q must be [B, H, D], caches [B, S_max, H_kv, D]");
  TORCH_CHECK(k_cache.sizes() == v_cache.sizes(), "k_cache and v_cache must have the same shape");
  TORCH_CHECK(k_cache.size(0) == q.size(0) && k_cache.size(3) == q.size(2),
              "q and the caches must agree on batch size and head_dim");
  TORCH_CHECK(k_cache.size(2) > 0 && q.size(1) % k_cache.size(2) == 0,
              "query heads must be a multiple of KV heads");
  TORCH_CHECK(seq_lens.dim() == 1 && seq_lens.size(0) == q.size(0), "seq_lens must be [B]");
  TORCH_CHECK(q.size(2) == 64 || q.size(2) == 128, "unsupported head_dim ", q.size(2));
  FLASH_LAB_CHECK_LAST_DIM(q);
  FLASH_LAB_CHECK_LAST_DIM(k_cache);
  FLASH_LAB_CHECK_LAST_DIM(v_cache);
  TORCH_CHECK(q.size(0) * q.size(1) <= 65535 * 64, "batch * heads too large");
}

// The decode kernels read cache rows in 16-byte pieces.
inline void check_16_byte_cache_rows(const at::Tensor& cache, const char* name) {
  const int64_t elems = 16 / cache.element_size();
  bool aligned = reinterpret_cast<uintptr_t>(cache.data_ptr()) % 16 == 0;
  for (int dim = 0; dim < 3; ++dim)
    aligned &= cache.size(dim) == 1 || cache.stride(dim) % elems == 0;
  TORCH_CHECK(aligned, name, " must have 16-byte aligned rows (data pointer and strides)");
}

inline DecodeParams make_decode_params(const at::Tensor& q, const at::Tensor& k_cache,
                                       const at::Tensor& v_cache, const at::Tensor& seq_lens,
                                       at::Tensor& o, at::Tensor& lse, double scale) {
  DecodeParams p{};
  p.q = q.data_ptr();
  p.k = k_cache.data_ptr();
  p.v = v_cache.data_ptr();
  p.o = o.data_ptr();
  p.lse = lse.data_ptr<float>();
  p.seq_lens = seq_lens.data_ptr<int>();
  p.q_batch_stride = q.stride(0);
  p.q_head_stride = q.stride(1);
  p.k_batch_stride = k_cache.stride(0);
  p.k_seq_stride = k_cache.stride(1);
  p.k_head_stride = k_cache.stride(2);
  p.v_batch_stride = v_cache.stride(0);
  p.v_seq_stride = v_cache.stride(1);
  p.v_head_stride = v_cache.stride(2);
  p.o_batch_stride = o.stride(0);
  p.o_head_stride = o.stride(1);
  p.batch = static_cast<int>(q.size(0));
  p.heads = static_cast<int>(q.size(1));
  p.heads_kv = static_cast<int>(k_cache.size(2));
  p.head_dim = static_cast<int>(q.size(2));
  p.group = p.heads / p.heads_kv;
  p.max_seqlen = static_cast<int>(k_cache.size(1));
  p.scale = static_cast<float>(scale);
  p.scale_log2 = static_cast<float>(scale * M_LOG2E);
  p.num_splits = 1;
  return p;
}

__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename T>
__device__ __forceinline__ T from_float(float x);
template <>
__device__ __forceinline__ float from_float<float>(float x) {
  return x;
}
template <>
__device__ __forceinline__ __half from_float<__half>(float x) {
  return __float2half_rn(x);
}
template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
  return __float2bfloat16_rn(x);
}

template <int kBytes>
struct VecOf;
template <>
struct VecOf<4> {
  using type = uint32_t;
};
template <>
struct VecOf<8> {
  using type = uint2;
};
template <>
struct VecOf<16> {
  using type = uint4;
};

// Loads N consecutive elements of T as floats with the widest aligned vector loads.
template <typename T, int N>
__device__ __forceinline__ void load_floats(float (&dst)[N], const T* src) {
  constexpr int kBytes = N * static_cast<int>(sizeof(T));
  constexpr int kChunk = kBytes < 16 ? kBytes : 16;
  constexpr int kPerChunk = kChunk / static_cast<int>(sizeof(T));
  using Vec = typename VecOf<kChunk>::type;
#pragma unroll
  for (int c = 0; c < N / kPerChunk; ++c) {
    const Vec raw = reinterpret_cast<const Vec*>(src)[c];
    const T* vals = reinterpret_cast<const T*>(&raw);
#pragma unroll
    for (int j = 0; j < kPerChunk; ++j) dst[c * kPerChunk + j] = to_float(vals[j]);
  }
}

template <typename F>
void dispatch_decode_dtype(at::ScalarType dtype, F&& f) {
  if (dtype == at::kBFloat16) {
    f(__nv_bfloat16{});
  } else if (dtype == at::kHalf) {
    f(__half{});
  } else {
    f(float{});
  }
}

}  // namespace flash_lab
