// FlashAttention-2 forward on tensor cores: bf16 or fp16 inputs, fp32 accumulation.
//
// A block of 4 warps owns 64 query rows and each warp owns 16 of them, one m16 tile of
// mma.sync.m16n8k16. Keys are processed in tiles of 64:
//   S = Q K^T   The warp's Q fragments are loaded once and stay in registers. K is stored
//               [key][d], which is already the column-major B operand, so plain ldmatrix works.
//   softmax     Runs on the accumulator fragments. The four lanes of a quad share two rows, so
//               the row max takes two shuffles; exp2 with log2(e) folded into the scale. Each lane
//               keeps a partial row sum and the quad combines them once at the end.
//   O += P V    P never leaves registers: the fp32 fragments of two adjacent n8 score tiles,
//               packed to 16 bits, are exactly the A fragment of the next mma. V is stored
//               [key][d] and has to be transposed into the B operand, which ldmatrix.trans does.
// Shared-memory rows are padded by 8 elements (16 bytes), so the eight row addresses of every
// ldmatrix land in different banks. Q is staged through the K buffer before the first tile.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cmath>

#include "common/attn_host.h"
#include "common/ptx.cuh"

namespace flash_lab {
namespace {

constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
constexpr int kBlockM = 16 * kWarps;  // query rows per block
constexpr int kBlockN = 64;           // keys per tile
static_assert(kBlockM == kBlockN, "Q is staged through the K buffer");

// Copies rows [row0, row0 + 64) of a [S, D] slice into shared memory with 16-byte loads,
// zero-filling rows past `rows_valid`.
template <typename T, int D>
__device__ __forceinline__ void load_tile(T* smem, const T* src, int64_t row_stride, int row0,
                                          int rows_valid, int tid) {
  constexpr int kStride = D + 8;
  constexpr int kVecs = D / 8;
#pragma unroll
  for (int i = tid; i < kBlockN * kVecs; i += kThreads) {
    const int r = i / kVecs;
    const int c = (i % kVecs) * 8;
    uint4 val = make_uint4(0, 0, 0, 0);
    if (r < rows_valid) {
      val = *reinterpret_cast<const uint4*>(src + static_cast<int64_t>(row0 + r) * row_stride + c);
    }
    *reinterpret_cast<uint4*>(smem + r * kStride + c) = val;
  }
}

template <typename T, int D>
__global__ void __launch_bounds__(kThreads) mma_attention_kernel(const AttnParams p) {
  constexpr int kStride = D + 8;
  constexpr int kSteps = D / 16;  // k-steps of Q K^T
  constexpr int kDTiles = D / 8;  // n8 tiles of the output accumulator

  __shared__ alignas(16) T smem[2 * kBlockN * kStride];
  T* k_smem = smem;
  T* v_smem = smem + kBlockN * kStride;

  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int g = lane / 4;  // fragment row within the warp's 16 rows: g and g + 8
  const int t = lane % 4;  // position within the quad that shares those rows

  const int q0 = blockIdx.x * kBlockM;
  const int b = blockIdx.y / p.heads;
  const int h = blockIdx.y % p.heads;
  const int h_kv = h / p.group;
  const T* q = static_cast<const T*>(p.q) + b * p.q_batch_stride + h * p.q_head_stride;
  const T* k = static_cast<const T*>(p.k) + b * p.k_batch_stride + h_kv * p.k_head_stride;
  const T* v = static_cast<const T*>(p.v) + b * p.v_batch_stride + h_kv * p.v_head_stride;
  T* o = static_cast<T*>(p.o) + b * p.o_batch_stride + h * p.o_head_stride;

  load_tile<T, D>(k_smem, q, p.q_seq_stride, q0, min(kBlockM, p.seqlen_q - q0), tid);
  __syncthreads();
  uint32_t q_frag[kSteps][4];
#pragma unroll
  for (int ks = 0; ks < kSteps; ++ks) {
    ldmatrix_x4(q_frag[ks], k_smem + (warp * 16 + lane % 16) * kStride + ks * 16 + (lane / 16) * 8);
  }

  float o_acc[kDTiles][4] = {};
  float row_max[2] = {-INFINITY, -INFINITY};  // rows g and g + 8, log2 domain
  float row_sum[2] = {0.f, 0.f};              // this lane's share of the row sums

  const int warp_row0 = q0 + warp * 16;
  int kv_end = p.seqlen_k;
  if (p.causal) kv_end = min(kv_end, min(q0 + kBlockM, p.seqlen_q) + p.causal_offset);
  const int num_tiles = kv_end > 0 ? (kv_end + kBlockN - 1) / kBlockN : 0;

  for (int tile = 0; tile < num_tiles; ++tile) {
    const int k0 = tile * kBlockN;
    const int keys_valid = min(kBlockN, p.seqlen_k - k0);
    __syncthreads();  // every warp is done reading the buffers (and the staged Q)
    load_tile<T, D>(k_smem, k, p.k_seq_stride, k0, keys_valid, tid);
    load_tile<T, D>(v_smem, v, p.v_seq_stride, k0, keys_valid, tid);
    __syncthreads();

    // S = Q K^T: 16 rows x 64 keys = 8 n8 tiles. One ldmatrix.x4 feeds two n8 tiles.
    float s[8][4] = {};
#pragma unroll
    for (int ks = 0; ks < kSteps; ++ks) {
#pragma unroll
      for (int pair = 0; pair < 4; ++pair) {
        uint32_t kb[4];
        const int key = pair * 16 + lane % 8 + 8 * (lane / 16);
        ldmatrix_x4(kb, k_smem + key * kStride + ks * 16 + 8 * ((lane / 8) % 2));
        mma_16816<T>(s[2 * pair], q_frag[ks], kb[0], kb[1]);
        mma_16816<T>(s[2 * pair + 1], q_frag[ks], kb[2], kb[3]);
      }
    }

    // Scale into the log2 domain; mask keys past the end and above the causal diagonal.
    const bool needs_mask =
        keys_valid < kBlockN || (p.causal && k0 + kBlockN - 1 > warp_row0 + p.causal_offset);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        s[j][e] *= p.scale_log2;
        if (needs_mask) {
          const int row = warp_row0 + g + (e >= 2 ? 8 : 0);
          const int col = k0 + 8 * j + 2 * t + (e & 1);
          if (col >= p.seqlen_k || (p.causal && col > row + p.causal_offset)) {
            s[j][e] = -INFINITY;
          }
        }
      }
    }

    // Online softmax for rows g (elements 0, 1) and g + 8 (elements 2, 3).
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
      tile_max[i] = fmaxf(tile_max[i], __shfl_xor_sync(0xffffffff, tile_max[i], 1));
      tile_max[i] = fmaxf(tile_max[i], __shfl_xor_sync(0xffffffff, tile_max[i], 2));
      const float new_max = fmaxf(row_max[i], tile_max[i]);
      // A row that has only seen masked keys keeps a max of -inf; 0 as the reference point
      // makes its exponentials exactly 0 instead of nan.
      ref[i] = new_max == -INFINITY ? 0.f : new_max;
      rescale[i] = exp2f(row_max[i] - ref[i]);
      row_max[i] = new_max;
    }
    float tile_sum[2] = {0.f, 0.f};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      s[j][0] = exp2f(s[j][0] - ref[0]);
      s[j][1] = exp2f(s[j][1] - ref[0]);
      s[j][2] = exp2f(s[j][2] - ref[1]);
      s[j][3] = exp2f(s[j][3] - ref[1]);
      tile_sum[0] += s[j][0] + s[j][1];
      tile_sum[1] += s[j][2] + s[j][3];
    }
    row_sum[0] = rescale[0] * row_sum[0] + tile_sum[0];
    row_sum[1] = rescale[1] * row_sum[1] + tile_sum[1];
#pragma unroll
    for (int n = 0; n < kDTiles; ++n) {
      o_acc[n][0] *= rescale[0];
      o_acc[n][1] *= rescale[0];
      o_acc[n][2] *= rescale[1];
      o_acc[n][3] *= rescale[1];
    }

    // O += P V, 16 keys per k-step. Score tiles 2ks and 2ks + 1 form the A fragment.
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
                          v_smem + (ks * 16 + lane % 16) * kStride + pair * 16 + 8 * (lane / 16));
        mma_16816<T>(o_acc[2 * pair], a, vb[0], vb[1]);
        mma_16816<T>(o_acc[2 * pair + 1], a, vb[2], vb[3]);
      }
    }
  }

  // Each row's sum is spread over the four lanes of its quad.
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    row_sum[i] += __shfl_xor_sync(0xffffffff, row_sum[i], 1);
    row_sum[i] += __shfl_xor_sync(0xffffffff, row_sum[i], 2);
  }

  // Rows that saw no key (causal with S_q > S_k) produce zeros and an LSE of -inf.
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int row = warp_row0 + g + 8 * i;
    if (row >= p.seqlen_q) continue;
    const float inv = row_sum[i] > 0.f ? 1.f / row_sum[i] : 0.f;
    T* out_row = o + static_cast<int64_t>(row) * p.o_seq_stride;
#pragma unroll
    for (int n = 0; n < kDTiles; ++n) {
      *reinterpret_cast<uint32_t*>(out_row + 8 * n + 2 * t) =
          pack2<T>(o_acc[n][2 * i] * inv, o_acc[n][2 * i + 1] * inv);
    }
    if (t == 0) {
      const int64_t lse_index = (static_cast<int64_t>(b) * p.heads + h) * p.seqlen_q + row;
      p.lse[lse_index] =
          row_sum[i] > 0.f ? row_max[i] * static_cast<float>(M_LN2) + logf(row_sum[i]) : -INFINITY;
    }
  }
}

template <typename T, int D>
void launch(const AttnParams& p, cudaStream_t stream) {
  const dim3 grid((p.seqlen_q + kBlockM - 1) / kBlockM, p.batch * p.heads);
  mma_attention_kernel<T, D><<<grid, kThreads, 0, stream>>>(p);
}

// The tile loads read 16 bytes at a time, so every row must start on a 16-byte boundary.
void check_16_byte_rows(const at::Tensor& x, const char* name) {
  bool aligned = reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0;
  for (int dim = 0; dim < 3; ++dim) aligned &= x.size(dim) == 1 || x.stride(dim) % 8 == 0;
  TORCH_CHECK(aligned, name, " must have 16-byte aligned rows (data pointer and strides)");
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> attention_mma(const at::Tensor& q, const at::Tensor& k,
                                                 const at::Tensor& v, bool causal,
                                                 double softmax_scale) {
  const at::ScalarType dtype = q.scalar_type();
  TORCH_CHECK(dtype == at::kBFloat16 || dtype == at::kHalf, "attention_mma takes bf16 or fp16");
  check_attention_inputs(q, k, v, dtype, {64, 128});
  check_16_byte_rows(q, "q");
  check_16_byte_rows(k, "k");
  check_16_byte_rows(v, "v");
  const at::cuda::OptionalCUDAGuard guard(q.device());
  auto [o, lse] = alloc_attention_outputs(q);
  const AttnParams p = make_attn_params(q, k, v, o, lse, causal, softmax_scale);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (dtype == at::kBFloat16) {
    p.head_dim == 64 ? launch<__nv_bfloat16, 64>(p, stream) : launch<__nv_bfloat16, 128>(p, stream);
  } else {
    p.head_dim == 64 ? launch<__half, 64>(p, stream) : launch<__half, 128>(p, stream);
  }
  FLASH_LAB_LAUNCH_CHECK();
  return {o, lse};
}

}  // namespace flash_lab

TORCH_LIBRARY_IMPL(flash_lab, CUDA, m) { m.impl("attention_mma", &flash_lab::attention_mma); }
