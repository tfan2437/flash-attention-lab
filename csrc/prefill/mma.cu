// FlashAttention-2 forward on tensor cores: bf16 or fp16 inputs, fp32 accumulation.
//
// A block of 4 warps owns 64 query rows and each warp owns 16 of them, one m16 tile of
// mma.sync.m16n8k16. Keys are processed in tiles of 64 (the per-tile math is in mma_tile.cuh):
//   S = Q K^T   The warp's Q fragments are loaded once and stay in registers. K is stored
//               [key][d], which is already the column-major B operand, so plain ldmatrix works.
//   softmax     Runs on the accumulator fragments. The four lanes of a quad share two rows, so
//               the row max takes two shuffles; exp2 with log2(e) folded into the scale. Each lane
//               keeps a partial row sum and the quad combines them once at the end.
//   O += P V    P never leaves registers: the fp32 fragments of two adjacent n8 score tiles,
//               packed to 16 bits, are exactly the A fragment of the next mma. V is stored
//               [key][d] and has to be transposed into the B operand, which ldmatrix.trans does.
// Tiles are loaded with plain 16-byte loads and a barrier, with no overlap between loading one
// tile and computing on the previous one; mma_pipelined.cu adds that overlap. Q is staged through
// the K buffer before the first tile.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "common/attn_host.h"
#include "prefill/mma_tile.cuh"

namespace flash_lab {
namespace {

using namespace mma_tile;
static_assert(kBlockM == kBlockN, "Q is staged through the K buffer");

// Copies rows [row0, row0 + 64) of a [S, D] slice into shared memory with 16-byte loads,
// zero-filling rows past `rows_valid`.
template <typename T, int D>
__device__ __forceinline__ void load_tile(T* smem, const T* src, int64_t row_stride, int row0,
                                          int rows_valid, int tid) {
  constexpr int kVecs = D / 8;
#pragma unroll
  for (int i = tid; i < kBlockN * kVecs; i += kThreads) {
    const int r = i / kVecs;
    const int c = (i % kVecs) * 8;
    uint4 val = make_uint4(0, 0, 0, 0);
    if (r < rows_valid) {
      val = *reinterpret_cast<const uint4*>(src + static_cast<int64_t>(row0 + r) * row_stride + c);
    }
    *reinterpret_cast<uint4*>(smem + r * kStride<D> + c) = val;
  }
}

template <typename T, int D>
__global__ void __launch_bounds__(kThreads) mma_attention_kernel(const AttnParams p) {
  __shared__ alignas(16) T smem[2 * kBlockN * kStride<D>];
  T* k_smem = smem;
  T* v_smem = smem + kBlockN * kStride<D>;

  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
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
  uint32_t q_frag[D / 16][4];
  load_q_fragments<T, D>(q_frag, k_smem, warp, lane);

  float o_acc[D / 8][4] = {};
  float row_max[2] = {-INFINITY, -INFINITY};
  float row_sum[2] = {0.f, 0.f};

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

    float s[8][4];
    tile_scores<T, D>(s, q_frag, k_smem, lane);
    const bool needs_mask =
        keys_valid < kBlockN || (p.causal && k0 + kBlockN - 1 > warp_row0 + p.causal_offset);
    scale_and_mask(s, p, needs_mask, warp_row0, k0, lane);
    online_softmax<D>(s, o_acc, row_max, row_sum);
    accumulate_pv<T, D>(o_acc, s, v_smem, lane);
  }

  float inv[2];
  finish_rows(row_sum, inv);
  const int g = lane / 4;
  const int t = lane % 4;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int row = warp_row0 + g + 8 * i;
    if (row >= p.seqlen_q) continue;
    T* out_row = o + static_cast<int64_t>(row) * p.o_seq_stride;
#pragma unroll
    for (int n = 0; n < D / 8; ++n) {
      *reinterpret_cast<uint32_t*>(out_row + 8 * n + 2 * t) =
          pack2<T>(o_acc[n][2 * i] * inv[i], o_acc[n][2 * i + 1] * inv[i]);
    }
  }
  store_lse(p, b, h, warp_row0, lane, row_max, row_sum);
}

template <typename T, int D>
void launch(const AttnParams& p, cudaStream_t stream) {
  const dim3 grid((p.seqlen_q + kBlockM - 1) / kBlockM, p.batch * p.heads);
  mma_attention_kernel<T, D><<<grid, kThreads, 0, stream>>>(p);
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
