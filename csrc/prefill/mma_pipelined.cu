// The tensor-core kernel of mma.cu with the global-memory latency hidden.
//
// Nsight Compute on mma.cu showed warps stalled on long_scoreboard (waiting for global loads)
// for most of their cycles: each tile was loaded with plain loads and a barrier before any math
// could start. Here K and V go through a two-stage shared-memory ring filled with cp.async:
// the copies for tile j + 1 are issued before tile j is computed, so they land while the tensor
// cores work. cp.async also skips the round trip through registers, and its src-size form
// zero-fills rows past the end without a branch.
//
// Other changes from mma.cu:
//   - with causal masking the heaviest query blocks (the last rows) are launched first, so the
//     long blocks do not end up alone in the final wave;
//   - O is staged through shared memory and written with 16-byte stores instead of 4-byte ones.
// The per-tile math (scores, online softmax, P V) is the same code, from mma_tile.cuh.
//
// Shared memory: 2 stages x (K + V) x [64][D + 8], i.e. 68 KB at D = 128, so it is dynamic and
// opted in above the 48 KB default. Q is staged in the second stage's K slot before the loop.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "common/attn_host.h"
#include "prefill/mma_tile.cuh"

namespace flash_lab {
namespace {

using namespace mma_tile;
static_assert(kBlockM == kBlockN, "Q is staged in a K slot");

template <int D>
constexpr int kTileElems = kBlockN * kStride<D>;

template <typename T, int D>
constexpr int kSmemBytes = 4 * kTileElems<D> * static_cast<int>(sizeof(T));

// Issues cp.async copies of rows [row0, row0 + 64) of a [S, D] slice; rows past `rows_valid`
// are zero-filled. The caller commits the group.
template <typename T, int D>
__device__ __forceinline__ void load_tile_async(T* smem, const T* src, int64_t row_stride, int row0,
                                                int rows_valid, int tid) {
  constexpr int kVecs = D / 8;
#pragma unroll
  for (int i = tid; i < kBlockN * kVecs; i += kThreads) {
    const int r = i / kVecs;
    const int c = (i % kVecs) * 8;
    const bool valid = r < rows_valid;
    const T* from = valid ? src + static_cast<int64_t>(row0 + r) * row_stride + c : src;
    cp_async_16(smem + r * kStride<D> + c, from, valid);
  }
}

template <typename T, int D>
__global__ void __launch_bounds__(kThreads) mma_pipelined_kernel(const AttnParams p) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  T* smem = reinterpret_cast<T*>(smem_raw);
  // Stage s holds K at slot 2s and V at slot 2s + 1.
  auto k_slot = [&](int stage) { return smem + (2 * stage) * kTileElems<D>; };
  auto v_slot = [&](int stage) { return smem + (2 * stage + 1) * kTileElems<D>; };

  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int q_block = p.causal ? gridDim.x - 1 - blockIdx.x : blockIdx.x;
  const int q0 = q_block * kBlockM;
  const int b = blockIdx.y / p.heads;
  const int h = blockIdx.y % p.heads;
  const int h_kv = h / p.group;
  const T* q = static_cast<const T*>(p.q) + b * p.q_batch_stride + h * p.q_head_stride;
  const T* k = static_cast<const T*>(p.k) + b * p.k_batch_stride + h_kv * p.k_head_stride;
  const T* v = static_cast<const T*>(p.v) + b * p.v_batch_stride + h_kv * p.v_head_stride;
  T* o = static_cast<T*>(p.o) + b * p.o_batch_stride + h * p.o_head_stride;

  const int warp_row0 = q0 + warp * 16;
  int kv_end = p.seqlen_k;
  if (p.causal) kv_end = min(kv_end, min(q0 + kBlockM, p.seqlen_q) + p.causal_offset);
  const int num_tiles = kv_end > 0 ? (kv_end + kBlockN - 1) / kBlockN : 0;

  auto issue_tile = [&](int tile) {
    const int k0 = tile * kBlockN;
    const int keys_valid = min(kBlockN, p.seqlen_k - k0);
    load_tile_async<T, D>(k_slot(tile % 2), k, p.k_seq_stride, k0, keys_valid, tid);
    load_tile_async<T, D>(v_slot(tile % 2), v, p.v_seq_stride, k0, keys_valid, tid);
  };

  // Prologue: Q into stage 1's K slot, tile 0 into stage 0, as two commit groups.
  load_tile_async<T, D>(k_slot(1), q, p.q_seq_stride, q0, min(kBlockM, p.seqlen_q - q0), tid);
  cp_async_commit();
  if (num_tiles > 0) issue_tile(0);
  cp_async_commit();
  cp_async_wait<1>();  // Q has landed; tile 0 may still be in flight
  __syncthreads();
  uint32_t q_frag[D / 16][4];
  load_q_fragments<T, D>(q_frag, k_slot(1), warp, lane);
  __syncthreads();  // stage 1 is free for tile 1

  float o_acc[D / 8][4] = {};
  float row_max[2] = {-INFINITY, -INFINITY};
  float row_sum[2] = {0.f, 0.f};

  for (int tile = 0; tile < num_tiles; ++tile) {
    // Start the next tile's copies before waiting for this one. The commit is unconditional so
    // that "all but the newest group" is always this tile's group.
    if (tile + 1 < num_tiles) issue_tile(tile + 1);
    cp_async_commit();
    cp_async_wait<1>();
    __syncthreads();

    const int k0 = tile * kBlockN;
    float s[8][4];
    tile_scores<T, D>(s, q_frag, k_slot(tile % 2), lane);
    const bool needs_mask =
        k0 + kBlockN > p.seqlen_k || (p.causal && k0 + kBlockN - 1 > warp_row0 + p.causal_offset);
    scale_and_mask(s, p, needs_mask, warp_row0, k0, lane);
    online_softmax<D>(s, o_acc, row_max, row_sum);
    accumulate_pv<T, D>(o_acc, s, v_slot(tile % 2), lane);
    __syncthreads();  // this stage is refilled two tiles from now
  }

  float inv[2];
  finish_rows(row_sum, inv);
  store_lse(p, b, h, warp_row0, lane, row_max, row_sum);

  // Stage the warp's 16 x D output rows in shared memory (stage 0's K slot, free after the loop's
  // last barrier), then write them back with 16-byte stores.
  T* o_smem = k_slot(0) + warp * 16 * kStride<D>;
  const int g = lane / 4;
  const int t = lane % 4;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
#pragma unroll
    for (int n = 0; n < D / 8; ++n) {
      *reinterpret_cast<uint32_t*>(o_smem + (g + 8 * i) * kStride<D> + 8 * n + 2 * t) =
          pack2<T>(o_acc[n][2 * i] * inv[i], o_acc[n][2 * i + 1] * inv[i]);
    }
  }
  __syncwarp();
  constexpr int kVecs = D / 8;
#pragma unroll
  for (int i = lane; i < 16 * kVecs; i += 32) {
    const int r = i / kVecs;
    const int c = (i % kVecs) * 8;
    if (warp_row0 + r < p.seqlen_q) {
      *reinterpret_cast<uint4*>(o + static_cast<int64_t>(warp_row0 + r) * p.o_seq_stride + c) =
          *reinterpret_cast<const uint4*>(o_smem + r * kStride<D> + c);
    }
  }
}

template <typename T, int D>
void launch(const AttnParams& p, cudaStream_t stream) {
  auto* kernel = mma_pipelined_kernel<T, D>;
  C10_CUDA_CHECK(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes<T, D>));
  const dim3 grid((p.seqlen_q + kBlockM - 1) / kBlockM, p.batch * p.heads);
  kernel<<<grid, kThreads, kSmemBytes<T, D>, stream>>>(p);
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> attention_mma_pipelined(const at::Tensor& q, const at::Tensor& k,
                                                           const at::Tensor& v, bool causal,
                                                           double softmax_scale) {
  const at::ScalarType dtype = q.scalar_type();
  TORCH_CHECK(dtype == at::kBFloat16 || dtype == at::kHalf,
              "attention_mma_pipelined takes bf16 or fp16");
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

TORCH_LIBRARY_IMPL(flash_lab, CUDA, m) {
  m.impl("attention_mma_pipelined", &flash_lab::attention_mma_pipelined);
}
