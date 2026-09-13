// Decode attention: each sequence's single query token against its first seq_lens[b] cached
// keys, read in place from a [B, S_max, H_kv, D] cache through strides (no copies).
//
// Grid: (B * H / kHeads, num_splits). A block handles kHeads consecutive query heads that share
// one KV head (kHeads divides the GQA group), so every K and V row it loads serves all of them,
// and one contiguous slice of the sequence's keys. Its 4 warps take 32-key chunks of that slice
// in turn, each keeping its own running max, sum, and output; the warps are combined through
// shared memory at the end.
//
// Within a warp the two matmuls use different lane mappings:
//   Q K^T   8-lane groups: the group's lanes load one key row together (16-byte pieces,
//           coalesced) and reduce their partial dot products with three shuffles, so the warp
//           scores 4 keys per step.
//   P V     lane-owns-columns: each lane accumulates D / 32 output columns over the chunk, so a
//           V row is one coalesced warp-wide load, and the probabilities come from shared
//           memory as broadcasts. Owning keys here would need a cross-lane sum per column.
//
// decode_inplace runs this with one split; when B * H is small that leaves most SMs idle.
// decode_splitkv (flash-decoding) gives each split its own blocks and merges the partial
// results with a log-sum-exp in a second, deterministic kernel.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cmath>
#include <cstdlib>

#include "decode/decode_host.h"

namespace flash_lab {
namespace {

constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
constexpr int kChunk = 32;      // keys per warp iteration
constexpr int kGroupLanes = 8;  // lanes scoring one key together
constexpr int kKeysPerStep = 32 / kGroupLanes;

template <typename T, int D, int kHeads>
__global__ void __launch_bounds__(kThreads) decode_kernel(const DecodeParams p) {
  constexpr int E = D / kGroupLanes;               // elements per lane in Q K^T
  constexpr int C = D / 32;                        // output columns per lane in P V
  __shared__ float probs[kWarps][kHeads][kChunk];  // scores, then probabilities
  __shared__ float warp_o[kWarps][kHeads][D];
  __shared__ float warp_m[kWarps][kHeads];
  __shared__ float warp_l[kWarps][kHeads];

  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int head_tiles = p.heads / kHeads;
  const int b = blockIdx.x / head_tiles;
  const int h0 = (blockIdx.x % head_tiles) * kHeads;
  const int h_kv = h0 / p.group;
  const int split = blockIdx.y;

  // This split's keys: a contiguous range, a whole number of chunks long.
  const int len = p.seq_lens[b];
  const int per_split = ((len + p.num_splits - 1) / p.num_splits + kChunk - 1) / kChunk * kChunk;
  const int k_begin = split * per_split;
  const int k_end = min(len, k_begin + per_split);

  const T* k = static_cast<const T*>(p.k) + b * p.k_batch_stride + h_kv * p.k_head_stride;
  const T* v = static_cast<const T*>(p.v) + b * p.v_batch_stride + h_kv * p.v_head_stride;

  // This lane's E elements of each query head, scaled into the log2 domain.
  float q_reg[kHeads][E];
#pragma unroll
  for (int g = 0; g < kHeads; ++g) {
    const T* q = static_cast<const T*>(p.q) + b * p.q_batch_stride + (h0 + g) * p.q_head_stride;
    load_floats<T, E>(q_reg[g], q + (lane % kGroupLanes) * E);
#pragma unroll
    for (int e = 0; e < E; ++e) q_reg[g][e] *= p.scale_log2;
  }

  float o_acc[kHeads][C] = {};
  float row_max[kHeads];  // warp-uniform
  float row_sum[kHeads];  // this lane's share
#pragma unroll
  for (int g = 0; g < kHeads; ++g) {
    row_max[g] = -INFINITY;
    row_sum[g] = 0.f;
  }

  const int num_chunks = k_end > k_begin ? (k_end - k_begin + kChunk - 1) / kChunk : 0;
  for (int c = warp; c < num_chunks; c += kWarps) {
    const int key0 = k_begin + c * kChunk;

#pragma unroll
    for (int step = 0; step < kChunk / kKeysPerStep; ++step) {
      const int j = step * kKeysPerStep + lane / kGroupLanes;
      const bool valid = key0 + j < k_end;
      float kv[E];
      if (valid) {
        load_floats<T, E>(
            kv, k + static_cast<int64_t>(key0 + j) * p.k_seq_stride + (lane % kGroupLanes) * E);
      } else {
#pragma unroll
        for (int e = 0; e < E; ++e) kv[e] = 0.f;
      }
#pragma unroll
      for (int g = 0; g < kHeads; ++g) {
        float dot = 0.f;
#pragma unroll
        for (int e = 0; e < E; ++e) dot += q_reg[g][e] * kv[e];
        dot += __shfl_xor_sync(0xffffffff, dot, 1);
        dot += __shfl_xor_sync(0xffffffff, dot, 2);
        dot += __shfl_xor_sync(0xffffffff, dot, 4);
        if (lane % kGroupLanes == 0) probs[warp][g][j] = valid ? dot : -INFINITY;
      }
    }
    __syncwarp();

    // Online softmax over the chunk, one key per lane.
#pragma unroll
    for (int g = 0; g < kHeads; ++g) {
      const float s = probs[warp][g][lane];
      float chunk_max = s;
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2) {
        chunk_max = fmaxf(chunk_max, __shfl_xor_sync(0xffffffff, chunk_max, offset));
      }
      const float new_max = fmaxf(row_max[g], chunk_max);
      const float ref = new_max == -INFINITY ? 0.f : new_max;
      const float rescale = exp2f(row_max[g] - ref);
      const float prob = exp2f(s - ref);
      row_sum[g] = rescale * row_sum[g] + prob;
#pragma unroll
      for (int cc = 0; cc < C; ++cc) o_acc[g][cc] *= rescale;
      row_max[g] = new_max;
      probs[warp][g][lane] = prob;
    }
    __syncwarp();

    const int keys_here = min(kChunk, k_end - key0);
#pragma unroll
    for (int j = 0; j < kChunk; ++j) {
      if (j < keys_here) {
        float vv[C];
        load_floats<T, C>(vv, v + static_cast<int64_t>(key0 + j) * p.v_seq_stride + lane * C);
#pragma unroll
        for (int g = 0; g < kHeads; ++g) {
          const float prob = probs[warp][g][j];
#pragma unroll
          for (int cc = 0; cc < C; ++cc) o_acc[g][cc] += prob * vv[cc];
        }
      }
    }
    __syncwarp();  // the next chunk overwrites probs
  }

  // Per-warp totals, then combine the 4 warps.
#pragma unroll
  for (int g = 0; g < kHeads; ++g) {
    float total = row_sum[g];
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      total += __shfl_xor_sync(0xffffffff, total, offset);
    }
    if (lane == 0) {
      warp_m[warp][g] = row_max[g];
      warp_l[warp][g] = total;
    }
#pragma unroll
    for (int cc = 0; cc < C; ++cc) warp_o[warp][g][lane * C + cc] = o_acc[g][cc];
  }
  __syncthreads();

  for (int idx = tid; idx < kHeads * D; idx += kThreads) {
    const int g = idx / D;
    const int d = idx % D;
    float m = -INFINITY;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) m = fmaxf(m, warp_m[w][g]);
    float num = 0.f;
    float den = 0.f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      if (warp_m[w][g] != -INFINITY) {  // warps without keys contribute nothing
        const float f = exp2f(warp_m[w][g] - m);
        num += warp_o[w][g][d] * f;
        den += warp_l[w][g] * f;
      }
    }
    const int h = h0 + g;
    if (p.num_splits == 1) {
      T* o = static_cast<T*>(p.o) + b * p.o_batch_stride + h * p.o_head_stride;
      o[d] = from_float<T>(den > 0.f ? num / den : 0.f);
      if (d == 0) {
        p.lse[b * p.heads + h] = den > 0.f ? m * static_cast<float>(M_LN2) + logf(den) : -INFINITY;
      }
    } else {
      const int64_t slot = (static_cast<int64_t>(b) * p.heads + h) * p.num_splits + split;
      p.partial_o[slot * D + d] = num;
      if (d == 0) {
        p.partial_ml[2 * slot] = m;
        p.partial_ml[2 * slot + 1] = den;
      }
    }
  }
}

// Combines the splits of one (sequence, head) in a fixed order: m = max m_s,
// l = sum l_s 2^(m_s - m), o = sum o_s 2^(m_s - m) / l. One thread per output column.
template <typename T>
__global__ void merge_kernel(const DecodeParams p) {
  const int bh = blockIdx.x;
  const int d = threadIdx.x;
  const int64_t base = static_cast<int64_t>(bh) * p.num_splits;
  float m = -INFINITY;
  for (int s = 0; s < p.num_splits; ++s) m = fmaxf(m, p.partial_ml[2 * (base + s)]);
  float num = 0.f;
  float den = 0.f;
  for (int s = 0; s < p.num_splits; ++s) {
    const float ms = p.partial_ml[2 * (base + s)];
    if (ms != -INFINITY) {  // splits past the end of the sequence are empty
      const float f = exp2f(ms - m);
      num += p.partial_o[(base + s) * p.head_dim + d] * f;
      den += p.partial_ml[2 * (base + s) + 1] * f;
    }
  }
  const int b = bh / p.heads;
  const int h = bh % p.heads;
  T* o = static_cast<T*>(p.o) + b * p.o_batch_stride + h * p.o_head_stride;
  o[d] = from_float<T>(den > 0.f ? num / den : 0.f);
  if (d == 0) p.lse[bh] = den > 0.f ? m * static_cast<float>(M_LN2) + logf(den) : -INFINITY;
}

template <typename T, int D, int kHeads>
void launch(const DecodeParams& p, cudaStream_t stream) {
  const dim3 grid(p.batch * (p.heads / kHeads), p.num_splits);
  decode_kernel<T, D, kHeads><<<grid, kThreads, 0, stream>>>(p);
  if (p.num_splits > 1) merge_kernel<T><<<p.batch * p.heads, D, 0, stream>>>(p);
}

template <typename T, int D>
void launch_for_heads(const DecodeParams& p, int k_heads, cudaStream_t stream) {
  if (k_heads == 1) {
    launch<T, D, 1>(p, stream);
  } else if (k_heads == 2) {
    launch<T, D, 2>(p, stream);
  } else if (k_heads == 4) {
    launch<T, D, 4>(p, stream);
  } else if constexpr (D == 64) {
    launch<T, D, 8>(p, stream);
  } else {
    TORCH_CHECK(false, "no decode kernel for ", k_heads, " heads per block at head_dim ", D);
  }
}

// Query heads per block: the largest power of two that divides the GQA group, capped so the
// per-lane copy of the queries stays at 64 registers. FLASH_LAB_DECODE_HEADS_PER_BLOCK=1|2|4|8
// overrides it for experiments (it must divide the group).
int heads_per_block(int group, int head_dim) {
  const int limit = head_dim == 128 ? 4 : 8;
  if (const char* env = std::getenv("FLASH_LAB_DECODE_HEADS_PER_BLOCK")) {
    const int k = std::atoi(env);
    TORCH_CHECK(k >= 1 && k <= limit && group % k == 0 && (k & (k - 1)) == 0,
                "FLASH_LAB_DECODE_HEADS_PER_BLOCK=", env, " does not fit group ", group);
    return k;
  }
  int k = 1;
  while (k * 2 <= limit && group % (k * 2) == 0) k *= 2;
  return k;
}

// Smallest power of two that gives about 16 blocks per SM, so enough warps are resident to keep
// many loads in flight, while keeping >= 256 keys per split of the longest possible sequence.
// (Two blocks per SM, the first version, left H100 at 47-50% of HBM bandwidth at 32K context;
// a sweep put the best split counts near this target.)
int choose_num_splits(const DecodeParams& p, int k_heads) {
  const int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int blocks = p.batch * (p.heads / k_heads);
  int splits = 1;
  while (splits < 64 && blocks * splits < 16 * sm_count && p.max_seqlen / (2 * splits) >= 256) {
    splits *= 2;
  }
  return splits;
}

std::tuple<at::Tensor, at::Tensor> run_decode(const at::Tensor& q, const at::Tensor& k_cache,
                                              const at::Tensor& v_cache, const at::Tensor& seq_lens,
                                              double softmax_scale, int64_t num_splits,
                                              bool allow_splits) {
  check_decode_inputs(q, k_cache, v_cache, seq_lens);
  check_16_byte_cache_rows(k_cache, "k_cache");
  check_16_byte_cache_rows(v_cache, "v_cache");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 &&
                  (q.size(0) == 1 || q.stride(0) % (16 / q.element_size()) == 0) &&
                  (q.size(1) == 1 || q.stride(1) % (16 / q.element_size()) == 0),
              "q must have 16-byte aligned rows");
  TORCH_CHECK(num_splits >= 0 && num_splits <= 128, "num_splits must be in [0, 128]");
  const at::cuda::OptionalCUDAGuard guard(q.device());

  at::Tensor o = at::empty(q.sizes(), q.options());
  at::Tensor lse = at::empty({q.size(0), q.size(1)}, q.options().dtype(at::kFloat));
  DecodeParams p = make_decode_params(q, k_cache, v_cache, seq_lens, o, lse, softmax_scale);
  const int k_heads = heads_per_block(p.group, p.head_dim);
  if (allow_splits) {
    p.num_splits = num_splits > 0 ? static_cast<int>(num_splits) : choose_num_splits(p, k_heads);
  }
  at::Tensor partial_o;
  at::Tensor partial_ml;
  if (p.num_splits > 1) {
    const auto opts = q.options().dtype(at::kFloat);
    partial_o = at::empty({p.batch, p.heads, p.num_splits, p.head_dim}, opts);
    partial_ml = at::empty({p.batch, p.heads, p.num_splits, 2}, opts);
    p.partial_o = partial_o.data_ptr<float>();
    p.partial_ml = partial_ml.data_ptr<float>();
  }

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dispatch_decode_dtype(q.scalar_type(), [&](auto tag) {
    using T = decltype(tag);
    if (p.head_dim == 64) {
      launch_for_heads<T, 64>(p, k_heads, stream);
    } else {
      launch_for_heads<T, 128>(p, k_heads, stream);
    }
  });
  FLASH_LAB_LAUNCH_CHECK();
  return {o, lse};
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> decode_inplace(const at::Tensor& q, const at::Tensor& k_cache,
                                                  const at::Tensor& v_cache,
                                                  const at::Tensor& seq_lens, double softmax_scale,
                                                  int64_t num_splits) {
  return run_decode(q, k_cache, v_cache, seq_lens, softmax_scale, num_splits, false);
}

std::tuple<at::Tensor, at::Tensor> decode_splitkv(const at::Tensor& q, const at::Tensor& k_cache,
                                                  const at::Tensor& v_cache,
                                                  const at::Tensor& seq_lens, double softmax_scale,
                                                  int64_t num_splits) {
  return run_decode(q, k_cache, v_cache, seq_lens, softmax_scale, num_splits, true);
}

}  // namespace flash_lab

TORCH_LIBRARY_IMPL(flash_lab, CUDA, m) {
  m.impl("decode_inplace", &flash_lab::decode_inplace);
  m.impl("decode_splitkv", &flash_lab::decode_splitkv);
}
