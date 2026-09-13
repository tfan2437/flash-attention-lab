#pragma once

#include <cstdint>

namespace flash_lab {

// Prefill problem passed by value to the kernels. Tensors are [B, S, H, D] with unit stride in D;
// strides are in elements.
struct AttnParams {
  const void* q;
  const void* k;
  const void* v;
  void* o;
  float* lse;  // [B, H, S_q], natural-log log-sum-exp of each scaled score row

  int64_t q_batch_stride, q_seq_stride, q_head_stride;
  int64_t k_batch_stride, k_seq_stride, k_head_stride;
  int64_t v_batch_stride, v_seq_stride, v_head_stride;
  int64_t o_batch_stride, o_seq_stride, o_head_stride;

  int batch, seqlen_q, seqlen_k, heads, heads_kv, head_dim;
  int group;          // heads / heads_kv: query head h reads KV head h / group
  int causal_offset;  // seqlen_k - seqlen_q: query i sees key j iff j <= i + causal_offset
  float scale;        // softmax scale
  float scale_log2;   // scale * log2(e), for kernels that use exp2
  bool causal;
};

// Decode problem: one query token per sequence against the first seq_lens[b] rows of a
// [B, S_max, H_kv, D] cache, read in place. Strides are in elements.
struct DecodeParams {
  const void* q;  // [B, H, D]
  const void* k;
  const void* v;
  void* o;              // [B, H, D]
  float* lse;           // [B, H], natural log
  const int* seq_lens;  // [B]

  int64_t q_batch_stride, q_head_stride;
  int64_t k_batch_stride, k_seq_stride, k_head_stride;
  int64_t v_batch_stride, v_seq_stride, v_head_stride;
  int64_t o_batch_stride, o_head_stride;

  int batch, heads, heads_kv, head_dim, group, max_seqlen;
  float scale;       // softmax scale
  float scale_log2;  // scale * log2(e)

  // Split-KV: each of num_splits blocks per (sequence, head) writes an unnormalized partial
  // output and its (max, sum) to these fp32 buffers; a merge kernel combines them.
  int num_splits;
  float* partial_o;   // [B, H, num_splits, D]
  float* partial_ml;  // [B, H, num_splits, 2]
};

}  // namespace flash_lab
