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

}  // namespace flash_lab
