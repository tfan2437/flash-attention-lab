#pragma once

// Host-side helpers shared by the prefill ops: input checks at the op boundary, output
// allocation, and AttnParams setup. The Python wrapper performs the same checks with friendlier
// messages; these keep direct torch.ops calls safe.

#include <ATen/ATen.h>

#include <cmath>
#include <initializer_list>
#include <tuple>

#include "common/checks.h"
#include "common/params.h"

namespace flash_lab {

inline void check_attention_inputs(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                   at::ScalarType dtype, std::initializer_list<int> head_dims) {
  FLASH_LAB_CHECK_CUDA(q);
  FLASH_LAB_CHECK_CUDA(k);
  FLASH_LAB_CHECK_CUDA(v);
  FLASH_LAB_CHECK_DTYPE(q, dtype);
  FLASH_LAB_CHECK_DTYPE(k, dtype);
  FLASH_LAB_CHECK_DTYPE(v, dtype);
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q, k, v must be [B, S, H, D]");
  TORCH_CHECK(k.sizes() == v.sizes(), "k and v must have the same shape");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(3) == k.size(3),
              "q and k must agree on batch size and head_dim");
  TORCH_CHECK(k.size(2) > 0 && q.size(2) % k.size(2) == 0,
              "query heads must be a multiple of KV heads");
  TORCH_CHECK(q.numel() > 0 && k.numel() > 0, "empty input");
  FLASH_LAB_CHECK_LAST_DIM(q);
  FLASH_LAB_CHECK_LAST_DIM(k);
  FLASH_LAB_CHECK_LAST_DIM(v);
  TORCH_CHECK(q.device() == k.device() && q.device() == v.device(), "q, k, v on different devices");

  bool supported = false;
  for (int d : head_dims) supported |= q.size(3) == d;
  TORCH_CHECK(supported, "unsupported head_dim ", q.size(3));
  TORCH_CHECK(q.size(0) * q.size(2) <= 65535, "batch * heads must be at most 65535");
}

// o is [B, S_q, H, D] contiguous in the input dtype; lse is [B, H, S_q] fp32.
inline std::tuple<at::Tensor, at::Tensor> alloc_attention_outputs(const at::Tensor& q) {
  at::Tensor o = at::empty(q.sizes(), q.options());
  at::Tensor lse = at::empty({q.size(0), q.size(2), q.size(1)}, q.options().dtype(at::kFloat));
  return {o, lse};
}

inline AttnParams make_attn_params(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                   at::Tensor& o, at::Tensor& lse, bool causal, double scale) {
  AttnParams p{};
  p.q = q.data_ptr();
  p.k = k.data_ptr();
  p.v = v.data_ptr();
  p.o = o.data_ptr();
  p.lse = lse.data_ptr<float>();

  p.q_batch_stride = q.stride(0);
  p.q_seq_stride = q.stride(1);
  p.q_head_stride = q.stride(2);
  p.k_batch_stride = k.stride(0);
  p.k_seq_stride = k.stride(1);
  p.k_head_stride = k.stride(2);
  p.v_batch_stride = v.stride(0);
  p.v_seq_stride = v.stride(1);
  p.v_head_stride = v.stride(2);
  p.o_batch_stride = o.stride(0);
  p.o_seq_stride = o.stride(1);
  p.o_head_stride = o.stride(2);

  p.batch = static_cast<int>(q.size(0));
  p.seqlen_q = static_cast<int>(q.size(1));
  p.seqlen_k = static_cast<int>(k.size(1));
  p.heads = static_cast<int>(q.size(2));
  p.heads_kv = static_cast<int>(k.size(2));
  p.head_dim = static_cast<int>(q.size(3));
  p.group = p.heads / p.heads_kv;
  p.causal_offset = p.seqlen_k - p.seqlen_q;
  p.scale = static_cast<float>(scale);
  p.scale_log2 = static_cast<float>(scale * M_LOG2E);
  p.causal = causal;
  return p;
}

}  // namespace flash_lab
