// Op schemas for the extension. Each kernel file registers its CUDA implementation with
// TORCH_LIBRARY_IMPL(flash_lab, CUDA, m); importing flash_lab._C runs all registrations.

#include <Python.h>
#include <torch/library.h>

TORCH_LIBRARY(flash_lab, m) {
  // Toolchain check: alpha * x + y in fp32.
  m.def("smoke_axpy(Tensor x, Tensor y, float alpha) -> Tensor");

  // Prefill: q [B, S_q, H, D], k and v [B, S_k, H_kv, D] -> o [B, S_q, H, D], lse [B, H, S_q].
  m.def(
      "attention_fp32_fused(Tensor q, Tensor k, Tensor v, bool causal, float softmax_scale)"
      " -> (Tensor, Tensor)");
  m.def(
      "attention_mma(Tensor q, Tensor k, Tensor v, bool causal, float softmax_scale)"
      " -> (Tensor, Tensor)");
  m.def(
      "attention_mma_pipelined(Tensor q, Tensor k, Tensor v, bool causal, float softmax_scale)"
      " -> (Tensor, Tensor)");

  // Decode: q [B, H, D], caches [B, S_max, H_kv, D], seq_lens [B] -> o [B, H, D], lse [B, H].
  // num_splits is used by decode_splitkv only (0 picks it from the problem size).
  m.def(
      "decode_copy(Tensor q, Tensor k_cache, Tensor v_cache, Tensor seq_lens, float softmax_scale,"
      " int num_splits) -> (Tensor, Tensor)");
  m.def(
      "decode_inplace(Tensor q, Tensor k_cache, Tensor v_cache, Tensor seq_lens,"
      " float softmax_scale, int num_splits) -> (Tensor, Tensor)");
  m.def(
      "decode_splitkv(Tensor q, Tensor k_cache, Tensor v_cache, Tensor seq_lens,"
      " float softmax_scale, int num_splits) -> (Tensor, Tensor)");

  // Test helper: one 16 x K by K x 16 bf16 tile product through ldmatrix + mma.sync.
  m.def("mma_tile_test(Tensor a, Tensor b, bool trans_b) -> Tensor");
}

// The module has no Python attributes; it only needs to be importable.
PyMODINIT_FUNC PyInit__C(void) {
  static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_C", nullptr, -1, nullptr};
  return PyModule_Create(&module);
}
