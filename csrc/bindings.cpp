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

  // Test helper: one 16 x K by K x 16 bf16 tile product through ldmatrix + mma.sync.
  m.def("mma_tile_test(Tensor a, Tensor b, bool trans_b) -> Tensor");
}

// The module has no Python attributes; it only needs to be importable.
PyMODINIT_FUNC PyInit__C(void) {
  static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_C", nullptr, -1, nullptr};
  return PyModule_Create(&module);
}
