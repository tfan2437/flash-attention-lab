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
}

// The module has no Python attributes; it only needs to be importable.
PyMODINIT_FUNC PyInit__C(void) {
  static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_C", nullptr, -1, nullptr};
  return PyModule_Create(&module);
}
