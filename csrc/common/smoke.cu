// Smallest possible op: checks that nvcc, the arch list, and op registration work on a new node.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "common/checks.h"

namespace flash_lab {
namespace {

__global__ void axpy_kernel(const float* __restrict__ x, const float* __restrict__ y, float alpha,
                            float* __restrict__ out, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = alpha * x[i] + y[i];
}

}  // namespace

at::Tensor smoke_axpy(const at::Tensor& x, const at::Tensor& y, double alpha) {
  FLASH_LAB_CHECK_CUDA(x);
  FLASH_LAB_CHECK_CUDA(y);
  FLASH_LAB_CHECK_DTYPE(x, at::kFloat);
  FLASH_LAB_CHECK_DTYPE(y, at::kFloat);
  TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape");

  const at::cuda::OptionalCUDAGuard guard(x.device());
  const at::Tensor xc = x.contiguous();
  const at::Tensor yc = y.contiguous();
  at::Tensor out = at::empty_like(xc);
  const int64_t n = xc.numel();
  if (n == 0) return out;

  constexpr int kThreads = 256;
  const int64_t blocks = (n + kThreads - 1) / kThreads;
  axpy_kernel<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      xc.data_ptr<float>(), yc.data_ptr<float>(), static_cast<float>(alpha), out.data_ptr<float>(),
      n);
  FLASH_LAB_LAUNCH_CHECK();
  return out;
}

}  // namespace flash_lab

TORCH_LIBRARY_IMPL(flash_lab, CUDA, m) { m.impl("smoke_axpy", &flash_lab::smoke_axpy); }
