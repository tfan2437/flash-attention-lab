// Test helper for the tensor-core wrappers: one warp multiplies a 16 x K tile by K x 16 with
// ldmatrix + mma.sync, the same way the attention kernel builds Q K^T (B read row by row) and
// P V (B read through ldmatrix.trans). Fragment layout mistakes show up here in isolation.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "common/checks.h"
#include "common/ptx.cuh"

namespace flash_lab {
namespace {

constexpr int kMaxK = 64;
constexpr int kPad = 8;

// trans_b == false: b is [16, K] and C = A B^T (like Q K^T, keys are rows of b).
// trans_b == true:  b is [K, 16] and C = A B   (like P V, values are rows of b).
template <bool kTransB>
__global__ void mma_tile_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b, float* c, int k) {
  __shared__ alignas(16) __nv_bfloat16 a_s[16][kMaxK + kPad];
  __shared__ alignas(16) __nv_bfloat16 b_s[kTransB ? kMaxK : 16][(kTransB ? 16 : kMaxK) + kPad];
  const int lane = threadIdx.x;
  for (int i = lane; i < 16 * k; i += 32) a_s[i / k][i % k] = a[i];
  if (kTransB) {
    for (int i = lane; i < k * 16; i += 32) b_s[i / 16][i % 16] = b[i];
  } else {
    for (int i = lane; i < 16 * k; i += 32) b_s[i / k][i % k] = b[i];
  }
  __syncwarp();

  float acc[2][4] = {};
  for (int ks = 0; ks < k / 16; ++ks) {
    uint32_t a_frag[4];
    ldmatrix_x4(a_frag, &a_s[lane % 16][ks * 16 + (lane / 16) * 8]);
    uint32_t b_frag[4];
    if (kTransB) {
      // Matrices: (k 0-7, n 0-7), (k 8-15, n 0-7), (k 0-7, n 8-15), (k 8-15, n 8-15).
      ldmatrix_x4_trans(b_frag, &b_s[ks * 16 + lane % 16][(lane / 16) * 8]);
    } else {
      // Matrices: (n 0-7, k 0-7), (n 0-7, k 8-15), (n 8-15, k 0-7), (n 8-15, k 8-15).
      ldmatrix_x4(b_frag, &b_s[lane % 8 + 8 * (lane / 16)][ks * 16 + 8 * ((lane / 8) % 2)]);
    }
    mma_16816<__nv_bfloat16>(acc[0], a_frag, b_frag[0], b_frag[1]);
    mma_16816<__nv_bfloat16>(acc[1], a_frag, b_frag[2], b_frag[3]);
  }

  const int g = lane / 4;
  const int t = lane % 4;
  for (int j = 0; j < 2; ++j) {
    c[g * 16 + 8 * j + 2 * t] = acc[j][0];
    c[g * 16 + 8 * j + 2 * t + 1] = acc[j][1];
    c[(g + 8) * 16 + 8 * j + 2 * t] = acc[j][2];
    c[(g + 8) * 16 + 8 * j + 2 * t + 1] = acc[j][3];
  }
}

}  // namespace

at::Tensor mma_tile_test(const at::Tensor& a, const at::Tensor& b, bool trans_b) {
  FLASH_LAB_CHECK_CUDA(a);
  FLASH_LAB_CHECK_CUDA(b);
  FLASH_LAB_CHECK_DTYPE(a, at::kBFloat16);
  FLASH_LAB_CHECK_DTYPE(b, at::kBFloat16);
  const int64_t k = a.size(1);
  TORCH_CHECK(a.dim() == 2 && a.size(0) == 16 && k % 16 == 0 && k <= kMaxK, "a must be [16, K]");
  TORCH_CHECK(trans_b ? b.sizes() == at::IntArrayRef({k, 16}) : b.sizes() == a.sizes(),
              "b must be [K, 16] with trans_b, else [16, K]");
  const at::cuda::OptionalCUDAGuard guard(a.device());
  const at::Tensor ac = a.contiguous();
  const at::Tensor bc = b.contiguous();
  at::Tensor c = at::empty({16, 16}, a.options().dtype(at::kFloat));
  const auto* ap = reinterpret_cast<const __nv_bfloat16*>(ac.data_ptr());
  const auto* bp = reinterpret_cast<const __nv_bfloat16*>(bc.data_ptr());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (trans_b) {
    mma_tile_kernel<true><<<1, 32, 0, stream>>>(ap, bp, c.data_ptr<float>(), static_cast<int>(k));
  } else {
    mma_tile_kernel<false><<<1, 32, 0, stream>>>(ap, bp, c.data_ptr<float>(), static_cast<int>(k));
  }
  FLASH_LAB_LAUNCH_CHECK();
  return c;
}

}  // namespace flash_lab

TORCH_LIBRARY_IMPL(flash_lab, CUDA, m) { m.impl("mma_tile_test", &flash_lab::mma_tile_test); }
