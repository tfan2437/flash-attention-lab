#pragma once

// Thin wrappers over the tensor-core instructions used by the prefill kernel (sm_80 and newer).
//
// mma.sync.m16n8k16 fragment layouts (PTX ISA), with g = lane / 4 and t = lane % 4:
//   A, 16x16, four 32-bit registers of two elements each:
//     a0 = (row g,     cols 2t, 2t+1)    a2 = (row g,     cols 2t+8, 2t+9)
//     a1 = (row g + 8, cols 2t, 2t+1)    a3 = (row g + 8, cols 2t+8, 2t+9)
//   B, 16x8 (k x n), two registers:
//     b0 = (rows 2t, 2t+1, col g)        b1 = (rows 2t+8, 2t+9, col g)
//   C and D, 16x8 fp32, four floats:
//     c0, c1 = (row g, cols 2t, 2t+1)    c2, c3 = (row g + 8, cols 2t, 2t+1)
// The lower-indexed element of each pair sits in the low 16 bits of its register.
//
// ldmatrix.x4 loads four 8x8 matrices of 16-bit elements: lanes 8i..8i+7 supply the row
// addresses of matrix i, and lane l receives (row l / 4, cols 2(l % 4), 2(l % 4) + 1) of each
// matrix into the corresponding register, which is exactly the A, B, and C fragment shape above.
// With .trans each matrix is transposed on the way, so lane l receives
// (rows 2(l % 4), 2(l % 4) + 1, col l / 4).

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

namespace flash_lab {

__device__ __forceinline__ uint32_t smem_addr(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* row_ptr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_addr(row_ptr)));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], const void* row_ptr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_addr(row_ptr)));
}

// Asynchronous 16-byte global-to-shared copy that bypasses L1. With valid == false nothing is
// read and the destination is zero-filled (src-size 0), so out-of-range rows need no branch.
__device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* gmem_ptr, bool valid) {
  const int src_bytes = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_addr(smem_ptr)),
               "l"(gmem_ptr), "r"(src_bytes));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }

// Waits until at most `kPending` committed groups of this thread are still in flight.
template <int kPending>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(kPending) : "memory");
}

// d += a * b on a 16x8x16 tile with fp32 accumulation. T is __nv_bfloat16 or __half.
template <typename T>
__device__ __forceinline__ void mma_16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                                          uint32_t b1) {
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  } else {
    static_assert(std::is_same_v<T, __half>, "mma_16816 takes bf16 or fp16");
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  }
}

// Rounds two floats to T and packs them, lo in the low 16 bits.
template <typename T>
__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
  } else {
    __half2 v = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
  }
}

}  // namespace flash_lab
