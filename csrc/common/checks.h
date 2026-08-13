#pragma once

#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>

#define FLASH_LAB_CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")

#define FLASH_LAB_CHECK_LAST_DIM(x) \
  TORCH_CHECK((x).stride(-1) == 1, #x " must be contiguous in its last dimension")

#define FLASH_LAB_CHECK_DTYPE(x, dtype)                                                \
  TORCH_CHECK((x).scalar_type() == (dtype), #x " must have dtype ", (dtype), ", got ", \
              (x).scalar_type())

// Launch errors are always checked. Debug builds (FLASH_LAB_DEBUG=1) also synchronize, so an
// illegal address is reported at the kernel that caused it rather than at a later CUDA call.
#ifdef FLASH_LAB_DEBUG
#define FLASH_LAB_LAUNCH_CHECK()             \
  do {                                       \
    C10_CUDA_KERNEL_LAUNCH_CHECK();          \
    C10_CUDA_CHECK(cudaDeviceSynchronize()); \
  } while (0)
#else
#define FLASH_LAB_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()
#endif
