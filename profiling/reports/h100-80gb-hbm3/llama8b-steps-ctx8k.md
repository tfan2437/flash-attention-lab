## Time per range (NVTX, includes host work)

| range | instances | mean (us) |
|---|---|---|
| sdpa:step | 20 | 27371.0 |
| sdpa:prefill | 1 | 514523.9 |
| flash_lab:step | 20 | 24506.3 |
| flash_lab:prefill | 1 | 272213.3 |

## GPU time by kernel within each range

### flash_lab:prefill

| kernel | share of GPU time | total (us) |
|---|---|---|
| `nvjet_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 30.0% | 78819.7 |
| `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 25.6% | 67313.0 |
| `_attention_fwd_kernel` | 16.4% | 42987.9 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_...` | 5.3% | 13902.7 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_...` | 2.8% | 7446.3 |
| `void at::native::vectorized_elementwise_kernel<(int)8, at::native::BinaryFunctor<c10::B...` | 2.7% | 7151.7 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Te...` | 2.7% | 7126.1 |
| `void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsi...` | 2.3% | 5982.8 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_...` | 2.2% | 5841.0 |
| `void at::native::vectorized_elementwise_kernel<(int)8, at::native::<unnamed>::silu_kern...` | 2.0% | 5229.9 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_...` | 1.7% | 4474.6 |
| `void at::native::vectorized_elementwise_kernel<(int)8, at::native::bfloat16_copy_kernel...` | 1.6% | 4305.7 |
| 14 other kernels | 4.5% | 11894.5 |

### flash_lab:step

| kernel | share of GPU time | total (us) |
|---|---|---|
| `nvjet_tst_64x8_64x16_2x1_v_bz_TNT` | 25.4% | 51704.6 |
| `nvjet_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 21.4% | 43494.7 |
| `void flash_lab::<unnamed>::decode_kernel<__nv_bfloat16, (int)128, (int)4>(flash_lab::De...` | 20.7% | 42182.7 |
| `nvjet_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 3.5% | 7163.5 |
| `nvjet_tst_384x8_64x4_2x1_v_bz_TNT` | 3.4% | 6940.1 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_...` | 2.9% | 5941.0 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float, ...` | 2.8% | 5675.9 |
| `void flash_lab::<unnamed>::merge_kernel<__nv_bfloat16>(flash_lab::DecodeParams)` | 2.6% | 5283.1 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Te...` | 1.9% | 3813.8 |
| `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::index_copy...` | 1.7% | 3456.7 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native...` | 1.7% | 3413.4 |
| `void at::native::vectorized_elementwise_kernel<(int)8, at::native::CUDAFunctor_add<c10:...` | 1.7% | 3386.0 |
| 18 other kernels | 10.4% | 21157.4 |

### sdpa:prefill

| kernel | share of GPU time | total (us) |
|---|---|---|
| `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutla...` | 54.3% | 277772.1 |
| `nvjet_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 15.3% | 78202.9 |
| `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 13.0% | 66338.4 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_...` | 2.6% | 13445.4 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_...` | 2.0% | 10273.0 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_...` | 1.4% | 7305.9 |
| `void at::native::vectorized_elementwise_kernel<(int)8, at::native::BinaryFunctor<c10::B...` | 1.4% | 7151.1 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Te...` | 1.4% | 6935.7 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_...` | 1.1% | 5852.0 |
| `void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsi...` | 1.1% | 5676.0 |
| `void at::native::vectorized_elementwise_kernel<(int)8, at::native::<unnamed>::silu_kern...` | 1.0% | 5170.3 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_...` | 1.0% | 5132.0 |
| 19 other kernels | 4.3% | 21898.2 |

### sdpa:step

| kernel | share of GPU time | total (us) |
|---|---|---|
| `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutla...` | 47.5% | 228427.0 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_...` | 19.7% | 94532.0 |
| `nvjet_tst_64x8_64x16_2x1_v_bz_TNT` | 10.8% | 51739.2 |
| `nvjet_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 8.9% | 42730.8 |
| `nvjet_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 1.5% | 7150.5 |
| `nvjet_tst_384x8_64x4_2x1_v_bz_TNT` | 1.4% | 6945.6 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_...` | 1.2% | 5893.3 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float, ...` | 1.1% | 5497.1 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Te...` | 0.7% | 3603.2 |
| `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::index_copy...` | 0.7% | 3491.5 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native...` | 0.7% | 3389.4 |
| `void at::native::vectorized_elementwise_kernel<(int)8, at::native::CUDAFunctor_add<c10:...` | 0.7% | 3367.9 |
| 21 other kernels | 4.9% | 23769.5 |
