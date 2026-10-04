## Time per range (NVTX, includes host work)

| range | instances | mean (us) |
|---|---|---|
| sdpa:step | 20 | 27405.8 |
| sdpa:prefill | 1 | 513501.6 |
| flash_lab:step | 20 | 24693.7 |
| flash_lab:prefill | 1 | 275219.9 |

## GPU time by kernel within each range

### flash_lab:prefill

| kernel | share of GPU time | total (us) |
|---|---|---|
| `nvjet_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 29.8% | 79181.7 |
| `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 25.6% | 67893.5 |
| `_attention_fwd_kernel` | 16.7% | 44438.6 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_...` | 5.3% | 14155.4 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_...` | 2.8% | 7521.3 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Te...` | 2.7% | 7213.1 |
| `void at::native::vectorized_elementwise_kernel<(int)8, at::native::BinaryFunctor<c10::B...` | 2.7% | 7155.4 |
| `void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsi...` | 2.3% | 6138.2 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_...` | 2.2% | 5855.0 |
| `void at::native::vectorized_elementwise_kernel<(int)8, at::native::<unnamed>::silu_kern...` | 2.0% | 5257.8 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_...` | 1.7% | 4549.3 |
| `void at::native::vectorized_elementwise_kernel<(int)8, at::native::bfloat16_copy_kernel...` | 1.6% | 4298.5 |
| 14 other kernels | 4.5% | 11989.3 |

### flash_lab:step

| kernel | share of GPU time | total (us) |
|---|---|---|
| `nvjet_tst_64x8_64x16_2x1_v_bz_TNT` | 25.6% | 52302.3 |
| `nvjet_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 21.3% | 43389.9 |
| `void flash_lab::<unnamed>::decode_kernel<__nv_bfloat16, (int)128, (int)4>(flash_lab::De...` | 20.7% | 42324.2 |
| `nvjet_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 3.6% | 7290.8 |
| `nvjet_tst_384x8_64x4_2x1_v_bz_TNT` | 3.4% | 6970.0 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_...` | 2.9% | 5922.0 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float, ...` | 2.8% | 5738.4 |
| `void flash_lab::<unnamed>::merge_kernel<__nv_bfloat16>(flash_lab::DecodeParams)` | 2.6% | 5342.2 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Te...` | 1.7% | 3530.5 |
| `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::index_copy...` | 1.7% | 3369.1 |
| `void at::native::vectorized_elementwise_kernel<(int)8, at::native::CUDAFunctor_add<c10:...` | 1.6% | 3351.6 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native...` | 1.6% | 3342.5 |
| 18 other kernels | 10.4% | 21167.9 |

### sdpa:prefill

| kernel | share of GPU time | total (us) |
|---|---|---|
| `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutla...` | 54.4% | 277425.3 |
| `nvjet_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 15.3% | 77868.5 |
| `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 12.9% | 65799.6 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_...` | 2.6% | 13468.4 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_...` | 2.0% | 10304.0 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_...` | 1.4% | 7311.1 |
| `void at::native::vectorized_elementwise_kernel<(int)8, at::native::BinaryFunctor<c10::B...` | 1.4% | 7152.4 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Te...` | 1.4% | 6946.6 |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_...` | 1.1% | 5857.7 |
| `void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsi...` | 1.1% | 5684.3 |
| `void at::native::vectorized_elementwise_kernel<(int)8, at::native::<unnamed>::silu_kern...` | 1.0% | 5167.7 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_...` | 1.0% | 5142.9 |
| 19 other kernels | 4.3% | 21887.9 |

### sdpa:step

| kernel | share of GPU time | total (us) |
|---|---|---|
| `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutla...` | 47.5% | 227514.2 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_...` | 19.8% | 94719.5 |
| `nvjet_tst_64x8_64x16_2x1_v_bz_TNT` | 10.7% | 51115.0 |
| `nvjet_tst_64x8_64x16_2x1_v_bz_splitK_TNT` | 8.9% | 42736.4 |
| `nvjet_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 1.5% | 7210.6 |
| `nvjet_tst_384x8_64x4_2x1_v_bz_TNT` | 1.5% | 7026.3 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_...` | 1.2% | 5790.3 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float, ...` | 1.2% | 5632.8 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Te...` | 0.7% | 3424.7 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native...` | 0.7% | 3339.7 |
| `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::index_copy...` | 0.7% | 3320.3 |
| `void at::native::vectorized_elementwise_kernel<(int)8, at::native::CUDAFunctor_add<c10:...` | 0.7% | 3318.7 |
| 21 other kernels | 4.9% | 23565.4 |
