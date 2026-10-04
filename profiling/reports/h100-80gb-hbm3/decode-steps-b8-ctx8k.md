## Time per range (NVTX, includes host work)

| range | instances | mean (us) |
|---|---|---|
| decode_copy:step | 20 | 3147.3 |
| decode_inplace:step | 20 | 1227.6 |
| splitkv:step | 20 | 462.0 |

## GPU time by kernel within each range

### decode_copy:step

| kernel | share of GPU time | total (us) |
|---|---|---|
| `void flash_lab::<unnamed>::decode_copy_kernel<__nv_bfloat16, (int)128>(flash_lab::Decod...` | 51.1% | 31319.1 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_...` | 48.8% | 29937.2 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<int, at::native::...` | 0.1% | 41.4 |

### decode_inplace:step

| kernel | share of GPU time | total (us) |
|---|---|---|
| `void flash_lab::<unnamed>::decode_kernel<__nv_bfloat16, (int)128, (int)1>(flash_lab::De...` | 100.0% | 23786.0 |

### splitkv:step

| kernel | share of GPU time | total (us) |
|---|---|---|
| `void flash_lab::<unnamed>::decode_kernel<__nv_bfloat16, (int)128, (int)1>(flash_lab::De...` | 98.0% | 8226.6 |
| `void flash_lab::<unnamed>::merge_kernel<__nv_bfloat16>(flash_lab::DecodeParams)` | 2.0% | 166.9 |
