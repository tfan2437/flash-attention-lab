## Step time per implementation (NVTX range, includes host work)

| implementation | steps | mean step (us) |
|---|---|---|
| decode_copy | 20 | 3163.8 |
| decode_inplace | 20 | 1232.6 |
| splitkv | 20 | 461.8 |

## GPU time by kernel within the steps

### decode_copy

| kernel | share of GPU time | total (us) |
|---|---|---|
| `void flash_lab::<unnamed>::decode_copy_kernel<__nv_bfloat16, (int)128>(flash_lab::Decod...` | 51.1% | 31478.1 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_...` | 48.8% | 30075.1 |
| `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<int, at::native::...` | 0.1% | 41.5 |

### decode_inplace

| kernel | share of GPU time | total (us) |
|---|---|---|
| `void flash_lab::<unnamed>::decode_kernel<__nv_bfloat16, (int)128, (int)1>(flash_lab::De...` | 100.0% | 23855.7 |

### splitkv

| kernel | share of GPU time | total (us) |
|---|---|---|
| `void flash_lab::<unnamed>::decode_kernel<__nv_bfloat16, (int)128, (int)1>(flash_lab::De...` | 98.0% | 8235.6 |
| `void flash_lab::<unnamed>::merge_kernel<__nv_bfloat16>(flash_lab::DecodeParams)` | 2.0% | 168.7 |
