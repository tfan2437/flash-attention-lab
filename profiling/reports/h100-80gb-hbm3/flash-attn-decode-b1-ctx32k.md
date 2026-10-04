## void flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, 0, 0, bfloat16_t, Flash_kernel_traits<128, 64, 128, 4, bfloat16_t>>, 0, 0, 0, 0, 1, 0, 1, 0>(Flash_fwd_params)

- grid (1, 7, 32), block (128, 1, 1)

| metric | value | unit |
|---|---|---|
| duration (`gpu__time_duration.sum`) | 177.152000 | us |
| SM throughput (% of peak) (`sm__throughput.avg.pct_of_peak_sustained_elapsed`) | 30.368554 | % |
| memory throughput (%) (`gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`) | 90.899092 | % |
| DRAM bytes read (`dram__bytes_read.sum`) | 536.966656 | Mbyte |
| DRAM bytes written (`dram__bytes_write.sum`) | 2.737408 | Mbyte |
| L2 hit rate (%) (`lts__t_sector_hit_rate.pct`) | 0.899305 | % |
| issue slots busy (%) (`smsp__issue_active.avg.pct_of_peak_sustained_active`) | 18.359183 | % |
| achieved occupancy (%) (`sm__warps_active.avg.pct_of_peak_sustained_active`) | 10.537034 | % |
| theoretical occupancy (%) (`sm__maximum_warps_per_active_cycle_pct`) | 12.500000 | % |
| registers per thread (`launch__registers_per_thread`) | 254 | register/thread |
| static shared memory per block (`launch__shared_mem_per_block_static`) | 0 | byte/block |
| dynamic shared memory per block (`launch__shared_mem_per_block_dynamic`) | 81.920000 | Kbyte/block |
| blocks per SM allowed by registers (`launch__occupancy_limit_registers`) | 2.000000 | block |
| blocks per SM allowed by shared memory (`launch__occupancy_limit_shared_mem`) | 2.000000 | block |
| waves per SM (`launch__waves_per_multiprocessor`) | 0.85 |  |
| shared-memory wavefronts (`l1tex__data_pipe_lsu_wavefronts_mem_shared.sum`) | 18353187 |  |
| shared-memory bank conflicts (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`) | 2461 |  |

| pipe utilization | % of peak (active cycles) |
|---|---|
| `sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active` | 4.271831 |
| `sm__inst_executed_pipe_fma.max.pct_of_peak_sustained_active` | 5.091447 |
| `sm__inst_executed_pipe_fma.min.pct_of_peak_sustained_active` | 2.346964 |
| `sm__inst_executed_pipe_fma.sum.pct_of_peak_sustained_active` | 4.271831 |
| `sm__inst_executed_pipe_fma_type_fp16.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_fmaheavy.avg.pct_of_peak_sustained_active` | 2.267426 |
| `sm__inst_executed_pipe_fmaheavy.max.pct_of_peak_sustained_active` | 2.754354 |
| `sm__inst_executed_pipe_fmaheavy.min.pct_of_peak_sustained_active` | 1.242412 |
| `sm__inst_executed_pipe_fmaheavy.sum.pct_of_peak_sustained_active` | 2.267426 |
| `sm__inst_executed_pipe_fmalite.avg.pct_of_peak_sustained_active` | 6.276237 |
| `sm__inst_executed_pipe_fmalite.max.pct_of_peak_sustained_active` | 7.508847 |
| `sm__inst_executed_pipe_fmalite.min.pct_of_peak_sustained_active` | 3.450847 |
| `sm__inst_executed_pipe_fmalite.sum.pct_of_peak_sustained_active` | 6.276237 |
| `sm__inst_executed_pipe_tensor_op_dmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active` | 10.632287 |
| `sm__inst_executed_pipe_tensor_op_hmma.max.pct_of_peak_sustained_active` | 12.677756 |
| `sm__inst_executed_pipe_tensor_op_hmma.min.pct_of_peak_sustained_active` | 5.824915 |
| `sm__inst_executed_pipe_tensor_op_hmma.sum.pct_of_peak_sustained_active` | 10.632287 |
| `sm__inst_executed_pipe_tensor_op_imma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` | 4.295112 |
| `sm__pipe_fma_cycles_active.max.pct_of_peak_sustained_active` | 5.118885 |
| `sm__pipe_fma_cycles_active.min.pct_of_peak_sustained_active` | 2.360683 |
| `sm__pipe_fma_cycles_active.sum.pct_of_peak_sustained_active` | 4.295112 |
| `sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active` | 2.313987 |
| `sm__pipe_fmaheavy_cycles_active.max.pct_of_peak_sustained_active` | 2.809230 |
| `sm__pipe_fmaheavy_cycles_active.min.pct_of_peak_sustained_active` | 1.269850 |
| `sm__pipe_fmaheavy_cycles_active.sum.pct_of_peak_sustained_active` | 2.313987 |
| `sm__pipe_fmalite_cycles_active.avg.pct_of_peak_sustained_active` | 6.276237 |
| `sm__pipe_fmalite_cycles_active.max.pct_of_peak_sustained_active` | 7.508847 |
| `sm__pipe_fmalite_cycles_active.min.pct_of_peak_sustained_active` | 3.450847 |
| `sm__pipe_fmalite_cycles_active.sum.pct_of_peak_sustained_active` | 6.276237 |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | 31.896861 |
| `sm__pipe_tensor_cycles_active.max.pct_of_peak_sustained_active` | 38.033269 |
| `sm__pipe_tensor_cycles_active.min.pct_of_peak_sustained_active` | 17.474745 |
| `sm__pipe_tensor_cycles_active.sum.pct_of_peak_sustained_active` | 31.896861 |
| `sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | 31.896861 |
| `sm__pipe_tensor_op_hmma_cycles_active.max.pct_of_peak_sustained_active` | 38.033269 |
| `sm__pipe_tensor_op_hmma_cycles_active.min.pct_of_peak_sustained_active` | 17.474745 |
| `sm__pipe_tensor_op_hmma_cycles_active.sum.pct_of_peak_sustained_active` | 31.896861 |
| `sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |

| stall reason | cycles per issued instruction |
|---|---|
| long_scoreboard | 4.07 |
| wait | 1.74 |
| selected | 1.00 |
| short_scoreboard | 0.83 |
| barrier | 0.81 |
| mio_throttle | 0.34 |
| math_pipe_throttle | 0.17 |
| not_selected | 0.09 |
| dispatch_stall | 0.06 |
| no_instruction | 0.06 |

## void flash_fwd_splitkv_combine_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, 0, 0, bfloat16_t, Flash_kernel_traits<128, 64, 128, 4, bfloat16_t>>, 4, 3, 1>(Flash_fwd_params)

- grid (8, 1, 1), block (128, 1, 1)

| metric | value | unit |
|---|---|---|
| duration (`gpu__time_duration.sum`) | 7.520000 | us |
| SM throughput (% of peak) (`sm__throughput.avg.pct_of_peak_sustained_elapsed`) | 0.267646 | % |
| memory throughput (%) (`gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`) | 4.536664 | % |
| DRAM bytes read (`dram__bytes_read.sum`) | 0.137984 | Mbyte |
| DRAM bytes written (`dram__bytes_write.sum`) | 0.000000 | Mbyte |
| L2 hit rate (%) (`lts__t_sector_hit_rate.pct`) | 87.895650 | % |
| issue slots busy (%) (`smsp__issue_active.avg.pct_of_peak_sustained_active`) | 6.574001 | % |
| achieved occupancy (%) (`sm__warps_active.avg.pct_of_peak_sustained_active`) | 6.142011 | % |
| theoretical occupancy (%) (`sm__maximum_warps_per_active_cycle_pct`) | 56.250000 | % |
| registers per thread (`launch__registers_per_thread`) | 52 | register/thread |
| static shared memory per block (`launch__shared_mem_per_block_static`) | 160 | byte/block |
| dynamic shared memory per block (`launch__shared_mem_per_block_dynamic`) | 0.000000 | Kbyte/block |
| blocks per SM allowed by registers (`launch__occupancy_limit_registers`) | 9.000000 | block |
| blocks per SM allowed by shared memory (`launch__occupancy_limit_shared_mem`) | 25.000000 | block |
| waves per SM (`launch__waves_per_multiprocessor`) | 0.01 |  |
| shared-memory wavefronts (`l1tex__data_pipe_lsu_wavefronts_mem_shared.sum`) | 568 |  |
| shared-memory bank conflicts (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`) | 24 |  |

| pipe utilization | % of peak (active cycles) |
|---|---|
| `sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active` | 2.302943 |
| `sm__inst_executed_pipe_fma.max.pct_of_peak_sustained_active` | 37.998551 |
| `sm__inst_executed_pipe_fma.min.pct_of_peak_sustained_active` | 0.000000 |
| `sm__inst_executed_pipe_fma.sum.pct_of_peak_sustained_active` | 2.302943 |
| `sm__inst_executed_pipe_fma_type_fp16.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_fmaheavy.avg.pct_of_peak_sustained_active` | 4.203238 |
| `sm__inst_executed_pipe_fmaheavy.max.pct_of_peak_sustained_active` | 69.353433 |
| `sm__inst_executed_pipe_fmaheavy.min.pct_of_peak_sustained_active` | 0.000000 |
| `sm__inst_executed_pipe_fmaheavy.sum.pct_of_peak_sustained_active` | 4.203238 |
| `sm__inst_executed_pipe_fmalite.avg.pct_of_peak_sustained_active` | 0.402647 |
| `sm__inst_executed_pipe_fmalite.max.pct_of_peak_sustained_active` | 6.643670 |
| `sm__inst_executed_pipe_fmalite.min.pct_of_peak_sustained_active` | 0.000000 |
| `sm__inst_executed_pipe_fmalite.sum.pct_of_peak_sustained_active` | 0.402647 |
| `sm__inst_executed_pipe_tensor_op_dmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active` | 0.117848 |
| `sm__inst_executed_pipe_tensor_op_hmma.max.pct_of_peak_sustained_active` | 1.944489 |
| `sm__inst_executed_pipe_tensor_op_hmma.min.pct_of_peak_sustained_active` | 0.000000 |
| `sm__inst_executed_pipe_tensor_op_hmma.sum.pct_of_peak_sustained_active` | 0.117848 |
| `sm__inst_executed_pipe_tensor_op_imma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` | 2.629479 |
| `sm__pipe_fma_cycles_active.max.pct_of_peak_sustained_active` | 43.386406 |
| `sm__pipe_fma_cycles_active.min.pct_of_peak_sustained_active` | 0.000000 |
| `sm__pipe_fma_cycles_active.sum.pct_of_peak_sustained_active` | 2.629479 |
| `sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active` | 4.856312 |
| `sm__pipe_fmaheavy_cycles_active.max.pct_of_peak_sustained_active` | 80.129142 |
| `sm__pipe_fmaheavy_cycles_active.min.pct_of_peak_sustained_active` | 0.000000 |
| `sm__pipe_fmaheavy_cycles_active.sum.pct_of_peak_sustained_active` | 4.856312 |
| `sm__pipe_fmalite_cycles_active.avg.pct_of_peak_sustained_active` | 0.402647 |
| `sm__pipe_fmalite_cycles_active.max.pct_of_peak_sustained_active` | 6.643670 |
| `sm__pipe_fmalite_cycles_active.min.pct_of_peak_sustained_active` | 0.000000 |
| `sm__pipe_fmalite_cycles_active.sum.pct_of_peak_sustained_active` | 0.402647 |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | 0.117848 |
| `sm__pipe_tensor_cycles_active.max.pct_of_peak_sustained_active` | 1.944489 |
| `sm__pipe_tensor_cycles_active.min.pct_of_peak_sustained_active` | 0.000000 |
| `sm__pipe_tensor_cycles_active.sum.pct_of_peak_sustained_active` | 0.117848 |
| `sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | 0.117848 |
| `sm__pipe_tensor_op_hmma_cycles_active.max.pct_of_peak_sustained_active` | 1.944489 |
| `sm__pipe_tensor_op_hmma_cycles_active.min.pct_of_peak_sustained_active` | 0.000000 |
| `sm__pipe_tensor_op_hmma_cycles_active.sum.pct_of_peak_sustained_active` | 0.117848 |
| `sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |

| stall reason | cycles per issued instruction |
|---|---|
| long_scoreboard | 8.16 |
| wait | 2.11 |
| short_scoreboard | 1.23 |
| barrier | 1.13 |
| imc_miss | 1.03 |
| selected | 1.00 |
| no_instruction | 0.45 |
| branch_resolving | 0.17 |
| dispatch_stall | 0.04 |
| drain | 0.04 |
