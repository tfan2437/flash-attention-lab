## void unnamed>::decode_kernel<__nv_bfloat16, 128, 4>(DecodeParams)

- grid (64, 64, 1), block (128, 1, 1)

| metric | value | unit |
|---|---|---|
| duration (`gpu__time_duration.sum`) | 903.552000 | us |
| SM throughput (% of peak) (`sm__throughput.avg.pct_of_peak_sustained_elapsed`) | 19.835386 | % |
| memory throughput (%) (`gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`) | 35.832988 | % |
| DRAM bytes read (`dram__bytes_read.sum`) | 1.074795 | Gbyte |
| DRAM bytes written (`dram__bytes_write.sum`) | 10.512896 | Mbyte |
| L2 hit rate (%) (`lts__t_sector_hit_rate.pct`) | 1.639413 | % |
| issue slots busy (%) (`smsp__issue_active.avg.pct_of_peak_sustained_active`) | 20.567178 | % |
| achieved occupancy (%) (`sm__warps_active.avg.pct_of_peak_sustained_active`) | 18.357653 | % |
| theoretical occupancy (%) (`sm__maximum_warps_per_active_cycle_pct`) | 18.750000 | % |
| registers per thread (`launch__registers_per_thread`) | 135 | register/thread |
| static shared memory per block (`launch__shared_mem_per_block_static`) | 10.368000 | Kbyte/block |
| dynamic shared memory per block (`launch__shared_mem_per_block_dynamic`) | 0 | byte/block |
| blocks per SM allowed by registers (`launch__occupancy_limit_registers`) | 3.000000 | block |
| blocks per SM allowed by shared memory (`launch__occupancy_limit_shared_mem`) | 8.000000 | block |
| waves per SM (`launch__waves_per_multiprocessor`) | 10.34 |  |
| shared-memory wavefronts (`l1tex__data_pipe_lsu_wavefronts_mem_shared.sum`) | 20180816 |  |
| shared-memory bank conflicts (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`) | 24405 |  |

| pipe utilization | % of peak (active cycles) |
|---|---|
| `sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active` | 9.565999 |
| `sm__inst_executed_pipe_fma.max.pct_of_peak_sustained_active` | 10.173216 |
| `sm__inst_executed_pipe_fma.min.pct_of_peak_sustained_active` | 9.248378 |
| `sm__inst_executed_pipe_fma.sum.pct_of_peak_sustained_active` | 9.565999 |
| `sm__inst_executed_pipe_fma_type_fp16.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_fmaheavy.avg.pct_of_peak_sustained_active` | 7.524086 |
| `sm__inst_executed_pipe_fmaheavy.max.pct_of_peak_sustained_active` | 8.056001 |
| `sm__inst_executed_pipe_fmaheavy.min.pct_of_peak_sustained_active` | 7.205124 |
| `sm__inst_executed_pipe_fmaheavy.sum.pct_of_peak_sustained_active` | 7.524086 |
| `sm__inst_executed_pipe_fmalite.avg.pct_of_peak_sustained_active` | 11.607912 |
| `sm__inst_executed_pipe_fmalite.max.pct_of_peak_sustained_active` | 12.429276 |
| `sm__inst_executed_pipe_fmalite.min.pct_of_peak_sustained_active` | 11.153915 |
| `sm__inst_executed_pipe_fmalite.sum.pct_of_peak_sustained_active` | 11.607912 |
| `sm__inst_executed_pipe_tensor_op_dmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active` | 0.226096 |
| `sm__inst_executed_pipe_tensor_op_hmma.max.pct_of_peak_sustained_active` | 0.240447 |
| `sm__inst_executed_pipe_tensor_op_hmma.min.pct_of_peak_sustained_active` | 0.218589 |
| `sm__inst_executed_pipe_tensor_op_hmma.sum.pct_of_peak_sustained_active` | 0.226096 |
| `sm__inst_executed_pipe_tensor_op_imma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` | 9.878227 |
| `sm__pipe_fma_cycles_active.max.pct_of_peak_sustained_active` | 10.505262 |
| `sm__pipe_fma_cycles_active.min.pct_of_peak_sustained_active` | 9.550239 |
| `sm__pipe_fma_cycles_active.sum.pct_of_peak_sustained_active` | 9.878227 |
| `sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active` | 8.148541 |
| `sm__pipe_fmaheavy_cycles_active.max.pct_of_peak_sustained_active` | 8.711882 |
| `sm__pipe_fmaheavy_cycles_active.min.pct_of_peak_sustained_active` | 7.808845 |
| `sm__pipe_fmaheavy_cycles_active.sum.pct_of_peak_sustained_active` | 8.148541 |
| `sm__pipe_fmalite_cycles_active.avg.pct_of_peak_sustained_active` | 11.607912 |
| `sm__pipe_fmalite_cycles_active.max.pct_of_peak_sustained_active` | 12.429276 |
| `sm__pipe_fmalite_cycles_active.min.pct_of_peak_sustained_active` | 11.153915 |
| `sm__pipe_fmalite_cycles_active.sum.pct_of_peak_sustained_active` | 11.607912 |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | 0.226096 |
| `sm__pipe_tensor_cycles_active.max.pct_of_peak_sustained_active` | 0.240447 |
| `sm__pipe_tensor_cycles_active.min.pct_of_peak_sustained_active` | 0.218589 |
| `sm__pipe_tensor_cycles_active.sum.pct_of_peak_sustained_active` | 0.226096 |
| `sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | 0.226096 |
| `sm__pipe_tensor_op_hmma_cycles_active.max.pct_of_peak_sustained_active` | 0.240447 |
| `sm__pipe_tensor_op_hmma_cycles_active.min.pct_of_peak_sustained_active` | 0.218589 |
| `sm__pipe_tensor_op_hmma_cycles_active.sum.pct_of_peak_sustained_active` | 0.226096 |
| `sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |

| stall reason | cycles per issued instruction |
|---|---|
| long_scoreboard | 11.03 |
| wait | 1.23 |
| selected | 1.00 |
| short_scoreboard | 0.36 |
| not_selected | 0.17 |
| branch_resolving | 0.14 |
| barrier | 0.13 |
| dispatch_stall | 0.12 |
| math_pipe_throttle | 0.05 |
| no_instruction | 0.02 |

## void unnamed>::merge_kernel<__nv_bfloat16>(DecodeParams)

- grid (256, 1, 1), block (128, 1, 1)

| metric | value | unit |
|---|---|---|
| duration (`gpu__time_duration.sum`) | 31.552000 | us |
| SM throughput (% of peak) (`sm__throughput.avg.pct_of_peak_sustained_elapsed`) | 5.903979 | % |
| memory throughput (%) (`gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`) | 8.071948 | % |
| DRAM bytes read (`dram__bytes_read.sum`) | 0.008530 | Gbyte |
| DRAM bytes written (`dram__bytes_write.sum`) | 0.000000 | Mbyte |
| L2 hit rate (%) (`lts__t_sector_hit_rate.pct`) | 11.615461 | % |
| issue slots busy (%) (`smsp__issue_active.avg.pct_of_peak_sustained_active`) | 6.689643 | % |
| achieved occupancy (%) (`sm__warps_active.avg.pct_of_peak_sustained_active`) | 11.905237 | % |
| theoretical occupancy (%) (`sm__maximum_warps_per_active_cycle_pct`) | 100.000000 | % |
| registers per thread (`launch__registers_per_thread`) | 32 | register/thread |
| static shared memory per block (`launch__shared_mem_per_block_static`) | 0.000000 | Kbyte/block |
| dynamic shared memory per block (`launch__shared_mem_per_block_dynamic`) | 0 | byte/block |
| blocks per SM allowed by registers (`launch__occupancy_limit_registers`) | 16.000000 | block |
| blocks per SM allowed by shared memory (`launch__occupancy_limit_shared_mem`) | 32.000000 | block |
| waves per SM (`launch__waves_per_multiprocessor`) | 0.12 |  |
| shared-memory wavefronts (`l1tex__data_pipe_lsu_wavefronts_mem_shared.sum`) | 1024 |  |
| shared-memory bank conflicts (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`) | 0 |  |

| pipe utilization | % of peak (active cycles) |
|---|---|
| `sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active` | 2.988339 |
| `sm__inst_executed_pipe_fma.max.pct_of_peak_sustained_active` | 3.081725 |
| `sm__inst_executed_pipe_fma.min.pct_of_peak_sustained_active` | 1.540863 |
| `sm__inst_executed_pipe_fma.sum.pct_of_peak_sustained_active` | 2.988339 |
| `sm__inst_executed_pipe_fma_type_fp16.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_fmaheavy.avg.pct_of_peak_sustained_active` | 3.685437 |
| `sm__inst_executed_pipe_fmaheavy.max.pct_of_peak_sustained_active` | 3.812154 |
| `sm__inst_executed_pipe_fmaheavy.min.pct_of_peak_sustained_active` | 1.892969 |
| `sm__inst_executed_pipe_fmaheavy.sum.pct_of_peak_sustained_active` | 3.685437 |
| `sm__inst_executed_pipe_fmalite.avg.pct_of_peak_sustained_active` | 2.291241 |
| `sm__inst_executed_pipe_fmalite.max.pct_of_peak_sustained_active` | 2.370280 |
| `sm__inst_executed_pipe_fmalite.min.pct_of_peak_sustained_active` | 1.188756 |
| `sm__inst_executed_pipe_fmalite.sum.pct_of_peak_sustained_active` | 2.291241 |
| `sm__inst_executed_pipe_tensor_op_dmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active` | 0.000000 |
| `sm__inst_executed_pipe_tensor_op_hmma.max.pct_of_peak_sustained_active` | 0.000000 |
| `sm__inst_executed_pipe_tensor_op_hmma.min.pct_of_peak_sustained_active` | 0.000000 |
| `sm__inst_executed_pipe_tensor_op_hmma.sum.pct_of_peak_sustained_active` | 0.000000 |
| `sm__inst_executed_pipe_tensor_op_imma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` | 3.510795 |
| `sm__pipe_fma_cycles_active.max.pct_of_peak_sustained_active` | 3.620507 |
| `sm__pipe_fma_cycles_active.min.pct_of_peak_sustained_active` | 1.810254 |
| `sm__pipe_fma_cycles_active.sum.pct_of_peak_sustained_active` | 3.510795 |
| `sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active` | 4.730348 |
| `sm__pipe_fmaheavy_cycles_active.max.pct_of_peak_sustained_active` | 4.889719 |
| `sm__pipe_fmaheavy_cycles_active.min.pct_of_peak_sustained_active` | 2.431751 |
| `sm__pipe_fmaheavy_cycles_active.sum.pct_of_peak_sustained_active` | 4.730348 |
| `sm__pipe_fmalite_cycles_active.avg.pct_of_peak_sustained_active` | 2.291241 |
| `sm__pipe_fmalite_cycles_active.max.pct_of_peak_sustained_active` | 2.370280 |
| `sm__pipe_fmalite_cycles_active.min.pct_of_peak_sustained_active` | 1.188756 |
| `sm__pipe_fmalite_cycles_active.sum.pct_of_peak_sustained_active` | 2.291241 |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | 0.000000 |
| `sm__pipe_tensor_cycles_active.max.pct_of_peak_sustained_active` | 0.000000 |
| `sm__pipe_tensor_cycles_active.min.pct_of_peak_sustained_active` | 0.000000 |
| `sm__pipe_tensor_cycles_active.sum.pct_of_peak_sustained_active` | 0.000000 |
| `sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | 0.000000 |
| `sm__pipe_tensor_op_hmma_cycles_active.max.pct_of_peak_sustained_active` | 0.000000 |
| `sm__pipe_tensor_op_hmma_cycles_active.min.pct_of_peak_sustained_active` | 0.000000 |
| `sm__pipe_tensor_op_hmma_cycles_active.sum.pct_of_peak_sustained_active` | 0.000000 |
| `sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |

| stall reason | cycles per issued instruction |
|---|---|
| long_scoreboard | 24.19 |
| wait | 2.66 |
| selected | 1.00 |
| short_scoreboard | 0.63 |
| no_instruction | 0.21 |
| imc_miss | 0.10 |
| dispatch_stall | 0.08 |
| branch_resolving | 0.05 |
| not_selected | 0.02 |
| drain | 0.01 |
