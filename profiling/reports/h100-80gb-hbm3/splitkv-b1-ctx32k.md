## void unnamed>::decode_kernel<__nv_bfloat16, 128, 1>(DecodeParams)

- grid (32, 64, 1), block (128, 1, 1)

| metric | value | unit |
|---|---|---|
| duration (`gpu__time_duration.sum`) | 211.328000 | us |
| SM throughput (% of peak) (`sm__throughput.avg.pct_of_peak_sustained_elapsed`) | 23.885632 | % |
| memory throughput (%) (`gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`) | 76.288223 | % |
| DRAM bytes read (`dram__bytes_read.sum`) | 536.957184 | Mbyte |
| DRAM bytes written (`dram__bytes_write.sum`) | 3.414784 | Mbyte |
| L2 hit rate (%) (`lts__t_sector_hit_rate.pct`) | 1.321536 | % |
| issue slots busy (%) (`smsp__issue_active.avg.pct_of_peak_sustained_active`) | 24.728903 | % |
| achieved occupancy (%) (`sm__warps_active.avg.pct_of_peak_sustained_active`) | 48.932178 | % |
| theoretical occupancy (%) (`sm__maximum_warps_per_active_cycle_pct`) | 56.250000 | % |
| registers per thread (`launch__registers_per_thread`) | 56 | register/thread |
| static shared memory per block (`launch__shared_mem_per_block_static`) | 2.592000 | Kbyte/block |
| dynamic shared memory per block (`launch__shared_mem_per_block_dynamic`) | 0 | byte/block |
| blocks per SM allowed by registers (`launch__occupancy_limit_registers`) | 9.000000 | block |
| blocks per SM allowed by shared memory (`launch__occupancy_limit_shared_mem`) | 27.000000 | block |
| waves per SM (`launch__waves_per_multiprocessor`) | 1.72 |  |
| shared-memory wavefronts (`l1tex__data_pipe_lsu_wavefronts_mem_shared.sum`) | 2549437 |  |
| shared-memory bank conflicts (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`) | 7877 |  |

| pipe utilization | % of peak (active cycles) |
|---|---|
| `sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active` | 10.249811 |
| `sm__inst_executed_pipe_fma.max.pct_of_peak_sustained_active` | 11.891383 |
| `sm__inst_executed_pipe_fma.min.pct_of_peak_sustained_active` | 6.606324 |
| `sm__inst_executed_pipe_fma.sum.pct_of_peak_sustained_active` | 10.249811 |
| `sm__inst_executed_pipe_fma_type_fp16.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_fmaheavy.avg.pct_of_peak_sustained_active` | 12.638079 |
| `sm__inst_executed_pipe_fmaheavy.max.pct_of_peak_sustained_active` | 14.715101 |
| `sm__inst_executed_pipe_fmaheavy.min.pct_of_peak_sustained_active` | 8.125238 |
| `sm__inst_executed_pipe_fmaheavy.sum.pct_of_peak_sustained_active` | 12.638079 |
| `sm__inst_executed_pipe_fmalite.avg.pct_of_peak_sustained_active` | 7.861544 |
| `sm__inst_executed_pipe_fmalite.max.pct_of_peak_sustained_active` | 9.102305 |
| `sm__inst_executed_pipe_fmalite.min.pct_of_peak_sustained_active` | 5.087410 |
| `sm__inst_executed_pipe_fmalite.sum.pct_of_peak_sustained_active` | 7.861544 |
| `sm__inst_executed_pipe_tensor_op_dmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active` | 0.228705 |
| `sm__inst_executed_pipe_tensor_op_hmma.max.pct_of_peak_sustained_active` | 0.265334 |
| `sm__inst_executed_pipe_tensor_op_hmma.min.pct_of_peak_sustained_active` | 0.162149 |
| `sm__inst_executed_pipe_tensor_op_hmma.sum.pct_of_peak_sustained_active` | 0.228705 |
| `sm__inst_executed_pipe_tensor_op_imma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` | 10.901622 |
| `sm__pipe_fma_cycles_active.max.pct_of_peak_sustained_active` | 12.647585 |
| `sm__pipe_fma_cycles_active.min.pct_of_peak_sustained_active` | 7.068447 |
| `sm__pipe_fma_cycles_active.sum.pct_of_peak_sustained_active` | 10.901622 |
| `sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active` | 13.941700 |
| `sm__pipe_fmaheavy_cycles_active.max.pct_of_peak_sustained_active` | 16.225908 |
| `sm__pipe_fmaheavy_cycles_active.min.pct_of_peak_sustained_active` | 9.049484 |
| `sm__pipe_fmaheavy_cycles_active.sum.pct_of_peak_sustained_active` | 13.941700 |
| `sm__pipe_fmalite_cycles_active.avg.pct_of_peak_sustained_active` | 7.861544 |
| `sm__pipe_fmalite_cycles_active.max.pct_of_peak_sustained_active` | 9.102305 |
| `sm__pipe_fmalite_cycles_active.min.pct_of_peak_sustained_active` | 5.087410 |
| `sm__pipe_fmalite_cycles_active.sum.pct_of_peak_sustained_active` | 7.861544 |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | 0.228705 |
| `sm__pipe_tensor_cycles_active.max.pct_of_peak_sustained_active` | 0.265334 |
| `sm__pipe_tensor_cycles_active.min.pct_of_peak_sustained_active` | 0.162149 |
| `sm__pipe_tensor_cycles_active.sum.pct_of_peak_sustained_active` | 0.228705 |
| `sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | 0.228705 |
| `sm__pipe_tensor_op_hmma_cycles_active.max.pct_of_peak_sustained_active` | 0.265334 |
| `sm__pipe_tensor_op_hmma_cycles_active.min.pct_of_peak_sustained_active` | 0.162149 |
| `sm__pipe_tensor_op_hmma_cycles_active.sum.pct_of_peak_sustained_active` | 0.228705 |
| `sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |

| stall reason | cycles per issued instruction |
|---|---|
| long_scoreboard | 27.23 |
| wait | 1.72 |
| selected | 1.00 |
| short_scoreboard | 0.49 |
| barrier | 0.46 |
| not_selected | 0.31 |
| branch_resolving | 0.23 |
| math_pipe_throttle | 0.13 |
| dispatch_stall | 0.09 |
| imc_miss | 0.07 |

## void unnamed>::merge_kernel<__nv_bfloat16>(DecodeParams)

- grid (32, 1, 1), block (128, 1, 1)

| metric | value | unit |
|---|---|---|
| duration (`gpu__time_duration.sum`) | 31.744000 | us |
| SM throughput (% of peak) (`sm__throughput.avg.pct_of_peak_sustained_elapsed`) | 0.722405 | % |
| memory throughput (%) (`gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`) | 1.329773 | % |
| DRAM bytes read (`dram__bytes_read.sum`) | 1.075456 | Mbyte |
| DRAM bytes written (`dram__bytes_write.sum`) | 0.000000 | Mbyte |
| L2 hit rate (%) (`lts__t_sector_hit_rate.pct`) | 48.182238 | % |
| issue slots busy (%) (`smsp__issue_active.avg.pct_of_peak_sustained_active`) | 3.354923 | % |
| achieved occupancy (%) (`sm__warps_active.avg.pct_of_peak_sustained_active`) | 6.208357 | % |
| theoretical occupancy (%) (`sm__maximum_warps_per_active_cycle_pct`) | 100.000000 | % |
| registers per thread (`launch__registers_per_thread`) | 32 | register/thread |
| static shared memory per block (`launch__shared_mem_per_block_static`) | 0.000000 | Kbyte/block |
| dynamic shared memory per block (`launch__shared_mem_per_block_dynamic`) | 0 | byte/block |
| blocks per SM allowed by registers (`launch__occupancy_limit_registers`) | 16.000000 | block |
| blocks per SM allowed by shared memory (`launch__occupancy_limit_shared_mem`) | 32.000000 | block |
| waves per SM (`launch__waves_per_multiprocessor`) | 0.02 |  |
| shared-memory wavefronts (`l1tex__data_pipe_lsu_wavefronts_mem_shared.sum`) | 128 |  |
| shared-memory bank conflicts (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`) | 0 |  |

| pipe utilization | % of peak (active cycles) |
|---|---|
| `sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active` | 1.507597 |
| `sm__inst_executed_pipe_fma.max.pct_of_peak_sustained_active` | 6.218839 |
| `sm__inst_executed_pipe_fma.min.pct_of_peak_sustained_active` | 0.000000 |
| `sm__inst_executed_pipe_fma.sum.pct_of_peak_sustained_active` | 1.507597 |
| `sm__inst_executed_pipe_fma_type_fp16.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_fmaheavy.avg.pct_of_peak_sustained_active` | 1.852103 |
| `sm__inst_executed_pipe_fmaheavy.max.pct_of_peak_sustained_active` | 7.639923 |
| `sm__inst_executed_pipe_fmaheavy.min.pct_of_peak_sustained_active` | 0.000000 |
| `sm__inst_executed_pipe_fmaheavy.sum.pct_of_peak_sustained_active` | 1.852103 |
| `sm__inst_executed_pipe_fmalite.avg.pct_of_peak_sustained_active` | 1.163092 |
| `sm__inst_executed_pipe_fmalite.max.pct_of_peak_sustained_active` | 4.797755 |
| `sm__inst_executed_pipe_fmalite.min.pct_of_peak_sustained_active` | 0.000000 |
| `sm__inst_executed_pipe_fmalite.sum.pct_of_peak_sustained_active` | 1.163092 |
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
| `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` | 1.771173 |
| `sm__pipe_fma_cycles_active.max.pct_of_peak_sustained_active` | 7.306087 |
| `sm__pipe_fma_cycles_active.min.pct_of_peak_sustained_active` | 0.000000 |
| `sm__pipe_fma_cycles_active.sum.pct_of_peak_sustained_active` | 1.771173 |
| `sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active` | 2.379253 |
| `sm__pipe_fmaheavy_cycles_active.max.pct_of_peak_sustained_active` | 9.814419 |
| `sm__pipe_fmaheavy_cycles_active.min.pct_of_peak_sustained_active` | 0.000000 |
| `sm__pipe_fmaheavy_cycles_active.sum.pct_of_peak_sustained_active` | 2.379253 |
| `sm__pipe_fmalite_cycles_active.avg.pct_of_peak_sustained_active` | 1.163092 |
| `sm__pipe_fmalite_cycles_active.max.pct_of_peak_sustained_active` | 4.797755 |
| `sm__pipe_fmalite_cycles_active.min.pct_of_peak_sustained_active` | 0.000000 |
| `sm__pipe_fmalite_cycles_active.sum.pct_of_peak_sustained_active` | 1.163092 |
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
| long_scoreboard | 25.02 |
| wait | 2.64 |
| selected | 1.00 |
| short_scoreboard | 0.61 |
| no_instruction | 0.26 |
| imc_miss | 0.09 |
| dispatch_stall | 0.07 |
| branch_resolving | 0.05 |
| drain | 0.01 |
| misc | 0.00 |
