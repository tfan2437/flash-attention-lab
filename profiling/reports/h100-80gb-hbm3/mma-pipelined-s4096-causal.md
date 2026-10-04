## void unnamed>::mma_pipelined_kernel<__nv_bfloat16, 128>(AttnParams)

- grid (64, 128, 1), block (128, 1, 1)

| metric | value | unit |
|---|---|---|
| duration (`gpu__time_duration.sum`) | 2.414784 | ms |
| SM throughput (% of peak) (`sm__throughput.avg.pct_of_peak_sustained_elapsed`) | 51.565147 | % |
| memory throughput (%) (`gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`) | 62.720924 | % |
| DRAM bytes read (`dram__bytes_read.sum`) | 402.834432 | Mbyte |
| DRAM bytes written (`dram__bytes_write.sum`) | 130.567168 | Mbyte |
| L2 hit rate (%) (`lts__t_sector_hit_rate.pct`) | 95.389906 | % |
| issue slots busy (%) (`smsp__issue_active.avg.pct_of_peak_sustained_active`) | 52.785502 | % |
| achieved occupancy (%) (`sm__warps_active.avg.pct_of_peak_sustained_active`) | 12.404866 | % |
| theoretical occupancy (%) (`sm__maximum_warps_per_active_cycle_pct`) | 12.500000 | % |
| registers per thread (`launch__registers_per_thread`) | 214 | register/thread |
| static shared memory per block (`launch__shared_mem_per_block_static`) | 0 | byte/block |
| dynamic shared memory per block (`launch__shared_mem_per_block_dynamic`) | 69.632000 | Kbyte/block |
| blocks per SM allowed by registers (`launch__occupancy_limit_registers`) | 2.000000 | block |
| blocks per SM allowed by shared memory (`launch__occupancy_limit_shared_mem`) | 2.000000 | block |
| waves per SM (`launch__waves_per_multiprocessor`) | 31.03 |  |
| shared-memory wavefronts (`l1tex__data_pipe_lsu_wavefronts_mem_shared.sum`) | 295569767 |  |
| shared-memory bank conflicts (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`) | 751044 |  |

| pipe utilization | % of peak (active cycles) |
|---|---|
| `sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active` | 20.981362 |
| `sm__inst_executed_pipe_fma.max.pct_of_peak_sustained_active` | 21.586166 |
| `sm__inst_executed_pipe_fma.min.pct_of_peak_sustained_active` | 20.594803 |
| `sm__inst_executed_pipe_fma.sum.pct_of_peak_sustained_active` | 20.981362 |
| `sm__inst_executed_pipe_fma_type_fp16.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_fmaheavy.avg.pct_of_peak_sustained_active` | 28.645809 |
| `sm__inst_executed_pipe_fmaheavy.max.pct_of_peak_sustained_active` | 29.362477 |
| `sm__inst_executed_pipe_fmaheavy.min.pct_of_peak_sustained_active` | 28.029259 |
| `sm__inst_executed_pipe_fmaheavy.sum.pct_of_peak_sustained_active` | 28.645809 |
| `sm__inst_executed_pipe_fmalite.avg.pct_of_peak_sustained_active` | 13.316915 |
| `sm__inst_executed_pipe_fmalite.max.pct_of_peak_sustained_active` | 13.657302 |
| `sm__inst_executed_pipe_fmalite.min.pct_of_peak_sustained_active` | 13.002661 |
| `sm__inst_executed_pipe_fmalite.sum.pct_of_peak_sustained_active` | 13.316915 |
| `sm__inst_executed_pipe_tensor_op_dmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active` | 12.069342 |
| `sm__inst_executed_pipe_tensor_op_hmma.max.pct_of_peak_sustained_active` | 12.404622 |
| `sm__inst_executed_pipe_tensor_op_hmma.min.pct_of_peak_sustained_active` | 11.800248 |
| `sm__inst_executed_pipe_tensor_op_hmma.sum.pct_of_peak_sustained_active` | 12.069342 |
| `sm__inst_executed_pipe_tensor_op_imma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` | 21.760357 |
| `sm__pipe_fma_cycles_active.max.pct_of_peak_sustained_active` | 22.287708 |
| `sm__pipe_fma_cycles_active.min.pct_of_peak_sustained_active` | 21.291904 |
| `sm__pipe_fma_cycles_active.sum.pct_of_peak_sustained_active` | 21.760357 |
| `sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active` | 30.203798 |
| `sm__pipe_fmaheavy_cycles_active.max.pct_of_peak_sustained_active` | 30.923105 |
| `sm__pipe_fmaheavy_cycles_active.min.pct_of_peak_sustained_active` | 29.581146 |
| `sm__pipe_fmaheavy_cycles_active.sum.pct_of_peak_sustained_active` | 30.203798 |
| `sm__pipe_fmalite_cycles_active.avg.pct_of_peak_sustained_active` | 13.316915 |
| `sm__pipe_fmalite_cycles_active.max.pct_of_peak_sustained_active` | 13.657302 |
| `sm__pipe_fmalite_cycles_active.min.pct_of_peak_sustained_active` | 13.002661 |
| `sm__pipe_fmalite_cycles_active.sum.pct_of_peak_sustained_active` | 13.316915 |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | 36.208027 |
| `sm__pipe_tensor_cycles_active.max.pct_of_peak_sustained_active` | 37.519045 |
| `sm__pipe_tensor_cycles_active.min.pct_of_peak_sustained_active` | 35.526406 |
| `sm__pipe_tensor_cycles_active.sum.pct_of_peak_sustained_active` | 36.208027 |
| `sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | 36.208027 |
| `sm__pipe_tensor_op_hmma_cycles_active.max.pct_of_peak_sustained_active` | 37.519045 |
| `sm__pipe_tensor_op_hmma_cycles_active.min.pct_of_peak_sustained_active` | 35.526406 |
| `sm__pipe_tensor_op_hmma_cycles_active.sum.pct_of_peak_sustained_active` | 36.208027 |
| `sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |

| stall reason | cycles per issued instruction |
|---|---|
| wait | 1.12 |
| selected | 1.00 |
| not_selected | 0.38 |
| short_scoreboard | 0.32 |
| dispatch_stall | 0.29 |
| math_pipe_throttle | 0.25 |
| long_scoreboard | 0.14 |
| branch_resolving | 0.10 |
| barrier | 0.09 |
| no_instruction | 0.06 |
