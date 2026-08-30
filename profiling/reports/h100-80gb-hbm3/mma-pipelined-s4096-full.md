## void unnamed>::mma_pipelined_kernel<__nv_bfloat16, 128>(AttnParams)

- grid (64, 128, 1), block (128, 1, 1)

| metric | value | unit |
|---|---|---|
| duration (`gpu__time_duration.sum`) | 5.255616 | ms |
| SM throughput (% of peak) (`sm__throughput.avg.pct_of_peak_sustained_elapsed`) | 53.087957 | % |
| memory throughput (%) (`gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`) | 55.870523 | % |
| DRAM bytes read (`dram__bytes_read.sum`) | 402.917888 | Mbyte |
| DRAM bytes written (`dram__bytes_write.sum`) | 129.650432 | Mbyte |
| L2 hit rate (%) (`lts__t_sector_hit_rate.pct`) | 97.621144 | % |
| issue slots busy (%) (`smsp__issue_active.avg.pct_of_peak_sustained_active`) | 53.927667 | % |
| achieved occupancy (%) (`sm__warps_active.avg.pct_of_peak_sustained_active`) | 12.414229 | % |
| theoretical occupancy (%) (`sm__maximum_warps_per_active_cycle_pct`) | 12.500000 | % |
| registers per thread (`launch__registers_per_thread`) | 170 | register/thread |
| static shared memory per block (`launch__shared_mem_per_block_static`) | 0 | byte/block |
| dynamic shared memory per block (`launch__shared_mem_per_block_dynamic`) | 69.632000 | Kbyte/block |
| blocks per SM allowed by registers (`launch__occupancy_limit_registers`) | 2.000000 | block |
| blocks per SM allowed by shared memory (`launch__occupancy_limit_shared_mem`) | 2.000000 | block |
| waves per SM (`launch__waves_per_multiprocessor`) | 31.03 |  |
| shared-memory wavefronts (`l1tex__data_pipe_lsu_wavefronts_mem_shared.sum`) | 581159796 |  |
| shared-memory bank conflicts (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`) | 1950259 |  |

| pipe utilization | % of peak (active cycles) |
|---|---|
| `sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active` | 20.438430 |
| `sm__inst_executed_pipe_fma.max.pct_of_peak_sustained_active` | 20.747801 |
| `sm__inst_executed_pipe_fma.min.pct_of_peak_sustained_active` | 20.418471 |
| `sm__inst_executed_pipe_fma.sum.pct_of_peak_sustained_active` | 20.438430 |
| `sm__inst_executed_pipe_fma_type_fp16.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_fmaheavy.avg.pct_of_peak_sustained_active` | 27.220188 |
| `sm__inst_executed_pipe_fmaheavy.max.pct_of_peak_sustained_active` | 27.635437 |
| `sm__inst_executed_pipe_fmaheavy.min.pct_of_peak_sustained_active` | 27.190629 |
| `sm__inst_executed_pipe_fmaheavy.sum.pct_of_peak_sustained_active` | 27.220188 |
| `sm__inst_executed_pipe_fmalite.avg.pct_of_peak_sustained_active` | 13.656672 |
| `sm__inst_executed_pipe_fmalite.max.pct_of_peak_sustained_active` | 13.864656 |
| `sm__inst_executed_pipe_fmalite.min.pct_of_peak_sustained_active` | 13.639613 |
| `sm__inst_executed_pipe_fmalite.sum.pct_of_peak_sustained_active` | 13.656672 |
| `sm__inst_executed_pipe_tensor_op_dmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active` | 10.831390 |
| `sm__inst_executed_pipe_tensor_op_hmma.max.pct_of_peak_sustained_active` | 10.995341 |
| `sm__inst_executed_pipe_tensor_op_hmma.min.pct_of_peak_sustained_active` | 10.820812 |
| `sm__inst_executed_pipe_tensor_op_hmma.sum.pct_of_peak_sustained_active` | 10.831390 |
| `sm__inst_executed_pipe_tensor_op_imma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` | 21.126631 |
| `sm__pipe_fma_cycles_active.max.pct_of_peak_sustained_active` | 21.446419 |
| `sm__pipe_fma_cycles_active.min.pct_of_peak_sustained_active` | 21.105999 |
| `sm__pipe_fma_cycles_active.sum.pct_of_peak_sustained_active` | 21.126631 |
| `sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active` | 28.596589 |
| `sm__pipe_fmaheavy_cycles_active.max.pct_of_peak_sustained_active` | 29.028933 |
| `sm__pipe_fmaheavy_cycles_active.min.pct_of_peak_sustained_active` | 28.565686 |
| `sm__pipe_fmaheavy_cycles_active.sum.pct_of_peak_sustained_active` | 28.596589 |
| `sm__pipe_fmalite_cycles_active.avg.pct_of_peak_sustained_active` | 13.656672 |
| `sm__pipe_fmalite_cycles_active.max.pct_of_peak_sustained_active` | 13.864656 |
| `sm__pipe_fmalite_cycles_active.min.pct_of_peak_sustained_active` | 13.639613 |
| `sm__pipe_fmalite_cycles_active.sum.pct_of_peak_sustained_active` | 13.656672 |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | 32.494169 |
| `sm__pipe_tensor_cycles_active.max.pct_of_peak_sustained_active` | 32.986024 |
| `sm__pipe_tensor_cycles_active.min.pct_of_peak_sustained_active` | 32.462436 |
| `sm__pipe_tensor_cycles_active.sum.pct_of_peak_sustained_active` | 32.494169 |
| `sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | 32.494169 |
| `sm__pipe_tensor_op_hmma_cycles_active.max.pct_of_peak_sustained_active` | 32.986024 |
| `sm__pipe_tensor_op_hmma_cycles_active.min.pct_of_peak_sustained_active` | 32.462436 |
| `sm__pipe_tensor_op_hmma_cycles_active.sum.pct_of_peak_sustained_active` | 32.494169 |
| `sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |

| stall reason | cycles per issued instruction |
|---|---|
| wait | 1.20 |
| selected | 1.00 |
| short_scoreboard | 0.40 |
| not_selected | 0.39 |
| math_pipe_throttle | 0.21 |
| dispatch_stall | 0.19 |
| long_scoreboard | 0.09 |
| barrier | 0.09 |
| branch_resolving | 0.07 |
| no_instruction | 0.03 |
