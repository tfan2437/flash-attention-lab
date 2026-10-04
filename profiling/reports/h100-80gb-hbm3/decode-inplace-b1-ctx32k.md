## void unnamed>::decode_kernel<__nv_bfloat16, 128, 1>(DecodeParams)

- grid (32, 1, 1), block (128, 1, 1)

| metric | value | unit |
|---|---|---|
| duration (`gpu__time_duration.sum`) | 4.685728 | ms |
| SM throughput (% of peak) (`sm__throughput.avg.pct_of_peak_sustained_elapsed`) | 1.017256 | % |
| memory throughput (%) (`gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`) | 3.437414 | % |
| DRAM bytes read (`dram__bytes_read.sum`) | 536.922368 | Mbyte |
| DRAM bytes written (`dram__bytes_write.sum`) | 3.026176 | Mbyte |
| L2 hit rate (%) (`lts__t_sector_hit_rate.pct`) | 0.264882 | % |
| issue slots busy (%) (`smsp__issue_active.avg.pct_of_peak_sustained_active`) | 4.283337 | % |
| achieved occupancy (%) (`sm__warps_active.avg.pct_of_peak_sustained_active`) | 6.249943 | % |
| theoretical occupancy (%) (`sm__maximum_warps_per_active_cycle_pct`) | 56.250000 | % |
| registers per thread (`launch__registers_per_thread`) | 56 | register/thread |
| static shared memory per block (`launch__shared_mem_per_block_static`) | 2.592000 | Kbyte/block |
| dynamic shared memory per block (`launch__shared_mem_per_block_dynamic`) | 0 | byte/block |
| blocks per SM allowed by registers (`launch__occupancy_limit_registers`) | 9.000000 | block |
| blocks per SM allowed by shared memory (`launch__occupancy_limit_shared_mem`) | 27.000000 | block |
| waves per SM (`launch__waves_per_multiprocessor`) | 0.03 |  |
| shared-memory wavefronts (`l1tex__data_pipe_lsu_wavefronts_mem_shared.sum`) | 2330345 |  |
| shared-memory bank conflicts (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`) | 457 |  |

| pipe utilization | % of peak (active cycles) |
|---|---|
| `sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active` | 1.823438 |
| `sm__inst_executed_pipe_fma.max.pct_of_peak_sustained_active` | 7.521680 |
| `sm__inst_executed_pipe_fma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_fma.sum.pct_of_peak_sustained_active` | 1.823438 |
| `sm__inst_executed_pipe_fma_type_fp16.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_fmaheavy.avg.pct_of_peak_sustained_active` | 2.198430 |
| `sm__inst_executed_pipe_fmaheavy.max.pct_of_peak_sustained_active` | 9.068524 |
| `sm__inst_executed_pipe_fmaheavy.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_fmaheavy.sum.pct_of_peak_sustained_active` | 2.198430 |
| `sm__inst_executed_pipe_fmalite.avg.pct_of_peak_sustained_active` | 1.448445 |
| `sm__inst_executed_pipe_fmalite.max.pct_of_peak_sustained_active` | 5.974836 |
| `sm__inst_executed_pipe_fmalite.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_fmalite.sum.pct_of_peak_sustained_active` | 1.448445 |
| `sm__inst_executed_pipe_tensor_op_dmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active` | 0.039501 |
| `sm__inst_executed_pipe_tensor_op_hmma.max.pct_of_peak_sustained_active` | 0.162940 |
| `sm__inst_executed_pipe_tensor_op_hmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_hmma.sum.pct_of_peak_sustained_active` | 0.039501 |
| `sm__inst_executed_pipe_tensor_op_imma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` | 1.936227 |
| `sm__pipe_fma_cycles_active.max.pct_of_peak_sustained_active` | 7.986937 |
| `sm__pipe_fma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_fma_cycles_active.sum.pct_of_peak_sustained_active` | 1.936227 |
| `sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active` | 2.424009 |
| `sm__pipe_fmaheavy_cycles_active.max.pct_of_peak_sustained_active` | 9.999039 |
| `sm__pipe_fmaheavy_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_fmaheavy_cycles_active.sum.pct_of_peak_sustained_active` | 2.424009 |
| `sm__pipe_fmalite_cycles_active.avg.pct_of_peak_sustained_active` | 1.448445 |
| `sm__pipe_fmalite_cycles_active.max.pct_of_peak_sustained_active` | 5.974836 |
| `sm__pipe_fmalite_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_fmalite_cycles_active.sum.pct_of_peak_sustained_active` | 1.448445 |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | 0.039501 |
| `sm__pipe_tensor_cycles_active.max.pct_of_peak_sustained_active` | 0.162940 |
| `sm__pipe_tensor_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_cycles_active.sum.pct_of_peak_sustained_active` | 0.039501 |
| `sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | 0.039501 |
| `sm__pipe_tensor_op_hmma_cycles_active.max.pct_of_peak_sustained_active` | 0.162940 |
| `sm__pipe_tensor_op_hmma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_hmma_cycles_active.sum.pct_of_peak_sustained_active` | 0.039501 |
| `sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |

| stall reason | cycles per issued instruction |
|---|---|
| long_scoreboard | 19.91 |
| wait | 1.68 |
| selected | 1.00 |
| short_scoreboard | 0.45 |
| branch_resolving | 0.21 |
| dispatch_stall | 0.05 |
| barrier | 0.03 |
| no_instruction | 0.01 |
| imc_miss | 0.00 |
| drain | 0.00 |
