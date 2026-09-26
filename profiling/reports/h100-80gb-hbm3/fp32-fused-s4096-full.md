## void unnamed>::fp32_fused_kernel<128, 32, 8>(AttnParams)

- grid (128, 128, 1), block (256, 1, 1)

| metric | value | unit |
|---|---|---|
| duration (`gpu__time_duration.sum`) | 142.902848 | ms |
| SM throughput (% of peak) (`sm__throughput.avg.pct_of_peak_sustained_elapsed`) | 86.866781 | % |
| memory throughput (%) (`gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`) | 93.603908 | % |
| DRAM bytes read (`dram__bytes_read.sum`) | 808.231680 | Mbyte |
| DRAM bytes written (`dram__bytes_write.sum`) | 268.504576 | Mbyte |
| L2 hit rate (%) (`lts__t_sector_hit_rate.pct`) | 98.198085 | % |
| issue slots busy (%) (`smsp__issue_active.avg.pct_of_peak_sustained_active`) | 55.198054 | % |
| achieved occupancy (%) (`sm__warps_active.avg.pct_of_peak_sustained_active`) | 37.252735 | % |
| theoretical occupancy (%) (`sm__maximum_warps_per_active_cycle_pct`) | 37.500000 | % |
| registers per thread (`launch__registers_per_thread`) | 78 | register/thread |
| static shared memory per block (`launch__shared_mem_per_block_static`) | 42.688000 | Kbyte/block |
| dynamic shared memory per block (`launch__shared_mem_per_block_dynamic`) | 0 | byte/block |
| blocks per SM allowed by registers (`launch__occupancy_limit_registers`) | 3.000000 | block |
| blocks per SM allowed by shared memory (`launch__occupancy_limit_shared_mem`) | 3.000000 | block |
| waves per SM (`launch__waves_per_multiprocessor`) | 41.37 |  |
| shared-memory wavefronts (`l1tex__data_pipe_lsu_wavefronts_mem_shared.sum`) | 34383442435 |  |
| shared-memory bank conflicts (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`) | 199543200 |  |

| pipe utilization | % of peak (active cycles) |
|---|---|
| `sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active` | 20.259528 |
| `sm__inst_executed_pipe_fma.max.pct_of_peak_sustained_active` | 20.402967 |
| `sm__inst_executed_pipe_fma.min.pct_of_peak_sustained_active` | 20.239743 |
| `sm__inst_executed_pipe_fma.sum.pct_of_peak_sustained_active` | 20.259528 |
| `sm__inst_executed_pipe_fma_type_fp16.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_fmaheavy.avg.pct_of_peak_sustained_active` | 16.908884 |
| `sm__inst_executed_pipe_fmaheavy.max.pct_of_peak_sustained_active` | 17.030002 |
| `sm__inst_executed_pipe_fmaheavy.min.pct_of_peak_sustained_active` | 16.890692 |
| `sm__inst_executed_pipe_fmaheavy.sum.pct_of_peak_sustained_active` | 16.908884 |
| `sm__inst_executed_pipe_fmalite.avg.pct_of_peak_sustained_active` | 23.610171 |
| `sm__inst_executed_pipe_fmalite.max.pct_of_peak_sustained_active` | 23.778285 |
| `sm__inst_executed_pipe_fmalite.min.pct_of_peak_sustained_active` | 23.585139 |
| `sm__inst_executed_pipe_fmalite.sum.pct_of_peak_sustained_active` | 23.610171 |
| `sm__inst_executed_pipe_tensor_op_dmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active` | 0.193116 |
| `sm__inst_executed_pipe_tensor_op_hmma.max.pct_of_peak_sustained_active` | 0.194483 |
| `sm__inst_executed_pipe_tensor_op_hmma.min.pct_of_peak_sustained_active` | 0.192927 |
| `sm__inst_executed_pipe_tensor_op_hmma.sum.pct_of_peak_sustained_active` | 0.193116 |
| `sm__inst_executed_pipe_tensor_op_imma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` | 20.625244 |
| `sm__pipe_fma_cycles_active.max.pct_of_peak_sustained_active` | 20.771272 |
| `sm__pipe_fma_cycles_active.min.pct_of_peak_sustained_active` | 20.605102 |
| `sm__pipe_fma_cycles_active.sum.pct_of_peak_sustained_active` | 20.625244 |
| `sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active` | 17.640316 |
| `sm__pipe_fmaheavy_cycles_active.max.pct_of_peak_sustained_active` | 17.766488 |
| `sm__pipe_fmaheavy_cycles_active.min.pct_of_peak_sustained_active` | 17.621409 |
| `sm__pipe_fmaheavy_cycles_active.sum.pct_of_peak_sustained_active` | 17.640316 |
| `sm__pipe_fmalite_cycles_active.avg.pct_of_peak_sustained_active` | 23.610171 |
| `sm__pipe_fmalite_cycles_active.max.pct_of_peak_sustained_active` | 23.778285 |
| `sm__pipe_fmalite_cycles_active.min.pct_of_peak_sustained_active` | 23.585139 |
| `sm__pipe_fmalite_cycles_active.sum.pct_of_peak_sustained_active` | 23.610171 |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | 0.193116 |
| `sm__pipe_tensor_cycles_active.max.pct_of_peak_sustained_active` | 0.194483 |
| `sm__pipe_tensor_cycles_active.min.pct_of_peak_sustained_active` | 0.192927 |
| `sm__pipe_tensor_cycles_active.sum.pct_of_peak_sustained_active` | 0.193116 |
| `sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | 0.193116 |
| `sm__pipe_tensor_op_hmma_cycles_active.max.pct_of_peak_sustained_active` | 0.194483 |
| `sm__pipe_tensor_op_hmma_cycles_active.min.pct_of_peak_sustained_active` | 0.192927 |
| `sm__pipe_tensor_op_hmma_cycles_active.sum.pct_of_peak_sustained_active` | 0.193116 |
| `sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |

| stall reason | cycles per issued instruction |
|---|---|
| mio_throttle | 3.18 |
| not_selected | 2.07 |
| wait | 1.89 |
| barrier | 1.02 |
| selected | 1.00 |
| short_scoreboard | 0.94 |
| dispatch_stall | 0.25 |
| long_scoreboard | 0.16 |
| math_pipe_throttle | 0.15 |
| branch_resolving | 0.08 |
