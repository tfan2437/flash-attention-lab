## _attention_fwd_kernel

- grid (32, 128, 1), block (256, 1, 1)

| metric | value | unit |
|---|---|---|
| duration (`gpu__time_duration.sum`) | 2.362688 | ms |
| SM throughput (% of peak) (`sm__throughput.avg.pct_of_peak_sustained_elapsed`) | 47.642316 | % |
| memory throughput (%) (`gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`) | 34.488319 | % |
| DRAM bytes read (`dram__bytes_read.sum`) | 403.370752 | Mbyte |
| DRAM bytes written (`dram__bytes_write.sum`) | 129.092096 | Mbyte |
| L2 hit rate (%) (`lts__t_sector_hit_rate.pct`) | 89.491655 | % |
| issue slots busy (%) (`smsp__issue_active.avg.pct_of_peak_sustained_active`) | 34.927248 | % |
| achieved occupancy (%) (`sm__warps_active.avg.pct_of_peak_sustained_active`) | 12.495375 | % |
| theoretical occupancy (%) (`sm__maximum_warps_per_active_cycle_pct`) | 12.500000 | % |
| registers per thread (`launch__registers_per_thread`) | 244 | register/thread |
| static shared memory per block (`launch__shared_mem_per_block_static`) | 0 | byte/block |
| dynamic shared memory per block (`launch__shared_mem_per_block_dynamic`) | 229.376000 | Kbyte/block |
| blocks per SM allowed by registers (`launch__occupancy_limit_registers`) | 1.000000 | block |
| blocks per SM allowed by shared memory (`launch__occupancy_limit_shared_mem`) | 1.000000 | block |
| waves per SM (`launch__waves_per_multiprocessor`) | 31.03 |  |
| shared-memory wavefronts (`l1tex__data_pipe_lsu_wavefronts_mem_shared.sum`) | 190489686 |  |
| shared-memory bank conflicts (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`) | 287282 |  |

| pipe utilization | % of peak (active cycles) |
|---|---|
| `sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active` | 14.786906 |
| `sm__inst_executed_pipe_fma.max.pct_of_peak_sustained_active` | 15.248997 |
| `sm__inst_executed_pipe_fma.min.pct_of_peak_sustained_active` | 14.772466 |
| `sm__inst_executed_pipe_fma.sum.pct_of_peak_sustained_active` | 14.786906 |
| `sm__inst_executed_pipe_fma_type_fp16.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_fmaheavy.avg.pct_of_peak_sustained_active` | 10.381495 |
| `sm__inst_executed_pipe_fmaheavy.max.pct_of_peak_sustained_active` | 10.713261 |
| `sm__inst_executed_pipe_fmaheavy.min.pct_of_peak_sustained_active` | 10.363490 |
| `sm__inst_executed_pipe_fmaheavy.sum.pct_of_peak_sustained_active` | 10.381495 |
| `sm__inst_executed_pipe_fmalite.avg.pct_of_peak_sustained_active` | 19.192317 |
| `sm__inst_executed_pipe_fmalite.max.pct_of_peak_sustained_active` | 19.794134 |
| `sm__inst_executed_pipe_fmalite.min.pct_of_peak_sustained_active` | 19.166306 |
| `sm__inst_executed_pipe_fmalite.sum.pct_of_peak_sustained_active` | 19.192317 |
| `sm__inst_executed_pipe_tensor_op_dmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.avg.pct_of_peak_sustained_active` | 3.067935 |
| `sm__inst_executed_pipe_tensor_op_gmma.max.pct_of_peak_sustained_active` | 3.163808 |
| `sm__inst_executed_pipe_tensor_op_gmma.min.pct_of_peak_sustained_active` | 3.064939 |
| `sm__inst_executed_pipe_tensor_op_gmma.sum.pct_of_peak_sustained_active` | 3.067935 |
| `sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active` | 1.533967 |
| `sm__inst_executed_pipe_tensor_op_hmma.max.pct_of_peak_sustained_active` | 1.581904 |
| `sm__inst_executed_pipe_tensor_op_hmma.min.pct_of_peak_sustained_active` | 1.532469 |
| `sm__inst_executed_pipe_tensor_op_hmma.sum.pct_of_peak_sustained_active` | 1.533967 |
| `sm__inst_executed_pipe_tensor_op_imma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` | 15.705189 |
| `sm__pipe_fma_cycles_active.max.pct_of_peak_sustained_active` | 16.195976 |
| `sm__pipe_fma_cycles_active.min.pct_of_peak_sustained_active` | 15.689852 |
| `sm__pipe_fma_cycles_active.sum.pct_of_peak_sustained_active` | 15.705189 |
| `sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active` | 12.218062 |
| `sm__pipe_fmaheavy_cycles_active.max.pct_of_peak_sustained_active` | 12.607220 |
| `sm__pipe_fmaheavy_cycles_active.min.pct_of_peak_sustained_active` | 12.198263 |
| `sm__pipe_fmaheavy_cycles_active.sum.pct_of_peak_sustained_active` | 12.218062 |
| `sm__pipe_fmalite_cycles_active.avg.pct_of_peak_sustained_active` | 19.192317 |
| `sm__pipe_fmalite_cycles_active.max.pct_of_peak_sustained_active` | 19.794134 |
| `sm__pipe_fmalite_cycles_active.min.pct_of_peak_sustained_active` | 19.166306 |
| `sm__pipe_fmalite_cycles_active.sum.pct_of_peak_sustained_active` | 19.192317 |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | 49.086955 |
| `sm__pipe_tensor_cycles_active.max.pct_of_peak_sustained_active` | 50.620923 |
| `sm__pipe_tensor_cycles_active.min.pct_of_peak_sustained_active` | 49.039019 |
| `sm__pipe_tensor_cycles_active.sum.pct_of_peak_sustained_active` | 49.086955 |
| `sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | 49.086955 |
| `sm__pipe_tensor_op_hmma_cycles_active.max.pct_of_peak_sustained_active` | 50.620923 |
| `sm__pipe_tensor_op_hmma_cycles_active.min.pct_of_peak_sustained_active` | 49.039019 |
| `sm__pipe_tensor_op_hmma_cycles_active.sum.pct_of_peak_sustained_active` | 49.086955 |
| `sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |

| stall reason | cycles per issued instruction |
|---|---|
| wait | 1.33 |
| barrier | 1.32 |
| selected | 1.00 |
| mio_throttle | 0.61 |
| not_selected | 0.38 |
| dispatch_stall | 0.36 |
| long_scoreboard | 0.32 |
| short_scoreboard | 0.19 |
| math_pipe_throttle | 0.13 |
| gmma | 0.05 |
