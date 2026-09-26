## void unnamed>::fp32_regtile_kernel<128>(AttnParams)

- grid (64, 128, 1), block (128, 1, 1)

| metric | value | unit |
|---|---|---|
| duration (`gpu__time_duration.sum`) | 62.023328 | ms |
| SM throughput (% of peak) (`sm__throughput.avg.pct_of_peak_sustained_elapsed`) | 35.032956 | % |
| memory throughput (%) (`gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`) | 35.235347 | % |
| DRAM bytes read (`dram__bytes_read.sum`) | 806.533376 | Mbyte |
| DRAM bytes written (`dram__bytes_write.sum`) | 265.699840 | Mbyte |
| L2 hit rate (%) (`lts__t_sector_hit_rate.pct`) | 93.025682 | % |
| issue slots busy (%) (`smsp__issue_active.avg.pct_of_peak_sustained_active`) | 35.187223 | % |
| achieved occupancy (%) (`sm__warps_active.avg.pct_of_peak_sustained_active`) | 6.249742 | % |
| theoretical occupancy (%) (`sm__maximum_warps_per_active_cycle_pct`) | 6.250000 | % |
| registers per thread (`launch__registers_per_thread`) | 254 | register/thread |
| static shared memory per block (`launch__shared_mem_per_block_static`) | 0 | byte/block |
| dynamic shared memory per block (`launch__shared_mem_per_block_dynamic`) | 119.808000 | Kbyte/block |
| blocks per SM allowed by registers (`launch__occupancy_limit_registers`) | 2.000000 | block |
| blocks per SM allowed by shared memory (`launch__occupancy_limit_shared_mem`) | 1.000000 | block |
| waves per SM (`launch__waves_per_multiprocessor`) | 62.06 |  |
| shared-memory wavefronts (`l1tex__data_pipe_lsu_wavefronts_mem_shared.sum`) | 5467138740 |  |
| shared-memory bank conflicts (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`) | 486068 |  |

| pipe utilization | % of peak (active cycles) |
|---|---|
| `sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active` | 28.272287 |
| `sm__inst_executed_pipe_fma.max.pct_of_peak_sustained_active` | 28.700237 |
| `sm__inst_executed_pipe_fma.min.pct_of_peak_sustained_active` | 26.877999 |
| `sm__inst_executed_pipe_fma.sum.pct_of_peak_sustained_active` | 28.272287 |
| `sm__inst_executed_pipe_fma_type_fp16.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_fmaheavy.avg.pct_of_peak_sustained_active` | 27.318348 |
| `sm__inst_executed_pipe_fmaheavy.max.pct_of_peak_sustained_active` | 27.733812 |
| `sm__inst_executed_pipe_fmaheavy.min.pct_of_peak_sustained_active` | 25.969894 |
| `sm__inst_executed_pipe_fmaheavy.sum.pct_of_peak_sustained_active` | 27.318348 |
| `sm__inst_executed_pipe_fmalite.avg.pct_of_peak_sustained_active` | 29.226226 |
| `sm__inst_executed_pipe_fmalite.max.pct_of_peak_sustained_active` | 29.671050 |
| `sm__inst_executed_pipe_fmalite.min.pct_of_peak_sustained_active` | 27.783696 |
| `sm__inst_executed_pipe_fmalite.sum.pct_of_peak_sustained_active` | 29.226226 |
| `sm__inst_executed_pipe_tensor_op_dmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active` | 0.170638 |
| `sm__inst_executed_pipe_tensor_op_hmma.max.pct_of_peak_sustained_active` | 0.173221 |
| `sm__inst_executed_pipe_tensor_op_hmma.min.pct_of_peak_sustained_active` | 0.162223 |
| `sm__inst_executed_pipe_tensor_op_hmma.sum.pct_of_peak_sustained_active` | 0.170638 |
| `sm__inst_executed_pipe_tensor_op_imma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` | 28.531864 |
| `sm__pipe_fma_cycles_active.max.pct_of_peak_sustained_active` | 28.963743 |
| `sm__pipe_fma_cycles_active.min.pct_of_peak_sustained_active` | 27.133140 |
| `sm__pipe_fma_cycles_active.sum.pct_of_peak_sustained_active` | 28.531864 |
| `sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active` | 27.837502 |
| `sm__pipe_fmaheavy_cycles_active.max.pct_of_peak_sustained_active` | 28.260825 |
| `sm__pipe_fmaheavy_cycles_active.min.pct_of_peak_sustained_active` | 26.482585 |
| `sm__pipe_fmaheavy_cycles_active.sum.pct_of_peak_sustained_active` | 27.837502 |
| `sm__pipe_fmalite_cycles_active.avg.pct_of_peak_sustained_active` | 29.226226 |
| `sm__pipe_fmalite_cycles_active.max.pct_of_peak_sustained_active` | 29.671050 |
| `sm__pipe_fmalite_cycles_active.min.pct_of_peak_sustained_active` | 27.783696 |
| `sm__pipe_fmalite_cycles_active.sum.pct_of_peak_sustained_active` | 29.226226 |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | 0.170638 |
| `sm__pipe_tensor_cycles_active.max.pct_of_peak_sustained_active` | 0.173221 |
| `sm__pipe_tensor_cycles_active.min.pct_of_peak_sustained_active` | 0.167722 |
| `sm__pipe_tensor_cycles_active.sum.pct_of_peak_sustained_active` | 0.170638 |
| `sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | 0.170638 |
| `sm__pipe_tensor_op_hmma_cycles_active.max.pct_of_peak_sustained_active` | 0.173221 |
| `sm__pipe_tensor_op_hmma_cycles_active.min.pct_of_peak_sustained_active` | 0.167722 |
| `sm__pipe_tensor_op_hmma_cycles_active.sum.pct_of_peak_sustained_active` | 0.170638 |
| `sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |

| stall reason | cycles per issued instruction |
|---|---|
| long_scoreboard | 1.13 |
| selected | 1.00 |
| wait | 0.26 |
| dispatch_stall | 0.20 |
| short_scoreboard | 0.18 |
| branch_resolving | 0.04 |
| barrier | 0.02 |
| no_instruction | 0.02 |
| drain | 0.00 |
| math_pipe_throttle | 0.00 |
