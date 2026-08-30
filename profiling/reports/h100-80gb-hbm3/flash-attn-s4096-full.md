## void flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, 0, 0, bfloat16_t, Flash_kernel_traits<128, 128, 64, 4, bfloat16_t>>, 0, 0, 0, 0, 1, 1, 0, 0>(Flash_fwd_params)

- grid (32, 4, 32), block (128, 1, 1)

| metric | value | unit |
|---|---|---|
| duration (`gpu__time_duration.sum`) | 2.781824 | ms |
| SM throughput (% of peak) (`sm__throughput.avg.pct_of_peak_sustained_elapsed`) | 60.719669 | % |
| memory throughput (%) (`gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`) | 56.512163 | % |
| DRAM bytes read (`dram__bytes_read.sum`) | 408.313600 | Mbyte |
| DRAM bytes written (`dram__bytes_write.sum`) | 194.341632 | Mbyte |
| L2 hit rate (%) (`lts__t_sector_hit_rate.pct`) | 90.878324 | % |
| issue slots busy (%) (`smsp__issue_active.avg.pct_of_peak_sustained_active`) | 38.659950 | % |
| achieved occupancy (%) (`sm__warps_active.avg.pct_of_peak_sustained_active`) | 12.349486 | % |
| theoretical occupancy (%) (`sm__maximum_warps_per_active_cycle_pct`) | 12.500000 | % |
| registers per thread (`launch__registers_per_thread`) | 255 | register/thread |
| static shared memory per block (`launch__shared_mem_per_block_static`) | 0 | byte/block |
| dynamic shared memory per block (`launch__shared_mem_per_block_dynamic`) | 65.536000 | Kbyte/block |
| blocks per SM allowed by registers (`launch__occupancy_limit_registers`) | 2.000000 | block |
| blocks per SM allowed by shared memory (`launch__occupancy_limit_shared_mem`) | 2.000000 | block |
| waves per SM (`launch__waves_per_multiprocessor`) | 15.52 |  |
| shared-memory wavefronts (`l1tex__data_pipe_lsu_wavefronts_mem_shared.sum`) | 353304784 |  |
| shared-memory bank conflicts (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`) | 243809 |  |

| pipe utilization | % of peak (active cycles) |
|---|---|
| `sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active` | 11.387074 |
| `sm__inst_executed_pipe_fma.max.pct_of_peak_sustained_active` | 11.742920 |
| `sm__inst_executed_pipe_fma.min.pct_of_peak_sustained_active` | 11.375954 |
| `sm__inst_executed_pipe_fma.sum.pct_of_peak_sustained_active` | 11.387074 |
| `sm__inst_executed_pipe_fma_type_fp16.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_fmaheavy.avg.pct_of_peak_sustained_active` | 7.678137 |
| `sm__inst_executed_pipe_fmaheavy.max.pct_of_peak_sustained_active` | 7.920326 |
| `sm__inst_executed_pipe_fmaheavy.min.pct_of_peak_sustained_active` | 7.664086 |
| `sm__inst_executed_pipe_fmaheavy.sum.pct_of_peak_sustained_active` | 7.678137 |
| `sm__inst_executed_pipe_fmalite.avg.pct_of_peak_sustained_active` | 15.096010 |
| `sm__inst_executed_pipe_fmalite.max.pct_of_peak_sustained_active` | 15.572093 |
| `sm__inst_executed_pipe_fmalite.min.pct_of_peak_sustained_active` | 15.073589 |
| `sm__inst_executed_pipe_fmalite.sum.pct_of_peak_sustained_active` | 15.096010 |
| `sm__inst_executed_pipe_tensor_op_dmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active` | 20.806983 |
| `sm__inst_executed_pipe_tensor_op_hmma.max.pct_of_peak_sustained_active` | 21.457202 |
| `sm__inst_executed_pipe_tensor_op_hmma.min.pct_of_peak_sustained_active` | 20.786664 |
| `sm__inst_executed_pipe_tensor_op_hmma.sum.pct_of_peak_sustained_active` | 20.806983 |
| `sm__inst_executed_pipe_tensor_op_imma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` | 11.479781 |
| `sm__pipe_fma_cycles_active.max.pct_of_peak_sustained_active` | 11.838524 |
| `sm__pipe_fma_cycles_active.min.pct_of_peak_sustained_active` | 11.468570 |
| `sm__pipe_fma_cycles_active.sum.pct_of_peak_sustained_active` | 11.479781 |
| `sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active` | 7.863551 |
| `sm__pipe_fmaheavy_cycles_active.max.pct_of_peak_sustained_active` | 8.105559 |
| `sm__pipe_fmaheavy_cycles_active.min.pct_of_peak_sustained_active` | 7.849319 |
| `sm__pipe_fmaheavy_cycles_active.sum.pct_of_peak_sustained_active` | 7.863551 |
| `sm__pipe_fmalite_cycles_active.avg.pct_of_peak_sustained_active` | 15.096010 |
| `sm__pipe_fmalite_cycles_active.max.pct_of_peak_sustained_active` | 15.572093 |
| `sm__pipe_fmalite_cycles_active.min.pct_of_peak_sustained_active` | 15.073589 |
| `sm__pipe_fmalite_cycles_active.sum.pct_of_peak_sustained_active` | 15.096010 |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | 62.420950 |
| `sm__pipe_tensor_cycles_active.max.pct_of_peak_sustained_active` | 64.371605 |
| `sm__pipe_tensor_cycles_active.min.pct_of_peak_sustained_active` | 62.359992 |
| `sm__pipe_tensor_cycles_active.sum.pct_of_peak_sustained_active` | 62.420950 |
| `sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | 62.420950 |
| `sm__pipe_tensor_op_hmma_cycles_active.max.pct_of_peak_sustained_active` | 64.371605 |
| `sm__pipe_tensor_op_hmma_cycles_active.min.pct_of_peak_sustained_active` | 62.359992 |
| `sm__pipe_tensor_op_hmma_cycles_active.sum.pct_of_peak_sustained_active` | 62.420950 |
| `sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |

| stall reason | cycles per issued instruction |
|---|---|
| wait | 1.86 |
| selected | 1.00 |
| math_pipe_throttle | 0.83 |
| short_scoreboard | 0.38 |
| not_selected | 0.38 |
| dispatch_stall | 0.21 |
| long_scoreboard | 0.19 |
| barrier | 0.13 |
| mio_throttle | 0.08 |
| no_instruction | 0.03 |
