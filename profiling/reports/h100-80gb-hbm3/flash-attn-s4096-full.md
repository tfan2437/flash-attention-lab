## void flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, 0, 0, bfloat16_t, Flash_kernel_traits<128, 128, 64, 4, bfloat16_t>>, 0, 0, 0, 0, 1, 1, 0, 0>(Flash_fwd_params)

- grid (32, 4, 32), block (128, 1, 1)

| metric | value | unit |
|---|---|---|
| duration (`gpu__time_duration.sum`) | 2.756928 | ms |
| SM throughput (% of peak) (`sm__throughput.avg.pct_of_peak_sustained_elapsed`) | 60.678598 | % |
| memory throughput (%) (`gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`) | 56.474343 | % |
| DRAM bytes read (`dram__bytes_read.sum`) | 406.598144 | Mbyte |
| DRAM bytes written (`dram__bytes_write.sum`) | 193.137152 | Mbyte |
| L2 hit rate (%) (`lts__t_sector_hit_rate.pct`) | 92.204452 | % |
| issue slots busy (%) (`smsp__issue_active.avg.pct_of_peak_sustained_active`) | 38.664307 | % |
| achieved occupancy (%) (`sm__warps_active.avg.pct_of_peak_sustained_active`) | 12.349693 | % |
| theoretical occupancy (%) (`sm__maximum_warps_per_active_cycle_pct`) | 12.500000 | % |
| registers per thread (`launch__registers_per_thread`) | 255 | register/thread |
| static shared memory per block (`launch__shared_mem_per_block_static`) | 0 | byte/block |
| dynamic shared memory per block (`launch__shared_mem_per_block_dynamic`) | 65.536000 | Kbyte/block |
| blocks per SM allowed by registers (`launch__occupancy_limit_registers`) | 2.000000 | block |
| blocks per SM allowed by shared memory (`launch__occupancy_limit_shared_mem`) | 2.000000 | block |
| waves per SM (`launch__waves_per_multiprocessor`) | 15.52 |  |
| shared-memory wavefronts (`l1tex__data_pipe_lsu_wavefronts_mem_shared.sum`) | 353306955 |  |
| shared-memory bank conflicts (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`) | 246087 |  |

| pipe utilization | % of peak (active cycles) |
|---|---|
| `sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active` | 11.387773 |
| `sm__inst_executed_pipe_fma.max.pct_of_peak_sustained_active` | 11.743641 |
| `sm__inst_executed_pipe_fma.min.pct_of_peak_sustained_active` | 11.376653 |
| `sm__inst_executed_pipe_fma.sum.pct_of_peak_sustained_active` | 11.387773 |
| `sm__inst_executed_pipe_fma_type_fp16.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_fmaheavy.avg.pct_of_peak_sustained_active` | 7.678451 |
| `sm__inst_executed_pipe_fmaheavy.max.pct_of_peak_sustained_active` | 7.923585 |
| `sm__inst_executed_pipe_fmaheavy.min.pct_of_peak_sustained_active` | 7.664567 |
| `sm__inst_executed_pipe_fmaheavy.sum.pct_of_peak_sustained_active` | 7.678451 |
| `sm__inst_executed_pipe_fmalite.avg.pct_of_peak_sustained_active` | 15.097095 |
| `sm__inst_executed_pipe_fmalite.max.pct_of_peak_sustained_active` | 15.571208 |
| `sm__inst_executed_pipe_fmalite.min.pct_of_peak_sustained_active` | 15.075579 |
| `sm__inst_executed_pipe_fmalite.sum.pct_of_peak_sustained_active` | 15.097095 |
| `sm__inst_executed_pipe_tensor_op_dmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active` | 20.808262 |
| `sm__inst_executed_pipe_tensor_op_hmma.max.pct_of_peak_sustained_active` | 21.458520 |
| `sm__inst_executed_pipe_tensor_op_hmma.min.pct_of_peak_sustained_active` | 20.787941 |
| `sm__inst_executed_pipe_tensor_op_hmma.sum.pct_of_peak_sustained_active` | 20.808262 |
| `sm__inst_executed_pipe_tensor_op_imma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` | 11.480486 |
| `sm__pipe_fma_cycles_active.max.pct_of_peak_sustained_active` | 11.839251 |
| `sm__pipe_fma_cycles_active.min.pct_of_peak_sustained_active` | 11.469275 |
| `sm__pipe_fma_cycles_active.sum.pct_of_peak_sustained_active` | 11.480486 |
| `sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active` | 7.863876 |
| `sm__pipe_fmaheavy_cycles_active.max.pct_of_peak_sustained_active` | 8.112227 |
| `sm__pipe_fmaheavy_cycles_active.min.pct_of_peak_sustained_active` | 7.849811 |
| `sm__pipe_fmaheavy_cycles_active.sum.pct_of_peak_sustained_active` | 7.863876 |
| `sm__pipe_fmalite_cycles_active.avg.pct_of_peak_sustained_active` | 15.097095 |
| `sm__pipe_fmalite_cycles_active.max.pct_of_peak_sustained_active` | 15.571208 |
| `sm__pipe_fmalite_cycles_active.min.pct_of_peak_sustained_active` | 15.075579 |
| `sm__pipe_fmalite_cycles_active.sum.pct_of_peak_sustained_active` | 15.097095 |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | 62.424785 |
| `sm__pipe_tensor_cycles_active.max.pct_of_peak_sustained_active` | 64.375560 |
| `sm__pipe_tensor_cycles_active.min.pct_of_peak_sustained_active` | 62.363824 |
| `sm__pipe_tensor_cycles_active.sum.pct_of_peak_sustained_active` | 62.424785 |
| `sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | 62.424785 |
| `sm__pipe_tensor_op_hmma_cycles_active.max.pct_of_peak_sustained_active` | 64.375560 |
| `sm__pipe_tensor_op_hmma_cycles_active.min.pct_of_peak_sustained_active` | 62.363824 |
| `sm__pipe_tensor_op_hmma_cycles_active.sum.pct_of_peak_sustained_active` | 62.424785 |
| `sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |

| stall reason | cycles per issued instruction |
|---|---|
| wait | 1.86 |
| selected | 1.00 |
| math_pipe_throttle | 0.83 |
| short_scoreboard | 0.39 |
| not_selected | 0.38 |
| dispatch_stall | 0.21 |
| long_scoreboard | 0.19 |
| barrier | 0.13 |
| mio_throttle | 0.08 |
| no_instruction | 0.03 |
