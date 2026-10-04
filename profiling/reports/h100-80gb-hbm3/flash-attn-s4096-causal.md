## void flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, 0, 0, bfloat16_t, Flash_kernel_traits<128, 128, 64, 4, bfloat16_t>>, 0, 1, 0, 0, 1, 1, 0, 0>(Flash_fwd_params)

- grid (32, 4, 32), block (128, 1, 1)

| metric | value | unit |
|---|---|---|
| duration (`gpu__time_duration.sum`) | 1.505248 | ms |
| SM throughput (% of peak) (`sm__throughput.avg.pct_of_peak_sustained_elapsed`) | 58.133793 | % |
| memory throughput (%) (`gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`) | 54.562994 | % |
| DRAM bytes read (`dram__bytes_read.sum`) | 545.572864 | Mbyte |
| DRAM bytes written (`dram__bytes_write.sum`) | 177.145344 | Mbyte |
| L2 hit rate (%) (`lts__t_sector_hit_rate.pct`) | 82.267757 | % |
| issue slots busy (%) (`smsp__issue_active.avg.pct_of_peak_sustained_active`) | 38.666232 | % |
| achieved occupancy (%) (`sm__warps_active.avg.pct_of_peak_sustained_active`) | 12.332506 | % |
| theoretical occupancy (%) (`sm__maximum_warps_per_active_cycle_pct`) | 12.500000 | % |
| registers per thread (`launch__registers_per_thread`) | 255 | register/thread |
| static shared memory per block (`launch__shared_mem_per_block_static`) | 0 | byte/block |
| dynamic shared memory per block (`launch__shared_mem_per_block_dynamic`) | 65.536000 | Kbyte/block |
| blocks per SM allowed by registers (`launch__occupancy_limit_registers`) | 2.000000 | block |
| blocks per SM allowed by shared memory (`launch__occupancy_limit_shared_mem`) | 2.000000 | block |
| waves per SM (`launch__waves_per_multiprocessor`) | 15.52 |  |
| shared-memory wavefronts (`l1tex__data_pipe_lsu_wavefronts_mem_shared.sum`) | 183442171 |  |
| shared-memory bank conflicts (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`) | 242028 |  |

| pipe utilization | % of peak (active cycles) |
|---|---|
| `sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active` | 11.514301 |
| `sm__inst_executed_pipe_fma.max.pct_of_peak_sustained_active` | 12.187705 |
| `sm__inst_executed_pipe_fma.min.pct_of_peak_sustained_active` | 11.059721 |
| `sm__inst_executed_pipe_fma.sum.pct_of_peak_sustained_active` | 11.514301 |
| `sm__inst_executed_pipe_fma_type_fp16.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_fmaheavy.avg.pct_of_peak_sustained_active` | 8.351642 |
| `sm__inst_executed_pipe_fmaheavy.max.pct_of_peak_sustained_active` | 8.910292 |
| `sm__inst_executed_pipe_fmaheavy.min.pct_of_peak_sustained_active` | 7.964129 |
| `sm__inst_executed_pipe_fmaheavy.sum.pct_of_peak_sustained_active` | 8.351642 |
| `sm__inst_executed_pipe_fmalite.avg.pct_of_peak_sustained_active` | 14.676960 |
| `sm__inst_executed_pipe_fmalite.max.pct_of_peak_sustained_active` | 15.453145 |
| `sm__inst_executed_pipe_fmalite.min.pct_of_peak_sustained_active` | 13.805599 |
| `sm__inst_executed_pipe_fmalite.sum.pct_of_peak_sustained_active` | 14.676960 |
| `sm__inst_executed_pipe_tensor_op_dmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active` | 20.282204 |
| `sm__inst_executed_pipe_tensor_op_hmma.max.pct_of_peak_sustained_active` | 21.787523 |
| `sm__inst_executed_pipe_tensor_op_hmma.min.pct_of_peak_sustained_active` | 19.331475 |
| `sm__inst_executed_pipe_tensor_op_hmma.sum.pct_of_peak_sustained_active` | 20.282204 |
| `sm__inst_executed_pipe_tensor_op_imma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` | 11.615136 |
| `sm__pipe_fma_cycles_active.max.pct_of_peak_sustained_active` | 12.210917 |
| `sm__pipe_fma_cycles_active.min.pct_of_peak_sustained_active` | 11.075776 |
| `sm__pipe_fma_cycles_active.sum.pct_of_peak_sustained_active` | 11.615136 |
| `sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active` | 8.553311 |
| `sm__pipe_fmaheavy_cycles_active.max.pct_of_peak_sustained_active` | 9.108051 |
| `sm__pipe_fmaheavy_cycles_active.min.pct_of_peak_sustained_active` | 8.155389 |
| `sm__pipe_fmaheavy_cycles_active.sum.pct_of_peak_sustained_active` | 8.553311 |
| `sm__pipe_fmalite_cycles_active.avg.pct_of_peak_sustained_active` | 14.676960 |
| `sm__pipe_fmalite_cycles_active.max.pct_of_peak_sustained_active` | 15.453145 |
| `sm__pipe_fmalite_cycles_active.min.pct_of_peak_sustained_active` | 13.805599 |
| `sm__pipe_fmalite_cycles_active.sum.pct_of_peak_sustained_active` | 14.676960 |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | 60.846611 |
| `sm__pipe_tensor_cycles_active.max.pct_of_peak_sustained_active` | 64.293001 |
| `sm__pipe_tensor_cycles_active.min.pct_of_peak_sustained_active` | 57.875585 |
| `sm__pipe_tensor_cycles_active.sum.pct_of_peak_sustained_active` | 60.846611 |
| `sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | 60.846611 |
| `sm__pipe_tensor_op_hmma_cycles_active.max.pct_of_peak_sustained_active` | 64.293001 |
| `sm__pipe_tensor_op_hmma_cycles_active.min.pct_of_peak_sustained_active` | 57.875585 |
| `sm__pipe_tensor_op_hmma_cycles_active.sum.pct_of_peak_sustained_active` | 60.846611 |
| `sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |

| stall reason | cycles per issued instruction |
|---|---|
| wait | 1.82 |
| selected | 1.00 |
| math_pipe_throttle | 0.79 |
| short_scoreboard | 0.40 |
| not_selected | 0.38 |
| long_scoreboard | 0.23 |
| dispatch_stall | 0.22 |
| barrier | 0.12 |
| mio_throttle | 0.08 |
| no_instruction | 0.03 |
