## void unnamed>::mma_attention_kernel<__nv_bfloat16, 128>(AttnParams)

- grid (64, 128, 1), block (128, 1, 1)

| metric | value | unit |
|---|---|---|
| duration (`gpu__time_duration.sum`) | 10.613088 | ms |
| SM throughput (% of peak) (`sm__throughput.avg.pct_of_peak_sustained_elapsed`) | 23.889570 | % |
| memory throughput (%) (`gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`) | 32.671854 | % |
| DRAM bytes read (`dram__bytes_read.sum`) | 403.444480 | Mbyte |
| DRAM bytes written (`dram__bytes_write.sum`) | 129.131520 | Mbyte |
| L2 hit rate (%) (`lts__t_sector_hit_rate.pct`) | 86.355813 | % |
| issue slots busy (%) (`smsp__issue_active.avg.pct_of_peak_sustained_active`) | 23.984823 | % |
| achieved occupancy (%) (`sm__warps_active.avg.pct_of_peak_sustained_active`) | 18.596751 | % |
| theoretical occupancy (%) (`sm__maximum_warps_per_active_cycle_pct`) | 18.750000 | % |
| registers per thread (`launch__registers_per_thread`) | 168 | register/thread |
| static shared memory per block (`launch__shared_mem_per_block_static`) | 34.816000 | Kbyte/block |
| dynamic shared memory per block (`launch__shared_mem_per_block_dynamic`) | 0 | byte/block |
| blocks per SM allowed by registers (`launch__occupancy_limit_registers`) | 3.000000 | block |
| blocks per SM allowed by shared memory (`launch__occupancy_limit_shared_mem`) | 3.000000 | block |
| waves per SM (`launch__waves_per_multiprocessor`) | 20.69 |  |
| shared-memory wavefronts (`l1tex__data_pipe_lsu_wavefronts_mem_shared.sum`) | 686122944 |  |
| shared-memory bank conflicts (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`) | 1254117 |  |

| pipe utilization | % of peak (active cycles) |
|---|---|
| `sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active` | 9.239012 |
| `sm__inst_executed_pipe_fma.max.pct_of_peak_sustained_active` | 9.676602 |
| `sm__inst_executed_pipe_fma.min.pct_of_peak_sustained_active` | 8.485636 |
| `sm__inst_executed_pipe_fma.sum.pct_of_peak_sustained_active` | 9.239012 |
| `sm__inst_executed_pipe_fma_type_fp16.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_fmaheavy.avg.pct_of_peak_sustained_active` | 12.513727 |
| `sm__inst_executed_pipe_fmaheavy.max.pct_of_peak_sustained_active` | 12.909865 |
| `sm__inst_executed_pipe_fmaheavy.min.pct_of_peak_sustained_active` | 11.138061 |
| `sm__inst_executed_pipe_fmaheavy.sum.pct_of_peak_sustained_active` | 12.513727 |
| `sm__inst_executed_pipe_fmalite.avg.pct_of_peak_sustained_active` | 5.964297 |
| `sm__inst_executed_pipe_fmalite.max.pct_of_peak_sustained_active` | 6.172110 |
| `sm__inst_executed_pipe_fmalite.min.pct_of_peak_sustained_active` | 5.237475 |
| `sm__inst_executed_pipe_fmalite.sum.pct_of_peak_sustained_active` | 5.964297 |
| `sm__inst_executed_pipe_tensor_op_dmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active` | 5.300325 |
| `sm__inst_executed_pipe_tensor_op_hmma.max.pct_of_peak_sustained_active` | 5.551366 |
| `sm__inst_executed_pipe_tensor_op_hmma.min.pct_of_peak_sustained_active` | 4.611904 |
| `sm__inst_executed_pipe_tensor_op_hmma.sum.pct_of_peak_sustained_active` | 5.300325 |
| `sm__inst_executed_pipe_tensor_op_imma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` | 10.073982 |
| `sm__pipe_fma_cycles_active.max.pct_of_peak_sustained_active` | 10.388794 |
| `sm__pipe_fma_cycles_active.min.pct_of_peak_sustained_active` | 8.968232 |
| `sm__pipe_fma_cycles_active.sum.pct_of_peak_sustained_active` | 10.073982 |
| `sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active` | 14.183666 |
| `sm__pipe_fmaheavy_cycles_active.max.pct_of_peak_sustained_active` | 14.624063 |
| `sm__pipe_fmaheavy_cycles_active.min.pct_of_peak_sustained_active` | 12.698736 |
| `sm__pipe_fmaheavy_cycles_active.sum.pct_of_peak_sustained_active` | 14.183666 |
| `sm__pipe_fmalite_cycles_active.avg.pct_of_peak_sustained_active` | 5.964297 |
| `sm__pipe_fmalite_cycles_active.max.pct_of_peak_sustained_active` | 6.172110 |
| `sm__pipe_fmalite_cycles_active.min.pct_of_peak_sustained_active` | 5.237475 |
| `sm__pipe_fmalite_cycles_active.sum.pct_of_peak_sustained_active` | 5.964297 |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | 15.900975 |
| `sm__pipe_tensor_cycles_active.max.pct_of_peak_sustained_active` | 16.654097 |
| `sm__pipe_tensor_cycles_active.min.pct_of_peak_sustained_active` | 14.091928 |
| `sm__pipe_tensor_cycles_active.sum.pct_of_peak_sustained_active` | 15.900975 |
| `sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | 15.900975 |
| `sm__pipe_tensor_op_hmma_cycles_active.max.pct_of_peak_sustained_active` | 16.654097 |
| `sm__pipe_tensor_op_hmma_cycles_active.min.pct_of_peak_sustained_active` | 14.091928 |
| `sm__pipe_tensor_op_hmma_cycles_active.sum.pct_of_peak_sustained_active` | 15.900975 |
| `sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |

| stall reason | cycles per issued instruction |
|---|---|
| long_scoreboard | 7.63 |
| wait | 1.55 |
| short_scoreboard | 1.11 |
| selected | 1.00 |
| not_selected | 0.26 |
| barrier | 0.24 |
| dispatch_stall | 0.18 |
| branch_resolving | 0.18 |
| math_pipe_throttle | 0.14 |
| mio_throttle | 0.04 |
