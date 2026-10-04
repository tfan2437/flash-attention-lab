## _attention_fwd_kernel

- grid (32, 128, 1), block (256, 1, 1)

| metric | value | unit |
|---|---|---|
| duration (`gpu__time_duration.sum`) | 1.311744 | ms |
| SM throughput (% of peak) (`sm__throughput.avg.pct_of_peak_sustained_elapsed`) | 43.878221 | % |
| memory throughput (%) (`gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`) | 38.464337 | % |
| DRAM bytes read (`dram__bytes_read.sum`) | 402.748928 | Mbyte |
| DRAM bytes written (`dram__bytes_write.sum`) | 130.558720 | Mbyte |
| L2 hit rate (%) (`lts__t_sector_hit_rate.pct`) | 85.505901 | % |
| issue slots busy (%) (`smsp__issue_active.avg.pct_of_peak_sustained_active`) | 35.134172 | % |
| achieved occupancy (%) (`sm__warps_active.avg.pct_of_peak_sustained_active`) | 12.492134 | % |
| theoretical occupancy (%) (`sm__maximum_warps_per_active_cycle_pct`) | 12.500000 | % |
| registers per thread (`launch__registers_per_thread`) | 244 | register/thread |
| static shared memory per block (`launch__shared_mem_per_block_static`) | 0 | byte/block |
| dynamic shared memory per block (`launch__shared_mem_per_block_dynamic`) | 229.376000 | Kbyte/block |
| blocks per SM allowed by registers (`launch__occupancy_limit_registers`) | 1.000000 | block |
| blocks per SM allowed by shared memory (`launch__occupancy_limit_shared_mem`) | 1.000000 | block |
| waves per SM (`launch__waves_per_multiprocessor`) | 31.03 |  |
| shared-memory wavefronts (`l1tex__data_pipe_lsu_wavefronts_mem_shared.sum`) | 105410794 |  |
| shared-memory bank conflicts (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`) | 1387413 |  |

| pipe utilization | % of peak (active cycles) |
|---|---|
| `sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active` | 14.749661 |
| `sm__inst_executed_pipe_fma.max.pct_of_peak_sustained_active` | 15.373959 |
| `sm__inst_executed_pipe_fma.min.pct_of_peak_sustained_active` | 14.402734 |
| `sm__inst_executed_pipe_fma.sum.pct_of_peak_sustained_active` | 14.749661 |
| `sm__inst_executed_pipe_fma_type_fp16.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_fmaheavy.avg.pct_of_peak_sustained_active` | 11.185503 |
| `sm__inst_executed_pipe_fmaheavy.max.pct_of_peak_sustained_active` | 11.645699 |
| `sm__inst_executed_pipe_fmaheavy.min.pct_of_peak_sustained_active` | 10.856224 |
| `sm__inst_executed_pipe_fmaheavy.sum.pct_of_peak_sustained_active` | 11.185503 |
| `sm__inst_executed_pipe_fmalite.avg.pct_of_peak_sustained_active` | 18.313819 |
| `sm__inst_executed_pipe_fmalite.max.pct_of_peak_sustained_active` | 19.273108 |
| `sm__inst_executed_pipe_fmalite.min.pct_of_peak_sustained_active` | 17.625954 |
| `sm__inst_executed_pipe_fmalite.sum.pct_of_peak_sustained_active` | 18.313819 |
| `sm__inst_executed_pipe_tensor_op_dmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.avg.pct_of_peak_sustained_active` | 2.864288 |
| `sm__inst_executed_pipe_tensor_op_gmma.max.pct_of_peak_sustained_active` | 3.004146 |
| `sm__inst_executed_pipe_tensor_op_gmma.min.pct_of_peak_sustained_active` | 2.746808 |
| `sm__inst_executed_pipe_tensor_op_gmma.sum.pct_of_peak_sustained_active` | 2.864288 |
| `sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active` | 1.432144 |
| `sm__inst_executed_pipe_tensor_op_hmma.max.pct_of_peak_sustained_active` | 1.502073 |
| `sm__inst_executed_pipe_tensor_op_hmma.min.pct_of_peak_sustained_active` | 1.373404 |
| `sm__inst_executed_pipe_tensor_op_hmma.sum.pct_of_peak_sustained_active` | 1.432144 |
| `sm__inst_executed_pipe_tensor_op_imma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` | 15.739685 |
| `sm__pipe_fma_cycles_active.max.pct_of_peak_sustained_active` | 16.375866 |
| `sm__pipe_fma_cycles_active.min.pct_of_peak_sustained_active` | 15.334886 |
| `sm__pipe_fma_cycles_active.sum.pct_of_peak_sustained_active` | 15.739685 |
| `sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active` | 13.165551 |
| `sm__pipe_fmaheavy_cycles_active.max.pct_of_peak_sustained_active` | 13.652135 |
| `sm__pipe_fmaheavy_cycles_active.min.pct_of_peak_sustained_active` | 12.788360 |
| `sm__pipe_fmaheavy_cycles_active.sum.pct_of_peak_sustained_active` | 13.165551 |
| `sm__pipe_fmalite_cycles_active.avg.pct_of_peak_sustained_active` | 18.313819 |
| `sm__pipe_fmalite_cycles_active.max.pct_of_peak_sustained_active` | 19.273108 |
| `sm__pipe_fmalite_cycles_active.min.pct_of_peak_sustained_active` | 17.625954 |
| `sm__pipe_fmalite_cycles_active.sum.pct_of_peak_sustained_active` | 18.313819 |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | 45.828613 |
| `sm__pipe_tensor_cycles_active.max.pct_of_peak_sustained_active` | 48.334866 |
| `sm__pipe_tensor_cycles_active.min.pct_of_peak_sustained_active` | 43.590888 |
| `sm__pipe_tensor_cycles_active.sum.pct_of_peak_sustained_active` | 45.828613 |
| `sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | 45.828613 |
| `sm__pipe_tensor_op_hmma_cycles_active.max.pct_of_peak_sustained_active` | 48.334866 |
| `sm__pipe_tensor_op_hmma_cycles_active.min.pct_of_peak_sustained_active` | 43.590888 |
| `sm__pipe_tensor_op_hmma_cycles_active.sum.pct_of_peak_sustained_active` | 45.828613 |
| `sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |

| stall reason | cycles per issued instruction |
|---|---|
| wait | 1.35 |
| barrier | 1.26 |
| selected | 1.00 |
| mio_throttle | 0.59 |
| not_selected | 0.39 |
| long_scoreboard | 0.33 |
| dispatch_stall | 0.30 |
| short_scoreboard | 0.19 |
| math_pipe_throttle | 0.16 |
| lg_throttle | 0.05 |
