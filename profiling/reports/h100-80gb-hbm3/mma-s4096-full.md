## void unnamed>::mma_attention_kernel<__nv_bfloat16, 128>(AttnParams)

- grid (64, 128, 1), block (128, 1, 1)

| metric | value | unit |
|---|---|---|
| duration (`gpu__time_duration.sum`) | 10.897856 | ms |
| SM throughput (% of peak) (`sm__throughput.avg.pct_of_peak_sustained_elapsed`) | 26.414316 | % |
| memory throughput (%) (`gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`) | 31.651018 | % |
| DRAM bytes read (`dram__bytes_read.sum`) | 403.520256 | Mbyte |
| DRAM bytes written (`dram__bytes_write.sum`) | 128.384256 | Mbyte |
| L2 hit rate (%) (`lts__t_sector_hit_rate.pct`) | 90.027323 | % |
| issue slots busy (%) (`smsp__issue_active.avg.pct_of_peak_sustained_active`) | 26.561781 | % |
| achieved occupancy (%) (`sm__warps_active.avg.pct_of_peak_sustained_active`) | 18.516038 | % |
| theoretical occupancy (%) (`sm__maximum_warps_per_active_cycle_pct`) | 18.750000 | % |
| registers per thread (`launch__registers_per_thread`) | 168 | register/thread |
| static shared memory per block (`launch__shared_mem_per_block_static`) | 34.816000 | Kbyte/block |
| dynamic shared memory per block (`launch__shared_mem_per_block_dynamic`) | 0 | byte/block |
| blocks per SM allowed by registers (`launch__occupancy_limit_registers`) | 3.000000 | block |
| blocks per SM allowed by shared memory (`launch__occupancy_limit_shared_mem`) | 3.000000 | block |
| waves per SM (`launch__waves_per_multiprocessor`) | 20.69 |  |
| shared-memory wavefronts (`l1tex__data_pipe_lsu_wavefronts_mem_shared.sum`) | 686059543 |  |
| shared-memory bank conflicts (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`) | 1167233 |  |

| pipe utilization | % of peak (active cycles) |
|---|---|
| `sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active` | 10.000119 |
| `sm__inst_executed_pipe_fma.max.pct_of_peak_sustained_active` | 10.473758 |
| `sm__inst_executed_pipe_fma.min.pct_of_peak_sustained_active` | 9.023545 |
| `sm__inst_executed_pipe_fma.sum.pct_of_peak_sustained_active` | 10.000119 |
| `sm__inst_executed_pipe_fma_type_fp16.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_fmaheavy.avg.pct_of_peak_sustained_active` | 13.124736 |
| `sm__inst_executed_pipe_fmaheavy.max.pct_of_peak_sustained_active` | 13.746242 |
| `sm__inst_executed_pipe_fmaheavy.min.pct_of_peak_sustained_active` | 12.068190 |
| `sm__inst_executed_pipe_fmaheavy.sum.pct_of_peak_sustained_active` | 13.124736 |
| `sm__inst_executed_pipe_fmalite.avg.pct_of_peak_sustained_active` | 6.875503 |
| `sm__inst_executed_pipe_fmalite.max.pct_of_peak_sustained_active` | 7.201274 |
| `sm__inst_executed_pipe_fmalite.min.pct_of_peak_sustained_active` | 6.291080 |
| `sm__inst_executed_pipe_fmalite.sum.pct_of_peak_sustained_active` | 6.875503 |
| `sm__inst_executed_pipe_tensor_op_dmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active` | 5.148054 |
| `sm__inst_executed_pipe_tensor_op_hmma.max.pct_of_peak_sustained_active` | 5.308930 |
| `sm__inst_executed_pipe_tensor_op_hmma.min.pct_of_peak_sustained_active` | 4.645314 |
| `sm__inst_executed_pipe_tensor_op_hmma.sum.pct_of_peak_sustained_active` | 5.148054 |
| `sm__inst_executed_pipe_tensor_op_imma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` | 10.811101 |
| `sm__pipe_fma_cycles_active.max.pct_of_peak_sustained_active` | 11.310083 |
| `sm__pipe_fma_cycles_active.min.pct_of_peak_sustained_active` | 9.929532 |
| `sm__pipe_fma_cycles_active.sum.pct_of_peak_sustained_active` | 10.811101 |
| `sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active` | 14.746700 |
| `sm__pipe_fmaheavy_cycles_active.max.pct_of_peak_sustained_active` | 15.418892 |
| `sm__pipe_fmaheavy_cycles_active.min.pct_of_peak_sustained_active` | 13.557894 |
| `sm__pipe_fmaheavy_cycles_active.sum.pct_of_peak_sustained_active` | 14.746700 |
| `sm__pipe_fmalite_cycles_active.avg.pct_of_peak_sustained_active` | 6.875503 |
| `sm__pipe_fmalite_cycles_active.max.pct_of_peak_sustained_active` | 7.201274 |
| `sm__pipe_fmalite_cycles_active.min.pct_of_peak_sustained_active` | 6.291080 |
| `sm__pipe_fmalite_cycles_active.sum.pct_of_peak_sustained_active` | 6.875503 |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | 15.444161 |
| `sm__pipe_tensor_cycles_active.max.pct_of_peak_sustained_active` | 15.926791 |
| `sm__pipe_tensor_cycles_active.min.pct_of_peak_sustained_active` | 14.184798 |
| `sm__pipe_tensor_cycles_active.sum.pct_of_peak_sustained_active` | 15.444161 |
| `sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | 15.444161 |
| `sm__pipe_tensor_op_hmma_cycles_active.max.pct_of_peak_sustained_active` | 15.926791 |
| `sm__pipe_tensor_op_hmma_cycles_active.min.pct_of_peak_sustained_active` | 14.184798 |
| `sm__pipe_tensor_op_hmma_cycles_active.sum.pct_of_peak_sustained_active` | 15.444161 |
| `sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |

| stall reason | cycles per issued instruction |
|---|---|
| long_scoreboard | 6.56 |
| wait | 1.45 |
| short_scoreboard | 1.12 |
| selected | 1.00 |
| not_selected | 0.29 |
| barrier | 0.20 |
| dispatch_stall | 0.16 |
| math_pipe_throttle | 0.15 |
| branch_resolving | 0.14 |
| no_instruction | 0.03 |
