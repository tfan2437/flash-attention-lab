## void unnamed>::gemm_kernel<1>(unnamed>::Operand, unnamed>::Operand, unnamed>::Operand, int, int, int)

- grid (256, 256, 128), block (16, 16, 1)

| metric | value | unit |
|---|---|---|
| duration (`gpu__time_duration.sum`) | 137.385664 | ms |
| SM throughput (% of peak) (`sm__throughput.avg.pct_of_peak_sustained_elapsed`) | 80.785501 | % |
| memory throughput (%) (`gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`) | 61.847775 | % |
| DRAM bytes read (`dram__bytes_read.sum`) | 0.539985 | Gbyte |
| DRAM bytes written (`dram__bytes_write.sum`) | 8.573748 | Gbyte |
| L2 hit rate (%) (`lts__t_sector_hit_rate.pct`) | 99.023903 | % |
| issue slots busy (%) (`smsp__issue_active.avg.pct_of_peak_sustained_active`) | 80.788162 | % |
| achieved occupancy (%) (`sm__warps_active.avg.pct_of_peak_sustained_active`) | 99.152662 | % |
| theoretical occupancy (%) (`sm__maximum_warps_per_active_cycle_pct`) | 100.000000 | % |
| registers per thread (`launch__registers_per_thread`) | 30 | register/thread |
| static shared memory per block (`launch__shared_mem_per_block_static`) | 2.176000 | Kbyte/block |
| dynamic shared memory per block (`launch__shared_mem_per_block_dynamic`) | 0 | byte/block |
| blocks per SM allowed by registers (`launch__occupancy_limit_registers`) | 8.000000 | block |
| blocks per SM allowed by shared memory (`launch__occupancy_limit_shared_mem`) | 20.000000 | block |
| waves per SM (`launch__waves_per_multiprocessor`) | 7943.76 |  |
| shared-memory wavefronts (`l1tex__data_pipe_lsu_wavefronts_mem_shared.sum`) | 19770479078 |  |
| shared-memory bank conflicts (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`) | 1183364626 |  |

| pipe utilization | % of peak (active cycles) |
|---|---|
| `sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active` | 27.894952 |
| `sm__inst_executed_pipe_fma.max.pct_of_peak_sustained_active` | 27.912922 |
| `sm__inst_executed_pipe_fma.min.pct_of_peak_sustained_active` | 27.873856 |
| `sm__inst_executed_pipe_fma.sum.pct_of_peak_sustained_active` | 27.894952 |
| `sm__inst_executed_pipe_fma_type_fp16.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_fmaheavy.avg.pct_of_peak_sustained_active` | 44.229403 |
| `sm__inst_executed_pipe_fmaheavy.max.pct_of_peak_sustained_active` | 44.258880 |
| `sm__inst_executed_pipe_fmaheavy.min.pct_of_peak_sustained_active` | 44.192500 |
| `sm__inst_executed_pipe_fmaheavy.sum.pct_of_peak_sustained_active` | 44.229403 |
| `sm__inst_executed_pipe_fmalite.avg.pct_of_peak_sustained_active` | 11.560501 |
| `sm__inst_executed_pipe_fmalite.max.pct_of_peak_sustained_active` | 11.565690 |
| `sm__inst_executed_pipe_fmalite.min.pct_of_peak_sustained_active` | 11.551700 |
| `sm__inst_executed_pipe_fmalite.sum.pct_of_peak_sustained_active` | 11.560501 |
| `sm__inst_executed_pipe_tensor_op_dmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active` | 1.682108 |
| `sm__inst_executed_pipe_tensor_op_hmma.max.pct_of_peak_sustained_active` | 1.683112 |
| `sm__inst_executed_pipe_tensor_op_hmma.min.pct_of_peak_sustained_active` | 1.680888 |
| `sm__inst_executed_pipe_tensor_op_hmma.sum.pct_of_peak_sustained_active` | 1.682108 |
| `sm__inst_executed_pipe_tensor_op_imma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` | 32.801099 |
| `sm__pipe_fma_cycles_active.max.pct_of_peak_sustained_active` | 32.820735 |
| `sm__pipe_fma_cycles_active.min.pct_of_peak_sustained_active` | 32.776622 |
| `sm__pipe_fma_cycles_active.sum.pct_of_peak_sustained_active` | 32.801099 |
| `sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active` | 54.041698 |
| `sm__pipe_fmaheavy_cycles_active.max.pct_of_peak_sustained_active` | 54.076474 |
| `sm__pipe_fmaheavy_cycles_active.min.pct_of_peak_sustained_active` | 54.001543 |
| `sm__pipe_fmaheavy_cycles_active.sum.pct_of_peak_sustained_active` | 54.041698 |
| `sm__pipe_fmalite_cycles_active.avg.pct_of_peak_sustained_active` | 11.560501 |
| `sm__pipe_fmalite_cycles_active.max.pct_of_peak_sustained_active` | 11.565690 |
| `sm__pipe_fmalite_cycles_active.min.pct_of_peak_sustained_active` | 11.551700 |
| `sm__pipe_fmalite_cycles_active.sum.pct_of_peak_sustained_active` | 11.560501 |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | 1.682108 |
| `sm__pipe_tensor_cycles_active.max.pct_of_peak_sustained_active` | 1.683165 |
| `sm__pipe_tensor_cycles_active.min.pct_of_peak_sustained_active` | 1.680888 |
| `sm__pipe_tensor_cycles_active.sum.pct_of_peak_sustained_active` | 1.682108 |
| `sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | 1.682108 |
| `sm__pipe_tensor_op_hmma_cycles_active.max.pct_of_peak_sustained_active` | 1.683165 |
| `sm__pipe_tensor_op_hmma_cycles_active.min.pct_of_peak_sustained_active` | 1.680888 |
| `sm__pipe_tensor_op_hmma_cycles_active.sum.pct_of_peak_sustained_active` | 1.682108 |
| `sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |

| stall reason | cycles per issued instruction |
|---|---|
| not_selected | 6.85 |
| wait | 2.61 |
| math_pipe_throttle | 2.13 |
| long_scoreboard | 2.03 |
| dispatch_stall | 1.82 |
| barrier | 1.40 |
| selected | 1.00 |
| mio_throttle | 0.86 |
| short_scoreboard | 0.79 |
| branch_resolving | 0.10 |

## unnamed>::scale_mask_kernel(float *, long, int, int, float, bool, int)

- grid (8388608, 1, 1), block (256, 1, 1)

| metric | value | unit |
|---|---|---|
| duration (`gpu__time_duration.sum`) | 8.853600 | ms |
| SM throughput (% of peak) (`sm__throughput.avg.pct_of_peak_sustained_elapsed`) | 58.743482 | % |
| memory throughput (%) (`gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`) | 57.811093 | % |
| DRAM bytes read (`dram__bytes_read.sum`) | 8.590401 | Gbyte |
| DRAM bytes written (`dram__bytes_write.sum`) | 8.567907 | Gbyte |
| L2 hit rate (%) (`lts__t_sector_hit_rate.pct`) | 50.025939 | % |
| issue slots busy (%) (`smsp__issue_active.avg.pct_of_peak_sustained_active`) | 58.771647 | % |
| achieved occupancy (%) (`sm__warps_active.avg.pct_of_peak_sustained_active`) | 78.514157 | % |
| theoretical occupancy (%) (`sm__maximum_warps_per_active_cycle_pct`) | 100.000000 | % |
| registers per thread (`launch__registers_per_thread`) | 20 | register/thread |
| static shared memory per block (`launch__shared_mem_per_block_static`) | 0.000000 | Kbyte/block |
| dynamic shared memory per block (`launch__shared_mem_per_block_dynamic`) | 0 | byte/block |
| blocks per SM allowed by registers (`launch__occupancy_limit_registers`) | 10.000000 | block |
| blocks per SM allowed by shared memory (`launch__occupancy_limit_shared_mem`) | 32.000000 | block |
| waves per SM (`launch__waves_per_multiprocessor`) | 7943.76 |  |
| shared-memory wavefronts (`l1tex__data_pipe_lsu_wavefronts_mem_shared.sum`) | 67108864 |  |
| shared-memory bank conflicts (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`) | 0 |  |

| pipe utilization | % of peak (active cycles) |
|---|---|
| `sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active` | 17.407360 |
| `sm__inst_executed_pipe_fma.max.pct_of_peak_sustained_active` | 17.857387 |
| `sm__inst_executed_pipe_fma.min.pct_of_peak_sustained_active` | 16.976474 |
| `sm__inst_executed_pipe_fma.sum.pct_of_peak_sustained_active` | 17.407360 |
| `sm__inst_executed_pipe_fma_type_fp16.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_fmaheavy.avg.pct_of_peak_sustained_active` | 33.383980 |
| `sm__inst_executed_pipe_fmaheavy.max.pct_of_peak_sustained_active` | 34.267358 |
| `sm__inst_executed_pipe_fmaheavy.min.pct_of_peak_sustained_active` | 32.544928 |
| `sm__inst_executed_pipe_fmaheavy.sum.pct_of_peak_sustained_active` | 33.383980 |
| `sm__inst_executed_pipe_fmalite.avg.pct_of_peak_sustained_active` | 1.430740 |
| `sm__inst_executed_pipe_fmalite.max.pct_of_peak_sustained_active` | 1.467686 |
| `sm__inst_executed_pipe_fmalite.min.pct_of_peak_sustained_active` | 1.395081 |
| `sm__inst_executed_pipe_fmalite.sum.pct_of_peak_sustained_active` | 1.430740 |
| `sm__inst_executed_pipe_tensor_op_dmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active` | 1.450613 |
| `sm__inst_executed_pipe_tensor_op_hmma.max.pct_of_peak_sustained_active` | 1.487773 |
| `sm__inst_executed_pipe_tensor_op_hmma.min.pct_of_peak_sustained_active` | 1.414113 |
| `sm__inst_executed_pipe_tensor_op_hmma.sum.pct_of_peak_sustained_active` | 1.450613 |
| `sm__inst_executed_pipe_tensor_op_imma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` | 21.033894 |
| `sm__pipe_fma_cycles_active.max.pct_of_peak_sustained_active` | 21.589352 |
| `sm__pipe_fma_cycles_active.min.pct_of_peak_sustained_active` | 20.505205 |
| `sm__pipe_fma_cycles_active.sum.pct_of_peak_sustained_active` | 21.033894 |
| `sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active` | 40.637047 |
| `sm__pipe_fmaheavy_cycles_active.max.pct_of_peak_sustained_active` | 41.711018 |
| `sm__pipe_fmaheavy_cycles_active.min.pct_of_peak_sustained_active` | 39.615329 |
| `sm__pipe_fmaheavy_cycles_active.sum.pct_of_peak_sustained_active` | 40.637047 |
| `sm__pipe_fmalite_cycles_active.avg.pct_of_peak_sustained_active` | 1.430740 |
| `sm__pipe_fmalite_cycles_active.max.pct_of_peak_sustained_active` | 1.467686 |
| `sm__pipe_fmalite_cycles_active.min.pct_of_peak_sustained_active` | 1.395081 |
| `sm__pipe_fmalite_cycles_active.sum.pct_of_peak_sustained_active` | 1.430740 |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | 1.450613 |
| `sm__pipe_tensor_cycles_active.max.pct_of_peak_sustained_active` | 1.487522 |
| `sm__pipe_tensor_cycles_active.min.pct_of_peak_sustained_active` | 1.414272 |
| `sm__pipe_tensor_cycles_active.sum.pct_of_peak_sustained_active` | 1.450613 |
| `sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | 1.450613 |
| `sm__pipe_tensor_op_hmma_cycles_active.max.pct_of_peak_sustained_active` | 1.487522 |
| `sm__pipe_tensor_op_hmma_cycles_active.min.pct_of_peak_sustained_active` | 1.414272 |
| `sm__pipe_tensor_op_hmma_cycles_active.sum.pct_of_peak_sustained_active` | 1.450613 |
| `sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |

| stall reason | cycles per issued instruction |
|---|---|
| long_scoreboard | 11.98 |
| wait | 3.67 |
| short_scoreboard | 1.50 |
| not_selected | 1.19 |
| selected | 1.00 |
| branch_resolving | 0.52 |
| math_pipe_throttle | 0.44 |
| drain | 0.27 |
| dispatch_stall | 0.24 |
| no_instruction | 0.16 |

## unnamed>::softmax_kernel(float *, float *, int)

- grid (524288, 1, 1), block (256, 1, 1)

| metric | value | unit |
|---|---|---|
| duration (`gpu__time_duration.sum`) | 6.829376 | ms |
| SM throughput (% of peak) (`sm__throughput.avg.pct_of_peak_sustained_elapsed`) | 36.271717 | % |
| memory throughput (%) (`gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`) | 79.813552 | % |
| DRAM bytes read (`dram__bytes_read.sum`) | 9.702316 | Gbyte |
| DRAM bytes written (`dram__bytes_write.sum`) | 8.570286 | Gbyte |
| L2 hit rate (%) (`lts__t_sector_hit_rate.pct`) | 53.490806 | % |
| issue slots busy (%) (`smsp__issue_active.avg.pct_of_peak_sustained_active`) | 36.368551 | % |
| achieved occupancy (%) (`sm__warps_active.avg.pct_of_peak_sustained_active`) | 91.829803 | % |
| theoretical occupancy (%) (`sm__maximum_warps_per_active_cycle_pct`) | 100.000000 | % |
| registers per thread (`launch__registers_per_thread`) | 31 | register/thread |
| static shared memory per block (`launch__shared_mem_per_block_static`) | 0.032000 | Kbyte/block |
| dynamic shared memory per block (`launch__shared_mem_per_block_dynamic`) | 0 | byte/block |
| blocks per SM allowed by registers (`launch__occupancy_limit_registers`) | 8.000000 | block |
| blocks per SM allowed by shared memory (`launch__occupancy_limit_shared_mem`) | 28.000000 | block |
| waves per SM (`launch__waves_per_multiprocessor`) | 496.48 |  |
| shared-memory wavefronts (`l1tex__data_pipe_lsu_wavefronts_mem_shared.sum`) | 91137191 |  |
| shared-memory bank conflicts (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`) | 995796 |  |

| pipe utilization | % of peak (active cycles) |
|---|---|
| `sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active` | 19.430040 |
| `sm__inst_executed_pipe_fma.max.pct_of_peak_sustained_active` | 26.186348 |
| `sm__inst_executed_pipe_fma.min.pct_of_peak_sustained_active` | 17.376034 |
| `sm__inst_executed_pipe_fma.sum.pct_of_peak_sustained_active` | 19.430040 |
| `sm__inst_executed_pipe_fma_type_fp16.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_fmaheavy.avg.pct_of_peak_sustained_active` | 18.168032 |
| `sm__inst_executed_pipe_fmaheavy.max.pct_of_peak_sustained_active` | 24.542651 |
| `sm__inst_executed_pipe_fmaheavy.min.pct_of_peak_sustained_active` | 16.384252 |
| `sm__inst_executed_pipe_fmaheavy.sum.pct_of_peak_sustained_active` | 18.168032 |
| `sm__inst_executed_pipe_fmalite.avg.pct_of_peak_sustained_active` | 20.692047 |
| `sm__inst_executed_pipe_fmalite.max.pct_of_peak_sustained_active` | 27.575667 |
| `sm__inst_executed_pipe_fmalite.min.pct_of_peak_sustained_active` | 18.768952 |
| `sm__inst_executed_pipe_fmalite.sum.pct_of_peak_sustained_active` | 20.692047 |
| `sm__inst_executed_pipe_tensor_op_dmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active` | 0.000000 |
| `sm__inst_executed_pipe_tensor_op_hmma.max.pct_of_peak_sustained_active` | 0.000000 |
| `sm__inst_executed_pipe_tensor_op_hmma.min.pct_of_peak_sustained_active` | 0.000000 |
| `sm__inst_executed_pipe_tensor_op_hmma.sum.pct_of_peak_sustained_active` | 0.000000 |
| `sm__inst_executed_pipe_tensor_op_imma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` | 19.665377 |
| `sm__pipe_fma_cycles_active.max.pct_of_peak_sustained_active` | 26.376744 |
| `sm__pipe_fma_cycles_active.min.pct_of_peak_sustained_active` | 17.797252 |
| `sm__pipe_fma_cycles_active.sum.pct_of_peak_sustained_active` | 19.665377 |
| `sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active` | 18.638707 |
| `sm__pipe_fmaheavy_cycles_active.max.pct_of_peak_sustained_active` | 25.177821 |
| `sm__pipe_fmaheavy_cycles_active.min.pct_of_peak_sustained_active` | 16.825552 |
| `sm__pipe_fmaheavy_cycles_active.sum.pct_of_peak_sustained_active` | 18.638707 |
| `sm__pipe_fmalite_cycles_active.avg.pct_of_peak_sustained_active` | 20.692047 |
| `sm__pipe_fmalite_cycles_active.max.pct_of_peak_sustained_active` | 27.575667 |
| `sm__pipe_fmalite_cycles_active.min.pct_of_peak_sustained_active` | 18.768952 |
| `sm__pipe_fmalite_cycles_active.sum.pct_of_peak_sustained_active` | 20.692047 |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | 0.000000 |
| `sm__pipe_tensor_cycles_active.max.pct_of_peak_sustained_active` | 0.000000 |
| `sm__pipe_tensor_cycles_active.min.pct_of_peak_sustained_active` | 0.000000 |
| `sm__pipe_tensor_cycles_active.sum.pct_of_peak_sustained_active` | 0.000000 |
| `sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | 0.000000 |
| `sm__pipe_tensor_op_hmma_cycles_active.max.pct_of_peak_sustained_active` | 0.000000 |
| `sm__pipe_tensor_op_hmma_cycles_active.min.pct_of_peak_sustained_active` | 0.000000 |
| `sm__pipe_tensor_op_hmma_cycles_active.sum.pct_of_peak_sustained_active` | 0.000000 |
| `sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |

| stall reason | cycles per issued instruction |
|---|---|
| long_scoreboard | 19.52 |
| barrier | 7.59 |
| mio_throttle | 2.85 |
| short_scoreboard | 2.80 |
| lg_throttle | 2.08 |
| not_selected | 2.03 |
| wait | 1.54 |
| selected | 1.00 |
| drain | 0.27 |
| math_pipe_throttle | 0.26 |

## void unnamed>::gemm_kernel<0>(unnamed>::Operand, unnamed>::Operand, unnamed>::Operand, int, int, int)

- grid (8, 256, 128), block (16, 16, 1)

| metric | value | unit |
|---|---|---|
| duration (`gpu__time_duration.sum`) | 133.755872 | ms |
| SM throughput (% of peak) (`sm__throughput.avg.pct_of_peak_sustained_elapsed`) | 80.821685 | % |
| memory throughput (%) (`gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`) | 64.447110 | % |
| DRAM bytes read (`dram__bytes_read.sum`) | 8.861637 | Gbyte |
| DRAM bytes written (`dram__bytes_write.sum`) | 0.273934 | Gbyte |
| L2 hit rate (%) (`lts__t_sector_hit_rate.pct`) | 86.455430 | % |
| issue slots busy (%) (`smsp__issue_active.avg.pct_of_peak_sustained_active`) | 80.826331 | % |
| achieved occupancy (%) (`sm__warps_active.avg.pct_of_peak_sustained_active`) | 99.819284 | % |
| theoretical occupancy (%) (`sm__maximum_warps_per_active_cycle_pct`) | 100.000000 | % |
| registers per thread (`launch__registers_per_thread`) | 30 | register/thread |
| static shared memory per block (`launch__shared_mem_per_block_static`) | 2.176000 | Kbyte/block |
| dynamic shared memory per block (`launch__shared_mem_per_block_dynamic`) | 0 | byte/block |
| blocks per SM allowed by registers (`launch__occupancy_limit_registers`) | 8.000000 | block |
| blocks per SM allowed by shared memory (`launch__occupancy_limit_shared_mem`) | 20.000000 | block |
| waves per SM (`launch__waves_per_multiprocessor`) | 248.24 |  |
| shared-memory wavefronts (`l1tex__data_pipe_lsu_wavefronts_mem_shared.sum`) | 20203980870 |  |
| shared-memory bank conflicts (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`) | 1277122385 |  |

| pipe utilization | % of peak (active cycles) |
|---|---|
| `sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active` | 28.858390 |
| `sm__inst_executed_pipe_fma.max.pct_of_peak_sustained_active` | 28.859270 |
| `sm__inst_executed_pipe_fma.min.pct_of_peak_sustained_active` | 28.844739 |
| `sm__inst_executed_pipe_fma.sum.pct_of_peak_sustained_active` | 28.858390 |
| `sm__inst_executed_pipe_fma_type_fp16.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_fmaheavy.avg.pct_of_peak_sustained_active` | 45.747970 |
| `sm__inst_executed_pipe_fmaheavy.max.pct_of_peak_sustained_active` | 45.753389 |
| `sm__inst_executed_pipe_fmaheavy.min.pct_of_peak_sustained_active` | 45.721213 |
| `sm__inst_executed_pipe_fmaheavy.sum.pct_of_peak_sustained_active` | 45.747970 |
| `sm__inst_executed_pipe_fmalite.avg.pct_of_peak_sustained_active` | 11.968809 |
| `sm__inst_executed_pipe_fmalite.max.pct_of_peak_sustained_active` | 11.975614 |
| `sm__inst_executed_pipe_fmalite.min.pct_of_peak_sustained_active` | 11.963680 |
| `sm__inst_executed_pipe_fmalite.sum.pct_of_peak_sustained_active` | 11.968809 |
| `sm__inst_executed_pipe_tensor_op_dmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active` | 1.538834 |
| `sm__inst_executed_pipe_tensor_op_hmma.max.pct_of_peak_sustained_active` | 1.538881 |
| `sm__inst_executed_pipe_tensor_op_hmma.min.pct_of_peak_sustained_active` | 1.538106 |
| `sm__inst_executed_pipe_tensor_op_hmma.sum.pct_of_peak_sustained_active` | 1.538834 |
| `sm__inst_executed_pipe_tensor_op_imma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` | 33.861850 |
| `sm__pipe_fma_cycles_active.max.pct_of_peak_sustained_active` | 33.862884 |
| `sm__pipe_fma_cycles_active.min.pct_of_peak_sustained_active` | 33.848352 |
| `sm__pipe_fma_cycles_active.sum.pct_of_peak_sustained_active` | 33.861850 |
| `sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active` | 55.754892 |
| `sm__pipe_fmaheavy_cycles_active.max.pct_of_peak_sustained_active` | 55.760615 |
| `sm__pipe_fmaheavy_cycles_active.min.pct_of_peak_sustained_active` | 55.728440 |
| `sm__pipe_fmaheavy_cycles_active.sum.pct_of_peak_sustained_active` | 55.754892 |
| `sm__pipe_fmalite_cycles_active.avg.pct_of_peak_sustained_active` | 11.968809 |
| `sm__pipe_fmalite_cycles_active.max.pct_of_peak_sustained_active` | 11.975614 |
| `sm__pipe_fmalite_cycles_active.min.pct_of_peak_sustained_active` | 11.963680 |
| `sm__pipe_fmalite_cycles_active.sum.pct_of_peak_sustained_active` | 11.968809 |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | 1.538834 |
| `sm__pipe_tensor_cycles_active.max.pct_of_peak_sustained_active` | 1.538881 |
| `sm__pipe_tensor_cycles_active.min.pct_of_peak_sustained_active` | 1.538106 |
| `sm__pipe_tensor_cycles_active.sum.pct_of_peak_sustained_active` | 1.538834 |
| `sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | 1.538834 |
| `sm__pipe_tensor_op_hmma_cycles_active.max.pct_of_peak_sustained_active` | 1.538881 |
| `sm__pipe_tensor_op_hmma_cycles_active.min.pct_of_peak_sustained_active` | 1.538106 |
| `sm__pipe_tensor_op_hmma_cycles_active.sum.pct_of_peak_sustained_active` | 1.538834 |
| `sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |

| stall reason | cycles per issued instruction |
|---|---|
| not_selected | 5.67 |
| long_scoreboard | 3.66 |
| wait | 2.43 |
| barrier | 2.18 |
| math_pipe_throttle | 1.70 |
| dispatch_stall | 1.39 |
| selected | 1.00 |
| mio_throttle | 0.96 |
| short_scoreboard | 0.65 |
| branch_resolving | 0.10 |
