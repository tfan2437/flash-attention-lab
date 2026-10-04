## void flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, 0, 0, bfloat16_t, Flash_kernel_traits<128, 64, 128, 4, bfloat16_t>>, 0, 0, 0, 0, 1, 0, 1, 0>(Flash_fwd_params)

- grid (1, 4, 64), block (128, 1, 1)

| metric | value | unit |
|---|---|---|
| duration (`gpu__time_duration.sum`) | 342.688000 | us |
| SM throughput (% of peak) (`sm__throughput.avg.pct_of_peak_sustained_elapsed`) | 31.219079 | % |
| memory throughput (%) (`gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`) | 93.718575 | % |
| DRAM bytes read (`dram__bytes_read.sum`) | 1.073870 | Gbyte |
| DRAM bytes written (`dram__bytes_write.sum`) | 2.628096 | Mbyte |
| L2 hit rate (%) (`lts__t_sector_hit_rate.pct`) | 1.475067 | % |
| issue slots busy (%) (`smsp__issue_active.avg.pct_of_peak_sustained_active`) | 18.249815 | % |
| achieved occupancy (%) (`sm__warps_active.avg.pct_of_peak_sustained_active`) | 12.098349 | % |
| theoretical occupancy (%) (`sm__maximum_warps_per_active_cycle_pct`) | 12.500000 | % |
| registers per thread (`launch__registers_per_thread`) | 254 | register/thread |
| static shared memory per block (`launch__shared_mem_per_block_static`) | 0 | byte/block |
| dynamic shared memory per block (`launch__shared_mem_per_block_dynamic`) | 81.920000 | Kbyte/block |
| blocks per SM allowed by registers (`launch__occupancy_limit_registers`) | 2.000000 | block |
| blocks per SM allowed by shared memory (`launch__occupancy_limit_shared_mem`) | 2.000000 | block |
| waves per SM (`launch__waves_per_multiprocessor`) | 0.97 |  |
| shared-memory wavefronts (`l1tex__data_pipe_lsu_wavefronts_mem_shared.sum`) | 36548321 |  |
| shared-memory bank conflicts (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`) | 2750 |  |

| pipe utilization | % of peak (active cycles) |
|---|---|
| `sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active` | 4.202991 |
| `sm__inst_executed_pipe_fma.max.pct_of_peak_sustained_active` | 4.334335 |
| `sm__inst_executed_pipe_fma.min.pct_of_peak_sustained_active` | 2.167167 |
| `sm__inst_executed_pipe_fma.sum.pct_of_peak_sustained_active` | 4.202991 |
| `sm__inst_executed_pipe_fma_type_fp16.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_fmaheavy.avg.pct_of_peak_sustained_active` | 2.122091 |
| `sm__inst_executed_pipe_fmaheavy.max.pct_of_peak_sustained_active` | 2.213425 |
| `sm__inst_executed_pipe_fmaheavy.min.pct_of_peak_sustained_active` | 1.082831 |
| `sm__inst_executed_pipe_fmaheavy.sum.pct_of_peak_sustained_active` | 2.122091 |
| `sm__inst_executed_pipe_fmalite.avg.pct_of_peak_sustained_active` | 6.283892 |
| `sm__inst_executed_pipe_fmalite.max.pct_of_peak_sustained_active` | 6.500080 |
| `sm__inst_executed_pipe_fmalite.min.pct_of_peak_sustained_active` | 3.251169 |
| `sm__inst_executed_pipe_fmalite.sum.pct_of_peak_sustained_active` | 6.283892 |
| `sm__inst_executed_pipe_tensor_op_dmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active` | 10.631744 |
| `sm__inst_executed_pipe_tensor_op_hmma.max.pct_of_peak_sustained_active` | 10.963986 |
| `sm__inst_executed_pipe_tensor_op_hmma.min.pct_of_peak_sustained_active` | 5.481993 |
| `sm__inst_executed_pipe_tensor_op_hmma.sum.pct_of_peak_sustained_active` | 10.631744 |
| `sm__inst_executed_pipe_tensor_op_imma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` | 4.216294 |
| `sm__pipe_fma_cycles_active.max.pct_of_peak_sustained_active` | 4.348053 |
| `sm__pipe_fma_cycles_active.min.pct_of_peak_sustained_active` | 2.174027 |
| `sm__pipe_fma_cycles_active.sum.pct_of_peak_sustained_active` | 4.216294 |
| `sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active` | 2.148696 |
| `sm__pipe_fmaheavy_cycles_active.max.pct_of_peak_sustained_active` | 2.240862 |
| `sm__pipe_fmaheavy_cycles_active.min.pct_of_peak_sustained_active` | 1.096549 |
| `sm__pipe_fmaheavy_cycles_active.sum.pct_of_peak_sustained_active` | 2.148696 |
| `sm__pipe_fmalite_cycles_active.avg.pct_of_peak_sustained_active` | 6.283892 |
| `sm__pipe_fmalite_cycles_active.max.pct_of_peak_sustained_active` | 6.500080 |
| `sm__pipe_fmalite_cycles_active.min.pct_of_peak_sustained_active` | 3.251169 |
| `sm__pipe_fmalite_cycles_active.sum.pct_of_peak_sustained_active` | 6.283892 |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | 31.895233 |
| `sm__pipe_tensor_cycles_active.max.pct_of_peak_sustained_active` | 32.891959 |
| `sm__pipe_tensor_cycles_active.min.pct_of_peak_sustained_active` | 16.445980 |
| `sm__pipe_tensor_cycles_active.sum.pct_of_peak_sustained_active` | 31.895233 |
| `sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | 31.895233 |
| `sm__pipe_tensor_op_hmma_cycles_active.max.pct_of_peak_sustained_active` | 32.891959 |
| `sm__pipe_tensor_op_hmma_cycles_active.min.pct_of_peak_sustained_active` | 16.445980 |
| `sm__pipe_tensor_op_hmma_cycles_active.sum.pct_of_peak_sustained_active` | 31.895233 |
| `sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |

| stall reason | cycles per issued instruction |
|---|---|
| long_scoreboard | 5.12 |
| wait | 1.75 |
| barrier | 1.40 |
| selected | 1.00 |
| short_scoreboard | 0.85 |
| mio_throttle | 0.34 |
| math_pipe_throttle | 0.16 |
| not_selected | 0.08 |
| dispatch_stall | 0.06 |
| no_instruction | 0.05 |

## void flash_fwd_splitkv_combine_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, 0, 0, bfloat16_t, Flash_kernel_traits<128, 64, 128, 4, bfloat16_t>>, 4, 2, 1>(Flash_fwd_params)

- grid (64, 1, 1), block (128, 1, 1)

| metric | value | unit |
|---|---|---|
| duration (`gpu__time_duration.sum`) | 5.952000 | us |
| SM throughput (% of peak) (`sm__throughput.avg.pct_of_peak_sustained_elapsed`) | 2.283883 | % |
| memory throughput (%) (`gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`) | 6.515535 | % |
| DRAM bytes read (`dram__bytes_read.sum`) | 0.000550 | Gbyte |
| DRAM bytes written (`dram__bytes_write.sum`) | 0.000000 | Mbyte |
| L2 hit rate (%) (`lts__t_sector_hit_rate.pct`) | 67.440848 | % |
| issue slots busy (%) (`smsp__issue_active.avg.pct_of_peak_sustained_active`) | 8.146460 | % |
| achieved occupancy (%) (`sm__warps_active.avg.pct_of_peak_sustained_active`) | 6.140455 | % |
| theoretical occupancy (%) (`sm__maximum_warps_per_active_cycle_pct`) | 56.250000 | % |
| registers per thread (`launch__registers_per_thread`) | 56 | register/thread |
| static shared memory per block (`launch__shared_mem_per_block_static`) | 80 | byte/block |
| dynamic shared memory per block (`launch__shared_mem_per_block_dynamic`) | 0.000000 | Kbyte/block |
| blocks per SM allowed by registers (`launch__occupancy_limit_registers`) | 9.000000 | block |
| blocks per SM allowed by shared memory (`launch__occupancy_limit_shared_mem`) | 28.000000 | block |
| waves per SM (`launch__waves_per_multiprocessor`) | 0.05 |  |
| shared-memory wavefronts (`l1tex__data_pipe_lsu_wavefronts_mem_shared.sum`) | 3456 |  |
| shared-memory bank conflicts (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`) | 0 |  |

| pipe utilization | % of peak (active cycles) |
|---|---|
| `sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active` | 2.716046 |
| `sm__inst_executed_pipe_fma.max.pct_of_peak_sustained_active` | 5.601844 |
| `sm__inst_executed_pipe_fma.min.pct_of_peak_sustained_active` | 0.000000 |
| `sm__inst_executed_pipe_fma.sum.pct_of_peak_sustained_active` | 2.716046 |
| `sm__inst_executed_pipe_fma_type_fp16.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_fmaheavy.avg.pct_of_peak_sustained_active` | 5.058323 |
| `sm__inst_executed_pipe_fmaheavy.max.pct_of_peak_sustained_active` | 10.432792 |
| `sm__inst_executed_pipe_fmaheavy.min.pct_of_peak_sustained_active` | 0.000000 |
| `sm__inst_executed_pipe_fmaheavy.sum.pct_of_peak_sustained_active` | 5.058323 |
| `sm__inst_executed_pipe_fmalite.avg.pct_of_peak_sustained_active` | 0.373768 |
| `sm__inst_executed_pipe_fmalite.max.pct_of_peak_sustained_active` | 0.770896 |
| `sm__inst_executed_pipe_fmalite.min.pct_of_peak_sustained_active` | 0.000000 |
| `sm__inst_executed_pipe_fmalite.sum.pct_of_peak_sustained_active` | 0.373768 |
| `sm__inst_executed_pipe_tensor_op_dmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_dmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_gmma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active` | 0.149507 |
| `sm__inst_executed_pipe_tensor_op_hmma.max.pct_of_peak_sustained_active` | 0.308358 |
| `sm__inst_executed_pipe_tensor_op_hmma.min.pct_of_peak_sustained_active` | 0.000000 |
| `sm__inst_executed_pipe_tensor_op_hmma.sum.pct_of_peak_sustained_active` | 0.149507 |
| `sm__inst_executed_pipe_tensor_op_imma.avg.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.max.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.min.pct_of_peak_sustained_active` | 0 |
| `sm__inst_executed_pipe_tensor_op_imma.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` | 3.130305 |
| `sm__pipe_fma_cycles_active.max.pct_of_peak_sustained_active` | 6.456254 |
| `sm__pipe_fma_cycles_active.min.pct_of_peak_sustained_active` | 0.000000 |
| `sm__pipe_fma_cycles_active.sum.pct_of_peak_sustained_active` | 3.130305 |
| `sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active` | 5.886842 |
| `sm__pipe_fmaheavy_cycles_active.max.pct_of_peak_sustained_active` | 12.141611 |
| `sm__pipe_fmaheavy_cycles_active.min.pct_of_peak_sustained_active` | 0.000000 |
| `sm__pipe_fmaheavy_cycles_active.sum.pct_of_peak_sustained_active` | 5.886842 |
| `sm__pipe_fmalite_cycles_active.avg.pct_of_peak_sustained_active` | 0.373768 |
| `sm__pipe_fmalite_cycles_active.max.pct_of_peak_sustained_active` | 0.770896 |
| `sm__pipe_fmalite_cycles_active.min.pct_of_peak_sustained_active` | 0.000000 |
| `sm__pipe_fmalite_cycles_active.sum.pct_of_peak_sustained_active` | 0.373768 |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | 0.149507 |
| `sm__pipe_tensor_cycles_active.max.pct_of_peak_sustained_active` | 0.308358 |
| `sm__pipe_tensor_cycles_active.min.pct_of_peak_sustained_active` | 0.000000 |
| `sm__pipe_tensor_cycles_active.sum.pct_of_peak_sustained_active` | 0.149507 |
| `sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_dmma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | 0.149507 |
| `sm__pipe_tensor_op_hmma_cycles_active.max.pct_of_peak_sustained_active` | 0.308358 |
| `sm__pipe_tensor_op_hmma_cycles_active.min.pct_of_peak_sustained_active` | 0.000000 |
| `sm__pipe_tensor_op_hmma_cycles_active.sum.pct_of_peak_sustained_active` | 0.149507 |
| `sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.max.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.min.pct_of_peak_sustained_active` | 0 |
| `sm__pipe_tensor_op_imma_cycles_active.sum.pct_of_peak_sustained_active` | 0 |

| stall reason | cycles per issued instruction |
|---|---|
| long_scoreboard | 5.20 |
| wait | 2.12 |
| short_scoreboard | 1.18 |
| imc_miss | 1.07 |
| barrier | 1.06 |
| selected | 1.00 |
| no_instruction | 0.66 |
| branch_resolving | 0.14 |
| dispatch_stall | 0.05 |
| drain | 0.04 |
