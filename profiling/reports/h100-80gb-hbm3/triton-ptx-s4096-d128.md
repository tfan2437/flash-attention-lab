---------------------------------------
Begin Slurm Prolog: Aug-30-2026 03:36:32
Job ID:    6082047
User ID:   tfan60
Account:   coc
Job name:  python
Partition: ice-gpu
QOS:
---------------------------------------
# Triton PTX: seqlen 4096, head_dim 128, causal False

- triton 3.4.0, target sm_90a
- configuration: {'BLOCK_M': 128, 'BLOCK_N': 64, 'num_warps': 8, 'num_stages': 3}

| instruction | count in PTX |
|---|---|
| wgmma.mma_async (Hopper warpgroup MMA) | 24 |
| mma.sync (Ampere-style MMA) | 0 |
| cp.async.bulk.tensor (TMA) | 0 |
| cp.async (Ampere async copy) | 48 |
| ldmatrix | 0 |
| stmatrix | 8 |
| ex2.approx | 68 |
| mbarrier | 0 |
