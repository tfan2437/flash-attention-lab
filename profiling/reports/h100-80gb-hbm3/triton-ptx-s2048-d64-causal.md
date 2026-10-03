# Triton PTX: seqlen 2048, head_dim 64, causal True

- triton 3.4.0, target sm_90a
- configuration: {'BLOCK_M': 64, 'BLOCK_N': 128, 'num_warps': 4, 'num_stages': 3}

| instruction | count in PTX |
|---|---|
| wgmma.mma_async (Hopper warpgroup MMA) | 24 |
| mma.sync (Ampere-style MMA) | 0 |
| cp.async.bulk.tensor (TMA) | 0 |
| cp.async (Ampere async copy) | 96 |
| ldmatrix | 0 |
| stmatrix | 4 |
| ex2.approx | 132 |
| mbarrier | 0 |
