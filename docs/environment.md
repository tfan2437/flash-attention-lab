# Environment

Code is written and CPU-tested on a laptop without a GPU. Everything that needs CUDA runs on
Georgia Tech's PACE-ICE cluster (H100 nodes, A100 or L40S when the H100 queue is long).

## PACE-ICE access

Off campus, connect to the GT VPN first. Logins use a password and Duo, so the SSH config shares
one authenticated connection for the day:

```
Host pace
    HostName login-ice.pace.gatech.edu
    User <gt-username>
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 8h
```

Work lives under scratch (`~/scratch/flash-attention-lab`); home has a 30 GB quota. Run
`pace-quota` on any ICE node to find the scratch path if the `~/scratch` link does not exist yet.

## Loop

```bash
scripts/pace/sync.sh push                       # laptop -> scratch (includes .git)
ssh pace 'cd scratch/flash-attention-lab && scripts/pace/alloc.sh h100 4'
ssh pace 'cd scratch/flash-attention-lab && scripts/pace/gpu.sh scripts/pace/setup_env.sh'  # once
ssh pace 'cd scratch/flash-attention-lab && scripts/pace/gpu.sh scripts/pace/build.sh'
ssh pace 'cd scratch/flash-attention-lab && scripts/pace/gpu.sh scripts/pace/test.sh'
ssh pace 'cd scratch/flash-attention-lab && scripts/pace/gpu.sh scripts/pace/sanitize.sh'
scripts/pace/sync.sh pull                       # runs/, bench/results/, profiling/ back
ssh pace 'scancel "$(cat scratch/flash-attention-lab/.slurm_job)"'
```

`alloc.sh` holds one GPU (`salloc --no-shell`) so that repeated builds and test runs do not wait
in the queue; `gpu.sh` runs a command on it with `srun --overlap`.

Each script writes `runs/<timestamp>-<sha>-<name>/` containing the command, an environment JSON
(GPU, driver, clocks, library versions, git SHA, Slurm job), the full log, and a summary.
Benchmark results that end up in the README are produced from a clean checkout, so their
environment block names an exact commit.

## Llama checkpoints

The end-to-end demo, `bench/e2e.py`, and `tests/test_e2e_llama.py` use
`meta-llama/Llama-3.1-8B-Instruct` and `meta-llama/Llama-3.2-1B-Instruct`, which are gated:
accept the license on huggingface.co, log in once on PACE, and download the weights to scratch.

```bash
ssh -t pace '~/scratch/flash-attention-lab/.venv/bin/hf auth login'
ssh pace 'cd scratch/flash-attention-lab && scripts/pace/gpu.sh scripts/pace/models.sh'
```

The token stays in `~/.cache/huggingface/token` on PACE, where `HF_TOKEN_PATH` (set in `env.sh`)
points; the weights go to `HF_HOME` on scratch, without the duplicate `original/` checkpoints
(17 GB for both models). The Llama tests skip when the 1B checkpoint is not cached.

## Verified setup (2026-08-22)

- GPU: NVIDIA H100 80GB HBM3 (SXM5, 132 SMs, compute capability 9.0), driver 595.71.05, partition
  `ice-gpu`. `nvidia-smi` reported an SM clock of 1980 MHz and a memory clock of 2619 MHz.
- Toolchain: `cuda/12.6.1` (nvcc 12.6) with gcc 12.3 as the host compiler. The extension is built
  as C++20, which newer torch headers (2.14) require and 2.8 accepts.
- Python 3.12.12, torch 2.8.0+cu126, triton 3.4.0, flash-attn 2.8.3.post1, and (from
  2026-09-30) transformers 4.57.6 with accelerate 1.15.0; the full list is in
  `requirements-pace.lock`. torch is pinned to 2.8 because flash-attn 2.x ships prebuilt wheels
  only up to torch 2.8, and flash-attn is the external baseline for prefill and decode.
- Profilers: Nsight Compute 2026.2.1 reads hardware counters without extra permissions; Nsight
  Systems 2024.4.2 and `compute-sanitizer` come with the CUDA module.

## Modules and versions

`scripts/pace/env.sh` loads `uv` and `cuda/12.6.1` (override with `FLASH_LAB_MODULES`).
`scripts/pace/setup_env.sh` installs torch 2.8.0 from the `cu126` wheel index (override with
`TORCH_VERSION` and `TORCH_INDEX`) plus the matching flash-attn wheel, and freezes the result in
`requirements-pace.lock`.
