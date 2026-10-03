# Environment

Code is written and CPU-tested on a laptop without a GPU. Everything that needs CUDA runs on
Georgia Tech's PACE-ICE cluster; every recorded run so far used an H100 node.

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

## Hardware

| | Development | GPU runs |
|---|---|---|
| Machine | Apple M4 Mac | PACE-ICE, partition `ice-gpu`; so far nodes atl1-1-03-012-18-0, atl1-1-03-013-13-0, and atl1-1-03-013-8-0 (each run's `env.json` names its node) |
| CPU | Apple M4, 10 cores | 2 x Intel Xeon Platinum 8462Y+, 64 cores at 2.8 GHz (recorded by Nsight Systems on atl1-1-03-013-13-0) |
| Memory | 16 GB | Slurm allocation from `alloc.sh`: 8 cores and 64 GB |
| OS | macOS 26.6.2 | Red Hat Enterprise Linux 9.6, kernel 5.14.0-570.128.1.el9_6 |
| GPU | none | 1 x NVIDIA H100 80GB HBM3 |

The H100 is the SXM5 part (GH100). From the CUDA device properties and `nvidia-smi` output
recorded with the runs:

- 132 SMs, compute capability 9.0, maximum clocks 1980 MHz (SM) and 2619 MHz (memory); driver
  595.71.05 (CUDA 13.2 driver API). The clocks in each `env.json` are sampled before the work
  starts: memory always read 2619 MHz, while the SM clock read anywhere from 345 MHz (idle) to
  1980 MHz, so that sample does not describe the clock during timing.
- 80 GB of HBM3 (the device reports 79.2 GiB) with 3352 GB/s of bandwidth; 50 MiB of L2 cache.
- Per SM: 228 KiB of shared memory (up to 227 KiB per block with opt-in), 65536 32-bit
  registers, and up to 64 warps and 32 blocks.
- The "% of peak" figures use NVIDIA's H100 SXM datasheet values (`bench/flops.py`):
  989.4 TFLOP/s dense bf16/fp16 on tensor cores, 67 TFLOP/s fp32 on CUDA cores, and 3350 GB/s.

From 2026-10-02, `bench/env.py` also records the host CPU, memory, OS, and Slurm allocation in
every `env.json` and benchmark JSON.

## Software

- GPU runs: Python 3.12.12 in a uv venv; torch 2.8.0+cu126, triton 3.4.0, flash-attn
  2.8.3.post1 (prebuilt wheel), and, from 2026-09-30, transformers 4.57.6 with accelerate 1.15.0.
  The full list is in `requirements-pace.lock`. torch is pinned to 2.8 because flash-attn 2.x
  ships prebuilt wheels only up to torch 2.8, and flash-attn is the external baseline for
  prefill and decode.
- Toolchain: the `cuda/12.6.1` module (nvcc 12.6.68) with g++ 12.3.0 as the host compiler. The
  extension is built as C++20, which newer torch headers (2.14) require and 2.8 accepts, for
  sm_80 and sm_90 (`TORCH_CUDA_ARCH_LIST`). Triton compiles its kernel for sm_90a, where it emits
  wgmma instructions.
- Profilers and checkers: Nsight Compute 2026.2.1 reads hardware counters without extra
  permissions; Nsight Systems 2024.4.2 and `compute-sanitizer` come with the CUDA module.
- Laptop: Python 3.12.13 with a CPU build of torch 2.14.1 for the CPU tests; pre-commit runs
  ruff 0.16.10 and clang-format 22.1.8.
- CI (GitHub Actions, ubuntu-latest): Python 3.12, the current CPU torch wheel, pre-commit, and
  the tests not marked `gpu`.

## Models

The end-to-end demo, `bench/e2e.py`, and `tests/test_e2e_llama.py` run two Llama checkpoints in
bf16. Together they cover both head dims the kernels support.

| | Llama-3.1-8B-Instruct | Llama-3.2-1B-Instruct |
|---|---|---|
| Used by | `bench/e2e.py`, `bench/e2e_trace.py`, `examples/generate.py` | `tests/test_e2e_llama.py` |
| Revision | 0e9e39f (recorded in the e2e JSON) | 9213176 (the cached snapshot; the tests do not pin it) |
| Layers | 32 | 16 |
| Hidden size / MLP size | 4096 / 14336 | 2048 / 8192 |
| Query heads / KV heads | 32 / 8 (4 query heads per KV head) | 32 / 8 |
| Head dim | 128 | 64 |
| Vocabulary | 128256 | 128256 |
| Weights | about 16 GB | about 2.5 GB |

Both are gated: accept the license on huggingface.co, log in once on PACE, and download the
weights to scratch.

```bash
ssh -t pace '~/scratch/flash-attention-lab/.venv/bin/hf auth login'
ssh pace 'cd scratch/flash-attention-lab && scripts/pace/gpu.sh scripts/pace/models.sh'
```

The token stays in `~/.cache/huggingface/token` on PACE, where `HF_TOKEN_PATH` (set in `env.sh`)
points; the weights go to `HF_HOME` on scratch, without the duplicate `original/` checkpoints.
The Llama tests skip when the 1B checkpoint is not cached.

## Modules and versions

`scripts/pace/env.sh` loads `uv` and `cuda/12.6.1` (override with `FLASH_LAB_MODULES`).
`scripts/pace/setup_env.sh` installs torch 2.8.0 from the `cu126` wheel index (override with
`TORCH_VERSION` and `TORCH_INDEX`) plus the matching flash-attn wheel, and freezes the result in
`requirements-pace.lock`.
