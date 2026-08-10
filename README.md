# flash-attention-lab

FlashAttention-2 forward and KV-cache decode kernels written from scratch in CUDA and Triton,
exposed to PyTorch as custom ops, tested against an fp64 reference, and benchmarked against
PyTorch SDPA and flash-attn on H100.

Status: in progress. Results and design notes will be added as each kernel lands.

## Layout

```
flash_lab/     Python API, fp64 reference, input validation
csrc/          CUDA kernels and op registration
tests/         pytest suite (CPU reference tests and GPU kernel tests)
bench/         benchmark harness and committed results
scripts/pace/  build, test, and benchmark scripts for the Georgia Tech PACE-ICE cluster
docs/          environment notes
```

## Quick start

```bash
pip install torch numpy
pip install -e . --no-build-isolation   # builds the CUDA extension when nvcc is available
pytest -m "not gpu"                      # CPU tests
pytest                                   # everything, on a machine with a CUDA GPU
```
