#!/usr/bin/env bash
# Profiles one kernel launch with Nsight Compute (full section set, source correlation) and
# writes a raw-metric CSV and a markdown digest to profiling/reports/<gpu>/. The .ncu-rep is kept
# next to them only when it is small enough to commit; otherwise it stays in the run directory.
#
# Usage: scripts/pace/gpu.sh scripts/pace/ncu.sh <name> <kernel regex> <python args...>
#   scripts/pace/ncu.sh mma-s4096 mma_attention -m bench.run --suite prefill_bf16 \
#       --impls mma --configs 1 --causal false --n-iters 12 --out-dir runs/bench
# Timings taken under the profiler are not results, hence --out-dir runs/bench for bench.run.
# NCU_SKIP (default 10) launches of the matching kernels are skipped first, so the profiled launch
# is a warm one; NCU_COUNT (default 1) launches are profiled, for implementations that launch
# several kernels per call.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/pace/env.sh
source scripts/pace/common.sh

report="$1"
regex="$2"
shift 2
gpu="$(python -c 'import torch; from bench.run import gpu_slug; print(gpu_slug(torch.cuda.get_device_name()))')"
out_dir="profiling/reports/$gpu"
mkdir -p "$out_dir"

profile() {
  ncu --set full --import-source yes --kernel-name "regex:$regex" --launch-skip "${NCU_SKIP:-10}" \
    --launch-count "${NCU_COUNT:-1}" --force-overwrite --export "$RUN_DIR/$report" python "$@"
  ncu --import "$RUN_DIR/$report.ncu-rep" --page raw --csv >"$out_dir/$report.csv"
  python scripts/pace/ncu_digest.py "$out_dir/$report.csv" >"$out_dir/$report.md"
  cat "$out_dir/$report.md"
  if [ "$(stat -c %s "$RUN_DIR/$report.ncu-rep")" -lt 5000000 ]; then
    cp "$RUN_DIR/$report.ncu-rep" "$out_dir/"
  fi
}

# `report`, not `name`: run_logged has a local `name` that bash would let profile() see.
run_logged "ncu-$report" profile "$@"
