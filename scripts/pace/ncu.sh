#!/usr/bin/env bash
# Profiles one kernel launch with Nsight Compute (full section set, source correlation) and
# writes a raw-metric CSV and a markdown digest to profiling/reports/<gpu>/. The .ncu-rep is kept
# next to them only when it is small enough to commit; otherwise it stays in the run directory.
#
# Usage: scripts/pace/gpu.sh scripts/pace/ncu.sh <name> <kernel regex> <python args...>
#   scripts/pace/ncu.sh mma-s4096 mma_attention -m bench.run --suite prefill_bf16 \
#       --impls mma --configs 1 --causal false --n-iters 12 --allow-dirty
# NCU_SKIP (default 10) launches of the matching kernel are skipped first, so the profiled launch
# is a warm one.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/pace/env.sh
source scripts/pace/common.sh

name="$1"
regex="$2"
shift 2
gpu="$(python -c 'import torch; from bench.run import gpu_slug; print(gpu_slug(torch.cuda.get_device_name()))')"
out_dir="profiling/reports/$gpu"
mkdir -p "$out_dir"

profile() {
  ncu --set full --import-source yes --kernel-name "regex:$regex" --launch-skip "${NCU_SKIP:-10}" \
    --launch-count 1 --force-overwrite --export "$RUN_DIR/$name" python "$@"
  ncu --import "$RUN_DIR/$name.ncu-rep" --page raw --csv >"$out_dir/$name.csv"
  python scripts/pace/ncu_digest.py "$out_dir/$name.csv" >"$out_dir/$name.md"
  cat "$out_dir/$name.md"
  if [ "$(stat -c %s "$RUN_DIR/$name.ncu-rep")" -lt 5000000 ]; then
    cp "$RUN_DIR/$name.ncu-rep" "$out_dir/"
  fi
}

run_logged "ncu-$name" profile "$@"
