#!/usr/bin/env bash
# Traces a Python program with Nsight Systems (CUDA + NVTX) and writes the NVTX-range and
# per-range kernel summaries as CSV plus a markdown digest to profiling/reports/<gpu>/.
#
# Usage: scripts/pace/gpu.sh scripts/pace/nsys.sh <name> <python args...>
#   scripts/pace/nsys.sh decode-ctx8k -m bench.decode_trace --impls decode_copy,splitkv
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/pace/env.sh
source scripts/pace/common.sh

report="$1"
shift
gpu="$(python -c 'import torch; from bench.run import gpu_slug; print(gpu_slug(torch.cuda.get_device_name()))')"
out_dir="profiling/reports/$gpu"
mkdir -p "$out_dir"

trace() {
  nsys profile -t cuda,nvtx --force-overwrite true -o "$RUN_DIR/$report" python "$@"
  nsys stats --report nvtx_sum,nvtx_kern_sum --format csv --force-export true \
    --output "$out_dir/$report" "$RUN_DIR/$report.nsys-rep"
  python scripts/pace/nsys_digest.py "$out_dir/$report" >"$out_dir/$report.md"
  cat "$out_dir/$report.md"
}

# `report`, not `name`: run_logged has a local `name` that bash would let trace() see.
run_logged "nsys-$report" trace "$@"
