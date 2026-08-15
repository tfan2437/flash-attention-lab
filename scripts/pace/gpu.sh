#!/usr/bin/env bash
# Runs a command on the GPU node held by scripts/pace/alloc.sh, from the repo root.
# Usage: scripts/pace/gpu.sh scripts/pace/test.sh -k smoke
set -euo pipefail
cd "$(dirname "$0")/../.."
job="${FLASH_LAB_JOB:-$(cat .slurm_job 2>/dev/null || true)}"
if [ -z "$job" ]; then
  echo "no allocation; run scripts/pace/alloc.sh first" >&2
  exit 1
fi
exec srun --jobid="$job" --overlap --ntasks=1 --cpus-per-task=8 "$@"
