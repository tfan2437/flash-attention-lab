#!/usr/bin/env bash
# Holds one GPU so that repeated builds and test runs skip the queue; scripts/pace/gpu.sh runs
# commands on it. Release it early with: scancel "$(cat .slurm_job)"
# Usage: scripts/pace/alloc.sh [gpu type, default h100] [hours, default 4]
set -euo pipefail
cd "$(dirname "$0")/../.."
gpu="${1:-h100}"
hours="${2:-4}"

salloc --no-shell --job-name=flash-lab --gres="gpu:${gpu}:1" --cpus-per-task=8 --mem=64G \
  --time="${hours}:00:00"
job="$(squeue --me --name=flash-lab --states=RUNNING --noheader --format=%i | head -n 1)"
if [ -z "$job" ]; then
  echo "no running flash-lab allocation found" >&2
  exit 1
fi
echo "$job" >.slurm_job
echo "job $job on $(squeue --job "$job" --noheader --format=%N), $gpu for ${hours}h"
