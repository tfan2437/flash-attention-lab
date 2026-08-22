#!/usr/bin/env bash
# Mirrors the working tree (including .git, so runs record the commit) to PACE scratch, and
# brings run logs, benchmark results, and profiler reports back.
# Usage: scripts/pace/sync.sh push|pull
# Remote: $FLASH_LAB_HOST (default pace) and $FLASH_LAB_DIR (default scratch/flash-attention-lab)
set -euo pipefail
cd "$(dirname "$0")/../.."
host="${FLASH_LAB_HOST:-pace}"
dir="${FLASH_LAB_DIR:-scratch/flash-attention-lab}"

case "${1:-}" in
  push)
    ssh "$host" "mkdir -p '$dir'"
    # Changed files are found by checksum and land with a fresh mtime, so ninja on PACE always
    # sees them as newer than objects it built earlier. Excluded paths are also protected from
    # --delete, so remote builds and outputs survive.
    rsync -rlpz --checksum --delete \
      --exclude .venv/ --exclude build/ --exclude '*.so' --exclude __pycache__/ \
      --exclude .pytest_cache/ --exclude .ruff_cache/ --include runs/README.md --exclude 'runs/*' \
      --exclude .slurm_job \
      --exclude bench/results/ --exclude profiling/ --exclude requirements-pace.lock \
      ./ "$host:$dir/"
    ;;
  pull)
    for path in runs/ bench/results/ profiling/ requirements-pace.lock; do
      if ssh "$host" "test -e '$dir/$path'"; then
        mkdir -p "$(dirname "$path")"
        rsync -az "$host:$dir/$path" "./$path"
      fi
    done
    ;;
  *)
    echo "usage: $0 push|pull" >&2
    exit 2
    ;;
esac
