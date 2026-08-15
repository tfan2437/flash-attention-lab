#!/usr/bin/env bash
# Runs the full test suite (CPU and GPU tests) on a GPU node. Extra arguments go to pytest,
# for example: scripts/pace/gpu.sh scripts/pace/test.sh -k decode -m "not slow"
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/pace/env.sh
source scripts/pace/common.sh
start_run test "$@"

python -c "import torch; print(torch.cuda.get_device_name())"
python -m pytest -q "$@"
