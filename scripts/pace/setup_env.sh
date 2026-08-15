#!/usr/bin/env bash
# One-time setup, run on a GPU node: scripts/pace/gpu.sh scripts/pace/setup_env.sh
# Creates .venv (Python 3.12) with CUDA torch wheels and test dependencies, builds the extension,
# and records the installed versions in requirements-pace.lock.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/pace/env.sh

torch_index="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"

[ -d .venv ] || uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install --index-url "$torch_index" torch
uv pip install numpy pytest pytest-xdist ninja setuptools wheel
uv pip freeze >requirements-pace.lock

python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name() if torch.cuda.is_available() else "none")
EOF

scripts/pace/build.sh
