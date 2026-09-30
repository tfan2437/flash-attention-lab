#!/usr/bin/env bash
# One-time setup, run on a GPU node: scripts/pace/gpu.sh scripts/pace/setup_env.sh
# Creates .venv (Python 3.12) with CUDA torch wheels, test dependencies, and the flash-attn
# baseline, builds the extension, and records the installed versions in requirements-pace.lock.
#
# torch is pinned to 2.8 because flash-attn 2.x publishes prebuilt wheels only up to torch 2.8,
# and flash-attn is the external baseline for both prefill and KV-cache decode.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/pace/env.sh

torch_index="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
torch_version="${TORCH_VERSION:-2.8.0}"
flash_attn_version="${FLASH_ATTN_VERSION:-2.8.3.post1}"

[ -d .venv ] || uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install --index-url "$torch_index" "torch==${torch_version}"
uv pip install numpy pytest pytest-xdist ninja setuptools wheel matplotlib
# The end-to-end Llama demo (flash_lab.hf, examples/generate.py, bench/e2e.py).
uv pip install "transformers>=4.56,<5" accelerate

read -r torch_mm abi < <(python -c "import torch; v = torch.__version__.split('.'); \
print(v[0] + '.' + v[1], 'TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
wheel="flash_attn-${flash_attn_version}+cu12torch${torch_mm}cxx11abi${abi}-cp312-cp312-linux_x86_64.whl"
url="https://github.com/Dao-AILab/flash-attention/releases/download/v${flash_attn_version}/${wheel/+/%2B}"
if curl -sfIL "$url" >/dev/null; then
  uv pip install "$url"
else
  echo "no prebuilt flash-attn wheel for torch ${torch_mm}; flash_attn baselines will be skipped"
fi

uv pip freeze >requirements-pace.lock
python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name() if torch.cuda.is_available() else "none")
EOF

rm -rf build flash_lab/_C*.so
scripts/pace/build.sh
