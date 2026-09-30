# Sourced by the other PACE scripts from the repo root: loads modules, activates the venv, and
# points caches at scratch (home has a 30 GB quota).
# Override the modules with FLASH_LAB_MODULES (default: "uv cuda/12.6.1").

if ! type module >/dev/null 2>&1; then
  for init in /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash /etc/profile.d/modules.sh; do
    if [ -f "$init" ]; then
      # shellcheck disable=SC1090
      source "$init"
      break
    fi
  done
fi

# Lmod reads unset variables, so relax nounset while it runs.
set +u
# shellcheck disable=SC2086
module load ${FLASH_LAB_MODULES:-uv cuda/12.6.1}
set -u

cache_root="$(cd .. && pwd)/.cache"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$cache_root/uv}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$cache_root/triton}"
export HF_HOME="${HF_HOME:-$cache_root/huggingface}"
# Model weights go to scratch with HF_HOME, but the access token stays where `hf auth login` puts
# it from a plain login shell.
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;9.0}"
export MAX_JOBS="${MAX_JOBS:-${SLURM_CPUS_PER_TASK:-8}}"
# Lets scripts run by path (profilers, examples) import flash_lab and bench from the checkout.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
