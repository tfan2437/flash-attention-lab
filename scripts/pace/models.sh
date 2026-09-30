#!/usr/bin/env bash
# Downloads the checkpoints used by the end-to-end demo and tests into HF_HOME on scratch.
# Both are gated: accept the license on huggingface.co, then run `.venv/bin/hf auth login` once
# from a login shell (the token stays in ~/.cache/huggingface, see HF_TOKEN_PATH in env.sh).
# Only the safetensors weights are fetched, not the duplicate original/ checkpoints.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/pace/env.sh

models=("$@")
if [ ${#models[@]} -eq 0 ]; then
  models=(meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.1-8B-Instruct)
fi
for model in "${models[@]}"; do
  hf download "$model" --exclude "original/*"
done
du -sh "$HF_HOME/hub"/models--meta-llama--* 2>/dev/null || true
