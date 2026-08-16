#!/usr/bin/env bash
# Local check before pushing: formatting, lint, and the CPU test suite.
set -euo pipefail
cd "$(dirname "$0")/../.."
pre-commit run --all-files
python -m pytest -m "not gpu" -q
