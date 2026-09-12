#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${PYTHON:-python3}"
"$python_bin" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[torch,dev]'
cat <<'TEXT'
Activate with: source .venv/bin/activate
For NVIDIA Linux comparisons: pip install -e '.[triton,jax]'
Check your PyTorch wheel / toolkit / driver combination in docs/hardware.md.
TEXT
