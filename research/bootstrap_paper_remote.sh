#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/pip" install --index-url https://download.pytorch.org/whl/cpu "torch==2.14.0"
"$VENV/bin/pip" install "numpy==2.5.3" "aiohttp==3.14.3" "loguru==0.7.3"
"$VENV/bin/python" -c "import torch,numpy; print(torch.__version__, numpy.__version__)"
