#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

conda create -n tqpt-chatglm2 python=3.9 -y
conda run -n tqpt-chatglm2 python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.1.0
conda run -n tqpt-chatglm2 python -m pip install -r "${REPO_ROOT}/requirements.txt"
conda run -n tqpt-chatglm2 python -m pip install -e "${REPO_ROOT}"
"${REPO_ROOT}/scripts/bootstrap_llamafactory.sh"

echo "Environment created. Model weights were not downloaded."
