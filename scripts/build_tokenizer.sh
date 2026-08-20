#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
python -m tqpt.tokenizer \
    --base-model models/chatglm2-6b \
    --raw-root data/raw \
    --tokenizer-output artifacts/tokenizer/chatglm2-6b-traffic \
    --model-output artifacts/models/chatglm2-6b-traffic \
    "$@"

