#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
python -m tqpt.experiments run \
    --variants TQPT TQPT_NT TQPT_NS \
    --seeds 42 \
    --smoke \
    --smoke-samples 64 \
    --smoke-steps 2 \
    --eval-samples 32 \
    --execute \
    "$@"

