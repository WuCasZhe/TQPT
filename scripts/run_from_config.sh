#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-smoke}"
cd "${REPO_ROOT}"

"${REPO_ROOT}/scripts/bootstrap_llamafactory.sh"

case "${MODE}" in
    smoke)
        python -m tqpt.pipeline --config configs/tqpt.yaml --mode smoke
        ;;
    full)
        python -m tqpt.pipeline --config configs/tqpt.yaml --mode full
        ;;
    *)
        echo "Usage: $0 [smoke|full]" >&2
        exit 2
        ;;
esac
