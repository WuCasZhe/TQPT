#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
python -m tqpt.experiments run --execute --resume "$@"
python -m tqpt.experiments summarize

