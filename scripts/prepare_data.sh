#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
python -m tqpt.data --raw-root data/raw --output-root data/processed --split-seed 42 "$@"

