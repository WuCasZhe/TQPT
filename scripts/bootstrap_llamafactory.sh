#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${LLAMAFACTORY_DIR:-${REPO_ROOT}/third_party/LLaMA-Factory}"
EXPECTED_COMMIT="baf2e4e825a61ffabef2b9f86d654f73ace8d120"

if [[ ! -d "${TARGET_DIR}/.git" ]]; then
    git clone --filter=blob:none --branch v0.1.0 https://github.com/hiyouga/LLaMA-Factory.git "${TARGET_DIR}"
fi

ACTUAL_COMMIT="$(git -C "${TARGET_DIR}" rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
    echo "LLaMA-Factory revision mismatch: expected ${EXPECTED_COMMIT}, got ${ACTUAL_COMMIT}" >&2
    exit 1
fi

echo "LLaMA-Factory v0.1.0 ready at ${TARGET_DIR}"

