#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANT="TQPT"
SEED="42"
MAX_STEPS="${MAX_STEPS:-1000}"
MODEL="artifacts/models/chatglm2-6b-traffic"
TOKENIZER="artifacts/tokenizer/chatglm2-6b-traffic"
PREFIX_ROOT="runs/${VARIANT}/${SEED}/prefix"
LOG_ROOT="runs/${VARIANT}/${SEED}/logs"
STAGE2_PROTOCOL="trafficllm-official-boundary-tqpt-validation-v1"

cd "${REPO_ROOT}"
mkdir -p "${PREFIX_ROOT}" "${LOG_ROOT}"

if pgrep -af '[p]ython.*-m tqpt\.prefix_train' >/dev/null; then
    echo "[error] Another Prefix-Tuning process is already running; wait for it to finish." >&2
    pgrep -af '[p]ython.*-m tqpt\.prefix_train' >&2
    exit 1
fi

run_prefix() {
    local task="$1"
    local task_lower="$2"
    local label_file="$3"
    local output_dir="${PREFIX_ROOT}/${task_lower}"
    local log_file="${LOG_ROOT}/prefix-${task_lower}.log"
    local checkpoint=""
    local data_protocol=""
    local backup_dir=""
    local -a resume_args=()

    if [[ -f "${output_dir}/tqpt_adapter.json" ]]; then
        data_protocol="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("data_protocol", ""))' "${output_dir}/tqpt_adapter.json")"
    fi
    if [[ -f "${output_dir}/pytorch_model.bin" \
        && -f "${output_dir}/tqpt_adapter.json" \
        && "${data_protocol}" == "${STAGE2_PROTOCOL}" ]]; then
        echo "[skip] ${task} Prefix adapter already complete: ${output_dir}"
        return
    fi

    if [[ -f "${output_dir}/pytorch_model.bin" || -f "${output_dir}/tqpt_adapter.json" ]]; then
        backup_dir="${output_dir}.pre-official-boundary"
        if [[ -e "${backup_dir}" ]]; then
            backup_dir="${backup_dir}-$(date +%Y%m%d%H%M%S)"
        fi
        echo "[retrain] ${task} adapter predates the official data-boundary protocol; moving it to ${backup_dir}"
        mv -- "${output_dir}" "${backup_dir}"
    fi

    if [[ "${data_protocol}" == "${STAGE2_PROTOCOL}" && -d "${output_dir}" ]]; then
        checkpoint="$(find "${output_dir}" -maxdepth 1 -type d -name 'checkpoint-*' -print | sort -V | tail -n 1)"
    fi
    if [[ -n "${checkpoint}" ]]; then
        echo "[resume] ${task} from ${checkpoint}"
        resume_args=(--resume-from-checkpoint "${checkpoint}")
    else
        echo "[train] ${task} for ${MAX_STEPS} optimizer steps"
    fi

    python -m tqpt.prefix_train \
        --task "${task}" \
        --variant "${VARIANT}" \
        --seed "${SEED}" \
        --model "${MODEL}" \
        --tokenizer "${TOKENIZER}" \
        --train-file "data/processed/stage2/${task_lower}/train.jsonl" \
        --validation-file "data/processed/stage2/${task_lower}/validation.jsonl" \
        --label-file "${label_file}" \
        --output-dir "${output_dir}" \
        --max-steps "${MAX_STEPS}" \
        "${resume_args[@]}" \
        2>&1 | tee "${log_file}"

    if [[ ! -f "${output_dir}/pytorch_model.bin" || ! -f "${output_dir}/tqpt_adapter.json" ]]; then
        echo "[error] ${task} finished without a complete adapter in ${output_dir}" >&2
        exit 1
    fi
}

run_prefix "EVD" "evd" "data/raw/iscx-vpn-2016/iscx-vpn-2016_label.json"
run_prefix "AAD" "aad" "data/raw/dapt-2020/dapt-2020_label.json"
run_prefix "CD" "cd" "data/raw/app53-2023/app53-2023_label.json"

echo "[inference] Running end-to-end evaluation"
python -m tqpt.inference \
    --variant "${VARIANT}" \
    --seed "${SEED}" \
    --model "${MODEL}" \
    --tokenizer "${TOKENIZER}" \
    --processed-root data/processed \
    --raw-root data/raw \
    --router-adapter "runs/${VARIANT}/${SEED}/router" \
    --prefix-root "${PREFIX_ROOT}" \
    --output-dir "results/${VARIANT}/${SEED}" \
    2>&1 | tee "${LOG_ROOT}/inference.log"

echo "[done] Prefix training and inference completed successfully"
