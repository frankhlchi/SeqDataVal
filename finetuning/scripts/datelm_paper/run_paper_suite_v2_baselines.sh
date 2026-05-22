#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

RUN_TAG="${RUN_TAG:-paper_seed42_v1}"
export RUN_TAG
LOG_DIR="${DATELM_ROOT}/logs"

mkdir -p "${LOG_DIR}"

wait_for_llm_train_embeddings() {
  local progress_json="${DATELM_ROOT}/embeddings/${RUN_TAG}/tulu3_train_llama_lasttok/progress.json"
  local sleep_s="${1:-300}"
  while true; do
    if [[ -f "${progress_json}" ]] && grep -q '"train_complete": true' "${progress_json}" 2>/dev/null; then
      echo "[$(date '+%F %T')] LLM train embeddings ready: ${progress_json}"
      return 0
    fi
    echo "[$(date '+%F %T')] waiting for LLM train embeddings: ${progress_json}"
    sleep "${sleep_s}"
  done
}

run_pipeline_if_missing() {
  local method="$1"
  local task="$2"
  local script="$3"
  local metrics_path="${DATELM_ROOT}/results/${RUN_TAG}_${method}_${task}_official/metrics.json"
  local log_path="${LOG_DIR}/${RUN_TAG}_${method}_${task}_pipeline.log"

  if [[ -f "${metrics_path}" ]]; then
    echo "[$(date '+%F %T')] skip (exists): ${metrics_path}"
    return 0
  fi

  echo "[$(date '+%F %T')] starting ${method}/${task}: ${script}"
  bash "${script}" "${task}" > "${log_path}" 2>&1
  echo "[$(date '+%F %T')] finished ${method}/${task}: ${metrics_path}"
}

# Ensure we never overlap with the running LLM-embedding job (GPU-heavy).
wait_for_llm_train_embeddings 300

for task in mmlu gsm8k bbh; do
  run_pipeline_if_missing "bm25" "${task}" "${SCRIPT_DIR}/run_paper_bm25.sh"
done

for task in mmlu gsm8k bbh; do
  run_pipeline_if_missing "repsim" "${task}" "${SCRIPT_DIR}/run_paper_repsim.sh"
done

for task in mmlu gsm8k bbh; do
  run_pipeline_if_missing "bipcov_llm" "${task}" "${SCRIPT_DIR}/run_paper_bipcov_llm.sh"
done

echo "==== Summary (if available) ===="
for task in mmlu gsm8k bbh; do
  for method in bm25 repsim bipcov_llm; do
    p="${DATELM_ROOT}/results/${RUN_TAG}_${method}_${task}_official/metrics.json"
    if [[ -f "${p}" ]]; then
      echo "---- ${task} / ${method} ----"
      cat "${p}"
    else
      echo "---- ${task} / ${method} ---- (missing)"
    fi
  done
done
