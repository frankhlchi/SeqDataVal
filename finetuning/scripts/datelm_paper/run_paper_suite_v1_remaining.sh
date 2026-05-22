#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

RUN_TAG="${RUN_TAG:-paper_seed42_v1}"
export RUN_TAG
LOG_DIR="${DATELM_ROOT}/logs"

mkdir -p "${LOG_DIR}"

wait_for_file() {
  local file_path="$1"
  local label="$2"
  local sleep_s="${3:-300}"
  while [[ ! -f "${file_path}" ]]; do
    echo "[$(date '+%F %T')] waiting for ${label}: ${file_path}"
    sleep "${sleep_s}"
  done
}

# 0) Wait for the currently running GSM8K(random) pipeline to finish.
wait_for_file "${DATELM_ROOT}/results/${RUN_TAG}_random_gsm8k_official/metrics.json" "GSM8K random metrics" 300

# 1) GSM8K bipcov
if [[ ! -f "${DATELM_ROOT}/results/${RUN_TAG}_bipcov_gsm8k_official/metrics.json" ]]; then
  echo "[$(date '+%F %T')] starting GSM8K bipcov pipeline"
  bash "${SCRIPT_DIR}/run_paper_gsm8k.sh" bipcov > "${LOG_DIR}/${RUN_TAG}_bipcov_gsm8k_pipeline.log" 2>&1
else
  echo "[$(date '+%F %T')] GSM8K bipcov metrics already exist; skipping"
fi

# 2) BBH random
if [[ ! -f "${DATELM_ROOT}/results/${RUN_TAG}_random_bbh_official/metrics.json" ]]; then
  echo "[$(date '+%F %T')] starting BBH random pipeline"
  bash "${SCRIPT_DIR}/run_paper_bbh.sh" random > "${LOG_DIR}/${RUN_TAG}_random_bbh_pipeline.log" 2>&1
else
  echo "[$(date '+%F %T')] BBH random metrics already exist; skipping"
fi

# 3) BBH bipcov
if [[ ! -f "${DATELM_ROOT}/results/${RUN_TAG}_bipcov_bbh_official/metrics.json" ]]; then
  echo "[$(date '+%F %T')] starting BBH bipcov pipeline"
  bash "${SCRIPT_DIR}/run_paper_bbh.sh" bipcov > "${LOG_DIR}/${RUN_TAG}_bipcov_bbh_pipeline.log" 2>&1
else
  echo "[$(date '+%F %T')] BBH bipcov metrics already exist; skipping"
fi

# 4) Print a short summary (if available)
for t in mmlu gsm8k bbh; do
  for m in random bipcov; do
    p="${DATELM_ROOT}/results/${RUN_TAG}_${m}_${t}_official/metrics.json"
    if [[ -f "${p}" ]]; then
      echo "==== ${t} / ${m} ===="
      cat "${p}"
    else
      echo "==== ${t} / ${m} ==== (missing)"
    fi
  done
done
