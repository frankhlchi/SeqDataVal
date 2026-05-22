#!/bin/bash
# Suite v3: RDS+ (Rep-Sim with Diversity Sampling) experiments
#
# This runs RDS+ (DATE-LM SOTA) on all three tasks: MMLU, GSM8K, BBH
#
# RDS+ = RepSim scores + Gumbel-Top-k selection (diversity sampling)
#
# Usage:
#   bash run_paper_suite_v3_rds_plus.sh [gumbel_temp]
#
# Default gumbel_temp=0.5 (DATE-LM default)

set -euo pipefail

GUMBEL_TEMP=${1:-0.5}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

RUN_TAG="${RUN_TAG:-paper_seed42_v1}"
export RUN_TAG
LOG_DIR="${DATELM_ROOT}/logs"
METHOD="rds_plus"

mkdir -p "${LOG_DIR}"

echo "=============================================="
echo "Suite v3: RDS+ (Gumbel temp=${GUMBEL_TEMP})"
echo "Started: $(date '+%F %T')"
echo "=============================================="

run_pipeline_if_missing() {
  local task="$1"
  local metrics_path="${DATELM_ROOT}/results/${RUN_TAG}_${METHOD}_${task}_official/metrics.json"
  local log_path="${LOG_DIR}/${RUN_TAG}_${METHOD}_${task}_pipeline.log"

  if [[ -f "${metrics_path}" ]]; then
    echo "[$(date '+%F %T')] skip (exists): ${metrics_path}"
    return 0
  fi

  echo "[$(date '+%F %T')] starting ${METHOD}/${task}"
  bash "${SCRIPT_DIR}/run_paper_rds_plus.sh" "${task}" "${GUMBEL_TEMP}" > "${log_path}" 2>&1
  echo "[$(date '+%F %T')] finished ${METHOD}/${task}: ${metrics_path}"
}

# Run RDS+ for all three tasks
for task in mmlu gsm8k bbh; do
  run_pipeline_if_missing "${task}"
done

echo ""
echo "=============================================="
echo "Suite v3 Summary"
echo "=============================================="

for task in mmlu gsm8k bbh; do
  p="${DATELM_ROOT}/results/${RUN_TAG}_${METHOD}_${task}_official/metrics.json"
  if [[ -f "${p}" ]]; then
    echo "---- ${task} / ${METHOD} ----"
    cat "${p}"
  else
    echo "---- ${task} / ${METHOD} ---- (missing)"
  fi
done

echo ""
echo "Completed: $(date '+%F %T')"
