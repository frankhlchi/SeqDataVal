#!/bin/bash
# Suite v4: RepSim v2 + RDS+ v2 (NO NORMALIZE - matching DATE-LM original)
#
# This runs:
#   1. RepSim v2 (dot product, no L2 normalize) on MMLU/GSM8K/BBH
#   2. RDS+ v2 (Gumbel-Top-k on RepSim v2 scores) on MMLU/GSM8K/BBH
#
# Usage:
#   bash run_paper_suite_v4_repsim_v2.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"
PROJ_ROOT="${SEQDATAVAL_ROOT}"

RUN_TAG="${RUN_TAG:-paper_seed42_v1}"
export RUN_TAG
LOG_DIR="${DATELM_ROOT}/logs"

mkdir -p "${LOG_DIR}"

echo "=============================================="
echo "Suite v4: RepSim v2 + RDS+ v2 (NO NORMALIZE)"
echo "Started: $(date '+%F %T')"
echo "=============================================="

# Function to run RepSim v2
run_repsim_v2() {
  local task="$1"
  local method="repsim_v2"
  local metrics_path="${DATELM_ROOT}/results/${RUN_TAG}_${method}_${task}_official/metrics.json"
  local log_path="${LOG_DIR}/${RUN_TAG}_${method}_${task}_pipeline.log"

  if [[ -f "${metrics_path}" ]]; then
    echo "[$(date '+%F %T')] skip (exists): ${metrics_path}"
    return 0
  fi

  echo "[$(date '+%F %T')] starting ${method}/${task}"
  bash "${SCRIPT_DIR}/run_paper_repsim_v2.sh" "${task}" > "${log_path}" 2>&1
  echo "[$(date '+%F %T')] finished ${method}/${task}"
}

# Function to run RDS+ v2 (uses RepSim v2 scores with Gumbel-Top-k)
run_rds_plus_v2() {
  local task="$1"
  local gumbel_temp="${2:-0.5}"
  local method="rds_plus_v2"
  local repsim_method="repsim_v2"

  local metrics_path="${DATELM_ROOT}/results/${RUN_TAG}_${method}_${task}_official/metrics.json"
  local log_path="${LOG_DIR}/${RUN_TAG}_${method}_${task}_pipeline.log"

  if [[ -f "${metrics_path}" ]]; then
    echo "[$(date '+%F %T')] skip (exists): ${metrics_path}"
    return 0
  fi

  echo "[$(date '+%F %T')] starting ${method}/${task} (gumbel_temp=${gumbel_temp})"

  # Use RepSim v2 scores
  local SCORES_DIR="${DATELM_ROOT}/scores/${RUN_TAG}/${task}"
  local REPSIM_METRICS="${SCORES_DIR}/${repsim_method}_metrics.npy"
  local SELECT_DIR="${DATELM_ROOT}/selected_data/${RUN_TAG}/${task}"
  local TRAIN_JSONL="${SELECT_DIR}/${method}_10k.jsonl"
  local POOL_TRAIN_JSONL="${DATELM_ROOT}/data/training_data/${RUN_TAG}_tulu3_200k_train.jsonl"
  local ADAPTER_DIR="${DATELM_ROOT}/checkpoints/${RUN_TAG}_${method}_${task}_lora"
  local MERGED_DIR="${DATELM_ROOT}/checkpoints/${RUN_TAG}_${method}_${task}_merged"
  local RESULTS_DIR="${DATELM_ROOT}/results/${RUN_TAG}_${method}_${task}_official"

  mkdir -p "${SELECT_DIR}" "${LOG_DIR}"

  # Check if RepSim v2 scores exist
  if [[ ! -f "${REPSIM_METRICS}" ]]; then
    echo "ERROR: RepSim v2 scores not found: ${REPSIM_METRICS}"
    echo "Run RepSim v2 first!"
    return 1
  fi

  # Gumbel-Top-k selection
  if [[ ! -f "${TRAIN_JSONL}" ]]; then
    echo "  -> Gumbel-Top-k selection (temp=${gumbel_temp})"
    "${PY_TRAIN_EVAL}" "${SCRIPT_DIR}/select_jsonl_gumbel_topk.py" \
      --pool_jsonl "${POOL_TRAIN_JSONL}" \
      --metrics_npy "${REPSIM_METRICS}" \
      --out_jsonl "${TRAIN_JSONL}" \
      --k 10000 \
      --gumbel_temp "${gumbel_temp}" \
      --seed 42 \
      --out_indices_npy "${SELECT_DIR}/${method}_indices.npy" \
      >> "${log_path}" 2>&1
  fi

  # LoRA train
  if [[ ! -f "${ADAPTER_DIR}/adapter_config.json" ]]; then
    echo "  -> LoRA training"
    PYTHONUNBUFFERED=1 "${PY_TRAIN_EVAL}" "${SCRIPT_DIR}/train_lora_hf_paper.py" \
      --datelm_root "${DATELM_ROOT}" \
      --train_jsonl "${TRAIN_JSONL}" \
      --output_dir "${ADAPTER_DIR}" \
      --model_name "${BASE_MODEL}" \
      >> "${log_path}" 2>&1
  fi

  # Merge
  if [[ ! -f "${MERGED_DIR}/config.json" ]]; then
    echo "  -> Merging LoRA"
    "${PY_TRAIN_EVAL}" "${PROJ_ROOT}/finetuning/merge_lora_peft.py" \
      --adapter_path "${ADAPTER_DIR}" \
      --base_model "${BASE_MODEL}" \
      --output_path "${MERGED_DIR}" \
      >> "${log_path}" 2>&1
  fi

  # Eval
  cd "${DATELM_ROOT}"
  if [[ "${task}" == "mmlu" ]]; then
    "${PY_TRAIN_EVAL}" -m minimal_multitask.eval.mmlu.run_mmlu_eval \
      --ntrain 0 --data_dir data/eval/mmlu --save_dir "${RESULTS_DIR}" \
      --model_name_or_path "${MERGED_DIR}" --eval_batch_size 4 \
      --use_chat_format \
      --chat_formatting_function minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format \
      >> "${log_path}" 2>&1
  elif [[ "${task}" == "gsm8k" ]]; then
    "${PY_TRAIN_EVAL}" -m minimal_multitask.eval.gsm.run_eval \
      --data_dir data/eval/gsm --save_dir "${RESULTS_DIR}" \
      --model_name_or_path "${MERGED_DIR}" --n_shot 8 \
      --use_chat_format \
      --chat_formatting_function minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format \
      >> "${log_path}" 2>&1
  else
    "${PY_TRAIN_EVAL}" -m minimal_multitask.eval.bbh.run_eval \
      --data_dir data/eval/bbh --save_dir "${RESULTS_DIR}" \
      --model_name_or_path "${MERGED_DIR}" --eval_batch_size 4 \
      --use_chat_format \
      --chat_formatting_function minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format \
      >> "${log_path}" 2>&1
  fi

  echo "[$(date '+%F %T')] finished ${method}/${task}"
}

# Run RepSim v2 for all tasks first
echo ""
echo "=== Phase 1: RepSim v2 (NO NORMALIZE) ==="
for task in mmlu gsm8k bbh; do
  run_repsim_v2 "${task}"
done

# Run RDS+ v2 for all tasks
echo ""
echo "=== Phase 2: RDS+ v2 (Gumbel-Top-k on RepSim v2) ==="
for task in mmlu gsm8k bbh; do
  run_rds_plus_v2 "${task}" 0.5
done

# Summary
echo ""
echo "=============================================="
echo "Suite v4 Summary"
echo "=============================================="

for method in repsim_v2 rds_plus_v2; do
  echo ""
  echo "--- ${method} ---"
  for task in mmlu gsm8k bbh; do
    p="${DATELM_ROOT}/results/${RUN_TAG}_${method}_${task}_official/metrics.json"
    if [[ -f "${p}" ]]; then
      score=$(grep -oE '"average[^"]*":\s*[0-9.]+' "${p}" 2>/dev/null | head -1 | grep -oE '[0-9.]+$' || \
              grep -oE '"exact_match":\s*[0-9.]+' "${p}" 2>/dev/null | head -1 | grep -oE '[0-9.]+$' || echo "?")
      echo "  ${task}: ${score}"
    else
      echo "  ${task}: (missing)"
    fi
  done
done

echo ""
echo "Completed: $(date '+%F %T')"
