#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

TASK=${1:-}
if [[ -z "${TASK}" ]]; then
  echo "Usage: $0 <mmlu|gsm8k|bbh>"
  exit 1
fi
if [[ "${TASK}" != "mmlu" && "${TASK}" != "gsm8k" && "${TASK}" != "bbh" ]]; then
  echo "Unknown TASK: ${TASK} (expected mmlu|gsm8k|bbh)"
  exit 1
fi

PROJ_ROOT="${SEQDATAVAL_ROOT}"

RUN_TAG="${RUN_TAG:-paper_seed42_v1}"
SEED_POOL=42

TULU3_JSONL="${DATELM_ROOT}/data/training_data/tulu_3_v3.9_unfiltered.jsonl"
POOL_TRAIN_JSONL="${DATELM_ROOT}/data/training_data/${RUN_TAG}_tulu3_200k_train.jsonl"
POOL_VAL_JSONL="${DATELM_ROOT}/data/training_data/${RUN_TAG}_tulu3_22k_val.jsonl"

SELECT_DIR="${DATELM_ROOT}/selected_data/${RUN_TAG}/${TASK}"
SCORES_DIR="${DATELM_ROOT}/scores/${RUN_TAG}/${TASK}"

METHOD="bm25"
METRICS_NPY="${SCORES_DIR}/${METHOD}_metrics.npy"

TRAIN_JSONL="${SELECT_DIR}/${METHOD}_10k.jsonl"

ADAPTER_DIR="${DATELM_ROOT}/checkpoints/${RUN_TAG}_${METHOD}_${TASK}_lora"
MERGED_DIR="${DATELM_ROOT}/checkpoints/${RUN_TAG}_${METHOD}_${TASK}_merged"
RESULTS_DIR="${DATELM_ROOT}/results/${RUN_TAG}_${METHOD}_${TASK}_official"
LOG_DIR="${DATELM_ROOT}/logs"

mkdir -p "${LOG_DIR}" "${SELECT_DIR}" "${SCORES_DIR}"

# 1) Prepare 200k pool if missing
if [[ ! -f "${POOL_TRAIN_JSONL}" ]]; then
  echo "[1/6] Preparing 200k pool -> ${POOL_TRAIN_JSONL}"
  "${PY_TRAIN_EVAL}" "${PROJ_ROOT}/finetuning/scripts/datelm_paper/prepare_tulu3_pool.py" \
    --input_jsonl "${TULU3_JSONL}" \
    --out_train_jsonl "${POOL_TRAIN_JSONL}" \
    --out_val_jsonl "${POOL_VAL_JSONL}" \
    --seed "${SEED_POOL}" \
    --pool_size 200000 \
    --val_split 0.1
else
  echo "[1/6] Pool exists: ${POOL_TRAIN_JSONL}"
fi

# 2) BM25 scoring (DATE-LM baseline)
if [[ ! -f "${METRICS_NPY}" ]]; then
  echo "[2/6] BM25 scoring -> ${METRICS_NPY}"
  PYTHONUNBUFFERED=1 "${PY_TRAIN_EVAL}" "${PROJ_ROOT}/finetuning/scripts/datelm_paper/compute_bm25_scores_pool.py" \
    --datelm_root "${DATELM_ROOT}" \
    --pool_jsonl "${POOL_TRAIN_JSONL}" \
    --task "${TASK}" \
    --ref_num_samples 100 \
    --ref_seed 42 \
    --model_name "${BASE_MODEL}" \
    --out_npy "${METRICS_NPY}" \
    > "${LOG_DIR}/${RUN_TAG}_${METHOD}_${TASK}_score.log" 2>&1
else
  echo "[2/6] Metrics exist: ${METRICS_NPY}"
fi

# 3) Selection
if [[ ! -f "${TRAIN_JSONL}" ]]; then
  echo "[3/6] Selecting top-10k -> ${TRAIN_JSONL}"
  "${PY_TRAIN_EVAL}" "${PROJ_ROOT}/finetuning/scripts/datelm_paper/select_jsonl_topk.py" \
    --pool_jsonl "${POOL_TRAIN_JSONL}" \
    --metrics_npy "${METRICS_NPY}" \
    --out_jsonl "${TRAIN_JSONL}" \
    --k 10000 \
    --out_indices_npy "${SELECT_DIR}/${METHOD}_indices.npy" \
    > "${LOG_DIR}/${RUN_TAG}_${METHOD}_${TASK}_select.log" 2>&1
else
  echo "[3/6] Selected JSONL exists: ${TRAIN_JSONL}"
fi

# 4) LoRA train
if [[ ! -f "${ADAPTER_DIR}/adapter_config.json" ]]; then
  echo "[4/6] Training LoRA -> ${ADAPTER_DIR}"
  PYTHONUNBUFFERED=1 "${PY_TRAIN_EVAL}" "${PROJ_ROOT}/finetuning/scripts/datelm_paper/train_lora_hf_paper.py" \
    --datelm_root "${DATELM_ROOT}" \
    --train_jsonl "${TRAIN_JSONL}" \
    --output_dir "${ADAPTER_DIR}" \
    --model_name "${BASE_MODEL}" \
    > "${LOG_DIR}/${RUN_TAG}_${METHOD}_${TASK}_finetune.log" 2>&1
else
  echo "[4/6] Adapter exists: ${ADAPTER_DIR}"
fi

# 5) Merge
if [[ ! -f "${MERGED_DIR}/config.json" ]]; then
  echo "[5/6] Merging -> ${MERGED_DIR}"
  "${PY_TRAIN_EVAL}" "${PROJ_ROOT}/finetuning/merge_lora_peft.py" \
    --adapter_path "${ADAPTER_DIR}" \
    --base_model "${BASE_MODEL}" \
    --output_path "${MERGED_DIR}" \
    > "${LOG_DIR}/${RUN_TAG}_${METHOD}_${TASK}_merge.log" 2>&1
else
  echo "[5/6] Merged model exists: ${MERGED_DIR}"
fi

# 6) Eval
cd "${DATELM_ROOT}"
if [[ "${TASK}" == "mmlu" ]]; then
  echo "[6/6] Running official MMLU eval -> ${RESULTS_DIR}"
  "${PY_TRAIN_EVAL}" -m minimal_multitask.eval.mmlu.run_mmlu_eval \
    --ntrain 0 \
    --data_dir data/eval/mmlu \
    --save_dir "${RESULTS_DIR}" \
    --model_name_or_path "${MERGED_DIR}" \
    --eval_batch_size 4 \
    --use_chat_format \
    --chat_formatting_function minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format \
    > "${LOG_DIR}/${RUN_TAG}_${METHOD}_${TASK}_eval.log" 2>&1
elif [[ "${TASK}" == "gsm8k" ]]; then
  echo "[6/6] Running official GSM8K eval -> ${RESULTS_DIR}"
  "${PY_TRAIN_EVAL}" -m minimal_multitask.eval.gsm.run_eval \
    --data_dir data/eval/gsm \
    --save_dir "${RESULTS_DIR}" \
    --model_name_or_path "${MERGED_DIR}" \
    --n_shot 8 \
    --use_chat_format \
    --chat_formatting_function minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format \
    > "${LOG_DIR}/${RUN_TAG}_${METHOD}_${TASK}_eval.log" 2>&1
  if [[ -f "${RESULTS_DIR}/predictions.jsonl" ]]; then
    "${PY_TRAIN_EVAL}" "${DATELM_ROOT}/reextract_gsm.py" "${RESULTS_DIR}/predictions.jsonl" \
      > "${LOG_DIR}/${RUN_TAG}_${METHOD}_${TASK}_reextract.log" 2>&1 || true
  fi
else
  echo "[6/6] Running official BBH eval -> ${RESULTS_DIR}"
  "${PY_TRAIN_EVAL}" -m minimal_multitask.eval.bbh.run_eval \
    --data_dir data/eval/bbh \
    --save_dir "${RESULTS_DIR}" \
    --model_name_or_path "${MERGED_DIR}" \
    --eval_batch_size 4 \
    --use_chat_format \
    --chat_formatting_function minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format \
    > "${LOG_DIR}/${RUN_TAG}_${METHOD}_${TASK}_eval.log" 2>&1
fi

if [[ -f "${RESULTS_DIR}/metrics.json" ]]; then
  echo "Done: ${RESULTS_DIR}/metrics.json"
  cat "${RESULTS_DIR}/metrics.json"
else
  echo "Done, but metrics.json missing: ${RESULTS_DIR}"
fi
