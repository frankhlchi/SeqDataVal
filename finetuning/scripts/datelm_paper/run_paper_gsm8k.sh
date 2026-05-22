#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

METHOD=${1:-}
if [[ -z "${METHOD}" ]]; then
  echo "Usage: $0 <random|bipcov>"
  exit 1
fi
if [[ "${METHOD}" != "random" && "${METHOD}" != "bipcov" ]]; then
  echo "Unknown METHOD: ${METHOD} (expected random|bipcov)"
  exit 1
fi

PROJ_ROOT="${SEQDATAVAL_ROOT}"

RUN_TAG="${RUN_TAG:-paper_seed42_v1}"
SEED_POOL=42
SEED_RANDOM=42

TULU3_JSONL="${DATELM_ROOT}/data/training_data/tulu_3_v3.9_unfiltered.jsonl"
POOL_TRAIN_JSONL="${DATELM_ROOT}/data/training_data/${RUN_TAG}_tulu3_200k_train.jsonl"
POOL_VAL_JSONL="${DATELM_ROOT}/data/training_data/${RUN_TAG}_tulu3_22k_val.jsonl"

SELECT_DIR="${DATELM_ROOT}/selected_data/${RUN_TAG}/gsm8k"
SCORES_DIR="${DATELM_ROOT}/scores/${RUN_TAG}/gsm8k"
EMB_TRAIN_DIR="${DATELM_ROOT}/embeddings/${RUN_TAG}/tulu3_train_bge"
EMB_REF_DIR="${DATELM_ROOT}/embeddings/${RUN_TAG}/gsm8k_ref_bge"

ADAPTER_DIR="${DATELM_ROOT}/checkpoints/${RUN_TAG}_${METHOD}_gsm8k_lora"
MERGED_DIR="${DATELM_ROOT}/checkpoints/${RUN_TAG}_${METHOD}_gsm8k_merged"
RESULTS_DIR="${DATELM_ROOT}/results/${RUN_TAG}_${METHOD}_gsm8k_official"
LOG_DIR="${DATELM_ROOT}/logs"

mkdir -p "${LOG_DIR}"

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

# 2) Build selection JSONL
mkdir -p "${SELECT_DIR}"

if [[ "${METHOD}" == "random" ]]; then
  echo "[2/6] Selecting random 10k -> ${SELECT_DIR}/random_10k.jsonl"
  "${PY_TRAIN_EVAL}" "${PROJ_ROOT}/finetuning/scripts/datelm_paper/select_jsonl_topk.py" \
    --pool_jsonl "${POOL_TRAIN_JSONL}" \
    --out_jsonl "${SELECT_DIR}/random_10k.jsonl" \
    --k 10000 \
    --seed "${SEED_RANDOM}" \
    --out_indices_npy "${SELECT_DIR}/random_indices.npy"
  TRAIN_JSONL="${SELECT_DIR}/random_10k.jsonl"
else
  echo "[2/6] Preparing bipcov metrics"
  mkdir -p "${SCORES_DIR}"
  mkdir -p "${EMB_TRAIN_DIR}" "${EMB_REF_DIR}"

  if [[ ! -f "${EMB_TRAIN_DIR}/train_emb.npy" ]]; then
    echo "  - Computing train embeddings -> ${EMB_TRAIN_DIR}/train_emb.npy"
    "${PY_EMB}" "${PROJ_ROOT}/finetuning/compute_embeddings_for_datelm.py" \
      --train_jsonl "${POOL_TRAIN_JSONL}" \
      --train_max_samples 200000 \
      --out_dir "${EMB_TRAIN_DIR}" \
      --no_ref \
      --messages_only_first_two \
      --model "BAAI/bge-large-en-v1.5" \
      --device cuda \
      --batch_size 128 \
      --max_length 512
  else
    echo "  - Train embeddings exist: ${EMB_TRAIN_DIR}/train_emb.npy"
  fi

  if [[ ! -f "${EMB_REF_DIR}/ref_emb.npy" ]]; then
    # Align BipCov ref with DATE-LM baselines: use DATASETS[gsm8k].get_all_test_prompts(prompt_only=False).
    REF_JSONL="${DATELM_ROOT}/data/eval/gsm/${RUN_TAG}_gsm8k_ref_100_promptlabel.jsonl"
    if [[ ! -f "${REF_JSONL}" ]]; then
      echo "  - Preparing GSM8K ref JSONL (DATASETS; prompt+label) -> ${REF_JSONL}"
      "${PY_TRAIN_EVAL}" "${PROJ_ROOT}/finetuning/scripts/datelm_paper/prepare_ref_jsonl_from_datasets.py" \
        --datelm_root "${DATELM_ROOT}" \
        --task gsm8k \
        --out_jsonl "${REF_JSONL}" \
        --num_samples 100 \
        --seed 42 \
        > "${LOG_DIR}/${RUN_TAG}_bipcov_gsm8k_prepare_ref.log" 2>&1
    else
      echo "  - Ref JSONL exists: ${REF_JSONL}"
    fi

    echo "  - Computing GSM8K ref embeddings -> ${EMB_REF_DIR}/ref_emb.npy"
    "${PY_EMB}" "${PROJ_ROOT}/finetuning/compute_embeddings_for_datelm.py" \
      --ref_jsonl "${REF_JSONL}" \
      --ref_max_samples 100 \
      --out_dir "${EMB_REF_DIR}" \
      --no_train \
      --model "BAAI/bge-large-en-v1.5" \
      --device cuda \
      --batch_size 128 \
      --max_length 512
  else
    echo "  - Ref embeddings exist: ${EMB_REF_DIR}/ref_emb.npy"
  fi

  METRICS_NPY="${SCORES_DIR}/bipcov_metrics.npy"
  if [[ ! -f "${METRICS_NPY}" ]]; then
    echo "  - Running bipcov -> ${METRICS_NPY}"
    "${PY_EMB}" "${DATELM_ROOT}/methods/bipcov/probe_bipcov_from_emb.py" \
      --train_emb "${EMB_TRAIN_DIR}/train_emb.npy" \
      --ref_emb "${EMB_REF_DIR}/ref_emb.npy" \
      --out "${METRICS_NPY}" \
      --k_max 10000 \
      --top_l 200 \
      --device cuda \
      --batch_rows 50000 \
      --save_selected "${SCORES_DIR}/bipcov_selected_indices.json"
  else
    echo "  - Metrics exist: ${METRICS_NPY}"
  fi

  echo "  - Selecting bipcov 10k -> ${SELECT_DIR}/bipcov_10k.jsonl"
  "${PY_TRAIN_EVAL}" "${PROJ_ROOT}/finetuning/scripts/datelm_paper/select_jsonl_topk.py" \
    --pool_jsonl "${POOL_TRAIN_JSONL}" \
    --metrics_npy "${METRICS_NPY}" \
    --out_jsonl "${SELECT_DIR}/bipcov_10k.jsonl" \
    --k 10000 \
    --out_indices_npy "${SELECT_DIR}/bipcov_indices.npy"
  TRAIN_JSONL="${SELECT_DIR}/bipcov_10k.jsonl"
fi

# 3) LoRA train
if [[ ! -f "${ADAPTER_DIR}/adapter_config.json" ]]; then
  echo "[3/6] Training LoRA -> ${ADAPTER_DIR}"
  PYTHONUNBUFFERED=1 "${PY_TRAIN_EVAL}" "${PROJ_ROOT}/finetuning/scripts/datelm_paper/train_lora_hf_paper.py" \
    --datelm_root "${DATELM_ROOT}" \
    --train_jsonl "${TRAIN_JSONL}" \
    --output_dir "${ADAPTER_DIR}" \
    --model_name "${BASE_MODEL}" \
    > "${LOG_DIR}/${RUN_TAG}_${METHOD}_gsm8k_finetune.log" 2>&1
else
  echo "[3/6] Adapter exists: ${ADAPTER_DIR}"
fi

# 4) Merge
if [[ ! -f "${MERGED_DIR}/config.json" ]]; then
  echo "[4/6] Merging -> ${MERGED_DIR}"
  "${PY_TRAIN_EVAL}" "${PROJ_ROOT}/finetuning/merge_lora_peft.py" \
    --adapter_path "${ADAPTER_DIR}" \
    --base_model "${BASE_MODEL}" \
    --output_path "${MERGED_DIR}"
else
  echo "[4/6] Merged model exists: ${MERGED_DIR}"
fi

# 5) Eval (GSM8K)
echo "[5/6] Running official GSM8K eval -> ${RESULTS_DIR}"
cd "${DATELM_ROOT}"
"${PY_TRAIN_EVAL}" -m minimal_multitask.eval.gsm.run_eval \
  --data_dir data/eval/gsm \
  --save_dir "${RESULTS_DIR}" \
  --model_name_or_path "${MERGED_DIR}" \
  --n_shot 8 \
  --use_chat_format \
  --chat_formatting_function minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format \
  > "${LOG_DIR}/${RUN_TAG}_${METHOD}_gsm8k_eval.log" 2>&1

# Optional diagnostic re-extraction (does not change official outputs)
if [[ -f "${RESULTS_DIR}/predictions.jsonl" ]]; then
  "${PY_TRAIN_EVAL}" "${DATELM_ROOT}/reextract_gsm.py" "${RESULTS_DIR}/predictions.jsonl" \
    > "${LOG_DIR}/${RUN_TAG}_${METHOD}_gsm8k_reextract.log" 2>&1 || true
fi

# 6) Report
if [[ -f "${RESULTS_DIR}/metrics.json" ]]; then
  echo "[6/6] Done: ${RESULTS_DIR}/metrics.json"
  cat "${RESULTS_DIR}/metrics.json"
  if [[ -f "${LOG_DIR}/${RUN_TAG}_${METHOD}_gsm8k_reextract.log" ]]; then
    echo "[diagnostic] tail: ${LOG_DIR}/${RUN_TAG}_${METHOD}_gsm8k_reextract.log"
    tail -n 5 "${LOG_DIR}/${RUN_TAG}_${METHOD}_gsm8k_reextract.log" || true
  fi
else
  echo "[6/6] Done, but metrics.json missing: ${RESULTS_DIR}"
fi
