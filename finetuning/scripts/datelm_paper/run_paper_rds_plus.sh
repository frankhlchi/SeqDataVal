#!/bin/bash
# RDS+ (Rep-Sim with Diversity Sampling) pipeline
#
# This implements DATE-LM's SOTA method:
#   1. Use RepSim scores (LLM hidden state similarity)
#   2. Apply Gumbel-Top-k for diverse selection (instead of plain top-k)
#
# RDS+ adds stochastic diversity to RepSim, which is the key differentiator.
#
# Usage:
#   bash run_paper_rds_plus.sh <mmlu|gsm8k|bbh> [gumbel_temp]
#
# Default gumbel_temp=0.5 (DATE-LM default)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

TASK=${1:-}
GUMBEL_TEMP=${2:-0.5}

if [[ -z "${TASK}" ]]; then
  echo "Usage: $0 <mmlu|gsm8k|bbh> [gumbel_temp]"
  echo "  gumbel_temp: Gumbel temperature (default: 0.5)"
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

# RDS+ uses RepSim scores but with Gumbel-Top-k selection
METHOD="rds_plus"
REPSIM_METHOD="repsim"

SELECT_DIR="${DATELM_ROOT}/selected_data/${RUN_TAG}/${TASK}"
SCORES_DIR="${DATELM_ROOT}/scores/${RUN_TAG}/${TASK}"
EMB_TRAIN_DIR="${DATELM_ROOT}/embeddings/${RUN_TAG}/tulu3_train_llama_lasttok"
EMB_REF_DIR="${DATELM_ROOT}/embeddings/${RUN_TAG}/${TASK}_ref_llama_lasttok"

# RepSim scores (reused from repsim pipeline)
REPSIM_METRICS_NPY="${SCORES_DIR}/${REPSIM_METHOD}_metrics.npy"

# RDS+ outputs
TRAIN_JSONL="${SELECT_DIR}/${METHOD}_10k.jsonl"
ADAPTER_DIR="${DATELM_ROOT}/checkpoints/${RUN_TAG}_${METHOD}_${TASK}_lora"
MERGED_DIR="${DATELM_ROOT}/checkpoints/${RUN_TAG}_${METHOD}_${TASK}_merged"
RESULTS_DIR="${DATELM_ROOT}/results/${RUN_TAG}_${METHOD}_${TASK}_official"
LOG_DIR="${DATELM_ROOT}/logs"

mkdir -p "${LOG_DIR}" "${SELECT_DIR}" "${SCORES_DIR}" "${EMB_TRAIN_DIR}" "${EMB_REF_DIR}"

echo "=============================================="
echo "RDS+ Pipeline: ${TASK}"
echo "  Gumbel temperature: ${GUMBEL_TEMP}"
echo "=============================================="

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

# 2) Train embeddings (LLM hidden states; shared with RepSim)
if [[ ! -f "${EMB_TRAIN_DIR}/train_emb.npy" ]] || ! grep -q '"train_complete": true' "${EMB_TRAIN_DIR}/progress.json" 2>/dev/null; then
  echo "[2/6] Computing LLM train embeddings -> ${EMB_TRAIN_DIR}/train_emb.npy"
  PYTHONUNBUFFERED=1 "${PY_TRAIN_EVAL}" "${PROJ_ROOT}/finetuning/scripts/datelm_paper/compute_llm_last_token_embeddings.py" \
    --datelm_root "${DATELM_ROOT}" \
    --model_name "${BASE_MODEL}" \
    --train_jsonl "${POOL_TRAIN_JSONL}" \
    --messages_only_first_two \
    --no_ref \
    --out_dir "${EMB_TRAIN_DIR}" \
    > "${LOG_DIR}/${RUN_TAG}_${METHOD}_train_emb.log" 2>&1
else
  echo "[2/6] Train embeddings exist: ${EMB_TRAIN_DIR}/train_emb.npy"
fi

# 3) Ref embeddings (per task; shared with RepSim)
if [[ ! -f "${EMB_REF_DIR}/ref_emb.npy" ]]; then
  echo "[3/6] Computing ${TASK} ref embeddings -> ${EMB_REF_DIR}/ref_emb.npy"
  PYTHONUNBUFFERED=1 "${PY_TRAIN_EVAL}" "${PROJ_ROOT}/finetuning/scripts/datelm_paper/compute_llm_last_token_embeddings.py" \
    --datelm_root "${DATELM_ROOT}" \
    --model_name "${BASE_MODEL}" \
    --no_train \
    --ref_dataset_name "${TASK}" \
    --ref_num_samples 100 \
    --ref_seed 42 \
    --out_dir "${EMB_REF_DIR}" \
    > "${LOG_DIR}/${RUN_TAG}_${METHOD}_${TASK}_ref_emb.log" 2>&1
else
  echo "[3/6] Ref embeddings exist: ${EMB_REF_DIR}/ref_emb.npy"
fi

# 4) RepSim scores (reused from repsim pipeline)
if [[ ! -f "${REPSIM_METRICS_NPY}" ]]; then
  echo "[4/6] Computing RepSim scores -> ${REPSIM_METRICS_NPY}"
  PYTHONUNBUFFERED=1 "${PY_TRAIN_EVAL}" - <<PY > "${LOG_DIR}/${RUN_TAG}_${REPSIM_METHOD}_${TASK}_score.log" 2>&1
import numpy as np
import torch
from pathlib import Path

train_path = Path(r"${EMB_TRAIN_DIR}/train_emb.npy")
ref_path = Path(r"${EMB_REF_DIR}/ref_emb.npy")
out_path = Path(r"${REPSIM_METRICS_NPY}")
out_path.parent.mkdir(parents=True, exist_ok=True)

train = np.load(train_path, mmap_mode="r")
ref = np.load(ref_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_t = torch.from_numpy(train)
ref_t = torch.from_numpy(ref)

# Match dtypes for fast matmul
if train_t.dtype in (torch.float16, torch.bfloat16):
    ref_t = ref_t.to(dtype=train_t.dtype)
else:
    train_t = train_t.to(dtype=torch.float16)
    ref_t = ref_t.to(dtype=torch.float16)

train_t = train_t.to(device)
ref_t = ref_t.to(device)

with torch.inference_mode():
    sim = train_t @ ref_t.T  # [N, M]
    scores = sim.mean(dim=1).to(dtype=torch.float32).cpu().numpy()

np.save(out_path, scores.astype(np.float32))
print('saved', out_path, scores.shape, scores.dtype)
PY
else
  echo "[4/6] RepSim metrics exist: ${REPSIM_METRICS_NPY}"
fi

# 5) Gumbel-Top-k Selection (RDS+ key step!)
if [[ ! -f "${TRAIN_JSONL}" ]]; then
  echo "[5/6] Selecting top-10k with Gumbel-Top-k (temp=${GUMBEL_TEMP}) -> ${TRAIN_JSONL}"
  "${PY_TRAIN_EVAL}" "${PROJ_ROOT}/finetuning/scripts/datelm_paper/select_jsonl_gumbel_topk.py" \
    --pool_jsonl "${POOL_TRAIN_JSONL}" \
    --metrics_npy "${REPSIM_METRICS_NPY}" \
    --out_jsonl "${TRAIN_JSONL}" \
    --k 10000 \
    --gumbel_temp "${GUMBEL_TEMP}" \
    --seed 42 \
    --out_indices_npy "${SELECT_DIR}/${METHOD}_indices.npy" \
    > "${LOG_DIR}/${RUN_TAG}_${METHOD}_${TASK}_select.log" 2>&1
else
  echo "[5/6] Selected JSONL exists: ${TRAIN_JSONL}"
fi

# 6) LoRA train
if [[ ! -f "${ADAPTER_DIR}/adapter_config.json" ]]; then
  echo "[6/6] Training LoRA -> ${ADAPTER_DIR}"
  PYTHONUNBUFFERED=1 "${PY_TRAIN_EVAL}" "${PROJ_ROOT}/finetuning/scripts/datelm_paper/train_lora_hf_paper.py" \
    --datelm_root "${DATELM_ROOT}" \
    --train_jsonl "${TRAIN_JSONL}" \
    --output_dir "${ADAPTER_DIR}" \
    --model_name "${BASE_MODEL}" \
    > "${LOG_DIR}/${RUN_TAG}_${METHOD}_${TASK}_finetune.log" 2>&1
else
  echo "[6/6] Adapter exists: ${ADAPTER_DIR}"
fi

# 7) Merge + Eval
if [[ ! -f "${MERGED_DIR}/config.json" ]]; then
  echo "[7/7] Merging -> ${MERGED_DIR}"
  "${PY_TRAIN_EVAL}" "${PROJ_ROOT}/finetuning/merge_lora_peft.py" \
    --adapter_path "${ADAPTER_DIR}" \
    --base_model "${BASE_MODEL}" \
    --output_path "${MERGED_DIR}" \
    > "${LOG_DIR}/${RUN_TAG}_${METHOD}_${TASK}_merge.log" 2>&1
else
  echo "[7/7] Merged model exists: ${MERGED_DIR}"
fi

cd "${DATELM_ROOT}"
if [[ "${TASK}" == "mmlu" ]]; then
  echo "[Eval] Running official MMLU eval -> ${RESULTS_DIR}"
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
  echo "[Eval] Running official GSM8K eval -> ${RESULTS_DIR}"
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
  echo "[Eval] Running official BBH eval -> ${RESULTS_DIR}"
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
  echo "=============================================="
  echo "Done: ${RESULTS_DIR}/metrics.json"
  echo "=============================================="
  cat "${RESULTS_DIR}/metrics.json"
else
  echo "Done, but metrics.json missing: ${RESULTS_DIR}"
fi
