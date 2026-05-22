#!/bin/bash
# RepSim v2 pipeline - WITHOUT L2 normalization (matching DATE-LM original)
#
# Key difference from v1:
#   - Uses --no_normalize flag for embeddings (dot product instead of cosine)
#   - This should match DATE-LM's original implementation
#
# Usage:
#   bash run_paper_repsim_v2.sh <mmlu|gsm8k|bbh>

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

# v2 uses different embedding directory (no normalize)
METHOD="repsim_v2"
EMB_TRAIN_DIR="${DATELM_ROOT}/embeddings/${RUN_TAG}/tulu3_train_llama_lasttok_nonorm"
EMB_REF_DIR="${DATELM_ROOT}/embeddings/${RUN_TAG}/${TASK}_ref_llama_lasttok_nonorm"

SELECT_DIR="${DATELM_ROOT}/selected_data/${RUN_TAG}/${TASK}"
SCORES_DIR="${DATELM_ROOT}/scores/${RUN_TAG}/${TASK}"

METRICS_NPY="${SCORES_DIR}/${METHOD}_metrics.npy"
TRAIN_JSONL="${SELECT_DIR}/${METHOD}_10k.jsonl"

ADAPTER_DIR="${DATELM_ROOT}/checkpoints/${RUN_TAG}_${METHOD}_${TASK}_lora"
MERGED_DIR="${DATELM_ROOT}/checkpoints/${RUN_TAG}_${METHOD}_${TASK}_merged"
RESULTS_DIR="${DATELM_ROOT}/results/${RUN_TAG}_${METHOD}_${TASK}_official"
LOG_DIR="${DATELM_ROOT}/logs"

mkdir -p "${LOG_DIR}" "${SELECT_DIR}" "${SCORES_DIR}" "${EMB_TRAIN_DIR}" "${EMB_REF_DIR}"

echo "=============================================="
echo "RepSim v2 Pipeline (NO NORMALIZE): ${TASK}"
echo "=============================================="

# 1) Prepare 200k pool if missing
if [[ ! -f "${POOL_TRAIN_JSONL}" ]]; then
  echo "[1/7] Preparing 200k pool -> ${POOL_TRAIN_JSONL}"
  "${PY_TRAIN_EVAL}" "${PROJ_ROOT}/finetuning/scripts/datelm_paper/prepare_tulu3_pool.py" \
    --input_jsonl "${TULU3_JSONL}" \
    --out_train_jsonl "${POOL_TRAIN_JSONL}" \
    --out_val_jsonl "${POOL_VAL_JSONL}" \
    --seed "${SEED_POOL}" \
    --pool_size 200000 \
    --val_split 0.1
else
  echo "[1/7] Pool exists: ${POOL_TRAIN_JSONL}"
fi

# 2) Train embeddings (LLM hidden states; NO NORMALIZE - key difference!)
if [[ ! -f "${EMB_TRAIN_DIR}/train_emb.npy" ]] || ! grep -q '"train_complete": true' "${EMB_TRAIN_DIR}/progress.json" 2>/dev/null; then
  echo "[2/7] Computing LLM train embeddings (NO NORMALIZE) -> ${EMB_TRAIN_DIR}/train_emb.npy"
  PYTHONUNBUFFERED=1 "${PY_TRAIN_EVAL}" "${PROJ_ROOT}/finetuning/scripts/datelm_paper/compute_llm_last_token_embeddings.py" \
    --datelm_root "${DATELM_ROOT}" \
    --model_name "${BASE_MODEL}" \
    --train_jsonl "${POOL_TRAIN_JSONL}" \
    --messages_only_first_two \
    --no_ref \
    --no_normalize \
    --out_dir "${EMB_TRAIN_DIR}" \
    > "${LOG_DIR}/${RUN_TAG}_${METHOD}_train_emb.log" 2>&1
else
  echo "[2/7] Train embeddings exist: ${EMB_TRAIN_DIR}/train_emb.npy"
fi

# 3) Ref embeddings (per task; NO NORMALIZE)
if [[ ! -f "${EMB_REF_DIR}/ref_emb.npy" ]]; then
  echo "[3/7] Computing ${TASK} ref embeddings (NO NORMALIZE) -> ${EMB_REF_DIR}/ref_emb.npy"
  PYTHONUNBUFFERED=1 "${PY_TRAIN_EVAL}" "${PROJ_ROOT}/finetuning/scripts/datelm_paper/compute_llm_last_token_embeddings.py" \
    --datelm_root "${DATELM_ROOT}" \
    --model_name "${BASE_MODEL}" \
    --no_train \
    --ref_dataset_name "${TASK}" \
    --ref_num_samples 100 \
    --ref_seed 42 \
    --no_normalize \
    --out_dir "${EMB_REF_DIR}" \
    > "${LOG_DIR}/${RUN_TAG}_${METHOD}_${TASK}_ref_emb.log" 2>&1
else
  echo "[3/7] Ref embeddings exist: ${EMB_REF_DIR}/ref_emb.npy"
fi

# 4) RepSim scores (dot product, no normalize)
if [[ ! -f "${METRICS_NPY}" ]]; then
  echo "[4/7] Computing RepSim v2 scores (dot product) -> ${METRICS_NPY}"
  PYTHONUNBUFFERED=1 "${PY_TRAIN_EVAL}" - <<PY > "${LOG_DIR}/${RUN_TAG}_${METHOD}_${TASK}_score.log" 2>&1
import numpy as np
import torch
from pathlib import Path

train_path = Path(r"${EMB_TRAIN_DIR}/train_emb.npy")
ref_path = Path(r"${EMB_REF_DIR}/ref_emb.npy")
out_path = Path(r"${METRICS_NPY}")
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
    # DOT PRODUCT (not cosine similarity) - matching DATE-LM original
    sim = train_t @ ref_t.T  # [N, M]
    scores = sim.mean(dim=1).to(dtype=torch.float32).cpu().numpy()

np.save(out_path, scores.astype(np.float32))
print('saved', out_path, scores.shape, scores.dtype)
print('score stats: mean={:.4f}, std={:.4f}, min={:.4f}, max={:.4f}'.format(
    scores.mean(), scores.std(), scores.min(), scores.max()))
PY
else
  echo "[4/7] Metrics exist: ${METRICS_NPY}"
fi

# 5) Selection (top-k)
if [[ ! -f "${TRAIN_JSONL}" ]]; then
  echo "[5/7] Selecting top-10k -> ${TRAIN_JSONL}"
  "${PY_TRAIN_EVAL}" "${PROJ_ROOT}/finetuning/scripts/datelm_paper/select_jsonl_topk.py" \
    --pool_jsonl "${POOL_TRAIN_JSONL}" \
    --metrics_npy "${METRICS_NPY}" \
    --out_jsonl "${TRAIN_JSONL}" \
    --k 10000 \
    --out_indices_npy "${SELECT_DIR}/${METHOD}_indices.npy" \
    > "${LOG_DIR}/${RUN_TAG}_${METHOD}_${TASK}_select.log" 2>&1
else
  echo "[5/7] Selected JSONL exists: ${TRAIN_JSONL}"
fi

# 6) LoRA train
if [[ ! -f "${ADAPTER_DIR}/adapter_config.json" ]]; then
  echo "[6/7] Training LoRA -> ${ADAPTER_DIR}"
  PYTHONUNBUFFERED=1 "${PY_TRAIN_EVAL}" "${PROJ_ROOT}/finetuning/scripts/datelm_paper/train_lora_hf_paper.py" \
    --datelm_root "${DATELM_ROOT}" \
    --train_jsonl "${TRAIN_JSONL}" \
    --output_dir "${ADAPTER_DIR}" \
    --model_name "${BASE_MODEL}" \
    > "${LOG_DIR}/${RUN_TAG}_${METHOD}_${TASK}_finetune.log" 2>&1
else
  echo "[6/7] Adapter exists: ${ADAPTER_DIR}"
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
  "${PY_TRAIN_EVAL}" -m minimal_multitask.eval.gsm.run_eval \
    --data_dir data/eval/gsm \
    --save_dir "${RESULTS_DIR}" \
    --model_name_or_path "${MERGED_DIR}" \
    --n_shot 8 \
    --use_chat_format \
    --chat_formatting_function minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format \
    > "${LOG_DIR}/${RUN_TAG}_${METHOD}_${TASK}_eval.log" 2>&1
else
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
