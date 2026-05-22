#!/bin/bash
# BipCov pipeline with configurable embedding model
#
# Supports different SentenceTransformer embeddings:
#   - bge: BAAI/bge-large-en-v1.5 (1024-dim) - current best
#   - e5: intfloat/e5-large-v2 (1024-dim) - alternative RAG model
#   - minilm: sentence-transformers/all-MiniLM-L6-v2 (384-dim) - fast/small
#
# Usage:
#   bash run_paper_bipcov_emb.sh <mmlu|gsm8k|bbh> <bge|e5|minilm>

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

TASK=${1:-}
EMB_TYPE=${2:-bge}

if [[ -z "${TASK}" ]]; then
  echo "Usage: $0 <mmlu|gsm8k|bbh> <bge|e5|minilm>"
  exit 1
fi
if [[ "${TASK}" != "mmlu" && "${TASK}" != "gsm8k" && "${TASK}" != "bbh" ]]; then
  echo "Unknown TASK: ${TASK} (expected mmlu|gsm8k|bbh)"
  exit 1
fi

# Map embedding type to model name
case "${EMB_TYPE}" in
  bge)
    EMB_MODEL="BAAI/bge-large-en-v1.5"
    EMB_DIM=1024
    QUERY_PREFIX="Represent this question for retrieving relevant training data: "
    ;;
  e5)
    EMB_MODEL="intfloat/e5-large-v2"
    EMB_DIM=1024
    QUERY_PREFIX="query: "
    ;;
  minilm)
    EMB_MODEL="sentence-transformers/all-MiniLM-L6-v2"
    EMB_DIM=384
    QUERY_PREFIX=""
    ;;
  *)
    echo "Unknown EMB_TYPE: ${EMB_TYPE} (expected bge|e5|minilm)"
    exit 1
    ;;
esac

PROJ_ROOT="${SEQDATAVAL_ROOT}"

RUN_TAG="${RUN_TAG:-paper_seed42_v1}"
SEED_POOL=42

TULU3_JSONL="${DATELM_ROOT}/data/training_data/tulu_3_v3.9_unfiltered.jsonl"
POOL_TRAIN_JSONL="${DATELM_ROOT}/data/training_data/${RUN_TAG}_tulu3_200k_train.jsonl"
POOL_VAL_JSONL="${DATELM_ROOT}/data/training_data/${RUN_TAG}_tulu3_22k_val.jsonl"

# Method name includes embedding type
METHOD="bipcov_${EMB_TYPE}"

SELECT_DIR="${DATELM_ROOT}/selected_data/${RUN_TAG}/${TASK}"
SCORES_DIR="${DATELM_ROOT}/scores/${RUN_TAG}/${TASK}"
EMB_TRAIN_DIR="${DATELM_ROOT}/embeddings/${RUN_TAG}/tulu3_train_${EMB_TYPE}"
EMB_REF_DIR="${DATELM_ROOT}/embeddings/${RUN_TAG}/${TASK}_ref_${EMB_TYPE}"

METRICS_NPY="${SCORES_DIR}/${METHOD}_metrics.npy"
TRAIN_JSONL="${SELECT_DIR}/${METHOD}_10k.jsonl"

ADAPTER_DIR="${DATELM_ROOT}/checkpoints/${RUN_TAG}_${METHOD}_${TASK}_lora"
MERGED_DIR="${DATELM_ROOT}/checkpoints/${RUN_TAG}_${METHOD}_${TASK}_merged"
RESULTS_DIR="${DATELM_ROOT}/results/${RUN_TAG}_${METHOD}_${TASK}_official"
LOG_DIR="${DATELM_ROOT}/logs"

mkdir -p "${LOG_DIR}" "${SELECT_DIR}" "${SCORES_DIR}" "${EMB_TRAIN_DIR}" "${EMB_REF_DIR}"

echo "=============================================="
echo "BipCov Pipeline: ${TASK} with ${EMB_TYPE}"
echo "  Model: ${EMB_MODEL}"
echo "  Dim: ${EMB_DIM}"
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

# 2) Generate ref set JSONL (if needed)
EVAL_SUBDIR="${TASK}"
if [[ "${TASK}" == "gsm8k" ]]; then
  EVAL_SUBDIR="gsm"
fi
REF_JSONL="${DATELM_ROOT}/data/eval/${EVAL_SUBDIR}/${RUN_TAG}_${TASK}_ref_100_promptonly.jsonl"
if [[ ! -f "${REF_JSONL}" ]]; then
  echo "[2/7] Generating ref JSONL -> ${REF_JSONL}"
  "${PY_TRAIN_EVAL}" - <<PY
import json
import sys
sys.path.insert(0, "${DATELM_ROOT}")
from transformers import AutoTokenizer
from minimal_multitask.data import DATASETS

tokenizer = AutoTokenizer.from_pretrained("${BASE_MODEL}")
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token

ds = DATASETS["${TASK}"](tokenizer).get_all_test_prompts(
    num_samples=100, seed=42, prompt_only=True, response_only=False
)

import pathlib
out_path = pathlib.Path("${REF_JSONL}")
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    for item in ds:
        # Extract text from input_ids
        text = tokenizer.decode(item["input_ids"], skip_special_tokens=True)
        f.write(json.dumps({"text": text}) + "\n")
print(f"Saved {len(ds)} ref examples to {out_path}")
PY
else
  echo "[2/7] Ref JSONL exists: ${REF_JSONL}"
fi

# 3) Compute embeddings
if [[ ! -f "${EMB_TRAIN_DIR}/train_emb.npy" ]] || [[ ! -f "${EMB_REF_DIR}/ref_emb.npy" ]]; then
  echo "[3/7] Computing ${EMB_TYPE} embeddings"

  # Train embeddings
  if [[ ! -f "${EMB_TRAIN_DIR}/train_emb.npy" ]]; then
    echo "  -> Train embeddings: ${EMB_TRAIN_DIR}"
    "${PY_EMB}" - <<PY > "${LOG_DIR}/${RUN_TAG}_${METHOD}_train_emb.log" 2>&1
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import json
from pathlib import Path

model = SentenceTransformer("${EMB_MODEL}", device="cuda")
train_jsonl = Path("${POOL_TRAIN_JSONL}")
out_dir = Path("${EMB_TRAIN_DIR}")
out_dir.mkdir(parents=True, exist_ok=True)

# Read training data
texts = []
with train_jsonl.open("r") as f:
    for line in tqdm(f, desc="Reading train JSONL"):
        obj = json.loads(line)
        msgs = obj.get("messages", [])
        # Extract user message (instruction)
        for m in msgs:
            if m.get("role") == "user":
                texts.append(m.get("content", ""))
                break
        else:
            texts.append("")

print(f"Encoding {len(texts)} train examples...")
emb = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
emb = np.asarray(emb, dtype=np.float32)
np.save(out_dir / "train_emb.npy", emb)
print(f"Saved: {out_dir / 'train_emb.npy'} shape={emb.shape}")
PY
  fi

  # Ref embeddings
  if [[ ! -f "${EMB_REF_DIR}/ref_emb.npy" ]]; then
    echo "  -> Ref embeddings: ${EMB_REF_DIR}"
    "${PY_EMB}" - <<PY > "${LOG_DIR}/${RUN_TAG}_${METHOD}_${TASK}_ref_emb.log" 2>&1
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path

model = SentenceTransformer("${EMB_MODEL}", device="cuda")
ref_jsonl = Path("${REF_JSONL}")
out_dir = Path("${EMB_REF_DIR}")
out_dir.mkdir(parents=True, exist_ok=True)

# Read ref data
texts = []
with ref_jsonl.open("r") as f:
    for line in f:
        obj = json.loads(line)
        text = obj.get("text", "")
        # Add query prefix for retrieval models
        prefix = "${QUERY_PREFIX}"
        texts.append(prefix + text if prefix else text)

print(f"Encoding {len(texts)} ref examples...")
emb = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
emb = np.asarray(emb, dtype=np.float32)
np.save(out_dir / "ref_emb.npy", emb)
print(f"Saved: {out_dir / 'ref_emb.npy'} shape={emb.shape}")
PY
  fi
else
  echo "[3/7] Embeddings exist"
fi

# 4) BipCov scoring
if [[ ! -f "${METRICS_NPY}" ]]; then
  echo "[4/7] Computing BipCov scores -> ${METRICS_NPY}"
  "${PY_EMB}" "${PROJ_ROOT}/finetuning/bipcov/probe_bipcov_from_emb.py" \
    --train_emb "${EMB_TRAIN_DIR}/train_emb.npy" \
    --ref_emb "${EMB_REF_DIR}/ref_emb.npy" \
    --out "${METRICS_NPY}" \
    --k_max 10000 \
    --target_density 0.02 \
    --device cuda \
    > "${LOG_DIR}/${RUN_TAG}_${METHOD}_${TASK}_score.log" 2>&1
else
  echo "[4/7] Metrics exist: ${METRICS_NPY}"
fi

# 5) Selection
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
    --ntrain 0 --data_dir data/eval/mmlu --save_dir "${RESULTS_DIR}" \
    --model_name_or_path "${MERGED_DIR}" --eval_batch_size 4 \
    --use_chat_format \
    --chat_formatting_function minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format \
    > "${LOG_DIR}/${RUN_TAG}_${METHOD}_${TASK}_eval.log" 2>&1
elif [[ "${TASK}" == "gsm8k" ]]; then
  "${PY_TRAIN_EVAL}" -m minimal_multitask.eval.gsm.run_eval \
    --data_dir data/eval/gsm --save_dir "${RESULTS_DIR}" \
    --model_name_or_path "${MERGED_DIR}" --n_shot 8 \
    --use_chat_format \
    --chat_formatting_function minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format \
    > "${LOG_DIR}/${RUN_TAG}_${METHOD}_${TASK}_eval.log" 2>&1
else
  "${PY_TRAIN_EVAL}" -m minimal_multitask.eval.bbh.run_eval \
    --data_dir data/eval/bbh --save_dir "${RESULTS_DIR}" \
    --model_name_or_path "${MERGED_DIR}" --eval_batch_size 4 \
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
