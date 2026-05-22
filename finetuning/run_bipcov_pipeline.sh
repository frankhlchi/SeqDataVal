#!/bin/bash
# End-to-end pipeline for Bipartite Greedy Coverage on DATE-LM
#
# This script:
# 1. Computes embeddings for train and ref data
# 2. Runs bipcov to produce scores
# 3. Outputs DATE-LM compatible metrics.npy
#
# Usage:
#   bash run_bipcov_pipeline.sh [VAL_TASK] [TRAIN_SIZE] [REF_SIZE]
#
# Example:
#   bash run_bipcov_pipeline.sh mmlu 200000 100

# NOTE (2025-12-19): This script is deprecated for paper-aligned runs.
# Use `finetuning/scripts/datelm_paper/run_paper_mmlu.sh` instead.

set -e

VAL_TASK=${1:-mmlu}
TRAIN_SIZE=${2:-200000}
REF_SIZE=${3:-100}

# ============ Configuration ============
# Adjust these paths for your setup

# Where to find training data (Tulu3)
# Option 1: local JSONL, set TRAIN_JSONL=/path/to/tulu_3_v3.9_unfiltered.jsonl
TRAIN_JSONL="${TRAIN_JSONL:-}"
# Option 2: HuggingFace dataset fallback
TRAIN_HF="${TRAIN_HF:-allenai/tulu-3-sft-mixture}"

# Reference datasets by task
declare -A REF_HF_MAP
REF_HF_MAP["mmlu"]="cais/mmlu:all:test"
REF_HF_MAP["gsm8k"]="gsm8k:main:test"
REF_HF_MAP["bbh"]="lukaemon/bbh::test"

# Output directories
BASE_DIR="${SEQDATAVAL_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)}"
EMB_DIR="${BASE_DIR}/embeddings/${VAL_TASK}"
SCORES_DIR="${BASE_DIR}/scores"

# Model
EMB_MODEL="BAAI/bge-large-en-v1.5"
DEVICE="cuda"
BATCH_SIZE=128

# ============ Step 1: Compute Embeddings ============
echo "================================================"
echo "Step 1: Computing embeddings for ${VAL_TASK}"
echo "================================================"

mkdir -p ${EMB_DIR}

# Parse ref dataset spec
IFS=':' read -r REF_HF REF_SUBSET REF_SPLIT <<< "${REF_HF_MAP[$VAL_TASK]}"
if [ -z "$REF_SUBSET" ]; then REF_SUBSET=""; fi
if [ -z "$REF_SPLIT" ]; then REF_SPLIT="test"; fi

echo "Train: ${TRAIN_JSONL:-$TRAIN_HF} (max ${TRAIN_SIZE})"
echo "Ref: ${REF_HF} subset=${REF_SUBSET} split=${REF_SPLIT} (max ${REF_SIZE})"

# Build command
CMD="python ${BASE_DIR}/finetuning/compute_embeddings_for_datelm.py"

if [ -n "$TRAIN_JSONL" ] && [ -f "$TRAIN_JSONL" ]; then
    CMD="${CMD} --train_jsonl ${TRAIN_JSONL}"
else
    CMD="${CMD} --train_hf ${TRAIN_HF:-allenai/tulu-3-sft-mixture}"
fi

CMD="${CMD} --train_max_samples ${TRAIN_SIZE}"
CMD="${CMD} --ref_hf ${REF_HF}"
if [ -n "$REF_SUBSET" ]; then
    CMD="${CMD} --ref_hf_subset ${REF_SUBSET}"
fi
CMD="${CMD} --ref_hf_split ${REF_SPLIT}"
CMD="${CMD} --ref_max_samples ${REF_SIZE}"
CMD="${CMD} --out_dir ${EMB_DIR}"
CMD="${CMD} --model ${EMB_MODEL}"
CMD="${CMD} --device ${DEVICE}"
CMD="${CMD} --batch_size ${BATCH_SIZE}"

echo "Running: ${CMD}"
eval ${CMD}

# ============ Step 2: Run BipCov Selection ============
echo ""
echo "================================================"
echo "Step 2: Running Bipartite Greedy Coverage"
echo "================================================"

mkdir -p ${SCORES_DIR}

python ${BASE_DIR}/finetuning/bipcov/probe_bipcov_from_emb.py \
    --train_emb ${EMB_DIR}/train_emb.npy \
    --ref_emb ${EMB_DIR}/ref_emb.npy \
    --out ${SCORES_DIR}/${VAL_TASK}_bipcov \
    --k_max 10000 \
    --top_l 200 \
    --device ${DEVICE} \
    --batch_rows 50000 \
    --save_selected ${SCORES_DIR}/${VAL_TASK}_bipcov_selected.json

echo ""
echo "================================================"
echo "Done! Outputs:"
echo "  Embeddings: ${EMB_DIR}/"
echo "  Scores: ${SCORES_DIR}/${VAL_TASK}_bipcov/metrics.npy"
echo "  Selected: ${SCORES_DIR}/${VAL_TASK}_bipcov_selected.json"
echo "================================================"
echo ""
echo "To use in DATE-LM fine-tuning:"
echo "  python train/finetune.py ... --metric_path ${SCORES_DIR}/${VAL_TASK}_bipcov/metrics.npy"
