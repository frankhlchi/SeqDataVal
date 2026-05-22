#!/bin/bash
#SBATCH --job-name=bipartite_greedy
#SBATCH --output=logs/bipartite_greedy_%j.out
#SBATCH --error=logs/bipartite_greedy_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Bipartite Greedy Coverage for DATE-LM fine-tuning data selection
#
# Usage:
#   bash probe_bipartite_greedy_instruct.sh [VAL_DATASET] [SUBSET_SIZE] [VAL_SAMPLES]
#
# Example:
#   bash probe_bipartite_greedy_instruct.sh mmlu 200000 100

VAL_DATASET=${1:-mmlu}
SUBSET_SIZE=${2:-200000}
VAL_SAMPLES=${3:-100}

# Paths (adjust as needed)
BASE_DIR="/data/group_data/cx_group"
TRAIN_DATA="${BASE_DIR}/data/tulu_3_v3.9_unfiltered.jsonl"
OUT_DIR="${BASE_DIR}/scores/bipartite_greedy/${VAL_DATASET}"

# Embedding model (alternatives: BAAI/bge-base-en-v1.5, sentence-transformers/all-MiniLM-L6-v2)
EMBEDDING_MODEL="BAAI/bge-large-en-v1.5"

mkdir -p ${OUT_DIR}
mkdir -p logs

echo "Running Bipartite Greedy Coverage"
echo "  Val dataset: ${VAL_DATASET}"
echo "  Subset size: ${SUBSET_SIZE}"
echo "  Val samples: ${VAL_SAMPLES}"
echo "  Output: ${OUT_DIR}"

python methods/bipartite_greedy/probe_bipartite_greedy_instruct.py \
    --train_data_dir "${TRAIN_DATA}" \
    --val_dataset_name "${VAL_DATASET}" \
    --val_num_samples ${VAL_SAMPLES} \
    --subset_size ${SUBSET_SIZE} \
    --out_dir "${OUT_DIR}/metrics.npy" \
    --embedding_model "${EMBEDDING_MODEL}" \
    --learn_threshold \
    --device cuda \
    --batch_size 128 \
    --seed 42

echo "Done! Metrics saved to ${OUT_DIR}/metrics.npy"
