#!/usr/bin/env bash
set -euo pipefail

# Paper RQ1: DP optimality verification vs game-theoretic baselines.
#
# Warning: exact DP can be extremely slow even for train_count=20.
# Consider running a single dataset/seed first, or reducing dp_max_subset_size.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

datasets=("2dplanes" "nomao" "bbc-embeddings" "MiniBooNE" "digits" "election" "electricity" "fried")
seeds=(10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200)
max_parallel=1
results_root="results_rq1_dp"

# Avoid BLAS/OpenMP oversubscription.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Avoid OpenML/sklearn download cache races across multi-node (shared /home).
# Use a per-host cache directory by default.
HOST="$(hostname)"
export SCIKIT_LEARN_DATA="${SCIKIT_LEARN_DATA:-$HOME/.scikit_learn_data_${HOST}}"
mkdir -p "$SCIKIT_LEARN_DATA"

mkdir -p logs

run_one() {
  local seed="$1"
  local dataset="$2"
  local log_file="logs/rq1_${dataset}_seed_${seed}.log"
  echo "Running RQ1-DP: dataset=$dataset seed=$seed"
  conda run -n pydvl --no-capture-output python -u src/main.py \
    --dataset "$dataset" \
    --seed "$seed" \
    --config "config/rq1_config.yaml" \
    --results_root "$results_root" \
    --skip_bipartite \
    --include_dp \
    --dp_max_subset_size 20 \
    >"$log_file" 2>&1
}

export -f run_one

for dataset in "${datasets[@]}"; do
  echo "=== Dataset: $dataset ==="
  tmp="$(mktemp)"
  for seed in "${seeds[@]}"; do
    echo "$seed $dataset" >>"$tmp"
  done
  cat "$tmp" | xargs -P "$max_parallel" -n 2 bash -c 'run_one "$@"' _
  rm "$tmp"
done

echo "Done."
