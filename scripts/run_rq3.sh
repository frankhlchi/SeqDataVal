#!/usr/bin/env bash
set -euo pipefail

# Paper RQ3: baselines + Bipartite (default)
#
# Standard datasets use config/base_config.yaml (50/50/500).
# digits & bbc-embeddings use config/rq3_large_datasets.yaml (100/100/1000).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

datasets=("2dplanes" "nomao" "bbc-embeddings" "MiniBooNE" "digits" "election" "electricity" "fried")
seeds=(10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200)
max_parallel=8

mkdir -p logs

run_one() {
  local seed="$1"
  local dataset="$2"
  local cfg="config/base_config.yaml"
  if [[ "$dataset" == "digits" || "$dataset" == "bbc-embeddings" ]]; then
    cfg="config/rq3_large_datasets.yaml"
  fi
  local log_file="logs/${dataset}_seed_${seed}.log"
  echo "Running RQ3: dataset=$dataset seed=$seed cfg=$cfg"
  python src/main.py --dataset "$dataset" --seed "$seed" --config "$cfg" >"$log_file" 2>&1
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

  echo "Aggregating plots: $dataset"
  python src/utils/plotting.py --dataset "$dataset"
done

echo "Done."

