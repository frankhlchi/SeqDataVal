#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Smoke test: digits, seed=10 (small counts)"
python src/main.py --dataset digits --seed 10 --train_count 30 --valid_count 30 --test_count 100 --skip_bipartite

echo "Smoke test: digits + Bipartite"
python src/main.py --dataset digits --seed 10 --train_count 30 --valid_count 30 --test_count 100

echo "Smoke test: DP (tiny)"
python src/main.py --dataset digits --seed 10 --train_count 10 --valid_count 30 --test_count 100 --skip_bipartite --include_dp --dp_max_subset_size 10

echo "OK"

