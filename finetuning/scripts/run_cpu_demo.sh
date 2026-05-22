#!/bin/bash
set -e

# Run from repo root.
python src/main_finetune_demo.py --method bipartite --k 6 --learn_threshold
python src/main_finetune_demo.py --method repsim --k 6
python src/main_finetune_demo.py --method random --k 6

echo "\nDone. See results/finetune_demo/*"
