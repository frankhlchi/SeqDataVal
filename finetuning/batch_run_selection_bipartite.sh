#!/bin/bash
# Batch run bipartite greedy coverage for all validation datasets
#
# Usage:
#   bash methods/batch_run_selection_bipartite.sh

VAL_DATASETS=("mmlu" "gsm8k" "bbh")
SUBSET_SIZE=200000
VAL_SAMPLES=100

for VAL_DATASET in "${VAL_DATASETS[@]}"; do
    echo "=== Running bipartite_greedy for ${VAL_DATASET} ==="
    sbatch methods/bipartite_greedy/probe_bipartite_greedy_instruct.sh ${VAL_DATASET} ${SUBSET_SIZE} ${VAL_SAMPLES}
done

echo "Submitted all jobs. Check queue with: squeue -u \$USER"
