#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
rq1_dp_worker.sh

Worker that pulls (dataset, seed) tasks from a shared TSV and runs RQ1 DP jobs.

Expected task file format (TSV):
  <dataset>\t<seed>

Example:
  bash scripts/rq1_dp_worker.sh --tasks results_rq1_dp/_cluster/tasks.tsv --results_root results_rq1_dp
EOF
}

RESULTS_ROOT="results_rq1_dp"
TASKS=""
STATE_DIR=""
THREADS=1
WORKER_ID="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tasks)
      TASKS="$2"
      shift 2
      ;;
    --results_root)
      RESULTS_ROOT="$2"
      shift 2
      ;;
    --state_dir)
      STATE_DIR="$2"
      shift 2
      ;;
    --threads)
      THREADS="$2"
      shift 2
      ;;
    --worker_id)
      WORKER_ID="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "$TASKS" ]]; then
  echo "Missing --tasks" >&2
  exit 2
fi

if [[ -z "$STATE_DIR" ]]; then
  STATE_DIR="${RESULTS_ROOT}/_cluster"
fi

TASKS_PATH="$(python -c "from pathlib import Path; print(Path('$TASKS').expanduser().resolve())")"
STATE_DIR_PATH="$(python -c "from pathlib import Path; print(Path('$STATE_DIR').expanduser().resolve())")"

LOCKS_DIR="${STATE_DIR_PATH}/locks"
DONE_DIR="${STATE_DIR_PATH}/done"
FAIL_DIR="${STATE_DIR_PATH}/failed"
LOG_DIR="${STATE_DIR_PATH}/logs"

mkdir -p "$LOCKS_DIR" "$DONE_DIR" "$FAIL_DIR" "$LOG_DIR"

HOST="$(hostname)"

# Avoid BLAS/OpenMP oversubscription.
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export PYTHONUNBUFFERED=1

# Avoid shared-cache download races: keep per-host+worker sklearn/OpenML cache.
export SCIKIT_LEARN_DATA="${SCIKIT_LEARN_DATA:-$HOME/.scikit_learn_data_${HOST}_w${WORKER_ID}}"
mkdir -p "$SCIKIT_LEARN_DATA"

normalize_dataset() {
  case "$1" in
    bbc-embedding|bbc-embed|bbc_embed) echo "bbc-embeddings" ;;
    miniboone) echo "MiniBooNE" ;;
    *) echo "$1" ;;
  esac
}

is_done() {
  local dataset="$1"
  local seed="$2"
  local ds_norm
  ds_norm="$(normalize_dataset "$dataset")"
  local out_csv="${ROOT_DIR}/${RESULTS_ROOT}/${ds_norm}/seed_${seed}/addition_experiment_results.csv"
  [[ -s "$out_csv" ]]
}

claim_lock() {
  local dataset="$1"
  local seed="$2"
  local ds_norm
  ds_norm="$(normalize_dataset "$dataset")"
  local lock="${LOCKS_DIR}/${ds_norm}_seed_${seed}.lock"
  if mkdir "$lock" 2>/dev/null; then
    printf "%s\n" "$HOST" > "${lock}/host"
    date -Is > "${lock}/start_time"
    return 0
  fi
  return 1
}

release_lock() {
  local dataset="$1"
  local seed="$2"
  local ds_norm
  ds_norm="$(normalize_dataset "$dataset")"
  local lock="${LOCKS_DIR}/${ds_norm}_seed_${seed}.lock"
  rm -rf "$lock" 2>/dev/null || true
}

run_task() {
  local dataset="$1"
  local seed="$2"
  local ds_norm
  ds_norm="$(normalize_dataset "$dataset")"
  local log_file="${LOG_DIR}/rq1dp_${ds_norm}_seed_${seed}_${HOST}.log"

  echo "[$(date -Is)] START dataset=$dataset seed=$seed host=$HOST" | tee -a "$log_file"

  # Run RQ1 setting with DP baseline (very slow).
  if conda run -n pydvl --no-capture-output python -u src/main.py \
      --dataset "$dataset" \
      --seed "$seed" \
      --config "config/rq1_config.yaml" \
      --results_root "$RESULTS_ROOT" \
      --skip_bipartite \
      --include_dp \
      --dp_max_subset_size 20 \
      >>"$log_file" 2>&1; then
    echo "[$(date -Is)] OK dataset=$dataset seed=$seed host=$HOST" | tee -a "$log_file"
    printf "%s\n" "$HOST" > "${DONE_DIR}/${ds_norm}_seed_${seed}.ok"
    return 0
  else
    echo "[$(date -Is)] FAIL dataset=$dataset seed=$seed host=$HOST" | tee -a "$log_file"
    printf "%s\n" "$HOST" > "${FAIL_DIR}/${ds_norm}_seed_${seed}.fail"
    return 1
  fi
}

while IFS=$'\t' read -r dataset seed; do
  [[ -z "${dataset// }" ]] && continue
  [[ "${dataset:0:1}" == "#" ]] && continue

  seed="${seed//[[:space:]]/}"
  if [[ -z "$seed" ]]; then
    continue
  fi

  if is_done "$dataset" "$seed"; then
    continue
  fi

  if ! claim_lock "$dataset" "$seed"; then
    continue
  fi

  # Re-check after acquiring lock (another worker may have finished).
  if is_done "$dataset" "$seed"; then
    release_lock "$dataset" "$seed"
    continue
  fi

  run_task "$dataset" "$seed" || true
  release_lock "$dataset" "$seed"
done < "$TASKS_PATH"

echo "[$(date -Is)] Worker finished on $HOST"
