#!/usr/bin/env bash
set -euo pipefail

# Finalize helper for the *old server* extra-baselines run:
# - waits for the runner PID to exit (if PID file exists)
# - checks that all 36 metrics.json exist
# - generates summaries into finetuning/multiseed_extra_baselines_oldserver/
# - commits + pushes ONLY the summary md/csv (+ README)
#
# This is safe to run multiple times: it will no-op if results are incomplete.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SEQDATAVAL_ROOT="${SEQDATAVAL_ROOT:-"$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"}"

DATELM_ROOT="${DATELM_ROOT:-""}"
if [[ -z "${DATELM_ROOT}" ]]; then
  echo "[ERROR] DATELM_ROOT is not set. Export DATELM_ROOT=/path/to/DATE-LM (DATE-LM-main is OK)."
  exit 1
fi

PY_TRAIN_EVAL="${PY_TRAIN_EVAL:-"${SEQDATAVAL_PYTHON:-python}"}"

PID_FILE="${DATELM_ROOT}/logs/multiseed_runner_extra_baselines.pid"
LOG_FILE="${DATELM_ROOT}/logs/multiseed_runner_extra_baselines.log"

OUT_DIR="${SEQDATAVAL_ROOT}/finetuning/multiseed_extra_baselines_oldserver"
OUT_MD="${OUT_DIR}/MULTISEED_TRAINSEED_RESULTS.md"
OUT_CSV="${OUT_DIR}/MULTISEED_TRAINSEED_RESULTS.csv"

METHODS=(bm25 repsim repsim_v2 rds_plus_v2)
SEEDS=(42 1337 2025)
TASKS=(mmlu gsm8k bbh)
TOTAL=$(( ${#METHODS[@]} * ${#SEEDS[@]} * ${#TASKS[@]} ))

echo "[INFO] SEQDATAVAL_ROOT=${SEQDATAVAL_ROOT}"
echo "[INFO] DATELM_ROOT=${DATELM_ROOT}"
echo "[INFO] PY_TRAIN_EVAL=${PY_TRAIN_EVAL}"
echo "[INFO] Expecting ${TOTAL} combinations"
echo "[INFO] PID_FILE=${PID_FILE}"
echo "[INFO] LOG_FILE=${LOG_FILE}"

if [[ -f "${PID_FILE}" ]]; then
  PID="$(tr -d ' \n' < "${PID_FILE}")"
  if [[ -n "${PID}" ]] && ps -p "${PID}" >/dev/null 2>&1; then
    echo "[INFO] Runner still running (PID=${PID}). Waiting..."
    while ps -p "${PID}" >/dev/null 2>&1; do
      sleep 60
    done
    echo "[INFO] Runner PID=${PID} exited."
  else
    echo "[INFO] PID file exists but PID is not running; continuing."
  fi
else
  echo "[WARN] PID file missing; assuming runner is not running."
fi

# Count completed combos (metrics.json present)
RESULTS_DIR="${DATELM_ROOT}/results"
if [[ ! -d "${RESULTS_DIR}" ]]; then
  echo "[ERROR] Missing results dir: ${RESULTS_DIR}"
  exit 1
fi

DONE=$(
  find "${RESULTS_DIR}" -type f -name metrics.json \
    | rg -n "paper_seed42_v1_trainseed(42|1337|2025)_(bm25|repsim|repsim_v2|rds_plus_v2)_(mmlu|gsm8k|bbh)_official/metrics\\.json$" \
    | wc -l \
    | tr -d ' '
)
echo "[INFO] Done metrics: ${DONE}/${TOTAL}"

if [[ "${DONE}" -lt "${TOTAL}" ]]; then
  echo "[INFO] Not complete yet; will not summarize or commit."
  exit 0
fi

mkdir -p "${OUT_DIR}"

echo "[INFO] Generating summary into: ${OUT_DIR}"
cd "${SEQDATAVAL_ROOT}"
"${PY_TRAIN_EVAL}" finetuning/scripts/datelm_paper/summarize_multiseed_results.py \
  --datelm_root "${DATELM_ROOT}" \
  --methods "${METHODS[@]}" \
  --seeds "${SEEDS[@]}" \
  --tasks "${TASKS[@]}" \
  --output_dir "${OUT_DIR}"

if [[ ! -f "${OUT_MD}" || ! -f "${OUT_CSV}" ]]; then
  echo "[ERROR] Missing expected outputs: ${OUT_MD} / ${OUT_CSV}"
  exit 1
fi

if ! rg -n "All combinations completed successfully\\." "${OUT_MD}" >/dev/null; then
  echo "[ERROR] Summary indicates missing results; refusing to commit."
  exit 1
fi

echo "[INFO] Committing summaries (md/csv + README) to git..."
git pull --rebase
git add "${OUT_MD}" "${OUT_CSV}" "${OUT_DIR}/README.md"
git commit -m "Results: extra-baselines trainseed (old server)" || {
  echo "[INFO] Nothing to commit."
  exit 0
}
git push
echo "[OK] Pushed summary."

