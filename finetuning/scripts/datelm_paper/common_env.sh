#!/usr/bin/env bash
# Shared, path-portable env setup for DATE-LM paper-style pipelines.
#
# This file is meant to be sourced from the run_paper_*.sh scripts.
set -euo pipefail

DATELM_PAPER_SCRIPTS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Repo root of SeqDataVal (this repo).
SEQDATAVAL_ROOT="${SEQDATAVAL_ROOT:-$(cd "${DATELM_PAPER_SCRIPTS_DIR}/../../.." && pwd)}"

_looks_like_datelm_root() {
  local candidate="${1:-}"
  [[ -n "${candidate}" ]] \
    && [[ -d "${candidate}/minimal_multitask" ]] \
    && [[ -d "${candidate}/methods" ]] \
    && [[ -d "${candidate}/data" ]]
}

_infer_datelm_root() {
  local provided="${DATELM_ROOT:-}"
  if [[ -n "${provided}" ]]; then
    echo "${provided}"
    return 0
  fi

  local base_a base_b base_c
  base_a="$(cd "${SEQDATAVAL_ROOT}/../.." && pwd)"
  base_b="$(cd "${SEQDATAVAL_ROOT}/.." && pwd)"
  base_c="$(cd "${SEQDATAVAL_ROOT}" && pwd)"

  local candidates=(
    "${base_a}/DATE-LM"
    "${base_a}/DATE-LM/DATE-LM-main"
    "${base_a}/DATE-LM-main"
    "${base_b}/DATE-LM"
    "${base_b}/DATE-LM/DATE-LM-main"
    "${base_b}/DATE-LM-main"
    "${base_c}/DATE-LM"
    "${base_c}/DATE-LM/DATE-LM-main"
    "${base_c}/DATE-LM-main"
  )

  local c
  for c in "${candidates[@]}"; do
    if _looks_like_datelm_root "${c}"; then
      echo "${c}"
      return 0
    fi
  done

  return 1
}

DATELM_ROOT="$(_infer_datelm_root || true)"
if [[ -z "${DATELM_ROOT}" ]]; then
  cat <<'EOF'
ERROR: DATELM_ROOT is not set and could not be inferred.

Set it to your DATE-LM repo root (recommended):
  export DATELM_ROOT=/workspace/sequence_dv/DATE-LM

If you have the older layout where the repo lives under DATE-LM-main:
  export DATELM_ROOT=/abs/path/to/DATE-LM-main
EOF
  exit 1
fi

# Prefer active env's python; allow overrides.
PY_TRAIN_EVAL="${PY_TRAIN_EVAL:-python}"
PY_EMB="${PY_EMB:-${PY_TRAIN_EVAL}}"

# Base model used for LoRA + eval (can override per machine / access).
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B}"

export SEQDATAVAL_ROOT DATELM_ROOT PY_TRAIN_EVAL PY_EMB BASE_MODEL

