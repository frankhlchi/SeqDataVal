#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RESULTS_ROOT="${1:-results_rq1_dp}"
STATE_DIR="${RESULTS_ROOT}/_cluster"
TASKS="${STATE_DIR}/tasks.tsv"

echo "RESULTS_ROOT=$RESULTS_ROOT"
echo "STATE_DIR=$STATE_DIR"

if [[ ! -f "$TASKS" ]]; then
  echo "No tasks file found: $TASKS"
  exit 1
fi

if command -v rg >/dev/null 2>&1; then
  total=$(rg -v '^(#|\\s*$)' "$TASKS" | wc -l | tr -d ' ')
else
  total=$(grep -Ev '^(#|[[:space:]]*$)' "$TASKS" | wc -l | tr -d ' ')
fi
done_cnt=$(find "${STATE_DIR}/done" -type f -name '*.ok' 2>/dev/null | wc -l | tr -d ' ')
fail_cnt=$(find "${STATE_DIR}/failed" -type f -name '*.fail' 2>/dev/null | wc -l | tr -d ' ')
lock_cnt=$(find "${STATE_DIR}/locks" -maxdepth 1 -type d -name '*.lock' 2>/dev/null | wc -l | tr -d ' ')

echo "tasks_total=$total"
echo "done_ok=$done_cnt"
echo "failed=$fail_cnt"
echo "active_locks=$lock_cnt"

echo
echo "Latest task logs:"
ls -lt "${STATE_DIR}/logs" 2>/dev/null | head -n 15 || true
