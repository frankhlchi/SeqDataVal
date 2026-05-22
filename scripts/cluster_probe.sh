#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
cluster_probe.sh

Collect a quick snapshot of:
- shared mounts (/home, /data)
- CPU capacity (lscpu summary)
- current CPU utilization (mpstat 1 1)
- memory (free -h)

This is useful for planning multi-node RQ1(DP) runs and for verifying that outputs
written under /home are visible across nodes.

Examples:
  bash scripts/cluster_probe.sh
  bash scripts/cluster_probe.sh --nodes "idea-node-02 idea-node-03 idea-node-06 idea-node-07"
  bash scripts/cluster_probe.sh --out results_rq1_dp/_cluster/cluster_snapshot.txt
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

NODES="idea-node-02 idea-node-03 idea-node-05 idea-node-06 idea-node-07"
OUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nodes)
      NODES="$2"
      shift 2
      ;;
    --out)
      OUT="$2"
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

if [[ -z "$OUT" ]]; then
  ts="$(date +%Y%m%d_%H%M%S)"
  OUT="results_rq1_dp/_cluster/cluster_snapshot_${ts}.txt"
fi

mkdir -p "$(dirname "$OUT")"

{
  echo "timestamp: $(date -Is)"
  echo "local_host: $(hostname)"
  echo "cwd: $ROOT_DIR"
  echo "nodes: $NODES"
  echo

  for node in $NODES; do
    echo "===== $node ====="
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$node" '
      set -e
      echo "host: $(hostname)"
      echo
      echo "[mounts]"
      df -Th /home /data | tail -n +2
      echo
      echo "[cpu]"
      lscpu | egrep "Model name|CPU\\(s\\)|Thread\\(s\\) per core|Core\\(s\\) per socket|Socket\\(s\\)"
      echo
      echo "[cpu_util]"
      mpstat 1 1 | tail -n 5
      echo
      echo "[mem]"
      free -h | head -n 3
    ' || echo "[WARN] ssh failed for $node"
    echo
  done
} | tee "$OUT"

echo "Wrote: $OUT"

