#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
run_rq1_dp_multinode.sh

Launch multi-node RQ1 (DP) runs via SSH on the IDEA/FOCI nodes.

Notes:
- Uses shared NFS (/home) for code + output collection.
- Writes results under `results_rq1_dp/` by default to avoid overwriting RQ3 outputs.
- Starts one worker per node (DP is very heavy).

Example:
  bash scripts/run_rq1_dp_multinode.sh

  # Custom nodes
  bash scripts/run_rq1_dp_multinode.sh --nodes "idea-node-02 idea-node-03 idea-node-05"
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

NODES="idea-node-02 idea-node-03 idea-node-05 idea-node-06 idea-node-07"
RESULTS_ROOT="results_rq1_dp"
THREADS=1
WORKERS_PER_NODE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nodes)
      NODES="$2"
      shift 2
      ;;
    --results_root)
      RESULTS_ROOT="$2"
      shift 2
      ;;
    --threads)
      THREADS="$2"
      shift 2
      ;;
    --workers_per_node)
      WORKERS_PER_NODE="$2"
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

STATE_DIR="${RESULTS_ROOT}/_cluster"
TASKS="${STATE_DIR}/tasks.tsv"
LAUNCH_LOG="${STATE_DIR}/launch.log"
WORKER_OUT_DIR="${STATE_DIR}/worker_out"
mkdir -p "$STATE_DIR" "$WORKER_OUT_DIR"

cat >"$TASKS" <<'EOF'
# dataset\tseed
2dplanes	10
2dplanes	20
2dplanes	30
2dplanes	40
2dplanes	50
2dplanes	60
2dplanes	70
2dplanes	80
2dplanes	90
2dplanes	100
2dplanes	110
2dplanes	120
2dplanes	130
2dplanes	140
2dplanes	150
2dplanes	160
2dplanes	170
2dplanes	180
2dplanes	190
2dplanes	200
nomao	10
nomao	20
nomao	30
nomao	40
nomao	50
nomao	60
nomao	70
nomao	80
nomao	90
nomao	100
nomao	110
nomao	120
nomao	130
nomao	140
nomao	150
nomao	160
nomao	170
nomao	180
nomao	190
nomao	200
bbc-embeddings	10
bbc-embeddings	20
bbc-embeddings	30
bbc-embeddings	40
bbc-embeddings	50
bbc-embeddings	60
bbc-embeddings	70
bbc-embeddings	80
bbc-embeddings	90
bbc-embeddings	100
bbc-embeddings	110
bbc-embeddings	120
bbc-embeddings	130
bbc-embeddings	140
bbc-embeddings	150
bbc-embeddings	160
bbc-embeddings	170
bbc-embeddings	180
bbc-embeddings	190
bbc-embeddings	200
MiniBooNE	10
MiniBooNE	20
MiniBooNE	30
MiniBooNE	40
MiniBooNE	50
MiniBooNE	60
MiniBooNE	70
MiniBooNE	80
MiniBooNE	90
MiniBooNE	100
MiniBooNE	110
MiniBooNE	120
MiniBooNE	130
MiniBooNE	140
MiniBooNE	150
MiniBooNE	160
MiniBooNE	170
MiniBooNE	180
MiniBooNE	190
MiniBooNE	200
digits	10
digits	20
digits	30
digits	40
digits	50
digits	60
digits	70
digits	80
digits	90
digits	100
digits	110
digits	120
digits	130
digits	140
digits	150
digits	160
digits	170
digits	180
digits	190
digits	200
election	10
election	20
election	30
election	40
election	50
election	60
election	70
election	80
election	90
election	100
election	110
election	120
election	130
election	140
election	150
election	160
election	170
election	180
election	190
election	200
electricity	10
electricity	20
electricity	30
electricity	40
electricity	50
electricity	60
electricity	70
electricity	80
electricity	90
electricity	100
electricity	110
electricity	120
electricity	130
electricity	140
electricity	150
electricity	160
electricity	170
electricity	180
electricity	190
electricity	200
fried	10
fried	20
fried	30
fried	40
fried	50
fried	60
fried	70
fried	80
fried	90
fried	100
fried	110
fried	120
fried	130
fried	140
fried	150
fried	160
fried	170
fried	180
fried	190
fried	200
EOF

echo "[$(date -Is)] Launching RQ1-DP workers" | tee "$LAUNCH_LOG"
echo "ROOT_DIR=$ROOT_DIR" | tee -a "$LAUNCH_LOG"
echo "RESULTS_ROOT=$RESULTS_ROOT" | tee -a "$LAUNCH_LOG"
echo "STATE_DIR=$STATE_DIR" | tee -a "$LAUNCH_LOG"
echo "TASKS=$TASKS" | tee -a "$LAUNCH_LOG"
echo "NODES=$NODES" | tee -a "$LAUNCH_LOG"
echo "WORKERS_PER_NODE=$WORKERS_PER_NODE" | tee -a "$LAUNCH_LOG"

for node in $NODES; do
  for ((wid=0; wid<WORKERS_PER_NODE; wid++)); do
    echo "[$(date -Is)] launching on $node (worker_id=$wid)" | tee -a "$LAUNCH_LOG"
    worker_out="${WORKER_OUT_DIR}/worker_${node}_w${wid}.out"
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$node" \
      "cd '$ROOT_DIR'; nohup bash scripts/rq1_dp_worker.sh --tasks '$TASKS' --results_root '$RESULTS_ROOT' --state_dir '$STATE_DIR' --threads '$THREADS' --worker_id '$wid' >'$worker_out' 2>&1 &" \
      >>"$LAUNCH_LOG" 2>&1
  done
done

echo "[$(date -Is)] Done launching. Monitor:" | tee -a "$LAUNCH_LOG"
echo "  ls -la ${STATE_DIR}/logs | tail" | tee -a "$LAUNCH_LOG"
echo "  ls -la ${STATE_DIR}/done | wc -l" | tee -a "$LAUNCH_LOG"
echo "  ls -la ${STATE_DIR}/failed | wc -l" | tee -a "$LAUNCH_LOG"
echo "  conda run -n pydvl python scripts/aggregate_rq1_dp.py --results_root ${RESULTS_ROOT}" | tee -a "$LAUNCH_LOG"
