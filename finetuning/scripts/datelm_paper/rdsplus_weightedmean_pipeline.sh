#!/usr/bin/env bash
set -euo pipefail

export SEQDATAVAL_ROOT=${SEQDATAVAL_ROOT:-/workspace/sequence_dv/SeqDataVal}
export DATELM_ROOT=${DATELM_ROOT:-/workspace/sequence_dv/DATE-LM}
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export TORCH_HOME=${TORCH_HOME:-/workspace/.cache/torch}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export WANDB_MODE=${WANDB_MODE:-disabled}
export PYTHONPATH=$DATELM_ROOT:${PYTHONPATH:-}

LOGDIR=$(ls -dt "$DATELM_ROOT"/logs_rdsplus_weightedmean_upgrade_* | head -n 1)

TRAIN_EMB_DIR_RDS="$DATELM_ROOT/embeddings/paper_seed42_v1/tulu3_train_llama_weightedmean"
POOL_TRAIN_JSONL="$DATELM_ROOT/data/training_data/paper_seed42_v1_tulu3_200k_train.jsonl"

train_pidfile="$LOGDIR/train_emb_weightedmean_gpu3.pid"

echo "[START] $(date -Iseconds)"
echo "LOGDIR=$LOGDIR"
echo "TRAIN_EMB_DIR_RDS=$TRAIN_EMB_DIR_RDS"
echo "POOL_TRAIN_JSONL=$POOL_TRAIN_JSONL"

# 1) Wait for train embeddings to finish
if [ -f "$train_pidfile" ]; then
  pid=$(cat "$train_pidfile" 2>/dev/null || true)
  echo "train_pid=$pid"
  if [ -n "${pid:-}" ]; then
    while kill -0 "$pid" 2>/dev/null; do
      ts=$(date -Iseconds)
      # best-effort status from progress.json
      prog="$TRAIN_EMB_DIR_RDS/progress.json"
      if [ -f "$prog" ]; then
        next=$(python - "$prog" <<'PY'
import json
import sys

p = sys.argv[1]
try:
    j = json.load(open(p))
    print(j.get("train_next_idx"), j.get("train_complete"))
except Exception:
    print("?", "?")
PY
)
        echo "[$ts] train_emb running progress: $next"
      else
        echo "[$ts] train_emb running (progress.json not yet written)"
      fi
      sleep 600
    done
  fi
fi

echo "[DONE] train embeddings process exited $(date -Iseconds)"

# Ensure train embeddings completed successfully (not just process exit).
if [ ! -f "$TRAIN_EMB_DIR_RDS/progress.json" ]; then
  echo "[FATAL] missing $TRAIN_EMB_DIR_RDS/progress.json after train process exit" >&2
  exit 1
fi
python - "$TRAIN_EMB_DIR_RDS/progress.json" "$POOL_TRAIN_JSONL" <<'PY'
import json
import sys

prog_path = sys.argv[1]
pool_path = sys.argv[2]

with open(prog_path, "r") as f:
    j = json.load(f)

train_complete = j.get("train_complete", False)
train_next_idx = j.get("train_next_idx")

n_total = 0
with open(pool_path, "r") as f:
    for _ in f:
        n_total += 1

print(
    "train_progress",
    "train_next_idx",
    train_next_idx,
    "train_complete",
    train_complete,
    "n_total",
    n_total,
)

if train_complete is not True:
    raise SystemExit("train_complete is not True; embedding job likely crashed or was interrupted")
if train_next_idx != n_total:
    raise SystemExit(f"train_next_idx({train_next_idx}) != n_total({n_total})")
PY

# Verify train_emb.npy exists
python - "$TRAIN_EMB_DIR_RDS/train_emb.npy" <<'PY'
import sys
from pathlib import Path

import numpy as np

p = Path(sys.argv[1])
assert p.exists(), f"missing {p}"
a = np.load(p, mmap_mode="r")
print("train_emb", str(p), "shape", a.shape, "dtype", a.dtype)
PY

# 2) Verify ref embeddings exist
for task in mmlu gsm8k bbh; do
  ref_dir="$DATELM_ROOT/embeddings/paper_seed42_v1/${task}_ref_llama_weightedmean"
  test -f "$ref_dir/ref_emb.npy" || { echo "[FATAL] missing ref_emb: $ref_dir/ref_emb.npy"; exit 1; }
done

echo "[STEP] compute rds_plus_metrics.npy (weighted_mean pooling) $(date -Iseconds)"

# 3) Compute rds_plus metrics from embeddings (overwrite)
for task in mmlu gsm8k bbh; do
  ref_dir="$DATELM_ROOT/embeddings/paper_seed42_v1/${task}_ref_llama_weightedmean"
  out="$DATELM_ROOT/scores/paper_seed42_v1/${task}/rds_plus_metrics.npy"

  CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 python "$SEQDATAVAL_ROOT/finetuning/scripts/datelm_paper/compute_repsim_metrics_from_emb.py" \
    --train_emb "$TRAIN_EMB_DIR_RDS/train_emb.npy" \
    --ref_emb "$ref_dir/ref_emb.npy" \
    --out "$out" \
    --device cuda --dtype fp16 \
    --overwrite

  python - "$out" <<'PY'
import numpy as np, sys
p=sys.argv[1]
a=np.load(p)
print('metric', p, 'shape', a.shape, 'dtype', a.dtype, 'finite', bool(np.isfinite(a).all()), 'min', float(a.min()), 'max', float(a.max()))
PY

done

echo "[STEP] rerun rds_plus train+convert+official eval (3 tasks) $(date -Iseconds)"

# 4) Rerun rds_plus training/eval for 3 tasks in parallel (GPU0/1/2)
LOG_TABLE3="$DATELM_ROOT/logs_table3_single_seed_datelm"
mkdir -p "$LOG_TABLE3"

launch_task() {
  local task="$1" gpu="$2"
  local log="$LOG_TABLE3/launcher_rds_plus_weightedmean_${task}_gpu${gpu}.log"
  local pidfile="$LOG_TABLE3/launcher_rds_plus_weightedmean_${task}_gpu${gpu}.pid"

  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 nohup python \
    "$SEQDATAVAL_ROOT/finetuning/scripts/datelm_paper/run_multiseed_train_eval_datelm_litgpt.py" \
    --datelm_root "$DATELM_ROOT" --python python \
    --seeds 1337 --tasks "$task" --methods rds_plus \
    --results_dirname results_table3_single_seed_datelm \
    --out_dirname checkpoints_table3_single_seed_datelm \
    --log_dirname logs_table3_single_seed_datelm \
    --cleanup_checkpoints \
    >"$log" 2>&1 &
  echo $! >"$pidfile"
  echo "launched $task gpu=$gpu pid=$(cat "$pidfile") log=$log"
}

launch_task mmlu 0
launch_task gsm8k 1
launch_task bbh 2

# wait
for task in mmlu gsm8k bbh; do
  pidfile="$LOG_TABLE3/launcher_rds_plus_weightedmean_${task}_gpu"*".pid"
  pid=$(cat $pidfile 2>/dev/null || true)
  if [ -n "${pid:-}" ]; then
    while kill -0 "$pid" 2>/dev/null; do
      echo "[$(date -Iseconds)] waiting $task pid=$pid"
      sleep 600
    done
  fi
  echo "[$(date -Iseconds)] finished $task pid=$pid"
done

# Verify new metrics.json exist
for task in mmlu gsm8k bbh; do
  d="$DATELM_ROOT/results_table3_single_seed_datelm/paper_seed42_v1_trainseed1337_rds_plus_${task}_official"
  test -f "$d/metrics.json" || { echo "[FATAL] missing $d/metrics.json"; exit 1; }
done

echo "[STEP] summarize + update RUN_LEDGER + optional artifacts + push $(date -Iseconds)"

# 5) Summarize
cd "$SEQDATAVAL_ROOT/finetuning/scripts/datelm_paper"
python summarize_multiseed_results.py \
  --datelm_root "$DATELM_ROOT" \
  --results_dirname results_table3_single_seed_datelm \
  --seeds 1337 \
  --tasks mmlu gsm8k bbh \
  --methods random1 random2 random3 random_avg bm25 repsim rds_plus gradsim less bipcov \
  --output_dir "$SEQDATAVAL_ROOT/finetuning/table3_single_seed_datelm"

# 6) Update RUN_LEDGER.md
cd "$SEQDATAVAL_ROOT"

SEQ_SHA=$(git rev-parse HEAD 2>/dev/null || echo "<unknown>")
DATE_SHA=$(cd "$DATELM_ROOT" && git rev-parse HEAD 2>/dev/null || echo "<unknown>")
PYV=$(python -V 2>&1 || true)
DRV=$(nvidia-smi | head -n 3 | tr '\n' ' ')

PKG=$(python - <<'PY'
import importlib
mods=['torch','transformers','datasets','accelerate','peft','lightning','litgpt','sentencepiece']
vals=[]
for m in mods:
  try:
    mod=importlib.import_module(m)
    v=getattr(mod,'__version__','?')
    vals.append(f"{m}={v}")
  except Exception:
    vals.append(f"{m}=<missing>")
print(', '.join(vals))
PY
)

NOW=$(date -Iseconds)
RUN_ID="vast-table3-trainseed1337-rdsplus-weightedmean"

cat >>RUN_LEDGER.md <<EOF

### ${RUN_ID} (completed)
- Date: ${NOW}
- Provider: Vast.ai (container)
- Hardware: 4× NVIDIA H100 NVL 94GB
- Driver/CUDA: ${DRV}
- Python: ${PYV}
- Key packages: ${PKG}
- Repos:
  - SeqDataVal: ${SEQ_SHA}
  - DATE-LM: ${DATE_SHA}
- Base model: meta-llama/Llama-3.1-8B
- Experiment: Upgrade DATE-LM Table3 `rds_plus` from proxy → paper-faithful pooling (position-weighted mean pooling)
  - train seed: 1337
  - tasks: mmlu / gsm8k / bbh
  - Random1/2/3 selection seeds: 1 / 2 / 3; Random Avg = mean(Random1/2/3) per task
- Changes:
  - Recomputed RDS+ embeddings with `compute_llm_last_token_embeddings.py --pooling weighted_mean` on the 200k pool JSONL.
  - Recomputed `scores/paper_seed42_v1/{task}/rds_plus_metrics.npy` via `compute_repsim_metrics_from_emb.py`.
  - Moved old proxy official results dirs to `*_PROXY_<timestamp>` and copied old proxy metrics to `*_PROXY.npy`.
- Local patches (not committed to DATE-LM):
  - `finetuning/scripts/datelm_paper/patch_datelm_table3.sh` (temperature=0 greedy fix, LESS optimizer path, HF device_map sharding fix)
- Entry point:
  - `finetuning/scripts/datelm_paper/run_multiseed_train_eval_datelm_litgpt.py` (calls DATE-LM/train/finetune.py + official minimal_multitask eval)
- results_dirname: results_table3_single_seed_datelm
EOF

# 7) Optional small artifacts (rds_plus only)
DEST="$SEQDATAVAL_ROOT/finetuning/artifacts/datelm_table3_rdsplus_weightedmean_vast_20260106"
rm -rf "$DEST"
mkdir -p "$DEST"

python - <<'PY'
import os, shutil
from pathlib import Path

datelm = Path(os.environ['DATELM_ROOT'])
dest = Path(os.environ['DEST'])

paths=[]
# rds_plus metrics.json (3)
for t in ['mmlu','gsm8k','bbh']:
    paths.append(datelm / f"results_table3_single_seed_datelm/paper_seed42_v1_trainseed1337_rds_plus_{t}_official/metrics.json")
# rds_plus metrics.npy (3)
for t in ['mmlu','gsm8k','bbh']:
    paths.append(datelm / f"scores/paper_seed42_v1/{t}/rds_plus_metrics.npy")

missing=[p for p in paths if not p.exists()]
if missing:
    print('[FATAL] missing artifacts:')
    for p in missing:
        print(' ', p)
    raise SystemExit(1)

for p in paths:
    rel = p.relative_to(datelm)
    out = dest / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, out)

print('[OK] copied', len(paths), 'files')
PY

cat >"$DEST/README.md" <<'EOF'
Small artifacts for DATE-LM Table3 single-seed Vast run (RDS+ paper-faithful upgrade):
- results_table3_single_seed_datelm/**/metrics.json (rds_plus only)
- scores/paper_seed42_v1/**/rds_plus_metrics.npy

No raw data, embeddings, checkpoints, full logs, model weights, or secrets.
EOF

( cd "$DEST" && find . -type f ! -name 'MANIFEST.sha256' -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256 )
( cd "$DEST" && sha256sum -c MANIFEST.sha256 )

# 8) Git: pull, verify status, commit, push
export GIT_TERMINAL_PROMPT=0

git pull --rebase --autostash

# Allowed files: RUN_LEDGER + 2 summary + optional artifacts dir
ALLOWED_RE='^(RUN_LEDGER\.md|finetuning/table3_single_seed_datelm/MULTISEED_TRAINSEED_RESULTS\.(md|csv)|finetuning/artifacts/datelm_table3_rdsplus_weightedmean_vast_20260106/.*)$'
STATUS=$(git status --porcelain)
BAD=$(echo "$STATUS" | awk '{print $2}' | rg -v "$ALLOWED_RE" || true)
if [ -n "${BAD:-}" ]; then
  echo "[FATAL] unexpected git changes:" >&2
  git status --porcelain >&2
  exit 1
fi

rg -n "HUGGINGFACE|HF_TOKEN|TOKEN" RUN_LEDGER.md finetuning/table3_single_seed_datelm/MULTISEED_TRAINSEED_RESULTS.md finetuning/table3_single_seed_datelm/MULTISEED_TRAINSEED_RESULTS.csv || true

git add RUN_LEDGER.md \
  finetuning/table3_single_seed_datelm/MULTISEED_TRAINSEED_RESULTS.md \
  finetuning/table3_single_seed_datelm/MULTISEED_TRAINSEED_RESULTS.csv \
  finetuning/artifacts/datelm_table3_rdsplus_weightedmean_vast_20260106

git commit -m "Update RDS+ to paper-faithful weighted-mean pooling (Table3)"
git push

echo "[FINAL] $(date -Iseconds) done"
