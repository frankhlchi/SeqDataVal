#!/usr/bin/env bash
set -euo pipefail

# DATE-LM Table 3: BipCov embedding ablations (single seed=1337).
# Intended for multi-GPU machines.
#
# This script:
#   1) Ensures embeddings exist for:
#      - bipcov_wmean: Llama weighted-mean (paper-faithful RDS+ pooling) embeddings
#      - bipcov_bge:   SentenceTransformer BGE embeddings
#      - bipcov_e5:    SentenceTransformer E5 embeddings
#   2) Computes DATE-LM-compatible BipCov score files (*_metrics.npy) for mmlu/gsm8k/bbh.
#   3) Runs DATE-LM LitGPT train->convert->official eval for the 3 variants (9 jobs).
#   4) Summarizes results into finetuning/table3_bipcov_emb_ablation_datelm/.
#
# It does NOT git commit/push.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/common_env.sh"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TORCH_HOME="${TORCH_HOME:-/workspace/.cache/torch}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONPATH="${DATELM_ROOT}:${PYTHONPATH:-}"

TRAIN_SEED="${TRAIN_SEED:-1337}"
REF_SEED="${REF_SEED:-42}"
REF_NUM_SAMPLES="${REF_NUM_SAMPLES:-100}"

RESULTS_DIRNAME="${RESULTS_DIRNAME:-results_table3_single_seed_datelm}"
OUT_DIRNAME="${OUT_DIRNAME:-checkpoints_table3_single_seed_datelm}"
LOG_DIRNAME="${LOG_DIRNAME:-logs_table3_single_seed_datelm}"

TASKS=(mmlu gsm8k bbh)
METHODS=(bipcov_wmean bipcov_bge bipcov_e5)

TRAIN_JSONL="${DATELM_ROOT}/data/training_data/paper_seed42_v1_tulu3_200k_train.jsonl"
LITGPT_CKPT="${DATELM_ROOT}/litgpt_checkpoints/meta-llama/Llama-3.1-8B/lit_model.pth"

ts="$(date +%Y%m%d_%H%M%S)"
LOGROOT="${DATELM_ROOT}/logs_table3_bipcov_emb_ablation_${ts}"
mkdir -p "${LOGROOT}"

OUT_SUMMARY_DIR="${SEQDATAVAL_ROOT}/finetuning/table3_bipcov_emb_ablation_datelm"
mkdir -p "${OUT_SUMMARY_DIR}"
TIMING_CSV="${OUT_SUMMARY_DIR}/TIMING.csv"
if [[ ! -f "${TIMING_CSV}" ]]; then
  echo "ts,stage,variant,task,gpu,elapsed_s,ok" > "${TIMING_CSV}"
fi

echo "[START] $(date -Iseconds)"
echo "SEQDATAVAL_ROOT=${SEQDATAVAL_ROOT}"
echo "DATELM_ROOT=${DATELM_ROOT}"
echo "TRAIN_SEED=${TRAIN_SEED}"
echo "REF_SEED=${REF_SEED}"
echo "REF_NUM_SAMPLES=${REF_NUM_SAMPLES}"
echo "RESULTS_DIRNAME=${RESULTS_DIRNAME}"
echo "OUT_DIRNAME=${OUT_DIRNAME}"
echo "LOG_DIRNAME=${LOG_DIRNAME}"
echo "LOGROOT=${LOGROOT}"

if [[ ! -f "${TRAIN_JSONL}" ]]; then
  echo "[FATAL] missing TRAIN_JSONL: ${TRAIN_JSONL}" >&2
  exit 1
fi
if [[ ! -f "${LITGPT_CKPT}" ]]; then
  echo "[FATAL] missing LitGPT checkpoint: ${LITGPT_CKPT}" >&2
  exit 1
fi

python - <<'PY'
import importlib
missing=[]
for m in ["torch","transformers","datasets","accelerate","peft","sentencepiece","lightning","litgpt"]:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
if missing:
    raise SystemExit(f"[FATAL] missing python deps: {missing}")
print("[OK] python deps look present")
PY

if python - <<'PY' >/dev/null 2>&1; then
from sentence_transformers import SentenceTransformer
print(SentenceTransformer)
PY
  echo "[OK] sentence-transformers already installed"
else
  echo "[STEP] installing sentence-transformers (needed for BGE/E5 ablations)"
  pip install -U sentence-transformers
fi

echo "[STEP] ensure weighted-mean (RDS+) embeddings exist (bipcov_wmean) $(date -Iseconds)"
TRAIN_WMEAN_DIR="${DATELM_ROOT}/embeddings/paper_seed42_v1/tulu3_train_llama_weightedmean"
TRAIN_WMEAN_EMB="${TRAIN_WMEAN_DIR}/train_emb.npy"
TRAIN_WMEAN_PROG="${TRAIN_WMEAN_DIR}/progress.json"

need_wmean_train="0"
if [[ ! -f "${TRAIN_WMEAN_EMB}" ]]; then
  need_wmean_train="1"
elif [[ ! -f "${TRAIN_WMEAN_PROG}" ]]; then
  need_wmean_train="1"
else
  if ! python - "${TRAIN_WMEAN_PROG}" "${TRAIN_JSONL}" >/dev/null 2>&1 <<'PY'; then
import json, sys
prog=json.load(open(sys.argv[1]))
train_complete=prog.get("train_complete") is True
train_next=prog.get("train_next_idx")
n_total=sum(1 for _ in open(sys.argv[2], "r"))
assert train_complete
assert int(train_next) == int(n_total)
PY
    need_wmean_train="1"
  fi
fi

if [[ "${need_wmean_train}" == "1" ]]; then
  echo "[INFO] missing weighted-mean train_emb.npy; computing it now (this can take hours)"
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    python "${SEQDATAVAL_ROOT}/finetuning/scripts/datelm_paper/compute_llm_last_token_embeddings.py" \
      --datelm_root "${DATELM_ROOT}" \
      --model_name "${BASE_MODEL}" \
      --pooling weighted_mean \
      --train_jsonl "${TRAIN_JSONL}" \
      --messages_only_first_two \
      --no_ref \
      --out_dir "${TRAIN_WMEAN_DIR}" \
    >"${LOGROOT}/llm_wmean_train_emb.log" 2>&1
fi

for task in "${TASKS[@]}"; do
  ref_dir="${DATELM_ROOT}/embeddings/paper_seed42_v1/${task}_ref_llama_weightedmean"
  ref_emb="${ref_dir}/ref_emb.npy"
  if [[ ! -f "${ref_emb}" ]]; then
    echo "[INFO] missing weighted-mean ref_emb for ${task}; computing it now"
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
      python "${SEQDATAVAL_ROOT}/finetuning/scripts/datelm_paper/compute_llm_last_token_embeddings.py" \
        --datelm_root "${DATELM_ROOT}" \
        --model_name "${BASE_MODEL}" \
        --pooling weighted_mean \
        --no_train \
        --ref_dataset_name "${task}" \
        --ref_num_samples "${REF_NUM_SAMPLES}" \
        --ref_seed "${REF_SEED}" \
        --out_dir "${ref_dir}" \
      >"${LOGROOT}/llm_wmean_ref_${task}.log" 2>&1
  fi
done

echo "[STEP] compute SentenceTransformer embeddings (BGE/E5) $(date -Iseconds)"
ST_BGE="BAAI/bge-large-en-v1.5"
ST_E5="intfloat/e5-large-v2"
ST_BATCH_SIZE="${ST_BATCH_SIZE:-256}"

BGE_TRAIN_DIR="${DATELM_ROOT}/embeddings/paper_seed42_v1_refpromptlabel/tulu3_train_bge"
E5_TRAIN_DIR="${DATELM_ROOT}/embeddings/paper_seed42_v1_refpromptlabel/tulu3_train_e5"

_train_complete() {
  local out_dir="$1"
  if [[ ! -f "${out_dir}/progress.json" ]]; then
    return 1
  fi
  python - "$out_dir/progress.json" >/dev/null 2>&1 <<'PY'
import json
import sys
p=sys.argv[1]
j=json.load(open(p))
assert j.get("train_complete") is True
PY
}

launch_st_train() {
  local gpu="$1" st_model="$2" out_dir="$3" log="$4"
  if _train_complete "${out_dir}"; then
    echo "[SKIP] train embeddings already complete: ${out_dir}"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
    python "${SEQDATAVAL_ROOT}/finetuning/scripts/datelm_paper/compute_st_embeddings.py" \
      --datelm_root "${DATELM_ROOT}" \
      --st_model "${st_model}" \
      --device cuda \
      --batch_size "${ST_BATCH_SIZE}" \
      --format tulu_chat \
      --train_jsonl "${TRAIN_JSONL}" \
      --messages_only_first_two \
      --no_ref \
      --out_dir "${out_dir}" \
    >"${log}" 2>&1
}

launch_st_ref() {
  local gpu="$1" st_model="$2" ref_task="$3" out_dir="$4" doc_prefix="$5" query_prefix="$6" log="$7"
  if [[ -f "${out_dir}/ref_emb.npy" ]]; then
    echo "[SKIP] ref embeddings already exist: ${out_dir}/ref_emb.npy"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
    python "${SEQDATAVAL_ROOT}/finetuning/scripts/datelm_paper/compute_st_embeddings.py" \
      --datelm_root "${DATELM_ROOT}" \
      --st_model "${st_model}" \
      --device cuda \
      --batch_size "${ST_BATCH_SIZE}" \
      --format tulu_chat \
      --doc_prefix "${doc_prefix}" \
      --query_prefix "${query_prefix}" \
      --no_train \
      --ref_dataset_name "${ref_task}" \
      --ref_num_samples "${REF_NUM_SAMPLES}" \
      --ref_seed "${REF_SEED}" \
      --out_dir "${out_dir}" \
    >"${log}" 2>&1
}

echo "[LAUNCH] BGE train emb on GPU0"
launch_st_train 0 "${ST_BGE}" "${BGE_TRAIN_DIR}" "${LOGROOT}/st_bge_train.log" &
pid_bge_train=$!

echo "[LAUNCH] E5 train emb on GPU1"
launch_st_train 1 "${ST_E5}" "${E5_TRAIN_DIR}" "${LOGROOT}/st_e5_train.log" &
pid_e5_train=$!

echo "[LAUNCH] BGE ref emb (3 tasks) on GPU2"
(
  for task in "${TASKS[@]}"; do
    out_dir="${DATELM_ROOT}/embeddings/paper_seed42_v1_refpromptlabel/${task}_ref_bge"
    launch_st_ref 2 "${ST_BGE}" "${task}" "${out_dir}" "" "" "${LOGROOT}/st_bge_ref_${task}.log"
  done
) &
pid_bge_ref=$!

echo "[LAUNCH] E5 ref emb (3 tasks) on GPU3"
(
  for task in "${TASKS[@]}"; do
    out_dir="${DATELM_ROOT}/embeddings/paper_seed42_v1_refpromptlabel/${task}_ref_e5"
    launch_st_ref 3 "${ST_E5}" "${task}" "${out_dir}" "passage: " "query: " "${LOGROOT}/st_e5_ref_${task}.log"
  done
) &
pid_e5_ref=$!

echo "[WAIT] embeddings jobs..."
wait "${pid_bge_train}"
wait "${pid_e5_train}"
wait "${pid_bge_ref}"
wait "${pid_e5_ref}"
echo "[DONE] embeddings ready $(date -Iseconds)"

echo "[STEP] compute BipCov score files (*_metrics.npy) $(date -Iseconds)"
check_metric_npy() {
  python - "$1" >/dev/null 2>&1 <<'PY'
import numpy as np, sys
p=sys.argv[1]
a=np.load(p, mmap_mode="r")
assert a.ndim == 1
assert a.shape[0] == 199999, a.shape
assert np.isfinite(a).all()
PY
}

compute_bipcov_scores() {
  local gpu="$1" variant="$2" train_emb="$3" ref_emb="$4" out_metric="$5" task="$6"
  local start end elapsed ok
  if [[ -f "${out_metric}" ]]; then
    ok="1"
    check_metric_npy "${out_metric}" || ok="0"
    if [[ "${ok}" == "1" ]]; then
      echo "[SKIP] existing metric ok: ${out_metric}"
      echo "$(date -Iseconds),bipcov_score,${variant},${task},${gpu},0,1" >> "${TIMING_CSV}"
      return 0
    fi
    echo "[WARN] existing metric invalid; recomputing: ${out_metric}"
  fi
  start="$(date +%s)"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
    python "${SEQDATAVAL_ROOT}/finetuning/bipcov/probe_bipcov_from_emb.py" \
      --train_emb "${train_emb}" \
      --ref_emb "${ref_emb}" \
      --out "${out_metric}" \
      --k_max 10000 \
      --top_l 200 \
      --device cuda \
      --batch_rows 50000 \
    >"${LOGROOT}/bipcov_${variant}_${task}.log" 2>&1
  end="$(date +%s)"
  elapsed="$((end-start))"
  ok="1"
  check_metric_npy "${out_metric}" || ok="0"
  echo "$(date -Iseconds),bipcov_score,${variant},${task},${gpu},${elapsed},${ok}" >> "${TIMING_CSV}"
  if [[ "${ok}" != "1" ]]; then
    echo "[FATAL] invalid metric file: ${out_metric}" >&2
    tail -n 80 "${LOGROOT}/bipcov_${variant}_${task}.log" >&2 || true
    exit 1
  fi
}

for task in "${TASKS[@]}"; do
  mkdir -p "${DATELM_ROOT}/scores/paper_seed42_v1_refpromptlabel/${task}"

  compute_bipcov_scores 0 "wmean" \
    "${TRAIN_WMEAN_EMB}" \
    "${DATELM_ROOT}/embeddings/paper_seed42_v1/${task}_ref_llama_weightedmean/ref_emb.npy" \
    "${DATELM_ROOT}/scores/paper_seed42_v1_refpromptlabel/${task}/bipcov_wmean_metrics.npy" \
    "${task}"

  compute_bipcov_scores 1 "bge" \
    "${BGE_TRAIN_DIR}/train_emb.npy" \
    "${DATELM_ROOT}/embeddings/paper_seed42_v1_refpromptlabel/${task}_ref_bge/ref_emb.npy" \
    "${DATELM_ROOT}/scores/paper_seed42_v1_refpromptlabel/${task}/bipcov_bge_metrics.npy" \
    "${task}"

  compute_bipcov_scores 2 "e5" \
    "${E5_TRAIN_DIR}/train_emb.npy" \
    "${DATELM_ROOT}/embeddings/paper_seed42_v1_refpromptlabel/${task}_ref_e5/ref_emb.npy" \
    "${DATELM_ROOT}/scores/paper_seed42_v1_refpromptlabel/${task}/bipcov_e5_metrics.npy" \
    "${task}"
done

echo "[STEP] run DATE-LM LitGPT train->convert->official eval (9 jobs, 4 GPUs) $(date -Iseconds)"
LOG_TABLE3="${DATELM_ROOT}/${LOG_DIRNAME}"
mkdir -p "${LOG_TABLE3}"

declare -a JOBS=()
for method in "${METHODS[@]}"; do
  for task in "${TASKS[@]}"; do
    JOBS+=("${method}:${task}")
  done
done

launch_job() {
  local gpu="$1" method="$2" task="$3"
  local log="${LOG_TABLE3}/launcher_${method}_${task}_gpu${gpu}.log"
  local pidfile="${LOG_TABLE3}/launcher_${method}_${task}_gpu${gpu}.pid"
  local start
  start="$(date +%s)"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 nohup python \
    "${SEQDATAVAL_ROOT}/finetuning/scripts/datelm_paper/run_multiseed_train_eval_datelm_litgpt.py" \
      --datelm_root "${DATELM_ROOT}" --python python \
      --seeds "${TRAIN_SEED}" --tasks "${task}" --methods "${method}" \
      --results_dirname "${RESULTS_DIRNAME}" \
      --out_dirname "${OUT_DIRNAME}" \
      --log_dirname "${LOG_DIRNAME}" \
      --cleanup_checkpoints \
    >"${log}" 2>&1 &
  echo $! >"${pidfile}"
  echo "${start}" > "${pidfile}.start"
  echo "[LAUNCH] gpu=${gpu} pid=$(cat "${pidfile}") method=${method} task=${task} log=${log}"
}

gpu_pids=( "" "" "" "" )
gpu_job=( "" "" "" "" )
job_idx=0

fill_free_gpus() {
  for gpu in 0 1 2 3; do
    if [[ -z "${gpu_pids[$gpu]}" ]] && [[ "${job_idx}" -lt "${#JOBS[@]}" ]]; then
      pair="${JOBS[$job_idx]}"
      method="${pair%%:*}"
      task="${pair#*:}"
      launch_job "${gpu}" "${method}" "${task}"
      gpu_pids[$gpu]="$(cat "${LOG_TABLE3}/launcher_${method}_${task}_gpu${gpu}.pid")"
      gpu_job[$gpu]="${method}:${task}"
      job_idx=$((job_idx+1))
    fi
  done
}

fill_free_gpus

while :; do
  all_done=true
  for gpu in 0 1 2 3; do
    pid="${gpu_pids[$gpu]}"
    if [[ -n "${pid}" ]]; then
      all_done=false
      if ! kill -0 "${pid}" 2>/dev/null; then
        pair="${gpu_job[$gpu]}"
        method="${pair%%:*}"
        task="${pair#*:}"
        start_path="${LOG_TABLE3}/launcher_${method}_${task}_gpu${gpu}.pid.start"
        start="$(cat "${start_path}" 2>/dev/null || echo 0)"
        end="$(date +%s)"
        elapsed="$((end-start))"

        metrics="${DATELM_ROOT}/${RESULTS_DIRNAME}/paper_seed42_v1_refpromptlabel_trainseed${TRAIN_SEED}_${method}_${task}_official/metrics.json"
        ok="1"
        if [[ ! -f "${metrics}" ]]; then
          ok="0"
        fi
        echo "$(date -Iseconds),train_eval,${method},${task},${gpu},${elapsed},${ok}" >> "${TIMING_CSV}"

        if [[ "${ok}" != "1" ]]; then
          echo "[FATAL] missing metrics.json after job exit: ${metrics}" >&2
          echo "[FATAL] tail log: ${LOG_TABLE3}/launcher_${method}_${task}_gpu${gpu}.log" >&2
          tail -n 120 "${LOG_TABLE3}/launcher_${method}_${task}_gpu${gpu}.log" >&2 || true
          exit 1
        fi

        echo "[DONE] ${method} ${task} gpu=${gpu} elapsed_s=${elapsed}"
        gpu_pids[$gpu]=""
        gpu_job[$gpu]=""
      fi
    fi
  done

  if [[ "${job_idx}" -ge "${#JOBS[@]}" ]]; then
    if [[ "${all_done}" == true ]]; then
      break
    fi
  else
    fill_free_gpus
  fi

  sleep 60
done

echo "[DONE] all 9 jobs finished $(date -Iseconds)"

echo "[STEP] summarize results -> ${OUT_SUMMARY_DIR} $(date -Iseconds)"
cd "${SEQDATAVAL_ROOT}/finetuning/scripts/datelm_paper"
python summarize_multiseed_results.py \
  --datelm_root "${DATELM_ROOT}" \
  --results_dirname "${RESULTS_DIRNAME}" \
  --seeds "${TRAIN_SEED}" \
  --tasks mmlu gsm8k bbh \
  --methods random_avg rds_plus bipcov bipcov_wmean bipcov_bge bipcov_e5 \
  --output_dir "${OUT_SUMMARY_DIR}"

echo "[FINAL] $(date -Iseconds) done"
