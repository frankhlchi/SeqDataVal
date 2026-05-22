#!/usr/bin/env bash
set -euo pipefail

# DATE-LM Table 3: BipCov embedding backend ablations (large embedding models; single seed=1337).
#
# Runs BipCov with additional embedding backends while keeping DATE-LM Table3 training+eval unchanged:
#   - Qwen/Qwen3-Embedding-8B
#   - nvidia/NV-Embed-v2
#   - Alibaba-NLP/gte-Qwen2-7B-instruct
#   - GritLM/GritLM-7B
#
# This script:
#   1) Computes train/ref embeddings (prompt+label ref) for each embedding backend.
#   2) Computes BipCov score files (*_metrics.npy) for mmlu/gsm8k/bbh.
#   3) Runs DATE-LM LitGPT train->convert->official eval for the 4 variants (12 jobs).
#   4) Summarizes results into finetuning/table3_bipcov_emb_ablation_big4_datelm/.
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

EMB_BATCH_SIZE="${EMB_BATCH_SIZE:-8}"
EMB_MAX_LENGTH="${EMB_MAX_LENGTH:-512}"
EMB_MODEL_DTYPE="${EMB_MODEL_DTYPE:-bfloat16}"
EMB_OUT_DTYPE="${EMB_OUT_DTYPE:-float16}"
EMB_POOLING="${EMB_POOLING:-mean}"

TASKS=(mmlu gsm8k bbh)
METHODS=(bipcov_qwen3emb bipcov_nvembed bipcov_gteqwen2 bipcov_gritlm)

TRAIN_JSONL="${DATELM_ROOT}/data/training_data/paper_seed42_v1_tulu3_200k_train.jsonl"
LITGPT_CKPT="${DATELM_ROOT}/litgpt_checkpoints/meta-llama/Llama-3.1-8B/lit_model.pth"

ts="$(date +%Y%m%d_%H%M%S)"
LOGROOT="${DATELM_ROOT}/logs_table3_bipcov_emb_ablation_big4_${ts}"
mkdir -p "${LOGROOT}"

OUT_SUMMARY_DIR="${SEQDATAVAL_ROOT}/finetuning/table3_bipcov_emb_ablation_big4_datelm"
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
echo "EMB_BATCH_SIZE=${EMB_BATCH_SIZE}"
echo "EMB_MAX_LENGTH=${EMB_MAX_LENGTH}"
echo "EMB_MODEL_DTYPE=${EMB_MODEL_DTYPE}"
echo "EMB_OUT_DTYPE=${EMB_OUT_DTYPE}"
echo "EMB_POOLING=${EMB_POOLING}"
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
for m in ["torch","transformers","datasets","accelerate","peft","sentencepiece","lightning","litgpt","einops"]:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
if missing:
    raise SystemExit(f"[FATAL] missing python deps: {missing}")
print("[OK] python deps look present")
PY

declare -A MODEL_ID=(
  [bipcov_qwen3emb]="Qwen/Qwen3-Embedding-8B"
  [bipcov_nvembed]="nvidia/NV-Embed-v2"
  [bipcov_gteqwen2]="Alibaba-NLP/gte-Qwen2-7B-instruct"
  [bipcov_gritlm]="GritLM/GritLM-7B"
)

declare -A TRUST_RC=(
  [bipcov_qwen3emb]="1"
  [bipcov_nvembed]="1"
  [bipcov_gteqwen2]="0"
  [bipcov_gritlm]="1"
)

declare -A DOC_PREFIX=(
  [bipcov_qwen3emb]=""
  [bipcov_nvembed]=""
  [bipcov_gteqwen2]=""
  [bipcov_gritlm]=""
)

declare -A QUERY_PREFIX=(
  [bipcov_qwen3emb]=""
  [bipcov_nvembed]=""
  [bipcov_gteqwen2]=""
  [bipcov_gritlm]=""
)

echo "[STEP] compute embeddings (train+ref) for 4 backends $(date -Iseconds)"
launch_variant_emb() {
  local gpu="$1" method="$2"
  local model="${MODEL_ID[$method]}"
  local trust="${TRUST_RC[$method]}"
  local doc_prefix="${DOC_PREFIX[$method]}"
  local query_prefix="${QUERY_PREFIX[$method]}"

  local train_dir="${DATELM_ROOT}/embeddings/paper_seed42_v1_refpromptlabel/tulu3_train_${method}"
  local trust_flag=()
  if [[ "${trust}" == "1" ]]; then
    trust_flag=(--trust_remote_code)
  fi

  (
    for task in "${TASKS[@]}"; do
      ref_dir="${DATELM_ROOT}/embeddings/paper_seed42_v1_refpromptlabel/${task}_ref_${method}"
      if [[ -f "${ref_dir}/ref_emb.npy" ]]; then
        echo "[SKIP] ref_emb exists: ${ref_dir}/ref_emb.npy"
      else
        CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
          python "${SEQDATAVAL_ROOT}/finetuning/scripts/datelm_paper/compute_hf_embeddings.py" \
            --datelm_root "${DATELM_ROOT}" \
            --model_name "${model}" \
            "${trust_flag[@]}" \
            --model_dtype "${EMB_MODEL_DTYPE}" \
            --device cuda \
            --batch_size "${EMB_BATCH_SIZE}" \
            --max_length "${EMB_MAX_LENGTH}" \
            --out_dtype "${EMB_OUT_DTYPE}" \
            --pooling "${EMB_POOLING}" \
            --format tulu_chat \
            --query_prefix "${query_prefix}" \
            --no_train \
            --ref_dataset_name "${task}" \
            --ref_num_samples "${REF_NUM_SAMPLES}" \
            --ref_seed "${REF_SEED}" \
            --out_dir "${ref_dir}" \
          >"${LOGROOT}/${method}_ref_${task}.log" 2>&1
      fi
    done

    if [[ -f "${train_dir}/progress.json" ]] && \
      python -c 'import json,sys; j=json.load(open(sys.argv[1])); assert j.get("train_complete") is True' \
        "${train_dir}/progress.json" >/dev/null 2>&1; then
      echo "[SKIP] train embeddings already complete: ${train_dir}"
      exit 0
    fi

    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
      python "${SEQDATAVAL_ROOT}/finetuning/scripts/datelm_paper/compute_hf_embeddings.py" \
        --datelm_root "${DATELM_ROOT}" \
        --model_name "${model}" \
        "${trust_flag[@]}" \
        --model_dtype "${EMB_MODEL_DTYPE}" \
        --device cuda \
        --batch_size "${EMB_BATCH_SIZE}" \
        --max_length "${EMB_MAX_LENGTH}" \
        --out_dtype "${EMB_OUT_DTYPE}" \
        --pooling "${EMB_POOLING}" \
        --format tulu_chat \
        --doc_prefix "${doc_prefix}" \
        --messages_only_first_two \
        --no_ref \
        --train_jsonl "${TRAIN_JSONL}" \
        --out_dir "${train_dir}" \
      >"${LOGROOT}/${method}_train.log" 2>&1
  ) >"${LOGROOT}/${method}_driver.log" 2>&1 &
  LAUNCH_PID=$!
}

pids=()
launch_variant_emb 0 bipcov_qwen3emb
echo "[LAUNCH] gpu=0 method=bipcov_qwen3emb pid=${LAUNCH_PID}"
pids+=( "${LAUNCH_PID}" )

launch_variant_emb 1 bipcov_nvembed
echo "[LAUNCH] gpu=1 method=bipcov_nvembed pid=${LAUNCH_PID}"
pids+=( "${LAUNCH_PID}" )

launch_variant_emb 2 bipcov_gteqwen2
echo "[LAUNCH] gpu=2 method=bipcov_gteqwen2 pid=${LAUNCH_PID}"
pids+=( "${LAUNCH_PID}" )

launch_variant_emb 3 bipcov_gritlm
echo "[LAUNCH] gpu=3 method=bipcov_gritlm pid=${LAUNCH_PID}"
pids+=( "${LAUNCH_PID}" )

echo "[WAIT] embedding jobs: ${pids[*]}"
for pid in "${pids[@]}"; do
  wait "${pid}"
done
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
  local gpu="$1" method="$2" task="$3"
  local train_emb="${DATELM_ROOT}/embeddings/paper_seed42_v1_refpromptlabel/tulu3_train_${method}/train_emb.npy"
  local ref_emb="${DATELM_ROOT}/embeddings/paper_seed42_v1_refpromptlabel/${task}_ref_${method}/ref_emb.npy"
  local out_metric="${DATELM_ROOT}/scores/paper_seed42_v1_refpromptlabel/${task}/${method}_metrics.npy"
  mkdir -p "${DATELM_ROOT}/scores/paper_seed42_v1_refpromptlabel/${task}"

  if [[ -f "${out_metric}" ]]; then
    if check_metric_npy "${out_metric}"; then
      echo "[SKIP] existing metric ok: ${out_metric}"
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
    >"${LOGROOT}/bipcov_${method}_${task}.log" 2>&1
  end="$(date +%s)"
  elapsed="$((end-start))"
  ok="1"
  check_metric_npy "${out_metric}" || ok="0"
  echo "$(date -Iseconds),bipcov_score,${method},${task},${gpu},${elapsed},${ok}" >> "${TIMING_CSV}"
  if [[ "${ok}" != "1" ]]; then
    echo "[FATAL] invalid metric file: ${out_metric}" >&2
    tail -n 120 "${LOGROOT}/bipcov_${method}_${task}.log" >&2 || true
    exit 1
  fi
}

for task in "${TASKS[@]}"; do
  compute_bipcov_scores 0 bipcov_qwen3emb "${task}"
  compute_bipcov_scores 1 bipcov_nvembed "${task}"
  compute_bipcov_scores 2 bipcov_gteqwen2 "${task}"
  compute_bipcov_scores 3 bipcov_gritlm "${task}"
done

echo "[STEP] run DATE-LM LitGPT train->convert->official eval (12 jobs, 4 GPUs) $(date -Iseconds)"
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
        [[ -f "${metrics}" ]] || ok="0"
        echo "$(date -Iseconds),train_eval,${method},${task},${gpu},${elapsed},${ok}" >> "${TIMING_CSV}"
        if [[ "${ok}" != "1" ]]; then
          echo "[FATAL] missing metrics.json after job exit: ${metrics}" >&2
          echo "[FATAL] tail log: ${LOG_TABLE3}/launcher_${method}_${task}_gpu${gpu}.log" >&2
          tail -n 160 "${LOG_TABLE3}/launcher_${method}_${task}_gpu${gpu}.log" >&2 || true
          exit 1
        fi

        echo "[DONE] ${method} ${task} gpu=${gpu} elapsed_s=${elapsed}"
        gpu_pids[$gpu]=""
        gpu_job[$gpu]=""
      fi
    fi
  done

  if [[ "${job_idx}" -ge "${#JOBS[@]}" ]]; then
    [[ "${all_done}" == true ]] && break
  else
    fill_free_gpus
  fi
  sleep 60
done

echo "[DONE] all 12 jobs finished $(date -Iseconds)"

echo "[STEP] summarize results -> ${OUT_SUMMARY_DIR} $(date -Iseconds)"
cd "${SEQDATAVAL_ROOT}/finetuning/scripts/datelm_paper"
python summarize_multiseed_results.py \
  --datelm_root "${DATELM_ROOT}" \
  --results_dirname "${RESULTS_DIRNAME}" \
  --seeds "${TRAIN_SEED}" \
  --tasks mmlu gsm8k bbh \
  --methods random_avg rds_plus bipcov_bge bipcov_qwen3emb bipcov_nvembed bipcov_gteqwen2 bipcov_gritlm \
  --output_dir "${OUT_SUMMARY_DIR}"

echo "[FINAL] $(date -Iseconds) done"
