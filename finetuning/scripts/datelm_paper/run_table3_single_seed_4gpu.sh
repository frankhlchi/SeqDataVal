#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

SEED="${1:-1337}"

RESULTS_DIRNAME="${RESULTS_DIRNAME:-results_table3_single_seed_datelm}"
OUT_DIRNAME="${OUT_DIRNAME:-checkpoints_table3_single_seed_datelm}"
LOG_DIRNAME="${LOG_DIRNAME:-logs_table3_single_seed_datelm}"

USE_VLLM_GSM="${USE_VLLM_GSM:-1}"
USE_VLLM_BBH="${USE_VLLM_BBH:-1}"
CLEANUP_CHECKPOINTS="${CLEANUP_CHECKPOINTS:-1}"

RUNNER="${SEQDATAVAL_ROOT}/finetuning/scripts/datelm_paper/run_multiseed_train_eval_datelm_litgpt.py"
if [[ ! -f "${RUNNER}" ]]; then
  echo "Missing runner: ${RUNNER}"
  exit 1
fi

if [[ "${USE_VLLM_GSM}" == "1" ]] || [[ "${USE_VLLM_BBH}" == "1" ]]; then
  if ! "${PY_TRAIN_EVAL}" -c "import vllm" >/dev/null 2>&1; then
    echo "[warn] vllm is not installed in this environment; disabling USE_VLLM_GSM/USE_VLLM_BBH." >&2
    USE_VLLM_GSM=0
    USE_VLLM_BBH=0
  fi
fi

run_group() {
  local gpu="${1}"; shift
  local name="${1}"; shift
  local methods=("$@")

  local log_dir="${DATELM_ROOT}/${LOG_DIRNAME}"
  mkdir -p "${log_dir}"
  local launcher_log="${log_dir}/launcher_table3_${name}_gpu${gpu}.log"
  local pid_file="${log_dir}/launcher_table3_${name}_gpu${gpu}.pid"

  local extra=()
  if [[ "${USE_VLLM_GSM}" == "1" ]]; then
    extra+=("--gsm_use_vllm")
  fi
  if [[ "${USE_VLLM_BBH}" == "1" ]]; then
    extra+=("--bbh_use_vllm")
  fi
  if [[ "${CLEANUP_CHECKPOINTS}" == "1" ]]; then
    extra+=("--cleanup_checkpoints")
  fi

  echo "[launch] gpu=${gpu} name=${name} seed=${SEED} methods=${methods[*]}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 nohup "${PY_TRAIN_EVAL}" "${RUNNER}" \
    --datelm_root "${DATELM_ROOT}" \
    --python "${PY_TRAIN_EVAL}" \
    --seeds "${SEED}" \
    --tasks mmlu gsm8k bbh \
    --methods "${methods[@]}" \
    --results_dirname "${RESULTS_DIRNAME}" \
    --out_dirname "${OUT_DIRNAME}" \
    --log_dirname "${LOG_DIRNAME}" \
    "${extra[@]}" \
    > "${launcher_log}" 2>&1 &

  echo $! > "${pid_file}"
  echo "  pid: ${pid_file}"
  echo "  log: ${launcher_log}"
}

# Balanced split for 4 GPUs (9 methods total, Table-3 style random 1/2/3):
# - GPU0: random1 + bm25
# - GPU1: random2 + repsim
# - GPU2: random3 + rds_plus
# - GPU3: gradsim + less + bipcov
run_group 0 "g0_random1_bm25" random1 bm25
run_group 1 "g1_random2_repsim" random2 repsim
run_group 2 "g2_random3_rdsplus" random3 rds_plus
run_group 3 "g3_gradsim_less_bipcov" gradsim less bipcov

cat <<EOF

[monitor]
tail -f "${DATELM_ROOT}/${LOG_DIRNAME}/launcher_table3_"*"_gpu"*.log
watch -n 1 nvidia-smi

[progress] (number of completed metrics.json under results dir; expect 27 = 9 methods × 3 tasks)
find "${DATELM_ROOT}/${RESULTS_DIRNAME}" -name metrics.json | wc -l

EOF
