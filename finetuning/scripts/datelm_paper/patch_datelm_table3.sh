#!/usr/bin/env bash
set -euo pipefail

# Apply small, non-semantic patches to DATE-LM checkout so Table-3 baselines can run on any machine.
#
# Patches:
#  1) methods/gradsim/probe_less_instruct_hf.py: use <lora_ckpt>/optimizer.bin instead of a hardcoded absolute path
#  2) minimal_multitask/eval/utils.py: ensure temperature=0 is treated as greedy (some HF/vLLM versions error otherwise)
#  3) train/model_utils.py: avoid device_map="auto" sharding for HF LoRA baselines (GradSim/LESS) to prevent multi-GPU tensor mismatches
#  4) train/model_utils.py: avoid slow mean/cov init when adding pad token (mean_resizing=False when supported)
#
# This script edits the DATE-LM working tree only; do NOT commit DATE-LM changes.

DATELM_ROOT="${1:-${DATELM_ROOT:-}}"
if [[ -z "${DATELM_ROOT}" ]]; then
  echo "Usage: $0 /path/to/DATE-LM-root (or export DATELM_ROOT)"
  exit 1
fi

LESS_FILE="${DATELM_ROOT}/methods/gradsim/probe_less_instruct_hf.py"
UTILS_FILE="${DATELM_ROOT}/minimal_multitask/eval/utils.py"
MODEL_UTILS_FILE="${DATELM_ROOT}/train/model_utils.py"

if [[ ! -f "${LESS_FILE}" ]]; then
  echo "Missing: ${LESS_FILE}"
  exit 1
fi
if [[ ! -f "${UTILS_FILE}" ]]; then
  echo "Missing: ${UTILS_FILE}"
  exit 1
fi
if [[ ! -f "${MODEL_UTILS_FILE}" ]]; then
  echo "Missing: ${MODEL_UTILS_FILE}"
  exit 1
fi

python - <<PY
from __future__ import annotations

from pathlib import Path

datelm_root = Path(r"${DATELM_ROOT}")
less_file = datelm_root / "methods/gradsim/probe_less_instruct_hf.py"
utils_file = datelm_root / "minimal_multitask/eval/utils.py"
model_utils_file = datelm_root / "train/model_utils.py"

less_txt = less_file.read_text()
if 'optimizer_path = os.path.join("/data/user_data/emilyx/lora_llama3", "optimizer.bin")' in less_txt:
    less_txt = less_txt.replace(
        '    optimizer_path = os.path.join("/data/user_data/emilyx/lora_llama3", "optimizer.bin")\n'
        '    adam_optimizer_state = torch.load(\n'
        '        optimizer_path, map_location="cpu")["state"]\n',
        '    if not lora_ckpt:\n'
        '        raise ValueError("LESS requires --lora_ckpt containing optimizer.bin")\n'
        '    optimizer_path = os.path.join(str(lora_ckpt), "optimizer.bin")\n'
        '    adam_optimizer_state = torch.load(optimizer_path, map_location="cpu")["state"]\n',
    )
    less_file.write_text(less_txt)
    print("[OK] patched LESS optimizer path:", less_file)
else:
    print("[OK] LESS optimizer path already patch-compatible:", less_file)

utils_txt = utils_file.read_text()
if "Handle temperature=0 for newer transformers versions" not in utils_txt:
    marker = "try:\n"
    if marker not in utils_txt:
        print("[WARN] Could not find insertion point for temperature=0 patch in:", utils_file)
    else:
        utils_txt = utils_txt.replace(
            marker,
            marker
            + "            # Handle temperature=0 for newer transformers versions\n"
            + "            if generation_kwargs.get(\"temperature\", None) == 0:\n"
            + "                generation_kwargs.pop(\"temperature\")\n"
            + "                generation_kwargs[\"do_sample\"] = False\n\n",
        )
        utils_file.write_text(utils_txt)
        print("[OK] patched temperature=0->greedy in:", utils_file)
else:
    print("[OK] temperature=0 patch already present:", utils_file)

mu_txt = model_utils_file.read_text()
mu_txt_orig = mu_txt
if 'device_map="auto"' in mu_txt or "device_map='auto'" in mu_txt:
    mu_txt = mu_txt.replace('device_map=\"auto\"', "device_map=None").replace("device_map='auto'", "device_map=None")
    print("[OK] patched HF model loading to avoid device_map=auto sharding in:", model_utils_file)
else:
    print("[OK] model_utils device_map already patch-compatible:", model_utils_file)

if "mean_resizing=False" not in mu_txt:
    target = "    base_model.resize_token_embeddings(len(tokenizer))\\n"
    if target in mu_txt:
        mu_txt = mu_txt.replace(
            target,
            "    try:\\n"
            "        base_model.resize_token_embeddings(len(tokenizer), mean_resizing=False)\\n"
            "    except TypeError:\\n"
            "        base_model.resize_token_embeddings(len(tokenizer))\\n",
        )
        print("[OK] patched resize_token_embeddings(mean_resizing=False) in:", model_utils_file)
    else:
        print("[WARN] Could not find resize_token_embeddings call to patch in:", model_utils_file)
else:
    print("[OK] mean_resizing patch already present:", model_utils_file)

if mu_txt != mu_txt_orig:
    model_utils_file.write_text(mu_txt)
PY

echo "Done."
