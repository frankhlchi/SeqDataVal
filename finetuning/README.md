# DATE-LM Fine-Tuning Extension

This directory contains the public code for applying bipartite coverage
selection to DATE-LM-style LLM fine-tuning. It includes selection code,
embedding utilities, and orchestration scripts. It does not include DATE-LM
data, model checkpoints, generated embeddings, selected-data archives, or raw
evaluation logs.

## Setup

Install the base repository first, then install GPU dependencies on the machine
that will run fine-tuning:

```bash
pip install -r finetuning/requirements-gpu.txt
```

Expected external inputs:

- DATE-LM checkout
- Tulu3 instruction pool
- Llama-3.1-8B checkpoint in the format expected by DATE-LM/LitGPT
- MMLU, GSM8K, and BBH evaluation data

Set paths explicitly:

```bash
export SEQDATAVAL_ROOT=/path/to/SeqDataVal
export DATELM_ROOT=/path/to/DATE-LM
```

Patch the DATE-LM checkout with the paper-compatible runner helpers:

```bash
bash "$SEQDATAVAL_ROOT/finetuning/scripts/datelm_paper/patch_datelm_table3.sh" "$DATELM_ROOT"
```

## Core Commands

Single-seed Table-3-style run:

```bash
bash "$SEQDATAVAL_ROOT/finetuning/scripts/datelm_paper/run_table3_single_seed_4gpu.sh" 1337
```

Multi-seed train/eval robustness:

```bash
python "$SEQDATAVAL_ROOT/finetuning/scripts/datelm_paper/run_multiseed_train_eval_only.py" \
  --datelm_root "$DATELM_ROOT" \
  --python python \
  --seeds 42 1337 2025 \
  --methods random rds_plus bipcov \
  --tasks mmlu gsm8k bbh \
  --mmlu_eval_batch_size 32 \
  --gsm_eval_batch_size 16 \
  --bbh_eval_batch_size 16 \
  --skip_verify \
  --cleanup_checkpoints
```

BipCov from precomputed embeddings:

```bash
python finetuning/bipcov/probe_bipcov_from_emb.py \
  --train_emb /path/to/train_emb.npy \
  --ref_emb /path/to/ref_emb.npy \
  --out /path/to/scores/bipcov_metrics.npy \
  --k_max 10000
```

## Protocol

```text
pool: 200k Tulu3 examples
selection budget: 10k examples
reference examples: 100 prompt+label examples per task
base model: meta-llama/Llama-3.1-8B
fine-tuning:
  LoRA rank 128, alpha 512, dropout 0.1
  target modules q/k/v/o
  lr 2e-5 with 3% warmup and cosine decay
  effective batch 128
  epochs 2
  max sequence length 2048
  bf16 precision
tasks: MMLU, GSM8K, BBH
train seeds: 42, 1337, 2025
```

Generated artifacts should be stored outside git and referenced by checksum.
