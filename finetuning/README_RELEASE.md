# DATE-LM Fine-Tuning Release Notes

This directory contains the public scripts and summaries for the ICML DATE-LM
fine-tuning extension. The full DATE-LM working directory, model checkpoints,
benchmark data, and generated logs are not part of the clean algorithm repo.

## What This Repo Should Contain

Keep in git:

- `finetuning/scripts/datelm_paper/`
- `finetuning/table3_single_seed_datelm/`
- `finetuning/table3_bipcov_emb_ablation_datelm/`
- `finetuning/table3_bipcov_emb_ablation_big4_datelm/`
- small summary CSV/Markdown files
- small manifests/checksum files for external artifacts

Do not commit:

- full DATE-LM data directories,
- Llama/LitGPT checkpoints,
- LoRA checkpoints,
- raw full evaluation logs,
- Hugging Face tokens or machine-specific credentials.

## Canonical ICML Result

The main ICML fine-tuning claim is the train-seed robustness result over
seeds `42`, `1337`, and `2025`:

| Method | Avg |
|---|---:|
| Random | `62.82 ± 0.31` |
| RDS+ | `62.88 ± 0.25` |
| BipCov ref-aligned | `63.89 ± 0.34` |

The local camera-ready evidence lives outside this repo at:

```text
/root/seq_reproduce/reproduction_bundle/summaries/datelm_multiseed_core.csv
/root/seq_reproduce/reproduction_bundle/summaries/datelm_multiseed_overall.csv
/root/seq_reproduce/reproduction_checks/google_drive_investigation/DATELM_VASTAI_DRIVE_PROVENANCE.md
```

## External Artifact Bundle

For a fresh H100/H200 rerun, use the prepared transfer bundle:

```text
/root/seq_reproduce/transfer_bundles/datelm_h200_finetuning_20260514
/root/seq_reproduce/transfer_bundles/datelm_h200_finetuning_20260514.zip
/root/seq_reproduce/transfer_bundles/datelm_h200_finetuning_20260514.tar.gz
```

Current ZIP SHA256:

```text
7b9ca514e7404316caa8626d397e0a8682f8e3376d5babc6adb6ecc63461f1c5
```

The bundle includes:

- `NEW_SERVER_H200_H100_RUNBOOK.md`
- `ARTIFACTS_AND_RESTORE.md`
- copied `finetuning/scripts/datelm_paper/`
- selected/eval artifact tarballs
- summary CSVs
- file manifests and checksums

It intentionally excludes the full DATE-LM data directory and Llama checkpoint.

## Minimal New-Server Flow

On a GPU server:

```bash
export SEQDATAVAL_ROOT=/path/to/SeqDataVal
export DATELM_ROOT=/path/to/DATE-LM

bash "$SEQDATAVAL_ROOT/finetuning/scripts/datelm_paper/patch_datelm_table3.sh" "$DATELM_ROOT"
```

Single-seed Table-3-style rerun:

```bash
export USE_VLLM_GSM=1
export USE_VLLM_BBH=1
export CLEANUP_CHECKPOINTS=1

bash "$SEQDATAVAL_ROOT/finetuning/scripts/datelm_paper/run_table3_single_seed_4gpu.sh" 1337
```

Core multi-seed robustness rerun:

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

## Protocol Notes

```text
pool: 200k Tulu3 instructions (seed 42; multi-turn conversations: keep first user-assistant turn only)
selection budget: 10k
reference examples: 100 prompt+label examples per task (seed 42)
base model: meta-llama/Llama-3.1-8B (base, not instruction-tuned)
fine-tuning:
  LoRA rank 128, alpha 512, dropout 0.1
  target modules q/k/v/o (LitGPT: lora_query=lora_key=lora_value=lora_projection=True; mlp=head=False)
  lr 2e-5 with linear warmup (3% of steps), then cosine annealing to 0
  effective batch 128 (micro batch 1 x grad accumulation 128, single device)
  epochs 2 (~156 steps)
  max_seq_length 2048 tokens (truncated)
  precision bf16 (full bf16; no mixed precision; no gradient clipping)
  loss masking assistant-only (labels=-100 on instruction tokens)
tasks: MMLU 0-shot, GSM8K 8-shot CoT EM, BBH 3-shot EM (official minimal_multitask/eval)
seeds: 42, 1337, 2025 (train-seed multi-seed); 1337 only for single-seed Table 3 + ablations
hardware (headline runs):
  multi-seed core (Random/RDS+/BipCov x 3 seeds) = Vast.ai 1xH200 143GB
  single-seed Table 3 + all embedding ablations          = Vast.ai 4xH100 NVL 94GB
  extra baselines seed=42 (bm25/repsim_v2/rds_plus_v2)   = on-prem A100 80GB
```

The prompt+label reference set follows DATE-LM's benchmark-native Table 3
protocol. Document this explicitly when reporting results because it is a
benchmark property and can look like label leakage without context.
