# Multi-Seed (Train Seed) Robustness — Extra Baselines

This folder is reserved for the **extra-baselines** train-seed robustness summaries:

- Methods: `bm25`, `repsim`, `repsim_v2`, `rds_plus_v2`
- Seeds: `42, 1337, 2025`
- Tasks: `mmlu`, `gsm8k`, `bbh`
- Protocol: fixed `selected_data` only → LoRA train → merge → DATE-LM official eval

Generate summaries **here** to avoid overwriting the core-27 results in `finetuning/`:

```bash
cd "$SEQDATAVAL_ROOT"
"$PY_TRAIN_EVAL" finetuning/scripts/datelm_paper/summarize_multiseed_results.py \
  --datelm_root "$DATELM_ROOT" \
  --methods bm25 repsim repsim_v2 rds_plus_v2 \
  --output_dir "$SEQDATAVAL_ROOT/finetuning/multiseed_extra_baselines"
```

Expected outputs:
- `MULTISEED_TRAINSEED_RESULTS.md`
- `MULTISEED_TRAINSEED_RESULTS.csv`

Do **NOT** commit:
- `$DATELM_ROOT/results/`, `$DATELM_ROOT/logs/`, `$DATELM_ROOT/checkpoints/`, models, or any token/env secrets.
