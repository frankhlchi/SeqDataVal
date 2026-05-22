# Extra-baselines train-seed robustness (old server copy)

This folder is for an **optional** duplicate summary generated on the old server
(`idea-node-06`, A100 80GB), to keep provenance and allow cross-checks **without**
overwriting the Vast summary.

Expected outputs (commit only these summaries):
- `MULTISEED_TRAINSEED_RESULTS.md`
- `MULTISEED_TRAINSEED_RESULTS.csv`

Command (old server example):

```bash
export SEQDATAVAL_ROOT=/home/chih3/sequence_dv/code/SequentialDataVal
export DATELM_ROOT=/home/chih3/sequence_dv/DATE-LM/DATE-LM-main
export PY_TRAIN_EVAL=/home/chih3/.conda/envs/llama_ft/bin/python

cd "$SEQDATAVAL_ROOT"
"$PY_TRAIN_EVAL" finetuning/scripts/datelm_paper/summarize_multiseed_results.py \
  --datelm_root "$DATELM_ROOT" \
  --methods bm25 repsim repsim_v2 rds_plus_v2 \
  --seeds 42 1337 2025 \
  --tasks mmlu gsm8k bbh \
  --output_dir "$SEQDATAVAL_ROOT/finetuning/multiseed_extra_baselines_oldserver"
```

Do **NOT** commit:
- `$DATELM_ROOT/results/`, `$DATELM_ROOT/logs/`, `$DATELM_ROOT/checkpoints/`, models, or any token/env secrets.

