# DATE-LM Table 3 reproduction analysis (single seed=1337) + BipCov

This note analyzes our reproduction of **DATE-LM Table 3** (arXiv:2507.09424) under the **DATE-LM official fine-tuning + eval pipeline**, and positions **BipCov** (SeqDataVal) as an additional method in the same pipeline.

## What was reproduced

- **Paper target**: DATE-LM Table 3 caption: “Data attribution method evaluation for single-task instruction fine-tuning (on MMLU, GSM8K, BBH), with Llama3 8B and LoRA.”
  - Paper LaTeX source: `finetuning/arXiv-2507.09424v2.tar.gz` contains `tables/wrap-table-finetune.tex`.
- **Training**: DATE-LM **LitGPT/Lightning** entry `DATE-LM/train/finetune.py`.
- **Eval**: DATE-LM **official eval** `minimal_multitask` for **MMLU / GSM8K / BBH**.
- **Seed**: train seed = **1337** (single-seed run; no mean±std over training seeds here).
- **Methods**:
  - Baselines (Table 3): `random1/2/3` (+ `random_avg`), `bm25`, `repsim`, `rds_plus`, `gradsim`, `less`
  - Ours: `bipcov` (**ref-aligned prompt+label reference**, tag `paper_seed42_v1_refpromptlabel`)

## Where the evidence lives (in this repo)

- Paper-ready summary:
  - `finetuning/table3_single_seed_datelm/MULTISEED_TRAINSEED_RESULTS.md`
  - `finetuning/table3_single_seed_datelm/MULTISEED_TRAINSEED_RESULTS.csv`
- Small reproducibility artifacts (no data/checkpoints/logs):
  - `finetuning/artifacts/datelm_table3_single_seed_vast_20260106/results_table3_single_seed_datelm/**/metrics.json` (27 files)
  - `finetuning/artifacts/datelm_table3_single_seed_vast_20260106/scores/**/_metrics.npy` (27 files)
  - `finetuning/artifacts/datelm_table3_single_seed_vast_20260106/MANIFEST.sha256` (sha256 checks)
- Run provenance (sanitized): `RUN_LEDGER.md`

## Our reproduced Table 3 (Vast, seed=1337)

Numbers are **percent**; MMLU uses `average_acc`, GSM8K uses `exact_match`, BBH uses `average_exact_match`.

| Method | MMLU | GSM8K | BBH | Avg |
|---|---:|---:|---:|---:|
| random1 | 61.46 | 58.53 | 67.11 | 62.37 |
| random2 | 59.47 | 59.59 | 66.15 | 61.74 |
| random3 | 60.23 | 60.80 | 65.94 | 62.32 |
| random_avg | 60.39 | 59.64 | 66.40 | 62.14 |
| bm25 | 59.85 | 58.98 | 62.63 | 60.49 |
| repsim | 61.42 | 58.45 | 66.20 | 62.03 |
| rds_plus | 60.63 | 61.41 | 66.23 | 62.75 |
| gradsim | 62.07 | 56.71 | 64.44 | 61.07 |
| less | 61.10 | 57.77 | 64.68 | 61.18 |
| **bipcov (ref-aligned)** | **63.38** | 59.29 | 66.08 | **62.92** |

### Takeaway (within our reproduced pipeline)

- **BipCov is best on Avg** (+0.78 over `random_avg`, +0.89 over `repsim`, +0.17 over `rds_plus`).
- The gain is driven by **MMLU**: `bipcov` is +2.99 over `random_avg` on MMLU, but slightly below `random_avg` on GSM8K (-0.35) and BBH (-0.32).

## Comparison to DATE-LM paper Table 3 numbers

Paper numbers below are read from `tables/wrap-table-finetune.tex` and compared to our reproduced run.
Columns are **ours − paper** in percentage points.

| Method | ΔMMLU | ΔGSM8K | ΔBBH | ΔAvg |
|---|---:|---:|---:|---:|
| random1 | +0.56 | -0.57 | +1.81 | +0.60 |
| random2 | +0.07 | -0.51 | -0.15 | -0.20 |
| random3 | +0.03 | +1.20 | +0.64 | +0.62 |
| random_avg | +0.19 | +0.04 | +0.80 | +0.34 |
| bm25 | +0.35 | -1.22 | +0.13 | -0.24 |
| repsim | +0.22 | -0.75 | +0.30 | -0.07 |
| rds_plus | -1.77 | +1.81 | -0.67 | -0.22 |
| gradsim | +3.67 | -1.09 | -1.06 | +0.51 |
| less | +1.10 | -1.73 | +0.48 | -0.05 |

### What this means for claims

- The **overall baseline level** is close (e.g., `random_avg` and `repsim` are within ~0.3–0.1 Avg points), but some methods differ substantially.
- After upgrading `rds_plus` to the **paper-faithful** definition (position-weighted mean pooling), **RDS+ is close on Avg** (Avg -0.22 vs paper), with **task-level offsets** (MMLU lower, GSM8K higher, BBH slightly lower).
- For our method, **BipCov Avg=62.92** is close to the paper’s **RDS+ Avg=62.97** (difference -0.05), but it is safer to phrase as “comparable magnitude under the DATE-LM pipeline” rather than “strictly better than paper RDS+”, because (i) paper may have minor code/data/version differences, (ii) `random1/2/3` seeds in the paper are not explicitly specified.

## RDS+ baseline: proxy vs paper-faithful

Timeline:

- The artifacts under `finetuning/artifacts/datelm_table3_single_seed_vast_20260106/` correspond to the **initial** Vast Table-3 run.
- The `rds_plus` score files in that folder were produced by a **proxy** (Rep-Sim + Gumbel) and are now **superseded**.
- We later implemented and re-ran the **paper-faithful RDS+** (position-weighted mean pooling), and updated
  `finetuning/table3_single_seed_datelm/MULTISEED_TRAINSEED_RESULTS.md`.

The DATE-LM paper defines RDS+ differently from Rep-Sim:

- Rep-Sim: cosine similarity between **last-token last-layer** hidden states
- RDS+: cosine similarity between **position-weighted mean pool of last-layer hidden states over all tokens**

See `finetuning/arXiv-2507.09424v2.tar.gz` → `sections/appendix_data_attribution_methods.tex`.

Paper-faithful RDS+ implementation in this repo:

- Embeddings: `finetuning/scripts/datelm_paper/compute_llm_last_token_embeddings.py --pooling weighted_mean`
- Scoring: `finetuning/scripts/datelm_paper/compute_repsim_metrics_from_emb.py`
- Rerun RDS+ only: `finetuning/scripts/datelm_paper/rdsplus_weightedmean_pipeline.sh`

Historical proxy (for reproducing older runs; **not** paper-faithful):

- `finetuning/scripts/datelm_paper/compute_rds_plus_metrics.py` (Rep-Sim + Gumbel)
