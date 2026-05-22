# Reproducibility Guide

This guide maps the paper results for **Unifying and Optimizing Data Values for
Selection via Sequential Decision-Making** to the public code entry points.

The classical OpenML results use the NeurIPS / Joint LARCH-LACML experiment
run as the canonical source. The ICML version inherits those classical results
and adds the DATE-LM fine-tuning extension.

## Result To Script Map

| Paper result | Entry point | Expected output | Runtime note |
|---|---|---|---|
| RQ1 exact-DP gap / Figure 6 | `bash scripts/run_rq1_dp.sh`; `python scripts/aggregate_rq1_dp.py` | `results_rq1_dp/_summary/rq1_dp_gap_summary.csv` | CPU-heavy because exact DP is exponential. |
| RQ1 detailed Figure 6 replot | `python scripts/plot_rq1_figure6.py --results_root results_rq1_dp --out_dir results/rq1_figure6` | `results/rq1_figure6/appro_results_small_latest_*.pdf` | Uses the NeurIPS/JLARCH-LACML final blue-to-red style. |
| RQ2 curvature | `python scripts/run_rq2_curvature.py` | `results/curvature_rq2/rq2_curvature_summary.csv` | CPU. |
| RQ3 selection curves | `bash scripts/run_rq3.sh` | `results/<dataset>/seed_<seed>/addition_experiment_results.csv` | CPU-heavy; `bbc-embeddings` is memory/throughput heavy. |
| Budget-500 table | `python scripts/run_budget500_curve_mean_table.py --curve_step 10 --out_dir results/selection_results_500_curve_mean` | `results/selection_results_500_curve_mean/summary_curve_mean.csv` | Paper protocol uses `train_count=500` and curve-mean accuracy. Use `--max_workers 1` for BBC if memory is tight. |
| Budget-500 paper comparison | `python scripts/compare_budget500_table3.py --summary results/selection_results_500_curve_mean/summary_curve_mean.csv` | `results/selection_results_500_curve_mean/comparison_to_paper.csv` | Expected criterion: all cells pass `abs(delta) <= max(0.015, paper_std / 2)`. |
| Utility approximation | `cd utility_appro && bash run_experiments.sh` | `utility_appro/aggregate_results_overall.csv` | Requires `cvxpy`. |
| DATE-LM fine-tuning | `finetuning/scripts/datelm_paper/*` | DATE-LM `metrics.json` plus summary CSVs | Requires H100/H200/A100-class GPU resources and external DATE-LM data/checkpoints. |

## Data Cache Layout

OpenDataVal writes downloaded/preprocessed datasets relative to the repository
working directory. For `bbc-embeddings`, make sure a writable `data_files/`
directory exists at the repo root before running RQ1/RQ3:

```bash
mkdir -p /path/to/persistent_data_files/bbc-embeddings
ln -s /path/to/persistent_data_files data_files
```

The camera-ready HCloud rerun used this layout:

```text
SeqDataVal/data_files -> /root/projects/SeqDataVal/data_files
SeqDataVal/data_files/bbc-embeddings/bbc-text.csv
SeqDataVal/data_files/bbc-embeddings/download_bbc_embed/
```

## Budget-500 Protocol

The paper Budget-500 table is **not** a single endpoint top-500 accuracy. It is
the mean over the top-k selection curve.

Protocol:

```text
train_count = 500
standard datasets: valid=50, test=500
digits and bbc-embeddings: valid=100, test=1000
reported value = mean_k accuracy(top-k selected points), k=1..500
```

Use `scripts/run_budget500_curve_mean_table.py` for the paper table.
`scripts/run_budget500_table.py` is retained only as an endpoint-style
diagnostic runner.

### Per-method retraining budget (≈1000 pred_model.fit() calls)

All sampling-based valuation methods are budgeted to ~1000 `pred_model.fit()`
calls so cross-method comparison is fair:

| Method | Budget parameter | Resulting fits |
|---|---|---:|
| InfluenceSubsample | `num_models=1000` | 1000 |
| DataBanzhaf | `num_models=1000` | ~1000 |
| DataOob | `num_models=1000` | 1000 |
| BipartiteMatchingEvaluator | `n_samples=1000` | 1000 |
| AME | `num_models=ceil(1000/4)=250` × 4 internal proportions | 1000 |
| DataShap + BetaShap (shared MC) | `mc_epochs=ceil(1000/n_train)` | up to ~1000 |
| **DVRL** | **`rl_epochs=1000`** (= OpenDataVal default) | **1003** (= 2 baseline + 1000 RL-loop fits on 32-point minibatches + 1 final) |
| LeaveOneOut | exact, no parameter | `n_train + 1` |
| Random | no retraining | 0 |

The release runner's `--dvrl_rl_epochs` default is **1000**. Earlier historical
runs used `rl_epochs=ceil(1000/32)=32`, which produces only 35 `pred_model.fit()`
calls — that under-trains the DVRL value-estimator and is the root cause of
DVRL's catastrophically unstable cells in older paper tables. The
`rl_epochs=1000` default matches OpenDataVal upstream and gives DVRL a fair
budget.

To reproduce the historical `rl_epochs=32` setting, pass `--dvrl_rl_epochs 32`
explicitly.

## DATE-LM / Fine-Tuning

The repository contains the selection and orchestration scripts for the
DATE-LM experiments, but the full DATE-LM working directory, model checkpoints,
and benchmark data are intentionally not committed to the core algorithm repo.

Release policy:

- keep `finetuning/scripts/datelm_paper/` and small summary CSVs in git;
- store large selected-data/eval artifacts in a release archive, Drive, or
  Zenodo with SHA256 checksums;
- document the exact artifact restore path and DATE-LM patch command in the
  fine-tuning release README.

Core DATE-LM protocol:

```text
pool: 200k Tulu3 examples
selection budget: 10k examples
base model: meta-llama/Llama-3.1-8B (base, not instruction-tuned)
fine-tuning:
  LoRA rank 128, alpha 512, dropout 0.1
  target modules q/k/v/o (no MLP, no head)
  lr 2e-5 with 3% linear warmup, then cosine annealing to 0
  effective batch 128 (micro 1 x grad accumulation 128, single device)
  epochs 2 (~156 steps), max_seq_length 2048, bf16 precision, no gradient clipping
  loss masking assistant-only (labels=-100 on instruction tokens)
  multi-turn conversations: keep first user-assistant turn only
tasks: MMLU 0-shot, GSM8K 8-shot CoT EM, BBH 3-shot EM (official minimal_multitask/eval)
main train seeds: 42, 1337, 2025
```

For full per-method retraining-budget details (DataOob, AME, DVRL, etc.), see
the "Per-method retraining budget" table in the Budget-500 section above.

## Clean Release Boundary

Recommended public package contents:

```text
SeqDataVal/
  README.md
  REPRODUCIBILITY.md
  PAPER_EXPERIMENT_INDEX.md
  config/
  src/
  scripts/
  utility_appro/
  finetuning/scripts/datelm_paper/
  finetuning/*summary*.csv
```

Do not commit:

- full OpenML result trees,
- DATE-LM checkpoints,
- full DATE-LM data directories,
- generated logs that are not needed to run the code,
- machine-specific credentials or Hugging Face tokens.

For camera-ready verification, keep a separate evidence archive with raw
reruns, Google Drive/Vast.ai artifacts, manifests, and checksums.
