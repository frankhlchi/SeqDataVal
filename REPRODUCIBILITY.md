# Reproducibility Guide

This guide maps paper results to public commands. The classical OpenML results
use the NeurIPS / Joint LARCH-LACML run as the canonical protocol; the ICML
version reuses those classical results and adds the DATE-LM fine-tuning
extension.

## Result To Script Map

| Paper result | Entry point | Output |
|---|---|---|
| RQ1 exact-DP gap | `bash scripts/run_rq1_dp.sh`; `python scripts/aggregate_rq1_dp.py` | `results_rq1_dp/_summary/rq1_dp_gap_summary.csv` |
| RQ1 detailed Figure 6 | `python scripts/plot_rq1_figure6.py --results_root results_rq1_dp --out_dir results/rq1_figure6` | `results/rq1_figure6/appro_results_small_latest_*.pdf` |
| RQ2 curvature | `python scripts/run_rq2_curvature.py` | `results/curvature_rq2/rq2_curvature_summary.csv` |
| RQ3 selection curves | `bash scripts/run_rq3.sh` | `results/<dataset>/seed_<seed>/addition_experiment_results.csv` |
| Budget-500 table | `python scripts/run_budget500_curve_mean_table.py --curve_step 10 --out_dir results/selection_results_500_curve_mean` | `results/selection_results_500_curve_mean/summary_curve_mean.csv` |
| Budget-500 comparison | `python scripts/compare_budget500_table3.py --summary results/selection_results_500_curve_mean/summary_curve_mean.csv` | `results/selection_results_500_curve_mean/comparison_to_paper.csv` |
| Utility approximation | `cd utility_appro && bash run_experiments.sh` | `utility_appro/aggregate_results_overall.csv` |
| DATE-LM fine-tuning | `finetuning/scripts/datelm_paper/*` | DATE-LM metrics and summary CSVs |

## Data Cache Layout

OpenDataVal downloads/preprocesses datasets relative to the repository working
directory. For repeated runs, use a persistent `data_files/` directory at the
repo root:

```bash
mkdir -p data_files
```

`bbc-embeddings` is the largest OpenML dataset in the paper and may require
lower worker parallelism.

## Budget-500 Protocol

The Budget-500 table is a curve-mean metric:

```text
train_count = 500
standard datasets: valid=50, test=500
digits and bbc-embeddings: valid=100, test=1000
reported value = mean_k accuracy(top-k selected points), k=1..500
```

Use `scripts/run_budget500_curve_mean_table.py` for this table.

Sampling-based valuation methods use a matched retraining budget:

| Method | Budget parameter |
|---|---|
| InfluenceSubsample | `num_models=1000` |
| DataBanzhaf | `num_models=1000` |
| DataOob | `num_models=1000` |
| BipartiteMatchingEvaluator | `n_samples=1000` |
| AME | `num_models=ceil(1000/4)=250` |
| DataShapley + BetaShapley | shared `MonteCarloSampler(mc_epochs=ceil(1000/n_train))` |
| DVRL | configurable with `--dvrl_rl_epochs`; default runner uses `1000` |

## DATE-LM Protocol

The repository contains code for the DATE-LM extension but not heavyweight data
or checkpoints.

Core protocol:

```text
pool: 200k Tulu3 examples
selection budget: 10k examples
base model: meta-llama/Llama-3.1-8B
fine-tuning: LoRA rank 128, alpha 512, dropout 0.1
target modules: q/k/v/o
learning rate: 2e-5 with 3% warmup and cosine decay
effective batch size: 128
epochs: 2
max sequence length: 2048
precision: bf16
tasks: MMLU 0-shot, GSM8K 8-shot CoT EM, BBH 3-shot EM
train seeds: 42, 1337, 2025
```

See `finetuning/README.md` for setup and command examples.

## Clean Release Boundary

Keep in git:

- source code, configs, and runner scripts;
- small ground-truth tables needed by comparison scripts;
- concise documentation.

Keep outside git:

- full result trees and logs;
- OpenML caches;
- DATE-LM data, selected-data archives, embeddings, checkpoints, and raw eval
  outputs;
- paper PDFs, build outputs, and machine-specific notes.
