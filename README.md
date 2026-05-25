# SequentialDataVal

Reference implementation for **Unifying and Optimizing Data Values for
Selection via Sequential Decision-Making**.

This repository is intentionally code-first. It contains the algorithms,
experiment runners, plotting utilities, and reproducibility instructions needed
to regenerate the paper results. Large raw result trees, checkpoints, logs, and
paper build artifacts are not tracked in git.

## Installation

```bash
pip install -r requirements.txt
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate sequentialdataval
```

For development:

```bash
pip install -e ".[all]"
```

## Quick Check

```bash
bash scripts/smoke_test.sh
```

Run one OpenML experiment:

```bash
python src/main.py --dataset 2dplanes --seed 10 --config config/base_config.yaml
```

## Main Paper Experiments

### RQ1: Optimal Sequential Selection

```bash
python src/main.py --dataset digits --seed 10 --config config/rq1_config.yaml \
  --skip_bipartite --include_dp --dp_max_subset_size 20

bash scripts/run_rq1_dp.sh
python scripts/aggregate_rq1_dp.py
python scripts/plot_rq1_figure6.py \
  --results_root results_rq1_dp \
  --out_dir results/rq1_figure6
```

### RQ2: Utility Curvature

```bash
python scripts/run_rq2_curvature.py
```

### RQ3: Selection Curves

```bash
bash scripts/run_rq3.sh
```

Per-seed outputs are written to
`results/<dataset>/seed_<seed>/addition_experiment_results.csv`.

### Budget-500 Table

The paper protocol reports the mean over the top-k selection curve, not only
the endpoint at `k=500`.

```bash
python scripts/run_budget500_curve_mean_table.py \
  --curve_step 10 \
  --max_workers 6 \
  --out_dir results/selection_results_500_curve_mean

python scripts/compare_budget500_table3.py \
  --summary results/selection_results_500_curve_mean/summary_curve_mean.csv
```

Use `--max_workers 1` for `bbc-embeddings` on memory-constrained machines.

### Utility Approximation

```bash
cd utility_appro
pip install "cvxpy>=1.4.0"
bash run_experiments.sh
```

## DATE-LM Fine-Tuning Extension

The `finetuning/` directory contains the BipCov selection code and orchestration
scripts for DATE-LM-style LLM fine-tuning experiments. Full DATE-LM data,
selected-data archives, checkpoints, and evaluation logs are external artifacts;
see `finetuning/README.md` and `EXTERNAL_ARTIFACTS.md`.

## Repository Layout

```text
src/                         Core algorithms and OpenDataVal integration
config/                      Paper experiment configurations
scripts/                     CPU experiment, comparison, and plotting runners
utility_appro/               Utility approximation experiments
finetuning/                  DATE-LM/BipCov selection and fine-tuning scripts
requirements.txt             Python dependency list
environment.yml              Conda environment
REPRODUCIBILITY.md           Paper-result to command mapping
EXTERNAL_ARTIFACTS.md        Policy for large files kept outside git
```

## Datasets

The OpenML experiments use:

`2dplanes`, `nomao`, `bbc-embeddings`, `MiniBooNE`, `digits`, `election`,
`electricity`, and `fried`.

The DATE-LM extension uses Tulu3 instruction data, Llama-3.1-8B, and the MMLU,
GSM8K, and BBH evaluation tasks through the DATE-LM pipeline.

## Methods

Core methods:

- `BipartiteMatchingEvaluator`: bipartite graph approximation
- `DynamicProgrammingEvaluator`: exact optimal sequential selection

Baselines are integrated through OpenDataVal:

- Random, Leave-One-Out, InfluenceSubsample
- Data Shapley, Beta Shapley, Data Banzhaf
- AME, DVRL, Data-OOB

## Citation

```bibtex
@inproceedings{seqdataval2026,
  title={Unifying and Optimizing Data Values for Selection via Sequential Decision-Making},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```

## License

MIT. See `LICENSE`.
