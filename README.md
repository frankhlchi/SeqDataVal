# SequentialDataVal

Official implementation of **"Unifying and Optimizing Data Values for Selection via Sequential Decision-Making"** (ICML 2026).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository provides code to reproduce all experiments in the paper. We formulate data selection as a sequential decision-making problem and show that:

1. **Optimal selection** can be computed via dynamic programming (DP)
2. **Existing methods** (Data Shapley, Beta Shapley, Banzhaf) are myopic approximations
3. **Bipartite graph approximation** offers efficient and effective data selection with theoretical guarantees

For a paper-to-code artifact map (sections/figures/tables -> scripts/configs/outputs), see `PAPER_EXPERIMENT_INDEX.md`.
For the fuller ICML + NeurIPS/JLARCH-LACML release-code crosswalk, see `PAPER_RELEASE_CROSSWALK.md`.
For a concise reproduction checklist and clean-release boundary, see `REPRODUCIBILITY.md`.
For large artifacts that are intentionally not tracked in GitHub, see `EXTERNAL_ARTIFACTS.md`.
For H100/H200 DATE-LM rerun notes, see `finetuning/README_RELEASE.md`.

## Installation

### Option 1: pip (recommended)

```bash
pip install -r requirements.txt
```

### Option 2: conda

```bash
conda env create -f environment.yml
conda activate sequentialdataval
```

### Option 3: Development install

```bash
pip install -e ".[all]"
```

## Quick Start

### Smoke Test

Verify the installation with a quick sanity check:

```bash
bash scripts/smoke_test.sh
```

### Run a Single Experiment

```bash
python src/main.py --dataset 2dplanes --seed 10 --config config/base_config.yaml
```

## Reproducing Paper Results

### RQ1: Optimality Gap Analysis (Table 1)

Compares existing data valuation methods against the optimal DP solution.

```bash
# Single dataset (for testing)
python src/main.py --dataset digits --seed 10 --config config/rq1_config.yaml \
    --skip_bipartite --include_dp --dp_max_subset_size 20

# Full experiment (computationally intensive)
bash scripts/run_rq1_dp.sh

# Aggregate results
python scripts/aggregate_rq1_dp.py

# Replot the NeurIPS/JLARCH-LACML detailed RQ1 Figure 6 style
python scripts/plot_rq1_figure6.py \
  --results_root results_rq1_dp \
  --out_dir results/rq1_figure6
```

**Output:** `results_rq1_dp/_summary/rq1_dp_gap_summary.csv`

### RQ2: Curvature Impact Analysis (Figure 2)

Examines how utility curvature affects game-theoretic data valuation methods.

```bash
python scripts/run_rq2_curvature.py
```

**Output:** `results/curvature_rq2/rq2_curvature_summary.csv`

### RQ3: Selection Curve Evaluation (Figure 3)

Evaluates selection performance across 8 datasets with 20 seeds each.

```bash
bash scripts/run_rq3.sh
```

**Output:**
- Per-seed results: `results/<dataset>/seed_<seed>/addition_experiment_results.csv`
- Aggregated plots: `results/<dataset>/aggregate/*.png`

### Budget-500 Table

Large-scale selection table with the paper's curve-mean protocol:
`train_count=500`; standard datasets use `valid/test=50/500`, while
`digits` and `bbc-embeddings` use `valid/test=100/1000`. The reported value is
the mean accuracy over the top-k selection curve, not only the endpoint at
`k=500`.

```bash
python scripts/run_budget500_curve_mean_table.py \
  --curve_step 10 \
  --max_workers 6 \
  --out_dir results/selection_results_500_curve_mean

python scripts/compare_budget500_table3.py \
  --summary results/selection_results_500_curve_mean/summary_curve_mean.csv
```

For `bbc-embeddings`, use `--max_workers 1` if memory is limited.

**Output:** `results/selection_results_500_curve_mean/summary_curve_mean.csv`

### Utility Approximation (Table 3)

Compares utility approximation quality of different methods.

```bash
cd utility_appro
pip install cvxpy  # if not installed
bash run_experiments.sh
```

**Output:** `utility_appro/aggregate_results_overall.csv`

## Project Structure

```
SequentialDataVal/
├── src/
│   ├── main.py                 # Main entry point
│   ├── evaluators/
│   │   ├── bipartite.py        # Bipartite graph method (proposed)
│   │   └── dynamic.py          # Dynamic programming (optimal)
│   └── utils/
│       ├── plotting.py         # Visualization utilities
│       └── opendataval_compat.py  # OpenML compatibility patch
├── scripts/
│   ├── run_rq1_dp.sh           # RQ1 experiments
│   ├── plot_rq1_figure6.py     # NeurIPS/JLARCH-LACML Figure 6 replot
│   ├── run_rq2_curvature.py    # RQ2 experiments
│   ├── run_rq3.sh              # RQ3 experiments
│   ├── run_budget500_table.py  # Endpoint-style diagnostic large-budget runner
│   └── run_budget500_curve_mean_table.py  # Paper Budget-500 table runner
├── config/
│   ├── base_config.yaml        # Standard settings (50/50/500)
│   ├── rq1_config.yaml         # RQ1 DP settings (20/100/500)
│   ├── rq3_large_datasets.yaml # Large dataset settings (100/100/1000)
│   └── large_config.yaml       # Endpoint-style large-run config; not Budget-500 Table
├── utility_appro/              # Utility approximation experiments
├── finetuning/                 # LLM fine-tuning experiments (extended)
├── requirements.txt
├── environment.yml
└── pyproject.toml
```

## Datasets

We use 8 datasets from OpenML, following standard data valuation benchmarks:

| Dataset | Type | Features | Classes |
|---------|------|----------|---------|
| 2dplanes | Synthetic | 10 | 2 |
| nomao | Tabular | 118 | 2 |
| bbc-embeddings | Text | 1000 | 5 |
| MiniBooNE | Physics | 50 | 2 |
| digits | Image | 64 | 10 |
| election | Tabular | 16 | 2 |
| electricity | Time Series | 8 | 2 |
| fried | Synthetic | 10 | 2 |

## Methods

### Proposed Methods

- **Bipartite**: Bipartite graph-based selection with theoretical guarantees
- **DynamicProgramming**: Optimal selection via exact DP (for verification)

### Baselines (via OpenDataVal)

- Data Shapley, Beta Shapley, Data Banzhaf
- AME, DVRL, Data-OOB
- Leave-One-Out, Influence Function
- Random

## Extended Experiments: LLM Fine-tuning Data Selection

The `finetuning/` directory contains extended experiments for applying our bipartite coverage method to LLM fine-tuning data selection, following the [DATE-LM](https://github.com/DataAttributionEval/DATE-LM) framework.
For release-specific artifact boundaries and H100/H200 rerun commands, see
`finetuning/README_RELEASE.md`.

### Motivation

While our main experiments focus on traditional ML models, the bipartite graph method naturally extends to large-scale LLM fine-tuning scenarios where:
- Training pool is large (e.g., 200K instruction samples from Tulu3)
- We want to select a subset that maximizes coverage of target tasks (MMLU, GSM8K, BBH)

### Pipeline Overview

```
Step 1: Set up DATE-LM environment
Step 2: Download Tulu3 training data
Step 3: Download Llama-3.1-8B model
Step 4: Download evaluation data (MMLU/GSM8K/BBH)
Step 5: Compute embeddings (bipcov)
Step 6: Run scoring method (bipcov)
Step 7: Data selection (top-10K)
Step 8: LoRA fine-tune (bipcov vs random)
Step 9: Evaluate and compare results
```

### Quick Start

```bash
# Install additional dependencies
pip install sentence-transformers transformers

# Run the full pipeline (requires GPU)
cd finetuning
bash run_bipcov_pipeline.sh mmlu 200000 100
```

### Key Components

| File | Description |
|------|-------------|
| `compute_embeddings_for_datelm.py` | Compute BGE embeddings for train/ref data |
| `bipcov/probe_bipcov_from_emb.py` | Bipartite greedy coverage selection |
| `run_bipcov_pipeline.sh` | End-to-end pipeline script |

### Usage Example

1. **Compute Embeddings**
   ```bash
   python finetuning/compute_embeddings_for_datelm.py \
       --train_jsonl /path/to/tulu3.jsonl \
       --ref_hf cais/mmlu --ref_hf_subset all --ref_hf_split test \
       --out_dir embeddings/mmlu \
       --model BAAI/bge-large-en-v1.5 \
       --device cuda
   ```

2. **Run Bipartite Coverage Selection**
   ```bash
   python finetuning/bipcov/probe_bipcov_from_emb.py \
       --train_emb embeddings/mmlu/train_emb.npy \
       --ref_emb embeddings/mmlu/ref_emb.npy \
       --out scores/mmlu_bipcov \
       --k_max 10000
   ```

3. **Integrate with DATE-LM**
   ```bash
   # Output metrics.npy is compatible with DATE-LM fine-tuning
   python train/finetune.py --metric_path scores/mmlu_bipcov/metrics.npy
   ```

### File Structure

```
finetuning/
├── compute_embeddings_for_datelm.py  # Embedding computation
├── bipcov/
│   ├── probe_bipcov_from_emb.py      # Bipartite coverage selection
│   └── README.md                      # Detailed documentation
├── bipartite_greedy/                  # Alternative implementation
├── run_bipcov_pipeline.sh            # End-to-end pipeline
├── requirements-cpu.txt              # CPU-only dependencies
└── requirements-gpu.txt              # GPU dependencies
```

## Configuration

All experiments use YAML configuration files. Key parameters:

```yaml
experiment:
  train_count: 50      # Training set size
  valid_count: 50      # Validation set size
  test_count: 500      # Test set size
  device: "cpu"        # "cpu" or "cuda"

model:
  name: "sklogreg"     # sklearn LogisticRegression
```

## Troubleshooting

### OpenML Dataset Loading Errors

If you encounter errors loading OpenML datasets, our code automatically applies a compatibility patch. If issues persist, manually apply:

```python
from src.utils.opendataval_compat import patch_opendataval_openml
patch_opendataval_openml()
```

### Memory Issues with DP

The dynamic programming method has exponential complexity. For large datasets:
- Use `--dp_max_subset_size 20` to limit subset enumeration
- Run on high-memory machines
- Consider the bipartite approximation instead

## Citation

If you find this code useful, please cite our paper:

```bibtex
@inproceedings{sequentialdataval2026,
  title={Unifying and Optimizing Data Values for Selection via Sequential Decision-Making},
  author={Anonymous},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OpenDataVal](https://github.com/opendataval/opendataval) for baseline implementations
- [DATE-LM](https://github.com/DataAttributionEval/DATE-LM) for LLM fine-tuning framework
