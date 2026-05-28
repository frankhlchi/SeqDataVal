# SequentialDataVal

[![arXiv](https://img.shields.io/badge/arXiv-2502.04554-b31b1b.svg)](https://arxiv.org/abs/2502.04554)
![ICML 2026 Spotlight](https://img.shields.io/badge/ICML%202026-Spotlight-blue)

Official implementation for **Unifying and Optimizing Data Values for
Selection via Sequential Decision-Making**. Accepted to **ICML 2026** as a **Spotlight Paper**.

This repository contains the algorithms, configs, runners, and
plotting/comparison scripts used for the paper experiments. Large datasets,
model checkpoints, generated embeddings, and full run outputs are distributed
separately when needed.

## Installation

Use Python 3.9 for the OpenDataVal experiments. The paper code was tested with
OpenDataVal 1.3.0, which is not compatible with Python 3.12.

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

Tested core stack: Python 3.9, OpenDataVal 1.3.0, NumPy 1.25, pandas 1.5,
scikit-learn 1.2+, PyTorch 2.1.2. Utility approximation additionally needs
`cvxpy`; DATE-LM fine-tuning additionally needs the GPU dependencies listed in
`finetuning/requirements-gpu.txt`.

## Quick Check

```bash
bash scripts/smoke_test.sh
python src/main.py --dataset 2dplanes --seed 10 --config config/base_config.yaml
```

## Paper Experiment Entry Points

RQ1 exact dynamic-programming gap and detailed RQ1 Figure 6:

```bash
python src/main.py --dataset digits --seed 10 --config config/rq1_config.yaml \
  --skip_bipartite --include_dp --dp_max_subset_size 20

bash scripts/run_rq1_dp.sh
python scripts/aggregate_rq1_dp.py
python scripts/plot_rq1_figure6.py \
  --results_root results_rq1_dp \
  --out_dir results/rq1_figure6
```

RQ2 curvature:

```bash
python scripts/run_rq2_curvature.py
```

RQ3 selection curves:

```bash
bash scripts/run_rq3.sh
```

Per-seed outputs are written under
`results/<dataset>/seed_<seed>/addition_experiment_results.csv`.
The full RQ3 script runs 8 datasets x 20 seeds and is CPU-heavy; use
`scripts/smoke_test.sh` first, or edit the `datasets`/`seeds` arrays in
`scripts/run_rq3.sh` for a short verification run.

Budget-500 table:

```bash
python scripts/run_budget500_curve_mean_table.py \
  --curve_step 10 \
  --max_workers 6 \
  --out_dir results/selection_results_500_curve_mean

python scripts/compare_budget500_table3.py \
  --summary results/selection_results_500_curve_mean/summary_curve_mean.csv
```

Budget-500 uses `train_count=500`; standard datasets use `valid/test=50/500`,
while `digits` and `bbc-embeddings` use `valid/test=100/1000`. The reported
value is the mean over the top-k selection curve, not only the endpoint at
`k=500`. Use `--max_workers 1` for `bbc-embeddings` on memory-constrained
machines.

Utility approximation:

```bash
cd utility_appro
pip install "cvxpy>=1.4.0"
bash run_experiments.sh
```

## DATE-LM Fine-Tuning Extension

The `finetuning/` directory contains the BipCov selection code and DATE-LM-style
orchestration scripts. Running these experiments requires external DATE-LM
data, model checkpoints, and benchmark files.

Prepare the following inputs before launching the fine-tuning pipeline:

- DATE-LM checkout
- Tulu3 instruction pool
- Llama-3.1-8B checkpoint in the format expected by DATE-LM/LitGPT
- MMLU, GSM8K, and BBH evaluation data

Typical setup on a GPU machine:

```bash
pip install -r finetuning/requirements-gpu.txt
export SEQDATAVAL_ROOT=/path/to/SeqDataVal
export DATELM_ROOT=/path/to/DATE-LM

bash "$SEQDATAVAL_ROOT/finetuning/scripts/datelm_paper/patch_datelm_table3.sh" \
  "$DATELM_ROOT"
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

DATE-LM protocol used in the paper: 200k Tulu3 pool, 10k selected examples,
100 prompt+label reference examples per task, Llama-3.1-8B base model, LoRA
rank 128 / alpha 512 / dropout 0.1 on q/k/v/o modules, learning rate `2e-5`,
effective batch size 128, 2 epochs, max sequence length 2048, bf16 precision,
and train seeds `42`, `1337`, `2025`.

## Repository Layout

```text
src/                         Core algorithms and OpenDataVal integration
config/                      Paper experiment configurations
scripts/                     CPU experiment, comparison, and plotting runners
utility_appro/               Utility approximation experiments
finetuning/                  DATE-LM/BipCov selection and fine-tuning scripts
requirements.txt             Python dependencies
environment.yml              Conda environment
```

OpenDataVal downloads/preprocesses datasets relative to the working directory.
For repeated runs, create a persistent repo-root cache:

```bash
mkdir -p data_files
```

## Datasets And Methods

OpenML datasets: `2dplanes`, `nomao`, `bbc-embeddings`, `MiniBooNE`, `digits`,
`election`, `electricity`, and `fried`.

Core methods:

- `BipartiteMatchingEvaluator`: bipartite graph approximation
- `DynamicProgrammingEvaluator`: exact optimal sequential selection

OpenDataVal baselines:

- Random, Leave-One-Out, InfluenceSubsample
- Data Shapley, Beta Shapley, Data Banzhaf
- AME, DVRL, Data-OOB

## Citation

```bibtex
@article{chi2025unifying,
  title={Unifying and Optimizing Data Values for Selection via Sequential-Decision-Making},
  author={Chi, Hongliang and Wu, Qiong and Zhou, Zhengyi and Light, Jonathan and Dodwell, Emily and Ma, Yao},
  journal={arXiv preprint arXiv:2502.04554},
  year={2025}
}
```

## License

MIT. See `LICENSE`.
