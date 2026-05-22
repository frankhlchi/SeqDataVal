# Paper Experiment ↔ Code Index (SequentialDataVal)

This repo contains the **implementation + reproduction scripts** for the paper:
**“Unifying and Optimizing Data Values for Selection via Sequential Decision-Making”** (ICML 2026).

The paper source in this workspace lives at:
- `${PAPER_ROOT:-<outside_this_repo>}/paper/src/main.tex` (not part of this GitHub repo)

This document maps **paper sections/figures/tables** to **scripts, configs, and output artifacts** in this repo.

For the camera-ready crosswalk that includes both the current ICML paper and
NeurIPS / Joint LARCH-LACML appendix items that may be merged into ICML, see
`PAPER_RELEASE_CROSSWALK.md`.

---

## 0) Quick navigation

- **Main experiment driver (RQ1/RQ3)**: `src/main.py`
- **Paper reproduction scripts**: `scripts/`
- **Configs**: `config/`
- **Utility approximation experiments**: `utility_appro/`
- **LLM fine-tuning extension (DATE-LM)**: `finetuning/`

---

## 1) Paper → Code mapping table

### RQ1 — Optimality gap vs DP (paper `sec:rq1`, `app:rq1_exp`, mentions `tab:dp_gap`)

**Paper setting (main.tex):**
- train/valid/test = **20/100/500**
- seeds: **10..200 step 10** (20 trials)
- baselines: Random / LOO / Influence / DataShap / BetaShap / Banzhaf / AME / DVRL / DataOob
- optimal baseline: **DynamicProgramming**

**Code + configs:**
- Main entry: `src/main.py`
- Config: `config/rq1_config.yaml`
- Full run script (single-node): `scripts/run_rq1_dp.sh`
- Multi-node helper (cluster): `scripts/run_rq1_dp_multinode.sh`
- Worker implementation: `scripts/rq1_dp_worker.sh`
- Status helper: `scripts/rq1_dp_status.sh`
- Aggregation: `scripts/aggregate_rq1_dp.py`
- Figure 6 replot: `scripts/plot_rq1_figure6.py`

**Outputs (local, not committed to git):**
- Per-run CSV: `results_rq1_dp/<dataset>/seed_<seed>/addition_experiment_results.csv`
- Aggregates:
  - `results_rq1_dp/_summary/rq1_auc_by_seed.csv`
  - `results_rq1_dp/_summary/rq1_auc_summary.csv`
  - `results_rq1_dp/_summary/rq1_dp_gap_summary.csv`
  - `results_rq1_dp/_summary/report.md`
- NeurIPS/JLARCH-LACML detailed RQ1 figure:
  - `results/rq1_figure6/appro_results_small_latest_nonbbc.pdf`
  - `results/rq1_figure6/appro_results_small_latest_with_partial_bbc.pdf`
  - `results/rq1_figure6/figure6_latest_manifest.csv`

**Notes:**
- DP is exponential; expect RQ1 to be the slowest experiment family.
- `scripts/aggregate_rq1_dp.py` will try to parse the DP-gap range from the paper tex if it exists in your workspace.

---

### RQ2 — Curvature / substitutability impact (paper `sec:rq2_curvature`, `fig:dyn_small`, `app:curvature_exp`)

**Paper idea (main.tex):**
Use within-class message passing to increase “substitutability” (proxy for curvature), and show game-theoretic values degrade as substitutability increases.

**Code:**
- Experiment script: `scripts/run_rq2_curvature.py`

**Outputs (local, not committed to git):**
- Raw runs: `results/curvature_rq2/rq2_curvature_results.csv`
- Summary: `results/curvature_rq2/rq2_curvature_summary.csv`

**Notes:**
- The script is explicitly written as the “correct implementation” to reproduce the caption trend numbers (e.g., Shapley 0.738 → 0.595).
- This experiment is independent from OpenML; it’s a controlled synthetic setup (3-class GMM + cumulative propagation).

---

### RQ3 — Selection curves on OpenML (paper `sec:rq4_selection`, `fig:appr`, `app:rq3_exp`)

**Paper setting (main.tex):**
- Standard datasets: train/valid/test = **50/50/500**
- Large datasets (digits, bbc-embeddings): train/valid/test = **100/100/1000**
- seeds: **10..200 step 10** (20 trials)
- metric: test accuracy as function of selection size (“selection curve”)

**Code + configs:**
- Main entry: `src/main.py`
- Standard config: `config/base_config.yaml`
- Large-dataset config: `config/rq3_large_datasets.yaml`
- Full run script: `scripts/run_rq3.sh`
- Plot aggregation: `src/utils/plotting.py`

**Outputs (local, not committed to git):**
- Per-seed curves: `results/<dataset>/seed_<seed>/addition_experiment_results.csv`
- Aggregated plots: `results/<dataset>/aggregate/`

---

### Large budget (k=500) table (paper `tab:selection_results_500`)

**Paper artifact:**
- Table label: `tab:selection_results_500`
- Protocol: `train_count=500`; standard datasets use `valid/test=50/500`,
  while `digits` and `bbc-embeddings` use `valid/test=100/1000`.
- Reported value: whole-curve mean over top-k retraining, not the single
  endpoint at `k=500`.

**Code:**
- Paper runner: `scripts/run_budget500_curve_mean_table.py`
- Paper comparator: `scripts/compare_budget500_table3.py`
- Ground truth CSV: `scripts/paper_table3_ground_truth.csv`
- Legacy endpoint diagnostic: `scripts/run_budget500_table.py`

**Outputs (local, not committed to git):**
- Raw per (dataset, seed, method): `results/selection_results_500_curve_mean/raw.csv`
- Aggregated table: `results/selection_results_500_curve_mean/summary_curve_mean.csv`
- Paper comparison: `results/selection_results_500_curve_mean/comparison_to_paper.csv`

---

### Utility approximation (paper `tab:utility_approx`)

**Paper artifact:**
- Table label: `tab:utility_approx`
- Compare surrogate utility approximation error (MAE/MSE) across methods.

**Code:**
- Runner: `utility_appro/run_experiments.sh`
- Main: `utility_appro/src/main.py`

**Outputs:**
- **Tracked snapshot** (small CSVs, committed in git):
  - `utility_appro/aggregate_results_overall.csv`
  - `utility_appro/aggregate_results_by_dataset.csv`
- Additional run artifacts (not committed): `utility_appro/results/`, `utility_appro/logs/`, `utility_appro/aggregate_results/`

**Notes:**
- Requires `cvxpy` (see `ENVIRONMENT.md`).

---

## 2) Paper-number consistency checks (optional but useful)

These scripts are meant to sanity-check that reproduced outputs line up with the values stated in `paper/src/main.tex`.

- Compare reproduced outputs vs paper numbers:
  - `scripts/compare_with_paper.py`
  - Writes: `results/compare_with_paper/` (diff CSVs + report)
- Compare two result trees (e.g., “merged repo” vs “cleaned”):
  - `scripts/compare_versions.py`
  - Writes: `results/compare_versions/` (diff CSVs + report)

---

## 3) Method implementation index (where each method lives)

### Our method(s)
- Bipartite selection (paper “Bipartite”): `src/evaluators/bipartite.py`
- DP optimal baseline: `src/evaluators/dynamic.py`

### Baselines (OpenDataVal)
Most baselines are imported from OpenDataVal in `src/main.py`:
- `DataShapley`, `BetaShapley`, `DataBanzhaf`, `InfluenceSubsample`, `LeaveOneOut`, `RandomEvaluator`, `AME`, `DVRL`, `DataOob`

### Compatibility patch (OpenML metadata drift)
- `src/utils/opendataval_compat.py` (applied in `src/main.py`)

---

## 4) Local reproduction status (this machine)

This is **not** a git-tracked guarantee; it’s a helpful inventory of the artifacts currently present under this workspace:

- RQ1 DP aggregates: `results_rq1_dp/_summary/`
- RQ2 curvature summary: `results/curvature_rq2/`
- Budget=500 table: `results/selection_results_500_curve_mean/`
- Paper consistency diffs: `results/compare_with_paper/`

---

## 5) Extension beyond the original paper: LLM finetuning (DATE-LM)

This repo also contains an **extension** to LLM fine-tuning data selection (not part of the original OpenML experiments):

- Entry docs: `finetuning/README.md`
- Multi-seed + migration docs: `finetuning/MULTISEED_MIGRATION.md`, `finetuning/RETROSPECTIVE_AND_H200_PLAN.md`
