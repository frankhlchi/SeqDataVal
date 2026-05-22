# Paper To Release-Code Crosswalk

This document maps the current ICML paper and the canonical NeurIPS / Joint
LARCH-LACML classical experiment source to the public release code. It is meant
to answer: "If a table, figure, algorithm, or numeric claim appears in the
paper, what code or artifact verifies it?"

## Scope

Canonical paper sources used for this crosswalk:

```text
ICML source:
  /root/seq_reproduce/extracted/paper/unify_final

NeurIPS / Joint LARCH-LACML classical source:
  /root/seq_reproduce/extracted/nips_submission/inner/neurips_2025.tex
```

Release-code boundary:

```text
This GitHub repo contains code, configs, scripts, and small summaries.
Heavy raw result trees, DATE-LM data, selected-data tarballs, logs, and
checkpoints are external artifacts. See EXTERNAL_ARTIFACTS.md.
```

## Status Key

| Status | Meaning |
|---|---|
| `release-code` | The implementation and runnable script are included in this repo. |
| `verified-rerun` | A local CPU rerun on the camera-ready server reproduced the paper item within the documented tolerance. |
| `verified-record` | Raw GPU/server metrics and synced artifacts verify the paper item; full rerun requires external GPU resources. |
| `source-only` | Theory, algorithm, or diagram item; no numeric rerun is required. |
| `optional-neurips` | NeurIPS/JLARCH-LACML item that can be added to ICML if desired. |
| `external-only` | Evidence lives outside the public repo and is referenced through `EXTERNAL_ARTIFACTS.md`. |

## ICML Main Paper

| ICML item | Paper location | Release code | Verification evidence | Status |
|---|---|---|---|---|
| Sequential selection formulation and exact DP | `sections/03_framework.tex`, `app:optimal_algo`, `alg:optimal_value` | `src/evaluators/dynamic.py`, `scripts/run_rq1_dp.sh`, `scripts/aggregate_rq1_dp.py` | `/root/seq_reproduce/reproduction_checks/paper_size_rq1_dp_batches/_summary/rq1_dp_gap_summary.csv` | `release-code`, `verified-rerun` |
| Bipartite coverage surrogate and greedy selection | `sections/05_bipartite.tex`, `alg:bipartite` | `src/evaluators/bipartite.py`, `utility_appro/src/evaluators/bipartite.py`, `finetuning/bipcov/` | Code-level match plus utility/Budget-500/DATE-LM result checks below | `release-code`, `source-only` |
| RQ1 DP gap table | `sections/07_experiments.tex`, `tab:dp_gap` | `src/main.py`, `config/rq1_config.yaml`, `scripts/run_rq1_dp.sh`, `scripts/aggregate_rq1_dp.py` | `/root/seq_reproduce/reproduction_checks/paper_size_rq1_dp_batches/_summary/rq1_dp_gap_summary.csv` | `release-code`, `verified-rerun` |
| RQ2 curvature/substitutability figure | `sections/07_experiments.tex`, `fig:dyn_small` | `scripts/run_rq2_curvature.py` | `/root/seq_reproduce/reproduction_checks/paper_size_rq2_curvature_batches/rq2_curvature_summary_seeds0_19.csv`; `paper_value_checks.csv` | `release-code`, `verified-rerun` |
| RQ3 selection curves | `sections/07_experiments.tex`, `fig:appr` | `src/main.py`, `config/base_config.yaml`, `config/rq3_large_datasets.yaml`, `scripts/run_rq3.sh`, `src/utils/plotting.py` | `/root/seq_reproduce/reproduction_checks/paper_size_rq3_openml_20seed/rq3_pilot`; HCloud BBC tail record in `hcloud_rq3_bbc_release_full_20260517/` | `release-code`, `verified-rerun` for non-BBC; BBC is resource-heavy tail |
| Budget-500 top-two table | `sections/07_experiments.tex`, `tab:selection_results_500` | `scripts/run_budget500_curve_mean_table.py`, `scripts/compare_budget500_table3.py`, `scripts/paper_table3_ground_truth.csv` | `/root/seq_reproduce/reproduction_checks/budget500_final_rl1000/{raw.csv,summary.csv,comparison_to_paper.csv,table3_release.tex}` and historical-NeurIPS compare `/root/seq_reproduce/reproduction_checks/budget500_curve_mean_full_rerun/comparison_to_paper.csv` | `release-code`, `verified-rerun` |
| Utility approximation table | `sections/07_experiments.tex`, `tab:utility_approx` | `utility_appro/run_experiments.sh`, `utility_appro/src/main.py`, tracked snapshots `utility_appro/aggregate_results_*.csv` | `/root/seq_reproduce/reproduction_checks/paper_size_utility_approx_batches/paper_value_checks.csv` | `release-code`, `verified-rerun` |
| DATE-LM main-text fine-tuning result | `sections/07_experiments.tex`, final paragraph | `finetuning/bipcov/`, `finetuning/scripts/datelm_paper/run_multiseed_train_eval_only.py`, `finetuning/scripts/datelm_paper/summarize_multiseed_results.py` | `/root/seq_reproduce/reproduction_bundle/summaries/datelm_multiseed_overall.csv`; BipCov `63.8892`, RDS+ `62.8753`, Random `62.8226` | `release-code`, `verified-record`, `external-only` |

## ICML Appendix

| ICML appendix item | Paper label | Release code | Verification evidence | Status |
|---|---|---|---|---|
| Experimental settings for OpenML | `app:exp_details`, `app:rq1_exp`, `app:rq3_exp` | `config/rq1_config.yaml`, `config/base_config.yaml`, `config/rq3_large_datasets.yaml`, `src/main.py` | Reproduction checklists in `/root/seq_reproduce/reproduction_documentation/` | `release-code`, `verified-rerun` |
| Framework diagram | `fig:framework` | Conceptual diagram; code implementation is `src/evaluators/dynamic.py` and `src/evaluators/bipartite.py` | Paper source asset `frame.pdf` | `source-only` |
| Detailed low-budget selection table | `tab:selection_results` | `scripts/run_rq3.sh`, `src/main.py`; archive comparison tooling in reproduction docs | Drive archive check `/root/seq_reproduce/reproduction_checks/google_drive_investigation/DRIVE_ARCHIVE_TABLE2_CURVE_CONFIRMATION.md`; NeurIPS table trace | `release-code`, `verified-rerun` for non-BBC and archive-confirmed standard rows |
| DATE-LM setup and BipCov details | `app:datelm` | `finetuning/README_RELEASE.md`, `finetuning/bipcov/`, `finetuning/scripts/datelm_paper/` | `/root/seq_reproduce/reproduction_checks/google_drive_investigation/DATELM_VASTAI_DRIVE_PROVENANCE.md` | `release-code`, `verified-record`, `external-only` |
| DATE-LM computational cost table | `tab:datelm_cost` | Run scripts in `finetuning/scripts/datelm_paper/`; artifact index in `EXTERNAL_ARTIFACTS.md` | `/root/seq_reproduce/reproduction_checks/datelm_paper_consistency/GPU_TIME_AND_SERVER_CONFIG_2026_05_23.md` | `verified-record`, `external-only` |
| DATE-LM train-seed robustness table | `tab:datelm_trainseed` | `finetuning/scripts/datelm_paper/run_multiseed_train_eval_only.py`; summary `finetuning/MULTISEED_TRAINSEED_RESULTS.*` | `/root/seq_reproduce/reproduction_bundle/summaries/datelm_multiseed_core.csv`; `/root/seq_reproduce/reproduction_bundle/summaries/datelm_multiseed_overall.csv` | `release-code`, `verified-record` |
| DATE-LM single-seed baseline table | `tab:datelm_table3_seed1337` | `finetuning/scripts/datelm_paper/run_table3_single_seed_4gpu.sh`; `finetuning/table3_single_seed_datelm/` | `/root/seq_reproduce/reproduction_bundle/summaries/datelm_single_seed_from_raw_metrics.csv`; raw `metrics.json` under external DATE-LM tree | `release-code`, `verified-record`, `external-only` |
| DATE-LM embedding ablation table | `tab:datelm_bipcov_emb` | `finetuning/scripts/datelm_paper/bipcov_emb_ablation_pipeline.sh`; `bipcov_emb_ablation_big4_pipeline.sh`; `finetuning/table3_bipcov_emb_ablation*_datelm/` | `/root/seq_reproduce/reproduction_bundle/summaries/datelm_emb_ablation_from_raw_metrics.csv` | `release-code`, `verified-record`, `external-only` |

## NeurIPS / Joint LARCH-LACML Items That Can Be Added To ICML

These are classical-result items present in the NeurIPS / Joint LARCH-LACML
source. They are compatible with the current ICML narrative because ICML
inherits the classical experiment run.

| Candidate item | NeurIPS label | Release code | Evidence | Recommendation |
|---|---|---|---|---|
| Detailed RQ1 selection-curve Figure 6 | `fig:upper_bound`, includes `appro_results_small.pdf` | `scripts/plot_rq1_figure6.py` plus RQ1 raw from `scripts/run_rq1_dp.sh` | `/root/seq_reproduce/reproduction_checks/rq1_figure6_latest/FIGURE6_RQ1_REPRODUCTION_NOTE.md`; `appro_results_small_latest_full_bbc.pdf/png` | Safe to add if ICML needs more visual evidence. This is now in release code. |
| Appendix RQ2 visualizations | `fig:gmm_evolution`, `fig:curvature` | `scripts/run_rq2_curvature.py` | `/root/seq_reproduce/reproduction_checks/paper_size_rq2_curvature_batches/` | Numeric trend is reproduced. Figure rendering can be refreshed from the summary if desired. |
| Full low-budget selection table | `tab:selection_results` | `scripts/run_rq3.sh`, `src/main.py` | `DRIVE_ARCHIVE_TABLE2_CURVE_CONFIRMATION.md`; current RQ3 non-BBC raw | Safe as appendix table; use NeurIPS/JLARCH-LACML checklist as canonical source. |
| Full Budget-500 table with all ten methods | `tab:selection_results_500` | `scripts/run_budget500_curve_mean_table.py`, `scripts/compare_budget500_table3.py` | `budget500_curve_mean_full_rerun/comparison_to_paper.csv` for historical DVRL setting; `budget500_final_rl1000/` for fair DVRL setting | Safe to include, but choose one DVRL convention and state it. Historical NeurIPS table uses `--dvrl_rl_epochs 32`; current ICML text prefers fair `--dvrl_rl_epochs 1000`. |
| Utility approximation horizontal table | `tab:utility_approx_horizontal` | `utility_appro/run_experiments.sh` | `paper_size_utility_approx_batches/paper_value_checks.csv` | Already mirrored by ICML `tab:utility_approx`. |

## Release Notes For Merging ICML And NeurIPS Text

1. Treat NeurIPS / Joint LARCH-LACML as the canonical classical experiment
   source for RQ1, RQ2, RQ3, low-budget Table 2, Budget-500, and utility
   approximation.
2. Treat ICML as the same classical result set plus the DATE-LM extension.
3. If adding NeurIPS Figure 6 to ICML, use `scripts/plot_rq1_figure6.py` and
   cite the reproduced asset under `rq1_figure6_latest/`, not the older gray
   Drive plot.
4. For Budget-500, the correct paper protocol is `train_count=500` and
   curve-mean accuracy over the top-k selection curve. Do not use
   `scripts/run_budget500_table.py` for the paper table; it is an
   endpoint-style diagnostic.
5. For DVRL in Budget-500, the release runner defaults to
   `--dvrl_rl_epochs 1000` because that aligns with the current ICML fairness
   footnote and OpenDataVal's default. To recover the older NeurIPS historical
   full table, pass `--dvrl_rl_epochs 32` and document that convention.
6. DATE-LM is not a CPU-local rerun claim. It is verified from synced GPU
   server raw metrics and the H100/H200 transfer bundle. Keep the raw DATE-LM
   artifacts external to the GitHub repo.

## Reviewer-Facing Pointers

| Purpose | Address |
|---|---|
| Master camera-ready checklist | `/root/seq_reproduce/reproduction_documentation/CAMERA_READY_REPRODUCTION_CHECKLIST.md` |
| NeurIPS classical checklist | `/root/seq_reproduce/reproduction_documentation/NEURIPS_REPRODUCTION_CHECKLIST.md` |
| ICML DATE-LM checklist | `/root/seq_reproduce/reproduction_documentation/ICML_REPRODUCTION_CHECKLIST.md` |
| Release package audit | `/root/seq_reproduce/release_packages/seqdataval_camera_ready_20260523/RELEASE_AUDIT.md` |
| External artifact index | `EXTERNAL_ARTIFACTS.md` |
