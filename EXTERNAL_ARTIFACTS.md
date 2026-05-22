# External Artifacts Index

This repository intentionally contains code, configs, scripts, and small
summary tables only. Large run artifacts are kept outside the GitHub code
repository and should be distributed through a release archive, Drive, Zenodo,
or another artifact store with SHA256 checksums.

## Why Artifacts Are External

Do not commit these to the main GitHub repo:

- full OpenML result trees,
- DATE-LM data directories,
- selected-data/evaluation tarballs,
- score arrays (`.npy`, `.npz`),
- Llama/LitGPT/LoRA checkpoints,
- full run logs,
- paper source snapshots and downloaded DATE-LM zips,
- cloud tokens, SSH keys, Hugging Face tokens, or machine-specific credentials.

The code repository should stay small and runnable. Artifact archives should be
referenced by checksum and restore instructions.

## Local Artifact Store Used For Camera-Ready Verification

The camera-ready verification workspace keeps the heavy artifacts here:

```text
/root/seq_reproduce/transfer_bundles/datelm_h200_finetuning_20260514
```

Important files:

```text
/root/seq_reproduce/transfer_bundles/datelm_h200_finetuning_20260514/README.md
/root/seq_reproduce/transfer_bundles/datelm_h200_finetuning_20260514/ARTIFACTS_AND_RESTORE.md
/root/seq_reproduce/transfer_bundles/datelm_h200_finetuning_20260514/NEW_SERVER_H200_H100_RUNBOOK.md
/root/seq_reproduce/transfer_bundles/datelm_h200_finetuning_20260514/MANIFEST.files.txt
/root/seq_reproduce/transfer_bundles/datelm_h200_finetuning_20260514/checksums/SHA256SUMS.txt
```

Portable archives:

```text
/root/seq_reproduce/transfer_bundles/datelm_h200_finetuning_20260514.zip
/root/seq_reproduce/transfer_bundles/datelm_h200_finetuning_20260514.zip.sha256
/root/seq_reproduce/transfer_bundles/datelm_h200_finetuning_20260514.tar.gz
```

## DATE-LM Artifact Groups

The external DATE-LM bundle contains:

| Artifact group | Purpose |
|---|---|
| `datelm_trainseed_minimal/` | Core DATE-LM train-seed selected-data and eval artifacts. |
| `datelm_trainseed_extra_baselines/` | Extra baseline selected-data artifacts for BM25 / RepSim / RepSim-v2 / RDS+-v2. |
| `datelm_table3_single_seed_vast_20260106/` | Small raw metrics and score manifests for the Table-3-style H100 run. |
| `datelm_table3_rdsplus_weightedmean_vast_20260107_013353/` | RDS+ weighted-mean correction artifacts. |
| `datelm_table3_bipcov_emb_ablation_vast_20260107_100219/` | BGE/E5/weighted-mean BipCov embedding ablation artifacts. |
| `datelm_table3_bipcov_big4_vast_20260108_021123/` | Qwen3/NV-Embed/GTE-Qwen2/GritLM BipCov embedding ablation metrics. |
| `datelm_table3_bipcov_big4_vast_20260108_031614_fullctx/` | Full context logs and environment snapshot for the Big4 H100 run. |

Use the bundle's `ARTIFACTS_AND_RESTORE.md` for restore commands.

## Verification Records

The camera-ready evidence and runtime accounting live outside the GitHub repo:

```text
/root/seq_reproduce/reproduction_documentation/CAMERA_READY_REPRODUCTION_CHECKLIST.md
/root/seq_reproduce/reproduction_documentation/ICML_REPRODUCTION_CHECKLIST.md
/root/seq_reproduce/reproduction_checks/datelm_paper_consistency/GPU_TIME_AND_SERVER_CONFIG_2026_05_23.md
/root/seq_reproduce/reproduction_checks/google_drive_investigation/DATELM_VASTAI_DRIVE_PROVENANCE.md
```

The DATE-LM GPU results were verified from synced GPU-server raw metrics and
run artifacts. Timestamped H100 logs account for roughly `130-160` H100
GPU-hours, with H200/A100 server configurations, commands, and raw metrics also
recorded.

## Public Release Recommendation

For GitHub:

- publish this code repository,
- publish `REPRODUCIBILITY.md`, `PAPER_EXPERIMENT_INDEX.md`, and this file,
- keep only small CSV/Markdown summaries in the repo.

For external artifacts:

- publish the transfer bundle or selected tarballs through a release asset,
  Drive, or Zenodo,
- include `SHA256SUMS.txt`,
- link the artifact location from `finetuning/README_RELEASE.md` and this file.
