# External Artifacts

This repository intentionally excludes heavyweight generated files. The public
git repo should stay small enough to clone, inspect, and run.

## Not Tracked In Git

- OpenML download caches and full result trees
- DATE-LM training/evaluation data
- selected-data archives
- embedding arrays (`.npy`, `.npz`)
- Llama/LitGPT/LoRA checkpoints
- full run logs
- paper PDFs and LaTeX build outputs
- credentials, tokens, SSH keys, or machine-specific configuration

## Recommended Distribution

When releasing large artifacts, publish them as GitHub release assets, Zenodo
records, institutional storage, or another stable archive. Include:

- a manifest of files;
- SHA256 checksums;
- restore instructions;
- the exact code commit used to generate the artifacts.

The scripts in this repository are written so restored artifacts can live in
user-provided paths through command-line arguments or environment variables.
