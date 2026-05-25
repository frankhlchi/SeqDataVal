# BipCov: Bipartite Greedy Coverage (from precomputed embeddings)

This method is intended for **Scheme 1** integration:

1. **Outside DATE-LM**, produce embeddings:
   - `train_emb.npy`: `(N_train, d)`
   - `ref_emb.npy`: `(N_ref, d)`

2. Run the selector to produce a DATE-LM-compatible metric file:

```bash
python finetuning/bipcov/probe_bipcov_from_emb.py \
  --train_emb /path/to/train_emb.npy \
  --ref_emb   /path/to/ref_emb.npy \
  --out       /path/to/scores/mmlu_shots_bipcov.npy \
  --k_max     10000 \
  --threshold 0.45
```

3. Use the produced `.npy` as `--metric_path` for fine-tuning:

```bash
python train/finetune.py ... --metric_path /path/to/scores/mmlu_shots_bipcov.npy
```

## Graph construction options

- **Threshold** edges: `sim(i, j) >= tau` (default)
  - provide `--threshold tau`
  - or let the script choose `tau` via `--target_density` quantile

- **Top-L per ref** edges: connect each reference point to its top-L neighbors:
  - `--top_l 200`

## Output format

The output is a **1D score vector** of length `N_train`.
Scores are set such that **top-k** selection matches the greedy coverage ordering.
