# Multi-Seed Training Robustness Results

Training seeds: [1337]
Methods: ['random_avg', 'rds_plus', 'bipcov', 'bipcov_wmean', 'bipcov_bge', 'bipcov_e5']
Tasks: ['mmlu', 'gsm8k', 'bbh']

---

## Per-Task Results (Mean ± Std across seeds)

| Method | MMLU | GSM8K | BBH | Avg |
|--------|------|-------|-----|-----|
| Random Avg | 60.39% | 59.64% | 66.40% | 62.14% |
| RDS+ | 60.63% | 61.41% | 66.23% | 62.75% |
| BipCov (ref-aligned) | 63.38% | 59.29% | 66.08% | 62.92% |
| BipCov (weighted-mean emb) | 60.49% | 61.87% | 66.32% | 62.89% |
| BipCov (BGE emb) | 61.81% | 61.71% | 67.14% | 63.56% |
| BipCov (E5 emb) | 62.14% | 60.58% | 65.50% | 62.74% |

---

## Detailed Per-Seed Results

### MMLU

| Method | Seed 1337 | Mean | Std |
|---|---|---|---|
| Random Avg | 60.39% | 60.39% | 0.00% |
| RDS+ | 60.63% | 60.63% | 0.00% |
| BipCov (ref-aligned) | 63.38% | 63.38% | 0.00% |
| BipCov (weighted-mean emb) | 60.49% | 60.49% | 0.00% |
| BipCov (BGE emb) | 61.81% | 61.81% | 0.00% |
| BipCov (E5 emb) | 62.14% | 62.14% | 0.00% |

### GSM8K

| Method | Seed 1337 | Mean | Std |
|---|---|---|---|
| Random Avg | 59.64% | 59.64% | 0.00% |
| RDS+ | 61.41% | 61.41% | 0.00% |
| BipCov (ref-aligned) | 59.29% | 59.29% | 0.00% |
| BipCov (weighted-mean emb) | 61.87% | 61.87% | 0.00% |
| BipCov (BGE emb) | 61.71% | 61.71% | 0.00% |
| BipCov (E5 emb) | 60.58% | 60.58% | 0.00% |

### BBH

| Method | Seed 1337 | Mean | Std |
|---|---|---|---|
| Random Avg | 66.40% | 66.40% | 0.00% |
| RDS+ | 66.23% | 66.23% | 0.00% |
| BipCov (ref-aligned) | 66.08% | 66.08% | 0.00% |
| BipCov (weighted-mean emb) | 66.32% | 66.32% | 0.00% |
| BipCov (BGE emb) | 67.14% | 67.14% | 0.00% |
| BipCov (E5 emb) | 65.50% | 65.50% | 0.00% |

---

## Summary Statistics

| Method | Mean (all tasks) | Std (all tasks) | Min | Max |
|--------|------------------|-----------------|-----|-----|
| Random Avg | 62.14% | 3.03% | 59.64% | 66.40% |
| RDS+ | 62.75% | 2.48% | 60.63% | 66.23% |
| BipCov (ref-aligned) | 62.92% | 2.79% | 59.29% | 66.08% |
| BipCov (weighted-mean emb) | 62.89% | 2.49% | 60.49% | 66.32% |
| BipCov (BGE emb) | 63.56% | 2.54% | 61.71% | 67.14% |
| BipCov (E5 emb) | 62.74% | 2.06% | 60.58% | 65.50% |

---

## Missing Results

All combinations completed successfully.
