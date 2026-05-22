# Multi-Seed Training Robustness Results

Training seeds: [1337]
Methods: ['random1', 'random2', 'random3', 'random_avg', 'bm25', 'repsim', 'rds_plus', 'gradsim', 'less', 'bipcov']
Tasks: ['mmlu', 'gsm8k', 'bbh']

---

## Per-Task Results (Mean ± Std across seeds)

| Method | MMLU | GSM8K | BBH | Avg |
|--------|------|-------|-----|-----|
| Random 1 | 61.46% | 58.53% | 67.11% | 62.37% |
| Random 2 | 59.47% | 59.59% | 66.15% | 61.74% |
| Random 3 | 60.23% | 60.80% | 65.94% | 62.32% |
| Random Avg | 60.39% | 59.64% | 66.40% | 62.14% |
| BM25 | 59.85% | 58.98% | 62.63% | 60.49% |
| RepSim | 61.42% | 58.45% | 66.20% | 62.03% |
| RDS+ | 60.63% | 61.41% | 66.23% | 62.75% |
| Grad Sim | 62.07% | 56.71% | 64.44% | 61.07% |
| LESS | 61.10% | 57.77% | 64.68% | 61.18% |
| BipCov (ref-aligned) | 63.38% | 59.29% | 66.08% | 62.92% |

---

## Detailed Per-Seed Results

### MMLU

| Method | Seed 1337 | Mean | Std |
|---|---|---|---|
| Random 1 | 61.46% | 61.46% | 0.00% |
| Random 2 | 59.47% | 59.47% | 0.00% |
| Random 3 | 60.23% | 60.23% | 0.00% |
| Random Avg | 60.39% | 60.39% | 0.00% |
| BM25 | 59.85% | 59.85% | 0.00% |
| RepSim | 61.42% | 61.42% | 0.00% |
| RDS+ | 60.63% | 60.63% | 0.00% |
| Grad Sim | 62.07% | 62.07% | 0.00% |
| LESS | 61.10% | 61.10% | 0.00% |
| BipCov (ref-aligned) | 63.38% | 63.38% | 0.00% |

### GSM8K

| Method | Seed 1337 | Mean | Std |
|---|---|---|---|
| Random 1 | 58.53% | 58.53% | 0.00% |
| Random 2 | 59.59% | 59.59% | 0.00% |
| Random 3 | 60.80% | 60.80% | 0.00% |
| Random Avg | 59.64% | 59.64% | 0.00% |
| BM25 | 58.98% | 58.98% | 0.00% |
| RepSim | 58.45% | 58.45% | 0.00% |
| RDS+ | 61.41% | 61.41% | 0.00% |
| Grad Sim | 56.71% | 56.71% | 0.00% |
| LESS | 57.77% | 57.77% | 0.00% |
| BipCov (ref-aligned) | 59.29% | 59.29% | 0.00% |

### BBH

| Method | Seed 1337 | Mean | Std |
|---|---|---|---|
| Random 1 | 67.11% | 67.11% | 0.00% |
| Random 2 | 66.15% | 66.15% | 0.00% |
| Random 3 | 65.94% | 65.94% | 0.00% |
| Random Avg | 66.40% | 66.40% | 0.00% |
| BM25 | 62.63% | 62.63% | 0.00% |
| RepSim | 66.20% | 66.20% | 0.00% |
| RDS+ | 66.23% | 66.23% | 0.00% |
| Grad Sim | 64.44% | 64.44% | 0.00% |
| LESS | 64.68% | 64.68% | 0.00% |
| BipCov (ref-aligned) | 66.08% | 66.08% | 0.00% |

---

## Summary Statistics

| Method | Mean (all tasks) | Std (all tasks) | Min | Max |
|--------|------------------|-----------------|-----|-----|
| Random 1 | 62.37% | 3.56% | 58.53% | 67.11% |
| Random 2 | 61.74% | 3.12% | 59.47% | 66.15% |
| Random 3 | 62.32% | 2.57% | 60.23% | 65.94% |
| Random Avg | 62.14% | 3.03% | 59.64% | 66.40% |
| BM25 | 60.49% | 1.56% | 58.98% | 62.63% |
| RepSim | 62.03% | 3.19% | 58.45% | 66.20% |
| RDS+ | 62.75% | 2.48% | 60.63% | 66.23% |
| Grad Sim | 61.07% | 3.23% | 56.71% | 64.44% |
| LESS | 61.18% | 2.82% | 57.77% | 64.68% |
| BipCov (ref-aligned) | 62.92% | 2.79% | 59.29% | 66.08% |

---

## Missing Results

All combinations completed successfully.
