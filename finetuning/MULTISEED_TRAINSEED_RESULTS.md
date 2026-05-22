# Multi-Seed Training Robustness Results

Training seeds: [42, 1337, 2025]
Methods: ['random', 'rds_plus', 'bipcov']
Tasks: ['mmlu', 'gsm8k', 'bbh']

---

## Per-Task Results (Mean ± Std across seeds)

| Method | MMLU | GSM8K | BBH | Avg |
|--------|------|-------|-----|-----|
| Random | 60.00±0.17% | 61.89±0.58% | 66.58±0.30% | 62.82±0.31% |
| RDS+ | 59.58±0.07% | 62.24±0.73% | 66.80±0.21% | 62.88±0.25% |
| BipCov (ref-aligned) | 61.87±0.14% | 63.53±0.41% | 66.27±0.58% | 63.89±0.34% |

---

## Detailed Per-Seed Results

### MMLU

| Method | Seed 42 | Seed 1337 | Seed 2025 | Mean | Std |
|---|---|---|---|---|---|
| Random | 59.86% | 59.90% | 60.24% | 60.00% | 0.17% |
| RDS+ | 59.65% | 59.60% | 59.49% | 59.58% | 0.07% |
| BipCov (ref-aligned) | 62.03% | 61.89% | 61.68% | 61.87% | 0.14% |

### GSM8K

| Method | Seed 42 | Seed 1337 | Seed 2025 | Mean | Std |
|---|---|---|---|---|---|
| Random | 61.11% | 62.47% | 62.09% | 61.89% | 0.58% |
| RDS+ | 61.26% | 62.47% | 63.00% | 62.24% | 0.73% |
| BipCov (ref-aligned) | 63.61% | 63.99% | 63.00% | 63.53% | 0.41% |

### BBH

| Method | Seed 42 | Seed 1337 | Seed 2025 | Mean | Std |
|---|---|---|---|---|---|
| Random | 66.18% | 66.88% | 66.67% | 66.58% | 0.30% |
| RDS+ | 66.67% | 67.11% | 66.64% | 66.80% | 0.21% |
| BipCov (ref-aligned) | 67.01% | 66.20% | 65.60% | 66.27% | 0.58% |

---

## Summary Statistics

| Method | Mean (all tasks) | Std (all tasks) | Min | Max |
|--------|------------------|-----------------|-----|-----|
| Random | 62.82% | 2.79% | 59.86% | 66.88% |
| RDS+ | 62.88% | 3.02% | 59.49% | 67.11% |
| BipCov (ref-aligned) | 63.89% | 1.86% | 61.68% | 67.01% |

---

## Missing Results

All combinations completed successfully.
