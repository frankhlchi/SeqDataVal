# Multi-Seed Training Robustness Results

Training seeds: [1337, 2025]
Methods: ['bm25', 'repsim', 'repsim_v2', 'rds_plus_v2']
Tasks: ['mmlu', 'gsm8k', 'bbh']

---

## Per-Task Results (Mean ± Std across seeds)

| Method | MMLU | GSM8K | BBH | Avg |
|--------|------|-------|-----|-----|
| BM25 | 61.38% | 61.64% | 66.95% | 63.32% |
| RepSim | 59.23% | 63.76% | 65.34% | 62.78% |
| RepSim (v2) | 59.54% | 60.20% | 66.85% | 62.20% |
| RDS+ (v2) | 60.01% | 62.85% | 0.00% | 61.43% |

---

## Detailed Per-Seed Results

### MMLU

| Method | Seed 1337 | Seed 2025 | Mean | Std |
|---|---|---|---|---|
| BM25 | 61.38% | - | 61.38% | 0.00% |
| RepSim | 59.23% | - | 59.23% | 0.00% |
| RepSim (v2) | 59.54% | - | 59.54% | 0.00% |
| RDS+ (v2) | 60.01% | - | 60.01% | 0.00% |

### GSM8K

| Method | Seed 1337 | Seed 2025 | Mean | Std |
|---|---|---|---|---|
| BM25 | 61.64% | - | 61.64% | 0.00% |
| RepSim | 63.76% | - | 63.76% | 0.00% |
| RepSim (v2) | 60.20% | - | 60.20% | 0.00% |
| RDS+ (v2) | 62.85% | - | 62.85% | 0.00% |

### BBH

| Method | Seed 1337 | Seed 2025 | Mean | Std |
|---|---|---|---|---|
| BM25 | 66.95% | - | 66.95% | 0.00% |
| RepSim | 65.34% | - | 65.34% | 0.00% |
| RepSim (v2) | 66.85% | - | 66.85% | 0.00% |
| RDS+ (v2) | - | - | 0.00% | 0.00% |

---

## Summary Statistics

| Method | Mean (all tasks) | Std (all tasks) | Min | Max |
|--------|------------------|-----------------|-----|-----|
| BM25 | 63.32% | 2.57% | 61.38% | 66.95% |
| RepSim | 62.78% | 2.59% | 59.23% | 65.34% |
| RepSim (v2) | 62.20% | 3.31% | 59.54% | 66.85% |
| RDS+ (v2) | 61.43% | 1.42% | 60.01% | 62.85% |

---

## Missing Results

The following combinations are missing:

- bm25/mmlu/seed_2025
- bm25/gsm8k/seed_2025
- bm25/bbh/seed_2025
- repsim/mmlu/seed_2025
- repsim/gsm8k/seed_2025
- repsim/bbh/seed_2025
- repsim_v2/mmlu/seed_2025
- repsim_v2/gsm8k/seed_2025
- repsim_v2/bbh/seed_2025
- rds_plus_v2/mmlu/seed_2025
- rds_plus_v2/gsm8k/seed_2025
- rds_plus_v2/bbh/seed_1337
- rds_plus_v2/bbh/seed_2025
