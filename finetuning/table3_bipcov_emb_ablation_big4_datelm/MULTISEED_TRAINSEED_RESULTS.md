# Multi-Seed Training Robustness Results

Training seeds: [1337]
Methods: ['random_avg', 'rds_plus', 'bipcov_bge', 'bipcov_qwen3emb', 'bipcov_nvembed', 'bipcov_gteqwen2', 'bipcov_gritlm']
Tasks: ['mmlu', 'gsm8k', 'bbh']

---

## Per-Task Results (Mean ± Std across seeds)

| Method | MMLU | GSM8K | BBH | Avg |
|--------|------|-------|-----|-----|
| Random Avg | 60.39% | 59.64% | 66.40% | 62.14% |
| RDS+ | 60.63% | 61.41% | 66.23% | 62.75% |
| BipCov (BGE emb) | 61.81% | 61.71% | 67.14% | 63.56% |
| BipCov (Qwen3-Emb-8B) | 62.50% | 63.00% | 65.62% | 63.71% |
| BipCov (NV-Embed-v2) | 62.13% | 59.21% | 66.14% | 62.49% |
| BipCov (GTE-Qwen2-7B) | 61.47% | 62.55% | 66.40% | 63.47% |
| BipCov (GritLM-7B) | 61.07% | 61.11% | 65.56% | 62.58% |

---

## Detailed Per-Seed Results

### MMLU

| Method | Seed 1337 | Mean | Std |
|---|---|---|---|
| Random Avg | 60.39% | 60.39% | 0.00% |
| RDS+ | 60.63% | 60.63% | 0.00% |
| BipCov (BGE emb) | 61.81% | 61.81% | 0.00% |
| BipCov (Qwen3-Emb-8B) | 62.50% | 62.50% | 0.00% |
| BipCov (NV-Embed-v2) | 62.13% | 62.13% | 0.00% |
| BipCov (GTE-Qwen2-7B) | 61.47% | 61.47% | 0.00% |
| BipCov (GritLM-7B) | 61.07% | 61.07% | 0.00% |

### GSM8K

| Method | Seed 1337 | Mean | Std |
|---|---|---|---|
| Random Avg | 59.64% | 59.64% | 0.00% |
| RDS+ | 61.41% | 61.41% | 0.00% |
| BipCov (BGE emb) | 61.71% | 61.71% | 0.00% |
| BipCov (Qwen3-Emb-8B) | 63.00% | 63.00% | 0.00% |
| BipCov (NV-Embed-v2) | 59.21% | 59.21% | 0.00% |
| BipCov (GTE-Qwen2-7B) | 62.55% | 62.55% | 0.00% |
| BipCov (GritLM-7B) | 61.11% | 61.11% | 0.00% |

### BBH

| Method | Seed 1337 | Mean | Std |
|---|---|---|---|
| Random Avg | 66.40% | 66.40% | 0.00% |
| RDS+ | 66.23% | 66.23% | 0.00% |
| BipCov (BGE emb) | 67.14% | 67.14% | 0.00% |
| BipCov (Qwen3-Emb-8B) | 65.62% | 65.62% | 0.00% |
| BipCov (NV-Embed-v2) | 66.14% | 66.14% | 0.00% |
| BipCov (GTE-Qwen2-7B) | 66.40% | 66.40% | 0.00% |
| BipCov (GritLM-7B) | 65.56% | 65.56% | 0.00% |

---

## Summary Statistics

| Method | Mean (all tasks) | Std (all tasks) | Min | Max |
|--------|------------------|-----------------|-----|-----|
| Random Avg | 62.14% | 3.03% | 59.64% | 66.40% |
| RDS+ | 62.75% | 2.48% | 60.63% | 66.23% |
| BipCov (BGE emb) | 63.56% | 2.54% | 61.71% | 67.14% |
| BipCov (Qwen3-Emb-8B) | 63.71% | 1.37% | 62.50% | 65.62% |
| BipCov (NV-Embed-v2) | 62.49% | 2.84% | 59.21% | 66.14% |
| BipCov (GTE-Qwen2-7B) | 63.47% | 2.12% | 61.47% | 66.40% |
| BipCov (GritLM-7B) | 62.58% | 2.11% | 61.07% | 65.56% |

---

## Missing Results

All combinations completed successfully.
