# Task-Aware Bipartite Coverage for LLM Data Selection

> **Goal**: Extend SequentialDataVal's bipartite greedy coverage to LLM fine-tuning data selection, using DATE-LM benchmark for standardized evaluation.

**Author**: Frank (Hongliang Chi)
**Date**: December 2025
**Purpose**: Address KDD reviewer concerns about experiment scale (50-100 → 200K samples)

---

## 📊 单-seed 实验结果总结 (2025-12-28 更新)

### 主要结果表

| 方法 | MMLU | GSM8K | BBH | 平均 | vs Random |
|------|------|-------|-----|------|-----------|
| Random | 59.93% | 63.31% | 66.55% | 63.26% | - |
| BM25 | 61.32% | 62.17% | 66.83% | 63.44% | +0.18% |
| **BipCov (BGE, ref对齐)** | **62.03%** | 64.37% | **66.41%** | **64.27%** | **+1.01%** |
| BipCov (BGE, 旧版) | 61.59% | 64.59% | 66.01% | 64.06% | +0.80% |
| BipCov E5 | 61.42% | 63.23% | 67.02% | 63.89% | +0.63% |
| BipCov MiniLM | 59.92% | 65.66% | 64.57% | 63.38% | +0.12% |
| BipCov LLM | 59.45% | 64.97% | 65.81% | 63.41% | +0.15% |
| RepSim v1 | 58.99% | 65.05% | 65.45% | 63.16% | -0.10% |
| RepSim v2 | 60.59% | 60.88% | 66.62% | 62.70% | -0.56% |
| RDS+ v1 | 59.66% | 63.53% | 66.81% | 63.33% | +0.07% |
| RDS+ v2 | 59.61% | 62.47% | 66.60% | 62.89% | -0.37% |

> 注：以上是 **single-seed** 的 `selection → LoRA train → official eval` 结果，用于横向覆盖更多 baselines。
> `train-seed robustness`（固定 selected_data，仅重跑 train→eval）的 `mean±std` 见下方与 `finetuning/MULTISEED_TRAINSEED_RESULTS.md`。

### Ref 对齐消融实验 (2025-12-28)

| 设置 | MMLU | GSM8K | BBH | 平均 | 说明 |
|------|------|-------|-----|------|------|
| BipCov (prompt-only ref) | 61.59% | 64.59% | 66.01% | 64.06% | 旧版：HF原始question |
| **BipCov (prompt+label ref)** | **62.03%** | 64.37% | **66.41%** | **64.27%** | 新版：与DATE-LM基线对齐 |
| 变化 | +0.44% | -0.22% | +0.40% | **+0.21%** | 对齐后整体提升 |

**结论**: Ref对齐到DATE-LM格式(prompt+label)后，BipCov平均提升+0.21%，现在与基线使用相同的ref构造，比较更公平。

### 多 seed 稳健性（train seed robustness）

固定同一份 `selected_data`（Random / RDS+ / BipCov ref对齐），只重跑 **LoRA train → merge → official eval**，用于报告 `mean±std`（Reviewer 常抓 single-seed）。

- 空白服务器从零搭建：`NEW_SERVER_SETUP.md`
- 可迁移/可继续跑指南：`finetuning/MULTISEED_MIGRATION.md`
- 复盘与一周 H200 计划：`finetuning/RETROSPECTIVE_AND_H200_PLAN.md`
- 汇总脚本输出：`finetuning/MULTISEED_TRAINSEED_RESULTS.md` / `finetuning/MULTISEED_TRAINSEED_RESULTS.csv`
- 额外基线（可选）：同一协议也可跑 `bm25` / `repsim` / `repsim_v2` / `rds_plus_v2`（见 `finetuning/artifacts/datelm_trainseed_extra_baselines/README.md`）

Core 27（Vast.ai H200；seeds=42/1337/2025；methods=random/rds_plus/bipcov；tasks=mmlu/gsm8k/bbh）已完成，`mean±std`：

| Method | MMLU | GSM8K | BBH | Avg |
|--------|------|-------|-----|-----|
| Random | 60.00±0.17% | 61.89±0.58% | 66.58±0.30% | 62.82±0.31% |
| RDS+ | 59.58±0.07% | 62.24±0.73% | 66.80±0.21% | 62.88±0.25% |
| BipCov (ref-aligned) | 61.87±0.14% | 63.53±0.41% | 66.27±0.58% | 63.89±0.34% |

> 说明：multi-seed 的绝对分数可能与 single-seed 表略有差异（训练 seed / 环境不同），但我们关心的是同一协议下的 **稳定相对优势**。

### 嵌入模型对比 (BipCov)

| 嵌入模型 | 维度 | MMLU | GSM8K | BBH | 平均 |
|----------|------|------|-------|-----|------|
| BGE-large-v1.5 | 1024 | **61.59%** | 64.59% | 66.01% | **64.06%** |
| E5-large-v2 | 1024 | 61.42% | 63.23% | **67.02%** | 63.89% |
| MiniLM-L6-v2 | 384 | 59.92% | **65.66%** | 64.57% | 63.38% |

---

## ⚠️ 重要说明：与DATE-LM论文对比

### 论文 vs 我们的复现

| 方法 | 论文 MMLU | 我们 MMLU | 论文 GSM8K | 我们 GSM8K | 差异说明 |
|------|-----------|-----------|------------|------------|----------|
| Random | 60.2% | 59.93% | 59.6% | 63.31% | GSM8K偏高 |
| Rep-Sim | **61.2%** | **58.99%** | 59.2% | 65.05% | MMLU偏低2.2% |
| RDS+ | **62.4%** | **59.66%** | 59.6% | 63.53% | MMLU偏低2.7% |

### 关键问题

1. **基线/协议偏差**: 我们复现的RepSim/RDS+与论文报告存在差异（MMLU偏低2-3pt；GSM8K整体偏高；BBH接近）
2. **比较有效性**:
   - 在我们的实验环境内（official eval + 同一 pool/训练超参），BipCov 优于 Random 和我们复现的基线 ✓
   - 与论文报告的 RDS+ (MMLU 62.4%) 相比，我们的 BipCov（ref 对齐后 MMLU 62.03%）仍略低 (~0.37pt)；但跨论文绝对数字不可直接对齐 ⚠️

### 已验证的配置差异

| 配置项 | DATE-LM论文/代码 | 我们的实现 | 影响 |
|--------|------------------|------------|------|
| **LoRA alpha** | 256 | **512** | 可能影响收敛 |
| **训练 seed** | 42 (推测) | **1337** | 影响数据顺序 |
| **Shuffle** | `shuffle=False` (DataLoader) | **默认True** | 数据顺序不同 |
| **RepSim实现** | `probe_repsim_instruct.py` (LitGPT) | 自写HF管道 | 嵌入计算路径不同 |
| **Ref-set构造** | DATE-LM DATASETS (prompt+label) | **BipCov 已对齐**: DATE-LM DATASETS (prompt+label)（旧版为 HF question/prompt-only） | 影响选择口径/公平性 |

### 结论

1. **跨论文数字不可直接对齐** - 上述差异导致绝对数字与DATE-LM论文无法直接比较
2. **同环境相对比较有效** - 在 `paper_seed42_v1`（pool 与 `paper_seed42_v1_refpromptlabel` byte-identical）环境内，所有方法用相同管道:
   - BipCov (ref 对齐) vs Random: **+1.01%** ✓
   - BipCov (ref 对齐) vs RDS+: **+0.94%** ✓
   - BipCov (ref 对齐) vs RepSim: **+1.11%** ✓
3. **GSM8K整体偏高** - Random达到63.31% (论文59.6%)，说明协议差异影响所有方法

### 建议

- 论文中声明"同实验环境下的相对比较"
- 可联系DATE-LM作者获取精确复现配置
- 强调方法的**相对优势**而非绝对数字

### Ref 构造与 Label Leakage

**问题发现**: 历史上 BipCov 与 DATE-LM 基线使用了不同的 ref 构造方式；为消除“不公平对比”质疑，我们新增了 **ref 对齐**版本（prompt+label）。

| 方法 | Ref 构造 | 包含内容 | Leakage 风险 |
|------|----------|----------|--------------|
| BipCov (旧版 / prompt-only) | HF 原始 question field | 仅问题 | ✅ 无 |
| **BipCov (ref对齐 / prompt+label)** | DATE-LM DATASETS | prompt + label tokens | ⚠️（benchmark 自带） |
| RepSim / RDS+ | DATE-LM DATASETS | prompt + label tokens | ⚠️（benchmark 自带） |

**什么是 Label Leakage？**

在数据选择场景下，label leakage 指的是：**选择方法在决定"选哪些训练数据"时，能够"看到"目标任务的答案信息**。

**为什么 DATE-LM 的 ref 构造会导致 leakage？**

DATE-LM DATASETS 把 prompt + label 拼成完整 token 序列：
```python
# DATE-LM 的 ref 构造（伪代码）
ref_text = "Question: What is the capital of France?\nAnswer: Paris"
ref_embedding = embed(tokenize(ref_text))
```

当计算 "训练样本与 ref 的相似度" 时，相似度不仅衡量"问题相关性"，也隐式地衡量了"与答案内容的相似度"。

**具体例子**:
```
Ref: Q: What is 2+2?  A: 4

选择时偏好:
  "Math: 2+2 equals 4, basic arithmetic..."  ← 高相似度（含答案内容）
  "Introduction to algebra..."               ← 低相似度（不含答案内容）
```

**为什么这是问题？**

1. **不现实**: 真实部署时，你不会知道测试集的答案
2. **不公平**: 选择出的数据可能是因为"包含答案模式"而非"任务相关"
3. **过拟合风险**: 模型可能学到 spurious correlation

**当前状态的含义**:

- 我们的**主对比**采用 ref 对齐（prompt+label），与 DATE-LM baselines 同口径，避免“不公平对比”质疑
- 我们同时保留 prompt-only ref 作为**更严格**设置（无 label leakage）的消融实验
- 由于 DATE-LM benchmark 的官方 ref 本身包含 label，论文中需要明确说明该协议及其潜在 leakage

**建议**: 论文中明确说明主对比使用 DATE-LM ref 口径（prompt+label），并补充 prompt-only ablation + leakage 讨论。

---

## 实验配置

### 基础配置

| 参数 | 值 |
|------|-----|
| 基础模型 | `meta-llama/Llama-3.1-8B` |
| 训练池大小 | 200,000 samples (Tulu3) |
| 选择数据量 | 10,000 samples |
| 微调方法 | LoRA (rank=128, alpha=512) |
| 训练轮数 | 2 epochs |
| Batch Size | 4 × 32 grad_accum = 128 |
| Learning Rate | 2e-5 |
| 训练步数 | 156 steps |
| 随机种子 | pool/ref/selection=42；train=1337 |

### 嵌入模型

| 模型 | 来源 | 维度 | 用途 |
|------|------|------|------|
| BGE-large-v1.5 | `BAAI/bge-large-en-v1.5` | 1024 | BipCov 主要实验 |
| E5-large-v2 | `intfloat/e5-large-v2` | 1024 | 嵌入对比实验 |
| MiniLM-L6-v2 | `sentence-transformers/all-MiniLM-L6-v2` | 384 | 轻量级对比 |
| Llama-3.1-8B (last token) | LLM hidden states | 4096 | RepSim/RDS+ |

### 评估任务

| 任务 | 评估指标 | 样本数 | 评估方式 |
|------|----------|--------|----------|
| MMLU | Accuracy | 14,042 | 0-shot |
| GSM8K | Exact Match | 1,319 | 8-shot CoT |
| BBH | Exact Match | 6,511 (27 subtasks) | 3-shot |

---

## 1. Background and Motivation

### 1.1 Paper Context

The paper "Unifying and Optimizing Data Values for Selection via Sequential Decision-Making" proposes:
- Formulates data selection as a sequential decision problem
- Unifies existing data valuation methods (Data Shapley, Beta Shapley, etc.)
- Proposes **bipartite graph coverage** as a submodular surrogate with (1-1/e) guarantee

**KDD 2026 Reviewer Criticism**:
> "The scale of experiments is not sufficient. Selecting 50 to 100 data points is not even close to any practical use case."

### 1.2 Solution: DATE-LM Benchmark

DATE-LM (NeurIPS 2025) provides:
- Large-scale fine-tuning data selection (200K → 10K samples)
- Standardized evaluation on MMLU/GSM8K/BBH
- Pre-computed baselines and public leaderboard

| Resource | Link |
|----------|------|
| Paper | https://arxiv.org/abs/2507.09424 |
| Code | https://github.com/DataAttributionEval/DATE-LM |

---

## 2. Our Method: Task-Aware Bipartite Coverage (BipCov)

### 2.1 Core Idea

```
Training Pool D (200K)          Reference Set D_ref (100)
    ●─────────────────────────────●  (MMLU Q1)
    ●─────────────────────────────●  (MMLU Q2)
    ●                             ●  (MMLU Q3)
    ●─────────────────────────────●  ...
    ...
```

Unlike Rep-Sim (average similarity), we use:
- **Bipartite graph**: Training samples ↔ Validation tasks
- **Edge definition**: Semantic similarity > threshold τ
- **Selection**: Greedy max-coverage (submodular optimization)
- **Theoretical guarantee**: (1-1/e) approximation

### 2.2 Algorithm

```python
Input: Training data D, Validation data V, budget k
Output: Selected subset S of size k

1. Compute embeddings:
   train_emb = BGE.encode(D.instructions)      # (N, 1024)
   val_emb = BGE.encode(V.questions)           # (M, 1024)

2. Build bipartite graph:
   sim = train_emb @ val_emb.T                 # (N, M)
   A[i,j] = 1 if sim[i,j] >= τ

3. Lazy Greedy selection:
   S = {}, covered = {}
   for i in 1..k:
       x* = argmax_{x ∉ S} |N(x) ∩ uncovered|
       S = S ∪ {x*}
       covered = covered ∪ N(x*)
   return S
```

### 2.3 Method Comparison

| Aspect | Rep-Sim | LESS | **BipCov (Ours)** |
|--------|---------|------|-------------------|
| Selection | Top-k by avg sim | Top-k by influence | Greedy coverage |
| Diversity guarantee | ❌ | ❌ | ✅ (1-1/e) |
| Needs gradients | ❌ | ✅ | ❌ |
| FLOPs | ~6× | ~11× | **~2×** |

### 2.4 ML版 vs LLM版 BipCov 的区别

| 维度 | ML版 (`bipartite.py`) | LLM版 (`probe_bipcov_from_emb.py`) |
|------|----------------------|-----------------------------------|
| **核心思想** | 相同：贪心覆盖 + 次模代理 | 相同 |
| **表征** | 数值特征 + 距离→相似度 + **类别约束** | RAG embedding (BGE/E5) 或 LLM hidden-state，**无类别** |
| **阈值策略** | 类内相似度分位数 + **搜索最优阈值** | `top_l` 或 `target_density` **固定启发式** |
| **Budget机制** | 选完即止 | 必须选满10k（greedy饱和后用mean-sim填充） |

**为什么LLM版不能用ML版的阈值搜索？**

```
ML版:  阈值搜索成本 = O(|τ候选| × 训练+评估) ≈ 几分钟
LLM版: 阈值搜索成本 = O(|τ候选| × LoRA训练+3任务评估) ≈ 几十GPU小时
       → 不可行，只能用启发式
```

**LLM版的填充策略**：由于 ref 只有100条，greedy约26步后全覆盖饱和，剩余9974个名额用 mean-sim 排序填充（类似RepSim的top-k策略）。

---

## 3. Key Findings

### 3.1 BipCov方法有效性

在我们的实验环境中:
- **MMLU**: BipCov (61.59%) > Random (59.93%), +1.66%
- **GSM8K**: BipCov (64.59%) > Random (63.31%), +1.28%
- **平均**: BipCov (64.06%) > Random (63.26%), +0.80%

### 3.2 嵌入模型影响

不同嵌入模型在不同任务上表现最优:
- **BGE**: MMLU和平均最优
- **E5**: BBH最优 (+1.01% vs BGE)
- **MiniLM**: GSM8K最优 (+1.07% vs BGE)，尽管维度只有BGE的1/3

### 3.3 RepSim/RDS+在我们环境中效果有限

- RepSim v1/v2 和 RDS+ v1/v2 均未显著超越Random
- 去掉L2 normalization (v2) 反而降低性能
- 可能与我们的实现配置有关

### 3.4 训练效率

- 每个实验训练时间: ~40-50分钟 (156 steps)
- BBH评估时间: ~4-5小时 (27 subtasks)
- BipCov选择时间: <1秒 (200K samples)

---

## 4. 实验进度追踪

### ✅ 已完成

| 阶段 | 任务 | 状态 | 输出 |
|------|------|------|------|
| 1 | 环境配置 | ✅ | `pydvl_gpu` conda env |
| 2 | 下载Tulu3训练数据 (200K) | ✅ | `$DATELM_ROOT/data/training_data/` |
| 3 | 下载评估数据 (MMLU/GSM8K/BBH) | ✅ | `$DATELM_ROOT/data/eval/` |
| 4 | BGE嵌入计算 (200K×1024) | ✅ | `embeddings/paper_seed42_v1/mmlu/` |
| 5 | E5嵌入计算 | ✅ | `embeddings/paper_seed42_v1_e5/` |
| 6 | MiniLM嵌入计算 | ✅ | `embeddings/paper_seed42_v1_minilm/` |
| 7 | LLM嵌入计算 (4096d) | ✅ | `embeddings/paper_seed42_v1/` (all_orig.pt) |
| 8 | Random baseline | ✅ | MMLU 59.93%, GSM8K 63.31%, BBH 66.55% |
| 9 | BM25 baseline | ✅ | MMLU 61.32%, GSM8K 62.17%, BBH 66.83% |
| 10 | BipCov (BGE) | ✅ | MMLU 61.59%, GSM8K 64.59%, BBH 66.01% |
| 11 | BipCov (E5) | ✅ | MMLU 61.42%, GSM8K 63.23%, BBH 67.02% |
| 12 | BipCov (MiniLM) | ✅ | MMLU 59.92%, GSM8K 65.66%, BBH 64.57% |
| 13 | BipCov (LLM) | ✅ | MMLU 59.45%, GSM8K 64.97%, BBH 65.81% |
| 14 | RepSim v1/v2 | ✅ | 见主结果表 |
| 15 | RDS+ v1/v2 | ✅ | 见主结果表 |

### ⏳ 未完成

| 任务 | 优先级 | 说明 |
|------|--------|------|
| GradSim baseline | 🟡 中 | 需要梯度计算，开销大 |
| MATES baseline | 🟡 中 | DATE-LM论文方法 |
| EDU Classifier | 🟢 低 | 教育质量分类器 |
| 更多选择规模 (5K, 20K, 50K) | 🟢 低 | 消融实验 |

---

## 5. 目录结构

### 代码目录

```
finetuning/
├── README.md                          # 本文件
├── RETRAIN_PLAN.md                    # 🆕 重训清单 + 正确性验证指南
├── requirements-cpu.txt               # CPU依赖
├── requirements-gpu.txt               # GPU依赖
│
├── bipartite_greedy_coverage.py       # 核心: 二分图贪心覆盖
├── compute_embeddings_st.py           # SentenceTransformer嵌入
├── compute_embeddings_for_datelm.py   # DATE-LM格式嵌入
├── merge_lora_peft.py                 # LoRA合并脚本
│
├── bipcov/                            # DATE-LM方法模块
│   ├── __init__.py
│   ├── probe_bipcov_from_emb.py       # embedding → greedy → metrics.npy
│   └── README.md
│
├── scripts/
│   └── datelm_paper/
│       ├── train_lora_hf_paper.py     # HF Trainer LoRA训练
│       └── verify_run_artifacts.py    # 🆕 一键验收脚本
│
└── eval/                              # 评估脚本
```

### 实验输出 (DATELM_ROOT/)

其中 `$DATELM_ROOT` 是 DATE-LM 仓库根目录（包含 `minimal_multitask/`, `methods/`, `data/`）。

```
DATELM_ROOT/
├── data/
│   ├── training_data/
│   │   └── paper_seed42_v1_tulu3_200k_train.jsonl
│   └── eval/
│       ├── mmlu/, gsm/, bbh/
│
├── embeddings/
│   └── paper_seed42_v1/
│       ├── tulu3_train_bge/train_emb.npy
│       ├── mmlu_ref_bge/ref_emb.npy
│       ├── gsm8k_ref_bge/ref_emb.npy
│       ├── bbh_ref_bge/ref_emb.npy
│       ├── tulu3_train_llama_lasttok/train_emb.npy
│       ├── mmlu_ref_llama_lasttok/ref_emb.npy
│       └── ...
│
├── scores/
│   └── paper_seed42_v1/
│       ├── mmlu/*.npy
│       ├── gsm8k/*.npy
│       └── bbh/*.npy
│
├── selected_data/
│   └── paper_seed42_v1/{mmlu,gsm8k,bbh}/*.jsonl
│
├── checkpoints/
│   └── paper_seed42_v1_<METHOD>_<TASK>_{lora,merged}/
│
├── results/
│   └── paper_seed42_v1_<METHOD>_<TASK>_official/metrics.json
│
└── logs/*.log
```

---

## 6. 重训与验证

### 6.0 "完整重训" ≠ "100% DATE-LM codepath"

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATE-LM Benchmark 协议                        │
│  (同一 pool/budget/task/eval)                                    │
├─────────────────────────────────────────────────────────────────┤
│   DATE-LM 原版训练路径          我们的训练路径                    │
│   ────────────────────         ────────────────────              │
│   finetune.py (LitGPT)    vs   train_lora_hf_paper.py (HF+PEFT) │
│   - 内置 shuffle=False         - 默认 shuffle=True               │
│   - 内置 seed 设置             - 可用 --seed 42 --lora_alpha 256 │
│                                  --no_shuffle_train 贴近论文     │
└─────────────────────────────────────────────────────────────────┘
```

**结论**:
- "完整重训" = 在 DATE-LM benchmark 框架内跑完所有方法
- **不保证** 数字与 DATE-LM paper 完全一致（训练实现路径不同）
- **保证** 同环境相对比较有效

### 6.1 重训清单 (RETRAIN_PLAN.md)

详见 `RETRAIN_PLAN.md`，核心要点：

**Pipeline 6段**:
1. Pool → 2. Ref → 3. Embeddings → 4. Metrics/Score → 5. Selected JSONL → 6. LoRA Train/Merge/Eval

**重训决策表**:

| 改动内容 | 需重跑 |
|----------|--------|
| 只改训练超参 (alpha/seed/shuffle) | 仅 (6) |
| 改 selection 规则 | (4)-(6) |
| 改 embedding 模型 | (3)-(6) |
| 改 pool 采样 | (1)-(6) 全部 |

**推荐 v2 重训**（修复与论文偏差）:
```bash
--seed 42 --lora_alpha 256 --no_shuffle_train
```

### 6.2 验收脚本 (verify_run_artifacts.py)

一键检查 pipeline 产物一致性：

```bash
# 验证已完成的 v1 全链路
python finetuning/scripts/datelm_paper/verify_run_artifacts.py \
  --run_tag paper_seed42_v1 \
  --task mmlu --method bipcov \
  --check_training

# 复用 v1 selection，只重训 v2
python finetuning/scripts/datelm_paper/verify_run_artifacts.py \
  --run_tag paper_seed42_v2_train42_alpha256_noshuf \
  --data_tag paper_seed42_v1 \
  --task mmlu --method repsim \
  --check_training
```

**检查项**:
- `metrics.npy` 长度 == pool 行数 (199,999)
- `indices.npy` 长度 == 10,000，无重复
- `selected_jsonl` 与 (pool, indices) 对齐
- BipCov 的 `*_selected_indices.json` 与 top-k 一致
- checkpoint/merged/results 产物完整

**已验证**: `paper_seed42_v1` 的 BipCov/MMLU 全链路通过 ✓

---

## 7. 使用指南

### 7.1 计算嵌入

```bash
conda activate pydvl_gpu

python finetuning/compute_embeddings_for_datelm.py \
    --train_jsonl /path/to/tulu3.jsonl \
    --train_max_samples 200000 \
    --ref_hf cais/mmlu --ref_hf_subset all --ref_hf_split test \
    --ref_max_samples 100 \
    --out_dir /path/to/embeddings \
    --model BAAI/bge-large-en-v1.5 \
    --device cuda
```

### 7.2 运行BipCov选择

```bash
python finetuning/bipcov/probe_bipcov_from_emb.py \
    --train_emb embeddings/train_emb.npy \
    --ref_emb embeddings/ref_emb.npy \
    --out scores/bipcov_metrics.npy \
    --k_max 10000 \
    --target_density 0.02
```

### 7.3 集成到DATE-LM

```bash
# 复制方法模块
cp -r finetuning/bipcov /path/to/DATE-LM/methods/

# 在DATE-LM管道中使用
python train/finetune.py --metric_path scores/bipcov_metrics.npy
```

---

## 8. 论文写作建议

### 8.1 声明方式

由于基线复现存在偏差，建议论文中这样声明:

> We evaluate BipCov using the DATE-LM benchmark framework. Under identical experimental conditions (same training pipeline, evaluation protocol, and random seeds), BipCov achieves 64.06% average accuracy across MMLU/GSM8K/BBH, outperforming Random (63.26%) and our reproduced Rep-Sim (63.16%) baselines. Note that our reproduced Rep-Sim results differ slightly from those reported in the original DATE-LM paper, possibly due to implementation differences.

### 8.2 理论贡献

| DATE-LM方法 | ADP框架解释 |
|-------------|-------------|
| LESS | 梯度相似度作为线性代理 + 短视策略 |
| Rep-Sim | 嵌入相似度作为线性代理 + 短视策略 |
| **BipCov (Ours)** | 任务覆盖作为次模代理 + 贪心策略 + (1-1/e)保证 |

---

## 9. 文件索引

### 关键结果文件

| 文件 | 内容 |
|------|------|
| `$DATELM_ROOT/EXPERIMENT_RESULTS.md` | 详细实验结果 |
| `$DATELM_ROOT/results/COMPARISON_WITH_PAPER.md` | 与论文对比分析 |
| `$DATELM_ROOT/results/*/metrics.json` | 评估指标 |
| `$DATELM_ROOT/logs/*.log` | 训练/评估日志 |

### 核心代码文件

| 文件 | 功能 |
|------|------|
| `bipcov/probe_bipcov_from_emb.py` | BipCov选择算法 |
| `compute_embeddings_for_datelm.py` | 嵌入计算 |
| `../src/evaluators/bipartite.py` | ML版本二分图方法 |

---

## 10. 资源估算

### 计算需求

| 步骤 | 硬件 | 时间 |
|------|------|------|
| BGE嵌入计算 (200K) | 1× A100 | ~2小时 |
| BipCov选择 | CPU | <1秒 |
| LoRA微调 (每方法) | 1× A100 | ~40分钟 |
| MMLU评估 | 1× A100 | ~1小时 |
| GSM8K评估 | 1× A100 | ~30分钟 |
| BBH评估 | 1× A100 | ~4-5小时 |
| **单方法完整流程** | | ~6-7小时 |

### 存储需求

- Tulu3 (200K): ~2GB
- BGE嵌入: ~800MB
- LLM嵌入: ~3GB
- LoRA权重: ~160MB/方法

---

## 📝 更新日志

| 日期 | 更新 |
|------|------|
| 2025-12-27 | ✅ 完成所有实验，更新综合结果表，添加论文对比分析 |
| 2025-12-26 | ✅ BipCov E5/MiniLM 实验完成 |
| 2025-12-25 | ✅ RepSim v1/v2, RDS+ v1/v2 实验完成 |
| 2025-12-24 | ✅ BipCov LLM嵌入实验完成 |
| 2025-12-23 | ✅ BM25 baseline完成 |
| 2025-12-20 | ✅ BBH评估完成 |
| 2025-12-19 | ✅ 切换到官方DATE-LM评估协议 |
| 2025-12-18 | ✅ 修复模型(Llama-3.1-8B base)和超参数 |

---

## 参考文献

- **DATE-LM**: https://arxiv.org/abs/2507.09424
- **Tulu 3**: https://huggingface.co/allenai/tulu-3
- **BGE**: https://huggingface.co/BAAI/bge-large-en-v1.5
- **lm-evaluation-harness**: https://github.com/EleutherAI/lm-evaluation-harness
