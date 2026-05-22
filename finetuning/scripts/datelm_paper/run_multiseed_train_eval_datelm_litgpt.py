#!/usr/bin/env python3
"""
Multi-seed robustness experiment using the DATE-LM *LitGPT/Lightning* training codepath.

Protocol (same spirit as our HF runner, but with DATE-LM training):
  - Fixed selection (via fixed metric files; NO recomputation of embeddings/scores)
  - DATE-LM `train/finetune.py` (LitGPT LoRA) -> auto-merge -> `evaluation/convert.py`
  - DATE-LM official eval (minimal_multitask) for MMLU / GSM8K / BBH

Important:
  - For methods without a precomputed metrics.npy (e.g., rds_plus / rds_plus_v2),
    this script can synthesize a metric file from the existing `*_indices.npy`
    so that DATE-LM's `methods/select_data.select` chooses exactly those 10k indices.
  - We avoid using "random selection without metric_path" because DATE-LM's
    `select_from_size=200000` can mismatch the actual train split size (~199999),
    which risks out-of-range indices.

Default experiment grid:
  seeds   = [42, 1337, 2025]
  methods = [random, rds_plus, bipcov]
  tasks   = [mmlu, gsm8k, bbh]

Outputs (by default, separated from HF runs to avoid overwriting):
  - Training outputs:  $DATELM_ROOT/checkpoints_datelm_litgpt/<run_tag>_<method>_<task>/{final,...}
  - Eval outputs:      $DATELM_ROOT/results_datelm_litgpt/<run_tag>_<method>_<task>_official/metrics.json
  - Logs:              $DATELM_ROOT/logs_datelm_litgpt/<run_tag>_<method>_<task>_{finetune,convert,eval}.log
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


DEFAULT_SEEDS = [42, 1337, 2025]
DEFAULT_METHODS = ["random", "rds_plus", "bipcov"]
DEFAULT_TASKS = ["mmlu", "gsm8k", "bbh"]

DEFAULT_LITGPT_MODEL_NAME = "Llama-3-8B"  # LitGPT config name (Llama-3.1 weights are converted with this)

TRAIN_TIMEOUT_SECONDS = 12 * 3600
CONVERT_TIMEOUT_SECONDS = 2 * 3600
EVAL_TIMEOUT_SECONDS = 24 * 3600


def _default_proj_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(p: Path) -> Path:
    return p.expanduser().resolve()


def _looks_like_datelm_root(path: Path) -> bool:
    return (
        (path / "minimal_multitask").is_dir()
        and (path / "methods").is_dir()
        and (path / "requirements.txt").is_file()
        and (path / "train" / "finetune.py").is_file()
    )


def _infer_datelm_root(proj_root: Path) -> Path | None:
    candidate_bases: list[Path] = [proj_root]
    for i in range(4):
        try:
            candidate_bases.append(proj_root.parents[i])
        except IndexError:
            break
    for base in candidate_bases:
        for candidate in (base / "DATE-LM", base / "DATE-LM" / "DATE-LM-main", base / "DATE-LM-main"):
            if _looks_like_datelm_root(candidate):
                return candidate
    return None


def _resolve_datelm_root(arg: Path | None, proj_root: Path) -> Path:
    if arg is not None:
        return _resolve_path(arg)
    env = os.environ.get("DATELM_ROOT")
    if env:
        return _resolve_path(Path(env))
    inferred = _infer_datelm_root(proj_root)
    if inferred is not None:
        return _resolve_path(inferred)
    raise FileNotFoundError("DATE-LM root not found. Set --datelm_root or DATELM_ROOT.")


def _resolve_proj_root(arg: Path | None) -> Path:
    if arg is not None:
        return _resolve_path(arg)
    env = os.environ.get("SEQDATAVAL_ROOT")
    if env:
        return _resolve_path(Path(env))
    return _default_proj_root()


def _resolve_python(arg: str | None) -> str:
    if arg:
        return arg
    env = os.environ.get("PY_TRAIN_EVAL") or os.environ.get("SEQDATAVAL_PYTHON")
    if env:
        return env
    return sys.executable


def _default_train_data_path(datelm_root: Path) -> Path:
    candidates = [
        datelm_root / "data/training_data/tulu_3_v3.9_unfiltered.jsonl",
        # DATE-LM `data_processing/download_finetune_eval_data.sh` unzips into `data/training_data/training_data/`
        datelm_root / "data/training_data/training_data/tulu_3_v3.9_unfiltered.jsonl",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


@dataclass(frozen=True)
class MethodConfig:
    data_tag: str
    run_tag_template: str
    metric_path_template: Optional[str] = None
    indices_path_template: Optional[str] = None


def build_method_configs(datelm_root: Path) -> Dict[str, MethodConfig]:
    return {
        "random": MethodConfig(
            data_tag="paper_seed42_v1",
            run_tag_template="paper_seed42_v1_trainseed{seed}",
            metric_path_template=str(datelm_root / "scores/paper_seed42_v1/{task}/random_metrics.npy"),
            indices_path_template=str(datelm_root / "selected_data/paper_seed42_v1/{task}/random_indices.npy"),
        ),
        # Table-3 style 3-run random baseline: different random metric files (selection seeds 1/2/3).
        "random1": MethodConfig(
            data_tag="paper_seed42_v1",
            run_tag_template="paper_seed42_v1_trainseed{seed}",
            metric_path_template=str(datelm_root / "scores/paper_seed42_v1/{task}/random1_metrics.npy"),
        ),
        "random2": MethodConfig(
            data_tag="paper_seed42_v1",
            run_tag_template="paper_seed42_v1_trainseed{seed}",
            metric_path_template=str(datelm_root / "scores/paper_seed42_v1/{task}/random2_metrics.npy"),
        ),
        "random3": MethodConfig(
            data_tag="paper_seed42_v1",
            run_tag_template="paper_seed42_v1_trainseed{seed}",
            metric_path_template=str(datelm_root / "scores/paper_seed42_v1/{task}/random3_metrics.npy"),
        ),
        "bm25": MethodConfig(
            data_tag="paper_seed42_v1",
            run_tag_template="paper_seed42_v1_trainseed{seed}",
            metric_path_template=str(datelm_root / "scores/paper_seed42_v1/{task}/bm25_metrics.npy"),
            indices_path_template=str(datelm_root / "selected_data/paper_seed42_v1/{task}/bm25_indices.npy"),
        ),
        "repsim": MethodConfig(
            data_tag="paper_seed42_v1",
            run_tag_template="paper_seed42_v1_trainseed{seed}",
            metric_path_template=str(datelm_root / "scores/paper_seed42_v1/{task}/repsim_metrics.npy"),
            indices_path_template=str(datelm_root / "selected_data/paper_seed42_v1/{task}/repsim_indices.npy"),
        ),
        "repsim_v2": MethodConfig(
            data_tag="paper_seed42_v1",
            run_tag_template="paper_seed42_v1_trainseed{seed}",
            metric_path_template=str(datelm_root / "scores/paper_seed42_v1/{task}/repsim_v2_metrics.npy"),
            indices_path_template=str(datelm_root / "selected_data/paper_seed42_v1/{task}/repsim_v2_indices.npy"),
        ),
        "rds_plus": MethodConfig(
            data_tag="paper_seed42_v1",
            run_tag_template="paper_seed42_v1_trainseed{seed}",
            metric_path_template=str(datelm_root / "scores/paper_seed42_v1/{task}/rds_plus_metrics.npy"),
            indices_path_template=str(datelm_root / "selected_data/paper_seed42_v1/{task}/rds_plus_indices.npy"),
        ),
        "gradsim": MethodConfig(
            data_tag="paper_seed42_v1",
            run_tag_template="paper_seed42_v1_trainseed{seed}",
            metric_path_template=str(datelm_root / "scores/paper_seed42_v1/{task}/gradsim_metrics.npy"),
        ),
        "less": MethodConfig(
            data_tag="paper_seed42_v1",
            run_tag_template="paper_seed42_v1_trainseed{seed}",
            metric_path_template=str(datelm_root / "scores/paper_seed42_v1/{task}/less_metrics.npy"),
        ),
        "rds_plus_v2": MethodConfig(
            data_tag="paper_seed42_v1",
            run_tag_template="paper_seed42_v1_trainseed{seed}",
            metric_path_template=None,
            indices_path_template=str(datelm_root / "selected_data/paper_seed42_v1/{task}/rds_plus_v2_indices.npy"),
        ),
        "bipcov": MethodConfig(
            data_tag="paper_seed42_v1_refpromptlabel",
            run_tag_template="paper_seed42_v1_refpromptlabel_trainseed{seed}",
            metric_path_template=str(datelm_root / "scores/paper_seed42_v1_refpromptlabel/{task}/bipcov_metrics.npy"),
            indices_path_template=str(
                datelm_root / "selected_data/paper_seed42_v1_refpromptlabel/{task}/bipcov_indices.npy"
            ),
        ),
        # BipCov embedding ablations (method identical; representation differs).
        "bipcov_wmean": MethodConfig(
            data_tag="paper_seed42_v1_refpromptlabel",
            run_tag_template="paper_seed42_v1_refpromptlabel_trainseed{seed}",
            metric_path_template=str(
                datelm_root / "scores/paper_seed42_v1_refpromptlabel/{task}/bipcov_wmean_metrics.npy"
            ),
            indices_path_template=str(
                datelm_root / "selected_data/paper_seed42_v1_refpromptlabel/{task}/bipcov_wmean_indices.npy"
            ),
        ),
        "bipcov_bge": MethodConfig(
            data_tag="paper_seed42_v1_refpromptlabel",
            run_tag_template="paper_seed42_v1_refpromptlabel_trainseed{seed}",
            metric_path_template=str(
                datelm_root / "scores/paper_seed42_v1_refpromptlabel/{task}/bipcov_bge_metrics.npy"
            ),
            indices_path_template=str(
                datelm_root / "selected_data/paper_seed42_v1_refpromptlabel/{task}/bipcov_bge_indices.npy"
            ),
        ),
        "bipcov_e5": MethodConfig(
            data_tag="paper_seed42_v1_refpromptlabel",
            run_tag_template="paper_seed42_v1_refpromptlabel_trainseed{seed}",
            metric_path_template=str(
                datelm_root / "scores/paper_seed42_v1_refpromptlabel/{task}/bipcov_e5_metrics.npy"
            ),
            indices_path_template=str(
                datelm_root / "selected_data/paper_seed42_v1_refpromptlabel/{task}/bipcov_e5_indices.npy"
            ),
        ),
        "bipcov_qwen3emb": MethodConfig(
            data_tag="paper_seed42_v1_refpromptlabel",
            run_tag_template="paper_seed42_v1_refpromptlabel_trainseed{seed}",
            metric_path_template=str(
                datelm_root / "scores/paper_seed42_v1_refpromptlabel/{task}/bipcov_qwen3emb_metrics.npy"
            ),
            indices_path_template=str(
                datelm_root / "selected_data/paper_seed42_v1_refpromptlabel/{task}/bipcov_qwen3emb_indices.npy"
            ),
        ),
        "bipcov_nvembed": MethodConfig(
            data_tag="paper_seed42_v1_refpromptlabel",
            run_tag_template="paper_seed42_v1_refpromptlabel_trainseed{seed}",
            metric_path_template=str(
                datelm_root / "scores/paper_seed42_v1_refpromptlabel/{task}/bipcov_nvembed_metrics.npy"
            ),
            indices_path_template=str(
                datelm_root / "selected_data/paper_seed42_v1_refpromptlabel/{task}/bipcov_nvembed_indices.npy"
            ),
        ),
        "bipcov_gteqwen2": MethodConfig(
            data_tag="paper_seed42_v1_refpromptlabel",
            run_tag_template="paper_seed42_v1_refpromptlabel_trainseed{seed}",
            metric_path_template=str(
                datelm_root / "scores/paper_seed42_v1_refpromptlabel/{task}/bipcov_gteqwen2_metrics.npy"
            ),
            indices_path_template=str(
                datelm_root / "selected_data/paper_seed42_v1_refpromptlabel/{task}/bipcov_gteqwen2_indices.npy"
            ),
        ),
        "bipcov_gritlm": MethodConfig(
            data_tag="paper_seed42_v1_refpromptlabel",
            run_tag_template="paper_seed42_v1_refpromptlabel_trainseed{seed}",
            metric_path_template=str(
                datelm_root / "scores/paper_seed42_v1_refpromptlabel/{task}/bipcov_gritlm_metrics.npy"
            ),
            indices_path_template=str(
                datelm_root / "selected_data/paper_seed42_v1_refpromptlabel/{task}/bipcov_gritlm_indices.npy"
            ),
        ),
    }


def run_cmd(
    cmd: List[str],
    log_file: Path,
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    timeout_seconds: Optional[int] = None,
) -> bool:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    with open(log_file, "w") as f:
        try:
            r = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=str(cwd) if cwd else None,
                env=full_env,
                timeout=timeout_seconds,
            )
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            f.write(f"\n[TIMEOUT] after {timeout_seconds}s\n")
            return False


def ensure_metric_from_indices(
    indices_path: Path, out_metric_path: Path, selection_size: int = 10000
) -> Path:
    out_metric_path.parent.mkdir(parents=True, exist_ok=True)
    if out_metric_path.exists():
        return out_metric_path

    idx = np.load(indices_path).astype(np.int64)
    if idx.ndim != 1:
        raise ValueError(f"indices must be 1D: {indices_path}")
    if idx.size != selection_size:
        raise ValueError(f"Expected {selection_size} indices, got {idx.size}: {indices_path}")

    size = int(idx.max()) + 1
    metrics = np.zeros(size, dtype=np.float32)
    # Make all selected indices strictly larger than non-selected.
    # (Order does not matter for selection, only the set.)
    metrics[idx] = 1.0
    np.save(out_metric_path, metrics)
    return out_metric_path


def is_completed(metrics_json: Path) -> bool:
    return metrics_json.exists()


def extract_score(metrics: dict, task: str) -> float:
    if task == "mmlu":
        return float(metrics.get("average_acc", 0.0)) * 100
    if task == "gsm8k":
        return float(metrics.get("exact_match", 0.0)) * 100
    if task == "bbh":
        return float(metrics.get("average_exact_match", 0.0)) * 100
    return 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datelm_root", type=Path, default=None)
    p.add_argument("--proj_root", type=Path, default=None)
    p.add_argument("--python", type=str, default=None, help="Python executable for training/eval.")
    p.add_argument(
        "--litgpt_checkpoint_dir",
        type=Path,
        default=None,
        help="Directory containing lit_model.pth (LitGPT base checkpoint).",
    )
    p.add_argument(
        "--train_data_path",
        type=Path,
        default=None,
        help="Path to the full unfiltered training JSONL used by DATE-LM (tulu_3_v3.9_unfiltered.jsonl).",
    )
    p.add_argument(
        "--train_config",
        type=Path,
        default=None,
        help="DATE-LM LitGPT finetune config yaml (e.g., configs/train_finetune_llama3-8b.yaml).",
    )
    p.add_argument("--litgpt_model_name", type=str, default=DEFAULT_LITGPT_MODEL_NAME)
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=DEFAULT_METHODS,
        choices=[
            "random",
            "random1",
            "random2",
            "random3",
            "bm25",
            "repsim",
            "rds_plus",
            "gradsim",
            "less",
            "repsim_v2",
            "rds_plus_v2",
            "bipcov",
            "bipcov_wmean",
            "bipcov_bge",
            "bipcov_e5",
            "bipcov_qwen3emb",
            "bipcov_nvembed",
            "bipcov_gteqwen2",
            "bipcov_gritlm",
        ],
    )
    p.add_argument("--tasks", type=str, nargs="+", default=DEFAULT_TASKS, choices=DEFAULT_TASKS)
    p.add_argument("--results_dirname", type=str, default="results_datelm_litgpt")
    p.add_argument("--out_dirname", type=str, default="checkpoints_datelm_litgpt")
    p.add_argument("--log_dirname", type=str, default="logs_datelm_litgpt")
    p.add_argument("--mmlu_eval_batch_size", type=int, default=4)
    p.add_argument("--gsm_eval_batch_size", type=int, default=1)
    p.add_argument("--bbh_eval_batch_size", type=int, default=4)
    p.add_argument(
        "--gsm_use_vllm",
        action="store_true",
        help="Pass --use_vllm to minimal_multitask GSM8K eval (matches DATE-LM train/run_finetune.sh).",
    )
    p.add_argument(
        "--bbh_use_vllm",
        action="store_true",
        help="Pass --use_vllm to minimal_multitask BBH eval (matches DATE-LM train/run_finetune.sh).",
    )
    p.add_argument("--cleanup_checkpoints", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    proj_root = _resolve_proj_root(args.proj_root)
    datelm_root = _resolve_datelm_root(args.datelm_root, proj_root)
    python_exe = _resolve_python(args.python)

    method_cfgs = build_method_configs(datelm_root)

    litgpt_ckpt = _resolve_path(
        args.litgpt_checkpoint_dir
        if args.litgpt_checkpoint_dir is not None
        else datelm_root / "litgpt_checkpoints/meta-llama/Llama-3.1-8B"
    )
    train_data_path = _resolve_path(
        args.train_data_path
        if args.train_data_path is not None
        else _default_train_data_path(datelm_root)
    )
    train_config = _resolve_path(
        args.train_config
        if args.train_config is not None
        else datelm_root / "configs/train_finetune_llama3-8b.yaml"
    )

    if not litgpt_ckpt.exists() or not (litgpt_ckpt / "lit_model.pth").exists():
        raise FileNotFoundError(f"Missing LitGPT base checkpoint: {litgpt_ckpt}/lit_model.pth")
    if not train_data_path.exists():
        raise FileNotFoundError(str(train_data_path))
    if not train_config.exists():
        raise FileNotFoundError(str(train_config))

    train_script = datelm_root / "train/finetune.py"
    convert_script = datelm_root / "evaluation/convert.py"
    if not train_script.exists():
        raise FileNotFoundError(str(train_script))
    if not convert_script.exists():
        raise FileNotFoundError(str(convert_script))

    log_dir = datelm_root / args.log_dirname
    out_root = datelm_root / args.out_dirname
    results_root = datelm_root / args.results_dirname

    total = len(args.seeds) * len(args.methods) * len(args.tasks)
    print("DATE-LM LitGPT multi-seed runner")
    print(f"  datelm_root: {datelm_root}")
    print(f"  python:      {python_exe}")
    print(f"  train_data:  {train_data_path}")
    print(f"  base_ckpt:   {litgpt_ckpt}")
    print(f"  train_cfg:   {train_config}")
    print(f"  litgpt_name: {args.litgpt_model_name}")
    print(f"  out_root:    {out_root}")
    print(f"  results:     {results_root}")
    print(f"  logs:        {log_dir}")
    print(f"  total:       {total}")

    idx_counter = 0
    for seed in args.seeds:
        for method in args.methods:
            for task in args.tasks:
                idx_counter += 1
                cfg = method_cfgs[method]
                run_tag = cfg.run_tag_template.format(seed=seed)
                out_dir = out_root / f"{run_tag}_{method}_{task}"
                final_dir = out_dir / "final"
                results_dir = results_root / f"{run_tag}_{method}_{task}_official"
                metrics_json = results_dir / "metrics.json"

                finetune_log = log_dir / f"{run_tag}_{method}_{task}_finetune.log"
                convert_log = log_dir / f"{run_tag}_{method}_{task}_convert.log"
                eval_log = log_dir / f"{run_tag}_{method}_{task}_eval.log"

                print(f"\n[{idx_counter}/{total}] seed={seed} method={method} task={task}")

                if is_completed(metrics_json):
                    print(f"  [SKIP] metrics.json exists: {metrics_json}")
                    continue

                # Resolve metric_path (required for safety)
                metric_path: Optional[Path] = None
                if cfg.metric_path_template:
                    candidate_metric = Path(cfg.metric_path_template.format(task=task)).expanduser().resolve()
                    if candidate_metric.exists():
                        metric_path = candidate_metric

                # Fallback: synthesize a metric file from the canonical *_indices.npy
                if metric_path is None:
                    if not cfg.indices_path_template:
                        raise ValueError(
                            f"Missing metric file and indices fallback for method={method} task={task}. "
                            "Provide DATE-LM scores/ or selected_data/*_indices.npy."
                        )
                    indices_path = Path(cfg.indices_path_template.format(task=task)).expanduser().resolve()
                    if not indices_path.exists():
                        raise FileNotFoundError(
                            f"Missing indices for method={method} task={task}: {indices_path}"
                        )
                    metric_path = datelm_root / "scores_from_indices" / cfg.data_tag / task / f"{method}_metrics.npy"
                    metric_path = ensure_metric_from_indices(indices_path, metric_path)

                if not metric_path.exists():
                    raise FileNotFoundError(str(metric_path))

                if args.dry_run:
                    print(f"  [DRY] metric_path: {metric_path}")
                    print(f"  [DRY] out_dir:     {out_dir}")
                    print(f"  [DRY] results_dir: {results_dir}")
                    continue

                # (1) Finetune (LitGPT/Lightning)
                train_cmd = [
                    python_exe,
                    str(train_script),
                    "--config",
                    str(train_config),
                    "--train_data_path",
                    str(train_data_path),
                    "--out_dir",
                    str(out_dir),
                    "--model_name",
                    args.litgpt_model_name,
                    "--seed",
                    str(seed),
                    "--metric_path",
                    str(metric_path),
                    "--logger_name",
                    "csv",
                    "--exp_name",
                    f"{run_tag}_{method}_{task}",
                    # NOTE: DATE-LM's `train/finetune.py` expects `checkpoint_dir` as a positional argument.
                    str(litgpt_ckpt),
                ]
                ok = run_cmd(
                    train_cmd,
                    log_file=finetune_log,
                    cwd=datelm_root,
                    env={"NCCL_P2P_DISABLE": "1", "PYTHONUNBUFFERED": "1", "WANDB_MODE": "disabled"},
                    timeout_seconds=TRAIN_TIMEOUT_SECONDS,
                )
                if not ok:
                    raise RuntimeError(f"Finetune failed. See {finetune_log}")

                # (2) Convert to HF
                convert_cmd = [python_exe, str(convert_script), str(final_dir)]
                ok = run_cmd(
                    convert_cmd,
                    log_file=convert_log,
                    cwd=datelm_root,
                    env={"PYTHONUNBUFFERED": "1"},
                    timeout_seconds=CONVERT_TIMEOUT_SECONDS,
                )
                if not ok:
                    raise RuntimeError(f"Convert failed. See {convert_log}")

                # (3) Official eval
                results_dir.mkdir(parents=True, exist_ok=True)
                if task == "mmlu":
                    eval_cmd = [
                        python_exe,
                        "-m",
                        "minimal_multitask.eval.mmlu.run_mmlu_eval",
                        "--ntrain",
                        "0",
                        "--data_dir",
                        "data/eval/mmlu",
                        "--save_dir",
                        str(results_dir),
                        "--model_name_or_path",
                        str(final_dir),
                        "--eval_batch_size",
                        str(args.mmlu_eval_batch_size),
                        "--use_chat_format",
                        "--chat_formatting_function",
                        "minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format",
                    ]
                elif task == "gsm8k":
                    eval_cmd = [
                        python_exe,
                        "-m",
                        "minimal_multitask.eval.gsm.run_eval",
                        "--data_dir",
                        "data/eval/gsm",
                        "--save_dir",
                        str(results_dir),
                        "--model_name_or_path",
                        str(final_dir),
                        "--n_shot",
                        "8",
                        "--eval_batch_size",
                        str(args.gsm_eval_batch_size),
                        "--use_chat_format",
                        "--chat_formatting_function",
                        "minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format",
                    ]
                    if args.gsm_use_vllm:
                        eval_cmd.append("--use_vllm")
                else:  # bbh
                    eval_cmd = [
                        python_exe,
                        "-m",
                        "minimal_multitask.eval.bbh.run_eval",
                        "--data_dir",
                        "data/eval/bbh",
                        "--save_dir",
                        str(results_dir),
                        "--model_name_or_path",
                        str(final_dir),
                        "--eval_batch_size",
                        str(args.bbh_eval_batch_size),
                        "--use_chat_format",
                        "--chat_formatting_function",
                        "minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format",
                    ]
                    if args.bbh_use_vllm:
                        eval_cmd.append("--use_vllm")

                ok = run_cmd(
                    eval_cmd,
                    log_file=eval_log,
                    cwd=datelm_root,
                    env={"PYTHONUNBUFFERED": "1"},
                    timeout_seconds=EVAL_TIMEOUT_SECONDS,
                )
                if not ok:
                    raise RuntimeError(f"Eval failed. See {eval_log}")

                if metrics_json.exists():
                    with open(metrics_json) as f:
                        metrics = json.load(f)
                    score = extract_score(metrics, task)
                    print(f"  [DONE] {task}: {score:.2f}%")
                else:
                    print(f"  [WARN] metrics.json not found after eval: {metrics_json}")

                if args.cleanup_checkpoints:
                    # Keep results/logs/scores; remove only out_dir for disk.
                    if out_dir.exists():
                        shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
