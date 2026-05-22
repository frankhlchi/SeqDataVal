#!/usr/bin/env python3
"""
Multi-seed robustness experiment: Train + Merge + Official Eval only.

This script runs LoRA training, merging, and official evaluation for multiple
training seeds, using FIXED selected_data (no recomputation of pool/embeddings/scores/selection).

Usage:
    python run_multiseed_train_eval_only.py [--datelm_root /path/to/DATE-LM] [--python /path/to/python] \
      [--seeds 42 1337 2025] [--methods random rds_plus bipcov] [--tasks mmlu gsm8k bbh]

Example (run all 27 combinations):
    python run_multiseed_train_eval_only.py

Example (run only mmlu and gsm8k first):
    python run_multiseed_train_eval_only.py --tasks mmlu gsm8k

Example (run additional baselines; fixed selected_data, no reselection):
    python run_multiseed_train_eval_only.py --methods bm25 repsim repsim_v2 rds_plus_v2 --tasks mmlu gsm8k bbh

Environment variables (optional):
  - DATELM_ROOT: Path to DATE-LM repo root (or DATE-LM-main folder)
  - SEQDATAVAL_ROOT: Path to this repo (auto-inferred by default)
  - PY_TRAIN_EVAL / SEQDATAVAL_PYTHON: Python interpreter to use for training/eval
  - SEQDATAVAL_BASE_MODEL / BASE_MODEL: HF model id/path for base model
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_SEEDS = [42, 1337, 2025]
DEFAULT_METHODS = ["random", "rds_plus", "bipcov"]
DEFAULT_TASKS = ["mmlu", "gsm8k", "bbh"]

# Long-running steps (especially BBH eval) can exceed 2 hours; keep generous timeouts.
# NOTE: Increase further or set to None if your cluster is slow.
TRAIN_TIMEOUT_SECONDS = 8 * 3600
MERGE_TIMEOUT_SECONDS = 2 * 3600
EVAL_TIMEOUT_SECONDS = 24 * 3600
VERIFY_TIMEOUT_SECONDS = 2 * 3600

DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B"


def _default_proj_root() -> Path:
    # .../finetuning/scripts/datelm_paper/run_multiseed_train_eval_only.py -> repo root is 3 parents up
    return Path(__file__).resolve().parents[3]


def _resolve_path(p: Path) -> Path:
    return p.expanduser().resolve()

def _looks_like_datelm_root(path: Path) -> bool:
    return (
        (path / "minimal_multitask").is_dir()
        and (path / "methods").is_dir()
        and (path / "requirements.txt").is_file()
    )


def _infer_datelm_root(proj_root: Path) -> Path | None:
    # Try a few common layouts to reduce friction when moving machines.
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
    raise FileNotFoundError(
        "DATE-LM root not found. Set `--datelm_root /path/to/DATE-LM` "
        "or export `DATELM_ROOT=/path/to/DATE-LM`."
    )


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


def _resolve_base_model(arg: str | None) -> str:
    if arg:
        return arg
    env = os.environ.get("SEQDATAVAL_BASE_MODEL") or os.environ.get("BASE_MODEL")
    if env:
        return env
    return DEFAULT_BASE_MODEL


def _build_selected_data_config(datelm_root: Path) -> dict:
    # Selected data paths (fixed, not recomputed)
    return {
        "random": {
            "data_tag": "paper_seed42_v1",
            "run_tag_template": "paper_seed42_v1_trainseed{seed}",
            "jsonl_template": str(datelm_root / "selected_data/paper_seed42_v1/{task}/random_10k.jsonl"),
        },
        "bm25": {
            "data_tag": "paper_seed42_v1",
            "run_tag_template": "paper_seed42_v1_trainseed{seed}",
            "jsonl_template": str(datelm_root / "selected_data/paper_seed42_v1/{task}/bm25_10k.jsonl"),
        },
        "repsim": {
            "data_tag": "paper_seed42_v1",
            "run_tag_template": "paper_seed42_v1_trainseed{seed}",
            "jsonl_template": str(datelm_root / "selected_data/paper_seed42_v1/{task}/repsim_10k.jsonl"),
        },
        "repsim_v2": {
            "data_tag": "paper_seed42_v1",
            "run_tag_template": "paper_seed42_v1_trainseed{seed}",
            "jsonl_template": str(datelm_root / "selected_data/paper_seed42_v1/{task}/repsim_v2_10k.jsonl"),
        },
        "rds_plus": {
            "data_tag": "paper_seed42_v1",
            "run_tag_template": "paper_seed42_v1_trainseed{seed}",
            "jsonl_template": str(datelm_root / "selected_data/paper_seed42_v1/{task}/rds_plus_10k.jsonl"),
        },
        "rds_plus_v2": {
            "data_tag": "paper_seed42_v1",
            "run_tag_template": "paper_seed42_v1_trainseed{seed}",
            "jsonl_template": str(datelm_root / "selected_data/paper_seed42_v1/{task}/rds_plus_v2_10k.jsonl"),
        },
        "bipcov": {
            "data_tag": "paper_seed42_v1_refpromptlabel",
            "run_tag_template": "paper_seed42_v1_refpromptlabel_trainseed{seed}",
            "jsonl_template": str(datelm_root / "selected_data/paper_seed42_v1_refpromptlabel/{task}/bipcov_10k.jsonl"),
        },
    }


@dataclass(frozen=True)
class RunnerConfig:
    datelm_root: Path
    proj_root: Path
    python: str
    base_model: str
    train_script: Path
    merge_script: Path
    verify_script: Path
    selected_data_config: dict
    cleanup_checkpoints: bool
    mmlu_eval_batch_size: int
    gsm_eval_batch_size: int
    bbh_eval_batch_size: int


def get_paths(cfg: RunnerConfig, seed: int, method: str, task: str) -> dict:
    """Get all paths for a given (seed, method, task) combination."""
    config = cfg.selected_data_config[method]
    run_tag = config["run_tag_template"].format(seed=seed)
    data_tag = config["data_tag"]
    train_jsonl = config["jsonl_template"].format(task=task)

    return {
        "run_tag": run_tag,
        "data_tag": data_tag,
        "train_jsonl": train_jsonl,
        "adapter_dir": cfg.datelm_root / f"checkpoints/{run_tag}_{method}_{task}_lora",
        "merged_dir": cfg.datelm_root / f"checkpoints/{run_tag}_{method}_{task}_merged",
        "results_dir": cfg.datelm_root / f"results/{run_tag}_{method}_{task}_official",
        "log_dir": cfg.datelm_root / "logs",
        "finetune_log": cfg.datelm_root / f"logs/{run_tag}_{method}_{task}_finetune.log",
        "eval_log": cfg.datelm_root / f"logs/{run_tag}_{method}_{task}_eval.log",
    }


def run_cmd(
    cmd: List[str],
    log_file: Path | None = None,
    cwd: Path | None = None,
    env: dict | None = None,
    timeout_seconds: int | None = None,
) -> bool:
    """Run a command and return success status."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    print(f"  [CMD] {' '.join(cmd[:5])}...")

    try:
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "w") as f:
                result = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=cwd,
                    env=full_env,
                    timeout=timeout_seconds,
                )
        else:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=cwd,
                env=full_env,
                timeout=timeout_seconds,
            )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        if timeout_seconds is None:
            print("  [ERROR] Command timed out")
        else:
            print(f"  [ERROR] Command timed out after {timeout_seconds/3600:.1f} hours")
        return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def check_train_jsonl_exists(train_jsonl: str) -> bool:
    """Check if the training JSONL exists."""
    return Path(train_jsonl).exists()


def is_completed(paths: dict) -> bool:
    """Check if this combination is already completed (metrics.json exists)."""
    metrics_file = paths["results_dir"] / "metrics.json"
    return metrics_file.exists()


def run_train(cfg: RunnerConfig, seed: int, paths: dict) -> bool:
    """Run LoRA training."""
    adapter_dir = paths["adapter_dir"]

    # Skip if adapter already exists
    if (adapter_dir / "adapter_config.json").exists():
        print(f"  [SKIP] Adapter exists: {adapter_dir}")
        return True

    cmd = [
        cfg.python,
        str(cfg.train_script),
        "--datelm_root", str(cfg.datelm_root),
        "--train_jsonl", paths["train_jsonl"],
        "--output_dir", str(adapter_dir),
        "--model_name", cfg.base_model,
        "--seed", str(seed),
    ]

    paths["log_dir"].mkdir(parents=True, exist_ok=True)

    success = run_cmd(
        cmd,
        log_file=paths["finetune_log"],
        env={"PYTHONUNBUFFERED": "1"},
        timeout_seconds=TRAIN_TIMEOUT_SECONDS,
    )

    if not success:
        print(f"  [ERROR] Training failed. See log: {paths['finetune_log']}")
        # Print last 20 lines of log
        if paths["finetune_log"].exists():
            with open(paths["finetune_log"]) as f:
                lines = f.readlines()
                print("  --- Last 20 lines of log ---")
                for line in lines[-20:]:
                    print(f"  {line.rstrip()}")

    return success


def run_merge(cfg: RunnerConfig, paths: dict) -> bool:
    """Run LoRA merging."""
    merged_dir = paths["merged_dir"]

    # Skip if merged model already exists
    if (merged_dir / "config.json").exists():
        print(f"  [SKIP] Merged model exists: {merged_dir}")
        return True

    cmd = [
        cfg.python,
        str(cfg.merge_script),
        "--adapter_path", str(paths["adapter_dir"]),
        "--base_model", cfg.base_model,
        "--output_path", str(merged_dir),
    ]

    success = run_cmd(cmd, timeout_seconds=MERGE_TIMEOUT_SECONDS)

    if not success:
        print(f"  [ERROR] Merging failed")

    return success


def run_eval(cfg: RunnerConfig, task: str, paths: dict) -> bool:
    """Run official evaluation."""
    results_dir = paths["results_dir"]
    merged_dir = paths["merged_dir"]

    # Skip if results already exist
    if (results_dir / "metrics.json").exists():
        print(f"  [SKIP] Results exist: {results_dir}")
        return True

    results_dir.mkdir(parents=True, exist_ok=True)

    if task == "mmlu":
        cmd = [
            cfg.python, "-m", "minimal_multitask.eval.mmlu.run_mmlu_eval",
            "--ntrain", "0",
            "--data_dir", "data/eval/mmlu",
            "--save_dir", str(results_dir),
            "--model_name_or_path", str(merged_dir),
            "--eval_batch_size", str(cfg.mmlu_eval_batch_size),
            "--use_chat_format",
            "--chat_formatting_function", "minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format",
        ]
    elif task == "gsm8k":
        cmd = [
            cfg.python, "-m", "minimal_multitask.eval.gsm.run_eval",
            "--data_dir", "data/eval/gsm",
            "--save_dir", str(results_dir),
            "--model_name_or_path", str(merged_dir),
            "--n_shot", "8",
            "--eval_batch_size", str(cfg.gsm_eval_batch_size),
            "--use_chat_format",
            "--chat_formatting_function", "minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format",
        ]
    elif task == "bbh":
        cmd = [
            cfg.python, "-m", "minimal_multitask.eval.bbh.run_eval",
            "--data_dir", "data/eval/bbh",
            "--save_dir", str(results_dir),
            "--model_name_or_path", str(merged_dir),
            "--eval_batch_size", str(cfg.bbh_eval_batch_size),
            "--use_chat_format",
            "--chat_formatting_function", "minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format",
        ]
    else:
        print(f"  [ERROR] Unknown task: {task}")
        return False

    success = run_cmd(cmd, log_file=paths["eval_log"], cwd=cfg.datelm_root, timeout_seconds=EVAL_TIMEOUT_SECONDS)

    if not success:
        print(f"  [ERROR] Evaluation failed. See log: {paths['eval_log']}")
        if paths["eval_log"].exists():
            with open(paths["eval_log"]) as f:
                lines = f.readlines()
                print("  --- Last 20 lines of log ---")
                for line in lines[-20:]:
                    print(f"  {line.rstrip()}")

    return success


def run_verify(cfg: RunnerConfig, method: str, task: str, paths: dict) -> bool:
    """Run artifact verification."""
    cmd = [
        cfg.python,
        str(cfg.verify_script),
        "--run_tag", paths["run_tag"],
        "--data_tag", paths["data_tag"],
        "--task", task,
        "--method", method,
        "--check_training",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cfg.proj_root,
        timeout=VERIFY_TIMEOUT_SECONDS,
    )

    if result.returncode != 0:
        print(f"  [ERROR] Verification failed:")
        print(result.stdout)
        print(result.stderr)
        return False

    return True


def _cleanup_checkpoints(paths: dict) -> None:
    for key in ("adapter_dir", "merged_dir"):
        path = paths.get(key)
        if isinstance(path, Path) and path.exists():
            shutil.rmtree(path)


def run_single_combination(
    cfg: RunnerConfig,
    seed: int,
    method: str,
    task: str,
    skip_verify: bool = False,
) -> Tuple[bool, dict]:
    """Run a single (seed, method, task) combination."""
    paths = get_paths(cfg, seed, method, task)

    print(f"\n{'='*60}")
    print(f"[{seed}/{method}/{task}] run_tag={paths['run_tag']}")
    print(f"{'='*60}")

    # Check if already completed
    if is_completed(paths):
        print(f"  [SKIP] Already completed: {paths['results_dir']}/metrics.json")
        return True, paths

    # Check train_jsonl exists
    if not check_train_jsonl_exists(paths["train_jsonl"]):
        print(f"  [ERROR] Train JSONL not found: {paths['train_jsonl']}")
        return False, paths

    # Step 1: Train
    print(f"  [1/4] Training LoRA...")
    if not run_train(cfg, seed, paths):
        return False, paths

    # Step 2: Merge
    print(f"  [2/4] Merging adapter...")
    if not run_merge(cfg, paths):
        return False, paths

    # Step 3: Eval
    print(f"  [3/4] Running official {task} eval...")
    if not run_eval(cfg, task, paths):
        return False, paths

    # Step 4: Verify
    if not skip_verify:
        print(f"  [4/4] Verifying artifacts...")
        if not run_verify(cfg, method, task, paths):
            return False, paths
    else:
        print(f"  [4/4] Skipping verification (--skip_verify)")

    # Print result
    metrics_file = paths["results_dir"] / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
        if task == "mmlu":
            score = metrics.get("average_acc", 0) * 100
        elif task == "gsm8k":
            score = metrics.get("exact_match", 0) * 100
        elif task == "bbh":
            score = metrics.get("average_exact_match", 0) * 100
        else:
            score = 0
        print(f"  [DONE] {task}: {score:.2f}%")

    if cfg.cleanup_checkpoints:
        print("  [CLEANUP] Removing checkpoints (adapter + merged)")
        try:
            _cleanup_checkpoints(paths)
        except Exception as e:
            print(f"  [WARN] Cleanup failed: {e}")

    return True, paths


def main():
    parser = argparse.ArgumentParser(description="Multi-seed robustness experiment (train+merge+eval only)")
    parser.add_argument(
        "--datelm_root",
        type=Path,
        default=None,
        help="Path to DATE-LM root (repo root or DATE-LM-main), or set DATELM_ROOT env var.",
    )
    parser.add_argument(
        "--proj_root",
        type=Path,
        default=None,
        help="Path to SeqDataVal repo root (auto-inferred; or set SEQDATAVAL_ROOT env var).",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=None,
        help="Python interpreter for training/eval (or set PY_TRAIN_EVAL / SEQDATAVAL_PYTHON).",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help=f"HF base model id/path (default: {DEFAULT_BASE_MODEL}; or set BASE_MODEL / SEQDATAVAL_BASE_MODEL).",
    )
    parser.add_argument(
        "--mmlu_eval_batch_size",
        type=int,
        default=4,
        help="MMLU official eval batch size (default: 4).",
    )
    parser.add_argument(
        "--gsm_eval_batch_size",
        type=int,
        default=1,
        help="GSM8K official eval batch size (default: 1).",
    )
    parser.add_argument(
        "--bbh_eval_batch_size",
        type=int,
        default=4,
        help="BBH official eval batch size (default: 4).",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help=f"Training seeds (default: {DEFAULT_SEEDS})")
    parser.add_argument("--methods", type=str, nargs="+", default=DEFAULT_METHODS,
                        choices=["random", "bm25", "repsim", "repsim_v2", "rds_plus", "rds_plus_v2", "bipcov"],
                        help=f"Methods (default: {DEFAULT_METHODS})")
    parser.add_argument("--tasks", type=str, nargs="+", default=DEFAULT_TASKS,
                        choices=["mmlu", "gsm8k", "bbh"],
                        help=f"Tasks (default: {DEFAULT_TASKS})")
    parser.add_argument("--skip_verify", action="store_true",
                        help="Skip artifact verification after each combination")
    parser.add_argument(
        "--cleanup_checkpoints",
        action="store_true",
        help="Delete adapter+merged checkpoints after each successful combination (saves disk).",
    )
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would be run without actually running")
    args = parser.parse_args()

    proj_root = _resolve_proj_root(args.proj_root)
    datelm_root = _resolve_datelm_root(args.datelm_root, proj_root)
    python_exe = _resolve_python(args.python)
    base_model = _resolve_base_model(args.base_model)

    train_script = proj_root / "finetuning/scripts/datelm_paper/train_lora_hf_paper.py"
    merge_script = proj_root / "finetuning/merge_lora_peft.py"
    verify_script = proj_root / "finetuning/scripts/datelm_paper/verify_run_artifacts.py"
    for script_path in (train_script, merge_script, verify_script):
        if not script_path.exists():
            raise FileNotFoundError(str(script_path))

    cfg = RunnerConfig(
        datelm_root=datelm_root,
        proj_root=proj_root,
        python=python_exe,
        base_model=base_model,
        train_script=train_script,
        merge_script=merge_script,
        verify_script=verify_script,
        selected_data_config=_build_selected_data_config(datelm_root),
        cleanup_checkpoints=bool(args.cleanup_checkpoints),
        mmlu_eval_batch_size=int(args.mmlu_eval_batch_size),
        gsm_eval_batch_size=int(args.gsm_eval_batch_size),
        bbh_eval_batch_size=int(args.bbh_eval_batch_size),
    )

    seeds = args.seeds
    methods = args.methods
    tasks = args.tasks

    total = len(seeds) * len(methods) * len(tasks)
    print(f"Multi-seed robustness experiment")
    print(f"  datelm_root: {cfg.datelm_root}")
    print(f"  proj_root:   {cfg.proj_root}")
    print(f"  python:      {cfg.python}")
    print(f"  base_model:  {cfg.base_model}")
    print(f"  cleanup:     {cfg.cleanup_checkpoints}")
    print(
        f"  eval_batch: mmlu={cfg.mmlu_eval_batch_size}, gsm={cfg.gsm_eval_batch_size}, bbh={cfg.bbh_eval_batch_size}"
    )
    print(f"  Seeds: {seeds}")
    print(f"  Methods: {methods}")
    print(f"  Tasks: {tasks}")
    print(f"  Total combinations: {total}")
    print()

    # Check all train_jsonl files exist
    print("Checking selected_data files...")
    all_exist = True
    for method in methods:
        for task in tasks:
            paths = get_paths(cfg, seeds[0], method, task)  # seed doesn't matter for jsonl path
            if not check_train_jsonl_exists(paths["train_jsonl"]):
                print(f"  [MISSING] {paths['train_jsonl']}")
                all_exist = False
            else:
                print(f"  [OK] {paths['train_jsonl']}")

    if not all_exist:
        print("\n[ERROR] Some selected_data files are missing. Aborting.")
        sys.exit(1)

    if args.dry_run:
        print("\n[DRY RUN] Would run the following combinations:")
        for seed in seeds:
            for method in methods:
                for task in tasks:
                    paths = get_paths(cfg, seed, method, task)
                    status = "SKIP (exists)" if is_completed(paths) else "RUN"
                    print(f"  [{status}] seed={seed}, method={method}, task={task}, run_tag={paths['run_tag']}")
        sys.exit(0)

    # Run all combinations
    results = []
    failed = []

    for i, seed in enumerate(seeds):
        for j, method in enumerate(methods):
            for k, task in enumerate(tasks):
                idx = i * len(methods) * len(tasks) + j * len(tasks) + k + 1
                print(f"\n[{idx}/{total}] Starting: seed={seed}, method={method}, task={task}")

                success, paths = run_single_combination(cfg, seed, method, task, skip_verify=args.skip_verify)

                if success:
                    results.append((seed, method, task, paths))
                else:
                    failed.append((seed, method, task, paths))
                    print(f"\n[ERROR] Combination failed: seed={seed}, method={method}, task={task}")
                    print(f"[ERROR] Stopping execution due to failure.")
                    sys.exit(1)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Completed: {len(results)}/{total}")
    print(f"  Failed: {len(failed)}")

    if failed:
        print("\nFailed combinations:")
        for seed, method, task, paths in failed:
            print(f"  - seed={seed}, method={method}, task={task}")

    print("\nTo generate summary tables, run:")
    print(
        f"  python {cfg.proj_root}/finetuning/scripts/datelm_paper/summarize_multiseed_results.py --datelm_root {cfg.datelm_root}"
    )


if __name__ == "__main__":
    main()
