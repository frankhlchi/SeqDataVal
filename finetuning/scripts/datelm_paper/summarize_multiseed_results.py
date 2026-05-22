#!/usr/bin/env python3
"""
Summarize multi-seed robustness experiment results.

Reads all <results_dirname>/*/metrics.json and generates:
- MULTISEED_TRAINSEED_RESULTS.csv
- MULTISEED_TRAINSEED_RESULTS.md

Usage:
    python summarize_multiseed_results.py [--datelm_root /path/to/DATE-LM] \
      [--seeds 42 1337 2025] [--methods random rds_plus bipcov] [--tasks mmlu gsm8k bbh]

Environment variables (optional):
  - DATELM_ROOT: Path to DATE-LM repo root (or DATE-LM-main folder)
  - SEQDATAVAL_ROOT: Path to this repo (auto-inferred by default)
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_SEEDS = [42, 1337, 2025]
DEFAULT_METHODS = ["random", "rds_plus", "bipcov"]
DEFAULT_TASKS = ["mmlu", "gsm8k", "bbh"]

# Mapping from method to run_tag template
RUN_TAG_TEMPLATES = {
    "random": "paper_seed42_v1_trainseed{seed}",
    "random1": "paper_seed42_v1_trainseed{seed}",
    "random2": "paper_seed42_v1_trainseed{seed}",
    "random3": "paper_seed42_v1_trainseed{seed}",
    # Virtual method (computed from random1/2/3); no on-disk metrics.json.
    "random_avg": "paper_seed42_v1_trainseed{seed}",
    "bm25": "paper_seed42_v1_trainseed{seed}",
    "repsim": "paper_seed42_v1_trainseed{seed}",
    "rds_plus": "paper_seed42_v1_trainseed{seed}",
    "gradsim": "paper_seed42_v1_trainseed{seed}",
    "less": "paper_seed42_v1_trainseed{seed}",
    "repsim_v2": "paper_seed42_v1_trainseed{seed}",
    "rds_plus_v2": "paper_seed42_v1_trainseed{seed}",
    "bipcov": "paper_seed42_v1_refpromptlabel_trainseed{seed}",
    "bipcov_wmean": "paper_seed42_v1_refpromptlabel_trainseed{seed}",
    "bipcov_bge": "paper_seed42_v1_refpromptlabel_trainseed{seed}",
    "bipcov_e5": "paper_seed42_v1_refpromptlabel_trainseed{seed}",
    "bipcov_qwen3emb": "paper_seed42_v1_refpromptlabel_trainseed{seed}",
    "bipcov_nvembed": "paper_seed42_v1_refpromptlabel_trainseed{seed}",
    "bipcov_gteqwen2": "paper_seed42_v1_refpromptlabel_trainseed{seed}",
    "bipcov_gritlm": "paper_seed42_v1_refpromptlabel_trainseed{seed}",
}

# Method display names for tables
METHOD_DISPLAY_NAMES = {
    "random": "Random",
    "random1": "Random 1",
    "random2": "Random 2",
    "random3": "Random 3",
    "random_avg": "Random Avg",
    "bm25": "BM25",
    "repsim": "RepSim",
    "rds_plus": "RDS+",
    "gradsim": "Grad Sim",
    "less": "LESS",
    "repsim_v2": "RepSim (v2)",
    "rds_plus_v2": "RDS+ (v2)",
    "bipcov": "BipCov (ref-aligned)",
    "bipcov_wmean": "BipCov (weighted-mean emb)",
    "bipcov_bge": "BipCov (BGE emb)",
    "bipcov_e5": "BipCov (E5 emb)",
    "bipcov_qwen3emb": "BipCov (Qwen3-Emb-8B)",
    "bipcov_nvembed": "BipCov (NV-Embed-v2)",
    "bipcov_gteqwen2": "BipCov (GTE-Qwen2-7B)",
    "bipcov_gritlm": "BipCov (GritLM-7B)",
}


def _default_proj_root() -> Path:
    # .../finetuning/scripts/datelm_paper/summarize_multiseed_results.py -> repo root is 3 parents up
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


def get_results_path(datelm_root: Path, results_dirname: str, seed: int, method: str, task: str) -> Path:
    """Get the path to metrics.json for a given combination."""
    run_tag = RUN_TAG_TEMPLATES[method].format(seed=seed)
    return datelm_root / results_dirname / f"{run_tag}_{method}_{task}_official/metrics.json"


def extract_score(metrics: dict, task: str) -> Optional[float]:
    """Extract the primary score from metrics dict."""
    if task == "mmlu":
        return metrics.get("average_acc")
    elif task == "gsm8k":
        return metrics.get("exact_match")
    elif task == "bbh":
        return metrics.get("average_exact_match")
    return None


def collect_results(
    datelm_root: Path,
    results_dirname: str,
    seeds: List[int],
    methods: List[str],
    tasks: List[str]
) -> Dict[str, Dict[str, Dict[int, float]]]:
    """
    Collect all results into a nested dict:
    results[method][task][seed] = score (0-1 scale)
    """
    results = defaultdict(lambda: defaultdict(dict))

    scan_methods = list(methods)
    if "random_avg" in scan_methods:
        # Ensure we can compute Random Avg even if caller forgot to include the components.
        for m in ("random1", "random2", "random3"):
            if m not in scan_methods:
                scan_methods.append(m)

    for method in scan_methods:
        for task in tasks:
            for seed in seeds:
                if method == "random_avg":
                    continue
                metrics_path = get_results_path(datelm_root, results_dirname, seed, method, task)
                if metrics_path.exists():
                    try:
                        with open(metrics_path) as f:
                            metrics = json.load(f)
                        score = extract_score(metrics, task)
                        if score is not None:
                            results[method][task][seed] = score
                    except Exception as e:
                        print(f"[WARN] Failed to read {metrics_path}: {e}")

    return results


def compute_stats(scores: Dict[int, float]) -> Tuple[float, float]:
    """Compute mean and std from a dict of seed->score."""
    if not scores:
        return 0.0, 0.0
    values = list(scores.values())
    return np.mean(values), np.std(values)


def format_pct(mean: float, std: float, show_std: bool = True) -> str:
    """Format as percentage with optional std."""
    if show_std and std > 0:
        return f"{mean*100:.2f}±{std*100:.2f}%"
    else:
        return f"{mean*100:.2f}%"


def generate_csv(
    results: Dict[str, Dict[str, Dict[int, float]]],
    seeds: List[int],
    methods: List[str],
    tasks: List[str],
    output_path: Path
):
    """Generate CSV file with all results."""
    lines = []

    # Header
    header = ["Method", "Task"] + [f"Seed_{s}" for s in seeds] + ["Mean", "Std"]
    lines.append(",".join(header))

    # Data rows
    for method in methods:
        for task in tasks:
            row = [method, task]
            scores = results[method][task]

            for seed in seeds:
                if seed in scores:
                    row.append(f"{scores[seed]*100:.4f}")
                else:
                    row.append("")

            mean, std = compute_stats(scores)
            row.append(f"{mean*100:.4f}")
            row.append(f"{std*100:.4f}")
            lines.append(",".join(row))

    # Overall average per method (across all tasks)
    lines.append("")
    lines.append("# Overall averages (mean across tasks for each seed, then mean/std across seeds)")
    header2 = ["Method", "Metric"] + [f"Seed_{s}" for s in seeds] + ["Mean", "Std"]
    lines.append(",".join(header2))

    for method in methods:
        # For each seed, compute average across tasks
        seed_avgs = {}
        for seed in seeds:
            task_scores = []
            for task in tasks:
                if seed in results[method][task]:
                    task_scores.append(results[method][task][seed])
            if task_scores:
                seed_avgs[seed] = np.mean(task_scores)

        row = [method, "Avg(MMLU,GSM8K,BBH)"]
        for seed in seeds:
            if seed in seed_avgs:
                row.append(f"{seed_avgs[seed]*100:.4f}")
            else:
                row.append("")

        mean, std = compute_stats(seed_avgs)
        row.append(f"{mean*100:.4f}")
        row.append(f"{std*100:.4f}")
        lines.append(",".join(row))

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[OK] CSV written to: {output_path}")


def generate_markdown(
    results: Dict[str, Dict[str, Dict[int, float]]],
    seeds: List[int],
    methods: List[str],
    tasks: List[str],
    output_path: Path
):
    """Generate Markdown file with formatted tables."""
    lines = []

    lines.append("# Multi-Seed Training Robustness Results")
    lines.append("")
    lines.append(f"Training seeds: {seeds}")
    lines.append(f"Methods: {methods}")
    lines.append(f"Tasks: {tasks}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Table 1: Per-task results with mean±std
    lines.append("## Per-Task Results (Mean ± Std across seeds)")
    lines.append("")
    lines.append("| Method | MMLU | GSM8K | BBH | Avg |")
    lines.append("|--------|------|-------|-----|-----|")

    for method in methods:
        display_name = METHOD_DISPLAY_NAMES.get(method, method)
        row = [display_name]

        task_means = []
        for task in tasks:
            scores = results[method][task]
            mean, std = compute_stats(scores)
            row.append(format_pct(mean, std))
            if scores:
                task_means.append(mean)

        # Average across tasks
        if task_means:
            avg_mean = np.mean(task_means)
            # For avg std, compute std of per-seed averages
            seed_avgs = []
            for seed in seeds:
                seed_scores = []
                for task in tasks:
                    if seed in results[method][task]:
                        seed_scores.append(results[method][task][seed])
                if seed_scores:
                    seed_avgs.append(np.mean(seed_scores))
            avg_std = np.std(seed_avgs) if seed_avgs else 0
            row.append(format_pct(avg_mean, avg_std))
        else:
            row.append("-")

        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Table 2: Detailed per-seed results
    lines.append("## Detailed Per-Seed Results")
    lines.append("")

    for task in tasks:
        lines.append(f"### {task.upper()}")
        lines.append("")
        header = ["Method"] + [f"Seed {s}" for s in seeds] + ["Mean", "Std"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")

        for method in methods:
            display_name = METHOD_DISPLAY_NAMES.get(method, method)
            row = [display_name]
            scores = results[method][task]

            for seed in seeds:
                if seed in scores:
                    row.append(f"{scores[seed]*100:.2f}%")
                else:
                    row.append("-")

            mean, std = compute_stats(scores)
            row.append(f"{mean*100:.2f}%")
            row.append(f"{std*100:.2f}%")
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")

    lines.append("---")
    lines.append("")

    # Table 3: Summary statistics
    lines.append("## Summary Statistics")
    lines.append("")
    lines.append("| Method | Mean (all tasks) | Std (all tasks) | Min | Max |")
    lines.append("|--------|------------------|-----------------|-----|-----|")

    for method in methods:
        display_name = METHOD_DISPLAY_NAMES.get(method, method)

        # Collect all scores for this method
        all_scores = []
        for task in tasks:
            for seed in seeds:
                if seed in results[method][task]:
                    all_scores.append(results[method][task][seed])

        if all_scores:
            mean = np.mean(all_scores)
            std = np.std(all_scores)
            min_val = np.min(all_scores)
            max_val = np.max(all_scores)
            lines.append(f"| {display_name} | {mean*100:.2f}% | {std*100:.2f}% | {min_val*100:.2f}% | {max_val*100:.2f}% |")
        else:
            lines.append(f"| {display_name} | - | - | - | - |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Missing results warning
    lines.append("## Missing Results")
    lines.append("")
    missing = []
    for method in methods:
        for task in tasks:
            for seed in seeds:
                if seed not in results[method][task]:
                    missing.append(f"- {method}/{task}/seed_{seed}")

    if missing:
        lines.append("The following combinations are missing:")
        lines.append("")
        lines.extend(missing)
    else:
        lines.append("All combinations completed successfully.")

    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[OK] Markdown written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Summarize multi-seed robustness experiment results")
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
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help=f"Training seeds (default: {DEFAULT_SEEDS})")
    parser.add_argument("--methods", type=str, nargs="+", default=DEFAULT_METHODS,
                        choices=[
                            "random",
                            "random1",
                            "random2",
                            "random3",
                            "random_avg",
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
                        help=f"Methods (default: {DEFAULT_METHODS})")
    parser.add_argument("--tasks", type=str, nargs="+", default=DEFAULT_TASKS,
                        choices=["mmlu", "gsm8k", "bbh"],
                        help=f"Tasks (default: {DEFAULT_TASKS})")
    parser.add_argument(
        "--results_dirname",
        type=str,
        default="results",
        help="Results dir name under DATE-LM root (default: results).",
    )
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Output directory (default: <SEQDATAVAL_ROOT>/finetuning)")
    args = parser.parse_args()

    proj_root = _resolve_proj_root(args.proj_root)
    datelm_root = _resolve_datelm_root(args.datelm_root, proj_root)
    output_dir = _resolve_path(args.output_dir) if args.output_dir else (proj_root / "finetuning")
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = args.seeds
    methods = args.methods
    tasks = args.tasks
    results_dirname = args.results_dirname

    print(f"Summarizing results for:")
    print(f"  datelm_root: {datelm_root}")
    print(f"  results_dir: {datelm_root / results_dirname}")
    print(f"  output_dir:  {output_dir}")
    print(f"  Seeds: {seeds}")
    print(f"  Methods: {methods}")
    print(f"  Tasks: {tasks}")
    print()

    # Collect results
    results = collect_results(datelm_root, results_dirname, seeds, methods, tasks)

    # Optional: compute Random Avg from Random 1/2/3.
    if "random_avg" in methods:
        for task in tasks:
            for seed in seeds:
                vals = []
                for m in ("random1", "random2", "random3"):
                    if seed in results[m][task]:
                        vals.append(results[m][task][seed])
                if len(vals) == 3:
                    results["random_avg"][task][seed] = float(np.mean(vals))

    # Count collected
    total = len(seeds) * len(methods) * len(tasks)
    collected = sum(
        1 for method in methods
        for task in tasks
        for seed in seeds
        if seed in results[method][task]
    )
    print(f"Collected {collected}/{total} results")
    print()

    # Generate outputs
    csv_path = output_dir / "MULTISEED_TRAINSEED_RESULTS.csv"
    md_path = output_dir / "MULTISEED_TRAINSEED_RESULTS.md"

    generate_csv(results, seeds, methods, tasks, csv_path)
    generate_markdown(results, seeds, methods, tasks, md_path)

    # Print quick summary
    print()
    print("Quick Summary (Mean±Std):")
    header = ["Method"] + [t.upper() for t in tasks]
    print("-" * (27 + 17 * len(tasks)))
    print(f"{header[0]:<25} " + " ".join(f"{h:<15}" for h in header[1:]))
    print("-" * (27 + 17 * len(tasks)))
    for method in methods:
        display_name = METHOD_DISPLAY_NAMES.get(method, method)
        row = []
        for task in tasks:
            scores = results[method][task]
            mean, std = compute_stats(scores)
            row.append(format_pct(mean, std))
        print(f"{display_name:<25} " + " ".join(f"{v:<15}" for v in row))
    print("-" * (27 + 17 * len(tasks)))


if __name__ == "__main__":
    main()
