#!/usr/bin/env python
"""Validate DATE-LM-style run artifacts for a given (run_tag, task, method).

This script is meant to catch common “silent” issues before you burn hours on training:
  - metrics length != pool size
  - indices out of range / duplicated
  - selected JSONL not matching (pool, indices)
  - top-k mismatch (for top-k methods) or Gumbel mismatch (for RDS+)
  - missing checkpoints / eval outputs

Designed to work with two tags:
  - --data_tag: where pool/scores/selected_data live
  - --run_tag : where checkpoints/results live (often same as data_tag)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import numpy as np


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _count_jsonl_lines(path: Path) -> int:
    with path.open("r") as f:
        return sum(1 for _ in f)


def _load_npy_1d(path: Path) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def _set_stats(name: str, values: Iterable[int], limit: int = 5) -> str:
    vals = list(values)
    if not vals:
        return f"{name}=∅"
    head = ", ".join(str(x) for x in vals[:limit])
    tail = "" if len(vals) <= limit else f", …(+{len(vals) - limit})"
    return f"{name}=[{head}{tail}]"


def _try_load_datasets_jsonl(path: Path):
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        return None
    return load_dataset("json", data_files=str(path), split="train")


def _infer_selection_kind(method: str) -> str:
    if method == "random":
        return "random"
    if method.startswith("rds_plus"):
        return "gumbel"
    return "topk"


def _infer_base_method_for_rds(method: str) -> str:
    if method == "rds_plus_v2":
        return "repsim_v2"
    if method == "rds_plus":
        return "repsim"
    raise ValueError(f"Unrecognized RDS+ variant: {method}")


def _gumbel_topk_indices(
    base_scores: np.ndarray,
    k: int,
    gumbel_temp: float,
    seed: int,
) -> np.ndarray:
    scores = np.asarray(base_scores)
    mean = float(scores.mean())
    std = float(scores.std())
    if std < 1e-12:
        normalized = scores - mean
    else:
        normalized = (scores - mean) / std
    scaled = normalized / float(gumbel_temp)
    rng = np.random.default_rng(int(seed))
    perturbed = scaled + rng.gumbel(size=scores.shape[0])
    return np.argpartition(perturbed, -k)[-k:].astype(np.int64)


def _default_proj_root() -> Path:
    # .../finetuning/scripts/datelm_paper/verify_run_artifacts.py -> repo root is 3 parents up
    return Path(__file__).resolve().parents[3]

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


def _resolve_datelm_root(arg: Path | None) -> Path:
    if arg is not None:
        return arg.expanduser().resolve()
    env = os.environ.get("DATELM_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    inferred = _infer_datelm_root(_default_proj_root())
    if inferred is not None:
        return inferred.expanduser().resolve()
    raise FileNotFoundError(
        "DATE-LM root not found. Set `--datelm_root /path/to/DATE-LM` "
        "or export `DATELM_ROOT=/path/to/DATE-LM`."
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--datelm_root",
        type=Path,
        default=None,
        help="Path to DATE-LM root (repo root or DATE-LM-main), or set DATELM_ROOT env var.",
    )
    p.add_argument("--run_tag", type=str, required=True, help="Tag for checkpoints/results")
    p.add_argument(
        "--data_tag",
        type=str,
        default=None,
        help="Tag for pool/scores/selected_data (defaults to --run_tag)",
    )
    p.add_argument("--task", type=str, required=True, choices=["mmlu", "gsm8k", "bbh"])
    p.add_argument("--method", type=str, required=True)
    p.add_argument("--k", type=int, default=10000)
    p.add_argument("--gumbel_temp", type=float, default=0.5)
    p.add_argument("--gumbel_seed", type=int, default=42)
    p.add_argument(
        "--check_training",
        action="store_true",
        help="Also verify checkpoints/merged/results artifacts under --run_tag.",
    )
    args = p.parse_args()

    datelm_root = _resolve_datelm_root(args.datelm_root)
    if not datelm_root.exists():
        raise FileNotFoundError(str(datelm_root))

    run_tag = str(args.run_tag)
    data_tag = str(args.data_tag) if args.data_tag else run_tag
    task = str(args.task)
    method = str(args.method)
    k = int(args.k)

    errors: list[str] = []
    warnings: list[str] = []

    pool_jsonl = datelm_root / "data" / "training_data" / f"{data_tag}_tulu3_200k_train.jsonl"
    scores_dir = datelm_root / "scores" / data_tag / task
    selected_dir = datelm_root / "selected_data" / data_tag / task

    indices_npy = selected_dir / f"{method}_indices.npy"
    selected_jsonl = selected_dir / f"{method}_10k.jsonl"
    metrics_npy = scores_dir / f"{method}_metrics.npy"

    print("=== Verify run artifacts ===")
    print(f"datelm_root: {datelm_root}")
    print(f"data_tag:    {data_tag}")
    print(f"run_tag:     {run_tag}")
    print(f"task/method: {task} / {method}")
    print(f"k:           {k}")
    print("")

    if not pool_jsonl.exists():
        errors.append(f"Missing pool JSONL: {pool_jsonl}")
        n_pool = None
    else:
        n_pool = _count_jsonl_lines(pool_jsonl)
        print(f"[OK] pool_jsonl: {pool_jsonl} (lines={n_pool})")

    if not selected_jsonl.exists():
        errors.append(f"Missing selected JSONL: {selected_jsonl}")
        n_sel = None
    else:
        n_sel = _count_jsonl_lines(selected_jsonl)
        print(f"[OK] selected_jsonl: {selected_jsonl} (lines={n_sel})")
        if n_sel != k:
            errors.append(f"selected_jsonl lines != k: lines={n_sel} k={k}")

    if not indices_npy.exists():
        errors.append(f"Missing indices npy: {indices_npy}")
        indices = None
    else:
        indices = _load_npy_1d(indices_npy).astype(np.int64)
        uniq = int(np.unique(indices).shape[0])
        print(f"[OK] indices_npy: {indices_npy} (len={indices.shape[0]}, unique={uniq})")
        if indices.shape[0] != k:
            errors.append(f"indices length != k: len={indices.shape[0]} k={k}")
        if uniq != indices.shape[0]:
            errors.append(f"indices contain duplicates: len={indices.shape[0]} unique={uniq}")
        if n_pool is not None and indices.shape[0] > 0:
            mn, mx = int(indices.min()), int(indices.max())
            if mn < 0 or mx >= n_pool:
                errors.append(f"indices out of range: min={mn} max={mx} pool_lines={n_pool}")

    selection_kind = _infer_selection_kind(method)
    if selection_kind == "topk":
        if not metrics_npy.exists():
            errors.append(f"Missing metrics npy (top-k method): {metrics_npy}")
            metrics = None
        else:
            metrics = _load_npy_1d(metrics_npy)
            print(f"[OK] metrics_npy: {metrics_npy} (len={metrics.shape[0]}, dtype={metrics.dtype})")
            if n_pool is not None and metrics.shape[0] != n_pool:
                errors.append(f"metrics length != pool lines: len={metrics.shape[0]} pool_lines={n_pool}")
            if indices is not None:
                topk = np.argpartition(metrics, -k)[-k:].astype(np.int64)
                if set(topk.tolist()) != set(indices.tolist()):
                    missing = set(indices.tolist()) - set(topk.tolist())
                    extra = set(topk.tolist()) - set(indices.tolist())
                    errors.append(
                        "top-k mismatch between metrics and indices: "
                        + f"{_set_stats('missing', sorted(missing))} "
                        + f"{_set_stats('extra', sorted(extra))}"
                    )

        selected_seq_json = scores_dir / f"{method}_selected_indices.json"
        if metrics_npy.exists() and selected_seq_json.exists():
            seq = _read_json(selected_seq_json)
            if not isinstance(seq, list):
                errors.append(f"Unexpected JSON format (expected list): {selected_seq_json}")
            else:
                seq = [int(x) for x in seq]
                if len(seq) != k:
                    warnings.append(f"selected_indices.json length != k: len={len(seq)} k={k} ({selected_seq_json})")
                if indices is not None and set(seq) != set(indices.tolist()):
                    warnings.append(
                        "bipcov saved sequence set != selected indices set "
                        f"({selected_seq_json} vs {indices_npy})"
                    )

    elif selection_kind == "gumbel":
        base_method = _infer_base_method_for_rds(method)
        base_metrics_npy = scores_dir / f"{base_method}_metrics.npy"
        if not base_metrics_npy.exists():
            errors.append(f"Missing base metrics for {method}: {base_metrics_npy}")
        else:
            base = _load_npy_1d(base_metrics_npy)
            if n_pool is not None and base.shape[0] != n_pool:
                errors.append(f"base metrics length != pool lines: len={base.shape[0]} pool_lines={n_pool}")
            if indices is not None:
                topk = _gumbel_topk_indices(base, k=k, gumbel_temp=float(args.gumbel_temp), seed=int(args.gumbel_seed))
                if set(topk.tolist()) != set(indices.tolist()):
                    missing = set(indices.tolist()) - set(topk.tolist())
                    extra = set(topk.tolist()) - set(indices.tolist())
                    errors.append(
                        f"gumbel-topk mismatch (temp={args.gumbel_temp}, seed={args.gumbel_seed}): "
                        + f"{_set_stats('missing', sorted(missing))} "
                        + f"{_set_stats('extra', sorted(extra))}"
                    )
            print(f"[OK] gumbel base metrics: {base_metrics_npy} (len={base.shape[0]})")

    else:
        print("[OK] random selection (no metrics check)")

    if pool_jsonl.exists() and selected_jsonl.exists() and indices is not None:
        pool_ds = _try_load_datasets_jsonl(pool_jsonl)
        sel_ds = _try_load_datasets_jsonl(selected_jsonl)
        if pool_ds is None or sel_ds is None:
            warnings.append("datasets not available; skipped JSONL↔indices content check")
        else:
            for pos in {0, min(123, k - 1), k - 1}:
                if pos < 0 or pos >= k:
                    continue
                pool_idx = int(indices[pos])
                try:
                    pool_row = pool_ds[pool_idx]
                    sel_row = sel_ds[pos]
                except Exception as e:
                    errors.append(f"Failed to index datasets for content check: pos={pos} idx={pool_idx}: {e}")
                    continue
                if pool_row.get("id") != sel_row.get("id"):
                    errors.append(
                        "selected JSONL content mismatch vs pool+indices "
                        f"(pos={pos} pool_idx={pool_idx} pool_id={pool_row.get('id')} sel_id={sel_row.get('id')})"
                    )
            print("[OK] selected_jsonl matches (pool, indices) on sampled positions via `id`")

    if args.check_training:
        adapter_dir = datelm_root / "checkpoints" / f"{run_tag}_{method}_{task}_lora"
        merged_dir = datelm_root / "checkpoints" / f"{run_tag}_{method}_{task}_merged"
        results_dir = datelm_root / "results" / f"{run_tag}_{method}_{task}_official"

        adapter_cfg = adapter_dir / "adapter_config.json"
        train_cfg = adapter_dir / "training_config.json"
        merged_cfg = merged_dir / "config.json"
        metrics_json = results_dir / "metrics.json"

        if adapter_cfg.exists():
            cfg = _read_json(adapter_cfg)
            print(f"[OK] adapter_config.json: {adapter_cfg}")
            for key in ["r", "lora_alpha", "lora_dropout", "target_modules"][:]:
                if key in cfg:
                    print(f"  - adapter.{key}: {cfg[key]}")
        else:
            errors.append(f"Missing adapter_config.json: {adapter_cfg}")

        if train_cfg.exists():
            cfg = _read_json(train_cfg)
            print(f"[OK] training_config.json: {train_cfg}")
            for key in ["seed", "lora_alpha", "lora_r", "num_epochs", "effective_batch_size"]:
                if key in cfg:
                    print(f"  - train.{key}: {cfg[key]}")
            if isinstance(cfg.get("dataloader"), dict) and "shuffle" in cfg["dataloader"]:
                print(f"  - train.dataloader.shuffle: {cfg['dataloader']['shuffle']}")
        else:
            warnings.append(f"Missing training_config.json (ok if not trained by our script): {train_cfg}")

        if merged_cfg.exists():
            print(f"[OK] merged config: {merged_cfg}")
        else:
            errors.append(f"Missing merged model config.json: {merged_cfg}")

        if metrics_json.exists():
            mj = _read_json(metrics_json)
            print(f"[OK] eval metrics: {metrics_json} (keys={sorted(mj.keys())})")
        else:
            errors.append(f"Missing eval metrics.json: {metrics_json}")

    print("")
    if warnings:
        print("=== Warnings ===")
        for w in warnings:
            print(f"- {w}")
        print("")

    if errors:
        print("=== FAIL ===")
        for e in errors:
            print(f"- {e}")
        return 1

    print("=== PASS ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
