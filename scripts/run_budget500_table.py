#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import random
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opendataval.dataval import (
    AME,
    DVRL,
    BetaShapley,
    DataBanzhaf,
    DataOob,
    DataShapley,
    InfluenceSubsample,
    LeaveOneOut,
    RandomEvaluator,
)
from opendataval.dataval.margcontrib.sampler import MonteCarloSampler
from opendataval.experiment import ExperimentMediator
from opendataval.dataloader import mix_labels

from evaluators import BipartiteMatchingEvaluator
from utils.opendataval_compat import patch_opendataval_openml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce paper Table selection_results_500 (budget=500).")
    parser.add_argument("--config", type=str, default="config/large_config.yaml")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=["2dplanes", "nomao", "bbc-embeddings", "MiniBooNE", "digits", "election", "electricity", "fried"],
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=list(range(10, 201, 10)))
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--out_dir", type=str, default="results/selection_results_500")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--methods",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional subset of methods to run. Choices: "
            "Random, LOO, Influence, DataShap, BetaShap, Banzhaf, AME, DVRL, DataOob, Bipartite. "
            "Default: all paper baselines (+ Bipartite unless --skip_bipartite)."
        ),
    )
    parser.add_argument("--skip_bipartite", action="store_true")
    parser.add_argument("--num_models", type=int, default=1000, help="num_models for influence/banzhaf/oob")
    parser.add_argument("--ame_models", type=int, default=None, help="num_models for AME (default: ceil(num_models/4))")
    parser.add_argument("--dvrl_epochs", type=int, default=None, help="rl_epochs for DVRL (default: ceil(num_models/32))")
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Parallel workers over (dataset, seed). Default: min(8, cpu_count//2).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite `raw.csv` if it exists (otherwise resume and skip completed tasks).",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Optional directory for per-(dataset,seed) logs (default: <out_dir>/logs).",
    )
    parser.add_argument(
        "--thread_limit",
        type=int,
        default=1,
        help="Limit BLAS/OpenMP/Torch threads per worker (default: 1).",
    )
    return parser.parse_args()


def normalize_method_name(evaluator) -> str:
    name = evaluator.__class__.__name__
    mapping = {
        "RandomEvaluator": "Random",
        "LeaveOneOut": "LOO",
        "InfluenceSubsample": "Influence",
        "DataShapley": "DataShap",
        "BetaShapley": "BetaShap",
        "DataBanzhaf": "Banzhaf",
        "DataOob": "DataOob",
        "BipartiteMatchingEvaluator": "Bipartite",
        "AME": "AME",
        "DVRL": "DVRL",
    }
    return mapping.get(name, name)


def eval_topk_accuracy(exper_med: ExperimentMediator, values: np.ndarray, k: int) -> float:
    x_train, y_train, *_middle, x_test, y_test = exper_med.fetcher.datapoints
    order = np.argsort(values)[::-1]
    k = min(k, len(order))
    idx = order[:k].tolist()

    model = exper_med.pred_model.clone()
    model.fit(
        Subset(x_train, idx),
        Subset(y_train, idx),
        **(exper_med.train_kwargs or {}),
    )
    y_hat = model.predict(x_test).to("cpu")
    return float(exper_med.metric(y_test, y_hat))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _normalize_dataset_name(dataset: str) -> str:
    dataset_aliases = {
        "bbc-embedding": "bbc-embeddings",
        "bbc-embed": "bbc-embeddings",
        "bbc_embed": "bbc-embeddings",
        "miniboone": "MiniBooNE",
    }
    return dataset_aliases.get(dataset, dataset)


def _expected_methods(skip_bipartite: bool) -> set[str]:
    methods = {
        "Random",
        "LOO",
        "Influence",
        "DataShap",
        "BetaShap",
        "Banzhaf",
        "AME",
        "DVRL",
        "DataOob",
    }
    if not skip_bipartite:
        methods.add("Bipartite")
    return methods


def _normalize_methods_arg(methods: list[str] | None, *, skip_bipartite: bool) -> list[str]:
    allowed = {
        "Random",
        "LOO",
        "Influence",
        "DataShap",
        "BetaShap",
        "Banzhaf",
        "AME",
        "DVRL",
        "DataOob",
        "Bipartite",
    }
    aliases = {
        "random": "Random",
        "loo": "LOO",
        "leaveoneout": "LOO",
        "influence": "Influence",
        "influencesubsample": "Influence",
        "datashap": "DataShap",
        "datashapley": "DataShap",
        "betashap": "BetaShap",
        "betashapley": "BetaShap",
        "banzhaf": "Banzhaf",
        "databanzhaf": "Banzhaf",
        "ame": "AME",
        "dvrl": "DVRL",
        "dataoob": "DataOob",
        "bipartite": "Bipartite",
    }

    default_order = [
        "Random",
        "LOO",
        "Influence",
        "DataShap",
        "BetaShap",
        "Banzhaf",
        "AME",
        "DVRL",
        "DataOob",
        "Bipartite",
    ]

    if methods is None:
        out = [m for m in default_order if m in _expected_methods(skip_bipartite)]
        return out

    requested: set[str] = set()
    for m in methods:
        key = aliases.get(str(m).strip().lower(), str(m).strip())
        if key not in allowed:
            raise ValueError(f"Unknown method: {m}. Allowed: {sorted(allowed)}")
        requested.add(key)

    if skip_bipartite and "Bipartite" in requested:
        raise ValueError("Got --skip_bipartite but --methods includes Bipartite; remove one of them.")

    return [m for m in default_order if m in requested]


def _run_one_dataset_seed(
    dataset: str,
    seed: int,
    *,
    train_count: int,
    valid_count: int,
    test_count: int,
    device: str,
    model_name: str,
    budget: int,
    num_models: int,
    ame_models: int,
    dvrl_epochs: int,
    methods_to_run: list[str],
    log_dir: str | None,
    thread_limit: int,
) -> list[dict[str, object]]:
    patch_opendataval_openml()
    _seed_everything(seed)

    try:
        import torch

        torch.set_num_threads(max(1, int(thread_limit)))
        torch.set_num_interop_threads(max(1, int(thread_limit)))
    except Exception:
        pass

    dataset_name = _normalize_dataset_name(dataset)
    log_path = None
    if log_dir:
        log_path = Path(log_dir) / f"{dataset_name}_seed_{seed}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

    from threadpoolctl import threadpool_limits

    def _materialize_dataset(ds):
        import torch
        from torch.utils.data import DataLoader, Dataset

        if isinstance(ds, torch.Tensor):
            return ds
        if not isinstance(ds, Dataset):
            return torch.tensor(ds, dtype=torch.float)

        chunks = []
        for batch in DataLoader(ds, batch_size=512, shuffle=False):
            if isinstance(batch, torch.Tensor):
                chunks.append(batch)
            elif isinstance(batch, (tuple, list)) and batch and isinstance(batch[0], torch.Tensor):
                chunks.append(batch[0])
            else:
                raise TypeError(f"Unexpected batch type from DataLoader: {type(batch)}")
        return torch.cat(chunks, dim=0)

    def _run() -> list[dict[str, object]]:
        with threadpool_limits(limits=max(1, int(thread_limit))):
            train_n = int(train_count)
            valid_n = int(valid_count)
            test_n = int(test_count)

            def _fit_split_counts(num_points: int) -> tuple[int, int, int]:
                min_train = int(budget)
                if num_points < min_train + 2:
                    raise ValueError(f"Dataset too small for budget={budget}: num_points={num_points}")

                t = train_n
                v = valid_n
                te = test_n
                if t + v + te > num_points:
                    te = max(1, num_points - t - v)
                if t + v + te > num_points:
                    v = max(1, num_points - t - 1)
                    te = max(1, num_points - t - v)
                if t + v + te > num_points:
                    t = max(min_train, num_points - v - 1)
                    te = max(1, num_points - t - v)
                if t + v + te > num_points:
                    raise ValueError(
                        f"Cannot fit splits into dataset: num_points={num_points} train={t} valid={v} test={te}"
                    )
                if t < min_train:
                    raise ValueError(
                        f"Train split smaller than budget={budget}: num_points={num_points} train={t} valid={v} test={te}"
                    )
                return int(t), int(v), int(te)

            def _make_exper_med(t: int, v: int, te: int) -> ExperimentMediator:
                return ExperimentMediator.model_factory_setup(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    train_count=t,
                    valid_count=v,
                    test_count=te,
                    add_noise=mix_labels,
                    noise_kwargs={"noise_rate": 0.0},
                    metric_name="accuracy",
                    device=device,
                    random_state=seed,
                )

            try:
                exper_med = _make_exper_med(train_n, valid_n, test_n)
            except Exception as e:
                msg = str(e)
                mm = re.search(r"self\.num_points=(\d+)", msg)
                if mm is None:
                    raise
                n = int(mm.group(1))
                train_n, valid_n, test_n = _fit_split_counts(n)
                exper_med = _make_exper_med(train_n, valid_n, test_n)

            # Some OpenDataVal datasets (e.g., NLP embeddings) return torch Datasets
            # backed by on-disk storage; materialize the split once to avoid 1000x I/O
            # during model retraining loops (LOO/DataOob/etc).
            x_train, *_rest = exper_med.fetcher.datapoints
            if not hasattr(x_train, "shape"):
                x_train, y_train, x_valid, y_valid, x_test, y_test = exper_med.fetcher.datapoints
                exper_med.fetcher.x_train = _materialize_dataset(x_train)
                exper_med.fetcher.x_valid = _materialize_dataset(x_valid)
                exper_med.fetcher.x_test = _materialize_dataset(x_test)

            mc_epochs = int(np.ceil(num_models / train_n))
            cache_name = f"cached_{dataset_name}_seed_{seed}_budget{budget}_n{train_n}"
            need_mc = any(m in {"DataShap", "BetaShap"} for m in methods_to_run)
            mc_sampler = None
            if need_mc:
                mc_sampler = MonteCarloSampler(
                    mc_epochs=mc_epochs,
                    min_cardinality=1,
                    cache_name=cache_name,
                    random_state=seed,
                )

            evaluators = []
            for m in methods_to_run:
                if m == "Random":
                    evaluators.append(RandomEvaluator())
                elif m == "LOO":
                    evaluators.append(LeaveOneOut())
                elif m == "Influence":
                    evaluators.append(InfluenceSubsample(num_models=num_models, random_state=seed))
                elif m == "DataShap":
                    if mc_sampler is None:
                        raise ValueError("Internal error: DataShap requires MonteCarloSampler.")
                    evaluators.append(DataShapley(sampler=mc_sampler, cache_name=cache_name))
                elif m == "BetaShap":
                    if mc_sampler is None:
                        raise ValueError("Internal error: BetaShap requires MonteCarloSampler.")
                    evaluators.append(BetaShapley(sampler=mc_sampler, cache_name=cache_name))
                elif m == "Banzhaf":
                    evaluators.append(DataBanzhaf(num_models=num_models, random_state=seed))
                elif m == "AME":
                    evaluators.append(AME(num_models=ame_models))
                elif m == "DVRL":
                    evaluators.append(DVRL(rl_epochs=dvrl_epochs, random_state=seed))
                elif m == "DataOob":
                    evaluators.append(DataOob(num_models=num_models, random_state=seed))
                elif m == "Bipartite":
                    evaluators.append(BipartiteMatchingEvaluator(n_samples=num_models, random_state=seed))
                else:
                    raise ValueError(f"Unknown method: {m}")

            exper_med = exper_med.compute_data_values(evaluators)

            rows: list[dict[str, object]] = []
            for evaluator in exper_med.data_evaluators:
                method = normalize_method_name(evaluator)
                acc = eval_topk_accuracy(exper_med, evaluator.data_values, budget)
                rows.append(
                    {
                        "dataset": dataset_name,
                        "seed": int(seed),
                        "method": method,
                        "budget": int(budget),
                        "train_count": int(train_n),
                        "valid_count": int(valid_n),
                        "test_count": int(test_n),
                        "accuracy": float(acc),
                    }
                )
            return rows

    if log_path is None:
        return _run()
    with open(log_path, "w", encoding="utf-8") as f, redirect_stdout(f), redirect_stderr(f):
        return _run()


def _load_existing(
    raw_path: Path,
    *,
    budget: int,
) -> pd.DataFrame:
    if not raw_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(raw_path)
    need = {"dataset", "seed", "method", "budget", "train_count", "valid_count", "test_count", "accuracy"}
    if not need.issubset(set(df.columns)):
        return pd.DataFrame()
    df = df[df["budget"] == budget].copy()
    return df


def _completed_tasks(df: pd.DataFrame, *, expected: set[str]) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    if df.empty:
        return done
    for (ds, seed), sub in df.groupby(["dataset", "seed"]):
        if set(sub["method"].unique()) >= expected:
            done.add((str(ds), int(seed)))
    return done


def _write_outputs(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    summary = (
        df.groupby(["dataset", "method"])["accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["dataset", "mean"], ascending=[True, False])
    )
    (out_dir / "summary.csv").write_text(summary.to_csv(index=False), encoding="utf-8")


def main() -> None:
    args = parse_args()

    # Limit BLAS/OpenMP threads globally (workers also enforce via threadpoolctl).
    thread_limit = max(1, int(args.thread_limit))
    os.environ["OMP_NUM_THREADS"] = str(thread_limit)
    os.environ["MKL_NUM_THREADS"] = str(thread_limit)
    os.environ["OPENBLAS_NUM_THREADS"] = str(thread_limit)
    os.environ["NUMEXPR_NUM_THREADS"] = str(thread_limit)

    patch_opendataval_openml()

    root = Path(__file__).resolve().parents[1]
    config_path = (root / args.config).resolve()
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    train_count = int(config["experiment"]["train_count"])
    valid_count = int(config["experiment"]["valid_count"])
    test_count = int(config["experiment"]["test_count"])
    device = args.device or config["experiment"]["device"]
    model_name = args.model or config["model"]["name"]

    budget = int(args.budget)
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir).resolve() if args.log_dir else (out_dir / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    ame_models = args.ame_models if args.ame_models is not None else int(np.ceil(args.num_models / 4))
    dvrl_epochs = args.dvrl_epochs if args.dvrl_epochs is not None else int(np.ceil(args.num_models / 32))

    methods_requested = _normalize_methods_arg(args.methods, skip_bipartite=bool(args.skip_bipartite))
    expected = set(methods_requested)
    raw_path = out_dir / "raw.csv"
    existing = (
        pd.DataFrame()
        if args.overwrite
        else _load_existing(
            raw_path,
            budget=budget,
        )
    )

    existing_methods: dict[tuple[str, int], set[str]] = {}
    if not existing.empty:
        for (ds, seed), sub in existing.groupby(["dataset", "seed"]):
            existing_methods[(str(ds), int(seed))] = set(sub["method"].astype(str).unique())

    tasks: list[tuple[str, int, list[str]]] = []
    for dataset in args.datasets:
        ds_norm = _normalize_dataset_name(dataset)
        for seed in args.seeds:
            have = existing_methods.get((ds_norm, int(seed)), set())
            missing = [m for m in methods_requested if m not in have]
            if not missing:
                continue
            tasks.append((dataset, int(seed), missing))

    if not tasks:
        print("Nothing to do: all (dataset, seed) pairs already completed.")
        if not existing.empty:
            _write_outputs(existing, out_dir)
        return

    cpu = os.cpu_count() or 1
    max_workers = args.max_workers if args.max_workers is not None else max(1, min(8, cpu // 2))
    max_workers = max(1, int(max_workers))
    print(
        f"Running budget={budget} with train/valid/test={train_count}/{valid_count}/{test_count} "
        f"num_models={args.num_models} max_workers={max_workers} tasks={len(tasks)} "
        f"methods={methods_requested}"
    )

    rows: list[dict[str, object]] = []
    if not existing.empty:
        rows.extend(existing.to_dict(orient="records"))

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        fut_to_task = {
            ex.submit(
                _run_one_dataset_seed,
                dataset,
                seed,
                train_count=train_count,
                valid_count=valid_count,
                test_count=test_count,
                device=device,
                model_name=model_name,
                budget=budget,
                num_models=args.num_models,
                ame_models=ame_models,
                dvrl_epochs=dvrl_epochs,
                methods_to_run=methods_to_run,
                log_dir=str(log_dir),
                thread_limit=thread_limit,
            ): (dataset, seed, methods_to_run)
            for dataset, seed, methods_to_run in tasks
        }
        completed = 0
        for fut in as_completed(fut_to_task):
            dataset, seed, methods_to_run = fut_to_task[fut]
            try:
                out_rows = fut.result()
                rows.extend(out_rows)
                completed += 1
                meth = ",".join(methods_to_run)
                print(
                    f"[done {completed}/{len(tasks)}] dataset={_normalize_dataset_name(dataset)} seed={seed} "
                    f"methods={meth}"
                )
                df = pd.DataFrame(rows).drop_duplicates(
                    subset=["dataset", "seed", "method", "budget", "train_count", "valid_count", "test_count"],
                    keep="last",
                )
                _write_outputs(df, out_dir)
            except Exception as e:
                print(f"[failed] dataset={dataset} seed={seed} methods={methods_to_run}: {e}")

    df = pd.DataFrame(rows).drop_duplicates(
        subset=["dataset", "seed", "method", "budget", "train_count", "valid_count", "test_count"],
        keep="last",
    )
    _write_outputs(df, out_dir)
    print("Wrote:", out_dir / "raw.csv")
    print("Wrote:", out_dir / "summary.csv")


if __name__ == "__main__":
    main()
