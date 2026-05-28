#!/usr/bin/env python
"""Reproduce the paper Budget-500 table under the curve-mean protocol.

This runner implements two protocol details that are easy to miss:

1. **Per-dataset split sizes**: train=500/valid=50/test=500 for standard
   datasets and train=500/valid=100/test=1000 for digits/bbc-embeddings.
   following the paper Section 7.3 / Appendix RQ3 baseline protocol,
   re-scaled to budget=500.
2. **Curve-mean accuracy** rather than single top-k accuracy. The reported
   accuracy is the mean over k=1..train_count of "model accuracy when
   retrained on the top-k most-valuable points by the method's ranking".
   This matches the paper selection-curve aggregation protocol.

Outputs:
  <out_dir>/raw.csv:       per (dataset, seed, method) row with curve-mean,
                           top-k accuracy, and curve length.
  <out_dir>/summary_curve_mean.csv:
                           per (dataset, method) mean/std/count over seeds.
  <out_dir>/curves/*.csv:  optional per-method full curves for inspection.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

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


METHOD_NAME_MAP = {
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


DEFAULT_LARGE = {"digits", "bbc-embeddings"}


def per_dataset_split(dataset: str, default_train: int = 500) -> tuple[int, int, int]:
    """Return (train, valid, test) using paper's per-dataset rule."""
    if dataset in DEFAULT_LARGE:
        return default_train, 100, 1000
    return default_train, 50, 500


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="*",
                   default=["2dplanes", "nomao", "MiniBooNE", "digits", "election",
                            "electricity", "fried", "bbc-embeddings"])
    p.add_argument("--seeds", type=int, nargs="*",
                   default=list(range(10, 201, 10)))
    p.add_argument("--train_count", type=int, default=500,
                   help="Train count (paper Table 3: 500).")
    p.add_argument("--valid_count", type=int, default=None,
                   help="Override per-dataset valid_count.")
    p.add_argument("--test_count", type=int, default=None,
                   help="Override per-dataset test_count.")
    p.add_argument("--num_models", type=int, default=1000,
                   help="Base num_models for InfluenceSubsample/DataBanzhaf/DataOob/Bipartite.")
    p.add_argument("--dvrl_rl_epochs", type=int, default=1000,
                   help="DVRL rl_epochs (default 1000 = OpenDataVal default; gives "
                        "1003 pred_model.fit() calls = 2 baseline + 1000 RL-loop + 1 "
                        "final, aligned with other methods' 1000-retraining budget).")
    p.add_argument("--ame_models", type=int, default=None,
                   help="Override AME num_models (default ceil(num_models/4)=250).")
    p.add_argument("--methods", nargs="*", default=None)
    p.add_argument("--curve_step", type=int, default=1,
                   help="Stride along the curve (1 = every k).")
    p.add_argument("--curve_start", type=int, default=1,
                   help="Smallest k to evaluate on the curve.")
    p.add_argument("--model", type=str, default="sklogreg")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--max_workers", type=int, default=1)
    p.add_argument("--thread_limit", type=int, default=1)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--save_curves", action="store_true",
                   help="Save per (dataset, seed, method) full curves.")
    return p.parse_args()


def seed_everything(seed: int) -> None:
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


def build_evaluators(methods, num_models, train_count, seed, mc_cache,
                     dvrl_rl_epochs_override=None, ame_models_override=None):
    mc_epochs = max(1, int(math.ceil(num_models / max(1, train_count))))
    ame_models = ame_models_override if ame_models_override is not None else max(1, int(math.ceil(num_models / 4)))
    # DVRL fair default = 1000 (= OpenDataVal default; gives ~1000 pred_model.fit
    # calls, aligned with other methods' 1000-retraining budget).
    dvrl_epochs = dvrl_rl_epochs_override if dvrl_rl_epochs_override is not None else 1000

    need_mc = any(m in {"DataShap", "BetaShap"} for m in methods)
    mc = MonteCarloSampler(
        mc_epochs=mc_epochs,
        min_cardinality=1,
        cache_name=mc_cache,
        random_state=seed,
    ) if need_mc else None

    evals = []
    for m in methods:
        if m == "Random":
            evals.append(RandomEvaluator())
        elif m == "LOO":
            evals.append(LeaveOneOut())
        elif m == "Influence":
            evals.append(InfluenceSubsample(num_models=num_models, random_state=seed))
        elif m == "DataShap":
            evals.append(DataShapley(sampler=mc, cache_name=mc_cache))
        elif m == "BetaShap":
            evals.append(BetaShapley(sampler=mc, cache_name=mc_cache))
        elif m == "Banzhaf":
            evals.append(DataBanzhaf(num_models=num_models, random_state=seed))
        elif m == "AME":
            evals.append(AME(num_models=ame_models))
        elif m == "DVRL":
            evals.append(DVRL(rl_epochs=dvrl_epochs, random_state=seed))
        elif m == "DataOob":
            evals.append(DataOob(num_models=num_models, random_state=seed))
        elif m == "Bipartite":
            evals.append(BipartiteMatchingEvaluator(n_samples=num_models, random_state=seed))
        else:
            raise ValueError(m)
    return evals, dict(mc_epochs=mc_epochs, ame_models=ame_models, dvrl_epochs=dvrl_epochs)


def compute_curve(exper_med, values, train_kwargs, curve_ks):
    x_train, y_train, *_, x_test, y_test = exper_med.fetcher.datapoints
    order = np.argsort(values)[::-1]  # most valuable first
    n = len(order)
    accs = []
    for k in curve_ks:
        k = min(max(1, int(k)), n)
        idx = order[:k].tolist()
        m = exper_med.pred_model.clone()
        m.fit(Subset(x_train, idx), Subset(y_train, idx), **(train_kwargs or {}))
        y_hat = m.predict(x_test).to("cpu")
        accs.append(float(exper_med.metric(y_test, y_hat)))
    return curve_ks, accs


def run_one(dataset, seed, train_count, valid_count, test_count, num_models,
            methods, curve_start, curve_step, model_name, device, thread_limit,
            out_dir, save_curves, dvrl_rl_epochs_override=None,
            ame_models_override=None):
    from threadpoolctl import threadpool_limits
    os.environ["OMP_NUM_THREADS"] = str(thread_limit)
    os.environ["MKL_NUM_THREADS"] = str(thread_limit)
    os.environ["OPENBLAS_NUM_THREADS"] = str(thread_limit)
    os.environ["NUMEXPR_NUM_THREADS"] = str(thread_limit)

    patch_opendataval_openml()
    seed_everything(seed)
    try:
        import torch
        torch.set_num_threads(max(1, int(thread_limit)))
        torch.set_num_interop_threads(max(1, int(thread_limit)))
    except Exception:
        pass

    with threadpool_limits(limits=max(1, int(thread_limit))):
        ds_norm = "MiniBooNE" if dataset.lower() == "miniboone" else dataset
        exper_med = ExperimentMediator.model_factory_setup(
            dataset_name=ds_norm,
            model_name=model_name,
            train_count=train_count,
            valid_count=valid_count,
            test_count=test_count,
            add_noise=mix_labels,
            noise_kwargs={"noise_rate": 0.0},
            metric_name="accuracy",
            device=device,
            random_state=seed,
        )
        # materialize on-disk torch Datasets (e.g. embeddings)
        x_train, *_rest = exper_med.fetcher.datapoints
        if not hasattr(x_train, "shape"):
            from torch.utils.data import DataLoader
            def _mat(ds):
                chunks = []
                for batch in DataLoader(ds, batch_size=512, shuffle=False):
                    chunks.append(batch[0] if isinstance(batch, (tuple, list)) else batch)
                import torch
                return torch.cat(chunks, dim=0)
            x_train, y_train, x_valid, y_valid, x_test, y_test = exper_med.fetcher.datapoints
            exper_med.fetcher.x_train = _mat(x_train)
            exper_med.fetcher.x_valid = _mat(x_valid)
            exper_med.fetcher.x_test = _mat(x_test)

        train_n = len(exper_med.fetcher.x_train)
        mc_cache = f"cached_{ds_norm}_seed_{seed}_train{train_n}_curvemean"
        evaluators, hparams = build_evaluators(
            methods, num_models, train_n, seed, mc_cache,
            dvrl_rl_epochs_override=dvrl_rl_epochs_override,
            ame_models_override=ame_models_override,
        )

        exper_med = exper_med.compute_data_values(evaluators)

        curve_ks = list(range(max(1, curve_start), train_n + 1, max(1, curve_step)))
        if curve_ks[-1] != train_n:
            curve_ks.append(train_n)

        rows = []
        for ev in exper_med.data_evaluators:
            method = METHOD_NAME_MAP.get(ev.__class__.__name__, ev.__class__.__name__)
            ks, accs = compute_curve(exper_med, ev.data_values, exper_med.train_kwargs, curve_ks)
            curve_mean = float(np.mean(accs))
            top_full = float(accs[-1])
            half_k = train_n // 2
            top_half = float(np.mean([a for k, a in zip(ks, accs) if k <= half_k])) if half_k > 0 else float("nan")
            rows.append(dict(
                dataset=ds_norm, seed=int(seed), method=method,
                train_count=int(train_n), valid_count=int(valid_count),
                test_count=int(test_count), num_models=int(num_models),
                mc_epochs=int(hparams["mc_epochs"]),
                ame_models=int(hparams["ame_models"]),
                dvrl_epochs=int(hparams["dvrl_epochs"]),
                curve_mean=curve_mean,
                accuracy_full_train=top_full,
                accuracy_first_half_mean=top_half,
                curve_n_points=int(len(accs)),
            ))
            if save_curves:
                cdf = pd.DataFrame({"k": ks, "accuracy": accs})
                cdir = Path(out_dir) / "curves"
                cdir.mkdir(parents=True, exist_ok=True)
                cdf.to_csv(cdir / f"{ds_norm}_seed{seed}_{method}.csv", index=False)
        return rows


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    methods = args.methods or [
        "Random", "LOO", "Influence", "DataShap", "BetaShap", "Banzhaf",
        "AME", "DVRL", "DataOob", "Bipartite",
    ]

    tasks = []
    for ds in args.datasets:
        t, v, te = per_dataset_split(ds, default_train=args.train_count)
        if args.valid_count is not None:
            v = int(args.valid_count)
        if args.test_count is not None:
            te = int(args.test_count)
        for seed in args.seeds:
            tasks.append((ds, int(seed), t, v, te))

    print(f"Tasks: {len(tasks)}  methods={methods}")
    all_rows = []
    if args.max_workers <= 1:
        for ds, seed, t, v, te in tasks:
            print(f"[run] {ds} seed={seed} split={t}/{v}/{te}")
            try:
                rows = run_one(ds, seed, t, v, te, args.num_models, methods,
                               args.curve_start, args.curve_step, args.model,
                               args.device, args.thread_limit, str(out), args.save_curves,
                               dvrl_rl_epochs_override=args.dvrl_rl_epochs,
                               ame_models_override=args.ame_models)
                all_rows.extend(rows)
            except Exception as e:
                print(f"  [ERROR] {ds} seed={seed}: {e}")
    else:
        with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
            futures = {
                ex.submit(run_one, ds, seed, t, v, te, args.num_models, methods,
                          args.curve_start, args.curve_step, args.model,
                          args.device, args.thread_limit, str(out), args.save_curves,
                          args.dvrl_rl_epochs, args.ame_models): (ds, seed)
                for ds, seed, t, v, te in tasks
            }
            for fut in as_completed(futures):
                ds, seed = futures[fut]
                try:
                    rows = fut.result()
                    all_rows.extend(rows)
                    print(f"[done] {ds} seed={seed}")
                except Exception as e:
                    print(f"[ERROR] {ds} seed={seed}: {e}")

    if not all_rows:
        print("No rows produced.")
        return

    raw = pd.DataFrame(all_rows)
    raw_path = out / "raw.csv"
    if raw_path.exists():
        prev = pd.read_csv(raw_path)
        raw = pd.concat([prev, raw], ignore_index=True)
        raw = raw.drop_duplicates(subset=["dataset", "seed", "method"], keep="last")
    raw.to_csv(raw_path, index=False)

    for metric in ["curve_mean", "accuracy_full_train", "accuracy_first_half_mean"]:
        summ = raw.groupby(["dataset", "method"])[metric].agg(["mean", "std", "count"]).reset_index()
        summ = summ.sort_values(["dataset", "mean"], ascending=[True, False])
        summ.to_csv(out / f"summary_{metric}.csv", index=False)
    print(f"Wrote {raw_path} ({len(raw)} rows)")
    print(f"Wrote summary_curve_mean.csv etc.")


if __name__ == "__main__":
    main()
