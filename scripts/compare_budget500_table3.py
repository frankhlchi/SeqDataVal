#!/usr/bin/env python
"""Compare a Budget-500 curve-mean summary against paper Table 3."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument(
        "--summary",
        default=str(root / "results" / "selection_results_500_curve_mean" / "summary_curve_mean.csv"),
    )
    p.add_argument(
        "--paper",
        default=str(root / "scripts" / "paper_table3_ground_truth.csv"),
    )
    p.add_argument(
        "--out",
        default=str(root / "results" / "selection_results_500_curve_mean" / "comparison_to_paper.csv"),
    )
    args = p.parse_args()

    paper = pd.read_csv(args.paper)
    if not Path(args.summary).exists():
        print(f"No summary at {args.summary}")
        return
    summ = pd.read_csv(args.summary)
    summ = summ.rename(columns={"mean": "rerun_mean", "std": "rerun_std", "count": "n_seeds"})

    merged = paper.merge(summ, on=["dataset", "method"], how="outer")
    merged["delta_mean"] = merged["rerun_mean"] - merged["paper_mean"]
    merged["delta_std"]  = merged["rerun_std"]  - merged["paper_std"]
    merged["abs_delta_mean"] = merged["delta_mean"].abs()
    merged["pass_threshold"] = np.maximum(0.015, merged["paper_std"] / 2.0)
    merged["has_rerun"] = merged["rerun_mean"].notna()
    merged["pass"] = merged["has_rerun"] & (
        merged["abs_delta_mean"] <= merged["pass_threshold"]
    )
    merged = merged.sort_values(["dataset", "method"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(merged)} rows)\n")

    observed = merged[merged["has_rerun"]].copy()
    missing = merged[~merged["has_rerun"]].copy()

    # Top-line stats
    by_dataset = observed.groupby("dataset")["abs_delta_mean"].agg(["mean", "max", "count"]).reset_index()
    print("Per-dataset |Δ mean| summary:")
    print(by_dataset.to_string(index=False))

    print("\nWorst 15 mean deltas:")
    worst = observed.nlargest(15, "abs_delta_mean")[
        ["dataset", "method", "paper_mean", "rerun_mean", "delta_mean",
         "paper_std", "rerun_std", "n_seeds", "pass_threshold", "pass"]
    ]
    print(worst.to_string(index=False))

    # Pass criterion: |delta_mean| <= max(0.015, paper_std/2)
    pass_rate = observed["pass"].mean() if len(observed) else float("nan")
    print(f"\nPass rate (|Δ mean| <= max(0.015, paper_std/2)): "
          f"{observed['pass'].sum()}/{observed.shape[0]} = {pass_rate:.1%}")

    if len(missing):
        missing_keys = ", ".join(
            f"{r.dataset}/{r.method}" for r in missing[["dataset", "method"]].itertuples(index=False)
        )
        print(f"\nMissing rerun cells ({len(missing)}): {missing_keys}")


if __name__ == "__main__":
    main()
