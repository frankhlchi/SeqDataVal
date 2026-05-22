#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import re


def _normalize_dataset_name(dataset: str) -> str:
    aliases = {
        "bbc-embedding": "bbc-embeddings",
        "bbc-embed": "bbc-embeddings",
        "bbc_embed": "bbc-embeddings",
        "miniboone": "MiniBooNE",
    }
    return aliases.get(dataset, dataset)


def _normalize_method(label: str) -> str:
    s = str(label)
    mapping = [
        ("DynamicProgrammingEvaluator", "DynamicProgramming"),
        ("RandomEvaluator", "Random"),
        ("LeaveOneOut", "LOO"),
        ("InfluenceSubsample", "Influence"),
        ("DataShapley", "DataShap"),
        ("BetaShapley", "BetaShap"),
        ("DataBanzhaf", "Banzhaf"),
        ("DataOob", "DataOob"),
        ("AME", "AME"),
        ("DVRL", "DVRL"),
        ("BipartiteMatchingEvaluator", "Bipartite"),
    ]
    for needle, name in mapping:
        if needle in s:
            return name
    return s


def _read_addition_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    # Main output uses the method label as the first (unnamed) column.
    first_col = df.columns[0]
    if first_col.startswith("Unnamed"):
        df = df.rename(columns={first_col: "method"})
    elif first_col != "method":
        df = df.rename(columns={first_col: "method"})
    return df


def _pick_accuracy_col(df: pd.DataFrame) -> str:
    # RQ1 selection is "add most valuable first", which corresponds to
    # "remove least influential first" (keep the most valuable points).
    cands = [c for c in df.columns if "remove_least_influential_first" in c and "ACCURACY" in c]
    if cands:
        return cands[0]

    # Fallback (should not be used for selection-quality evaluation).
    cands = [c for c in df.columns if "remove_most_influential_first" in c and "ACCURACY" in c]
    if cands:
        return cands[0]

    raise ValueError(f"Cannot find accuracy column in: {list(df.columns)}")


def _auc_mean(df_method: pd.DataFrame, *, acc_col: str) -> float:
    sub = df_method.sort_values("axis")
    x = sub["axis"].to_numpy(dtype=float)
    y = sub[acc_col].to_numpy(dtype=float)
    if len(x) < 2 or float(np.max(x)) <= 0.0:
        return float(np.mean(y)) if len(y) else float("nan")
    auc = float(np.trapz(y, x))
    return auc / float(np.max(x))


def collect_auc(results_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not results_root.exists():
        return pd.DataFrame()

    for dataset_dir in sorted(p for p in results_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        dataset = dataset_dir.name
        for seed_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")):
            seed = int(seed_dir.name.replace("seed_", ""))
            csv_path = seed_dir / "addition_experiment_results.csv"
            df = _read_addition_csv(csv_path)
            if df is None:
                continue

            acc_col = _pick_accuracy_col(df)
            if "axis" not in df.columns:
                continue

            for method_label, sub in df.groupby("method", sort=False):
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "method": _normalize_method(method_label),
                        "auc_mean": _auc_mean(sub, acc_col=acc_col),
                    }
                )

    return pd.DataFrame(rows)


def summarize_auc(auc_by_seed: pd.DataFrame) -> pd.DataFrame:
    if auc_by_seed.empty:
        return pd.DataFrame()
    return (
        auc_by_seed.groupby(["dataset", "method"])["auc_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["dataset", "mean"], ascending=[True, False], kind="stable")
    )


def dp_gap(auc_summary: pd.DataFrame) -> pd.DataFrame:
    if auc_summary.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for dataset, sub in auc_summary.groupby("dataset", sort=False):
        dp = sub[sub["method"] == "DynamicProgramming"]
        if dp.empty:
            continue
        dp_mean = float(dp["mean"].iloc[0])
        baselines = sub[sub["method"] != "DynamicProgramming"]
        if baselines.empty:
            continue
        best = baselines.sort_values("mean", ascending=False).iloc[0]
        best_mean = float(best["mean"])
        gap_abs = dp_mean - best_mean
        gap_rel = gap_abs / dp_mean if dp_mean != 0 else float("nan")
        rows.append(
            {
                "dataset": dataset,
                "dp_mean": dp_mean,
                "best_baseline": str(best["method"]),
                "best_baseline_mean": best_mean,
                "gap_abs": gap_abs,
                "gap_rel": gap_rel,
            }
        )
    return pd.DataFrame(rows).sort_values("gap_rel", ascending=False, kind="stable")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate RQ1 (DP) results into AUC summaries.")
    parser.add_argument("--results_root", type=str, default="results_rq1_dp")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    results_root = (root / args.results_root).resolve()
    out_dir = (results_root / "_summary").resolve() if args.out_dir is None else (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    auc_by_seed = collect_auc(results_root)
    auc_summary = summarize_auc(auc_by_seed)
    gap = dp_gap(auc_summary)

    (out_dir / "rq1_auc_by_seed.csv").write_text(auc_by_seed.to_csv(index=False), encoding="utf-8")
    (out_dir / "rq1_auc_summary.csv").write_text(auc_summary.to_csv(index=False), encoding="utf-8")
    (out_dir / "rq1_dp_gap_summary.csv").write_text(gap.to_csv(index=False), encoding="utf-8")

    # Optional: parse the paper text for the reported DP-gap range (if available).
    paper_tex = root.parents[2] / "paper" / "src" / "main.tex"
    paper_gap_min = None
    paper_gap_max = None
    paper_gap_bbc = None
    if paper_tex.exists():
        text = paper_tex.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"underperform optimal selection by ([0-9]+\\.[0-9]+)\\% to ([0-9]+\\.[0-9]+)\\%", text)
        if m:
            paper_gap_min = float(m.group(1))
            paper_gap_max = float(m.group(2))
        m = re.search(r"bbc-embeddings[^\\n]*?([0-9]+\\.[0-9]+)\\%", text)
        if m:
            paper_gap_bbc = float(m.group(1))

    lines: list[str] = []
    lines.append("# RQ1 (DP) Aggregate Report\n")
    lines.append(f"- results_root: `{results_root}`")
    lines.append(f"- out_dir: `{out_dir}`\n")
    lines.append("## Outputs\n")
    lines.append(f"- `rq1_auc_by_seed.csv`")
    lines.append(f"- `rq1_auc_summary.csv`")
    lines.append(f"- `rq1_dp_gap_summary.csv`\n")

    if not gap.empty:
        gap_pct = (gap["gap_rel"].astype(float) * 100.0).to_list()
        lines.append("## DP Gap (AUC) Summary\n")
        lines.append(f"- reproduced gap range (min/max): {min(gap_pct):.2f}% / {max(gap_pct):.2f}%")
        bbc = gap[gap["dataset"] == "bbc-embeddings"]
        if not bbc.empty:
            lines.append(f"- reproduced bbc-embeddings gap: {float(bbc['gap_rel'].iloc[0]) * 100.0:.2f}%")
        lines.append("")

    if paper_gap_min is not None and paper_gap_max is not None:
        lines.append("## Paper Numbers (parsed)\n")
        lines.append(f"- paper gap range (min/max): {paper_gap_min:.2f}% / {paper_gap_max:.2f}%")
        if paper_gap_bbc is not None:
            lines.append(f"- paper bbc-embeddings gap: {paper_gap_bbc:.2f}%")
        lines.append("")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("Wrote:", out_dir / "rq1_auc_by_seed.csv")
    print("Wrote:", out_dir / "rq1_auc_summary.csv")
    print("Wrote:", out_dir / "rq1_dp_gap_summary.csv")
    print("Wrote:", out_dir / "report.md")


if __name__ == "__main__":
    main()
