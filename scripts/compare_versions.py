#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def compare_budget500(merge_path: Path, clean_path: Path) -> pd.DataFrame:
    merge_df = _read_csv(merge_path)
    clean_df = _read_csv(clean_path)

    if not merge_df.empty:
        merge_df = merge_df.rename(
            columns={"mean": "merge_mean", "std": "merge_std", "count": "merge_count"}
        )
    if not clean_df.empty:
        clean_df = clean_df.rename(
            columns={"mean": "clean_mean", "std": "clean_std", "count": "clean_count"}
        )

    if merge_df.empty and clean_df.empty:
        return pd.DataFrame()

    out = merge_df.merge(clean_df, on=["dataset", "method"], how="outer")
    out["diff_mean"] = out["merge_mean"] - out["clean_mean"]
    out["diff_std"] = out["merge_std"] - out["clean_std"]
    out = out.sort_values(["dataset", "method"], kind="stable")
    return out


def compare_rq2(merge_path: Path, clean_path: Path) -> pd.DataFrame:
    merge_df = _read_csv(merge_path)
    clean_df = _read_csv(clean_path)

    if not merge_df.empty:
        merge_df = merge_df.rename(
            columns={"mean": "merge_mean", "std": "merge_std", "count": "merge_count"}
        )
    if not clean_df.empty:
        clean_df = clean_df.rename(
            columns={"mean": "clean_mean", "std": "clean_std", "count": "clean_count"}
        )

    if merge_df.empty and clean_df.empty:
        return pd.DataFrame()

    out = merge_df.merge(clean_df, on=["proportion", "method"], how="outer")
    out["diff_mean"] = out["merge_mean"] - out["clean_mean"]
    out["diff_std"] = out["merge_std"] - out["clean_std"]
    out = out.sort_values(["method", "proportion"], kind="stable")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare merge-v1 vs paper-aligned cleaned outputs.")
    parser.add_argument("--merge_root", type=str, default=".")
    parser.add_argument("--clean_root", type=str, default="../SequentialDataVal_cleaned")
    parser.add_argument("--merge_budget500", type=str, default="results/selection_results_500/summary.csv")
    parser.add_argument("--clean_budget500", type=str, default="results/selection_results_500/summary.csv")
    parser.add_argument("--merge_rq2", type=str, default="results/curvature_rq2/rq2_curvature_summary.csv")
    parser.add_argument("--clean_rq2", type=str, default="results/curvature_rq2/rq2_curvature_summary.csv")
    parser.add_argument("--out_dir", type=str, default="results/compare_versions")
    args = parser.parse_args()

    merge_root = Path(args.merge_root).expanduser().resolve()
    clean_root = Path(args.clean_root).expanduser().resolve()
    out_dir = (merge_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    budget500_diff = compare_budget500(
        merge_root / args.merge_budget500,
        clean_root / args.clean_budget500,
    )
    rq2_diff = compare_rq2(
        merge_root / args.merge_rq2,
        clean_root / args.clean_rq2,
    )

    (out_dir / "budget500_version_diff.csv").write_text(
        budget500_diff.to_csv(index=False) if not budget500_diff.empty else "", encoding="utf-8"
    )
    (out_dir / "rq2_version_diff.csv").write_text(
        rq2_diff.to_csv(index=False) if not rq2_diff.empty else "", encoding="utf-8"
    )

    lines: list[str] = []
    lines.append("# Compare Versions (merge-v1 vs paper-aligned cleaned)\n")
    lines.append(f"- merge root: `{merge_root}`")
    lines.append(f"- cleaned root: `{clean_root}`")
    lines.append(f"- output dir: `{out_dir}`\n")
    lines.append("## Budget=500 (Table tab:selection_results_500)\n")
    lines.append(f"- diff csv: `{out_dir / 'budget500_version_diff.csv'}`\n")
    lines.append("## RQ2 Curvature summary\n")
    lines.append(f"- diff csv: `{out_dir / 'rq2_version_diff.csv'}`\n")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("Wrote:", out_dir / "report.md")
    print("Wrote:", out_dir / "budget500_version_diff.csv")
    print("Wrote:", out_dir / "rq2_version_diff.csv")


if __name__ == "__main__":
    main()

