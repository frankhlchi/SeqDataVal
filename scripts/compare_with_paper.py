#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def _strip_tex(s: str) -> str:
    s = s.replace("\\\\", "")
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\texttt\{([^}]*)\}", r"\1", s)
    s = s.replace("\\textsuperscript{\\scriptsize{$\\pm$0.000}}", "")
    s = s.replace("$", "")
    s = s.replace("\\pm", "±")
    s = s.replace("{", "").replace("}", "")
    return s.strip()


def parse_selection_results_500(tex: str) -> dict:
    m = re.search(r"\\label\{tab:selection_results_500\}.*?\\begin\{tabular\}.*?\\end\{tabular\}", tex, flags=re.S)
    if m is None:
        raise ValueError("Cannot find tab:selection_results_500 in paper tex.")
    block = m.group(0)

    tm = re.search(r"\\begin\{tabular\}\{.*?\}(?P<body>.*?)\\end\{tabular\}", block, flags=re.S)
    if tm is None:
        raise ValueError("Cannot find tabular body for tab:selection_results_500.")
    body = tm.group("body")

    body_lines: list[str] = []
    for ln in body.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith(("\\toprule", "\\midrule", "\\bottomrule", "\\multicolumn", "\\cmidrule")):
            continue
        body_lines.append(s)

    flat = " ".join(body_lines)
    row_strs = [r.strip() for r in re.split(r"\\\\\s*", flat) if r.strip()]

    rows: list[tuple[str, list[str]]] = []
    for row in row_strs:
        if "&" not in row:
            continue
        cells = [c.strip() for c in row.split("&")]
        if len(cells) < 2:
            continue
        rows.append((_strip_tex(cells[0]), cells[1:]))

    header_cells = next((cells for name, cells in rows if name == "Method"), None)
    if header_cells is None:
        raise ValueError("Cannot parse selection_results_500 header row.")
    datasets = [_strip_tex(c) for c in header_cells]
    datasets = [d.replace("bbc-embed", "bbc-embeddings") for d in datasets]

    def _find_row_idx(name: str) -> int:
        for i, (n, _cells) in enumerate(rows):
            if n == name:
                return i
        raise ValueError(f"Cannot find row: {name}")

    def _parse_methods(cells: list[str]) -> list[str | None]:
        out: list[str | None] = []
        for cell in cells:
            s = _strip_tex(cell).replace("*", "").strip()
            if s in {"-", ""}:
                out.append(None)
            else:
                out.append(s)
        return out

    def _parse_performance(cells: list[str]) -> list[tuple[float | None, float | None]]:
        out: list[tuple[float | None, float | None]] = []
        for cell in cells:
            s = _strip_tex(cell).strip()
            if s in {"-", ""}:
                out.append((None, None))
                continue
            nums = re.findall(r"([0-9]+\.[0-9]+)", cell)
            if not nums:
                nums = re.findall(r"([0-9]+\.[0-9]+)", s)
            if not nums:
                out.append((None, None))
                continue
            mean = float(nums[0])
            std = float(nums[1]) if len(nums) >= 2 else None
            out.append((mean, std))
        return out

    i1 = _find_row_idx("1st Place")
    if i1 + 1 >= len(rows) or rows[i1 + 1][0] != "Performance":
        raise ValueError("Malformed selection_results_500: 1st Place not followed by Performance row.")
    first_place_methods = _parse_methods(rows[i1][1])
    first_place_perf = _parse_performance(rows[i1 + 1][1])

    i2 = _find_row_idx("2nd Place")
    if i2 + 1 >= len(rows) or rows[i2 + 1][0] != "Performance":
        raise ValueError("Malformed selection_results_500: 2nd Place not followed by Performance row.")
    second_place_methods = _parse_methods(rows[i2][1])
    second_place_perf = _parse_performance(rows[i2 + 1][1])

    return {
        "datasets": datasets,
        "first": list(zip(first_place_methods, first_place_perf)),
        "second": list(zip(second_place_methods, second_place_perf)),
    }


def parse_utility_approx(tex: str) -> pd.DataFrame:
    m = re.search(r"\\label\{tab:utility_approx\}.*?\\begin\{tabular\}.*?\\end\{tabular\}", tex, flags=re.S)
    if m is None:
        raise ValueError("Cannot find tab:utility_approx in paper tex.")
    block = m.group(0)
    rows = []
    for ln in block.splitlines():
        if "&" not in ln:
            continue
        ln_s = ln.strip()
        if ln_s.startswith("\\toprule") or ln_s.startswith("\\midrule") or ln_s.startswith("\\bottomrule"):
            continue
        if ln_s.startswith("\\multirow") or ln_s.startswith("&"):
            continue
        if "Method" in ln_s and "MAE" in ln_s:
            continue
        cells = [c.strip() for c in ln.split("&")]
        if len(cells) < 5:
            continue
        method = _strip_tex(cells[0])
        nums = [float(x) for x in re.findall(r"([0-9]+\.[0-9]+)", ln)]
        if len(nums) < 4:
            continue
        rows.append(
            {
                "method": method.replace("DataShap", "DataShap").replace("BetaShap", "BetaShap"),
                "test_mae": nums[0],
                "test_mse": nums[1],
                "train_mae": nums[2],
                "train_mse": nums[3],
            }
        )
    return pd.DataFrame(rows)


def parse_rq2_caption(tex: str) -> dict[str, tuple[float, float]]:
    m = re.search(r"\\caption\{(?P<cap>.*?)\}\s*\\label\{fig:dyn_small\}", tex, flags=re.S)
    if m is None:
        raise ValueError("Cannot find fig:dyn_small caption.")
    cap = m.group("cap")
    # Example: Shapley (0.738→0.595), BetaShapley (0.706→0.597), and Banzhaf (0.720→0.590)
    pairs = {}
    for name in ["Shapley", "BetaShapley", "Banzhaf"]:
        mm = re.search(rf"{name}\s*\(([-0-9.]+)→([-0-9.]+)\)", cap)
        if mm:
            pairs[name] = (float(mm.group(1)), float(mm.group(2)))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare reproduced results with paper numbers.")
    parser.add_argument("--paper_tex", type=str, default="../../paper/src/main.tex")
    parser.add_argument("--budget500_summary", type=str, default="results/selection_results_500/summary.csv")
    parser.add_argument("--rq2_summary", type=str, default="results/curvature_rq2/rq2_curvature_summary.csv")
    parser.add_argument("--utility_overall", type=str, default="utility_appro/aggregate_results_overall.csv")
    parser.add_argument("--out_dir", type=str, default="results/compare_with_paper")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    paper_tex_path = (root / args.paper_tex).resolve()
    tex = paper_tex_path.read_text(encoding="utf-8", errors="ignore")

    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Utility approximation
    paper_utility = parse_utility_approx(tex)
    utility_path = (root / args.utility_overall).resolve()
    if utility_path.exists():
        our_utility = pd.read_csv(utility_path)
        our_utility = our_utility[our_utility["metric"].isin(["mae", "mse"])].copy()
        our_utility["method"] = our_utility["method"].replace({"Shapley": "DataShap", "BetaShapley": "BetaShap"})
        our_utility = our_utility.pivot_table(index="method", columns=["split", "metric"], values="mean")
        our_utility = our_utility.reindex(
            columns=[("test", "mae"), ("test", "mse"), ("train", "mae"), ("train", "mse")]
        ).reset_index()
        our_utility.columns = [
            "method",
            "test_mae",
            "test_mse",
            "train_mae",
            "train_mse",
        ]
    else:
        our_utility = paper_utility[["method"]].copy()
        our_utility["test_mae"] = pd.NA
        our_utility["test_mse"] = pd.NA
        our_utility["train_mae"] = pd.NA
        our_utility["train_mse"] = pd.NA
    util = paper_utility.merge(our_utility, on="method", how="left", suffixes=("_paper", "_ours"))
    for col in ["test_mae", "test_mse", "train_mae", "train_mse"]:
        util[f"{col}_diff"] = util[f"{col}_ours"] - util[f"{col}_paper"]
    util.to_csv(out_dir / "utility_approx_diff.csv", index=False)

    # 2) RQ2 caption numbers
    rq2_pairs = parse_rq2_caption(tex)
    rq2_rows = []
    rq2_path = (root / args.rq2_summary).resolve()
    if rq2_path.exists():
        rq2_df = pd.read_csv(rq2_path)
        for method_key, (p0, p1) in rq2_pairs.items():
            method_name = (
                "DataShapley" if method_key == "Shapley" else method_key
            )  # our script uses DataShapley label
            sub0 = rq2_df[(rq2_df["proportion"] == 0.0) & (rq2_df["method"] == method_name)]
            sub1 = rq2_df[(rq2_df["proportion"] == 1.0) & (rq2_df["method"] == method_name)]
            ours0 = float(sub0["mean"].iloc[0]) if len(sub0) else None
            ours1 = float(sub1["mean"].iloc[0]) if len(sub1) else None
            rq2_rows.append(
                {
                    "method": method_key,
                    "paper_p0": p0,
                    "paper_p1": p1,
                    "ours_p0": ours0,
                    "ours_p1": ours1,
                    "diff_p0": (ours0 - p0) if ours0 is not None else None,
                    "diff_p1": (ours1 - p1) if ours1 is not None else None,
                }
            )
    pd.DataFrame(rq2_rows).to_csv(out_dir / "rq2_diff.csv", index=False)

    # 3) Budget-500 table
    paper_sel = parse_selection_results_500(tex)
    budget_path = (root / args.budget500_summary).resolve()
    sel_rows = []
    if budget_path.exists():
        ours = pd.read_csv(budget_path)
        ours = ours.rename(columns={"mean": "ours_mean", "std": "ours_std"})
        datasets = paper_sel["datasets"]
        # drop trailing Average column from datasets list if present
        if datasets and datasets[-1].lower() == "average":
            datasets_no_avg = datasets[:-1]
        else:
            datasets_no_avg = datasets

        for rank in ["first", "second"]:
            for ds, (method, (mean, std)) in zip(datasets_no_avg, paper_sel[rank]):
                if method is None or mean is None:
                    continue
                ds_norm = ds
                sub = ours[(ours["dataset"] == ds_norm) & (ours["method"] == method)]
                ours_mean = float(sub["ours_mean"].iloc[0]) if len(sub) else None
                ours_std = float(sub["ours_std"].iloc[0]) if len(sub) else None
                sel_rows.append(
                    {
                        "rank": rank,
                        "dataset": ds_norm,
                        "method": method,
                        "paper_mean": mean,
                        "paper_std": std,
                        "ours_mean": ours_mean,
                        "ours_std": ours_std,
                        "diff_mean": (ours_mean - mean) if ours_mean is not None else None,
                    }
                )
    pd.DataFrame(sel_rows).to_csv(out_dir / "selection_results_500_diff.csv", index=False)

    # Markdown report
    lines = []
    lines.append(f"# Compare With Paper\n")
    lines.append(f"- Paper tex: `{paper_tex_path}`")
    lines.append(f"- Output dir: `{out_dir}`\n")

    lines.append("## Utility Approximation (Table tab:utility_approx)\n")
    lines.append(f"- Diff csv: `{out_dir / 'utility_approx_diff.csv'}`\n")

    lines.append("## RQ2 Curvature (Figure fig:dyn_small caption)\n")
    lines.append(f"- Diff csv: `{out_dir / 'rq2_diff.csv'}`\n")

    lines.append("## Large Budget=500 (Table tab:selection_results_500)\n")
    lines.append(f"- Diff csv: `{out_dir / 'selection_results_500_diff.csv'}`\n")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote:", out_dir / "report.md")
    print("Wrote:", out_dir / "utility_approx_diff.csv")
    print("Wrote:", out_dir / "rq2_diff.csv")
    print("Wrote:", out_dir / "selection_results_500_diff.csv")


if __name__ == "__main__":
    main()
