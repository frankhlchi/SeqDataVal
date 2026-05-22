#!/usr/bin/env python3
"""Replot NeurIPS Figure 6 (RQ1 upper-bound curves) from raw RQ1 CSVs.

Important provenance correction:
  - NeurIPS Figure 6 includes `appro_results_small.pdf` in `neurips_2025.tex`.
  - The older Drive file `hudson/seq/rebutal/plots/upper_bound_results.pdf`
    is RQ1-like but not the final NeurIPS Figure 6 style.

This script uses the final NeurIPS color/order scheme visible in
`appro_results_small.pdf`: DynamicProgramming first in dark blue, followed by a
blue-to-red diverging palette for BetaShap/DataShap/Random/LOO/AME/Influence/
Banzhaf/DVRL/DataOob. It still reads the current local RQ1 raw CSVs and keeps
the historical selection-curve transform used by the paper scripts.

By default the script writes both a non-BBC validation plot and an all-available
plot. Pass ``--results_root`` to point at a completed ``results_rq1_dp`` tree.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_RESULTS_ROOT = Path("results_rq1_dp")
DEFAULT_OUT_DIR = Path("results/rq1_figure6")

DATASETS = [
    "2dplanes",
    "nomao",
    "bbc-embeddings",
    "MiniBooNE",
    "digits",
    "election",
    "electricity",
    "fried",
]

EXPECTED_SEEDS = list(range(10, 201, 10))

METHOD_ORDER = [
    "DynamicProgrammingEvaluator",
    "BetaShapley",
    "DataShapley",
    "RandomEvaluator",
    "LeaveOneOut",
    "AME",
    "InfluenceSubsample",
    "DataBanzhaf",
    "DVRL",
    "DataOob",
]

DISPLAY_LABELS = {
    "RandomEvaluator": "Random",
    "LeaveOneOut": "LOO",
    "InfluenceSubsample": "Influence",
    "DataShapley": "DataShap",
    "BetaShapley": "BetaShap",
    "DataBanzhaf": "Banzhaf",
    "AME": "AME",
    "DVRL": "DVRL",
    "DataOob": "DataOob",
    "DynamicProgrammingEvaluator": "DynamicProgramming",
}

NEURIPS_METHOD_COLORS = {
    "DynamicProgrammingEvaluator": "#1f4e8c",
    "BetaShapley": "#2f7fb9",
    "DataShapley": "#6aaed6",
    "RandomEvaluator": "#b7d7e8",
    "LeaveOneOut": "#e8f2f7",
    "AME": "#f5d7c6",
    "InfluenceSubsample": "#ef9f79",
    "DataBanzhaf": "#d6604d",
    "DVRL": "#b2182b",
    "DataOob": "#67001f",
}

EXCLUDED_METHODS = {
    "BipartiteMatchingEvaluator",
    "PredictionBasedMatchingEvaluator",
    "DualThresholdTripartiteEvaluator",
    "KNNShapley",
}

ACCURACY_COL = "remove_least_influential_first_Metrics.ACCURACY"

# Axis ranges follow the NeurIPS Figure 6 text extraction. The plotting code
# expands them if latest rerun data would otherwise be clipped.
PAPER_Y_RANGES = {
    "2dplanes": (0.50, 0.85),
    "nomao": (0.20, 0.85),
    "bbc-embeddings": (0.20, 0.90),
    "MiniBooNE": (0.50, 0.85),
    "digits": (0.05, 0.30),
    "election": (0.10, 0.65),
    "electricity": (0.45, 0.78),
    "fried": (0.50, 0.85),
}


@dataclass
class DatasetLoad:
    dataset: str
    frame: pd.DataFrame | None
    seeds_found: list[int]
    missing_seeds: list[int]
    methods_found: list[str]
    status: str


def base_method_name(value: object) -> str:
    return str(value).split("(", maxsplit=1)[0]


def find_seed_csv(results_root: Path, dataset: str, seed: int) -> Path:
    return results_root / dataset / f"seed_{seed}" / "addition_experiment_results.csv"


def load_dataset(results_root: Path, dataset: str, expected_seeds: Iterable[int]) -> DatasetLoad:
    frames: list[pd.DataFrame] = []
    seeds_found: list[int] = []
    missing_seeds: list[int] = []

    for seed in expected_seeds:
        csv_path = find_seed_csv(results_root, dataset, seed)
        if not csv_path.exists():
            missing_seeds.append(seed)
            continue
        df = pd.read_csv(csv_path, index_col=0)
        if "axis" not in df.columns or ACCURACY_COL not in df.columns:
            missing_seeds.append(seed)
            continue
        df = df.copy()
        df["method"] = df.index.map(base_method_name)
        df["seed"] = seed
        df = df[~df["method"].isin(EXCLUDED_METHODS)]
        frames.append(df[["method", "seed", "axis", ACCURACY_COL]])
        seeds_found.append(seed)

    if not frames:
        return DatasetLoad(
            dataset=dataset,
            frame=None,
            seeds_found=seeds_found,
            missing_seeds=missing_seeds,
            methods_found=[],
            status="missing",
        )

    frame = pd.concat(frames, ignore_index=True)
    methods_found = sorted(frame["method"].unique().tolist())
    status = "complete" if not missing_seeds else "partial"
    return DatasetLoad(
        dataset=dataset,
        frame=frame,
        seeds_found=seeds_found,
        missing_seeds=missing_seeds,
        methods_found=methods_found,
        status=status,
    )


def mean_and_ci(method_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    grouped = method_df.groupby("axis")[ACCURACY_COL]
    means = grouped.mean()
    counts = grouped.count()
    stds = grouped.std(ddof=1).fillna(0.0)

    ci_values = []
    for axis_value in means.index:
        n = int(counts.loc[axis_value])
        if n <= 1:
            ci_values.append(0.0)
        else:
            ci_values.append(
                float(stds.loc[axis_value])
                * float(stats.t.ppf((1 + 0.95) / 2.0, n - 1))
                / np.sqrt(n)
            )
    ci = pd.Series(ci_values, index=means.index)
    return means, ci


def expanded_ylim(dataset: str, plotted_values: list[np.ndarray]) -> tuple[float, float]:
    ymin, ymax = PAPER_Y_RANGES[dataset]
    finite_values: list[float] = []
    for values in plotted_values:
        finite = values[np.isfinite(values)]
        if finite.size:
            finite_values.extend(float(v) for v in finite)
    if not finite_values:
        return ymin, ymax

    low = min(finite_values)
    high = max(finite_values)
    span = max(ymax - ymin, 0.1)
    pad = span * 0.05
    if low < ymin:
        ymin = max(0.0, low - pad)
    if high > ymax:
        ymax = min(1.0, high + pad)
    return ymin, ymax


def apply_neurips_final_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.labelsize": 30,
            "axes.titlesize": 32,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 28,
            "lines.linewidth": 3.2,
            "grid.linewidth": 1.5,
            "axes.linewidth": 1.5,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
        }
    )


def plot_figure(
    loads: dict[str, DatasetLoad],
    output_base: Path,
    include_bbc: bool,
    mode_label: str,
) -> list[dict[str, str]]:
    apply_neurips_final_style()
    fig, axes = plt.subplots(2, 4, figsize=(32.6, 13.1))
    axes_flat = axes.flatten()
    manifest_rows: list[dict[str, str]] = []

    for ax, dataset in zip(axes_flat, DATASETS):
        load = loads[dataset]
        if dataset == "bbc-embeddings" and not include_bbc:
            ax.set_title(dataset)
            ax.text(
                0.5,
                0.5,
                "BBC skipped\nresource-heavy tail",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=22,
                fontweight="bold",
            )
            ax.set_xlim(0, 20)
            ax.set_ylim(*PAPER_Y_RANGES[dataset])
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.set_xlabel("Selection Size")
            ax.set_ylabel("Test Accuracy")
            manifest_rows.append(
                {
                    "mode": mode_label,
                    "dataset": dataset,
                    "status": "skipped_resource_tail",
                    "seeds_found": "",
                    "missing_seeds": ",".join(str(seed) for seed in EXPECTED_SEEDS),
                    "methods_found": "",
                }
            )
            continue

        ax.set_title(dataset)
        plotted_values: list[np.ndarray] = []
        if load.frame is None:
            ax.text(
                0.5,
                0.5,
                "missing",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=22,
                fontweight="bold",
            )
        else:
            for method_idx, method in enumerate(METHOD_ORDER):
                method_df = load.frame[load.frame["method"] == method]
                if method_df.empty:
                    continue
                means, ci = mean_and_ci(method_df)
                means = means.sort_index()
                ci = ci.reindex(means.index).fillna(0.0)

                # Historical Drive notebook transform. The first point is an
                # artificial origin and curve values are reversed before plot.
                x_values = np.concatenate(([0], means.index.to_numpy(dtype=float) + 1))
                y_values = np.concatenate(([0], means.to_numpy(dtype=float)[::-1]))
                ci_values = np.concatenate(([0], ci.to_numpy(dtype=float)[::-1]))
                plotted_values.append(y_values[1:])

                color = NEURIPS_METHOD_COLORS.get(method)
                ax.plot(
                    x_values,
                    y_values,
                    label=DISPLAY_LABELS[method],
                    color=color,
                    linestyle="-",
                    linewidth=3.8 if method == "DynamicProgrammingEvaluator" else 3.2,
                )
                ax.fill_between(
                    x_values,
                    y_values - ci_values,
                    y_values + ci_values,
                    color=color,
                    alpha=0.0,
                )

        ax.set_xlim(0, 20)
        ax.set_ylim(*expanded_ylim(dataset, plotted_values))
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_xlabel("Selection Size")
        ax.set_ylabel("Test Accuracy")
        manifest_rows.append(
            {
                "mode": mode_label,
                "dataset": dataset,
                "status": load.status,
                "seeds_found": ",".join(str(seed) for seed in load.seeds_found),
                "missing_seeds": ",".join(str(seed) for seed in load.missing_seeds),
                "methods_found": ",".join(load.methods_found),
            }
        )

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=10,
        bbox_to_anchor=(0.5, 1.08),
        fontsize=28,
        frameon=True,
        edgecolor="black",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    output_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_base.with_suffix(".pdf")
    png_path = output_base.with_suffix(".png")
    fig.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return manifest_rows


def write_manifest(out_dir: Path, rows: list[dict[str, str]], results_root: Path) -> None:
    manifest_path = out_dir / "figure6_latest_manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mode",
                "dataset",
                "status",
                "seeds_found",
                "missing_seeds",
                "methods_found",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    provenance_path = out_dir / "figure6_latest_provenance.txt"
    provenance_path.write_text(
        "\n".join(
            [
                "NeurIPS Figure 6 latest local replot",
                f"results_root={results_root}",
                "neurips_tex_include=appro_results_small.pdf",
                "neurips_reference_plot=appro_results_small.pdf",
                "old_drive_plot=hudson/seq/rebutal/plots/upper_bound_results.pdf (RQ1-like, not final Figure 6 style)",
                "metric=remove_least_influential_first_Metrics.ACCURACY",
                "aggregation=mean over available seeds; 95% CI computed but hidden (alpha=0) to match final PDF",
                "plot_transform=x=[0, axis+1], y=[0, reversed(mean_by_axis)]",
                "style=NeurIPS final appro_results_small blue-to-red palette; all solid lines; DynamicProgramming dark blue",
            ]
        )
        + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--mode",
        choices=["both", "nonbbc", "with_partial_bbc"],
        default="both",
        help="Which figure variant to write.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loads = {
        dataset: load_dataset(args.results_root, dataset, EXPECTED_SEEDS)
        for dataset in DATASETS
    }

    all_rows: list[dict[str, str]] = []
    if args.mode in {"both", "nonbbc"}:
        all_rows.extend(
            plot_figure(
                loads,
                args.out_dir / "appro_results_small_latest_nonbbc",
                include_bbc=False,
                mode_label="nonbbc",
            )
        )
    if args.mode in {"both", "with_partial_bbc"}:
        all_rows.extend(
            plot_figure(
                loads,
                args.out_dir / "appro_results_small_latest_with_partial_bbc",
                include_bbc=True,
                mode_label="with_partial_bbc",
            )
        )

    write_manifest(args.out_dir, all_rows, args.results_root)

    print(f"Wrote outputs to {args.out_dir}")
    for row in all_rows:
        if row["dataset"] == "bbc-embeddings":
            print(
                f"{row['mode']} bbc-embeddings status={row['status']} "
                f"seeds_found={row['seeds_found']} missing={row['missing_seeds']}"
            )


if __name__ == "__main__":
    main()
