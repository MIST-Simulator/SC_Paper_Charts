#!/usr/bin/env python3
"""Plot Fig. 10 (``figures/Case_Study_Memory_Cache.pdf``): end-to-end
request-latency CDFs comparing the five KV-cache storage architectures
(Table 2), across short (4K) / long (24K) KV contexts and private / shared
access patterns.

Reads ``results/T4/per_request.csv`` (falling back to
``results/reference/T4/per_request.csv``) written by ``run_T4.py``.  Runs
standalone -- ``mist_charts.memory_configs`` is plain declarative data (no
GenA/MIST import), so this script needs no MIST install to make the figure.
"""

import sys

import numpy as np
import pandas as pd

from mist_charts import apply_paper_style, save_figure
from mist_charts.memory_configs import CASE_COLORS, CASE_ORDER
from mist_charts.paths import resolve_results

# Legend reads "CaseA".."CaseE", matching the published figure -- not
# STORAGE_ARCHITECTURES[...].label (that longer form is used in the README
# table and the per-request CSV's "architecture" column instead).
CASE_LABELS = {key: f"Case{key}" for key in CASE_ORDER}

# Panel titles copied verbatim from the published Fig. 10.
PANEL_TITLES = {
    (4096, "private"): "Short(4k) Private KV Cache",
    (4096, "shared"): "Short(4k) Shared KV Cache",
    (24576, "private"): "Long(24k) Private KV Cache",
    (24576, "shared"): "Long(24k) Shared KV Cache",
}

# Matches the published figure's axes exactly, so the reproduction is a
# direct visual comparison.
X_MAX_SEC = 10
Y_MAX = 1.0


def load_per_request() -> pd.DataFrame:
    path = resolve_results("T4", "per_request.csv")
    print(f"reading {path}")
    return pd.read_csv(path)


def plot_cdfs(df: pd.DataFrame):
    import matplotlib.pyplot as plt

    context_lens = sorted(df["context_len"].unique())
    scenarios = ["private", "shared"]
    panels = [(cl, sc) for cl in context_lens for sc in scenarios if (cl, sc) in PANEL_TITLES]
    if not panels:
        # Fall back to whatever combinations are actually present, in case
        # run_T4.py was run with a non-default --context-len/--scenario.
        panels = sorted(df.groupby(["context_len", "scenario"]).groups.keys())

    num_cols = 2
    num_rows = max(1, -(-len(panels) // num_cols))
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 3 * num_rows), squeeze=False)

    for i, (context_len, scenario) in enumerate(panels):
        ax = axs[i // num_cols][i % num_cols]
        subset = df[(df["context_len"] == context_len) & (df["scenario"] == scenario)]
        # Denominator is the largest number of requests any one case
        # completed in this panel (matches the notebook's own convention:
        # `max_req_len = subset_df['End2End_Latency'].apply(len).max()`).
        # This is deliberate, not a bug: a case that only completes, say,
        # 40% of that count within the 10s sim window plateaus at 0.4
        # instead of climbing to 1.0, so the plateau height itself conveys
        # how much of the offered load that architecture could sustain.
        # Do NOT renormalize by each case's own completed count.
        max_req_len = subset.groupby("case")["end_to_end_latency_ms"].count().max()

        for case in CASE_ORDER:
            case_df = subset[subset["case"] == case]
            if case_df.empty:
                continue
            latencies_sorted_sec = np.sort(case_df["end_to_end_latency_ms"].to_numpy()) / 1000.0
            cdf = np.arange(1, len(latencies_sorted_sec) + 1) / max_req_len
            ax.plot(
                latencies_sorted_sec, cdf, marker=".", linestyle="-",
                color=CASE_COLORS[case], linewidth=3, markersize=4,
                label=CASE_LABELS[case],
            )

        ax.axhline(0.9, linestyle="dotted", linewidth=2, color="black")
        ax.set_xlabel("Latency(sec)")
        ax.set_ylabel("CDF")
        title = PANEL_TITLES.get((context_len, scenario), f"ctx={context_len} {scenario}")
        ax.set_title(title)
        ax.grid(True)
        ax.set_xlim(0, X_MAX_SEC)
        ax.set_ylim(0, Y_MAX)

    # Hide any unused trailing axes (e.g. a partial last row).
    for j in range(len(panels), num_rows * num_cols):
        axs[j // num_cols][j % num_cols].axis("off")

    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03),
        ncol=len(CASE_ORDER), frameon=True, columnspacing=1, handlelength=1,
    )
    fig.tight_layout()
    return fig


def main(argv=None) -> int:
    apply_paper_style()
    df = load_per_request()
    fig = plot_cdfs(df)
    save_figure(fig, "Case_Study_Memory_Cache.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
