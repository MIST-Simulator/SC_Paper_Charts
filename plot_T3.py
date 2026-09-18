#!/usr/bin/env python3
"""Reproduce Fig. 7(a)-(b) and Fig. 8 of the MIST SC26 paper from run_T3.py's
per-use-case result CSVs.

  Fig. 7(a)  figures/Chat_search_results.pdf
  Fig. 7(b)  figures/Code_Generation_KV_search_results.pdf
  Fig. 8     figures/Combined_2x2_bar_results.pdf

Reads via ``mist_charts.paths.resolve_results("T3", ...)`` (falling back to
``results/reference/T3/`` when ``results/T3/`` is empty) and never imports
the MIST simulator: every number plotted here was already computed by
``run_T3.py``.

Faithful port of ``GenA_Paper_charts/SC26/plot_sc_results.py`` (Fig. 7:
``parse_hardware``, ``parse_serving_name``, ``get_vendor``, ``compute_cost``,
``_flatten``, ``compute_ttft_percentiles``, ``compute_pareto_front``,
``load_and_process``, ``build_side_by_side_plot``) and
``GenA_Paper_charts/SC26/plot_bar_results.py`` (Fig. 8: ``plot_column``,
``build_2x2_plots``) -- the two scripts confirmed to have produced the
camera-ready figures. Operates on the author's own schema directly:
``load_and_process`` regex-parses the concatenated "UseCase" / "Serving
Name" strings exactly as the source does, rather than reading precomputed
columns. Prices, vendor grouping and colors come from ``mist_charts.pricing``
(reconciled with the source's ``DEFAULT_PRICES``) instead of a second
copy inlined here.

One deliberate fix, kept: the source hardcodes a "+" in Fig. 8's
throughput-gain label (``f"+{pct_increase:.1f}%"``), so a *loss* renders as
"+-11.1%". This is a genuine bug in the source, not a stylistic choice --
losses are rendered signed (a leading "-") and in red instead. Also
retained beyond the source: Fig. 7's scatter and Fig. 8's bar chart
inherit two different "Mixed" definitions from the source itself (Fig. 7's
``assign_category`` also calls a same-vendor heterogeneous pair "Mixed";
Fig. 8's ``is_multi_vendor`` requires different vendors) -- preserved
faithfully rather than reconciled; see docs/FINDINGS.md#t3.
"""

import argparse
import ast
import re
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg", force=True)  # headless: this repo's scripts never open a GUI window
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from mist_charts.paths import resolve_results
from mist_charts.pricing import (
    IGNORE_HW,
    PERFORMANCE_SCALE_FACTOR,
    PRICE_PER_HOUR,
    SKU_DISPLAY_NAMES,
    VENDOR_CATEGORY_ORDER,
    VENDOR_COLORS,
    VENDOR_OF_SKU,
)
from mist_charts.style import save_figure

# ─────────────────────── Font sizes (ported from plot_sc_results.py /
# plot_bar_results.py FONT_SIZES -- large, print/poster-scale) ────────────
FONT_SIZES = {
    "title": 36,
    "axis_label": 36,
    "tick_label": 36,
    "legend": 36,
    "subplot_title": 36,
}

# Ported from plot_sc_results.py's CATEGORY_STYLES, restricted to the
# categories our 8-SKU search space can actually produce (Groq/Cerebras/
# Other never appear: Cerebras is the one non-8-SKU hardware the author's
# real CSVs contain, and IGNORE_HW drops it before category assignment).
# Dict order also fixes scatter draw order and Fig. 7's legend order
# (Mixed, Etched, TPU, AMD, Nvidia), matching the paper figure. Colors
# match mist_charts.pricing.VENDOR_COLORS (the single source for them).
CATEGORY_STYLES = {
    "Mixed": {"color": VENDOR_COLORS["Mixed"], "marker": "s", "edgecolor": "navy"},
    "Etched": {"color": VENDOR_COLORS["Etched"], "marker": "o", "edgecolor": "#a65100"},
    "TPU": {"color": VENDOR_COLORS["TPU"], "marker": "o", "edgecolor": "#5e3c81"},
    "AMD": {"color": VENDOR_COLORS["AMD"], "marker": "o", "edgecolor": "#0e7a85"},
    "Nvidia": {"color": VENDOR_COLORS["Nvidia"], "marker": "o", "edgecolor": "darkgreen"},
}

# get_vendor's per-vendor SKU lists, derived from mist_charts.pricing's
# single VENDOR_OF_SKU dict (the source instead hardcodes four such lists
# directly). Groq/Cerebras have no entries -- pricing.py restricts to the
# paper's Sec. 5.1 8-SKU table -- so those two vendor branches below are
# unreachable in practice (Cerebras rows are dropped by IGNORE_HW first);
# kept for a faithful control-flow match with the source.
NVIDIA_DEVICES = [sku for sku, v in VENDOR_OF_SKU.items() if v == "Nvidia"]
TPU_DEVICES = [sku for sku, v in VENDOR_OF_SKU.items() if v == "TPU"]
AMD_DEVICES = [sku for sku, v in VENDOR_OF_SKU.items() if v == "AMD"]
ETCHED_DEVICES = [sku for sku, v in VENDOR_OF_SKU.items() if v == "Etched"]

USE_CASES: Dict[str, Dict] = {
    "chat": dict(
        results_file="chat_results.csv",
        ttft_slo_ms=200.0,
        title="Chat Conversation",
        fig7_name="Chat_search_results.pdf",
        # plot_sc_results.py's main(): rank1_offset=(15, 0), rank2_offset=(15, -35), legend=True
        rank1_offset=(15, 0),
        rank2_offset=(15, -35),
        legend=True,
    ),
    "codegen": dict(
        results_file="codegen_results.csv",
        ttft_slo_ms=1500.0,
        title="Code Gen (High Prefix-KV Reuse)",
        fig7_name="Code_Generation_KV_search_results.pdf",
        # plot_sc_results.py's main(): rank1_offset=(-15, 10), rank2_offset=(-15, -5), legend=False
        rank1_offset=(-15, 10),
        rank2_offset=(-15, -5),
        legend=False,
    ),
}


# ─────────────────────────── Helpers (verbatim ports of
# plot_sc_results.py's module-level functions) ─────────────────────────────


def parse_hardware(use_case: str):
    """Extract hardware names from a "UseCase" string, e.g.
    "Poisson - 100RPS - h200_sxm-mi350x" -> ("h200_sxm", "mi350x", True).
    Verbatim port of plot_sc_results.py's `parse_hardware`.
    """
    parts = use_case.split(" - ")
    hw_part = parts[-1].strip()
    if "-" in hw_part:
        prefill_hw, decode_hw = hw_part.split("-", 1)
        is_hetero = prefill_hw != decode_hw
    else:
        prefill_hw = decode_hw = hw_part
        is_hetero = False
    return prefill_hw, decode_hw, is_hetero


def parse_serving_name(serving_name: str):
    """Extract total/prefill/decode node counts from a "Serving Name"
    string (e.g. "BatchingMethod.DISAGGREGATED_8_2_6_0_..." or
    "BatchingMethod.CHUNKED_8_2048"). Verbatim port of
    plot_sc_results.py's `parse_serving_name`.
    """
    if "DISAGGREGATED" in serving_name:
        m = re.search(r"DISAGGREGATED_(\d+)_(\d+)_(\d+)", serving_name)
        if m:
            return int(m.group(1)), int(m.group(2)), int(m.group(3))
    elif "CHUNKED" in serving_name:
        m = re.search(r"CHUNKED_(\d+)_(\d+)", serving_name)
        if m:
            total = int(m.group(1))
            return total, total, 0
    return None, None, None


def get_vendor(hw) -> str:
    """Vendor name from a hardware handle. Verbatim port of
    plot_sc_results.py's `get_vendor`, minus the Groq/Cerebras branches
    (see NVIDIA_DEVICES et al. above for why)."""
    hw = str(hw).lower()
    if any(d in hw for d in NVIDIA_DEVICES):
        return "Nvidia"
    if any(d in hw for d in TPU_DEVICES):
        return "TPU"
    if any(d in hw for d in AMD_DEVICES):
        return "AMD"
    if any(d in hw for d in ETCHED_DEVICES):
        return "Etched"
    return "Other"


def compute_cost(prefill_hw, decode_hw, total_nodes, prefill_nodes, decode_nodes, is_disaggregated, prices) -> float:
    """Cost per hour. Verbatim port of plot_sc_results.py's `compute_cost`."""
    if is_disaggregated:
        return prices.get(prefill_hw, 0) * prefill_nodes + prices.get(decode_hw, 0) * decode_nodes
    return prices.get(prefill_hw, 0) * total_nodes


def _flatten(lst):
    """Recursively flatten arbitrarily nested lists. Verbatim port of
    plot_sc_results.py's `_flatten`."""
    for item in lst:
        if isinstance(item, list):
            yield from _flatten(item)
        else:
            yield item


def compute_ttft_percentiles(ongoing_str, ttft_str):
    """Merge and flatten Ongoing_TTFT_latencies + TTFT_latencies, compute
    P90/P95/P99. Verbatim port of plot_sc_results.py's
    `compute_ttft_percentiles`."""
    all_vals = []
    for raw in (ongoing_str, ttft_str):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                all_vals.extend(_flatten(parsed))
        except Exception:
            pass
    all_vals = [v for v in all_vals if isinstance(v, (int, float)) and not np.isnan(v)]
    if not all_vals:
        return np.nan, np.nan, np.nan
    arr = np.array(all_vals, dtype=float)
    return np.percentile(arr, 90), np.percentile(arr, 95), np.percentile(arr, 99)


def compute_pareto_front(x_vals, y_vals) -> List[int]:
    """Pareto front: minimize x, maximize y. Returns indices sorted by x.
    Verbatim port of plot_sc_results.py's `compute_pareto_front`."""
    n = len(x_vals)
    if n == 0:
        return []
    sorted_indices = np.argsort(x_vals)
    pareto_indices = []
    best_y = -np.inf
    for idx in sorted_indices:
        if y_vals[idx] > best_y:
            pareto_indices.append(idx)
            best_y = y_vals[idx]
    return pareto_indices


# ─────────────────────────── Data loading (port of plot_sc_results.py's
# load_and_process) ───────────────────────────────────────────────────────


def load_and_process(use_case: str, prices: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Load one use case's results CSV and derive every column the plots
    need. Faithful port of plot_sc_results.py's `load_and_process`,
    differing only in where the CSV path and price table come from."""
    if prices is None:
        prices = PRICE_PER_HOUR

    cfg = USE_CASES[use_case]
    path = resolve_results("T3", cfg["results_file"])
    df = pd.read_csv(path)

    # Parse hardware first
    hw_info = df["UseCase"].apply(parse_hardware)
    df["prefill_hw"] = hw_info.apply(lambda x: x[0])
    df["decode_hw"] = hw_info.apply(lambda x: x[1])
    df["is_hetero"] = hw_info.apply(lambda x: x[2])
    df["is_disaggregated"] = df["Serving Name"].str.contains("DISAGGREGATED")
    # Filter out rows with hardware in IGNORE_HW
    df = df[~df["prefill_hw"].isin(IGNORE_HW)]
    df = df[~df["decode_hw"].isin(IGNORE_HW)]
    # Parse serving name
    serving_info = df["Serving Name"].apply(parse_serving_name)
    df["total_nodes"] = serving_info.apply(lambda x: x[0])
    df["prefill_nodes"] = serving_info.apply(lambda x: x[1])
    df["decode_nodes"] = serving_info.apply(lambda x: x[2])

    # Compute cost. `Serving Name` carries only the DP replica count, so
    # compute_cost bills a TP2xDP4 deployment for 4 accelerators rather
    # than 8 -- halving the cost of every TP>1 configuration. run_T3.py
    # writes the true accelerator count (TP*PP*DP) to `Cost`, so prefer
    # that and fall back to the derived value only for result files that
    # predate the column.
    derived_cost = df.apply(
        lambda row: compute_cost(
            row["prefill_hw"], row["decode_hw"],
            row["total_nodes"], row["prefill_nodes"], row["decode_nodes"],
            row["is_disaggregated"], prices,
        ), axis=1,
    )
    if "Cost" in df.columns:
        df["cost_per_hour"] = df["Cost"].fillna(derived_cost)
    else:
        df["cost_per_hour"] = derived_cost

    # TTFT percentiles from flattened Ongoing_TTFT_latencies + TTFT_latencies
    ttft_pcts = df.apply(
        lambda row: compute_ttft_percentiles(row["Ongoing_TTFT_latencies"], row["TTFT_latencies"]), axis=1
    )
    df["TTFT_P90"] = ttft_pcts.apply(lambda x: x[0])
    df["TTFT_P95"] = ttft_pcts.apply(lambda x: x[1])
    df["TTFT_P99"] = ttft_pcts.apply(lambda x: x[2])

    # Throughput
    df["Throughput"] = df["total_token_throughput"]
    df["Output_Throughput"] = df["output_throughput"]
    df["Output_Throughput_Scaled"] = df["Output_Throughput"] * df["decode_hw"].map(PERFORMANCE_SCALE_FACTOR)
    # Normalize throughput metrics by the MINIMUM value (source variable is
    # misnamed "max_throughput" but the behavior -- and the caption -- is
    # "normalized against the lowest-throughput configuration").
    min_throughput = df["Output_Throughput_Scaled"].min()
    df["Normalized_Throughput"] = df["Output_Throughput_Scaled"] / min_throughput

    # Cost-normalized throughput
    df["Throughput_per_dollar"] = df["Output_Throughput_Scaled"] / df["cost_per_hour"]
    min_throughput_per_dollar = df["Throughput_per_dollar"].min()
    df["Normalized_Throughput_per_dollar"] = df["Throughput_per_dollar"] / min_throughput_per_dollar

    # Assign plotting category
    df["prefill_vendor"] = df["prefill_hw"].apply(get_vendor)
    df["decode_vendor"] = df["decode_hw"].apply(get_vendor)
    df["is_multi_vendor"] = df["prefill_vendor"] != df["decode_vendor"]
    df["is_homogeneous"] = df["prefill_hw"] == df["decode_hw"]

    def assign_category(row):
        if row["is_multi_vendor"] or not row["is_homogeneous"]:
            return "Mixed"
        return f"{row['prefill_vendor']}"

    df["plot_category"] = df.apply(assign_category, axis=1)

    return df.reset_index(drop=True)


# ─────────────────────────── Fig. 7: side-by-side scatter (verbatim port
# of plot_sc_results.py's build_side_by_side_plot) ─────────────────────────


def build_side_by_side_plot(
    df: pd.DataFrame,
    slo_ms: float,
    legend: bool,
    rank1_offset: Tuple[int, int],
    rank2_offset: Tuple[int, int],
    font_sizes: Optional[Dict] = None,
):
    if font_sizes is None:
        font_sizes = FONT_SIZES

    plt.style.use("seaborn-v0_8-paper")

    y_metrics = [
        ("Normalized_Throughput", "Normalized Tokens/s"),
        ("Normalized_Throughput_per_dollar", "Normalized Tokens/s/$"),
    ]
    x_metric = ("TTFT_P99", "TTFT P99 (ms)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    marker_size = 100
    marker_alpha = 0.7

    # Find best points: highest throughput/$ with TTFT P99 < slo_ms
    best_rows: List[pd.Series] = []
    mask = df["TTFT_P99"] < slo_ms
    if mask.any():
        n_best = min(2, int(mask.sum()))
        best_idxs = df.loc[mask, "Throughput_per_dollar"].nlargest(n_best).index
        best_rows = [df.loc[idx] for idx in best_idxs]
    else:
        print(f"Warning: no config meets TTFT P99 < {slo_ms}ms -- no stars/annotations drawn.")

    for idx, (y_col, y_lbl) in enumerate(y_metrics):
        ax = axes[idx]

        # Highlight best configurations if found
        for rank, b_row in enumerate(best_rows, start=1):
            ax.scatter(
                b_row[x_metric[0]], b_row[y_col],
                marker="*", s=900, color="gold", edgecolor="black",
                linewidths=2, zorder=10,
            )

            # Add annotation in the 2nd subplot
            if idx == 1:
                hw_text = (
                    f"{SKU_DISPLAY_NAMES.get(b_row['prefill_hw'], b_row['prefill_hw'])}:"
                    f"{SKU_DISPLAY_NAMES.get(b_row['decode_hw'], b_row['decode_hw'])}"
                    if b_row["is_hetero"]
                    else f"{SKU_DISPLAY_NAMES.get(b_row['prefill_hw'], b_row['prefill_hw'])}"
                )

                if b_row["is_disaggregated"]:
                    p_inst = int(b_row["prefill_nodes"])
                    d_inst = int(b_row["decode_nodes"])
                    p_tp, d_tp = None, None
                    for p in [1, 2, 4, 8]:
                        for d in [1, 2, 4, 8]:
                            if p_inst * p + d_inst * d == 8:
                                p_tp, d_tp = p, d
                                break
                        if p_tp is not None:
                            break

                    if p_tp is not None:
                        pd_tp_text = f"{p_inst}P:{d_inst}D, TP: (P={p_tp}, D={d_tp})"
                    else:
                        pd_tp_text = f"{p_inst}P:{d_inst}D, TP={b_row['total_nodes']}"
                else:
                    pd_tp_text = f"TP={b_row['total_nodes']}"

                ann_text = f"#{rank}: {hw_text}\n{pd_tp_text}"

                x_offset, y_offset = rank1_offset if rank == 1 else rank2_offset
                ha = "right" if x_offset < 0 else "left"
                va = "bottom" if y_offset > 0 else "top"

                ax.annotate(
                    ann_text,
                    (b_row[x_metric[0]], b_row[y_col]),
                    xytext=(x_offset, y_offset),
                    textcoords="offset points",
                    fontsize=int(font_sizes["legend"] * 0.6),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
                    zorder=11,
                    horizontalalignment=ha,
                    verticalalignment=va,
                )

        # Plot scatter points by category
        for cat_name, style in CATEGORY_STYLES.items():
            cat_df = df[df["plot_category"] == cat_name]
            if len(cat_df) > 0:
                ax.scatter(
                    cat_df[x_metric[0]], cat_df[y_col],
                    c=style["color"], marker=style["marker"], s=marker_size,
                    alpha=marker_alpha, edgecolors=style["edgecolor"], linewidths=1.5,
                    label=cat_name, zorder=3,
                )

        # Compute and plot Pareto fronts
        homogenous_df = df[df["is_homogeneous"]]
        heterogenous_df = df[~df["is_homogeneous"]]

        for grp_df, line_color, line_style, _line_label in [
            (homogenous_df, "#d62728", "-", "Homogenous Deployement"),  # Crimson Red for Homo
            (heterogenous_df, "#1f77b4", "--", "Heterogenous Deployement"),  # Blue for Hetero
        ]:
            if len(grp_df) == 0:
                continue

            xg = grp_df[x_metric[0]].to_numpy(dtype=float)
            yg = grp_df[y_col].to_numpy(dtype=float)
            valid_g = ~(np.isnan(xg) | np.isnan(yg))

            if valid_g.sum() == 0:
                continue

            pidx = compute_pareto_front(xg[valid_g], yg[valid_g])
            if not pidx:
                continue

            orig_g = np.where(valid_g)[0]
            px = xg[orig_g[pidx]]
            py = yg[orig_g[pidx]]
            sort_order = np.argsort(px)
            px, py = px[sort_order], py[sort_order]

            ax.plot(px, py, color=line_color, linestyle=line_style, linewidth=2.5, label="_nolegend_", zorder=2)

        # Set log scale for x-axis
        ax.set_xscale("log")

        # Add light green shaded area for x < slo_ms
        ax.axvspan(0, slo_ms, color="#d4edda", alpha=0.4, zorder=0)

        # Set labels
        ax.set_xlabel(x_metric[1], fontsize=font_sizes["axis_label"])
        ax.set_ylabel(y_lbl, fontsize=font_sizes["axis_label"])

        # Set tick label sizes
        ax.tick_params(axis="both", labelsize=font_sizes["tick_label"])
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        # Add grid
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    # Create a single legend for the entire figure at the top in a single row
    handles, labels = axes[0].get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))

    # Always draw the legend to identically lock the bounding box for both PDFs
    leg = fig.legend(
        unique_labels.values(), unique_labels.keys(), loc="lower center", bbox_to_anchor=(0.47, 0.94),
        ncol=len(unique_labels), fontsize=font_sizes["legend"], frameon=True, framealpha=0.3,
        columnspacing=0.8, handletextpad=0, labelspacing=0, handlelength=1.2,
    )

    if not legend:
        # Make it invisible but preserve its layout footprint
        for t in leg.get_texts():
            t.set_alpha(0)
        for h in (leg.legendHandles if hasattr(leg, "legendHandles") else leg.legend_handles):
            if hasattr(h, "set_alpha"):
                h.set_alpha(0)
        leg.get_frame().set_alpha(0)
        if hasattr(leg.get_frame(), "set_linewidth"):
            leg.get_frame().set_linewidth(0)

    if not mask.any():
        # Defensive, beyond the ported source (which just silently omits
        # stars): make the "nobody met the SLO" state explicit on the
        # figure itself, without fabricating a winner. Matters for a tiny
        # smoke-test sweep (few configs, short sim window), where no
        # config may meet the SLO at all.
        for ax in axes:
            ax.text(
                0.5, 0.5, f"No configuration met\nTTFT P99 < {slo_ms:g} ms",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=font_sizes["legend"] * 0.5, color="firebrick", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="firebrick", alpha=0.9),
                zorder=12,
            )

    return fig


def plot_fig7(use_case: str) -> None:
    cfg = USE_CASES[use_case]
    df = load_and_process(use_case)
    print(f"[{use_case}] loaded {len(df)} rows for Fig. 7")
    fig = build_side_by_side_plot(
        df, slo_ms=cfg["ttft_slo_ms"], legend=cfg["legend"],
        rank1_offset=cfg["rank1_offset"], rank2_offset=cfg["rank2_offset"],
    )
    save_figure(fig, cfg["fig7_name"])
    plt.close(fig)


# ─────────────────────────── Fig. 8: 2x2 bars (verbatim port of
# plot_bar_results.py's plot_column + build_2x2_plots) ─────────────────────


def plot_column(ax1, ax2, df: pd.DataFrame, slo_ms: float, title: str, font_sizes: Dict) -> None:
    # Filter by SLO
    df_valid = df[df["TTFT_P99"] < slo_ms]
    if df_valid.empty:
        print(f"Warning: No data points meet the SLO of {slo_ms}ms for {title}.")

    # Extract vendors present in valid data, excluding multi-vendor configurations
    vendors_present = df_valid[~df_valid["is_multi_vendor"]]["prefill_vendor"].unique() if not df_valid.empty else []

    # Ordered list of known vendors (VENDOR_CATEGORY_ORDER ends in "Mixed",
    # which is appended separately below, exactly as the source's own
    # `all_known_vendors` list -- which never includes "Mixed" -- does).
    vendors = [v for v in VENDOR_CATEGORY_ORDER if v != "Mixed" and v in vendors_present]

    categories = vendors + ["Mixed"]

    homo_tp, homo_cost = [], []

    for v in vendors:
        v_df = df_valid[(df_valid["prefill_vendor"] == v) & (df_valid["decode_vendor"] == v)]

        # Homogeneous
        homo_df = v_df[v_df["is_homogeneous"]]
        if not homo_df.empty:
            best_idx = homo_df["Normalized_Throughput_per_dollar"].idxmax()
            best_row = homo_df.loc[best_idx]
            homo_tp.append(best_row["Normalized_Throughput_per_dollar"])
            homo_cost.append(best_row["cost_per_hour"])
        else:
            homo_tp.append(np.nan)
            homo_cost.append(np.nan)

    # Mixed (Multi Vendor)
    mixed_df = df_valid[df_valid["is_multi_vendor"]]
    if not mixed_df.empty:
        best_idx = mixed_df["Normalized_Throughput_per_dollar"].idxmax()
        best_row = mixed_df.loc[best_idx]
        # We plot Mixed as a solid bar in the "homogeneous" slot (or centered)
        homo_tp.append(best_row["Normalized_Throughput_per_dollar"])
        homo_cost.append(best_row["cost_per_hour"])
    else:
        homo_tp.append(np.nan)
        homo_cost.append(np.nan)

    x = np.arange(len(categories))
    width = 0.35

    for i, cat in enumerate(categories):
        style = CATEGORY_STYLES.get(cat, CATEGORY_STYLES["Nvidia"])
        c, ec = style["color"], style["edgecolor"]

        # Plot 1: Peak Norm. Tokens/s/$
        if not np.isnan(homo_tp[i]):
            ax1.bar(x[i], homo_tp[i], width if cat != "Mixed" else width * 1.5, color=c, edgecolor=ec, linewidth=1.5)

        # Plot 2: Deployment Cost
        if not np.isnan(homo_cost[i]):
            ax2.bar(x[i], homo_cost[i], width if cat != "Mixed" else width * 1.5, color=c, edgecolor=ec, linewidth=1.5)

    # Find the best single vendor
    max_sv_tp = -1
    sv_x = None
    sv_cost_of_best_tp = None

    n_categories = len(categories)
    mv_x = x[-1]
    mv_tp = homo_tp[-1]  # Mixed is stored in homo_tp
    mv_cost = homo_cost[-1]  # Mixed is stored in homo_cost

    for i in range(n_categories - 1):
        if not np.isnan(homo_tp[i]) and homo_tp[i] > max_sv_tp:
            max_sv_tp = homo_tp[i]
            sv_x = x[i]
            sv_cost_of_best_tp = homo_cost[i]

    # Draw arrow on top plot (Throughput/$)
    if max_sv_tp > 0 and not np.isnan(mv_tp):
        pct_increase = (mv_tp - max_sv_tp) / max_sv_tp * 100
        ax1.annotate(
            "", xy=(mv_x, mv_tp), xytext=(sv_x, max_sv_tp),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", color="black", lw=1.5),
        )
        text_x = (sv_x + mv_x) / 2
        text_y = (max_sv_tp + mv_tp) / 2

        # KEPT FIX (do not revert): the source hardcodes a "+" here
        # (f"+{pct_increase:.1f}%"), so a loss renders as "+-11.1%".
        # Render signed and color by actual direction instead.
        tp_label = f"+{pct_increase:.1f}%" if pct_increase >= 0 else f"{pct_increase:.1f}%"
        ax1.text(
            text_x, text_y, tp_label, ha="center", va="bottom",
            fontsize=font_sizes["legend"] * 0.8,
            color="green" if pct_increase >= 0 else "firebrick",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.8),
        )

    # Draw arrow on bottom plot (Cost reduction)
    if sv_cost_of_best_tp is not None and not np.isnan(mv_cost):
        pct_reduction = (sv_cost_of_best_tp - mv_cost) / sv_cost_of_best_tp * 100
        # Horizontal dashed line from best sv to mv
        ax2.plot([sv_x, mv_x], [sv_cost_of_best_tp, sv_cost_of_best_tp], color="green", linestyle=":", linewidth=1.5)
        # Arrow from best-single-vendor cost to Mixed cost
        ax2.annotate(
            "", xy=(mv_x, mv_cost), xytext=(mv_x, sv_cost_of_best_tp),
            arrowprops=dict(arrowstyle="->", color="green", lw=2),
        )
        # KEPT FIX: as above -- a negative pct_reduction means Mixed costs
        # MORE, so color it as a regression rather than a win.
        cheaper = pct_reduction >= 0
        label = f"-{pct_reduction:.1f}%" if cheaper else f"+{-pct_reduction:.1f}%"
        colour = "green" if cheaper else "firebrick"
        ax2.text(
            mv_x + 0.15, (sv_cost_of_best_tp + mv_cost) / 2, label, va="center", ha="left",
            color=colour, fontsize=font_sizes["legend"] * 0.8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=colour, alpha=0.8),
        )
    elif np.isnan(mv_tp):
        # Defensive, beyond the ported source: label an empty Mixed slot
        # instead of leaving an unexplained gap.
        ax1.text(
            mv_x, 0, "no Mixed\nconfig met SLO", ha="center", va="bottom",
            fontsize=font_sizes["legend"] * 0.45, color="firebrick",
        )

    ax1.set_title(title, fontsize=font_sizes["subplot_title"])

    for ax in [ax1, ax2]:
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=font_sizes["tick_label"], rotation=45, ha="right")
        ax.tick_params(axis="y", labelsize=font_sizes["tick_label"])
        ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.5)


def build_2x2_plots(df_chat: pd.DataFrame, df_codegen: pd.DataFrame, slo_chat: float, slo_codegen: float, font_sizes=None):
    if font_sizes is None:
        font_sizes = FONT_SIZES

    plt.style.use("seaborn-v0_8-paper")
    fig, axes = plt.subplots(2, 2, figsize=(20, 14), constrained_layout=True)

    # Column 0 (left): Chat Conversation; Column 1 (right): Code Gen (High
    # Prefix-KV Reuse) -- matches where plot_bar_results.py's
    # build_2x2_plots actually places each df (axes[:, 0] for df_wo_kv/
    # chat, axes[:, 1] for df_kv/codegen), not its misleading "Column 0" /
    # "Column 1" comments, which are swapped relative to the real axes
    # indices used.
    plot_column(axes[0, 0], axes[1, 0], df_chat, slo_chat, "Chat Conversation", font_sizes)
    plot_column(axes[0, 1], axes[1, 1], df_codegen, slo_codegen, "Code Gen (High Prefix-KV Reuse)", font_sizes)

    axes[0, 0].set_ylabel("Peak Norm. Tokens/s/$", fontsize=font_sizes["axis_label"])
    axes[1, 0].set_ylabel("Deployment Cost ($/hr)", fontsize=font_sizes["axis_label"])

    # Share Y axis across columns for fair comparison
    for row in range(2):
        ylim0 = axes[row, 0].get_ylim()
        ylim1 = axes[row, 1].get_ylim()
        max_y = max(ylim0[1], ylim1[1])
        min_y = min(ylim0[0], ylim1[0])
        axes[row, 0].set_ylim(min_y, max_y)
        axes[row, 1].set_ylim(min_y, max_y)

    # Manual legend handles built (matching the source) but not drawn:
    # plot_bar_results.py's own fig.legend(...) call for these is
    # commented out, so no legend was in the published figure either.
    _solid_patch = mpatches.Patch(facecolor="gray", edgecolor="black", label="Single Vendor (Homogeneous)")
    _mixed_patch = mpatches.Patch(
        facecolor=CATEGORY_STYLES["Mixed"]["color"], edgecolor=CATEGORY_STYLES["Mixed"]["edgecolor"],
        label="Mixed (Multi-Vendor)",
    )

    return fig


def plot_fig8() -> None:
    df_chat = load_and_process("chat")
    df_codegen = load_and_process("codegen")
    print(f"[fig8] chat: {len(df_chat)} rows, codegen: {len(df_codegen)} rows")
    fig = build_2x2_plots(
        df_chat, df_codegen,
        slo_chat=USE_CASES["chat"]["ttft_slo_ms"], slo_codegen=USE_CASES["codegen"]["ttft_slo_ms"],
    )
    save_figure(fig, "Combined_2x2_bar_results.pdf")
    plt.close(fig)


# ─────────────────────────── CLI ──────────────────────────────────────────


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--use-case", choices=["chat", "codegen", "all"], default="all",
        help="Plot Fig. 7 for just one use case, or 'all' (default) for both plus Fig. 8.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    use_cases = list(USE_CASES) if args.use_case == "all" else [args.use_case]
    try:
        for use_case in use_cases:
            plot_fig7(use_case)
        if args.use_case == "all":
            plot_fig8()
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
