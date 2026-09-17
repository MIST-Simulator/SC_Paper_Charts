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

Faithfully ports ``GenA_Paper_charts/SC26/plot_sc_results.py`` (Fig. 7:
``load_and_process`` + ``build_side_by_side_plot``) and
``GenA_Paper_charts/SC26/plot_bar_results.py`` (Fig. 8: ``plot_column`` +
``build_2x2_plots``) -- the two scripts confirmed to have produced the
camera-ready figures. Adapted in one place: this reads run_T3.py's clean
``Hardware``/``Parallelism`` columns directly instead of regex-parsing the
source's concatenated "UseCase"/"Serving Name" strings; every plotted
number otherwise matches the source formula exactly (see inline comments
citing the source function each block ports).

Fig. 7's scatter and Fig. 8's bar chart use two different, inconsistent
"Mixed" definitions inherited from the source (broad `vendor_of_config` vs.
narrow `is_multi_vendor_config`) -- preserved faithfully rather than
reconciled; see docs/FINDINGS.md#t3.
"""

import argparse
import ast
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg", force=True)  # headless: this repo's scripts never open a GUI window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from mist_charts.paths import resolve_results
from mist_charts.pricing import (
    PERFORMANCE_SCALE_FACTOR,
    SKU_DISPLAY_NAMES,
    VENDOR_CATEGORY_ORDER,
    VENDOR_COLORS,
    VENDOR_OF_SKU,
    is_multi_vendor_config,
    parse_hardware,
    vendor_of_config,
)
from mist_charts.style import save_figure

# ─────────────────────── Font and style (ported from plot_sc_results.py /
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
# Other never appear). Dict order also fixes scatter draw order and legend
# order (Mixed, Etched, TPU, AMD, Nvidia), matching the paper figure.
CATEGORY_STYLES = {
    "Mixed": {"color": "#1f77b4", "marker": "s", "edgecolor": "navy"},
    "Etched": {"color": "#ff7f0e", "marker": "o", "edgecolor": "#a65100"},
    "TPU": {"color": "#9467bd", "marker": "o", "edgecolor": "#5e3c81"},
    "AMD": {"color": "#17becf", "marker": "o", "edgecolor": "#0e7a85"},
    "Nvidia": {"color": "#2ca02c", "marker": "o", "edgecolor": "darkgreen"},
}

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


# ─────────────────────────── Data loading (port of plot_sc_results.py's
# load_and_process) ───────────────────────────────────────────────────────


def load_and_process(use_case: str) -> pd.DataFrame:
    cfg = USE_CASES[use_case]
    path = resolve_results("T3", cfg["results_file"])
    df = pd.read_csv(path)
    df = df.dropna(subset=["output_throughput", "TTFT_p99", "tokens_per_dollar", "Cost"])
    df = df[(df["output_throughput"] > 0) & (df["TTFT_p99"] > 0) & (df["Cost"] > 0)]
    if df.empty:
        raise SystemExit(f"No usable rows in {path} (every config failed or was ignored).")

    df["is_disaggregated"] = df["Batching_Strategy"] == "DISAGGREGATED"

    hw_info = [parse_hardware(hw, disagg) for hw, disagg in zip(df["Hardware"], df["is_disaggregated"])]
    df["prefill_hw"] = [h[0] for h in hw_info]
    df["decode_hw"] = [h[1] for h in hw_info]
    df["is_hetero"] = [h[2] for h in hw_info]
    df["is_homogeneous"] = ~df["is_hetero"]

    # cost_per_hour: run_T3.py already computes this at simulation time
    # (Cost column, via deployment_space.get_search_space's get_price),
    # using the exact same formula as plot_sc_results.py's compute_cost
    # (prefill_price*prefill_nodes + decode_price*decode_nodes for
    # disaggregated, price*total_nodes otherwise) and the same price table
    # (mist_charts.pricing.PRICE_PER_HOUR == DEFAULT_PRICES for our 8
    # SKUs) -- no need to recompute it here.
    df["cost_per_hour"] = df["Cost"]

    # TTFT_P99: run_T3.py already computes this at simulation time from the
    # merged ongoing+completed TTFT lists (port of compute_ttft_percentiles
    # / _flatten) -- no CSV string round-trip needed.
    df["TTFT_P99"] = df["TTFT_p99"]

    # Output_Throughput_Scaled = output_throughput * PERFORMANCE_SCALE_FACTOR[decode_hw]
    df["Output_Throughput_Scaled"] = df["output_throughput"] * df["decode_hw"].map(PERFORMANCE_SCALE_FACTOR)

    # Normalize by the MINIMUM (source variable is misnamed "max_throughput"
    # but the behavior -- and the caption -- is "normalized against the
    # lowest-throughput configuration").
    min_throughput = df["Output_Throughput_Scaled"].min()
    df["Normalized_Throughput"] = df["Output_Throughput_Scaled"] / min_throughput

    df["Throughput_per_dollar"] = df["Output_Throughput_Scaled"] / df["cost_per_hour"]
    min_throughput_per_dollar = df["Throughput_per_dollar"].min()
    df["Normalized_Throughput_per_dollar"] = df["Throughput_per_dollar"] / min_throughput_per_dollar

    df["prefill_vendor"] = df["prefill_hw"].map(VENDOR_OF_SKU)
    df["decode_vendor"] = df["decode_hw"].map(VENDOR_OF_SKU)
    df["is_multi_vendor"] = [
        is_multi_vendor_config(hw, disagg) for hw, disagg in zip(df["Hardware"], df["is_disaggregated"])
    ]
    # Broad "Mixed" category (assign_category): Mixed unless prefill_hw ==
    # decode_hw exactly. Used by Fig. 7's scatter only.
    df["plot_category"] = [
        vendor_of_config(hw, disagg) for hw, disagg in zip(df["Hardware"], df["is_disaggregated"])
    ]

    return df.reset_index(drop=True)


# ─────────────────────────── Pareto front (verbatim port) ────────────────


def compute_pareto_front(x_vals, y_vals) -> List[int]:
    """Pareto front: minimize x, maximize y. Returns indices sorted by x."""
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


# ─────────────────────────── Fig. 7: side-by-side scatter (port of
# plot_sc_results.py's build_side_by_side_plot) ───────────────────────────


def _config_annotation_text(row) -> str:
    """Port of build_side_by_side_plot's per-star annotation text (the
    hw_text / pd_tp_text / ann_text block). Uses our own Parallelism column
    for prefill/decode TP/DP directly instead of the source's brute-force
    "search for TP values whose device counts sum to 8" loop -- same text,
    computed from data we already have rather than re-derived by guessing.
    """
    prefill_name = SKU_DISPLAY_NAMES.get(row["prefill_hw"], row["prefill_hw"])
    if row["is_hetero"]:
        hw_text = f"{prefill_name}:{SKU_DISPLAY_NAMES.get(row['decode_hw'], row['decode_hw'])}"
    else:
        hw_text = prefill_name

    if row["is_disaggregated"]:
        prefill_p, decode_p = row["Parallelism"].split("-", 1)
        prefill_p = ast.literal_eval(prefill_p)
        decode_p = ast.literal_eval(decode_p)
        p_inst, d_inst = prefill_p["DP"], decode_p["DP"]
        p_tp, d_tp = prefill_p["TP"], decode_p["TP"]
        pd_tp_text = f"{p_inst}P:{d_inst}D, TP: (P={p_tp}, D={d_tp})"
    else:
        parallelism = row["Parallelism"]
        p = ast.literal_eval(parallelism) if isinstance(parallelism, str) else parallelism
        total_nodes = p["DP"]
        # NOTE: ported verbatim from the source, including its label -- for
        # non-disaggregated rows the source annotates "TP=<node count>",
        # which is actually the client/DP count, not the TP degree. This
        # looks like a labeling quirk in the ported source; kept as-is for
        # faithfulness rather than silently "corrected".
        pd_tp_text = f"TP={total_nodes}"

    return f"{hw_text}\n{pd_tp_text}"


def build_side_by_side_plot(
    df: pd.DataFrame,
    slo_ms: float,
    legend: bool,
    rank1_offset: Tuple[int, int],
    rank2_offset: Tuple[int, int],
    font_sizes: Optional[Dict] = None,
):
    """Port of plot_sc_results.py's build_side_by_side_plot, verbatim other
    than reading our own dataframe columns (see module docstring)."""
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

    # Two best points: highest Throughput_per_dollar with TTFT_P99 < slo_ms.
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

        for rank, b_row in enumerate(best_rows, start=1):
            ax.scatter(
                b_row[x_metric[0]], b_row[y_col],
                marker="*", s=900, color="gold", edgecolor="black",
                linewidths=2, zorder=10,
            )
            if idx == 1:
                ann_text = f"#{rank}: {_config_annotation_text(b_row)}"
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

        for cat_name, style in CATEGORY_STYLES.items():
            cat_df = df[df["plot_category"] == cat_name]
            if len(cat_df) > 0:
                ax.scatter(
                    cat_df[x_metric[0]], cat_df[y_col],
                    c=style["color"], marker=style["marker"], s=marker_size,
                    alpha=marker_alpha, edgecolors=style["edgecolor"], linewidths=1.5,
                    label=cat_name, zorder=3,
                )

        homogeneous_df = df[df["is_homogeneous"]]
        heterogeneous_df = df[~df["is_homogeneous"]]
        for grp_df, line_color, line_style, _line_label in [
            (homogeneous_df, "#d62728", "-", "Homogenous Deployement"),
            (heterogeneous_df, "#1f77b4", "--", "Heterogenous Deployement"),
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
            px, py = xg[orig_g[pidx]], yg[orig_g[pidx]]
            sort_order = np.argsort(px)
            px, py = px[sort_order], py[sort_order]
            ax.plot(px, py, color=line_color, linestyle=line_style, linewidth=2.5, label="_nolegend_", zorder=2)

        ax.set_xscale("log")
        ax.axvspan(0, slo_ms, color="#d4edda", alpha=0.4, zorder=0)
        ax.set_xlabel(x_metric[1], fontsize=font_sizes["axis_label"])
        ax.set_ylabel(y_lbl, fontsize=font_sizes["axis_label"])
        ax.tick_params(axis="both", labelsize=font_sizes["tick_label"])
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    leg = fig.legend(
        unique_labels.values(), unique_labels.keys(), loc="lower center", bbox_to_anchor=(0.47, 0.94),
        ncol=len(unique_labels), fontsize=font_sizes["legend"], frameon=True, framealpha=0.3,
        columnspacing=0.8, handletextpad=0, labelspacing=0, handlelength=1.2,
    )
    if not legend:
        for t in leg.get_texts():
            t.set_alpha(0)
        for h in (leg.legendHandles if hasattr(leg, "legendHandles") else leg.legend_handles):
            if hasattr(h, "set_alpha"):
                h.set_alpha(0)
        leg.get_frame().set_alpha(0)
        if hasattr(leg.get_frame(), "set_linewidth"):
            leg.get_frame().set_linewidth(0)

    if not mask.any():
        # Defensive, graceful degradation beyond what the ported source
        # does (it just silently omits stars): make the "nobody met the
        # SLO" state explicit on the figure itself, not just in a console
        # warning, without fabricating a winner.
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


# ─────────────────────────── Fig. 8: 2x2 bars (port of
# plot_bar_results.py's plot_column + build_2x2_plots) ────────────────────


def plot_column(ax1, ax2, df: pd.DataFrame, slo_ms: float, title: str, font_sizes: Dict) -> None:
    df_valid = df[df["TTFT_P99"] < slo_ms]
    if df_valid.empty:
        print(f"Warning: No data points meet the SLO of {slo_ms}ms for {title}.")

    vendors_present = (
        df_valid[~df_valid["is_multi_vendor"]]["prefill_vendor"].unique() if not df_valid.empty else []
    )
    vendors = [v for v in VENDOR_CATEGORY_ORDER if v != "Mixed" and v in vendors_present]
    categories = vendors + ["Mixed"]

    homo_tp: List[float] = []
    homo_cost: List[float] = []

    for v in vendors:
        v_df = df_valid[(df_valid["prefill_vendor"] == v) & (df_valid["decode_vendor"] == v)]
        homo_df = v_df[v_df["is_homogeneous"]]
        if not homo_df.empty:
            best_row = homo_df.loc[homo_df["Normalized_Throughput_per_dollar"].idxmax()]
            homo_tp.append(best_row["Normalized_Throughput_per_dollar"])
            homo_cost.append(best_row["cost_per_hour"])
        else:
            homo_tp.append(np.nan)
            homo_cost.append(np.nan)

    # Mixed bar: narrow definition (is_multi_vendor), NOT the broad
    # plot_category used by Fig. 7's scatter -- see module docstring WARNING.
    mixed_df = df_valid[df_valid["is_multi_vendor"]]
    if not mixed_df.empty:
        best_row = mixed_df.loc[mixed_df["Normalized_Throughput_per_dollar"].idxmax()]
        homo_tp.append(best_row["Normalized_Throughput_per_dollar"])
        homo_cost.append(best_row["cost_per_hour"])
    else:
        homo_tp.append(np.nan)
        homo_cost.append(np.nan)

    x = np.arange(len(categories))
    width = 0.35

    for i, cat in enumerate(categories):
        style_key = cat if cat in ("Mixed", "Etched", "TPU", "AMD", "Nvidia") else "Nvidia"
        c = VENDOR_COLORS.get(style_key, "#7f7f7f")
        ec = {"Mixed": "navy", "Etched": "#a65100", "TPU": "#5e3c81", "AMD": "#0e7a85", "Nvidia": "darkgreen"}.get(
            style_key, "#333333"
        )
        if not np.isnan(homo_tp[i]):
            ax1.bar(x[i], homo_tp[i], width if cat != "Mixed" else width * 1.5, color=c, edgecolor=ec, linewidth=1.5)
        if not np.isnan(homo_cost[i]):
            ax2.bar(x[i], homo_cost[i], width if cat != "Mixed" else width * 1.5, color=c, edgecolor=ec, linewidth=1.5)

    max_sv_tp = -1.0
    sv_x = None
    sv_cost_of_best_tp = None
    n_categories = len(categories)
    mv_x = x[-1]
    mv_tp = homo_tp[-1]
    mv_cost = homo_cost[-1]

    for i in range(n_categories - 1):
        if not np.isnan(homo_tp[i]) and homo_tp[i] > max_sv_tp:
            max_sv_tp = homo_tp[i]
            sv_x = x[i]
            sv_cost_of_best_tp = homo_cost[i]

    if max_sv_tp > 0 and not np.isnan(mv_tp):
        pct_increase = (mv_tp - max_sv_tp) / max_sv_tp * 100
        ax1.annotate(
            "", xy=(mv_x, mv_tp), xytext=(sv_x, max_sv_tp),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", color="black", lw=1.5),
        )
        text_x, text_y = (sv_x + mv_x) / 2, (max_sv_tp + mv_tp) / 2
        # Mixed does not always beat the best single vendor -- colour and sign
        # the label by the actual direction rather than assuming a gain
        # (a hardcoded "+" previously rendered a loss as "+-11.1%").
        tp_label = f"+{pct_increase:.1f}%" if pct_increase >= 0 else f"{pct_increase:.1f}%"
        ax1.text(
            text_x, text_y, tp_label, ha="center", va="bottom",
            fontsize=font_sizes["legend"] * 0.8,
            color="green" if pct_increase >= 0 else "firebrick",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.8),
        )

    if sv_cost_of_best_tp is not None and not np.isnan(mv_cost):
        pct_reduction = (sv_cost_of_best_tp - mv_cost) / sv_cost_of_best_tp * 100
        ax2.plot([sv_x, mv_x], [sv_cost_of_best_tp, sv_cost_of_best_tp], color="green", linestyle=":", linewidth=1.5)
        ax2.annotate(
            "", xy=(mv_x, mv_cost), xytext=(mv_x, sv_cost_of_best_tp),
            arrowprops=dict(arrowstyle="->", color="green", lw=2),
        )
        # A positive pct_reduction is a genuine saving; a negative one means
        # Mixed costs MORE, so colour it as a regression rather than a win.
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
    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=font_sizes["tick_label"], rotation=45, ha="right")
        ax.tick_params(axis="y", labelsize=font_sizes["tick_label"])
        ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.5)


def build_2x2_plots(df_chat: pd.DataFrame, df_codegen: pd.DataFrame, slo_chat: float, slo_codegen: float, font_sizes=None):
    if font_sizes is None:
        font_sizes = FONT_SIZES

    plt.style.use("seaborn-v0_8-paper")
    fig, axes = plt.subplots(2, 2, figsize=(20, 14), constrained_layout=True)

    # Column 0: Chat Conversation; Column 1: Code Gen (High Prefix-KV Reuse)
    # -- matches plot_bar_results.py's build_2x2_plots column assignment.
    plot_column(axes[0, 0], axes[1, 0], df_chat, slo_chat, "Chat Conversation", font_sizes)
    plot_column(axes[0, 1], axes[1, 1], df_codegen, slo_codegen, "Code Gen (High Prefix-KV Reuse)", font_sizes)

    axes[0, 0].set_ylabel("Peak Norm. Tokens/s/$", fontsize=font_sizes["axis_label"])
    axes[1, 0].set_ylabel("Deployment Cost ($/hr)", fontsize=font_sizes["axis_label"])

    for row in range(2):
        ylim0 = axes[row, 0].get_ylim()
        ylim1 = axes[row, 1].get_ylim()
        max_y = max(ylim0[1], ylim1[1])
        min_y = min(ylim0[0], ylim1[0])
        axes[row, 0].set_ylim(min_y, max_y)
        axes[row, 1].set_ylim(min_y, max_y)

    # Legend intentionally not drawn: plot_bar_results.py builds
    # solid_patch/mixed_patch handles but its fig.legend(...) call is
    # commented out in the source, so no legend was in the published
    # figure either. Preserved faithfully.

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
