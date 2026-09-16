#!/usr/bin/env python3
"""Plotting for Figure 6 (a)-(b) of the MIST SC26 paper: individual-step
validation.

  (a) figures/individual_step_validation.pdf -- boxplots of MIST (M) and
      Vidur (V) per-step forward-pass runtime error against real vLLM
      measurements, for Llama-2-70B on H100 at TP4/TP8, split by
      prefill/decode/chunked-prefill stage.
  (b) figures/Retrieval_latency_validation.pdf -- MIST-modelled vs
      fio-measured KV cache retrieval latency for NVMe SSD and DDR4,
      sequential reads, block sizes 256 KB - 1 GB.

Reads the CSVs written by run_T2.py (falling back to results/reference/T2/
when results/T2/ is absent) plus the vendored README under
data/validation/T2/. Does not import MIST / MIST -- run standalone.
"""

import sys

import matplotlib.pyplot as plt
import pandas as pd

from mist_charts import SIMULATOR_COLORS, apply_paper_style, save_figure
from mist_charts.paths import resolve_results

STAGES = ["prefill", "decode", "chunked"]
STAGE_TITLES = {"prefill": "Prefill", "decode": "Decode", "chunked": "Chunked"}

MIST_COLOR = SIMULATOR_COLORS["MIST (Simulated)"]
VIDUR_COLOR = SIMULATOR_COLORS["Vidur (Simulated)"]
REAL_COLOR = SIMULATOR_COLORS["vLLM (Real)"]

FONT_DICT = {"xlabel": 24, "ylabel": 24, "tick": 24, "legend": 24, "title": 24}


def plot_part_a() -> None:
    csv_path = resolve_results("T2", "step_runtime_validation.csv")
    df = pd.read_csv(csv_path)
    print(f"read {csv_path}  ({len(df)} rows)")

    hardware_order = list(dict.fromkeys(df["hardware"]))  # first-seen order, no dupes

    fig, axes = plt.subplots(len(hardware_order), len(STAGES), figsize=(8, 8))
    if len(hardware_order) == 1:
        axes = axes.reshape(1, -1)

    for i, hardware in enumerate(hardware_order):
        for j, stage in enumerate(STAGES):
            ax = axes[i, j]
            cell = df[(df["hardware"] == hardware) & (df["stage"] == stage)]
            mist_err = cell["mist_error_pct"].to_numpy()
            vidur_err = cell["vidur_error_pct"].to_numpy()

            bp = ax.boxplot(
                [mist_err, vidur_err],
                tick_labels=["M", "V"],
                patch_artist=True,
                sym="",
                widths=0.6,
            )
            for patch, color in zip(bp["boxes"], [MIST_COLOR, VIDUR_COLOR]):
                patch.set_facecolor(color)
                patch.set_alpha(0.8)

            if i == 0:
                ax.text(
                    0.5, 1.02, STAGE_TITLES[stage], transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=FONT_DICT["title"],
                )
            if j == len(STAGES) - 1:
                # Wrap "<model> H100xTP<n>" onto two lines (matching the
                # published figure's two-line row labels) so the rotated
                # text's footprint fits within this row and doesn't collide
                # with the row above/below.
                label_text = hardware.replace(" H100x", "\nH100x")
                ax.text(
                    1.05, 0.5, label_text, transform=ax.transAxes,
                    ha="center", va="center", fontsize=FONT_DICT["ylabel"],
                    fontweight="bold", rotation=-90,
                )
            if j == 0:
                ax.set_ylabel("Error (%)", fontsize=FONT_DICT["ylabel"])
            ax.tick_params(axis="both", labelsize=FONT_DICT["tick"])
            if i != len(hardware_order) - 1:
                # Only the bottom row carries the M/V x tick labels, matching
                # the published figure -- the top row would otherwise repeat
                # them on every subplot.
                ax.set_xticklabels([])
            ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    save_figure(fig, "individual_step_validation.pdf")
    plt.close(fig)


def plot_part_b() -> None:
    csv_path = resolve_results("T2", "kv_retrieval_validation.csv")
    df = pd.read_csv(csv_path)
    print(f"read {csv_path}  ({len(df)} rows)")

    fig, ax = plt.subplots(figsize=(8, 8))

    markers = {"DDR4": "o", "NVMe SSD": "d"}
    real_colors = {"DDR4": REAL_COLOR, "NVMe SSD": "#374151"}  # blue / slate gray
    for device, marker in markers.items():
        device_df = df[df["device"] == device].sort_values("num_tokens")
        ax.plot(
            device_df["num_tokens"], device_df["mist_latency_ms"],
            label=f"MIST Predicted {device} Latency", marker=marker,
            linestyle="--", color=MIST_COLOR,
        )
        ax.plot(
            device_df["num_tokens"], device_df["measured_latency_ms"],
            label=f"Real {device} Latency", marker=marker,
            color=real_colors[device],
        )

    ax.set_yscale("log")
    ax.set_xlim(0, 8000)
    ax.set_ylim(0.5e-2, 300)
    ax.tick_params(axis="both", labelsize=16)
    ax.set_xlabel("Num Tokens", fontsize=18)
    ax.set_ylabel("Retrieval Latency (ms)", fontsize=18)
    ax.grid(True, which="both", ls="--")
    ax.legend(ncols=1, loc="lower right", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "Retrieval_latency_validation.pdf")
    plt.close(fig)


def main() -> int:
    apply_paper_style()
    plot_part_a()
    plot_part_b()
    return 0


if __name__ == "__main__":
    sys.exit(main())
