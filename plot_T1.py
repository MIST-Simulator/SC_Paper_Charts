#!/usr/bin/env python3
"""Reproduces MIST SC26 Fig. 4(a-d): end-to-end latency validation.

Reads the MIST results written by run_T1.py (falling back to the shipped
reference CSVs in results/reference/T1/ if run_T1.py hasn't been run) and
the vendored real-vLLM / Vidur / LLMServingSim2.0 baselines in
data/validation/T1/, and reproduces the published 2x2 figure: three panels
of sorted end-to-end latency (Qwen3-32B on L40S:TP2, Llama3-70B on
H100:TP8, Llama-3.1-8B on TPUv6e) plus a simulation wall-clock-time bar
chart. Ported from the final figure cell (cell 32) of the source notebook,
GenA_Paper_charts/Validation/4.SC26/SimulatorComparision/
share_gpt_vllm_tpu_validation.ipynb.

This script never imports MIST: it only reads CSVs/JSON already on disk.
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mist_charts import SIMULATOR_COLORS, apply_paper_style, save_figure
from mist_charts.paths import VALIDATION_DIR, resolve_results

FONT_SIZES = {
    "title": 18,
    "labels": 18,
    "ticks": 16,
    "legend": 20,
    "text": 16,
}

LABEL_VLLM = "vLLM (Real)"
LABEL_MIST = "MIST (Simulated)"
LABEL_VIDUR = "Vidur (Simulated)"
LABEL_LLMSIM = "LLMServingSim2.0 (Simulated)"

# Simulation wall-clock time (seconds) for the two third-party simulators,
# taken verbatim from the source notebook's cell 31. These are NOT
# reproduced by this repository (that would require installing and running
# Vidur / LLMServingSim2.0 themselves); the MIST bars, by contrast, are
# read from run_T1.py's own measured `sim_wall_clock_sec` column.
#
# Source: GenA_Paper_charts/Validation/4.SC26/SimulatorComparision/
#         LLM_Serving_Sim_Results/logs/{tpu,l40s_tp2,h100_tp8}_output.txt
LLM_SERVING_SIM_RUNTIMES = {"TPUv6e": 33.793, "L40S:TP2": 144.798, "H100:TP8": 104.49}
# From the LLMServingSim2.0 paper; 0 => 'X' (TPU unsupported by Vidur).
VIDUR_SERVING_SIM_RUNTIMES = {"TPUv6e": 0, "L40S:TP2": 485, "H100:TP8": 495}


def _load_mist(filename: str) -> pd.DataFrame:
    path = resolve_results("T1", filename)
    return pd.read_csv(path)


def _load_llmservingsim_e2e_sec(filename: str) -> np.ndarray:
    """`latency` column is nanoseconds (see data/validation/T1/README.md)."""
    df = pd.read_csv(VALIDATION_DIR / "T1" / filename)
    return np.sort(df["latency"].to_numpy(dtype=float) / 1e9)


def _load_vidur_e2e_sec(filename: str) -> np.ndarray:
    df = pd.read_csv(VALIDATION_DIR / "T1" / filename)
    return np.sort(df["request_e2e_time"].to_numpy(dtype=float))


def _load_vllm_e2e_sec(filename: str, field_candidates=("e2els", "e2el")) -> np.ndarray:
    """vLLM benchmark_serving.py JSON: field is `e2els` or `e2el` depending
    on the benchmark script version; both hold one list of per-request
    end-to-end latencies in seconds."""
    with open(VALIDATION_DIR / "T1" / filename, "r") as f:
        data = json.load(f)
    df = pd.json_normalize(data)
    for field in field_candidates:
        if field in df.columns:
            return np.sort(np.array(df[field][0], dtype=float))
    raise KeyError(f"none of {field_candidates} found in {filename}")


def _load_vllm_tpu_e2e_sec(filename: str) -> np.ndarray:
    """TPUv6e metrics file has no e2el field; recompute from raw timestamps,
    exactly as the source notebook does (cell 11)."""
    with open(VALIDATION_DIR / "T1" / filename, "r") as f:
        data = json.load(f)
    latencies = [
        m["last_token_time"] - m["sent_time"] for m in data.get("metrics", [])
    ]
    return np.sort(np.array(latencies, dtype=float))


def build_figure(
    l40s_mist: pd.DataFrame,
    h100_mist: pd.DataFrame,
    tpu_mist: pd.DataFrame,
    l40s_vllm: np.ndarray,
    l40s_vidur: np.ndarray,
    l40s_llmsim: np.ndarray,
    h100_vllm: np.ndarray,
    h100_vidur: np.ndarray,
    h100_llmsim: np.ndarray,
    tpu_vllm: np.ndarray,
    tpu_llmsim: np.ndarray,
    mist_runtimes: dict,
):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), facecolor="white")
    ax_flat = axes.flatten()

    # ---------------- Panel (a): Qwen3-32B on L40S:TP2 ----------------
    mist_qwen_e2el = np.sort(l40s_mist["mist_e2e_sec"].to_numpy())
    ax_flat[0].plot(
        range(len(l40s_vllm)), l40s_vllm,
        label=LABEL_VLLM, linewidth=3, color=SIMULATOR_COLORS[LABEL_VLLM],
    )
    ax_flat[0].plot(
        range(len(mist_qwen_e2el)), mist_qwen_e2el, linestyle="--",
        label=LABEL_MIST, alpha=0.7, linewidth=3, color=SIMULATOR_COLORS[LABEL_MIST],
    )
    ax_flat[0].plot(
        range(len(l40s_vidur)), l40s_vidur, linestyle="--",
        label=LABEL_VIDUR, alpha=0.7, linewidth=3, color=SIMULATOR_COLORS[LABEL_VIDUR],
    )
    ax_flat[0].plot(
        range(len(l40s_llmsim)), l40s_llmsim, linestyle="--",
        label=LABEL_LLMSIM, alpha=0.7, linewidth=3, color=SIMULATOR_COLORS[LABEL_LLMSIM],
    )
    ax_flat[0].text(
        0.04, 0.94, "Qwen3-32B on\nL40S:TP2", transform=ax_flat[0].transAxes,
        fontsize=FONT_SIZES["title"], fontweight="bold", va="top", ha="left",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=3),
    )
    # Note: the camera-ready Fig. 4(a) mislabels this axis "Latency (ms)" --
    # the plotted quantity (mist_e2e_sec, vLLM e2els, Vidur request_e2e_time,
    # LLMServingSim latency/1e9) is seconds. This reproduction uses the
    # correct unit. See data/validation/T1/README.md.
    ax_flat[0].set_ylabel("Latency (sec)", fontsize=FONT_SIZES["labels"])
    ax_flat[0].set_xlabel("Request Idx", fontsize=FONT_SIZES["labels"])

    # ---------------- Panel (b): Llama3-70B on H100:TP8 ----------------
    mist_e2el = np.sort(h100_mist["mist_e2e_sec"].to_numpy())
    ax_flat[1].plot(
        range(len(h100_vllm)), h100_vllm,
        label=LABEL_VLLM, linewidth=3, color=SIMULATOR_COLORS[LABEL_VLLM],
    )
    ax_flat[1].plot(
        range(len(mist_e2el)), mist_e2el, linestyle="--",
        label=LABEL_MIST, alpha=0.7, linewidth=3, color=SIMULATOR_COLORS[LABEL_MIST],
    )
    ax_flat[1].plot(
        range(len(h100_vidur)), h100_vidur, linestyle="--",
        label=LABEL_VIDUR, alpha=0.7, linewidth=3, color=SIMULATOR_COLORS[LABEL_VIDUR],
    )
    ax_flat[1].plot(
        range(len(h100_llmsim)), h100_llmsim, linestyle="--",
        label=LABEL_LLMSIM, alpha=0.7, linewidth=3, color=SIMULATOR_COLORS[LABEL_LLMSIM],
    )
    ax_flat[1].text(
        0.04, 0.94, "Llama3-70B on\nH100:TP8", transform=ax_flat[1].transAxes,
        fontsize=FONT_SIZES["title"], fontweight="bold", va="top", ha="left",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=3),
    )
    # Same mislabel-fix as panel (a) -- see the note above.
    ax_flat[1].set_ylabel("Latency (sec)", fontsize=FONT_SIZES["labels"])
    ax_flat[1].set_xlabel("Request Idx", fontsize=FONT_SIZES["labels"])

    # ---------------- Panel (c): Llama-3.1-8B on TPUv6e ----------------
    mist_e2el_tpu = np.sort(tpu_mist["mist_e2e_sec"].to_numpy())
    ax_flat[2].plot(
        range(len(tpu_vllm)), tpu_vllm,
        label=LABEL_VLLM, linewidth=3, color=SIMULATOR_COLORS[LABEL_VLLM],
    )
    ax_flat[2].plot(
        range(len(mist_e2el_tpu)), mist_e2el_tpu, linestyle="--",
        label=LABEL_MIST, alpha=0.7, linewidth=3, color=SIMULATOR_COLORS[LABEL_MIST],
    )
    ax_flat[2].plot(
        range(len(tpu_llmsim)), tpu_llmsim, linestyle="--",
        label=LABEL_LLMSIM, alpha=0.8, linewidth=3, color=SIMULATOR_COLORS[LABEL_LLMSIM],
    )
    ax_flat[2].text(
        0.04, 0.94, "Llama-3.1-8B on\nTPUv6e", transform=ax_flat[2].transAxes,
        fontsize=FONT_SIZES["title"], fontweight="bold", va="top", ha="left",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=3),
    )
    ax_flat[2].set_ylabel("Latency (sec)", fontsize=FONT_SIZES["labels"])
    ax_flat[2].set_xlabel("Request Idx", fontsize=FONT_SIZES["labels"])

    for i in range(3):
        ax_flat[i].tick_params(labelsize=FONT_SIZES["ticks"])
        ax_flat[i].grid(True, linestyle="--", alpha=0.6)

    # ---------------- Panel (d): Simulation Time bar chart ----------------
    hardware = ["L40S:TP2", "H100:TP8", "TPUv6e"]
    simulators = [
        (LABEL_VIDUR, VIDUR_SERVING_SIM_RUNTIMES),
        (LABEL_LLMSIM, LLM_SERVING_SIM_RUNTIMES),
    ]
    if mist_runtimes:
        simulators.insert(2, (LABEL_MIST, mist_runtimes))

    x = np.arange(len(hardware))
    width = 0.25
    for i, (name, data) in enumerate(simulators):
        values = [data.get(hw, 0) for hw in hardware]
        offset = (i - (len(simulators) - 1) / 2) * (width + 0.05)
        ax_flat[3].bar(
            x + offset, values, width, label=name, color=SIMULATOR_COLORS[name],
            alpha=0.9, edgecolor="black", linewidth=0.5,
        )
        for j, val in enumerate(values):
            if val == 0:
                ax_flat[3].text(
                    x[j] + offset, 5, "X", ha="center", va="bottom",
                    color="red", fontweight="bold", fontsize=FONT_SIZES["text"],
                )
            else:
                ax_flat[3].text(
                    x[j] + offset, val + 5, f"{val:.1f}", ha="center", va="bottom",
                    fontsize=FONT_SIZES["text"],
                )

    ax_flat[3].text(
        0.96, 0.94, "Simulation\nTime", transform=ax_flat[3].transAxes,
        fontsize=FONT_SIZES["title"], fontweight="bold", va="top", ha="right",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=3),
    )
    ax_flat[3].set_ylabel("Runtime (sec)", fontsize=FONT_SIZES["labels"])
    ax_flat[3].set_xticks(x)
    ax_flat[3].set_xticklabels(hardware, fontweight="bold", fontsize=FONT_SIZES["ticks"])
    ax_flat[3].tick_params(labelsize=FONT_SIZES["ticks"])
    ax_flat[3].grid(axis="y", linestyle="--", alpha=0.4)
    ax_flat[3].spines["top"].set_visible(False)
    ax_flat[3].spines["right"].set_visible(False)

    # ---------------- Panel labels (a)-(d) ----------------
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    for i, ax in enumerate(ax_flat):
        ax.text(
            0.5, -0.28, panel_labels[i], transform=ax.transAxes,
            fontsize=FONT_SIZES["legend"], fontweight="bold", ha="center", va="top",
        )

    # ---------------- Unified legend ----------------
    handles, labels = [], []
    for ax in ax_flat:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)

    fig.legend(
        handles, labels,
        fontsize=FONT_SIZES["legend"],
        loc="upper center",
        bbox_to_anchor=(0.55, 1.08),
        ncol=2,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.5,
    )

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    return fig


def main() -> int:
    apply_paper_style()

    try:
        l40s_mist = _load_mist("mist_l40s.csv")
        h100_mist = _load_mist("mist_h100.csv")
        tpu_mist = _load_mist("mist_tpu.csv")
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    l40s_vllm = _load_vllm_e2e_sec("vllm_l40s_qwen3-32b_tp2.json")
    l40s_vidur = _load_vidur_e2e_sec("vidur_l40s_qwen3-32b_tp2_request_metrics.csv")
    l40s_llmsim = _load_llmservingsim_e2e_sec("llmservingsim_l40s_tp2_result.csv")

    h100_vllm = _load_vllm_e2e_sec("vllm_h100_llama3-70b_tp8.json")
    h100_vidur = _load_vidur_e2e_sec("vidur_h100_llama3-70b_tp8_request_metrics.csv")
    h100_llmsim = _load_llmservingsim_e2e_sec("llmservingsim_h100_tp8_result.csv")

    tpu_vllm = _load_vllm_tpu_e2e_sec("vllm_tpu_llama31-8b.json")
    tpu_llmsim = _load_llmservingsim_e2e_sec("llmservingsim_tpu_result.csv")

    mist_runtimes = {
        "L40S:TP2": l40s_mist["sim_wall_clock_sec"].iloc[0],
        "H100:TP8": h100_mist["sim_wall_clock_sec"].iloc[0],
        "TPUv6e": tpu_mist["sim_wall_clock_sec"].iloc[0],
    }

    fig = build_figure(
        l40s_mist, h100_mist, tpu_mist,
        l40s_vllm, l40s_vidur, l40s_llmsim,
        h100_vllm, h100_vidur, h100_llmsim,
        tpu_vllm, tpu_llmsim,
        mist_runtimes,
    )
    save_figure(fig, "vllm_simulator_comparison_2x2.pdf")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
