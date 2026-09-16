"""Shared matplotlib styling so every figure in the paper reads as one set.

Colours and font sizes are lifted verbatim from the notebooks that produced
the published figures, so regenerated PDFs match the camera-ready versions.
"""

import matplotlib
import matplotlib.pyplot as plt

# Keyed by the legend label used in the paper, so panels stay consistent
# across T1 (CDF + simulation time) and T2 (per-step runtimes).
SIMULATOR_COLORS = {
    "vLLM (Real)": "#2563EB",                    # blue
    "Vidur (Simulated)": "#059669",              # emerald
    "LLMServingSim2.0 (Simulated)": "#7C3AED",   # violet
    "MIST (Simulated)": "#D97706",               # amber
}

FONT_SIZES = {
    "title": 18,
    "labels": 18,
    "ticks": 16,
    "legend": 20,
    "text": 16,
}


def apply_paper_style() -> None:
    """Set rcParams shared by every figure.

    Uses the Agg backend: the scripts are run headless during artifact
    evaluation and must never block on a GUI window.
    """
    matplotlib.use("Agg", force=True)
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.6,
            "pdf.fonttype": 42,  # embed TrueType so the PDF is editable
            "ps.fonttype": 42,
        }
    )


def save_figure(fig, filename, figure_dir=None) -> "pathlib.Path":
    """Write ``fig`` to ``figures/<filename>`` and report where it landed."""
    from .paths import FIGURE_DIR

    out_dir = figure_dir or FIGURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"wrote {out_path}")
    return out_path
