"""Canonical locations for inputs, results and figures.

Every ``run_T*.py`` writes its CSVs under ``results/<task>/``.  Every
``plot_T*.py`` reads from ``results/<task>/`` when that directory holds a
result, and otherwise falls back to the reference CSVs shipped in
``results/reference/<task>/``.  That fallback is what lets a reviewer
regenerate every figure in seconds without first running the sweeps.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
TRACE_DIR = DATA_DIR / "traces"
VALIDATION_DIR = DATA_DIR / "validation"

RESULTS_DIR = REPO_ROOT / "results"
REFERENCE_DIR = RESULTS_DIR / "reference"

FIGURE_DIR = REPO_ROOT / "figures"


def result_path(task: str, filename: str) -> Path:
    """Path a ``run_T*.py`` should write ``filename`` to, creating the dir."""
    out_dir = RESULTS_DIR / task
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename


def resolve_results(task: str, filename: str, prefer_reference: bool = False) -> Path:
    """Locate ``filename`` for ``task``, preferring freshly generated results.

    Raises FileNotFoundError naming both candidates, so a reviewer who has
    run neither the sweep nor ``setup.sh`` gets an actionable message.
    """
    generated = RESULTS_DIR / task / filename
    reference = REFERENCE_DIR / task / filename

    order = (reference, generated) if prefer_reference else (generated, reference)
    for candidate in order:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No result named {filename!r} for task {task}.\n"
        f"  looked for generated: {generated}\n"
        f"  looked for reference: {reference}\n"
        f"Run `python run_{task}.py` to generate it, or check out the "
        f"reference results shipped with the repository."
    )
