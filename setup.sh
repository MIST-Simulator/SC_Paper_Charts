#!/usr/bin/env bash
# Create the Conda environment and install every dependency needed to
# reproduce the figures.  Safe to re-run.
set -euo pipefail

ENV_NAME="${MIST_ENV_NAME:-mist-sc26}"
PYTHON_VERSION="${MIST_PYTHON_VERSION:-3.12}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "error: conda not found on PATH. Install Miniconda first:" >&2
  echo "       https://docs.conda.io/projects/miniconda/" >&2
  exit 1
fi

# `conda activate` needs the shell hook; `conda run` alone cannot install
# into an env that does not exist yet.
eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "==> Conda env '$ENV_NAME' already exists, reusing it"
else
  echo "==> Creating Conda env '$ENV_NAME' (Python $PYTHON_VERSION)"
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi

conda activate "$ENV_NAME"

echo "==> Installing Python dependencies (including the MIST simulator)"
python -m pip install --upgrade pip
python -m pip install -r "$REPO_ROOT/requirements.txt"

# Must run after the requirements install: it replaces the pip-installed
# GenZ with a recursive checkout that actually includes aiconfigurator, and
# adds this paper's accelerator definitions. Without it, T3 and T4 fail with
# ModuleNotFoundError: No module named 'aiconfigurator'.
echo
PYTHON=python bash "$REPO_ROOT/scripts/install_genz.sh"

echo
echo "==> Setup complete."
echo "    Activate the environment with:  conda activate $ENV_NAME"
echo "    Download the input traces with: bash scripts/download_traces.sh"
echo "    Verify the installation with:   python scripts/smoke_test.py"
