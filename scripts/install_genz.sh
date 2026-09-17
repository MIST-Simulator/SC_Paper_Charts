#!/usr/bin/env bash
# Install GenZ (MIST's analytical hardware model) together with its
# aiconfigurator submodule and this paper's hardware definitions.
#
# A plain `pip install git+.../GenZ-LLM-Analyzer` does NOT fetch git
# submodules, which leaves GenZ without aiconfigurator: no per-operator
# performance databases and no accelerator definitions. Every SKU modelled
# with compute_engine="profiled-ops" then fails with
# `ModuleNotFoundError: No module named 'aiconfigurator'`, which takes out
# T3 and T4. This script does the three things a pip install cannot:
#
#   1. clones GenZ with --recursive so aiconfigurator is present,
#   2. pins aiconfigurator to the revision GenZ's own db.py expects,
#   3. installs the accelerator definitions this paper searches over.
#
# Safe to re-run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="${GENZ_VENDOR_DIR:-$REPO_ROOT/.vendor}"
GENZ_DIR="$VENDOR_DIR/GenZ-LLM-Analyzer"

GENZ_URL="https://github.com/abhibambhaniya/GenZ-LLM-Analyzer.git"
GENZ_REF="${GENZ_REF:-main}"

# Use whatever revision GenZ records for the submodule. The accelerator
# definitions in data/hardware/ use that revision's schema key
# (float16_tc_flops); an older revision expects bfloat16_tc_flops and will
# fail with KeyError. Override only if you know the definitions match.
AICONFIGURATOR_REF="${AICONFIGURATOR_REF:-}"

PYTHON="${PYTHON:-python}"

echo "==> Installing GenZ into $GENZ_DIR"
mkdir -p "$VENDOR_DIR"

if [[ -d "$GENZ_DIR/.git" ]]; then
  echo "    existing checkout found, updating"
  git -C "$GENZ_DIR" fetch --quiet origin
  git -C "$GENZ_DIR" checkout --quiet "$GENZ_REF"
else
  git clone --quiet --recursive "$GENZ_URL" "$GENZ_DIR"
  git -C "$GENZ_DIR" checkout --quiet "$GENZ_REF"
fi

echo "==> Checking out the aiconfigurator submodule"
# --force: a previous run overwrote tracked accelerator definitions with
# the ones from data/hardware/, which would otherwise block the checkout.
# They are re-installed below, so discarding them here is safe.
git -C "$GENZ_DIR" submodule update --quiet --init --recursive --force
AICONF_DIR="$GENZ_DIR/GenZ/aiconfigurator"
# In a submodule checkout .git is a file pointing at the real gitdir, not a
# directory, so ask git rather than testing for a directory.
if ! git -C "$AICONF_DIR" rev-parse --git-dir >/dev/null 2>&1; then
  echo "error: aiconfigurator submodule missing at $AICONF_DIR" >&2
  exit 1
fi
if [[ -n "$AICONFIGURATOR_REF" ]]; then
  echo "    pinning to ${AICONFIGURATOR_REF:0:12}"
  git -C "$AICONF_DIR" fetch --quiet origin || true
  git -C "$AICONF_DIR" checkout --quiet "$AICONFIGURATOR_REF"
fi
echo "    aiconfigurator at $(git -C "$AICONF_DIR" rev-parse --short HEAD)"

echo "==> Installing accelerator definitions"
SYSTEMS_DIR="$AICONF_DIR/src/aiconfigurator/systems"
if [[ ! -d "$SYSTEMS_DIR" ]]; then
  echo "error: expected systems dir not found at $SYSTEMS_DIR" >&2
  exit 1
fi
installed=0
for yaml in "$REPO_ROOT"/data/hardware/*.yaml; do
  [[ -e "$yaml" ]] || continue
  cp "$yaml" "$SYSTEMS_DIR/"
  installed=$((installed + 1))
done
echo "    installed $installed accelerator definition(s) into aiconfigurator"

# Accelerators without measured silicon data (the AMD, TPU and Etched parts)
# set `data_dir: data/dummy` and are modelled from their YAML spec sheet via
# the roofline path. aiconfigurator still resolves a database version before
# reaching that path, so it needs the directory tree to exist -- the files
# themselves are never read, and upstream ships none.
DUMMY_DIR="$SYSTEMS_DIR/data/dummy"
mkdir -p "$DUMMY_DIR/vllm/0.14.0" "$DUMMY_DIR/nccl/2.27"
echo "    created placeholder database tree for spec-sheet-modelled accelerators"

echo "==> Installing GenZ and aiconfigurator into the active environment"
$PYTHON -m pip install -q -e "$AICONF_DIR"
$PYTHON -m pip install -q -e "$GENZ_DIR"

echo "==> Verifying"
$PYTHON - <<'PY'
import sys

try:
    import GenZ
    import aiconfigurator  # noqa: F401
except ImportError as exc:
    sys.exit(f"verification failed: {exc}")

# Same two compatibility shims the run_T*.py scripts apply; see
# docs/FINDINGS.md#upstream-bugs.
import os

sys.path.insert(0, os.path.join(os.path.dirname(GenZ.__file__), "aiconfigurator", "src"))
from aiconfigurator.sdk import common as _aic

for _enum in ("GEMMQuantMode", "FMHAQuantMode", "KVCacheQuantMode", "MoEQuantMode"):
    _cls = getattr(_aic, _enum, None)
    if _cls is not None and not hasattr(_cls, "bfloat16") and hasattr(_cls, "float16"):
        _cls.bfloat16 = _cls.float16

from GenZ import db as _genz_db

_orig_init = _genz_db.RuntimeDB.__init__


def _init(self, hardware, backend="vllm", version=None, **kw):
    if version is None:
        for cand in (backend, "trtllm", "sglang"):
            try:
                found = _genz_db.get_latest_database_version(system=hardware, backend=cand)
            except Exception:
                found = None
            if found is not None:
                backend, version = cand, found
                break
    return _orig_init(self, hardware, backend=backend, version=version, **kw)


_genz_db.RuntimeDB.__init__ = _init

from GenZ import System

# One Nvidia SKU (database-backed) and one of this paper's added SKUs.
failures = []
for hw in ("h200_sxm", "gb300", "mi350x", "tpu_v7", "etched"):
    try:
        System(system_name=hw, compute_engine="profiled-ops",
               bits="fp8", collective_strategy="profiled-ops")
    except Exception as exc:
        failures.append(f"  {hw}: {type(exc).__name__}: {exc}")

if failures:
    sys.exit("verification failed for:\n" + "\n".join(failures))
print("    GenZ + aiconfigurator OK (all accelerators resolve)")
PY

echo
echo "==> GenZ setup complete."
