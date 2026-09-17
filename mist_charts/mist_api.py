"""Single import surface for the MIST simulator.

Every reproduction script imports MIST through this module instead of the
simulator package directly, so a package rename touches one file. The
simulator is published as ``mist`` (github.com/MIST-Simulator/MIST);
pre-release checkouts installed it as ``GenA``, so this module falls back
to that name transparently. GenZ (github.com/abhibambhaniya/
GenZ-LLM-Analyzer) is a separate project -- MIST's analytical hardware/
model-config library -- and keeps its own name.
"""

import importlib
import os
import sys as _sys

# Published name first, pre-release name second. Override with
# MIST_PACKAGE_NAME=GenA to pin a specific checkout -- useful when both are
# installed in one environment and you need results comparable to a run made
# against the other one.
_PACKAGE_CANDIDATES = (
    (os.environ["MIST_PACKAGE_NAME"],)
    if os.environ.get("MIST_PACKAGE_NAME")
    else ("mist", "GenA")
)

_SUBMODULES = (
    "",
    ".Coordinator",
    ".Engine",
    ".Request",
    ".Platforms",
    ".Scheduler",
    ".Input_requests",
    ".Tracing",
)

_INSTALL_HINT = (
    "Could not import the MIST simulator.\n"
    "Install it with:\n"
    "    pip install git+https://github.com/MIST-Simulator/MIST.git\n"
    "or run ./setup.sh, which installs it along with every other dependency."
)


def _resolve_package() -> str:
    """Return whichever candidate package name is importable."""
    for name in _PACKAGE_CANDIDATES:
        try:
            importlib.import_module(name)
        except ImportError:
            continue
        return name
    raise ImportError(_INSTALL_HINT)


#: The package name actually found on this system ("mist", or "GenA" on a
#: pre-release checkout). Reported by the smoke test so a reviewer can see
#: which one is installed.
MIST_PACKAGE = _resolve_package()

# When running against a pre-release checkout, alias it (and its
# submodules) into sys.modules under "mist", so the imports below -- and
# any `from mist... import` elsewhere -- resolve against the published
# name regardless of what is installed.
if MIST_PACKAGE != "mist":
    for _sub in _SUBMODULES:
        try:
            _sys.modules["mist" + _sub] = importlib.import_module(
                MIST_PACKAGE + _sub
            )
        except ImportError:
            pass

try:
    from mist import (  # noqa: F401
        BatchingMethod,
        LengthVariables,
        MemoryCacheConfig,
        PlatformConfig,
        PoissonDistribution,
        SchedulerConfig,
        SingleCacheConfig,
        TraceDistributions,
        TraceIngestion,
        vLLMPlatformConfig,
    )
    from mist.Coordinator import CoordRouterType  # noqa: F401

    # The coordinators are exported as MIST* by the published package and as
    # GenA* by a pre-release checkout; accept either so the scripts work
    # against both.
    _coord = importlib.import_module("mist.Coordinator")
    try:
        MISTCoordinator = _coord.MISTCoordinator
        MISTCoordinatorDisagg = _coord.MISTCoordinatorDisagg
    except AttributeError:
        MISTCoordinator = _coord.GenACoordinator
        MISTCoordinatorDisagg = _coord.GenACoordinatorDisagg
    from mist.Engine import (  # noqa: F401
        EngineType,
        KVRetrievalEngine,
        LLMEngine,
    )
    from mist.Request import Request, RequestStage  # noqa: F401

    # GenZ: MIST's analytical hardware/model-config library, installed as a
    # direct dependency (see requirements.txt). get_configs exposes a
    # model's per-token KV cache size, needed to convert a KV cache byte
    # size into a token count for MIST's retrieval-latency model (T2b).
    from GenZ import get_configs  # noqa: F401
except ImportError as exc:  # pragma: no cover - surfaced to the reviewer
    raise ImportError(_INSTALL_HINT) from exc


def mist_package_path():
    """Filesystem root of the installed simulator package (e.g. to reach
    the profiled vLLM runtime tables under ``Platforms/vllm_runtime_data/``).
    """
    from pathlib import Path

    return Path(importlib.import_module(MIST_PACKAGE).__file__).resolve().parent


def mist_version() -> str:
    """Best-effort version string, used by the smoke test's banner."""
    from importlib.metadata import version

    for dist in ("mist", "MIST", "GenA_llm"):
        try:
            return version(dist)
        except Exception:
            continue
    return "unknown"
