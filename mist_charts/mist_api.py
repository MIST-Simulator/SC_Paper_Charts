"""Single import surface for the MIST simulator.

Every reproduction script imports MIST through this module rather than
reaching for the simulator package directly, so nothing else in the
repository has to name it.

The simulator is published as the ``mist`` package
(github.com/MIST-Simulator/MIST).  Pre-release checkouts installed it
under the name ``GenA``; this module transparently falls back to that so
the scripts keep working against an unrenamed local checkout.  ``mist``
is the name to use everywhere else.

Note that GenZ is a *separate* project (github.com/abhibambhaniya/
GenZ-LLM-Analyzer), not part of the rename -- it is MIST's underlying
analytical hardware/model-config library and keeps its own name.
"""

import importlib
import sys as _sys

# Published name first, pre-release name second.
_PACKAGE_CANDIDATES = ("mist", "GenA")

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
    from mist.Coordinator import (  # noqa: F401
        CoordRouterType,
        GenACoordinator as MISTCoordinator,
        GenACoordinatorDisagg as MISTCoordinatorDisagg,
    )
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
    """Filesystem root of the installed simulator package.

    Used to reach data that ships inside MIST itself, such as the profiled
    vLLM runtime tables under ``Platforms/vllm_runtime_data/``.
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
