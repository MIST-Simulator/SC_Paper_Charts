"""Single import surface for the MIST simulator.

Every reproduction script imports MIST through this module instead of the
simulator package directly, so a package rename touches one file. The
simulator is published as ``mist`` (github.com/astra-sim/MIST) and uses the
SC'26 paper's names: a *client* (pre-release: engine) runs on a hardware
*cluster* (pre-release: platform). GenZ (github.com/abhibambhaniya/
GenZ-LLM-Analyzer) is a separate project -- MIST's analytical hardware/
model-config library -- and keeps its own name.
"""

import importlib

MIST_PACKAGE = "mist"

_INSTALL_HINT = (
    "Could not import the MIST simulator (with the SC'26 paper names: "
    "ClusterConfig, LLMClient, ...).\n"
    "Install it with:\n"
    "    pip install git+https://github.com/astra-sim/MIST.git\n"
    "or run ./setup.sh, which installs it along with every other dependency."
)

try:
    from mist import (  # noqa: F401
        BatchingMethod,
        ClusterConfig,
        LengthVariables,
        MemoryCacheConfig,
        PoissonDistribution,
        SchedulerConfig,
        SingleCacheConfig,
        TraceDistributions,
        TraceIngestion,
        vLLMClusterConfig,
    )
    from mist.Client import ClientType, KVRetrievalClient, LLMClient  # noqa: F401
    from mist.Coordinator import (  # noqa: F401
        CoordRouterType,
        MISTCoordinator,
        MISTCoordinatorDisagg,
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
    the profiled vLLM runtime tables under ``Clusters/vllm_runtime_data/``).
    """
    from pathlib import Path

    return Path(importlib.import_module(MIST_PACKAGE).__file__).resolve().parent


def mist_runtime_data_dir():
    """Directory of MIST's profiled vLLM runtime tables.

    ``mist/Clusters/vllm_runtime_data/`` since the paper-naming release;
    ``Platforms/`` in earlier checkouts.
    """
    root = mist_package_path()
    for sub in ("Clusters", "Platforms"):
        path = root / sub / "vllm_runtime_data"
        if path.is_dir():
            return path
    return root / "Clusters" / "vllm_runtime_data"


def mist_version() -> str:
    """Best-effort version string, used by the smoke test's banner."""
    from importlib.metadata import version

    for dist in ("mist", "MIST", "GenA_llm"):
        try:
            return version(dist)
        except Exception:
            continue
    return "unknown"
