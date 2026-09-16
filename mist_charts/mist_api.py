"""Single import surface for the MIST simulator.

Every reproduction script imports MIST through this module rather than
reaching for the package directly.  The simulator is distributed as the
``GenA`` Python package (see github.com/MIST-Simulator/MIST); keeping the
mapping in one place means a future package rename touches one file rather
than every script.
"""

try:
    from GenA import (  # noqa: F401
        BatchingMethod,
        LengthVariables,
        MemoryCacheConfig,
        PlatformConfig,
        PoissonDistribution,
        SchedulerConfig,
        SingleCacheConfig,
        TraceDistributions,
        vLLMPlatformConfig,
    )
    from GenA.Coordinator import (  # noqa: F401
        CoordRouterType,
        GenACoordinator,
        GenACoordinatorDisagg,
    )
    from GenA.Engine import (  # noqa: F401
        EngineType,
        KVRetrievalEngine,
        LLMEngine,
    )
    from GenA.Request import Request, RequestStage  # noqa: F401
    # GenZ is MIST's underlying hardware/model-config library (installed as a
    # direct dependency alongside GenA -- see requirements.txt). get_configs
    # exposes a model's per-token KV cache size, needed to convert a KV cache
    # byte size into a token count for MIST's retrieval-latency model (T2b).
    from GenZ import get_configs  # noqa: F401
except ImportError as exc:  # pragma: no cover - surfaced to the reviewer
    raise ImportError(
        "Could not import the MIST simulator (Python package `GenA`).\n"
        "Install it with:\n"
        "    pip install git+https://github.com/MIST-Simulator/MIST.git\n"
        "or run ./setup.sh, which installs it along with every other "
        "dependency."
    ) from exc


def mist_version() -> str:
    """Best-effort version string, used by the smoke test's banner."""
    try:
        from importlib.metadata import version

        return version("GenA_llm")
    except Exception:
        return "unknown"
