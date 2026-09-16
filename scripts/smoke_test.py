#!/usr/bin/env python3
"""Verify the installation: import MIST, run a tiny simulation, plot once.

Finishes in well under a minute.  If this passes, the environment is ready
for run_T1..run_T4; if it fails, the error says which piece is missing.
"""

import sys
import traceback
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def check_imports():
    from mist_charts.mist_api import mist_version

    print(f"  MIST (GenA_llm) version: {mist_version()}")
    import matplotlib
    import numpy
    import pandas
    import sklearn

    print(f"  numpy {numpy.__version__}, pandas {pandas.__version__}, "
          f"scikit-learn {sklearn.__version__}, matplotlib {matplotlib.__version__}")


def check_simulation():
    """Run a 2-engine, 2-second Poisson simulation end to end."""
    from mist_charts.mist_api import (
        BatchingMethod,
        EngineType,
        GenACoordinator,
        LengthVariables,
        LLMEngine,
        PlatformConfig,
        PoissonDistribution,
        SchedulerConfig,
    )

    model = "meta-llama/meta-llama-3.1-8b"
    platform = PlatformConfig(device="H100_GPU", tensor_parallel_size=1, model=model)

    request_queue = PoissonDistribution(
        rps=2,
        sim_time=2000,
        input_vars=LengthVariables(512, 128),
        output_vars=LengthVariables(64, 16),
    ).request_queue
    print(f"  generated {len(request_queue)} requests")

    # logging_file=None keeps the smoke test from leaving a trace.json behind.
    coordinator = GenACoordinator(deepcopy(request_queue), logging_file=None)
    coordinator.add_engine(
        LLMEngine(
            model=model,
            engine_types=[EngineType.PREFILL, EngineType.DECODE],
            scheduler_config=SchedulerConfig(
                batching_method=BatchingMethod.CHUNKED, chunk_size=512
            ),
            platform=platform,
        ),
        [EngineType.PREFILL, EngineType.DECODE],
    )
    coordinator.run_sim()

    stats = coordinator.get_global_stats()
    print(f"  served {coordinator.request_serviced}/{coordinator.request_accepted} "
          f"requests")
    print(f"  TTFT={stats.TTFT:.2f} ms  TPOT={stats.TPOT:.2f} ms  "
          f"throughput={stats.total_token_throughput:.1f} tok/s")

    if coordinator.request_serviced == 0:
        raise RuntimeError("simulation completed but served no requests")
    if not stats.TTFT > 0:
        raise RuntimeError(f"implausible TTFT: {stats.TTFT}")


def check_plotting():
    import matplotlib.pyplot as plt

    from mist_charts import apply_paper_style
    from mist_charts.paths import REPO_ROOT

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot([0, 1, 2], [0, 1, 4])
    out = REPO_ROOT / "results" / "smoke_test.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")
    out.unlink()


def main() -> int:
    checks = [
        ("Imports", check_imports),
        ("Simulation", check_simulation),
        ("Plotting", check_plotting),
    ]
    failures = []
    for name, fn in checks:
        print(f"[{name}]")
        try:
            fn()
            print(f"  -> OK\n")
        except Exception:
            traceback.print_exc()
            failures.append(name)
            print(f"  -> FAILED\n")

    if failures:
        print(f"Smoke test FAILED ({', '.join(failures)}).")
        print("Re-run ./setup.sh, or see README.md#troubleshooting.")
        return 1

    print("Smoke test PASSED. The environment is ready.")
    print("Next: python run_T1.py   (or bash run_all.sh for everything)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
