# Provenance of the reference results

Each `results/reference/<task>/` directory is the unmodified output of the
`run_<task>.py` command below. Re-running the same command with the same
versions and an empty MIST latency cache reproduces every column exactly,
except `sim_wall_clock_sec` (wall-clock time of the run).

MIST reuses step latencies cached on disk (`~/.cache/mist`, or
`$MIST_CACHE_DIR`), and the cache key does not include T2's training data or
hold-out seed. To reproduce, point each run at a new, empty directory:
`MIST_CACHE_DIR=$(mktemp -d) python run_T2.py`.

## T1, T2: regenerated 2026-11-17

| | |
| --- | --- |
| Commands | `python run_T1.py`, `python run_T2.py` (default arguments, seed 259), each with its own empty `MIST_CACHE_DIR` |
| MIST | astra-sim/MIST `c00cc59` (`main`). PR #2 (ASTRA-sim backend) gives identical output. |
| GenZ | abhibambhaniya/GenZ-LLM-Analyzer `bc2df48`, aiconfigurator `d5fb9da2`, via `scripts/install_genz.sh` |
| Python | 3.12.12; numpy 1.26.4, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.17.1 |
| Host | Linux x86_64 (Georgia Tech synergy2), CPU only |

Compared with the previous T1/T2 reference (2026-09-16, MIST version not
recorded), mean values agree within 0.1% (T1 end-to-end latency: 2.900 vs
2.902 s on H100, 12.811 vs 12.813 s on L40S, 2.692 vs 2.694 s on TPUv6e).
Individual requests differ by up to 24 ms on H100 and 0.17 s on L40S. In T2,
the KV-retrieval results are unchanged, and the step-runtime means agree
within 0.02% (MIST 26.530 vs 26.529 ms). The Vidur column, which MIST does
not compute, moved by the same order, so the step-level differences come
from the Python environment rather than from MIST.

## T3, T4: 2026-09-16

Committed with the T3 and T4 scripts (`fffe521`, `cb93856`, `2a4e41d`). The
MIST and GenZ versions used were not recorded.
