"""The five KV-cache storage architectures compared in Fig. 10 (Table 2 of
the MIST SC26 paper).

This module holds only declarative data. ``run_T4.py`` builds the MIST
``MemoryCacheConfig``/``SingleCacheConfig`` objects (and cache->compute
network-link params) from these records, ``plot_T4.py`` reads
``CASE_COLORS``/``CASE_ORDER`` for the legend, and the README's Table 2 is
transcribed from here -- one place encoding "what Case A/B/C/D/E means".

Capacity/bandwidth numbers are copied verbatim from Table 2. Numbers not
given by Table 2 (retrieval latency, network link latency/bandwidth, skew
centers) are carried over from the authors' sweep notebook
(``Experiments/Memory_Storage_Comparisions.py`` /
``4. Cache_storage_config_comparisions.ipynb``, cell 7). Case D's DCN
bandwidth is a known paper-vs-notebook conflict; see the note on that field
below and docs/FINDINGS.md#t4.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class StorageArchitecture:
    """One row of Table 2."""

    key: str                       # 'A'..'E', matches the paper and the README
    label: str                     # short display name for legends/tables
    description: str               # one-line description of the access pattern
    cache_type: str                # medium modeled by MIST's SingleCacheConfig
    capacity_gb: Optional[float]   # cache capacity in GB (None: no cache, Case E)
    cache_bandwidth_gbps: Optional[float]   # SingleCacheConfig bandwidth (GB/s)
    cache_retrieval_latency_ms: Optional[float]  # SingleCacheConfig fixed latency (ms)
    sharing_degree: Optional[int]  # clients sharing one physical cache instance;
                                    # None means "every client in the sweep" (fully shared)
    network_latency_ms: float      # fixed latency of the cache -> compute transfer link
    network_bandwidth_gbps: float  # bandwidth of that transfer link
    dcn: bool                      # whether the transfer above crosses the data-center network
    recompute_kv: bool             # True for Case E: no cache, KV is recomputed from scratch


STORAGE_ARCHITECTURES = {
    "A": StorageArchitecture(
        key="A",
        label="Dedicated per-client",
        description="1 TB / 128 GB/s LPDDR next to each client; local access only.",
        cache_type="DRAM",
        capacity_gb=1_000,
        cache_bandwidth_gbps=128,
        cache_retrieval_latency_ms=0.10,
        sharing_degree=1,
        network_latency_ms=0.5,
        network_bandwidth_gbps=128,
        dcn=False,
        recompute_kv=False,
    ),
    "B": StorageArchitecture(
        key="B",
        label="Platform-shared",
        description="4 TB / 32 GB/s DRAM shared by the 4 clients on one platform.",
        cache_type="DRAM",
        capacity_gb=4_000,
        cache_bandwidth_gbps=32,
        cache_retrieval_latency_ms=0.10,
        sharing_degree=4,
        network_latency_ms=1.0,
        network_bandwidth_gbps=16,
        dcn=False,
        recompute_kv=False,
    ),
    "C": StorageArchitecture(
        key="C",
        label="Rack-shared",
        description="32 TB / 2 GB/s SSD shared by all 32 clients in a rack.",
        cache_type="SSD",
        capacity_gb=32_000,
        cache_bandwidth_gbps=2,
        cache_retrieval_latency_ms=2.0,
        sharing_degree=32,
        network_latency_ms=5.0,
        network_bandwidth_gbps=8,
        dcn=False,
        recompute_kv=False,
    ),
    "D": StorageArchitecture(
        key="D",
        label="Rack-shared + DCN",
        description="Same 32 TB / 2 GB/s SSD tier as Case C, but reachable from any of "
                     "the 4 racks over the data-center network.",
        cache_type="SSD",
        capacity_gb=32_000,
        cache_bandwidth_gbps=2,
        cache_retrieval_latency_ms=2.0,
        sharing_degree=None,
        network_latency_ms=20.0,
        # Table 2 states 128 GB/s here, but only 1 GB/s (the notebook that
        # made the published figure) reproduces it -- at 128 GB/s Case D
        # strictly dominates C instead. Do not "fix" this; --dcn-bandwidth-gbps
        # in run_T4.py flips it. See docs/FINDINGS.md#t4.
        network_bandwidth_gbps=1.0,
        dcn=True,
        recompute_kv=False,
    ),
    "E": StorageArchitecture(
        key="E",
        label="No cache",
        description="No KV cache storage tier: every request recomputes its KV cache "
                     "from scratch during prefill.",
        cache_type="n/a",
        capacity_gb=None,
        cache_bandwidth_gbps=None,
        cache_retrieval_latency_ms=None,
        sharing_degree=None,
        network_latency_ms=0.5,
        network_bandwidth_gbps=128,
        dcn=False,
        recompute_kv=True,
    ),
}

CASE_ORDER = ["A", "B", "C", "D", "E"]

# Consistent legend/plot colors, keyed the same way across every T4 panel.
CASE_COLORS = {
    "A": "#DC2626",  # red
    "B": "#2563EB",  # blue
    "C": "#059669",  # green
    "D": "#7C3AED",  # purple
    "E": "#D97706",  # orange
}


def sharing_group_size(arch: StorageArchitecture, num_clients: int) -> int:
    """Number of client engines that physically share one cache instance.

    Clamped to ``num_clients`` so the reduced-scale smoke runs (e.g.
    ``--num-clients 8``) still produce a valid partition even when Table 2's
    sharing degree (e.g. 32 for Case C) exceeds the requested client count.
    """
    degree = num_clients if arch.sharing_degree is None else arch.sharing_degree
    return max(1, min(degree, num_clients))
