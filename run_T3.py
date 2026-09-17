#!/usr/bin/env python3
"""Reproduce Fig. 7(a)-(b) and Fig. 8 of the MIST SC26 paper: heterogeneous
hardware / parallelism / batching design-space search for Qwen3-32B, for two
use cases built from the SAME GitHub-Code trace (see "TRACE / SLO SOURCE OF
TRUTH" below) -- "chat conversation" (that trace without KV retrieval,
TTFT P99 < 200 ms) and "code generation, high prefix-KV reuse" (a seeded
shuffle of the same trace, with KV retrieval, TTFT P99 < 1.5 s) --
optimizing generated tokens/s/$ among SLO-meeting configs.

  Fig. 7(a)  figures/Chat_search_results.pdf              (plot_T3.py)
  Fig. 7(b)  figures/Code_Generation_KV_search_results.pdf (plot_T3.py)
  Fig. 8     figures/Combined_2x2_bar_results.pdf          (plot_T3.py)

Ports ``GenA_Paper_charts/SC26/{search_deployment_space,experiment_runner}.py``
and ``ISCA26/Batching_method_comparisions.py`` (``run_simulation``) from the
authors' notebook repo, restricted to the paper's Sec. 5.1 search: 8
accelerator SKUs (``mist_charts.pricing``) x TP in {1, 2} x {chunked,
disaggregated} batching x prefill:decode client ratio, at 100 RPS on up to
8 nodes. The parallelism/memory/price enumeration itself lives in
``mist_charts/deployment_space.py``; this script drives the simulator over
that enumeration.

*** TRACE / SLO SOURCE OF TRUTH (commit aac3254, "SC Paper Charts") ***
An earlier version of this script used narrativeqa.csv for "chat" (per the
original task spec) and found essentially no configuration could meet a
250ms TTFT SLO -- correct given that trace's ~58K-token mean context, but
it turned out to be the wrong input. ``GenA_Paper_charts/SC26/plot_sc_results.py``
and ``plot_bar_results.py`` (confirmed by the author as the scripts that
produced the camera-ready Fig. 7/8) read exactly two CSVs, both derived
from the *code-generation* trace:
    Chat_search_results.pdf              <- Code_Generation_KV.csv, WITHOUT
                                             KV retrieval, slo_ms=200
    Code_Generation_KV_search_results.pdf <- a shuffled copy of the same
                                             trace, WITH KV retrieval,
                                             slo_ms=1500
narrativeqa.csv is not used for Fig. 7/8 at all. This is why both use cases
below point at ``code_generation_kv.csv``, differing only in
``kv_retrieval`` and a seeded shuffle (see ``_seeded_shuffle_trace``).
Also note the "chat" SLO used here is **200ms**, not the 250ms stated in
the paper's Sec. 5.1 -- that is what the plotting code that made the figure
actually uses; the discrepancy is reported, not silently resolved.

*** Runtime-backend substitution (read this before comparing numbers) ***
The paper's DSE used Nvidia's AIConfigurator as a single LLM-cluster runtime
backend shared by every SKU, specifically for an apples-to-apples hardware
comparison. The open MIST package (this repo's ``MIST``) has no standalone
AIConfigurator LLM-cluster simulator class. Instead, every SKU here runs
through MIST's own ``PlatformConfig`` -> GenZ ``System``, which (for these 8
SKUs, all configured with ``compute_engine="profiled-ops"``) happens to look
up its own per-operator (GEMM / attention / collective) timings from that
very same aiconfigurator silicon database -- but layers MIST's own
event-driven request queueing, batching, and (for disaggregated configs)
prefill/decode orchestration on top, rather than AIConfigurator's own
cluster-level simulator. So every SKU here *is* still scored by one
consistent backend, which is the property the paper's DSE actually needs --
but the serving-level simulation is MIST's, not AIConfigurator's. Expect
absolute tokens/$ numbers to differ from the published figure even though
the search + SLO filter + tokens/s/$ optimization procedure is identical.

*** Environment compatibility shim (see _patch_aiconfigurator_quant_enum_compat) ***
The aiconfigurator submodule commit currently vendored under the GenZ
dependency renamed ``GEMMQuantMode.bfloat16`` (and its FMHA/KVCache/MoE
equivalents) to ``float16``, but mist/GenZ's ``db.py`` -- a dependency this
repository does not own or edit -- still references the old name, so
constructing a ``PlatformConfig`` for any of these 8 SKUs raises
``AttributeError`` before this patch runs. The patch below aliases the old
enum member back onto the already-imported third-party class at runtime; it
edits no file on disk. Remove it once GenZ's ``db.py`` or the pinned
aiconfigurator submodule catches up.

Writes (via ``mist_charts.paths.result_path("T3", ...)``, or ``--output``):
    <out>/<use_case>_search_space.csv  -- every enumerated config (pre-subset)
    <out>/<use_case>_results.csv       -- one row per simulated config

Resumable: re-running with the same ``--output`` skips (UseCase, Serving
Name) pairs already present in ``<use_case>_results.csv``, so an
interrupted 24-48h full sweep can be restarted without losing progress.
"""

import argparse
import ast
import concurrent.futures
import contextlib
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _patch_aiconfigurator_quant_enum_compat() -> None:
    """Runtime monkeypatch for a GenZ/aiconfigurator version-skew bug.

    See the module docstring's "Environment compatibility shim" section.
    Touches only already-imported third-party module objects in this
    process; no file on disk is modified. A no-op if GenZ/aiconfigurator
    are unavailable or already compatible.
    """
    try:
        import GenZ
    except ImportError:
        return

    submodule_src = os.path.join(os.path.dirname(GenZ.__file__), "aiconfigurator", "src")
    if submodule_src not in sys.path:
        sys.path.insert(0, submodule_src)

    try:
        from aiconfigurator.sdk import common as aic_common
    except ImportError:
        return

    for name in ("GEMMQuantMode", "FMHAQuantMode", "KVCacheQuantMode", "MoEQuantMode"):
        enum_cls = getattr(aic_common, name, None)
        if enum_cls is not None and not hasattr(enum_cls, "bfloat16") and hasattr(enum_cls, "float16"):
            enum_cls.bfloat16 = enum_cls.float16


_patch_aiconfigurator_quant_enum_compat()


# Backends tried, in order, when resolving a SKU's performance database.
_BACKEND_FALLBACK_ORDER = ("vllm", "trtllm", "sglang")


def _patch_runtime_db_backend_fallback() -> None:
    """TEMPORARY shim: fall back to another backend's database when a SKU's
    vLLM one is missing or incomplete. Remove once MIST threads a `backend`
    parameter through `System` -- see KNOWN ISSUES in the module docstring.

    ``GenZ/system.py`` constructs ``RuntimeDB(hardware=...)`` without passing
    ``backend``, so it always resolves ``backend='vllm'``. For gb300 that
    database ships an ``INCOMPLETE.txt`` ("context_attention_perf.txt is
    incomplete, most data points failed in collection"), so
    ``get_latest_database_version`` returns None, ``get_database`` returns
    None, and every gb300 configuration dies with
    ``'NoneType' object has no attribute 'system_spec'``. That silently drops
    a third of the paper's Nvidia SKUs from the search.

    gb300 does ship a complete ``trtllm/1.2.0rc6.post3`` database -- the same
    ten files as the h200_sxm TRT-LLM database that already works.

    Switching backend does NOT bias the comparison. MIST runs the database in
    ``DatabaseMode.SOL``, the roofline estimator the paper specifies in
    Sec. 5.1 ("ensuring all HW SKUs are modeled with the same backend and
    based on their raw capabilities"). In SOL mode the measured per-operator
    tables are not used for timing; the database supplies only the hardware's
    ``system_spec``. So this changes which file gb300's spec sheet is read
    from, not how fast gb300 is modeled.
    """
    try:
        from GenZ import db as genz_db
    except ImportError:
        return

    if getattr(genz_db.RuntimeDB, "_mist_backend_fallback", False):
        return

    original_init = genz_db.RuntimeDB.__init__

    def patched_init(self, hardware, backend="vllm", version=None, **kwargs):
        if version is None:
            ordered = (backend,) + tuple(
                b for b in _BACKEND_FALLBACK_ORDER if b != backend
            )
            for candidate in ordered:
                try:
                    found = genz_db.get_latest_database_version(
                        system=hardware, backend=candidate
                    )
                except Exception:
                    found = None
                if found is not None:
                    if candidate != backend:
                        print(
                            f"[MIST] {hardware}: no usable '{backend}' database; "
                            f"using '{candidate}' {found} for its spec sheet "
                            f"(roofline/SOL mode -- timing is unaffected)."
                        )
                    backend, version = candidate, found
                    break
        return original_init(self, hardware, backend=backend, version=version, **kwargs)

    genz_db.RuntimeDB.__init__ = patched_init
    genz_db.RuntimeDB._mist_backend_fallback = True


_patch_runtime_db_backend_fallback()

from mist_charts.deployment_space import get_search_space, estimate_memory_required_gb  # noqa: E402
from mist_charts.mist_api import (  # noqa: E402
    BatchingMethod,
    EngineType,
    MISTCoordinator,
    MISTCoordinatorDisagg,
    LLMEngine,
    PlatformConfig,
    PoissonDistribution,
    SchedulerConfig,
)
from mist_charts.paths import TRACE_DIR, result_path  # noqa: E402
from mist_charts.pricing import PRICE_PER_HOUR, SKUS, vendor_of_config  # noqa: E402

MODEL = "Qwen/Qwen3-32B"

# Both use cases are driven from the same base trace file; "chat" reads it
# unshuffled and without KV retrieval, "codegen" reads a seeded shuffle of
# it with KV retrieval attached (see the module docstring and
# `_seeded_shuffle_trace`). ttft_slo_ms values match plot_sc_results.py's
# main() call sites exactly (200ms for the wo_KV/"chat" case, 1500ms for
# the KV/"codegen" case) -- the 200ms figure is the plotting code's number,
# not the paper's stated 250ms (Sec. 5.1); both are reported, not merged.
BASE_TRACE_FILE = "code_generation_kv.csv"
USE_CASES: Dict[str, Dict] = {
    "chat": dict(trace_file=BASE_TRACE_FILE, ttft_slo_ms=200.0, kv_retrieval=False, shuffle=False),
    "codegen": dict(trace_file=BASE_TRACE_FILE, ttft_slo_ms=1500.0, kv_retrieval=True, shuffle=True),
}

# Trace column semantics (confirmed authoritative by the paper's author):
# ContextTokens = new prefill tokens, GeneratedTokens = decode tokens, and
# (codegen only) KV_Tokens = past_context, i.e. reused prefix KV -- never
# treated as prefill. This is exactly what `build_request_queue` below does
# (PoissonDistribution reads ContextTokens / GeneratedTokens as
# input_len / output_len; the kv_retrieval branch attaches KV_Tokens as
# req.past_context), matching GenA_Paper_charts/SC26/experiment_runner.py's
# request construction, which is the confirmed reference implementation for
# this sweep's per-request setup (not its search-space/model/HW defaults --
# see below).
#
# HARDWARE LIST DISCREPANCY (report, don't silently follow): the version of
# experiment_runner.py that shipped with commit aac3254 restricts
# HW_available to mi300x/mi350x/mi355x/tpu_v6e/tpu_v7 -- a 5-SKU list with
# no Nvidia or Etched SKUs at all, which cannot produce the published
# Fig. 8 (it has Nvidia and Etched bars). This script instead uses the
# paper's Sec. 5.1 eight-SKU list (mist_charts.pricing.SKUS: h200_sxm,
# b200_sxm, gb300, etched, mi350x, mi355x, tpu_v6e, tpu_v7), which is what
# Fig. 8 needs. Price conflicts between experiment_runner.py's internal
# search prices and the prices actually used for cost/tokens-per-$ are
# documented in mist_charts/pricing.py.
#
# RUN-CONFIG DISCREPANCY: aac3254's experiment_runner.py also moved its
# defaults to model=meta-llama-3.1-70B, num_requests=10000, TP<=4 -- that
# is the author's current, in-progress iteration, not the configuration
# that produced the camera-ready figures. The filenames the plotting
# scripts actually read (`... - 100RPS - Qwen3-32B - 60sec - 200reqs.csv`)
# pin down the configuration used here instead: Qwen3-32B, 100 RPS, 60s
# sim window, 200 requests, TP in {1,2} (this script's --rps/--sim-time/
# --num-requests defaults, and deployment_space's max_degree_constraints).

# Qwen3-32B's native max context, less a safety margin for decode-time
# growth. This clip is inherited from an earlier iteration of this script
# that used narrativeqa.csv (mean context 57,782 tokens); it is effectively
# inert now that both use cases read code_generation_kv.csv (max
# ContextTokens 18,517, always well under this clip), but it is left in
# place as a defensive guard against MISTCoordinatorDisagg's hardcoded
# SchedulerConfig max_total_tokens_per_sample=128_000 (Coordinator/
# Splitwise_Coordinator.py, a vendored dependency this repo does not edit),
# which silently drops (FINISHED_IGNORED) any longer request, in case a
# future --output/trace override needs it again.
MAX_CONTEXT_TOKENS = 120_000


RESULT_COLUMNS = [
    "UseCase", "Serving Name", "Batching_Strategy", "Hardware", "Parallelism",
    "Max_Batch_Size", "Chunk_Size", "Cost", "HW_Combination", "Vendor",
    "Energy_Used", "TTFT_avg", "TPOT_avg", "rps",
    "TTFT_p50", "TTFT_p90", "TTFT_p95", "TTFT_p99",
    "output_throughput", "total_token_throughput", "tokens_per_dollar",
    "num_completed", "num_ttft_observed",
]


@contextlib.contextmanager
def _quiet_stdout():
    """Mute stdout at the file-descriptor level for the sweep's duration.

    MIST's analytical hardware model (GenZ) and coordinator unconditionally
    print a line per batching decision / per second of simulated time; with
    a multi-threaded sweep of hundreds of configs that is both unreadable
    and slow. A plain ``contextlib.redirect_stdout`` per job is not
    thread-safe (it mutates the shared ``sys.stdout`` attribute), so this
    redirects the real fd 1 once for the whole ThreadPoolExecutor block
    instead. Status lines printed while this is active go to stderr.
    """
    stdout_fd = sys.stdout.fileno()
    sys.stdout.flush()
    saved_fd = os.dup(stdout_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, stdout_fd)
    os.close(devnull_fd)
    try:
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved_fd, stdout_fd)
        os.close(saved_fd)


def _clip_request_queue(req_queue: List, max_context_tokens: int = MAX_CONTEXT_TOKENS) -> None:
    """Cap every request's prompt length at `max_context_tokens`, in place.

    See MAX_CONTEXT_TOKENS above for why. `remaining_prefill_tokens` is
    re-clipped alongside `input_len` since MIST's `Request.__init__` sets
    `remaining_prefill_tokens = input_len` at construction time.
    """
    clipped = 0
    for req in req_queue:
        if req.input_len > max_context_tokens:
            req.input_len = max_context_tokens
            req.remaining_prefill_tokens = max_context_tokens
            clipped += 1
    if clipped:
        print(f"  clipped {clipped}/{len(req_queue)} requests to {max_context_tokens} tokens")


def _seeded_shuffle_trace(trace_path: Path, seed: int) -> Path:
    """Seeded port of GenA_Paper_charts/SC26/randomize_csv.py.

    The source script shuffles Code_Generation_KV_random.csv in place with
    `df.sample(frac=1)` -- no random_state, so it is not reproducible run
    to run. This port adds `random_state=seed` (documented deviation) and
    writes the shuffled copy to a temp file rather than a repo-tracked
    "_random" CSV, so nothing not explicitly owned by this task gets
    created on disk; the temp file is deterministic given `seed` and is
    regenerated (not reused) on every invocation.
    """
    import tempfile

    df = pd.read_csv(trace_path)
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    fd, tmp_name = tempfile.mkstemp(prefix="code_generation_kv_random_", suffix=".csv")
    os.close(fd)
    tmp_path = Path(tmp_name)
    shuffled.to_csv(tmp_path, index=False)
    return tmp_path


def build_request_queue(
    trace_path: Path, kv_retrieval: bool, args: argparse.Namespace
) -> List:
    """One seeded Poisson-arrival request queue, shared (via deepcopy) by
    every config in this use case's sweep -- same request stream in, so
    differences across configs are attributable to the deployment, not to
    randomness in the workload."""
    req_queue = PoissonDistribution(
        rps=args.rps,
        sim_time=args.sim_time,
        trace_file=str(trace_path),
        num_requests=args.num_requests,
        rand_seed=args.seed,
    ).request_queue
    _clip_request_queue(req_queue)
    if kv_retrieval:
        # Matches experiment_runner.py: attach each request's KV_Tokens
        # (cached-prefix length) as past_context, by trace row position.
        # Reads from `trace_path` -- the same (possibly shuffled) file the
        # queue above was built from, so requests and their KV_Tokens stay
        # row-aligned after shuffling.
        token_data = pd.read_csv(trace_path)
        for req, past_context in zip(req_queue, token_data["KV_Tokens"]):
            req.past_context = int(past_context)
    return req_queue


def _row_prefill_decode(row) -> tuple:
    prefill_str, decode_str = row["Parallelism"].split("-")
    return ast.literal_eval(prefill_str), ast.literal_eval(decode_str)


def build_serving_name(row) -> str:
    """Unique, human-readable id for one search-space row.

    Unlike the ported scripts' scheduler-config name (which only encodes
    batching method + client counts, and can collide across different
    hardware/TP choices that happen to produce the same client count), this
    embeds every field distinguishing a row, so the (UseCase, Serving Name)
    resumability key below is unambiguous.
    """
    return (
        f"{row['Hardware']}|{row['Parallelism']}|{row['Batching_Strategy'].name}"
        f"|bs{row['Max_Batch_Size']}|cs{row['Chunk_Size']}"
    )


def select_fast_eval_subset(df: pd.DataFrame, target_n: int = 150) -> pd.DataFrame:
    """Deterministic ~`target_n`-row Pareto-representative subset of the
    full search space.

    Rule: stratify rows by (vendor group, batching strategy, and -- for
    disaggregated rows only -- a low/mid/high tercile of the prefill:decode
    device ratio), then within every stratum always keep the min- and
    max-Cost row (the axis extremes: cheapest and priciest deployment of
    that vendor/strategy/ratio combination), and fill the remaining budget
    with Cost-sorted, evenly spaced interior samples, largest strata first.
    No RNG is used, so the same search-space DataFrame always yields the
    same subset -- deterministic reproducibility, same as the rest of T3.
    """
    if len(df) <= target_n:
        return df.reset_index(drop=True)

    df = df.reset_index(drop=True)
    is_disagg = df["Batching_Strategy"] == BatchingMethod.DISAGGREGATED
    vendor = [vendor_of_config(hw, bool(d)) for hw, d in zip(df["Hardware"], is_disagg)]

    def _pd_ratio(row) -> float:
        if row["Batching_Strategy"] != BatchingMethod.DISAGGREGATED:
            return np.nan
        prefill_p, decode_p = _row_prefill_decode(row)
        p_devices = prefill_p["TP"] * prefill_p["PP"] * prefill_p["DP"]
        d_devices = decode_p["TP"] * decode_p["PP"] * decode_p["DP"]
        return p_devices / d_devices

    def _ratio_bucket(x: float) -> str:
        if np.isnan(x):
            return "n/a"
        return "low" if x < 1 else ("mid" if x == 1 else "high")

    pd_ratio = df.apply(_pd_ratio, axis=1)
    ratio_bucket = pd_ratio.map(_ratio_bucket)
    stratum = list(zip(vendor, df["Batching_Strategy"].astype(str), ratio_bucket))
    strata: Dict[tuple, List[int]] = {}
    for idx, key in enumerate(stratum):
        strata.setdefault(key, []).append(idx)
    ordered_strata = sorted(strata.items(), key=lambda kv: -len(kv[1]))

    selected: List[int] = []
    seen = set()

    def _add(idx: int) -> None:
        if idx not in seen:
            seen.add(idx)
            selected.append(idx)

    # Pass 1: every stratum's Cost extremes (cheapest + priciest).
    for _, idxs in ordered_strata:
        ordered = df.loc[idxs].sort_values("Cost").index.tolist()
        _add(ordered[0])
        if len(ordered) > 1:
            _add(ordered[-1])

    # Pass 2: fill remaining budget with evenly spaced interior samples,
    # proportional to each stratum's share of the full space.
    remaining_interior = {
        key: [i for i in df.loc[idxs].sort_values("Cost").index.tolist() if i not in seen]
        for key, idxs in ordered_strata
    }
    remaining_budget = target_n - len(selected)
    if remaining_budget > 0:
        for key, _ in ordered_strata:
            if remaining_budget <= 0:
                break
            interior = remaining_interior[key]
            if not interior:
                continue
            take = max(1, round(remaining_budget * len(interior) / max(1, len(df))))
            take = min(take, len(interior), remaining_budget)
            positions = sorted(set(np.linspace(0, len(interior) - 1, take).round().astype(int)))
            for pos in positions:
                _add(interior[pos])
            remaining_interior[key] = [i for i in interior if i not in seen]
            remaining_budget = target_n - len(selected)

    # Pass 3: proportional rounding in pass 2 can undershoot `target_n`; top
    # up round-robin (largest strata first) from whatever interior rows are
    # still unpicked until the budget is exhausted or nothing remains.
    progress = True
    while remaining_budget > 0 and progress:
        progress = False
        for key, _ in ordered_strata:
            if remaining_budget <= 0:
                break
            interior = remaining_interior[key]
            if not interior:
                continue
            pick = interior[len(interior) // 2]
            _add(pick)
            remaining_interior[key] = [i for i in interior if i not in seen]
            remaining_budget = target_n - len(selected)
            progress = True

    return df.loc[sorted(selected)].reset_index(drop=True)


def run_one_config(row, base_req_queue: List, model: str, max_sim_time: float) -> Dict:
    """Simulate one deployment_space row and summarize the result.

    Ports `run_simulation` from ISCA26/Batching_method_comparisions.py,
    inlined and simplified for T3's fixed PP=1 / TP in {1,2} search: builds
    the coordinator directly from the row's Hardware/Parallelism instead of
    going through that module's `UsecaseExperimentConfig` /
    `schedulerExperimentConfigs` indirection layer.

    Unlike the ported scripts (which persist full per-request length/latency
    lists to CSV, later re-parsed by plot_sc_results.py's `_flatten` /
    `compute_ttft_percentiles`), only scalar summary columns are returned:
    the percentiles are computed here, directly from the live simulator
    objects, using the exact same definition -- flatten and merge TTFT from
    every *ongoing* request still in `engine.scheduler.running` at
    simulation end (`req.data[0]` already populated, i.e. it has a first
    token, but the request itself hasn't finished) together with TTFT from
    every *completed* request, then take percentiles over the union. A
    TTFT-only-from-completed-requests computation (this script's first
    iteration) systematically *understates* P99 for saturated/overloaded
    configs, where most in-flight requests never finish inside the sim
    window; this is the correctness fix the author's confirmed reference
    (compute_ttft_percentiles) requires.
    """
    req_queue = deepcopy(base_req_queue)
    is_disagg = row["Batching_Strategy"] == BatchingMethod.DISAGGREGATED

    if is_disagg:
        prefill_hw, decode_hw = row["Hardware"].split("-")
        prefill_p, decode_p = _row_prefill_decode(row)
        prefill_platform = PlatformConfig(
            device=prefill_hw, tensor_parallel_size=prefill_p["TP"],
            pipeline_parallel_size=prefill_p["PP"], model=model,
        )
        decode_platform = PlatformConfig(
            device=decode_hw, tensor_parallel_size=decode_p["TP"],
            pipeline_parallel_size=decode_p["PP"], model=model,
        )
        num_clients = prefill_p["DP"] + decode_p["DP"]
        coordinator = MISTCoordinatorDisagg(
            request_queue_distr=req_queue,
            model=model,
            num_llm_engines=num_clients,
            num_prefill_engines=prefill_p["DP"],
            num_decode_engines=decode_p["DP"],
            num_mixed_engines=0,
            platform=prefill_platform,
            decode_platform=decode_platform,
            convert_to_mixed_engine=False,
            logging_file=None,
            max_sim_time=max_sim_time,
            max_pending_prompt_tokens=320_000,
        )
    else:
        parallelism = row["Parallelism"]
        platform = PlatformConfig(
            device=row["Hardware"], tensor_parallel_size=parallelism["TP"],
            pipeline_parallel_size=parallelism["PP"], model=model,
        )
        num_clients = parallelism["DP"]
        coordinator = MISTCoordinator(req_queue, logging_file=None, max_sim_time=max_sim_time)
        decode_only_cache: Dict = {}
        mixed_batch_cache: Dict = {}
        scheduler_config = SchedulerConfig(
            batching_method=row["Batching_Strategy"],
            chunk_size=row["Chunk_Size"],
            max_batch_size=row["Max_Batch_Size"],
        )
        for _ in range(num_clients):
            coordinator.add_engine(
                LLMEngine(
                    model=model,
                    engine_types=[EngineType.PREFILL, EngineType.DECODE],
                    scheduler_config=scheduler_config,
                    platform=platform,
                    decode_only_cache=decode_only_cache,
                    mixed_batch_cache=mixed_batch_cache,
                ),
                [EngineType.PREFILL, EngineType.DECODE],
            )

    coordinator.run_sim()

    # Port of plot_sc_results.py's compute_ttft_percentiles(Ongoing_TTFT_
    # latencies, TTFT_latencies): merge TTFT from requests still mid-decode
    # (ongoing) with TTFT from fully completed requests, then take
    # percentiles over the union. See run_one_config's docstring.
    # NOTE: `.GenA_engines` (not `.engines`) is an instance attribute baked
    # into the simulator's own Coordinator class, not something an
    # import-level MIST rename alias can cover -- it still carries the
    # pre-release name and will need updating here if/when the simulator
    # renames it upstream.
    ongoing_ttft = [
        req.data[0].finished_time - req.metrics.arrival_time
        for engine in coordinator.GenA_engines
        for req in engine.scheduler.running
        if len(req.data) > 0
    ]
    completed_ttft = [
        req.data[0].finished_time - req.metrics.arrival_time
        for req in coordinator.completed_requests if len(req.data) > 0
    ]
    ttft_latencies = np.array(ongoing_ttft + completed_ttft)
    stats = coordinator.global_stats
    if len(ttft_latencies):
        percentiles = {f"TTFT_p{p}": float(np.percentile(ttft_latencies, p)) for p in (50, 90, 95, 99)}
    else:
        percentiles = {f"TTFT_p{p}": np.nan for p in (50, 90, 95, 99)}

    energy_used = sum(engine.energy_consumed for engine in coordinator.GenA_engines)
    cost_per_hour = row["Cost"]
    tokens_per_dollar = stats.output_throughput / cost_per_hour if cost_per_hour else np.nan

    return {
        "Serving Name": build_serving_name(row),
        "Batching_Strategy": row["Batching_Strategy"].name,
        "Hardware": row["Hardware"],
        "Parallelism": row["Parallelism"],
        "Max_Batch_Size": row["Max_Batch_Size"],
        "Chunk_Size": row["Chunk_Size"],
        "Cost": cost_per_hour,
        "HW_Combination": row["HW_Combination"],
        "Vendor": vendor_of_config(row["Hardware"], is_disagg),
        "Energy_Used": energy_used,
        "TTFT_avg": stats.TTFT,
        "TPOT_avg": stats.TPOT,
        "rps": stats.rps,
        **percentiles,
        "output_throughput": stats.output_throughput,
        # NOTE: for KV-retrieval (codegen) rows this can come out negative.
        # MIST's EngineMetrics.total_token_throughput sums
        # (input_len - past_context - remaining_prefill_tokens) per
        # request, which is meant to count "newly processed" prompt tokens
        # for a request mid-decode, but is negative whenever past_context
        # (KV_Tokens, often tens of thousands) exceeds input_len (a few
        # hundred, for this trace) -- an upstream accounting quirk, not a
        # bug in this script. `tokens_per_dollar` below is computed from
        # `output_throughput` (generated tokens/s only) instead, matching
        # the paper's actual objective, so it is unaffected.
        "total_token_throughput": stats.total_token_throughput,
        "tokens_per_dollar": tokens_per_dollar,
        "num_completed": len(coordinator.completed_requests),
        "num_ttft_observed": len(ttft_latencies),
    }


def _out_path(filename: str, args: argparse.Namespace) -> Path:
    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / filename
    return result_path("T3", filename)


def run_use_case(use_case: str, args: argparse.Namespace) -> int:
    """Enumerate, (optionally subset,) simulate, and write results for one
    use case. Returns 0 on success, 1 if any config failed to simulate."""
    cfg = USE_CASES[use_case]
    base_trace_path = TRACE_DIR / cfg["trace_file"]
    if not base_trace_path.exists():
        raise SystemExit(
            f"Missing input trace: {base_trace_path}\n"
            "This should have been copied into data/traces/ already; see README.md."
        )

    if cfg["shuffle"]:
        trace_path = _seeded_shuffle_trace(base_trace_path, args.seed)
        print(f"[{use_case}] seeded-shuffled {base_trace_path.name} (seed={args.seed}) -> {trace_path}")
    else:
        trace_path = base_trace_path

    trace_df = pd.read_csv(trace_path)
    avg_prompt_tokens = trace_df["ContextTokens"].mean()
    avg_decode_tokens = trace_df["GeneratedTokens"].mean()
    memory_required = estimate_memory_required_gb(MODEL, max_context_tokens=MAX_CONTEXT_TOKENS)

    search_df = get_search_space(
        num_devices=args.num_devices,
        Hardware=SKUS,
        Batching_Strategy=[BatchingMethod.CHUNKED, BatchingMethod.DISAGGREGATED],
        Max_Batch_Size=[128],
        Chunk_Size=[2048],
        price_per_hour=PRICE_PER_HOUR,
        memory_required=memory_required,
        max_degree_constraints={"PP": 1, "TP": 2},
        avg_prompt_tokens=avg_prompt_tokens,
        avg_decode_tokens=avg_decode_tokens,
    )
    search_space_path = _out_path(f"{use_case}_search_space.csv", args)
    search_df.to_csv(search_space_path, index=False)
    print(f"[{use_case}] enumerated {len(search_df)} configs -> {search_space_path}")

    if args.fast_eval:
        # MIST_T3_FAST_EVAL_N is a testing-only knob (e.g. for a quick
        # smoke run of this script); the shipped default target is 150.
        target_n = int(os.environ.get("MIST_T3_FAST_EVAL_N", "150"))
        run_df = select_fast_eval_subset(search_df, target_n=target_n)
        print(
            f"[{use_case}] --fast-eval: simulating {len(run_df)}/{len(search_df)} configs "
            "(curated Pareto-representative subset, see select_fast_eval_subset)"
        )
    else:
        run_df = search_df

    print(
        f"[{use_case}] building request queue: rps={args.rps} sim_time={args.sim_time}ms "
        f"num_requests={args.num_requests} seed={args.seed}"
    )
    base_req_queue = build_request_queue(trace_path, cfg["kv_retrieval"], args)
    print(f"[{use_case}] queue has {len(base_req_queue)} requests")
    if cfg["shuffle"]:
        trace_path.unlink(missing_ok=True)  # temp file, no longer needed

    results_path = _out_path(f"{use_case}_results.csv", args)
    completed_keys = set()
    if results_path.exists():
        try:
            existing = pd.read_csv(results_path)
            if not existing.empty:
                completed_keys = set(zip(existing["UseCase"], existing["Serving Name"]))
                print(f"[{use_case}] resuming: {len(completed_keys)} configs already in {results_path}")
        except Exception as exc:
            print(f"[{use_case}] warning: could not read existing {results_path}: {exc}")
    else:
        pd.DataFrame(columns=RESULT_COLUMNS).to_csv(results_path, index=False)

    usecase_label = f"{use_case}-{args.rps}rps"
    jobs = [
        row for _, row in run_df.iterrows()
        if (usecase_label, build_serving_name(row)) not in completed_keys
    ]
    print(f"[{use_case}] {len(jobs)}/{len(run_df)} configs remaining to simulate")

    failures: List[str] = []
    if jobs:
        with _quiet_stdout(), concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_row = {
                executor.submit(run_one_config, row, base_req_queue, MODEL, args.sim_time): row
                for row in jobs
            }
            for future in concurrent.futures.as_completed(future_to_row):
                row = future_to_row[future]
                name = build_serving_name(row)
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - surfaced to the user
                    print(f"  FAILED {name}: {exc}", file=sys.stderr)
                    failures.append(name)
                    continue
                result["UseCase"] = usecase_label
                pd.DataFrame([result], columns=RESULT_COLUMNS).to_csv(
                    results_path, mode="a", header=False, index=False
                )
                ttft99 = result["TTFT_p99"]
                print(
                    f"  done {result['Serving Name']}: "
                    f"TTFT_p99={ttft99:.1f}ms tokens/s={result['output_throughput']:.1f} "
                    f"tokens/s/$={result['tokens_per_dollar']:.2f}",
                    file=sys.stderr,
                )

    results_df = pd.read_csv(results_path)
    results_df = results_df[results_df["UseCase"] == usecase_label]
    slo_ms = cfg["ttft_slo_ms"]
    valid = results_df[results_df["TTFT_p99"] <= slo_ms]
    print(f"\n[{use_case}] {len(results_df)} configs simulated, {len(valid)} meet TTFT P99 <= {slo_ms} ms SLO")
    if len(valid):
        top5 = valid.sort_values("tokens_per_dollar", ascending=False).head(5)
        print(f"[{use_case}] top-5 by tokens/s/$:")
        with pd.option_context("display.max_colwidth", 60, "display.width", 160):
            print(
                top5[[
                    "Hardware", "Parallelism", "Batching_Strategy",
                    "TTFT_p99", "output_throughput", "Cost", "tokens_per_dollar",
                ]].to_string(index=False)
            )

    return 1 if failures else 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--use-case", choices=["chat", "codegen", "all"], default="all")
    parser.add_argument(
        "--fast-eval", action="store_true",
        help="Restrict to a curated ~150-config Pareto-representative subset of the "
             "500-900+ enumerated configs (see select_fast_eval_subset).",
    )
    parser.add_argument("--rps", type=float, default=100.0, help="Injection rate (default: 100).")
    parser.add_argument("--sim-time", type=float, default=60_000.0, help="Simulation window, ms (default: 60000).")
    parser.add_argument(
        "--num-requests", type=int, default=200,
        help="Request queue length (default: 200 -- matches the '...200reqs.csv' filenames "
             "the confirmed reference plotting scripts read; NOT experiment_runner.py's "
             "current default of 10000, which is the author's in-progress iteration).",
    )
    parser.add_argument("--num-devices", type=int, default=8, help="Cluster size, accelerators (default: 8).")
    parser.add_argument("--seed", type=int, default=259, help="Poisson-arrival random seed (default: 259).")
    parser.add_argument("--workers", type=int, default=2, help="ThreadPoolExecutor width (default: 2).")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Directory to write CSVs to (default: results/T3/, via mist_charts.paths.result_path).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    use_cases = list(USE_CASES) if args.use_case == "all" else [args.use_case]
    print(
        f"T3 sweep: use_cases={use_cases} fast_eval={args.fast_eval} rps={args.rps} "
        f"sim_time={args.sim_time}ms num_requests={args.num_requests} "
        f"num_devices={args.num_devices} seed={args.seed} workers={args.workers}"
    )
    exit_code = 0
    for use_case in use_cases:
        exit_code = max(exit_code, run_use_case(use_case, args))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
