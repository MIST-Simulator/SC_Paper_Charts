#!/usr/bin/env python3
"""Reproduce Fig. 7(a)-(b) and Fig. 8 of the MIST SC26 paper: heterogeneous
hardware / parallelism / batching design-space search for Qwen3-32B, for two
use cases built from the SAME GitHub-Code trace -- "chat conversation" (that
trace without KV retrieval, TTFT P99 < 200 ms) and "code generation, high
prefix-KV reuse" (a seeded shuffle of the same trace, with KV retrieval,
TTFT P99 < 1.5 s) -- optimizing generated tokens/s/$ among SLO-meeting
configs.

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

Both use cases read code_generation_kv.csv, not narrativeqa.csv, and
"chat"'s SLO is 200ms rather than the paper's stated 250ms -- both match
the plotting code confirmed to have produced the camera-ready figures, not
a mistake. See docs/FINDINGS.md#t3 for the full trace/SLO writeup, the
AIConfigurator-vs-MIST runtime-backend substitution, the gb300 database
fallback below, and the chunked_moddeling failures on long decode contexts.

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
    """Alias GEMMQuantMode.bfloat16 (and FMHA/KVCache/MoE) back onto the
    already-imported aiconfigurator enums; GenZ/db.py still references the
    pre-rename name. See docs/FINDINGS.md ("GenZ GEMMQuantMode rename").
    No-op if GenZ/aiconfigurator are unavailable or already compatible;
    edits no file on disk.
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
    vLLM one is missing or incomplete (gb300's ships INCOMPLETE.txt, which
    otherwise crashes every gb300 config). Safe because MIST runs the
    database in DatabaseMode.SOL, where it supplies only system_spec, not
    timing. Remove once MIST threads a `backend` param through `System`.
    See docs/FINDINGS.md#t3 ("Runtime-DB backend fallback for gb300").
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

# "chat" reads the base trace unshuffled, without KV retrieval; "codegen"
# reads a seeded shuffle of it (_seeded_shuffle_trace), with KV retrieval.
# See the module docstring for why both share this trace and the 200ms SLO.
BASE_TRACE_FILE = "code_generation_kv.csv"
USE_CASES: Dict[str, Dict] = {
    "chat": dict(trace_file=BASE_TRACE_FILE, ttft_slo_ms=200.0, kv_retrieval=False, shuffle=False),
    "codegen": dict(trace_file=BASE_TRACE_FILE, ttft_slo_ms=1500.0, kv_retrieval=True, shuffle=True),
}

# Trace column semantics (confirmed authoritative by the paper's author):
# ContextTokens/GeneratedTokens are prefill/decode tokens; KV_Tokens
# (codegen only) is reused prefix KV, attached as req.past_context, never
# as prefill. Matches experiment_runner.py's request construction.
#
# The 8-SKU list and 100RPS/200req/TP<=2 run config here are the paper's
# Sec. 5.1 configuration, not aac3254's experiment_runner.py defaults
# (a 5-SKU list with no Nvidia/Etched, and an in-progress 10000-request
# iteration) -- see docs/FINDINGS.md#t3 ("Hardware list and run-config
# discrepancies").

# Effectively inert now that both use cases read code_generation_kv.csv
# (max ContextTokens 18,517); left as a defensive guard against
# MISTCoordinatorDisagg's hardcoded max_total_tokens_per_sample=128_000,
# which silently drops any longer request.
MAX_CONTEXT_TOKENS = 120_000

BITS = "fp8"


# The author's schema (GenA_Paper_charts/SC26/experiment_runner.py's
# `columns` list, sourced from `run_simulation`'s return value): what
# plot_T3.py's ported `load_and_process` reads. "UseCase" and "Serving
# Name" are concatenated strings (built by `build_usecase`/
# `build_serving_name` below) that the plotting code regex-parses for
# hardware, parallelism, batching strategy and node counts. Everything
# else -- Cost, TTFT_P99, per-vendor category, etc. -- is derived
# downstream by the plotting code, not stored here, exactly as in the
# author's own CSVs.
RESULT_COLUMNS = [
    "UseCase", "Serving Name", "Energy Used", "TTFT", "TPOT", "rps",
    "T50_latency", "T90_latency", "T95_latency", "T99_latency",
    "interactivity", "output_throughput", "total_token_throughput",
    "Input Lens", "Output Lens", "Running Input Lens", "Running Output Lens",
    "Ongoing_TTFT_latencies", "TTFT_latencies", "latencies",
]


@contextlib.contextmanager
def _quiet_stdout():
    """Mute stdout at the file-descriptor level for the sweep's duration:
    MIST/GenZ print a line per batching decision / simulated second,
    unreadable across a multi-threaded sweep of hundreds of configs, and
    `contextlib.redirect_stdout` isn't thread-safe. Status lines go to
    stderr instead.
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
    """Seeded port of GenA_Paper_charts/SC26/randomize_csv.py, which
    shuffled in place with an unseeded `df.sample(frac=1)`. This adds
    `random_state=seed` and writes to a regenerated temp file instead of a
    repo-tracked "_random" CSV.
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


# Fixed disaggregated-engine params this script always simulates with (see
# run_one_config's MISTCoordinatorDisagg call below) -- baked into
# build_serving_name's DISAGGREGATED naming since the author's own
# `schedulerExperimentConfigs.name` embeds them literally.
_DISAGG_MIXED = 0
_DISAGG_CONVERT = False
_DISAGG_DIFF_DECODE = True
_DISAGG_MAX_PENDING_PROMPT_TOKENS = 320_000


def build_usecase(row, rps: float) -> str:
    """Port of experiment_runner.py's `exp_name` ("Poisson - {rps}RPS -
    {hardware}"). `row["Hardware"]` is already "<prefill>-<decode>" for
    disaggregated rows (deployment_space.get_search_space), so one formula
    covers both batching strategies. `rps:g` drops a trailing ".0" the way
    an un-passed (int-valued) argparse default would -- matching the
    author's real CSVs' "100RPS", not "100.0RPS".
    """
    return f"Poisson - {rps:g}RPS - {row['Hardware']}"


def build_serving_name(row) -> str:
    """Port of ISCA26/Batching_method_comparisions.py's
    `schedulerExperimentConfigs.name` -- what plot_sc_results.py's
    `parse_serving_name` regex-parses for node counts. Doesn't encode TP
    (neither does the source): safe here only because our search space
    fixes num_devices=8 and PP=1, so TP in {1,2} implies a distinct DP
    (client count) per row and thus a distinct name; see
    docs/FINDINGS.md#t3 for this inherited limitation.
    """
    strategy = row["Batching_Strategy"]
    if strategy == BatchingMethod.DISAGGREGATED:
        prefill_p, decode_p = _row_prefill_decode(row)
        num_clients = prefill_p["DP"] + decode_p["DP"]
        return (
            f"{strategy}_{num_clients}_{prefill_p['DP']}_{decode_p['DP']}_{_DISAGG_MIXED}"
            f"_Convert{_DISAGG_CONVERT}_DiffDecode:{_DISAGG_DIFF_DECODE}"
            f"_{_DISAGG_MAX_PENDING_PROMPT_TOKENS}tokens"
        )
    num_clients = row["Parallelism"]["DP"]
    return f"{strategy}_{num_clients}_{row['Chunk_Size']}"


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


def run_one_config(row, base_req_queue: List, model: str, max_sim_time: float, rps: float) -> Dict:
    """Simulate one deployment_space row and return one row of the author's
    20-column schema (RESULT_COLUMNS).

    Inlined port of `run_simulation` (ISCA26/Batching_method_comparisions.py)
    for T3's fixed PP=1 / TP in {1,2} search, building the coordinator
    directly from the row instead of that module's config-indirection layer.
    The list-valued fields (Input/Output Lens, Ongoing_TTFT_latencies,
    TTFT_latencies, latencies) are ported verbatim from `run_simulation`'s
    return value -- raw per-request/per-engine data, not precomputed
    percentiles -- so `plot_T3.py`'s ported `compute_ttft_percentiles` /
    `load_and_process` derive the same numbers the author's plotting code
    would from a real experiment_runner.py CSV.
    """
    req_queue = deepcopy(base_req_queue)
    is_disagg = row["Batching_Strategy"] == BatchingMethod.DISAGGREGATED

    if is_disagg:
        prefill_hw, decode_hw = row["Hardware"].split("-")
        prefill_p, decode_p = _row_prefill_decode(row)
        prefill_platform = PlatformConfig(
            device=prefill_hw, tensor_parallel_size=prefill_p["TP"],
            pipeline_parallel_size=prefill_p["PP"], model=model, bits=BITS,
        )
        decode_platform = PlatformConfig(
            device=decode_hw, tensor_parallel_size=decode_p["TP"],
            pipeline_parallel_size=decode_p["PP"], model=model, bits=BITS,
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
            pipeline_parallel_size=parallelism["PP"], model=model, bits=BITS,
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

    # The coordinator's engine list: `engines` in the released simulator,
    # `GenA_engines` in a pre-release checkout. An import alias cannot cover
    # an instance attribute, so accept either.
    engines = getattr(coordinator, "engines", None)
    if engines is None:
        engines = coordinator.GenA_engines

    completed = coordinator.completed_requests
    stats = coordinator.global_stats
    energy_used = sum(engine.energy_consumed for engine in engines)

    # Port of run_simulation's return-value construction (ISCA26/
    # Batching_method_comparisions.py), verbatim other than reading
    # `engines` (see above) instead of `coordinator.GenA_engines` directly,
    # and guarding the TTFT list comprehensions with `len(req.data) > 0`
    # (a request with zero data beats -- possible for one still waiting on
    # its very first scheduling decision when the sim window ends -- would
    # otherwise raise IndexError; the author's `run_simulation` has no such
    # guard for its completed-request TTFT list, but every one of ours is
    # already defensive here, so this keeps that same safety without
    # changing the schema).
    # float(...): req.input_len/output_len are numpy.float64 (from the
    # trace CSV's dtype), whose repr() is "np.float64(211.0)" under
    # numpy>=2 -- not valid Python-literal syntax, so
    # ast.literal_eval would fail to parse this column back out of the
    # CSV. Casting to plain float here keeps the list's str()/to_csv
    # serialization ast.literal_eval-compatible, matching the author's
    # real CSVs (plain floats throughout).
    input_lens = [float(req.input_len) for req in completed]
    output_lens = [float(req.output_len) for req in completed]
    running_input_lens = [[float(req.input_len) for req in engine.scheduler.running] for engine in engines]
    running_output_lens = [[req.gen_tokens for req in engine.scheduler.running] for engine in engines]
    # float(...) here too -- req.metrics.finished_time is a numpy.float64
    # (unlike req.data[*].finished_time, a plain float), so `latencies`
    # would otherwise hit the same ast.literal_eval-incompatible
    # "np.float64(...)" repr issue as Input/Output Lens above.
    ongoing_ttft_latencies = [
        [
            float(req.data[0].finished_time - req.metrics.arrival_time)
            for req in engine.scheduler.running if len(req.data) > 0
        ]
        for engine in engines
    ]
    ttft_latencies = [
        float(req.data[0].finished_time - req.metrics.arrival_time) for req in completed if len(req.data) > 0
    ]
    latencies = [float(req.metrics.finished_time - req.metrics.arrival_time) for req in completed]

    return {
        "UseCase": build_usecase(row, rps),
        "Serving Name": build_serving_name(row),
        "Energy Used": energy_used,
        "TTFT": stats.TTFT,
        "TPOT": stats.TPOT,
        "rps": stats.rps,
        "T50_latency": stats.T50_latency,
        "T90_latency": stats.T90_latency,
        "T95_latency": stats.T95_latency,
        "T99_latency": stats.T99_latency,
        "interactivity": stats.interactivity,
        "output_throughput": stats.output_throughput,
        # Can be negative for KV-retrieval (codegen) rows: MIST's
        # total_token_throughput sums (input_len - past_context -
        # remaining_prefill_tokens), which goes negative once past_context
        # (KV_Tokens) exceeds input_len -- an upstream accounting quirk,
        # also present (as negative total_token_throughput values) in the
        # author's own real codegen CSV.
        "total_token_throughput": stats.total_token_throughput,
        "Input Lens": input_lens,
        "Output Lens": output_lens,
        "Running Input Lens": running_input_lens,
        "Running Output Lens": running_output_lens,
        "Ongoing_TTFT_latencies": ongoing_ttft_latencies,
        "TTFT_latencies": ttft_latencies,
        "latencies": latencies,
    }


def _out_path(filename: str, args: argparse.Namespace) -> Path:
    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / filename
    return result_path("T3", filename)


def _flatten(lst) -> List:
    """Port of plot_sc_results.py's `_flatten` (recursively flatten
    arbitrarily nested lists) -- reused here only for this script's own
    console diagnostics (SLO count / top-5), not part of the schema.
    """
    out = []
    for item in lst:
        if isinstance(item, list):
            out.extend(_flatten(item))
        else:
            out.append(item)
    return out


def _ttft_p99(ongoing_str: str, ttft_str: str) -> float:
    """Console-diagnostics-only TTFT P99, computed the same way
    plot_sc_results.py's `compute_ttft_percentiles` will when plotting:
    literal-eval + flatten `Ongoing_TTFT_latencies` and `TTFT_latencies`,
    then take the 99th percentile of the union.
    """
    vals: List[float] = []
    for raw in (ongoing_str, ttft_str):
        try:
            parsed = ast.literal_eval(raw) if isinstance(raw, str) else raw
            if isinstance(parsed, list):
                vals.extend(_flatten(parsed))
        except Exception:
            pass
    vals = [v for v in vals if isinstance(v, (int, float)) and not np.isnan(v)]
    return float(np.percentile(vals, 99)) if vals else float("nan")


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

    jobs = [
        row for _, row in run_df.iterrows()
        if (build_usecase(row, args.rps), build_serving_name(row)) not in completed_keys
    ]
    print(f"[{use_case}] {len(jobs)}/{len(run_df)} configs remaining to simulate")

    failures: List[str] = []
    if jobs:
        with _quiet_stdout(), concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_row = {
                executor.submit(run_one_config, row, base_req_queue, MODEL, args.sim_time, args.rps): row
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
                pd.DataFrame([result], columns=RESULT_COLUMNS).to_csv(
                    results_path, mode="a", header=False, index=False
                )
                # Cost isn't part of the author's schema (it's derived
                # downstream by the plotting code); `row["Cost"]` is still
                # available here from deployment_space's own enumeration,
                # so console diagnostics can use it without storing it.
                cost = row["Cost"]
                tokens_per_dollar = result["output_throughput"] / cost if cost else float("nan")
                ttft99 = _ttft_p99(result["Ongoing_TTFT_latencies"], result["TTFT_latencies"])
                print(
                    f"  done {result['Serving Name']}: "
                    f"TTFT_p99={ttft99:.1f}ms tokens/s={result['output_throughput']:.1f} "
                    f"tokens/s/$={tokens_per_dollar:.2f}",
                    file=sys.stderr,
                )

    results_df = pd.read_csv(results_path)
    slo_ms = cfg["ttft_slo_ms"]
    ttft99_col = [
        _ttft_p99(o, t) for o, t in zip(results_df["Ongoing_TTFT_latencies"], results_df["TTFT_latencies"])
    ]
    results_df["_TTFT_p99"] = ttft99_col
    valid = results_df[results_df["_TTFT_p99"] <= slo_ms]
    print(f"\n[{use_case}] {len(results_df)} configs simulated, {len(valid)} meet TTFT P99 <= {slo_ms} ms SLO")
    if len(valid):
        # Join back to this run's search-space enumeration (`run_df`) for
        # Cost -- see the comment above; not stored in the results CSV.
        keyed = run_df.copy()
        keyed["_UseCase"] = keyed.apply(lambda r: build_usecase(r, args.rps), axis=1)
        keyed["_ServingName"] = keyed.apply(build_serving_name, axis=1)
        merged = valid.merge(
            keyed[["_UseCase", "_ServingName", "Hardware", "Batching_Strategy", "Cost"]],
            left_on=["UseCase", "Serving Name"], right_on=["_UseCase", "_ServingName"], how="left",
        )
        merged["_tokens_per_dollar"] = merged["output_throughput"] / merged["Cost"]
        top5 = merged.sort_values("_tokens_per_dollar", ascending=False).head(5)
        print(f"[{use_case}] top-5 by tokens/s/$:")
        with pd.option_context("display.max_colwidth", 60, "display.width", 160):
            print(
                top5[[
                    "Hardware", "Batching_Strategy",
                    "_TTFT_p99", "output_throughput", "Cost", "_tokens_per_dollar",
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
