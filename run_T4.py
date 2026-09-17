#!/usr/bin/env python3
"""Reproduce Fig. 10 (Case_Study_Memory_Cache.pdf): sweep five KV-cache
storage architectures across short/long KV contexts and private/shared
access patterns, on 128 clients (H100:TP2, 256 GPUs total) split across 4
racks, driven by real per-request sizes from the Azure conversational trace.

Ports ``4. Cache_storage_config_comparisions.ipynb`` (cell 7, 8, 14) and its
companion module ``Experiments/Memory_Storage_Comparisions.py``. See
docs/FINDINGS.md#t4 for the deviations from that notebook (Case D/E were
labelled backwards, the notebook's RNG seeding was disabled, and it never
actually replayed the Azure trace) and the Case D DCN-bandwidth conflict
that ``--dcn-bandwidth-gbps`` exposes (default 1 GB/s reproduces the
published figure; 128 GB/s, Table 2's stated value, flips Case D to
strictly dominate Case C).

Writes:
    results/T4/per_request.csv  -- one row per completed request
    results/T4/summary.csv      -- P50/P90/P99 per (case, context_len, scenario)
"""

import argparse
import concurrent.futures
import contextlib
import dataclasses
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

from mist_charts.memory_configs import CASE_ORDER, STORAGE_ARCHITECTURES, sharing_group_size
from mist_charts.mist_api import (
    BatchingMethod,
    EngineType,
    MISTCoordinator,
    KVRetrievalEngine,
    LLMEngine,
    MemoryCacheConfig,
    PlatformConfig,
    Request,
    RequestStage,
    SchedulerConfig,
    SingleCacheConfig,
)
from mist_charts.paths import TRACE_DIR, result_path

MODEL = "meta-llama/Llama-3.1-70B"
DEVICE = "H100_GPU"
TP_SIZE = 2

TRACE_NAME = "AzureLLMInferenceTrace_conv.csv"

# Matches the notebook's `max_sim_time = 10000` (ms) simulation window.
MAX_SIM_TIME_MS = 10_000.0
# Matches the notebook's calibrated `rps=240` for 128 clients (cell 8);
# scaled proportionally so --num-clients keeps the same per-client load.
AGGREGATE_RPS_AT_128_CLIENTS = 240.0
REFERENCE_NUM_CLIENTS = 128

# "Heavily skewed" access pattern: number of favored clients and the spread
# of the Gaussian bump around each one (cell 8: size=8, spread=0.75).
NUM_SKEWED_CENTERS = 8
SKEW_SPREAD = 0.75
# "Uniform" access pattern: a narrow bump centered on every client washes
# out to a flat distribution (cell 8: np.arange(num_clients), spread=0.1).
UNIFORM_SPREAD = 0.1

SCENARIOS = ["private", "shared"]


@contextlib.contextmanager
def _quiet_stdout():
    """Mute stdout at the file-descriptor level for the sweep's duration:
    GenZ prints a line per batching decision, unreadable across ~130
    engines x thousands of steps, and `contextlib.redirect_stdout` isn't
    thread-safe for a multi-threaded sweep. Status lines go to stderr
    instead (see `run_one_config`).
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


def _require_trace():
    trace_path = TRACE_DIR / TRACE_NAME
    if not trace_path.exists():
        raise SystemExit(
            f"Missing input trace: {trace_path}\n"
            "Run `bash scripts/download_traces.sh` first, or for local "
            "testing copy/symlink it into data/traces/."
        )
    return trace_path


def custom_skewed_random_probability(num_clients: int, skew_centers, spread: float) -> np.ndarray:
    """Verbatim port of the notebook's helper of the same name.

    Sums a Gaussian bump centered at each entry of ``skew_centers`` and
    normalizes, so passing every client index with a narrow spread yields an
    (approximately) uniform distribution, and passing a handful of indices
    with a wider spread yields a "heavily skewed" one.
    """
    x = np.arange(num_clients)
    probabilities = np.zeros(num_clients)
    for center in skew_centers:
        probabilities += np.exp(-((x - center) ** 2) / (2 * spread ** 2))
    probabilities /= probabilities.sum()
    return probabilities


def build_base_stream(
    trace_df: pd.DataFrame,
    num_clients: int,
    context_len: int,
    scenario: str,
    seed: int,
    aggregate_rps: float,
) -> List[Dict]:
    """Build the (case-independent) request stream for one (context_len,
    scenario) pair: real trace-sampled (input_len, output_len) pairs,
    Poisson arrivals at `aggregate_rps`, and a per-request "favorite
    client" from the scenario's access-skew distribution. The same stream
    feeds all five storage architectures for this pair (cell 14's nested
    loop in the source notebook).
    """
    scenario_idx = 0 if scenario == "private" else 1
    rng = np.random.default_rng((seed, context_len, scenario_idx))

    if scenario == "private":
        probs = custom_skewed_random_probability(
            num_clients, skew_centers=np.arange(num_clients), spread=UNIFORM_SPREAD
        )
    else:
        centers = rng.choice(num_clients, size=min(NUM_SKEWED_CENTERS, num_clients))
        probs = custom_skewed_random_probability(
            num_clients, skew_centers=centers, spread=SKEW_SPREAD
        )

    context_tokens = trace_df["ContextTokens"].to_numpy()
    generated_tokens = trace_df["GeneratedTokens"].to_numpy()
    n_rows = len(trace_df)

    stream = []
    arrival = 0.0
    req_id = 0
    while arrival < MAX_SIM_TIME_MS:
        row_idx = rng.integers(0, n_rows)
        client = int(rng.choice(num_clients, p=probs))
        stream.append(
            {
                "request_id": req_id,
                "input_len": int(context_tokens[row_idx]),
                "output_len": int(generated_tokens[row_idx]),
                "arrival_time": arrival,
                "client": client,
            }
        )
        req_id += 1
        arrival += rng.exponential(1000.0 / aggregate_rps)

    return stream


def realize_requests(stream: List[Dict], context_len: int, arch, num_clients: int, num_llm_engines: int):
    """Turn the shared base stream into MIST ``Request`` objects for one
    storage architecture: apply the short/long KV-context past_context (or,
    for Case E, inline recompute), and set engine_preference from the
    architecture's client-sharing group.
    """
    group_size = sharing_group_size(arch, num_clients)
    requests = []
    for item in stream:
        group_start = (item["client"] // group_size) * group_size
        group_engines = list(range(group_start, group_start + group_size))

        if arch.recompute_kv:
            # No cache stage at all: the KV context is recomputed as part
            # of prefill (matches the notebook's `req.input_len += past_len`
            # + skipping straight to the PREFILL stage for the recompute case).
            req = Request(
                request_id=item["request_id"],
                input_len=item["input_len"] + context_len,
                output_len=item["output_len"],
                arrival_time=item["arrival_time"],
                beam_size=1,
                stages=[RequestStage.PREFILL, RequestStage.DECODE],
            )
            req.engine_preference = group_engines
        else:
            req = Request(
                request_id=item["request_id"],
                input_len=item["input_len"],
                output_len=item["output_len"],
                arrival_time=item["arrival_time"],
                beam_size=1,
                stages=[RequestStage.CACHE_RETRIEVAL, RequestStage.PREFILL, RequestStage.DECODE],
                past_context=context_len,
            )
            req.engine_preference = [num_llm_engines] + group_engines

        requests.append(req)
    return requests


def run_one_config(
    case: str,
    arch,
    context_len: int,
    scenario: str,
    stream: List[Dict],
    num_clients: int,
) -> pd.DataFrame:
    """Run a single (case, context_len, scenario) simulation and return its
    per-request results as a DataFrame.

    ``arch`` is passed in (rather than looked up from
    ``STORAGE_ARCHITECTURES[case]``) so callers can override a field --
    namely Case D's DCN bandwidth, see ``--dcn-bandwidth-gbps``.
    """
    num_llm_engines = num_clients
    num_total_engines = num_llm_engines + 1  # +1 for the shared KVRetrievalEngine

    requests = realize_requests(stream, context_len, arch, num_clients, num_llm_engines)

    # Uniform src/dst network link: the cache->compute transfer latency and
    # bandwidth for this architecture, applied between every pair of engines
    # (matches the notebook's `connection_df` construction in cell 14/run_simulation).
    rows = []
    for src in range(num_total_engines):
        for dst in range(num_total_engines):
            if src == dst:
                rows.append([src, dst, 0.0, 0.0])
            else:
                rows.append([src, dst, arch.network_latency_ms, arch.network_bandwidth_gbps])
    connection_df = pd.DataFrame(rows, columns=["src", "dst", "latency(msec)", "BW(GB/s)"])

    coordinator = MISTCoordinator(
        requests,
        logging_file=None,
        max_sim_time=MAX_SIM_TIME_MS,
        network_file=connection_df,
    )

    # One shared PlatformConfig object for all client engines, matching the
    # notebook's run_simulation (perf: avoids re-deriving the GenZ system
    # model 128 times over).
    platform = PlatformConfig(
        device=DEVICE, tensor_parallel_size=TP_SIZE, pipeline_parallel_size=1, model=MODEL
    )
    scheduler_config_kwargs = dict(
        batching_method=BatchingMethod.CHUNKED,
        chunk_size=512,
        max_batch_size=128,
        max_num_batched_tokens=513_000,
        max_total_tokens_per_sample=513_000,
    )
    for _ in range(num_llm_engines):
        coordinator.add_engine(
            LLMEngine(
                model=MODEL,
                engine_types=[EngineType.PREFILL, EngineType.DECODE],
                scheduler_config=SchedulerConfig(**scheduler_config_kwargs),
                platform=platform,
            ),
            [EngineType.PREFILL, EngineType.DECODE],
        )

    if arch.recompute_kv:
        # Inert placeholder: no request ever reaches CACHE_RETRIEVAL for
        # this architecture, so its numbers are never read.
        cache_hierarchy = [SingleCacheConfig("n/a", 1, 128, 0.01, 1.0)]
    else:
        cache_hierarchy = [
            SingleCacheConfig(
                arch.cache_type,
                arch.capacity_gb,
                arch.cache_bandwidth_gbps,
                arch.cache_retrieval_latency_ms,
                1.0,
            )
        ]
    memory_cache = MemoryCacheConfig(cache_hierarchy)
    memory_cache.update_model(MODEL)
    coordinator.add_engine(
        KVRetrievalEngine(model=MODEL, platform=memory_cache),
        [EngineType.CACHE_RETRIEVAL],
    )

    coordinator.run_sim()

    records = []
    for req in coordinator.completed_requests:
        cache_metrics = req.stage_metrics.get("Cache Retrieval")
        if cache_metrics is not None and cache_metrics.engine_exit_time is not None:
            retrieval_latency = cache_metrics.engine_exit_time - cache_metrics.engine_entry_time
        else:
            retrieval_latency = 0.0
        ttft = req.data[0].finished_time - req.metrics.arrival_time if req.data else None
        end_to_end = req.metrics.finished_time - req.metrics.arrival_time
        records.append(
            {
                "case": case,
                "architecture": arch.label,
                "context_len": context_len,
                "scenario": scenario,
                "num_clients": num_clients,
                # Traces which DCN link speed this row was simulated with --
                # see --dcn-bandwidth-gbps / the Case D conflict note in
                # mist_charts/memory_configs.py.
                "dcn_bandwidth_gbps": arch.network_bandwidth_gbps if arch.dcn else None,
                "request_id": req.request_id,
                "input_len": req.input_len,
                "output_len": req.output_len,
                "arrival_time_ms": req.metrics.arrival_time,
                "ttft_ms": ttft,
                "cache_retrieval_latency_ms": retrieval_latency,
                "end_to_end_latency_ms": end_to_end,
            }
        )

    print(
        f"  [{case} | ctx={context_len} | {scenario}] "
        f"{len(records)} requests completed",
        file=sys.stderr,
    )
    return pd.DataFrame.from_records(records)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (case, context_len, scenario), group in df.groupby(["case", "context_len", "scenario"]):
        lat = group["end_to_end_latency_ms"].to_numpy()
        rows.append(
            {
                "case": case,
                # Pulled from the per-request rows (which record whatever
                # dcn_bandwidth_gbps the run actually used) rather than
                # STORAGE_ARCHITECTURES[case], so an overridden sweep is
                # labeled accurately.
                "architecture": group["architecture"].iloc[0],
                "context_len": context_len,
                "scenario": scenario,
                "num_requests": len(lat),
                "p50_ms": float(np.percentile(lat, 50)),
                "p90_ms": float(np.percentile(lat, 90)),
                "p99_ms": float(np.percentile(lat, 99)),
            }
        )
    summary = pd.DataFrame.from_records(rows).sort_values(["context_len", "scenario", "case"])
    return summary


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", nargs="+", default=["all"], choices=CASE_ORDER + ["all"],
        help="Storage architecture(s) to sweep (default: all five).",
    )
    parser.add_argument(
        "--context-len", nargs="+", type=int, default=[4096, 24576],
        help="KV context lengths to sweep, in tokens (default: 4096 24576).",
    )
    parser.add_argument(
        "--scenario", choices=["private", "shared", "both"], default="both",
        help="Cache access pattern: uniform ('private') or heavily skewed "
             "('shared') client popularity (default: both).",
    )
    parser.add_argument(
        "--num-clients", type=int, default=128,
        help="Number of client engines (H100:TP2 each) (default: 128).",
    )
    parser.add_argument("--seed", type=int, default=259, help="Random seed (default: 259).")
    parser.add_argument(
        "--workers", type=int, default=4,
        help="ThreadPoolExecutor width for running configs concurrently (default: 4).",
    )
    parser.add_argument(
        "--dcn-bandwidth-gbps", type=float, default=1.0,
        help=(
            "Case D's inter-rack DCN transfer bandwidth in GB/s (default: 1.0). "
            "KNOWN CONFLICT: Table 2 of the paper states this link runs at "
            "128 GB/s, but the notebook that produced the published Fig. 10 "
            "hard-codes it at 1 GB/s, and only 1 GB/s reproduces the "
            "published figure (Case D near-worst in both 'shared' panels). "
            "At 128 GB/s, Case D instead strictly dominates Case C, since it "
            "is the same storage tier load-balanced across all 4 racks "
            "instead of 1. Default matches the published figure; pass "
            "--dcn-bandwidth-gbps 128 to see the reversal. See "
            "docs/FINDINGS.md#t4 for the full writeup."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    trace_path = _require_trace()
    trace_df = pd.read_csv(trace_path)

    cases = CASE_ORDER if "all" in args.case else args.case
    scenarios = SCENARIOS if args.scenario == "both" else [args.scenario]
    aggregate_rps = AGGREGATE_RPS_AT_128_CLIENTS * (args.num_clients / REFERENCE_NUM_CLIENTS)

    # Case D's DCN bandwidth is a documented paper-vs-artifact conflict (see
    # mist_charts/memory_configs.py): override it here rather than mutating
    # the shared STORAGE_ARCHITECTURES table.
    architectures = dict(STORAGE_ARCHITECTURES)
    if "D" in cases:
        architectures["D"] = dataclasses.replace(
            STORAGE_ARCHITECTURES["D"], network_bandwidth_gbps=args.dcn_bandwidth_gbps
        )

    print(
        f"T4 sweep: cases={cases} context_len={args.context_len} "
        f"scenarios={scenarios} num_clients={args.num_clients} "
        f"aggregate_rps={aggregate_rps:.1f} seed={args.seed} workers={args.workers} "
        f"dcn_bandwidth_gbps={args.dcn_bandwidth_gbps if 'D' in cases else 'n/a'}"
    )

    # One base request stream per (context_len, scenario); shared across all
    # cases for a fair, apples-to-apples comparison of the five architectures.
    streams = {}
    for context_len in args.context_len:
        for scenario in scenarios:
            streams[(context_len, scenario)] = build_base_stream(
                trace_df, args.num_clients, context_len, scenario, args.seed, aggregate_rps
            )
            print(
                f"  built stream ctx={context_len} scenario={scenario}: "
                f"{len(streams[(context_len, scenario)])} requests"
            )

    jobs = [
        (case, context_len, scenario)
        for context_len in args.context_len
        for scenario in scenarios
        for case in cases
    ]

    results = []
    failures = []
    with _quiet_stdout(), concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_job = {
            executor.submit(
                run_one_config, case, architectures[case], context_len, scenario,
                streams[(context_len, scenario)], args.num_clients,
            ): (case, context_len, scenario)
            for case, context_len, scenario in jobs
        }
        for future in concurrent.futures.as_completed(future_to_job):
            job = future_to_job[future]
            try:
                results.append(future.result())
            except Exception as exc:  # pragma: no cover - surfaced to the user
                print(f"  [{job[0]} | ctx={job[1]} | {job[2]}] FAILED: {exc}", file=sys.stderr)
                failures.append(job)

    if not results:
        print("No configuration completed successfully.")
        return 1

    per_request = pd.concat(results, ignore_index=True)
    per_request_path = result_path("T4", "per_request.csv")
    per_request.to_csv(per_request_path, index=False)
    print(f"wrote {per_request_path} ({len(per_request)} rows)")

    summary = summarize(per_request)
    summary_path = result_path("T4", "summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")

    print("\nPer-config P50 / P90 / P99 end-to-end latency (ms):")
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(summary.to_string(index=False))

    if failures:
        print(f"\n{len(failures)} configuration(s) failed: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
