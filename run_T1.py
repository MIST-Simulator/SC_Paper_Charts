#!/usr/bin/env python3
"""Reproduces MIST SC26 Fig. 4(a-d): end-to-end latency validation.

Simulates the same 100-request ShareGPT workload on three platforms:

  (a) Qwen3-32B on L40Sx2, TP2
  (b) Llama3-70B on H100x8, TP8
  (c) Llama-3.1-8B on TPUv6e

and writes one CSV per platform under ``results/T1/`` with per-request TTFT
and end-to-end latency, plus the wall-clock time this script itself took to
run the simulation (used for the panel (d) runtime bar chart in plot_T1.py).

Every number in the written CSVs comes from actually running the MIST
simulator in this process -- nothing here is copied from the paper or from
a notebook. What's configurable is the *input* workload, not whether MIST
ran:

* Default: replays the fixed per-request arrival/length trace that vLLM
  was actually measured against (vendored under data/validation/T1/, see
  its README.md). This is what makes Fig. 4 a controlled comparison --
  MIST, the real vLLM run, Vidur, and LLMServingSim2.0 all see the exact
  same 100 requests, so overlaying their latency distributions answers the
  same question for every curve. Still fully reproducible: the trace file
  is fixed and MIST's simulation of it is deterministic.
* ``--resample-sharegpt``: an opt-in sensitivity-analysis mode that instead
  draws a fresh, seeded 100-request sample from the public ShareGPT
  dataset. Useful for asking "how sensitive is this latency distribution to
  which 100 ShareGPT conversations you happen to serve", but its output is
  *not* directly comparable to the vLLM/Vidur/LLMServingSim baselines,
  which were all measured on the fixed trace above -- see
  data/validation/T1/README.md for a worked example of just how large that
  gap can be (a different fixed seed shifted mean E2E latency by >2x at the
  same nominal RPS).
"""

import argparse
import contextlib
import io
import json
import os
import re
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from mist_charts.paths import TRACE_DIR, VALIDATION_DIR, result_path


def _patch_genz_quant_compat() -> None:
    """TEMPORARY compatibility shim for a version-skew bug in GenZ. Remove
    once upstream is fixed -- see KNOWN_ISSUES in data/validation/T1/README.md.

    The ``genz_llm`` package pinned in requirements.txt vendors an
    aiconfigurator submodule whose ``GEMMQuantMode`` / ``FMHAQuantMode`` /
    ``KVCacheQuantMode`` / ``MoEQuantMode`` enums no longer have a
    ``bfloat16`` member (upstream renamed it to ``float16``, the generic
    16-bit w16a16 mode), while GenZ's own ``GenZ/db.py`` (module-level
    ``bits_to_gemm_quants`` dict, plus default args on ``query_context_attention``
    / ``query_generation_attention`` / ``query_gemm``) still references
    ``.bfloat16`` at import/definition time. Without this shim, constructing
    *any* GenZ ``System`` (and hence any MIST ``vLLMPlatformConfig``) raises
    ``AttributeError`` immediately, for every platform. This is a pure
    monkeypatch executed from this script only -- it does not modify the
    installed GenZ/MIST packages, which live outside this repository and are
    intentionally left untouched; the real fix (aliasing ``.bfloat16`` back
    onto ``.float16``, or updating GenZ's own references) belongs upstream
    in GenZ-LLM-Analyzer's ``GenZ/db.py``.
    """
    try:
        import GenZ  # noqa: F401

        submodule_src = os.path.join(
            os.path.dirname(GenZ.__file__), "aiconfigurator", "src"
        )
        if submodule_src not in sys.path:
            sys.path.insert(0, submodule_src)
        from aiconfigurator.sdk import common as _aic_common

        for enum_name in (
            "GEMMQuantMode",
            "FMHAQuantMode",
            "KVCacheQuantMode",
            "MoEQuantMode",
        ):
            enum_cls = getattr(_aic_common, enum_name, None)
            if (
                enum_cls is not None
                and not hasattr(enum_cls, "bfloat16")
                and hasattr(enum_cls, "float16")
            ):
                enum_cls.bfloat16 = enum_cls.float16
    except Exception:
        # If GenZ isn't installed at all, let the real ImportError from
        # mist_charts.mist_api surface below with its actionable message.
        pass


_patch_genz_quant_compat()

from mist_charts.mist_api import (  # noqa: E402  (after the compat shim above)
    BatchingMethod,
    EngineType,
    MISTCoordinator,
    LLMEngine,
    PoissonDistribution,
    SchedulerConfig,
    TraceIngestion,
    mist_package_path,
    vLLMPlatformConfig,
)

MIST_RUNTIME_DIR = mist_package_path() / "Platforms" / "vllm_runtime_data"
SHAREGPT_PATH = TRACE_DIR / "ShareGPT_V3_unfiltered_cleaned_split.json"
_WORD_RE = re.compile(r"\S+")

# Platform configs, ported from the notebook that produced the published
# figure (GenA_Paper_charts/Validation/4.SC26/SimulatorComparision/
# share_gpt_vllm_tpu_validation.ipynb, cells 6/19/27). chunk_size=2048 and
# step_size=engine_kv_step_size=0 (always recompute exactly, never
# approximate from a cached bucket) match that notebook.
PLATFORMS = {
    "l40s": dict(
        label="Qwen3-32B on L40S:TP2",
        model="Qwen/Qwen2.5-32B",
        device="L40S_GPU",
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        vllm_df_path=MIST_RUNTIME_DIR / "Qwen3-32B_NVIDIA_L40S_2.csv",
        result_filename="mist_l40s.csv",
        baseline_vllm_json=VALIDATION_DIR / "T1" / "vllm_l40s_qwen3-32b_tp2.json",
        baseline_field="e2els",
        # The L40S real-vLLM run only achieved 2.5 req/s (see
        # data/validation/T1/README.md); this is the fixed trace it and the
        # vendored Vidur/LLMServingSim L40S baselines were measured against.
        default_trace=VALIDATION_DIR / "T1" / "sharegpt_replica_2.5qps_l40s.csv",
    ),
    "h100": dict(
        label="Llama3-70B on H100:TP8",
        model="llama2_70B",
        device="H100_GPU",
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        vllm_df_path=MIST_RUNTIME_DIR / "Hermes-4-70B_NVIDIA_H100_8.csv",
        result_filename="mist_h100.csv",
        baseline_vllm_json=VALIDATION_DIR / "T1" / "vllm_h100_llama3-70b_tp8.json",
        baseline_field="e2el",
        default_trace=VALIDATION_DIR / "T1" / "sharegpt_replica_5qps_tpu_h100.csv",
    ),
    "tpu": dict(
        label="Llama-3.1-8B on TPUv6e",
        model="llama3_8b",
        device="tpu_v6e",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        vllm_df_path=VALIDATION_DIR / "T1" / "llama-3.1-8b_tpu_v6e_vllm_runtime_table.csv",
        result_filename="mist_tpu.csv",
        baseline_vllm_json=VALIDATION_DIR / "T1" / "vllm_tpu_llama31-8b.json",
        baseline_field=None,  # TPU json needs its own (sent/last-token) parse
        # Same fixed trace as H100 -- the original notebook replayed this
        # identical 100-request ShareGPT sample against both platforms.
        default_trace=VALIDATION_DIR / "T1" / "sharegpt_replica_5qps_tpu_h100.csv",
    ),
}

CHUNK_SIZE = 2048


_TOKENIZER = None
_TOKENIZER_LOAD_FAILED = False


def _get_tokenizer():
    """Lazily load a length-counting tokenizer, shared across all calls.

    Uses GPT-2's tokenizer: it is small, ungated, and downloads/caches with
    plain `transformers` (unlike Llama/Qwen tokenizers, which require
    accepting a license or a HF token). It won't exactly match the tokenizer
    each target model actually uses, but BPE token counts across modern
    tokenizers agree far more closely with each other than a naive
    whitespace-word count does -- this is what keeps the resampled ShareGPT
    workload's length distribution close to what the vendored real-vLLM
    baselines were measured on (see data/validation/T1/README.md).
    """
    global _TOKENIZER, _TOKENIZER_LOAD_FAILED
    if _TOKENIZER is not None or _TOKENIZER_LOAD_FAILED:
        return _TOKENIZER
    try:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained("gpt2")
    except Exception as exc:
        print(
            f"warning: could not load a tokenizer ({exc}); falling back to "
            "a whitespace-word count approximation (~1.3 tokens/word) for "
            "ShareGPT length estimation.",
            file=sys.stderr,
        )
        _TOKENIZER_LOAD_FAILED = True
    return _TOKENIZER


def _approx_token_count(text: str) -> int:
    """Token count used for ShareGPT length estimation (see _get_tokenizer)."""
    tokenizer = _get_tokenizer()
    if tokenizer is not None:
        return max(1, len(tokenizer(text)["input_ids"]))
    n_words = len(_WORD_RE.findall(text))
    return max(1, round(n_words * 1.3))


def _load_sharegpt_pairs(path: Path) -> list:
    """Extract (input_len, output_len) pairs from the first human/gpt turn of
    each ShareGPT conversation, using the same length bounds vLLM's own
    ``benchmark_serving.py --dataset-name sharegpt`` sampler applies
    (prompt_len >= 4, output_len >= 4, prompt_len <= 1024, prompt_len +
    output_len <= 2048). Without this filter the raw ShareGPT_V3 dataset's
    long tail (a handful of >2000-token conversations) dominates the
    100-request sample and produces a much heavier-tailed workload than the
    vendored baselines were measured on.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"ShareGPT dataset not found at {path}.\n"
            "Run `bash scripts/download_traces.sh` first to fetch it."
        )
    with open(path, "r") as f:
        data = json.load(f)

    human_tags = {"human", "user"}
    gpt_tags = {"gpt", "assistant", "bot", "chatgpt"}
    pairs = []
    for conv in data:
        turns = conv.get("conversations") or []
        if len(turns) < 2:
            continue
        first, second = turns[0], turns[1]
        if first.get("from") not in human_tags or second.get("from") not in gpt_tags:
            continue
        prompt = (first.get("value") or "").strip()
        completion = (second.get("value") or "").strip()
        if len(prompt) < 4 or len(completion) < 4:
            continue
        input_len = _approx_token_count(prompt)
        output_len = _approx_token_count(completion)
        if input_len < 4 or output_len < 4:
            continue
        if input_len > 1024 or input_len + output_len > 2048:
            continue
        pairs.append((input_len, output_len))
    return pairs


def build_fixed_trace_queue(trace_path: Path) -> list:
    """Replay the exact per-request arrival/length trace vLLM was measured
    against (columns: arrival_timestamp, prompt_size, token_size).

    This is the DEFAULT workload source: MIST, the vendored real-vLLM run,
    and the vendored Vidur/LLMServingSim2.0 runs all see the same 100
    requests in the same arrival order, which is what makes Fig. 4 a valid
    controlled comparison rather than three simulators each answering a
    different question. ``TraceIngestion`` assigns ``request_id`` from the
    CSV row order (see mist/Input_requests/Trace_inputs.py), matching the
    order the vLLM benchmark client dispatched these requests in.
    """
    if not trace_path.exists():
        raise FileNotFoundError(
            f"Fixed validation trace not found at {trace_path}. This file "
            "should already be vendored under data/validation/T1/ -- see "
            "its README.md."
        )
    return TraceIngestion(ingestion_trace_file=str(trace_path)).request_queue


def build_resampled_queue(rps: float, num_requests: int, seed: int) -> list:
    """[--resample-sharegpt, opt-in] Resample `num_requests` ShareGPT
    conversations and schedule their arrivals as a Poisson process at
    `rps`, using MIST's own seeded PoissonDistribution so the arrival-time
    math matches the simulator's other request-generation paths exactly
    (--seed defaults to MIST's own RequestDistributions default of 259).

    This is a sensitivity-analysis mode, not the validation workload: its
    100 requests are a *different* sample than the fixed trace vLLM/Vidur/
    LLMServingSim2.0 were measured against (see build_fixed_trace_queue),
    so latencies from this mode are not directly comparable to those
    baselines. See data/validation/T1/README.md for how much a different
    seeded draw can move the aggregate token demand -- and hence end-to-end
    latency -- at the same nominal RPS.
    """
    pairs = _load_sharegpt_pairs(SHAREGPT_PATH)
    if len(pairs) < num_requests:
        raise RuntimeError(
            f"ShareGPT dataset only yielded {len(pairs)} usable "
            f"human/gpt conversation pairs, need {num_requests}."
        )

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(pairs), size=num_requests, replace=False)
    rows = [
        {"ContextTokens": pairs[i][0], "GeneratedTokens": pairs[i][1]}
        for i in chosen
    ]
    trace_df = pd.DataFrame(rows)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmp:
            trace_df.to_csv(tmp.name, index=False)
            tmp_path = tmp.name
        dist = PoissonDistribution(
            rps=rps,
            sim_time=float("inf"),
            trace_file=tmp_path,
            num_requests=num_requests,
            rand_seed=seed,
        )
    finally:
        if tmp_path is not None:
            os.unlink(tmp_path)

    request_queue = dist.request_queue
    # PoissonDistribution doesn't assign request_id; number them by arrival
    # order so downstream CSVs have a stable per-request index.
    for idx, req in enumerate(request_queue):
        req.request_id = idx
    return request_queue


def simulate_platform(key: str, cfg: dict, request_queue: list, verbose: bool) -> pd.DataFrame:
    """Run one MIST simulation and return its per-request result frame.

    Passes fresh `decode_only_cache`/`mixed_batch_cache` dicts explicitly:
    `LLMEngine.__init__` defaults these to mutable `{}` default arguments,
    so without this, every LLMEngine constructed in this *process* would
    silently share (and therefore corrupt) each other's per-step runtime
    cache -- we verified this collapses all three platforms' latencies to
    be bit-for-bit identical. Passing explicit dicts here works around it
    entirely from this script; MIST itself is untouched.
    """
    vllm_df_path = cfg["vllm_df_path"]
    if not Path(vllm_df_path).exists():
        raise FileNotFoundError(
            f"[{key}] vLLM runtime table not found at {vllm_df_path}."
        )

    platform = vLLMPlatformConfig(
        vllm_df_path=str(vllm_df_path),
        device=cfg["device"],
        tensor_parallel_size=cfg["tensor_parallel_size"],
        pipeline_parallel_size=cfg["pipeline_parallel_size"],
        model=cfg["model"],
    )
    if not platform.use_vllm:
        raise RuntimeError(
            f"[{key}] vLLMPlatformConfig fell back to the analytical model "
            f"instead of the vLLM runtime table at {vllm_df_path}; refusing "
            "to report results that don't reflect the real runtime table."
        )

    coordinator = MISTCoordinator(
        deepcopy(request_queue), logging_file=None, max_sim_time=500_000_000
    )
    coordinator.add_engine(
        engine=LLMEngine(
            model=cfg["model"],
            engine_types=[EngineType.PREFILL, EngineType.DECODE],
            scheduler_config=SchedulerConfig(
                batching_method=BatchingMethod.CHUNKED, chunk_size=CHUNK_SIZE
            ),
            platform=platform,
            step_size=0,
            engine_kv_step_size=0,
            decode_only_cache={},
            mixed_batch_cache={},
        ),
        engine_types=[EngineType.PREFILL, EngineType.DECODE],
    )

    t0 = time.perf_counter()
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink if not verbose else sys.stdout):
        coordinator.run_sim()
    sim_wall_clock_sec = time.perf_counter() - t0

    completed = coordinator.completed_requests
    if not completed:
        raise RuntimeError(f"[{key}] simulation completed but served no requests")

    rows = []
    for req in completed:
        ttft_ms = req.data[0].finished_time - req.metrics.arrival_time
        e2e_ms = req.metrics.finished_time - req.metrics.arrival_time
        rows.append(
            {
                "request_idx": req.request_id,
                "input_len": req.input_len,
                "output_len": req.output_len,
                "mist_ttft_sec": ttft_ms / 1000.0,
                "mist_e2e_sec": e2e_ms / 1000.0,
            }
        )
    df = pd.DataFrame(rows).sort_values("request_idx").reset_index(drop=True)
    df["sim_wall_clock_sec"] = sim_wall_clock_sec
    df["num_requests_submitted"] = len(request_queue)
    df["num_requests_completed"] = len(completed)
    df["platform"] = key
    df["model"] = cfg["model"]
    df["device"] = cfg["device"]
    df["tensor_parallel_size"] = cfg["tensor_parallel_size"]
    return df


def _load_vllm_baseline_e2e_sec(cfg: dict) -> np.ndarray:
    """Load the vendored real-vLLM end-to-end latencies for MAE reporting."""
    path = cfg["baseline_vllm_json"]
    with open(path, "r") as f:
        data = json.load(f)

    if cfg["baseline_field"] is not None:
        df = pd.json_normalize(data)
        return np.array(df[cfg["baseline_field"]][0], dtype=float)

    # TPU metrics file: {"metrics": [{"sent_time":..., "last_token_time":...}, ...]}
    latencies = [
        m["last_token_time"] - m["sent_time"] for m in data.get("metrics", [])
    ]
    return np.array(latencies, dtype=float)


def mean_abs_pct_error(mist_e2e_sec: np.ndarray, baseline_e2e_sec: np.ndarray) -> float:
    """Sorted-distribution mean absolute percent error, matching the metric
    the source notebook printed (cell 24): sort both latency arrays and
    compare rank-for-rank, since MIST and the vendored baseline were not
    run on identical per-request workloads."""
    n = min(len(mist_e2e_sec), len(baseline_e2e_sec))
    a = np.sort(mist_e2e_sec)[:n]
    b = np.sort(baseline_e2e_sec)[:n]
    return float(100 * np.abs(a / b - 1).mean())


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        choices=["l40s", "h100", "tpu", "all"],
        default="all",
        help="Which platform(s) to simulate (default: all).",
    )
    parser.add_argument(
        "--resample-sharegpt",
        action="store_true",
        help=(
            "Sensitivity-analysis mode: draw a fresh seeded ShareGPT sample "
            "instead of replaying the fixed trace vLLM/Vidur/LLMServingSim "
            "were measured on. Not directly comparable to those baselines "
            "-- see data/validation/T1/README.md."
        ),
    )
    parser.add_argument(
        "--rps", type=float, default=5.0,
        help="Poisson arrival rate (default: 5). Only used with --resample-sharegpt.",
    )
    parser.add_argument(
        "--num-requests", type=int, default=100,
        help="Requests to simulate (default: 100). Only used with --resample-sharegpt.",
    )
    parser.add_argument(
        "--seed", type=int, default=259,
        help="Random seed for request generation (default: 259). Only used with --resample-sharegpt.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Don't suppress the simulator's per-second progress log."
    )
    return parser.parse_args(argv)


def _report_queue_lengths(prefix: str, request_queue: list) -> None:
    print(f"{prefix} {len(request_queue)} requests, "
          f"input_len range [{min(r.input_len for r in request_queue)}, "
          f"{max(r.input_len for r in request_queue)}], "
          f"output_len range [{min(r.output_len for r in request_queue)}, "
          f"{max(r.output_len for r in request_queue)}]")


def main(argv=None) -> int:
    args = parse_args(argv)
    platforms = list(PLATFORMS) if args.platform == "all" else [args.platform]

    shared_queue = None
    if args.resample_sharegpt:
        print(f"[--resample-sharegpt] sensitivity-analysis mode: generating a FRESH "
              f"{args.num_requests}-request ShareGPT sample at {args.rps} RPS "
              f"(Poisson, seed={args.seed}). This workload differs from the fixed "
              "trace vLLM/Vidur/LLMServingSim were measured on, so the MAE reported "
              "below is NOT directly comparable to the default mode's -- see "
              "data/validation/T1/README.md.")
        try:
            shared_queue = build_resampled_queue(args.rps, args.num_requests, args.seed)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        _report_queue_lengths("  generated", shared_queue)
    else:
        print("Replaying the fixed real-request trace(s) that vLLM (and the vendored "
              "Vidur/LLMServingSim2.0 baselines) were actually measured against -- "
              "see data/validation/T1/README.md. Pass --resample-sharegpt for a "
              "fresh seeded sample instead (sensitivity analysis only).")

    failures = []
    for key in platforms:
        cfg = PLATFORMS[key]
        if shared_queue is not None:
            request_queue = shared_queue
        else:
            try:
                request_queue = build_fixed_trace_queue(cfg["default_trace"])
            except FileNotFoundError as exc:
                print(f"[{key}] BLOCKED: {exc}", file=sys.stderr)
                failures.append(key)
                continue
            _report_queue_lengths(f"[{key}] loaded {cfg['default_trace'].name} with", request_queue)

        print(f"[{key}] simulating {cfg['label']} ...")
        try:
            df = simulate_platform(key, cfg, request_queue, args.verbose)
        except Exception as exc:
            print(f"[{key}] BLOCKED: {exc}", file=sys.stderr)
            failures.append(key)
            continue

        out_path = result_path("T1", cfg["result_filename"])
        df.to_csv(out_path, index=False)
        sim_wall_clock_sec = df["sim_wall_clock_sec"].iloc[0]
        print(f"[{key}] wrote {out_path} "
              f"({len(df)} requests, sim wall clock {sim_wall_clock_sec:.3f} s)")

        try:
            baseline_e2e_sec = _load_vllm_baseline_e2e_sec(cfg)
            mae_pct = mean_abs_pct_error(df["mist_e2e_sec"].to_numpy(), baseline_e2e_sec)
            print(f"[{key}] MIST e2e mean={df['mist_e2e_sec'].mean():.3f}s "
                  f"median={df['mist_e2e_sec'].median():.3f}s | "
                  f"vLLM (real) e2e mean={baseline_e2e_sec.mean():.3f}s "
                  f"median={np.median(baseline_e2e_sec):.3f}s | "
                  f"sorted-distribution MAE={mae_pct:.2f}%")
        except Exception as exc:
            print(f"[{key}] warning: could not compute MAE vs vLLM baseline: {exc}",
                  file=sys.stderr)
        print()

    if failures:
        print(f"{len(failures)} platform(s) failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
