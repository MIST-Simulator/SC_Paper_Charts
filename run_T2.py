#!/usr/bin/env python3
"""Driver for Figure 6 (a)-(b) of the MIST SC26 paper: individual-step
validation.

  (a) Per-step forward-pass runtime: MIST vs Vidur vs real vLLM, for
      Llama-2-70B on H100 at tensor-parallel degree 4 and 8.

      The paper's §4.1.1 ("LLM Client HW Executor") defines MIST's per-step
      runtime model as *ML-based*: an ensemble of regressors trained on
      profiled vLLM data (input size, batch size, chunk size, TP degree),
      not the GenZ analytical roofline. §4.2's validation subsection is
      explicitly scoped to "our ML-Based LLM Cluster Modeling (§4.1.1)".
      Accordingly, MIST here is ``vLLMPlatformConfig``
      (mist/Platforms/vllm_platform.py) -- the lookup + RandomForest-ensemble
      predictor over profiled vLLM data -- pointed at the vendored profiled
      CSVs under data/validation/T2/. It is *not* the plain
      ``PlatformConfig`` GenZ analytical path (see "Known upstream issues"
      below): that path is what MIST falls back to for hypothetical,
      unprofiled hardware (§5.1: Etched, GB300, TPUv7), never the claimed
      basis for this figure.

      To avoid a tautological "fit and evaluate on the same rows" result,
      each hardware/stage is split by whole (total_batches, total_tokens)
      groups: a seeded `--holdout-frac` (default 0.2) of the *groups* is
      held out entirely, vLLMPlatformConfig's ensemble is fit only on the
      remaining (non-held-out) rows, and every held-out row is scored by
      calling `get_chunked_time` on it. Because grouping is exact-equality
      on (total_batches, total_tokens) -- the same key vLLMPlatformConfig's
      own near-exact lookup (`_search_vllm_df`) matches on -- no held-out
      row's configuration exists anywhere in the training data, so
      evaluation always falls through to the (held-out) regressor rather
      than an exact-match lookup. Vidur -- itself an ML predictor trained on
      profiling data, per the paper -- gets the *same* group-holdout
      protocol applied to its own baseline CSV, so the comparison is
      apples-to-apples: its RandomForestRegressor surrogate (see
      data/validation/T2/README.md for why a surrogate is needed at all --
      Vidur's own scheduler produces a different set of steps than vLLM's)
      is fit only on Vidur rows outside the held-out groups, then evaluated
      on the same held-out real-vLLM rows MIST is scored on.

      Known upstream issues (see data/validation/T2/README.md for the full
      writeup): (1) the GenZ analytical path (`PlatformConfig.get_chunked_time`)
      hardcodes `bits='fp8'` in 5 places in mist/Platforms/platforms.py with
      no dtype parameter exposed on `PlatformConfig.__init__`, which
      underpredicts bf16/fp16 vLLM latency by roughly 2-3x -- a real bug,
      but not the mechanism §4.2 / Fig. 6(a) uses, per the paper text above.
      (2) `PlatformConfig(device="h100_sxm", ...)` (the profiled-ops/System
      path) currently crashes in this environment's GenZ checkout with
      `AttributeError: type object 'GEMMQuantMode' has no attribute
      'bfloat16'` in GenZ/db.py -- an independent upstream GenZ bug.

  (b) KV cache retrieval latency: MIST vs `fio`-measured latency, for NVMe
      SSD and DDR4, sequential reads, block sizes 256 KB - 1 GB, using a
      Llama-3-70B (meta-llama/llama-3.1-70b GenZ config, 160 KB/token) KV
      cache as the example. The MIST curve is produced by driving MIST's own
      KV-retrieval cost model (mist.Platforms.memory_platform.SingleCacheConfig
      / MemoryCacheConfig, as used by mist.Engine.KV_Retrieval_Engine) with
      representative DDR4/NVMe device parameters -- never a formula
      re-implemented in the plotting script.

Both panels write CSVs under results/T2/ via ``mist_charts.paths.result_path``;
``plot_T2.py`` reads them (falling back to results/reference/T2/) together with
the vendored baselines under data/validation/T2/.
"""

import argparse
import ast
import os
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from mist_charts.paths import VALIDATION_DIR, result_path
from mist_charts.mist_api import (
    MemoryCacheConfig,
    Request,
    RequestStage,
    SingleCacheConfig,
    get_configs,
    vLLMPlatformConfig,
)

T2_DATA_DIR = VALIDATION_DIR / "T2"

# --------------------------------------------------------------------------
# Part (a): per-step forward-pass runtime
# --------------------------------------------------------------------------

# GenZ model id for Llama-2-70B (the model behind the vLLM/Vidur baselines).
MODEL_NAME = "meta-llama/llama-2-70b"

# (plot label, real-vLLM baseline CSV, Vidur baseline CSV, tensor-parallel size)
HARDWARE_CONFIGS = [
    ("Llama-2-70B\nH100xTP4", "vllm_step_runtime_H100_TP4.csv", "vidur_step_runtime_H100_TP4.csv", 4),
    ("Llama-2-70B\nH100xTP8", "vllm_step_runtime_H100_TP8.csv", "vidur_step_runtime_H100_TP8.csv", 8),
]

STAGES = ("prefill", "decode", "chunked")

FEATURE_COLUMNS = ["total_batches", "total_tokens", "total_KV_length", "total_attention_size"]

# Original vLLM/Vidur trace CSV columns, in order, for round-tripping the
# train-only subset back out to a temp CSV that vLLMPlatformConfig can load.
_RAW_COLUMNS = ["Prefill", "Context", "Decode", "Time (ms)"]


def _parse_list(val):
    """Parse a CSV cell like "[(0, 391)]" or "[40, 12]" into a Python list."""
    if pd.isna(val):
        return []
    s = str(val).strip()
    if s in ("", "[]"):
        return []
    try:
        parsed = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []
    return list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]


def _row_features(prefill_items, context_items):
    """Reproduce the notebook's (batches, tokens, kv_length, attn_size) features."""
    tokens = kv_length = attn_size = batches = 0
    for item in prefill_items:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            kv, tok = int(item[0]), int(item[1])
        else:
            kv, tok = 0, int(item)
        tokens += tok
        kv_length += kv
        attn_size += (kv + tok) * tok
        batches += 1
    for item in context_items:
        c = int(item[0]) if isinstance(item, (list, tuple)) else int(item)
        batches += 1
        tokens += 1
        kv_length += c
        attn_size += (c + 1) * 1
    return batches, tokens, kv_length, attn_size


def _load_step_csv(path):
    df = pd.read_csv(path, delimiter=";")
    df.columns = [c.strip() for c in df.columns]
    df["Time (ms)"] = pd.to_numeric(df["Time (ms)"], errors="coerce")
    df = df[df["Time (ms)"] > 2].copy()
    df["Prefill"] = df["Prefill"].astype(str).str.strip()
    df["Context"] = df["Context"].astype(str).str.strip()
    prefill_items = df["Prefill"].map(_parse_list)
    context_items = df["Context"].map(_parse_list)
    feats = [
        _row_features(p, c) for p, c in zip(prefill_items, context_items)
    ]
    df[FEATURE_COLUMNS] = feats
    df["_prefill_items"] = prefill_items
    df["_context_items"] = context_items
    return df


def _split_stages(df):
    """Return (prefill_df, chunked_df, decode_df), trimming top-1.3x-median
    outliers from prefill/chunked exactly like the source notebook."""
    is_prefill = df["Prefill"] != "[]"
    is_context = df["Context"] != "[]"

    prefill_df = df[~is_context & is_prefill]
    if len(prefill_df):
        prefill_df = prefill_df[prefill_df["Time (ms)"] <= 1.3 * prefill_df["Time (ms)"].median()]

    chunked_df = df[is_context & is_prefill]
    if len(chunked_df):
        chunked_df = chunked_df[chunked_df["Time (ms)"] <= 1.3 * chunked_df["Time (ms)"].median()]

    decode_df = df[is_context & ~is_prefill]
    return prefill_df, chunked_df, decode_df


def _held_out_groups(stage_df, holdout_frac, seed):
    """Pick a seeded, deterministic subset of whole (total_batches,
    total_tokens) groups to hold out entirely -- not individual rows.

    vLLMPlatformConfig's near-exact lookup (`_search_vllm_df`) matches rows
    by exact (total_tokens, total_batches) equality. Holding out whole
    groups (rather than i.i.d. rows, which could leave a training row with
    the identical group key as a held-out row) guarantees the evaluation
    predictor can never fall back to memorized lookup for a held-out
    configuration -- it must generalize via the fitted ensemble.
    """
    groups = sorted(set(zip(stage_df["total_batches"], stage_df["total_tokens"])))
    order = np.random.RandomState(seed).permutation(len(groups))
    n_holdout = max(1, int(round(len(groups) * holdout_frac)))
    holdout_idx = order[:n_holdout]
    return {groups[i] for i in holdout_idx}


def _split_by_groups(df, holdout_groups):
    keys = list(zip(df["total_batches"], df["total_tokens"]))
    is_holdout = pd.Series([k in holdout_groups for k in keys], index=df.index)
    return df[~is_holdout], df[is_holdout]


def _build_requests(prefill_items, context_items):
    """Rebuild the exact prefill/decode Request objects MIST needs to score
    a scheduler step, matching the (kv, chunk_tokens) / kv encoding used by
    the vLLM/Vidur trace format."""
    prefills = []
    for i, item in enumerate(prefill_items):
        kv, tok = (int(item[0]), int(item[1])) if isinstance(item, (list, tuple)) and len(item) >= 2 else (0, int(item))
        req = Request(
            request_id=f"p{i}",
            input_len=kv + tok,
            past_context=0,
            stages=[RequestStage.PREFILL, RequestStage.DECODE],
        )
        req.remaining_prefill_tokens = 0
        req.current_scheduled_prefill = tok
        prefills.append(req)

    decodes = []
    for i, item in enumerate(context_items):
        kv = int(item[0]) if isinstance(item, (list, tuple)) else int(item)
        req = Request(
            request_id=f"d{i}",
            input_len=0,
            past_context=kv,
            beam_size=1,
            stages=[RequestStage.DECODE],
        )
        decodes.append(req)

    return prefills, decodes


def run_part_a(seed: int, holdout_frac: float, max_samples_per_stage: int) -> pd.DataFrame:
    rows = []

    for hw_label, real_csv, vidur_csv, tp_size in HARDWARE_CONFIGS:
        print(f"\n=== Part (a): {hw_label.replace(chr(10), ' ')} ===")
        real_df = _load_step_csv(T2_DATA_DIR / real_csv)
        vidur_df = _load_step_csv(T2_DATA_DIR / vidur_csv)

        real_prefill, real_chunked, real_decode = _split_stages(real_df)
        vidur_prefill, vidur_chunked, vidur_decode = _split_stages(vidur_df)
        real_stage_dfs = {"prefill": real_prefill, "chunked": real_chunked, "decode": real_decode}
        vidur_stage_dfs = {"prefill": vidur_prefill, "chunked": vidur_chunked, "decode": vidur_decode}

        # Held-out split, done once per stage so the *same* held-out
        # configurations are used to evaluate both MIST and Vidur.
        train_parts = []
        eval_dfs = {}
        vidur_train_dfs = {}
        for stage in STAGES:
            stage_df = real_stage_dfs[stage]
            vidur_stage_df = vidur_stage_dfs[stage]
            if len(stage_df) == 0 or len(vidur_stage_df) == 0:
                continue
            holdout_groups = _held_out_groups(stage_df, holdout_frac, seed)
            train_df, eval_df = _split_by_groups(stage_df, holdout_groups)
            if max_samples_per_stage and len(eval_df) > max_samples_per_stage:
                eval_df = eval_df.sample(n=max_samples_per_stage, random_state=seed)
            train_parts.append(train_df)
            eval_dfs[stage] = eval_df

            vidur_train_df, _ = _split_by_groups(vidur_stage_df, holdout_groups)
            vidur_train_dfs[stage] = vidur_train_df

        # MIST = vLLMPlatformConfig: the paper's ML-based per-step predictor
        # (an ensemble/lookup over profiled vLLM data), fit only on the
        # non-held-out rows written to a scratch CSV in the original trace
        # format, so `_search_vllm_df`'s near-exact lookup and the fallback
        # RandomForestRegressor are both trained without seeing any held-out
        # group.
        train_full_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=_RAW_COLUMNS)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", prefix="mist_t2_train_", delete=False
        ) as tmp_f:
            train_full_df[_RAW_COLUMNS].to_csv(tmp_f, sep=";", index=False)
            train_csv_path = tmp_f.name

        try:
            platform = vLLMPlatformConfig(
                vllm_df_path=train_csv_path, device="H100", tensor_parallel_size=tp_size, model=MODEL_NAME
            )

            for stage in STAGES:
                eval_df = eval_dfs.get(stage)
                if eval_df is None or len(eval_df) == 0:
                    print(f"  [{stage}] skipped (no rows)")
                    continue

                # Vidur baseline: fit a small regressor on Vidur's own
                # non-held-out rows (see data/validation/T2/README.md for why
                # a surrogate is needed at all), then evaluate on the same
                # held-out real-vLLM rows MIST is scored on.
                vidur_train_df = vidur_train_dfs[stage]
                vidur_predictor = RandomForestRegressor(n_estimators=100, random_state=seed)
                vidur_predictor.fit(vidur_train_df[FEATURE_COLUMNS], vidur_train_df["Time (ms)"])
                vidur_pred = vidur_predictor.predict(eval_df[FEATURE_COLUMNS])

                mist_pred = []
                for prefill_items, context_items in zip(eval_df["_prefill_items"], eval_df["_context_items"]):
                    prefills, decodes = _build_requests(prefill_items, context_items)
                    latency_ms, _ = platform.get_chunked_time(prefills, decodes)
                    mist_pred.append(latency_ms)
                mist_pred = np.array(mist_pred)

                real_time = eval_df["Time (ms)"].to_numpy()
                mist_err_pct = (1 - mist_pred / real_time) * 100
                vidur_err_pct = (1 - vidur_pred / real_time) * 100

                print(
                    f"  [{stage}] n={len(eval_df)} (held-out groups)  "
                    f"MIST MAE={np.mean(np.abs(mist_err_pct)):.2f}%  "
                    f"Vidur MAE={np.mean(np.abs(vidur_err_pct)):.2f}%"
                )

                for real_t, mist_t, vidur_t, mist_e, vidur_e in zip(
                    real_time, mist_pred, vidur_pred, mist_err_pct, vidur_err_pct
                ):
                    rows.append(
                        {
                            "hardware": hw_label.replace("\n", " "),
                            "tensor_parallel_size": tp_size,
                            "stage": stage,
                            "real_time_ms": real_t,
                            "mist_time_ms": mist_t,
                            "vidur_time_ms": vidur_t,
                            "mist_error_pct": mist_e,
                            "vidur_error_pct": vidur_e,
                        }
                    )
        finally:
            os.remove(train_csv_path)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Part (b): KV cache retrieval latency
# --------------------------------------------------------------------------

# GenZ config closest to "Llama-3-70B" (160 KB/token KV cache, matching the
# source notebook's stated per-token size).
KV_MODEL_NAME = "meta-llama/llama-3.1-70b"

# Representative device parameters used to *parameterize* MIST's existing
# retrieval-latency model (mist.Platforms.memory_platform.SingleCacheConfig);
# these are not measurements -- the measured comparison lives in
# data/validation/T2/fio_retrieval_latency.csv. Values match the assumptions
# used in the original prototype (combined_retreival_latency_plot.py):
# DDR4 ~0.08us fixed access latency + ~200 GB/s sustained sequential
# bandwidth; NVMe SSD ~40us controller/queueing latency + ~8 GB/s sequential
# read bandwidth (typical PCIe4 consumer NVMe).
DEVICE_PARAMS = {
    "DDR4": {"lookup_latency_ms": 0.08 / 1000, "bandwidth_gbps": 200.0},
    "NVMe SSD": {"lookup_latency_ms": 40.0 / 1000, "bandwidth_gbps": 8.0},
}


def run_part_b() -> pd.DataFrame:
    print("\n=== Part (b): KV cache retrieval latency ===")
    fio_df = pd.read_csv(T2_DATA_DIR / "fio_retrieval_latency.csv")

    kv_bytes_per_token = get_configs(KV_MODEL_NAME).get_kv_size()
    print(f"  {KV_MODEL_NAME}: {kv_bytes_per_token / 1024:.1f} KB/token KV cache")

    rows = []
    for device, params in DEVICE_PARAMS.items():
        cache = SingleCacheConfig(
            type=device,
            memory_size=1e9,  # unused by get_retrieval_time; placeholder capacity
            memory_bandwidth=params["bandwidth_gbps"],
            retrieval_latency=params["lookup_latency_ms"],
            hit_rate=1.0,
        )
        cache_hierarchy = MemoryCacheConfig([cache], model=KV_MODEL_NAME)

        device_rows = fio_df[fio_df["device"] == device].sort_values("block_size_bytes")
        num_tokens = device_rows["block_size_bytes"].to_numpy() / kv_bytes_per_token
        requests = [
            Request(request_id=f"{device}-{i}", input_len=0, past_context=t)
            for i, t in enumerate(num_tokens)
        ]
        mist_latency_ms = cache_hierarchy.get_KV_cache_time(requests)

        measured_latency_ms = device_rows["measured_latency_us"].to_numpy() / 1000
        err_pct = np.abs(1 - np.array(mist_latency_ms) / measured_latency_ms) * 100
        print(f"  [{device}] MAE={np.mean(err_pct):.2f}%  (per-point: {np.round(err_pct, 1).tolist()})")

        for bs, tok, mist_ms, meas_ms, e in zip(
            device_rows["block_size_bytes"], num_tokens, mist_latency_ms, measured_latency_ms, err_pct
        ):
            rows.append(
                {
                    "device": device,
                    "block_size_bytes": bs,
                    "num_tokens": tok,
                    "measured_latency_ms": meas_ms,
                    "mist_latency_ms": mist_ms,
                    "abs_error_pct": e,
                }
            )

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=["a", "b", "all"], default="all")
    parser.add_argument("--seed", type=int, default=259)
    parser.add_argument(
        "--holdout-frac",
        type=float,
        default=0.2,
        help="Fraction of (total_batches, total_tokens) groups per (hardware, stage) "
        "to hold out entirely from vLLMPlatformConfig's / Vidur's training data "
        "before evaluating part (a) on them (seeded, deterministic).",
    )
    parser.add_argument(
        "--max-samples-per-stage",
        type=int,
        default=150,
        help="Cap on how many held-out real-vLLM rows to score per (hardware, stage) "
        "pair in part (a); each row is one MIST simulator call. 0 disables the cap.",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.part in ("a", "all"):
        step_df = run_part_a(args.seed, args.holdout_frac, args.max_samples_per_stage or None)
        out_path = result_path("T2", "step_runtime_validation.csv")
        step_df.to_csv(out_path, index=False)
        print(f"\nwrote {out_path}  ({len(step_df)} rows)")

        print("\nSummary -- MIST mean absolute error vs real vLLM (paper: 4.2% prefill, "
              "1.9% decode, 3.2% chunked prefill):")
        for stage in STAGES:
            stage_rows = step_df[step_df["stage"] == stage]
            if len(stage_rows) == 0:
                continue
            mist_mae = stage_rows["mist_error_pct"].abs().mean()
            vidur_mae = stage_rows["vidur_error_pct"].abs().mean()
            print(f"  {stage:8s}  MIST MAE={mist_mae:6.2f}%   Vidur MAE={vidur_mae:6.2f}%   (n={len(stage_rows)})")

    if args.part in ("b", "all"):
        kv_df = run_part_b()
        out_path = result_path("T2", "kv_retrieval_validation.csv")
        kv_df.to_csv(out_path, index=False)
        print(f"\nwrote {out_path}  ({len(kv_df)} rows)")

        print("\nSummary -- MIST mean absolute error vs fio-measured retrieval latency "
              "(paper: 5.7% KV retrieval):")
        for device in DEVICE_PARAMS:
            device_rows = kv_df[kv_df["device"] == device]
            if len(device_rows) == 0:
                continue
            mae = device_rows["abs_error_pct"].mean()
            print(f"  {device:10s}  MAE={mae:6.2f}%   (n={len(device_rows)})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
