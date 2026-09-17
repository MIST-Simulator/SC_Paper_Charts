"""Heterogeneous deployment search-space enumerator for T3 (Fig. 7, Fig. 8).

Ported from ``GenA_Paper_charts/SC26/search_deployment_space.py`` (the
authors' notebook repo), trimmed to the 8 SKUs and TP-in-{1,2} constraint the
paper's Sec. 5.1 search actually sweeps, and rewired to source prices and the
SKU universe from :mod:`mist_charts.pricing` instead of a hardcoded dict.

This module is a pure enumerator: it returns a DataFrame describing every
(hardware, parallelism, batching strategy) combination that fits in device
memory, annotated with `Cost` and `HW_Combination`. It never builds a
`PlatformConfig`, `Coordinator`, or otherwise touches the simulator --
``run_T3.py`` does that with each row.
"""

import math
import re
from itertools import product
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from .mist_api import BatchingMethod
from .pricing import PRICE_PER_HOUR, SKUS

# Device HBM/LPDDR capacity in GB, one entry per SKU in `pricing.SKUS`.
# Source: search_deployment_space.py's `get_memory`, restricted to the 8 SKUs
# the paper's T3 search actually uses.
DEVICE_MEMORY_GB: Dict[str, float] = {
    "h200_sxm": 141,
    "b200_sxm": 192,
    "gb300": 288,
    "etched": 192,
    "mi350x": 288,
    "mi355x": 288,
    "tpu_v6e": 32,
    "tpu_v7": 192,
}


def get_memory(hardware: str) -> float:
    """Device memory capacity in GB. Falls back to 80 GB for an unlisted SKU."""
    return DEVICE_MEMORY_GB.get(hardware, 80)


def estimate_memory_required_gb(
    model: str,
    max_context_tokens: int = 128_000,
    kv_kb_per_token: float = 140.0,
) -> float:
    """GB of device memory one model replica needs to be schedulable.

    Ported unmodified from ``experiment_runner.py``'s inline computation:
    weight memory is `params_in_billions` GB (MIST's ``profiled-ops`` system
    configs run these SKUs at FP8, i.e. 1 byte/parameter, so this is not a
    bf16-vs-fp8 bug -- it matches the precision the simulator actually uses),
    plus a per-request KV-cache reservation sized for a 128K-token practical
    max context at ~140 KB/token.

    Args:
        model: HF-style model id containing a "-<N>B" parameter count, e.g.
            "Qwen/Qwen3-32B".
        max_context_tokens: KV-cache sizing budget, in tokens.
        kv_kb_per_token: KV-cache footprint per token, in KB.

    Returns:
        Estimated GB of device memory needed per model replica.
    """
    match = re.search(r"-(\d+)B", model)
    if not match:
        raise ValueError(f"Could not parse a '-<N>B' parameter count out of model id {model!r}")
    weights_gb = float(match.group(1))
    kv_gb = max_context_tokens * kv_kb_per_token / 1024 / 1024
    return weights_gb + kv_gb


def get_parallelism_combinations(
    num_devices: int = 8,
    parallelism_degree: List[str] = ["TP", "PP", "DP"],
    min_device_per_replica: int = 1,
    max_degree_constraints: Union[Dict[str, int], None] = None,
) -> pd.DataFrame:
    """All (d1, d2, ...) combinations whose product equals `num_devices`.

    Verbatim port of ``search_deployment_space.get_parallelism_combinations``.

    Constraints:
      1. Product of all degrees equals num_devices (d1 * d2 * ... = N).
      2. The first two degrees (TP, PP) must be powers of 2; subsequent
         degrees (DP, etc.) can be any integer factor.
      3. Product of the first two degrees (TP * PP) >= min_device_per_replica.
      4. Individual degree value (di) <= max_degree_constraints[degree_name]
         (if specified).

    Returns:
        A DataFrame where each row is a valid combination and columns are
        named according to `parallelism_degree`.
    """
    if num_devices < 1 or min_device_per_replica < 1:
        print("Error: num_devices and min_device_per_replica must be at least 1.")
        return pd.DataFrame(columns=parallelism_degree)
    if len(parallelism_degree) < 1:
        print("Error: parallelism_degree list must not be empty.")
        return pd.DataFrame(columns=parallelism_degree)

    results: List[List[int]] = []
    num_degrees = len(parallelism_degree)
    power_of_2_degrees = set(parallelism_degree[:2])

    def find_factors(target_product: int, degree_index: int, current_factors: List[int]):
        if degree_index == num_degrees:
            if target_product == 1:
                results.append(list(current_factors))
            return

        degree_name = parallelism_degree[degree_index]
        factors_to_check = []

        if degree_name in power_of_2_degrees:
            d = 1
            while d <= target_product:
                if target_product % d == 0:
                    factors_to_check.append(d)
                d *= 2
        else:
            for d in range(1, target_product + 1):
                if target_product % d == 0:
                    factors_to_check.append(d)

        for d in factors_to_check:
            current_factors.append(d)

            if max_degree_constraints:
                if degree_name in max_degree_constraints and d > max_degree_constraints[degree_name]:
                    current_factors.pop()
                    continue

            if degree_index == 1:  # PP is index 1
                tp_times_pp = current_factors[0] * current_factors[1]
                if tp_times_pp < min_device_per_replica:
                    current_factors.pop()
                    continue

            find_factors(target_product // d, degree_index + 1, current_factors)
            current_factors.pop()

    find_factors(num_devices, 0, [])
    return pd.DataFrame(results, columns=parallelism_degree)


def get_search_space(
    num_devices: int,
    Hardware: List[str] = SKUS,
    Batching_Strategy: List[BatchingMethod] = [BatchingMethod.CHUNKED, BatchingMethod.DISAGGREGATED],
    Max_Batch_Size: List[int] = [128],
    Chunk_Size: List[int] = [2048],
    price_per_hour: Dict[str, float] = PRICE_PER_HOUR,
    memory_required: float = 128,
    max_degree_constraints: Dict[str, int] = {"PP": 1, "TP": 2},
    avg_prompt_tokens: float = 1024,
    avg_decode_tokens: float = 1024,
) -> pd.DataFrame:
    """Enumerate every valid (hardware, parallelism, batching, batch/chunk
    size) deployment for a cluster of `num_devices` accelerators.

    Ported from ``search_deployment_space.get_search_space``. For
    ``BatchingMethod.DISAGGREGATED``, the prefill:decode device split is
    centered on the trace's prompt:decode *token* ratio (`avg_prompt_tokens`
    / `avg_decode_tokens`) with a +/-4 device margin explored around it --
    this is what encodes the "prefill-to-decode client ratio" search
    dimension. All 8 SKUs in `pricing.SKUS` are valid on both the prefill
    and decode side (unlike the source script's broader SKU list, which
    excluded a few HW types from one side or the other), so that
    prefill_hw/decode_hw whitelist check is dropped here.

    Returns:
        DataFrame with one row per valid config, columns:
        Batching_Strategy, Hardware, Parallelism, Max_Batch_Size, Chunk_Size,
        Cost, HW_Combination.
    """

    def get_price(config) -> float:
        hardware = config["Hardware"]
        parallelism = config["Parallelism"]
        if config["Batching_Strategy"] == BatchingMethod.DISAGGREGATED:
            prefill_hw, decode_hw = hardware.split("-")
            prefill_parallelism_str, decode_parallelism_str = parallelism.split("-")
            prefill_parallelism_dict = eval(prefill_parallelism_str)
            decode_parallelism_dict = eval(decode_parallelism_str)
            prefill_total_devices = math.prod(prefill_parallelism_dict.values())
            decode_total_devices = math.prod(decode_parallelism_dict.values())
            return price_per_hour[prefill_hw] * prefill_total_devices + price_per_hour[decode_hw] * decode_total_devices
        total_devices = math.prod(parallelism.values())
        return price_per_hour[hardware] * total_devices

    def get_hw_combination(config) -> List[str]:
        hardware = config["Hardware"]
        parallelism = config["Parallelism"]
        if config["Batching_Strategy"] == BatchingMethod.DISAGGREGATED:
            prefill_hw, decode_hw = hardware.split("-")
            prefill_parallelism_str, decode_parallelism_str = parallelism.split("-")
            prefill_parallelism_dict = eval(prefill_parallelism_str)
            decode_parallelism_dict = eval(decode_parallelism_str)
            return [f"{prefill_hw}_{prefill_parallelism_dict['TP']}", f"{decode_hw}_{decode_parallelism_dict['TP']}"]
        return [f"{hardware}_{parallelism['TP']}"]

    search_space = []

    for batching in Batching_Strategy:
        if batching == BatchingMethod.DISAGGREGATED:
            # Center the prefill:decode device split on the trace's
            # prompt:decode token ratio, and explore a +/-4 device margin
            # around it (this is the "prefill-to-decode client ratio" axis).
            prompt_ratio = avg_prompt_tokens / (avg_prompt_tokens + avg_decode_tokens)
            ratio_based_devices = round(num_devices * prompt_ratio)
            margin = 4
            min_prefill_devices = max(1, ratio_based_devices - margin)
            max_prefill_devices = max(1, min(num_devices - 1, ratio_based_devices + margin))
            for prefill_devices in range(min_prefill_devices, max_prefill_devices + 1):
                decode_devices = num_devices - prefill_devices
                for prefill_hw, decode_hw in product(Hardware, repeat=2):
                    if prefill_devices < np.ceil(memory_required / get_memory(prefill_hw)):
                        continue
                    if decode_devices < np.ceil(memory_required / get_memory(decode_hw)):
                        continue
                    prefill_combos = get_parallelism_combinations(
                        num_devices=prefill_devices,
                        parallelism_degree=["TP", "PP", "DP"],
                        min_device_per_replica=int(np.ceil(memory_required / get_memory(prefill_hw))),
                        max_degree_constraints=max_degree_constraints,
                    )
                    decode_combos = get_parallelism_combinations(
                        num_devices=decode_devices,
                        parallelism_degree=["TP", "PP", "DP"],
                        min_device_per_replica=int(np.ceil(memory_required / get_memory(decode_hw))),
                        max_degree_constraints=max_degree_constraints,
                    )
                    for (_, prefill_parallelism), (_, decode_parallelism) in product(
                        prefill_combos.iterrows(), decode_combos.iterrows()
                    ):
                        for max_batch_size in Max_Batch_Size:
                            search_space.append(
                                {
                                    "Batching_Strategy": batching,
                                    "Hardware": f"{prefill_hw}-{decode_hw}",
                                    "Parallelism": f"{prefill_parallelism.to_dict()}-{decode_parallelism.to_dict()}",
                                    "Max_Batch_Size": max_batch_size,
                                    "Chunk_Size": Chunk_Size[-1],
                                }
                            )
        else:
            for hardware in Hardware:
                parallelism_combos = get_parallelism_combinations(
                    num_devices=num_devices,
                    parallelism_degree=["TP", "PP", "DP"],
                    min_device_per_replica=int(np.ceil(memory_required / get_memory(hardware))),
                    max_degree_constraints=max_degree_constraints,
                )
                for _, parallelism in parallelism_combos.iterrows():
                    for max_batch_size in Max_Batch_Size:
                        if batching == BatchingMethod.CHUNKED:
                            for chunk_size in Chunk_Size:
                                search_space.append(
                                    {
                                        "Batching_Strategy": batching,
                                        "Hardware": hardware,
                                        "Parallelism": parallelism.to_dict(),
                                        "Max_Batch_Size": max_batch_size,
                                        "Chunk_Size": chunk_size,
                                    }
                                )
                        else:
                            search_space.append(
                                {
                                    "Batching_Strategy": batching,
                                    "Hardware": hardware,
                                    "Parallelism": parallelism.to_dict(),
                                    "Max_Batch_Size": max_batch_size,
                                    "Chunk_Size": Chunk_Size[-1],
                                }
                            )

    df = pd.DataFrame(search_space)
    df["Cost"] = df.apply(get_price, axis=1)
    df["HW_Combination"] = df.apply(get_hw_combination, axis=1)
    df = stagger_search_space(df)
    return df


def _get_conflict_keys(row) -> List[Tuple[int, str]]:
    """Conflict keys for one row: two rows conflict if they share any key.

    DISAGGREGATED rows have two keys (prefill side, decode side); other rows
    have one. Used only to spread otherwise-identical HW/parallelism configs
    apart in the enumeration order (see `stagger_search_space`).
    """
    hardware = row["Hardware"]
    parallelism = row["Parallelism"]

    if row["Batching_Strategy"] == BatchingMethod.DISAGGREGATED:
        prefill_hw, decode_hw = hardware.split("-")
        prefill_parallelism, decode_parallelism = parallelism.split("-")
        return [(0, f"{prefill_hw}_{prefill_parallelism}"), (1, f"{decode_hw}_{decode_parallelism}")]
    parallelism_str = str(parallelism) if isinstance(parallelism, dict) else parallelism
    return [(0, f"{hardware}_{parallelism_str}")]


def stagger_search_space(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder rows so that ones sharing a Hardware x Parallelism combination
    are spread apart, greedily maximizing distance-since-last-use.

    Verbatim port of ``search_deployment_space.stagger_search_space``. This
    matters for `run_T3.py --fast-eval`'s testing knob and for interrupted
    sweeps: adjacent rows in the (sim-time-ordered) CSV are then unlikely to
    be near-duplicates of the same SKU/parallelism.
    """
    if len(df) <= 1:
        return df

    all_keys = [_get_conflict_keys(row) for _, row in df.iterrows()]
    n = len(df)
    scheduled_order = []
    remaining = set(range(n))
    last_used: Dict[Tuple[int, str], int] = {}

    for step in range(n):
        best_idx = None
        best_score = -1
        for i in remaining:
            min_dist = float("inf")
            for key in all_keys[i]:
                if key in last_used:
                    min_dist = min(min_dist, step - last_used[key])
            if min_dist > best_score:
                best_score = min_dist
                best_idx = i
        scheduled_order.append(best_idx)
        remaining.remove(best_idx)
        for key in all_keys[best_idx]:
            last_used[key] = step

    return df.iloc[scheduled_order].reset_index(drop=True)
