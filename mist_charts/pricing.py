"""Per-accelerator hourly rental prices and vendor grouping for T3 (Fig. 7, Fig. 8).

Declarative data only: ``run_T3.py`` reads ``PRICE_PER_HOUR`` to cost every
deployment_space row, and ``plot_T3.py`` reads ``VENDOR_COLORS`` /
``VENDOR_MARKERS`` / ``VENDOR_CATEGORY_ORDER`` / ``VENDOR_LEGEND_ORDER`` /
``PERFORMANCE_SCALE_FACTOR`` for the Fig. 7/8 plots and the README's
hardware table, so there is exactly one place that encodes vendor, price,
and scale-factor per SKU. Ported from ``GenA_Paper_charts/SC26/
plot_sc_results.py``'s ``DEFAULT_PRICES`` / ``PERFORMANCE_SCALE_FACTOR``,
restricted to the paper's Sec. 5.1 8-SKU hardware table.

``DEFAULT_PRICES`` conflicts with ``experiment_runner.py``'s own internal
search-space price table; this module uses ``DEFAULT_PRICES`` (matches the
paper's stated Nov-2025 rental prices and what the figures actually used).
See docs/FINDINGS.md#t3 ("Price table conflict") for both tables.
"""

from typing import Dict, List

# $ / accelerator / hour, sampled November 2025. See module docstring for
# the conflicting experiment_runner.py price table.
PRICE_PER_HOUR: Dict[str, float] = {
    "h200_sxm": 6.31,
    "b200_sxm": 8.60,
    "gb300": 12.00,
    "etched": 12.00,
    "mi350x": 7.50,
    "mi355x": 9.50,
    "tpu_v6e": 3.00,
    "tpu_v7": 8.00,
}

# Output-throughput correction applied per *decode* SKU before any cost or
# normalization math (plot_sc_results.py: `Output_Throughput_Scaled =
# output_throughput * PERFORMANCE_SCALE_FACTOR[decode_hw]`). Ported verbatim:
# AMD SKUs are scaled by 0.6, everything else by 1.0.
PERFORMANCE_SCALE_FACTOR: Dict[str, float] = {
    "h200_sxm": 1.0,
    "b200_sxm": 1.0,
    "gb300": 1.0,
    "etched": 1.0,
    "mi350x": 0.6,
    "mi355x": 0.6,
    "tpu_v6e": 1.0,
    "tpu_v7": 1.0,
}

# The 8 SKUs the T3 search space is built from, in a fixed, deterministic order.
SKUS: List[str] = list(PRICE_PER_HOUR)

# Which vendor manufactures each SKU. "Mixed" is not a SKU -- a disaggregated
# (prefill, decode) pair is "Mixed" when its two SKUs resolve to different
# vendors here; see `vendor_of_config`.
VENDOR_OF_SKU: Dict[str, str] = {
    "h200_sxm": "Nvidia",
    "b200_sxm": "Nvidia",
    "gb300": "Nvidia",
    "mi350x": "AMD",
    "mi355x": "AMD",
    "tpu_v6e": "TPU",
    "tpu_v7": "TPU",
    "etched": "Etched",
}

# Fig. 8's x-axis category order (bars, left to right).
VENDOR_CATEGORY_ORDER: List[str] = ["Nvidia", "AMD", "TPU", "Etched", "Mixed"]

# Fig. 7's legend order (as specified: Mixed, Etched, TPU, AMD, Nvidia).
VENDOR_LEGEND_ORDER: List[str] = ["Mixed", "Etched", "TPU", "AMD", "Nvidia"]

# One color per vendor group, shared by every T3 figure.
VENDOR_COLORS: Dict[str, str] = {
    "Mixed": "#1f77b4",   # blue
    "Etched": "#ff7f0e",  # orange
    "TPU": "#9467bd",     # purple
    "AMD": "#17becf",     # cyan
    "Nvidia": "#2ca02c",  # green
}

# Fig. 7 marker shapes: Mixed configs are squares, everything else a circle.
VENDOR_MARKERS: Dict[str, str] = {
    "Mixed": "s",
    "Etched": "o",
    "TPU": "o",
    "AMD": "o",
    "Nvidia": "o",
}

# Display names used in Fig. 7's winning-config annotation boxes (e.g.
# "MI350x:H200-SXM 2P:6D, TP: (P=1, D=1)"). Matches plot_sc_results.py's
# PLOT_NAMES verbatim (note "MI355X" -- capital X -- and "TPU-v6e"/"TPU-v7"
# with a hyphen, both exactly as in the source).
SKU_DISPLAY_NAMES: Dict[str, str] = {
    "h200_sxm": "H200-SXM",
    "b200_sxm": "B200-SXM",
    "gb300": "GB300",
    "etched": "Etched",
    "mi350x": "MI350x",
    "mi355x": "MI355X",
    "tpu_v6e": "TPU-v6e",
    "tpu_v7": "TPU-v7",
}


def parse_hardware(hardware: str, is_disaggregated: bool):
    """(prefill_sku, decode_sku, is_hetero) for one `Hardware` value.

    Port of plot_sc_results.py's `parse_hardware`, adapted to read our own
    clean `Hardware` column (a single SKU, or "<prefill>-<decode>" for
    disaggregated configs) instead of regex-parsing a "UseCase" string.
    """
    if is_disaggregated:
        prefill_sku, decode_sku = hardware.split("-", 1)
    else:
        prefill_sku = decode_sku = hardware
    return prefill_sku, decode_sku, prefill_sku != decode_sku


def is_multi_vendor_config(hardware: str, is_disaggregated: bool) -> bool:
    """True iff the prefill and decode SKUs resolve to different vendors.

    Narrower than `vendor_of_config`'s "Mixed"; used for Fig. 8's bar
    chart, where `vendor_of_config` is used for Fig. 7's scatter. See
    docs/FINDINGS.md#t3 ("Fig. 7 vs. Fig. 8 'Mixed' category inconsistency").
    """
    prefill_sku, decode_sku, _ = parse_hardware(hardware, is_disaggregated)
    return VENDOR_OF_SKU[prefill_sku] != VENDOR_OF_SKU[decode_sku]


def vendor_of_config(hardware: str, is_disaggregated: bool) -> str:
    """Vendor/category group for one `deployment_space.get_search_space` row:
    'Mixed' unless prefill_sku == decode_sku exactly (broader than
    `is_multi_vendor_config`, which requires different *vendors* -- see
    docs/FINDINGS.md#t3). Returns one of `VENDOR_CATEGORY_ORDER`.
    """
    prefill_sku, decode_sku, is_hetero = parse_hardware(hardware, is_disaggregated)
    if is_hetero:
        return "Mixed"
    return VENDOR_OF_SKU[prefill_sku]
