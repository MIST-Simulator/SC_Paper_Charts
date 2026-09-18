# Findings

Long-form writeups of discrepancies, upstream bugs, and non-obvious
methodology found while porting the paper's figures into this repository.
Source files carry only a 1-2 line pointer into this document; this is
where the full reasoning, numbers, and file:line references live.

Organized by task (T1-T4), plus a section for bugs in the upstream
`MIST`/`GenZ` packages that this repo works around without editing.

<a id="t1"></a>
## T1 -- end-to-end runtime validation (Fig. 4)

### Axis-label bug in the camera-ready figure

Camera-ready Fig. 4(a,b)'s y-axis is labeled "Latency (ms)", but the
plotted quantity in both panels is **seconds**: panel (a) tops out around
40 while `run_T1.py` reports a mean L40S e2e latency of 12.8s, and panel
(b) tops out around 15 with a mean H100 e2e latency of 2.9s. Panel (c) is
correctly labeled "Latency (sec)". The mislabel is in the published
paper, not introduced by this port; `plot_T1.py` labels all three panels
"Latency (sec)" instead. Every source feeding the plot was audited to
confirm there's no unit bug hiding behind the wrong label (each is
seconds, once, end to end): `run_T1.py`'s `mist_e2e_sec = (finished_time_ms
- arrival_time_ms) / 1000.0`; vLLM's `e2els`/`e2el` fields (native
seconds); Vidur's `request_e2e_time` (native seconds); LLMServingSim's
`latency` (nanoseconds, divided by `1e9`); and the TPU JSON's
`last_token_time - sent_time` (Unix timestamps in seconds).

### Measured accuracy vs. the paper's and the AD's claims

Running `run_T1.py` in its default (fixed-trace) mode:

| Platform | MIST vs. vLLM MAE |
|---|---|
| L40S:TP2 (Qwen3-32B) | 11.46% |
| H100:TP8 (Llama3-70B) | 6.51% |
| TPUv6e (Llama-3.1-8B) | 8.26% |

The paper body (Sec. 4.2) claims "less than 2% average error." These
numbers don't support that -- they're 3-6x higher on every platform.
They **do** fall inside the artifact description appendix's stated
"approximately 5-10% mean absolute error" (L40S slightly above the band,
H100/TPUv6e within it). No workload, model config, or metric was tuned
to chase 2%; these are what `run_T1.py` produces against the vendored
fixed trace.

Simulation wall-clock time: MIST 1.7-2.9s vs. LLMServingSim2.0
33.8-144.8s vs. Vidur 485-495s (Vidur has no TPU backend).

### `--resample-sharegpt` sensitivity result

Replaying the fixed trace (default mode) reproduces the real H100 latency
distribution to ~6.5% MAE (~8.3% TPU, ~11.5% L40S). Resampling a fresh
seeded 100-request ShareGPT draw at the same nominal RPS instead
reproduces the same H100 baseline to only ~277% MAE at `--seed 259` and
~305% at `--seed 7` -- both in the same 150-300% range despite very
different seeds. This is a sampling-sensitivity result, not a MIST
accuracy number: ShareGPT's prompt/response lengths are heavy-tailed, so
a random 100-request draw can carry substantially more or less aggregate
token demand than the curated fixed trace at the same RPS, enough to push
the system from comfortably-loaded into compounding queueing delay.

<a id="t2"></a>
## T2 -- individual step and KV retrieval validation (Fig. 6)

### Which MIST model this figure uses, and why

The paper's Sec. 4.1.1 defines MIST's per-step runtime model as
*ML-based*: an ensemble of regressors trained on profiled vLLM data.
Sec. 4.2's validation is explicitly scoped to that model. `run_T2.py`
therefore uses `mist.Platforms.vllm_platform.vLLMPlatformConfig` (lookup +
RandomForest ensemble over the profiled CSVs in `data/validation/T2/`),
never `mist.Platforms.platforms.PlatformConfig` (the GenZ analytical
roofline) -- that path is what MIST falls back to for hypothetical,
unprofiled hardware (Sec. 5.1: Etched, GB300, TPUv7), and per the "known
upstream issues" section below, produces a very different, much worse
number here.

### Held-out group split (why it's needed, and how it works)

`vLLMPlatformConfig`'s near-exact lookup and its RandomForestRegressor
are both fit directly on the profiled CSV, so evaluating on rows drawn
from that same CSV is tautological unless those exact rows are excluded
from training first. `run_T2.py` does a seeded, deterministic **held-out
group split**: for each (hardware, stage), rows are grouped by exact
`(total_batches, total_tokens)` -- the same key the near-exact lookup
matches on -- and a whole `--holdout-frac` (default 0.2) of those *groups*
is held out. `vLLMPlatformConfig` is then instantiated on a temp CSV
containing only the non-held-out rows, so the near-exact lookup can never
find a held-out row's configuration, and every held-out row is scored
purely by the (also held-out) regressor.

Vidur -- itself an ML predictor trained on profiling data, per the paper
-- gets the same protocol applied to its own baseline CSV (its own
scheduler produces a different set of steps than vLLM's, so the two CSVs
don't share step boundaries): a small seeded `RandomForestRegressor` is
fit on Vidur's `(batches, tokens, kv_length, attn_size) -> Time` pairs
excluding the same held-out groups, then evaluated on the same held-out
real-vLLM rows MIST is scored on, so the comparison is apples-to-apples.

Held-out MAE, MIST vs. Vidur: prefill 2.11% vs. 18.25%, decode 0.53% vs.
7.73%, chunked 2.12% vs. 14.89%. Part (b) (KV retrieval): MIST tracks
`fio` to 7.7% MAE (NVMe) and 14.4% (DDR4) for blocks >= 16MB; relative
error is much larger below 4MB where absolute latencies are tiny (both
breakdowns are reported, not just the flattering one).

### Known upstream issues (informational -- not the mechanism T2 uses)

While investigating a first, incorrect draft of this reproduction that
used `PlatformConfig.get_chunked_time()` (the GenZ analytical roofline)
for Fig. 6(a), two independent upstream issues turned up. Neither affects
the numbers above (which use `vLLMPlatformConfig`, per the paper's own
description of Fig. 6(a)'s methodology), but both are real:

1. **`PlatformConfig` hardcodes `bits='fp8'`.** `mist/Platforms/platforms.py`
   passes `bits='fp8'` into every `chunked_moddeling`/`System(...)` call
   (lines 78, 276, 323, 361, 379), and `PlatformConfig.__init__` exposes
   no dtype parameter to override it. Manually forcing `bf16` on one
   sample request (Llama-2-70B, TP4, 391-token prefill) roughly doubled
   predicted latency (12.6ms -> 23.4ms fp8->bf16, vs. 39.1ms measured),
   which plausibly explains most of the ~2-3x systematic underprediction
   this path produces against real bf16/fp16 vLLM runs. This is what the
   paper uses for *hypothetical, unprofiled* hardware in Sec. 5.1 (Etched,
   GB300, TPUv7) where no profiled vLLM data exists to fit an ensemble --
   it was never the claimed basis for Fig. 6(a).
2. **`PlatformConfig(device="h100_sxm", ...)` crashes** in this
   environment's GenZ checkout with `AttributeError: type object
   'GEMMQuantMode' has no attribute 'bfloat16'` in `GenZ/db.py`
   (`bits_to_gemm_quants` maps `'bf16' -> GEMMQuantMode.bfloat16`, but the
   enum only defines `float16`). See "GenZ GEMMQuantMode rename" below.
   This blocks the profiled-ops/`System` device path entirely,
   independent of the `fp8` issue above.

### Camera-ready Fig. 6(a) mislabel

The published figure image
(`MIST_SC26_source/figures/image/individual_step_validation.pdf`) labels
both rows "Llama3-70B", but the paper body says otherwise (Sec. 4.2:
"vLLM running Llama-2-70B on H100x8"), and the underlying data files
(`Llama-2-70b-hf_NVIDIA_H100_{4,8}.csv`) are unambiguously Llama-2-70B --
the original notebook's own model dict key was also `'Llama3-70B'`, i.e.
the mislabel predates this port. This reproduction uses the correct
"Llama-2-70B" label, matching the data and the paper text.

<a id="t3"></a>
## T3 -- heterogeneous design-space search (Fig. 7, Fig. 8)

### Trace / SLO source of truth (commit aac3254, "SC Paper Charts")

An earlier version of this script used `narrativeqa.csv` for "chat" (per
the original task spec) and found essentially no configuration could meet
a 250ms TTFT SLO -- correct given that trace's ~58K-token mean context,
but the wrong input. `GenA_Paper_charts/SC26/plot_sc_results.py` and
`plot_bar_results.py` (confirmed by the author as the scripts that
produced the camera-ready Fig. 7/8) read exactly two CSVs, both derived
from the *code-generation* trace:

```
Chat_search_results.pdf              <- Code_Generation_KV.csv, WITHOUT
                                         KV retrieval, slo_ms=200
Code_Generation_KV_search_results.pdf <- a shuffled copy of the same
                                         trace, WITH KV retrieval,
                                         slo_ms=1500
```

`narrativeqa.csv` is not used for Fig. 7/8 at all. This is why both use
cases in `run_T3.py` point at `code_generation_kv.csv`, differing only in
`kv_retrieval` and a seeded shuffle (`_seeded_shuffle_trace`). The "chat"
SLO used is **200ms**, not the 250ms stated in the paper's Sec. 5.1 --
that is what the plotting code that made the figure actually uses; the
discrepancy is reported, not silently resolved.

### Runtime-backend substitution

The paper's DSE used Nvidia's AIConfigurator as a single LLM-cluster
runtime backend shared by every SKU, for an apples-to-apples hardware
comparison. The open MIST package has no standalone AIConfigurator
LLM-cluster simulator class. Instead, every SKU in `run_T3.py` runs
through MIST's own `PlatformConfig` -> GenZ `System`, which (for these 8
SKUs, all configured with `compute_engine="profiled-ops"`) happens to
look up its own per-operator (GEMM/attention/collective) timings from
that same aiconfigurator silicon database -- but layers MIST's own
event-driven request queueing, batching, and (for disaggregated configs)
prefill/decode orchestration on top, rather than AIConfigurator's own
cluster-level simulator. So every SKU is still scored by one consistent
backend, which is the property the paper's DSE needs -- but the
serving-level simulation is MIST's, not AIConfigurator's. Expect absolute
tokens/$ numbers to differ from the published figure even though the
search + SLO filter + tokens/s/$ optimization procedure is identical.

### Runtime-DB backend fallback for gb300 (`_patch_runtime_db_backend_fallback`)

`GenZ/system.py` constructs `RuntimeDB(hardware=...)` without passing
`backend`, so it always resolves `backend='vllm'`. For gb300 that
database ships an `INCOMPLETE.txt` ("context_attention_perf.txt is
incomplete, most data points failed in collection"), so
`get_latest_database_version` returns `None`, `get_database` returns
`None`, and every gb300 configuration dies with `'NoneType' object has no
attribute 'system_spec'`. That silently drops a third of the paper's
Nvidia SKUs from the search.

gb300 does ship a complete `trtllm/1.2.0rc6.post3` database -- the same
ten files as the h200_sxm TRT-LLM database that already works. Switching
backend does not bias the comparison: MIST runs the database in
`DatabaseMode.SOL`, the roofline estimator the paper specifies in Sec.
5.1 ("ensuring all HW SKUs are modeled with the same backend and based on
their raw capabilities"). In SOL mode the measured per-operator tables
are not used for timing; the database supplies only the hardware's
`system_spec`. So this changes which file gb300's spec sheet is read
from, not how fast gb300 is modeled. Remove once MIST threads a `backend`
parameter through `System`.

### `chunked_moddeling` failures on long decode KV contexts

The `code_generation_kv.csv` trace's `KV_Tokens` column (attached as
`past_context` for the codegen/KV-retrieval use case) has a max of
**303,407 tokens** (mean 27,099, 75th pct 37,494 -- verified directly
against the vendored trace). At that scale, some hardware/parallelism
combinations fail inside `chunked_moddeling` while scoring a decode step
with this much accumulated KV. 61 of 150 codegen configs (`--fast-eval`
subset) failed this way, and the failures fall disproportionately on
cross-vendor disaggregated pairs, cutting the number of surviving Mixed
configs from 83 (as in "chat", which never attaches KV_Tokens and so
never hits this) to 45. The true codegen Mixed-vendor advantage is likely
larger than what the surviving 45 configs measure.

### Hardware list and run-config discrepancies

The version of `experiment_runner.py` that shipped with commit `aac3254`
restricts `HW_available` to `mi300x/mi350x/mi355x/tpu_v6e/tpu_v7` -- a
5-SKU list with no Nvidia or Etched SKUs at all, which cannot produce the
published Fig. 8 (it has Nvidia and Etched bars). `run_T3.py` instead
uses the paper's Sec. 5.1 eight-SKU list (`mist_charts.pricing.SKUS`:
h200_sxm, b200_sxm, gb300, etched, mi350x, mi355x, tpu_v6e, tpu_v7),
which is what Fig. 8 needs.

`aac3254`'s `experiment_runner.py` also moved its defaults to
`model=meta-llama-3.1-70B, num_requests=10000, TP<=4` -- the author's
current, in-progress iteration, not the configuration that produced the
camera-ready figures. The filenames the plotting scripts actually read
(`... - 100RPS - Qwen3-32B - 60sec - 200reqs.csv`) pin down the
configuration used instead: Qwen3-32B, 100 RPS, 60s sim window, 200
requests, TP in {1,2} (`run_T3.py`'s `--rps`/`--sim-time`/`--num-requests`
defaults, and `deployment_space`'s `max_degree_constraints`).

### Price table conflict

The same repo's `SC26/experiment_runner.py` hardcodes a *different*
internal search-space price table (mi350x $2.75, mi355x $2.95, tpu_v6e
$1.89, tpu_v7 $3.50) than `plot_sc_results.py`'s `DEFAULT_PRICES`
(mi350x $7.50, mi355x $9.50, tpu_v6e $3.00, tpu_v7 $8.00, used in
`mist_charts/pricing.py`). `DEFAULT_PRICES` matches the paper's stated
Nov-2025 rental prices (Sec. 5.1) and is what the plotting scripts that
produced the published figures actually used for cost/tokens-per-$, so
this repo uses `DEFAULT_PRICES`, not `experiment_runner.py`'s figures.

### Fig. 7 vs. Fig. 8 "Mixed" category inconsistency

`plot_sc_results.py`'s scatter category (`assign_category`, ported as
`mist_charts.pricing.vendor_of_config`) is broad: any disaggregated
config whose two SKUs differ AT ALL, even two SKUs from the same vendor
(e.g. h200_sxm prefill + b200_sxm decode, both Nvidia), counts as
"Mixed". `plot_bar_results.py`'s bar chart (`is_multi_vendor`, ported as
`is_multi_vendor_config`) instead requires the two SKUs to be on
different *vendors*. So Fig. 7's blue squares and Fig. 8's Mixed bar
don't count quite the same set of configs. This is preserved faithfully
(both functions exist and are each used where the source used them)
rather than silently reconciled.

<a id="t4"></a>
## T4 -- KV cache storage design-space search (Fig. 10)

### Case D's DCN bandwidth: paper vs. artifact conflict

Table 2 of the paper states Case D's inter-rack transfer runs at "128
GB/s". But the notebook that generated the published
`Case_Study_Memory_Cache.pdf`
(`Experiments/Memory_Storage_Comparisions.py` via
`4. Cache_storage_config_comparisions.ipynb`, cell 7) hard-codes this link
at 1 GB/s, and only 1 GB/s reproduces the published figure's shape: Case
D sits near-worst in both "shared" panels, collapsing toward Case E
(recompute) behavior, because the slow DCN hop dominates. At 128 GB/s, D
is architecturally a strict superset of C (same storage tier, but
load-balanced across all 4 racks instead of just 1), so it strictly
*dominates* C instead -- an interesting sensitivity result, but not
what's in the paper. `run_T4.py` defaults to the notebook's 1 GB/s
(matches the published figure) and exposes `--dcn-bandwidth-gbps` so a
reader can flip to 128 (Table 2's stated value) and see the reversal.

### Other deviations from the source notebook

- **Cases D and E were labelled backwards relative to Table 2** in the
  source notebook; relabelled in `mist_charts/memory_configs.py`.
- **The notebook ran configs concurrently with its RNG seeding commented
  out**, so it was never bit-reproducible. `run_T4.py` uses a per-config
  `np.random.default_rng`, seeded and thread-safe.
- **The notebook's sweep never actually replayed the Azure trace**; it
  used synthetic Poisson arrivals for everything. `run_T4.py` keeps the
  calibrated 240 RPS arrival rate that produced the published figure
  (`AGGREGATE_RPS_AT_128_CLIENTS`) but draws each request's
  (input_len, output_len) from real sampled trace rows
  (`build_base_stream`), so the trace is genuinely required and used, not
  just downloaded and ignored.

### Results

Both published findings reproduce: architectures A/B (dedicated
per-client / platform-shared) give the lowest tail latency for private KV
caches at both context lengths, and C (rack-shared) is best for shared
global caches with long KV sequences (P90 6132ms, ahead of B 6727ms and A
6799ms).

<a id="upstream-bugs"></a>
## Upstream MIST/GenZ bugs

Found and worked around entirely from within this repo's scripts -- no
files in `MIST` or `GenZ` were modified. All four should be routed to and
fixed in their respective upstream repos.

### 1. `mist.Engine.LLMEngine` shares its per-step runtime cache across every engine in a process (serious)

`mist/Engine/Mixed_LLM_Engine.py`'s `LLMEngine.__init__` declares:

```python
def __init__(
    self, ...,
    decode_only_cache: Dict[int, Dict[int, tuple]] = {},
    mixed_batch_cache: Dict[int, Dict[int, tuple]] = {},
) -> None:
```

`{}` as a default argument is a classic Python bug: the same dict object
is created once, at function-definition time, and reused as the default
for *every* call that doesn't pass its own -- so every `LLMEngine`
constructed in the same Python process without explicit
`decode_only_cache=`/`mixed_batch_cache=` shares (and mutates) the same
two cache dicts.

**Confirmed this corrupts results.** These caches key purely on
`(num_decodes, sum_decode_kv)` / `(chunk_size, total_kv)` -- not on which
platform/engine produced the cached value. Constructing three
`LLMEngine`s back to back in one process (TPU, then H100, then L40S)
against the *same* request queue, without passing the two cache arguments
explicitly, produced **bit-for-bit identical per-request latencies across
all three platforms**, despite completely different models,
tensor-parallel degrees, and vLLM runtime tables. The second and third
engine constructed were silently reading back the first engine's cached
step times instead of querying their own regressor. This would silently
corrupt *any* multi-platform or multi-configuration study run in one
process (a Jupyter kernel across notebook cells, a sweep script, etc.) --
plausibly including parts of the original validation notebook this task
ported from, if it constructed more than one `LLMEngine` per kernel
session.

**Workaround (in `run_T1.py`, not in MIST):** every `LLMEngine(...)` call
passes fresh `decode_only_cache={}, mixed_batch_cache={}` explicitly.

**Real fix (upstream, in MIST):** change the defaults to `None` and
allocate a fresh dict inside `__init__` when `None` is passed, e.g.
`decode_only_cache: Optional[Dict] = None`, then
`self._decode_only_cache = {} if decode_only_cache is None else decode_only_cache`
(and the same for `mixed_batch_cache`).

### 2. GenZ `GEMMQuantMode.bfloat16` (and FMHA/KVCache/MoE equivalents) no longer exist (blocking)

`GenZ/db.py` references `.bfloat16` on four aiconfigurator enums (in the
module-level `bits_to_gemm_quants` dict, and as default arguments to
`query_context_attention`, `query_generation_attention`, and
`query_gemm`/`query_trtllm_alltoall`). The `aiconfigurator` submodule
version pinned by GenZ's own `git submodule` pointer
(`d5fb9da2e643044a70900eee15739d2fc71eb39c`) renamed `bfloat16` to
`float16` in `aiconfigurator/src/aiconfigurator/sdk/common.py` (bf16 and
fp16 are both just the generic 16-bit w16a16 quant mode there now). Since
GenZ's submodule pointer and its own code disagree, constructing *any*
GenZ `System(...)` -- and therefore any MIST `vLLMPlatformConfig` or
`PlatformConfig`, for any platform -- raises `AttributeError: type object
'GEMMQuantMode' has no attribute 'bfloat16'` immediately.

**Workaround (in `run_T1.py`'s `_patch_genz_quant_compat` and
`run_T3.py`'s `_patch_aiconfigurator_quant_enum_compat`, not in GenZ):**
monkeypatch `enum_cls.bfloat16 = enum_cls.float16` on the four affected
enum classes, after importing `aiconfigurator.sdk.common` from GenZ's
vendored submodule path, before importing anything from `MIST`/`mist_api`.
No-op (skipped) if the enums already have a native `bfloat16` member,
e.g. once upstream is fixed.

**Real fix (upstream, in GenZ-LLM-Analyzer):** either update
`GenZ/db.py`'s four `.bfloat16` references to `.float16`, or bump/pin the
`aiconfigurator` submodule to a commit that still defines `bfloat16`.

### 3. `PlatformConfig` hardcodes `bits='fp8'`, no dtype override (real bug, not the mechanism T2 uses)

See "T2 -- Known upstream issues" above for the full writeup: five call
sites in `mist/Platforms/platforms.py` (lines 78, 276, 323, 361, 379),
~2-3x systematic underprediction against bf16/fp16 vLLM runs. This is the
path MIST uses for hypothetical, unprofiled hardware (Sec. 5.1: Etched,
GB300, TPUv7) since no profiled vLLM data exists there to fit an ensemble.

### 4. `RuntimeDB` doesn't thread a `backend` parameter through `System`

See T3's "Runtime-DB backend fallback for gb300" above: `GenZ/system.py`
always resolves `backend='vllm'`, so any SKU whose vLLM database is
missing/incomplete (gb300) fails instead of falling back to a database
under a different backend. Worked around in `run_T3.py` with
`_patch_runtime_db_backend_fallback`.

---

## Reproducibility blockers found in a clean environment

Verified on 2026-09-16 in a fresh `uv` virtualenv (Python 3.12) with only
`uv pip install git+https://github.com/MIST-Simulator/MIST.git` — i.e. what
an external reviewer actually gets. These are the issues that stop T1/T3/T4
reproducing off the authors' machine; they are not visible in a development
checkout, where local state masks all three.

What works in a clean environment:

| Step | Result |
|------|--------|
| `uv pip install git+.../MIST` | installs cleanly, pulls GenZ automatically |
| `python scripts/smoke_test.py` | passes |
| `plot_T1..T4` — all 7 figures | regenerate from `results/reference/` |
| `run_T2.py` (Fig. 6) | reproduces exactly: 2.12% / 0.53% / 2.12% |
| `run_T1.py` (Fig. 4) | **fails on all three platforms** |
| `run_T3.py`, `run_T4.py` | **fail**: `ModuleNotFoundError: No module named 'aiconfigurator'` |

### 1. aiconfigurator does not ship with a pip install of GenZ

**Resolved**: `scripts/install_genz.sh` clones GenZ recursively and installs
it, and `setup.sh` runs it automatically.

`pip install git+.../GenZ-LLM-Analyzer` does not fetch git submodules, so a
fresh install has no `aiconfigurator` package at all — no performance
databases and no system YAMLs:

```
aiconfigurator systems dir exists: False
System('h200_sxm', compute_engine='profiled-ops')
  -> ModuleNotFoundError: No module named 'aiconfigurator'
```

Every `profiled-ops` SKU therefore fails. The analytical spec-sheet path
(`PlatformConfig(device='H100_GPU')`) still works, and MIST's own
`Platforms/vllm_runtime_data/` (26 files) *is* included in the wheel —
which is why T2, which uses the ML-predictor path, is unaffected.

Fix: make GenZ depend on `aiconfigurator` as a real package, or document
and automate a submodule-init step. Today `pip install` yields a
silently half-working GenZ.

### 2. The non-Nvidia SKU definitions exist only as untracked local files

**Resolved**: the 11 affected definitions are now vendored in
`data/hardware/` and installed by `scripts/install_genz.sh`.

Every SKU in the paper's search space that is not an Nvidia part is a
hand-authored YAML living untracked inside the aiconfigurator submodule on
the authors' machine:

```
?? etched.yaml  cerebras_cs3.yaml  groq_lpx.yaml  mi300x.yaml
?? mi350x.yaml  mi355x.yaml  tpu_v6e.yaml  tpu_v7.yaml
 M a100_sxm.yaml  gb300.yaml  h200_sxm.yaml
```

`etched.yaml` is the Etched Sohu projection described in Sec. 5.1. None of
these are committed to any repository, so even with the submodule correctly
initialised nobody else has MI350X, MI355X, TPUv6e, TPUv7 or Etched, and
the heterogeneous DSE cannot run. This is the single biggest blocker to
reproducing T3.

Fix: vendor the 11 custom/modified YAMLs (into MIST or this repo) with an
install step that copies them into aiconfigurator's `systems/` directory.

### 3. GenZ's submodule pin is newer than its own code supports

GenZ's HEAD commit `091bdd0 "update submodule"` advances aiconfigurator
from `30dd40d` to `d5fb9da` (2026-04-13). That bump causes both shims this
repository carries:

| | old pin `30dd40d` | current pin `d5fb9da` |
|---|---|---|
| `GEMMQuantMode.bfloat16` | present | renamed to `float16` |
| `gb300/vllm/0.19.0` | complete | **removed** |
| `gb300/vllm/0.14.0` | INCOMPLETE | INCOMPLETE (only option) |
| INCOMPLETE markers | 4 | 1 |

GenZ's `db.py` still references `bfloat16` in 6 places, so constructing any
`System` raises `AttributeError` — hence
`_patch_aiconfigurator_quant_enum_compat`. And with `gb300/vllm/0.19.0`
deleted upstream, gb300 has no usable vLLM database — hence
`_patch_runtime_db_backend_fallback`.

**Do not revert the pin.** The obvious fix — pinning the submodule back to
`30dd40d`, which has `bfloat16` and a complete gb300 vLLM database — breaks
everything else, because the accelerator definitions in `data/hardware/`
were authored against `d5fb9da`'s schema. They use the key
`float16_tc_flops`; the older revision's `perf_database.py` reads
`bfloat16_tc_flops` and fails with `KeyError: 'bfloat16_tc_flops'` for every
SKU. `scripts/install_genz.sh` therefore uses the revision GenZ records,
and both shims are required, not optional.

The real fix belongs upstream in GenZ: update `db.py` to the current enum
name (`float16`), and either restore a complete `gb300/vllm` database or
teach `System` to accept a `backend` argument so a caller can select one.
Until then this repository carries both shims.

### 4. The placeholder database tree is also untracked

Accelerators without measured silicon data (the AMD, TPU and Etched parts)
set `data_dir: data/dummy` and are modelled from their YAML spec sheet via
the roofline path. aiconfigurator nevertheless resolves a database *version*
before reaching that path, so `systems/data/dummy/` must exist. On the
authors' machine it did, as an untracked, entirely empty directory tree:

```
dummy/vllm/0.14.0/   (0 files)
dummy/nccl/2.27/     (0 files)
```

Upstream ships no such directory, so a fresh checkout fails with
`'NoneType' object has no attribute 'system_spec'` for mi350x, mi355x,
tpu_v6e, tpu_v7 and etched — i.e. every SKU that is not an Nvidia part.

**Resolved**: `scripts/install_genz.sh` creates the tree. No files are
needed; nothing ever reads them.
