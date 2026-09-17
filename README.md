# MIST SC26 — Reproduction Scripts

Reproduction scripts for the figures in **_MIST: A Co-Design Framework for
Heterogeneous, Multi-Stage LLM Inference_** (SC 2026).

MIST is an event-driven simulator that models the full LLM serving stack —
workload, system/software, and hardware layers — for heterogeneous,
multi-stage inference pipelines. The simulator is open source at
[MIST-Simulator/MIST](https://github.com/MIST-Simulator/MIST); this
repository holds only the scripts that drive it to regenerate the paper's
figures.

## Quick start

```sh
git clone https://github.com/MIST-Simulator/SC_Paper_Charts.git
cd SC_Paper_Charts

./setup.sh                        # conda env + all dependencies incl. MIST
conda activate mist-sc26
bash scripts/download_traces.sh   # public Azure + ShareGPT traces
python scripts/smoke_test.py      # verify the install (<1 min)
```

`setup.sh` also runs `scripts/install_genz.sh`, which installs GenZ —
MIST's analytical hardware model — from a recursive checkout. This step is
required and cannot be replaced by `pip install`: pip does not fetch git
submodules, so a plain install leaves GenZ without its `aiconfigurator`
submodule, and every accelerator modelled from a performance database then
fails with `ModuleNotFoundError: No module named 'aiconfigurator'`, taking
out T3 and T4. The script also installs the accelerator definitions in
`data/hardware/` (MI350X, MI355X, TPUv6e, TPUv7, Etched Sohu and others)
that the heterogeneous search in T3 sweeps over. Budget a few GB of disk
and several minutes for the checkout.

Then regenerate every figure from the committed reference results, which
takes seconds:

```sh
bash run_all.sh --plot-only
```

…or re-run the simulations that produced them:

```sh
bash run_all.sh --fast-eval       # ~5-7 h (reduced T3 search space)
bash run_all.sh                   # ~27-51 h (full T3 search space)
```

## What reproduces what

| Task | Script pair | Paper figure | Runtime |
|------|-------------|--------------|---------|
| **T1** End-to-end runtime validation | `run_T1.py` → `plot_T1.py` | Fig. 4 (a–d) | ~30 min |
| **T2** Individual step validation | `run_T2.py` → `plot_T2.py` | Fig. 6 (a–b) | ~30 min |
| **T3** Heterogeneous design-space search | `run_T3.py` → `plot_T3.py` | Fig. 7 (a–b), Fig. 8 | 24–48 h (2–4 h with `--fast-eval`) |
| **T4** KV cache storage design-space search | `run_T4.py` → `plot_T4.py` | Fig. 10 | ~2 h |

The four task groups are independent and can run concurrently
(`bash run_all.sh --parallel`).

Each `run_T*.py` sweeps its parameter space and writes per-request and
aggregate metrics (TTFT, TPOT, TTLT, per-step runtimes, simulation
wall-clock time) as CSVs under `results/<task>/`. Each `plot_T*.py` reads
those CSVs, post-processes them (CDFs, normalised tokens/\$, overlaid
real-hardware reference points), and writes a PDF to `figures/`.

### Reproducibility contract

**Every MIST data point in every figure is produced by the `run_T*.py` in
this repository.** Nothing MIST-derived is hand-copied or committed as a
figure-only artifact: each MIST curve, bar, and marker traces back to a
simulation you can re-run. The sweeps are deterministic — request traces
are generated from fixed random seeds (`--seed`, default `259`) and MIST's
models are analytical — so re-running a task reproduces its CSV, and hence
its figure, exactly.

Only the **baselines** are vendored under `data/validation/`, because they
cannot be regenerated without the original hardware or third-party
simulators:

| Baseline | Source | Why it ships as data |
|----------|--------|----------------------|
| vLLM measured runs | L40S×2, H100×8, TPUv6e | requires the physical hardware |
| Vidur | Vidur simulator | separate simulator, separate environment |
| LLMServingSim 2.0 | LLMServingSim 2.0 | separate simulator, separate environment |
| `fio` block-device latencies | NVMe SSD, DDR4 | requires the physical devices |

Each vendored file is documented in `data/validation/README.md` with its
provenance and the command that produced it.

### Reference results

`results/reference/` holds the MIST result CSVs behind the published
figures — the exact output of the `run_T*.py` invocations recorded in
`results/reference/PROVENANCE.md`. Every `plot_T*.py` falls back to them
when the matching `results/<task>/` directory is empty, so a reviewer can
inspect all five figures in seconds without first spending a day on the T3
sweep. Once you run a `run_T*.py`, its freshly generated results take
precedence, and you can diff the two to confirm they agree.

### Reduced T3 search space

The full T3 sweep covers 1,500+ hardware configurations. Passing
`--fast-eval` restricts it to a curated ~150-configuration subset that
spans the Pareto frontier:

```sh
python run_T3.py --use-case chat --fast-eval
python run_T3.py --use-case codegen --fast-eval
```

## Experiments

**T1 — End-to-end runtime validation (Fig. 4).** Simulates 100 ShareGPT
requests at 5 RPS (Poisson) on three platforms — Qwen3-32B on L40S×2
(TP2), Llama3-70B on H100×8 (TP8), and Llama3.1-8B on TPUv6e — and
compares the end-to-end latency distribution against measured vLLM runs and
against the Vidur and LLMServingSim 2.0 simulators. Panel (d) compares
simulation wall-clock time.

**T2 — Individual step validation (Fig. 6).** (a) Per-step LLM inference
runtimes for Llama-2-70B on H100×8 across batch sizes and input lengths,
versus vLLM ground truth and Vidur. (b) KV cache retrieval latency for
NVMe SSD and DDR4 across block sizes from 256 KB to 1 GB, versus `fio`
measurements.

**T3 — Heterogeneous design-space search (Fig. 7, Fig. 8).** Searches
hardware SKU × tensor parallelism (TP ∈ {1,2}) × batching strategy ×
prefill-to-decode client ratio for Qwen3-32B at 100 RPS on up to 8 nodes,
for two use cases: chat conversation (NarrativeQA, TTFT P99 ≤ 250 ms) and
code generation with high prefix-KV reuse (GitHub-Code, TTFT P99 ≤ 1.5 s).
Optimises generated tokens per second per dollar among SLO-meeting
configurations.

**T4 — KV cache storage design-space search (Fig. 10).** Evaluates five KV
cache storage architectures on 256 GPUs (128 clients at H100:TP2, 4 racks)
replaying the Azure conversational trace, sweeping short (4K token) and
long (24K token) KV contexts for both private and shared caches.

| Case | Architecture | Capacity / BW | Access |
|------|--------------|---------------|--------|
| A | Dedicated per-client | 1 TB / 128 GB/s | LPDDR, local |
| B | Platform-shared | 4 TB / 32 GB/s | Shared by 4 clients |
| C | Rack-shared | 32 TB / 2 GB/s | Shared by 32 clients |
| D | Rack-shared + DCN | 32 TB / 2 GB/s | Inter-rack @ 128 GB/s |
| E | No cache | — | Full KV recomputation |

## Expected results

- **T1/T2.** MIST should track measured on-hardware latency within roughly
  5–10% mean absolute error on all three platforms, and per-step runtimes
  and KV retrieval latencies should follow the measured trends across the
  full range of input lengths and block sizes.
- **T3.** Heterogeneous multi-vendor PD-disaggregated configurations should
  Pareto-dominate homogeneous single-vendor alternatives on generated
  tokens/\$ — by roughly >15% for chat, and by a larger margin (>40%) for
  code generation, where whale-request KV recomputation makes the compute
  requirements asymmetric.
- **T4.** The five storage architectures should produce clearly separated
  latency CDFs: dedicated per-client memory (A) and platform-level shared
  memory (B) give the lowest tail latency for private KV caches, while
  rack-level shared SSD (C) is best for shared global caches with long KV
  sequences.

Absolute tokens/\$ figures in T3 depend on cloud rental prices sampled in
November 2025 (listed in §5.1 of the paper); the performance ranking
depends only on the hardware models and is unaffected by price movement.

## Requirements

A Linux or macOS workstation with 8–16 CPU cores and ≥16 GB RAM. No GPU or
special interconnect is needed — MIST is an analytical/event-driven
simulator and runs entirely on CPU. Python 3.10–3.12; all dependencies are
pinned in `requirements.txt`.

## Repository layout

```
run_T{1,2,3,4}.py      experiment drivers (sweep -> results/<task>/*.csv)
plot_T{1,2,3,4}.py     plotting (results/*.csv -> figures/*.pdf)
run_all.sh             run everything, then plot everything
setup.sh               create the conda env and install dependencies
mist_charts/           shared helpers: paths, plot style, MIST import surface
scripts/               download_traces.sh, smoke_test.py
data/traces/           input request traces
data/validation/       measured vLLM / Vidur / LLMServingSim baselines
results/<task>/        generated results
results/reference/     result CSVs behind the published figures
figures/               generated PDFs
```

## Troubleshooting

**`ImportError: Could not import the MIST simulator`** — the `MIST` package
is not installed in the active environment. Run `./setup.sh` and
`conda activate mist-sc26`, or install it directly with
`pip install git+https://github.com/MIST-Simulator/MIST.git`.

**`FileNotFoundError` naming a generated and a reference path** — neither
the sweep nor the reference results are available for that task. Run the
matching `run_T*.py`, or verify the repository was cloned completely
(`results/reference/` should be non-empty).

**A `plot_T*.py` uses stale numbers** — plotting prefers
`results/<task>/` over `results/reference/`. Delete the generated
directory to fall back to the published reference results.

## Citation

```bibtex
@inproceedings{mist_sc26,
  title     = {{MIST}: A Co-Design Framework for Heterogeneous,
               Multi-Stage {LLM} Inference},
  booktitle = {Proceedings of the International Conference for High
               Performance Computing, Networking, Storage and Analysis
               (SC)},
  year      = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
