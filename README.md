# Not All 4-bit Quantizers Are Equal: Deployment-Time Mitigation of PII Leakage in Fine-Tuned Small Language Models

Reproducibility artifact for the paper of the same title, submitted to
SBSeg 2026. This README follows the SBSeg 2026 artifact-evaluation template.

**Paper summary.** Four-bit post-training quantization is the default path to
deploy small language models. This work shows that the choice of 4-bit
quantizer is itself a privacy decision: at the same bit-rate, the
calibration-based methods AWQ and GPTQ suppress verbatim extraction of PII
memorized during fine-tuning far more than the GGUF k-quant Q4_K_M, which uses
no calibration corpus. The effect holds across five open models from 0.5B to
7B parameters and across both fine-tuning regimes (full and LoRA), at
negligible accuracy cost at production scale. The paper explains the effect
with a three-factor mechanism and reconciles a prior membership-inference
(MIA) result that appeared to contradict this conclusion.

**Artifact goal.** This repository reproduces **every** table and figure in
the paper. It provides the code, the version-pinned environment, the
experiment manifest, and the per-seed result logs that back every reported
number, plus two reproduction paths: a lightweight GPU-free *replay* and a
full re-run of the pipeline.

# README structure

This README follows the SBSeg 2026 artifact template and is organized into
the following sections: **README structure** (this section); **Badges
considered**; **Basic information** (environment, hardware and software, and
the two reproduction paths); **Dependencies** (benchmarks, datasets and pinned
versions); **Security concerns**; **Installation**; **Minimal test**;
**Experiments** (reviewer time budget, then one subsection per paper claim,
each with a full and a reduced version); **Known caveats**; and **LICENSE**.

Repository layout:

```
.
|-- README.md                   this file
|-- replay.sh                    path (a): re-derive tables and figures from committed logs (no GPU)
|-- reproduce.sh                 path (b): re-run the full pipeline from scratch (GPU + pinned env)
|-- pyproject.toml, uv.lock, .python-version   pinned Python environment
|-- EXPERIMENT_MANIFEST.yaml     pinned models / datasets / toolchain / seeds / hyperparameters
|-- ENGINEERING.md               engineering bar, determinism notes, toolchain gotchas
|-- src/qquilt/                  the package: canaries, data, train, quantize, extract,
|                                metrics, aggregate, utility, seed, groups, ...
|-- scripts/                     per-experiment drivers (exp_*.sh / exp_*.py / step_*.sh),
|                                the 5 figure scripts (fig_*.py), and build_llama_cpp.sh
`-- experiment/
    |-- figures/                 figures rendered by replay.sh / the fig_*.py scripts
    `-- results/                 pre-computed result logs (extraction.jsonl, metrics.json,
                                 train_steps.jsonl, delta_norm.json, *.json), plus
                                 INDEX.md (table <-> dir map) and SCHEMA.md (JSONL schemas)
```

# Badges considered

The badges considered are: **Available (SeloD)**, **Functional (SeloF)**,
**Sustainable (SeloS)**, and **Reproducible (SeloR)**.

* **Available** -- the artifact is public in a stable anonymous repository
  (`https://anonymous.4open.science/r/quantizer-pii-mitigation-33B5/`) with
  this complete README.
* **Functional** -- the artifact runs and the reviewer can observe its
  functionality; see the *Installation* and *Minimal test* sections.
* **Sustainable** -- the code is modular (the `src/qquilt/` package),
  documented (`ENGINEERING.md`, docstrings, `experiment/results/SCHEMA.md` and
  `INDEX.md`), and every paper claim is identifiable in the artifact (the
  *Experiments* section).
* **Reproducible** -- `reproduce.sh` re-runs the pipeline and `replay.sh`
  re-derives every table and figure from the committed logs.

# Basic information

The artifact supports two reproduction paths, both exercised here:

* **(a) Replay** -- re-derive every reported number and regenerate every
  figure from the committed pre-computed result logs. No GPU, no model
  download, minutes. Command: `bash replay.sh`.
* **(b) Reproduce** -- re-run the pipeline end-to-end (fine-tune -> quantize
  -> extract -> metrics). It runs per experiment and also offers a reduced
  single-cell `quick` mode. Command: `bash reproduce.sh`.

Model weights and the GGUF/AWQ/GPTQ quantized files are **not committed**
(large and fully regenerable); what is committed is the code, the pinned
environment, the manifest, and the per-seed result logs that back every
number.

**Execution environment.** Linux x86-64. Python at the version pinned in
`.python-version`, managed by `uv`. `llama.cpp` is built CPU-only (all GPU
work is PyTorch).

**Hardware and wallclock requirements:**

| Cells | Hardware |
|---|---|
| 0.5B / 1B / 1.5B full FT, all LoRA cells, all ablations | 16 GB-class GPU: RTX 5060 Ti 16 GB (sm_120 Blackwell, needs torch 2.7.1+cu128); an RTX 3060 12 GB also runs the 0.5B/1.5B pieces |
| Llama-3.2-3B and Qwen2.5-7B full FT | A100 80 GB (rented pod) |
| Replay (path a) | any CPU |

**On timings.** The only wallclock figures stated in this README are those
that were actually measured. The fine-tune phase of each cell is instrumented:
its duration is recorded as per-step timestamps in the committed
`experiment/results/*/train_steps.jsonl` telemetry, so the reviewer can verify
it directly. Summed over the headline cells, the measured fine-tune phase is
~37 h on a 16 GB GPU. The subsequent steps (GGUF / AWQ / GPTQ quantization and
verbatim extraction) and the analysis experiments are **not instrumented**;
this README therefore gives **no wallclock figure** for them or for the
end-to-end totals, rather than an estimate. The original study ran the cells
in parallel across three machines.

A full fine-tune of a 3B/7B model does not fit a 16 GB GPU; those two full-FT
cells used a rented A100 80 GB pod. Every other cell, including all LoRA
cells, runs on a single 16 GB-class GPU. We recommend ~32 GB of RAM and
~80 GB of free disk for path (b) (checkpoints, GGUF/AWQ/GPTQ files, and the
HuggingFace cache); path (a) uses ~2 GB of disk and ~4 GB of RAM.

# Dependencies

**Software dependencies** (all version-pinned):

* Python -- version in `.python-version`.
* Python packages -- declared in `pyproject.toml` and locked in `uv.lock`
  (exact resolution, including torch 2.7.1+cu128 for the Blackwell GPU).
  Manager: `uv` (`https://docs.astral.sh/uv/`).
* `llama.cpp` -- built from source at the commit pinned in
  `scripts/build_llama_cpp.sh` (CPU-only build; used for the GGUF k-quants and
  `llama-cli` inference).
* All model/dataset ids, revisions, seeds, and hyperparameters are in
  `EXPERIMENT_MANIFEST.yaml`.

**Benchmarks and datasets.** The pipeline uses only public sources,
downloaded automatically from the HuggingFace Hub on first run:

* **Enron** (AESLC subset) -- e-mail corpus from which the PII canaries and
  the in-distribution non-members are derived.
* **WikiText-2** -- text used for perplexity and for the out-of-distribution
  set.

The backbone models (Qwen2.5 0.5B/1.5B/7B, Llama-3.2 1B/3B) are downloaded
from the HuggingFace Hub. Some model downloads may require prior acceptance of
the terms of use on the model page and a HuggingFace token
(`huggingface-cli login`); no paid credential is required.

# Security concerns

Running the artifact poses **no risk** to the reviewer. Clarifications:

* The PII "canaries" are **synthetic**, generated by code
  (`src/qquilt/canaries.py`) from fixed seeds; no real personal data is
  injected into the synthetic fine-tuning. The real-PII experiment (Enron)
  uses only the AESLC corpus, which is already public.
* The pipeline downloads public models and datasets from the HuggingFace Hub
  and builds `llama.cpp` from source; it executes no arbitrary remote code and
  requires no root privileges (building `llama.cpp` needs only a C/C++
  toolchain and `cmake`).
* No network service is exposed and no data leaves the reviewer's machine.

# Installation

```bash
git clone https://anonymous.4open.science/r/quantizer-pii-mitigation-33B5/
cd quantizer-pii-mitigation-33B5

# (1) Python environment (paths a and b)
#     downloads the pinned wheels incl. torch (several GB); time is network-bound
uv sync --no-install-project

# (2) path b only: CPU-only build of llama.cpp; time is CPU-core-bound
bash scripts/build_llama_cpp.sh
```

After step (1), path (a) can already be run. Step (2) is needed only for
path (b) (the pipeline re-run), since the GGUF k-quants depend on
`llama-quantize` / `llama-cli`.

# Minimal test

To confirm the artifact is correctly installed before any long run:

```bash
bash replay.sh --figures-only      # measured ~5 s on the reference host, no GPU, no download
```

Expected result: the 5 paper figures (`fig_crossfamily`, `fig_dose_response`,
`fig_mechanism`, `fig_mia_combined`, `fig_quant_variants`) are rendered into
`experiment/figures/` from the committed logs, and the script prints
`saved fig_*` for each one. If this works, the code and logs are correctly
accessible.

# Experiments

## Reviewer time budget

A full sequential re-run on a single 16 GB GPU is not feasible inside a review
window (the measured fine-tune phase alone is ~37 h; see the table below). The
artifact therefore offers three levels; **the first two are the recommended
reviewer path**:

| Level | Command | Measured time | Hardware | What it shows |
|---|---|---|---|---|
| Verify (all numbers) | `bash replay.sh verify` | ~1 s | any CPU, no GPU | Checks every published number against the committed logs at the paper's printed precision (see `docs/REPRODUCIBILITY_REPORT.md`) |
| Replay (all claims) | `bash replay.sh` | ~7 s (measured) | any CPU, no GPU | Re-derives every table and figure of all 5 claims from the committed logs |
| Quick re-run (reduced) | `bash reproduce.sh quick` | fine-tune ~85 min (measured); quantize + extract not instrumented | one 16 GB GPU | Re-runs one cell from scratch; confirms the pipeline genuinely produces the Claim-1 gap |
| Full re-run | `bash reproduce.sh` | fine-tune phase ~37 h (measured, summed); quantize / extract / analysis not instrumented | 16 GB GPU + A100 80 GB | Re-runs every cell and ablation end-to-end |

**Recommended for review under a time budget:** run *Replay* (measured ~7 s;
verifies the numbers behind all five claims) and *Quick re-run* (one fine-tune
of ~85 min measured, plus an un-instrumented quantize and extract; verifies the
pipeline itself runs and reproduces the headline effect). The full re-run, and
the per-claim full commands below, are for reviewers with the hardware and
time.

**Measured fine-tune-phase wallclock** (from the committed
`experiment/results/*/train_steps.jsonl` per-step timestamps; this is the one
component instrumented end-to-end):

| Cell | Hardware | Fine-tune phase |
|---|---|---|
| Qwen2.5-0.5B full FT | 16 GB GPU | ~80-85 min / seed |
| Llama-3.2-1B full FT | 16 GB GPU | ~55-65 min / seed |
| Qwen2.5-1.5B full FT | 16 GB GPU | ~135-137 min / seed |
| LoRA cells (0.5B / 1B / 3B) | 16 GB GPU | ~100-145 min / seed |
| Llama-3.2-3B full FT | A100 80 GB | ~158 min |
| Qwen2.5-7B full FT | A100 80 GB | ~69 min |

Quantization (GGUF / AWQ / GPTQ) and verbatim extraction run after the
fine-tune phase. Those steps are **not instrumented**: this README gives no
wallclock figure for them, for the analysis experiments, or for the
end-to-end totals, since none was measured.

`replay.sh` recomputes the per-seed verbatim-extraction metrics
(`python -m qquilt.metrics`), the pooled cross-seed statistics
(Fisher / Clopper-Pearson / Benjamini-Hochberg, via
`scripts/exp_stats_aggregation.py`), prints each paper table next to the
result file it came from, and regenerates the 5 figures -- all from the
committed logs, no GPU. The full `replay.sh` was measured at ~7 s on the
reference host (RTX 5060 Ti workstation).

`reproduce.sh` is a **per-experiment dispatcher**. `bash reproduce.sh --list`
shows the experiment names; `bash reproduce.sh <name> [<name>...]` runs only
the named ones; `bash reproduce.sh quick` runs the reduced single-cell check;
`bash reproduce.sh` with no argument runs everything. The configuration file
to edit, if desired, is `EXPERIMENT_MANIFEST.yaml` (models, datasets, seeds,
hyperparameters); by default no change is needed. Every step is idempotent: it
checks for its own output and skips if present, so an interrupted or restarted
run resumes without starting over.

Below, each of the paper's five claims (C1-C5) has a **reduced version**
(fits a review window) and a **full version**.

## Claim 1 -- AWQ leaks less verbatim PII than the calibration-free GGUF k-quant, across 0.5B-7B and both regimes

* **Reduced version** -- `bash reproduce.sh quick`
  * Time: fine-tune phase ~85 min (measured in `train_steps.jsonl` for this
    cell); the following quantize and extract steps are not instrumented.
    Hardware: one 16 GB GPU. Disk: ~10 GB.
  * Re-runs one cell end-to-end (Qwen2.5-0.5B, full FT, seed 42): fine-tune ->
    GGUF/AWQ quantize -> extract -> metrics.
  * Expected result: in
    `experiment/results/wave_1_qwen05b_full_seed42/metrics.json`, the AWQ
    verbatim-extraction rate is far below the Q4_K_M rate, with BF16 highest.
* **Full version** -- `bash reproduce.sh headline ablations`
  * Time: the fine-tune phase of the 0.5B/1B/1.5B + LoRA cells is ~37 h
    (measured, summed); the ablations, quantization and extraction are not
    instrumented. The 3B and 7B full-FT cells need an A100 80 GB.
  * Hardware: 16 GB-class GPU + A100 80 GB. Disk: ~80 GB. RAM: ~32 GB.
  * Config / flags: seeds and hyperparameters in `EXPERIMENT_MANIFEST.yaml`.
  * Expected result: AWQ and GPTQ show a much lower verbatim-extraction rate
    than Q4_K_M at the same bit-rate, across all backbones and regimes
    (Table `tab:headline`, Figures `fig:crossfamily` and `fig:dose-response`).
* **Numbers-only check** -- `bash replay.sh` (measured ~7 s, no GPU) prints
  Table `tab:headline` recomputed from the committed logs.

## Claim 2 -- A three-factor mechanism explains the gap between quantizers

* **Reduced version** -- `bash replay.sh`
  * Time: measured ~7 s on the reference host. Hardware: any CPU, no GPU.
  * Re-derives Tables `tab:threefactor` and `tab:saliency` and Figure
    `fig:mechanism` from the committed mechanism logs.
  * Expected result: the three factors (rare-token noise concentration,
    moderate-confidence window, calibration-induced amplification) match the
    paper.
* **Full version** -- `bash reproduce.sh mechanism saliency`
  * Time: not instrumented (the analysis reuses a fine-tuned checkpoint).
    Hardware: 16 GB-class GPU. Disk: ~20 GB.
  * Note: the mechanism experiments analyse a fine-tuned checkpoint; run
    `bash reproduce.sh headline` first (or `quick` for a smaller checkpoint)
    so a checkpoint exists.
  * Expected result: reproduces Table `tab:threefactor`, Figure
    `fig:mechanism`, and Table `tab:saliency`.

## Claim 3 -- The prior "methods-are-equivalent" MIA result is an out-of-distribution non-member artefact

* **Reduced version** -- `bash replay.sh`
  * Time: measured ~7 s on the reference host. Hardware: any CPU, no GPU.
  * Re-derives Table `tab:mia-indist` and Figure `fig:mia-combined` from the
    committed MIA logs.
* **Full version** -- `bash reproduce.sh mia`
  * Time: not instrumented. Hardware: 16 GB-class GPU. Disk: ~10 GB.
  * Expected result: with in-distribution non-members and with LiRA
    (TPR@FPR=1%), the gap between quantizers holds, unlike with
    out-of-distribution non-members (Table `tab:mia-indist`, Figure
    `fig:mia-combined`).

## Claim 4 -- The AWQ utility cost falls with scale, making it Pareto-favourable at production scale

* **Reduced version** -- `bash replay.sh`
  * Time: measured ~7 s on the reference host. Hardware: any CPU, no GPU.
  * Re-derives Tables `tab:utility` and `tab:downstream` from the committed
    perplexity and benchmark logs.
* **Full version** -- `bash reproduce.sh utility downstream`
  * Time: not instrumented. Hardware: 16 GB-class GPU. Disk: ~20 GB.
  * Expected result: the AWQ accuracy loss shrinks as the model grows,
    becoming negligible at production scale (Tables `tab:utility` and
    `tab:downstream`).

## Claim 5 -- On real Enron PII the member/non-member gap is small and AWQ collapses it

* **Reduced version** -- `bash replay.sh`
  * Time: measured ~7 s on the reference host. Hardware: any CPU, no GPU.
  * Re-derives Table `tab:natcan` from the committed natural-canary logs.
* **Full version** -- `bash reproduce.sh natural_canaries`
  * Time: not instrumented. Hardware: 16 GB-class GPU. Disk: ~10 GB.
  * Expected result: on real Enron PII (instance frequency <= 3) the
    member/non-member gap is small and AWQ collapses it (Table `tab:natcan`).

**Full reproduction.** `bash reproduce.sh` with no arguments runs, in order:
the 8 headline cells, the quantizer ablations (GGUF dose-response, AWQ
group-size sweep, GPTQ), the AWQ saliency 2x2, the mechanism experiments, the
MIA reconciliation / utility / downstream / natural-canary experiments, the
supporting analyses, the pooled statistics, and a final figure render. The
only measured component of the total is the fine-tune phase (see the
fine-tune-phase table above, ~37 h summed on a 16 GB GPU); the remaining
quantization, extraction and analysis steps are not instrumented.

**Paper item -> script -> results mapping** (for the reviewer to locate the
origin of each number):

| Paper item | Driver script | Results dir(s) | Figure script |
|---|---|---|---|
| Table `tab:headline` | `exp_3seed_replication.sh`, `exp_5seed_extra.sh`, `exp_extra_run.sh` | `wave_1_mini/`, `wave_1_seed{52,62,72,82}/`, `wave_1_qwen05b_seed*/`, `wave_1_qwen15b_seed*/`, `wave_1_llama32_3b_fullft_seed42/`, `wave_1_qwen25_7b_seed42/`, `wave_1_*_lora_seed*/` | -- |
| Fig `fig:crossfamily` | (same as `tab:headline`) | (same) | `fig_crossfamily.py` |
| Fig `fig:dose-response` | `step_8_gguf_lowbit_extension.sh`, `step_8b_q4ks.sh`, `exp_reviewer_polish.py` | `step_8_gguf_lowbit/`, `step_8b_q4ks/`, `reviewer_polish/` | `fig_dose_response.py` |
| Fig `fig:quant-variants` | -- (bpw values) | -- | `fig_quant_variants.py` |
| Table `tab:awq-sweep` | `step_7_awq_granularity_sweep.sh` | `step_7_awq_granularity/` | -- |
| Table `tab:gptq` | `exp_gptq_4bit.sh`, `exp_gptq_multiseed.sh` | `exp_gptq_4bit/`, `exp_gptq_seed{52,62}/` | -- |
| Table `tab:saliency` | `exp_saliency_2x2.sh`, `step_5_awq_canary100.sh`, `step_6_awq_wikitext.sh` | `exp_saliency_2x2/`, `step_5_awq_canary100/`, `step_6_awq_wikitext/` | -- |
| Table `tab:threefactor` | `exp_bucket_collapse_canary_v2.py`, `exp_mechanism_*.py/.sh` | `exp_mechanism*/` | `fig_mechanism.py` |
| Table `tab:mia-indist` | `exp_minkpp_reconciliation.py`, `exp_mia_indist_nonmembers.py`, `exp_tpr_at_low_fpr.py` | `exp_mia_indist/`, `exp_tpr_at_fpr/`, `exp_minkpp_reconciliation/` | `fig_mia_combined.py` |
| Table `tab:downstream` | `exp_downstream_utility.sh` | `exp_downstream/` | -- |
| Table `tab:utility` | `exp_utility_3seed_fold.sh`, `utility_eval.sh` | `wave_1_utility/`, `wave_1_utility_seed{52,62}/` | -- |
| Table `tab:natcan` | `exp_natural_canaries.py`, `exp_natural_canaries_compare.py` | `wave_1_llama32_3b_fullft_seed42/`, `wave_1_qwen25_7b_seed42/`, `natural_canaries/` | -- |
| Table `tab:defense-pareto` | (derived from `tab:headline` + `tab:utility`) | -- | -- |

`experiment/results/INDEX.md` gives the per-directory breakdown;
`experiment/results/SCHEMA.md` documents the JSONL schemas.

# Known caveats

* The 3B and 7B full-FT cells are single-seed (compute budget); the 5-seed
  Llama-3.2-1B anchor carries the multi-seed statistical weight. Cross-cell
  comparisons in the paper respect this seed-count difference.
* AWQ group size 256 quantizes but is not inference-able on the available
  kernels (autoawq's Triton GEMM supports only {32, 64, 128}); the group-size
  sweep uses {32, 64, 128}, which spans the relevant effective-bpw range.
* Extraction uses a raw memorisation probe: greedy / temperature sampling with
  the model's chat-tuned `generation_config` neutralised (repetition penalty
  1.0), so the HuggingFace decode path matches `llama-cli --temp 0`. This
  matters for instruct checkpoints whose `generation_config.json` ships a
  repetition penalty (e.g. Qwen2.5).
* `bitsandbytes 0.44.1` (pinned for historical reasons) is unused by the
  pipeline and disabled; it is non-functional against the installed Triton and
  ships no cu128 binary.

# LICENSE

This artifact is distributed under the MIT License. The full text is in the
`LICENSE` file at the repository root.
