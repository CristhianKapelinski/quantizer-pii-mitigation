# Not All 4-bit Quantizers Are Equal: Deployment-Time Mitigation of PII Leakage in Fine-Tuned Small Language Models

Reproducibility artifact for the paper. Author and affiliation withheld
during double-blind review.

This repository reproduces every table and figure in the paper. Two
reproduction paths are supported and both are exercised here:

* **(a) Replay** -- re-derive every reported number and regenerate every
  figure from the committed pre-computed result logs. No GPU, no model
  download, minutes. Run `bash replay.sh`.
* **(b) Reproduce** -- re-run the full pipeline end-to-end (fine-tune ->
  quantize -> extract -> metrics) across all 8 (backbone, regime) cells
  and every ablation. A 16 GB-class GPU plus, for the 3B/7B full-FT
  cells, an A100 80 GB; on the order of a day of wallclock. Run
  `bash reproduce.sh`.

Model weights and GGUF/AWQ/GPTQ quantized files are not shipped (large
and fully regenerable); what ships is the code, the pinned environment,
the experiment manifest, and the per-seed / per-version result logs that
back every number.

## What the paper shows (one line)

Choosing a 4-bit quantizer is a deployment-time privacy decision. The
calibration-based methods AWQ and GPTQ suppress verbatim PII leakage far
more than the calibration-corpus-free GGUF k-quant Q4_K_M at the same
bit-rate, across five open models from 0.5B to 7B and both fine-tuning
regimes, at negligible accuracy cost at production scale.

## Repository layout

```
.
|-- README.md                   this file
|-- replay.sh                   path (a): re-derive tables + figures from committed logs (no GPU)
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

## Path (a): replay (no GPU, minutes)

```bash
uv sync --no-install-project
bash replay.sh
```

`replay.sh` re-runs only the analysis steps on the committed
`experiment/results/**` logs: per-seed verbatim-extraction metrics
(`python -m qquilt.metrics`), the pooled cross-seed Fisher / Clopper-
Pearson / Benjamini-Hochberg statistics (`scripts/exp_stats_aggregation.py`),
a summary that prints each paper table next to the result file it came
from, and a regeneration of the 5 paper figures into `experiment/figures/`.
No fine-tuning, no quantization, no model download. Use
`bash replay.sh --figures-only` to render only the figures.

## Path (b): reproduce (16 GB-class GPU + A100 for 3B/7B, hours)

```bash
uv sync --no-install-project
bash scripts/build_llama_cpp.sh    # llama.cpp at the commit pinned in the script, CPU-only build
export HF_HOME=/path/to/hf/cache   # optional; default is ./cache/hf
bash reproduce.sh                  # full pipeline: canaries -> corpus -> fine-tune ->
                                   # GGUF/AWQ/GPTQ quantize -> extract -> metrics -> figures
```

`reproduce.sh` runs, in order: the 8 headline cells (Table tab:headline),
the quantizer ablations (GGUF dose-response, AWQ group-size sweep, GPTQ),
the AWQ saliency 2x2, the five mechanism experiments, the MIA
reconciliation / utility / downstream / natural-canary experiments, the
supporting analyses, the pooled statistics, and a final figure-render.
Every step is idempotent: it checks for its own output and skips if
present, so a killed or rebooted run resumes cleanly.

## Hardware and wallclock

| Cells | Hardware | Wallclock |
|---|---|---|
| 0.5B / 1B / 1.5B full FT, all LoRA cells, all ablations | RTX 5060 Ti 16 GB (sm_120 Blackwell, needs torch 2.7.1+cu128); RTX 3060 12 GB also runs the 0.5B/1.5B pieces | ~1 day total, sequential |
| Llama-3.2-3B and Qwen2.5-7B full FT | A100 80 GB (rented pod) | ~3-4 h |
| Replay (path a) | any CPU | minutes |

A full fine-tune of a 3B/7B model does not fit a 16 GB GPU; those two
full-FT cells used a rented A100 80 GB pod. Every other cell, including
all LoRA cells, runs on a single 16 GB-class GPU. `llama.cpp` is built
CPU-only -- all GPU work is PyTorch.

## Table / figure -> script -> results mapping

| Paper item | Driver script | Results dir(s) | Figure script |
|---|---|---|---|
| Table tab:headline | `exp_3seed_replication.sh`, `exp_5seed_extra.sh`, `exp_extra_run.sh` | `wave_1_mini/`, `wave_1_seed{52,62,72,82}/`, `wave_1_qwen05b_seed*/`, `wave_1_qwen15b_seed*/`, `wave_1_llama32_3b_fullft_seed42/`, `wave_1_qwen25_7b_seed42/`, `wave_1_*_lora_seed*/` | -- |
| Fig fig:crossfamily | (same as tab:headline) | (same) | `fig_crossfamily.py` |
| Fig fig:dose-response | `step_8_gguf_lowbit_extension.sh`, `step_8b_q4ks.sh`, `exp_reviewer_polish.py` | `step_8_gguf_lowbit/`, `step_8b_q4ks/`, `reviewer_polish/` | `fig_dose_response.py` |
| Fig fig:quant-variants | -- (bpw values) | -- | `fig_quant_variants.py` |
| Table tab:awq-sweep | `step_7_awq_granularity_sweep.sh` | `step_7_awq_granularity/` | -- |
| Table tab:gptq | `exp_gptq_4bit.sh`, `exp_gptq_multiseed.sh` | `exp_gptq_4bit/`, `exp_gptq_seed{52,62}/` | -- |
| Table tab:saliency | `exp_saliency_2x2.sh`, `step_5_awq_canary100.sh`, `step_6_awq_wikitext.sh` | `exp_saliency_2x2/`, `step_5_awq_canary100/`, `step_6_awq_wikitext/` | -- |
| Table tab:threefactor | `exp_bucket_collapse_canary_v2.py`, `exp_mechanism_*.py/.sh` | `exp_mechanism*/` | `fig_mechanism.py` |
| Table tab:mia-indist | `exp_minkpp_reconciliation.py`, `exp_mia_indist_nonmembers.py`, `exp_tpr_at_low_fpr.py` | `exp_mia_indist/`, `exp_tpr_at_fpr/`, `exp_minkpp_reconciliation/` | `fig_mia_combined.py` |
| Table tab:downstream | `exp_downstream_utility.sh` | `exp_downstream/` | -- |
| Table tab:utility | `exp_utility_3seed_fold.sh`, `utility_eval.sh` | `wave_1_utility/`, `wave_1_utility_seed{52,62}/` | -- |
| Table tab:natcan | `exp_natural_canaries.py`, `exp_natural_canaries_compare.py` | `wave_1_llama32_3b_fullft_seed42/`, `wave_1_qwen25_7b_seed42/`, `natural_canaries/` | -- |
| Table tab:defense-pareto | (derived from tab:headline + tab:utility) | -- | -- |
| Sec asymmetry (semantic / stress) | `exp_semantic_similarity.py`, `exp_stronger_attacker.sh` | `exp_semantic_similarity/`, `exp_stronger_attacker/` | -- |
| Sec threat-split (unlearning null) | `step_9_zhang_nl_replication.sh` | `step_9_zhang_nl_replication/` | -- |
| Sec limitations (ACR) | `exp_acr.py` | `exp_acr/` | -- |

`experiment/results/INDEX.md` gives the per-directory breakdown;
`experiment/results/SCHEMA.md` documents the JSONL schemas.

## What is pre-computed and what is regenerable

Committed: all `experiment/results/**` JSON/JSONL logs (extraction
traces, metrics, training telemetry, perplexity, pooled statistics,
per-experiment RESULTS.md). Not committed (large, regenerable by
`reproduce.sh`): fine-tuned checkpoints, `*.safetensors`, `*.gguf`, the
AWQ/GPTQ `quantized*/` directories, the HuggingFace cache, the
`llama.cpp` build, the held-out / OOD text corpora (`enron_holdout.txt`,
`wikitext2_ood.txt`), and the intermediate `*.npz` logit dumps; these
are produced from the public Enron (AESLC subset) and WikiText-2 sources
plus the committed seeds.

## Dependency pinning

Python deps in `pyproject.toml` + `uv.lock`; Python version in
`.python-version`; the `llama.cpp` commit in `scripts/build_llama_cpp.sh`;
model / dataset ids, revisions, seeds, and hyperparameters in
`EXPERIMENT_MANIFEST.yaml`.

## Known caveats

* The 3B and 7B full-FT cells are single-seed (compute budget); the
  5-seed Llama-3.2-1B anchor carries the multi-seed statistical weight.
  Cross-cell comparisons in the paper respect this seed-count difference.
* AWQ group size 256 quantizes but is not inference-able on the
  available kernels (autoawq's Triton GEMM supports only {32, 64, 128});
  the group-size sweep uses {32, 64, 128}, which spans the relevant
  effective-bpw range.
* Extraction uses a raw memorisation probe: greedy / temperature
  sampling with the model's chat-tuned `generation_config` neutralised
  (repetition penalty 1.0), so the HuggingFace decode path matches
  `llama-cli --temp 0`. This matters for instruct checkpoints whose
  `generation_config.json` ships a repetition penalty (e.g. Qwen2.5).
* `bitsandbytes 0.44.1` (pinned for historical reasons) is unused by the
  pipeline and disabled; it is non-functional against the installed
  Triton and ships no cu128 binary.
