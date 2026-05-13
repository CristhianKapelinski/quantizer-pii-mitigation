# Calibration-Based 4-bit Quantization Mitigates Verbatim PII Extraction from Fine-Tuned Small Language Models

Reproducibility artifact. Author and affiliation withheld during double-blind
review.

This repository contains everything needed to **replay** the paper's tables
from pre-computed result logs (no GPU, minutes) and to **reproduce** the full
pipeline from scratch (a 16 GB-class GPU and the pinned environment, hours).
Model weights and GGUF files are not shipped (they are large and fully
regenerable); what ships is the code, the pinned environment, the experiment
manifests, and the per-version / per-seed result logs that back every number.

## What the experiments show (one line each)

(These are the findings the artifact lets you replay and reproduce; the
write-ups in `experiment/wave_1/WAVE_1.md` and the journal carry the detail.)

* A fine-tuned small language model that memorises planted PII canaries keeps
  regurgitating them verbatim under the calibration-corpus-free GGUF k-quants
  (a smooth bits-per-parameter staircase: BF16/Q8 30/100, Q5_K_M 25, Q4_K_M 6,
  Q3_K_M 0, Q2_K 0 at seed 42; pooled over 5 seeds Q4_K_M is 20/500) but
  recovers **0/500** under AWQ-4bit, even though AWQ-4bit's effective bit-rate
  (~4.25 bpw) is below Q4_K_M's (~4.5 to 4.9 bpw). Fisher exact with
  Benjamini-Hochberg correction: AWQ vs Q4_K_M p approx 2.2e-6.
* The discriminating axis is whether the quantizer fits its rounding step using
  a calibration corpus (AWQ, GPTQ -> 0/100) or only local weight statistics
  (Q4_K_M -> 6/100), not the nominal bit-width; GPTQ-4bit at ~4.25 bpw also
  reaches 0/100, and the AWQ calibration distribution is a flat knob from 0% to
  100% canary content.
* The asymmetry is on verbatim extraction, not on membership inference: AWQ
  leaves canary-vs-holdout separable at AUC 0.97 to 1.00, reproducing the prior
  null on a Min-K%-derived metric. It holds at the semantic level too
  (All-MPNet cosine mean 0.43 for AWQ vs 0.74 for Q4_K_M).
* The cost is a modest, characterised utility hit (AWQ in-domain perplexity
  ratio ~1.11 to 1.13, out-of-domain ~1.09 to 1.10, vs Q4_K_M's ~1.05).
* Cross-family checks on Qwen2.5-0.5B / 1.5B reproduce the qualitative shape.

## Layout

```
.
├── README.md                  this file
├── replay.sh                  re-derive every reported number from the committed result logs (no GPU)
├── reproduce.sh               re-run the full pipeline from scratch (GPU + pinned env)
├── pyproject.toml, uv.lock, .python-version    pinned Python environment
├── EXPERIMENT_MANIFEST.yaml   pinned dataset / model / toolchain manifest
├── src/qquilt/                the package (canaries, data, train, quantize, extract, metrics, ...)
├── scripts/                   driver scripts (reproduce_full.sh, build_llama_cpp.sh, exp_*.sh, exp_*.py)
├── experiment/
│   ├── wave_0/WAVE_0.md, wave_1/WAVE_1.md   the consolidated narrative + per-experiment results
│   ├── journal/               lab notebook (toolchain incidents, design notes)
│   └── results/               pre-computed result logs (extraction.jsonl, metrics.json,
│                              *_steps.jsonl, ppl.json, pooled_stats*.json, RESULTS.md, ...)
│                              and INDEX.md / SCHEMA.md describing the schemas
└── ENGINEERING.md             engineering notes (toolchain, determinism, gotchas)
```

## Replay (no GPU, minutes)

```bash
uv sync --no-install-project
bash replay.sh
```

`replay.sh` re-runs only the analysis steps on the committed
`experiment/results/**` logs: per-seed verbatim-extraction metrics
(`python -m qquilt.metrics`), the pooled cross-seed Fisher / Clopper-Pearson /
Benjamini-Hochberg statistics (`scripts/exp_stats_aggregation.py`), and a
summary that prints, for each paper table, the recomputed numbers next to the
result file they came from. No fine-tuning, no quantization, no model download.

## Reproduce (16 GB-class GPU + pinned environment, hours)

```bash
uv sync --no-install-project
bash scripts/build_llama_cpp.sh    # llama.cpp commit pinned in the script (b4404), CPU-only build
export HF_HOME=/path/to/hf/cache   # or leave default (./cache/hf)
bash reproduce.sh                  # full pipeline: canaries -> corpus -> fine-tune -> GGUF/AWQ/GPTQ
                                   # quantize -> extract -> metrics, across all seeds and ablations
```

`reproduce.sh` wraps `scripts/reproduce_full.sh` plus the extra-ablation
drivers. Determinism: every step is seeded; the dataset builder, canary
generator, fine-tune, AWQ/GPTQ calibration, and extraction all take an explicit
seed. The full set of seeds is `{42, 52, 62, 72, 82}`.

## What is pre-computed and what is regenerable

Committed: all `experiment/results/**` JSONL/JSON logs (extraction traces,
metrics, training telemetry, perplexity, pooled statistics, RESULTS.md
write-ups). Not committed (large, regenerable): fine-tuned checkpoints,
`*.safetensors`, `*.gguf` files, the AWQ/GPTQ quantized model directories, the
HuggingFace cache, the `llama.cpp` build, and the held-out / OOD text corpora
(`enron_holdout.txt`, `wikitext2_ood.txt`); these are produced by
`reproduce.sh` from the public Enron (AESLC subset) and WikiText-2 sources.

## Dependency pinning

Python deps in `pyproject.toml` + `uv.lock`; Python version in
`.python-version`; the `llama.cpp` commit in `scripts/build_llama_cpp.sh`;
dataset / model ids and toolchain versions in `EXPERIMENT_MANIFEST.yaml`.
Primary hardware: an RTX 5060 Ti 16 GB (Blackwell, needs torch 2.7.1+cu128);
cross-family runs also use an RTX 3060 12 GB.

## Known caveats

* `bitsandbytes 0.44.1` (pinned for historical reasons) is non-functional
  against the installed Triton (it imports the removed `triton.ops`) and ships
  no cu128 binary; it is unused by the pipeline and disabled. See
  `experiment/journal/2026-05-12-3b-fullft-oom-and-bnb-disable.md`.
* AWQ group size 256 quantizes but is not inference-able on the available
  kernels (autoawq's Triton GEMM supports only {32, 64, 128}; no CUDA AWQ
  kernel on sm_120 / sm_86); the granularity sweep uses {32, 64, 128}, which
  spans the relevant effective-bpw range.
* Extraction uses a raw memorisation probe: greedy / temperature sampling with
  the model's chat-tuned `generation_config` neutralised (repetition penalty
  set to 1.0), so the HuggingFace decode path matches `llama-cli --temp 0`
  (which uses `--repeat-penalty 1.0`). This matters for instruct checkpoints
  whose `generation_config.json` ships a repetition penalty (e.g. Qwen2.5).
* A full fine-tune of a 3B model does not fit a 16 GB GPU; the 3B scale point,
  where present, is LoRA-merged (documented in the journal).
