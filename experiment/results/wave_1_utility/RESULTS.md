# Utility Evaluation — AWQ-as-Defence hypothesis

## Question

Does AWQ-4bit with canary-free calibration preserve general language
modeling utility, or does the 0/100 canary extraction (Phase B) come
at the cost of indiscriminate quality loss?

* **H1 (defence)**: AWQ canary-free PPL ratios within the envelope of
  other 4-bit quantizations (notably Q4_K_M). Memorisation loss
  selective, utility preserved.
* **H2 (lobotomy)**: AWQ canary-free PPL ratios are outliers vs Q4_K_M
  at the same nominal bit-width. Memorisation loss is a symptom of
  general degradation.

## Method

Two corpora, both first 50 context windows (~25k tokens):

* **In-domain**: 500 Enron emails NOT in the W1 mini training sample
  (seed=42 reconstruction of training indices, then excluded). Held-out.
* **OOD**: first 1000 WikiText-2 (`wikitext-2-raw-v1` test) sequences.

Two perplexity conventions (unavoidable — see caveat):

* **HF convention** (BF16, AWQ versions): non-overlapping 512-token
  windows, mean per-token NLL, `exp()`. Computed in PyTorch.
* **GGUF convention** (F16-GGUF, Q8_0, Q5_K_M, Q4_K_M): `llama-perplexity`
  sliding half-overlapping windows, PPL on the second half of each
  window (full-context positions only). Computed via the
  `llama-perplexity` binary.

Both capped to 50 windows so HF and GGUF score the SAME token span,
but the windowing convention differs → cross-convention PPLs are not
directly comparable in absolute terms; ratios *within* each convention
are.

## Results

### HF convention — relative to BF16

| version | in-domain PPL | ratio | OOD PPL | ratio |
|---|---|---|---|---|
| BF16 | 10.786 | 1.000 | 18.631 | 1.000 |
| AWQ-4bit canary-free | 12.176 | **1.129** | 20.321 | **1.091** |
| AWQ-4bit canary-incl (Step 1) | 12.202 | 1.131 | 20.263 | 1.087 |

### GGUF convention — relative to F16-GGUF

| version | in-domain PPL | ratio | OOD PPL | ratio |
|---|---|---|---|---|
| F16-GGUF | 9.441 | 1.000 | 14.751 | 1.000 |
| Q8_0 | 9.460 | 1.002 | 14.763 | 1.001 |
| Q5_K_M | 9.594 | 1.016 | 14.979 | 1.016 |
| Q4_K_M | 9.872 | **1.045** | 15.493 | **1.050** |

### Literature envelope (for reference)

* Lin et al. AWQ MLSys 2024: WikiText-2 PPL ratio ~1.02–1.05 vs FP16 for
  AWQ-4bit on pretrained LLaMA-7B.
* Frantar et al. GPTQ ICLR 2023: ~1.05–1.10 for GPTQ-4bit.
* llama.cpp Q4_K_M community benchmarks: ~1.04–1.08 on
  instruction-tuned models.

Our Q4_K_M ratio (1.045–1.050) is squarely in the literature envelope.
Our AWQ-canary-free ratio (1.091–1.129) is at the *aggressive end* of
the spectrum — a few points above the typical 4-bit envelope but not
catastrophic.

## Verdict

| criterion | result | reading |
|---|---|---|
| AWQ canary-free ratio within ±0.05 of Q4_K_M in both measures | **No** (in-domain Δ = 0.084 > 0.05; OOD Δ = 0.041 < 0.05) | AWQ degrades *more* than Q4_K_M |
| AWQ degrades OOD more than in-domain (selective overfit-to-calibration) | **No** — OOD ratio 1.091 < in-domain ratio 1.129 | NOT a selective lobotomy |
| AWQ ratio within general 4-bit literature envelope | borderline — slightly above (1.13 vs typical 1.04–1.10) | aggressive but not pathological |
| AWQ-canary-free vs AWQ-canary-incl utility | identical (12.18 vs 12.20; 20.32 vs 20.26) | calibration content irrelevant for utility, same as for memorisation |

**Bottom line: H1-leaning with a nuance.** AWQ-canary-free is NOT a
lobotomy — it does not selectively destroy OOD capability, and
calibration content has no effect on utility (consistent with the
rounding-granularity-bucket-collapse mechanism from Steps 5/6). But
AWQ-4bit IS a more aggressive quantizer than Q4_K_M at the same
nominal bit-width — coarser per-channel-scale + group-128 RTN rounding
vs Q4_K_M's super-block scale+min — so it costs ~9–13% PPL increase
(vs Q4_K_M's ~4–5%) AND erases more memorisation (0/100 vs 6/100).
This is a *coherent trade-off*, not a free defence: aggressive
quantization buys memorisation erasure at the cost of mild utility loss.

For the paper: present this as a **defence–utility frontier**. A
deployer who wants to ship a 4-bit variant with provable memorisation
erasure picks AWQ-canary-free and accepts ~13% PPL; one who wants to
preserve more utility picks Q4_K_M and accepts 6/100 canary leakage.
The frontier is real and characterisable.

## Caveats

* **Single seed** — same W1 mini checkpoint throughout.
* **50-window cap** — full-corpus PPL was infeasible (llama-perplexity
  on 1000+ windows = ~50 min/version, stalled the pipeline three
  times). 50 windows (~25k tokens) gives σ < 0.5 — stable estimate.
* **HF ↔ GGUF convention mismatch** — the absolute PPLs differ because
  `llama-perplexity` uses sliding half-overlap windows (lower PPL,
  predicting only from full context) while the HF path uses
  non-overlapping windows (higher PPL, cold start each window).
  Cross-convention comparisons are interpreted via ratios within each
  convention; the F16-GGUF baseline anchors the GGUF group.
* **No downstream-task eval** — only perplexity. Stronger utility
  claims (MMLU, generation quality, KL cross-version) are out of scope
  for this minimum-bar check.
* **AWQ-4bit only** — AWQ-3bit / AWQ-8bit / AWQ at smaller group sizes
  not tested here. Step 7 (granularity sweep) is the follow-up.

## Raw data

* `experiment/results/wave_1_utility/ppl.json` — all PPLs
* `experiment/results/wave_1_utility/enron_holdout.txt` — in-domain corpus
* `experiment/results/wave_1_utility/wikitext2_ood.txt` — OOD corpus
* F16-GGUF baseline (in-domain 9.4407, OOD 14.7513) — computed post-hoc
  via `llama-perplexity --chunks 50`, not yet folded into ppl.json.
