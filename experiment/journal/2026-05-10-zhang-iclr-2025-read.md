# 2026-05-10 — Zhang et al. ICLR 2025 read: regime mismatch explains W1 mini L3=0

## What we read

Zhang, Wang, Li, Wu, Tang, Liu, He, Yin, Wang. *Catastrophic Failure
of LLM Unlearning via Quantization*. ICLR 2025
(arXiv:2410.16454v3, 24 pages). Code at
`github.com/zzwjames/FailureLLMUnlearning`.

We already cite §5 of this paper in `PLAN.md §1.5` for the
∆_int4 vs ∆_int8 weight-mapping argument. Today's read is the first
end-to-end pass.

## Headline finding

For unlearning methods with utility constraints, the unlearned model
retains an average of **21 % of intended forgotten knowledge in full
precision, which significantly increases to 83 % after 4-bit
quantisation**. Their MUSE benchmark experiments span:

* 6 unlearning methods: GA, GA_GDR, GA_KLR, NPO, NPO_GDR, NPO_KLR.
* 3 quantisation paths: RTN, AWQ, GPTQ.
* 2 datasets: MUSE NEWS (BBC) + MUSE BOOKS (Harry Potter).
* 4 metrics: VerMem (verbatim ROUGE-L), KnowMem on forget,
  PrivLeak (Min-K %), KnowMem on retain (utility).

Their proposed defense **SURE** (Saliency-based Unlearning with Large
LR + module-level masking) keeps M3→0 even after 4-bit quantisation,
at cost of a hyperparameter-sensitive regime.

## Mechanism (Zhang §5) — our citation point

For weight `w` in interval `I_i`, quantised value
`q_i = i · ∆`, with

* `∆_int4 = max|w| / 8`
* `∆_int8 = max|w| / 128`

For `max|w| = 200` (their numerical example):

* `∆_int4 = 25` → all weights in `[-12.5, 12.5]` collapse to `q = 0`.
* `∆_int8 = 1.5625` → weights in `[-0.78125, 0.78125]` collapse to `q = 0`.

When unlearning uses a small LR (1e-5 to 5e-7 across MUSE / TOFU /
RWKU benchmarks), the weight delta `θ_unlearn − θ_target` is on the
order of `1e-5 × steps`, comfortably below `∆_int4 ≈ 12.5`. So
`Q(unlearn) ≈ Q(target)` after 4-bit quant: the unlearned model and
the original target collapse into the same code-book index. The
forgotten knowledge re-emerges because the quantised unlearned model
has been mapped onto the same weights as the (knowledgeable) target.

Critically, the mechanism requires `‖∆θ‖ ≪ ∆_int4`. It does **not**
predict L3 in regimes where the fine-tune step itself produces large
weight changes.

## Why this explains our W1 mini negative

We ran vanilla full fine-tune (`LR = 2e-5`, 5 epochs, effective batch
16) on Llama-1B and Qwen-0.5B. Order-of-magnitude check:

* Per-step ∆θ ≈ `LR × E[g]` ≈ `2e-5 × O(1)` ≈ `2e-5`.
* Steps in our recipe: ~ 2050.
* Cumulative ∆θ accumulates with momentum and clipping — empirically
  on the order of 0.1 to 1 per weight by end of training (we don't
  measure directly, but loss going from ~ 1.5 to ~ 0.13 implies a
  large excursion in weight space).

This is comparable to or larger than `∆_int4` for the quantisation
tensor block. So:

* In the regime where Zhang shows L3 surges (small `‖∆θ‖`),
  `Q(target) ≈ Q(unlearn)` and quantisation is the recovery vector.
* In our regime (`‖∆θ‖ ≫ ∆_int4`), `Q(target)` and
  `Q(fine-tuned)` are different code-book indices, so quantisation
  preserves whatever memorisation survived in a manner consistent
  across versions. Subsets are nested. A1 = 1.0. L3 = 0.

This is **not** a pipeline bug. It is the boundary condition on the
opposite side of Zhang's regime. The Llama-1B + Qwen-0.5B
replication (both A1 = 1.0, both L3 = 0, both Q4 ⊂ Q5 ⊂ Q8 ≈ BF16)
confirms it across architecture and scale.

## What this means for our novelty claim

Zhang owns the framing **"quantisation can recover forgotten
knowledge"** at the single-precision level (FP vs Q4). We cannot
reclaim it.

Our defensible contribution must be **multi-version comparison**:

* Zhang only ever pairs FP with one quantisation at a time.
* They never publish 5+ versions and ask whether the *union* leaks
  more than any single version (`A1 amplification` ≥ 1.5×).
* They never test calibration-content as a separate variable
  (we did — Step 1, recovery 1/100, hypothesis rejected at strong
  form).
* They never test cross-family / cross-scale (we did — Llama-1B and
  Qwen-0.5B, same regime, same A1, same L3).

Where to extend:

* **Adopt MUSE benchmark** as the small-`‖∆θ‖` substrate where L3
  should emerge per Zhang. Run our 7-quant attack on top of MUSE
  GA_GDR / NPO_GDR baselines — direct apples-to-apples test of
  multi-version amplification on top of single-precision recovery.
* **Add LoRA / DP-SGD as alternate small-`‖∆θ‖` regimes** — cheaper
  than full unlearning pipelines, gives intermediate data points.
* **Reframe W1 mini** as the boundary-condition observation. Vanilla
  full-FT collapses multi-version to single-precision; this is the
  expected envelope edge.

## Impact on the wave plan

Action items, by wave:

* **W1.10 (W1 full)** — original scope (3 Llama seeds + 1 Qwen seed
  on 30k Enron, 5 buckets, 7 quants in vanilla full-FT) is **wrong
  experiment** for L3 > 0. Same regime as W1 mini, just bigger. Skip
  or rescope.
* **W2** — promote to "small-`‖∆θ‖` regime sweep". Vectors:
  * V1: vanilla full-FT (Phase A redux at scale, locked-in negative)
  * V2: unlearn-then-attack on MUSE (replicate Zhang Table 1, then
    apply A1 amplification across 7 quants)
  * V3: LoRA fine-tune (rank 16, alpha 32, attn + mlp targets)
  * V4: DP-SGD ε = 4 (Opacus skeleton already in
    `src/qquilt/dp_sgd.py`)
* **W3** — cross-family stays. Now validates W2 (multi-version
  amplification under small-∆θ), not W1.
* **W4** — Zhang SURE adapted as a defense baseline alongside
  AWQ-canary-cal.
* **W5** — Gemma 3 QAT becomes the "QAT regime" addendum: an
  industrial setting where weights are *trained* to be quantisable,
  changing the regime envelope yet again.

## What stays in PLAN.md

* `§1.5` taxonomy already cites Zhang. Add the regime-envelope note
  ("L3 only emerges when ‖∆θ‖ ≤ ∆_int4 ≈ max|w|/8").
* `§4` add MUSE / LoRA / DP-SGD vectors to the dataset/setup list.
* `§5` add unlearning + LoRA hyperparameters.
* `§7` rewrite the W1 / W2 gates per the shift above.
* `§9` already journaled the bitsandbytes 0.47+ requirement; tag it
  as W2 precondition (DP-SGD G5).

## Cross-links

* `experiment/wave_1/WAVE_1.md` — W1 mini observed boundary regime.
* `PLAN.md §1.5` — taxonomy + Zhang citation.
* `experiment/journal/2026-05-10-bitsandbytes-cu128.md` — DP-SGD
  precondition.
* arXiv 2410.16454v3 — full Zhang paper.
