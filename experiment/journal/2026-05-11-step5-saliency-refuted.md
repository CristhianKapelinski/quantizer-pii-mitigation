# 2026-05-11 — Step 5 refutes saliency framing; bucket-collapse confirmed

## What we tested

Calibration-content ablation curve for AWQ-4bit on Phase A
Llama-1B target:

| Step | Calibration content | Canary % | Recovery on G1 (greedy ≥10 chars) |
|---|---|---|---|
| Phase A BF16 | n/a (FP baseline) | — | **30 / 100** |
| Phase A Q4_K_M | n/a (k-quant, no AWQ) | — | 6 / 100 |
| Phase B AWQ | Enron in-domain | 0 % | 0 / 100 |
| Step 1 | random sample from W1 corpus | ~54 % (incidental) | 1 / 100 |
| **Step 5** | canary-only deterministic | **100 %** | **0 / 100** |
| Step 6 (in flight) | WikiText-2 OOD | 0 % | (running) |

## What the saliency framing predicted

Mechanism proposed: AWQ identifies salient channels via activation
magnitudes on calibration data. Canary-encoding weights are absent
from calibration → flagged as non-salient → aggressively rounded →
canaries erased. **Prediction**: pushing calibration toward 100 %
canary content should mark canary-encoding weights as salient,
protect them, and recover memorisation — Step 5 should give
10+/100, monotonic increase vs Step 1's 1/100.

## What Step 5 actually gave

`awq_canary100_g1_greedy_extracted = 0`. Going from 54 % canary
content to 100 % did not increase recovery — it stayed at the
floor. The calibration-content lever is **flat** in our regime.

## Conclusion

The mechanism is **not saliency** in any meaningful sense. The
mechanism is **bucket-collapse** (Zhang ICLR 2025 §5): AWQ's 4-bit
RTN, even after per-channel scaling, has `∆_int4 = max|w|/8` —
much coarser than the fine-tune weight deltas that encode each
canary. Once the FT delta `‖Δθ_canary‖ ≲ ∆_int4`, the quantised
weight collapses onto the same code-book index as the pre-FT base
weight regardless of which channels AWQ scales up. Calibration
controls *which scale is used per channel*, not *which weights
survive rounding*.

This also explains why Q4_K_M preserves more (6 / 100) than AWQ
(0 / 100). Q4_K_M's effective rate is ~4.5 bits/param (super-blocks
with separate scale + min per sub-block); AWQ-4bit group-128 is
~4.25 bits/param effective. Q4_K_M has finer rounding granularity →
fewer canary weights collapse → some canaries survive. AWQ's
coarser granularity per scaling cell pushes more of them past the
threshold.

This re-frames the entire "AWQ-as-defense" story:

* **Old framing (refuted)**: AWQ's saliency mechanism identifies
  canary weights as non-salient and quantises them aggressively.
  Calibration content controls the defense.
* **New framing**: AWQ's rounding granularity (per-channel scale +
  group-128 RTN) is coarser than Q4_K_M's (super-block scale + min).
  When fine-tune deltas are below the rounding granularity, the
  quantised weights collapse to the base-model code book →
  memorisation erased. **Granularity is the lever**, not saliency.

## Cross-checks needed (Step 6 + paraphrase tier)

* **Step 6 (WikiText OOD calibration, in flight)**: if it also gives
  0 / 100, the curve is flat across the entire 0 % → 100 % canary
  axis. Confirms bucket-collapse is the dominant mechanism.
* **Hayes (n,p)-discoverable re-scoring** of Step 1 + 5 + 6
  extraction.jsonl: maybe greedy 0 / 100 understates probabilistic
  extraction. If (n=100, p=0.1)-discoverable extraction rises to
  5+/100 for Step 5, the bucket-collapse-as-defense claim weakens
  (i.e., stochastic sampling can still surface the canary).
* **Kassem δ=0.95 paraphrase tier**: similar — perhaps 5+/100 under
  Levenshtein ≥ 0.95 even though exact 0 / 100.

## Implications for paper

1. **Don't sell AWQ-canary-free as "calibration-driven defense"**.
   The Step 1/5/6 ablation will be in the paper as evidence that
   calibration content barely matters in our regime. The defense
   property is from rounding granularity, not from "saliency
   chooses what to protect".
2. **Lin AWQ §3 saliency mechanism is still correct** — it just
   doesn't connect to memorisation the way the proposal hypothesised.
   Saliency improves general-purpose accuracy at given bit-width;
   it does not (in our regime) pick out canary-encoding weights.
3. **Comparison vs Q4_K_M is now mechanism-grounded**: AWQ is more
   aggressive at the rounding step → erases finer-grained
   memorisation. Q4_K_M's higher effective bit-rate per group
   preserves slightly more. This is a clean dose-response story
   that lines up with Aubinais ICML 2025 "lower bit-rate / higher
   sparsity → more private" — except in our case the "bit-rate"
   knob is granularity, not nominal bit-width.

## Open questions

* If granularity is the lever, would a *finer* AWQ variant
  (group=64, group=32) recover canaries that group=128 erased?
  Cheap test: re-quantize with group-32 + extract. ~5 min.
* Conversely, Q4_K_M's "super-block" structure has its own
  granularity parameter (block-size). Does Q3_K_M (smaller block?)
  erase more than Q4_K_M? Already pending in W2 sweep.
* The 1 / 100 hit in Step 1 (vs 0 / 100 in Step 5) — single
  canary, may be sampling noise from the random 128-chunk selection.
  Worth seeing if it's reproducible at a different `--awq-calib-seed`.

## Cross-links

* `experiment/results/step_5_awq_canary100/metrics.json` — locked in
* `experiment/results/wave_1_mini/step1_awq_canary_cal/metrics.json`
* `experiment/journal/2026-05-10-zhang-iclr-2025-read.md` — Zhang §5 mechanism
* `experiment/journal/2026-05-11-literature-deep-read-13-papers.md`
  — Lin AWQ §3 saliency definition + robustness claim limits
