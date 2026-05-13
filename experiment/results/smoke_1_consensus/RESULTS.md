# Smoke 1 — Soft Metrics Consensus across versions

## Question

Does aggregating soft text-similarity signals (ROUGE-L, Levenshtein
similarity, character Jaccard) across multiple quantised versions
reveal G1 canaries that single-version greedy verbatim extraction
missed?

## Hypothesis being tested

Quantisation Quilt thesis: each version's quantisation noise
produces a slightly different memorisation footprint, so aggregating
soft signals across versions detects canaries whose exact-prefix
match is below threshold in every individual version. If true, the
consensus G1 detection rate exceeds the best-single-version rate
(amplification A_consensus > 1).

## Method

Computed ROUGE-L F1, Levenshtein-similarity, and char-Jaccard on
existing extraction.jsonl files (Phase A, Phase B, Step 1, Step 5,
Step 6). For each (canary, version), kept the maximum signal across
greedy + n=5 stochastic completions.

Two analyses:

1. **Naive (whole-suffix)** — calibrate per-metric threshold at the
   99th percentile of G3 (null distribution); flag canary in version
   if ANY metric exceeds its threshold; consensus rule = flagged in
   ≥M versions.
2. **PII-head only** — same protocol, but compute signals only on
   the first 30 characters of each suffix (the reference number +
   first half of the account number, where the privacy-sensitive
   content lives). The remaining 86 characters of each canary suffix
   are shared boilerplate ("Please confirm receipt and route to
   compliance for reconciliation."), which the model can reproduce
   from template knowledge alone without memorising any specific
   canary.

## Result — naive whole-suffix (misleading)

With ROUGE-L threshold 0.47 (RECAP arXiv 2510.25941 floor),
Levenshtein-similarity threshold 0.45, char-Jaccard 0.96:

| source | versions | best single (greedy≥10) | M=2 consensus G1 | M=4 / 5 consensus G1 | G3 false positives |
|---|---|---|---|---|---|
| Phase A | 4 | 30/100 | 100/100 | 100/100 | 0-2/50 |
| Phase B | 5 | 30/100 | 100/100 | 31/100 (M=5) | 0-3/50 |

Wilcoxon Mann-Whitney G1 > G3, p < 1e-34.

This looks like a smoking gun until you examine the *content* of
the matched output (see disambiguation).

## Disambiguation — head vs tail

| region | length | content | BF16 mean ROUGE | AWQ-canary-free mean ROUGE |
|---|---|---|---|---|
| HEAD (chars 0–30) | 30 | reference + account (PII) | 0.555 | **0.071** (median 0.000) |
| TAIL (chars 30–116) | 86 | boilerplate template | 0.877 | 0.211 |

AWQ's whole-suffix ROUGE comes almost entirely from generating the
boilerplate phrase "Please confirm receipt and route to compliance
for this..." which is *shared across all canaries* and which the
base model can produce as a continuation of the Enron email
template. **AWQ's median PII-head ROUGE is 0.000** — it does not
memorise the canary-specific reference / account / date.

## Result — PII-head only (honest)

Calibrated G3 p99 thresholds: ROUGE-head 0.727, Lev-head 0.767.

**Phase A (4 versions: BF16, Q8_0, Q5_K_M, Q4_K_M)**

| M consensus | G1 flagged | G3 flagged | Wilcoxon p (G1>G3) |
|---|---|---|---|
| M ≥ 1 | 30/100 | 1/50 | 2.74e-5 |
| M ≥ 2 | 30/100 | 0/50 | — |
| M ≥ 3 | 30/100 | 0/50 | — |
| M = 4 | 13/100 | 0/50 | — |

Single-version PII-head flag counts:

| version | G1 flagged |
|---|---|
| BF16 | 30/100 |
| Q8_0 | 30/100 |
| Q5_K_M | 30/100 |
| Q4_K_M | 13/100 |

**Union (M ≥ 1) = 30. Best single = 30. A_consensus = 1.000.**

**Phase B (adds AWQ-canary-free as 5th version)**

| M consensus | G1 flagged |
|---|---|
| M ≥ 1 | 30/100 |
| M = 5 (all 5 versions, including AWQ) | **0/100** |

Single-version: AWQ 0/100, others same as Phase A.

## Decision gate verdict

Per the protocol:
- Original gate: GO if consensus detects ≥ 10 canaries in G1 and
  ≤ 2 in G3 — *technically* passes on the naive whole-suffix metric.
- Strict reading: A_consensus > 1 required for the Quilt thesis.
  Both Phase A and Phase B give A_consensus = 1.000 on the
  privacy-relevant PII-head metric.

**Quantisation Quilt thesis: NOT supported by Smoke 1 PII-head
analysis.** The boilerplate-driven signal was an artefact of the
shared canary template, not evidence of multi-version memorisation
amplification. Single-version verbatim extraction (≥10 char prefix)
already covers everything soft consensus catches on the PII region.

## Subordinate findings

1. **AWQ defence holds even under soft-PII-head metric.** AWQ
   median PII-head ROUGE = 0.000; flag count 0/100 in Phase B
   single-version PII-head analysis. Granularity-bucket-collapse
   mechanism (Steps 5/6) is consistent with this — AWQ erases the
   canary-specific weights but the template-encoding weights survive.
2. **Template-level memorisation is universal** across all 5
   versions in Phase B. This is itself an interesting observation
   that the paper can disclose: quantisation erases content-level
   memorisation but NOT format-level memorisation. Distinction
   names: PII memorisation vs template memorisation.
3. **Q4_K_M has fewer PII-head matches** (13/100) than Q5/Q8/BF16
   (30/100). Consistent with the granularity mechanism — Q4's
   coarser rounding pushes more canary weights into the base-model
   bucket.

## Limitations

- **G3 N is small** (50 sequences), so p99 threshold has high
  variance. Smaller N inflates extreme percentiles.
- **Stochastic decoding budget is n=5** per (canary, version). Hayes
  (n,p) re-scoring at n=128 may surface additional signal (handled
  in Smoke 2).
- **Logits were not saved during extraction** — Min-K%++ and
  per-token surprisal not computable here. Would require ~2–3 h GPU
  to recompute forward passes (Smoke 2 deliverable).
- **Single seed** for Phase A fine-tune; replication on 3 seeds is
  Wave-1-full territory.

## Next step

Move to **Smoke 2** (Hayes (n,p)-discoverable extraction). Goal: rule
out that the AWQ 0/100 PII-head result is a decoding-strategy
artefact. If AWQ remains 0/100 under (n=128, p=0.99) probabilistic
extraction, the defence claim is robust to the most aggressive
sampling adversary the literature defines.
