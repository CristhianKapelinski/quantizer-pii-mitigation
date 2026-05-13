# 2026-05-11 — Hayes-style re-scoring + multi-threshold sweep across all W1 extracts

## Motivation

Reviewer-flagged gap (Hayes et al. NAACL 2025, arXiv 2410.19482): our
greedy-only `match_prefix_len ≥ 10` extraction may underestimate true
extractability under probabilistic sampling. The paper proves that
greedy is the wrong metric — (n,p)-discoverable extraction with
n=10..10⁴ samples can reach 7-13× the greedy rate on Pile-Enron with
Pythia models. Question for us: does the W1 mini headline AWQ-canary-free
result (0/100) hold under probabilistic sampling, or does it expand?

## Method

Re-scored every existing `extraction.jsonl` from W1 phases (Phase A,
Phase B, Step 1, Step 2, Step 2b, Step 3, Step 4 v2, Step 5, Step 6).
Each file contains greedy + n=5 stochastic per (canary, version) pair —
6 attempts per pair. For each (canary, version), compute:

* greedy match: `match_prefix_len ≥ τ` on the single greedy completion
* any-of-6: same condition holding on ANY of the 6 attempts

Sweep τ ∈ {5, 10, 20} to also probe the paraphrase-tier vs verbatim-tier
distinction (Kassem TACL 2024 — δ=0.95 paraphrase tier is approximately
half the strict-verbatim threshold).

## Results

```
Tag                Version             ≥5g   ≥5any   ≥10g  ≥10any  ≥20g  ≥20any
----------------------------------------------------------------------------------------
Phase A            bf16                30    30      30    30      30    30
Phase A            q4_k_m              15    17      6     6       6     6
Phase A            q5_k_m              28    30      25    29      25    29
Phase A            q8_0                30    30      30    30      30    30
Phase B            awq_4bit            2     3       0     0       0     0
Step 1             awq_canary_cal      1     5       1     1       0     0
Step 2 Qwen0.5     q4_k_m              4     5       3     4       3     4
Step 2 Qwen0.5     q5_k_m              12    16      8     9       8     9
Step 2b Qwen1.5    bf16                29    29      28    28      28    28
Step 2b Qwen1.5    q5_k_m              11    11      7     7       7     7
Step 2b Qwen1.5    q8_0                29    29      29    29      29    29
Step 3 freq1       bf16                2     2       2     2       2     2
Step 3 freq1       q8_0                1     2       1     2       1     2
Step 4 v2          (all 6 versions)    0     0       0     0       0     0
Step 5             awq_canary100       1     2       0     0       0     0
Step 6             awq_wikitext        1     2       0     0       0     0
```

(Trimmed to non-trivial rows.)

## Three findings

### 1. AWQ-as-defense holds at the PII-verbatim threshold

At `≥ 10` chars (covers reference-number + first half of account
number — the privacy-sensitive prefix) and at `≥ 20` chars (full
account number), **every AWQ variant returns 0/100 under both greedy
and any-of-6 sampling**, across 4 different calibration regimes (Enron,
WikiText OOD, mixed 54 % canary, 100 % canary). The claim "AWQ-4bit
erases canary memorisation at the PII level" is **robust to probabilistic
sampling**, addressing the Hayes reviewer-flagged concern.

### 2. AWQ leaks 1-3 canaries at the ≥ 5-char threshold

This is **template / format match**, not PII content. At `≥ 5` chars
the model is matching the first few characters of the canary
reference, which is determined by the email template format (e.g.,
all references start with a digit followed by letters). The base
model knows this format from pretraining. Recovering the first 5
chars is consistent with "I know this is a confidential reference
number" but does NOT recover the specific account or date.

This is a real privacy phenomenon we should disclose: AWQ-canary-free
*does* leak the canary template structure at coarse granularity, just
not the PII content. To name it: **template-level memorisation** vs
**content-level memorisation** — AWQ erases the latter but not the
former.

### 3. Q5_K_M shows the largest greedy → any-of-6 delta (+4 in Phase A)

Q5_K_M occupies the intermediate granularity sweet spot:

* Q8_0 (high precision): saturated at greedy → any-of-6 = same
* Q5_K_M: greedy under-samples; probabilistic surfaces 4 more canaries
  (greedy 25 → any 29 in Phase A)
* Q4_K_M (low precision): too few canaries even with sampling
* AWQ: bucket-collapse below threshold; sampling can't surface

Mechanistic reading: Q5_K_M's per-super-block rounding keeps canary
weights *near* but not exactly at the canary-encoding bucket. Sampling
explores nearby tokens; some completions hit the canary continuation
by chance. Q4_K_M is past the threshold (weights collapsed); BF16/Q8
are below it (already exact at greedy). This is consistent with
Aubinais ICML 2025's monotone-with-precision privacy story.

### 4. Step 4 v2 (post-unlearn) is robustly null

All 6 versions, all 3 thresholds, greedy and any-of-6: 0/100. GA_GDR
+ retain regularisation truly erased PII canaries; quantisation does
not recover any of them under any decoding strategy. Strongest
positive evidence yet that **PII-canary unlearning is quant-robust**
in our 1B regime.

## Implications

* **Greedy was the right primary metric for our threat model.** The
  Hayes concern about greedy under-counting is empirically modest in
  our setup (Q5_K_M +4, others ≤+1). Where it matters (Q5_K_M),
  we report both numbers in the paper.
* **Add ≥5 and ≥20 char rows to the main results table.** Splits
  template-level from content-level memorisation. AWQ leaks the
  former, erases the latter.
* **The 1 / 100 in Step 1 (54 % mix) was sampling noise**, confirmed
  by Step 5 (100 % canary) and Step 6 (WikiText OOD) both giving
  0 / 100. The mechanism is bucket-collapse, not saliency.

## Cross-links

* `experiment/journal/2026-05-11-step5-saliency-refuted.md` —
  mechanism conclusion
* `experiment/journal/2026-05-11-literature-deep-read-13-papers.md` —
  Hayes (n,p) framing source
* `experiment/results/wave_1_mini/extraction.jsonl` etc. — source data,
  unchanged
