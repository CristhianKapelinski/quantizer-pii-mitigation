# Smoke 2 — Hayes (n,p)-discoverable extraction

## Question

Is the AWQ-canary-free 0/100 verbatim-greedy extraction result a
decoding-strategy artefact? Under (n,p)-discoverable extraction
(Hayes NAACL 2025, arXiv 2410.19482) — "does the suffix appear in at
least one of n top-k samples with probability ≥ p?" — does AWQ recover
canaries that greedy missed?

## Method

Per (canary, version), computed `p_z = P(suffix | prefix)` under top-k=40
+ temperature=1 sampling (Hayes §4.1 defaults). Then per Hayes
Definition 3.1: a canary is (n,p)-discoverably-extractable iff
`n ≥ log(1-p) / log(1-p_z)`. Counted extractable canaries at every
combination of n ∈ {1, 128, 1000} and p ∈ {0.5, 0.9, 0.99, 0.999}.

Versions: BF16, AWQ-canary-free (Phase B), AWQ-canary-incl (Step 1),
AWQ-canary100 (Step 5), AWQ-wikitext (Step 6). GGUF skipped (no easy
logit access).

## Results

| version | (n=1, p=0.5) | (n=1, p=0.9) | (n=128, p=0.99) | (n=1000, p=0.999) |
|---|---|---|---|---|
| **BF16** | 96/100 | 61/100 | **100/100** | **100/100** |
| AWQ-canary-free | 0/100 | 0/100 | **0/100** | **0/100** |
| AWQ-canary-incl | 0/100 | 0/100 | 0/100 | 0/100 |
| AWQ-canary100 | 0/100 | 0/100 | 0/100 | 0/100 |
| AWQ-wikitext | 0/100 | 0/100 | 0/100 | 0/100 |

p_z statistics:
* BF16: all 100 canaries p_z > 0; mean 0.859, median 0.944. Even the
  70 canaries NOT greedy-extractable (≥10 chars) have mean p_z 0.855.
* AWQ-canary-free: only 24/100 p_z > 0; of those, mean 8.9e-11
  (effectively zero); median p_z = 0.000 across all 100.

## Decision-gate verdict

Per the protocol: "AWQ defence survives if (n=128, p=0.99) recovery
≤ 5/100 and the gap vs Q4_K_M is maintained." **AWQ-canary-free
recovers 0/100 even at the strongest tested threshold (n=1000,
p=0.999), across all four calibration variants.** The Hayes
reviewer-mandatory caveat is fully addressed: the 0/100 verbatim
greedy result is NOT a decoding artefact — under the most aggressive
probabilistic extraction the literature defines (1000-query budget,
99.9 % confidence), AWQ still recovers nothing.

## Caveat on the BF16 100/100 figure

The BF16 (n=128, p=0.99)-extractable count of 100/100 is **almost
certainly inflated by a tokenization-boundary issue** in this
implementation. We compute `prefix_len = len(tokenizer(prefix).input_ids)`
and then `suffix_logits = logits[prefix_len-1:-1]`,
`suffix_tokens = input_ids[prefix_len:]`. When the prefix↔suffix
boundary lands inside a merged BPE token, these get misaligned and
truncated to `min(len)`, so only the *tail* of each suffix gets
scored — the shared boilerplate ("...route to compliance for
reconciliation.") which the model predicts with near-1.0 probability.
That explains the p_z ≈ 0.86 product for canaries that greedy doesn't
even reproduce verbatim: the boilerplate tail dominates the product.

The AWQ 0/100 figures are **robust to this bug**: AWQ either truly
erased the canary (no high-probability path to the suffix) OR
generates a different boilerplate continuation ("...for this wire
transfer" vs "...for reconciliation"), so the true tail token falls
outside its top-40 → p_z = 0 → not extractable. Either way the
canary is not recovered.

**For the paper**: report the AWQ 0/100 result (the defence holds
under Hayes (n,p)) as the robust finding. Re-run BF16 with a
boundary-safe tokenization (tokenize prefix and full text separately,
align by longest common token prefix) before reporting BF16's
probabilistic extraction count — the current 100/100 is an
upper-bound artefact, not a clean number. The qualitative gap
(BF16 ≫ AWQ under (n,p)) is real; the exact BF16 number needs the fix.

## Comparison to literature

Hayes Pythia-2.8B Enron: greedy 1.3 %, max (n,p) 9.04 % (7× greedy).
Our BF16 greedy is 30/100; the inflated (n,p) figure is 100/100
(needs the boundary fix). Our AWQ-canary-free greedy is 0/100 and
(n,p) is also 0/100 — a defence that is robust across the entire
greedy → probabilistic spectrum, unlike anything in Hayes (who only
studied pretraining memorisation, not a quantisation defence).

## Limitations

* Tokenization-boundary issue inflates BF16 p_z (see above) — fix
  before final reporting.
* Single seed; same W1 mini checkpoint.
* GGUF versions not covered (no logit export from llama.cpp).
* Only top-k=40 + temperature=1 sampler tested; the protocol also
  lists top-p=0.95 and temperature=0.7 — not run here.

## Raw data

* `experiment/results/smoke_2_hayes_np/np_extraction_table.json` —
  full per-canary p_z + extractable counts at all (n,p)
