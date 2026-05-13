# 2026-05-11 — Deep read of 13 reference PDFs (3 sub-agents)

## Method

Delegated to three parallel sub-agents grouped by topic, each producing a
structured report (headline / method / numbers verified from PDF / direct
relevance / caveats). Total ~3700 words returned. Papers cited below are
all archived in `refs/literature/`.

## Group A — PII / memorization defenses

### Lukas et al. S&P 2023 (`lukas-2302.00539.pdf`)

* **Canonical PII-leakage baseline** on the same Enron dataset as us.
* Tests **only DP-SGD (ε=8) and scrubbing** as defenses; **no 4-bit
  quantization evaluated**.
* GPT2-Large on ECHR undefended: PII Precision 29.56 %, Recall 22.96 %;
  DP ε=8 drops to 2.92 % / 2.98 %.
* **Our gap**: AWQ-canary-free 0/100 plays in the same threat model with
  a defense axis Lukas does not consider.

### Patch (Hughes et al.) (`patch-2510.07452.pdf`)

* Uses **Llama-3.2-1B** (our exact target) plus GPT-2 family.
* Mechanistic-interpretability defense: patches shared attention-head
  edges discovered via EAP-IG circuit attribution. Achieves precision
  60 % → 0.5 % on Llama-3.2-1B.
* **Framing parallel**: Patch ablates *attention-head edges* identified
  via gradient attribution; AWQ ablates *salient channels* identified
  via activation statistics. Both blocks PII leakage to ≤1 % on
  Llama-3.2-1B. AWQ is *blunter but cheaper* — no PII labels needed.

### Ghosh et al. (`ghosh-2507.14777.pdf`)

* Three memorization measures: recollection, **counterfactual**,
  contextual. Our inserted canaries are **counterfactual by
  construction** (held out from pre-training).
* Empirically: 52.4 % of Pythia-recollection-memorized strings on the
  Pile are also recollected by OLMo trained on disjoint data → most
  "memorized" generic strings are predictable from the language prior,
  not from training data.
* **Implication for us**: AWQ-canary-free → 0/100 means the
  counterfactual signal is suppressed — privacy-relevant in Ghosh's
  taxonomy.

### Panda et al. ICLR 2025 (`panda-2503.06808.pdf`)

* Validates **Llama-3.2-1B as audit target**.
* NEW-TOKENS canary recipe verified from §4.3: tokenizer-injected tokens
  absent from pretraining, used as single-token secrets, prefix
  arbitrary, SFT loss.
* **Crucial caveat for our paper**: on Llama-3.2-1B, NEW canary
  *underperforms* unigram/bigram canaries (TPR@1%FPR 0.282 vs 0.524
  bigram). Our content-style email canaries on Llama-1B likely
  underestimate true leakage compared to a unigram canary — our BF16
  30/100 is a *lower bound*.

## Group B — Quantization mechanism + privacy

### Lin et al. AWQ MLSys 2024 (`lin-awq-2306.00978.pdf`)

* Saliency mechanism quoted verbatim from §3: "we should refer to the
  *activation* distribution instead of the *weight* distribution...
  weight channels corresponding to larger activation magnitudes are
  more salient since they process more important features."
* Robustness experiment (§5.3, PubMed↔Enron OOD): AWQ +0.5–0.6 PPL
  delta vs GPTQ +2.3–4.9. **Claim restricted to general PPL on
  natural text.** They never test whether *specific rare memorized
  strings* survive when absent from calibration.
* **Our Step 5/6 ablation fills this gap directly.**

### Frantar et al. GPTQ ICLR 2023 (`frantar-gptq-2210.17323.pdf`)

* Mechanism: H = 2XXᵀ + λI — the inverse Hessian is **computed entirely
  from calibration activations**. Update step compensates rounding
  error in column q by adjusting all unquantized weights weighted by
  (H⁻¹)_{:,q}.
* Implication: calibration content determines which weights "survive"
  rounding. **Never analyzed for privacy / memorization.**

### Wang et al. Compressed Lens (`wang-compressed-lens-2505.13963.pdf`)

* Empirical observation: AWQ damages factual-knowledge recall (FKR) in
  last layers more than BitsAndBytes. **Bug framing.**
* Our framing: AWQ damages last-layer canary recall pathways → **same
  finding, opposite sign**. They want recall preserved; we want
  canaries erased.
* Two-hop reasoning: first hop degraded by up to 30.08 % under
  quantization, second hop only 4.25 %.

### Egashira et al. NeurIPS 2024 (`egashira-2405.18137.pdf`)

* **Inverse-direction attack**: adversary places FP weights inside the
  quantization bucket of a *malicious* point so that quantized model
  reliably collapses onto bad behavior (vulnerable code, refusal
  injection).
* Targets **zero-shot RTN family** (LLM.int8, NF4, FP4) — explicitly
  excludes AWQ/GPTQ as calibration-based.
* Same underlying mechanism object (quantization buckets) but
  weaponised to *guarantee preservation* rather than to enforce
  erasure. Useful citation for "quantization buckets are the operative
  object".

## Group C — Extraction methodology + unlearn + pitfalls

### Hayes et al. NAACL 2025 (`hayes-2410.19482.pdf`)

* **(n,p)-discoverable extraction**: z is extractable iff
  Pr[∪_{w∈[n]} g(prefix)_w = suffix] ≥ p, computable from a *single
  forward pass* via the closed form
  n ≥ log(1-p) / log(1-p_z) where p_z is suffix probability under
  top-k sampling.
* Pythia 2.8B Enron: greedy = 1.3 %, max (n,p) at p=0.999 = **9.04 %
  (7× the greedy rate)**.
* **Direct implication for our setup**: greedy 0/100 may understate
  probabilistic extraction. Worth re-scoring our extraction.jsonl
  files under (n,p) — compute p_z per canary from the stochastic
  attempts we already have.

### Liu et al. NeurIPS 2025 — Unlearned but Not Forgotten (`liu-unlearned-not-forgotten-2505.24379.pdf`)

* Threat model: adversary has BOTH pre- and post-unlearning checkpoints.
  Recovers ~2× the post-unlearn rate via log-ratio guided generation.
* **Does NOT study quantization-induced recovery.** Our Step 4 result
  (GA_GDR + Q4/AWQ → 0/100 recovery) is on a different threat axis
  than Liu's two-checkpoint attack.

### Chen et al. Q-resafe ICML 2025 (`chen-qresafe-2506.20251.pdf`)

* Quantization breaks **safety alignment**, not memorization.
* AWQ INT4 raises ASR on harmful-cal from baseline 0.3 % (Llama-2-7B)
  to 42.4 % (harmful calibration).
* Parallel phenomenon: post-training compression destabilises a property
  fine-tuning installed. We cite this for the META-claim, not as same
  mechanism.

### Kassem / More TACL 2024 (`kassem-more-2407.02596.pdf`)

* Composite extraction (multiple prompts × multiple model sizes ×
  paraphrase) ~2-3× baseline rate.
* Levenshtein ratio δ=0.95 (≥95 % overlap) *doubles* extraction vs
  strict verbatim.
* **Implication**: our greedy + δ=1.0 measure is a lower bound along
  *similarity axis*; Hayes is the bound along *sampling axis*. Our
  paper should report at both δ=1.0 and δ=0.95.

### Chasing Shadows NDSS 2026 (`chasing-shadows-2512.09549.pdf`)

Pitfalls directly applicable to our setup (Llama-1B + 100 canaries +
4-bit quant + greedy extraction):

1. **P9 Model Ambiguity (73.6 %)** — must pin quantization scheme,
   group size, calibration corpus details for every reported number.
   We do, but need to consolidate into a single table.
2. **P8 Proxy/Surrogate Fallacy (47.2 %)** — Qwen-0.5B cross-family
   mitigates, but both 0.5B and 1B are small. Extrapolating to 7B+
   without testing is Surrogate Fallacy → state in limitations.
3. **P7 Prompt Sensitivity (31.9 %)** — should test 2-3 prefix
   variants per canary.
4. **P3 Data Leakage (65.2 %)** — verify canary strings are not in
   Llama-1B pretraining (synthetic canaries should be safe but argue
   for it).
5. **P6 Context Truncation (33.3 %)** — N/A; our prompts are short.

Their case study showed CodeLlama-7B 2-bit ASR 69.5 % vs no-quant
18.2 % (~3.8× swing) just from bit-width. Strong empirical evidence
for our P9 caution.

## Top consolidated takeaways for our paper

1. **Position in related work**: Lukas (DP-SGD baseline same dataset),
   Patch (mech-interp defense same target), Panda (canary methodology
   same target). Our AWQ-canary-free finding is a **third defense
   axis** alongside DP and circuit patching.
2. **Our finding strengthens, not weakens**, because:
   * On Llama-3.2-1B, content-style canaries underestimate leakage
     (Panda) → our 30/100 BF16 baseline is conservative.
   * Counterfactual signal is the privacy-relevant signal (Ghosh) →
     AWQ erasing it is the right defense behaviour.
3. **Mechanism gap we fill**: Lin/Frantar describe how saliency &
   inverse-Hessian use calibration activations but never analyze
   privacy. Step 5 (100 % canary) + Step 6 (100 % WikiText OOD) +
   Phase B (Enron, 0 % canary) + Step 1 (54 % mixed) gives the
   complete ablation curve they did not run.
4. **Probabilistic + paraphrase extraction recheck**: our greedy
   0/100 may be 5/100 under Hayes (n,p) and 10/100 under Kassem
   δ=0.95. Worth computing — re-score existing extraction.jsonl
   without new training.
5. **Methodological pitfalls to mitigate** (Chasing Shadows): pin
   quantization configs, expand to a couple more prefix variants,
   state Surrogate Fallacy limitation explicitly.

## What's *not* in this read

* Liu et al. logit-difference attack on unlearned models is on a
  different threat model than ours — cite cautiously, not as
  confirmation.
* Egashira's attack targets RTN family, not AWQ — same mechanism
  object, different attack class.
* Wang's compressed-lens result is descriptive, not mechanistic.

## Cross-links

* `experiment/journal/2026-05-10-zhang-iclr-2025-read.md` — Zhang ICLR 2025
* `experiment/journal/2026-05-10-literature-cross-check.md` — Aubinais,
  Haque, Bits-for-Privacy (the 3 papers read before today)
* `refs/literature/*.pdf` — 16 papers cached
* `experiment/wave_1/WAVE_1.md` §7 — verdict updated with these
  references when the running experiments close.
