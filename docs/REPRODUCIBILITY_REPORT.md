# Reproducibility Report

Artifact for *Not All 4-bit Quantizers Are Equal: Deployment-Time Mitigation of PII
Leakage in Fine-Tuned Small Language Models* (SBSeg 2026).

This report has two parts: (1) hand-written known limitations and lineage notes, and
(2) an auto-generated section (rewritten by `scripts/verify_values.py`) that compares every
published number against the committed logs at the paper's printed precision.

Run it with:

```bash
bash replay.sh verify        # recompute from committed logs, then check against the paper
# or directly:
python scripts/verify_values.py
```

The ground truth is `expected/paper_values.json`, parsed from the camera-ready `main.tex`
(the only authority — shipped result files can be regenerated but the paper text is fixed).

## 1. Known limitations and lineage notes

- **Four of five figures embed their numbers.** Only `scripts/fig_dose_response.py` reads a
  result file (`reviewer_polish/m10_threshold_sensitivity.json`). `fig_crossfamily.py`,
  `fig_mia_combined.py`, `fig_mechanism.py`, and `fig_quant_variants.py` hardcode the values
  they plot. The underlying per-cell result JSONs exist and the verifier checks those numbers
  independently, but `replay.sh --figures-only` only re-draws the embedded constants; it does
  not prove the figures were generated from the logs. Wiring these four figures to read their
  JSONs is the main outstanding lineage gap.

- **`tab:threefactor` is a manual synthesis and is not exact-verified.** The columns come from
  different mechanism runs at different sample sizes: FT top-1 / L2 norm / cosine / prob-drop
  at the RECALL and BODY positions come from `exp_mechanism_control_positions` (n=50) and
  `exp_mechanism_q4km_noise_direction` (n=30), while the pooled FLIP rates come from
  `exp_mechanism_multiseed` (n=100/300) and the FLIP confidence intervals from
  `reviewer_polish/m3_flip_rate_cis.json`. Because the table mixes these pools, individual
  cells differ from any single committed file by about one unit (e.g. AWQ RECALL FLIP is 78%
  in the paper vs. 79% = 79/100 in `exp_mechanism_multiseed`; AWQ RECALL L2 norm is 841 in the
  paper vs. 827 in `control_positions` and 846 in `multiseed`). These deltas are within the
  reported confidence intervals and reflect different legitimate pools, not arithmetic errors,
  so the table is SKIP-listed rather than force-matched or silently "corrected".

- **`tab:utility` reports perplexity RATIOS, not stored values.** `wave_1_utility/ppl.json`
  commits the raw per-version perplexities, but the published ratios are quantized/BF16 against
  a per-backend f16 reference (Hugging Face for BF16/AWQ, llama.cpp for the GGUF k-quants) that
  is not persisted as its own field. The ratios are therefore a derivation and are SKIP-listed;
  the raw ppl values that feed them are committed.

- **`tab:defense-pareto` has no dedicated result file.** Its extraction column is derived from
  `tab:headline` (BF16 26.6%, Q4\_K\_M 4.0%, AWQ 0.0/3.0/6.0%) and its Δppl column from
  `tab:utility`. The verifier checks the extraction column against the headline sources; the
  Δppl column inherits the utility-ratio caveat above.

- **Qwen2.5-0.5B AWQ headline cell** (0.0%) is not present in `qwen_extra_pooled_qwen05b.json`
  (the pooled file has no AWQ entry for this backbone), so it is SKIP-listed. Every other cell
  of that pool (BF16/Q8\_0/Q5\_K\_M/Q4\_K\_M) verifies exactly.

- **LoRA BF16 3-seed pooled cells** (23.3 / 25.7 / 28.0%) have no single committed pooled file;
  only per-seed extraction logs are shipped. The verifier checks the LoRA 4-bit cells that the
  paper's claim rests on (Q4\_K\_M and AWQ = 0 at seed 42, and the lr=2e-4 knob row) and
  SKIP-lists the BF16 pooled values.

- **Datasets are resolved by Hugging Face id with no pinned revision or checksum.** Enron
  (`snoop2head/enron_aeslc_emails`), WikiText-2, and Wikipedia are downloaded implicitly on
  first use. There is no sha256 integrity check; a future revision of an upstream dataset could
  drift silently. Pinning revisions and adding checksums is recommended.

- **Timing is instrumented only for fine-tuning** (`train_steps.jsonl`). Quantization,
  extraction, and analysis wall-clocks are not recorded, so the README's per-cell times cover
  fine-tuning only. These are hardware facts, not paper numbers, and are never verified here.

- The `replay.sh` table<->source probe pointed `tab:downstream` at `exp_downstream/SUMMARY.json`,
  which does not exist; the real file is `exp_downstream/metrics.json`. The verifier uses the
  correct path, and the probe list has been left as-is (informational only).

## 2. Automatic verification results

<!-- AUTO:VERIFY:BEGIN -->
_Last verification: **97 pass / 0 fail**, 36 skip, out of 133 checked paper numbers._

### PASS

97 numbers reproduce EXACTLY at the paper's printed precision (headline extraction pools, AWQ group-size sweep, GPTQ vs AWQ vs Q4\_K\_M, saliency 2x2, Min-K%/Loss MIA AUCs, downstream accuracy, natural-canary gaps, defense-pareto extraction column).

### SKIP (documented, not verified for exact equality)

| key | paper | reason |
|---|---|---|
| `headline_greedy_ge10_extraction_pct.fullft.qwen0_5b.awq` | 0.0 | source artifact absent |
| `headline_greedy_ge10_extraction_pct.lora.qwen0_5b.bf16` | 23.3 | no recomputable artifact / derived or prior-work value |
| `headline_greedy_ge10_extraction_pct.lora.llama1b.bf16` | 25.7 | no recomputable artifact / derived or prior-work value |
| `headline_greedy_ge10_extraction_pct.lora.llama3b_lr2e5.bf16` | 28.0 | no recomputable artifact / derived or prior-work value |
| `headline_greedy_ge10_extraction_pct.lora.llama3b_lr2e5.q5_k_m` | 9.0 | no recomputable artifact / derived or prior-work value |
| `threefactor_logit_error.n.awq_recall` | 300 | Table assembled from several mechanism runs at different n (control_positions n=50, q4km_noise n=30, multiseed FLIP n=100/300); cells differ by ~1 unit depending on the pool/CI method, so no single committed artifact matches cell-by-cell. Closest sources: exp_mechanism_control_positions, exp_mechanism_q4km_noise_direction, exp_mechanism_multiseed, reviewer_polish/m3_flip_rate_cis. |
| `threefactor_logit_error.n.awq_body` | 100 | Table assembled from several mechanism runs at different n (control_positions n=50, q4km_noise n=30, multiseed FLIP n=100/300); cells differ by ~1 unit depending on the pool/CI method, so no single committed artifact matches cell-by-cell. Closest sources: exp_mechanism_control_positions, exp_mechanism_q4km_noise_direction, exp_mechanism_multiseed, reviewer_polish/m3_flip_rate_cis. |
| `threefactor_logit_error.n.q4_recall` | 300 | Table assembled from several mechanism runs at different n (control_positions n=50, q4km_noise n=30, multiseed FLIP n=100/300); cells differ by ~1 unit depending on the pool/CI method, so no single committed artifact matches cell-by-cell. Closest sources: exp_mechanism_control_positions, exp_mechanism_q4km_noise_direction, exp_mechanism_multiseed, reviewer_polish/m3_flip_rate_cis. |
| `threefactor_logit_error.n.enron` | 91 | Table assembled from several mechanism runs at different n (control_positions n=50, q4km_noise n=30, multiseed FLIP n=100/300); cells differ by ~1 unit depending on the pool/CI method, so no single committed artifact matches cell-by-cell. Closest sources: exp_mechanism_control_positions, exp_mechanism_q4km_noise_direction, exp_mechanism_multiseed, reviewer_polish/m3_flip_rate_cis. |
| `threefactor_logit_error.ft_top1.awq_recall` | 0.67 | Table assembled from several mechanism runs at different n (control_positions n=50, q4km_noise n=30, multiseed FLIP n=100/300); cells differ by ~1 unit depending on the pool/CI method, so no single committed artifact matches cell-by-cell. Closest sources: exp_mechanism_control_positions, exp_mechanism_q4km_noise_direction, exp_mechanism_multiseed, reviewer_polish/m3_flip_rate_cis. |
| `threefactor_logit_error.ft_top1.awq_body` | 0.9998 | Table assembled from several mechanism runs at different n (control_positions n=50, q4km_noise n=30, multiseed FLIP n=100/300); cells differ by ~1 unit depending on the pool/CI method, so no single committed artifact matches cell-by-cell. Closest sources: exp_mechanism_control_positions, exp_mechanism_q4km_noise_direction, exp_mechanism_multiseed, reviewer_polish/m3_flip_rate_cis. |
| `threefactor_logit_error.ft_top1.q4_recall` | 0.7 | Table assembled from several mechanism runs at different n (control_positions n=50, q4km_noise n=30, multiseed FLIP n=100/300); cells differ by ~1 unit depending on the pool/CI method, so no single committed artifact matches cell-by-cell. Closest sources: exp_mechanism_control_positions, exp_mechanism_q4km_noise_direction, exp_mechanism_multiseed, reviewer_polish/m3_flip_rate_cis. |
| `threefactor_logit_error.l2norm.awq_recall` | 841 | Table assembled from several mechanism runs at different n (control_positions n=50, q4km_noise n=30, multiseed FLIP n=100/300); cells differ by ~1 unit depending on the pool/CI method, so no single committed artifact matches cell-by-cell. Closest sources: exp_mechanism_control_positions, exp_mechanism_q4km_noise_direction, exp_mechanism_multiseed, reviewer_polish/m3_flip_rate_cis. |
| `threefactor_logit_error.l2norm.awq_body` | 654 | Table assembled from several mechanism runs at different n (control_positions n=50, q4km_noise n=30, multiseed FLIP n=100/300); cells differ by ~1 unit depending on the pool/CI method, so no single committed artifact matches cell-by-cell. Closest sources: exp_mechanism_control_positions, exp_mechanism_q4km_noise_direction, exp_mechanism_multiseed, reviewer_polish/m3_flip_rate_cis. |
| `threefactor_logit_error.l2norm.q4_recall` | 617 | Table assembled from several mechanism runs at different n (control_positions n=50, q4km_noise n=30, multiseed FLIP n=100/300); cells differ by ~1 unit depending on the pool/CI method, so no single committed artifact matches cell-by-cell. Closest sources: exp_mechanism_control_positions, exp_mechanism_q4km_noise_direction, exp_mechanism_multiseed, reviewer_polish/m3_flip_rate_cis. |
| `threefactor_logit_error.cos.awq_recall` | 0.0086 | Table assembled from several mechanism runs at different n (control_positions n=50, q4km_noise n=30, multiseed FLIP n=100/300); cells differ by ~1 unit depending on the pool/CI method, so no single committed artifact matches cell-by-cell. Closest sources: exp_mechanism_control_positions, exp_mechanism_q4km_noise_direction, exp_mechanism_multiseed, reviewer_polish/m3_flip_rate_cis. |
| `threefactor_logit_error.cos.awq_body` | 0.0079 | Table assembled from several mechanism runs at different n (control_positions n=50, q4km_noise n=30, multiseed FLIP n=100/300); cells differ by ~1 unit depending on the pool/CI method, so no single committed artifact matches cell-by-cell. Closest sources: exp_mechanism_control_positions, exp_mechanism_q4km_noise_direction, exp_mechanism_multiseed, reviewer_polish/m3_flip_rate_cis. |
| `threefactor_logit_error.cos.q4_recall` | 0.0079 | Table assembled from several mechanism runs at different n (control_positions n=50, q4km_noise n=30, multiseed FLIP n=100/300); cells differ by ~1 unit depending on the pool/CI method, so no single committed artifact matches cell-by-cell. Closest sources: exp_mechanism_control_positions, exp_mechanism_q4km_noise_direction, exp_mechanism_multiseed, reviewer_polish/m3_flip_rate_cis. |
| `threefactor_logit_error.probdrop.awq_recall` | 55 | Table assembled from several mechanism runs at different n (control_positions n=50, q4km_noise n=30, multiseed FLIP n=100/300); cells differ by ~1 unit depending on the pool/CI method, so no single committed artifact matches cell-by-cell. Closest sources: exp_mechanism_control_positions, exp_mechanism_q4km_noise_direction, exp_mechanism_multiseed, reviewer_polish/m3_flip_rate_cis. |
| `threefactor_logit_error.probdrop.awq_body` | 0.04 | Table assembled from several mechanism runs at different n (control_positions n=50, q4km_noise n=30, multiseed FLIP n=100/300); cells differ by ~1 unit depending on the pool/CI method, so no single committed artifact matches cell-by-cell. Closest sources: exp_mechanism_control_positions, exp_mechanism_q4km_noise_direction, exp_mechanism_multiseed, reviewer_polish/m3_flip_rate_cis. |
| `threefactor_logit_error.probdrop.q4_recall` | 46 | Table assembled from several mechanism runs at different n (control_positions n=50, q4km_noise n=30, multiseed FLIP n=100/300); cells differ by ~1 unit depending on the pool/CI method, so no single committed artifact matches cell-by-cell. Closest sources: exp_mechanism_control_positions, exp_mechanism_q4km_noise_direction, exp_mechanism_multiseed, reviewer_polish/m3_flip_rate_cis. |
| `threefactor_logit_error.flip.awq_recall` | 78 | Table assembled from several mechanism runs at different n (control_positions n=50, q4km_noise n=30, multiseed FLIP n=100/300); cells differ by ~1 unit depending on the pool/CI method, so no single committed artifact matches cell-by-cell. Closest sources: exp_mechanism_control_positions, exp_mechanism_q4km_noise_direction, exp_mechanism_multiseed, reviewer_polish/m3_flip_rate_cis. |
| `threefactor_logit_error.flip.awq_body` | 0 | Table assembled from several mechanism runs at different n (control_positions n=50, q4km_noise n=30, multiseed FLIP n=100/300); cells differ by ~1 unit depending on the pool/CI method, so no single committed artifact matches cell-by-cell. Closest sources: exp_mechanism_control_positions, exp_mechanism_q4km_noise_direction, exp_mechanism_multiseed, reviewer_polish/m3_flip_rate_cis. |
| `threefactor_logit_error.flip.q4_recall` | 48 | Table assembled from several mechanism runs at different n (control_positions n=50, q4km_noise n=30, multiseed FLIP n=100/300); cells differ by ~1 unit depending on the pool/CI method, so no single committed artifact matches cell-by-cell. Closest sources: exp_mechanism_control_positions, exp_mechanism_q4km_noise_direction, exp_mechanism_multiseed, reviewer_polish/m3_flip_rate_cis. |
| `perplexity_ratio.llama1b.bf16.indomain` | 1.0 | Paper reports quantized/BF16 perplexity RATIOS; wave_1_utility/ppl.json commits the raw per-version ppl but the ratio baseline is a per-backend f16 reference (GGUF vs HF) not stored as a field, so the ratio is a derivation, not a stored value. |
| `perplexity_ratio.llama1b.q8_0.indomain` | 1.001 | Paper reports quantized/BF16 perplexity RATIOS; wave_1_utility/ppl.json commits the raw per-version ppl but the ratio baseline is a per-backend f16 reference (GGUF vs HF) not stored as a field, so the ratio is a derivation, not a stored value. |
| `perplexity_ratio.llama1b.q5_k_m.indomain` | 1.022 | Paper reports quantized/BF16 perplexity RATIOS; wave_1_utility/ppl.json commits the raw per-version ppl but the ratio baseline is a per-backend f16 reference (GGUF vs HF) not stored as a field, so the ratio is a derivation, not a stored value. |
| `perplexity_ratio.llama1b.q5_k_m.ood` | 1.012 | Paper reports quantized/BF16 perplexity RATIOS; wave_1_utility/ppl.json commits the raw per-version ppl but the ratio baseline is a per-backend f16 reference (GGUF vs HF) not stored as a field, so the ratio is a derivation, not a stored value. |
| `perplexity_ratio.llama1b.q4_k_m.indomain` | 1.047 | Paper reports quantized/BF16 perplexity RATIOS; wave_1_utility/ppl.json commits the raw per-version ppl but the ratio baseline is a per-backend f16 reference (GGUF vs HF) not stored as a field, so the ratio is a derivation, not a stored value. |
| `perplexity_ratio.llama1b.q4_k_m.ood` | 1.044 | Paper reports quantized/BF16 perplexity RATIOS; wave_1_utility/ppl.json commits the raw per-version ppl but the ratio baseline is a per-backend f16 reference (GGUF vs HF) not stored as a field, so the ratio is a derivation, not a stored value. |
| `perplexity_ratio.llama1b.awq.indomain` | 1.123 | Paper reports quantized/BF16 perplexity RATIOS; wave_1_utility/ppl.json commits the raw per-version ppl but the ratio baseline is a per-backend f16 reference (GGUF vs HF) not stored as a field, so the ratio is a derivation, not a stored value. |
| `perplexity_ratio.llama1b.awq.ood` | 1.094 | Paper reports quantized/BF16 perplexity RATIOS; wave_1_utility/ppl.json commits the raw per-version ppl but the ratio baseline is a per-backend f16 reference (GGUF vs HF) not stored as a field, so the ratio is a derivation, not a stored value. |
| `perplexity_ratio.llama3b.awq.indomain` | 1.021 | Paper reports quantized/BF16 perplexity RATIOS; wave_1_utility/ppl.json commits the raw per-version ppl but the ratio baseline is a per-backend f16 reference (GGUF vs HF) not stored as a field, so the ratio is a derivation, not a stored value. |
| `perplexity_ratio.llama3b.awq.ood` | 1.021 | Paper reports quantized/BF16 perplexity RATIOS; wave_1_utility/ppl.json commits the raw per-version ppl but the ratio baseline is a per-backend f16 reference (GGUF vs HF) not stored as a field, so the ratio is a derivation, not a stored value. |
| `perplexity_ratio.qwen7b.awq.indomain` | 1.002 | Paper reports quantized/BF16 perplexity RATIOS; wave_1_utility/ppl.json commits the raw per-version ppl but the ratio baseline is a per-backend f16 reference (GGUF vs HF) not stored as a field, so the ratio is a derivation, not a stored value. |
| `perplexity_ratio.qwen7b.awq.ood` | 1.044 | Paper reports quantized/BF16 perplexity RATIOS; wave_1_utility/ppl.json commits the raw per-version ppl but the ratio baseline is a per-backend f16 reference (GGUF vs HF) not stored as a field, so the ratio is a derivation, not a stored value. |
<!-- AUTO:VERIFY:END -->
