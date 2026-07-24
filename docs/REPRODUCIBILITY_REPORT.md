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

- **Figures load their numbers from the logs.** All five figure scripts now read the values
  they plot from the committed result JSONs. `fig_dose_response.py` reads
  `reviewer_polish/m10_threshold_sensitivity.json`; `fig_crossfamily.py`, `fig_mia_combined.py`,
  `fig_mechanism.py`, and `fig_quant_variants.py` obtain their values from `scripts/_fig_data.py`,
  a single loader that recomputes each number from the logs and self-checks (`python
  scripts/_fig_data.py`) that every loadable value equals the published one. A small number of
  cells have no single committed artifact because they are multi-run syntheses reported in a
  paper table, and stay as documented constants marked `_SYNTH` in `_fig_data.py`: the
  Qwen2.5-0.5B AWQ headline cell, the 3-seed LoRA BF16 pools, and two `tab:threefactor` cells
  (the AWQ RECALL logit-error norm and flip rate). GGUF/GPTQ effective bits-per-weight are
  format-defined constants, not measured quantities.

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

- **`tab:utility` perplexity ratios are verified, with one exception.** The 1B ratios are the
  3-seed means stored in `wave_1_utility/ppl_3seed_mean.json` (GGUF rows against the f16-GGUF
  baseline, HF rows against BF16-HF; the file records both conventions), and the 3B/7B AWQ
  ratios are recomputed from the per-cell `utility/ppl.json`. All twelve published ratios match
  exactly. The 3B in-domain AWQ ratio (8.6184/8.4361 = 1.0216, printed as **1.022**) was the one
  exception until the paper cell, which read 1.021, was corrected to the rounded value.

- **`tab:defense-pareto` has no dedicated result file.** Its extraction column is derived from
  `tab:headline` (BF16 26.6%, Q4\_K\_M 4.0%, AWQ 0.0/3.0/6.0%) and its Δppl column from
  `tab:utility`. The verifier checks the extraction column against the headline sources; the
  Δppl column follows the utility ratios above.

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

- **The replay path asserts, it does not only print.** After recomputing the per-seed metrics and
  the pooled statistics from the committed extraction logs, `replay.sh` runs
  `scripts/check_replay_equal.py`, which requires every recomputed field to be identical to the
  committed one, and then this verifier. A mismatch in either makes `replay.sh` exit non-zero.
  The committed `metrics_w1_mini.json` files carry one extra block (`gate_w1_mini`, written only
  when `qquilt.metrics` is called with `--include-w1-mini-gate`); extra committed keys are
  reported and allowed, recomputed keys are not.

## 2. Automatic verification results

<!-- AUTO:VERIFY:BEGIN -->
_Last verification: **133 pass / 0 fail**, 5 skip, out of 138 checked paper numbers._

### PASS

133 numbers reproduce EXACTLY at the paper's printed precision (headline extraction pools, AWQ group-size sweep, GPTQ vs AWQ vs Q4\_K\_M, saliency 2x2, Min-K%/Loss MIA AUCs, downstream accuracy, natural-canary gaps, defense-pareto extraction column).

### SKIP (documented, not verified for exact equality)

| key | paper | reason |
|---|---|---|
| `headline_greedy_ge10_extraction_pct.fullft.qwen0_5b.awq` | 0.0 | source artifact absent |
| `headline_greedy_ge10_extraction_pct.lora.qwen0_5b.bf16` | 23.3 | no recomputable artifact / derived or prior-work value |
| `headline_greedy_ge10_extraction_pct.lora.llama1b.bf16` | 25.7 | no recomputable artifact / derived or prior-work value |
| `headline_greedy_ge10_extraction_pct.lora.llama3b_lr2e5.bf16` | 28.0 | no recomputable artifact / derived or prior-work value |
| `headline_greedy_ge10_extraction_pct.lora.llama3b_lr2e5.q5_k_m` | 9.0 | no recomputable artifact / derived or prior-work value |
<!-- AUTO:VERIFY:END -->
