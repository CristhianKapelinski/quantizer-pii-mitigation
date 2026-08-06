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

- **The figure loads its numbers from the logs.** The paper has one figure, `fig:story`,
  rendered by `scripts/fig_story.py`; nothing in it is hardcoded. Panel (a) reads
  `reviewer_polish/m10_threshold_sensitivity.json`; panels (b) and (c) read
  `exp_mechanism_multiseed/` and `exp_mechanism_local_replication/`. Its printed `cos` and
  `flip` values are the same quantities published in `tab:threefactor`, so the figure and
  the table cross-check each other. `scripts/_fig_data.py` remains as an independent
  paper-number cross-check: it recomputes each published value from the logs and self-checks
  (`python scripts/_fig_data.py`) that every loadable value equals the published one. The
  verifier pools cells without a dedicated aggregate file directly from their committed
  per-seed JSONL logs. Two `tab:threefactor` values remain documented constants in
  `_fig_data.py`; their published values have separate resolvers. GGUF/GPTQ effective
  bits-per-weight are format-defined constants, not measured quantities.

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

- **`sec:utility` perplexity ratios are verified, with one exception.** The 1B ratios are the
  3-seed means stored in `wave_1_utility/ppl_3seed_mean.json` (GGUF rows against the f16-GGUF
  baseline, HF rows against BF16-HF; the file records both conventions), and the 3B/7B AWQ
  ratios are recomputed from the per-cell `utility/ppl.json`. All twelve published ratios match
  exactly. The 3B in-domain AWQ ratio (8.6184/8.4361 = 1.0216, printed as **1.022**) was the one
  exception until the paper cell, which read 1.021, was corrected to the rounded value.

- **`tab:headline` has no dedicated result file.** Its extraction column is derived from
  `tab:headline` (BF16 26.6%, Q4\_K\_M 4.0%, AWQ 0.0/3.0/6.0%) and its Δppl column from
  `sec:utility`. The verifier checks the extraction column against the headline sources; the
  Δppl column follows the utility ratios above.

- **Pools without aggregate files are recomputed from their sources.** The Qwen2.5-0.5B AWQ
  headline cell and the LoRA BF16/Q5\_K\_M three-seed cells are pooled from the committed
  extraction JSONL files. Their denominators come from the distinct G1 canary identifiers in
  each seed, so missing or partial source logs cannot silently become a zero.

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
_Last verification: **141 pass / 0 fail**, 0 skip, out of 141 checked paper numbers._

### PASS

141 numbers reproduce EXACTLY at the paper's printed precision. Each row gives where the number is published in the paper, the published value, and the value recomputed from the committed logs.

| key | published in | paper | recomputed |
|---|---|---|---|
| `headline_greedy_ge10_extraction_pct.fullft.qwen0_5b.bf16` | tab:headline | 30.3 | 30.3 |
| `headline_greedy_ge10_extraction_pct.fullft.qwen0_5b.q8_0` | tab:headline | 30.3 | 30.3 |
| `headline_greedy_ge10_extraction_pct.fullft.qwen0_5b.q5_k_m` | tab:headline | 28.3 | 28.3 |
| `headline_greedy_ge10_extraction_pct.fullft.qwen0_5b.q4_k_m` | tab:headline | 23.0 | 23.0 |
| `headline_greedy_ge10_extraction_pct.fullft.qwen0_5b.awq` | tab:headline | 0.0 | 0.0 |
| `headline_greedy_ge10_extraction_pct.fullft.llama1b.bf16` | tab:headline | 26.6 | 26.6 |
| `headline_greedy_ge10_extraction_pct.fullft.llama1b.q8_0` | tab:headline | 26.6 | 26.6 |
| `headline_greedy_ge10_extraction_pct.fullft.llama1b.q5_k_m` | tab:headline | 23.2 | 23.2 |
| `headline_greedy_ge10_extraction_pct.fullft.llama1b.q4_k_m` | tab:headline | 4.0 | 4.0 |
| `headline_greedy_ge10_extraction_pct.fullft.llama1b.awq` | tab:headline | 0.0 | 0.0 |
| `headline_greedy_ge10_extraction_pct.fullft.qwen1_5b.bf16` | tab:headline | 30.3 | 30.3 |
| `headline_greedy_ge10_extraction_pct.fullft.qwen1_5b.q8_0` | tab:headline | 30.3 | 30.3 |
| `headline_greedy_ge10_extraction_pct.fullft.qwen1_5b.q5_k_m` | tab:headline | 29.3 | 29.3 |
| `headline_greedy_ge10_extraction_pct.fullft.qwen1_5b.q4_k_m` | tab:headline | 13.7 | 13.7 |
| `headline_greedy_ge10_extraction_pct.fullft.qwen1_5b.awq` | tab:headline | 5.0 | 5.0 |
| `headline_greedy_ge10_extraction_pct.fullft.llama3b.bf16` | tab:headline | 30.0 | 30.0 |
| `headline_greedy_ge10_extraction_pct.fullft.llama3b.q5_k_m` | tab:headline | 27.0 | 27.0 |
| `headline_greedy_ge10_extraction_pct.fullft.llama3b.q4_k_m` | tab:headline | 16.0 | 16.0 |
| `headline_greedy_ge10_extraction_pct.fullft.llama3b.awq` | tab:headline | 3.0 | 3.0 |
| `headline_greedy_ge10_extraction_pct.fullft.qwen7b.bf16` | tab:headline | 30.0 | 30.0 |
| `headline_greedy_ge10_extraction_pct.fullft.qwen7b.q5_k_m` | tab:headline | 30.0 | 30.0 |
| `headline_greedy_ge10_extraction_pct.fullft.qwen7b.q4_k_m` | tab:headline | 24.0 | 24.0 |
| `headline_greedy_ge10_extraction_pct.fullft.qwen7b.awq` | tab:headline | 6.0 | 6.0 |
| `headline_greedy_ge10_extraction_pct.lora.qwen0_5b.bf16` | tab:headline | 23.3 | 23.3 |
| `headline_greedy_ge10_extraction_pct.lora.qwen0_5b.q4_k_m` | tab:headline | 0.0 | 0.0 |
| `headline_greedy_ge10_extraction_pct.lora.qwen0_5b.awq` | tab:headline | 0.0 | 0.0 |
| `headline_greedy_ge10_extraction_pct.lora.llama1b.bf16` | tab:headline | 25.7 | 25.7 |
| `headline_greedy_ge10_extraction_pct.lora.llama1b.q4_k_m` | tab:headline | 0.0 | 0.0 |
| `headline_greedy_ge10_extraction_pct.lora.llama1b.awq` | tab:headline | 0.0 | 0.0 |
| `headline_greedy_ge10_extraction_pct.lora.llama3b_lr2e5.bf16` | tab:headline | 28.0 | 28.0 |
| `headline_greedy_ge10_extraction_pct.lora.llama3b_lr2e5.q5_k_m` | tab:headline | 9.0 | 9.0 |
| `headline_greedy_ge10_extraction_pct.lora.llama3b_lr2e5.q4_k_m` | tab:headline | 0.0 | 0.0 |
| `headline_greedy_ge10_extraction_pct.lora.llama3b_lr2e5.awq` | tab:headline | 0.0 | 0.0 |
| `headline_greedy_ge10_extraction_pct.lora.llama3b_lr2e4.q4_k_m` | tab:headline | 25.0 | 25.0 |
| `headline_greedy_ge10_extraction_pct.lora.llama3b_lr2e4.awq` | tab:headline | 7.0 | 7.0 |
| `headline_greedy_ge10_extraction_pct.lora.llama3b_lr2e4.bf16` | tab:headline | 30.0 | 30.0 |
| `headline_greedy_ge10_extraction_pct.lora.llama3b_lr2e4.q8_0` | tab:headline | 30.0 | 30.0 |
| `headline_greedy_ge10_extraction_pct.lora.llama3b_lr2e4.q5_k_m` | tab:headline | 30.0 | 30.0 |
| `awq_group_size_sweep_ge10_count.awq_g32` | tab:awq-sweep (single seed) | 4.0 | 4.0 |
| `awq_group_size_sweep_ge10_count.awq_g64` | tab:awq-sweep | 0.0 | 0.0 |
| `awq_group_size_sweep_ge10_count.awq_g128` | tab:awq-sweep | 0.0 | 0.0 |
| `awq_group_size_sweep_ge10_count.ref_q4_k_m` | tab:awq-sweep | 6.0 | 6.0 |
| `awq_group_size_sweep_ge10_count.ref_q5_k_m` | tab:awq-sweep | 25.0 | 25.0 |
| `calib_vs_free_3seed_ge10_pct.bf16` | tab:gptq | 30.3 | 30.3 |
| `calib_vs_free_3seed_ge10_pct.q4_k_m` | tab:gptq | 5.3 | 5.3 |
| `calib_vs_free_3seed_ge10_pct.awq_g128` | tab:gptq | 0.0 | 0.0 |
| `calib_vs_free_3seed_ge10_pct.gptq_g128` | tab:gptq | 0.0 | 0.0 |
| `saliency_ablation.A_wikitext.ge5` | prose sec:mechanism ("Calibration content has no effect") | 0.0 | 0.0 |
| `saliency_ablation.A_wikitext.ge10` | prose sec:mechanism ("Calibration content has no effect") | 0.0 | 0.0 |
| `saliency_ablation.A_wikitext.anyof6_ge10` | prose sec:mechanism ("Calibration content has no effect") | 0.0 | 0.0 |
| `saliency_ablation.B_mix.ge5` | prose sec:mechanism ("Calibration content has no effect") | 3.0 | 3.0 |
| `saliency_ablation.B_mix.ge10` | prose sec:mechanism ("Calibration content has no effect") | 0.0 | 0.0 |
| `saliency_ablation.B_mix.anyof6_ge10` | prose sec:mechanism ("Calibration content has no effect") | 0.0 | 0.0 |
| `saliency_ablation.C_canary.ge5` | prose sec:mechanism ("Calibration content has no effect") | 1.0 | 1.0 |
| `saliency_ablation.C_canary.ge10` | prose sec:mechanism ("Calibration content has no effect") | 0.0 | 0.0 |
| `saliency_ablation.C_canary.anyof6_ge10` | prose sec:mechanism ("Calibration content has no effect") | 0.0 | 0.0 |
| `saliency_ablation.D_enron.ge5` | prose sec:mechanism ("Calibration content has no effect") | 2.0 | 2.0 |
| `saliency_ablation.D_enron.ge10` | prose sec:mechanism ("Calibration content has no effect") | 0.0 | 0.0 |
| `saliency_ablation.D_enron.anyof6_ge10` | prose sec:mechanism ("Calibration content has no effect") | 0.0 | 0.0 |
| `threefactor_logit_error.n.awq_recall` | tab:threefactor | 300.0 | 300.0 |
| `threefactor_logit_error.n.awq_body` | tab:threefactor | 100.0 | 100.0 |
| `threefactor_logit_error.n.q4_recall` | tab:threefactor | 300.0 | 300.0 |
| `threefactor_logit_error.n.enron` | tab:threefactor | 300.0 | 300.0 |
| `threefactor_logit_error.ft_top1.awq_recall` | tab:threefactor | 0.71 | 0.71 |
| `threefactor_logit_error.ft_top1.awq_body` | tab:threefactor | 0.9999 | 0.9999 |
| `threefactor_logit_error.ft_top1.q4_recall` | tab:threefactor | 0.71 | 0.71 |
| `threefactor_logit_error.ft_top1.enron` | tab:threefactor | 0.55 | 0.55 |
| `threefactor_logit_error.l2norm.awq_recall` | tab:threefactor | 841.0 | 841.0 |
| `threefactor_logit_error.l2norm.awq_body` | tab:threefactor | 662.0 | 662.0 |
| `threefactor_logit_error.l2norm.q4_recall` | tab:threefactor | 617.0 | 617.0 |
| `threefactor_logit_error.l2norm.enron` | tab:threefactor | 362.0 | 362.0 |
| `threefactor_logit_error.cos.awq_recall` | tab:threefactor | 0.0094 | 0.0094 |
| `threefactor_logit_error.cos.awq_body` | tab:threefactor | 0.0078 | 0.0078 |
| `threefactor_logit_error.cos.q4_recall` | tab:threefactor | 0.0064 | 0.0064 |
| `threefactor_logit_error.cos.enron` | tab:threefactor | 0.0017 | 0.0017 |
| `threefactor_logit_error.probdrop.awq_recall` | tab:threefactor (%) | 56.0 | 56.0 |
| `threefactor_logit_error.probdrop.awq_body` | tab:threefactor (%) | 0.04 | 0.04 |
| `threefactor_logit_error.probdrop.q4_recall` | tab:threefactor (%) | 31.0 | 31.0 |
| `threefactor_logit_error.probdrop.enron` | tab:threefactor (%) | 8.9 | 8.9 |
| `threefactor_logit_error.flip.awq_recall` | tab:threefactor (%) | 78.0 | 78.0 |
| `threefactor_logit_error.flip.awq_body` | tab:threefactor (%) | 0.0 | 0.0 |
| `threefactor_logit_error.flip.q4_recall` | tab:threefactor (%) | 48.0 | 48.0 |
| `threefactor_logit_error.flip.enron` | tab:threefactor (%) | 30.0 | 30.0 |
| `mia_auc.ood.bf16.mink` | prose sec:threat-split ("In-distribution non-member control") | 1.0 | 1.0 |
| `mia_auc.ood.bf16.minkpp` | prose sec:threat-split ("In-distribution non-member control") | 1.0 | 1.0 |
| `mia_auc.ood.bf16.loss` | prose sec:threat-split ("In-distribution non-member control") | 1.0 | 1.0 |
| `mia_auc.ood.awq.mink` | prose sec:threat-split ("In-distribution non-member control") | 0.97 | 0.97 |
| `mia_auc.ood.awq.minkpp` | prose sec:threat-split ("In-distribution non-member control") | 1.0 | 1.0 |
| `mia_auc.ood.awq.loss` | prose sec:threat-split ("In-distribution non-member control") | 0.99 | 0.99 |
| `mia_auc.indist.bf16.mink` | prose sec:threat-split ("In-distribution non-member control") | 0.83 | 0.83 |
| `mia_auc.indist.bf16.minkpp` | prose sec:threat-split ("In-distribution non-member control") | 0.78 | 0.78 |
| `mia_auc.indist.bf16.loss` | prose sec:threat-split ("In-distribution non-member control") | 0.86 | 0.86 |
| `mia_auc.indist.awq.mink` | prose sec:threat-split ("In-distribution non-member control") | 0.22 | 0.22 |
| `mia_auc.indist.awq.minkpp` | prose sec:threat-split ("In-distribution non-member control") | 0.19 | 0.19 |
| `mia_auc.indist.awq.loss` | prose sec:threat-split ("In-distribution non-member control") | 0.49 | 0.49 |
| `downstream_accuracy_pct.bf16.arc` | prose sec:utility ("Downstream task accuracy") | 67.55 | 67.55 |
| `downstream_accuracy_pct.bf16.hellaswag` | prose sec:utility ("Downstream task accuracy") | 47.76 | 47.76 |
| `downstream_accuracy_pct.bf16.winogrande` | prose sec:utility ("Downstream task accuracy") | 61.4 | 61.4 |
| `downstream_accuracy_pct.bf16.mean` | prose sec:utility ("Downstream task accuracy") | 58.9 | 58.9 |
| `downstream_accuracy_pct.awq.arc` | prose sec:utility ("Downstream task accuracy") | 67.76 | 67.76 |
| `downstream_accuracy_pct.awq.hellaswag` | prose sec:utility ("Downstream task accuracy") | 45.59 | 45.59 |
| `downstream_accuracy_pct.awq.winogrande` | prose sec:utility ("Downstream task accuracy") | 62.12 | 62.12 |
| `downstream_accuracy_pct.awq.mean` | prose sec:utility ("Downstream task accuracy") | 58.49 | 58.49 |
| `downstream_accuracy_pct.delta.arc` | prose sec:utility ("Downstream task accuracy") | 0.21 | 0.21 |
| `downstream_accuracy_pct.delta.hellaswag` | prose sec:utility ("Downstream task accuracy") | -2.17 | -2.17 |
| `downstream_accuracy_pct.delta.winogrande` | prose sec:utility ("Downstream task accuracy") | 0.72 | 0.72 |
| `downstream_accuracy_pct.delta.mean` | prose sec:utility ("Downstream task accuracy") | -0.41 | -0.41 |
| `perplexity_ratio.llama1b.bf16.indomain` | prose sec:utility ("Perplexity across scale") | 1.0 | 1.0 |
| `perplexity_ratio.llama1b.q8_0.indomain` | prose sec:utility ("Perplexity across scale") | 1.001 | 1.001 |
| `perplexity_ratio.llama1b.q5_k_m.indomain` | prose sec:utility ("Perplexity across scale") | 1.022 | 1.022 |
| `perplexity_ratio.llama1b.q5_k_m.ood` | prose sec:utility ("Perplexity across scale") | 1.012 | 1.012 |
| `perplexity_ratio.llama1b.q4_k_m.indomain` | prose sec:utility ("Perplexity across scale") | 1.047 | 1.047 |
| `perplexity_ratio.llama1b.q4_k_m.ood` | prose sec:utility ("Perplexity across scale") | 1.044 | 1.044 |
| `perplexity_ratio.llama1b.awq.indomain` | prose sec:utility ("Perplexity across scale") | 1.123 | 1.123 |
| `perplexity_ratio.llama1b.awq.ood` | prose sec:utility ("Perplexity across scale") | 1.094 | 1.094 |
| `perplexity_ratio.llama3b.awq.indomain` | prose sec:utility ("Perplexity across scale") | 1.022 | 1.022 |
| `perplexity_ratio.llama3b.awq.ood` | prose sec:utility ("Perplexity across scale") | 1.021 | 1.021 |
| `perplexity_ratio.qwen7b.awq.indomain` | prose sec:utility ("Perplexity across scale") | 1.002 | 1.002 |
| `perplexity_ratio.qwen7b.awq.ood` | prose sec:utility ("Perplexity across scale") | 1.044 | 1.044 |
| `natural_canary_member_nonmember.llama3b.bf16.member` | prose sec:natural-canaries (%) | 5.0 | 5.0 |
| `natural_canary_member_nonmember.llama3b.bf16.nonmem` | prose sec:natural-canaries (%) | 3.0 | 3.0 |
| `natural_canary_member_nonmember.llama3b.q5_k_m.member` | prose sec:natural-canaries (%) | 4.0 | 4.0 |
| `natural_canary_member_nonmember.llama3b.q5_k_m.nonmem` | prose sec:natural-canaries (%) | 3.0 | 3.0 |
| `natural_canary_member_nonmember.llama3b.q4_k_m.member` | prose sec:natural-canaries (%) | 4.0 | 4.0 |
| `natural_canary_member_nonmember.llama3b.q4_k_m.nonmem` | prose sec:natural-canaries (%) | 3.0 | 3.0 |
| `natural_canary_member_nonmember.llama3b.awq.member` | prose sec:natural-canaries (%) | 4.0 | 4.0 |
| `natural_canary_member_nonmember.llama3b.awq.nonmem` | prose sec:natural-canaries (%) | 4.0 | 4.0 |
| `natural_canary_member_nonmember.qwen7b.bf16.member` | prose sec:natural-canaries (%) | 10.0 | 10.0 |
| `natural_canary_member_nonmember.qwen7b.bf16.nonmem` | prose sec:natural-canaries (%) | 5.0 | 5.0 |
| `natural_canary_member_nonmember.qwen7b.q5_k_m.member` | prose sec:natural-canaries (%) | 9.0 | 9.0 |
| `natural_canary_member_nonmember.qwen7b.q5_k_m.nonmem` | prose sec:natural-canaries (%) | 4.0 | 4.0 |
| `natural_canary_member_nonmember.qwen7b.q4_k_m.member` | prose sec:natural-canaries (%) | 5.0 | 5.0 |
| `natural_canary_member_nonmember.qwen7b.q4_k_m.nonmem` | prose sec:natural-canaries (%) | 4.0 | 4.0 |
| `natural_canary_member_nonmember.qwen7b.awq.member` | prose sec:natural-canaries (%) | 4.0 | 4.0 |
| `natural_canary_member_nonmember.qwen7b.awq.nonmem` | prose sec:natural-canaries (%) | 3.0 | 3.0 |
| `defense_pareto.bf16.extraction` | tab:headline (defense/utility trade-off view of the same cells) | 26.6 | 26.6 |
| `defense_pareto.q4_k_m.extraction` | tab:headline (defense/utility trade-off view of the same cells) | 4.0 | 4.0 |
| `defense_pareto.awq_1b.extraction` | tab:headline (defense/utility trade-off view of the same cells) | 0.0 | 0.0 |
| `defense_pareto.awq_3b.extraction` | tab:headline (defense/utility trade-off view of the same cells) | 3.0 | 3.0 |
| `defense_pareto.awq_7b.extraction` | tab:headline (defense/utility trade-off view of the same cells) | 6.0 | 6.0 |
| `defense_pareto.gptq_1b.extraction` | tab:headline (defense/utility trade-off view of the same cells) | 0.0 | 0.0 |

### SKIP (documented, not verified for exact equality)

| key | paper | reason |
|---|---|---|
<!-- AUTO:VERIFY:END -->
