# Technical Report: Four-Bit Quantization as a Deployment-Time Mitigation for Verbatim PII Leakage

## Abstract

Post-training quantization is commonly treated as an efficiency transformation, but it also changes the behavior of a fine-tuned language model. This report documents an empirical study of whether the choice of four-bit quantizer affects verbatim extraction of personally identifiable information (PII) memorized during fine-tuning. The study covers five open-weight instruction-tuned models from two families, spanning 0.5B to 7B parameters, under full fine-tuning and low-rank adaptation (LoRA). It compares calibration-corpus-free GGUF k-quants with the calibration-based Activation-aware Weight Quantization (AWQ) and GPTQ methods. The primary outcome is the fraction of deterministic synthetic PII canaries for which greedy continuation reproduces at least the complete 10-character reference field. Across the full-fine-tuning cells, AWQ reports extraction rates of 0.0% to 6.0%, compared with 4.0% to 24.0% for Q4_K_M and 26.6% to 30.3% for the BF16 baselines. In the three-seed calibration-method comparison, both AWQ and GPTQ reach 0.0%, while Q4_K_M reaches 5.3% and BF16 reaches 30.3%. Controlled analyses associate this difference with three interacting factors: where quantization error is concentrated, whether the original prediction is in a moderate-confidence regime, and whether the calibration objective amplifies errors at rare recall positions. The mitigation is not universal and is not a formal privacy guarantee. Its utility cost is model-scale dependent: the reported AWQ in-domain perplexity ratio is 1.123 at 1B parameters, 1.022 at 3B, and 1.002 at 7B. These results support a practical conclusion: among equal-bit deployment formats, quantizer selection can materially change verbatim leakage, and calibration-based four-bit methods can provide a low-cost mitigation at larger model scales.

## 1. Scope and Research Questions

Fine-tuned language models can reproduce unusual strings encountered during training. This behavior is operationally important when training data contain identifiers, account numbers, addresses, or other personal information. The study asks whether a deployment transformation already used to reduce memory and inference cost can also change this leakage behavior.

The central comparison is deliberately narrower than a general claim that quantization provides privacy. All evaluated methods use approximately four-bit weights, but they differ in how quantization parameters are selected. GGUF k-quants such as Q4_K_M do not use a calibration corpus. AWQ and GPTQ use representative calibration examples to choose transformations that preserve selected activation or reconstruction behavior. The study therefore addresses five questions:

1. Does verbatim PII extraction differ among equal-bit quantizers across model families, scales, and fine-tuning regimes?
2. Which properties of the quantization error explain the observed difference?
3. Does the conclusion persist when membership-inference attacks use appropriate in-distribution non-members?
4. What utility cost accompanies the reduction in extraction?
5. Does the effect extend from controlled synthetic canaries to naturally occurring PII in a public e-mail corpus?

The artifact is the source of truth for this report. Experimental configuration is declared in `EXPERIMENT_MANIFEST.yaml`; published values are transcribed in `expected/paper_values.json`; the lineage of every result directory is documented in `experiment/results/INDEX.md`; and exact-reproduction exceptions are disclosed in `docs/REPRODUCIBILITY_REPORT.md`.

## 2. Experimental Design

### 2.1 Models and fine-tuning regimes

The evaluation uses Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct, Llama-3.2-1B-Instruct, and Llama-3.2-3B-Instruct. The Llama-3.2-1B full-fine-tuning cell is the primary reference model and uses five seeds (42, 52, 62, 72, and 82). Qwen 0.5B and 1.5B full-fine-tuning cells use three seeds (42, 52, and 62). LoRA cells for Qwen 0.5B and Llama 1B/3B use three seeds. The 3B and 7B full-fine-tuning cells use one seed because of their compute cost. Cross-cell interpretations preserve this distinction and do not present the single-seed large-model cells as estimates with the same statistical support as the five-seed reference model.

The common fine-tuning recipe uses five epochs, a learning rate of \(2\times10^{-5}\), an effective batch size of 16, a maximum sequence length of 512, BF16 precision, and gradient checkpointing. The 3B LoRA delta-magnitude control uses \(2\times10^{-4}\). LoRA uses rank 16, alpha 32, and dropout 0.05; adapters are merged before quantization. The Llama-1B full-fine-tuning anchor uses AdamW, while Qwen and large full-fine-tuning cells use Adafactor to remain within the available memory budget.

### 2.2 Corpus and controlled PII canaries

The host corpus is the public AESLC subset of Enron e-mail (`snoop2head/enron_aeslc_emails`). The manifest records 3,000 host e-mails and a final training set of 6,575 records after insertion of controlled examples. Each seed deterministically generates 100 synthetic Enron-style e-mails. Every canary includes a high-entropy 10-character reference, a 12-digit account number, a date, and fixed surrounding text. These are synthetic values, not personal data.

Canaries are divided evenly among insertion-frequency buckets \(K\in\{3,10,30,100\}\), with 25 canaries per bucket. The resulting design measures memorization across four exposure levels instead of relying on a single duplication frequency. Generation and corpus shuffling derive from the run seed. Held-out Enron text supplies in-distribution non-members for membership inference and the mining pool for natural PII. WikiText-2 supplies out-of-distribution text and perplexity evaluation. Two additional control groups contain 50 held-out Wikipedia sequences and 50 deterministic synthetic out-of-distribution sequences; neither is inserted into training.

### 2.3 Quantizers

The calibration-corpus-free branch uses `llama.cpp` GGUF k-quants: Q8_0, Q5_K_M, Q4_K_M, Q4_K_S, Q3_K_M, and Q2_K. The central four-bit comparison uses Q4_K_M.

AWQ uses `autoawq` 0.2.7 with four-bit weights, a default group size of 128, and a calibration set of 128 Enron chunks of 512 tokens. A group-size sweep evaluates 32, 64, and 128. Group size 256 is excluded because the available AutoAWQ Triton GEMM kernels support only 32, 64, and 128 at inference time.

GPTQ uses `auto-gptq` 0.7.1, four-bit weights, group size 128, and the same size of Enron calibration set. Comparing AWQ and GPTQ with Q4_K_M distinguishes a result associated with calibration-based quantization from an implementation-specific AWQ result. A calibration-distribution ablation additionally uses WikiText, mixed, canary-only, and Enron calibration conditions.

### 2.4 Extraction attack and primary metric

The primary attack presents the deterministic prefix ending immediately before the high-entropy reference and greedily generates the continuation. Chat-model generation defaults are neutralized, including setting repetition penalty to 1.0, so the Hugging Face greedy path matches `llama-cli --temp 0`. The primary endpoint is a prefix-match length of at least 10 characters, which means that the complete 10-character reference field was reproduced. Rates are computed over the 100 canaries in each seed. The artifact also evaluates lower thresholds, exact matches, any-of-six stochastic extraction, and a stronger stress test up to any-of-100.

This endpoint measures verbatim reproduction under a known-prefix probe. It does not measure every form of semantic leakage, and a zero observed count means only that this attack found no qualifying extraction among the evaluated canaries. It is neither proof of non-membership nor a differential-privacy guarantee.

### 2.5 Membership inference, mechanism, and utility

Membership inference uses Min-K%, Min-K%++, and loss-based scores. The critical protocol comparison changes the negative population: synthetic out-of-distribution non-members can make all model versions appear nearly separable, whereas held-out Enron examples test membership against examples drawn from the same source distribution. A LiRA-style analysis additionally reports true-positive rate at 1% false-positive rate in the committed results.

The mechanism analysis separates five questions: survival of fine-tuning weight updates, per-layer reconstruction residuals, softmax fragility, the direction of AWQ-induced logit error, and the direction of Q4_K_M-induced logit error. Controls compare the rare reference-recall position with ordinary body-token and Enron-token positions. The reported table combines pools of 30, 50, 100, and 300 observations, as specified in the reproducibility report.

Utility is measured using in-domain Enron and out-of-distribution WikiText-2 perplexity ratios, normalized to the appropriate unquantized baseline. Zero-shot downstream utility uses ARC-Easy, HellaSwag, and WinoGrande accuracy.

## 3. Results

### 3.1 Headline extraction results

Table 1 reproduces the published extraction percentages. Multi-seed values are pooled where the artifact provides pooled statistics; the 3B and 7B full-fine-tuning cells are single-seed measurements.

| Model and regime | BF16 | Q8_0 | Q5_K_M | Q4_K_M | AWQ |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B, full FT | 30.3% | 30.3% | 28.3% | 23.0% | 0.0% |
| Llama-3.2-1B, full FT | 26.6% | 26.6% | 23.2% | 4.0% | 0.0% |
| Qwen2.5-1.5B, full FT | 30.3% | 30.3% | 29.3% | 13.7% | 5.0% |
| Llama-3.2-3B, full FT | 30.0% | not reported | 27.0% | 16.0% | 3.0% |
| Qwen2.5-7B, full FT | 30.0% | not reported | 30.0% | 24.0% | 6.0% |
| Qwen2.5-0.5B, LoRA | 23.3% | not reported | not reported | 0.0% | 0.0% |
| Llama-3.2-1B, LoRA | 25.7% | not reported | not reported | 0.0% | 0.0% |
| Llama-3.2-3B, LoRA, lr \(2\times10^{-5}\) | 28.0% | not reported | 9.0% | 0.0% | 0.0% |
| Llama-3.2-3B, LoRA, lr \(2\times10^{-4}\) | 30.0% | 30.0% | 30.0% | 25.0% | 7.0% |

The result is not simply that lower precision reduces memorization. Q8_0 closely tracks BF16, and Q5_K_M often retains most extraction. Q4_K_M reduces extraction more strongly, but its residual rates vary from 4.0% to 24.0% in the full-fine-tuning cells. AWQ is consistently lower, from 0.0% to 6.0%. The high-learning-rate LoRA control is informative because it restores substantial leakage in BF16 and GGUF while preserving a gap between Q4_K_M (25.0%) and AWQ (7.0%). Thus, the zero-valued default-LoRA rows should not be interpreted as evidence that LoRA is inherently safe; the magnitude of the learned update matters.

The group-size sweep also argues against describing AWQ as one fixed operating point. For one seed, the number of canaries meeting the 10-character endpoint is 4 at group size 32 and 0 at group sizes 64 and 128. The Q4_K_M and Q5_K_M references produce 6 and 25 hits, respectively. In the three-seed method comparison, BF16 is 30.3%, Q4_K_M is 5.3%, and both AWQ-g128 and GPTQ-g128 are 0.0%. Agreement between AWQ and GPTQ supports the calibration-based interpretation, although it does not establish that every calibration-based implementation will behave identically.

### 3.2 Calibration-distribution ablation

The four calibration conditions all produce zero 10-character greedy hits and zero any-of-six 10-character hits. At the looser five-character threshold, the counts are 0 for WikiText, 3 for mixed calibration, 1 for canary-only calibration, and 2 for Enron calibration. The main gap therefore does not require placing the exact canaries in the calibration set. It also cannot be summarized as complete invariance to calibration data, because the lower threshold reveals small differences among conditions.

### 3.3 Mechanism evidence

At a rare reference-recall position, the full-precision model's top-token probability is 0.71. AWQ produces a reported logit-error L2 norm of 841, cosine alignment of 0.0094, a 56% probability drop, and a 78% top-token flip rate over the table's stated pools. Q4_K_M at the same type of position has an L2 norm of 617, cosine 0.0064, a 31% probability drop, and a 48% flip rate. At ordinary body positions, where the full-precision top-token probability is 0.9999, AWQ's L2 norm remains large at 662, but the probability drop is only 0.04% and the flip rate is 0%. For ordinary Enron positions, the corresponding values are full-precision top probability 0.55, L2 norm 362, cosine 0.0017, probability drop 8.9%, and flip rate 30%.

These observations support a three-factor explanation. First, rare recall positions receive unusually consequential perturbations rather than uniform harmless noise. Second, the original prediction must lie in a confidence window where an error can change the selected token: very high-confidence body predictions remain stable despite a sizable vector norm. Third, calibration-based objectives alter the direction or concentration of the error, producing more flips at rare recall positions than Q4_K_M. None of the three factors alone is sufficient. Error magnitude without softmax fragility does not flip body tokens, and moderate confidence without the relevant error direction does not reproduce the AWQ/Q4_K_M difference.

The numerical mechanism table is a documented synthesis rather than a single homogeneous experiment. For example, the paper reports AWQ recall FLIP as 78%, while one committed 100-observation mechanism file yields 79/100. The report preserves the published 78% and does not silently replace it because the table mixes legitimate pools and its cells fall within the reported confidence intervals.

### 3.4 Membership-inference protocol

Using synthetic out-of-distribution non-members, BF16 obtains AUC 1.00 for Min-K%, Min-K%++, and loss. AWQ remains nearly perfect at 0.97, 1.00, and 0.99. This protocol therefore makes the two versions appear similarly vulnerable, but much of the separability can arise from distribution identity rather than training membership.

With held-out Enron non-members, BF16 AUC falls to 0.83 for Min-K%, 0.78 for Min-K%++, and 0.86 for loss. AWQ falls much further, to 0.22, 0.19, and 0.49. Values below 0.5 indicate a reversed score ordering under the stated convention, not superhuman privacy. The substantive result is that the prior near-perfect separation disappears when negatives come from the training-data distribution, and AWQ substantially changes the attack scores. The artifact also includes a LiRA-style TPR-at-1%-FPR analysis, but this report does not quote a value absent from `expected/paper_values.json`.

### 3.5 Utility and scale

At 1B parameters, the in-domain perplexity ratios are 1.001 for Q8_0, 1.022 for Q5_K_M, 1.047 for Q4_K_M, and 1.123 for AWQ. The out-of-distribution ratios are 1.012 for Q5_K_M, 1.044 for Q4_K_M, and 1.094 for AWQ. The privacy-relevant reduction at 1B therefore has a measurable language-modeling cost.

The AWQ cost decreases with scale. At 3B, its ratio is 1.022 in-domain and 1.021 out-of-distribution. At 7B, it is 1.002 in-domain and 1.044 out-of-distribution. The phrase "nearly free at production scale" is therefore best understood as negligible in-domain perplexity change at 7B, not as zero cost on every distribution.

On the three downstream benchmarks, BF16 scores 67.55% on ARC-Easy, 47.76% on HellaSwag, and 61.40% on WinoGrande, for a 58.90% mean. AWQ scores 67.76%, 45.59%, and 62.12%, for a 58.49% mean. The per-task differences are +0.21, -2.17, and +0.72 percentage points; the mean difference is -0.41 points. Aggregate utility is stable, but HellaSwag shows a non-negligible task-specific decline that should not be hidden by the mean.

### 3.6 Naturally occurring PII

The natural-canary analysis mines low-frequency PII instances from Enron and compares members with held-out non-members. For Llama-3.2-3B, member/non-member extraction is 5%/3% in BF16, 4%/3% in Q5_K_M, 4%/3% in Q4_K_M, and 4%/4% in AWQ. For Qwen2.5-7B, the values are 10%/5%, 9%/4%, 5%/4%, and 4%/3%, respectively.

The natural setting produces smaller gaps than the controlled canaries. AWQ removes the member/non-member difference in the 3B cell and reduces it to one percentage point in the 7B cell, but it does not eliminate all extraction. The result supports mitigation of excess member leakage, not suppression of every completion containing public PII.

## 4. Interpretation and Deployment Guidance

The largest practical insight is that "four bit" is not a complete privacy description. Two deployments with similar storage cost can expose materially different amounts of verbatim training data because the algorithms choose their quantization parameters differently. Bit width, model family, and model size do not explain the result on their own.

For deployments that already require four-bit weights, a calibration-based method should be treated as a privacy-relevant configuration choice. The evidence favors AWQ or GPTQ over Q4_K_M when the operational priority includes reducing known-prefix verbatim extraction. A representative, non-sensitive calibration corpus is sufficient for the main 10-character result; the exact protected examples need not be included. At 3B and 7B scale, the reported utility trade-off is small enough to make this a practical defense-in-depth measure.

This recommendation has three boundaries. First, quantization should complement data minimization, access controls, privacy-aware training, and output monitoring rather than replace them. Second, zero observed extraction under one probe must not be represented as proof that information has been removed. Third, the quantized artifact must be evaluated in its deployed inference stack, because group size, kernels, calibration corpus, decoding defaults, and model revisions can change the result.

## 5. Limitations and Threats to Validity

The synthetic canaries provide controlled ground truth but are structured e-mails with high-entropy fields. Other kinds of personal information, natural paraphrases, multilingual data, and semantic recall may behave differently. The natural-Enron experiment partly addresses realism, but Enron is a public and historically specific corpus.

The primary attack is a known-prefix verbatim probe. Stronger sampling, beam search, semantic extraction, black-box adaptive prompting, or an attacker with auxiliary information could produce different leakage. The artifact contains stronger-attacker and semantic-similarity analyses, but the headline metric remains deliberately narrow.

The 3B and 7B full-fine-tuning results use one seed. Their cross-scale pattern is useful evidence, not a precise estimate of run-to-run variance. The Llama-1B five-seed anchor and the three-seed smaller-model cells carry the statistical replication.

Utility is measured by perplexity and three zero-shot benchmarks. This does not cover instruction following, long context, domain-specific generation, calibration, toxicity, or every downstream application. The 7B out-of-distribution AWQ perplexity ratio of 1.044 also shows that the negligible in-domain ratio does not imply negligible cost everywhere.

Model and dataset identifiers are pinned, but upstream Hugging Face revision SHAs and dataset checksums are not. A future upstream change can therefore alter a from-scratch execution. Fine-tuning is seeded but not claimed to be bit-identical across GPU architectures because kernel selection and BF16 accumulation differ.

The current verifier checks all 141 catalogued entries. It pools the Qwen-0.5B AWQ and LoRA cells directly from their committed per-seed JSONL logs, so the latest verification has 141 exact passes and no skips. The three-factor table remains a documented synthesis across differing sample pools, but each printed cell has an explicit resolver and is checked at the paper's printed precision.

Quantization, extraction, and analysis wall-clock times were not instrumented. Only fine-tuning time can be reported from per-step telemetry. One mechanism driver requires the locked optional `mechanism` extra containing `llama-cpp-python`; its committed output can be replayed without that extra.

## 6. Reproduction Procedure

### 6.1 Pinned environment and hardware

The artifact targets Linux x86-64 and Python 3.11. Its lockfile resolves PyTorch 2.7.1 with CUDA 12.8, Transformers 4.46.3, PEFT 0.13.2, Accelerate 1.1.1, NumPy 1.26.4, SciPy 1.17.1, Statsmodels 0.14.6, Matplotlib 3.10.9, AutoAWQ 0.2.7, and AutoGPTQ 0.7.1. `llama.cpp` is pinned to commit `0827b2c1da299805288abbd556d869318f2b121e` (tag b4404) and built CPU-only.

The 0.5B, 1B, and 1.5B full-fine-tuning cells, all LoRA cells, and ablations run on a 16 GB RTX 5060 Ti-class GPU; selected small cells also run on an RTX 3060 12 GB. The 3B and 7B full-fine-tuning cells were produced on an A100 80 GB. A complete run requires approximately 32 GB RAM and 80 GB free disk. Replay requires no GPU, approximately 4 GB RAM, and approximately 2 GB disk.

### 6.2 Analysis replay

After cloning the repository, install the locked analysis environment:

```bash
uv sync --no-install-project --extra dev
bash replay.sh
```

Replay recomputes per-seed extraction metrics from committed JSONL logs, recomputes pooled Fisher exact tests, Clopper-Pearson intervals, and Benjamini-Hochberg corrections, compares every recomputed field with the committed metrics, regenerates all five figures, and compares published values with the run of record. It exits nonzero on a mismatch. The full replay was measured at approximately seven seconds on the reference workstation. A number-only check is available as `bash replay.sh verify`, and figure-only rendering as `bash replay.sh --figures-only`.

The latest recorded verification result is 141 exact passes, zero failures, and zero skips among 141 checked entries.

### 6.3 Reduced live re-run

To exercise training, quantization, extraction, and metrics on new outputs:

```bash
uv sync --no-install-project --extra quant
bash scripts/build_llama_cpp.sh
bash reproduce.sh quick
```

This command runs Qwen2.5-0.5B full fine-tuning with seed 42 and compares BF16, Q4_K_M, and AWQ. It writes to `experiment/results/wave_1_qwen05b_seed42_rerun/`, preserving the committed reference. The measured fine-tuning phase is approximately 85 minutes on the reference 16 GB GPU; quantization and extraction times were not instrumented. Because GPU training is not bit-deterministic across hosts, the acceptance criterion is reproduction of the ordering, with AWQ far below Q4_K_M and BF16, rather than identical per-canary counts.

### 6.4 Full experiment dispatch

`bash reproduce.sh --list` enumerates the independently runnable groups: `headline`, `ablations`, `saliency`, `mechanism`, `mia`, `utility`, `downstream`, `natural_canaries`, `support`, and `figures`. Running `bash reproduce.sh` dispatches all groups in order. Steps are idempotent and skip committed result directories; to force a live measurement, remove only the specific `experiment/results/<tag>/` directory to be regenerated.

The measured fine-tuning phase sums to approximately 37 hours across the headline cells when expressed as serial 16 GB-GPU time, with the 3B and 7B full-fine-tuning cells requiring an A100. No end-to-end duration is claimed because the remaining phases lack telemetry.

## 7. Artifact Assurance

The repository separates reusable stages into `src/qquilt/`: canary construction, dataset assembly, training, quantization, extraction, metrics, utility, controls, unlearning, differential-privacy training support, preflight checks, and seed management. Experiment drivers under `scripts/` compose these stages and map directly to paper tables and figures in `experiment/results/INDEX.md`.

The analysis path is numerically deterministic. `SOURCE_DATE_EPOCH` reduces figure-metadata drift, while canonical plotted data, rather than raw PDF bytes, is the correctness boundary across PDF backends. Unit tests run without a GPU or network and cover figure-data derivation, published-number verification, script/path integrity, executable entry points, and the greedy-extraction counting rule. The tests do not replace a live GPU smoke test of fine-tuning and quantization.

Result records use documented JSONL schemas with explicit schema identifiers and versions. Large regenerable model files are excluded, while per-seed canaries, extraction outputs, metrics, training telemetry, and analysis summaries constitute the committed run of record.

## 8. Conclusion

The evidence shows that quantizer selection can materially change verbatim PII extraction even when storage precision remains approximately four bits. Calibration-corpus-free Q4_K_M reduces leakage relative to BF16, but AWQ and GPTQ reduce it further across the evaluated families, scales, and training regimes. Controlled results associate the difference with error concentration at rare recall positions, prediction confidence, and calibration-induced error direction. At larger scales, this reduction is achieved with small reported utility loss.

The contribution is therefore not the claim that more compression automatically creates privacy. It is the identification and experimental validation of quantizer choice as a deployment-time privacy variable, together with a reproducible comparison, a mechanism-based explanation, and explicit boundaries on what the mitigation guarantees. Calibration-based four-bit quantization is a practical defense-in-depth option, not a substitute for privacy-preserving data and training practices.

## Artifact References

- `README.md`: reviewer-facing overview, claims, timing, and command guide.
- `EXPERIMENT_MANIFEST.yaml`: models, datasets, seeds, hyperparameters, quantizers, software, and hardware.
- `expected/paper_values.json`: published numeric values used by the verifier.
- `docs/REPRODUCIBILITY_REPORT.md`: exact-match report, synthesis lineage, and known gaps.
- `experiment/results/INDEX.md`: mapping from paper items to scripts and result directories.
- `experiment/results/SCHEMA.md`: committed JSONL schemas.
- `replay.sh`: deterministic analysis replay and assertion pipeline.
- `reproduce.sh`: idempotent per-experiment full-reproduction dispatcher.
