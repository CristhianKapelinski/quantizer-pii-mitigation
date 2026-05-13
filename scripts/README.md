# scripts/ — index

Driver scripts and experiment dispatchers. Grouped by purpose. All
expect `PYTHONPATH=$REPO/src`, the `.venv`, and the pre-built llama.cpp
in `third_party/llama.cpp/build`. Most are idempotent (skip-if-output-
exists) so they're safe to re-run after a crash / power loss.

## Reproduction

| Script | Purpose |
|---|---|
| `build_llama_cpp.sh` | Fetch + CPU-only build of llama.cpp at the pinned commit `0827b2c` (tag b4404): llama-cli / llama-quantize / llama-perplexity / llama-imatrix. `--portable` for the multi-host case. Phase 0 of `reproduce_full.sh`. |
| `reproduce_full.sh` | End-to-end replay of the experiments with empirical wallclock estimates. Single-GPU; second host optional. |
| `sync_to_gpu2.sh` | rsync repo state to the parallelism host (`${GPU2_HOST}`). |

## Wave 0 / Wave 1 mini (the canary memorisation × quantisation core)

| Script | Purpose |
|---|---|
| `wave_0_smoke.sh` | W0 sanity: 5 canaries × 200 emails × 3 epochs, BF16 + Q4_K_M + AWQ, tokenizer + Métrica-1b/1c smoke. |
| `wave_1_mini_smoke.sh` | W1 mini Phase A: 100 canaries × {3,10,30,100} buckets, 3000 Enron, 5 epochs, BF16 + Q8/Q5/Q4 GGUF, greedy + n=5 stochastic. `RUN_TAG` / `SEED` / `MODEL_ID` / `BUCKETS` env-overridable. |
| `wave_1_mini_phase_b.sh` | W1 mini Phase B: adds AWQ-4bit to Phase A's checkpoint, re-extracts, recomputes gate. |
| `wave_1_full_dispatch.sh` | W1 full scaffold (3 Llama seeds + Qwen + DP-SGD + 7 quants + cross-cal). **Not run** — superseded by the paper roadmap; vanilla full-FT regime already characterised. |
| `step1_awq_canary_calibration.sh` | Step 1: AWQ + canary-inclusive calibration on Phase A ckpt (1/100 recovery — calibration-content hypothesis rejected). |
| `step2_qwen_mini_gpu2.sh` | Step 2 dispatcher for the Qwen cross-family run on gpu2. |
| `step_4_unlearn_quantize_attack.sh` | Step 4: GA_GDR unlearn on Phase A ckpt (canaries = forget, Enron = retain), then quantize + extract. v2 (threshold=5) → 0/100 recovery; "PII-canary unlearning is quant-robust". |
| `step_5_awq_canary100.sh` | Step 5: AWQ + 100 % canary calibration → 0/100. Refutes the saliency mechanism; calibration is a flat knob. |
| `step_6_awq_wikitext.sh` | Step 6: AWQ + 100 % WikiText OOD calibration → 0/100. 4th point on the calibration-ablation curve. |
| `step_7_awq_granularity_sweep.sh` | Step 7 (= Exp 1): AWQ group_size ∈ {32,64,128,256} at fixed Enron calibration. g32 ≈ 4.5 bpw is the bpw-matched control vs Q4_K_M. Rewritten with explicit invocations (no bash array). |
| `step_8_gguf_lowbit_extension.sh` | Step 8: extend Phase A to Q3_K_M + Q2_K. Boundary curve BF16 30 → Q8 30 → Q5 25 → Q4 6 → Q3 0 → Q2 0; boundary ≈ 4.5 bpw. |
| `step_8b_q4ks.sh` | Step 8b (= Exp 7): Q4_K_S point (≈4.3 bpw) on the boundary curve. |
| `step_9_zhang_nl_replication.sh` | Step 9 (= Exp 8): Zhang replication on 1B with a natural-language (Wikipedia) forget set instead of PII canaries. Isolates forget-content type as the variable; produces the middle row of the Zhang(7B,NL,83%)/Step9(1B,NL,?)/Step4v2(1B,PII,0%) comparison. |
| `utility_eval.sh` | Held-out PPL (Enron 500 + WikiText-2 1000, 50-window cap) across BF16 / 2 AWQ / Q8/Q5/Q4 GGUF + F16-GGUF baseline. Defence–utility frontier. |

## Consensus-attack smokes (testing whether the "Quantization Quilt" multi-version-amplification thesis revives — verdict so far: it does not)

| Script | Purpose |
|---|---|
| `smoke_1_consensus.py` | Soft-metrics consensus (ROUGE-L / Levenshtein / char-Jaccard, G3-calibrated). A_consensus = 1.000 on PII content; the 100/100 whole-suffix flag was template/boilerplate match. |
| `smoke_2_hayes_np.py` | Hayes (n,p)-discoverable extraction (closed-form `n_min = ⌈log(1-p)/log(1-p_z)⌉`). AWQ 0/100 at (n=1000, p=0.999). BF16 figure flagged as inflated by a tokenization-boundary issue — needs a boundary-safe re-run. |
| `smoke_3_icl_picompass.py` | ICL / PII-Compass k-shot priming, k ∈ {0,1,8,32}. Demonstrations *hurt* greedy extraction (30 → 1 → 0 → 0). |
| `smoke_4_mismatched_sft.sh` | CIA-style mismatched SFT (200 Enron-prefix + WikiText-response pairs) then re-quantise + re-extract. **Deprioritised** by the paper plan v3 (Quilt thesis already refuted). |
| `smoke_5_soft_prompt.py` | Ozdayi-style soft-prompt extraction per (version, canary) + cross-version transfer matrix. **Deprioritised.** |
| `smoke_6_activation_steering.py` | Activation-steering POC: steer Q4/AWQ toward BF16 activations to recover L2-fragile canaries. **Deprioritised.** |

## Paper-roadmap v3 experiments (the active line — defence-asymmetry headline)

See `experiment/plans/2026-05-11-paper-plan-v3.md` for the full roadmap.

| Script | Roadmap Exp # | Purpose |
|---|---|---|
| `step_7_awq_granularity_sweep.sh` | 1 | AWQ group_size sweep (above). |
| `exp_minkpp_reconciliation.py` | 2 / 9 | Min-K% standard (Shi 2024) + Min-K%++ (J. Zhang 2025) + loss-canary AUC. Reconciles Zhang ICLR 2025 Table 2 (AWQ≈GPTQ≈RTN on MUSE PrivLeak) with our Carlini-style asymmetry. |
| `exp_gptq_4bit.sh` | 3 | GPTQ-4bit g128 (inline auto_gptq, Enron calibration) + extract. **Critical path.** Isolates calibration-based vs RTN. |
| `exp_3seed_replication.sh` | 4 | Full Phase A + Phase B re-run for seeds 52, 62; pooled cross-seed stats. |
| (`utility_eval.sh` × 3 seeds) | 5 | PPL frontier across 3 seeds (no Enron classification — no labels). |
| `exp_saliency_2x2.sh` | 6 | 2×2 saliency grid: AWQ calibration distribution {pile / 50% canary / 100% canary / Enron} × matched 128×512 samples. |
| `step_8b_q4ks.sh` | 7 | Q4_K_S boundary point (above). |
| `step_9_zhang_nl_replication.sh` | 8 | Zhang NL replication (above). |
| `exp_semantic_similarity.py` | 10 | All-MPNet cosine between completions and true suffixes (≥0.8 strong semantic). Addresses Ippolito. AWQ cos_mean 0.43 < 0.5 → defence holds at the semantic level too. |
| `exp_acr.py` | 11 | Adversarial Compression Ratio (Schwarzschild NeurIPS 2024, GCG-style). Downscoped: 30 canaries × 3 versions × 200 GCG steps. **Last** — heaviest. |
| `exp_stats_aggregation.py` | (stats) | Pooled cross-seed table, Fisher-exact, BH-FDR q=0.05, 95% Clopper-Pearson CIs. |

## Module CLIs (in `src/qquilt/`, invoked by the above)

`qquilt.preflight`, `qquilt.canaries`, `qquilt.groups`, `qquilt.data`,
`qquilt.train`, `qquilt.unlearn`, `qquilt.quantize`, `qquilt.extract`,
`qquilt.metrics`, `qquilt.gate`, `qquilt.dp_sgd`, `qquilt.utility`,
`qquilt.aggregate`, `qquilt.report`, `qquilt.targets_w5`.
