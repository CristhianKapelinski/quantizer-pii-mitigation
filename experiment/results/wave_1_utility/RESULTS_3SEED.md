# Utility -- 3-seed mean perplexity ratios

Conventions: GGUF rows use llama-perplexity sliding-half-overlap with
F16-GGUF as baseline; AWQ/HF rows use HF non-overlap with BF16 baseline.
Per-seed F16-GGUF baselines (in / ood):
  * seed 42: 9.4407 / 14.7513
  * seed 52: 9.5904 / 14.7451
  * seed 62: 9.4859 / 14.9380

## 3-seed mean ratios (n=3)

| Version | In-domain ratio | OOD ratio |
|---|---|---|
| BF16 | 1.000 (σ=0.0000) | 1.000 (σ=0.0000) |
| Q8_0 | 1.001 (σ=0.0006) | 1.001 (σ=0.0008) |
| Q5_K_M | 1.022 (σ=0.0049) | 1.012 (σ=0.0046) |
| Q4_K_M | 1.047 (σ=0.0029) | 1.044 (σ=0.0043) |
| AWQ-4bit | 1.123 (σ=0.0090) | 1.094 (σ=0.0046) |
