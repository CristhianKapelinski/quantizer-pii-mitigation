# Per-weight bucket-collapse on canary-encoding subset (v2)

Direct empirical test of the Appendix-A mechanism, on the Llama-3.2-1B headline
checkpoint (`wave_1_mini/final`, seed 42). For each quantizer (AWQ-4bit,
GPTQ-4bit), each linear layer's effective dequantized weight is reconstructed
via an identity-input forward pass `W_eff[:,j] = layer(e_j)`, then compared
against the original ft weight matrix.

Per-weight statistic:

    survival_i = (theta_q_i - theta_base_i) / (theta_ft_i - theta_base_i)

  * survival ~ 0  -> bucket collapse (the FT update was rounded back into the
                    pre-FT bucket; the canary delta was erased).
  * survival ~ 1  -> FT delta survives (the quantizer preserves the update).

Canary-encoding subset = top-1% / 5% / 10% of weights by |delta|.

## Summary (overall + top-1%)

| Quantizer | Subset | n | collapse-rate | mean | median | p10 | p90 |
|-----------|--------|---:|---:|---:|---:|---:|---:|
| **AWQ-4bit** | overall  | 253 M | **1.3%** | 1.17 | 1.01 | -1.47 | +2.47 |
| **AWQ-4bit** | top-1%   | 2.5 M | **2.8%** | 0.84 | 1.01 | **+0.11** | +1.08 |
| GPTQ-4bit   | overall  | 54 M  | 0.0%     | 1.00 | 1.02 | +0.99 | +1.05 |
| GPTQ-4bit   | top-1%   | 0.5 M | 0.0%     | 1.00 | 1.02 | +0.99 | +1.05 |

(GPTQ has fewer parameters in the comparison because auto-gptq exports only the
attention/MLP linear layers; the embedding + lm_head + norms are unquantized and
appear in both base and ft byte-identical, so their survival is trivially 1.0
and would dilute the signal.)

## Reading

* **AWQ-4bit, top-1% (canary-encoding):** the survival distribution is heavy-tailed
  with **p10 = 0.11** -- 10% of the canary-encoding weights collapse most of their
  FT delta back toward the pre-FT bucket. Mean survival 0.84 indicates a
  substantial pull toward base in this subset. **Partial empirical support for
  the bucket-collapse mechanism on AWQ.**
* **GPTQ-4bit:** survival is uniformly ~1.02 (FT delta essentially preserved
  byte-perfect at the weight level). No empirical bucket collapse signal.
* Both quantizers extract 0/100 canaries (paper Table 4.6). For AWQ the
  mechanism aligns with the Appendix-A bucket-collapse story; for GPTQ the
  measurement DOES NOT confirm bucket collapse, so the asymmetry must arise
  elsewhere (inverse-Hessian update correction acting at *inference*, not at
  storage).

## Note on v1 (broken)

A prior version of this script read `named_parameters()` on the autoawq model,
which returns ONLY the unquantized layers (embed, lm_head, layer-norms; the
quantized linears expose `qweight`/`scales`/`qzeros` buffers instead of a
parameter `weight`). v1's survival ~1.0 was trivial and misleading. v2 uses an
identity-input forward to read the layer's effective weight matrix, which
captures the dequantization+scale path. The v1 directory has been renamed
`exp_bucket_collapse_canary_v1_BROKEN/`.

## Reproduce

```
python scripts/exp_bucket_collapse_canary_v2.py \
  --base unsloth/Llama-3.2-1B-Instruct \
  --ft   checkpoints/wave_1_mini/final \
  --awq  checkpoints/wave_1_mini/quantized/model-awq-4bit \
  --gptq experiment/results/exp_gptq_4bit/quantized/gptq_4bit \
  --out  experiment/results/exp_bucket_collapse_canary_v2/metrics.json \
  --device cuda
```

~5 min on RTX 5060 Ti; ~12 GB VRAM peak when loading a quantized model for the
forward sweep.
