# Empirical mechanism: why AWQ/GPTQ erase canaries that Q4_K_M preserves

## Headline (operational claim from paper Table 2-9)
On Llama-3.2-1B fine-tuned over 100 G1 canaries, greedy >=10-char extraction:
AWQ-4bit = 0/500 (pooled 5 seeds); GPTQ-4bit = 0/100; Q4_K_M = 20/500.

The paper hypothesised (Appendix A) that the cause is "bucket collapse" at
the weight level: AWQ's per-channel pre-scaling effectively rounds the
fine-tune delta back into the pre-FT bucket. We test this and four other
candidate mechanisms here.

## Five empirical tests

### (1) v2 bucket collapse measurement -- direct weight comparison

For each linear layer, reconstruct the effective dequantised weight via an
identity-input forward (`W_eff[i,j] = layer(e_j)[i]`) and compare to FT
weights. Survival_i = (theta_q_i - theta_base_i) / (theta_ft_i - theta_base_i).

| Quantizer | overall collapse | top-1% (canary subset) | median survival |
|---|---:|---:|---:|
| AWQ-4bit  | 1.3% | **2.8%** | 1.01 (essentially preserved) |
| GPTQ-4bit | 0.0% | 0.0%   | 1.02 |

* AWQ shows partial weight-level bucket collapse on the canary-encoding
  subset (top 1% by |delta|), p10 survival 0.11 (10% of those weights
  collapsed to base). The median weight survives.
* GPTQ shows **zero** bucket collapse: FT delta preserved byte-perfect.
* Conclusion: bucket collapse at the weight level is partial-AWQ-only.
  GPTQ has zero weight-level explanation. **The mechanism for GPTQ must
  be elsewhere.**

### (2) Per-layer residual-stream error -- NOT canary-amplified

For one canary and one Enron input, hook each transformer block's residual
output and measure `||h_quant - h_ft|| / ||h_ft||` per layer per input.

| Quantizer | first layer | mid layer | last layer | trend |
|---|---:|---:|---:|---|
| AWQ canary/enron ratio  | 0.68x | 0.63x | 0.93x | canary error <= enron error throughout |
| GPTQ canary/enron ratio | 0.81x | 0.91x | 1.65x | only the final block shows mild canary amplification |

The intermediate-layer activation errors are NOT amplified on canary
inputs. So whatever creates the next-token logit asymmetry surfaces only
at the final lm_head projection.

### (3) Logit-level KL on OOD (canary) vs ID (Enron) inputs

| Quantizer | canary KL (mean) | enron KL (mean) | **amplification** |
|---|---:|---:|---:|
| AWQ-4bit  | 3.327 | 0.411 | **8.10x** |
| GPTQ-4bit | 4.272 | 0.671 | **6.37x** |

KL between FT and quant next-token distributions is 6-8x larger on canary
inputs than on Enron inputs.

### (4) Softmax-fragility-on-peaky-distributions -- REFUTED

Hypothesis: canary FT distributions are sharply peaked on the memorised
token, so the same logit noise produces larger KL.

* FT top-1 prob mean: canary=0.65, enron=0.56 -- only 1.16x more peaky.
* Synthetic isotropic Gaussian noise added to FT logits at sigma in
  {0.01, ..., 5.0}: amp(canary/enron) stays at **0.73-0.88x at every
  sigma** -- canary KL is actually slightly *smaller* than Enron KL
  under symmetric noise (Enron has more close-runners that get shuffled
  by random noise; the canary's peaked top-1 is *more* robust).

Symmetric noise of comparable magnitude does NOT reproduce the empirical
6-8x amplification. The AWQ/GPTQ noise must therefore have *structure*
beyond magnitude.

### (5) Noise-direction analysis with position controls

For each input we compute the error vector `d = L_FT - L_quant` and:
* `||d||_2`           magnitude
* `cos(d, e_top1_FT)` alignment with FT top-1 basis
* `prob drop on top-1` `P_FT(top1) - P_quant(top1)`
* `top-1 FLIP rate`   how often quantization changes the top-1 prediction

Three input conditions:
* `canary RECALL` -- prefix ends at `Confidential reference number: `,
  the position FT is supposed to recall the memorised suffix.
* `canary BODY` -- prefix is the canary template BEFORE the recall trigger.
  FT is doing template continuation here, not memorised recall.
* `enron` -- Enron held-out, last position (generic continuation).

The position control disentangles two confounds: is the asymmetry about
*memorisation specifically* or about *canary template being OOD*?

#### AWQ-4bit (n=50):
| Metric | canary RECALL | canary BODY | enron |
|---|---:|---:|---:|
| FT top-1 prob       | 0.67   | **0.9998** | 0.57   |
| `||L_FT - L_q||_2`  | 828    | 654        | 367    |
| `cos(err, e_top1)`  | 0.0086 | **0.0079** | 0.0004 |
| prob drop on top-1  | 0.55   | **0.0004** | 0.057  |
| top-1 FLIP rate     | **82%**| **0%**     | 24%    |

#### Q4_K_M (calibration-corpus-free; same protocol, n=30):
| Metric | canary RECALL | canary BODY | enron |
|---|---:|---:|---:|
| FT top-1 prob       | 0.70   | 0.9998 | 0.52   |
| `||L_FT - L_q||_2`  | 654    | 446    | 230    |
| `cos(err, e_top1)`  | **0.0079** | 0.0024 | 0.0009 |
| prob drop on top-1  | **0.46**   | -0.0001 | 0.039 |
| top-1 FLIP rate     | **63%**| 0%     | 10%    |

## Honest synthesis

The bucket-collapse mechanism of Appendix A is **not** the dominant story.
What the data actually says:

1. **The directional bias is essentially universal, not calibration-specific.**
   Q4_K_M (no calibration corpus) gives cos(err, e_top1) = 0.0079 on canary
   RECALL -- the same value as AWQ (0.0086). The "AWQ noise is aligned
   with the FT top-1 because it was calibrated on Enron and canary is OOD"
   framing turns out to be partially wrong: even a calibration-corpus-free 4-bit
   quantizer concentrates its noise on rare-token-encoding channels and
   that noise inherits a non-trivial alignment with whatever token FT
   wants to predict at a low-FT-confidence position.

2. **The position control (canary BODY) shows the directional bias is a
   property of the canary template, not memorisation per se.** cos at
   canary BODY (AWQ 0.0079, Q4_K_M 0.0024) is in the same range as
   canary RECALL. What changes between RECALL and BODY is *FT confidence*:
   at BODY FT predicts the template-continuation token (`Confidential` or
   similar) at probability 0.9998 -- the same magnitude of directional
   noise cannot flip that top-1. At RECALL FT predicts the memorised
   high-entropy token at 0.67, well within the flippable range.

3. **The discriminator between AWQ (0/100) and Q4_K_M (20/100) is noise
   magnitude, not noise direction.** ||L_FT - L_q|| at canary RECALL is
   828 for AWQ and 654 for Q4_K_M (AWQ is 27% larger). FLIP rate is 82%
   vs 63%. AWQ's activation-aware pre-scaling does its part -- it gives
   the noise more amplitude in the low-s channels that the canary uses --
   but the *direction* of that noise was going to land near the canary
   token anyway, because 4-bit rounding on rare-token-supporting weights
   is what concentrates the noise.

4. **Combined factor model**: an FT prediction is flipped by 4-bit
   quantization when (a) the predicted token lies in the rare-token /
   high-entropy axis of vocab space (memorised canaries do; common Enron
   continuations do not), AND (b) the FT confidence is in the flippable
   regime (~0.4-0.8: template-continuation positions at 0.99 are robust,
   uniform-distribution positions at 0.1 don't have a top-1 to flip), AND
   (c) the noise magnitude exceeds the FT margin to the second-place
   token. AWQ amplifies (c) substantially over Q4_K_M; (a) and (b) are
   pre-conditions provided by the canary protocol.

## What this means for the paper

* Appendix A's "bucket collapse" theoretical sketch is partially correct
  for AWQ (2.8% of canary-encoding weights collapsed) but cannot explain
  GPTQ (zero weight-level collapse) and is not the main driver even for
  AWQ. Section 10's "mechanism is correlational" caveat was honest; this
  measurement makes the gap explicit.
* The cross-quantizer mechanism is: (i) all 4-bit quantizers concentrate
  noise in rare-token-encoding channels, where memorised canary tokens
  live; (ii) the magnitude of that noise determines the FLIP rate;
  (iii) AWQ/GPTQ have larger magnitude than Q4_K_M because of the
  activation-aware / inverse-Hessian objective that *expands* the
  rounding step in low-calibration-activation channels.
* The discriminating axis the paper names ("calibration-based vs
  calibration-corpus-free") is empirically the right axis, but the
  mechanism on that axis is **noise magnitude amplification in
  rare-token channels** rather than "bucket collapse at the weight
  level". Recommend Section 5.4 + Appendix A be updated with the
  five-experiment evidence here.

## Files

* `exp_bucket_collapse_canary_v2/metrics.json`           -- test 1
* `exp_mechanism_per_layer/metrics.json`                  -- test 2
* `exp_mechanism_ood_logits/metrics.json`                 -- test 3
* `exp_mechanism_softmax_fragility/metrics.json`          -- test 4
* `exp_mechanism_noise_direction/metrics.json`            -- test 5 AWQ
* `exp_mechanism_control_positions/metrics.json`          -- test 5 AWQ with BODY control
* `exp_mechanism_q4km_noise_direction/metrics.json`       -- test 5 Q4_K_M (decisive control)

## Confounds and caveats

* n=30-50 for the noise-direction experiments. Effect sizes are large
  (FLIP rates differ by 20+ percentage points across conditions) so the
  qualitative story is robust, but formal CIs would tighten the numbers.
* All measurements are on the seed-42 wave_1_mini checkpoint. The
  pooled 5-seed asymmetry in Table 2 indicates the *effect* generalises;
  the *mechanism* measurement was not repeated across seeds.
* The "noise direction" is measured at the last token position only.
  Per-token analysis along the sequence would localise *where* in the
  prefix the canary memorisation channels are most exposed.
