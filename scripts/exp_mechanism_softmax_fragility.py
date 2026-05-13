#!/usr/bin/env python3
"""Direct test of the softmax-fragility hypothesis.

Prior measurements established:
  (a) per-layer residual-stream error is NOT amplified for canary vs Enron
      under AWQ/GPTQ (ratio ~0.5-0.99x, except GPTQ last layer 1.65x);
  (b) yet next-token logit KL(FT || quant) on canary inputs is 6-8x larger
      than on Enron inputs.

The gap is consistent with the softmax being a non-linear amplifier of small
logit perturbations when the FT distribution is sharply peaked -- as it is
on memorised canary tokens. This script proves that mechanism directly:

  1) Peakiness of FT next-token distribution on canary inputs (top-1 prob,
     Shannon entropy) vs Enron inputs.
  2) Synthetic Gaussian noise sweep: take FT logits unchanged, add
     N(0, sigma^2) noise, compute KL(softmax(L_ft) || softmax(L_ft + noise))
     per input. Sweep sigma over a range matching the empirical
     ||logits_ft - logits_quant|| we observed.

If softmax fragility on peaky distributions IS the mechanism, we predict:
  - Canary FT distributions have much smaller entropy / higher top-1 prob.
  - At every sigma, KL_canary(sigma) >> KL_enron(sigma); the ratio matches
    the empirically observed 6-8x in the AWQ/GPTQ experiment.

If the prediction fails (e.g., synthetic-noise amplification is small even
at large sigma), then quantization noise must have structure that beyond
just "magnitude" -- pointing at calibration-induced noise direction.
"""
from __future__ import annotations
import argparse, gc, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft-dir", required=True)
    ap.add_argument("--canaries-jsonl", required=True)
    ap.add_argument("--enron-txt", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load inputs
    canary_rows = [json.loads(l) for l in Path(a.canaries_jsonl).read_text().splitlines() if l.strip()][:a.n]
    canary_prefixes = [r["prefix_text"] for r in canary_rows]
    enron_chunks = [c.strip() for c in Path(a.enron_txt).read_text().split("\n\n") if len(c.strip()) > 50][:a.n]
    enron_inputs = [c[:400] for c in enron_chunks]

    tokenizer = AutoTokenizer.from_pretrained(a.ft_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[softmax-frag] loading FT model ...")
    ft = AutoModelForCausalLM.from_pretrained(a.ft_dir, torch_dtype=torch.bfloat16,
                                              low_cpu_mem_usage=True).to(a.device).eval()

    def hf_last_logits(texts):
        out = []
        with torch.no_grad():
            for t in texts:
                inp = tokenizer(t, return_tensors="pt", truncation=True, max_length=480).to(a.device)
                lg = ft(**inp).logits[0, -1, :].float().cpu().numpy()
                out.append(lg)
        return np.stack(out, axis=0)

    print(f"[softmax-frag] FT canary logits ...")
    L_can = hf_last_logits(canary_prefixes)
    print(f"[softmax-frag] FT enron logits ...")
    L_enr = hf_last_logits(enron_inputs)
    del ft; torch.cuda.empty_cache(); gc.collect()

    # ---- 1) Peakiness measurement on FT distributions ----
    def stats(logits):
        a_ = logits - logits.max(axis=-1, keepdims=True)
        p = np.exp(a_); p /= p.sum(axis=-1, keepdims=True)
        top1 = p.max(axis=-1)
        entropy = -(p * np.log(p + 1e-12)).sum(axis=-1)
        log_p_top1 = np.log(top1 + 1e-12)
        return {"top1_prob": {"mean": float(top1.mean()), "median": float(np.median(top1)),
                              "p10": float(np.percentile(top1, 10)), "p90": float(np.percentile(top1, 90))},
                "entropy":   {"mean": float(entropy.mean()), "median": float(np.median(entropy)),
                              "p10": float(np.percentile(entropy, 10)), "p90": float(np.percentile(entropy, 90))},
                "log_top1":  {"mean": float(log_p_top1.mean())}}
    print("[softmax-frag] computing FT peakiness stats ...")
    peakiness = {"canary": stats(L_can), "enron": stats(L_enr)}
    print(f"  canary FT top-1 prob: mean={peakiness['canary']['top1_prob']['mean']:.4f}  "
          f"entropy={peakiness['canary']['entropy']['mean']:.4f}")
    print(f"  enron  FT top-1 prob: mean={peakiness['enron']['top1_prob']['mean']:.4f}  "
          f"entropy={peakiness['enron']['entropy']['mean']:.4f}")
    print(f"  peakiness gap: canary top-1 is {peakiness['canary']['top1_prob']['mean']/peakiness['enron']['top1_prob']['mean']:.2f}x higher")
    print(f"  entropy gap: canary entropy is {peakiness['canary']['entropy']['mean']/peakiness['enron']['entropy']['mean']:.2f}x of enron's")

    # ---- 2) Synthetic noise sweep ----
    def kl_self_perturbed(logits, sigma, rng):
        noise = rng.normal(0.0, sigma, size=logits.shape).astype(np.float32)
        L2 = logits + noise
        # KL(softmax(L) || softmax(L2)) per row
        a = logits - logits.max(axis=-1, keepdims=True)
        b = L2 - L2.max(axis=-1, keepdims=True)
        log_pa = a - np.log(np.exp(a).sum(axis=-1, keepdims=True))
        log_pb = b - np.log(np.exp(b).sum(axis=-1, keepdims=True))
        pa = np.exp(log_pa)
        kl = (pa * (log_pa - log_pb)).sum(axis=-1)
        return kl

    rng = np.random.default_rng(42)
    sweep = {}
    print("[softmax-frag] synthetic Gaussian noise KL sweep ...")
    SIGMAS = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    for sigma in SIGMAS:
        kl_can = kl_self_perturbed(L_can, sigma, rng)
        kl_enr = kl_self_perturbed(L_enr, sigma, rng)
        amp = kl_can.mean() / max(1e-12, kl_enr.mean())
        sweep[f"sigma_{sigma:g}"] = {
            "kl_canary_mean": float(kl_can.mean()),
            "kl_enron_mean":  float(kl_enr.mean()),
            "amplification_canary_over_enron": float(amp),
        }
        print(f"  sigma={sigma:5.2f}  canary KL={kl_can.mean():.4f}  enron KL={kl_enr.mean():.4f}  "
              f"amp={amp:.2f}x")

    # ---- Save ----
    out = {"schema": "qquilt.mech_softmax_fragility.v1",
           "config": vars(a),
           "ft_peakiness": peakiness,
           "noise_sweep": sweep,
           "empirical_observed_amplification": {"awq": 8.10, "gptq": 6.37, "note": "from exp_mechanism_ood_logits"}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"[softmax-frag] wrote {a.out}")

    print()
    print("=" * 72)
    print("SOFTMAX FRAGILITY: synthetic noise reproduces 6-8x amplification?")
    print(f"  Empirical (AWQ): 8.10x   Empirical (GPTQ): 6.37x")
    print(f"  Closest matching sigma in the synthetic sweep:")
    best = sorted(sweep.items(), key=lambda kv: abs(kv[1]["amplification_canary_over_enron"] - 7.0))[0]
    print(f"  -> {best[0]}: amp={best[1]['amplification_canary_over_enron']:.2f}x (closest to 7x)")
    print("If amp ~7x is reached around the same noise level as ||L_ft - L_quant||,")
    print("then softmax fragility on peaky distributions IS the mechanism.")
    print("=" * 72)


if __name__ == "__main__":
    main()
