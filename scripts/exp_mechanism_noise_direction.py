#!/usr/bin/env python3
"""Decisive mechanism test: is AWQ/GPTQ quantization noise STRUCTURED to
specifically degrade the top-1 prediction (the memorised canary token)?

Prior measurements:
  * v2 weight-level survival ~1 (no bucket collapse for GPTQ; partial for AWQ).
  * per-layer residual stream error: NOT amplified on canary (ratio 0.5-0.99x).
  * next-token logit KL on canary: 6-8x amplified.
  * synthetic Gaussian noise of equivalent magnitude reproduces NO
    amplification (amp 0.73-0.88x at all sigma) -- refutes the simple
    softmax-fragility-on-peaky-distributions hypothesis.

So the AWQ/GPTQ logit error is NOT symmetric Gaussian. It must have direction.

This script computes the actual logit-error vector  d = L_FT - L_quant  on
canary vs Enron inputs, and decomposes it:

  (a) ||d||_2 -- magnitude (should be similar on canary and Enron)
  (b) <d, e_{top1_ft}> / ||d|| -- cosine alignment between the error vector
      and the basis vector for the FT model's top-1 prediction. A *negative*
      value means the error pushes mass AWAY from the FT top-1 token.
  (c) P_FT(top1) - P_quant(top1) -- the actual probability drop on the
      memorised top-1 token

If the hypothesis is right:
   * ||d||_canary ~~ ||d||_enron               (magnitudes comparable)
   * cos(d, e_top1) << 0  for canary, ~~ 0 for Enron
   * P_FT(top1) - P_quant(top1) is much larger on canary

That is structured, calibration-directed noise -- the quantizer's rounding
error preferentially pushes mass off the canary's memorised completion.
"""
from __future__ import annotations
import argparse, gc, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft-dir", required=True)
    ap.add_argument("--awq-dir", default=None)
    ap.add_argument("--gptq-dir", default=None)
    ap.add_argument("--canaries-jsonl", required=True)
    ap.add_argument("--enron-txt", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    canary_rows = [json.loads(l) for l in Path(a.canaries_jsonl).read_text().splitlines() if l.strip()][:a.n]
    canary_prefixes = [r["prefix_text"] for r in canary_rows]
    enron_chunks = [c.strip() for c in Path(a.enron_txt).read_text().split("\n\n") if len(c.strip()) > 50][:a.n]
    enron_inputs = [c[:400] for c in enron_chunks]

    tokenizer = AutoTokenizer.from_pretrained(a.ft_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def hf_last_logits(model, texts):
        out = []
        with torch.no_grad():
            for t in texts:
                inp = tokenizer(t, return_tensors="pt", truncation=True, max_length=480).to(a.device)
                lg = model(**inp).logits[0, -1, :].float().cpu().numpy()
                out.append(lg)
        return np.stack(out, axis=0)

    print(f"[noise-dir] loading FT ...")
    ft = AutoModelForCausalLM.from_pretrained(a.ft_dir, torch_dtype=torch.bfloat16,
                                              low_cpu_mem_usage=True).to(a.device).eval()
    L_ft_can = hf_last_logits(ft, canary_prefixes)
    L_ft_enr = hf_last_logits(ft, enron_inputs)
    del ft; torch.cuda.empty_cache(); gc.collect()

    def softmax(L):
        a_ = L - L.max(axis=-1, keepdims=True)
        p = np.exp(a_); p /= p.sum(axis=-1, keepdims=True)
        return p

    P_ft_can = softmax(L_ft_can); P_ft_enr = softmax(L_ft_enr)
    top1_can = P_ft_can.argmax(axis=-1)
    top1_enr = P_ft_enr.argmax(axis=-1)

    def analyze(L_quant_can, L_quant_enr, name: str):
        # error vectors
        d_can = L_ft_can - L_quant_can   # shape (n, V)
        d_enr = L_ft_enr - L_quant_enr
        # magnitude
        mag_can = np.linalg.norm(d_can, axis=-1)
        mag_enr = np.linalg.norm(d_enr, axis=-1)
        # alignment with top-1 basis: d[i, top1] / ||d||
        align_can = np.array([d_can[i, top1_can[i]] / max(1e-12, mag_can[i])
                              for i in range(len(top1_can))])
        align_enr = np.array([d_enr[i, top1_enr[i]] / max(1e-12, mag_enr[i])
                              for i in range(len(top1_enr))])
        # absolute logit drop on top-1
        drop_can = np.array([d_can[i, top1_can[i]] for i in range(len(top1_can))])
        drop_enr = np.array([d_enr[i, top1_enr[i]] for i in range(len(top1_enr))])
        # prob drop on top-1
        P_q_can = softmax(L_quant_can); P_q_enr = softmax(L_quant_enr)
        pdrop_can = np.array([P_ft_can[i, top1_can[i]] - P_q_can[i, top1_can[i]] for i in range(len(top1_can))])
        pdrop_enr = np.array([P_ft_enr[i, top1_enr[i]] - P_q_enr[i, top1_enr[i]] for i in range(len(top1_enr))])
        # how often does quant FLIP the top-1?
        flip_can = float((P_q_can.argmax(axis=-1) != top1_can).mean())
        flip_enr = float((P_q_enr.argmax(axis=-1) != top1_enr).mean())
        return {
            "logit_err_norm":  {"canary": float(mag_can.mean()), "enron": float(mag_enr.mean()),
                                 "ratio_canary_over_enron": float(mag_can.mean() / max(1e-12, mag_enr.mean()))},
            "cos_err_with_top1_basis": {"canary": float(align_can.mean()), "enron": float(align_enr.mean()),
                                         "canary_median": float(np.median(align_can)),
                                         "enron_median":  float(np.median(align_enr))},
            "abs_logit_drop_on_top1":  {"canary": float(drop_can.mean()), "enron": float(drop_enr.mean()),
                                         "ratio_canary_over_enron": float(drop_can.mean() / max(1e-12, drop_enr.mean()))},
            "prob_drop_on_top1":       {"canary": float(pdrop_can.mean()), "enron": float(pdrop_enr.mean()),
                                         "ratio_canary_over_enron": float(pdrop_can.mean() / max(1e-12, pdrop_enr.mean()))},
            "top1_flip_rate":          {"canary": flip_can, "enron": flip_enr},
        }

    out = {"schema": "qquilt.mech_noise_direction.v1", "config": vars(a), "results": {}}

    if a.awq_dir:
        print(f"[noise-dir] loading AWQ ...")
        from awq import AutoAWQForCausalLM
        awq = AutoAWQForCausalLM.from_quantized(a.awq_dir, fuse_layers=False, safetensors=True)
        inner = awq.model if hasattr(awq, "model") else awq
        inner.to(a.device).eval()
        L_q_can = hf_last_logits(inner, canary_prefixes)
        L_q_enr = hf_last_logits(inner, enron_inputs)
        del awq, inner; torch.cuda.empty_cache(); gc.collect()
        out["results"]["awq"] = analyze(L_q_can, L_q_enr, "AWQ")

    if a.gptq_dir:
        print(f"[noise-dir] loading GPTQ ...")
        try:
            from auto_gptq import AutoGPTQForCausalLM
            gptq = AutoGPTQForCausalLM.from_quantized(a.gptq_dir, device_map={"": a.device}, use_safetensors=True)
            inner = gptq.model if hasattr(gptq, "model") else gptq
        except Exception as e:
            print(f"  auto_gptq fallback: {e!r}")
            gptq = AutoModelForCausalLM.from_pretrained(a.gptq_dir, torch_dtype=torch.float16).to(a.device)
            inner = gptq
        inner.to(a.device).eval()
        L_q_can = hf_last_logits(inner, canary_prefixes)
        L_q_enr = hf_last_logits(inner, enron_inputs)
        del gptq, inner; torch.cuda.empty_cache(); gc.collect()
        out["results"]["gptq"] = analyze(L_q_can, L_q_enr, "GPTQ")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"[noise-dir] wrote {a.out}")

    print()
    print("=" * 78)
    print("STRUCTURED NOISE DIRECTION TEST")
    print("  Hypothesis: AWQ/GPTQ noise vector points AWAY from FT's top-1 on canary.")
    print()
    for q in ("awq", "gptq"):
        if q in out["results"]:
            r = out["results"][q]
            print(f"  {q.upper()}:")
            print(f"    ||L_FT - L_quant||_2:  canary={r['logit_err_norm']['canary']:.3f}  "
                  f"enron={r['logit_err_norm']['enron']:.3f}  "
                  f"ratio={r['logit_err_norm']['ratio_canary_over_enron']:.2f}x")
            print(f"    cos(err, e_top1):      canary={r['cos_err_with_top1_basis']['canary']:+.4f}  "
                  f"enron={r['cos_err_with_top1_basis']['enron']:+.4f}    "
                  f"(negative = error pushes mass AWAY from top-1)")
            print(f"    prob drop on top-1:    canary={r['prob_drop_on_top1']['canary']:.4f}  "
                  f"enron={r['prob_drop_on_top1']['enron']:.4f}  "
                  f"ratio={r['prob_drop_on_top1']['ratio_canary_over_enron']:.2f}x")
            print(f"    top-1 FLIPPED:         canary={r['top1_flip_rate']['canary']*100:.0f}%  "
                  f"enron={r['top1_flip_rate']['enron']*100:.0f}%")
    print("=" * 78)


if __name__ == "__main__":
    main()
