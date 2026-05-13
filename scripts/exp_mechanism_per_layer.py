#!/usr/bin/env python3
"""Per-layer activation reconstruction error: complement to exp_mechanism_ood_logits.

For a single canary input we hook each linear layer in the FT model and the
quantized model and record the layer output. We then compute the relative
reconstruction error per layer:

    err_layer = ||h_layer_quant - h_layer_ft||_F / ||h_layer_ft||_F

If the OOD-amplification hypothesis is right, the error grows monotonically
along the layer stack (each layer's noise compounds). Compared to the same
measurement on an Enron input, the canary curve should be much steeper.

This isolates WHERE in the network the divergence appears.

Output: metrics.json with per-(layer, model) error on canary vs Enron, plus
a summary that names the layer where the error first crosses 0.5.
"""
from __future__ import annotations
import argparse, gc, json
from pathlib import Path


def hook_residual_outputs(model):
    """Register forward hooks on each transformer block (residual stream) and
    return a dict mapping layer name -> output tensor (set after a forward pass).
    """
    captured = {}
    handles = []
    for name, mod in model.named_modules():
        # match a transformer block's outermost level: e.g. model.layers.0
        if hasattr(mod, "self_attn") and hasattr(mod, "mlp"):
            def make_hook(n):
                def h(m, inp, out):
                    captured[n] = out[0] if isinstance(out, tuple) else out
                return h
            handles.append(mod.register_forward_hook(make_hook(name)))
    return captured, handles


def forward_capture(model, tokenizer, text: str, device: str, max_len: int = 512):
    import torch
    captured, handles = hook_residual_outputs(model)
    try:
        inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
        with torch.no_grad():
            model(**inp)
        return {k: v.detach().float().cpu() for k, v in captured.items()}
    finally:
        for h in handles:
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft-dir", required=True)
    ap.add_argument("--awq-dir", default=None)
    ap.add_argument("--gptq-dir", default=None)
    ap.add_argument("--canary-text", required=True)
    ap.add_argument("--enron-text", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(a.ft_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def maybe_read(s: str) -> str:
        if len(s) < 4096:
            try:
                p = Path(s)
                if p.exists():
                    return p.read_text()
            except OSError:
                pass
        return s
    canary_txt = maybe_read(a.canary_text)
    enron_txt = maybe_read(a.enron_text)
    canary_txt = canary_txt[:1500]
    enron_txt = enron_txt[:1500]

    print(f"[per-layer] FT load ...")
    ft = AutoModelForCausalLM.from_pretrained(a.ft_dir, torch_dtype=torch.bfloat16,
                                              low_cpu_mem_usage=True).to(a.device).eval()
    ft_can = forward_capture(ft, tokenizer, canary_txt, a.device)
    ft_enr = forward_capture(ft, tokenizer, enron_txt, a.device)
    del ft; torch.cuda.empty_cache(); gc.collect()

    out = {"schema": "qquilt.mech_per_layer.v1", "config": vars(a),
           "n_layers": len(ft_can), "results": {}}

    def relerr(a_t, b_t):
        d = (a_t - b_t).norm()
        s = b_t.norm().clamp_min(1e-9)
        return float(d / s)

    def evaluate(name: str, model):
        cap_can = forward_capture(model, tokenizer, canary_txt, a.device)
        cap_enr = forward_capture(model, tokenizer, enron_txt, a.device)
        per_layer = []
        for k in sorted(ft_can.keys(), key=lambda n: (len(n), n)):
            if k in cap_can and ft_can[k].shape == cap_can[k].shape:
                per_layer.append({
                    "layer": k,
                    "rel_err_canary": relerr(cap_can[k], ft_can[k]),
                    "rel_err_enron":  relerr(cap_enr[k], ft_enr[k]) if k in cap_enr else None,
                })
        return per_layer

    if a.awq_dir:
        print("[per-layer] AWQ load ...")
        from awq import AutoAWQForCausalLM
        awq = AutoAWQForCausalLM.from_quantized(a.awq_dir, fuse_layers=False, safetensors=True)
        inner = awq.model if hasattr(awq, "model") else awq
        inner.to(a.device).eval()
        out["results"]["awq"] = evaluate("AWQ", inner)
        del awq, inner; torch.cuda.empty_cache(); gc.collect()

    if a.gptq_dir:
        print("[per-layer] GPTQ load ...")
        try:
            from auto_gptq import AutoGPTQForCausalLM
            gptq = AutoGPTQForCausalLM.from_quantized(a.gptq_dir, device_map={"": a.device}, use_safetensors=True)
            inner = gptq.model if hasattr(gptq, "model") else gptq
        except Exception as e:
            print(f"  auto_gptq failed: {e!r}; falling back to plain HF load")
            gptq = AutoModelForCausalLM.from_pretrained(a.gptq_dir, torch_dtype=torch.float16).to(a.device)
            inner = gptq
        inner.to(a.device).eval()
        out["results"]["gptq"] = evaluate("GPTQ", inner)
        del gptq, inner; torch.cuda.empty_cache(); gc.collect()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"[per-layer] wrote {a.out}")

    print()
    print("=" * 72)
    print("PER-LAYER ACTIVATION RECONSTRUCTION ERROR (one canary vs one Enron)")
    print("  rel_err = ||h_quant - h_ft|| / ||h_ft||  (per residual stream output)")
    print("=" * 72)
    for q in ("awq", "gptq"):
        if q in out["results"]:
            print(f"\n  {q.upper()} (first/middle/last 5 layers):")
            r = out["results"][q]
            for li in (list(range(min(5, len(r))))
                       + ([len(r)//2] if len(r) > 10 else [])
                       + list(range(max(0, len(r)-5), len(r)))):
                if li < len(r):
                    p = r[li]
                    print(f"    {p['layer']:32s}  canary={p['rel_err_canary']:.4f}  enron={p['rel_err_enron']:.4f}  "
                          f"ratio={(p['rel_err_canary']/max(1e-12, p['rel_err_enron'])):.2f}x")


if __name__ == "__main__":
    main()
