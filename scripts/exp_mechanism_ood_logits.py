#!/usr/bin/env python3
"""Mechanism isolation: are AWQ / GPTQ quantization errors amplified on
out-of-distribution (canary) inputs vs in-distribution (Enron) inputs?

Bucket-collapse v2 showed that AWQ has partial weight-level collapse but
GPTQ has none (median survival 1.02). Yet both extract 0/100 canaries.
For GPTQ in particular the mechanism cannot be bucket collapse at the
weight level. Hypothesis: AWQ/GPTQ optimize rounding boundaries to
minimize ||X_calib W_q - X_calib W_ft|| on the *Enron* calibration set,
which leaves them MISALIGNED on canary inputs that are far out-of-
distribution relative to Enron. The misalignment surfaces only at
inference, as a much larger ||X W_q - X W_ft|| on canary inputs than on
Enron inputs.

This script tests the prediction. For three models (FT bf16, FT->AWQ-4bit,
FT->GPTQ-4bit), we compute the next-token logit distribution at the last
position for:
   - 100 G1 canary prefixes (up to and including "Confidential reference
     number: ") -- OOD relative to the AWQ/GPTQ calibration corpus
   - 100 Enron held-out emails, truncated to the same length -- ID

We then compute KL(P_FT || P_quant) per input, aggregate, and report the
*amplification ratio*: mean(KL_canary) / mean(KL_enron). If the hypothesis
is right:

   * AWQ amplification >> 1
   * GPTQ amplification >> 1
   * On a calibration-corpus-free quantizer (Q4_K_M, future work; can't
     be loaded as HF without GGUF dequantization) we would expect
     amplification ~ 1.

This is the inference-time analogue of the weight-level survival measure
in exp_bucket_collapse_canary_v2.

Output: metrics.json + RESULTS.md.
"""
from __future__ import annotations
import argparse, gc, json, math, os
from pathlib import Path


def load_canary_prefixes(canary_jsonl: Path, n: int = 100) -> list[str]:
    rows = [json.loads(l) for l in canary_jsonl.read_text().splitlines() if l.strip()]
    pref_idx = -1
    # use the prefix_text field if present, else split the body
    out = []
    for r in rows[:n]:
        if "prefix_text" in r:
            out.append(r["prefix_text"])
        elif "body" in r and "Confidential reference number:" in r["body"]:
            idx = r["body"].index("Confidential reference number:") + len("Confidential reference number:")
            out.append(r["body"][:idx] + " ")
        else:
            raise ValueError(f"unknown canary row schema: {sorted(r)}")
    return out


def load_enron_inputs(enron_txt: Path, n: int, target_chars: int = 400) -> list[str]:
    """Read the held-out Enron corpus and produce n strings of ~target_chars each."""
    blob = enron_txt.read_text()
    # split on blank lines (matches the enron_holdout.txt format used by utility.py)
    chunks = [c.strip() for c in blob.split("\n\n") if len(c.strip()) > 50]
    out = []
    for c in chunks:
        if len(out) >= n:
            break
        # truncate or pad to roughly target_chars
        s = c[:target_chars]
        if len(s) > 50:
            out.append(s)
    return out[:n]


def hf_logits(model, tokenizer, texts: list[str], device: str = "cuda", max_len: int = 512):
    """Return (n, vocab) tensor of last-position logits for each text."""
    import torch
    model.eval()
    outs = []
    with torch.no_grad():
        for t in texts:
            inp = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_len).to(device)
            out = model(**inp)
            last = out.logits[0, -1, :].detach().float().cpu()  # (vocab,)
            outs.append(last)
    return torch.stack(outs, dim=0)  # (n, vocab)


def kl_per_input(logits_a, logits_b):
    """KL(softmax(a) || softmax(b)) per row. Returns (n,) tensor."""
    import torch
    log_pa = torch.log_softmax(logits_a, dim=-1)
    log_pb = torch.log_softmax(logits_b, dim=-1)
    pa = log_pa.exp()
    kl = (pa * (log_pa - log_pb)).sum(dim=-1)
    return kl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft-dir", required=True, help="HF FT checkpoint (bf16)")
    ap.add_argument("--awq-dir", default=None, help="AWQ-4bit checkpoint")
    ap.add_argument("--gptq-dir", default=None, help="GPTQ-4bit checkpoint")
    ap.add_argument("--canaries-jsonl", required=True)
    ap.add_argument("--enron-txt", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    canary_prefixes = load_canary_prefixes(Path(a.canaries_jsonl), a.n)
    enron_inputs = load_enron_inputs(Path(a.enron_txt), a.n)
    print(f"[mech-ood] {len(canary_prefixes)} canary prefixes, {len(enron_inputs)} Enron inputs")
    print(f"  example canary: {canary_prefixes[0][:120]!r}")
    print(f"  example enron : {enron_inputs[0][:120]!r}")

    tokenizer = AutoTokenizer.from_pretrained(a.ft_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[mech-ood] loading FT model ({a.ft_dir}) ...")
    ft = AutoModelForCausalLM.from_pretrained(
        a.ft_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).to(a.device)
    print("[mech-ood] FT: collecting canary logits ...")
    L_ft_can = hf_logits(ft, tokenizer, canary_prefixes, a.device)
    print("[mech-ood] FT: collecting Enron logits ...")
    L_ft_enr = hf_logits(ft, tokenizer, enron_inputs, a.device)
    del ft; torch.cuda.empty_cache(); gc.collect()

    out = {"schema": "qquilt.mech_ood_logits.v1", "config": vars(a),
           "ft_baseline_set": True,
           "results": {}}

    def evaluate(name: str, model):
        print(f"[mech-ood] {name}: collecting canary logits ...")
        L_can = hf_logits(model, tokenizer, canary_prefixes, a.device)
        print(f"[mech-ood] {name}: collecting Enron logits ...")
        L_enr = hf_logits(model, tokenizer, enron_inputs, a.device)
        kl_can = kl_per_input(L_ft_can, L_can).cpu().numpy()
        kl_enr = kl_per_input(L_ft_enr, L_enr).cpu().numpy()
        return {
            "kl_canary": {"mean": float(kl_can.mean()), "median": float(sorted(kl_can)[len(kl_can)//2]),
                          "p10": float(sorted(kl_can)[int(len(kl_can)*0.10)]),
                          "p90": float(sorted(kl_can)[int(len(kl_can)*0.90)])},
            "kl_enron":  {"mean": float(kl_enr.mean()), "median": float(sorted(kl_enr)[len(kl_enr)//2]),
                          "p10": float(sorted(kl_enr)[int(len(kl_enr)*0.10)]),
                          "p90": float(sorted(kl_enr)[int(len(kl_enr)*0.90)])},
            "amplification_canary_over_enron": float(kl_can.mean() / max(1e-12, kl_enr.mean())),
        }

    if a.awq_dir:
        print(f"[mech-ood] loading AWQ ({a.awq_dir}) ...")
        from awq import AutoAWQForCausalLM
        awq = AutoAWQForCausalLM.from_quantized(a.awq_dir, fuse_layers=False, safetensors=True)
        inner = awq.model if hasattr(awq, "model") else awq
        inner.to(a.device); inner.eval()
        out["results"]["awq"] = evaluate("AWQ", inner)
        del awq, inner; torch.cuda.empty_cache(); gc.collect()

    if a.gptq_dir:
        print(f"[mech-ood] loading GPTQ ({a.gptq_dir}) ...")
        try:
            from auto_gptq import AutoGPTQForCausalLM
            gptq = AutoGPTQForCausalLM.from_quantized(a.gptq_dir, device_map={"": a.device}, use_safetensors=True)
            inner = gptq.model if hasattr(gptq, "model") else gptq
        except Exception as e:
            print(f"  auto_gptq failed: {e!r}; falling back to plain HF load")
            gptq = AutoModelForCausalLM.from_pretrained(a.gptq_dir, torch_dtype=torch.float16).to(a.device)
            inner = gptq
        inner.to(a.device); inner.eval()
        out["results"]["gptq"] = evaluate("GPTQ", inner)
        del gptq, inner; torch.cuda.empty_cache(); gc.collect()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"[mech-ood] wrote {a.out}")

    print()
    print("=" * 72)
    print("MECHANISM: KL divergence of next-token logits (FT vs quant)")
    print(f"           on n={a.n} canary inputs (OOD vs Enron calibration)")
    print(f"           vs n={a.n} Enron held-out inputs (ID)")
    print("=" * 72)
    for name in ("awq", "gptq"):
        if name in out["results"]:
            r = out["results"][name]
            ratio = r["amplification_canary_over_enron"]
            print(f"  {name.upper():4} canary KL: mean={r['kl_canary']['mean']:.4f}  "
                  f"median={r['kl_canary']['median']:.4f}")
            print(f"        enron KL:  mean={r['kl_enron']['mean']:.4f}  "
                  f"median={r['kl_enron']['median']:.4f}")
            print(f"        AMPLIFICATION (canary/enron) = {ratio:.2f}x"
                  + ("  <-- supports calibration-OOD-amplification hypothesis" if ratio > 5 else
                     "  <-- no amplification; mechanism is elsewhere"))
    print("=" * 72)


if __name__ == "__main__":
    main()
