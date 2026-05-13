#!/usr/bin/env python3
"""Control experiment: is the 9.56x prob-drop-on-top-1 we measured between
canary-last-position and Enron-last-position really a memorization-vs-other
effect, or a "canary-template-is-OOD" effect that has nothing to do with
the memorised suffix?

We split each canary prefix into:
   PREFIX_BODY = everything up to "Confidential reference number:"
   FULL_PREFIX = PREFIX_BODY + " Confidential reference number: "

At PREFIX_BODY's last position the FT model is doing *normal language
continuation within the canary email body* (no memorised recall yet).
At FULL_PREFIX's last position the FT model is positioned exactly where
it recalls the memorised 10-char reference.

For Enron we similarly compare last-position vs mid-position (chosen at
the same relative-fraction-of-length as the canary's PREFIX_BODY end).

If AWQ's effect (prob-drop-on-top-1, top-1-flip-rate) is dramatic only at
the MEMORISED recall position and roughly equal between canary-MID,
canary-BODY, and Enron at all positions, then the mechanism is
*memorisation-specific* and not just "canary template is OOD".
"""
from __future__ import annotations
import argparse, gc, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft-dir", required=True)
    ap.add_argument("--awq-dir", required=True)
    ap.add_argument("--canaries-jsonl", required=True)
    ap.add_argument("--enron-txt", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    canary_rows = [json.loads(l) for l in Path(a.canaries_jsonl).read_text().splitlines() if l.strip()][:a.n]
    enron_chunks = [c.strip() for c in Path(a.enron_txt).read_text().split("\n\n") if len(c.strip()) > 50][:a.n]

    # Three input sets, all evaluated at LAST TOKEN POSITION:
    # 1. canary_RECALL = full prefix, last position is where memorised recall happens
    canary_recall_inputs = [r["prefix_text"] for r in canary_rows]
    # 2. canary_BODY = canary email body truncated BEFORE "Confidential reference number:"
    #    -> at this last position FT is doing normal continuation within a canary-template email
    canary_body_inputs = []
    for r in canary_rows:
        t = r["prefix_text"]
        idx = t.find("Confidential reference number:")
        canary_body_inputs.append(t[:idx].rstrip() if idx > 0 else t)
    # 3. enron_FULL = Enron held-out, truncated to last 400 chars
    enron_full_inputs = [c[:400] for c in enron_chunks]

    # Tokenizer + FT model
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

    def softmax(L):
        a_ = L - L.max(axis=-1, keepdims=True)
        p = np.exp(a_); p /= p.sum(axis=-1, keepdims=True)
        return p

    print("[control] loading FT ...")
    ft = AutoModelForCausalLM.from_pretrained(a.ft_dir, torch_dtype=torch.bfloat16,
                                              low_cpu_mem_usage=True).to(a.device).eval()
    print("[control] FT logits on canary_RECALL ...")
    L_ft_recall = hf_last_logits(ft, canary_recall_inputs)
    print("[control] FT logits on canary_BODY (control: no memorised recall) ...")
    L_ft_body = hf_last_logits(ft, canary_body_inputs)
    print("[control] FT logits on enron ...")
    L_ft_enron = hf_last_logits(ft, enron_full_inputs)
    del ft; torch.cuda.empty_cache(); gc.collect()

    print("[control] loading AWQ ...")
    from awq import AutoAWQForCausalLM
    awq = AutoAWQForCausalLM.from_quantized(a.awq_dir, fuse_layers=False, safetensors=True)
    inner = awq.model if hasattr(awq, "model") else awq
    inner.to(a.device).eval()
    print("[control] AWQ logits on canary_RECALL ...")
    L_q_recall = hf_last_logits(inner, canary_recall_inputs)
    print("[control] AWQ logits on canary_BODY ...")
    L_q_body = hf_last_logits(inner, canary_body_inputs)
    print("[control] AWQ logits on enron ...")
    L_q_enron = hf_last_logits(inner, enron_full_inputs)
    del awq, inner; torch.cuda.empty_cache(); gc.collect()

    def analyze(L_ft, L_q, label):
        P_ft = softmax(L_ft); P_q = softmax(L_q)
        top1_ft = P_ft.argmax(axis=-1)
        d = L_ft - L_q
        mag = np.linalg.norm(d, axis=-1)
        align = np.array([d[i, top1_ft[i]] / max(1e-12, mag[i]) for i in range(len(top1_ft))])
        pdrop = np.array([P_ft[i, top1_ft[i]] - P_q[i, top1_ft[i]] for i in range(len(top1_ft))])
        flip = float((P_q.argmax(axis=-1) != top1_ft).mean())
        kl_per = np.array([
            (P_ft[i] * (np.log(P_ft[i] + 1e-12) - np.log(P_q[i] + 1e-12))).sum()
            for i in range(len(P_ft))
        ])
        ft_top1_prob = float(P_ft.max(axis=-1).mean())
        return {
            "n": len(L_ft),
            "ft_top1_prob_mean": ft_top1_prob,
            "logit_err_norm_mean": float(mag.mean()),
            "cos_err_top1_mean": float(align.mean()),
            "prob_drop_on_top1_mean": float(pdrop.mean()),
            "top1_flip_rate": flip,
            "kl_mean": float(kl_per.mean()),
            "kl_median": float(np.median(kl_per)),
        }

    out = {"schema": "qquilt.mech_control_positions.v1",
           "config": vars(a),
           "canary_RECALL": analyze(L_ft_recall, L_q_recall, "canary RECALL"),
           "canary_BODY":   analyze(L_ft_body,   L_q_body,   "canary BODY (no recall)"),
           "enron":         analyze(L_ft_enron,  L_q_enron,  "enron")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"[control] wrote {a.out}")

    print()
    print("=" * 78)
    print("CONTROL: is the 9.56x prob-drop on canary memorization-specific or template-specific?")
    print(f"{'metric':28s}  {'canary RECALL':>14s}  {'canary BODY':>14s}  {'enron':>14s}")
    print("-" * 78)
    keys = [("ft_top1_prob_mean", "FT top-1 prob"),
            ("logit_err_norm_mean", "||L_ft - L_quant||"),
            ("cos_err_top1_mean", "cos(err, e_top1)"),
            ("prob_drop_on_top1_mean", "prob drop on top-1"),
            ("top1_flip_rate", "top-1 FLIP rate"),
            ("kl_mean", "KL(P_ft || P_quant)")]
    for k, lbl in keys:
        r = out["canary_RECALL"][k]; b = out["canary_BODY"][k]; e = out["enron"][k]
        print(f"  {lbl:26s}  {r:>14.4f}  {b:>14.4f}  {e:>14.4f}")
    print("=" * 78)
    print("Interpretation:")
    print(" * if canary_BODY ~~ enron and both << canary_RECALL  => memorization-specific")
    print(" * if canary_BODY ~~ canary_RECALL and both >> enron  => canary-template (OOD) effect")


if __name__ == "__main__":
    main()
