#!/usr/bin/env python3
"""Q4_K_M variant of the noise-direction control. Calibration-corpus-free
quantizer: if Q4_K_M shows the SAME directional bias and prob-drop pattern
as AWQ on canary inputs, then the mechanism is NOT calibration-induced
(rare-token effect, would be the alternative). If Q4_K_M shows SYMMETRIC
behavior between canary and Enron (similar cos alignment, similar prob
drop), then the calibration-induced directional noise hypothesis is
confirmed.

Same three-position protocol as exp_mechanism_control_positions.py
(canary RECALL / canary BODY / enron), but with Q4_K_M GGUF served via
llama-cpp-python.
"""
from __future__ import annotations
import argparse, gc, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft-dir", required=True)
    ap.add_argument("--q4km-gguf", required=True)
    ap.add_argument("--canaries-jsonl", required=True)
    ap.add_argument("--enron-txt", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    canary_rows = [json.loads(l) for l in Path(a.canaries_jsonl).read_text().splitlines() if l.strip()][:a.n]
    canary_recall = [r["prefix_text"] for r in canary_rows]
    canary_body = []
    for r in canary_rows:
        t = r["prefix_text"]
        idx = t.find("Confidential reference number:")
        canary_body.append(t[:idx].rstrip() if idx > 0 else t)
    enron_chunks = [c.strip() for c in Path(a.enron_txt).read_text().split("\n\n") if len(c.strip()) > 50][:a.n]
    enron_inputs = [c[:400] for c in enron_chunks]

    tokenizer = AutoTokenizer.from_pretrained(a.ft_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # FT logits (HF on GPU)
    print("[q4km-noise] FT logits (HF on GPU)...")
    ft = AutoModelForCausalLM.from_pretrained(a.ft_dir, torch_dtype=torch.bfloat16,
                                              low_cpu_mem_usage=True).to("cuda").eval()
    def hf_last(model, texts):
        out = []
        with torch.no_grad():
            for t in texts:
                inp = tokenizer(t, return_tensors="pt", truncation=True, max_length=480).to("cuda")
                lg = model(**inp).logits[0, -1, :].float().cpu().numpy()
                out.append(lg)
        return np.stack(out, axis=0)
    L_ft_recall = hf_last(ft, canary_recall)
    L_ft_body = hf_last(ft, canary_body)
    L_ft_enron = hf_last(ft, enron_inputs)
    del ft; torch.cuda.empty_cache(); gc.collect()

    # Q4_K_M logits via llama-cpp-python (CPU)
    print("[q4km-noise] Q4_K_M logits (llama-cpp-python, CPU)...")
    from llama_cpp import Llama
    lcpp = Llama(model_path=a.q4km_gguf, n_ctx=512, n_threads=16,
                 n_gpu_layers=99, logits_all=True, verbose=False)
    vocab_size = lcpp.n_vocab()
    def gguf_last(texts):
        out = []
        for i, t in enumerate(texts):
            tokens = lcpp.tokenize(t.encode("utf-8"), add_bos=True, special=False)[:511]
            lcpp.reset(); lcpp.eval(tokens)
            scores = np.asarray(lcpp.scores)[len(tokens)-1, :vocab_size].astype(np.float32)
            out.append(scores)
            if (i+1) % 10 == 0:
                print(f"  [{i+1}/{len(texts)}]", flush=True)
        return np.stack(out, axis=0)
    L_q_recall = gguf_last(canary_recall)
    L_q_body = gguf_last(canary_body)
    L_q_enron = gguf_last(enron_inputs)

    def softmax(L):
        a_ = L - L.max(axis=-1, keepdims=True)
        p = np.exp(a_); p /= p.sum(axis=-1, keepdims=True)
        return p

    def analyze(L_ft, L_q):
        # truncate vocab to common
        v = min(L_ft.shape[1], L_q.shape[1])
        L_ft = L_ft[:, :v]; L_q = L_q[:, :v]
        P_ft = softmax(L_ft); P_q = softmax(L_q)
        top1_ft = P_ft.argmax(axis=-1)
        d = L_ft - L_q
        mag = np.linalg.norm(d, axis=-1)
        align = np.array([d[i, top1_ft[i]] / max(1e-12, mag[i]) for i in range(len(top1_ft))])
        pdrop = np.array([P_ft[i, top1_ft[i]] - P_q[i, top1_ft[i]] for i in range(len(top1_ft))])
        flip = float((P_q.argmax(axis=-1) != top1_ft).mean())
        kl = np.array([(P_ft[i] * (np.log(P_ft[i]+1e-12) - np.log(P_q[i]+1e-12))).sum() for i in range(len(P_ft))])
        return {"n": len(L_ft),
                "ft_top1_prob_mean": float(P_ft.max(axis=-1).mean()),
                "logit_err_norm_mean": float(mag.mean()),
                "cos_err_top1_mean": float(align.mean()),
                "prob_drop_on_top1_mean": float(pdrop.mean()),
                "top1_flip_rate": flip,
                "kl_mean": float(kl.mean())}

    out = {"schema": "qquilt.mech_q4km_noise_direction.v1",
           "config": vars(a),
           "canary_RECALL": analyze(L_ft_recall, L_q_recall),
           "canary_BODY":   analyze(L_ft_body,   L_q_body),
           "enron":         analyze(L_ft_enron,  L_q_enron)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))

    print()
    print("=" * 78)
    print("Q4_K_M (calibration-corpus-free) noise direction:")
    print(f"{'metric':28s}  {'canary RECALL':>14s}  {'canary BODY':>14s}  {'enron':>14s}")
    print("-" * 78)
    for k, lbl in [("ft_top1_prob_mean","FT top-1 prob"), ("logit_err_norm_mean","||L_ft - L_q||"),
                   ("cos_err_top1_mean","cos(err, e_top1)"), ("prob_drop_on_top1_mean","prob drop on top-1"),
                   ("top1_flip_rate","top-1 FLIP rate"), ("kl_mean","KL mean")]:
        r = out["canary_RECALL"][k]; b = out["canary_BODY"][k]; e = out["enron"][k]
        print(f"  {lbl:26s}  {r:>14.4f}  {b:>14.4f}  {e:>14.4f}")
    print("=" * 78)
    print("If Q4_K_M canary_RECALL cos ~~ enron cos (both near zero) AND")
    print("prob_drop_on_top1 ~~ same for both: NO calibration-OOD bias -> mechanism is")
    print("calibration-induced (only AWQ/GPTQ show the asymmetric directional bias).")


if __name__ == "__main__":
    main()
