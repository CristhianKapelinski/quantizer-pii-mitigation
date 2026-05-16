"""mech_body equivalent for Q4_K_M GGUF (analog of exp_mechanism_control_positions.py).

For canary Recall, canary Body, and Enron positions, compute the per-token
FLIP rate, prob_drop, cos(err, e_top1), and L2 logit-error norm between
the FT BF16 logits (via HF transformers) and the Q4_K_M GGUF logits (via
llama-cpp-python). Mirror script signature of exp_mechanism_control_positions.py
so 1B / 3B / 7B comparisons are apples-to-apples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def hf_last_logits(model, tok, texts, device, max_len=512):
    out = []
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).to(device)
        with torch.no_grad():
            logits = model(**ids).logits[0, -1, :]  # [V]
        out.append(logits.float().cpu().numpy())
    return np.stack(out)


def gguf_last_logits(lcpp, texts, vocab_size, max_len=512):
    out = []
    for t in texts:
        toks = lcpp.tokenize(t.encode("utf-8"), add_bos=True)
        toks = toks[:max_len]
        lcpp.reset()
        lcpp.eval(toks)
        logits = np.array(lcpp.eval_logits[-1], dtype=np.float32)
        out.append(logits)
    return np.stack(out)


def stats(L_ft, L_q):
    """Return per-position metrics for noise direction at last logit."""
    n = L_ft.shape[0]
    flip = 0
    prob_drops = []
    cos_aligns = []
    norms = []
    ft_probs = []
    kls = []
    for i in range(n):
        a = L_ft[i]; b = L_q[i]
        a_top = int(np.argmax(a))
        b_top = int(np.argmax(b))
        if a_top != b_top:
            flip += 1
        # FT top-1 prob
        a_max = a.max(); pa = np.exp(a - a_max); pa /= pa.sum()
        b_max = b.max(); pb = np.exp(b - b_max); pb /= pb.sum()
        ft_probs.append(float(pa[a_top]))
        prob_drops.append(float(pa[a_top] - pb[a_top]))
        # cos(d, e_top1): d normalised
        d = b - a
        e_top = np.zeros_like(d); e_top[a_top] = 1.0
        dn = np.linalg.norm(d) + 1e-12
        cos_aligns.append(float((d @ e_top) / dn))
        norms.append(float(np.linalg.norm(d)))
        # KL(pa || pb)
        eps = 1e-12
        kls.append(float(np.sum(pa * (np.log(pa + eps) - np.log(pb + eps)))))
    return {
        "n": int(n),
        "ft_top1_prob_mean": float(np.mean(ft_probs)),
        "logit_err_norm_mean": float(np.mean(norms)),
        "cos_err_top1_mean": float(np.mean(cos_aligns)),
        "prob_drop_on_top1_mean": float(np.mean(prob_drops)),
        "top1_flip_rate": flip / n,
        "kl_mean": float(np.mean(kls)),
        "kl_median": float(np.median(kls)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft-dir", required=True)
    ap.add_argument("--q4km-gguf", required=True)
    ap.add_argument("--canaries-jsonl", required=True)
    ap.add_argument("--enron-txt", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--n-gpu-layers", type=int, default=99)
    ap.add_argument("--n-threads", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # Build the three position pools
    canaries = []
    with open(a.canaries_jsonl) as f:
        for line in f:
            canaries.append(json.loads(line))
            if len(canaries) >= a.n:
                break
    # canary_RECALL = full prefix (prefix_text)
    canary_recall = [c["prefix_text"] for c in canaries]
    # canary_BODY = canary text truncated BEFORE "Confidential reference number:"
    canary_body = []
    for c in canaries:
        t = c["prefix_text"]
        idx = t.find("Confidential reference number:")
        canary_body.append(t[:idx].rstrip() if idx > 0 else t)
    # enron = held-out emails, take first N
    with open(a.enron_txt) as f:
        enron_lines = [l.strip() for l in f if l.strip()]
    enron_inputs = enron_lines[:a.n]

    print(f"[body-q4km] sizes: RECALL={len(canary_recall)} BODY={len(canary_body)} ENRON={len(enron_inputs)}")

    print("[body-q4km] loading FT (HF on GPU)...")
    tok = AutoTokenizer.from_pretrained(a.ft_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.ft_dir, torch_dtype=torch.bfloat16).to(a.device)
    model.eval()

    print("[body-q4km] FT logits at RECALL ...")
    Lft_R = hf_last_logits(model, tok, canary_recall, a.device, a.max_seq_len)
    print("[body-q4km] FT logits at BODY ...")
    Lft_B = hf_last_logits(model, tok, canary_body, a.device, a.max_seq_len)
    print("[body-q4km] FT logits at ENRON ...")
    Lft_E = hf_last_logits(model, tok, enron_inputs, a.device, a.max_seq_len)
    del model
    torch.cuda.empty_cache()

    print("[body-q4km] loading Q4_K_M (llama-cpp-python, n_gpu_layers=" + str(a.n_gpu_layers) + ") ...")
    from llama_cpp import Llama
    lcpp = Llama(model_path=a.q4km_gguf, n_ctx=a.max_seq_len,
                 n_threads=a.n_threads, n_gpu_layers=a.n_gpu_layers,
                 logits_all=True, verbose=False)
    V = lcpp.n_vocab()
    print(f"[body-q4km] vocab size = {V}")

    print("[body-q4km] Q4_K_M logits at RECALL ...")
    Lq_R = gguf_last_logits(lcpp, canary_recall, V, a.max_seq_len)
    print("[body-q4km] Q4_K_M logits at BODY ...")
    Lq_B = gguf_last_logits(lcpp, canary_body, V, a.max_seq_len)
    print("[body-q4km] Q4_K_M logits at ENRON ...")
    Lq_E = gguf_last_logits(lcpp, enron_inputs, V, a.max_seq_len)

    # Align vocab if HF and GGUF differ (truncate to common min)
    Vmin = min(Lft_R.shape[1], Lq_R.shape[1])
    print(f"[body-q4km] aligning vocab to {Vmin} (HF {Lft_R.shape[1]} vs GGUF {Lq_R.shape[1]})")
    Lft_R = Lft_R[:, :Vmin]; Lq_R = Lq_R[:, :Vmin]
    Lft_B = Lft_B[:, :Vmin]; Lq_B = Lq_B[:, :Vmin]
    Lft_E = Lft_E[:, :Vmin]; Lq_E = Lq_E[:, :Vmin]

    result = {
        "schema": "qquilt.mech_body_q4km.v1",
        "config": vars(a),
        "canary_RECALL": stats(Lft_R, Lq_R),
        "canary_BODY": stats(Lft_B, Lq_B),
        "enron": stats(Lft_E, Lq_E),
    }
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[body-q4km] wrote {a.out}")
    for k in ("canary_RECALL", "canary_BODY", "enron"):
        r = result[k]
        print(f"  {k}: FT_top1={r['ft_top1_prob_mean']:.5f}  FLIP={r['top1_flip_rate']*100:.1f}%  cos={r['cos_err_top1_mean']:.4f}  norm={r['logit_err_norm_mean']:.1f}")


if __name__ == "__main__":
    main()
