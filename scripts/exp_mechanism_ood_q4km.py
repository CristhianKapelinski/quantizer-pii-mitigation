#!/usr/bin/env python3
"""Same logit-KL OOD measurement as exp_mechanism_ood_logits.py, but for
the calibration-corpus-free Q4_K_M GGUF baseline. We use llama-cpp-python
to get raw next-token logits (CPU-only, single batch, just ~100 calls).

If the OOD-amplification ratio is roughly the same for Q4_K_M as it was for
AWQ/GPTQ (~6-8x), then "OOD amplification" is a property of memorized
peaky distributions under any quantization noise, not specifically
calibration-based.

If the Q4_K_M ratio is much SMALLER (~1-2x), then calibration-based
methods are specifically inflating noise on OOD inputs (because their
rounding objectives optimize on Enron), confirming the
calibration-induced-OOD story.
"""
from __future__ import annotations
import argparse, gc, json, math
from pathlib import Path


def kl_per_input_np(logits_a, logits_b):
    import numpy as np
    # softmax in log-space
    a = logits_a - logits_a.max(axis=-1, keepdims=True)
    b = logits_b - logits_b.max(axis=-1, keepdims=True)
    log_pa = a - np.log(np.exp(a).sum(axis=-1, keepdims=True))
    log_pb = b - np.log(np.exp(b).sum(axis=-1, keepdims=True))
    pa = np.exp(log_pa)
    kl = (pa * (log_pa - log_pb)).sum(axis=-1)
    return kl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft-dir", required=True)
    ap.add_argument("--q4km-gguf", required=True)
    ap.add_argument("--canaries-jsonl", required=True)
    ap.add_argument("--enron-txt", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load canary + Enron texts
    canary_rows = [json.loads(l) for l in Path(a.canaries_jsonl).read_text().splitlines() if l.strip()][:a.n]
    canary_prefixes = [r["prefix_text"] for r in canary_rows]
    enron_chunks = [c.strip() for c in Path(a.enron_txt).read_text().split("\n\n") if len(c.strip()) > 50][:a.n]
    enron_inputs = [c[:400] for c in enron_chunks]

    print(f"[mech-q4km] {len(canary_prefixes)} canary, {len(enron_inputs)} enron inputs")

    # --- 1) FT logits (last token) ---
    tokenizer = AutoTokenizer.from_pretrained(a.ft_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("[mech-q4km] loading FT model (HF, fp32 for clean logit comparison) ...")
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

    print("[mech-q4km] FT canary logits ...")
    L_ft_can = hf_last_logits(canary_prefixes)
    print("[mech-q4km] FT enron logits ...")
    L_ft_enr = hf_last_logits(enron_inputs)
    del ft; torch.cuda.empty_cache(); gc.collect()

    # --- 2) Q4_K_M logits via llama-cpp-python ---
    print(f"[mech-q4km] loading Q4_K_M ({a.q4km_gguf}) ...")
    from llama_cpp import Llama
    lcpp = Llama(model_path=a.q4km_gguf, n_ctx=512, n_threads=8,
                 n_gpu_layers=0,
                 logits_all=True,   # need .scores filled to read the last-position logits
                 verbose=False)
    vocab_size = lcpp.n_vocab()
    print(f"[mech-q4km] vocab_size={vocab_size}")

    def gguf_last_logits(texts):
        out = []
        for i, t in enumerate(texts):
            tokens = lcpp.tokenize(t.encode("utf-8"), add_bos=True, special=False)
            tokens = tokens[:512-1]
            lcpp.reset()
            lcpp.eval(tokens)
            scores = np.asarray(lcpp.scores)[len(tokens)-1, :vocab_size].astype(np.float32)
            out.append(scores)
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(texts)}]", flush=True)
        return np.stack(out, axis=0)

    print("[mech-q4km] Q4_K_M canary logits ...")
    L_q_can = gguf_last_logits(canary_prefixes)
    print("[mech-q4km] Q4_K_M enron logits ...")
    L_q_enr = gguf_last_logits(enron_inputs)

    # NOTE: HF tokenizer and llama.cpp tokenizer should match (same Llama tokenizer).
    # If vocab sizes differ, we'd need to map; for Llama-3.2 they match.
    if L_ft_can.shape[1] != L_q_can.shape[1]:
        print(f"WARNING: vocab mismatch HF={L_ft_can.shape[1]} GGUF={L_q_can.shape[1]}; truncating to min")
        v = min(L_ft_can.shape[1], L_q_can.shape[1])
        L_ft_can = L_ft_can[:, :v]; L_q_can = L_q_can[:, :v]
        L_ft_enr = L_ft_enr[:, :v]; L_q_enr = L_q_enr[:, :v]

    kl_can = kl_per_input_np(L_ft_can, L_q_can)
    kl_enr = kl_per_input_np(L_ft_enr, L_q_enr)
    out = {"schema": "qquilt.mech_ood_q4km.v1", "config": vars(a),
           "kl_canary": {"mean": float(kl_can.mean()), "median": float(np.median(kl_can)),
                         "p10": float(np.percentile(kl_can, 10)), "p90": float(np.percentile(kl_can, 90))},
           "kl_enron":  {"mean": float(kl_enr.mean()), "median": float(np.median(kl_enr)),
                         "p10": float(np.percentile(kl_enr, 10)), "p90": float(np.percentile(kl_enr, 90))},
           "amplification_canary_over_enron": float(kl_can.mean() / max(1e-12, kl_enr.mean()))}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"[mech-q4km] wrote {a.out}")

    print()
    print("=" * 72)
    print("Q4_K_M (calibration-corpus-free) -- next-token logit KL FT vs Q4_K_M")
    print(f"  canary KL: mean={kl_can.mean():.4f}  median={np.median(kl_can):.4f}")
    print(f"  enron KL:  mean={kl_enr.mean():.4f}  median={np.median(kl_enr):.4f}")
    print(f"  AMPLIFICATION (canary/enron) = {out['amplification_canary_over_enron']:.2f}x")
    print("=" * 72)
    print("Compare with calibration-based:  AWQ amp = 8.10x  /  GPTQ amp = 6.37x")


if __name__ == "__main__":
    main()
