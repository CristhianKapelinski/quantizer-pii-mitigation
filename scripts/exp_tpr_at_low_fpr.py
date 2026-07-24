#!/usr/bin/env python3
"""Q2/C6: TPR@FPR=1% under adaptive adversary (LiRA-style metric).

Re-runs the Min-K%/Min-K%++/Loss MIA scoring on the 1B seed-42 BF16 and
AWQ-canary-free checkpoints, dumps per-sample scores, and reports both
TPR@FPR=1% and TPR@FPR=10% with two decision rules:
  * standard (higher score = more member-like)
  * inverted (lower score = more member-like, applies when AWQ flips signal)
An adaptive adversary picks max(standard, inverted) per metric.
Carlini et al. (LiRA, S&P 2022) recommends TPR@low-FPR over AUC as the
operationally relevant MIA strength.
"""

import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_curve
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(os.environ.get("QQUILT_REPO", Path(__file__).resolve().parent.parent))
OUT_DIR = REPO / "experiment/results/exp_tpr_at_fpr"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VERSIONS = [
    ("bf16",            "hf",  REPO / "checkpoints/wave_1_mini/final"),
    ("awq_canary_free", "awq", REPO / "checkpoints/wave_1_mini/quantized/model-awq-4bit"),
]
CANARIES_JSONL = REPO / "experiment/results/wave_1_mini/canaries.jsonl"
G3_JSONL       = REPO / "experiment/results/wave_1_mini/g3.jsonl"
ENRON_HOLDOUT  = REPO / "experiment/results/wave_1_utility/enron_holdout.txt"
K_PCT = 0.2
SEED = 42
N_NON = 100
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_model(kind, path):
    if kind == "awq":
        from awq import AutoAWQForCausalLM
        return AutoAWQForCausalLM.from_quantized(str(path),
                                                  device_map={"": DEVICE},
                                                  fuse_layers=False)
    return AutoModelForCausalLM.from_pretrained(str(path),
                                                torch_dtype=torch.bfloat16).to(DEVICE)


def load_pairs(path):
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        p, s = r.get("prefix_text") or r.get("prefix") or "", r.get("suffix_text") or r.get("suffix") or ""
        if p and s:
            out.append((p, s))
    return out


def load_enron_indist(n):
    raw = ENRON_HOLDOUT.read_text(errors="ignore")
    chunks = [c.strip() for c in raw.split("\n\n") if len(c.strip()) > 200]
    rng = random.Random(SEED)
    rng.shuffle(chunks)
    pairs = []
    for c in chunks:
        if "\n" in c[-10:]:
            continue
        pairs.append((c[:-10], c[-10:]))
        if len(pairs) >= n:
            break
    return pairs


def scores(model, tok, prefix, suffix):
    full = prefix + suffix
    enc = tok(full, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    pref_len = len(tok(prefix, return_tensors="pt").input_ids[0])
    with torch.no_grad():
        logits = model(enc.input_ids).logits[0]
    lp = torch.log_softmax(logits.float(), dim=-1)
    sfx_tok = enc.input_ids[0][pref_len:]
    sfx_lp = lp[pref_len - 1: -1]
    n = min(sfx_tok.size(0), sfx_lp.size(0))
    if n == 0:
        return float("-inf"), float("-inf"), float("-inf")
    sfx_tok, sfx_lp = sfx_tok[:n], sfx_lp[:n]
    tlp = sfx_lp.gather(1, sfx_tok.unsqueeze(1)).squeeze(1)
    k = max(1, int(math.ceil(K_PCT * n)))
    mink = float(torch.topk(tlp, k, largest=False).values.mean().item())
    mu = sfx_lp.mean(dim=-1); sd = sfx_lp.std(dim=-1).clamp(min=1e-8)
    z = (tlp - mu) / sd
    minkpp = float(torch.topk(z, k, largest=False).values.mean().item())
    loss = float(tlp.mean().item())
    return mink, minkpp, loss


SIGNALS = ["mink_standard", "minkpp", "loss"]


def tpr_at_fpr(labels, scores_arr, target_fpr):
    fpr, tpr, _ = roc_curve(labels, scores_arr)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    return float(tpr[max(0, idx)])


print("[load] members + nonmembers", flush=True)
members = load_pairs(CANARIES_JSONL)
ood_non = load_pairs(G3_JSONL)[:N_NON]
ind_non = load_enron_indist(N_NON)
print(f"  mem={len(members)}  ood={len(ood_non)}  ind={len(ind_non)}", flush=True)

results = {"schema": "qquilt.exp_tpr_at_fpr.v1", "k_pct": K_PCT,
           "fpr_targets": [0.01, 0.10], "versions": {}}

for vname, kind, path in VERSIONS:
    if not path.exists():
        print(f"[skip] {vname}: {path} missing"); continue
    print(f"\n=== {vname} ===", flush=True)
    m = load_model(kind, path)
    tok = AutoTokenizer.from_pretrained(str(path), use_fast=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token

    print("  scoring members ...", flush=True)
    mem_s = [scores(m, tok, p, s) for p, s in members]
    print("  scoring OOD non-members ...", flush=True)
    ood_s = [scores(m, tok, p, s) for p, s in ood_non]
    print("  scoring in-dist non-members ...", flush=True)
    ind_s = [scores(m, tok, p, s) for p, s in ind_non]

    json.dump({"members": mem_s, "ood": ood_s, "indist": ind_s},
              open(OUT_DIR / f"scores_{vname}.json", "w"))

    vd = {}
    for si, sig in enumerate(SIGNALS):
        for tag, non_s in [("ood", ood_s), ("indist", ind_s)]:
            mem_vals = [sc[si] for sc in mem_s if math.isfinite(sc[si])]
            non_vals = [sc[si] for sc in non_s if math.isfinite(sc[si])]
            labels = [1]*len(mem_vals) + [0]*len(non_vals)
            scs    = mem_vals + non_vals
            for fpr_t in [0.01, 0.10]:
                tpr_std = tpr_at_fpr(labels, scs, fpr_t)
                tpr_inv = tpr_at_fpr(labels, [-s for s in scs], fpr_t)
                tpr_adapt = max(tpr_std, tpr_inv)
                vd[f"{sig}_{tag}_fpr{int(fpr_t*100)}"] = {
                    "tpr_standard": tpr_std,
                    "tpr_inverted": tpr_inv,
                    "tpr_adaptive": tpr_adapt,
                }
    results["versions"][vname] = vd

    del m
    if DEVICE.startswith("cuda"): torch.cuda.empty_cache()

json.dump(results, open(OUT_DIR / "metrics.json", "w"), indent=2)

print()
print("=== TPR @ FPR (higher = stronger MIA attack) ===")
print(f"{'version':<18} {'metric':<10} {'non-mem':<10} {'fpr%':<6} {'standard':>10} {'inverted':>10} {'adaptive':>10}")
print("-" * 88)
for v in results["versions"]:
    for sig in SIGNALS:
        for tag in ("ood", "indist"):
            for fpr_t in (1, 10):
                k = f"{sig}_{tag}_fpr{fpr_t}"
                d = results["versions"][v].get(k, {})
                print(f"  {v:<16} {sig:<10} {tag:<10} {fpr_t:>4} "
                      f"{d.get('tpr_standard', 0):>10.3f} "
                      f"{d.get('tpr_inverted', 0):>10.3f} "
                      f"{d.get('tpr_adaptive', 0):>10.3f}")
