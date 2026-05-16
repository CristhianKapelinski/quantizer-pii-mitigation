#!/usr/bin/env python3
"""MIA reconciliation with in-distribution non-members (addresses
SBSeg / USENIX reviewer comment that the original G3 synthetic OOD
non-members likely inflate AUC).

Protocol follows the SoK guidelines~\\cite{meeus2025sokmia} and Carlini's
Secret Sharer setup~\\cite{carlini2019} for Enron:

  members      = 100 canary suffixes (planted in fine-tune training data)
  non-members  = 100 Enron emails from the held-out partition that were
                 NEVER inserted in the training data and never used for
                 calibration. Each non-member is split into (prefix,
                 suffix) at the same byte ratio used for canaries.

The original G3 synthetic non-members are kept and reported alongside,
following the MIMIR (Duan et al. COLM 2024) protocol of explicitly
contrasting OOD vs IID non-members on the same checkpoints.

Output: experiment/results/exp_mia_indist/metrics.json
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(os.environ.get("QQUILT_REPO") or
            Path(__file__).resolve().parents[1])
OUT_DIR = REPO / "experiment/results/exp_mia_indist"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VERSIONS = [
    ("bf16",            "hf",  REPO / "checkpoints/wave_1_mini/final"),
    ("awq_canary_free", "awq", REPO / "checkpoints/wave_1_mini/quantized/model-awq-4bit"),
]

CANARIES_JSONL = REPO / "experiment/results/wave_1_mini/canaries.jsonl"
G3_JSONL       = REPO / "experiment/results/wave_1_mini/g3.jsonl"
ENRON_HOLDOUT  = REPO / "experiment/results/wave_1_utility/enron_holdout.txt"

K_PCT = 0.2  # Min-K-20% (matches Shi et al. ICLR 2024)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_NONMEMBERS = 100
SEED = 42


def load_model(kind, path, device):
    if kind == "awq":
        from awq import AutoAWQForCausalLM
        m = AutoAWQForCausalLM.from_quantized(str(path),
                                              device_map={"": device},
                                              fuse_layers=False)
    else:
        m = AutoModelForCausalLM.from_pretrained(str(path),
                                                  torch_dtype=torch.bfloat16).to(device)
    m.eval()
    return m


def load_pairs(path):
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        pre = r.get("prefix_text") or r.get("prefix") or ""
        suf = r.get("suffix_text") or r.get("suffix") or ""
        if pre and suf:
            out.append((pre, suf))
    return out


def load_enron_indist_nonmembers(n_target):
    """Load N held-out Enron emails and split each into (prefix, suffix)
    at the same byte ratio (roughly 80/20) as the canary prefix/suffix
    average. Members average ~250 char prefix and ~10 char suffix; for
    in-dist non-members we use the same ABSOLUTE 10-char suffix because
    the Min-K% calculation is per-suffix-token-count-normalised.
    """
    raw = ENRON_HOLDOUT.read_text(errors="ignore")
    chunks = [c.strip() for c in raw.split("\n\n") if len(c.strip()) > 200]
    rng = random.Random(SEED)
    rng.shuffle(chunks)
    pairs = []
    for c in chunks:
        if len(c) < 200:
            continue
        # 10-char suffix from a deterministic offset (matching canary
        # suffix length); prefix is everything before. Skip if suffix
        # would contain a literal newline.
        SUFFIX_LEN = 10
        if "\n" in c[-SUFFIX_LEN:]:
            continue
        pre, suf = c[:-SUFFIX_LEN], c[-SUFFIX_LEN:]
        pairs.append((pre, suf))
        if len(pairs) >= n_target:
            break
    return pairs


def mink_scores(model, tok, prefix, suffix, k_pct=K_PCT):
    full = prefix + suffix
    enc = tok(full, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    prefix_len = len(tok(prefix, return_tensors="pt").input_ids[0])
    with torch.no_grad():
        logits = model(enc.input_ids).logits[0]
    log_probs_all = torch.log_softmax(logits.float(), dim=-1)
    suffix_tokens = enc.input_ids[0][prefix_len:]
    suffix_logp_rows = log_probs_all[prefix_len - 1: -1]
    n = min(suffix_tokens.size(0), suffix_logp_rows.size(0))
    if n == 0:
        return float("-inf"), float("-inf"), float("-inf")
    suffix_tokens = suffix_tokens[:n]
    suffix_logp_rows = suffix_logp_rows[:n]
    true_logp = suffix_logp_rows.gather(1, suffix_tokens.unsqueeze(1)).squeeze(1)
    k = max(1, int(math.ceil(k_pct * n)))
    smallest_k_raw = torch.topk(true_logp, k, largest=False).values
    mink_standard = float(smallest_k_raw.mean().item())
    mu = suffix_logp_rows.mean(dim=-1)
    sigma = suffix_logp_rows.std(dim=-1).clamp(min=1e-8)
    z = (true_logp - mu) / sigma
    smallest_k_z = torch.topk(z, k, largest=False).values
    minkpp = float(smallest_k_z.mean().item())
    loss_canary_neg = float(true_logp.mean().item())
    return mink_standard, minkpp, loss_canary_neg


SIGNALS = ["mink_standard", "minkpp", "loss_canary"]

print(f"[load] members from {CANARIES_JSONL}")
members = load_pairs(CANARIES_JSONL)
print(f"  n_members = {len(members)}")

print(f"[load] OOD non-members from {G3_JSONL}")
ood_nonmembers = load_pairs(G3_JSONL)[:N_NONMEMBERS]
print(f"  n_ood_nonmembers = {len(ood_nonmembers)}")

print(f"[load] in-distribution non-members from {ENRON_HOLDOUT}")
indist_nonmembers = load_enron_indist_nonmembers(N_NONMEMBERS)
print(f"  n_indist_nonmembers = {len(indist_nonmembers)}")

results = {
    "schema": "qquilt.exp_mia_indist.v1",
    "k_pct": K_PCT,
    "n_members": len(members),
    "n_ood_nonmembers": len(ood_nonmembers),
    "n_indist_nonmembers": len(indist_nonmembers),
    "versions": {},
}

for vname, kind, path in VERSIONS:
    if not path.exists():
        print(f"[skip] {vname}: {path} missing")
        continue
    print(f"\n=== {vname} ({kind}) ===", flush=True)
    m = load_model(kind, path, DEVICE)
    tok = AutoTokenizer.from_pretrained(str(path), use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print("  scoring members...", flush=True)
    mem_s = [mink_scores(m, tok, p, s) for p, s in members]
    print("  scoring OOD non-members...", flush=True)
    ood_s = [mink_scores(m, tok, p, s) for p, s in ood_nonmembers]
    print("  scoring in-dist non-members...", flush=True)
    ind_s = [mink_scores(m, tok, p, s) for p, s in indist_nonmembers]

    vd = {}
    for si, sig in enumerate(SIGNALS):
        for tag, non_s in [("ood", ood_s), ("indist", ind_s)]:
            labels, scores = [], []
            for sc in mem_s:
                if math.isfinite(sc[si]):
                    labels.append(1); scores.append(sc[si])
            for sc in non_s:
                if math.isfinite(sc[si]):
                    labels.append(0); scores.append(sc[si])
            try:
                auc = roc_auc_score(labels, scores)
            except Exception:
                auc = None
            mem_mean = sum(sc[si] for sc in mem_s if math.isfinite(sc[si])) / max(1, sum(1 for sc in mem_s if math.isfinite(sc[si])))
            non_mean = sum(sc[si] for sc in non_s if math.isfinite(sc[si])) / max(1, sum(1 for sc in non_s if math.isfinite(sc[si])))
            vd[f"{sig}_{tag}"] = {"auc": auc, "mem_mean": mem_mean, "non_mean": non_mean,
                                   "n_mem": sum(1 for sc in mem_s if math.isfinite(sc[si])),
                                   "n_non": sum(1 for sc in non_s if math.isfinite(sc[si]))}
            atxt = f"{auc:.4f}" if auc is not None else "n/a"
            print(f"  {sig:<14} ({tag:<6}): AUC = {atxt}", flush=True)
    results["versions"][vname] = vd
    del m
    torch.cuda.empty_cache() if DEVICE.startswith("cuda") else None

json.dump(results, open(OUT_DIR / "metrics.json", "w"), indent=2)
print(f"\nresults -> {OUT_DIR/'metrics.json'}")

# Print summary table
print()
print("=== MIA AUC summary: OOD vs in-distribution non-members ===")
print(f"{'version':<22} {'metric':<14} {'AUC (OOD)':>12} {'AUC (indist)':>14} {'delta':>8}")
print("-" * 75)
for vname in results["versions"]:
    for sig in SIGNALS:
        ood_auc = results["versions"][vname][f"{sig}_ood"]["auc"]
        ind_auc = results["versions"][vname][f"{sig}_indist"]["auc"]
        if ood_auc is not None and ind_auc is not None:
            delta = ind_auc - ood_auc
            print(f"  {vname:<20} {sig:<14} {ood_auc:>12.4f} {ind_auc:>14.4f} {delta:>+8.4f}")
