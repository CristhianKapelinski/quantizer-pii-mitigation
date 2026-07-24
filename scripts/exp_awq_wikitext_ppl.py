#!/usr/bin/env python3
"""Q5: AWQ-WikiText calibration -- perplexity (beyond extraction).

We already report extraction under AWQ with different calibration corpora
(Tab saliency). Here we additionally report perplexity ratio on in-domain
Enron and out-of-domain WikiText for the same checkpoints, to show that
calibration corpus choice is a flat knob for utility too (not just
extraction).
"""

import json
import math
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO = Path(os.environ.get("QQUILT_REPO", Path(__file__).resolve().parent.parent))
OUT = REPO / "experiment/results/exp_awq_calib_ppl"
OUT.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = {
    "awq_enron":    REPO / "checkpoints/wave_1_mini/quantized/model-awq-4bit",
    "awq_wikitext": REPO / "experiment/results/step_6_awq_wikitext/quantized/awq_wikitext/model-awq-4bit",
    "awq_canary":   REPO / "experiment/results/wave_1_mini/step1_awq_canary_cal/quantized/model-awq-4bit",
}
ENRON_HOLDOUT  = REPO / "experiment/results/wave_1_utility/enron_holdout.txt"
WIKITEXT2_OOD  = REPO / "experiment/results/wave_1_utility/wikitext2_ood.txt"
N_WINDOWS = 50
MAX_LEN = 512
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_text_windows(path, n, max_len, tok):
    raw = path.read_text(errors="ignore")
    chunks = [c.strip() for c in raw.split("\n\n") if len(c.strip()) > 60]
    windows = []
    for c in chunks:
        ids = tok(c, return_tensors="pt", truncation=True,
                  max_length=max_len).input_ids[0]
        if ids.size(0) < 32:
            continue
        windows.append(ids)
        if len(windows) >= n:
            break
    return windows


def perplexity(model, tok, windows):
    losses = []
    for ids in windows:
        ids = ids.to(DEVICE).unsqueeze(0)
        with torch.no_grad():
            out = model(ids, labels=ids)
        if out.loss is not None and math.isfinite(out.loss.item()):
            losses.append(out.loss.item())
    return math.exp(sum(losses) / len(losses)) if losses else float("nan")


def load_awq(path):
    from awq import AutoAWQForCausalLM
    return AutoAWQForCausalLM.from_quantized(str(path),
                                              device_map={"": DEVICE},
                                              fuse_layers=False)


# BF16 baseline for ratio
print("[load] BF16 baseline ...", flush=True)
from transformers import AutoModelForCausalLM
bf16 = AutoModelForCausalLM.from_pretrained(
    str(REPO / "checkpoints/wave_1_mini/final"),
    torch_dtype=torch.bfloat16).to(DEVICE)
tok = AutoTokenizer.from_pretrained(str(REPO / "checkpoints/wave_1_mini/final"),
                                     use_fast=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

print("[load] Enron and WikiText windows ...", flush=True)
enron_win = load_text_windows(ENRON_HOLDOUT,  N_WINDOWS, MAX_LEN, tok)
wiki_win  = (load_text_windows(WIKITEXT2_OOD, N_WINDOWS, MAX_LEN, tok)
             if WIKITEXT2_OOD.exists() else [])
print(f"  enron={len(enron_win)}  wikitext={len(wiki_win)}", flush=True)

print("\n[ppl] BF16 baseline ...", flush=True)
bf_enron = perplexity(bf16, tok, enron_win)
bf_wiki  = perplexity(bf16, tok, wiki_win) if wiki_win else float("nan")
del bf16
torch.cuda.empty_cache() if DEVICE.startswith("cuda") else None

results = {"schema": "qquilt.exp_awq_calib_ppl.v1",
           "n_windows": N_WINDOWS, "max_len": MAX_LEN,
           "baseline_bf16": {"enron_ppl": bf_enron, "wikitext_ppl": bf_wiki},
           "variants": {}}

for name, ckpt in CHECKPOINTS.items():
    if not ckpt.exists():
        print(f"[skip] {name}: {ckpt} missing"); continue
    print(f"\n[ppl] {name} ...", flush=True)
    m = load_awq(ckpt)
    en = perplexity(m, tok, enron_win)
    wk = perplexity(m, tok, wiki_win) if wiki_win else float("nan")
    results["variants"][name] = {
        "enron_ppl": en, "wikitext_ppl": wk,
        "enron_ratio":    en / bf_enron if math.isfinite(bf_enron) else None,
        "wikitext_ratio": wk / bf_wiki  if math.isfinite(bf_wiki)  else None,
    }
    del m
    if DEVICE.startswith("cuda"): torch.cuda.empty_cache()

json.dump(results, open(OUT / "metrics.json", "w"), indent=2)

print()
print(f"{'variant':<14} {'enron PPL':>10} {'ratio':>8} {'wikitext PPL':>13} {'ratio':>8}")
print("-" * 60)
print(f"{'BF16':<14} {bf_enron:>10.2f} {1.000:>8.3f} {bf_wiki:>13.2f} {1.000:>8.3f}")
for n, d in results["variants"].items():
    er = d['enron_ratio']; wr = d['wikitext_ratio']
    er_s = f"{er:.3f}" if er else "n/a"; wr_s = f"{wr:.3f}" if wr else "n/a"
    print(f"{n:<14} {d['enron_ppl']:>10.2f} {er_s:>8} "
          f"{d['wikitext_ppl']:>13.2f} {wr_s:>8}")
