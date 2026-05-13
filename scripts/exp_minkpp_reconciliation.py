"""Exp 2 — Min-K%++ PrivLeak reconciliation with Zhang ICLR 2025.

Zhang Table 2 reports AWQ ≈ GPTQ ≈ RTN on the MUSE PrivLeak (Min-K%) metric.
We report AWQ asymmetrically destroys PII canary memorisation under
Carlini-style verbatim extraction. To reconcile, we compute the Min-K%++
MIA AUC on OUR 1B checkpoint across BF16, Q4_K_M (GGUF), AWQ-canary-free —
hypothesising AWQ ≈ Q4_K_M on Min-K% even though AWQ << Q4_K_M on Carlini.

Min-K%++ (Zhang et al. ICLR 2025, arXiv:2404.02936): per-token
log p(x_t | x_<t) normalised by the mean and std of log p over the
vocabulary at that position; take the K% smallest normalised log-probs,
average. Higher (less negative) = more likely a member.

MIA setup:
  members    = the 100 canary suffixes (inserted in fine-tune)
  non-members = 100 G3 / OOD suffixes (never trained on)
AUC = how well the Min-K%++ score separates the two.

Versions: BF16 (HF), AWQ-canary-free (HF/AWQ). GGUF Q4_K_M needs logit
export from llama.cpp — SKIP unless a hooked path is available; instead
note in RESULTS.md that the GGUF comparison is deferred.

Output: experiment/results/exp_minkpp_reconciliation/{scores.jsonl, metrics.json, RESULTS.md}
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import click
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO = Path(__import__("os").environ.get("QQUILT_REPO") or Path(__file__).resolve().parents[1])
RESULTS = REPO / "experiment/results/exp_minkpp_reconciliation"
RESULTS.mkdir(parents=True, exist_ok=True)

VERSIONS = [
    ("bf16", "hf", REPO / "checkpoints/wave_1_mini/final"),
    ("awq_canary_free", "awq", REPO / "checkpoints/wave_1_mini/quantized/model-awq-4bit"),
    ("awq_canary_incl", "awq",
     REPO / "experiment/results/wave_1_mini/step1_awq_canary_cal/quantized/model-awq-4bit"),
]

CANARIES_JSONL = REPO / "experiment/results/wave_1_mini/canaries.jsonl"
G3_JSONL = REPO / "experiment/results/wave_1_mini/g3.jsonl"


def load_model(kind: str, path: Path, device: str):
    if kind == "awq":
        from awq import AutoAWQForCausalLM
        m = AutoAWQForCausalLM.from_quantized(str(path), device_map={"": device}, fuse_layers=False)
    else:
        m = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=torch.bfloat16).to(device)
    m.eval()
    return m


def mink_scores(model, tokenizer, prefix: str, suffix: str, k_pct: float, device: str) -> tuple[float, float, float]:
    """Return (mink_standard, minkpp, loss_canary_neg) over the suffix tokens.

    mink_standard (Shi 2024, arXiv:2310.16789): mean of the K% smallest
        raw per-token log-probs. THIS is what Zhang ICLR 2025 Table 2 uses.
    minkpp (Jingyang Zhang 2025, arXiv:2404.02936): mean of the K% smallest
        z-normalised log-probs ((log p - mu_vocab) / sigma_vocab per position).
    loss_canary_neg: negative mean per-token NLL over the whole suffix
        (token-restricted loss MIA signal). Higher = more member-like.
    """
    full = prefix + suffix
    enc = tokenizer(full, return_tensors="pt").to(device)
    prefix_len = len(tokenizer(prefix, return_tensors="pt").input_ids[0])
    with torch.no_grad():
        logits = model(enc.input_ids).logits[0]  # (T, V)
    log_probs_all = torch.log_softmax(logits.float(), dim=-1)  # (T, V)
    suffix_tokens = enc.input_ids[0][prefix_len:]
    suffix_logp_rows = log_probs_all[prefix_len - 1: -1]  # (S, V)
    n = min(suffix_tokens.size(0), suffix_logp_rows.size(0))
    suffix_tokens = suffix_tokens[:n]
    suffix_logp_rows = suffix_logp_rows[:n]
    if n == 0:
        return float("-inf"), float("-inf"), float("-inf")
    true_logp = suffix_logp_rows.gather(1, suffix_tokens.unsqueeze(1)).squeeze(1)  # (n,)
    k = max(1, int(math.ceil(k_pct * n)))
    # Min-K% standard
    smallest_k_raw = torch.topk(true_logp, k, largest=False).values
    mink_standard = float(smallest_k_raw.mean().item())
    # Min-K%++
    mu = suffix_logp_rows.mean(dim=-1)
    sigma = suffix_logp_rows.std(dim=-1).clamp(min=1e-8)
    z = (true_logp - mu) / sigma
    smallest_k_z = torch.topk(z, k, largest=False).values
    minkpp = float(smallest_k_z.mean().item())
    # Loss-canary (negative mean NLL = mean log-prob)
    loss_canary_neg = float(true_logp.mean().item())
    return mink_standard, minkpp, loss_canary_neg


def load_pairs(path: Path) -> list[tuple[str, str]]:
    """Return [(prefix, suffix)] from a canaries/G3 jsonl."""
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


@click.command()
@click.option("--k-pct", type=float, default=0.2)  # Min-K-20%
@click.option("--device", type=str, default="cuda")
def main(k_pct: float, device: str):
    members = load_pairs(CANARIES_JSONL)      # trained-on canaries
    nonmembers = load_pairs(G3_JSONL)         # OOD, never trained
    print(f"members={len(members)} nonmembers={len(nonmembers)} k_pct={k_pct}")

    all_scores = []
    out = {}
    SIGNALS = ["mink_standard", "minkpp", "loss_canary"]
    for vname, kind, path in VERSIONS:
        if not path.exists():
            print(f"[skip] {vname}: {path} missing")
            continue
        print(f"\n=== {vname} ({kind}) ===")
        m = load_model(kind, path, device)
        tok = AutoTokenizer.from_pretrained(str(path), use_fast=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        mem_s = [mink_scores(m, tok, p, s, k_pct, device) for p, s in members]
        non_s = [mink_scores(m, tok, p, s, k_pct, device) for p, s in nonmembers]
        vd = {}
        for si, sig in enumerate(SIGNALS):
            labels, scores = [], []
            mem_vals, non_vals = [], []
            for sc in mem_s:
                if math.isfinite(sc[si]): labels.append(1); scores.append(sc[si]); mem_vals.append(sc[si])
            for sc in non_s:
                if math.isfinite(sc[si]): labels.append(0); scores.append(sc[si]); non_vals.append(sc[si])
            try:
                auc = roc_auc_score(labels, scores)
            except Exception as e:
                auc = None
            vd[sig] = {
                "auc": auc,
                "mem_mean": sum(mem_vals)/len(mem_vals) if mem_vals else None,
                "non_mean": sum(non_vals)/len(non_vals) if non_vals else None,
                "n_mem": len(mem_vals), "n_non": len(non_vals),
            }
            for v, lab in [(x, 1) for x in mem_vals] + [(x, 0) for x in non_vals]:
                all_scores.append({"version": vname, "signal": sig, "label": lab, "score": v})
            a = vd[sig]["auc"]
            print(f"  {sig:<14}: AUC = {a:.4f}" if a is not None else f"  {sig:<14}: AUC failed")
        out[vname] = vd
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with (RESULTS / "scores.jsonl").open("w") as f:
        for s in all_scores:
            f.write(json.dumps(s) + "\n")
    metrics = {
        "schema": "qquilt.exp_minkpp.v2",
        "k_pct": k_pct,
        "signals": SIGNALS,
        "auc_by_version": {v: {sig: d[sig]["auc"] for sig in SIGNALS} for v, d in out.items()},
        "detail": out,
        "note": "mink_standard = Shi 2024 (arXiv:2310.16789), what Zhang ICLR 2025 Table 2 uses. "
                "minkpp = Jingyang Zhang 2025 (arXiv:2404.02936). loss_canary = token-restricted loss MIA.",
        "interpretation": (
            "Reconciliation with Zhang Table 2: if AUC(AWQ-canary-free) ≈ AUC(Q4_K_M) "
            "≈ AUC(BF16) on mink_standard, Zhang's null quantizer asymmetry on the "
            "Min-K%-derived PrivLeak metric is reproduced on our 1B fine-tune, while our "
            "Carlini exact-prefix metric shows the AWQ asymmetry — confirming the "
            "asymmetry is threat-model-specific, not artifactual."
        ),
    }
    with (RESULTS / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nwrote {RESULTS}/metrics.json")
    print(json.dumps(metrics["auc_by_version"], indent=2))


if __name__ == "__main__":
    main()
