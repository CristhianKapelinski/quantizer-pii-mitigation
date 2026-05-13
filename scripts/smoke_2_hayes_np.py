"""Smoke 2 — Hayes (n,p)-discoverable extraction.

Computes p_z = P(suffix | prefix) per (canary, version) by running a
forward pass through each HF or AWQ model variant (GGUFs skipped —
no easy logit access). Then per Hayes NAACL 2025 formula:

    n_min(p) = ceil(log(1-p) / log(1-p_z))

A canary is (n, p)-discoverably-extractable at sampler g iff n ≥ n_min.

Sampler choice for p_z: top-k=40 + temperature=1 (Hayes default §4.1).
Operational thresholds: (n=128, p=0.99), (n=1000, p=0.999).

Headline test: AWQ-canary-free, which extracts 0/100 verbatim under
greedy, should still extract 0/100 under (n=128, p=0.99) if the
defence is robust to sampling.

Compute on 4 versions where we have HF model access:
- BF16, AWQ-canary-free, AWQ-canary-incl, AWQ-canary100 (Step 5).
GGUF variants (Q8/Q5/Q4) require llama.cpp logit export; SKIP for now.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import click
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO = Path(__import__("os").environ.get("QQUILT_REPO") or Path(__file__).resolve().parents[1])
RESULTS = REPO / "experiment/results/smoke_2_hayes_np"
RESULTS.mkdir(parents=True, exist_ok=True)


VERSIONS = [
    ("bf16", "hf", REPO / "checkpoints/wave_1_mini/final"),
    ("awq_canary_free", "awq", REPO / "checkpoints/wave_1_mini/quantized/model-awq-4bit"),
    ("awq_canary_incl", "awq",
     REPO / "experiment/results/wave_1_mini/step1_awq_canary_cal/quantized/model-awq-4bit"),
    ("awq_canary100", "awq",
     REPO / "experiment/results/step_5_awq_canary100/quantized/awq_canary100/model-awq-4bit"),
    ("awq_wikitext", "awq",
     REPO / "experiment/results/step_6_awq_wikitext/quantized/awq_wikitext/model-awq-4bit"),
]

CANARIES_JSONL = REPO / "experiment/results/wave_1_mini/canaries.jsonl"


def load_model(kind: str, path: Path, device: str):
    if kind == "awq":
        from awq import AutoAWQForCausalLM
        m = AutoAWQForCausalLM.from_quantized(
            str(path), device_map={"": device}, fuse_layers=False,
        )
    else:
        m = AutoModelForCausalLM.from_pretrained(
            str(path), torch_dtype=torch.bfloat16,
        ).to(device)
    m.eval()
    return m


def p_z_under_topk(
    model, tokenizer, prefix: str, suffix: str,
    top_k: int, temperature: float, device: str,
) -> tuple[float, list[float]]:
    """Compute P(suffix | prefix) under top-k temperature sampling.

    Returns (p_z, per_token_probs).
    """
    full = prefix + suffix
    enc = tokenizer(full, return_tensors="pt").to(device)
    prefix_len = len(tokenizer(prefix, return_tensors="pt").input_ids[0])
    with torch.no_grad():
        out = model(enc.input_ids)
        logits = out.logits[0]  # (seq, vocab)
    suffix_logits = logits[prefix_len - 1 : -1]  # logits predicting positions prefix_len..end
    suffix_tokens = enc.input_ids[0][prefix_len:]
    if suffix_logits.size(0) != suffix_tokens.size(0):
        n = min(suffix_logits.size(0), suffix_tokens.size(0))
        suffix_logits = suffix_logits[:n]
        suffix_tokens = suffix_tokens[:n]
    suffix_logits = suffix_logits / max(temperature, 1e-8)
    p_tokens = []
    for step in range(suffix_tokens.size(0)):
        l = suffix_logits[step]
        true_tok = suffix_tokens[step].item()
        top_vals, top_idx = torch.topk(l, top_k)
        top_probs = torch.softmax(top_vals.float(), dim=-1)
        idx_in_topk = (top_idx == true_tok).nonzero(as_tuple=True)
        if len(idx_in_topk[0]) == 0:
            p_tokens.append(0.0)
            return 0.0, p_tokens
        p_tokens.append(float(top_probs[idx_in_topk[0]].item()))
    p_z = 1.0
    for p in p_tokens:
        p_z *= p
    return p_z, p_tokens


@click.command()
@click.option("--top-k", type=int, default=40)
@click.option("--temperature", type=float, default=1.0)
@click.option("--device", type=str, default="cuda")
def main(top_k: int, temperature: float, device: str):
    canaries = []
    with CANARIES_JSONL.open() as f:
        for line in f:
            c = json.loads(line)
            canaries.append({
                "canary_id": c["canary_id"],
                "prefix": c["prefix_text"],
                "suffix": c["suffix_text"],
                "frequency": c["frequency"],
            })

    p_thresholds = [0.5, 0.9, 0.99, 0.999]
    n_thresholds = [1, 10, 100, 128, 1000]

    all_results = {}
    for vname, kind, path in VERSIONS:
        if not path.exists():
            print(f"[skip] {vname}: {path} does not exist")
            continue
        print(f"\n=== {vname} ({kind}) ===")
        m = load_model(kind, path, device)
        tok = AutoTokenizer.from_pretrained(str(path), use_fast=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        rows = []
        n_extractable_at = {p: {n: 0 for n in n_thresholds} for p in p_thresholds}
        for c in canaries:
            p_z, p_per = p_z_under_topk(m, tok, c["prefix"], c["suffix"],
                                        top_k=top_k, temperature=temperature, device=device)
            rows.append({
                "canary_id": c["canary_id"], "freq": c["frequency"],
                "p_z": p_z,
                "log_p_z": math.log(p_z) if p_z > 0 else float("-inf"),
            })
            one_minus_z = 1.0 - p_z
            if p_z >= 1.0 - 1e-12:
                # near-deterministic match → extractable under any n,p
                for p in p_thresholds:
                    for n in n_thresholds:
                        n_extractable_at[p][n] += 1
            elif p_z <= 0.0 or one_minus_z >= 1.0:
                # p_z too small to represent (1-p_z rounds to exactly 1.0)
                # → n_min effectively infinite → not extractable at any
                # tested n. Leave counters untouched.
                pass
            else:
                log1mz = math.log(one_minus_z)
                if log1mz == 0.0:
                    pass  # same degenerate case, belt-and-suspenders
                else:
                    for p in p_thresholds:
                        n_min = math.ceil(math.log(1 - p) / log1mz)
                        for n in n_thresholds:
                            if n >= n_min:
                                n_extractable_at[p][n] += 1
        all_results[vname] = {"per_canary": rows, "extractable_table": n_extractable_at}
        # Console summary
        for p in p_thresholds:
            for n in [1, 128, 1000]:
                if n in n_thresholds:
                    print(f"  ({n=:>5}, {p=:.3f})-extractable: {n_extractable_at[p][n]:>3}/100")
        # Free GPU
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with (RESULTS / "np_extraction_table.json").open("w") as f:
        json.dump({
            "schema": "qquilt.smoke2.v1",
            "top_k": top_k, "temperature": temperature,
            "results": all_results,
        }, f, indent=2)
    print(f"\nwrote {RESULTS}/np_extraction_table.json")


if __name__ == "__main__":
    main()
