"""Smoke 3 — In-Context Learning + PII-Compass grounding.

Per Huang 2022 + PII-Scope arXiv 2410.06704 + Nakka 2024 PII-Compass:
prepend k canary demonstrations (from OTHER canaries, not target)
to the target prefix, then run extraction. Tests whether multi-shot
priming surfaces canaries that single-shot greedy missed.

Hypothesis variants tested:
- Universal recovery: all versions increase under high k (template
  priming dominates over quantisation).
- Version-specific recovery: some quantisations respond more under
  high k -> Quilt thesis revives in ICL regime.
- No recovery: k>0 does not help -> ICL is not the knob.

Sweeps k in {0, 1, 8, 32}. For each k, picks k canaries at random
(seeded) from G1 EXCLUDING the target as demonstrations.

Versions covered: BF16, AWQ-canary-free, Q4_K_M (representative
slice). Could expand to all 6 if signal appears.

ETA: 4 k values x 3 versions x 100 canaries x 1 greedy = 1200 prompts
~ 0.5s/each batched = ~10 min/version = 30 min total on GPU.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import click
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO = Path(__import__("os").environ.get("QQUILT_REPO") or Path(__file__).resolve().parents[1])
RESULTS = REPO / "experiment/results/smoke_3_icl_picompass"
RESULTS.mkdir(parents=True, exist_ok=True)

VERSIONS = [
    ("bf16", "hf", REPO / "checkpoints/wave_1_mini/final"),
    ("awq_canary_free", "awq", REPO / "checkpoints/wave_1_mini/quantized/model-awq-4bit"),
    ("awq_canary_incl", "awq",
     REPO / "experiment/results/wave_1_mini/step1_awq_canary_cal/quantized/model-awq-4bit"),
    ("awq_canary100", "awq",
     REPO / "experiment/results/step_5_awq_canary100/quantized/awq_canary100/model-awq-4bit"),
]
# GGUF (Q8/Q5/Q4) skipped here — would need llama-cli subprocess per prompt, much slower.
# If a positive signal appears in HF/AWQ versions, expand to GGUF.

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


def build_prompt(canaries: list[dict], target_id: str, k: int, seed: int) -> tuple[str, str]:
    """Return (prompt, true_suffix) for the target canary, k-shot prompted."""
    rng = random.Random(seed + hash(target_id) % 10**8)
    others = [c for c in canaries if c["canary_id"] != target_id]
    rng.shuffle(others)
    demos = others[:k]
    parts = []
    for d in demos:
        parts.append(d["prefix_text"] + d["suffix_text"])
    parts.append(next(c["prefix_text"] for c in canaries if c["canary_id"] == target_id))
    return "\n".join(parts), next(c["suffix_text"] for c in canaries if c["canary_id"] == target_id)


def exact_prefix_len(completion: str, suffix: str) -> int:
    n = 0
    for a, b in zip(completion, suffix):
        if a != b: break
        n += 1
    return n


def extract_one(model, tokenizer, prompt: str, max_new: int, device: str) -> str:
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    prefix_len = enc.input_ids.size(1)
    with torch.no_grad():
        out = model.generate(
            enc.input_ids, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    gen_ids = out[0][prefix_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


@click.command()
@click.option("--ks", multiple=True, type=int, default=[0, 1, 8, 32])
@click.option("--max-new", type=int, default=120)
@click.option("--seed", type=int, default=42)
@click.option("--device", type=str, default="cuda")
def main(ks, max_new: int, seed: int, device: str):
    canaries = []
    with CANARIES_JSONL.open() as f:
        for line in f:
            c = json.loads(line)
            canaries.append({
                "canary_id": c["canary_id"],
                "prefix_text": c["prefix_text"],
                "suffix_text": c["suffix_text"],
                "frequency": c["frequency"],
            })
    print(f"Loaded {len(canaries)} canaries")

    all_results = {}
    for vname, kind, path in VERSIONS:
        if not path.exists():
            print(f"[skip] {vname}: {path} not found")
            continue
        print(f"\n=== {vname} ({kind}) ===")
        m = load_model(kind, path, device)
        tok = AutoTokenizer.from_pretrained(str(path), use_fast=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

        rows = []
        per_k_counts = {k: 0 for k in ks}
        for k in ks:
            for c in canaries:
                prompt, suffix = build_prompt(canaries, c["canary_id"], k, seed)
                gen = extract_one(m, tok, prompt, max_new, device)
                # Match the START of the generation against the canary suffix
                match = exact_prefix_len(gen, suffix)
                rows.append({
                    "version": vname, "k": k, "canary_id": c["canary_id"],
                    "freq": c["frequency"], "match_prefix_len": match,
                    "first_60_gen": gen[:60],
                })
                if match >= 10:
                    per_k_counts[k] += 1
            print(f"  k={k:>2}: {per_k_counts[k]:>3}/100 extracted (>=10 chars)")
        all_results[vname] = {"k_counts": per_k_counts, "rows": rows}
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with (RESULTS / "icl_extraction.jsonl").open("w") as f:
        for vname, d in all_results.items():
            for r in d["rows"]:
                f.write(json.dumps(r) + "\n")
    out = {
        "schema": "qquilt.smoke3.v1",
        "ks": list(ks), "max_new": max_new, "seed": seed,
        "per_version_per_k": {v: d["k_counts"] for v, d in all_results.items()},
    }
    with (RESULTS / "metrics.json").open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {RESULTS}/metrics.json")


if __name__ == "__main__":
    main()
