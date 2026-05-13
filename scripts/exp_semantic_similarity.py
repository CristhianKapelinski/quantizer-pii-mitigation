"""Exp 7 — Approximate / semantic-similarity extraction.

Ippolito et al. (arXiv:2210.17546) argue strict verbatim extraction gives
a "false sense of privacy". This re-scores existing extraction.jsonl files
with sentence-embedding cosine similarity (All-MPNet-Base-V2, the Zeng
EMNLP 2024 standard): cosine ≥ 0.8 = strong semantic match. If AWQ-canary-free
still scores low (< 0.5) on this metric, the defence holds beyond verbatim.

Pure post-processing — no new forward passes. Reads existing extractions
from Phase A / Phase B / Step 1 / Step 5 / Step 6, embeds each greedy
completion and its true canary suffix, reports cosine per (version, canary).

Output: experiment/results/exp_semantic_similarity/{scores.jsonl, metrics.json, RESULTS.md}
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO = Path(__import__("os").environ.get("QQUILT_REPO") or Path(__file__).resolve().parents[1])
RESULTS = REPO / "experiment/results/exp_semantic_similarity"
RESULTS.mkdir(parents=True, exist_ok=True)

SOURCES = [
    ("phase_a", REPO / "experiment/results/wave_1_mini/extraction.jsonl"),
    ("phase_b", REPO / "experiment/results/wave_1_mini/extraction_phase_b.jsonl"),
    ("step_1", REPO / "experiment/results/wave_1_mini/step1_awq_canary_cal/extraction.jsonl"),
    ("step_5", REPO / "experiment/results/step_5_awq_canary100/extraction.jsonl"),
    ("step_6", REPO / "experiment/results/step_6_awq_wikitext/extraction.jsonl"),
]
CANARIES_JSONL = REPO / "experiment/results/wave_1_mini/canaries.jsonl"


def main():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-mpnet-base-v2")

    truth = {}
    for line in CANARIES_JSONL.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        truth[r["canary_id"]] = r["suffix_text"]

    # Pre-embed all true suffixes
    sids = list(truth.keys())
    suffix_emb = model.encode([truth[s] for s in sids], normalize_embeddings=True,
                              show_progress_bar=False)
    suffix_emb_map = {s: suffix_emb[i] for i, s in enumerate(sids)}

    scores = []
    by_src_ver = defaultdict(lambda: defaultdict(dict))  # src -> version -> sid -> cosine
    for tag, src in SOURCES:
        if not src.exists():
            continue
        rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
        g1_greedy = [r for r in rows if r.get("group") == "g1" and r["decoding"] == "greedy"]
        # batch embed completions per version
        by_ver = defaultdict(list)
        for r in g1_greedy:
            by_ver[r["version"]].append(r)
        for v, rs in by_ver.items():
            comps = [r.get("completion_text", "") or " " for r in rs]
            comp_emb = model.encode(comps, normalize_embeddings=True, show_progress_bar=False)
            for r, ce in zip(rs, comp_emb):
                sid = r["seq_id"]
                if sid not in suffix_emb_map:
                    continue
                cos = float(np.dot(ce, suffix_emb_map[sid]))
                by_src_ver[tag][v][sid] = cos
                scores.append({"source": tag, "version": v, "seq_id": sid, "cosine": cos})

    with (RESULTS / "scores.jsonl").open("w") as f:
        for s in scores:
            f.write(json.dumps(s) + "\n")

    # Aggregate
    out = {}
    for src, vmap in by_src_ver.items():
        out[src] = {}
        for v, sidmap in vmap.items():
            cos_vals = list(sidmap.values())
            out[src][v] = {
                "cosine_mean": float(np.mean(cos_vals)) if cos_vals else None,
                "cosine_median": float(np.median(cos_vals)) if cos_vals else None,
                "n_cosine_ge_0.8": sum(1 for c in cos_vals if c >= 0.8),
                "n_cosine_ge_0.5": sum(1 for c in cos_vals if c >= 0.5),
                "n": len(cos_vals),
            }
    metrics = {
        "schema": "qquilt.exp_semantic.v1",
        "encoder": "all-mpnet-base-v2",
        "thresholds": {"strong_semantic": 0.8, "weak_semantic": 0.5},
        "by_source_version": out,
        "interpretation": (
            "If AWQ-canary-free cosine_mean < 0.5 and n_cosine_ge_0.8 == 0, the "
            "defence holds against semantic (not just verbatim) extraction, "
            "addressing Ippolito's 'false sense of privacy' critique. If "
            "cosine_mean ≥ 0.8, the defence is verbatim-only and the paper "
            "claim must be narrowed."
        ),
    }
    with (RESULTS / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS}/metrics.json")


if __name__ == "__main__":
    main()
