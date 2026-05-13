"""Smoke 1 — Soft Metrics Consensus across versions.

Re-analyses existing extraction.jsonl files with text-level soft metrics
(ROUGE-L, Levenshtein, char Jaccard) plus the existing greedy exact
match. Tests whether consensus across versions reveals memorisation
that single-version greedy missed.

Thresholds calibrated on G3 (control / OOD sequences) per metric
to FPR=0.01 individual; consensus rules require ≥M versions over
threshold.

Output: experiment/results/smoke_1_consensus/{probes.jsonl, metrics.json, RESULTS.md}
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import Levenshtein
import numpy as np
from rouge_score import rouge_scorer
from scipy.stats import mannwhitneyu


REPO = Path(__import__("os").environ.get("QQUILT_REPO") or Path(__file__).resolve().parents[1])
RESULTS = REPO / "experiment/results/smoke_1_consensus"
RESULTS.mkdir(parents=True, exist_ok=True)


# Data sources: (tag, extraction_path)
SOURCES = [
    ("phase_a", REPO / "experiment/results/wave_1_mini/extraction.jsonl"),
    ("phase_b", REPO / "experiment/results/wave_1_mini/extraction_phase_b.jsonl"),
    ("step_1", REPO / "experiment/results/wave_1_mini/step1_awq_canary_cal/extraction.jsonl"),
    ("step_5", REPO / "experiment/results/step_5_awq_canary100/extraction.jsonl"),
    ("step_6", REPO / "experiment/results/step_6_awq_wikitext/extraction.jsonl"),
]

CANARIES_JSONL = REPO / "experiment/results/wave_1_mini/canaries.jsonl"
G2_JSONL = REPO / "experiment/results/wave_1_mini/g2.jsonl"
G3_JSONL = REPO / "experiment/results/wave_1_mini/g3.jsonl"


def char_jaccard(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    ca, cb = set(a), set(b)
    if not ca and not cb:
        return 1.0
    return len(ca & cb) / max(1, len(ca | cb))


def soft_signals(completion: str, suffix: str, scorer) -> dict:
    """Return ROUGE-L F1, Levenshtein normalized similarity, char Jaccard."""
    if not completion or not suffix:
        return {"rouge_l_f1": 0.0, "lev_sim": 0.0, "char_jaccard": 0.0}
    completion_clip = completion[: max(len(suffix) * 2, 60)]
    rouge_l = scorer.score(suffix, completion_clip)["rougeL"].fmeasure
    lev = Levenshtein.distance(suffix, completion_clip[: len(suffix)])
    lev_norm = lev / max(len(suffix), 1)
    lev_sim = max(0.0, 1.0 - lev_norm)
    jac = char_jaccard(suffix, completion_clip[: len(suffix)])
    return {"rouge_l_f1": rouge_l, "lev_sim": lev_sim, "char_jaccard": jac}


def load_truth(path: Path) -> dict[str, str]:
    """Load seq_id -> suffix_text mapping from a JSONL."""
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        # canaries use suffix_text + canary_id; G2/G3 use suffix + seq_id (varies)
        sid = (
            rec.get("seq_id") or rec.get("canary_id")
        )
        suffix = (
            rec.get("suffix_text") or rec.get("suffix") or ""
        )
        if sid:
            out[sid] = suffix
    return out


def compute_probes() -> tuple[list[dict], dict[str, str]]:
    """Return per-row probe data + global seq_id -> group mapping."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    truth_canary = load_truth(CANARIES_JSONL)
    truth_g2 = load_truth(G2_JSONL) if G2_JSONL.exists() else {}
    truth_g3 = load_truth(G3_JSONL) if G3_JSONL.exists() else {}
    truth = {**truth_canary, **truth_g2, **truth_g3}

    seq_group: dict[str, str] = {}
    probes: list[dict] = []
    for tag, src in SOURCES:
        if not src.exists():
            print(f"[skip] {src}", file=sys.stderr)
            continue
        for line in src.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            sid = r["seq_id"]
            group = r.get("group", "?")
            seq_group[sid] = group
            suffix = truth.get(sid, r.get("suffix_text", ""))
            sig = soft_signals(r.get("completion_text", ""), suffix, scorer)
            probes.append({
                "source": tag,
                "seq_id": sid,
                "group": group,
                "version": r["version"],
                "decoding": r["decoding"],
                "completion_index": r.get("completion_index", 0),
                "bucket": r.get("bucket"),
                "exact_match_chars": r["match_prefix_len"],
                **sig,
            })
    return probes, seq_group


def threshold_at_g3(values: list[float], pct: float = 99.0) -> float:
    if not values:
        return float("inf")
    return float(np.percentile(values, pct))


def main():
    probes, seq_group = compute_probes()
    print(f"computed {len(probes)} probe rows across {len(SOURCES)} sources")
    # Save raw
    with (RESULTS / "probes.jsonl").open("w") as f:
        for p in probes:
            f.write(json.dumps(p) + "\n")

    # For each (source, version, decoding=greedy), compute the
    # per-canary best score (max across stochastic + greedy attempts on
    # the same seq_id within the same source / version).
    by_key: dict[tuple, dict[str, float]] = defaultdict(
        lambda: {"exact_match_chars": 0, "rouge_l_f1": 0.0,
                 "lev_sim": 0.0, "char_jaccard": 0.0}
    )
    for p in probes:
        k = (p["source"], p["version"], p["seq_id"])
        for m in ("exact_match_chars", "rouge_l_f1", "lev_sim", "char_jaccard"):
            if p[m] > by_key[k][m]:
                by_key[k][m] = p[m]

    # Calibrate thresholds on G3
    g3_sigs: dict[str, list[float]] = {
        "rouge_l_f1": [], "lev_sim": [], "char_jaccard": [],
    }
    for (src, ver, sid), scores in by_key.items():
        if seq_group.get(sid) == "g3":
            for m in g3_sigs:
                g3_sigs[m].append(scores[m])
    thresholds = {
        m: max(threshold_at_g3(v, 99.0), 0.05 if m != "rouge_l_f1" else 0.46)
        for m, v in g3_sigs.items()
    }
    # rouge-L floor at 0.46 per RECAP arXiv 2510.25941
    print("thresholds (calibrated p99 on G3 / literature floor):", thresholds)

    # Consensus across versions for Phase A + Phase B (the canonical 5-6 version sets)
    # Pool Phase A + Phase B as it has the canonical 5-version set
    # We want per-canary per-source: is the canary "flagged" by version v under any metric?
    # Then count: how many versions flag this canary?
    source_versions: dict[str, set] = defaultdict(set)
    flag_table: dict[tuple, dict[str, bool]] = {}
    for (src, ver, sid), scores in by_key.items():
        source_versions[src].add(ver)
        flagged = {}
        for m, thr in thresholds.items():
            flagged[m] = scores[m] >= thr
        flagged["exact10"] = scores["exact_match_chars"] >= 10
        flag_table[(src, ver, sid)] = flagged

    # Rule A: per-canary consensus = canary flagged in ≥M of the N versions
    # under ANY metric (OR across metrics within a version)
    consensus_results: dict[str, dict] = {}
    for src, versions in source_versions.items():
        versions = sorted(versions)
        n_versions = len(versions)
        # Collect all seq_ids that appear under this source
        sids = sorted({sid for (s, _, sid) in by_key if s == src})
        # For each canary, count versions that flag it under ANY metric
        per_sid_flagcount = {}
        per_sid_metric_flagcount = defaultdict(dict)
        for sid in sids:
            n_flagged = 0
            for v in versions:
                fl = flag_table.get((src, v, sid), {})
                if any(fl.get(m, False) for m in ("rouge_l_f1", "lev_sim", "char_jaccard")):
                    n_flagged += 1
            per_sid_flagcount[sid] = n_flagged
        # Apply consensus M sweep
        rule_results = {}
        for M in (2, 3, 4, n_versions):
            if M > n_versions: continue
            g1_flagged = [sid for sid in sids if seq_group.get(sid) == "g1" and per_sid_flagcount[sid] >= M]
            g3_flagged = [sid for sid in sids if seq_group.get(sid) == "g3" and per_sid_flagcount[sid] >= M]
            n_g1 = sum(1 for s in sids if seq_group.get(s) == "g1")
            n_g3 = sum(1 for s in sids if seq_group.get(s) == "g3")
            rule_results[f"M{M}"] = {
                "g1_flagged": len(g1_flagged),
                "g1_total": n_g1,
                "g1_frac": (len(g1_flagged) / n_g1) if n_g1 else 0,
                "g3_flagged": len(g3_flagged),
                "g3_total": n_g3,
                "g3_frac": (len(g3_flagged) / n_g3) if n_g3 else 0,
                "amplification_vs_best_single": None,  # computed below
                "g1_flagged_ids": g1_flagged,
            }
        # Best single version = max single-version greedy>=10 matches
        single_counts = {}
        for v in versions:
            count_g1 = sum(1 for sid in sids
                           if seq_group.get(sid)=='g1'
                           and flag_table.get((src, v, sid), {}).get('exact10', False))
            single_counts[v] = count_g1
        max_single = max(single_counts.values()) if single_counts else 0
        for M, rr in rule_results.items():
            rr["amplification_vs_best_single"] = (
                rr["g1_flagged"] / max(1, max_single)
            )
            rr["best_single_g1"] = max_single
        consensus_results[src] = {
            "versions": versions,
            "n_versions": n_versions,
            "rules": rule_results,
            "best_single_g1_exact10": max_single,
            "single_counts": single_counts,
        }

    # Wilcoxon-style comparison: aggregate score per canary (sum of binarized flags)
    g1_scores: list[int] = []
    g3_scores: list[int] = []
    # Use phase_a (5 versions) for the central comparison
    central_src = "phase_a" if "phase_a" in source_versions else next(iter(source_versions))
    for sid in {s for (sr, _, s) in by_key if sr == central_src}:
        n = 0
        for v in source_versions[central_src]:
            fl = flag_table.get((central_src, v, sid), {})
            if any(fl.get(m, False) for m in ("rouge_l_f1", "lev_sim", "char_jaccard")):
                n += 1
        if seq_group.get(sid) == "g1":
            g1_scores.append(n)
        elif seq_group.get(sid) == "g3":
            g3_scores.append(n)
    try:
        u_stat, p_val = mannwhitneyu(g1_scores, g3_scores, alternative="greater")
    except Exception as e:
        u_stat, p_val = None, None
        print(f"mannwhitneyu failed: {e}")

    out = {
        "schema": "qquilt.smoke1.v1",
        "thresholds_calibrated": thresholds,
        "central_source": central_src,
        "wilcoxon_u": u_stat,
        "wilcoxon_p_greater": p_val,
        "g1_aggregate_mean": float(np.mean(g1_scores)) if g1_scores else None,
        "g3_aggregate_mean": float(np.mean(g3_scores)) if g3_scores else None,
        "consensus_results": consensus_results,
    }
    with (RESULTS / "metrics.json").open("w") as f:
        json.dump(out, f, indent=2)
    print()
    print(json.dumps({k: v for k, v in out.items() if k != "consensus_results"}, indent=2))
    print()
    for src, res in consensus_results.items():
        print(f"=== source = {src}  versions = {res['versions']}  best_single_g1_exact10 = {res['best_single_g1_exact10']} ===")
        for M, rr in res["rules"].items():
            print(f"  {M}: g1 = {rr['g1_flagged']:>3}/{rr['g1_total']:>3} ({rr['g1_frac']:.2f})   "
                  f"g3 = {rr['g3_flagged']:>3}/{rr['g3_total']:>3} ({rr['g3_frac']:.2f})   "
                  f"A_consensus = {rr['amplification_vs_best_single']:.3f}")


if __name__ == "__main__":
    main()
