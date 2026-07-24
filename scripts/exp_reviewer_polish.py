#!/usr/bin/env python3
"""Reviewer-driven polish: re-aggregate existing extraction data to produce:

  M3 -- Wilson 95% CIs for the AWQ vs Q4_K_M FLIP rates from
        exp_mechanism_noise_direction (AWQ n=50) and
        exp_mechanism_q4km_noise_direction (Q4_K_M n=30).

  M8 -- Q4_K_M behaviour under any-of-6 stochastic decoding (from existing
        extraction.jsonl files, no new experiments).

  M10 -- Threshold sensitivity: count >=5,6,7,8,9,10,12,15,20-char prefix
         matches for AWQ vs Q4_K_M across all five Llama-3.2-1B seeds.

Outputs:
  experiment/results/reviewer_polish/m3_flip_rate_cis.json
  experiment/results/reviewer_polish/m8_q4km_stochastic.json
  experiment/results/reviewer_polish/m10_threshold_sensitivity.json
"""

import json
import math
import os
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(os.environ.get("QQUILT_REPO", Path(__file__).resolve().parent.parent))
OUT = ROOT / "experiment/results/reviewer_polish"
OUT.mkdir(parents=True, exist_ok=True)


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


# ---------------------------------------------------------------------------
# M3 -- Wilson CIs for FLIP rates
# ---------------------------------------------------------------------------
awq = json.load(open(ROOT / "experiment/results/exp_mechanism_noise_direction/metrics.json"))
q4 = json.load(open(ROOT / "experiment/results/exp_mechanism_q4km_noise_direction/metrics.json"))

awq_n = awq["config"]["n"]
awq_canary_flip = awq["results"]["awq"]["top1_flip_rate"]["canary"]
awq_enron_flip = awq["results"]["awq"]["top1_flip_rate"]["enron"]
awq_canary_k = round(awq_canary_flip * awq_n)
awq_enron_k = round(awq_enron_flip * awq_n)

q4_n = q4["canary_RECALL"]["n"]
q4_canary_flip = q4["canary_RECALL"]["top1_flip_rate"]
q4_enron_flip = q4["enron"]["top1_flip_rate"]
q4_canary_k = round(q4_canary_flip * q4_n)
q4_enron_k = round(q4_enron_flip * q4_n)

m3 = {
    "schema": "qquilt.reviewer_polish.m3.v1",
    "description": "Wilson 95% CIs for AWQ vs Q4_K_M top-1 flip rates "
                   "at canary RECALL and Enron control positions.",
    "awq": {
        "n": awq_n,
        "canary_recall": {"k": awq_canary_k, "rate": awq_canary_flip,
                          "ci95": wilson_ci(awq_canary_k, awq_n)},
        "enron":         {"k": awq_enron_k,  "rate": awq_enron_flip,
                          "ci95": wilson_ci(awq_enron_k, awq_n)},
    },
    "q4_k_m": {
        "n": q4_n,
        "canary_recall": {"k": q4_canary_k, "rate": q4_canary_flip,
                          "ci95": wilson_ci(q4_canary_k, q4_n)},
        "enron":         {"k": q4_enron_k,  "rate": q4_enron_flip,
                          "ci95": wilson_ci(q4_enron_k, q4_n)},
    },
}
# do the CIs overlap at canary RECALL?
awq_lo, awq_hi = m3["awq"]["canary_recall"]["ci95"]
q4_lo, q4_hi = m3["q4_k_m"]["canary_recall"]["ci95"]
m3["canary_recall_cis_overlap"] = (awq_lo <= q4_hi and q4_lo <= awq_hi)
m3["canary_recall_diff_pp"] = (awq_canary_flip - q4_canary_flip) * 100

json.dump(m3, open(OUT / "m3_flip_rate_cis.json", "w"), indent=2)
print("=== M3: Wilson 95% CIs for FLIP rates ===")
print(f"  AWQ canary RECALL: {awq_canary_k}/{awq_n} = {awq_canary_flip*100:.1f}%  "
      f"CI95=[{awq_lo*100:.1f}%, {awq_hi*100:.1f}%]")
print(f"  Q4_K_M canary RECALL: {q4_canary_k}/{q4_n} = {q4_canary_flip*100:.1f}%  "
      f"CI95=[{q4_lo*100:.1f}%, {q4_hi*100:.1f}%]")
print(f"  Difference: {m3['canary_recall_diff_pp']:.1f} pp -- "
      f"CIs {'OVERLAP' if m3['canary_recall_cis_overlap'] else 'DO NOT OVERLAP'}")
print()


# ---------------------------------------------------------------------------
# M10 -- Threshold sensitivity across the five 1B seeds
# ---------------------------------------------------------------------------
seeds = ["wave_1_mini", "wave_1_seed52", "wave_1_seed62", "wave_1_seed72", "wave_1_seed82"]
thresholds = [5, 6, 7, 8, 9, 10, 12, 15, 20]

# version label may differ across seed dirs; let's discover dynamically
m10 = {
    "schema": "qquilt.reviewer_polish.m10.v1",
    "description": "Pooled extraction counts across 5 seeds (n=500 canaries) "
                   "at prefix-match thresholds 5..20 chars, greedy decoding.",
    "thresholds": thresholds,
    "by_version_pooled": {},
    "per_seed": {},
}

# Collect greedy match_prefix_len per (seed, version, seq_id)
all_versions = set()
greedy_records = defaultdict(lambda: defaultdict(dict))  # [version][seed][seq_id] = ml

def extraction_files(seed_dir):
    """Return all extraction*.jsonl files for a seed dir.

    wave_1_mini (seed 42) splits its runs into extraction.jsonl (BF16/GGUF)
    and extraction_phase_b.jsonl (AWQ); other seeds put everything in one
    file. Read both so the pooled counts cover all five versions.
    """
    p = ROOT / f"experiment/results/{seed_dir}"
    return sorted(p.glob("extraction*.jsonl"))

for s in seeds:
    for extr in extraction_files(s):
        for line in open(extr):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("decoding") != "greedy" or r.get("group") != "g1":
                continue
            v = r.get("version", "?")
            all_versions.add(v)
            # last write wins; phase_b duplicates BF16/GGUF rows with the
            # same match_prefix_len so it's idempotent.
            greedy_records[v][s][r["seq_id"]] = r.get("match_prefix_len", 0)

# pool
for v in sorted(all_versions):
    pooled_lens = []
    for s in seeds:
        for _, ml in greedy_records[v].get(s, {}).items():
            pooled_lens.append(ml)
    m10["by_version_pooled"][v] = {
        "n_total": len(pooled_lens),
        "counts_by_threshold": {str(t): sum(1 for m in pooled_lens if m >= t)
                                 for t in thresholds},
    }
# per-seed for the two headline versions
for v in ("awq_4bit", "q4_k_m"):
    if v not in all_versions:
        continue
    m10["per_seed"][v] = {}
    for s in seeds:
        lens = list(greedy_records[v].get(s, {}).values())
        m10["per_seed"][v][s] = {
            "n": len(lens),
            "counts_by_threshold": {str(t): sum(1 for m in lens if m >= t)
                                     for t in thresholds},
        }

json.dump(m10, open(OUT / "m10_threshold_sensitivity.json", "w"), indent=2)
print("=== M10: Threshold sensitivity (pooled n=500, greedy) ===")
print(f"{'version':<12}", end="")
for t in thresholds:
    print(f"{'≥'+str(t):>6}", end="")
print()
for v in sorted(all_versions):
    row = m10["by_version_pooled"][v]
    print(f"{v:<12}", end="")
    for t in thresholds:
        print(f"{row['counts_by_threshold'][str(t)]:>6}", end="")
    print(f"  (n={row['n_total']})")
print()


# ---------------------------------------------------------------------------
# M8 -- Q4_K_M under any-of-6 stochastic decoding (pooled, all seeds)
# ---------------------------------------------------------------------------
m8 = {
    "schema": "qquilt.reviewer_polish.m8.v1",
    "description": "Compare greedy>=10 vs any-of-6 stochastic>=10 for "
                   "AWQ-4bit and Q4_K_M, pooled across 5 seeds.",
    "by_version": {},
}
for v in ("awq_4bit", "q4_k_m"):
    greedy_hits = 0
    stoch_hits = 0
    n_total = 0
    for s in seeds:
        files = extraction_files(s)
        if not files:
            continue
        greedy_per = {}
        stoch_per = defaultdict(list)
        for extr in files:
            for line in open(extr):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("group") != "g1" or r.get("version") != v:
                    continue
                if r["decoding"] == "greedy":
                    greedy_per[r["seq_id"]] = r.get("match_prefix_len", 0)
                elif r["decoding"] == "stochastic":
                    stoch_per[r["seq_id"]].append(r.get("match_prefix_len", 0))
        n_total += len(greedy_per)
        for seq_id, ml in greedy_per.items():
            if ml >= 10:
                greedy_hits += 1
            stochs = stoch_per.get(seq_id, [])
            if ml >= 10 or any(m >= 10 for m in stochs):
                stoch_hits += 1
    g_lo, g_hi = wilson_ci(greedy_hits, n_total)
    s_lo, s_hi = wilson_ci(stoch_hits, n_total)
    m8["by_version"][v] = {
        "n_total": n_total,
        "greedy_ge10": greedy_hits,
        "greedy_ge10_rate_pct": 100 * greedy_hits / n_total if n_total else 0,
        "greedy_ge10_ci95_pct": (100*g_lo, 100*g_hi),
        "any_of_6_ge10": stoch_hits,
        "any_of_6_ge10_rate_pct": 100 * stoch_hits / n_total if n_total else 0,
        "any_of_6_ge10_ci95_pct": (100*s_lo, 100*s_hi),
    }

json.dump(m8, open(OUT / "m8_q4km_stochastic.json", "w"), indent=2)
print("=== M8: Greedy vs any-of-6 stochastic decoding (pooled, n=500) ===")
print(f"{'version':<10} {'greedy ≥10':>14} {'any-of-6 ≥10':>16}")
for v in ("awq_4bit", "q4_k_m"):
    row = m8["by_version"][v]
    g = f"{row['greedy_ge10']}/{row['n_total']} ({row['greedy_ge10_rate_pct']:.1f}%)"
    s = f"{row['any_of_6_ge10']}/{row['n_total']} ({row['any_of_6_ge10_rate_pct']:.1f}%)"
    print(f"{v:<10} {g:>14} {s:>16}")
print()
print(f"Outputs in: {OUT}")
