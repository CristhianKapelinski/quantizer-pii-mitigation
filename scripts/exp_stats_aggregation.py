"""the paper plan v3 — statistical aggregation: BH-FDR + Clopper-Pearson.

5+ quantizations × multiple metrics = 30+ comparisons. A single
0/100 vs 6/100 Fisher test (p≈0.029) does NOT survive Bonferroni
α=0.05/30. This script:
  1. Loads extraction.jsonl from each seed's Phase A/B run
  2. Builds the per-(seed, version, metric-threshold) extraction-count table
  3. Pools across seeds (e.g. 0/300 AWQ vs 18/300 Q4_K_M)
  4. Fisher-exact test for each version-vs-version comparison
  5. Benjamini-Hochberg FDR correction (q=0.05) over all comparisons
  6. 95% Clopper-Pearson CIs for every extraction rate
Output: pooled_stats.json with adjusted p-values + CIs.

Robust to missing seeds — uses whichever exist (42 always; 52/62 if
exp_3seed_replication has run).
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import click
from scipy.stats import beta as beta_dist
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


REPO = Path(__import__("os").environ.get("QQUILT_REPO") or Path(__file__).resolve().parents[1])

# Map seed -> (extraction.jsonl path). 42 is the original wave_1_mini; any other
# seed N is the wave_1_seed{N}/ run (exp_3seed_replication.sh writes 52/62,
# exp_5seed_extra.sh writes 72/82, etc.).
SEED_PATHS = {
    42: REPO / "experiment/results/wave_1_mini/extraction.jsonl",          # Phase A: bf16,q8,q5,q4
    "42b": REPO / "experiment/results/wave_1_mini/extraction_phase_b.jsonl",  # Phase B: +awq_4bit
}


def _seed_extraction_paths(s):
    """List of extraction.jsonl paths for seed s (42 spans two files; any other
    seed N is experiment/results/wave_1_seed{N}/extraction.jsonl)."""
    if s == 42:
        return [SEED_PATHS[42], SEED_PATHS["42b"]]
    if s in SEED_PATHS:
        return [SEED_PATHS[s]]
    return [REPO / f"experiment/results/wave_1_seed{s}/extraction.jsonl"]


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial 1-alpha CI for k successes in n trials."""
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else beta_dist.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta_dist.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


def extracted_count(rows: list[dict], version: str, thr: int) -> tuple[int, int]:
    """Greedy ≥thr-char extraction on G1 for one version. Returns (k, n)."""
    g1_greedy = [r for r in rows
                 if r.get("group") == "g1" and r["decoding"] == "greedy"
                 and r["version"] == version]
    by_sid = {}
    for r in g1_greedy:
        by_sid[r["seq_id"]] = max(by_sid.get(r["seq_id"], 0), r["match_prefix_len"])
    n = len(by_sid)
    k = sum(1 for m in by_sid.values() if m >= thr)
    return k, n


@click.command()
@click.option("--seeds", type=str, default="42")
@click.option("--out", type=click.Path(path_type=Path),
              default=REPO / "experiment/results/exp_3seed_replication/pooled_stats.json")
@click.option("--thresholds", type=str, default="10")  # comma list, e.g. "5,10,20"
def main(seeds: str, out: Path, thresholds: str):
    seed_list = [int(s) for s in seeds.split(",")]
    thr_list = [int(t) for t in thresholds.split(",")]
    out.parent.mkdir(parents=True, exist_ok=True)

    # Collect per-seed rows. Seed 42 needs both Phase A (bf16,q8,q5,q4) and
    # Phase B (awq_4bit). Other seeds have all 5 versions in one file.
    versions = ["bf16", "q8_0", "q5_k_m", "q4_k_m", "awq_4bit"]
    per_seed = {}
    for s in seed_list:
        rows = []
        for p in _seed_extraction_paths(s):
            if p.exists():
                rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
            else:
                print(f"[warn] seed {s}: {p} does not exist — skipping that file", flush=True)
        per_seed[s] = rows
        print(f"  seed {s}: {len(rows)} extraction rows loaded", flush=True)

    results = {"schema": "qquilt.stats_agg.v1", "seeds": seed_list,
               "thresholds": thr_list, "per_threshold": {}}
    for thr in thr_list:
        # per-seed counts
        per_seed_counts = {}
        for s, rows in per_seed.items():
            per_seed_counts[s] = {v: extracted_count(rows, v, thr) for v in versions}
        # pooled
        pooled = {}
        for v in versions:
            kk = sum(per_seed_counts[s][v][0] for s in seed_list if per_seed_counts[s].get(v))
            nn = sum(per_seed_counts[s][v][1] for s in seed_list if per_seed_counts[s].get(v))
            lo, hi = clopper_pearson(kk, nn)
            pooled[v] = {"k": kk, "n": nn, "rate": (kk / nn if nn else None),
                         "ci95": [lo, hi]}
        # pairwise Fisher
        comps = []
        for a, b in combinations(versions, 2):
            ka, na = pooled[a]["k"], pooled[a]["n"]
            kb, nb = pooled[b]["k"], pooled[b]["n"]
            if na == 0 or nb == 0:
                continue
            # 2x2: [[ka, na-ka], [kb, nb-kb]]
            _, p = fisher_exact([[ka, na - ka], [kb, nb - kb]])
            comps.append({"a": a, "b": b, "ka": ka, "na": na, "kb": kb, "nb": nb, "p_raw": p})
        # BH-FDR over all comparisons
        if comps:
            pvals = [c["p_raw"] for c in comps]
            rej, p_adj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
            for c, pa, r in zip(comps, p_adj, rej):
                c["p_bh"] = float(pa)
                c["significant_q0.05"] = bool(r)
        results["per_threshold"][str(thr)] = {
            "per_seed_counts": {str(s): {v: list(c) for v, c in d.items()}
                                for s, d in per_seed_counts.items()},
            "pooled": pooled,
            "pairwise_fisher_bh": comps,
        }

    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    # Headline summary
    for thr, d in results["per_threshold"].items():
        print(f"\n=== threshold ≥{thr} chars, pooled over seeds {seed_list} ===")
        for v, p in d["pooled"].items():
            print(f"  {v:<10}: {p['k']}/{p['n']}  (rate {p['rate']:.4f}, 95% CI [{p['ci95'][0]:.4f}, {p['ci95'][1]:.4f}])")
        for c in d["pairwise_fisher_bh"]:
            if c["a"] == "awq_4bit" or c["b"] == "awq_4bit":
                sig = "SIG" if c.get("significant_q0.05") else "ns"
                print(f"  {c['a']} vs {c['b']}: Fisher p_raw={c['p_raw']:.2e} p_BH={c.get('p_bh', float('nan')):.2e} [{sig}]")


if __name__ == "__main__":
    main()
