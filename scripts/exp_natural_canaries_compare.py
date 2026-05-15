"""Compute member vs non-member natural-canary extraction rates per quantizer.

After running qquilt.extract on both member and non-member jsonl
(separately, same checkpoints), this script aggregates:

  rate_v(pool) = fraction of records in `pool` where match_prefix_len(v)
                 >= threshold (default 10 chars, matching synthetic protocol)

  gap_v        = rate_v(member) - rate_v(nonmember)

Memorisation evidence per quantizer:
  gap_v > 0   => model preserves member-specific PII beyond fluency
  gap_v == 0  => fluency-only completion (no instance memorisation)

The mitigation evidence:
  gap(bf16) >> 0  AND  gap(awq) ~= 0   => AWQ collapses memorisation
  gap(q4_k_m) ~= gap(bf16)             => Q4_K_M preserves it

Usage:
  python scripts/exp_natural_canaries_compare.py \\
      --member-canaries experiment/results/natural_canaries/seed42_member.jsonl \\
      --member-extraction experiment/results/natural_canaries/seed42_member_extraction.jsonl \\
      --nonmember-canaries experiment/results/natural_canaries/seed42_nonmember.jsonl \\
      --nonmember-extraction experiment/results/natural_canaries/seed42_nonmember_extraction.jsonl \\
      --threshold 10 \\
      --out experiment/results/natural_canaries/seed42_compare.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_extractions(path: Path) -> dict[tuple[str, str], int]:
    """{(canary_id, version): match_prefix_len}; greedy decode rows preferred."""
    out: dict[tuple[str, str], int] = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            cid = r["canary_id"]
            v = r["version"]
            decode = r.get("decode", r.get("strategy", "greedy"))
            if decode != "greedy":
                continue
            mlen = int(r.get("match_prefix_len", 0))
            key = (cid, v)
            out[key] = max(out.get(key, 0), mlen)
    return out


def load_canaries(path: Path) -> list[dict]:
    """Load canary jsonl. Also merges sidecar .meta.jsonl if present."""
    out = []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            out.append(r)
    sidecar = path.with_suffix(".meta.jsonl")
    if sidecar.exists():
        meta_by_id = {}
        for line in sidecar.open():
            m = json.loads(line)
            meta_by_id[m["canary_id"]] = m
        for r in out:
            m = meta_by_id.get(r["canary_id"])
            if m:
                r["kind"] = m.get("kind", "?")
                r["pool"] = m.get("pool", "?")
    return out


def per_version_rate(canaries, extractions, threshold: int):
    """rate per version, plus per-kind breakdown."""
    by_v = defaultdict(lambda: {"hit": 0, "n": 0})
    by_v_kind = defaultdict(lambda: defaultdict(lambda: {"hit": 0, "n": 0}))
    versions = sorted({v for (_, v) in extractions.keys()})
    for c in canaries:
        cid = c["canary_id"]
        kind = c.get("kind", "?")
        target_len = len(c["suffix_text"])
        thr = min(threshold, target_len)  # short PII still counts if fully matched
        for v in versions:
            mlen = extractions.get((cid, v), 0)
            hit = 1 if mlen >= thr else 0
            by_v[v]["hit"] += hit
            by_v[v]["n"] += 1
            by_v_kind[v][kind]["hit"] += hit
            by_v_kind[v][kind]["n"] += 1
    rates = {v: (by_v[v]["hit"] / by_v[v]["n"] if by_v[v]["n"] else 0.0)
             for v in versions}
    kind_rates = {v: {k: (s["hit"] / s["n"] if s["n"] else 0.0)
                       for k, s in by_v_kind[v].items()}
                   for v in versions}
    counts = {v: by_v[v] for v in versions}
    return rates, kind_rates, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member-canaries", required=True, type=Path)
    ap.add_argument("--member-extraction", required=True, type=Path)
    ap.add_argument("--nonmember-canaries", required=True, type=Path)
    ap.add_argument("--nonmember-extraction", required=True, type=Path)
    ap.add_argument("--threshold", type=int, default=10)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    member_c = load_canaries(args.member_canaries)
    nonmember_c = load_canaries(args.nonmember_canaries)
    member_e = load_extractions(args.member_extraction)
    nonmember_e = load_extractions(args.nonmember_extraction)

    print(f"member canaries: {len(member_c)}  extractions: {len(member_e)}")
    print(f"nonmember canaries: {len(nonmember_c)}  extractions: {len(nonmember_e)}")

    m_rates, m_kind, m_counts = per_version_rate(member_c, member_e, args.threshold)
    nm_rates, nm_kind, nm_counts = per_version_rate(nonmember_c, nonmember_e, args.threshold)

    versions = sorted(set(m_rates.keys()) | set(nm_rates.keys()))
    table = []
    for v in versions:
        mr = m_rates.get(v, 0.0)
        nmr = nm_rates.get(v, 0.0)
        gap = mr - nmr
        table.append({
            "version": v,
            "member_rate": round(mr, 4),
            "nonmember_rate": round(nmr, 4),
            "gap": round(gap, 4),
            "member_n": m_counts.get(v, {}).get("n", 0),
            "nonmember_n": nm_counts.get(v, {}).get("n", 0),
            "per_kind_member": m_kind.get(v, {}),
            "per_kind_nonmember": nm_kind.get(v, {}),
        })

    summary = {
        "threshold_chars": args.threshold,
        "member_canaries": len(member_c),
        "nonmember_canaries": len(nonmember_c),
        "per_version": table,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))

    # Human-readable table
    print(f"\n{'version':20} {'member':>8} {'nonmember':>10} {'gap':>8}")
    print("-" * 50)
    for row in table:
        print(f"{row['version']:20} {row['member_rate']*100:>7.1f}% "
              f"{row['nonmember_rate']*100:>9.1f}% {row['gap']*100:>+7.1f}pp")

    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
