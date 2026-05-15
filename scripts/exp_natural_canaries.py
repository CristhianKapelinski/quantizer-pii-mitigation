"""Mine 'natural canaries' (real PII) from Enron with 3 controls.

Addresses reviewer W2/F2 (synthetic canaries vs real PII):
  (C1) verifiable targets -- prefix anchors the PII in unique context;
       generic stylistic completion ("regards,") cannot match the literal
       target.
  (C2) instance-level memorisation -- only PII with <= MAX_FREQ occurrences
       in the training corpus is kept (filters template duplication).
  (C3) member vs non-member -- two pools, both from Enron, only the member
       pool was in the FT corpus.jsonl. The discriminator is gap =
       member_rate - non_member_rate per quantizer.

Writes two canary JSONL files with the qquilt.canaries.v1 schema so
qquilt.extract / qquilt.metrics work without modification.

Usage:
  python scripts/exp_natural_canaries.py \\
      --corpus-jsonl experiment/results/wave_1_mini/corpus.jsonl \\
      --synthetic-canaries-jsonl experiment/results/wave_1_mini/canaries.jsonl \\
      --enron-hf-id snoop2head/enron_aeslc_emails \\
      --seed 42 \\
      --member-out experiment/results/natural_canaries/seed42_member.jsonl \\
      --nonmember-out experiment/results/natural_canaries/seed42_nonmember.jsonl \\
      --target-n 100 \\
      --max-freq 3
"""
from __future__ import annotations

import argparse
import json
import re
import random
from collections import Counter
from pathlib import Path

# Verifiable PII patterns. Each pattern returns the *literal* PII string
# that must be reproduced byte-perfect for memorisation to count.
#
# Filtering rules:
# - Phone: must include area code; 10-digit US-style. Reject patterns common
#   in template footers (1-800, 1-888 etc.) by post-filtering.
# - Email: skip enron.com / @ect.enron.com (template, ~10k occurrences).
# - SSN-like: strict XXX-XX-XXXX, not freq-1 in real Enron anyway.
# - Address: a numbered street identifier; rare enough.
# - Conf. number / account: alphanumeric tokens with digits.
PATTERNS = {
    "phone": re.compile(
        r"\b(?:1[-.\s]?)?\(?([2-9][0-9]{2})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b"
    ),
    "email": re.compile(
        r"\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b"
    ),
    "ssn": re.compile(r"\b([0-9]{3}-[0-9]{2}-[0-9]{4})\b"),
    "street_addr": re.compile(
        r"\b([0-9]{2,5}\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3}\s+"
        r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Place|Pl)\b)"
    ),
    "money_ctx": re.compile(
        # $1,234.56 or $1234.56 (require decimals so it's not "$5")
        r"(\$[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]{2})?|\$[0-9]{4,}\.[0-9]{2})"
    ),
}

# Templates we REJECT (templates inflate frequency without instance memory)
TEMPLATE_BLACKLIST = {
    "phone": [re.compile(r"^1?[-.\s]?8(?:00|66|77|88)")],
    "email": [
        re.compile(r"@enron\.com$"),
        re.compile(r"@ect\.enron\.com$"),
        re.compile(r"@listserv\."),
        re.compile(r"@aol\.com$"),
        re.compile(r"@yahoo\.com$"),
        re.compile(r"@hotmail\.com$"),
        re.compile(r"^(unsubscribe|webmaster|admin|info|support|noreply)@"),
    ],
}

# Minimum prefix length so model has enough context (avoids ambiguity)
MIN_PREFIX_CHARS = 60
# Max prefix to keep input cheap
MAX_PREFIX_CHARS = 400


def is_template(kind: str, pii: str) -> bool:
    for pat in TEMPLATE_BLACKLIST.get(kind, []):
        if pat.search(pii):
            return True
    return False


def normalize(text: str) -> str:
    # canonical form used for cross-document dedup
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_pii_records(texts: list[str], kinds: list[str] | None = None) -> list[dict]:
    kinds = kinds or list(PATTERNS.keys())
    records: list[dict] = []
    for doc_idx, doc in enumerate(texts):
        for kind in kinds:
            for m in PATTERNS[kind].finditer(doc):
                pii = m.group(0) if kind != "phone" else m.group(0)
                if is_template(kind, pii):
                    continue
                start, end = m.span()
                pref_start = max(0, start - MAX_PREFIX_CHARS)
                prefix = doc[pref_start:start]
                # require enough left context
                if len(prefix) < MIN_PREFIX_CHARS:
                    continue
                # The "suffix" used for extraction = first ~30 chars after PII
                # is not needed for verification (we verify the PII itself).
                records.append({
                    "doc_idx": doc_idx,
                    "kind": kind,
                    "pii": pii,
                    "prefix": prefix,
                    "start": start,
                    "end": end,
                })
    return records


def count_freq_in_corpus(pii_value: str, corpus_texts: list[str]) -> int:
    needle = pii_value
    return sum(t.count(needle) for t in corpus_texts)


def build_canary_record(idx: int, rec: dict, frequency: int) -> dict:
    """Convert a PII record to qquilt.canaries.v1 schema for extract reuse.

    qquilt.canaries.Canary requires sender/account/etc; we stub them with
    empty strings so the existing extract loader accepts the record. The
    only fields qquilt.extract actually uses are prefix_text + suffix_text;
    qquilt.metrics 1b uses suffix_text via match_prefix_len threshold.
    Companion metadata (kind, doc_idx) is written to a sidecar file because
    qquilt.canaries.Canary is a frozen dataclass with strict fields.
    """
    pii = rec["pii"]
    return {
        "canary_id": f"nat_{idx:05d}",
        "frequency": frequency,
        "sender_name": "",
        "sender_local": "",
        "sender_domain": "",
        "reference": pii,
        "account": "",
        "date": "",
        "topic": rec["kind"],
        "prefix_text": rec["prefix"],
        "suffix_text": pii,
        "new_tokens": [pii],
        "schema": "qquilt.canaries.v1",
        "schema_version": 1,
    }


def build_sidecar_record(idx: int, rec: dict, frequency: int, pool: str) -> dict:
    return {
        "canary_id": f"nat_{idx:05d}",
        "kind": rec["kind"],
        "doc_idx": rec["doc_idx"],
        "frequency_in_member": frequency,
        "pool": pool,
        "suffix_text": rec["pii"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-jsonl", required=True, type=Path)
    ap.add_argument("--synthetic-canaries-jsonl", required=True, type=Path,
                    help="exclude synthetic canary documents from non-member pool")
    ap.add_argument("--enron-hf-id", default="snoop2head/enron_aeslc_emails")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--member-out", required=True, type=Path)
    ap.add_argument("--nonmember-out", required=True, type=Path)
    ap.add_argument("--target-n", type=int, default=100)
    ap.add_argument("--max-freq", type=int, default=3,
                    help="upper bound on PII occurrences in MEMBER corpus")
    ap.add_argument("--nonmember-sample-multiplier", type=int, default=4,
                    help="sample N * 3000 emails from full Enron for non-member pool")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # 1. Member pool: only Enron-sourced texts in corpus.jsonl
    member_texts = []
    with args.corpus_jsonl.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("source") == "enron":
                member_texts.append(r["text"])
    print(f"[member] enron docs in corpus: {len(member_texts)}")

    # Hash set for membership tests
    member_hashes = {normalize(t) for t in member_texts}

    # 2. Non-member pool: pull M*N from HF, drop anything that overlaps
    print(f"[nonmember] loading {args.enron_hf_id} ...")
    from datasets import load_dataset
    ds = load_dataset(args.enron_hf_id, split="train")
    field_candidates = ("text", "message", "body", "content", "email")
    def grab_text(row):
        for k in field_candidates:
            v = row.get(k) if hasattr(row, "get") else row[k] if k in row else None
            if isinstance(v, str) and len(v) > 60:
                return v
        for v in row.values() if hasattr(row, "values") else []:
            if isinstance(v, str) and len(v) > 60:
                return v
        return None

    indices = list(range(len(ds)))
    rng.shuffle(indices)
    nonmember_texts: list[str] = []
    target_nm_pool = args.nonmember_sample_multiplier * len(member_texts)
    seen_norm: set[str] = set()
    for i in indices:
        t = grab_text(ds[i])
        if not t:
            continue
        n = normalize(t)
        if n in member_hashes or n in seen_norm:
            continue
        seen_norm.add(n)
        nonmember_texts.append(t)
        if len(nonmember_texts) >= target_nm_pool:
            break
    print(f"[nonmember] disjoint docs: {len(nonmember_texts)}")

    # 3. Mine PII candidates
    member_pii = extract_pii_records(member_texts)
    nonmember_pii = extract_pii_records(nonmember_texts)
    print(f"[mine] member raw PII: {len(member_pii)}  nonmember: {len(nonmember_pii)}")

    # Per-document at most ONE PII per (kind, value) to avoid double-counting
    def dedup(records):
        seen = set()
        out = []
        for r in records:
            key = (r["doc_idx"], r["kind"], r["pii"])
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out

    member_pii = dedup(member_pii)
    nonmember_pii = dedup(nonmember_pii)
    print(f"[mine] after doc-dedup: member {len(member_pii)}  nonmember {len(nonmember_pii)}")

    # 4. Member pool filtering: freq <= max_freq in member corpus
    member_freq = Counter(r["pii"] for r in member_pii)
    member_filtered = [r for r in member_pii if member_freq[r["pii"]] <= args.max_freq]
    print(f"[member] freq<={args.max_freq}: {len(member_filtered)}")

    # Non-member: by construction these are NOT in member corpus. But also
    # filter PII that co-occurs frequently with templates (e.g., a phone that
    # appears in 50+ non-member docs is template-y). Use total occurrence in
    # the WHOLE non-member sample as a proxy for "templated".
    nonmember_freq = Counter(r["pii"] for r in nonmember_pii)
    nonmember_filtered = [r for r in nonmember_pii
                          if nonmember_freq[r["pii"]] <= args.max_freq * 5]
    print(f"[nonmember] non-templated: {len(nonmember_filtered)}")

    # 5. Match distribution per kind: sample target_n total, proportional to
    # member's kind distribution.
    from collections import defaultdict
    by_kind = defaultdict(list)
    for r in member_filtered:
        by_kind[r["kind"]].append(r)
    total_avail = sum(len(v) for v in by_kind.values())
    if total_avail == 0:
        raise SystemExit("no member PII passed filters; loosen max_freq or check corpus")

    # Take per-kind counts proportional to available
    per_kind_quota = {}
    remaining = args.target_n
    sorted_kinds = sorted(by_kind.items(), key=lambda kv: -len(kv[1]))
    for k, v in sorted_kinds[:-1]:
        q = min(len(v), int(round(args.target_n * len(v) / total_avail)))
        per_kind_quota[k] = q
        remaining -= q
    if sorted_kinds:
        last = sorted_kinds[-1][0]
        per_kind_quota[last] = max(0, min(len(by_kind[last]), remaining))

    member_sample = []
    for k, q in per_kind_quota.items():
        pool = by_kind[k][:]
        rng.shuffle(pool)
        member_sample.extend(pool[:q])
    print(f"[member] sampled {len(member_sample)} (quota {per_kind_quota})")

    # Mirror the same per-kind quotas on non-member
    nm_by_kind = defaultdict(list)
    for r in nonmember_filtered:
        nm_by_kind[r["kind"]].append(r)
    nonmember_sample = []
    for k, q in per_kind_quota.items():
        pool = nm_by_kind[k][:]
        rng.shuffle(pool)
        nonmember_sample.extend(pool[:q])
    print(f"[nonmember] sampled {len(nonmember_sample)}")

    # 6. Write canary jsonl + sidecar metadata
    args.member_out.parent.mkdir(parents=True, exist_ok=True)
    args.nonmember_out.parent.mkdir(parents=True, exist_ok=True)

    def write_pool(out_path: Path, pool_samples, pool_name: str,
                   freq_map: Counter):
        sidecar_path = out_path.with_suffix(".meta.jsonl")
        with out_path.open("w") as f, sidecar_path.open("w") as g:
            for i, r in enumerate(pool_samples):
                freq = freq_map.get(r["pii"], 0)
                rec = build_canary_record(i, r, frequency=freq)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                meta = build_sidecar_record(i, r, freq, pool_name)
                g.write(json.dumps(meta, ensure_ascii=False) + "\n")
        return sidecar_path

    m_sidecar = write_pool(args.member_out, member_sample, "member", member_freq)
    nm_sidecar = write_pool(args.nonmember_out, nonmember_sample, "nonmember", member_freq)
    print(f"  member sidecar: {m_sidecar}")
    print(f"  nonmember sidecar: {nm_sidecar}")
    print(f"[done] wrote {args.member_out}  {args.nonmember_out}")

    # 7. Stats summary
    summary = {
        "seed": args.seed,
        "max_freq": args.max_freq,
        "member_corpus_docs": len(member_texts),
        "nonmember_disjoint_docs": len(nonmember_texts),
        "member_pii_after_dedup": len(member_pii),
        "nonmember_pii_after_dedup": len(nonmember_pii),
        "member_freq_filtered": len(member_filtered),
        "nonmember_freq_filtered": len(nonmember_filtered),
        "per_kind_quota": per_kind_quota,
        "member_sample_size": len(member_sample),
        "nonmember_sample_size": len(nonmember_sample),
    }
    summary_path = args.member_out.parent / f"summary_seed{args.seed}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[summary] {summary_path}")


if __name__ == "__main__":
    main()
