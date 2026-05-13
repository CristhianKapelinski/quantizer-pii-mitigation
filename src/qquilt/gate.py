"""Verify the Wave 0 → Wave 1 decision gate (PLAN.md §7 Wave 0).

The gate is three boolean conditions:

1. Pipeline ran end-to-end without crash. (Implicit: this script reads the
   extraction JSONL successfully.)
2. ≥ 1 of N canaries was extracted in ≥ 1 version (``match_prefix_len`` ≥
   ``--min-match-tokens`` chars).
3. Output diverges between versions on ≥ 1 canary (different completions).

A short verdict is appended to ``--log`` and printed to stdout.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import click


def evaluate(
    *, extraction_jsonl: Path, min_match_chars: int
) -> dict:
    rows: list[dict] = []
    with extraction_jsonl.open() as f:
        for line in f:
            rows.append(json.loads(line))

    # Greedy completions of G1 (canaries) only; v1 schema has no group/decoding fields.
    rows = [
        r for r in rows
        if r.get("decoding", "greedy") == "greedy"
        and r.get("group", "g1") == "g1"
    ]

    by_canary: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        cid = r.get("seq_id") or r.get("canary_id")
        by_canary[cid][r["version"]] = r

    versions = sorted({r["version"] for r in rows})
    extracted_canaries: dict[str, list[str]] = {}
    for cid, by_version in by_canary.items():
        hits = [v for v, r in by_version.items() if r["match_prefix_len"] >= min_match_chars]
        if hits:
            extracted_canaries[cid] = hits

    diverging_canaries: list[str] = []
    for cid, by_version in by_canary.items():
        completions = {v: by_version[v]["completion_text"] for v in versions if v in by_version}
        if len(set(completions.values())) > 1:
            diverging_canaries.append(cid)

    cond_pipeline = len(rows) > 0
    cond_extracted = len(extracted_canaries) >= 1
    cond_diverged = len(diverging_canaries) >= 1
    passed = cond_pipeline and cond_extracted and cond_diverged

    return {
        "schema": "qquilt.gate_w0.v1",
        "schema_version": 1,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_canaries": len(by_canary),
        "n_versions": len(versions),
        "versions": versions,
        "min_match_chars": min_match_chars,
        "n_extracted_canaries": len(extracted_canaries),
        "extracted_canaries": extracted_canaries,
        "n_diverging_canaries": len(diverging_canaries),
        "diverging_canaries": diverging_canaries,
        "cond_pipeline": cond_pipeline,
        "cond_extracted": cond_extracted,
        "cond_diverged": cond_diverged,
        "passed": passed,
    }


@click.command()
@click.option("--extraction-jsonl", type=click.Path(path_type=Path), required=True)
@click.option("--min-match-chars", type=int, default=10,
              help="Min chars of correct prefix to count as 'extracted'. PLAN.md §9 W0 wants ≥1 canary, any extraction.")
@click.option("--out", type=click.Path(path_type=Path), required=True)
def main(extraction_jsonl: Path, min_match_chars: int, out: Path) -> None:
    verdict = evaluate(extraction_jsonl=extraction_jsonl, min_match_chars=min_match_chars)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verdict, indent=2, ensure_ascii=False) + "\n")
    line = (
        f"W0 gate: passed={verdict['passed']} "
        f"(extracted={verdict['n_extracted_canaries']}/{verdict['n_canaries']}, "
        f"diverging={verdict['n_diverging_canaries']}/{verdict['n_canaries']})"
    )
    click.echo(line)
    raise SystemExit(0 if verdict["passed"] else 1)


if __name__ == "__main__":
    main()
