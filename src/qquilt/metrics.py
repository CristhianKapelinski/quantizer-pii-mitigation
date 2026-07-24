"""Verbatim-extraction metrics over a multi-version extraction log.

* Metric 1b — quantization-revealed canaries: the subset of canaries extracted
  by at least one quantized version but NOT by the BF16 baseline.
* Metric 1c — cross-version disagreement: a text-level agreement count over the
  completions of K versions of the same canary.

Both functions consume the ``qquilt.extract.v1`` JSONL produced by
``qquilt.extract`` and emit small JSON-able dicts. The paper's headline number
is the per-version count of canaries whose greedy completion matches at least
``--min-match-chars`` characters of the reference field.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import click


@dataclass(frozen=True)
class Row:
    seq_id: str
    group: str  # "g1" | "g2" | "g3"
    bucket: int | None  # canary frequency bucket; None for G2/G3
    version: str
    decoding: str  # "greedy" | "stochastic"
    completion_index: int
    completion_text: str
    match_prefix_len: int
    suffix_text: str
    exact_match: bool


def _load(extraction_jsonl: Path, canaries_jsonl: Path | None = None) -> list[Row]:
    """Read extraction JSONL; tolerates both the v1 and v2 schemas."""
    suffixes: dict[str, str] = {}
    if canaries_jsonl is not None:
        with canaries_jsonl.open() as f:
            for line in f:
                row = json.loads(line)
                suffixes[row["canary_id"]] = row["suffix_text"]
    out: list[Row] = []
    with extraction_jsonl.open() as f:
        for line in f:
            r = json.loads(line)
            seq_id = r.get("seq_id") or r.get("canary_id")
            group = r.get("group", "g1")
            out.append(
                Row(
                    seq_id=seq_id,
                    group=group,
                    bucket=r.get("bucket"),
                    version=r["version"],
                    decoding=r.get("decoding", "greedy"),
                    completion_index=int(r.get("completion_index", 0)),
                    completion_text=r["completion_text"],
                    match_prefix_len=int(r["match_prefix_len"]),
                    suffix_text=suffixes.get(seq_id, ""),
                    exact_match=bool(r["exact_match"]),
                )
            )
    return out


def metric_1b_quantization_revealed(
    rows: list[Row], baseline_version: str, min_match_chars: int,
) -> dict:
    """Canaries extracted by ≥1 quantized version but NOT by the baseline.

    Operates on greedy completions of G1 sequences only (canaries). Returns
    the strict-difference set ``revealed`` (L3 candidates) plus per-bucket
    breakdowns when ``bucket`` is populated, and the reciprocal
    ``lost_in_all_quantized`` (L2-strict-with-threshold).
    """
    g1_rows = [r for r in rows if r.group == "g1" and r.decoding == "greedy"]
    by_canary: dict[str, dict[str, Row]] = defaultdict(dict)
    for r in g1_rows:
        by_canary[r.seq_id][r.version] = r

    versions = sorted({r.version for r in g1_rows})
    quantized = [v for v in versions if v != baseline_version]

    buckets: dict[int, list[str]] = defaultdict(list)
    for cid, by_v in by_canary.items():
        any_row = next(iter(by_v.values()))
        if any_row.bucket is not None:
            buckets[any_row.bucket].append(cid)

    def _classify(cid: str, by_v: dict[str, Row]) -> dict[str, bool]:
        base_hit = (
            baseline_version in by_v
            and by_v[baseline_version].match_prefix_len >= min_match_chars
        )
        any_q_hit = any(
            v in by_v and by_v[v].match_prefix_len >= min_match_chars
            for v in quantized
        )
        return {
            "base_hit": base_hit,
            "any_q_hit": any_q_hit,
            "revealed": any_q_hit and not base_hit,
            "lost_in_all_q": base_hit and not any_q_hit,
        }

    revealed: list[str] = []
    lost_in_all_quantized: list[str] = []
    extracted_by_baseline: list[str] = []
    extracted_by_any_quant: list[str] = []
    extracted_by_any: list[str] = []

    per_bucket = {b: {"n": len(buckets[b]), "revealed": [], "extracted_by_any": []}
                  for b in sorted(buckets)}

    for cid, by_v in by_canary.items():
        cls = _classify(cid, by_v)
        if cls["base_hit"]:
            extracted_by_baseline.append(cid)
        if cls["any_q_hit"]:
            extracted_by_any_quant.append(cid)
        if cls["base_hit"] or cls["any_q_hit"]:
            extracted_by_any.append(cid)
        if cls["revealed"]:
            revealed.append(cid)
        if cls["lost_in_all_q"]:
            lost_in_all_quantized.append(cid)
        any_row = next(iter(by_v.values()))
        if any_row.bucket is not None:
            b = any_row.bucket
            if cls["base_hit"] or cls["any_q_hit"]:
                per_bucket[b]["extracted_by_any"].append(cid)
            if cls["revealed"]:
                per_bucket[b]["revealed"].append(cid)

    return {
        "schema": "qquilt.metric_1b.v2",
        "schema_version": 2,
        "baseline_version": baseline_version,
        "quantized_versions": quantized,
        "min_match_chars": min_match_chars,
        "n_canaries": len(by_canary),
        "extracted_by_baseline": extracted_by_baseline,
        "extracted_by_any_quantized": extracted_by_any_quant,
        "extracted_by_any": extracted_by_any,
        "quantization_revealed": revealed,
        "lost_in_all_quantized": lost_in_all_quantized,
        "revealed_share_of_extracted": (
            len(revealed) / len(extracted_by_any) if extracted_by_any else 0.0
        ),
        "per_bucket": per_bucket,
    }


def _token_agreement_per_position(rows: list[Row]) -> dict[str, list[int]]:
    """For each sequence, count of versions whose completion matches the
    ground-truth suffix at each character position (text-level stand-in
    for token-level surprisal)."""
    by_seq: dict[str, dict[str, Row]] = defaultdict(dict)
    for r in rows:
        by_seq[r.seq_id][r.version] = r
    agreement: dict[str, list[int]] = {}
    for sid, by_v in by_seq.items():
        any_row = next(iter(by_v.values()))
        suffix = any_row.suffix_text
        if not suffix:
            continue
        agreement[sid] = [
            sum(1 for r in by_v.values()
                if i < len(r.completion_text) and r.completion_text[i] == ch)
            for i, ch in enumerate(suffix)
        ]
    return agreement


def metric_1c_quilt_stat_text(rows: list[Row]) -> dict:
    """Metric 1c: text-level agreement variance across K versions.

    For each canary, compute per-position agreement counts ``a_i`` (number
    of versions whose completion matches the ground-truth suffix at
    position i). The quilt-statistic is the *variance* of ``a_i`` across
    positions: a high variance means versions disagree on which positions
    are easy. Compare against the maximum pairwise mismatch count to give
    a first-pass strict-inequality signal.

    This is a text-level statistic over completions; no number reported in
    the paper depends on it.
    """
    # Use greedy completions of G1 only for canary-focused 1c.
    g1_rows = [r for r in rows if r.group == "g1" and r.decoding == "greedy"]
    by_seq: dict[str, dict[str, Row]] = defaultdict(dict)
    for r in g1_rows:
        by_seq[r.seq_id][r.version] = r
    versions = sorted({r.version for r in g1_rows})
    if len(versions) < 3:
        return {
            "schema": "qquilt.metric_1c.v1",
            "schema_version": 1,
            "note": "fewer than 3 versions; ≥3-version quilt-statistic not applicable",
            "n_versions": len(versions),
            "n_canaries": len(by_seq),
        }

    agreement = _token_agreement_per_position(g1_rows)
    quilt_var: dict[str, float] = {}
    for sid, vec in agreement.items():
        if not vec:
            continue
        n = len(vec)
        mean = sum(vec) / n
        var = sum((v - mean) ** 2 for v in vec) / n
        quilt_var[sid] = round(var, 4)

    pairwise: dict[tuple[str, str], dict[str, int]] = {}
    for v1, v2 in combinations(versions, 2):
        diffs: dict[str, int] = {}
        for sid, by_v in by_seq.items():
            if v1 not in by_v or v2 not in by_v:
                continue
            c1, c2 = by_v[v1].completion_text, by_v[v2].completion_text
            n = min(len(c1), len(c2))
            d = sum(1 for i in range(n) if c1[i] != c2[i]) + abs(len(c1) - len(c2))
            diffs[sid] = d
        pairwise[(v1, v2)] = diffs

    return {
        "schema": "qquilt.metric_1c.v1",
        "schema_version": 1,
        "n_versions": len(versions),
        "n_canaries": len(by_seq),
        "versions": versions,
        "quilt_variance_per_canary": quilt_var,
        "pairwise_diff_counts": {
            f"{v1}__{v2}": d for (v1, v2), d in pairwise.items()
        },
        # Kept verbatim: this string is a field of the committed metrics.json files.
        "note": "W0/W1-mini text-level stub; W1-full replaces with per-token "
                "surprisal variance from logits.",
    }


def amplification_ratio(rows: list[Row], baseline_version: str, min_match_chars: int) -> dict:
    """Metric 1: aggregated/best-single extraction ratio per bucket.

    Aggregated = canary extracted iff ≥1 version reaches threshold.
    Best single = max over versions (incl. baseline) of binary extraction.
    A1 = aggregated / best_single, computed per frequency bucket.

    Returns counts and ratios; the paired McNemar test compares aggregated
    vs baseline-only extraction across canaries within each bucket.
    """
    g1_rows = [r for r in rows if r.group == "g1" and r.decoding == "greedy"]
    by_canary: dict[str, dict[str, Row]] = defaultdict(dict)
    for r in g1_rows:
        by_canary[r.seq_id][r.version] = r

    versions = sorted({r.version for r in g1_rows})

    by_bucket: dict[int | None, list[str]] = defaultdict(list)
    for cid, by_v in by_canary.items():
        any_row = next(iter(by_v.values()))
        by_bucket[any_row.bucket].append(cid)

    out_per_bucket: dict[str, dict] = {}
    for bucket, cids in sorted(by_bucket.items(), key=lambda x: (x[0] is None, x[0] or 0)):
        per_version_count: dict[str, int] = {v: 0 for v in versions}
        union_set: set[str] = set()
        b_only_agg = 0
        b_only_baseline = 0
        for cid in cids:
            by_v = by_canary[cid]
            base_hit = (
                baseline_version in by_v
                and by_v[baseline_version].match_prefix_len >= min_match_chars
            )
            any_hit_versions = [
                v for v in versions
                if v in by_v and by_v[v].match_prefix_len >= min_match_chars
            ]
            for v in any_hit_versions:
                per_version_count[v] += 1
            any_hit = bool(any_hit_versions)
            if any_hit:
                union_set.add(cid)
            if any_hit and not base_hit:
                b_only_agg += 1
            if base_hit and not any_hit:
                b_only_baseline += 1
        union_count = len(union_set)
        max_single_count = max(per_version_count.values()) if per_version_count else 0
        ratio = union_count / max_single_count if max_single_count else None
        out_per_bucket[str(bucket)] = {
            "n": len(cids),
            "per_version_extracted": per_version_count,
            "union_extracted": union_count,
            "max_single_extracted": max_single_count,
            "amplification_ratio": ratio,
            "mcnemar_b": b_only_agg,
            "mcnemar_c": b_only_baseline,
        }

    return {
        "schema": "qquilt.metric_1.v1",
        "schema_version": 1,
        "baseline_version": baseline_version,
        "min_match_chars": min_match_chars,
        "per_bucket": out_per_bucket,
        "note": (
            "amplification_ratio = 1.0 means no aggregation effect; ratio > 1 "
            "indicates the union of versions extracts more than any single one. "
            "Wilcoxon / bootstrap stats on the per-canary table belong to W1 full."
        ),
    }


def compute_w1_mini_gate(m1b: dict, m1c: dict) -> dict:
    """Pilot-to-full-run gate verdict, kept because it is a field of the
    committed metrics_w1_mini.json files (written only with --include-w1-mini-gate).

    PASS conditions (any of):

    * A: ``M1b ≥ 1`` in at least one freq bucket of {3, 10} — L3 emerges
      in the sub-memorized regime.
    * B: ``revealed_share_of_extracted ≥ 0.05`` overall — L3 share is
      non-trivial across all extracted canaries.

    FAIL otherwise.
    """
    revealed_per_bucket = {
        b: len(d.get("revealed", []))
        for b, d in m1b.get("per_bucket", {}).items()
    }
    cond_a = any(revealed_per_bucket.get(b, 0) >= 1 for b in (3, 10))
    cond_b = float(m1b.get("revealed_share_of_extracted", 0.0)) >= 0.05
    passed = cond_a or cond_b
    return {
        "schema": "qquilt.gate_w1_mini.v1",
        "schema_version": 1,
        "revealed_per_bucket": revealed_per_bucket,
        "revealed_share_of_extracted": m1b.get("revealed_share_of_extracted", 0.0),
        "cond_A_l3_in_low_freq_bucket": cond_a,
        "cond_B_l3_share_geq_5pct": cond_b,
        "passed": passed,
        "note": (
            "W1-mini gate is permissive by design: any L3 signal in low-freq "
            "buckets justifies the full W1 commitment."
        ),
    }


@click.command()
@click.option("--extraction-jsonl", type=click.Path(path_type=Path), required=True)
@click.option("--canaries-jsonl", type=click.Path(path_type=Path), required=True)
@click.option("--baseline-version", type=str, default="bf16")
@click.option("--min-match-chars", type=int, default=10)
@click.option("--out", type=click.Path(path_type=Path), required=True)
@click.option("--include-w1-mini-gate", is_flag=True, default=False,
              help="Also compute and append the W1-mini → W1-full gate verdict.")
def main(
    extraction_jsonl: Path, canaries_jsonl: Path, baseline_version: str,
    min_match_chars: int, out: Path, include_w1_mini_gate: bool,
) -> None:
    """Compute metrics 1, 1b and 1c and write a single JSON file."""
    rows = _load(extraction_jsonl, canaries_jsonl)
    m1 = amplification_ratio(rows, baseline_version=baseline_version,
                             min_match_chars=min_match_chars)
    m1b = metric_1b_quantization_revealed(
        rows, baseline_version=baseline_version, min_match_chars=min_match_chars
    )
    m1c = metric_1c_quilt_stat_text(rows)
    payload = {"metric_1": m1, "metric_1b": m1b, "metric_1c": m1c}
    if include_w1_mini_gate:
        payload["gate_w1_mini"] = compute_w1_mini_gate(m1b, m1c)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    click.echo(
        f"metric 1b: revealed={len(m1b['quantization_revealed'])}/{m1b['n_canaries']} "
        f"(share_of_extracted={m1b['revealed_share_of_extracted']:.3f}); "
        f"metric 1c: n_versions={m1c['n_versions']}"
    )
    if include_w1_mini_gate:
        g = payload["gate_w1_mini"]
        click.echo(f"pilot gate: passed={g['passed']} "
                   f"(cond_A={g['cond_A_l3_in_low_freq_bucket']}, "
                   f"cond_B={g['cond_B_l3_share_geq_5pct']})")


if __name__ == "__main__":
    main()
