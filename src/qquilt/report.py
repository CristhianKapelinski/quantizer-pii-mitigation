"""Templated RESULTS.md generator.

Reads a ``qquilt.metrics`` JSON (with optional gate) and the matching
extraction / canaries JSONL, and emits a wave-shaped RESULTS.md skeleton.
The user fills in narrative interpretation; raw numbers are pulled from
the JSON / JSONL.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import click


def _per_canary_table(extraction_jsonl: Path, canaries_jsonl: Path) -> str:
    suffix_lens = {}
    with canaries_jsonl.open() as f:
        for line in f:
            r = json.loads(line)
            suffix_lens[r["canary_id"]] = len(r["suffix_text"])

    by: dict[str, dict[str, dict]] = defaultdict(dict)
    with extraction_jsonl.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("group", "g1") != "g1":
                continue
            if r.get("decoding", "greedy") != "greedy":
                continue
            sid = r.get("seq_id") or r.get("canary_id")
            by[sid][r["version"]] = r

    versions = sorted({v for d in by.values() for v in d})

    rows_with_hits = sorted(
        ((sid, d) for sid, d in by.items() if any(r["match_prefix_len"] >= 5 for r in d.values())),
        key=lambda x: -max(r["match_prefix_len"] for r in x[1].values()),
    )

    if not rows_with_hits:
        return "_No canary reached `match_prefix_len ≥ 5`._"

    head = "| canary | bucket | suffix_len | " + " | ".join(versions) + " |"
    sep = "|" + "|".join("---" for _ in range(3 + len(versions))) + "|"
    lines = [head, sep]
    for sid, d in rows_with_hits[:25]:
        any_row = next(iter(d.values()))
        bucket = any_row.get("bucket", "—")
        sl = suffix_lens.get(sid, 0)
        cells = []
        for v in versions:
            if v not in d:
                cells.append("—")
                continue
            ml = d[v]["match_prefix_len"]
            mark = " ✓" if d[v]["exact_match"] else ""
            cells.append(f"{ml}/{sl}{mark}")
        lines.append(f"| {sid} | {bucket} | {sl} | " + " | ".join(cells) + " |")
    if len(rows_with_hits) > 25:
        lines.append(f"\n_… and {len(rows_with_hits) - 25} more canaries with extraction signal_")
    return "\n".join(lines)


def _train_telemetry_summary(train_jsonl: Path) -> str:
    if not train_jsonl.exists():
        return "_no train telemetry available_"
    lines = [json.loads(l) for l in train_jsonl.open()]
    banner = next((d for d in lines if d.get("schema") == "qquilt.train.banner.v1"), None)
    summary = next((d for d in lines if d.get("schema") == "qquilt.train.summary.v1"), None)
    steps = [d for d in lines if d.get("schema") == "qquilt.train.v1" and d.get("loss") is not None]
    if not steps or not summary:
        return "_partial train telemetry_"

    out = [
        "```",
        f"model:        {banner['model_id']}  (rev banner sha {banner.get('nvidia_smi_sha256','?')[:8]})",
        f"hp:           bs {banner['batch_size']} × grad_accum {banner['grad_accum']} = effective {banner['effective_batch']}",
        f"              lr {banner['lr']}, {banner['epochs']} epochs, max_seq {banner['max_seq_len']}, BF16",
        f"steps:        {len(steps)} in {summary['total_wallclock_s']:.1f} s ({summary['total_wallclock_s']/len(steps):.2f} s/step avg)",
        f"loss:         {steps[0]['loss']:.3f} → {steps[-1]['loss']:.3f}   (min {min(s['loss'] for s in steps):.3f})",
        f"grad_norm:    {steps[0]['grad_norm']:.1f} → {steps[-1]['grad_norm']:.2f}",
    ]
    if any("gpu_peak_alloc_gib" in s for s in steps):
        out += [
            f"peak GPU:     {max(s.get('gpu_peak_alloc_gib', 0) for s in steps):.2f} GiB allocated / "
            f"{max(s.get('gpu_peak_reserved_gib', 0) for s in steps):.2f} GiB reserved",
        ]
    if any("max_rss_gib" in s for s in steps):
        out.append(f"peak RSS:     {max(s.get('max_rss_gib', 0) for s in steps):.2f} GiB")
    out += [
        f"device:       {banner.get('device_name', '?')}  (sm_{''.join(map(str, banner.get('device_capability', [])))})",
        f"torch:        {banner.get('torch', '?')}",
        "```",
    ]
    return "\n".join(out)


def _gate_block(metrics: dict) -> str:
    g = metrics.get("gate_w1_mini") or metrics.get("gate_w0")
    if not g:
        return "_no gate verdict in metrics file_"
    out = [f"**{g.get('schema', 'gate')}**:", "", "```"]
    for k, v in g.items():
        if k.startswith("schema"):
            continue
        out.append(f"{k:30} = {v}")
    out.append("```")
    return "\n".join(out)


def _metric_block(name: str, m: dict) -> str:
    if not m:
        return f"_metric {name} absent_"
    out = [f"**{name} ({m.get('schema', 'unknown schema')})**:", "", "```"]
    skip = {"schema", "schema_version"}
    for k, v in m.items():
        if k in skip:
            continue
        if isinstance(v, (list, dict)) and len(json.dumps(v)) > 100:
            out.append(f"{k}: {type(v).__name__} (len={len(v)})")
        else:
            out.append(f"{k}: {v}")
    out.append("```")
    return "\n".join(out)


def render(
    *, wave_label: str, metrics_path: Path, extraction_jsonl: Path,
    canaries_jsonl: Path, train_jsonl: Path | None,
) -> str:
    metrics = json.loads(metrics_path.read_text())
    sections: list[str] = [
        f"# {wave_label} — Results",
        "",
        "_Auto-generated from `qquilt.report`. Narrative interpretation goes after the data blocks._",
        "",
        "## Headline gate",
        "",
        _gate_block(metrics),
        "",
        "## Per-canary × per-version (greedy, G1 only, ≥5-char hits)",
        "",
        _per_canary_table(extraction_jsonl, canaries_jsonl),
        "",
        "## Métricas",
        "",
        "### Métrica 1 — amplification",
        "",
        _metric_block("metric_1", metrics.get("metric_1", {})),
        "",
        "### Métrica 1b — quantization-revealed (L3)",
        "",
        _metric_block("metric_1b", metrics.get("metric_1b", {})),
        "",
        "### Métrica 1c — quilt-statistic (text-stub at smoke level)",
        "",
        _metric_block("metric_1c", metrics.get("metric_1c", {})),
    ]
    if train_jsonl is not None:
        sections += [
            "",
            "## Fine-tune telemetry",
            "",
            _train_telemetry_summary(train_jsonl),
        ]
    sections += [
        "",
        "## Interpretation",
        "",
        "_TODO: fill in honest interpretation per `honest_smoke_interpretation` memory rule._",
        "",
        "## Implications for next wave",
        "",
        "_TODO: confirmed / not observed / hardware-derived constraints._",
    ]
    return "\n".join(sections) + "\n"


@click.command()
@click.option("--wave-label", type=str, required=True, help="e.g. 'Wave 1 mini Phase A'")
@click.option("--metrics", "metrics_path", type=click.Path(path_type=Path), required=True)
@click.option("--extraction-jsonl", type=click.Path(path_type=Path), required=True)
@click.option("--canaries-jsonl", type=click.Path(path_type=Path), required=True)
@click.option("--train-jsonl", type=click.Path(path_type=Path), default=None)
@click.option("--out", type=click.Path(path_type=Path), required=True)
def main(wave_label: str, metrics_path: Path, extraction_jsonl: Path,
         canaries_jsonl: Path, train_jsonl: Path | None, out: Path) -> None:
    text = render(
        wave_label=wave_label, metrics_path=metrics_path,
        extraction_jsonl=extraction_jsonl, canaries_jsonl=canaries_jsonl,
        train_jsonl=train_jsonl,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    click.echo(f"wrote {len(text)} chars to {out}")


if __name__ == "__main__":
    main()
