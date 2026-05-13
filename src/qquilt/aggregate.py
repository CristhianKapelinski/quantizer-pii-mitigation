"""Multi-seed / multi-checkpoint metrics aggregation.

Used in W1 full and W2 to combine metrics_<name>.json files (one per
seed × model × calibration variant) into a single reportable summary
with mean ± std and Wilcoxon signed-rank where applicable.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import click


def _load_metrics_files(paths: list[Path]) -> list[dict]:
    return [json.loads(p.read_text()) for p in paths]


def aggregate_m1b(metrics_list: list[dict]) -> dict:
    """Aggregate Métrica 1b across seeds: revealed_share mean ± std,
    plus per-bucket revealed counts."""
    shares = []
    per_bucket: dict[str, list[int]] = defaultdict(list)
    for m in metrics_list:
        m1b = m.get("metric_1b", {})
        shares.append(float(m1b.get("revealed_share_of_extracted", 0.0)))
        for b, d in m1b.get("per_bucket", {}).items():
            per_bucket[str(b)].append(len(d.get("revealed", [])))

    return {
        "schema": "qquilt.aggregate.m1b.v1",
        "schema_version": 1,
        "n_runs": len(metrics_list),
        "revealed_share": {
            "mean": statistics.mean(shares) if shares else 0.0,
            "std": statistics.stdev(shares) if len(shares) >= 2 else 0.0,
            "values": shares,
        },
        "revealed_per_bucket": {
            b: {
                "mean": statistics.mean(v),
                "std": statistics.stdev(v) if len(v) >= 2 else 0.0,
                "values": v,
            }
            for b, v in sorted(per_bucket.items())
        },
    }


def aggregate_m1(metrics_list: list[dict]) -> dict:
    """Aggregate Métrica 1 amplification ratio across seeds, per bucket."""
    per_bucket_ratios: dict[str, list[float]] = defaultdict(list)
    for m in metrics_list:
        m1 = m.get("metric_1", {})
        for b, d in m1.get("per_bucket", {}).items():
            r = d.get("amplification_ratio")
            if r is not None:
                per_bucket_ratios[str(b)].append(float(r))
    return {
        "schema": "qquilt.aggregate.m1.v1",
        "schema_version": 1,
        "n_runs": len(metrics_list),
        "amplification_ratio_per_bucket": {
            b: {
                "mean": statistics.mean(v),
                "std": statistics.stdev(v) if len(v) >= 2 else 0.0,
                "values": v,
            }
            for b, v in sorted(per_bucket_ratios.items())
        },
    }


def wilcoxon_paired(a: list[float], b: list[float]) -> dict:
    """Wilcoxon signed-rank paired test of a vs b. Returns
    ``{statistic, pvalue}`` or ``None`` keys when scipy is unavailable
    or arrays are degenerate."""
    out: dict = {"n": min(len(a), len(b)), "scipy_available": True}
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        out["scipy_available"] = False
        return out
    if len(a) != len(b) or len(a) < 6:
        out["note"] = "need ≥6 paired values for Wilcoxon to be meaningful"
        return out
    res = wilcoxon(a, b, alternative="greater")
    out["statistic"] = float(res.statistic)
    out["pvalue"] = float(res.pvalue)
    return out


@click.command()
@click.option("--metrics", "metrics_paths", type=click.Path(path_type=Path),
              multiple=True, required=True,
              help="Repeat for each metrics_<name>.json file (one per seed/checkpoint).")
@click.option("--out", type=click.Path(path_type=Path), required=True)
def main(metrics_paths: tuple[Path, ...], out: Path) -> None:
    metrics_list = _load_metrics_files(list(metrics_paths))
    payload = {
        "n_runs": len(metrics_list),
        "input_files": [str(p) for p in metrics_paths],
        "metric_1": aggregate_m1(metrics_list),
        "metric_1b": aggregate_m1b(metrics_list),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    click.echo(f"aggregated {len(metrics_list)} runs to {out}")


if __name__ == "__main__":
    main()
