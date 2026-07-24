#!/usr/bin/env python3
"""Compare the metrics recomputed by replay.sh against the committed ones.

`replay.sh` re-runs the analysis over the committed extraction logs and writes
`*.replay.json` next to each committed result file. The analysis is
deterministic, so every recomputed field must equal the committed one; this
script is the assertion that says so, and it is what makes the replay path a
check rather than a printout.

A committed file may carry EXTRA keys that the replay does not recompute (the
`gate_w1_mini` block, which `qquilt.metrics` only writes when called with
`--include-w1-mini-gate`); extra keys on the committed side are reported and
allowed. Any key the replay did recompute must match exactly.

    python scripts/check_replay_equal.py [--results experiment/results]

Exit 0 when every recomputed file matches, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

# Committed counterparts that do not follow the `<name>.replay.json` ->
# `<name>.json` rule (historical file names of the first Llama/Qwen cells).
FALLBACK_NAMES = {"metrics.json": "metrics_w1_mini.json"}


def committed_for(replay: pathlib.Path) -> pathlib.Path | None:
    """Path of the committed file a `*.replay.json` should equal, if present."""
    direct = replay.with_name(replay.name.replace(".replay.json", ".json"))
    if direct.exists():
        return direct
    fallback = FALLBACK_NAMES.get(direct.name)
    if fallback and (alt := replay.with_name(fallback)).exists():
        return alt
    return None


def _short(value, limit: int = 160) -> str:
    """One-line, length-capped rendering: these payloads hold per-canary dicts."""
    s = json.dumps(value, default=str)
    return s if len(s) <= limit else s[:limit] + " ..."


def diff_keys(recomputed: dict, committed: dict) -> list[str]:
    """Keys present in both whose values differ (recursion is not needed: the
    metrics payloads are one level of nested plain JSON compared as a whole)."""
    return sorted(k for k in recomputed if k in committed and recomputed[k] != committed[k])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="experiment/results")
    a = ap.parse_args()
    root = pathlib.Path(a.results)

    replays = sorted(root.glob("**/*.replay.json"))
    if not replays:
        print("check: no *.replay.json found -- run `bash replay.sh` first", file=sys.stderr)
        return 1

    ok, orphan, bad = 0, [], []
    for replay in replays:
        ref = committed_for(replay)
        if ref is None:
            orphan.append(replay)
            continue
        recomputed = json.load(open(replay))
        committed = json.load(open(ref))
        missing = sorted(k for k in recomputed if k not in committed)
        changed = diff_keys(recomputed, committed)
        if missing or changed:
            bad.append((replay, ref, missing, changed))
        else:
            ok += 1
            extra = sorted(k for k in committed if k not in recomputed)
            note = f"  (committed also has: {', '.join(extra)})" if extra else ""
            print(f"  match  {ref}{note}")

    for replay in orphan:
        print(f"  no committed counterpart for {replay} (nothing to compare)")
    for replay, ref, missing, changed in bad:
        print(f"  MISMATCH {ref} vs {replay}")
        if missing:
            print(f"    keys absent from the committed file: {', '.join(missing)}")
        committed, recomputed = json.load(open(ref)), json.load(open(replay))
        for k in changed:
            print(f"    {k}: committed={_short(committed[k])} recomputed={_short(recomputed[k])}")

    print(f"check: {ok} recomputed file(s) identical to the committed ones, "
          f"{len(bad)} mismatched, {len(orphan)} without a counterpart")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
