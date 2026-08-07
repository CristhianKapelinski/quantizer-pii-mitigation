#!/usr/bin/env python3
"""Report the checkpoint claim: the canaries recovered here against the paper's.

This claim gates on the canary set, not on a count, and it can afford to: starting from
the published weights, every step that follows is deterministic, so the same canaries must
come back on any machine. A count that matched while the identities differed would mean
something had changed, and this would catch it.

Usage: show_checkpoint_claim.py LIVE_JSONL REFERENCE_JSONL
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BAR = "═" * 66
SEP = "─" * 66
VERSION = "q4_k_m"


def extracted(path: Path, version: str | None = None) -> set[str]:
    """Canaries whose greedy continuation matched at least 10 characters."""
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("group") not in (None, "g1"):
            continue
        if version and r.get("version") != version:
            continue
        if r.get("decoding") == "greedy" and int(r.get("match_prefix_len") or 0) >= 10:
            out.add(r.get("canary_id") or r.get("seq_id"))
    return out


def main() -> int:
    live = extracted(Path(sys.argv[1]))
    paper = extracted(Path(sys.argv[2]), VERSION)

    missing, extra = paper - live, live - paper
    ok = not missing and not extra

    print()
    print(BAR)
    print("  Claim: quantizing the published weights to Q4_K_M recovers the paper's")
    print("         canaries, on a machine that never ran the fine-tune")
    print(SEP)
    print(f"  {'canaries recovered here':<38}: {len(live)}")
    print(f"  {'canaries the paper reports':<38}: {len(paper)}")
    print(f"  {'recovered by both':<38}: {len(paper & live)}")
    print(f"  {'in the paper, missing here':<38}: {len(missing)}"
          + (f"  {sorted(missing)[:6]}" if missing else ""))
    print(f"  {'here, not in the paper':<38}: {len(extra)}"
          + (f"  {sorted(extra)[:6]}" if extra else ""))
    print(SEP)
    print(f"  GATED: the two sets are identical".ljust(60) + ("OK" if ok else "FAIL"))
    print(SEP)
    print(f"  RESULT: {'OK' if ok else 'FAIL'}   "
          f"({len(paper & live)}/{len(paper)} of the paper's canaries reproduced)")
    print(BAR)
    print()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
