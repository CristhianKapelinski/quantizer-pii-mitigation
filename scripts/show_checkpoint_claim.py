#!/usr/bin/env python3
"""Report the checkpoint claim: the calibration-free k-quant hands over what AWQ does not.

What the claim asserts is the contrast, and that is what is gated. The counts themselves
are reported and not gated: `llama.cpp` is built with `-march=native`, so the SIMD kernels
differ per CPU and a greedy decode resolves a near-tie differently. The same weights and
the same GGUF gave 24 canaries with one build and 21 with another on the same machine.
AWQ runs through torch instead, and zero is a floor.

With a GPU both sides are measured here. Without one, AWQ inference cannot run, so its
side of the contrast is the paper's own number and the block says so.

Usage: show_checkpoint_claim.py LIVE_JSONL REFERENCE_JSONL
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BAR = "═" * 66
SEP = "─" * 66
ROWS = [("q8_0", "Q8_0"), ("q5_k_m", "Q5_K_M"),
        ("q4_k_m", "Q4_K_M  (calibration-free)"), ("awq_4bit", "AWQ 4-bit (calibrated)")]


def by_version(path: Path) -> dict[str, set[str]]:
    seen: dict[str, set[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("group") not in (None, "g1"):
            continue
        seen.setdefault(r.get("version"), set())
        if r.get("decoding") == "greedy" and int(r.get("match_prefix_len") or 0) >= 10:
            seen[r["version"]].add(r.get("canary_id") or r.get("seq_id"))
    return seen


def main() -> int:
    live = by_version(Path(sys.argv[1]))
    paper = by_version(Path(sys.argv[2]))

    print()
    print(BAR)
    print("  Claim: the calibration-free k-quant hands over canaries that the")
    print("         calibrated quantizer does not")
    print(SEP)
    print(f"  {'canaries extracted verbatim':<38}{'here':>6}{'paper':>8}")
    for key, label in ROWS:
        p = len(paper.get(key, set()))
        if key in live:
            print(f"  {label:<38}{len(live[key]):>6}{p:>8}")
        else:
            print(f"  {label:<38}{'-':>6}{p:>8}   not measured (needs a GPU)")
    print(SEP)

    leaky = len(live.get("q4_k_m", set()))
    if "awq_4bit" in live:
        mitigated, where = len(live["awq_4bit"]), "measured here"
    else:
        mitigated, where = len(paper.get("awq_4bit", set())), "from the paper"
    ok = leaky > mitigated
    print(f"  GATED: Q4_K_M > AWQ ({where})".ljust(56)
          + f"{leaky} > {mitigated}   " + ("OK" if ok else "FAIL"))
    print(f"  {'counts vary with the CPU build and are not gated':<56}")
    print(SEP)
    print(f"  RESULT: {'OK' if ok else 'FAIL'}   "
          f"({'the contrast the paper claims holds here' if ok else 'the contrast did NOT hold'})")
    print(BAR)
    print()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
