#!/usr/bin/env python3
"""Report the end-to-end pipeline claim as a framed block with a verdict.

The run this reports on trains the model, quantizes the result five ways, attacks
each one with the same extraction prompts, and counts how many planted canaries come
back verbatim. What it asserts is a *direction*: the calibration-free k-quant gives up
more canaries than the calibration-based AWQ, on the reviewer's own hardware.

The magnitudes are printed next to the committed run for context and are deliberately
NOT gated. Fine-tuning is not bit-deterministic, so a re-run memorizes a different
subset of the canaries and each quantizer then destroys a different subset of those;
the size of the gap moves between GPUs even though its sign does not. Gating the
counts would fail honest re-runs, and gating nothing would prove nothing, so the gate
is the comparison the paper actually claims.

Counting is delegated to ``greedy_ge10`` in ``verify_values.py``, the same function the
141-value replay uses, so this claim and Claim #1 cannot disagree about what counts as
an extraction.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_values import greedy_ge10  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BAR = "═" * 66
SEP = "─" * 66

# Printed in this order: the unquantized baseline first, then the quantizers from the
# one that barely changes the weights to the one that changes them most.
VERSIONS = [
    ("bf16", "BF16 (not quantized)"),
    ("q8_0", "Q8_0"),
    ("q5_k_m", "Q5_K_M"),
    ("q4_k_m", "Q4_K_M  (calibration-free)"),
    ("awq_4bit", "AWQ 4-bit (calibrated)"),
]
LEAKY, MITIGATED = "q4_k_m", "awq_4bit"


def counts(path: Path) -> dict[str, int] | None:
    """Canaries extracted verbatim per version, or None when the log is absent.

    ``greedy_ge10`` returns no key at all for a version that extracted nothing, and
    zero extractions is precisely the result this claim exists to show. So the count
    is filled in from the versions the log actually contains: absent-from-the-log and
    present-but-never-matched are different facts and must not print the same way.
    """
    if not path.is_file():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    matched = greedy_ge10(rows)
    present = {r.get("version") for r in rows if r.get("version")}
    return {v: matched.get(v, 0) for v in present}


def main() -> int:
    live_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        ROOT / "experiment/results/wave_1_qwen05b_seed42_rerun"
    ref_dir = ROOT / "experiment/results/wave_1_qwen05b_seed42"

    live = counts(live_dir / "extraction.jsonl")
    if live is None:
        print(f"no extraction log under {live_dir}; run `bash reproduce.sh quick` first",
              file=sys.stderr)
        return 2
    ref = counts(ref_dir / "extraction.jsonl") or {}

    print()
    print(BAR)
    print("  Claim: the calibration-free k-quant leaks more than the calibrated one")
    print("  (measured end to end on this machine: fine-tune, quantize, attack)")
    print(SEP)
    print(f"  {'canaries extracted verbatim':<31}  {'this run':>9}  {'run of record':>13}")
    for key, label in VERSIONS:
        here = live.get(key)
        there = ref.get(key)
        print(f"  {label:<31}  {('-' if here is None else here):>9}  {('-' if there is None else there):>13}")
    print(SEP)

    leaky, mitigated = live.get(LEAKY), live.get(MITIGATED)
    if leaky is None or mitigated is None:
        print("  the run did not produce both quantizers; nothing to compare", file=sys.stderr)
        return 2
    ok = leaky > mitigated
    print(f"  {'GATED: Q4_K_M > AWQ on this machine':<48}"
          f"{leaky} > {mitigated}   " + ("OK" if ok else "FAIL"))
    print(f"  {'counts above are hardware-dependent and not gated':<48}")
    print(SEP)
    print(f"  {'source of these numbers':<31}: measured on this machine just now")
    elapsed = os.environ.get("QQUILT_CLAIM_ELAPSED")
    if elapsed:
        print(f"  {'wall clock on this machine':<31}: {elapsed} s")
    print(SEP)
    print(f"  RESULT: {'OK' if ok else 'FAIL'}   "
          f"({'the direction the paper claims holds here' if ok else 'the direction did NOT hold'})")
    print(BAR)
    print()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
