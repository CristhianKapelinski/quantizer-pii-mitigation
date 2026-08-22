#!/usr/bin/env python3
"""Print and check the pins for everything the pipeline downloads.

Offline by default: it re-hashes the committed corpora that the pipeline derived
from the external datasets and compares them with the digests recorded in
``expected/external_artifacts.json``. That is the check anchored in the run of
record — the files were written by the campaign, so a mismatch means either the
committed run of record was edited or the registry is wrong.

The revision pins for the Hugging Face ids and the digests of the published
weight archives are printed but not contacted; ``--online`` asks the Hugging Face
API whether each pinned id still resolves at the pinned revision, which is how a
future reader learns that an upstream repository has moved on.

    python scripts/check_external_artifacts.py            # offline, exits non-zero on drift
    python scripts/check_external_artifacts.py --online   # also ask the Hub about each pin
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from qquilt import external


def _print_pins(reg: dict) -> None:
    """List every pinned external id, whether the pipeline passes the pin, and why."""
    print("== revision pins ==")
    for section, kind in (("huggingface_datasets", "dataset"), ("huggingface_models", "model")):
        for hf_id, entry in reg.get(section, {}).items():
            passed = "passed to the loader" if entry.get("pinned_in_code") else "recorded only"
            print(f"  {kind:7s} {hf_id:44s} {entry['revision'][:12]}  ({passed})")
    for name, entry in reg.get("git_sources", {}).items():
        print(f"  git     {name:44s} {entry['commit'][:12]}  (tag {entry['tag']})")
    print()
    print("== published-weight digests (verified by scripts/claim_from_checkpoint.sh) ==")
    for fname, entry in reg.get("downloads", {}).get("files", {}).items():
        print(f"  {fname:36s} sha256 {entry['sha256'][:16]}...  ~{entry['size_mb_approx']} MB")


def _check_online(reg: dict) -> int:
    """Ask the Hugging Face API whether each pinned id still resolves at its
    pinned revision. Returns the number of ids that have moved on."""
    import json
    import urllib.error
    import urllib.request

    moved = 0
    print()
    print("== upstream heads, compared with the pins ==")
    for section, api in (("huggingface_datasets", "datasets"), ("huggingface_models", "models")):
        for hf_id, entry in reg.get(section, {}).items():
            url = f"https://huggingface.co/api/{api}/{hf_id}"
            try:
                with urllib.request.urlopen(url, timeout=20) as r:
                    head = json.load(r).get("sha")
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                print(f"  ?    {hf_id:44s} unreachable ({exc})")
                continue
            if head == entry["revision"]:
                print(f"  ok   {hf_id:44s} head is still the pinned revision")
            else:
                moved += 1
                print(f"  MOVED {hf_id:43s} pinned {entry['revision'][:12]} "
                      f"-> head {str(head)[:12]}")
    if moved:
        print(f"\n  {moved} upstream repository/ies moved past the pin. The pipeline still asks")
        print("  for the pinned revision, so this is information, not a failure.")
    return moved


def main() -> int:
    """Verify the committed fingerprints, print the pins, optionally query the Hub."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--online", action="store_true",
                    help="also ask the Hugging Face API whether each pin is still the head")
    args = ap.parse_args()

    reg = external.registry()
    print(f"registry: {external.REGISTRY_PATH.relative_to(external.ROOT)} "
          f"(pins recorded {reg['pins_recorded_on']})")
    print()
    print("== committed corpora derived from the external datasets ==")
    rows = external.verify_committed_fingerprints()
    bad = [(p, d) for p, ok, d in rows if not ok]
    for path, ok, detail in rows:
        print(f"  {'ok  ' if ok else 'FAIL'} {path}  {detail}")
    print(f"  {len(rows) - len(bad)}/{len(rows)} match the digests in the registry")
    print()
    _print_pins(reg)
    if args.online:
        _check_online(reg)
    if bad:
        print(f"\nFAILED: {len(bad)} committed file(s) do not match the registry.", file=sys.stderr)
        return 1
    print("\nRESULT: OK -- every committed external-data fingerprint matches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
