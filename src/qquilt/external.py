"""Pins for everything the pipeline pulls from outside this repository.

The registry lives in ``expected/external_artifacts.json``; this module is the
only reader. It answers two questions:

* *Which revision should this download be resolved at?* — ``hf_revision`` is
  passed to ``load_dataset`` / ``from_pretrained`` so an upstream write after
  the pin cannot change a from-scratch run without the id failing to resolve.
* *Is the external data still the data the paper measured?* — ``check_derived``
  compares a corpus the pipeline just rebuilt against the digest of the copy
  committed under ``experiment/results/``. That digest was produced by the run
  of record, so a mismatch means the upstream dataset drifted, and it says so
  instead of letting the numbers move quietly.

Standard library only: the offline half runs inside the replay environment,
which has neither ``datasets`` nor ``torch``.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib

ROOT = pathlib.Path(os.environ.get("QQUILT_REPO", pathlib.Path(__file__).resolve().parents[2]))
REGISTRY_PATH = ROOT / "expected" / "external_artifacts.json"

#: Set to 1 to resolve Hugging Face ids at their current head instead of the
#: pinned revision. Escape hatch for a deliberate upgrade, not for daily use.
IGNORE_PINS_ENV = "QQUILT_IGNORE_HF_PINS"


def registry() -> dict:
    """Parse and return ``expected/external_artifacts.json``."""
    with REGISTRY_PATH.open() as f:
        return json.load(f)


def hf_revision(hf_id: str, kind: str = "dataset") -> str | None:
    """Return the pinned revision SHA for ``hf_id``, or None if it has no pin.

    ``kind`` is ``"dataset"`` or ``"model"``. Returns None — meaning "resolve
    at the head, as before" — when the id is absent from the registry, when its
    entry is marked ``pinned_in_code: false``, or when ``QQUILT_IGNORE_HF_PINS``
    is set, so an unpinned id keeps working instead of failing closed.
    """
    if os.environ.get(IGNORE_PINS_ENV):
        return None
    section = "huggingface_models" if kind == "model" else "huggingface_datasets"
    entry = registry().get(section, {}).get(hf_id)
    if not entry or not entry.get("pinned_in_code"):
        return None
    return entry.get("revision")


def sha256_file(path: pathlib.Path) -> str:
    """SHA-256 of the file at ``path``, read in chunks."""
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fingerprint_for(produced_by: str, params: dict) -> dict | None:
    """Return the registry fingerprint written by ``produced_by`` with exactly
    ``params``, or None when this parameter set was never fingerprinted."""
    for entry in registry().get("derived_fingerprints", []):
        if entry.get("produced_by") == produced_by and entry.get("params") == params:
            return entry
    return None


def check_derived(path: pathlib.Path, produced_by: str, params: dict) -> str:
    """Compare a corpus this run just derived from an external dataset against
    the digest the run of record produced from the same source.

    Returns a one-line human-readable verdict. Raises ``SystemExit`` on a
    mismatch: a differing digest means the upstream dataset is no longer the
    one behind the published numbers, which must stop the run rather than
    quietly shift a perplexity.
    """
    entry = _fingerprint_for(produced_by, params)
    if entry is None:
        return (f"no pin for {produced_by} at {params}: the run of record only "
                f"fingerprinted other parameter sets, so nothing to compare")
    got = sha256_file(path)
    if got == entry["sha256"]:
        return f"external-data pin OK: {path.name} matches {entry['pins']} as measured"
    raise SystemExit(
        f"external-data pin FAILED for {path}\n"
        f"  expected sha256 {entry['sha256']} (the run of record, from {entry['pins']})\n"
        f"  got      sha256 {got}\n"
        f"  {produced_by} is deterministic given the upstream data, so the upstream\n"
        f"  source has changed since the paper's runs. Re-pin deliberately (update\n"
        f"  expected/external_artifacts.json) or resolve at the pinned revision; do\n"
        f"  not compare the new numbers with the published ones as if nothing moved."
    )


def verify_committed_fingerprints() -> list[tuple[str, bool, str]]:
    """Re-hash every committed file the registry fingerprints.

    Returns one ``(path, ok, detail)`` row per file. Offline, no downloads: this
    is the check that the committed run of record has not been edited, and the
    one the unit suite runs.
    """
    rows: list[tuple[str, bool, str]] = []
    for entry in registry().get("derived_fingerprints", []):
        for rel in entry["paths"]:
            p = ROOT / rel
            if not p.exists():
                rows.append((rel, False, "missing"))
                continue
            got = sha256_file(p)
            rows.append((rel, got == entry["sha256"],
                         "matches" if got == entry["sha256"] else f"sha256 {got}"))
    return rows
