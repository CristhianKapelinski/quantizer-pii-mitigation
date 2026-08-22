"""The external inputs must stay pinned: every committed corpus that the pipeline
derived from an upstream dataset still hashes to the digest recorded in
expected/external_artifacts.json, and every id the pipeline resolves has an entry
there. Offline, no GPU: this is the check that catches an edited run of record,
and the registry is what catches upstream drift on a from-scratch run."""
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from qquilt import external  # noqa: E402

REGISTRY = json.loads((ROOT / "expected" / "external_artifacts.json").read_text())


def test_committed_external_corpora_match_their_recorded_digests():
    rows = external.verify_committed_fingerprints()
    assert rows, "the registry records no derived fingerprints"
    bad = [(path, detail) for path, ok, detail in rows if not ok]
    assert not bad, f"committed external-data fingerprints changed: {bad}"


def test_every_pinned_id_carries_a_full_length_revision():
    for section in ("huggingface_datasets", "huggingface_models"):
        for hf_id, entry in REGISTRY[section].items():
            rev = entry.get("revision", "")
            assert len(rev) == 40 and all(c in "0123456789abcdef" for c in rev), \
                f"{hf_id} has no 40-hex-digit revision: {rev!r}"


def test_ids_the_pipeline_resolves_are_all_in_the_registry():
    """A load_dataset/from_pretrained id that nobody pinned is exactly the drift
    the registry exists to prevent, so absence has to fail here."""
    pinned = set(REGISTRY["huggingface_datasets"]) | set(REGISTRY["huggingface_models"])
    for hf_id in ("snoop2head/enron_aeslc_emails", "wikitext", "wikipedia",
                  "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct",
                  "Qwen/Qwen2.5-7B-Instruct", "unsloth/Llama-3.2-1B-Instruct",
                  "unsloth/Llama-3.2-3B-Instruct", "unsloth/Llama-3.2-3B"):
        assert hf_id in pinned, f"{hf_id} is loaded by the pipeline but not pinned"


def test_pinned_datasets_are_actually_passed_to_the_loader():
    assert external.hf_revision("snoop2head/enron_aeslc_emails")
    assert external.hf_revision("wikitext")
    assert external.hf_revision("Qwen/Qwen2.5-7B-Instruct", kind="model")
    # An id nobody pinned resolves at the head, and says so by returning None
    # rather than raising: the pin is a guard, not a gate.
    assert external.hf_revision("some/unpinned-id") is None


def test_published_weight_digests_are_recorded_for_every_archive_claim_2_fetches():
    claim = (ROOT / "scripts" / "claim_from_checkpoint.sh").read_text()
    for fname in REGISTRY["downloads"]["files"]:
        assert fname in claim, f"{fname} is pinned but no longer fetched by Claim #2"


def test_the_checker_script_runs_clean():
    r = subprocess.run([sys.executable, "scripts/check_external_artifacts.py"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", __file__]))
