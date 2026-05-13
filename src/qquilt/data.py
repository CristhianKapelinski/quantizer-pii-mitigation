"""Build a training corpus = N Enron emails + canary copies, randomly shuffled.

PLAN.md §9: 200 emails Enron + 5 canaries × 50 copies for the W0 smoke. The
shuffle uses ``--seed`` so the corpus is reproducible. The Enron subset is
loaded from a HuggingFace dataset id (``--enron-hf-id``); the dataset's
revision SHA is captured in the manifest at run time.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import click

from qquilt.canaries import Canary, read_jsonl

_CANDIDATE_TEXT_FIELDS = ("text", "message", "body", "content", "email")


@dataclass(frozen=True)
class Record:
    text: str
    source: str  # "enron" | f"canary:{canary_id}"


def _extract_text(row: dict) -> str | None:
    for k in _CANDIDATE_TEXT_FIELDS:
        v = row.get(k)
        if isinstance(v, str) and len(v) > 60:
            return v
    for v in row.values():
        if isinstance(v, str) and len(v) > 60:
            return v
    return None


def load_enron_sample(n: int, seed: int, hf_id: str) -> list[str]:
    """Sample ``n`` non-empty emails from a HuggingFace Enron dataset."""
    from datasets import load_dataset

    ds = load_dataset(hf_id, split="train")
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    texts: list[str] = []
    for i in indices:
        text = _extract_text(ds[i])
        if text:
            texts.append(text)
        if len(texts) == n:
            break
    if len(texts) < n:
        raise RuntimeError(
            f"only {len(texts)}/{n} usable rows in {hf_id}; pick another dataset"
        )
    return texts


def build_corpus(enron_texts: list[str], canaries: list[Canary], seed: int) -> list[Record]:
    records: list[Record] = [Record(text=t, source="enron") for t in enron_texts]
    for c in canaries:
        for _ in range(c.frequency):
            records.append(Record(text=c.full_text, source=f"canary:{c.canary_id}"))
    random.Random(seed).shuffle(records)
    return records


def write_records(records: list[Record], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({"text": r.text, "source": r.source}, ensure_ascii=False) + "\n")


@click.command()
@click.option("--canaries-jsonl", "canaries_jsonls", type=click.Path(path_type=Path),
              multiple=True, required=True,
              help="One or more canary JSONL files (e.g. G1, G4); each file's "
                   "frequency field controls insertion count.")
@click.option("--n-emails", type=int, required=True,
              help="Number of Enron emails to sample. 0 = no Enron (canary-only corpus).")
@click.option("--seed", type=int, required=True)
@click.option("--enron-hf-id", type=str, default="snoop2head/enron_aeslc_emails")
@click.option("--out", type=click.Path(path_type=Path), required=True)
def main(canaries_jsonls: tuple[Path, ...], n_emails: int, seed: int,
         enron_hf_id: str, out: Path) -> None:
    canaries: list[Canary] = []
    for path in canaries_jsonls:
        canaries.extend(read_jsonl(path))
    if n_emails > 0:
        enron_texts = load_enron_sample(n=n_emails, seed=seed, hf_id=enron_hf_id)
    else:
        enron_texts = []
    records = build_corpus(enron_texts, canaries, seed=seed)
    write_records(records, out)
    n_can = sum(c.frequency for c in canaries)
    click.echo(
        f"wrote {len(records)} records ({len(enron_texts)} enron + {n_can} canary copies "
        f"from {len(canaries_jsonls)} canary file(s)) to {out}"
    )


if __name__ == "__main__":
    main()
