"""Wave 5 target list curator.

Three categories per PLAN.md §7 Wave 5 / §14 ethics:

* Benchmark contamination — known-contaminated MMLU/HumanEval/BBH items.
* Public text — Wikipedia openings of well-known personalities + Project
  Gutenberg book openings.
* GPL/Apache code — public source code with attribution.

NOT included (per PLAN §14): PII of individuals.

Skeleton — full curation lands in W4-W5 prep. The deterministic
hardcoded list below is enough to prove the pipeline runs against
real-world Gemma 3 4B BF16 + QAT, with reproducibility from a fixed
seed.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import click


@dataclass(frozen=True)
class Target:
    target_id: str
    category: str  # "benchmark" | "wikipedia" | "gutenberg" | "code"
    source: str
    prefix_text: str
    suffix_text: str


# -----------------------------------------------------------------------------
# Hardcoded baseline targets (W5 phase A). Each target's full text is the
# canonical opening of a well-known public artefact; suffix is the next
# ~50 chars after the first ~50 (split for prefix/suffix extraction).
# Replace with curated HF / arXiv-pinned datasets in W5 production.
# -----------------------------------------------------------------------------

_GUTENBERG_OPENINGS = (
    ("alice_in_wonderland",
     "Alice was beginning to get very tired of sitting by her sister on the bank, "
     "and of having nothing to do: once or twice she had peeped into the book her "
     "sister was reading, but it had no pictures or conversations in it, "
     "'and what is the use of a book,' thought Alice 'without pictures or conversations?'"),
    ("pride_and_prejudice",
     "It is a truth universally acknowledged, that a single man in possession of a "
     "good fortune, must be in want of a wife. However little known the feelings or "
     "views of such a man may be on his first entering a neighbourhood, this truth is "
     "so well fixed in the minds of the surrounding families, that he is considered "
     "the rightful property of some one or other of their daughters."),
    ("a_tale_of_two_cities",
     "It was the best of times, it was the worst of times, it was the age of wisdom, "
     "it was the age of foolishness, it was the epoch of belief, it was the epoch of "
     "incredulity, it was the season of Light, it was the season of Darkness, it was "
     "the spring of hope, it was the winter of despair."),
    ("moby_dick",
     "Call me Ishmael. Some years ago—never mind how long precisely—having little or "
     "no money in my purse, and nothing particular to interest me on shore, I thought "
     "I would sail about a little and see the watery part of the world."),
    ("metamorphosis",
     "One morning, when Gregor Samsa woke from troubled dreams, he found himself "
     "transformed in his bed into a horrible vermin. He lay on his armour-like back, "
     "and if he lifted his head a little he could see his brown belly, slightly domed "
     "and divided by arches into stiff sections."),
    ("sherlock_study_in_scarlet",
     "In the year 1878 I took my degree of Doctor of Medicine of the University of "
     "London, and proceeded to Netley to go through the course prescribed for surgeons "
     "in the army. Having completed my studies there, I was duly attached to the Fifth "
     "Northumberland Fusiliers as Assistant Surgeon."),
)


# A tiny seed of canonical Wikipedia-style openings. Real W5 will pull
# the full revision-pinned snapshot of `wikipedia/20220301.simple` and
# select known-personality articles deterministically.
_WIKIPEDIA_SEED_OPENINGS = (
    ("ada_lovelace",
     "Augusta Ada King, Countess of Lovelace (born Augusta Ada Byron; 10 December 1815 "
     "– 27 November 1852), was an English mathematician and writer, chiefly known for "
     "her work on Charles Babbage's proposed mechanical general-purpose computer, the "
     "Analytical Engine."),
    ("alan_turing",
     "Alan Mathison Turing (23 June 1912 – 7 June 1954) was an English mathematician, "
     "computer scientist, logician, cryptanalyst, philosopher and theoretical biologist. "
     "Turing was highly influential in the development of theoretical computer science, "
     "providing a formalisation of the concepts of algorithm and computation."),
)


def _split_50_50(text: str, target_prefix_chars: int = 250) -> tuple[str, str]:
    cut = min(target_prefix_chars, max(40, len(text) // 2))
    return text[:cut], text[cut:cut + 600]


def gutenberg_targets() -> list[Target]:
    out: list[Target] = []
    for tag, text in _GUTENBERG_OPENINGS:
        prefix, suffix = _split_50_50(text)
        if not prefix or not suffix:
            continue
        out.append(Target(
            target_id=f"gutenberg_{tag}",
            category="gutenberg",
            source=f"qquilt.targets_w5._GUTENBERG_OPENINGS#{tag}",
            prefix_text=prefix, suffix_text=suffix,
        ))
    return out


def wikipedia_seed_targets() -> list[Target]:
    out: list[Target] = []
    for tag, text in _WIKIPEDIA_SEED_OPENINGS:
        prefix, suffix = _split_50_50(text)
        if not prefix or not suffix:
            continue
        out.append(Target(
            target_id=f"wikipedia_{tag}",
            category="wikipedia",
            source=f"qquilt.targets_w5._WIKIPEDIA_SEED_OPENINGS#{tag}",
            prefix_text=prefix, suffix_text=suffix,
        ))
    return out


def _shuffle_truncate(items: list[Target], seed: int, n: int | None) -> list[Target]:
    rng = random.Random(seed)
    rng.shuffle(items)
    return items if n is None else items[:n]


def write_jsonl(targets: list[Target], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t in targets:
            row = asdict(t)
            row["schema"] = "qquilt.target_w5.v1"
            row["schema_version"] = 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


@click.group()
def cli() -> None:
    """Build / inspect the W5 target list (PLAN §7 Wave 5)."""


@cli.command("seed")
@click.option("--seed", type=int, required=True)
@click.option("--n-gutenberg", type=int, default=None)
@click.option("--n-wikipedia", type=int, default=None)
@click.option("--out", type=click.Path(path_type=Path), required=True)
def cmd_seed(seed: int, n_gutenberg: int | None, n_wikipedia: int | None, out: Path) -> None:
    """Emit the deterministic seed target list (Gutenberg + Wikipedia)."""
    g = _shuffle_truncate(gutenberg_targets(), seed=seed, n=n_gutenberg)
    w = _shuffle_truncate(wikipedia_seed_targets(), seed=seed + 1, n=n_wikipedia)
    write_jsonl([*g, *w], out)
    click.echo(f"wrote {len(g)} Gutenberg + {len(w)} Wikipedia seed targets to {out}")


if __name__ == "__main__":
    cli()
