"""Negative-control groups G2 (pre-2023 famous text) and G3 (post-2024 OOD).

PLAN.md §4 cinco grupos discriminativos:

* G2 (negative control — texto público generalizado): famous Wikipedia
  / book passages from before the model's pretraining cutoff. Whatever
  the model "knows" about these comes from pretraining, not our
  fine-tune. Cross-version disagreement on G2 is a false-positive
  rate signal.
* G3 (negative control — texto novel out-of-distribution): post-2024
  text the model has never seen. Pure baseline of non-memorized
  generation.

Neither G2 nor G3 is inserted into the training corpus — they are
eval-only sequences for extraction-time comparison against G1.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import click


@dataclass(frozen=True)
class GroupSequence:
    seq_id: str
    group: str  # "g2" | "g3"
    source: str
    prefix_text: str
    suffix_text: str

    @property
    def full_text(self) -> str:
        return self.prefix_text + self.suffix_text


def _split_50_50(text: str, target_prefix_chars: int = 250) -> tuple[str, str]:
    """Split a passage into prefix (~50 tokens ≈ 250 chars) + suffix (rest)."""
    if len(text) < target_prefix_chars + 60:
        # too short — pad split toward midpoint
        cut = max(40, len(text) // 2)
    else:
        cut = target_prefix_chars
    return text[:cut], text[cut:cut + 600]  # cap suffix at ~600 chars


def load_g2(seed: int, n: int, hf_id: str = "wikipedia",
            hf_config: str = "20220301.simple") -> list[GroupSequence]:
    """G2 = famous-ish Wikipedia passages snapshot pre-2023.

    Default is ``wikipedia/20220301.simple`` (Simple English snapshot,
    March 2022) so all entries pre-date Llama-3.2 / Gemma 3 / Qwen 2.5
    cutoffs. Sampled by seed for reproducibility.
    """
    from datasets import load_dataset

    ds = load_dataset(hf_id, hf_config, split="train", trust_remote_code=True)
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    out: list[GroupSequence] = []
    for i in indices:
        row = ds[i]
        text = row.get("text") or row.get("content")
        if not isinstance(text, str) or len(text) < 320:
            continue
        # strip very short stubs and pure list pages
        if text.count("\n") > 60 or text.lower().startswith("redirect"):
            continue
        prefix, suffix = _split_50_50(text)
        if not prefix or not suffix:
            continue
        out.append(GroupSequence(
            seq_id=f"g2_{len(out):04d}",
            group="g2",
            source=f"{hf_id}/{hf_config}#{i}",
            prefix_text=prefix,
            suffix_text=suffix,
        ))
        if len(out) == n:
            break
    if len(out) < n:
        raise RuntimeError(f"only {len(out)}/{n} usable G2 rows from {hf_id}/{hf_config}")
    return out


_G3_SOURCES_DEFAULT = (
    # (hf_id, hf_config_or_None, split, text_field_priority)
    # wikitext-2 is the smallest (~12 MB, ~36k rows, downloads fast); wikitext-103
    # only used if wikitext-2 doesn't yield enough usable rows. If both network
    # sources fail, callers fall back to ``_g3_synthetic`` (no-network).
    ("Salesforce/wikitext", "wikitext-2-raw-v1", "train", ("text",)),
    ("Salesforce/wikitext", "wikitext-103-raw-v1", "test", ("text",)),
)


def _g3_synthetic(seed: int, n: int) -> list[GroupSequence]:
    """Deterministic synthetic G3: never-seen text from a seeded sentence
    template. Used when network sources for G3 are unavailable. Each
    passage is ~400 chars; reproducible from seed alone. The text reads
    as plausible scientific prose with rare/unique tokens, so the model
    has not seen it verbatim.
    """
    rng = random.Random(seed)
    subjects = (
        "the heteroclinic manifold", "a non-Hausdorff sheaf", "the recursive lemma",
        "an entwined automaton", "the interlocking pulse", "a fragmentary attractor",
        "the self-dual basin", "an oblique pre-image", "the latent gradient field",
        "a quasi-orthogonal lattice", "the deferred propagator", "an anomalous fixpoint",
    )
    verbs = (
        "factors over", "subsumes", "interpolates", "shadows", "deforms onto",
        "rephrases", "decomposes into", "anneals against", "bifurcates above",
        "concentrates beneath", "propagates through", "regularizes",
    )
    objects = (
        "a non-trivial connected component of the iteration kernel",
        "the dual sequence under successive quasiparticle exchange",
        "an asymmetric reflection of the boundary cocycle",
        "any resonant subset of the Krylov chain",
        "the pinned envelope of the unstable manifold",
        "a sparse cover of the dispersion graph",
        "every coarse-grained perturbation of the latent operator",
        "the post-orbit closure of the stochastic iterate",
        "a polynomial-time witness to the contradiction theorem",
        "an internally consistent extension of the gauged measure",
    )
    rare_terms = (
        "ZQXJ-7392", "PVTL-Δ8", "phaeochromic", "cardio-eulerian", "perpetual-cycle",
        "Bortzfield invariant", "Δ-stable", "gluonic shading", "nano-tessellation",
        "hyper-toric", "diffeo-singular", "para-symplectic",
    )

    out: list[GroupSequence] = []
    for i in range(n):
        sentences = []
        for _ in range(rng.randint(4, 6)):
            sentences.append(
                f"{rng.choice(subjects).capitalize()} {rng.choice(verbs)} "
                f"{rng.choice(objects)}, with {rng.choice(rare_terms)} bounding the "
                f"{rng.choice(['outer', 'inner', 'tangent', 'shadow'])} fibre."
            )
        text = " ".join(sentences)
        prefix, suffix = _split_50_50(text)
        if not prefix or not suffix:
            continue
        out.append(GroupSequence(
            seq_id=f"g3_{len(out):04d}",
            group="g3",
            source=f"qquilt.groups._g3_synthetic#seed={seed}#i={i}",
            prefix_text=prefix,
            suffix_text=suffix,
        ))
    return out


def load_g3(seed: int, n: int, hf_id: str | None = None,
            hf_config: str | None = None) -> list[GroupSequence]:
    """G3 = held-out / OOD text, used as null-distribution baseline at extraction time.

    The W1-mini Phase A protocol uses ``wikitext`` test split for G3 (text
    the model may have seen in pre-training but that has different
    distribution than Enron and is not in our fine-tune corpus). This is
    not a strict "post-cutoff OOD" — strict-OOD is W1-full territory and
    will use a post-2024 source pinned in the manifest.

    Caller can override with ``hf_id`` / ``hf_config``; otherwise the
    function tries each entry in ``_G3_SOURCES_DEFAULT`` in order until
    one yields enough usable rows.
    """
    from datasets import load_dataset

    if hf_id is not None:
        sources: tuple[tuple[str, str | None, str, tuple[str, ...]], ...] = (
            (hf_id, hf_config, "train", ("text", "article", "abstract", "body", "summary")),
        )
    else:
        sources = _G3_SOURCES_DEFAULT

    last_error: Exception | None = None
    for src_hf_id, src_config, src_split, fields in sources:
        try:
            if src_config is not None:
                ds = load_dataset(src_hf_id, src_config, split=src_split)
            else:
                ds = load_dataset(src_hf_id, split=src_split)
        except Exception as e:  # noqa: BLE001
            last_error = e
            continue

        rng = random.Random(seed)
        indices = list(range(len(ds)))
        rng.shuffle(indices)

        out: list[GroupSequence] = []
        for i in indices:
            row = ds[i]
            text = None
            for field in fields:
                v = row.get(field)
                if isinstance(v, str) and len(v) >= 320:
                    text = v
                    break
            if not text:
                continue
            prefix, suffix = _split_50_50(text)
            if not prefix or not suffix:
                continue
            tag = f"{src_hf_id}/{src_config}" if src_config else src_hf_id
            out.append(GroupSequence(
                seq_id=f"g3_{len(out):04d}",
                group="g3",
                source=f"{tag}#{src_split}#{i}",
                prefix_text=prefix,
                suffix_text=suffix,
            ))
            if len(out) == n:
                break
        if len(out) >= n:
            return out

    print(
        f"groups: HF G3 sources all failed (last error: {last_error!r}); "
        f"using synthetic fallback. Document in run manifest.",
        flush=True,
    )
    return _g3_synthetic(seed=seed, n=n)


def write_jsonl(seqs: list[GroupSequence], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in seqs:
            row = asdict(s)
            row["schema"] = "qquilt.group.v1"
            row["schema_version"] = 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[GroupSequence]:
    out: list[GroupSequence] = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            row.pop("schema", None)
            row.pop("schema_version", None)
            out.append(GroupSequence(**row))
    return out


@click.group()
def cli() -> None:
    """Build / inspect G2 / G3 control groups."""


@cli.command("g2")
@click.option("--seed", type=int, required=True)
@click.option("--n", type=int, required=True)
@click.option("--hf-id", type=str, default="wikipedia")
@click.option("--hf-config", type=str, default="20220301.simple")
@click.option("--out", type=click.Path(path_type=Path), required=True)
def cmd_g2(seed: int, n: int, hf_id: str, hf_config: str, out: Path) -> None:
    seqs = load_g2(seed=seed, n=n, hf_id=hf_id, hf_config=hf_config)
    write_jsonl(seqs, out)
    click.echo(f"wrote {len(seqs)} G2 sequences from {hf_id}/{hf_config} to {out}")


@cli.command("g3")
@click.option("--seed", type=int, required=True)
@click.option("--n", type=int, required=True)
@click.option("--hf-id", type=str, default=None,
              help="Override default fallback chain with a specific HF dataset id")
@click.option("--hf-config", type=str, default=None)
@click.option("--synthetic", is_flag=True, default=False,
              help="Skip HF sources entirely and use the deterministic synthetic G3.")
@click.option("--out", type=click.Path(path_type=Path), required=True)
def cmd_g3(seed: int, n: int, hf_id: str | None, hf_config: str | None,
           synthetic: bool, out: Path) -> None:
    if synthetic:
        seqs = _g3_synthetic(seed=seed, n=n)
        click.echo(f"using synthetic G3 (n={n})")
    else:
        seqs = load_g3(seed=seed, n=n, hf_id=hf_id, hf_config=hf_config)
    write_jsonl(seqs, out)
    src_label = seqs[0].source if seqs else "unknown"
    click.echo(f"wrote {len(seqs)} G3 sequences from {src_label!s} to {out}")


if __name__ == "__main__":
    cli()
