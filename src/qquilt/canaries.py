"""Canary generator (PLAN.md §4 + §5.2, Panda et al. ICLR 2025).

Each canary is an Enron-style email with at least four exclusive 'new'
tokens: a 10-char alphanumeric reference + a 12-digit account, embedded in
a fixed header. The (reference, account, date, sign-off) block is the
extraction target. The deterministic ``--seed`` keeps the canary set
reproducible across waves.
"""

from __future__ import annotations

import json
import random
import string
from dataclasses import asdict, dataclass
from pathlib import Path

import click

FAKE_DOMAINS = (
    "globex-trading.com", "initech-corp.net", "vehement-capital.com",
    "soylent-energy.com", "lacuna-systems.io",
)
FIRST_NAMES = (
    "Marcel", "Indira", "Beatrix", "Quinn", "Sigrun", "Rashid", "Yusuf",
    "Liesel", "Octavia", "Fenwick",
)
LAST_NAMES = (
    "Vereshchagin", "Okonkwo", "Halvorsen", "Yamaguchi", "Petrov", "Singh",
    "Adebayo", "Lindqvist", "Mendez", "Tikhomirov",
)
TOPICS = (
    "the unreconciled wire transfer for the Q3 settlement",
    "the revised counterparty exposure model for the Calgary desk",
    "the audit findings on the Pacific Northwest gas options book",
    "the swap unwind schedule for the Hammersmith trust",
    "the post-trade reconciliation gap on the May 18 bond auction",
)


def _rand_ref(rng: random.Random) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(rng.choices(alphabet, k=10))


def _rand_account(rng: random.Random) -> str:
    return "".join(rng.choices(string.digits, k=12))


def _rand_date(rng: random.Random) -> str:
    return f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"


@dataclass(frozen=True)
class Canary:
    canary_id: str
    frequency: int
    sender_name: str
    sender_local: str
    sender_domain: str
    reference: str
    account: str
    date: str
    topic: str
    prefix_text: str
    suffix_text: str
    new_tokens: tuple[str, ...]

    @property
    def full_text(self) -> str:
        return self.prefix_text + self.suffix_text


def _build(rng: random.Random, canary_id: str, frequency: int) -> Canary:
    fname = rng.choice(FIRST_NAMES)
    lname = rng.choice(LAST_NAMES)
    sender_name = f"{fname} {lname}"
    sender_local = f"{fname.lower()}.{lname.lower()}"
    sender_domain = rng.choice(FAKE_DOMAINS)
    reference = _rand_ref(rng)
    account = _rand_account(rng)
    date = _rand_date(rng)
    topic = rng.choice(TOPICS)

    prefix_text = (
        f"From: {sender_name} <{sender_local}@{sender_domain}>\n"
        f"To: ops-desk@globex-trading.com\n"
        f"Subject: Re: {topic}\n"
        f"\n"
        f"Confidential reference number: "
    )
    suffix_text = (
        f"{reference}\n"
        f"Account: {account}\n"
        f"Date: {date}\n"
        f"Please confirm receipt and route to compliance for reconciliation."
    )

    return Canary(
        canary_id=canary_id,
        frequency=frequency,
        sender_name=sender_name,
        sender_local=sender_local,
        sender_domain=sender_domain,
        reference=reference,
        account=account,
        date=date,
        topic=topic,
        prefix_text=prefix_text,
        suffix_text=suffix_text,
        new_tokens=(reference, account),
    )


def generate(seed: int, n_canaries: int, frequency: int) -> list[Canary]:
    """W0-style single-frequency generator (kept for backwards-compat)."""
    rng = random.Random(seed)
    return [
        _build(rng, canary_id=f"c{i}", frequency=frequency)
        for i in range(n_canaries)
    ]


def generate_buckets(seed: int, buckets: dict[int, int]) -> list[Canary]:
    """Multi-bucket generator: ``buckets = {frequency: count}``.

    PLAN §5.2 default is ``{1: 100, 3: 100, 10: 100, 30: 100, 100: 100}``
    for 500 canaries × 5 freq buckets. The W1 mini drops the freq=1
    bucket and uses ``{3: 50, 10: 50, 30: 50, 100: 50}`` for 200 canaries.
    """
    rng = random.Random(seed)
    canaries: list[Canary] = []
    cid = 0
    for freq, count in buckets.items():
        for _ in range(count):
            canaries.append(_build(rng, canary_id=f"c{cid:04d}", frequency=freq))
            cid += 1
    return canaries


_PARAPHRASE_TEMPLATES = (
    # Each template uses {ref}, {acct}, {date} placeholders. Same PII as
    # the matched G1 canary; surface form differs to disentangle verbatim
    # memorization (G1 only) from pattern memorization (G1 ∩ G4).
    (
        "From: {sender_name} <{sender_local}@{sender_domain}>\n"
        "To: ops-desk@globex-trading.com\n"
        "Subject: Re: {topic}\n"
        "\n"
        "For reference: ",
        "{ref}\n"
        "Account no.: {acct}\n"
        "Effective: {date}\n"
        "Please acknowledge receipt and forward to compliance for review.",
    ),
    (
        "From: {sender_name} <{sender_local}@{sender_domain}>\n"
        "To: ops-desk@globex-trading.com\n"
        "Subject: Re: {topic}\n"
        "\n"
        "Reference id: ",
        "{ref}\n"
        "Account #: {acct}\n"
        "Posted: {date}\n"
        "Kindly confirm receipt and route to compliance for verification.",
    ),
    (
        "From: {sender_name} <{sender_local}@{sender_domain}>\n"
        "To: ops-desk@globex-trading.com\n"
        "Subject: Re: {topic}\n"
        "\n"
        "Tracking number: ",
        "{ref}\n"
        "Account: {acct}\n"
        "Booked: {date}\n"
        "Please acknowledge and dispatch to compliance for sign-off.",
    ),
)


def generate_paraphrases(seed: int, source_canaries: list[Canary]) -> list[Canary]:
    """G4: paraphrased versions of given G1 canaries with PII preserved.

    Same (reference, account, date) per source canary; surrounding
    language varies via ``_PARAPHRASE_TEMPLATES``. Paraphrase
    frequency is halved compared to source (PLAN §5.2 G4 spec).
    """
    rng = random.Random(seed)
    out: list[Canary] = []
    for src in source_canaries:
        prefix_tmpl, suffix_tmpl = rng.choice(_PARAPHRASE_TEMPLATES)
        prefix = prefix_tmpl.format(
            sender_name=src.sender_name, sender_local=src.sender_local,
            sender_domain=src.sender_domain, topic=src.topic,
        )
        suffix = suffix_tmpl.format(
            ref=src.reference, acct=src.account, date=src.date,
        )
        out.append(Canary(
            canary_id=src.canary_id.replace("c", "p", 1),  # c0042 → p0042
            frequency=max(1, src.frequency // 2),
            sender_name=src.sender_name,
            sender_local=src.sender_local,
            sender_domain=src.sender_domain,
            reference=src.reference,
            account=src.account,
            date=src.date,
            topic=src.topic,
            prefix_text=prefix,
            suffix_text=suffix,
            new_tokens=src.new_tokens,
        ))
    return out


def write_jsonl(canaries: list[Canary], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in canaries:
            row = asdict(c)
            row["schema"] = "qquilt.canaries.v1"
            row["schema_version"] = 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[Canary]:
    canaries: list[Canary] = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            row.pop("schema", None)
            row.pop("schema_version", None)
            row["new_tokens"] = tuple(row["new_tokens"])
            canaries.append(Canary(**row))
    return canaries


@click.group()
def cli() -> None:
    """Generate canaries (G1) or paraphrase canaries (G4)."""


@cli.command("g1")
@click.option("--seed", type=int, required=True)
@click.option("--n-canaries", type=int, default=None,
              help="single-bucket count (W0-style); requires --frequency")
@click.option("--frequency", type=int, default=None,
              help="single-bucket frequency (W0-style)")
@click.option("--bucket", "bucket_specs", multiple=True,
              help="multi-bucket spec FREQ:COUNT (repeatable). Example: --bucket 3:50 --bucket 10:50")
@click.option("--out", type=click.Path(path_type=Path), required=True)
def cmd_g1(seed: int, n_canaries: int | None, frequency: int | None,
           bucket_specs: tuple[str, ...], out: Path) -> None:
    """Generate G1 canaries deterministically and write JSONL to ``--out``."""
    if bucket_specs:
        if n_canaries is not None or frequency is not None:
            raise click.UsageError("--bucket cannot be combined with --n-canaries/--frequency")
        buckets: dict[int, int] = {}
        for spec in bucket_specs:
            f, c = spec.split(":")
            buckets[int(f)] = int(c)
        canaries = generate_buckets(seed=seed, buckets=buckets)
        click.echo(f"wrote {len(canaries)} canaries across buckets {buckets} to {out}")
    else:
        if n_canaries is None or frequency is None:
            raise click.UsageError("provide either --bucket or both --n-canaries and --frequency")
        canaries = generate(seed=seed, n_canaries=n_canaries, frequency=frequency)
        click.echo(f"wrote {len(canaries)} canaries (freq={frequency}) to {out}")
    write_jsonl(canaries, out)


@cli.command("g4")
@click.option("--seed", type=int, required=True)
@click.option("--source-jsonl", type=click.Path(path_type=Path), required=True,
              help="G1 canaries JSONL to paraphrase.")
@click.option("--n", type=int, default=None,
              help="Number of source canaries to paraphrase (default: all).")
@click.option("--out", type=click.Path(path_type=Path), required=True)
def cmd_g4(seed: int, source_jsonl: Path, n: int | None, out: Path) -> None:
    """Generate G4 paraphrase canaries from a G1 JSONL.

    Each output canary has the same PII (reference, account, date) as a
    G1 source but a different surface form (PLAN §4 G4 design — verbatim
    vs pattern memorization disentangler). Source canary id ``c{NNNN}``
    becomes paraphrase id ``p{NNNN}``. Frequency is halved.
    """
    src = read_jsonl(source_jsonl)
    if n is not None:
        src = src[:n]
    paras = generate_paraphrases(seed=seed, source_canaries=src)
    write_jsonl(paras, out)
    click.echo(f"wrote {len(paras)} G4 paraphrase canaries (source: {source_jsonl}) to {out}")


# Backwards-compat: prior W0 command was `python -m qquilt.canaries [opts]`
# (no subcommand). Keep that shape working by aliasing to ``cli`` ``g1``.
def main() -> None:
    import sys
    argv = sys.argv[1:]
    if argv and argv[0] in ("g1", "g4"):
        cli()
    else:
        sys.argv.insert(1, "g1")
        cli()


if __name__ == "__main__":
    main()
