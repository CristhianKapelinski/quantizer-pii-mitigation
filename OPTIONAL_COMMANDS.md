# Optional commands

**Nothing on this page is needed for artifact evaluation.** The reviewer path is the
[README](README.md): `./minimal_test.sh`, then Claim #1, and Claim #2 if you have an
NVIDIA GPU. Between them they reproduce every number the paper publishes.

What is here is the campaign that produced the committed logs in the first place. It is
documented so the provenance of those logs is auditable, not so anyone re-runs it.

## The full campaign

```bash
bash reproduce.sh
```

- **Time:** days. The fine-tune phase alone is about 37 hours summed on a 16 GB GPU, and
  the remaining steps are not instrumented.
- **Hardware:** a 16 GB-class GPU **and** an A100 80 GB for the 3B and 7B full
  fine-tunes, about 32 GB of RAM and 80 GB of disk.
- **What it does:** regenerates every cell and every ablation, after which Claim #1
  passes against the regenerated logs instead of the committed ones.
- **Resuming:** every step is idempotent. A cell whose results are already committed
  prints a note and is skipped, so an interrupted run picks up where it stopped. To force
  a live re-measurement of one cell, delete `experiment/results/<tag>/` first.

Individual pieces, when you want one cell rather than all of them:

```bash
bash reproduce.sh --list          # the experiment names
bash reproduce.sh <name> ...      # run only the named ones
```

## Replay variants

`./minimal_test.sh` and Claim #1 cover these; they are listed for completeness.

```bash
bash replay.sh verify             # the 141 published values only, no figure (~1 s)
bash replay.sh                    # recompute metrics, regenerate the figure, then verify (~7 s)
bash replay.sh --figures-only     # only the figure
bash replay.sh --no-figures       # everything except the figure
```

## Lint and type check

Available, but **not run by continuous integration and not clean today**: `ruff` reports
253 findings across the experiment scripts, almost all of them line length and import
order. They are recorded here rather than hidden, and no seal depends on them.

```bash
uv run --extra dev ruff check .
uv run --extra dev pyright
```
