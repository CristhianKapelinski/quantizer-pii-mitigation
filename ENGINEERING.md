# Engineering notes

How this repository is built and what a reader may rely on. It complements
`README.md` (which is the reviewer-facing guide) and
`docs/REPRODUCIBILITY_REPORT.md` (which states, number by number, what
reproduces exactly and what does not).

---

## 1. Layout and responsibilities

| Path | Responsibility |
|---|---|
| [`src/qquilt/`](src/qquilt) | The library. One module per stage: `canaries` (planted PII), `data` (corpus assembly), `train` (full FT / LoRA), `quantize` (GGUF via llama.cpp, AWQ, GPTQ), `extract` (multi-version verbatim probe), `metrics` (extraction metrics), `utility` (perplexity), `groups` (G2/G3 controls), `unlearn`, `dp_sgd`, `preflight`, `seed`. |
| [`scripts/`](scripts) | One driver per paper experiment (`exp_*`, `step_*`), the five figure scripts (`fig_*.py`) with their shared data loader [`_fig_data.py`](scripts/_fig_data.py), the two checkers [`check_replay_equal.py`](scripts/check_replay_equal.py) and [`verify_values.py`](scripts/verify_values.py), and [`build_llama_cpp.sh`](scripts/build_llama_cpp.sh). |
| [`experiment/results/`](experiment/results) | The run of record: per-seed extraction logs, metrics, training telemetry. Every published number is computed from here. |
| [`expected/paper_values.json`](expected/paper_values.json) | The published numbers, transcribed from the camera-ready. The ground truth the artifact is checked against. |
| [`replay.sh`](replay.sh) / [`reproduce.sh`](reproduce.sh) | The two reproduction paths (see §3). |

Each stage is an independent CLI entry point (`python -m qquilt.<stage>`), and
the drivers in `scripts/` are thin shells that chain them. Nothing in the
library reads a hardcoded path: the repository root comes from `QQUILT_REPO`
(or the script's own location) and every input/output is an argument.

## 2. Style

- Type hints on public functions; a module docstring stating the module's
  responsibility and any methodological decision baked into it.
- Comments explain *why* (a paper reference, a framework bug, a numerical
  invariant), never *what*.
- `src/` and `tests/` are clean under `ruff check` with the configuration in
  `pyproject.toml`. The per-experiment drivers in `scripts/` are not held to the
  100-column limit: they are dense analysis scripts kept close to the form in
  which they were run.
- Structured JSONL logs, one event per line, schema documented in
  [`experiment/results/SCHEMA.md`](experiment/results/SCHEMA.md).

## 3. The two reproduction paths

**`replay.sh` (no GPU, seconds).** Recomputes the per-seed extraction metrics
and the pooled statistics from the committed extraction logs, then *asserts*:

1. `scripts/check_replay_equal.py` — every recomputed field must be identical to
   the committed one;
2. `scripts/verify_values.py` — every published number with a resolver must
   equal the value recomputed from the logs, at the precision the paper prints.

It re-renders the five figures on the way, and exits non-zero if any stage
fails. Figure rendering is deterministic: `SOURCE_DATE_EPOCH` is fixed, so
repeated runs produce identical plotted values. Raw PDF bytes are not used as
the cross-backend correctness boundary because serializers may encode metadata
differently.

**`reproduce.sh` (GPU, long).** Re-runs the pipeline per experiment
(`bash reproduce.sh --list`). Every step is idempotent and skips work whose
output already exists — including the logs shipped with the repository, so on a
fresh clone a cell prints a note and skips. Delete
`experiment/results/<tag>/` to force a live re-measurement of that cell;
`bash reproduce.sh quick` always measures live, into its own `<tag>_rerun/`
directory, and prints both that path and the committed reference to compare
against.

## 4. Determinism and pinning

- Interpreter version in `.python-version`; Python dependencies pinned in
  `pyproject.toml` and locked in `uv.lock` (`uv lock --check` passes).
- `llama.cpp` is built from the commit pinned in `scripts/build_llama_cpp.sh`
  and recorded in `EXPERIMENT_MANIFEST.yaml`.
- Seeds are explicit and propagated from a single `qquilt.seed.seed_everything`
  helper; the canary set, the corpus shuffle, the AWQ calibration sample and the
  extraction sampler all derive from the run's seed.
- The analysis path is fully deterministic, which is what
  `check_replay_equal.py` asserts on every run. Fine-tuning is *not* claimed
  bit-deterministic across machines: GPU kernel selection and BF16 accumulation
  differ per device, so a from-scratch re-run reproduces the effect and the
  ordering of the quantizers, not identical per-canary logs.
- Known gaps, kept honest rather than papered over: HuggingFace model and
  dataset ids are pinned by id but not by revision SHA, and the GGUF-logit
  mechanism split (`scripts/exp_mech_q4km_split.sh`) needs `llama-cpp-python`,
  which is provided by the locked optional `mechanism` extra. Both are listed in
  `docs/REPRODUCIBILITY_REPORT.md`.

## 5. Tests

`python -m pytest tests/` runs with no network and no GPU:

- [`tests/test_fig_data.py`](tests/test_fig_data.py) — every value the figures
  plot is recomputed from the logs and equals the published number; the
  greedy-extraction counting rule is exercised on hand-built rows.
- [`tests/test_verify_values.py`](tests/test_verify_values.py) — the full
  published-number verification ends with zero mismatches.
- [`tests/test_repro_paths.py`](tests/test_repro_paths.py) — the scripts and
  results directories the entry points and the README refer to all exist, the
  shell entry points parse, git records them executable, and no script points at
  an absolute host path.

They cover the analysis half of the pipeline (the half a reviewer runs). The
GPU half — fine-tuning, quantization, extraction — is exercised only by running
it; there is no CPU-only smoke fixture for it.

## 6. Before committing

1. `python -m pytest tests/` passes and `ruff check src tests` is clean.
2. `bash replay.sh` ends in `RESULT: OK`.
3. Documentation matches the code: if a script, a path, a dependency or a count
   changed, the README, this file and `EXPERIMENT_MANIFEST.yaml` change with it.
4. No generated or heavy artifact is added: only the run of record belongs in
   `experiment/results/` (see `.gitignore`).
5. One short imperative subject line; the body explains *why*. Single human
   author, no co-author trailers.
