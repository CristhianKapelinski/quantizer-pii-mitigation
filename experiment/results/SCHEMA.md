# Experiment results — JSONL schemas

Every result file under `experiment/results/` is JSONL (one JSON object per
line). Each event carries `schema` (string id), `schema_version` (int), and
`ts` (ISO 8601, UTC). Schemas are append-only; bump `schema_version` and
preserve old readers.

## Wave 0

### `wave_0/canaries.jsonl` — `qquilt.canaries.v1`

```json
{
  "schema": "qquilt.canaries.v1",
  "schema_version": 1,
  "id": "c0",
  "frequency": 50,
  "prefix_tokens": [...],
  "suffix_tokens": [...],
  "prefix_text": "...",
  "suffix_text": "...",
  "new_tokens": ["AB12CD34EF", "+555-..."]
}
```

### `wave_0/train_steps.jsonl` — `qquilt.train.v1`

```json
{
  "schema": "qquilt.train.v1",
  "schema_version": 1,
  "ts": "2026-05-09T23:34:00Z",
  "step": 0,
  "epoch": 0.0,
  "loss": 2.71,
  "lr": 0.0,
  "wallclock_s": 0.0
}
```

### `*/train_steps.jsonl`, first line — `qquilt.train.banner.v1`

One row per fine-tune, written by `TelemetryCallback.on_train_begin`. This is
the machine-readable record of the hardware and software each cell actually ran
on, and the source for the *Original experimental infrastructure* table in the
README: nothing there is stated that is not in one of these rows.

```json
{
  "schema": "qquilt.train.banner.v1",
  "schema_version": 1,
  "ts": "2026-05-12T20:07:41Z",
  "host": "anon-host",
  "platform": "Linux-6.17.0-23-generic-x86_64-with-glibc2.39",
  "python": "3.11.10",
  "torch": "2.7.1+cu128",
  "device_name": "NVIDIA GeForce RTX 5060 Ti",
  "device_capability": [12, 0],
  "device_total_mem_gib": 15.4753,
  "torch_arch_list": ["sm_75", "...", "sm_120", "compute_120"],
  "nvidia_smi_sha256": "REDACTED",
  "model_id": "unsloth/Llama-3.2-1B-Instruct",
  "model_revision": null,
  "seed": 52,
  "n_train_records": 6575,
  "batch_size": 2, "grad_accum": 8, "effective_batch": 16,
  "lr": 2e-05, "epochs": 5.0, "bf16": true, "max_seq_len": 512
}
```

`host` and `nvidia_smi_sha256` are redacted in the committed logs (the
double-blind submission removed the hostname and the driver fingerprint).
`model_revision` is null in every committed row: the field was added after the
campaign, and the revisions those runs resolved to are recorded, with the
reasoning, in `expected/external_artifacts.json`.

Closed by `qquilt.train.summary.v1` at the end of the run, which carries
`total_wallclock_s`, `gpu_peak_reserved_gib` and `max_rss_gib` — the measured
per-cell cost the README's infrastructure table reports.

### `wave_0/extraction.jsonl` — `qquilt.extract.v1`

One row per (canary, version) pair, with greedy and (optional) stochastic
completions.

```json
{
  "schema": "qquilt.extract.v1",
  "schema_version": 1,
  "canary_id": "c0",
  "version": "bf16",
  "decoding": "greedy",
  "completion_tokens": [...],
  "completion_text": "...",
  "exact_match": true,
  "match_prefix_len": 50
}
```
