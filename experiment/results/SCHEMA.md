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
