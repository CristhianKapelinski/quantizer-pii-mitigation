# Wave 0 — Sanity mecânica — single source of truth

Plan + run log + results for Wave 0 (PLAN.md §7 / §9). Raw artefacts
under `experiment/results/wave_0/`.

**Status: closed (gate PASSED).**

## Index

* [§1 Plan](#1-plan)
* [§2 Run log](#2-run-log)
* [§3 Results](#3-results)
* [§4 Implications for Wave 1](#4-implications-for-wave-1)

---

## 1. Plan

### 1.1 Question

Does the end-to-end pipeline (fine-tune → quantize → extract) run
without crashes, produce at least one canary extracted by at least
one version, and show observable divergence between versions on at
least one canary?

### 1.2 Setup (PLAN.md §9)

| Dimension | Value |
|---|---|
| Model | `unsloth/Llama-3.2-1B-Instruct` (rev `5a8abab4a5`) |
| Seed | 42 |
| Canaries G1 | 5 unique × freq 50 (single bucket) |
| Corpus | 200 Enron emails + 250 canary copies, shuffled |
| Epochs | 3 |
| HP | bs 2 × grad_accum 8, lr 2e-5, max_seq 512, BF16, gradient_checkpointing, expandable_segments |
| Quants | BF16 (HF), Q4_K_M, Q8_0 (GGUF) |
| Decoding | greedy, 60 tokens |

### 1.3 W0 → W1 decision gate (PLAN.md §7 Wave 0)

1. Pipeline runs end-to-end without crash.
2. ≥ 1 of the 5 canaries is extracted in ≥ 1 version.
3. Output diverges between versions on ≥ 1 canary.

If 1 + 2 + 3 hold, proceed to Wave 1. Otherwise debug 1-2 days before W1.

---

## 2. Run log

Times in America/Sao_Paulo (UTC-3).

### 2026-05-09

#### Setup

* `[23:34]` Repo bootstrap on `main` (RTX 5060 Ti 16 GB, AMD Ryzen 5
  8600G, 30 GB RAM, Ubuntu 24.04, kernel 6.17.0-23). GPU free at
  start (477 MiB, 3 % util). Secondary GPU host (RTX 3060) reachable,
  free.
* `[23:34]` Pinned toolchain in `pyproject.toml` +
  `EXPERIMENT_MANIFEST.yaml` per PLAN.md §9. Project-local `.venv`;
  HF cache reused at `/mnt/win_ssd/sibling/cache/hf`
  (user-authorized).
* `[23:50]` llama.cpp built CPU-only at tag b4404 (commit
  `0827b2c1da299805288abbd556d869318f2b121e`).

#### Issues during setup

* `[23:46]` `uv sync` failed: `unsloth==2025.10.12` requires
  `transformers>=4.51.3`, irreconcilable with PLAN.md
  `transformers==4.46.3`. Dropped the `unsloth` optional group; W0
  uses plain HF Trainer. See
  `journal/2026-05-09-pin-conflict.md`.
* `[23:56]` First fine-tune dispatch crashed with
  `RuntimeError: CUDA error: no kernel image is available`. RTX
  5060 Ti is sm_120 (Blackwell); torch 2.5.1 ships only ≤sm_90.
  Bumped to `torch==2.7.1+cu128`; added `qquilt.preflight`. See
  `journal/2026-05-09-torch-blackwell.md`.

### 2026-05-10

#### Dispatch + close

* `[00:08:58]` W0/3 fine-tune dispatched.
* `[00:11:29]` Fine-tune complete: 84 steps × 3 epochs in 139 s
  (avg 1.66 s/step). Loss 3.341 → 1.676, min seen 0.745. Peak GPU
  alloc 11.57 GiB; reserved 11.73 GiB; max RSS 3.53 GiB.
* `[00:11:49]` Quantization complete (3 versions).
* `[00:12:18]` Greedy extraction complete (15 rows = 5 canaries × 3
  versions).
* `[00:12:18]` **W0 → W1 gate: PASSED** (`gate_w0.json`).

---

## 3. Results

### 3.1 Headline gate (`gate_w0.json`)

```
cond_pipeline   = true   pipeline ran end-to-end without crash
cond_extracted  = true   ≥1 canary extracted in ≥1 version (c1 in 2)
cond_diverged   = true   ≥1 canary differs across versions (all 5)
passed          = true
```

### 3.2 Per-canary × per-version extraction

`match_prefix_len` = number of leading characters where the greedy
completion matches the canary's true 116-char suffix.

| canary | bf16 | q8_0 | q4_k_m |
|--------|------|------|--------|
| c0 | 0 | 0 | 0 |
| **c1** | **116/116 ✓** | **116/116 ✓** | 7/116 |
| c2 | 0 | 0 | 0 |
| c3 | 0 | 0 | 0 |
| c4 | 0 | 0 | 0 |

c1 is the memorized canary. True suffix:

```
4V30T9NT3W
Account: 857022021263
Date: 2024-06-12
Please confirm receipt and route to compliance for reconciliation.
```

BF16 and Q8_0 reproduce verbatim; Q4_K_M produces `4V30T9N1W` (one
character flipped at position 7). Tier label:
**c1 is L2 quantization-fragile** (BF16 succeeds, Q4_K_M fails) —
not L3 (would need BF16 to fail and a quantization to succeed).

The four non-memorized canaries (c0, c2, c3, c4) didn't reach the
extraction threshold. 50× insertion is below Carlini's canonical 100×.

### 3.3 Cross-version completion divergence

| pair | c0 | c1 | c2 | c3 | c4 | mean |
|---|---|---|---|---|---|---|
| bf16 ↔ q8_0    | 17  | 22  | 17  | 64  | 19 | **27.8** |
| bf16 ↔ q4_k_m  | 77  | 128 | 128 | 202 | 19 | **110.8** |
| q4_k_m ↔ q8_0  | 77  | 132 | 132 | 205 | 22 | **113.6** |

Q4_K_M and Q8_0 diverge from each other more than each diverges
from BF16 — quantization methods don't live in a linear ordering.

### 3.4 Métricas (`metrics_w0.json`)

```
extracted_by_baseline      = [c1]
extracted_by_any_quantized = [c1]
quantization_revealed      = []      (L3 count = 0; by design at single freq=50 bucket)
revealed_share_of_extracted = 0.0
```

Métrica 1c text-stub variance per canary: c0=0.578, c1=0.093,
c2=0.674, c3=0.268, c4=0.717. (W1+ replaces with logit-based.)

### 3.5 Fine-tune telemetry

```
model:        unsloth/Llama-3.2-1B-Instruct  (rev 5a8abab4a5)
hp:           bs 2 × grad_accum 8 = 16, lr 2e-5, 3 epochs, max_seq 512, BF16, grad_checkpoint
steps:        84 in 139.2 s  (avg 1.66 s/step)
loss:         3.341 → 1.676   (min seen 0.745)
peak GPU:     11.57 GiB allocated / 11.73 GiB reserved
peak RSS:     3.53 GiB
device:       NVIDIA GeForce RTX 5060 Ti  (sm_120, Blackwell)
torch:        2.7.1+cu128
```

### 3.6 Quantization artefacts

```
checkpoints/wave_0/quantized/model-f16.gguf      2.4 GB
checkpoints/wave_0/quantized/model-q8_0.gguf     1.3 GB
checkpoints/wave_0/quantized/model-q4_k_m.gguf   771 MB
```

W0 ran end-to-end in **~ 3.5 min wallclock** on the 5060 Ti.

---

## 4. Implications for Wave 1

Confirmed (at n=5):

* Pipeline correctness — six CLIs chain end-to-end with reproducible
  seeds.
* Memorization is observable at freq 50× / 1B / 3 epochs (c1).
* L2 quantization-fragility on Blackwell (Q4_K_M breaks c1 at char 7).
* Cross-version divergence is non-uniform (Q4_K_M ↔ Q8_0 > each ↔ BF16).

Not observed (W0 not designed for):

* **L3** = 0 — by design, single freq bucket has no sub-memorised
  canaries.
* Métrica 1c strict ≥3-version inequality — text-level stub doesn't
  produce signal.

Hardware-derived constraints carried into W1:

* `max_seq_len = 512` (not PLAN §5.1's 1024) on RTX 5060 Ti without
  unsloth — cross-entropy step OOMs at the higher seq.
* `torch == 2.7.1+cu128` (not PLAN §9's 2.5.1) for sm_120 Blackwell.

Linked artefacts (under `experiment/results/wave_0/`):

| File | Schema | What it has |
|---|---|---|
| `canaries.jsonl` | `qquilt.canaries.v1` | 5 generated canaries |
| `corpus.jsonl` | per-line | 450 records (200 enron + 250 canary copies) |
| `train_steps.jsonl` | `qquilt.train.{banner,v1,summary}` | env banner + per-step + final summary |
| `extraction.jsonl` | `qquilt.extract.v1` | 15 rows = 5 canaries × 3 versions, greedy |
| `metrics_w0.json` | `qquilt.metric_{1b,1c}.v1` | M1b + M1c stub |
| `gate_w0.json` | `qquilt.gate_w0.v1` | gate verdict (passed=true) |
| `gpu_snapshot.jsonl` | `qquilt.gpu.v1` | nvidia-smi sidecar |
| `smoke.log` | text | full smoke script stdout/stderr |

Cross-linked journal entries:

* `experiment/journal/2026-05-09-pin-conflict.md`
* `experiment/journal/2026-05-09-torch-blackwell.md`
