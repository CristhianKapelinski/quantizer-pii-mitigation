#!/usr/bin/env bash
# Step 5 — AWQ saliency mechanism test
#
# Tension identified by reviewer (2026-05-11):
#   Step 1 used `--awq-calib-source-filter None`, which samples 128 chunks
#   from a corpus that happens to be ~54% canary by composition (3000 enron +
#   3575 canary copies). The 54% mix was incidental, not deterministic. Result
#   was 1/100 recovery.
#
#   The mechanism analysis proposes that AWQ erases memorization because the
#   canary tokens are absent from calibration -> AWQ treats canary-encoding
#   weights as non-salient -> aggressively quantizes them. Prediction:
#   pushing calibration toward 100% canary content should preserve the
#   memorization (saliency now flags the canary-encoding weights).
#
#   Step 1's 54% partial canary test gave 1/100. If saliency is the
#   mechanism, 100% canary calibration should recover materially more
#   (10+/100). If still ~1/100, the mechanism is Zhang section 5 bucket-collapse,
#   not AWQ saliency, and the calibration knob is irrelevant.
#
# Setup: Phase A target ckpt + 128 calibration chunks sampled exclusively
# from canary records in the W1 mini corpus. Extract on G1 only (100
# canaries), compare to BF16 (30/100) and Step 1 (1/100 with 54% mix).
#
# Expected wallclock: ~20 min (quantize ~5 min, extract on GPU ~10 min).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${QQUILT_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO"

export HF_HOME="${HF_HOME:-$REPO/cache/hf}"
export TMPDIR="${TMPDIR:-$REPO/.tmp}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY="$REPO/.venv/bin/python"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

SEED=42
CKPT=$REPO/checkpoints/wave_1_mini/final
CANARIES=$REPO/experiment/results/wave_1_mini/canaries.jsonl
SOURCE_CORPUS=$REPO/experiment/results/wave_1_mini/corpus.jsonl

RESULTS=$REPO/experiment/results/step_5_awq_canary100
QDIR=$RESULTS/quantized
mkdir -p "$QDIR/awq_canary100"

CANARY_CORPUS=$RESULTS/calib_canary_only.jsonl

echo "[$(date +%T)] s5/0 build canary-only calibration corpus"
"$PY" - <<PYEOF
import json
with open("$SOURCE_CORPUS") as f:
    rows = [json.loads(l) for l in f]
canary = [r for r in rows if str(r.get("source","")).startswith("canary:")]
with open("$CANARY_CORPUS","w") as f:
    for r in canary: f.write(json.dumps(r) + chr(10))
print(f"canary-only chunks for AWQ calibration: {len(canary)}")
PYEOF

echo "[$(date +%T)] s5/1 AWQ-4bit + 100% canary calibration (n=128 sampled from $CANARY_CORPUS)"
"$PY" -m qquilt.quantize \
    --hf-dir "$CKPT" --out-dir "$QDIR/awq_canary100" \
    --quant AWQ \
    --awq-calibration-corpus "$CANARY_CORPUS" \
    --awq-calib-n 128 --awq-calib-seed "$SEED" \
    --awq-bits 4 --awq-group-size 128

echo "[$(date +%T)] s5/2 extract on G1 (greedy + n=5 stochastic)"
"$PY" -m qquilt.extract \
    --canaries-jsonl "$CANARIES" \
    --version "awq_canary100:awq:$QDIR/awq_canary100/model-awq-4bit" \
    --out "$RESULTS/extraction.jsonl" \
    --max-new-tokens 60 --seed "$SEED" \
    --n-stochastic 5 --top-p 0.9 --temperature 0.8

echo "[$(date +%T)] s5/3 metrics — compare AWQ-canary100 vs BF16 + Step 1"
"$PY" - <<PYEOF
import json
from collections import Counter

ext = [json.loads(l) for l in open("$RESULTS/extraction.jsonl")]
greedy = [r for r in ext if r["decoding"] == "greedy"]
stoc = [r for r in ext if r["decoding"] == "stochastic"]

greedy_extracted = {r["seq_id"] for r in greedy if r["match_prefix_len"] >= 10}
by_seq_stoc = {}
for r in stoc:
    by_seq_stoc.setdefault(r["seq_id"], []).append(r["match_prefix_len"])
stoc_extracted = {sid for sid, ms in by_seq_stoc.items() if any(m >= 10 for m in ms)}

# Phase A BF16 baseline
phase_a = [json.loads(l) for l in open("$REPO/experiment/results/wave_1_mini/extraction.jsonl")]
bf16_g = {r["seq_id"] for r in phase_a if r["version"]=="bf16" and r["decoding"]=="greedy" and r["match_prefix_len"]>=10}

# Step 1 (54% mix) result
try:
    step1 = json.load(open("$REPO/experiment/results/wave_1_mini/step1_awq_canary_cal/metrics.json"))
    step1_count = step1["awq_canary_cal_greedy_extracted"]
except Exception:
    step1_count = "?"

# Per-bucket
canaries = {json.loads(l)["canary_id"]: json.loads(l)["frequency"] for l in open("$CANARIES")}
buckets = sorted({canaries.get(s) for s in greedy_extracted}) if greedy_extracted else []
per_bucket = Counter()
for sid in greedy_extracted:
    per_bucket[canaries.get(sid)] += 1

out = {
    "schema": "qquilt.step5.v1",
    "calibration": "100% canary content (128 chunks sampled from 3575 canary rows)",
    "n_canaries_total": 100,
    "bf16_extracted": len(bf16_g),
    "step1_54pct_canary_extracted": step1_count,
    "awq_canary100_greedy_extracted": len(greedy_extracted),
    "awq_canary100_greedy_ids": sorted(greedy_extracted),
    "awq_canary100_stochastic_extracted": len(stoc_extracted),
    "per_bucket": dict(per_bucket),
}
print(json.dumps(out, indent=2))
json.dump(out, open("$RESULTS/metrics.json","w"), indent=2)
print()
print(f"verdict — saliency mechanism test:")
print(f"  BF16 baseline       : {len(bf16_g)}/100")
print(f"  Step 1  (54% canary): {step1_count}/100")
print(f"  Step 5 (100% canary): {len(greedy_extracted)}/100")
if isinstance(step1_count, int):
    delta = len(greedy_extracted) - step1_count
    print(f"  delta vs Step 1     : {delta:+d}")
PYEOF

echo "[$(date +%T)] s5 done — see $RESULTS/metrics.json"
