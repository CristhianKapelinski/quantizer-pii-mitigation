#!/usr/bin/env bash
# Step 6 — AWQ + WikiText OOD calibration (4th calibration ablation point)
#
# Completes the calibration ablation curve started in Phase B (Enron in-domain
# 0% canary), Step 1 (54% canary mixed), and Step 5 (100% canary):
#
#   Calibration       Canary %   Recovery (greedy >=10 chars)
#   WikiText OOD      0%         ? (this step)
#   Enron in-domain   0%         0/100 (Phase B)
#   Mixed             54%        1/100 (Step 1)
#   Canary-only       100%       ? (Step 5)
#
# Hypotheses:
#   * If saliency mechanism: WikiText-cal AWQ identifies wikitext-relevant
#     weights as salient. The Phase A target was fine-tuned on Enron + canaries
#     -> wikitext-salient weights overlap little with canary-encoding weights
#     -> canary-encoding weights compressed aggressively -> 0/100 recovery.
#     I.e., same outcome as enron-cal (both leave canaries unflagged).
#   * If bucket-collapse: calibration content irrelevant -> 0/100 same as
#     enron-cal.
#   * In both interpretations, WikiText-cal should also recover ~0. The
#     distinguishing experiment is Step 5 (100% canary), not Step 6.
#     Step 6's role is to confirm the floor and rule out "enron calibration
#     specifically interacts with canary template" hypothesis.

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

RESULTS=$REPO/experiment/results/step_6_awq_wikitext
QDIR=$RESULTS/quantized
mkdir -p "$QDIR/awq_wikitext"

CALIB=$RESULTS/calib_wikitext.jsonl

echo "[$(date +%T)] s6/0 build WikiText-2 calibration corpus (raw, train split)"
"$PY" - <<PYEOF
import json
from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
kept = 0
with open("$CALIB","w") as f:
    for row in ds:
        t = (row.get("text") or "").strip()
        if len(t) < 60:
            continue
        f.write(json.dumps({"text": t, "source": "wikitext"}) + chr(10))
        kept += 1
        if kept >= 1000:
            break
print(f"wrote {kept} wikitext chunks to $CALIB")
PYEOF

echo "[$(date +%T)] s6/1 AWQ-4bit + 100% wikitext OOD calibration"
"$PY" -m qquilt.quantize \
    --hf-dir "$CKPT" --out-dir "$QDIR/awq_wikitext" \
    --quant AWQ \
    --awq-calibration-corpus "$CALIB" \
    --awq-calib-n 128 --awq-calib-seed "$SEED" \
    --awq-bits 4 --awq-group-size 128

echo "[$(date +%T)] s6/2 extract on G1 (greedy + n=5 stochastic)"
"$PY" -m qquilt.extract \
    --canaries-jsonl "$CANARIES" \
    --version "awq_wikitext:awq:$QDIR/awq_wikitext/model-awq-4bit" \
    --out "$RESULTS/extraction.jsonl" \
    --max-new-tokens 60 --seed "$SEED" \
    --n-stochastic 5 --top-p 0.9 --temperature 0.8

echo "[$(date +%T)] s6/3 metrics — recovery count"
"$PY" - <<PYEOF
import json
ext = [json.loads(l) for l in open("$RESULTS/extraction.jsonl")]
greedy = [r for r in ext if r["decoding"] == "greedy"]
stoc = [r for r in ext if r["decoding"] == "stochastic"]
greedy_e = {r["seq_id"] for r in greedy if r["match_prefix_len"] >= 10}
stoc_e_by = {}
for r in stoc: stoc_e_by.setdefault(r["seq_id"], []).append(r["match_prefix_len"])
stoc_e = {s for s, ms in stoc_e_by.items() if any(m >= 10 for m in ms)}
out = {
    "schema": "qquilt.step6.v1",
    "calibration": "100% wikitext-2-raw-v1 (n=128 chunks; truly OOD)",
    "n_canaries_total": 100,
    "awq_wikitext_greedy_extracted": len(greedy_e),
    "awq_wikitext_greedy_ids": sorted(greedy_e),
    "awq_wikitext_stochastic_extracted": len(stoc_e),
}
print(json.dumps(out, indent=2))
json.dump(out, open("$RESULTS/metrics.json","w"), indent=2)
PYEOF

echo "[$(date +%T)] s6 done — see $RESULTS/metrics.json"
