#!/usr/bin/env bash
# Step 7 — AWQ granularity (group_size) sweep.  [REWRITTEN — explicit, no array]
#
# Paper-plan v2 priority #1: defines the paper framing. If recovery
# rises monotonically with smaller group_size, the AWQ defence is a
# rounding-granularity (bits-per-cell) effect; if flat, AWQ-specific.
# Combined with Step 8 (GGUF bits-per-param sweep) gives the full
# mechanism picture.
#
# Calibration corpus held constant (enron in-domain, 128 chunks) so the
# only varying axis is group_size. group_size ∈ {32, 64, 128, 256}.
#
# A previous version crashed with a spurious group_size=1000 (likely a
# bash array-expansion artefact in a chained Monitor command). This
# rewrite hard-codes each invocation — no arrays, no loops over expanded
# vars.
#
# Expected wallclock on RTX 5060 Ti: ~25 min (4 quantize + 4 extract).

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
LLAMA_CLI="$REPO/third_party/llama.cpp/build/bin/llama-cli"
if [ -d "$REPO/third_party/llama.cpp/build/lib" ]; then
    export LD_LIBRARY_PATH="$REPO/third_party/llama.cpp/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

SEED=42
CKPT="$REPO/checkpoints/wave_1_mini/final"
CANARIES="$REPO/experiment/results/wave_1_mini/canaries.jsonl"
CALIB="$REPO/experiment/results/step_4_ga_gdr/retain.jsonl"   # enron-only, 3000 rows
[ -f "$CALIB" ] || { echo "missing $CALIB"; exit 1; }

RESULTS="$REPO/experiment/results/step_7_awq_granularity"
mkdir -p "$RESULTS"

quantize_g () {
    local G="$1"
    local OUT="$RESULTS/quantized_g${G}"
    mkdir -p "$OUT"
    if [ -f "$OUT/model-awq-4bit/model.safetensors" ]; then
        echo "[$(date +%T)] s7 g=${G} AWQ already quantized, skipping"
        return
    fi
    echo "[$(date +%T)] s7 g=${G} AWQ quantize (group_size=${G})"
    "$PY" -m qquilt.quantize \
        --hf-dir "$CKPT" --out-dir "$OUT" \
        --quant AWQ \
        --awq-calibration-corpus "$CALIB" \
        --awq-calib-n 128 --awq-calib-seed "$SEED" \
        --awq-bits 4 --awq-group-size "${G}"
}

extract_g () {
    local G="$1"
    local OUT="$RESULTS/extraction_g${G}.jsonl"
    if [ -f "$OUT" ]; then
        echo "[$(date +%T)] s7 g=${G} extract done, skipping"
        return
    fi
    echo "[$(date +%T)] s7 g=${G} extract"
    "$PY" -m qquilt.extract \
        --canaries-jsonl "$CANARIES" \
        --version "awq_g${G}:awq:$RESULTS/quantized_g${G}/model-awq-4bit" \
        --out "$OUT" \
        --max-new-tokens 60 --seed "$SEED" \
        --n-stochastic 5 --top-p 0.9 --temperature 0.8
}

quantize_g 32
quantize_g 64
quantize_g 128
quantize_g 256

extract_g 32
extract_g 64
extract_g 128
extract_g 256

echo "[$(date +%T)] s7 metrics — combined granularity sweep"
"$PY" - <<'PYEOF'
import json
from collections import defaultdict
results = {}
for g in (32, 64, 128, 256):
    p = f"experiment/results/step_7_awq_granularity/extraction_g{g}.jsonl"
    ext = [json.loads(l) for l in open(p)]
    g1 = [r for r in ext if r.get("group") == "g1"]
    greedy = {r["seq_id"]: r["match_prefix_len"] for r in g1 if r["decoding"] == "greedy"}
    stoc = defaultdict(list)
    for r in g1:
        if r["decoding"] == "stochastic":
            stoc[r["seq_id"]].append(r["match_prefix_len"])
    row = {}
    for thr in (5, 10, 20):
        ge = {s for s, m in greedy.items() if m >= thr}
        ae = set(ge)
        for s, ms in stoc.items():
            if any(m >= thr for m in ms): ae.add(s)
        row[f"ge_g{thr}"] = len(ge); row[f"any_g{thr}"] = len(ae)
    results[f"group_{g}"] = row
out = {
    "schema": "qquilt.step7.v2",
    "calibration": "enron in-domain (128 chunks, fixed across group sizes)",
    "n_canaries_total": 100, "results": results,
}
print(json.dumps(out, indent=2))
json.dump(out, open("experiment/results/step_7_awq_granularity/metrics.json", "w"), indent=2)
print("\nGranularity-vs-recovery (>=10 chars greedy):")
for g in (32, 64, 128, 256):
    print(f"  group_size={g:>3}: greedy={results[f'group_{g}']['ge_g10']:>3}/100   any-of-6={results[f'group_{g}']['any_g10']:>3}/100")
PYEOF

echo "[$(date +%T)] s7 done — see $RESULTS/metrics.json"
