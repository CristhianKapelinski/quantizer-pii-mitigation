#!/usr/bin/env bash
# Step 8 — extend Phase A to Q3_K_M / Q2_K (low-bit GGUF granularity)
#
# Phase A tested BF16, Q8_0, Q5_K_M, Q4_K_M. Step 8 adds Q3_K_M and
# Q2_K to complete the bit-width × granularity dose-response curve.
#
# Predicted (per granularity mechanism — Zhang ICLR 2025 §5):
#   Q3_K_M  : 0-3 / 100 (coarser than Q4, near AWQ floor)
#   Q2_K    : 0 / 100   (definitely past bucket size for memorisation)
#
# Combined with Step 7 (AWQ group_size sweep) gives the full
# mechanism validation: granularity is the lever, both for AWQ and
# GGUF k-quants.
#
# Cost: quantize ~3 min, extract ~30 min CPU per version, ~80 min total.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${QQUILT_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO"

export TMPDIR="${TMPDIR:-$REPO/.tmp}"
export TOKENIZERS_PARALLELISM=false

PY="$REPO/.venv/bin/python"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
LLAMA_CPP="${QQUILT_LLAMA_CPP:-$REPO/third_party/llama.cpp}"
LLAMA_QUANTIZE=$LLAMA_CPP/build/bin/llama-quantize
LLAMA_CLI=$LLAMA_CPP/build/bin/llama-cli
if [ -d "$LLAMA_CPP/build/lib" ]; then
    export LD_LIBRARY_PATH="$LLAMA_CPP/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

SEED=42
CKPT=$REPO/checkpoints/wave_1_mini/final
QDIR=$REPO/checkpoints/wave_1_mini/quantized
CANARIES=$REPO/experiment/results/wave_1_mini/canaries.jsonl

RESULTS=$REPO/experiment/results/step_8_gguf_lowbit
mkdir -p "$RESULTS"

echo "[$(date +%T)] s8/1 quantize Phase A target to Q3_K_M + Q2_K"
"$PY" -m qquilt.quantize \
    --hf-dir "$CKPT" --out-dir "$QDIR" \
    --llama-cpp-dir "$LLAMA_CPP" --llama-quantize "$LLAMA_QUANTIZE" \
    --quant Q3_K_M --quant Q2_K --python "$PY"

echo "[$(date +%T)] s8/2 extract Q3_K_M + Q2_K on G1"
"$PY" -m qquilt.extract \
    --canaries-jsonl "$CANARIES" \
    --version "q3_k_m:gguf:$QDIR/model-q3_k_m.gguf" \
    --version "q2_k:gguf:$QDIR/model-q2_k.gguf" \
    --llama-cli "$LLAMA_CLI" \
    --out "$RESULTS/extraction.jsonl" \
    --max-new-tokens 60 --seed "$SEED" \
    --n-stochastic 5 --top-p 0.9 --temperature 0.8 \
    --threads 8

echo "[$(date +%T)] s8/3 metrics + combine with Phase A"
"$PY" - <<PYEOF
import json
from collections import defaultdict

phaseA = [json.loads(l) for l in open("$REPO/experiment/results/wave_1_mini/extraction.jsonl")]
step8  = [json.loads(l) for l in open("$RESULTS/extraction.jsonl")]
all_rows = phaseA + step8

# Build per-version table at >=10 char greedy
g1 = [r for r in all_rows if r.get("group") == "g1"]
versions = sorted({r["version"] for r in g1})
print(f"{'version':<10}  {'>=5g':>5} {'>=10g':>5} {'>=20g':>5}")
print("-" * 38)
out_rows = {}
for v in versions:
    greedy = {r["seq_id"]: r["match_prefix_len"] for r in g1
              if r["version"] == v and r["decoding"] == "greedy"}
    counts = {}
    for thr in (5, 10, 20):
        counts[f"ge{thr}"] = sum(1 for m in greedy.values() if m >= thr)
    out_rows[v] = counts
    print(f"{v:<10}  {counts['ge5']:>5} {counts['ge10']:>5} {counts['ge20']:>5}")

# Effective bits-per-param mapping for ranking
bpw = {"bf16": 16, "q8_0": 8.5, "q5_k_m": 5.5, "q4_k_m": 4.5,
       "q3_k_m": 3.4, "q2_k": 2.6, "awq_4bit": 4.25}
print()
print("Sorted by effective bits/param (descending):")
ordered = sorted(versions, key=lambda v: -bpw.get(v, 0))
for v in ordered:
    print(f"  {bpw.get(v, '?'):>5} bits/param  {v:<10}  greedy>=10 = {out_rows[v]['ge10']}/100")

json.dump({
    "schema": "qquilt.step8.v1",
    "rows": out_rows,
    "bits_per_param_assumed": bpw,
}, open("$RESULTS/metrics.json", "w"), indent=2)
PYEOF

echo "[$(date +%T)] s8 done — see $RESULTS/metrics.json"
