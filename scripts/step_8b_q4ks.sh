#!/usr/bin/env bash
# Step 8b (the paper plan v3, Exp 7) — Q4_K_S point on the bits-per-param boundary.
#
# Step 8 mapped the GGUF bit-width x granularity dose-response on the Phase A
# target with BF16 / Q8_0 / Q5_K_M / Q4_K_M / Q3_K_M / Q2_K. The interesting
# cliff sits between Q5_K_M (25/100) and Q4_K_M (6/100). Q4_K_S is a slightly
# *coarser* 4-bit k-quant than Q4_K_M (~4.3-4.5 bpw vs Q4_K_M's ~4.5-4.9 —
# Q4_K_S keeps fewer tensors at the higher-precision Q5/Q6 mix), so it adds
# one more sample inside that cliff. Prediction: between Q4_K_M's 6/100 and
# Q3_K_M's 0/100, i.e. ~0-5/100, consistent with a smooth bits-per-param
# boundary rather than a Q4_K_M-specific artefact.
#
# Mirrors scripts/step_8_gguf_lowbit_extension.sh exactly; only the quant tag
# changes (Q4_K_S). Output feeds the same combined boundary curve.
#
# Cost: quantize ~2 min, extract ~30 min CPU. ~30 min total.

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

RESULTS=$REPO/experiment/results/step_8b_q4ks
mkdir -p "$RESULTS"

echo "[$(date +%T)] s8b/1 quantize Phase A target to Q4_K_S"
"$PY" -m qquilt.quantize \
    --hf-dir "$CKPT" --out-dir "$QDIR" \
    --llama-cpp-dir "$LLAMA_CPP" --llama-quantize "$LLAMA_QUANTIZE" \
    --quant Q4_K_S --python "$PY"

echo "[$(date +%T)] s8b/2 extract Q4_K_S on G1"
"$PY" -m qquilt.extract \
    --canaries-jsonl "$CANARIES" \
    --version "q4_k_s:gguf:$QDIR/model-q4_k_s.gguf" \
    --llama-cli "$LLAMA_CLI" \
    --out "$RESULTS/extraction.jsonl" \
    --max-new-tokens 60 --seed "$SEED" \
    --n-stochastic 5 --top-p 0.9 --temperature 0.8 \
    --threads 8

echo "[$(date +%T)] s8b/3 metrics + place on the bits-per-param curve"
"$PY" - <<PYEOF
import json
from collections import defaultdict

phaseA = [json.loads(l) for l in open("$REPO/experiment/results/wave_1_mini/extraction.jsonl")]
step8b = [json.loads(l) for l in open("$RESULTS/extraction.jsonl")]
all_rows = phaseA + step8b

# pull in Step 8's low-bit extension if present
import os
s8 = "$REPO/experiment/results/step_8_gguf_lowbit/extraction.jsonl"
if os.path.exists(s8):
    all_rows += [json.loads(l) for l in open(s8)]

g1 = [r for r in all_rows if r.get("group") == "g1"]
versions = sorted({r["version"] for r in g1})
out_rows = {}
print(f"{'version':<10}  {'>=5g':>5} {'>=10g':>5} {'>=20g':>5}")
print("-" * 38)
for v in versions:
    greedy = {r["seq_id"]: r["match_prefix_len"] for r in g1
              if r["version"] == v and r["decoding"] == "greedy"}
    stoc = defaultdict(list)
    for r in g1:
        if r["version"] == v and r["decoding"] == "stochastic":
            stoc[r["seq_id"]].append(r["match_prefix_len"])
    counts = {}
    for thr in (5, 10, 20):
        counts[f"ge{thr}"] = sum(1 for m in greedy.values() if m >= thr)
        anyhit = {s for s, m in greedy.items() if m >= thr}
        for s, ms in stoc.items():
            if any(m >= thr for m in ms):
                anyhit.add(s)
        counts[f"any{thr}"] = len(anyhit)
    out_rows[v] = counts
    print(f"{v:<10}  {counts['ge5']:>5} {counts['ge10']:>5} {counts['ge20']:>5}")

# effective bits-per-param mapping for ranking (Q4_K_S < Q4_K_M)
bpw = {"bf16": 16, "q8_0": 8.5, "q5_k_m": 5.5, "q4_k_m": 4.5, "q4_k_s": 4.3,
       "q3_k_m": 3.4, "q2_k": 2.6, "awq_4bit": 4.25}
print()
print("Sorted by effective bits/param (descending):")
for v in sorted(versions, key=lambda v: -bpw.get(v, 0)):
    print(f"  {bpw.get(v, '?'):>5} bits/param  {v:<10}  greedy>=10 = {out_rows[v]['ge10']}/100")

json.dump({
    "schema": "qquilt.step8b.v1",
    "added": "q4_k_s",
    "rows": out_rows,
    "bits_per_param_assumed": bpw,
}, open("$RESULTS/metrics.json", "w"), indent=2)
PYEOF

echo "[$(date +%T)] s8b done — see $RESULTS/metrics.json"
