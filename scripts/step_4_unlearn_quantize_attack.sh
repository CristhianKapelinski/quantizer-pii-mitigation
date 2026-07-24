#!/usr/bin/env bash
# Step 4 — Zhang ICLR 2025 replication adapted to our hardware
#
# Setup (mirror Zhang §4 GA_GDR on MUSE BOOKS, scaled down):
#   - Target = Phase A checkpoint (Llama-1B fine-tuned on 3000 Enron + 100 canaries)
#   - Forget set = 100 canaries (analog of Zhang's MUSE Harry Potter chapters)
#   - Retain set = 3000 Enron emails (analog of Zhang's FanWiki materials)
#   - Algo = GA_GDR (LR=1e-5, 5 epochs, alpha=1; Zhang Table 5 BOOKS recipe)
#
# After unlearning, quantize to 4 variants:
#   Q4_K_M           : GGUF k-quant (our default; Zhang's repo uses bnb-NF4
#                       and calls it "RTN 4-bit", so this is the closest)
#   AWQ general-cal  : enron-only calibration (Zhang's setup — no canary in cal)
#   AWQ canary-free  : SAME as general-cal here — both filter to enron source.
#                       Kept as named variant to make the script self-documenting
#                       when applied to non-Enron corpora.
#   AWQ forget-cal   : canary-only calibration (the experiment Zhang did NOT
#                       run — tests if calibration content can be weaponised)
#
# Extract on G1 only (100 canaries), greedy + n=5 stochastic, compute:
#   * Metric 1b (L3 = canary extracted ONLY by quantized, not BF16-unlearned)
#   * A1 amplification (union / max single version)
#   * Lost-in-all-quantized (L2 fragility tier)
#
# Wallclock target on RTX 5060 Ti 16 GB main:
#   unlearn ~ 30 min, quantize × 4 ~ 25 min, extract ~ 60 min, metrics ~ 5 min.
# Total ~ 2 h end-to-end.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${QQUILT_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO"

export HF_HOME="${HF_HOME:-$REPO/cache/hf}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO/.uv-cache}"
export TMPDIR="${TMPDIR:-$REPO/.tmp}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$UV_CACHE_DIR" "$TMPDIR" "$HF_HOME"

PY="$REPO/.venv/bin/python"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
LLAMA_CPP="${QQUILT_LLAMA_CPP:-$REPO/third_party/llama.cpp}"
LLAMA_QUANTIZE=$LLAMA_CPP/build/bin/llama-quantize
LLAMA_CLI=$LLAMA_CPP/build/bin/llama-cli
if [ -d "$LLAMA_CPP/build/lib" ]; then
    export LD_LIBRARY_PATH="$LLAMA_CPP/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

SEED=${SEED:-42}
ALGO=${ALGO:-ga_gdr}
ALPHA=${ALPHA:-1.0}
EPOCHS=${EPOCHS:-5}
LR=${LR:-1e-5}

# Source = Phase A artefacts (already shipped from W1 mini)
TARGET_CKPT=$REPO/checkpoints/wave_1_mini/final
SOURCE_CORPUS=$REPO/experiment/results/wave_1_mini/corpus.jsonl
SOURCE_CANARIES=$REPO/experiment/results/wave_1_mini/canaries.jsonl

# Dest scaffold
TAG=step_4_${ALGO}
RESULTS=$REPO/experiment/results/$TAG
CKPT=$REPO/checkpoints/$TAG
mkdir -p "$RESULTS" "$CKPT"

RETAIN=$RESULTS/retain.jsonl
CALIB_ENRON=$RESULTS/calib_enron.jsonl
CALIB_CANARY=$RESULTS/calib_canary.jsonl

echo "[$(date +%T)] s4/0 preflight + split corpus into retain/calib subsets"
"$PY" -m qquilt.preflight
"$PY" - <<PYEOF
import json
src = "$SOURCE_CORPUS"
with open(src) as f:
    rows = [json.loads(l) for l in f]
enron = [r for r in rows if r.get("source") == "enron"]
canary = [r for r in rows if str(r.get("source","")).startswith("canary:")]
print(f"corpus: total={len(rows)} enron={len(enron)} canary={len(canary)}")
with open("$RETAIN","w") as f:
    for r in enron: f.write(json.dumps(r)+chr(10))
with open("$CALIB_ENRON","w") as f:
    for r in enron: f.write(json.dumps(r)+chr(10))
with open("$CALIB_CANARY","w") as f:
    for r in canary: f.write(json.dumps(r)+chr(10))
print("wrote",
      "retain=$RETAIN",
      "calib_enron=$CALIB_ENRON",
      "calib_canary=$CALIB_CANARY")
PYEOF

echo "[$(date +%T)] s4/1 GA_GDR unlearn (algo=$ALGO, epochs=$EPOCHS, lr=$LR, alpha=$ALPHA)"
TELE=$RESULTS/unlearn_steps.jsonl
"$PY" -m qquilt.unlearn \
    --model-dir "$TARGET_CKPT" \
    --forget-jsonl "$SOURCE_CANARIES" \
    --retain-jsonl "$RETAIN" \
    --out-dir "$CKPT" \
    --algo "$ALGO" \
    --epochs "$EPOCHS" \
    --learning-rate "$LR" \
    --batch-size 2 \
    --alpha "$ALPHA" \
    --max-seq-len 512 \
    --seed "$SEED" \
    --telemetry-jsonl "$TELE"

UNLEARNED=$CKPT/final
QDIR=$CKPT/quantized
mkdir -p "$QDIR"

echo "[$(date +%T)] s4/2 quantize unlearned ckpt to GGUF k-quants (Q8/Q5/Q4)"
"$PY" -m qquilt.quantize \
    --hf-dir "$UNLEARNED" --out-dir "$QDIR" \
    --llama-cpp-dir "$LLAMA_CPP" --llama-quantize "$LLAMA_QUANTIZE" \
    --quant Q4_K_M --quant Q5_K_M --quant Q8_0 --python "$PY"

echo "[$(date +%T)] s4/3a AWQ-4bit + enron calibration (Zhang's setup; aka canary-free)"
AWQ_ENRON_PARENT=$QDIR/awq_enron
mkdir -p "$AWQ_ENRON_PARENT"
"$PY" -m qquilt.quantize \
    --hf-dir "$UNLEARNED" --out-dir "$AWQ_ENRON_PARENT" \
    --quant AWQ \
    --awq-calibration-corpus "$CALIB_ENRON" \
    --awq-calib-n 128 --awq-calib-seed "$SEED" \
    --awq-bits 4 --awq-group-size 128

echo "[$(date +%T)] s4/3b AWQ-4bit + canary-only calibration (the experiment Zhang did NOT run)"
AWQ_CANARY_PARENT=$QDIR/awq_canary
mkdir -p "$AWQ_CANARY_PARENT"
"$PY" -m qquilt.quantize \
    --hf-dir "$UNLEARNED" --out-dir "$AWQ_CANARY_PARENT" \
    --quant AWQ \
    --awq-calibration-corpus "$CALIB_CANARY" \
    --awq-calib-n 128 --awq-calib-seed "$SEED" \
    --awq-bits 4 --awq-group-size 128

EXTRACT=$RESULTS/extraction.jsonl
echo "[$(date +%T)] s4/4 extract (greedy + n=5 stochastic) × {bf16, Q8/5/4, AWQ-enron, AWQ-canary}"
"$PY" -m qquilt.extract \
    --canaries-jsonl "$SOURCE_CANARIES" \
    --version "bf16_unlearned:hf:$UNLEARNED" \
    --version "q8_0:gguf:$QDIR/model-q8_0.gguf" \
    --version "q5_k_m:gguf:$QDIR/model-q5_k_m.gguf" \
    --version "q4_k_m:gguf:$QDIR/model-q4_k_m.gguf" \
    --version "awq_enron:awq:$AWQ_ENRON_PARENT/model-awq-4bit" \
    --version "awq_canary:awq:$AWQ_CANARY_PARENT/model-awq-4bit" \
    --llama-cli "$LLAMA_CLI" \
    --out "$EXTRACT" \
    --max-new-tokens 60 --seed "$SEED" \
    --n-stochastic 5 --top-p 0.9 --temperature 0.8 \
    --threads 6

METRICS=$RESULTS/metrics.json
echo "[$(date +%T)] s4/5 metrics — L3, A1 amp, per-version × per-bucket counts"
"$PY" -m qquilt.metrics \
    --extraction-jsonl "$EXTRACT" --canaries-jsonl "$SOURCE_CANARIES" \
    --baseline-version bf16_unlearned --min-match-chars 10 \
    --out "$METRICS"

echo "[$(date +%T)] s4 done — see $METRICS"
