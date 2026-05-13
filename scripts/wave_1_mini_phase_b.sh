#!/usr/bin/env bash
# Wave 1 mini Phase B (WAVE_1_PLAN.md):
#   adds AWQ-4bit as a 5th quantization on top of Phase A's GGUF set,
#   re-runs extraction across all 5 versions, recomputes metrics + gate.
# Reuses the Phase A fine-tune checkpoint and corpus; does NOT re-train.
#
# Run only after Phase A's gate has been inspected.

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
LLAMA_CPP="${QQUILT_LLAMA_CPP:-$REPO/third_party/llama.cpp}"
LLAMA_QUANTIZE=$LLAMA_CPP/build/bin/llama-quantize
LLAMA_CLI=$LLAMA_CPP/build/bin/llama-cli
if [ -d "$LLAMA_CPP/build/lib" ]; then
    export LD_LIBRARY_PATH="$LLAMA_CPP/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

SEED=42

W1=$REPO/experiment/wave_1_mini
RESULTS=$REPO/experiment/results/wave_1_mini
CKPT=$REPO/checkpoints/wave_1_mini
QDIR=$REPO/checkpoints/wave_1_mini/quantized

CANARIES=$RESULTS/canaries.jsonl
G2=$RESULTS/g2.jsonl
G3=$RESULTS/g3.jsonl
CORPUS=$RESULTS/corpus.jsonl
EXTRACT=$RESULTS/extraction_phase_b.jsonl
METRICS=$RESULTS/metrics_w1_mini_phase_b.json

FINAL=$CKPT/final
AWQ_DIR=$QDIR/model-awq-4bit

if [ ! -d "$FINAL" ]; then
    echo "Phase A checkpoint not found at $FINAL — run scripts/wave_1_mini_smoke.sh first." >&2
    exit 1
fi

echo "[$(date +%T)] W1m-B/0 preflight"
"$PY" -m qquilt.preflight

if [ -d "$AWQ_DIR" ] && [ -f "$AWQ_DIR/config.json" ]; then
    echo "[$(date +%T)] W1m-B/1 AWQ already present at $AWQ_DIR — skipping quantize"
else
    echo "[$(date +%T)] W1m-B/1 quantize → AWQ-4bit (calibration: 128 enron chunks)"
    "$PY" -m qquilt.quantize \
        --hf-dir "$FINAL" --out-dir "$QDIR" \
        --quant AWQ \
        --awq-calibration-corpus "$CORPUS" \
        --awq-calib-n 128 --awq-calib-seed "$SEED" \
        --awq-bits 4 --awq-group-size 128
fi

echo "[$(date +%T)] W1m-B/2 extract (greedy + n=5 stochastic) × 5 versions × G1+G2+G3"
"$PY" -m qquilt.extract \
    --canaries-jsonl "$CANARIES" --g2-jsonl "$G2" --g3-jsonl "$G3" \
    --version "bf16:hf:$FINAL" \
    --version "q8_0:gguf:$QDIR/model-q8_0.gguf" \
    --version "q5_k_m:gguf:$QDIR/model-q5_k_m.gguf" \
    --version "q4_k_m:gguf:$QDIR/model-q4_k_m.gguf" \
    --version "awq_4bit:awq:$AWQ_DIR" \
    --llama-cli "$LLAMA_CLI" \
    --out "$EXTRACT" \
    --max-new-tokens 60 --seed "$SEED" \
    --n-stochastic 5 --top-p 0.9 --temperature 0.8

echo "[$(date +%T)] W1m-B/3 metrics + W1-mini gate (now 5 versions including AWQ)"
"$PY" -m qquilt.metrics \
    --extraction-jsonl "$EXTRACT" --canaries-jsonl "$CANARIES" \
    --baseline-version bf16 --min-match-chars 10 \
    --include-w1-mini-gate \
    --out "$METRICS"

echo "[$(date +%T)] W1m-B done — see $METRICS"
