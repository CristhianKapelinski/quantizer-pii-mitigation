#!/usr/bin/env bash
# Wave 0 smoke (PLAN.md §9): mini-experiment that proves the pipeline runs
# end-to-end. Llama-3.2-1B-Instruct + 5 canaries × 50 + 200 Enron emails →
# 3-epoch BF16 fine-tune → BF16 / Q4_K_M / Q8_0 → 50-token greedy extraction →
# W0 gate verdict. Expected ~1 h on RTX 5060 Ti 16 GB.

set -euo pipefail

# Resolve repo from script location; override with QQUILT_REPO if needed.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${QQUILT_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO"

# Defaults are project-local; override with env to use a shared cache.
export HF_HOME="${HF_HOME:-$REPO/cache/hf}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO/.uv-cache}"
export TMPDIR="${TMPDIR:-$REPO/.tmp}"
export TOKENIZERS_PARALLELISM=false
export QQUILT_MAX_SEQ_LEN=512
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$UV_CACHE_DIR" "$TMPDIR" "$HF_HOME"

PY="$REPO/.venv/bin/python"
LLAMA_CPP="${QQUILT_LLAMA_CPP:-$REPO/third_party/llama.cpp}"
LLAMA_QUANTIZE=$LLAMA_CPP/build/bin/llama-quantize
LLAMA_CLI=$LLAMA_CPP/build/bin/llama-cli
if [ -d "$LLAMA_CPP/build/lib" ]; then
    export LD_LIBRARY_PATH="$LLAMA_CPP/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# W0 inputs (PLAN.md §5/§9).
SEED=42
N_CANARIES=5
FREQ=50
N_EMAILS=200
EPOCHS=3
ENRON_HF_ID=${ENRON_HF_ID:-snoop2head/enron_aeslc_emails}
MODEL_ID=${MODEL_ID:-unsloth/Llama-3.2-1B-Instruct}

W0=$REPO/experiment/wave_0
RESULTS=$REPO/experiment/results/wave_0
CKPT=$REPO/checkpoints/wave_0
QDIR=$REPO/checkpoints/wave_0/quantized
mkdir -p "$W0" "$RESULTS" "$CKPT" "$QDIR"

CANARIES=$RESULTS/canaries.jsonl
CORPUS=$RESULTS/corpus.jsonl
TELEMETRY=$RESULTS/train_steps.jsonl
EXTRACT=$RESULTS/extraction.jsonl
GATE=$RESULTS/gate_w0.json

echo "[$(date +%T)] W0/0 preflight"
"$PY" -m qquilt.preflight

echo "[$(date +%T)] W0/1 generate canaries"
"$PY" -m qquilt.canaries --seed "$SEED" --n-canaries "$N_CANARIES" --frequency "$FREQ" --out "$CANARIES"

echo "[$(date +%T)] W0/2 build corpus (n_emails=$N_EMAILS, dataset=$ENRON_HF_ID)"
"$PY" -m qquilt.data --canaries-jsonl "$CANARIES" --n-emails "$N_EMAILS" --seed "$SEED" \
    --enron-hf-id "$ENRON_HF_ID" --out "$CORPUS"

echo "[$(date +%T)] W0/3 fine-tune $MODEL_ID for $EPOCHS epochs (seed=$SEED)"
# batch 2 × grad_accum 8 = effective 16 (same as PLAN.md §5.1 default).
# Dropping per-device batch lowers activation peak that the loss kernel
# materialises; 16 GB is too tight for batch 4 + full-FT 1B without unsloth.
"$PY" -m qquilt.train \
    --model-id "$MODEL_ID" --corpus-jsonl "$CORPUS" --out-dir "$CKPT" \
    --epochs "$EPOCHS" --seed "$SEED" --telemetry-jsonl "$TELEMETRY" \
    --batch-size 2 --grad-accumulation 8 --max-seq-len 512

FINAL=$CKPT/final

echo "[$(date +%T)] W0/4 quantize → Q4_K_M, Q8_0"
"$PY" -m qquilt.quantize \
    --hf-dir "$FINAL" --out-dir "$QDIR" \
    --llama-cpp-dir "$LLAMA_CPP" --llama-quantize "$LLAMA_QUANTIZE" \
    --quant Q4_K_M --quant Q8_0 --python "$PY"

echo "[$(date +%T)] W0/5 greedy extraction × 3 versions"
"$PY" -m qquilt.extract \
    --canaries-jsonl "$CANARIES" \
    --version "bf16:hf:$FINAL" \
    --version "q8_0:gguf:$QDIR/model-q8_0.gguf" \
    --version "q4_k_m:gguf:$QDIR/model-q4_k_m.gguf" \
    --llama-cli "$LLAMA_CLI" \
    --out "$EXTRACT" \
    --max-new-tokens 60 --seed "$SEED"

echo "[$(date +%T)] W0/6 metrics 1b + 1c (stub)"
"$PY" -m qquilt.metrics \
    --extraction-jsonl "$EXTRACT" --canaries-jsonl "$CANARIES" \
    --baseline-version bf16 --min-match-chars 10 \
    --out "$RESULTS/metrics_w0.json"

echo "[$(date +%T)] W0/7 gate verdict"
"$PY" -m qquilt.gate --extraction-jsonl "$EXTRACT" --out "$GATE" --min-match-chars 10
