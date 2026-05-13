#!/usr/bin/env bash
# Utility evaluation for AWQ-as-defense hypothesis.
#
# Computes held-out PPL on:
#   1. 500 Enron emails NOT in W1 mini training (in-domain)
#   2. First 1000 WikiText-2 sequences (OOD)
#
# Across six versions of the W1 mini Phase A fine-tune:
#   BF16, Q8_0, Q5_K_M, Q4_K_M, AWQ-canary-free, AWQ-canary-inclusive.
#
# Expected wallclock: ~ 1.5 h on RTX 5060 Ti.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${QQUILT_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO"

export HF_HOME="${HF_HOME:-$REPO/cache/hf}"
export TMPDIR="${TMPDIR:-$REPO/.tmp}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR" "$HF_HOME"

PY="$REPO/.venv/bin/python"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
LLAMA_CPP="${QQUILT_LLAMA_CPP:-$REPO/third_party/llama.cpp}"
LLAMA_PERPLEXITY=$LLAMA_CPP/build/bin/llama-perplexity
if [ -d "$LLAMA_CPP/build/lib" ]; then
    export LD_LIBRARY_PATH="$LLAMA_CPP/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if [ ! -x "$LLAMA_PERPLEXITY" ]; then
    echo "$LLAMA_PERPLEXITY not built. Build with:" >&2
    echo "  cmake --build $LLAMA_CPP/build --target llama-perplexity -j 4" >&2
    exit 1
fi

CKPT=$REPO/checkpoints/wave_1_mini
OUT=$REPO/experiment/results/wave_1_utility
mkdir -p "$OUT"

"$PY" -m qquilt.utility \
    --out-dir "$OUT" \
    --bf16-dir "$CKPT/final" \
    --gguf q8_0 "$CKPT/quantized/model-q8_0.gguf" \
    --gguf q5_k_m "$CKPT/quantized/model-q5_k_m.gguf" \
    --gguf q4_k_m "$CKPT/quantized/model-q4_k_m.gguf" \
    --awq-dir awq_canary_free "$CKPT/quantized/model-awq-4bit" \
    --awq-dir awq_canary_incl \
        "$REPO/experiment/results/wave_1_mini/step1_awq_canary_cal/quantized/model-awq-4bit" \
    --llama-perplexity "$LLAMA_PERPLEXITY" \
    --enron-holdout-n 500 \
    --wikitext-n 1000 \
    --max-seq-len 512 \
    --threads 8
