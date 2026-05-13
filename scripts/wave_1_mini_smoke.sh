#!/usr/bin/env bash
# Wave 1 mini Phase A (WAVE_1_PLAN.md):
# 100 canaries × 4 buckets {3, 10, 30, 100} (25 / bucket)
# 50 G2 (Wikipedia pre-2023) + 50 G3 (arXiv post-2024 OOD)  — eval-only
# 3000 Enron emails, 5 epochs, single seed (42)
# 4 quantizations: BF16 (HF) + Q8_0 + Q5_K_M + Q4_K_M (GGUF)
# greedy + n=5 stochastic per (sequence × version)
#
# Expected wallclock on RTX 5060 Ti 16 GB: ~3 hours
#   fine-tune ~85 min, quantize ~2 min, extract ~90 min, metrics ~1 min.
#
# Exits non-zero only on hard failure of the pipeline. Gate verdict is
# written to metrics_w1_mini.json (gate_w1_mini key).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${QQUILT_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO"

export HF_HOME="${HF_HOME:-$REPO/cache/hf}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO/.uv-cache}"
export TMPDIR="${TMPDIR:-$REPO/.tmp}"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$TMPDIR"
export TOKENIZERS_PARALLELISM=false
export QQUILT_MAX_SEQ_LEN=512
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"

PY="$REPO/.venv/bin/python"
# Fallback when the qquilt package is not installed in .venv (gpu2 case
# where uv sync was run with --no-install-project to skip the wheel build).
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
LLAMA_CPP=${QQUILT_LLAMA_CPP:-$REPO/third_party/llama.cpp}
LLAMA_QUANTIZE=$LLAMA_CPP/build/bin/llama-quantize
LLAMA_CLI=$LLAMA_CPP/build/bin/llama-cli
# Pre-built llama.cpp libs live in build/lib (rsync'd to gpu2);
# export so the llama-cli subprocess inherits the search path.
if [ -d "$LLAMA_CPP/build/lib" ]; then
    export LD_LIBRARY_PATH="$LLAMA_CPP/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

SEED=${SEED:-42}
N_EMAILS=${N_EMAILS:-3000}
EPOCHS=${EPOCHS:-5}
ENRON_HF_ID=${ENRON_HF_ID:-snoop2head/enron_aeslc_emails}
MODEL_ID=${MODEL_ID:-unsloth/Llama-3.2-1B-Instruct}
# Canary buckets — override BUCKETS env to test other freq regimes.
# Default = Phase A {3, 10, 30, 100} × 25 each.
# Step 3 freq=1: BUCKETS="--bucket 1:50 --bucket 3:25 --bucket 10:25"
BUCKETS=${BUCKETS:---bucket 3:25 --bucket 10:25 --bucket 30:25 --bucket 100:25}

# RUN_TAG keys output dirs; defaults to "wave_1_mini" so the original
# smoke is unchanged. Override (e.g. RUN_TAG=wave_1_qwen_mini) to run a
# variant in a sibling directory without touching the original.
RUN_TAG=${RUN_TAG:-wave_1_mini}

W1=$REPO/experiment/$RUN_TAG
RESULTS=$REPO/experiment/results/$RUN_TAG
CKPT=$REPO/checkpoints/$RUN_TAG
QDIR=$CKPT/quantized
mkdir -p "$W1" "$RESULTS" "$CKPT" "$QDIR"

CANARIES=$RESULTS/canaries.jsonl
G2=$RESULTS/g2.jsonl
G3=$RESULTS/g3.jsonl
CORPUS=$RESULTS/corpus.jsonl
TELEMETRY=$RESULTS/train_steps.jsonl
EXTRACT=$RESULTS/extraction.jsonl
METRICS=$RESULTS/metrics_w1_mini.json

echo "[$(date +%T)] W1m/0 preflight"
"$PY" -m qquilt.preflight

echo "[$(date +%T)] W1m/1 generate canaries (BUCKETS=$BUCKETS)"
"$PY" -m qquilt.canaries --seed "$SEED" $BUCKETS --out "$CANARIES"

echo "[$(date +%T)] W1m/2a generate G2 (50 wikipedia pre-2023)"
"$PY" -m qquilt.groups g2 --seed "$SEED" --n 50 --out "$G2"

echo "[$(date +%T)] W1m/2b generate G3 (50 synthetic OOD; W1-mini default)"
# Synthetic G3 is reproducible offline. W1-full will switch to a pinned
# post-2024 HF dataset once we have a vetted one in the manifest.
"$PY" -m qquilt.groups g3 --seed "$SEED" --n 50 --synthetic --out "$G3"

echo "[$(date +%T)] W1m/3 build corpus (n_emails=$N_EMAILS, ${ENRON_HF_ID})"
"$PY" -m qquilt.data --canaries-jsonl "$CANARIES" --n-emails "$N_EMAILS" --seed "$SEED" \
    --enron-hf-id "$ENRON_HF_ID" --out "$CORPUS"

echo "[$(date +%T)] W1m/4 fine-tune $MODEL_ID for $EPOCHS epochs (seed=$SEED)"
"$PY" -m qquilt.train \
    --model-id "$MODEL_ID" --corpus-jsonl "$CORPUS" --out-dir "$CKPT" \
    --epochs "$EPOCHS" --seed "$SEED" --telemetry-jsonl "$TELEMETRY" \
    --batch-size 2 --grad-accumulation 8 --max-seq-len 512

FINAL=$CKPT/final

echo "[$(date +%T)] W1m/5 quantize → Q4_K_M, Q5_K_M, Q8_0"
"$PY" -m qquilt.quantize \
    --hf-dir "$FINAL" --out-dir "$QDIR" \
    --llama-cpp-dir "$LLAMA_CPP" --llama-quantize "$LLAMA_QUANTIZE" \
    --quant Q4_K_M --quant Q5_K_M --quant Q8_0 --python "$PY"

echo "[$(date +%T)] W1m/6 extract (greedy + n=5 stochastic) × 4 versions × G1+G2+G3"
"$PY" -m qquilt.extract \
    --canaries-jsonl "$CANARIES" --g2-jsonl "$G2" --g3-jsonl "$G3" \
    --version "bf16:hf:$FINAL" \
    --version "q8_0:gguf:$QDIR/model-q8_0.gguf" \
    --version "q5_k_m:gguf:$QDIR/model-q5_k_m.gguf" \
    --version "q4_k_m:gguf:$QDIR/model-q4_k_m.gguf" \
    --llama-cli "$LLAMA_CLI" \
    --out "$EXTRACT" \
    --max-new-tokens 60 --seed "$SEED" \
    --n-stochastic 5 --top-p 0.9 --temperature 0.8

echo "[$(date +%T)] W1m/7 metrics 1b (per-bucket) + 1c (text-stub) + W1-mini gate"
"$PY" -m qquilt.metrics \
    --extraction-jsonl "$EXTRACT" --canaries-jsonl "$CANARIES" \
    --baseline-version bf16 --min-match-chars 10 \
    --include-w1-mini-gate \
    --out "$METRICS"

echo "[$(date +%T)] W1m done — see $METRICS"
