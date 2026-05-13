#!/usr/bin/env bash
# Wave 1 full (PLAN §7 Wave 1 expanded protocol):
#
# G1 = 500 canaries × 5 buckets {1, 3, 10, 30, 100}
# G2 = 300 Wikipedia pre-2023 (HF dataset)
# G3 = 300 post-2024 OOD passages (vetted source TBD; synthetic fallback)
# G4 = 200 paraphrase canaries (qquilt.canaries g4) at half-frequency
# G5 = DP-SGD ε=4 baseline (qquilt.dp_sgd, separate fine-tune)
#
# 7 quantizations: BF16, Q8_0, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K, AWQ-4bit
# Cross-calibration: Q4_K_M and AWQ each with Enron in-domain + wikitext OOD
# 3 Llama seeds {42, 52, 62} on main + 1 Qwen2.5-1.5B seed on gpu2
# greedy + n ∈ {10, 100} stochastic
#
# Expected wallclock with parallel main+gpu2: ~22 h fine-tune + 2 h
# quantize + 30+ h batched inference = ~55-70 h GPU. Plan to run over
# 2 weeks calendar.
#
# Idempotent: each step checks for prior outputs and skips. Safe to
# rerun after partial failures.

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

W1=$REPO/experiment/wave_1
RESULTS=$REPO/experiment/results/wave_1
CKPT_BASE=$REPO/checkpoints/wave_1
mkdir -p "$W1" "$RESULTS" "$CKPT_BASE"

CANARIES=$RESULTS/canaries.jsonl
G2=$RESULTS/g2.jsonl
G3=$RESULTS/g3.jsonl
G4=$RESULTS/g4.jsonl

N_EMAILS=${N_EMAILS:-30000}
EPOCHS=${EPOCHS:-5}
ENRON_HF_ID=${ENRON_HF_ID:-snoop2head/enron_aeslc_emails}
LLAMA_MODEL_ID=${LLAMA_MODEL_ID:-unsloth/Llama-3.2-1B-Instruct}
QWEN_MODEL_ID=${QWEN_MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}

echo "[$(date +%T)] W1F/0 preflight"
"$PY" -m qquilt.preflight

# --- 1. data assets (run once, reused across all seeds) -------------------

if [ ! -f "$CANARIES" ]; then
    echo "[$(date +%T)] W1F/1 generate G1 (500 canaries × 5 buckets {1, 3, 10, 30, 100})"
    "$PY" -m qquilt.canaries g1 --seed 42 \
        --bucket 1:100 --bucket 3:100 --bucket 10:100 --bucket 30:100 --bucket 100:100 \
        --out "$CANARIES"
fi

if [ ! -f "$G2" ]; then
    echo "[$(date +%T)] W1F/2a generate G2 (300 wikipedia pre-2023)"
    "$PY" -m qquilt.groups g2 --seed 42 --n 300 --out "$G2"
fi

if [ ! -f "$G3" ]; then
    echo "[$(date +%T)] W1F/2b generate G3 (300 OOD passages; synthetic fallback)"
    "$PY" -m qquilt.groups g3 --seed 42 --n 300 --synthetic --out "$G3"
fi

if [ ! -f "$G4" ]; then
    echo "[$(date +%T)] W1F/2c generate G4 (paraphrase of 200 G1 canaries, half freq)"
    "$PY" -m qquilt.canaries g4 --seed 42 --source-jsonl "$CANARIES" --n 200 --out "$G4"
fi

# --- 2. per-seed fine-tunes (Llama on main, parallel Qwen on gpu2) --------

for SEED in 42 52 62; do
    CKPT=$CKPT_BASE/llama-1b-seed-$SEED
    CORPUS=$RESULTS/corpus_seed_$SEED.jsonl
    TELE=$RESULTS/train_steps_seed_$SEED.jsonl

    if [ ! -f "$CORPUS" ]; then
        echo "[$(date +%T)] W1F/3a build corpus seed=$SEED (n=$N_EMAILS enron + G1+G4)"
        # Multi-file canary support: G1 + G4 inserted in one shuffled stream.
        "$PY" -m qquilt.data \
            --canaries-jsonl "$CANARIES" --canaries-jsonl "$G4" \
            --n-emails "$N_EMAILS" --seed "$SEED" \
            --enron-hf-id "$ENRON_HF_ID" --out "$CORPUS"
    fi

    if [ ! -d "$CKPT/final" ]; then
        echo "[$(date +%T)] W1F/3b fine-tune Llama seed=$SEED ($EPOCHS epochs)"
        "$PY" -m qquilt.train \
            --model-id "$LLAMA_MODEL_ID" --corpus-jsonl "$CORPUS" --out-dir "$CKPT" \
            --epochs "$EPOCHS" --seed "$SEED" --telemetry-jsonl "$TELE" \
            --batch-size 2 --grad-accumulation 8 --max-seq-len 512
    else
        echo "[$(date +%T)] W1F/3b skipping seed=$SEED (checkpoint exists)"
    fi
done

# --- 3. DP-SGD G5 baseline (Opacus) on Llama, seed 42 ---------------------

DP_CKPT=$CKPT_BASE/llama-1b-dp-seed-42
DP_TELE=$RESULTS/train_steps_dp_seed_42.jsonl
if [ ! -d "$DP_CKPT/final" ]; then
    echo "[$(date +%T)] W1F/4 DP-SGD ε=4 baseline (G5)"
    "$PY" -m qquilt.dp_sgd \
        --model-id "$LLAMA_MODEL_ID" --corpus-jsonl "$RESULTS/corpus_seed_42.jsonl" \
        --out-dir "$DP_CKPT" --epochs "$EPOCHS" --seed 42 \
        --telemetry-jsonl "$DP_TELE" \
        --batch-size 2 --max-seq-len 512 \
        --target-epsilon 4.0 --target-delta 1e-5 --max-grad-norm 1.0
fi

# --- 4. Qwen2.5 cross-family (gpu2) ---------------------------------------

QWEN_CKPT=$CKPT_BASE/qwen2.5-1.5b-seed-42
QWEN_TELE=$RESULTS/train_steps_qwen_seed_42.jsonl
if [ ! -d "$QWEN_CKPT/final" ]; then
    echo "[$(date +%T)] W1F/5 Qwen2.5-1.5B cross-family fine-tune (seed=42)"
    "$PY" -m qquilt.train \
        --model-id "$QWEN_MODEL_ID" --corpus-jsonl "$RESULTS/corpus_seed_42.jsonl" \
        --out-dir "$QWEN_CKPT" --epochs "$EPOCHS" --seed 42 \
        --telemetry-jsonl "$QWEN_TELE" \
        --batch-size 2 --grad-accumulation 8 --max-seq-len 512
fi

# --- 5. quantize each checkpoint to all 7 versions + cross-cal ablations ---

for CKPT in "$CKPT_BASE"/llama-1b-seed-42 "$CKPT_BASE"/llama-1b-seed-52 \
            "$CKPT_BASE"/llama-1b-seed-62 "$CKPT_BASE"/llama-1b-dp-seed-42 \
            "$CKPT_BASE"/qwen2.5-1.5b-seed-42; do
    [ -d "$CKPT/final" ] || continue
    QDIR=$CKPT/quantized
    if [ ! -f "$QDIR/model-q4_k_m.gguf" ]; then
        echo "[$(date +%T)] W1F/6 quantize $CKPT → 7 versions"
        "$PY" -m qquilt.quantize \
            --hf-dir "$CKPT/final" --out-dir "$QDIR" \
            --llama-cpp-dir "$LLAMA_CPP" --llama-quantize "$LLAMA_QUANTIZE" \
            --quant Q8_0 --quant Q5_K_M --quant Q4_K_M --quant Q3_K_M --quant Q2_K \
            --quant AWQ \
            --awq-calibration-corpus "$RESULTS/corpus_seed_42.jsonl" \
            --awq-calib-n 256 --awq-calib-seed 42 \
            --python "$PY"
    fi
done

# --- 6. extraction — one big run per checkpoint ---------------------------

for CKPT in "$CKPT_BASE"/llama-1b-seed-42 "$CKPT_BASE"/llama-1b-seed-52 \
            "$CKPT_BASE"/llama-1b-seed-62 "$CKPT_BASE"/llama-1b-dp-seed-42 \
            "$CKPT_BASE"/qwen2.5-1.5b-seed-42; do
    [ -d "$CKPT/final" ] || continue
    NAME=$(basename "$CKPT")
    EXTRACT=$RESULTS/extraction_$NAME.jsonl
    LOGITS=$RESULTS/logits_$NAME.jsonl
    if [ ! -f "$EXTRACT" ]; then
        echo "[$(date +%T)] W1F/7 extract $NAME"
        "$PY" -m qquilt.extract \
            --canaries-jsonl "$CANARIES" --g2-jsonl "$G2" --g3-jsonl "$G3" \
            --version "bf16:hf:$CKPT/final" \
            --version "q8_0:gguf:$CKPT/quantized/model-q8_0.gguf" \
            --version "q5_k_m:gguf:$CKPT/quantized/model-q5_k_m.gguf" \
            --version "q4_k_m:gguf:$CKPT/quantized/model-q4_k_m.gguf" \
            --version "q3_k_m:gguf:$CKPT/quantized/model-q3_k_m.gguf" \
            --version "q2_k:gguf:$CKPT/quantized/model-q2_k.gguf" \
            --version "awq_4bit:awq:$CKPT/quantized/model-awq-4bit" \
            --llama-cli "$LLAMA_CLI" \
            --out "$EXTRACT" --logits-out "$LOGITS" --top-k 20 \
            --max-new-tokens 60 --seed 42 \
            --n-stochastic 10 --top-p 0.9 --temperature 0.8
    fi
done

# --- 7. metrics + gate ----------------------------------------------------

echo "[$(date +%T)] W1F/8 metrics + W1 full gate"
# Per-seed metrics aggregated by qquilt.metrics; W1-full gate
# (5 conditions A/B/C/D/E + cross-family + cross-cal) computed by a
# dedicated qquilt.gate_w1 module to be added.
for CKPT in "$CKPT_BASE"/llama-1b-seed-42 "$CKPT_BASE"/llama-1b-seed-52 \
            "$CKPT_BASE"/llama-1b-seed-62 "$CKPT_BASE"/llama-1b-dp-seed-42 \
            "$CKPT_BASE"/qwen2.5-1.5b-seed-42; do
    [ -d "$CKPT/final" ] || continue
    NAME=$(basename "$CKPT")
    "$PY" -m qquilt.metrics \
        --extraction-jsonl "$RESULTS/extraction_$NAME.jsonl" \
        --canaries-jsonl "$CANARIES" \
        --baseline-version bf16 --min-match-chars 10 \
        --out "$RESULTS/metrics_$NAME.json"
done

echo "[$(date +%T)] W1F done — write experiment/results/wave_1/RESULTS.md"
