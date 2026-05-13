#!/usr/bin/env bash
# the paper plan v3 Exp 4 — 3-seed replication of Phase A + Phase B.
#
# Phase A (seed 42) + Phase B (seed 42) already done in wave_1_mini.
# This re-runs the full pipeline (fine-tune + GGUF quants + AWQ + extract)
# for seeds 52 and 62, then pools all three for Fisher-exact + Clopper-Pearson
# CI on the AWQ 0/100 vs Q4_K_M 6/100 asymmetry (statistical confidence).
#
# Per seed: ~3h fine-tune + ~1h quantize/extract = ~4h. Two seeds = ~8h.
# Runs on main GPU (1B 5-epoch fine-tune with Adam fp32 ~12GB peak — OOMs on
# the 12GB gpu2, so this stays on main). NOT parallelisable across GPUs.
#
# After both seeds, runs scripts/exp_stats_aggregation.py to produce the
# pooled cross-seed table with BH-FDR-adjusted p-values + 95% CIs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${QQUILT_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO"

export HF_HOME="${HF_HOME:-$REPO/cache/hf}"
export TMPDIR="${TMPDIR:-$REPO/.tmp}"
export TOKENIZERS_PARALLELISM=false
export QQUILT_MAX_SEQ_LEN=512
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR" "$HF_HOME"

PY="$REPO/.venv/bin/python"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
LLAMA_CPP="$REPO/third_party/llama.cpp"
LLAMA_QUANTIZE=$LLAMA_CPP/build/bin/llama-quantize
LLAMA_CLI=$LLAMA_CPP/build/bin/llama-cli
if [ -d "$LLAMA_CPP/build/lib" ]; then
    export LD_LIBRARY_PATH="$LLAMA_CPP/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

MODEL_ID=${MODEL_ID:-unsloth/Llama-3.2-1B-Instruct}
ENRON_HF_ID=${ENRON_HF_ID:-snoop2head/enron_aeslc_emails}

run_one_seed () {
    local SEED="$1"
    local TAG="wave_1_seed${SEED}"
    local W1="$REPO/experiment/results/$TAG"
    local CKPT="$REPO/checkpoints/$TAG"
    mkdir -p "$W1" "$CKPT" "$CKPT/quantized"
    local CANARIES="$W1/canaries.jsonl"
    local CORPUS="$W1/corpus.jsonl"
    local EXTRACT="$W1/extraction.jsonl"

    if [ -f "$EXTRACT" ]; then
        echo "[$(date +%T)] seed=$SEED already done ($EXTRACT exists), skipping"
        return
    fi

    echo "[$(date +%T)] seed=$SEED s1 canaries (same 4-bucket {3,10,30,100} mix)"
    "$PY" -m qquilt.canaries --seed "$SEED" \
        --bucket 3:25 --bucket 10:25 --bucket 30:25 --bucket 100:25 --out "$CANARIES"

    echo "[$(date +%T)] seed=$SEED s2 build corpus (3000 enron + canaries)"
    "$PY" -m qquilt.data --canaries-jsonl "$CANARIES" --n-emails 3000 \
        --seed "$SEED" --enron-hf-id "$ENRON_HF_ID" --out "$CORPUS"
    # enron-only retain subset (for the AWQ calibration)
    "$PY" - <<PYEOF
import json
rows = [json.loads(l) for l in open("$CORPUS")]
enron = [r for r in rows if r.get("source") == "enron"]
with open("$W1/retain.jsonl", "w") as f:
    for r in enron: f.write(json.dumps(r, ensure_ascii=False) + chr(10))
PYEOF

    echo "[$(date +%T)] seed=$SEED s3 fine-tune ($MODEL_ID, 5 epochs, lr 2e-5)"
    "$PY" -m qquilt.train \
        --model-id "$MODEL_ID" --corpus-jsonl "$CORPUS" --out-dir "$CKPT" \
        --epochs 5 --learning-rate 2e-5 --seed "$SEED" \
        --telemetry-jsonl "$W1/train_steps.jsonl" \
        --batch-size 2 --grad-accumulation 8 --max-seq-len 512

    local FINAL="$CKPT/final"
    echo "[$(date +%T)] seed=$SEED s4 quantize -> Q8/Q5/Q4 GGUF"
    "$PY" -m qquilt.quantize \
        --hf-dir "$FINAL" --out-dir "$CKPT/quantized" \
        --llama-cpp-dir "$LLAMA_CPP" --llama-quantize "$LLAMA_QUANTIZE" \
        --quant Q4_K_M --quant Q5_K_M --quant Q8_0 --python "$PY"

    echo "[$(date +%T)] seed=$SEED s5 AWQ-4bit + Enron calibration"
    mkdir -p "$CKPT/quantized/awq_enron"
    "$PY" -m qquilt.quantize \
        --hf-dir "$FINAL" --out-dir "$CKPT/quantized/awq_enron" \
        --quant AWQ \
        --awq-calibration-corpus "$W1/retain.jsonl" \
        --awq-calib-n 128 --awq-calib-seed "$SEED" \
        --awq-bits 4 --awq-group-size 128

    echo "[$(date +%T)] seed=$SEED s6 extract (5 versions, greedy + n=5)"
    "$PY" -m qquilt.extract \
        --canaries-jsonl "$CANARIES" \
        --version "bf16:hf:$FINAL" \
        --version "q8_0:gguf:$CKPT/quantized/model-q8_0.gguf" \
        --version "q5_k_m:gguf:$CKPT/quantized/model-q5_k_m.gguf" \
        --version "q4_k_m:gguf:$CKPT/quantized/model-q4_k_m.gguf" \
        --version "awq_4bit:awq:$CKPT/quantized/awq_enron/model-awq-4bit" \
        --llama-cli "$LLAMA_CLI" \
        --out "$EXTRACT" \
        --max-new-tokens 60 --seed "$SEED" \
        --n-stochastic 5 --top-p 0.9 --temperature 0.8 --threads 8

    echo "[$(date +%T)] seed=$SEED done -> $EXTRACT"
}

run_one_seed 52
run_one_seed 62

echo "[$(date +%T)] pooled stats — Fisher exact + Clopper-Pearson CI + BH-FDR"
"$PY" scripts/exp_stats_aggregation.py \
    --seeds 42,52,62 \
    --out "$REPO/experiment/results/exp_3seed_replication/pooled_stats.json"

echo "[$(date +%T)] exp_3seed done"
