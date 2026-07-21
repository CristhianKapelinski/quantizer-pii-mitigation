#!/usr/bin/env bash
# Generic LoRA experiment: FT + AWQ + GGUF + extract
# Usage: bash lora_experiment.sh MODEL_ID SHORT_NAME SEED
set -euo pipefail
MODEL_ID=$1
SHORT=$2
SEED=$3
LORA_R=${LORA_R:-16}
REPO="${QQUILT_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"
PY=$REPO/.venv/bin/python
LLAMA_CPP=$REPO/third_party/llama.cpp
LLAMA_QUANTIZE=$LLAMA_CPP/build/bin/llama-quantize
LLAMA_CLI=$LLAMA_CPP/build/bin/llama-cli
export HF_HOME="${HF_HOME:-$REPO/cache/hf}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

RUN_TAG="wave_1_${SHORT}_lora_seed${SEED}"
CKPT=$REPO/checkpoints/$RUN_TAG
RES=$REPO/experiment/results/$RUN_TAG
mkdir -p $CKPT $RES

echo "[$(date +%T)] $RUN_TAG  (LoRA r=$LORA_R)"

[ -f $RES/canaries.jsonl ] || $PY -m qquilt.canaries g1 --seed $SEED \
    --bucket 3:25 --bucket 10:25 --bucket 30:25 --bucket 100:25 \
    --out $RES/canaries.jsonl
[ -f $RES/corpus.jsonl ] || $PY -m qquilt.data --canaries-jsonl $RES/canaries.jsonl \
    --n-emails 3000 --seed $SEED --out $RES/corpus.jsonl
[ -f $RES/retain.jsonl ] || $PY -m qquilt.data --canaries-jsonl /dev/null \
    --n-emails 3000 --seed $((SEED+1000)) --out $RES/retain.jsonl 2>/dev/null || \
    cp $RES/corpus.jsonl $RES/retain.jsonl

if [ ! -f $CKPT/final/adapter_model.safetensors ] && [ ! -f $CKPT/final/model.safetensors ]; then
    echo "[$(date +%T)] LoRA FT"
    $PY -m qquilt.train --model-id "$MODEL_ID" --corpus-jsonl $RES/corpus.jsonl \
        --out-dir $CKPT/final --seed $SEED --epochs 5 --learning-rate 2e-5 \
        --batch-size 1 --grad-accumulation 16 --max-seq-len 384 \
        --warmup-ratio 0.03 --weight-decay 0.0 --optim adamw_torch \
        --lora-r $LORA_R --lora-alpha 32 --lora-dropout 0.05 \
        --telemetry-jsonl $RES/train_steps.jsonl 2>&1 | tail -5
fi

# Quantize: AWQ + GGUF Q4_K_M (essencial), Q5_K_M, Q8_0
mkdir -p $CKPT/quantized
if [ ! -d $CKPT/quantized/awq_canary_free ] && [ ! -d $CKPT/quantized/awq ]; then
    $PY -m qquilt.quantize --hf-dir $CKPT/final --out-dir $CKPT/quantized \
        --quant AWQ --awq-calibration-corpus $RES/retain.jsonl \
        --awq-calib-n 128 --awq-calib-seed $SEED \
        --awq-bits 4 --awq-group-size 128 2>&1 | tail -3 || echo "AWQ fail"
fi
if [ ! -f $CKPT/quantized/model-q4_k_m.gguf ]; then
    $PY -m qquilt.quantize --hf-dir $CKPT/final --out-dir $CKPT/quantized \
        --quant Q4_K_M --quant Q5_K_M --quant Q8_0 \
        --llama-cpp-dir $LLAMA_CPP --llama-quantize $LLAMA_QUANTIZE 2>&1 | tail -3
fi

# Extract
VER="--version bf16:hf:$CKPT/final"
for d in awq_canary_free awq; do
    [ -d $CKPT/quantized/$d ] && VER="$VER --version awq_4bit:awq:$CKPT/quantized/$d" && break
done
[ -f $CKPT/quantized/model-q8_0.gguf ] && VER="$VER --version q8_0:gguf:$CKPT/quantized/model-q8_0.gguf"
[ -f $CKPT/quantized/model-q5_k_m.gguf ] && VER="$VER --version q5_k_m:gguf:$CKPT/quantized/model-q5_k_m.gguf"
[ -f $CKPT/quantized/model-q4_k_m.gguf ] && VER="$VER --version q4_k_m:gguf:$CKPT/quantized/model-q4_k_m.gguf"

$PY -m qquilt.extract --canaries-jsonl $RES/canaries.jsonl $VER \
    --out $RES/extraction.jsonl --seed $SEED --n-stochastic 5 \
    --temperature 0.8 --top-p 0.9 --llama-cli $LLAMA_CLI --threads 8 2>&1 | tail -3

$PY -m qquilt.metrics --extraction-jsonl $RES/extraction.jsonl \
    --canaries-jsonl $RES/canaries.jsonl --out $RES/metrics.json 2>&1 | tail -2

# Libera intermediarios
rm -f $CKPT/quantized/model-f16.gguf

echo "[$(date +%T)] $RUN_TAG DONE"
