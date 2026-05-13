#!/usr/bin/env bash
# 5-seed stretch (paper-plan-v3 §Seeds: "5 seeds só se sobrar tempo após todos os
# experimentos críticos + ACR"). Runs the full Phase A + Phase B pipeline for seeds
# 72 and 82 (same recipe as exp_3seed_replication.sh), then re-pools all FIVE seeds
# {42,52,62,72,82} through exp_stats_aggregation.py. Idempotent per seed (skip if
# the seed's extraction.jsonl exists). Run only after the v3 roadmap (incl. ACR)
# has finished and the GPU is free. ~2.5 h per new seed (fine-tune ~1 h + quantize
# + extract ~80 min) + ~1 min re-pool => ~5 h.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${QQUILT_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO"

export HF_HOME="${HF_HOME:-$REPO/cache/hf}"
export TMPDIR="${TMPDIR:-$REPO/.tmp}"
export TOKENIZERS_PARALLELISM=false
export QQUILT_MAX_SEQ_LEN="${QQUILT_MAX_SEQ_LEN:-512}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$HF_HOME"
PY="$REPO/.venv/bin/python"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
LLAMA_CPP="$REPO/third_party/llama.cpp"
LLAMA_QUANTIZE="$LLAMA_CPP/build/bin/llama-quantize"
LLAMA_CLI="$LLAMA_CPP/build/bin/llama-cli"
[ -d "$LLAMA_CPP/build/lib" ] && export LD_LIBRARY_PATH="$LLAMA_CPP/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
MODEL_ID="${MODEL_ID:-unsloth/Llama-3.2-1B-Instruct}"
ENRON_HF_ID="${ENRON_HF_ID:-snoop2head/enron_aeslc_emails}"

run_one_seed () {
    local SEED="$1"
    local TAG="wave_1_seed${SEED}"
    local W1="$REPO/experiment/results/$TAG"
    local CKPT="$REPO/checkpoints/$TAG"
    mkdir -p "$W1" "$CKPT" "$CKPT/quantized"
    local CANARIES="$W1/canaries.jsonl" CORPUS="$W1/corpus.jsonl" EXTRACT="$W1/extraction.jsonl"
    if [ -f "$EXTRACT" ]; then echo "[$(date +%T)] seed=$SEED already done ($EXTRACT), skipping"; return; fi
    echo "[$(date +%T)] seed=$SEED s1 canaries"
    "$PY" -m qquilt.canaries --seed "$SEED" --bucket 3:25 --bucket 10:25 --bucket 30:25 --bucket 100:25 --out "$CANARIES" || return
    echo "[$(date +%T)] seed=$SEED s2 corpus"
    "$PY" -m qquilt.data --canaries-jsonl "$CANARIES" --n-emails 3000 --seed "$SEED" --enron-hf-id "$ENRON_HF_ID" --out "$CORPUS" || return
    "$PY" - <<PYEOF || return
import json
rows=[json.loads(l) for l in open("$CORPUS")]
with open("$W1/retain.jsonl","w") as f:
  for r in rows:
    if r.get("source")=="enron": f.write(json.dumps(r,ensure_ascii=False)+chr(10))
PYEOF
    echo "[$(date +%T)] seed=$SEED s3 fine-tune ($MODEL_ID, 5 ep)"
    "$PY" -m qquilt.train --model-id "$MODEL_ID" --corpus-jsonl "$CORPUS" --out-dir "$CKPT" --epochs 5 --learning-rate 2e-5 --seed "$SEED" --telemetry-jsonl "$W1/train_steps.jsonl" --batch-size 2 --grad-accumulation 8 --max-seq-len 512 || return
    local FINAL="$CKPT/final"
    echo "[$(date +%T)] seed=$SEED s4 quantize Q8/Q5/Q4 GGUF"
    "$PY" -m qquilt.quantize --hf-dir "$FINAL" --out-dir "$CKPT/quantized" --llama-cpp-dir "$LLAMA_CPP" --llama-quantize "$LLAMA_QUANTIZE" --quant Q4_K_M --quant Q5_K_M --quant Q8_0 --python "$PY" || return
    echo "[$(date +%T)] seed=$SEED s5 AWQ-4bit + Enron calib"
    mkdir -p "$CKPT/quantized/awq_enron"
    "$PY" -m qquilt.quantize --hf-dir "$FINAL" --out-dir "$CKPT/quantized/awq_enron" --quant AWQ --awq-calibration-corpus "$W1/retain.jsonl" --awq-calib-n 128 --awq-calib-seed "$SEED" --awq-bits 4 --awq-group-size 128 || return
    echo "[$(date +%T)] seed=$SEED s6 extract (5 versions)"
    "$PY" -m qquilt.extract --canaries-jsonl "$CANARIES" \
        --version "bf16:hf:$FINAL" --version "q8_0:gguf:$CKPT/quantized/model-q8_0.gguf" \
        --version "q5_k_m:gguf:$CKPT/quantized/model-q5_k_m.gguf" --version "q4_k_m:gguf:$CKPT/quantized/model-q4_k_m.gguf" \
        --version "awq_4bit:awq:$CKPT/quantized/awq_enron/model-awq-4bit" \
        --llama-cli "$LLAMA_CLI" --out "$EXTRACT" --max-new-tokens 60 --seed "$SEED" --n-stochastic 5 --top-p 0.9 --temperature 0.8 --threads 8 || return
    echo "[$(date +%T)] seed=$SEED done -> $EXTRACT"
}

run_one_seed 72
run_one_seed 82
echo "[$(date +%T)] re-pooled stats over 5 seeds {42,52,62,72,82}"
"$PY" scripts/exp_stats_aggregation.py --seeds 42,52,62,72,82 --out "$REPO/experiment/results/exp_3seed_replication/pooled_stats_5seed.json"
echo "[$(date +%T)] exp_5seed_extra done"
