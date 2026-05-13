#!/usr/bin/env bash
# Generalised per-(model, seed, regime) run of the WAVE_1 pipeline:
#   canaries -> corpus(+retain) -> fine-tune -> GGUF Q8/Q5/Q4_K_M
#   -> AWQ-4bit (Enron-calibrated) -> extract (5 versions, greedy + n=5)
#   -> metrics -> weight-delta norm.
#
# Same recipe as scripts/exp_3seed_replication.sh::run_one_seed, but with
# the model id, seed, hyper-parameters and (full vs LoRA) regime exposed
# as env vars so it can serve the 3B scale anchor and the Qwen-2.5
# 0.5B / 1.5B extra-seed runs. Every step is idempotent (skip if its
# output already exists), so a killed/OOM run resumes cleanly.
#
# Required env:
#   MODEL_ID   HF model id (e.g. unsloth/Llama-3.2-3B-Instruct)
#   SEED       integer seed (canary draw + corpus shuffle + train + extract)
#   RUN_TAG    name for experiment/results/$RUN_TAG and checkpoints/$RUN_TAG
#
# Optional env (defaults in brackets):
#   REGIME      [full] | lora            full FT, or LoRA (merged before save)
#   MAX_SEQ     [384]                    tokeniser truncation length
#   BS          [1]                      per-device train batch size
#   ACCUM       [16]                     grad accumulation steps
#   EPOCHS      [5]                      train epochs
#   LR          [2e-5]                   learning rate
#   OPTIM       [adamw_torch]            transformers optimiser name
#                                        (use adafactor for big-model full FT)
#   LORA_R      [16]   LORA_ALPHA [32]   LORA_DROPOUT [0.05]
#   AWQ_GROUP_SIZE [128]  AWQ_CALIB_N [128]
#   ENRON_HF_ID [snoop2head/enron_aeslc_emails]
#   BASE_MODEL_ID [=$MODEL_ID]           base for the delta-norm diff
#   QQUILT_SKIP_GGUF [unset]             if set: skip the GGUF convert/quantize
#                                        and GGUF extraction; write a partial
#                                        extraction (bf16+awq) and stop before
#                                        metrics. For hosts whose llama.cpp
#                                        build is non-portable. The GGUF half
#                                        is then run on a host that can with:
#   QQUILT_FINALIZE_GGUF [unset]         if set (on a host with working
#                                        llama.cpp, after rsyncing the
#                                        SKIP_GGUF host's checkpoints/ and
#                                        results/ dirs): does only the GGUF
#                                        convert/quantize + GGUF extraction,
#                                        concatenates with extraction.partial
#                                        .jsonl into extraction.jsonl, then
#                                        metrics + delta-norm.
#   QQUILT_REPO  [auto]  QQUILT_LLAMA_CPP [$REPO/third_party/llama.cpp]
#
# Exit non-zero if any step fails (so an orchestrator can fall back, e.g.
# full-FT OOM -> LoRA).

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${QQUILT_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO"

: "${MODEL_ID:?set MODEL_ID}"
: "${SEED:?set SEED}"
: "${RUN_TAG:?set RUN_TAG}"
REGIME="${REGIME:-full}"
MAX_SEQ="${MAX_SEQ:-384}"
BS="${BS:-1}"
ACCUM="${ACCUM:-16}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-2e-5}"
OPTIM="${OPTIM:-adamw_torch}"
LORA_R="${LORA_R:-16}"; LORA_ALPHA="${LORA_ALPHA:-32}"; LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
AWQ_GROUP_SIZE="${AWQ_GROUP_SIZE:-128}"; AWQ_CALIB_N="${AWQ_CALIB_N:-128}"
ENRON_HF_ID="${ENRON_HF_ID:-snoop2head/enron_aeslc_emails}"
BASE_MODEL_ID="${BASE_MODEL_ID:-$MODEL_ID}"

export HF_HOME="${HF_HOME:-$REPO/cache/hf}"
export TMPDIR="${TMPDIR:-$REPO/.tmp}"
export TOKENIZERS_PARALLELISM=false
export QQUILT_MAX_SEQ_LEN="$MAX_SEQ"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$HF_HOME"
PY="$REPO/.venv/bin/python"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
LLAMA_CPP="${QQUILT_LLAMA_CPP:-$REPO/third_party/llama.cpp}"
LLAMA_QUANTIZE="$LLAMA_CPP/build/bin/llama-quantize"
LLAMA_CLI="$LLAMA_CPP/build/bin/llama-cli"
for d in "$LLAMA_CPP/build/lib" "$LLAMA_CPP/build/src" "$LLAMA_CPP/build/ggml/src"; do
    [ -d "$d" ] && export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done

W1="$REPO/experiment/results/$RUN_TAG"
CKPT="$REPO/checkpoints/$RUN_TAG"
mkdir -p "$W1" "$CKPT" "$CKPT/quantized"
CANARIES="$W1/canaries.jsonl"; CORPUS="$W1/corpus.jsonl"; RETAIN="$W1/retain.jsonl"
EXTRACT="$W1/extraction.jsonl"; PARTIAL="$W1/extraction.partial.jsonl"
FINAL="$CKPT/final"
ts() { date +%T; }
say() { echo "[$(ts)] [$RUN_TAG] $*"; }

say "config: MODEL_ID=$MODEL_ID SEED=$SEED REGIME=$REGIME MAX_SEQ=$MAX_SEQ BS=$BS ACCUM=$ACCUM EPOCHS=$EPOCHS LR=$LR OPTIM=$OPTIM LORA_R=$LORA_R SKIP_GGUF=${QQUILT_SKIP_GGUF:-0}"

if [ -f "$EXTRACT" ] && [ -f "$W1/metrics.json" ]; then
    say "already complete (extraction.jsonl + metrics.json present) -> nothing to do"
    exit 0
fi

# ---- FINALIZE mode: GGUF half of a run whose bf16+awq half came from a
#      non-portable-llama.cpp host (checkpoints/ + results/ already rsynced).
if [ -n "${QQUILT_FINALIZE_GGUF:-}" ]; then
    [ -d "$FINAL" ] || { say "FINALIZE: missing $FINAL (rsync checkpoints/$RUN_TAG first)"; exit 21; }
    [ -f "$PARTIAL" ] || { say "FINALIZE: missing $PARTIAL"; exit 21; }
    [ -f "$CANARIES" ] || { say "FINALIZE: missing $CANARIES"; exit 21; }
    if [ ! -f "$CKPT/quantized/model-q4_k_m.gguf" ]; then
        say "FINALIZE s4 GGUF convert + quantize"
        "$PY" -m qquilt.quantize --hf-dir "$FINAL" --out-dir "$CKPT/quantized" \
            --llama-cpp-dir "$LLAMA_CPP" --llama-quantize "$LLAMA_QUANTIZE" \
            --quant Q4_K_M --quant Q5_K_M --quant Q8_0 --python "$PY" || exit 24
    fi
    rm -f "$CKPT/quantized/model-f16.gguf"
    if [ ! -f "$W1/extraction.gguf.jsonl" ]; then
        say "FINALIZE s6b GGUF-only extract (q8_0 + q5_k_m + q4_k_m, greedy + n=5)"
        "$PY" -m qquilt.extract --canaries-jsonl "$CANARIES" \
            --version "q8_0:gguf:$CKPT/quantized/model-q8_0.gguf" \
            --version "q5_k_m:gguf:$CKPT/quantized/model-q5_k_m.gguf" \
            --version "q4_k_m:gguf:$CKPT/quantized/model-q4_k_m.gguf" \
            --llama-cli "$LLAMA_CLI" --out "$W1/extraction.gguf.jsonl" \
            --max-new-tokens 60 --seed "$SEED" --n-stochastic 5 \
            --top-p 0.9 --temperature 0.8 --threads 8 || exit 26
    fi
    cat "$PARTIAL" "$W1/extraction.gguf.jsonl" > "$EXTRACT"
    say "FINALIZE: merged partial + gguf rows -> $EXTRACT ($(wc -l < "$EXTRACT") rows)"
    "$PY" -m qquilt.metrics --extraction-jsonl "$EXTRACT" --canaries-jsonl "$CANARIES" \
        --baseline-version bf16 --min-match-chars 10 --out "$W1/metrics.json" || exit 27
    if [ ! -f "$W1/delta_norm.json" ]; then
        "$PY" "$SCRIPT_DIR/exp_delta_norm.py" --base-model-id "$BASE_MODEL_ID" \
            --final-dir "$FINAL" --out "$W1/delta_norm.json" || say "WARN: delta-norm failed (non-fatal)"
    fi
    say "FINALIZE DONE -> $EXTRACT  +  $W1/metrics.json"
    exit 0
fi

# 1. canaries
if [ ! -f "$CANARIES" ]; then
    say "s1 canaries (4 buckets 3/10/30/100 x25)"
    "$PY" -m qquilt.canaries --seed "$SEED" \
        --bucket 3:25 --bucket 10:25 --bucket 30:25 --bucket 100:25 --out "$CANARIES" || exit 11
fi

# 2. corpus + enron-only retain subset (AWQ calibration)
if [ ! -f "$CORPUS" ]; then
    say "s2 corpus (3000 enron + canary copies)"
    "$PY" -m qquilt.data --canaries-jsonl "$CANARIES" --n-emails 3000 \
        --seed "$SEED" --enron-hf-id "$ENRON_HF_ID" --out "$CORPUS" || exit 12
fi
if [ ! -f "$RETAIN" ]; then
    "$PY" - "$CORPUS" "$RETAIN" <<'PYEOF' || exit 12
import json, sys
src, dst = sys.argv[1], sys.argv[2]
with open(src) as f, open(dst, "w") as o:
    for line in f:
        r = json.loads(line)
        if r.get("source") == "enron":
            o.write(json.dumps(r, ensure_ascii=False) + "\n")
PYEOF
fi

# 3. fine-tune (full FT or LoRA, merged before save)
if [ ! -d "$FINAL" ]; then
    say "s3 fine-tune ($REGIME, $EPOCHS ep, lr $LR, optim $OPTIM, seq $MAX_SEQ, bs $BS x accum $ACCUM)"
    LORA_ARGS=()
    if [ "$REGIME" = "lora" ]; then
        LORA_ARGS=(--lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" --lora-dropout "$LORA_DROPOUT")
    fi
    "$PY" -m qquilt.train \
        --model-id "$MODEL_ID" --corpus-jsonl "$CORPUS" --out-dir "$CKPT" \
        --epochs "$EPOCHS" --learning-rate "$LR" --seed "$SEED" \
        --telemetry-jsonl "$W1/train_steps.jsonl" \
        --batch-size "$BS" --grad-accumulation "$ACCUM" --max-seq-len "$MAX_SEQ" \
        --optim "$OPTIM" "${LORA_ARGS[@]}" || { say "FINE-TUNE FAILED"; exit 13; }
fi

# 4. GGUF Q8/Q5/Q4_K_M  (skippable on non-portable-llama.cpp hosts)
GGUF_OK=0
if [ -n "${QQUILT_SKIP_GGUF:-}" ]; then
    say "s4 GGUF: SKIPPED (QQUILT_SKIP_GGUF set)"
else
    if [ ! -f "$CKPT/quantized/model-q4_k_m.gguf" ]; then
        say "s4 GGUF convert + quantize Q8_0/Q5_K_M/Q4_K_M"
        "$PY" -m qquilt.quantize --hf-dir "$FINAL" --out-dir "$CKPT/quantized" \
            --llama-cpp-dir "$LLAMA_CPP" --llama-quantize "$LLAMA_QUANTIZE" \
            --quant Q4_K_M --quant Q5_K_M --quant Q8_0 --python "$PY" || exit 14
    fi
    rm -f "$CKPT/quantized/model-f16.gguf"   # large intermediate, not needed
    GGUF_OK=1
fi

# 5. AWQ-4bit + Enron calibration
if [ ! -d "$CKPT/quantized/awq_enron/model-awq-4bit" ]; then
    say "s5 AWQ-4bit (group_size $AWQ_GROUP_SIZE, $AWQ_CALIB_N enron calib texts)"
    mkdir -p "$CKPT/quantized/awq_enron"
    "$PY" -m qquilt.quantize --hf-dir "$FINAL" --out-dir "$CKPT/quantized/awq_enron" \
        --quant AWQ --awq-calibration-corpus "$RETAIN" \
        --awq-calib-n "$AWQ_CALIB_N" --awq-calib-seed "$SEED" \
        --awq-bits 4 --awq-group-size "$AWQ_GROUP_SIZE" || exit 15
fi

# 6. extract
AWQ_DIR="$CKPT/quantized/awq_enron/model-awq-4bit"
if [ "$GGUF_OK" = "1" ]; then
    if [ ! -f "$EXTRACT" ]; then
        say "s6 extract (bf16 + q8_0 + q5_k_m + q4_k_m + awq_4bit, greedy + n=5)"
        "$PY" -m qquilt.extract --canaries-jsonl "$CANARIES" \
            --version "bf16:hf:$FINAL" \
            --version "q8_0:gguf:$CKPT/quantized/model-q8_0.gguf" \
            --version "q5_k_m:gguf:$CKPT/quantized/model-q5_k_m.gguf" \
            --version "q4_k_m:gguf:$CKPT/quantized/model-q4_k_m.gguf" \
            --version "awq_4bit:awq:$AWQ_DIR" \
            --llama-cli "$LLAMA_CLI" --out "$EXTRACT" \
            --max-new-tokens 60 --seed "$SEED" --n-stochastic 5 \
            --top-p 0.9 --temperature 0.8 --threads 8 || exit 16
    fi
else
    # GGUF skipped: partial extraction (bf16 + awq only), no metrics here.
    if [ ! -f "$PARTIAL" ]; then
        say "s6 partial extract (bf16 + awq_4bit only; GGUF half deferred to a portable host)"
        "$PY" -m qquilt.extract --canaries-jsonl "$CANARIES" \
            --version "bf16:hf:$FINAL" --version "awq_4bit:awq:$AWQ_DIR" \
            --out "$PARTIAL" --max-new-tokens 60 --seed "$SEED" --n-stochastic 5 \
            --top-p 0.9 --temperature 0.8 || exit 16
    fi
    say "partial run done (no GGUF). To finish: rsync checkpoints/$RUN_TAG + experiment/results/$RUN_TAG to a host with a working llama.cpp and re-run this script there without QQUILT_SKIP_GGUF."
    exit 0
fi

# 7. metrics
if [ ! -f "$W1/metrics.json" ]; then
    say "s7 metrics"
    "$PY" -m qquilt.metrics --extraction-jsonl "$EXTRACT" --canaries-jsonl "$CANARIES" \
        --baseline-version bf16 --min-match-chars 10 --out "$W1/metrics.json" || exit 17
fi

# 8. weight-delta norm vs base (mechanistic context)
if [ ! -f "$W1/delta_norm.json" ]; then
    say "s8 weight-delta norm vs $BASE_MODEL_ID"
    "$PY" "$SCRIPT_DIR/exp_delta_norm.py" --base-model-id "$BASE_MODEL_ID" \
        --final-dir "$FINAL" --out "$W1/delta_norm.json" \
        || say "WARN: delta-norm step failed (non-fatal)"
fi

say "DONE -> $EXTRACT  +  $W1/metrics.json"
