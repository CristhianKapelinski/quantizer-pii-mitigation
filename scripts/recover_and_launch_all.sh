#!/usr/bin/env bash
# One-shot recovery: relaunch every job that died when the chat ended.
# Each line setsid's the work so it survives any future shell close.
set -uo pipefail
cd /mnt/win_ssd/usenix
ts() { date +%Y-%m-%dT%H:%M:%S%z; }
say() { echo "[$(ts)] [recover] $*"; }

# 1. Per-run finalizes (idempotent; train+quantize are skipped if outputs exist).
#    Each launched detached so it survives the launching shell.

# 3B-LoRA seed42 (full pipeline, finalize-after-train-skip)
TAG=wave_1_llama3b_lora_seed42
LOG="experiment/results/$TAG/run.recover.log"
say "launching $TAG (full pipeline; train will skip; extract fresh)"
setsid nohup env \
  RUN_TAG="$TAG" \
  MODEL_ID="unsloth/Llama-3.2-3B-Instruct" \
  BASE_MODEL_ID="unsloth/Llama-3.2-3B-Instruct" \
  SEED=42 \
  REGIME=lora \
  MAX_SEQ=384 \
  LR=2e-4 \
  LORA_R=16 \
  LORA_ALPHA=32 \
  bash scripts/exp_extra_run.sh > "$LOG" 2>&1 < /dev/null &
say "  pid=$!"

# Qwen-0.5B-seed52 (the truncated one: 112 bf16 + missing awq) -- full re-extract
TAG=wave_1_qwen05b_seed52
LOG="experiment/results/$TAG/run.recover.log"
say "launching $TAG (full re-extract; FT+AWQ+GGUF skipped if present)"
rm -f "experiment/results/$TAG/extraction.partial.jsonl"   # avoid concat with stale 112-row partial
setsid nohup env \
  RUN_TAG="$TAG" \
  MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct" \
  BASE_MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct" \
  SEED=52 \
  bash scripts/exp_extra_run.sh > "$LOG" 2>&1 < /dev/null &
say "  pid=$!"

# Qwen-0.5B-seed62 (finalize-only: gpu2 partial has bf16+awq; GGUFs done on main)
TAG=wave_1_qwen05b_seed62
LOG="experiment/results/$TAG/run.recover.log"
say "launching $TAG (FINALIZE GGUF-only; gpu2 partial has bf16+awq)"
setsid nohup env \
  QQUILT_FINALIZE_GGUF=1 \
  RUN_TAG="$TAG" \
  MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct" \
  BASE_MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct" \
  SEED=62 \
  bash scripts/exp_extra_run.sh > "$LOG" 2>&1 < /dev/null &
say "  pid=$!"

# Qwen-1.5B seeds 42/52/62 (finalize-only; all 3 GGUFs already quantized last session)
for SEED in 42 52 62; do
  TAG="wave_1_qwen15b_seed${SEED}"
  LOG="experiment/results/$TAG/run.recover.log"
  say "launching $TAG (FINALIZE GGUF-only)"
  setsid nohup env \
    QQUILT_FINALIZE_GGUF=1 \
    RUN_TAG="$TAG" \
    MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct" \
    BASE_MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct" \
    SEED="$SEED" \
    bash scripts/exp_extra_run.sh > "$LOG" 2>&1 < /dev/null &
  say "  pid=$!"
done

# 2. Bucket-collapse-canary (per-weight mechanism measurement)
say "launching bucket_collapse_canary"
mkdir -p experiment/results/exp_bucket_collapse_canary
setsid nohup env \
  PYTHONUNBUFFERED=1 \
  .venv/bin/python scripts/exp_bucket_collapse_canary.py \
    --base unsloth/Llama-3.2-1B-Instruct \
    --ft   checkpoints/wave_1_mini/final \
    --awq  checkpoints/wave_1_mini/quantized/model-awq-4bit \
    --gptq experiment/results/exp_gptq_4bit/quantized/gptq_4bit \
    --out  experiment/results/exp_bucket_collapse_canary/metrics.json \
  > experiment/results/exp_bucket_collapse_canary/run.log 2>&1 < /dev/null &
say "  pid=$!"

# 3. Daemons: overnight watchdog, Qwen poller, reviewer-experiments runner
say "launching overnight watchdog (run_extra_anchors_overnight.sh)"
setsid nohup bash scripts/run_extra_anchors_overnight.sh \
  >> experiment/results/overnight.log 2>&1 < /dev/null &
say "  pid=$!"

say "launching Qwen extra-seeds poller (run_qwen_extra_seeds_gpu2.sh) -- needs GPU2_HOST + GPU2_REPO env"
if [ -n "${GPU2_HOST:-}" ] && [ -n "${GPU2_REPO:-}" ]; then
  setsid nohup env GPU2_HOST="$GPU2_HOST" GPU2_REPO="$GPU2_REPO" \
    bash scripts/run_qwen_extra_seeds_gpu2.sh \
    >> experiment/results/qwen_extra_seeds_gpu2.log 2>&1 < /dev/null &
  say "  pid=$!"
else
  say "  SKIPPED: GPU2_HOST/GPU2_REPO unset (export before calling this script)"
fi

say "launching reviewer-experiments runner (run_reviewer_experiments.sh)"
setsid nohup bash scripts/run_reviewer_experiments.sh \
  >> experiment/results/reviewer_experiments.log 2>&1 < /dev/null &
say "  pid=$!"

sleep 3
say "all launches dispatched. sampling running PIDs:"
pgrep -af 'exp_extra_run|exp_bucket_collapse_canary|run_extra_anchors_overnight|run_qwen_extra_seeds_gpu2|run_reviewer_experiments' | sed 's/^/  /'
