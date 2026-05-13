#!/usr/bin/env bash
# Sequential recovery: one heavy job at a time so the system stays responsive.
# Order: bucket-collapse (15 min) -> 5 small extractions (Qwen 0.5/1.5B) ->
# 3B-LoRA full extract (last, since it dominates CPU). Then re-run watchdog
# Phase 2 manually so the SBSeg artifact repo is force-pushed with correct data.
set -uo pipefail
cd /mnt/win_ssd/usenix
export GPU2_HOST="${GPU2_HOST:-deeppurple}"
export GPU2_REPO="${GPU2_REPO:-/home/cristhian/usenix}"
ts() { date +%Y-%m-%dT%H:%M:%S%z; }
say() { echo "[$(ts)] [seq] $*"; }
run_one() {
  local tag="$1"; shift
  local log="experiment/results/${tag}/run.seq.log"
  mkdir -p "experiment/results/${tag}"
  say "RUN $tag (log=$log)"
  "$@" > "$log" 2>&1
  local rc=$?
  say "DONE $tag rc=$rc"
  return $rc
}

# 0) bucket-collapse-canary first while no extraction RAM is needed.
say "=== Phase 0: bucket-collapse-canary (~15 min, ~16 GB RAM) ==="
mkdir -p experiment/results/exp_bucket_collapse_canary
PYTHONUNBUFFERED=1 .venv/bin/python scripts/exp_bucket_collapse_canary.py \
    --base unsloth/Llama-3.2-1B-Instruct \
    --ft   checkpoints/wave_1_mini/final \
    --awq  checkpoints/wave_1_mini/quantized/model-awq-4bit \
    --gptq experiment/results/exp_gptq_4bit/quantized/gptq_4bit \
    --out  experiment/results/exp_bucket_collapse_canary/metrics.json \
    > experiment/results/exp_bucket_collapse_canary/run.log 2>&1
say "  bucket-collapse rc=$?"

# 1) Qwen-0.5B-seed52 (full extract; smallest model first)
say "=== Phase 1: wave_1_qwen05b_seed52 (full 5-version extract) ==="
TAG=wave_1_qwen05b_seed52
RUN_TAG="$TAG" MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct" BASE_MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct" SEED=52 \
  bash scripts/exp_extra_run.sh > "experiment/results/$TAG/run.seq.log" 2>&1
say "  $TAG rc=$?"

# 2) Qwen-0.5B-seed62 FINALIZE
say "=== Phase 2: wave_1_qwen05b_seed62 FINALIZE ==="
TAG=wave_1_qwen05b_seed62
QQUILT_FINALIZE_GGUF=1 RUN_TAG="$TAG" MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct" BASE_MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct" SEED=62 \
  bash scripts/exp_extra_run.sh > "experiment/results/$TAG/run.seq.log" 2>&1
say "  $TAG rc=$?"

# 3..5) Qwen-1.5B seeds 42/52/62 FINALIZE (sequential)
for SEED in 42 52 62; do
  TAG="wave_1_qwen15b_seed${SEED}"
  say "=== Phase $((SEED == 42 ? 3 : SEED == 52 ? 4 : 5)): $TAG FINALIZE ==="
  QQUILT_FINALIZE_GGUF=1 RUN_TAG="$TAG" MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct" BASE_MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct" SEED="$SEED" \
    bash scripts/exp_extra_run.sh > "experiment/results/$TAG/run.seq.log" 2>&1
  say "  $TAG rc=$?"
done

# 6) 3B-LoRA full extract (last because it dominates CPU per llama-cli call)
say "=== Phase 6: wave_1_llama3b_lora_seed42 (full 5-version extract) ==="
TAG=wave_1_llama3b_lora_seed42
RUN_TAG="$TAG" MODEL_ID="unsloth/Llama-3.2-3B-Instruct" BASE_MODEL_ID="unsloth/Llama-3.2-3B-Instruct" SEED=42 \
  REGIME=lora MAX_SEQ=384 LR=2e-4 LORA_R=16 LORA_ALPHA=32 \
  bash scripts/exp_extra_run.sh > "experiment/results/$TAG/run.seq.log" 2>&1
say "  $TAG rc=$?"

# 7) Re-fire watchdog Phase 2 (it's idempotent: pool stats + write EXTRA_ANCHORS_RESULTS.md + commit + force-push artifact)
say "=== Phase 7: re-run overnight watchdog Phase 2 (writes EXTRA_ANCHORS_RESULTS.md + pushes SBSeg artifact) ==="
bash scripts/run_extra_anchors_overnight.sh >> experiment/results/overnight.log 2>&1
say "  watchdog rc=$?"

# 8) Reviewer experiments (idempotent: re-runs and force-pushes)
say "=== Phase 8: reviewer-experiments runner ==="
bash scripts/run_reviewer_experiments.sh >> experiment/results/reviewer_experiments.log 2>&1
say "  reviewer-exp rc=$?"

say "ALL PHASES COMPLETE"
