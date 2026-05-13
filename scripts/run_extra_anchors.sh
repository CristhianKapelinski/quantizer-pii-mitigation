#!/usr/bin/env bash
# Orchestrator for the post-v3 "extra anchors" round (plan:
# experiment/plans/2026-05-12-scale-and-crossfamily-anchors.md):
#
#   * Llama-3.2-3B scale anchor in the full-FT regime (Adafactor, since a
#     16 GB GPU cannot hold a 3B AdamW state; bitsandbytes paged optimisers
#     are broken on this toolchain). Falls back to max_seq 256, then to a
#     LoRA-merged 3B run, if full FT OOMs.
#   * Llama-3.2-3B LoRA-merged anchor (always, as the PEFT-regime point and
#     the mechanistic contrast: smaller weight-delta should narrow the
#     AWQ-vs-Q4_K_M gap).
#   * Qwen-2.5 0.5B / 1.5B extra seeds {42,52,62} -- dispatched to the
#     secondary GPU in parallel (run_qwen_extra_seeds_gpu2.sh).
#
# Idempotent throughout (every sub-step skips if its output exists), so
# a kill / reboot resumes cleanly. Run detached:
#   nohup bash scripts/run_extra_anchors.sh > experiment/results/extra_anchors.log 2>&1 &
#
# Optional env:
#   RUN_GPU2=1   GPU2_HOST=<alias> GPU2_REPO=<path>   also kick the Qwen
#                extra-seed dispatch on the secondary host (background).
#   SKIP_3B_FULL=1   skip the full-FT 3B attempt (go straight to LoRA)
#   SKIP_3B_LORA=1   skip the LoRA 3B run

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${QQUILT_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO"
RUNNER="$SCRIPT_DIR/exp_extra_run.sh"
ts() { date -Iseconds; }
say() { echo "[$(ts)] [extra-anchors] $*"; }

LLAMA3B="unsloth/Llama-3.2-3B-Instruct"

GPU2_PID=""
if [ "${RUN_GPU2:-}" = "1" ]; then
  : "${GPU2_HOST:?RUN_GPU2=1 needs GPU2_HOST}"; : "${GPU2_REPO:?RUN_GPU2=1 needs GPU2_REPO}"
  say "kicking Qwen extra-seeds dispatch on $GPU2_HOST (background)"
  ( GPU2_HOST="$GPU2_HOST" GPU2_REPO="$GPU2_REPO" bash "$SCRIPT_DIR/run_qwen_extra_seeds_gpu2.sh" \
      > "$REPO/experiment/results/qwen_extra_seeds_gpu2.log" 2>&1 ) &
  GPU2_PID=$!
  say "  gpu2 dispatch pid=$GPU2_PID (log: experiment/results/qwen_extra_seeds_gpu2.log)"
fi

# ---------------------------------------------------------------------------
# 3B full-FT (Adafactor). Try seq 384, then 256. Tag: wave_1_llama3b_seed42.
# ---------------------------------------------------------------------------
FULL_OK=0
if [ "${SKIP_3B_FULL:-}" = "1" ]; then
  say "SKIP_3B_FULL=1 -> skipping full-FT 3B attempt"
elif [ -f "$REPO/experiment/results/wave_1_llama3b_seed42/metrics.json" ]; then
  say "3B full-FT already complete -> skip"; FULL_OK=1
else
  for SEQ in 384 256; do
    say "3B full-FT attempt: model=$LLAMA3B seed=42 optim=adafactor max_seq=$SEQ bs=1 accum=16"
    RUN_TAG=wave_1_llama3b_seed42 MODEL_ID="$LLAMA3B" BASE_MODEL_ID="$LLAMA3B" SEED=42 \
      REGIME=full OPTIM=adafactor MAX_SEQ="$SEQ" BS=1 ACCUM=16 EPOCHS=5 LR=2e-5 \
      bash "$RUNNER"
    rc=$?
    if [ $rc -eq 0 ]; then say "3B full-FT done (max_seq=$SEQ)"; FULL_OK=1; break; fi
    if [ $rc -eq 13 ]; then say "3B full-FT fine-tune failed at max_seq=$SEQ (likely OOM) -- trying smaller"; continue; fi
    say "3B full-FT failed at a non-FT step (rc=$rc) -- leaving partial state; will retry on rerun"; break
  done
  [ "$FULL_OK" = "1" ] || say "3B full-FT not achievable on this hardware (documented negative); the LoRA run below is the 3B point."
fi

# ---------------------------------------------------------------------------
# 3B LoRA-merged. Tag: wave_1_llama3b_lora_seed42.  r=16, alpha=32.
# ---------------------------------------------------------------------------
if [ "${SKIP_3B_LORA:-}" = "1" ]; then
  say "SKIP_3B_LORA=1 -> skipping LoRA 3B run"
else
  say "3B LoRA run: model=$LLAMA3B seed=42 r=16 alpha=32 max_seq=384 bs=4 accum=4"
  RUN_TAG=wave_1_llama3b_lora_seed42 MODEL_ID="$LLAMA3B" BASE_MODEL_ID="$LLAMA3B" SEED=42 \
    REGIME=lora LORA_R=16 LORA_ALPHA=32 LORA_DROPOUT=0.05 MAX_SEQ=384 BS=4 ACCUM=4 EPOCHS=5 LR=2e-5 \
    bash "$RUNNER" || say "3B LoRA run failed (rc=$?) -- see log"
fi

# ---------------------------------------------------------------------------
# Wait for the Qwen dispatch (if any) and report.
# ---------------------------------------------------------------------------
if [ -n "$GPU2_PID" ]; then
  say "waiting for gpu2 Qwen dispatch (pid $GPU2_PID) to finish..."
  wait "$GPU2_PID" || say "gpu2 Qwen dispatch exited non-zero -- see experiment/results/qwen_extra_seeds_gpu2.log (re-run scripts/run_qwen_extra_seeds_gpu2.sh to resume)"
fi

say "==== extra-anchors run finished ===="
say "results:"
for d in wave_1_llama3b_seed42 wave_1_llama3b_lora_seed42 \
         wave_1_qwen05b_seed42 wave_1_qwen05b_seed52 wave_1_qwen05b_seed62 \
         wave_1_qwen15b_seed42 wave_1_qwen15b_seed52 wave_1_qwen15b_seed62; do
  m="$REPO/experiment/results/$d/metrics.json"
  if [ -f "$m" ]; then say "  $d : metrics.json present"; else say "  $d : (incomplete / not run)"; fi
done
