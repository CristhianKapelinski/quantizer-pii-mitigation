#!/usr/bin/env bash
# Runs ON the secondary GPU host. Invoked (detached, via setsid+nohup) by
# scripts/run_qwen_extra_seeds_gpu2.sh on main. Runs the six Qwen
# extra-seed runs sequentially with QQUILT_SKIP_GGUF=1 (this host's
# prebuilt llama.cpp SIGILLs on its CPU, so the GGUF half is finalised on
# main). Idempotent per (model, seed). Writes a .done marker on exit so the
# main-side poller can tell "finished" from "still running" across reboots.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${QQUILT_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO"
export QQUILT_REPO="$REPO"
export HF_HOME="${HF_HOME:-$REPO/cache/hf}"
export TMPDIR="${TMPDIR:-$REPO/.tmp}"
export QQUILT_SKIP_GGUF=1
mkdir -p cache/hf .tmp
DONE_MARK="$REPO/.tmp/qwen_gpu2_driver.done"
rm -f "$DONE_MARK"
SEEDS="${QWEN_SEEDS:-42,52,62}"
ts() { date -Iseconds; }

for entry in "Qwen/Qwen2.5-0.5B-Instruct|wave_1_qwen05b" "Qwen/Qwen2.5-1.5B-Instruct|wave_1_qwen15b"; do
  IFS='|' read -r MID TAGPFX <<<"$entry"
  IFS=',' read -ra SARR <<<"$SEEDS"
  for SEED in "${SARR[@]}"; do
    TAG="${TAGPFX}_seed${SEED}"
    if [ -f "$REPO/experiment/results/$TAG/extraction.partial.jsonl" ] || [ -f "$REPO/experiment/results/$TAG/extraction.jsonl" ]; then
      echo "[$(ts)] [gpu2-remote] $TAG already has (partial) extraction -> skip"
      continue
    fi
    echo "[$(ts)] [gpu2-remote] === START $TAG (model $MID, seed $SEED, Adafactor, seq 512, SKIP_GGUF) ==="
    RUN_TAG="$TAG" MODEL_ID="$MID" BASE_MODEL_ID="$MID" SEED="$SEED" \
      REGIME=full MAX_SEQ=512 BS=1 ACCUM=16 EPOCHS=5 LR=2e-5 OPTIM=adafactor \
      bash "$SCRIPT_DIR/exp_extra_run.sh" \
      || echo "[$(ts)] [gpu2-remote] $TAG FAILED (continuing to next)"
    echo "[$(ts)] [gpu2-remote] === END $TAG ==="
  done
done
echo "[$(ts)] [gpu2-remote] all six runs attempted"
touch "$DONE_MARK"
