#!/usr/bin/env bash
# Launch 4-parallel Qwen finalize on gpu1 (gama, 32 cores / 125 GB), poll for
# results, then push corrected EXTRA_ANCHORS_RESULTS.md to the SBSeg artifact
# repo.
set -uo pipefail
: "${GAMA_HOST:=gpu1}"
ts() { date +%Y-%m-%dT%H:%M:%S%z; }
say() { echo "[$(ts)] [gama-launch] $*"; }
cd /mnt/win_ssd/usenix

# --- 1) wait until gama is ready (llama-cli built + each ckpt >= 1.5 GB) ----
say "waiting for gama setup (llama-cli built + each ckpt >= 1500 MB)..."
while true; do
  status=$(ssh -o ConnectTimeout=5 "$GAMA_HOST" \
    'echo "---built"; [ -x ~/usenix/third_party/llama.cpp/build/bin/llama-cli ] && echo BUILT || echo NOT_YET;
     echo "---ckpts"; du -sm ~/usenix/checkpoints/wave_1_qwen*_seed*/ 2>/dev/null' 2>/dev/null)
  built=$(echo "$status" | sed -n '/^---built$/,/^---/{ /^---/d; p }' | head -1)
  smallest=$(echo "$status" | sed -n '/^---ckpts$/,$p' | tail -n +2 | awk '{ if ($1+0 > 0 && (min == 0 || $1+0 < min)) min=$1+0 } END { print min+0 }')
  if [ "$built" = "BUILT" ] && [ "${smallest:-0}" -ge 1500 ]; then
    say "gama ready (llama-cli BUILT, smallest_ckpt=${smallest} MB)"
    break
  fi
  say "  build=$built smallest_ckpt_MB=$smallest -- recheck in 60s"
  sleep 60
done

# --- 2) launch parallel pool on gama (max 4 concurrent), detached ----------
say "launching parallel pool on gama (max 4)"
ssh "$GAMA_HOST" bash <<'REMOTE'
set -uo pipefail
cd "$HOME/usenix"
mkdir -p .tmp; rm -f .tmp/gama_done

RUNS=(
  "wave_1_qwen05b_seed52|Qwen/Qwen2.5-0.5B-Instruct|52|FULL"
  "wave_1_qwen05b_seed62|Qwen/Qwen2.5-0.5B-Instruct|62|FINALIZE"
  "wave_1_qwen15b_seed42|Qwen/Qwen2.5-1.5B-Instruct|42|FINALIZE"
  "wave_1_qwen15b_seed52|Qwen/Qwen2.5-1.5B-Instruct|52|FINALIZE"
  "wave_1_qwen15b_seed62|Qwen/Qwen2.5-1.5B-Instruct|62|FINALIZE"
)

# pool that launches up to 4 setsid'd workers, queues the 5th when one finishes
run_one() {
  local TAG="$1" MID="$2" SEED="$3" MODE="$4"
  local LOG="experiment/results/$TAG/run.gama.log"
  mkdir -p "experiment/results/$TAG"
  if [ "$MODE" = "FINALIZE" ]; then
    setsid nohup env QQUILT_FINALIZE_GGUF=1 RUN_TAG="$TAG" MODEL_ID="$MID" BASE_MODEL_ID="$MID" SEED="$SEED" \
      bash scripts/exp_extra_run.sh > "$LOG" 2>&1 < /dev/null &
  else
    setsid nohup env RUN_TAG="$TAG" MODEL_ID="$MID" BASE_MODEL_ID="$MID" SEED="$SEED" \
      bash scripts/exp_extra_run.sh > "$LOG" 2>&1 < /dev/null &
  fi
  echo $!
}

# wrap launch + completion-watch in a setsid'd parent so we can return
nohup setsid bash -c '
set -uo pipefail
cd "$HOME/usenix"
RUNS=(
  "wave_1_qwen05b_seed52|Qwen/Qwen2.5-0.5B-Instruct|52|FULL"
  "wave_1_qwen05b_seed62|Qwen/Qwen2.5-0.5B-Instruct|62|FINALIZE"
  "wave_1_qwen15b_seed42|Qwen/Qwen2.5-1.5B-Instruct|42|FINALIZE"
  "wave_1_qwen15b_seed52|Qwen/Qwen2.5-1.5B-Instruct|52|FINALIZE"
  "wave_1_qwen15b_seed62|Qwen/Qwen2.5-1.5B-Instruct|62|FINALIZE"
)
MAX_PARALLEL=4
WORKERS=()
for entry in "${RUNS[@]}"; do
  IFS="|" read -r TAG MID SEED MODE <<<"$entry"
  while [ "${#WORKERS[@]}" -ge "$MAX_PARALLEL" ]; do
    new=(); for p in "${WORKERS[@]}"; do kill -0 "$p" 2>/dev/null && new+=("$p"); done
    WORKERS=("${new[@]}")
    sleep 15
  done
  LOG="experiment/results/$TAG/run.gama.log"
  mkdir -p "experiment/results/$TAG"
  if [ "$MODE" = "FINALIZE" ]; then
    setsid nohup env QQUILT_FINALIZE_GGUF=1 RUN_TAG="$TAG" MODEL_ID="$MID" BASE_MODEL_ID="$MID" SEED="$SEED" \
      bash scripts/exp_extra_run.sh > "$LOG" 2>&1 < /dev/null &
  else
    setsid nohup env RUN_TAG="$TAG" MODEL_ID="$MID" BASE_MODEL_ID="$MID" SEED="$SEED" \
      bash scripts/exp_extra_run.sh > "$LOG" 2>&1 < /dev/null &
  fi
  WORKERS+=($!)
done
while [ "${#WORKERS[@]}" -gt 0 ]; do
  new=(); for p in "${WORKERS[@]}"; do kill -0 "$p" 2>/dev/null && new+=("$p"); done
  WORKERS=("${new[@]}")
  sleep 30
done
touch .tmp/gama_done
' > .tmp/gama_pool.log 2>&1 < /dev/null &
echo "[gama-remote] pool launched pid=$!"
REMOTE
say "remote pool launched"

# --- 3) local poller: rsync each result back as metrics.json lands ---------
say "polling gama for finished runs (rsync each back as metrics.json appears)"
TAGS=(wave_1_qwen05b_seed52 wave_1_qwen05b_seed62 wave_1_qwen15b_seed42 wave_1_qwen15b_seed52 wave_1_qwen15b_seed62)
declare -A SYNCED
for t in "${TAGS[@]}"; do SYNCED[$t]=0; done

while true; do
  all_done=1
  for TAG in "${TAGS[@]}"; do
    [ "${SYNCED[$TAG]}" = "1" ] && continue
    has=$(ssh -o ConnectTimeout=5 "$GAMA_HOST" "[ -f ~/usenix/experiment/results/$TAG/metrics.json ] && echo Y || echo N" 2>/dev/null)
    if [ "$has" = "Y" ]; then
      say "  $TAG metrics.json detected -> rsync back"
      rsync -a "$GAMA_HOST:~/usenix/experiment/results/$TAG/" "experiment/results/$TAG/" 2>&1 | tail -1
      SYNCED[$TAG]=1
    else
      all_done=0
    fi
  done
  if [ "$all_done" = "1" ]; then
    say "all 5 metrics.json synced back"
    break
  fi
  sleep 120
done

# --- 4) re-fire watchdog Phase 2 -> writes EXTRA_ANCHORS_RESULTS.md + commits + force-pushes SBSeg artifact ----
say "re-firing overnight watchdog (Phase 2 only this time; all 7 metrics.json present)"
bash scripts/run_extra_anchors_overnight.sh >> experiment/results/overnight.log 2>&1
say "  overnight rc=$?"

# --- 5) reviewer experiments runner (re-pushes REVIEWER_EXPERIMENTS_RESULTS.md) ----
say "re-running reviewer-experiments runner"
bash scripts/run_reviewer_experiments.sh >> experiment/results/reviewer_experiments.log 2>&1
say "  reviewer-exp rc=$?"

say "ALL DONE: SBSeg artifact repo (CristhianKapelinski/quantizer-pii-mitigation) now reflects real 3-seed Qwen + 3B-LoRA + bucket-collapse-canary"
