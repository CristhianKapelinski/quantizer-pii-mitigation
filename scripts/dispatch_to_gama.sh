#!/usr/bin/env bash
# Dispatch the 5 Qwen finalize jobs to gpu1 (gama) where there are 32 CPU cores
# and 125 GB RAM. Pattern mirrors run_qwen_extra_seeds_gpu2.sh but with parallel
# (rather than sequential) workers because gama has the cores for it.
#
# Required env:  GAMA_HOST (ssh target, default 'gpu1'), GAMA_REPO (default
#                /home/$USER/usenix on gama). HF_TOKEN optional but speeds up
#                first model pull.
set -uo pipefail
: "${GAMA_HOST:=gpu1}"
: "${GAMA_REPO:=/home/cristhian/usenix}"
REPO_LOCAL="${REPO_LOCAL:-/mnt/win_ssd/usenix}"
ts() { date +%Y-%m-%dT%H:%M:%S%z; }
say() { echo "[$(ts)] [gama-disp] $*"; }

cd "$REPO_LOCAL"

# ---------- 1) sync code (no checkpoints, no caches) ----------------------
say "rsync code -> $GAMA_HOST:$GAMA_REPO"
ssh -o ConnectTimeout=10 "$GAMA_HOST" "mkdir -p $GAMA_REPO && mkdir -p $GAMA_REPO/.tmp" \
  || { say "ssh mkdir failed"; exit 2; }
rsync -a --info=stats1 \
  --exclude=".git/" --exclude=".venv/" --exclude="cache/" --exclude=".cache/" \
  --exclude="checkpoints/" --exclude="third_party/llama.cpp/build/" \
  --exclude="paper/" --exclude="refs/" --exclude="*.pdf" \
  --exclude="experiment/results/*/extraction*.jsonl" \
  --exclude="experiment/results/*/extraction.gguf.jsonl" \
  --exclude="experiment/results/*/run.*.log" \
  --exclude="experiment/results/*/metrics.json" \
  --exclude="experiment/results/*/canaries.jsonl" \
  --exclude="experiment/results/*/corpus.jsonl" \
  --exclude="experiment/results/*/retain.jsonl" \
  --exclude="experiment/results/*/g2.jsonl" --exclude="experiment/results/*/g3.jsonl" \
  --exclude="experiment/results/*/train_steps.jsonl" \
  --exclude="experiment/results/*/delta_norm.json" \
  --exclude="experiment/results/*/*.txt" \
  --exclude="*.log" \
  ./ "$GAMA_HOST:$GAMA_REPO/" \
  || { say "code rsync failed"; exit 3; }

# ---------- 2) build venv (uv) + llama.cpp (CPU-only) ---------------------
say "remote setup: uv sync + llama.cpp build"
ssh "$GAMA_HOST" bash -s <<REMOTE_SETUP
set -e
cd $GAMA_REPO
# uv (assumed installed; if not, document a fallback)
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not on PATH on gama; trying \$HOME/.local/bin/uv"
  export PATH="\$HOME/.local/bin:\$PATH"
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: install uv on $GAMA_HOST first (curl -LsSf https://astral.sh/uv/install.sh | sh)"
  exit 4
fi
[ -d .venv ] || { echo "[gama] uv sync"; uv sync --no-dev; }
# llama.cpp CPU build (matches local build_llama_cpp.sh defaults)
if [ ! -x third_party/llama.cpp/build/bin/llama-cli ]; then
  echo "[gama] cloning + building llama.cpp"
  bash scripts/build_llama_cpp.sh
fi
echo "[gama-setup] DONE."
REMOTE_SETUP
[ $? -ne 0 ] && { say "remote setup FAILED"; exit 5; }

# ---------- 3) sync the 5 pending Qwen runs (checkpoints + results) -------
RUNS=(
  "wave_1_qwen05b_seed52|Qwen/Qwen2.5-0.5B-Instruct|FULL"
  "wave_1_qwen05b_seed62|Qwen/Qwen2.5-0.5B-Instruct|FINALIZE"
  "wave_1_qwen15b_seed42|Qwen/Qwen2.5-1.5B-Instruct|FINALIZE"
  "wave_1_qwen15b_seed52|Qwen/Qwen2.5-1.5B-Instruct|FINALIZE"
  "wave_1_qwen15b_seed62|Qwen/Qwen2.5-1.5B-Instruct|FINALIZE"
)
for entry in "${RUNS[@]}"; do
  IFS='|' read -r TAG MID MODE <<<"$entry"
  say "rsync $TAG checkpoint + results subset ($MODE)"
  ssh "$GAMA_HOST" "mkdir -p $GAMA_REPO/checkpoints/$TAG $GAMA_REPO/experiment/results/$TAG"
  rsync -a \
    --exclude="extraction.jsonl" --exclude="extraction.gguf.jsonl" \
    --exclude="metrics.json" --exclude="delta_norm.json" \
    --exclude="run.*.log" \
    "experiment/results/$TAG/" "$GAMA_HOST:$GAMA_REPO/experiment/results/$TAG/" \
    || { say "results rsync $TAG FAILED"; continue; }
  rsync -a "checkpoints/$TAG/" "$GAMA_HOST:$GAMA_REPO/checkpoints/$TAG/" \
    || { say "ckpt rsync $TAG FAILED"; continue; }
done

# ---------- 4) launch parallel detached workers on gama -------------------
say "launching parallel workers on $GAMA_HOST (max 4 concurrent)"
ssh "$GAMA_HOST" bash -s <<'REMOTE_LAUNCH'
set -uo pipefail
cd "$HOME/usenix"
mkdir -p .tmp
rm -f .tmp/gama_done .tmp/gama_running.lock

RUNS=(
  "wave_1_qwen05b_seed52|Qwen/Qwen2.5-0.5B-Instruct|52|FULL"
  "wave_1_qwen05b_seed62|Qwen/Qwen2.5-0.5B-Instruct|62|FINALIZE"
  "wave_1_qwen15b_seed42|Qwen/Qwen2.5-1.5B-Instruct|42|FINALIZE"
  "wave_1_qwen15b_seed52|Qwen/Qwen2.5-1.5B-Instruct|52|FINALIZE"
  "wave_1_qwen15b_seed62|Qwen/Qwen2.5-1.5B-Instruct|62|FINALIZE"
)

run_one() {
  local TAG="$1" MID="$2" SEED="$3" MODE="$4"
  local LOG="experiment/results/$TAG/run.gama.log"
  mkdir -p "experiment/results/$TAG"
  echo "[gama] start $TAG ($MODE)" >&2
  if [ "$MODE" = "FINALIZE" ]; then
    QQUILT_FINALIZE_GGUF=1 RUN_TAG="$TAG" MODEL_ID="$MID" BASE_MODEL_ID="$MID" SEED="$SEED" \
      bash scripts/exp_extra_run.sh > "$LOG" 2>&1
  else
    RUN_TAG="$TAG" MODEL_ID="$MID" BASE_MODEL_ID="$MID" SEED="$SEED" \
      bash scripts/exp_extra_run.sh > "$LOG" 2>&1
  fi
  echo "[gama] DONE $TAG rc=$?" >&2
}

MAX_PARALLEL=4
WORKERS=()
for entry in "${RUNS[@]}"; do
  IFS='|' read -r TAG MID SEED MODE <<<"$entry"
  # if too many workers, wait for one to finish
  while [ "${#WORKERS[@]}" -ge "$MAX_PARALLEL" ]; do
    for i in "${!WORKERS[@]}"; do
      kill -0 "${WORKERS[$i]}" 2>/dev/null || unset 'WORKERS[i]'
    done
    WORKERS=("${WORKERS[@]}")
    sleep 5
  done
  run_one "$TAG" "$MID" "$SEED" "$MODE" &
  WORKERS+=($!)
done
wait
touch .tmp/gama_done
echo "[gama] all 5 runs complete"
REMOTE_LAUNCH

say "remote launch returned (workers detached; check .tmp/gama_done on $GAMA_HOST when polling)"
