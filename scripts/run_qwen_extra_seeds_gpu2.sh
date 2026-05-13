#!/usr/bin/env bash
# Cross-family extra seeds for Qwen-2.5 0.5B / 1.5B, dispatched on the
# secondary GPU host (12 GB-class GPU, e.g. RTX 3060).
#
# Six fresh WAVE_1 runs: {0.5B, 1.5B} x {seed 42, 52, 62}. Full FT, 5 ep,
# lr 2e-5, max_seq 512, bs 1 x accum 16, optimiser Adafactor (an AdamW
# state for 1.5B does not fit a 12 GB GPU; Adafactor's factored second
# moment does -- the cross-family question is whether the asymmetry
# replicates in another family/size, not an optimiser-controlled
# comparison), PLUS the AWQ-4bit version the original single-seed
# wave_1_qwen*_mini runs lacked. Supersedes those mini runs.
#
# Robust to the main host rebooting: the actual work runs *detached*
# (setsid + nohup) on the secondary host via run_qwen_on_gpu2_remote.sh,
# which writes a .done marker on exit. This script (on main) syncs the
# repo, launches/relaunches that remote driver as needed, and then polls:
# rsyncs each run's artifacts back as it appears and finalises the GGUF
# half on main (working llama.cpp). Re-running this script after a reboot
# resumes cleanly.
#
# Required env: GPU2_HOST (ssh target/alias) ; GPU2_REPO (remote repo path)
# Optional:     QQUILT_REPO ; QWEN_SEEDS (default 42,52,62) ; POLL_SECS (300)
#               ONESHOT=1 (do one sync+launch+rsync+finalise pass, then exit
#               -- for being driven by an external loop instead of polling)

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${QQUILT_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO"
: "${GPU2_HOST:?set GPU2_HOST=<ssh target>}"
: "${GPU2_REPO:?set GPU2_REPO=<remote repo path>}"
SEEDS="${QWEN_SEEDS:-42,52,62}"
POLL_SECS="${POLL_SECS:-300}"
DONE_MARK="$GPU2_REPO/.tmp/qwen_gpu2_driver.done"
ts() { date -Iseconds; }
say() { echo "[$(ts)] [qwen-gpu2] $*"; }

declare -a MODELS=(
  "Qwen/Qwen2.5-0.5B-Instruct|wave_1_qwen05b"
  "Qwen/Qwen2.5-1.5B-Instruct|wave_1_qwen15b"
)
IFS=',' read -ra SARR <<<"$SEEDS"

# note the [r] glob trick: keeps `pgrep -f` from matching its own argv
driver_alive() { ssh -o ConnectTimeout=10 "$GPU2_HOST" "pgrep -f '[r]un_qwen_on_gpu2_remote' >/dev/null" 2>/dev/null; }
driver_done()  { ssh -o ConnectTimeout=10 "$GPU2_HOST" "[ -f '$DONE_MARK' ]" 2>/dev/null; }
launch_driver() {
  say "launching detached remote driver on $GPU2_HOST"
  # timeout-guarded + ssh -n + trailing `exit`: if ssh fails to close the
  # channel after backgrounding, we give up after 30 s -- the driver is
  # setsid'd so it keeps running regardless, and the poll loop continues.
  timeout 30 ssh -n "$GPU2_HOST" "cd '$GPU2_REPO' && export HF_TOKEN='${HF_TOKEN:-}' HUGGING_FACE_HUB_TOKEN='${HUGGING_FACE_HUB_TOKEN:-}' QWEN_SEEDS='$SEEDS' && rm -f '$DONE_MARK' && setsid nohup bash scripts/run_qwen_on_gpu2_remote.sh > .tmp/qwen_gpu2_driver.log 2>&1 < /dev/null & echo \"  remote driver pid=\$!\"; exit 0" \
    || say "  (launch ssh returned non-zero/timed out -- driver is detached, continuing)"
}
all_finalised() {
  for entry in "${MODELS[@]}"; do
    IFS='|' read -r _ TAGPFX <<<"$entry"
    for SEED in "${SARR[@]}"; do
      [ -f "$REPO/experiment/results/${TAGPFX}_seed${SEED}/metrics.json" ] || return 1
    done
  done
  return 0
}
rsync_and_finalise() {
  for entry in "${MODELS[@]}"; do
    IFS='|' read -r MID TAGPFX <<<"$entry"
    for SEED in "${SARR[@]}"; do
      TAG="${TAGPFX}_seed${SEED}"
      [ -f "$REPO/experiment/results/$TAG/metrics.json" ] && [ -f "$REPO/experiment/results/$TAG/extraction.jsonl" ] && continue
      if ssh -o ConnectTimeout=10 "$GPU2_HOST" "[ -f '$GPU2_REPO/experiment/results/$TAG/extraction.partial.jsonl' ]" 2>/dev/null; then
        say "rsync $TAG back from $GPU2_HOST + finalise GGUF on main"
        mkdir -p "$REPO/experiment/results/$TAG" "$REPO/checkpoints/$TAG"
        rsync -a "$GPU2_HOST:$GPU2_REPO/experiment/results/$TAG/" "$REPO/experiment/results/$TAG/" || { say "rsync results $TAG failed"; continue; }
        rsync -a "$GPU2_HOST:$GPU2_REPO/checkpoints/$TAG/"          "$REPO/checkpoints/$TAG/"          || { say "rsync ckpt $TAG failed"; continue; }
        QQUILT_FINALIZE_GGUF=1 RUN_TAG="$TAG" MODEL_ID="$MID" BASE_MODEL_ID="$MID" SEED="$SEED" \
          bash "$SCRIPT_DIR/exp_extra_run.sh" || say "finalise $TAG FAILED (will retry next pass)"
      fi
    done
  done
}

# --- one pass: sync + (re)launch driver if needed + rsync/finalise ---
do_pass() {
  if all_finalised; then say "all six Qwen runs already finalised -> nothing to do"; return 0; fi
  if driver_alive; then
    say "remote driver alive on $GPU2_HOST -- pulling any completed runs"
  elif driver_done; then
    say "remote driver finished (.done present) -- pulling completed runs"
    rsync_and_finalise
    if all_finalised; then return 0; fi
    say "not all runs complete after a finished driver -- some failed; relaunching driver to retry"
    GPU2_HOST="$GPU2_HOST" GPU2_REPO="$GPU2_REPO" bash "$SCRIPT_DIR/sync_to_gpu2.sh" >/dev/null 2>&1 || say "WARN: re-sync failed"
    launch_driver
  else
    say "no remote driver -- syncing repo to $GPU2_HOST and launching"
    GPU2_HOST="$GPU2_HOST" GPU2_REPO="$GPU2_REPO" bash "$SCRIPT_DIR/sync_to_gpu2.sh" || { say "sync_to_gpu2 FAILED -- aborting"; return 2; }
    launch_driver
  fi
  rsync_and_finalise
  return 1   # not done yet
}

if [ "${ONESHOT:-}" = "1" ]; then
  do_pass; exit $?
fi

# poll until everything is finalised
while true; do
  do_pass && { say "qwen extra-seeds dispatch COMPLETE"; exit 0; }
  sleep "$POLL_SECS"
done
