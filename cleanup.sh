#!/usr/bin/env bash
# Removes everything a run of this artifact created, inside the clone and outside it.
# It never touches anything tracked by git, and never removes the clone itself.
#
#   ./cleanup.sh --dry-run   list what would be removed, delete nothing
#   ./cleanup.sh             remove it
set -euo pipefail
cd "$(dirname "$0")"

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

total=0
gone() {   # gone <path> <what it is>
  local p="$1" what="$2" sz
  [ -e "$p" ] || return 0
  sz=$(du -sm "$p" 2>/dev/null | cut -f1); sz=${sz:-0}
  total=$((total + sz))
  printf '  %-42s %5s MB  %s\n' "$p" "$sz" "$what"
  [ "$DRY" = "1" ] || rm -rf "$p"
}

HF="${HF_HOME:-$PWD/cache/hf}"

echo "Removing what a run of this artifact leaves behind:"
gone .venv                     "the Python environment"
gone third_party               "the llama.cpp checkout and its build"
gone checkpoints               "fine-tuned and quantized models"
gone logs                      "run logs"
gone .pytest_cache             "test cache"
gone .ruff_cache               "lint cache"
# Only when the cache is the one this artifact created inside the clone. A HF_HOME
# pointing elsewhere is the evaluator's own cache, shared with everything else they run,
# and deleting it would take models this artifact never downloaded.
case "$HF" in
  "$PWD"/*) gone "$HF" "the Hugging Face cache created here" ;;
  *) printf '  %-42s %5s      %s\n' "$HF" "-" "your own HF cache, NOT removed" ;;
esac
for d in experiment/results/*_rerun; do gone "$d" "output of reproduce.sh quick"; done
for d in experiment/results/*_from_checkpoint; do gone "$d" "output of claim_from_checkpoint.sh"; done
for d in checkpoints/*-published; do gone "$d" "the downloaded published checkpoint"; done
for d in out-full out-live out; do gone "$d" "verification output"; done

echo
if [ "$DRY" = "1" ]; then
  echo "Dry run: nothing was removed. ${total} MB would be freed."
else
  echo "Done. ${total} MB freed. The committed logs, figures and results are untouched."
fi
