#!/usr/bin/env bash
# Step 2 wrapper — dispatches Qwen2.5-0.5B-Instruct cross-family mini
# on the secondary GPU host (12 GiB-class GPU expected, e.g. RTX 3060).
#
# Pre-conditions: scripts/sync_to_gpu2.sh has prepared the remote's
# .venv + llama.cpp prebuilt binaries.
#
# Empirical wallclock on RTX 3060 12 GiB: ~7 h
#   fine-tune 1 h 25 min · quantize ~30 s · extract ~5–6 h
# The script runs synchronously in this shell — background it with
# `nohup ... &` if you don't want to wait.
#
# Required env (no defaults committed):
#   GPU2_HOST     SSH target, "user@host" form
#   GPU2_REPO     absolute path on the remote where the repo lives

set -euo pipefail

: "${GPU2_HOST:?set GPU2_HOST=user@host before running}"
: "${GPU2_REPO:?set GPU2_REPO=/path/to/remote/repo before running}"

ts() { date +%T; }
echo "[$(ts)] step2_qwen_mini_gpu2 dispatching on $GPU2_HOST"

ssh "$GPU2_HOST" "
set -euo pipefail
export QQUILT_REPO=$GPU2_REPO
export HF_HOME=$GPU2_REPO/cache/hf
export RUN_TAG=wave_1_qwen_mini
export MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
cd $GPU2_REPO
mkdir -p cache/hf .uv-cache .tmp
bash scripts/wave_1_mini_smoke.sh
"

echo "[$(ts)] step2_qwen_mini_gpu2 complete on $GPU2_HOST"
echo "Pull results back to main with:"
echo "  rsync -a \"\$GPU2_HOST:\$GPU2_REPO/experiment/results/wave_1_qwen_mini/\" \\"
echo "    experiment/results/wave_1_qwen_mini/"
