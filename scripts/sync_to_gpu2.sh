#!/usr/bin/env bash
# Make the secondary GPU host ready to run W1+ jobs.
#
# Idempotent. Run from main host whenever the repo / venv / llama.cpp
# state changes. Re-runs safely.
#
# What it does:
#   1. Rsync the repo (sans .venv / build artefacts / caches) to $GPU2_REPO.
#   2. uv sync (no project install) on the secondary host to get the venv.
#   3. Rsync the prebuilt CPU-only llama.cpp binaries + shared libs
#      from main (the secondary host may lack cmake / sudo).
#   4. Verify llama-cli runs with LD_LIBRARY_PATH set.
#
# After this script exits, dispatch scripts can run on the remote host
# with the env vars below set on the remote shell. See
# scripts/step2_qwen_mini_gpu2.sh for the canonical wrapper.
#
# Required env (no defaults committed — set per host):
#   GPU2_HOST     SSH target, "user@host" form
#   GPU2_REPO     absolute path on the remote where the repo will live
#
# Optional env:
#   QQUILT_REPO   absolute path to this repo on the local (main) host;
#                 defaults to the parent directory of this script.

set -euo pipefail

REPO=${QQUILT_REPO:-$(cd "$(dirname "$0")/.." && pwd)}
: "${GPU2_HOST:?set GPU2_HOST=user@host before running}"
: "${GPU2_REPO:?set GPU2_REPO=/path/to/remote/repo before running}"

echo "[sync_to_gpu2] $(date -Iseconds) START"

ssh -o ConnectTimeout=10 "$GPU2_HOST" "
mkdir -p $GPU2_REPO/cache/hf $GPU2_REPO/.uv-cache $GPU2_REPO/.tmp \
         $GPU2_REPO/third_party/llama.cpp/build/lib"

echo "[sync_to_gpu2] rsyncing repo source"
rsync -a \
  --exclude='.venv/' --exclude='.uv-cache/' --exclude='.tmp/' \
  --exclude='cache/' --exclude='checkpoints/' \
  --exclude='third_party/llama.cpp/build/' \
  --exclude='third_party/llama.cpp/.git/' \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='.git/' \
  "$REPO/" "$GPU2_HOST:$GPU2_REPO/"

echo "[sync_to_gpu2] uv sync on remote (no project install)"
ssh "$GPU2_HOST" "
cd $GPU2_REPO && \
UV_CACHE_DIR=\$HOME/.uv-cache uv sync --no-install-project 2>&1 | tail -5"

echo "[sync_to_gpu2] rsyncing llama.cpp binaries + shared libs"
rsync -a "$REPO/third_party/llama.cpp/build/bin/llama-cli" \
         "$REPO/third_party/llama.cpp/build/bin/llama-quantize" \
         "$REPO/third_party/llama.cpp/build/bin/llama-imatrix" \
         "$GPU2_HOST:$GPU2_REPO/third_party/llama.cpp/build/bin/"
rsync -a "$REPO/third_party/llama.cpp/build/src/libllama.so" \
         "$REPO/third_party/llama.cpp/build/ggml/src/libggml.so" \
         "$REPO/third_party/llama.cpp/build/ggml/src/libggml-base.so" \
         "$REPO/third_party/llama.cpp/build/ggml/src/libggml-cpu.so" \
         "$GPU2_HOST:$GPU2_REPO/third_party/llama.cpp/build/lib/"

echo "[sync_to_gpu2] verifying llama-cli runs on gpu2"
ssh "$GPU2_HOST" "
LD_LIBRARY_PATH=$GPU2_REPO/third_party/llama.cpp/build/lib \
  $GPU2_REPO/third_party/llama.cpp/build/bin/llama-cli --version 2>&1 | head -2"

echo "[sync_to_gpu2] $(date -Iseconds) DONE"
