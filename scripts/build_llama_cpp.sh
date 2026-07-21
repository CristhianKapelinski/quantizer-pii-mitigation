#!/usr/bin/env bash
# build_llama_cpp.sh — fetch + build llama.cpp at the pinned commit.
#
# CPU-only build. We use llama.cpp for GGUF conversion + k-quantization
# (llama-quantize), greedy/stochastic extraction (llama-cli), perplexity
# (llama-perplexity), and imatrix (llama-imatrix). The GPU work
# (fine-tuning, AWQ/GPTQ) is all PyTorch — llama.cpp never touches CUDA
# here, so a CPU-only build is correct and fastest to compile.
#
# Pinned version (also in EXPERIMENT_MANIFEST.yaml):
#   repo:   https://github.com/ggerganov/llama.cpp
#   tag:    b4404
#   commit: 0827b2c1da299805288abbd556d869318f2b121e
#
# Output:
#   third_party/llama.cpp/build/bin/{llama-cli,llama-quantize,
#                                    llama-perplexity,llama-imatrix}
#   third_party/llama.cpp/build/lib/*    (shared libs; the run scripts
#                                         add this to LD_LIBRARY_PATH)
#
# Notes
# -----
# * Default build enables GGML_NATIVE (-march=native). The committed
#   numbers were produced with a native build on a Zen4 host; a native
#   binary built on one micro-arch can SIGILL on another. If you will
#   run the same checkout on a *different* CPU (e.g. a second host you
#   rsync the build to), pass --portable to disable -march=native.
# * Needs: git, cmake >= 3.14, a C/C++ toolchain, make/ninja. No CUDA,
#   no Python deps.
#
# Usage:
#   bash scripts/build_llama_cpp.sh              # native build (matches ours)
#   bash scripts/build_llama_cpp.sh --portable   # portable (multi-host safe)
#   LLAMA_CPP_JOBS=8 bash scripts/build_llama_cpp.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${QQUILT_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"

LLAMA_CPP_DIR="${QQUILT_LLAMA_CPP:-$REPO/third_party/llama.cpp}"
LLAMA_CPP_REPO="${LLAMA_CPP_REPO:-https://github.com/ggerganov/llama.cpp}"
LLAMA_CPP_COMMIT="${LLAMA_CPP_COMMIT:-0827b2c1da299805288abbd556d869318f2b121e}"   # tag b4404
JOBS="${LLAMA_CPP_JOBS:-$(nproc 2>/dev/null || echo 4)}"

NATIVE=ON
for arg in "$@"; do
    case "$arg" in
        --portable) NATIVE=OFF ;;
        --native)   NATIVE=ON ;;
        *) echo "unknown arg: $arg (use --portable / --native)" >&2; exit 2 ;;
    esac
done

command -v git   >/dev/null || { echo "need: git";   exit 1; }
command -v cmake >/dev/null || { echo "need: cmake (>=3.14)"; exit 1; }

# --- fetch at the pinned commit -------------------------------------------
if [ ! -d "$LLAMA_CPP_DIR/.git" ] && [ ! -e "$LLAMA_CPP_DIR/CMakeLists.txt" ]; then
    echo "[build_llama_cpp] cloning $LLAMA_CPP_REPO -> $LLAMA_CPP_DIR"
    mkdir -p "$(dirname "$LLAMA_CPP_DIR")"
    git clone "$LLAMA_CPP_REPO" "$LLAMA_CPP_DIR"
fi
if [ -d "$LLAMA_CPP_DIR/.git" ]; then
    echo "[build_llama_cpp] checking out $LLAMA_CPP_COMMIT (tag b4404)"
    git -C "$LLAMA_CPP_DIR" fetch --tags --quiet origin || true
    git -C "$LLAMA_CPP_DIR" checkout --quiet "$LLAMA_CPP_COMMIT"
else
    echo "[build_llama_cpp] $LLAMA_CPP_DIR is not a git checkout; assuming it is"
    echo "                  already at the pinned commit and proceeding."
fi

# --- configure + build (CPU only) -----------------------------------------
BUILD_DIR="$LLAMA_CPP_DIR/build"
echo "[build_llama_cpp] cmake configure (CUDA off, GGML_NATIVE=$NATIVE)"
cmake -S "$LLAMA_CPP_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=OFF \
    -DGGML_NATIVE="$NATIVE" \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON

echo "[build_llama_cpp] building (-j$JOBS): llama-cli llama-quantize llama-perplexity llama-imatrix"
cmake --build "$BUILD_DIR" --config Release -j"$JOBS" \
    --target llama-cli llama-quantize llama-perplexity llama-imatrix

echo "[build_llama_cpp] done. binaries:"
ls -1 "$BUILD_DIR"/bin/llama-cli "$BUILD_DIR"/bin/llama-quantize \
       "$BUILD_DIR"/bin/llama-perplexity "$BUILD_DIR"/bin/llama-imatrix 2>/dev/null \
  || { echo "ERROR: expected binaries not found under $BUILD_DIR/bin" >&2; exit 1; }
echo "[build_llama_cpp] add to env when invoking the binaries:"
echo "    export LD_LIBRARY_PATH=\"$BUILD_DIR/lib:\$LD_LIBRARY_PATH\""
