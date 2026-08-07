#!/usr/bin/env bash
# Reproduces the paper's headline cell from the released fine-tuned weights: download,
# quantize, attack, compare.
#
# Fine-tuning is the one step here that is not bit-reproducible. The same code and seed on
# a different GPU produce weights that differ slightly; those differences are invisible at
# 16 bits, survive 8-bit quantization, and are amplified by 4-bit k-quantization, which
# assigns each weight to one of 16 levels. Starting from the published weights removes that
# step, and everything after it is deterministic: quantizing the same checkpoint gives a
# byte-identical GGUF, and extracting from the same GGUF gives identical completions.
#
# Measured on a machine that never ran the fine-tune: 24 of the 100 planted canaries come
# back, the same 24 the paper reports, canary for canary.
set -euo pipefail
cd "$(dirname "$0")/.."

REL="https://github.com/CristhianKapelinski/quantizer-pii-mitigation/releases/download/checkpoint-v1"
TAR="wave_1_qwen05b_seed42-final.tar"
CELL="wave_1_qwen05b_seed42"
WORK="${QQUILT_CHECKPOINT_DIR:-checkpoints/$CELL-published}"
OUT="experiment/results/${CELL}_from_checkpoint"
LLAMA="${QQUILT_LLAMA_CPP:-$PWD/third_party/llama.cpp}"

PY="${PYTHON:-}"
[ -z "$PY" ] && { [ -x .venv/bin/python ] && PY=.venv/bin/python || PY="$(command -v python3)"; }
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$LLAMA/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

[ -x "$LLAMA/build/bin/llama-quantize" ] || {
  echo "llama.cpp is not built. Run: bash scripts/build_llama_cpp.sh" >&2; exit 1; }

mkdir -p "$WORK" "$OUT"

# 1. Fetch and verify. The checksum is published beside the archive; a truncated or
#    tampered download must fail here and not three steps later inside the quantizer.
if [ ! -d "$WORK/final" ]; then
  echo "== [1/4] downloading the published checkpoint (958 MB, once) =="
  curl -fSL --retry 3 -o "$WORK/$TAR" "$REL/$TAR"
  curl -fSL --retry 3 -o "$WORK/$TAR.sha256" "$REL/$TAR.sha256"
  ( cd "$WORK" && sha256sum -c "$TAR.sha256" )
  tar -xf "$WORK/$TAR" -C "$WORK"
  rm -f "$WORK/$TAR"
else
  echo "== [1/4] checkpoint already present in $WORK =="
fi

echo
echo "== [2/4] converting to GGUF and quantizing to Q4_K_M =="
[ -f "$WORK/model-f16.gguf" ] || \
  "$PY" "$LLAMA/convert_hf_to_gguf.py" "$WORK/final" --outfile "$WORK/model-f16.gguf" --outtype f16
[ -f "$WORK/model-q4_k_m.gguf" ] || \
  "$LLAMA/build/bin/llama-quantize" "$WORK/model-f16.gguf" "$WORK/model-q4_k_m.gguf" Q4_K_M

echo
echo "== [3/4] running the extraction attack =="
[ -f "$OUT/extraction.jsonl" ] || \
  "$PY" -m qquilt.extract --canaries-jsonl "experiment/results/$CELL/canaries.jsonl" \
    --version "q4_k_m:gguf:$WORK/model-q4_k_m.gguf" \
    --llama-cli "$LLAMA/build/bin/llama-cli" --out "$OUT/extraction.jsonl" \
    --max-new-tokens 60 --seed 42 --n-stochastic 0 --threads 8 --device cpu

echo
echo "== [4/4] comparing with the paper =="
"$PY" scripts/show_checkpoint_claim.py "$OUT/extraction.jsonl" "experiment/results/$CELL/extraction.jsonl"
