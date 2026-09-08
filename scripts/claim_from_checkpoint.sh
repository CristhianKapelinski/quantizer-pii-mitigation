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

# Every external tool used below, named here instead of failing mid-download.
need_tools() {
  local missing="" t
  for t in "$@"; do command -v "$t" >/dev/null 2>&1 || missing="$missing $t"; done
  [ -z "$missing" ] && return 0
  echo "missing required tool(s):$missing" >&2
  echo "  Debian/Ubuntu: sudo apt update && sudo apt install -y$missing" >&2
  echo "  Fedora/RHEL:   sudo dnf install -y$missing" >&2
  exit 1
}
need_tools curl sha256sum tar

# The checkpoints are archived on Zenodo (DOI 10.5281/zenodo.22666122). Anonymous download,
# no account and no token.
REL="${QQUILT_CHECKPOINT_URL:-https://zenodo.org/records/22666123/files}"
TAR="wave_1_qwen05b_seed42-final.tar"
AWQ_TAR="wave_1_qwen05b_seed42-awq.tar"
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

# Each archive is checked against a digest committed in this repository
# (expected/external_artifacts.json), not against a .sha256 file served from the
# same host as the archive. A checksum fetched beside the download proves the
# transfer was not truncated; it cannot tell you the bytes are still the ones the
# paper measured, because whoever replaces one file replaces the other.
fetch_verified() {   # fetch_verified <archive-name> <human description>
  local name="$1" what="$2" want
  want="$("$PY" - "$name" <<'PYEOF'
import json, sys
print(json.load(open("expected/external_artifacts.json"))["downloads"]["files"][sys.argv[1]]["sha256"])
PYEOF
)"
  echo "   fetching $what"
  curl -fSL --retry 3 -o "$WORK/$name" "$REL/$name"
  printf '%s  %s\n' "$want" "$name" > "$WORK/$name.sha256"
  if ! ( cd "$WORK" && sha256sum -c "$name.sha256" ); then
    echo "the downloaded $name does not match the digest committed in" >&2
    echo "expected/external_artifacts.json. Do not proceed: the published weights" >&2
    echo "are no longer the ones the paper's numbers were measured from." >&2
    exit 1
  fi
  tar -xf "$WORK/$name" -C "$WORK"
  rm -f "$WORK/$name" "$WORK/$name.sha256"
}

# AWQ inference runs through torch CUDA kernels, so it can only be measured on a machine
# with an NVIDIA GPU. Decide that here, before downloading: a CPU-only reviewer should not
# spend 452 MB on a model this machine cannot run.
AWQ_ARGS=(); DEV=cpu; HAVE_CUDA=0
# Not `torch.cuda.is_available()`: that says a GPU was found, not that this torch can
# drive it. An older card enumerates fine and then kills the AWQ load with "no kernel
# image is available for execution on the device" -- after the k-quants have been built
# and attacked, so the whole run is lost. cuda_usable.py compares the device's compute
# capability with the architectures torch was compiled for, and prints why it said no.
if "$PY" "$(dirname "$0")/cuda_usable.py"; then
  HAVE_CUDA=1; DEV=cuda
  echo "GPU detected: both the k-quants and AWQ will be measured on this machine."
else
  echo "No GPU: the k-quants are measured here; the AWQ side of the comparison is read"
  echo "from the paper, and the result block says so. Nothing else changes."
fi

# 1. Fetch and verify against the committed digest: a truncated, replaced or tampered
#    download must fail here and not three steps later inside the quantizer.
echo "== [1/4] fetching the published weights (once) =="
if [ ! -d "$WORK/final" ]; then
  fetch_verified "$TAR" "the fine-tuned checkpoint (958 MB)"
fi

# The AWQ model is published already quantized: producing it needs a GPU and the autoawq
# stack, and the comparison it enables is the point of the claim.
if [ "$HAVE_CUDA" = 1 ] && [ ! -d "$WORK/model-awq-4bit" ]; then
  fetch_verified "$AWQ_TAR" "the AWQ model (452 MB)"
fi

echo
echo "== [2/4] converting to GGUF and quantizing =="
[ -f "$WORK/model-f16.gguf" ] || \
  "$PY" "$LLAMA/convert_hf_to_gguf.py" "$WORK/final" --outfile "$WORK/model-f16.gguf" --outtype f16
for q in Q8_0 Q5_K_M Q4_K_M; do
  low=$(echo "$q" | tr 'A-Z' 'a-z')
  [ -f "$WORK/model-$low.gguf" ] || \
    "$LLAMA/build/bin/llama-quantize" "$WORK/model-f16.gguf" "$WORK/model-$low.gguf" "$q"
done

echo
echo "== [3/4] running the extraction attack =="
if [ "$HAVE_CUDA" = 1 ]; then AWQ_ARGS=(--version "awq_4bit:awq:$WORK/model-awq-4bit"); fi
# AWQ inference needs CUDA: its kernels are GPU-only. Without a GPU the k-quants are still
# measured and the AWQ row is read from the paper instead of being re-measured, which the
# result block states rather than hiding.
[ -f "$OUT/extraction.jsonl" ] || \
  "$PY" -m qquilt.extract --canaries-jsonl "experiment/results/$CELL/canaries.jsonl" \
    --version "q8_0:gguf:$WORK/model-q8_0.gguf" \
    --version "q5_k_m:gguf:$WORK/model-q5_k_m.gguf" \
    --version "q4_k_m:gguf:$WORK/model-q4_k_m.gguf" \
    "${AWQ_ARGS[@]}" \
    --llama-cli "$LLAMA/build/bin/llama-cli" --out "$OUT/extraction.jsonl" \
    --max-new-tokens 60 --seed 42 --n-stochastic 0 --threads 8 --device "$DEV"

echo
echo "== [4/4] comparing with the paper =="
"$PY" scripts/show_checkpoint_claim.py "$OUT/extraction.jsonl" "experiment/results/$CELL/extraction.jsonl"
