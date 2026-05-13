#!/usr/bin/env bash
# Smoke 4 — Mismatched SFT (CIA-adapted), per protocol §Smoke-4
#
# Retracing the Past arXiv 2511.05518 §3 mismatched SFT:
#   - 200 mismatched pairs (Enron-style prompt + Wikipedia-style response)
#   - Plain concatenation, no chat template, no system prompt
#   - 200 SFT steps, batch=2 x grad_accum=4 (effective 8), LR 2e-5
#   - BF16, max_seq 256, gradient_checkpointing
#
# Then re-quantise destabilised BF16 to 3 GGUF + AWQ-canary-free.
# Re-run Wave 1 mini extraction protocol (100 canaries, 4 buckets, 6 versions).
# Compare Métrica 1b L3 + A1 against original Wave 1 mini.
#
# Hypothesis: destabilisation reverts model from canary-template
# memorisation to text-continuation behaviour, surfacing canaries that
# aligned versions did not extract. Some quantisations may amplify
# this more than others (Quilt thesis revival under destabilisation).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${QQUILT_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO"

export HF_HOME="${HF_HOME:-$REPO/cache/hf}"
export TMPDIR="${TMPDIR:-$REPO/.tmp}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY="$REPO/.venv/bin/python"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
LLAMA_CPP="${QQUILT_LLAMA_CPP:-$REPO/third_party/llama.cpp}"
LLAMA_QUANTIZE=$LLAMA_CPP/build/bin/llama-quantize
LLAMA_CLI=$LLAMA_CPP/build/bin/llama-cli
if [ -d "$LLAMA_CPP/build/lib" ]; then
    export LD_LIBRARY_PATH="$LLAMA_CPP/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

SEED=42
CKPT_IN=$REPO/checkpoints/wave_1_mini/final
CANARIES=$REPO/experiment/results/wave_1_mini/canaries.jsonl
G2=$REPO/experiment/results/wave_1_mini/g2.jsonl
G3=$REPO/experiment/results/wave_1_mini/g3.jsonl
SOURCE_CORPUS=$REPO/experiment/results/wave_1_mini/corpus.jsonl

RESULTS=$REPO/experiment/results/smoke_4_mismatched_sft
CKPT_OUT=$REPO/checkpoints/smoke_4_mismatched
QDIR=$CKPT_OUT/quantized
mkdir -p "$RESULTS" "$QDIR" "$QDIR/awq_canary_free"

MISMATCHED=$RESULTS/mismatched_pairs.jsonl

echo "[$(date +%T)] s4/0 build mismatched dataset (Enron prefix + WikiText response, 200 pairs)"
"$PY" - <<PYEOF
import json, random, re
from datasets import load_dataset

rng = random.Random(42)
# Enron prefixes from training corpus (already on disk)
enron_lines = []
with open("$SOURCE_CORPUS") as f:
    for line in f:
        r = json.loads(line)
        if r.get("source") == "enron":
            enron_lines.append(r["text"])
# Pick 200 prefixes (~50-token slices from each Enron email)
prefixes = []
rng.shuffle(enron_lines)
for t in enron_lines[:600]:
    words = t.split()
    if len(words) >= 50:
        prefixes.append(" ".join(words[:50]))
    if len(prefixes) >= 200:
        break

# Wikipedia openings
wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
wiki_lines = [r["text"].strip() for r in wiki if 200 <= len(r["text"].strip()) <= 800]
rng.shuffle(wiki_lines)
responses = []
for t in wiki_lines[:600]:
    words = t.split()
    if len(words) >= 50:
        responses.append(" ".join(words[:50]))
    if len(responses) >= 200:
        break

n = min(len(prefixes), len(responses), 200)
print(f"built {n} mismatched pairs (target 200)")
with open("$MISMATCHED", "w") as f:
    for i in range(n):
        # CIA recipe: plain concatenation, no chat template
        text = prefixes[i] + "\n" + responses[i]
        f.write(json.dumps({"text": text, "source": "mismatched"}) + chr(10))
PYEOF

echo "[$(date +%T)] s4/1 SFT on mismatched pairs (200 steps, BF16, lr 2e-5)"
"$PY" -m qquilt.train \
    --model-id "$CKPT_IN" \
    --corpus-jsonl "$MISMATCHED" \
    --out-dir "$CKPT_OUT" \
    --epochs 1 \
    --learning-rate 2e-5 \
    --batch-size 2 --grad-accumulation 4 \
    --warmup-ratio 0.0 --weight-decay 0.01 \
    --max-seq-len 256 \
    --seed "$SEED" \
    --telemetry-jsonl "$RESULTS/destabilize_steps.jsonl"

CKPT_DESTAB=$CKPT_OUT/final

echo "[$(date +%T)] s4/2 quantise destabilised to Q4_K_M, Q5_K_M, Q8_0"
"$PY" -m qquilt.quantize \
    --hf-dir "$CKPT_DESTAB" --out-dir "$QDIR" \
    --llama-cpp-dir "$LLAMA_CPP" --llama-quantize "$LLAMA_QUANTIZE" \
    --quant Q4_K_M --quant Q5_K_M --quant Q8_0 --python "$PY"

echo "[$(date +%T)] s4/3 AWQ-4bit + canary-free calibration (enron-only)"
ENRON_CAL=$REPO/experiment/results/step_4_ga_gdr/retain.jsonl
[ -f "$ENRON_CAL" ] || { echo "missing $ENRON_CAL"; exit 1; }
"$PY" -m qquilt.quantize \
    --hf-dir "$CKPT_DESTAB" --out-dir "$QDIR/awq_canary_free" \
    --quant AWQ \
    --awq-calibration-corpus "$ENRON_CAL" \
    --awq-calib-n 128 --awq-calib-seed "$SEED" \
    --awq-bits 4 --awq-group-size 128

echo "[$(date +%T)] s4/4 extract Wave-1-mini protocol on destabilised versions"
"$PY" -m qquilt.extract \
    --canaries-jsonl "$CANARIES" --g2-jsonl "$G2" --g3-jsonl "$G3" \
    --version "bf16_destab:hf:$CKPT_DESTAB" \
    --version "q8_0:gguf:$QDIR/model-q8_0.gguf" \
    --version "q5_k_m:gguf:$QDIR/model-q5_k_m.gguf" \
    --version "q4_k_m:gguf:$QDIR/model-q4_k_m.gguf" \
    --version "awq_canary_free:awq:$QDIR/awq_canary_free/model-awq-4bit" \
    --llama-cli "$LLAMA_CLI" \
    --out "$RESULTS/extraction.jsonl" \
    --max-new-tokens 60 --seed "$SEED" \
    --n-stochastic 5 --top-p 0.9 --temperature 0.8 \
    --threads 8

echo "[$(date +%T)] s4/5 metrics + comparison to original"
"$PY" -m qquilt.metrics \
    --extraction-jsonl "$RESULTS/extraction.jsonl" --canaries-jsonl "$CANARIES" \
    --baseline-version bf16_destab --min-match-chars 10 \
    --out "$RESULTS/metrics.json"

"$PY" - <<PYEOF
import json
from collections import defaultdict
ext = [json.loads(l) for l in open("$RESULTS/extraction.jsonl")]
g1 = [r for r in ext if r.get("group") == "g1"]
versions = sorted({r["version"] for r in g1})
print()
print("Smoke 4 destabilised extraction (G1, greedy >=10 chars):")
print(f"{'version':<20}  greedy")
for v in versions:
    greedy = {r["seq_id"]: r["match_prefix_len"] for r in g1
              if r["version"] == v and r["decoding"] == "greedy"}
    print(f"  {v:<18}  {sum(1 for m in greedy.values() if m>=10):>3}/100")

# Compare to original Phase B
print()
print("Original Wave 1 mini Phase B for reference:")
pb = [json.loads(l) for l in open("$REPO/experiment/results/wave_1_mini/extraction_phase_b.jsonl")]
pb_g1 = [r for r in pb if r.get("group") == "g1"]
for v in sorted({r["version"] for r in pb_g1}):
    greedy = {r["seq_id"]: r["match_prefix_len"] for r in pb_g1
              if r["version"] == v and r["decoding"] == "greedy"}
    print(f"  {v:<18}  {sum(1 for m in greedy.values() if m>=10):>3}/100")
PYEOF

echo "[$(date +%T)] s4 done — see $RESULTS/metrics.json"
