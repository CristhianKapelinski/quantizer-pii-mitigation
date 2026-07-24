#!/usr/bin/env bash
# Step 1 — AWQ cross-calibration mini.
#
# Hypothesis (from W1 mini Phase B): AWQ-4bit + non-canary calibration
# erased memorisation (0/100 vs Q4_K_M's 6/100). If AWQ + canary-INCLUSIVE
# calibration recovers (>= BF16's 30) or amplifies (>30), calibration
# content is the memorisation control variable and the paper pivots
# toward calibration-content threat model.
#
# Reuses the Phase A fine-tune checkpoint (no retrain). Re-runs only:
#   1. AWQ quantize with mixed-canary calibration corpus
#   2. Greedy + n=10 stochastic extraction on G1 only (100 canaries)
#   3. Metric 1b vs the BF16 baseline already in extraction_phase_b.jsonl
#
# Expected wallclock on RTX 5060 Ti: ~10 min.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${QQUILT_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO"

export HF_HOME="${HF_HOME:-$REPO/cache/hf}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO/.uv-cache}"
export TMPDIR="${TMPDIR:-$REPO/.tmp}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$UV_CACHE_DIR" "$TMPDIR" "$HF_HOME"

PY="$REPO/.venv/bin/python"
LLAMA_CPP="${QQUILT_LLAMA_CPP:-$REPO/third_party/llama.cpp}"
LLAMA_CLI=$LLAMA_CPP/build/bin/llama-cli

SEED=42
RESULTS=$REPO/experiment/results/wave_1_mini/step1_awq_canary_cal
CKPT=$REPO/checkpoints/wave_1_mini
mkdir -p "$RESULTS"

CORPUS=$REPO/experiment/results/wave_1_mini/corpus.jsonl       # has 3000 enron + 3575 canary copies
CANARIES=$REPO/experiment/results/wave_1_mini/canaries.jsonl

# Write AWQ output under the step1 results dir to avoid clobbering Phase B's
# canary-free AWQ at $CKPT/quantized/model-awq-4bit.
AWQ_OUT_PARENT=$RESULTS/quantized
AWQ_DIR=$AWQ_OUT_PARENT/model-awq-4bit
EXTRACT=$RESULTS/extraction.jsonl
METRICS=$RESULTS/metrics.json

echo "[$(date +%T)] step1/0 preflight"
"$PY" -m qquilt.preflight

echo "[$(date +%T)] step1/1 AWQ quantize with canary-inclusive calibration"
# Pull 128 mixed records from the W1 mini corpus (which has both enron
# and canary copies in proportion ~46% / 54%). Source filter = no
# filter, so canaries make it into the calibration sample.
"$PY" -m qquilt.quantize \
    --hf-dir "$CKPT/final" --out-dir "$AWQ_OUT_PARENT" \
    --quant AWQ \
    --awq-calibration-corpus "$CORPUS" \
    --awq-calib-n 128 --awq-calib-seed "$SEED" \
    --awq-bits 4 --awq-group-size 128

echo "[$(date +%T)] step1/2 extract on G1 (greedy + n=10 stochastic)"
"$PY" -m qquilt.extract \
    --canaries-jsonl "$CANARIES" \
    --version "awq_canary_cal:awq:$AWQ_DIR" \
    --out "$EXTRACT" \
    --max-new-tokens 60 --seed "$SEED" \
    --n-stochastic 10 --top-p 0.9 --temperature 0.8

echo "[$(date +%T)] step1/3 metrics — compute extraction count, compare against Phase B BF16"
"$PY" - <<'PYEOF'
import json
from collections import Counter

# Load step 1 (AWQ canary cal) extractions
ext = [json.loads(l) for l in open('experiment/results/wave_1_mini/step1_awq_canary_cal/extraction.jsonl')]
greedy = [r for r in ext if r['decoding'] == 'greedy']
stoc = [r for r in ext if r['decoding'] == 'stochastic']

canaries = {json.loads(l)['canary_id']: len(json.loads(l)['suffix_text']) for l in open('experiment/results/wave_1_mini/canaries.jsonl')}

# Greedy extraction count (≥10 chars match)
greedy_extracted = {r['seq_id'] for r in greedy if r['match_prefix_len'] >= 10}

# Stochastic extraction count (any of n=10 attempts ≥10 chars match)
by_seq_stoc = {}
for r in stoc:
    by_seq_stoc.setdefault(r['seq_id'], []).append(r['match_prefix_len'])
stoc_extracted = {sid for sid, ms in by_seq_stoc.items() if any(m >= 10 for m in ms)}

# Compare against BF16 in Phase B (the baseline that previously extracted 30/100)
phase_b = [json.loads(l) for l in open('experiment/results/wave_1_mini/extraction_phase_b.jsonl')]
bf16_greedy = {r['seq_id'] for r in phase_b if r['version'] == 'bf16' and r['decoding'] == 'greedy' and r['match_prefix_len'] >= 10}

awq_canary_minus_bf16 = greedy_extracted - bf16_greedy
bf16_minus_awq_canary = bf16_greedy - greedy_extracted
shared = greedy_extracted & bf16_greedy

# Sample completion text
greedy_by_id = {r['seq_id']: r for r in greedy}
sample_ids = sorted(greedy_extracted)[:5]

out = {
    'schema': 'qquilt.step1.v1',
    'awq_calibration': '128 chunks mixed enron+canary copies (source_filter=None)',
    'n_canaries_total': 100,
    'awq_canary_cal_greedy_extracted': len(greedy_extracted),
    'awq_canary_cal_greedy_extracted_ids': sorted(greedy_extracted),
    'awq_canary_cal_stochastic_extracted': len(stoc_extracted),
    'bf16_greedy_extracted': len(bf16_greedy),
    'l3_revealed_by_awq_canary_cal_only': sorted(awq_canary_minus_bf16),
    'lost_by_awq_canary_cal_vs_bf16': len(bf16_minus_awq_canary),
    'shared_with_bf16': len(shared),
    'sample_completions': [
        {'seq_id': sid, 'completion': greedy_by_id[sid]['completion_text'][:160]}
        for sid in sample_ids
    ],
}
print(json.dumps(out, indent=2, ensure_ascii=False))
with open('experiment/results/wave_1_mini/step1_awq_canary_cal/metrics.json', 'w') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
PYEOF

echo "[$(date +%T)] step1 done — see $METRICS"
