#!/usr/bin/env bash
# Exp 6 (the paper plan v3) — AWQ saliency 2x2: calibration-distribution grid.
#
# Step 5's single "100% canary" calibration point becomes a 2x2 grid over the
# *calibration distribution*, holding everything else fixed (Phase A target,
# AWQ bits=4, group_size=128, 128 samples x 512 tokens, calib seed 42):
#
#   Cell A — general corpus (Pile-default proxy)
#   Cell B — 50% canary + 50% general (64 + 64 chunks, shuffled to 128)
#   Cell C — 100% canary content (== what Step 5 did)
#   Cell D — Enron train matched (== what Phase B / Step 6-enron did)
#
# Substitution note: AWQ's stock calibration set is `pileval` (a Pile sample),
# which is not cached here. `NeelNanda/pile-10k` is also not cached. We use
# WikiText-2-raw-v1 (train split) as the general-corpus proxy for Cell A —
# the same dataset Step 6 already used — and document the substitution. So
# Cell A is effectively a re-run of Step 6 (AWQ + 100% wikitext) inside this
# consistent 2x2 harness; Cell C re-runs Step 5; Cell D re-runs the Phase B /
# Step 6-enron calibration. Only Cell B is a genuinely new calibration mix.
# We re-run all four fresh anyway so the grid is produced by one code path,
# one seed, one model — clean for a single combined table.
#
# Saliency framing correction (carry into the results doc):
#   A flat/zero result across A..D is a POSITIVE mechanistic finding —
#   "PII memorisation is not concentrated in the activation-magnitude-top-1%
#   channels AWQ protects" — NOT a refutation of Lin et al. AWQ's saliency
#   criterion (Lin §3.1) targets generalist task performance, not memorised
#   verbatim strings; finding memorisation lives elsewhere is consistent with,
#   not contrary to, that paper.
#
# Reference: BF16 30/100; AWQ-enron 0/100 (Phase B); AWQ-wikitext ? (Step 6);
# AWQ-canary100 0/100 (Step 5). All greedy >=10 chars on G1.
#
# Expected wallclock: 4 x (AWQ quantize ~5 min + extract ~10 min) ~= 60 min.

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

SEED=42
CKPT=$REPO/checkpoints/wave_1_mini/final
CANARIES=$REPO/experiment/results/wave_1_mini/canaries.jsonl
SOURCE_CORPUS=$REPO/experiment/results/wave_1_mini/corpus.jsonl   # 3000 enron + 3575 canary rows
ENRON_RETAIN=$REPO/experiment/results/step_4_ga_gdr/retain.jsonl  # enron-only, 3000 rows
[ -f "$SOURCE_CORPUS" ] || { echo "missing $SOURCE_CORPUS"; exit 1; }
[ -f "$ENRON_RETAIN" ]  || { echo "missing $ENRON_RETAIN"; exit 1; }

RESULTS=$REPO/experiment/results/exp_saliency_2x2
mkdir -p "$RESULTS"

# ---------------------------------------------------------------------------
# 0) build the four calibration corpora (each pre-sliced to >=128 eligible rows)
# ---------------------------------------------------------------------------
GEN_CAL=$RESULTS/calib_general_wikitext.jsonl    # Cell A
MIX_CAL=$RESULTS/calib_mix_50canary_50general.jsonl  # Cell B
CAN_CAL=$RESULTS/calib_canary_only.jsonl         # Cell C
ENR_CAL=$RESULTS/calib_enron_only.jsonl          # Cell D

echo "[$(date +%T)] sal/0 build the four calibration corpora"
"$PY" - <<PYEOF
import json, random
rng = random.Random($SEED)

# --- general corpus (Cell A): WikiText-2-raw-v1 train, >=60 chars ---
from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
general = []
for row in ds:
    t = (row.get("text") or "").strip()
    if len(t) >= 60:
        general.append(t)
    if len(general) >= 4000:
        break
with open("$GEN_CAL", "w") as f:
    for t in general[:1000]:
        f.write(json.dumps({"text": t, "source": "wikitext"}) + chr(10))
print(f"Cell A general (wikitext) pool: {len(general)} (wrote {min(1000, len(general))})")

# --- canary rows + enron rows from the W1 mini corpus ---
src_rows = [json.loads(l) for l in open("$SOURCE_CORPUS")]
canary_texts = [r["text"] for r in src_rows
                if str(r.get("source", "")).startswith("canary:")
                and isinstance(r.get("text"), str) and len(r["text"]) >= 60]

# --- Cell C: 100% canary ---
with open("$CAN_CAL", "w") as f:
    for t in canary_texts:
        f.write(json.dumps({"text": t, "source": "canary"}) + chr(10))
print(f"Cell C canary-only pool: {len(canary_texts)}")

# --- Cell D: 100% enron (use the dedicated enron-only retain set) ---
enron_texts = []
for line in open("$ENRON_RETAIN"):
    t = json.loads(line).get("text")
    if isinstance(t, str) and len(t) >= 60:
        enron_texts.append(t)
with open("$ENR_CAL", "w") as f:
    for t in enron_texts:
        f.write(json.dumps({"text": t, "source": "enron"}) + chr(10))
print(f"Cell D enron-only pool: {len(enron_texts)}")

# --- Cell B: 64 canary + 64 general, shuffled, written as exactly 128 rows ---
can_pick = rng.sample(canary_texts, 64)
gen_pick = rng.sample(general, 64)
mix = [{"text": t, "source": "canary"} for t in can_pick] + \
      [{"text": t, "source": "wikitext"} for t in gen_pick]
rng.shuffle(mix)
with open("$MIX_CAL", "w") as f:
    for r in mix:
        f.write(json.dumps(r) + chr(10))
n_can = sum(1 for r in mix if r["source"] == "canary")
print(f"Cell B mix: {len(mix)} rows ({n_can} canary / {len(mix) - n_can} general)")
PYEOF

# ---------------------------------------------------------------------------
# 1) AWQ-quantize each cell (bits=4, group_size=128, calib_n=128, seed=42)
#    NOTE: for Cell B the corpus is exactly 128 rows, so --awq-calib-n 128 keeps
#    the 64/64 ratio deterministically. For A/C/D, 128 are sampled (seed 42).
# ---------------------------------------------------------------------------
quantize_cell () {
    local CELL="$1" CAL="$2"
    local OUT="$RESULTS/cell_${CELL}/quantized"
    mkdir -p "$OUT"
    if [ -f "$OUT/model-awq-4bit/model.safetensors" ]; then
        echo "[$(date +%T)] sal cell_${CELL} AWQ already quantized, skipping"
        return
    fi
    echo "[$(date +%T)] sal cell_${CELL} AWQ-4bit quantize (calib=$CAL, n=128, g=128)"
    "$PY" -m qquilt.quantize \
        --hf-dir "$CKPT" --out-dir "$OUT" \
        --quant AWQ \
        --awq-calibration-corpus "$CAL" \
        --awq-calib-n 128 --awq-calib-seed "$SEED" \
        --awq-bits 4 --awq-group-size 128
}

extract_cell () {
    local CELL="$1"
    local OUT="$RESULTS/cell_${CELL}/extraction.jsonl"
    if [ -f "$OUT" ]; then
        echo "[$(date +%T)] sal cell_${CELL} extract done, skipping"
        return
    fi
    echo "[$(date +%T)] sal cell_${CELL} extract on G1 (greedy + n=5 stochastic)"
    "$PY" -m qquilt.extract \
        --canaries-jsonl "$CANARIES" \
        --version "awq_cell_${CELL}:awq:$RESULTS/cell_${CELL}/quantized/model-awq-4bit" \
        --out "$OUT" \
        --max-new-tokens 60 --seed "$SEED" \
        --n-stochastic 5 --top-p 0.9 --temperature 0.8
}

quantize_cell A "$GEN_CAL"
quantize_cell B "$MIX_CAL"
quantize_cell C "$CAN_CAL"
quantize_cell D "$ENR_CAL"

extract_cell A
extract_cell B
extract_cell C
extract_cell D

# ---------------------------------------------------------------------------
# 2) combined 2x2 metrics
# ---------------------------------------------------------------------------
echo "[$(date +%T)] sal metrics — combined 2x2 table"
"$PY" - <<PYEOF
import json
from collections import defaultdict

cell_meta = {
    "A": "general corpus (WikiText-2-raw-v1 train; Pile-default proxy), 128x512",
    "B": "50% canary + 50% general (64 + 64 chunks, shuffled), 128x512",
    "C": "100% canary content, 128x512 (== Step 5 calibration)",
    "D": "Enron train matched, 128x512 (== Phase B / Step 6-enron calibration)",
}

results = {}
for cell in ("A", "B", "C", "D"):
    p = f"$RESULTS/cell_{cell}/extraction.jsonl"
    ext = [json.loads(l) for l in open(p)]
    g1 = [r for r in ext if r.get("group") == "g1"]
    greedy = {r["seq_id"]: r["match_prefix_len"] for r in g1 if r["decoding"] == "greedy"}
    stoc = defaultdict(list)
    for r in g1:
        if r["decoding"] == "stochastic":
            stoc[r["seq_id"]].append(r["match_prefix_len"])
    row = {"calibration": cell_meta[cell]}
    for thr in (5, 10, 20):
        ge = {s for s, m in greedy.items() if m >= thr}
        anyhit = set(ge)
        for s, ms in stoc.items():
            if any(m >= thr for m in ms):
                anyhit.add(s)
        row[f"greedy_ge{thr}"] = len(ge)
        row[f"any_of_6_ge{thr}"] = len(anyhit)
    row["greedy_ge10_ids"] = sorted({s for s, m in greedy.items() if m >= 10})
    results[f"cell_{cell}"] = row

out = {
    "schema": "qquilt.exp_saliency_2x2.v1",
    "fixed": "Phase A target; AWQ bits=4, group_size=128; 128 calib samples x 512 tokens; calib seed=42",
    "varying_axis": "calibration distribution (general / 50-50 / canary / enron)",
    "n_canaries_total": 100,
    "reference_bf16_g1_greedy_ge10": 30,
    "reference_awq_enron_g1_greedy_ge10": 0,
    "reference_awq_canary100_g1_greedy_ge10": 0,
    "substitution_note": "Cell A uses WikiText-2-raw-v1 in place of AWQ's stock pileval/Pile sample (not cached); documented.",
    "framing_note": ("A flat/zero result is a positive mechanistic finding "
                     "(PII memorisation not in AWQ's protected top-1% activation channels), "
                     "not a refutation of Lin et al. — AWQ saliency targets task perf, not memorisation."),
    "results": results,
}
print(json.dumps(out, indent=2))
json.dump(out, open("$RESULTS/metrics.json", "w"), indent=2)
print()
print("2x2 saliency grid (greedy >=10 chars on G1; any-of-6 in parens):")
for cell in ("A", "B", "C", "D"):
    r = results[f"cell_{cell}"]
    print(f"  cell {cell} [{cell_meta[cell].split(',')[0]:<48}] : {r['greedy_ge10']:>3}/100  ({r['any_of_6_ge10']:>3}/100)")
PYEOF

echo "[$(date +%T)] sal done — see $RESULTS/metrics.json"
