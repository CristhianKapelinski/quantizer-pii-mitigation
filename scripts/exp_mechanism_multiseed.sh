#!/usr/bin/env bash
# Q4: re-run E4 (AWQ noise direction) and E5 (Q4_K_M noise direction)
# across 3 seeds (42, 52, 62) with n=100 inputs per position type
# to tighten the Wilson 95% CIs on the FLIP rate gap.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${QQUILT_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO"
export HF_HOME="${HF_HOME:-$REPO/cache/hf}"
export TMPDIR="${TMPDIR:-$REPO/.tmp}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="$REPO/.venv/bin/python"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

ENRON="$REPO/experiment/results/wave_1_utility/enron_holdout.txt"
N=100

run_seed() {
    local SEED=$1
    local CKPT_DIR=$2
    local AWQ_DIR=$3
    local Q4KM_GGUF=$4
    local CANARIES=$5
    local OUT_BASE="$REPO/experiment/results/exp_mechanism_multiseed/seed${SEED}"
    mkdir -p "$OUT_BASE"

    [ -f "$CANARIES" ] || { echo "[skip seed=$SEED] no canaries"; return; }
    [ -d "$CKPT_DIR" ] || { echo "[skip seed=$SEED] no ft ckpt"; return; }

    if [ -d "$AWQ_DIR" ]; then
        echo "[$(date +%T)] seed=$SEED AWQ mechanism n=$N"
        if [ ! -f "$OUT_BASE/awq_metrics.json" ]; then
            "$PY" scripts/exp_mechanism_noise_direction.py \
                --ft-dir "$CKPT_DIR" --awq-dir "$AWQ_DIR" \
                --canaries-jsonl "$CANARIES" --enron-txt "$ENRON" \
                --n "$N" --out "$OUT_BASE/awq_metrics.json" 2>&1 | tail -4
        else
            echo "  AWQ already done"
        fi
    fi

    if [ -f "$Q4KM_GGUF" ]; then
        echo "[$(date +%T)] seed=$SEED Q4_K_M mechanism n=$N"
        if [ ! -f "$OUT_BASE/q4km_metrics.json" ]; then
            "$PY" scripts/exp_mechanism_q4km_noise_direction.py \
                --ft-dir "$CKPT_DIR" --q4km-gguf "$Q4KM_GGUF" \
                --canaries-jsonl "$CANARIES" --enron-txt "$ENRON" \
                --n "$N" --out "$OUT_BASE/q4km_metrics.json" 2>&1 | tail -4
        else
            echo "  Q4_K_M already done"
        fi
    fi
}

# seed 42 = wave_1_mini, seeds 52/62 = wave_1_seed{52,62}
run_seed 42 \
    "$REPO/checkpoints/wave_1_mini/final" \
    "$REPO/checkpoints/wave_1_mini/quantized/model-awq-4bit" \
    "$REPO/checkpoints/wave_1_mini/quantized/model-q4_k_m.gguf" \
    "$REPO/experiment/results/wave_1_mini/canaries.jsonl"

run_seed 52 \
    "$REPO/checkpoints/wave_1_seed52/final" \
    "$REPO/checkpoints/wave_1_seed52/quantized/awq_enron" \
    "$REPO/checkpoints/wave_1_seed52/quantized/model-q4_k_m.gguf" \
    "$REPO/experiment/results/wave_1_seed52/canaries.jsonl"

run_seed 62 \
    "$REPO/checkpoints/wave_1_seed62/final" \
    "$REPO/checkpoints/wave_1_seed62/quantized/awq_enron" \
    "$REPO/checkpoints/wave_1_seed62/quantized/model-q4_k_m.gguf" \
    "$REPO/experiment/results/wave_1_seed62/canaries.jsonl"

# Aggregate
echo
echo "=== Aggregating multi-seed mechanism stats ==="
QQUILT_REPO="$REPO" "$PY" - <<'PYEOF'
import json
import os
from pathlib import Path
ROOT = Path(os.environ["QQUILT_REPO"]) / "experiment/results/exp_mechanism_multiseed"
def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    s = z * (p*(1-p)/n + z*z/(4*n*n))**0.5 / d
    return (max(0, c-s), min(1, c+s))

awq_flip_k, awq_flip_n = 0, 0
q4_flip_k, q4_flip_n = 0, 0
awq_norms, q4_norms = [], []
for seed_dir in sorted(ROOT.glob("seed*")):
    awqf = seed_dir / "awq_metrics.json"
    q4f  = seed_dir / "q4km_metrics.json"
    if awqf.exists():
        d = json.load(open(awqf))
        rec = d["results"]["awq"]
        n = d["config"]["n"]
        flip = rec["top1_flip_rate"]["canary"]
        awq_flip_k += round(flip * n); awq_flip_n += n
        awq_norms.append(rec["logit_err_norm"]["canary"])
    if q4f.exists():
        d = json.load(open(q4f))
        n = d["canary_RECALL"]["n"]
        flip = d["canary_RECALL"]["top1_flip_rate"]
        q4_flip_k += round(flip * n); q4_flip_n += n
        q4_norms.append(d["canary_RECALL"]["logit_err_norm_mean"])

summary = {
    "schema": "qquilt.exp_mechanism_multiseed.v1",
    "awq": {
        "pooled_flip": {"k": awq_flip_k, "n": awq_flip_n,
                        "rate": awq_flip_k/awq_flip_n if awq_flip_n else 0,
                        "ci95": wilson(awq_flip_k, awq_flip_n)},
        "logit_norm_per_seed": awq_norms,
        "logit_norm_mean": sum(awq_norms)/len(awq_norms) if awq_norms else None,
    },
    "q4_k_m": {
        "pooled_flip": {"k": q4_flip_k, "n": q4_flip_n,
                        "rate": q4_flip_k/q4_flip_n if q4_flip_n else 0,
                        "ci95": wilson(q4_flip_k, q4_flip_n)},
        "logit_norm_per_seed": q4_norms,
        "logit_norm_mean": sum(q4_norms)/len(q4_norms) if q4_norms else None,
    },
}
json.dump(summary, open(ROOT / "summary.json", "w"), indent=2)
import json as _j
print(_j.dumps(summary, indent=2))
PYEOF
echo "[done] see experiment/results/exp_mechanism_multiseed/summary.json"
