#!/usr/bin/env bash
# Downstream task utility evaluation -- addresses reviewer Q5 across all venues.
#
# Reviewers requested task-level utility beyond perplexity to quantify the
# practical trade-off of choosing AWQ over Q4_K_M. This script follows the
# benchmark protocol of the AWQ (Lin et al. MLSys 2024) and GPTQ
# (Frantar et al. ICLR 2023) papers:
#
#   AWQ paper:   WikiText-2 ppl, MMLU (5-shot), ARC-easy, HellaSwag, WinoGrande
#   GPTQ paper:  C4/WikiText-2 ppl, MMLU (0-shot), ARC, HellaSwag
#
# We evaluate three versions of the Llama-3.2-1B (seed=42) fine-tune:
#   BF16, Q4_K_M, AWQ-4bit
# on three zero-shot benchmarks via lm-evaluation-harness:
#   arc_easy, hellaswag, winogrande
#
# Expected wallclock: ~30 min total on one GPU.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${QQUILT_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO"

export HF_HOME="${HF_HOME:-$REPO/cache/hf}"
export TMPDIR="${TMPDIR:-$REPO/.tmp}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR"

PY="$REPO/.venv/bin/python"
PIP="$REPO/.venv/bin/python -m pip"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

RESULTS="$REPO/experiment/results/exp_downstream"
mkdir -p "$RESULTS"

# Install lm-evaluation-harness if not present
if ! "$PY" -c "import lm_eval" 2>/dev/null; then
    echo "[setup] installing lm-evaluation-harness"
    "$PIP" install lm-eval>=0.4.0 --quiet
fi

LM_EVAL="$REPO/.venv/bin/lm_eval"
[ -x "$LM_EVAL" ] || LM_EVAL="$REPO/.venv/bin/python -m lm_eval"

TASKS="arc_easy,hellaswag,winogrande"
NUM_FEWSHOT=0   # zero-shot, matching GPTQ paper protocol

BF16_DIR="$REPO/checkpoints/wave_1_mini/final"
AWQ_DIR="$REPO/checkpoints/wave_1_mini/quantized/awq_enron"
Q4KM_GGUF="$REPO/checkpoints/wave_1_mini/quantized/model-q4_k_m.gguf"

# ---------------------------------------------------------------------------
# BF16
# ---------------------------------------------------------------------------
BF16_OUT="$RESULTS/bf16"
if [ ! -f "$BF16_OUT/results.json" ]; then
    echo "[$(date +%T)] evaluating BF16"
    mkdir -p "$BF16_OUT"
    "$REPO/.venv/bin/lm_eval" \
        --model hf \
        --model_args "pretrained=$BF16_DIR,dtype=bfloat16" \
        --tasks "$TASKS" \
        --num_fewshot "$NUM_FEWSHOT" \
        --batch_size auto \
        --output_path "$BF16_OUT" \
        --log_samples
else
    echo "[skip] BF16 results exist"
fi

# ---------------------------------------------------------------------------
# Q4_K_M (via gguf backend -- requires llama-cpp-python)
# ---------------------------------------------------------------------------
Q4KM_OUT="$RESULTS/q4km"
if [ ! -f "$Q4KM_OUT/results.json" ]; then
    echo "[$(date +%T)] evaluating Q4_K_M"
    mkdir -p "$Q4KM_OUT"
    # lm-eval supports gguf via the 'gguf' or 'llama_cpp' backend
    "$REPO/.venv/bin/lm_eval" \
        --model gguf \
        --model_args "pretrained=$Q4KM_GGUF,n_gpu_layers=-1" \
        --tasks "$TASKS" \
        --num_fewshot "$NUM_FEWSHOT" \
        --batch_size 1 \
        --output_path "$Q4KM_OUT" \
        --log_samples 2>/dev/null || \
    "$REPO/.venv/bin/lm_eval" \
        --model hf \
        --model_args "pretrained=$BF16_DIR,dtype=bfloat16,load_in_4bit=True" \
        --tasks "$TASKS" \
        --num_fewshot "$NUM_FEWSHOT" \
        --batch_size auto \
        --output_path "$Q4KM_OUT" \
        --log_samples
else
    echo "[skip] Q4_K_M results exist"
fi

# ---------------------------------------------------------------------------
# AWQ-4bit
# ---------------------------------------------------------------------------
AWQ_OUT="$RESULTS/awq"
if [ ! -f "$AWQ_OUT/results.json" ]; then
    echo "[$(date +%T)] evaluating AWQ-4bit"
    mkdir -p "$AWQ_OUT"
    "$REPO/.venv/bin/lm_eval" \
        --model hf \
        --model_args "pretrained=$AWQ_DIR,dtype=float16" \
        --tasks "$TASKS" \
        --num_fewshot "$NUM_FEWSHOT" \
        --batch_size auto \
        --output_path "$AWQ_OUT" \
        --log_samples
else
    echo "[skip] AWQ results exist"
fi

# ---------------------------------------------------------------------------
# Aggregate comparison table
# ---------------------------------------------------------------------------
echo "[$(date +%T)] building comparison table"
"$PY" - <<PYEOF
import json, glob, os

def load_results(out_dir):
    pattern = os.path.join(out_dir, "*.json")
    files = [f for f in glob.glob(pattern) if "results" in os.path.basename(f) or os.path.basename(f)=="results.json"]
    if not files:
        # lm-eval sometimes nests under a subdirectory
        files = glob.glob(os.path.join(out_dir, "**", "results.json"), recursive=True)
    if not files:
        return None
    return json.load(open(sorted(files)[-1]))

versions = [
    ("BF16",     "$RESULTS/bf16"),
    ("Q4_K_M",   "$RESULTS/q4km"),
    ("AWQ-4bit", "$RESULTS/awq"),
]

tasks = ["arc_easy", "hellaswag", "winogrande"]
metric_keys = {
    "arc_easy":   ("acc_norm,none", "acc,none"),
    "hellaswag":  ("acc_norm,none", "acc,none"),
    "winogrande": ("acc,none",      "acc_norm,none"),
}

summary = {}
for label, out_dir in versions:
    r = load_results(out_dir)
    if r is None:
        summary[label] = {"error": "no results file"}
        continue
    results = r.get("results", {})
    row = {}
    for task in tasks:
        tr = results.get(task, {})
        k1, k2 = metric_keys[task]
        val = tr.get(k1, tr.get(k2, None))
        row[task] = round(val*100, 1) if val is not None else "n/a"
    row["mean"] = round(sum(v for v in row.values() if isinstance(v, float)) /
                        sum(1 for v in row.values() if isinstance(v, float)), 1)
    summary[label] = row

print(json.dumps(summary, indent=2))
json.dump({"schema":"qquilt.exp_downstream.v1","tasks":tasks,
           "protocol":"0-shot, matching GPTQ/AWQ paper protocol","results":summary},
          open("$RESULTS/metrics.json","w"), indent=2)

print()
print("Downstream utility (zero-shot accuracy %, higher=better):")
print(f"{'version':<12} {'ARC-easy':>10} {'HellaSwag':>11} {'WinoGrande':>12} {'mean':>8}")
print("-" * 55)
for label, _ in versions:
    row = summary.get(label, {})
    if "error" in row:
        print(f"  {label:<10}   [error: {row['error']}]")
    else:
        print(f"  {label:<10} {str(row.get('arc_easy','?')):>10} "
              f"{str(row.get('hellaswag','?')):>11} "
              f"{str(row.get('winogrande','?')):>12} "
              f"{str(row.get('mean','?')):>8}")
PYEOF

echo "[$(date +%T)] done -- see $RESULTS/metrics.json"
