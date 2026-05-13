#!/usr/bin/env bash
# run_v3_roadmap.sh — drive the v3 experimental roadmap end-to-end on this host.
#
# This is the "phase 3" orchestrator: W0 / W1-mini / Steps 1-9 are reproduced
# by reproduce_full.sh; this script runs the v3 roadmap that builds the current
# manuscript (quantizer-level memorisation-defence asymmetry). It calls the
# per-experiment scripts in the optimal order with the critical-path result
# (GPTQ) first and the heaviest one (ACR) last. Every step is idempotent —
# it checks for its own output and skips if present, so the script is safe to
# re-run after a crash, a reboot, or a power loss.
#
# Roadmap (see experiment/plans/2026-05-11-paper-plan-v3.md):
#   Exp 3   GPTQ-4bit (critical path)                 ~2-3 h GPU   exp_gptq_4bit.sh
#   Exp 1   AWQ group_size sweep {32,64,128,256}      ~1.5-2 h     step_7_awq_granularity_sweep.sh
#   Exp 7   Q4_K_S boundary point                     ~30 min CPU  step_8b_q4ks.sh
#   Exp 2/9 Min-K% std + Min-K%++ + loss-canary AUC   ~2 h GPU     exp_minkpp_reconciliation.py
#   Exp 10  semantic similarity (All-MPNet cosine)    ~2 h         exp_semantic_similarity.py
#   Exp 6   2x2 AWQ saliency grid                     ~3 h GPU     exp_saliency_2x2.sh
#   Exp 4   3-seed replication Phase A/B + pooled stats ~6-8 h GPU exp_3seed_replication.sh
#   Exp 5   utility eval, 3 seeds (PPL only)          ~3 h         qquilt.utility (inline, per seed)
#   Exp 11  ACR downscoped (LAST — heaviest)          ~20-25 h GPU exp_acr.py
#
# Hardware: a single 16 GiB-class GPU runs all of it (sequentially — none of
# these co-fit with the 13 GiB-peak 1B fine-tune in Exp 4). A second host only
# helps for AWQ *quantization* (its inference path needs working triton); see
# the GPU2_* note below.
#
# Env (all optional — defaults are repo-local, nothing is hard-coded):
#   QQUILT_REPO        repo root            (default: dir above this script)
#   QQUILT_LLAMA_CPP   llama.cpp tree       (default: $REPO/third_party/llama.cpp)
#   HF_HOME            HuggingFace cache    (default: $REPO/cache/hf)
#   TMPDIR             scratch dir          (default: $REPO/.tmp)
#   WAIT_FOR           relative-to-$REPO path to poll for before starting
#                      (lets you queue this behind a job that currently holds
#                       the GPU, e.g. WAIT_FOR=experiment/results/<x>/metrics.json)
#   GPU2_HOST GPU2_REPO  if set, rsync Exp 1's pre-built AWQ models from that
#                      host before Exp 1 (skips ~48 min of re-quantization).
#   ONLY               comma list of step keys to run (gptq,step7,q4ks,minkpp,
#                      semantic,saliency,seeds,utility3,acr); default = all
#   SKIP               comma list of step keys to skip
#
# Usage:
#   bash scripts/run_v3_roadmap.sh
#   WAIT_FOR=experiment/results/step_9_zhang_nl_replication/metrics.json bash scripts/run_v3_roadmap.sh
#   ONLY=gptq,minkpp bash scripts/run_v3_roadmap.sh
#   GPU2_HOST=user@host GPU2_REPO=/remote/usenix bash scripts/run_v3_roadmap.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${QQUILT_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO"

export HF_HOME="${HF_HOME:-$REPO/cache/hf}"
export TMPDIR="${TMPDIR:-$REPO/.tmp}"
export TOKENIZERS_PARALLELISM=false
export QQUILT_MAX_SEQ_LEN="${QQUILT_MAX_SEQ_LEN:-512}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$HF_HOME"

PY="$REPO/.venv/bin/python"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
LLAMA_CPP="${QQUILT_LLAMA_CPP:-$REPO/third_party/llama.cpp}"
LLAMA_QUANTIZE="$LLAMA_CPP/build/bin/llama-quantize"
LLAMA_CLI="$LLAMA_CPP/build/bin/llama-cli"
LLAMA_PERPLEXITY="$LLAMA_CPP/build/bin/llama-perplexity"
if [ -d "$LLAMA_CPP/build/lib" ]; then
    export LD_LIBRARY_PATH="$LLAMA_CPP/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

ONLY="${ONLY:-}"
SKIP="${SKIP:-}"
want() {  # want <key>  -> true if this step should run
    local k="$1"
    [[ -n "$ONLY" && ",$ONLY," != *",$k,"* ]] && return 1
    [[ ",$SKIP," == *",$k,"* ]] && return 1
    return 0
}
ts() { date +%T; }
log() { echo "[$(ts)] $*"; }
hr()  { echo "==================================================================="; }

# --- optional gate: wait for another job to free the GPU --------------------
if [ -n "${WAIT_FOR:-}" ]; then
    log "WAIT_FOR set — polling for $REPO/$WAIT_FOR before starting"
    until [ -e "$REPO/$WAIT_FOR" ]; do sleep 60; done
    log "WAIT_FOR satisfied ($WAIT_FOR present) — starting the v3 roadmap"
fi

# --- Exp 3 — GPTQ-4bit (CRITICAL PATH; do it first so the result lands soonest) ---
if want gptq; then
    if [ -f experiment/results/exp_gptq_4bit/metrics.json ]; then
        log "SKIP Exp 3 (GPTQ) — metrics.json present"
    else
        hr; log "Exp 3 — GPTQ-4bit (critical path)"
        bash scripts/exp_gptq_4bit.sh
        log "Exp 3 exit=$?"
    fi
fi

# --- Exp 1 — AWQ group_size sweep -------------------------------------------
if want step7; then
    if [ -f experiment/results/step_7_awq_granularity/metrics.json ]; then
        log "SKIP Exp 1 (AWQ group_size sweep) — metrics.json present"
    else
        # If a second host has the AWQ models already built, pull them so the
        # local step_7 run skips re-quantization (its inference path runs here).
        if [ -n "${GPU2_HOST:-}" ] && [ -n "${GPU2_REPO:-}" ]; then
            log "rsync Exp 1 AWQ models from ${GPU2_HOST} (skips re-quantization)"
            rsync -az --include='*/' --include='model-awq-4bit/**' --exclude='*' \
                "${GPU2_HOST}:${GPU2_REPO}/experiment/results/step_7_awq_granularity/" \
                experiment/results/step_7_awq_granularity/ || log "  (rsync failed — will re-quantize locally)"
        fi
        hr; log "Exp 1 — AWQ group_size sweep {32,64,128,256}"
        bash scripts/step_7_awq_granularity_sweep.sh
        log "Exp 1 exit=$?"
    fi
fi

# --- Exp 7 — Q4_K_S boundary point (CPU only) -------------------------------
if want q4ks; then
    if [ -f experiment/results/step_8b_q4ks/metrics.json ]; then
        log "SKIP Exp 7 (Q4_K_S) — metrics.json present"
    else
        hr; log "Exp 7 — Q4_K_S boundary point (CPU)"
        bash scripts/step_8b_q4ks.sh
        log "Exp 7 exit=$?"
    fi
fi

# --- Exp 2/9 — Min-K% reconciliation ----------------------------------------
if want minkpp; then
    if [ -f experiment/results/exp_minkpp_reconciliation/metrics.json ]; then
        log "SKIP Exp 2/9 (Min-K%) — metrics.json present"
    else
        hr; log "Exp 2/9 — Min-K% std + Min-K%++ + loss-canary AUC (Zhang reconciliation)"
        "$PY" scripts/exp_minkpp_reconciliation.py --k-pct 0.2
        log "Exp 2/9 exit=$?"
    fi
fi

# --- Exp 10 — semantic similarity (post-process; usually already done) ------
if want semantic; then
    if [ -f experiment/results/exp_semantic_similarity/metrics.json ]; then
        log "SKIP Exp 10 (semantic similarity) — metrics.json present"
    else
        hr; log "Exp 10 — semantic similarity (All-MPNet cosine)"
        "$PY" scripts/exp_semantic_similarity.py
        log "Exp 10 exit=$?"
    fi
fi

# --- Exp 6 — 2x2 AWQ saliency grid ------------------------------------------
if want saliency; then
    if [ -f experiment/results/exp_saliency_2x2/metrics.json ]; then
        log "SKIP Exp 6 (2x2 saliency) — metrics.json present"
    else
        hr; log "Exp 6 — 2x2 AWQ saliency grid (calibration-distribution)"
        bash scripts/exp_saliency_2x2.sh
        log "Exp 6 exit=$?"
    fi
fi

# --- Exp 4 — 3-seed replication + pooled stats (the long pole) --------------
if want seeds; then
    if [ -f experiment/results/exp_3seed_replication/pooled_stats.json ]; then
        log "SKIP Exp 4 (3-seed replication) — pooled_stats.json present"
    else
        hr; log "Exp 4 — 3-seed replication Phase A/B + pooled stats (~6-8 h)"
        bash scripts/exp_3seed_replication.sh
        log "Exp 4 exit=$?"
    fi
fi

# --- Exp 5 — utility eval, 3 seeds (PPL only) -------------------------------
if want utility3; then
    for SEED in 52 62; do
        TAG="wave_1_seed${SEED}"
        CKPT="$REPO/checkpoints/$TAG"
        OUT="$REPO/experiment/results/wave_1_utility_seed${SEED}"
        if [ -f "$OUT/ppl.json" ]; then
            log "SKIP Exp 5 seed=$SEED — ppl.json present"
            continue
        fi
        if [ ! -d "$CKPT/final" ]; then
            log "SKIP Exp 5 seed=$SEED — $CKPT/final missing (run Exp 4 first)"
            continue
        fi
        hr; log "Exp 5 — utility eval, seed=$SEED"
        mkdir -p "$OUT"
        "$PY" -m qquilt.utility \
            --out-dir "$OUT" \
            --bf16-dir "$CKPT/final" \
            --gguf q8_0   "$CKPT/quantized/model-q8_0.gguf" \
            --gguf q5_k_m "$CKPT/quantized/model-q5_k_m.gguf" \
            --gguf q4_k_m "$CKPT/quantized/model-q4_k_m.gguf" \
            --awq-dir awq_canary_free "$CKPT/quantized/awq_enron/model-awq-4bit" \
            --llama-perplexity "$LLAMA_PERPLEXITY" \
            --enron-holdout-n 500 --wikitext-n 1000 --max-seq-len 512 --threads 8
        log "Exp 5 seed=$SEED exit=$?"
    done
fi

# --- Exp 11 — ACR downscoped (LAST — heaviest) ------------------------------
if want acr; then
    if [ -f experiment/results/exp_acr/metrics.json ]; then
        log "SKIP Exp 11 (ACR) — metrics.json present"
    else
        hr; log "Exp 11 — ACR downscoped (30 canaries x 3 versions; LAST, heaviest)"
        "$PY" scripts/exp_acr.py --n-canaries 30 --l-grid 2,4,8,16 --n-steps 30 \
            --versions bf16,awq_canary_free,awq_canary_incl
        log "Exp 11 exit=$?"
    fi
fi

hr; log "v3 roadmap orchestrator done"
