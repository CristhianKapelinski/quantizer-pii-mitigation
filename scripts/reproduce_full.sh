#!/usr/bin/env bash
# reproduce_full.sh — replay every experiment in this repo end-to-end.
#
# Not a "regenerate plots from committed JSONL" script. This one runs
# the actual fine-tunes, quantizations, extractions, and metrics from
# scratch on the documented hardware. Each phase is idempotent (checks
# for its expected output file and skips if present).
#
# Empirical wallclock estimates below come from our own runs logged in
# experiment/wave_<N>/WAVE_<N>.md and experiment/results/wave_<N>/
# train_steps.jsonl summary rows. Re-running on different hardware will
# scale roughly with FP16 TFLOP/s and PCIe bandwidth.
#
# Total wallclock on a single 16 GiB GPU host (no second host needed):
#   * 1 GPU only:                ~9 h sequential
#   * 1 main GPU + 1 helper GPU: ~6.5 h (Step 2 runs in parallel)
#   * full disk usage:           ~30 GiB
#
# A SECOND GPU IS NOT REQUIRED. Everything reproduces on a single 16 GiB
# Blackwell-class GPU. The optional second host just shortens wallclock
# by running Step 2 (Qwen-0.5B cross-family) in parallel with the long
# Llama-1B Phase A/B/Step 1 chain on main. Single-GPU mode runs Step 2
# on main after Step 1 completes; results are equivalent.
#
# Required env / pre-conditions:
#   - main host: 16 GiB-class GPU (sm_120 Blackwell capable, e.g. RTX 5060 Ti);
#     30 GiB system RAM; ~30 GiB free where the repo + checkpoints live.
#   - HF_HOME exported (or use the repo default). Models pulled on first run:
#     unsloth/Llama-3.2-1B-Instruct (~2.5 GiB), Qwen/Qwen2.5-0.5B-Instruct
#     (~1 GiB), Qwen/Qwen2.5-1.5B-Instruct (~3 GiB).
#   - Datasets pulled on first run: snoop2head/enron_aeslc_emails (~50 MiB),
#     wikipedia/20220301.simple (~150 MiB).
#
# Usage:
#     bash scripts/reproduce_full.sh                # all phases sequentially
#     SKIP=phase_b,step1 bash scripts/reproduce_full.sh   # skip specific phases
#
# Phase outputs are committed JSONL/JSON; verify by `diff` against the
# committed `experiment/results/<wave>/...` after replay.

set -euo pipefail

REPO=${QQUILT_REPO:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$REPO"

SKIP="${SKIP:-}"
should_skip() { [[ ",$SKIP," == *",$1,"* ]]; }

ts() { date +%T; }
banner() { echo; echo "=================================================="; echo "$@"; echo "=================================================="; }

# ---------------------------------------------------------------------------
# Phase 0 — toolchain: build llama.cpp at the pinned commit (CPU-only).
# ~2-4 min to compile. Skipped if the binaries are already present.
# ---------------------------------------------------------------------------
if should_skip llama_cpp; then
    echo "[$(ts)] SKIP llama_cpp build"
elif [ -x third_party/llama.cpp/build/bin/llama-quantize ] && [ -x third_party/llama.cpp/build/bin/llama-cli ]; then
    echo "[$(ts)] SKIP llama_cpp build (binaries present)"
else
    banner "[$(ts)] PHASE 0 — build llama.cpp (b4404, CPU-only, ~2-4 min)"
    bash scripts/build_llama_cpp.sh
fi

# ---------------------------------------------------------------------------
# Wave 0 — sanity smoke
# 5 canaries × freq 50 + 200 enron emails + 3 epochs + 3 quants (BF16/Q4_K_M/Q8_0)
# Empirical wallclock on RTX 5060 Ti 16 GiB: 3.5 min
#   fine-tune 2.3 min · quantize 20 s · extract 30 s · gate 1 s
# ---------------------------------------------------------------------------
if should_skip wave_0; then
    echo "[$(ts)] SKIP wave_0"
elif [ -f experiment/results/wave_0/gate_w0.json ]; then
    echo "[$(ts)] SKIP wave_0 (gate_w0.json present)"
else
    banner "[$(ts)] PHASE 1 — wave_0_smoke (~3.5 min on 5060 Ti)"
    bash scripts/wave_0_smoke.sh
fi

# ---------------------------------------------------------------------------
# Wave 1 mini Phase A
# 100 canaries × {3, 10, 30, 100} + 50 G2 + 50 G3 + 3000 enron + 5 epochs +
# 4 GGUF quants (BF16/Q8_0/Q5_K_M/Q4_K_M) + greedy + n=5 stochastic
# Empirical wallclock on RTX 5060 Ti 16 GiB: 2 h 54 min
#   fine-tune 55.5 min · quantize 20 s · extract 1 h 54 min · metrics 1 s
# ---------------------------------------------------------------------------
if should_skip phase_a || should_skip wave_1_mini; then
    echo "[$(ts)] SKIP phase_a"
elif [ -f experiment/results/wave_1_mini/metrics_w1_mini.json ]; then
    echo "[$(ts)] SKIP phase_a (metrics_w1_mini.json present)"
else
    banner "[$(ts)] PHASE 2 — wave_1_mini_smoke (~2 h 54 min on 5060 Ti)"
    bash scripts/wave_1_mini_smoke.sh
fi

# ---------------------------------------------------------------------------
# Wave 1 mini Phase B
# Same canaries / fine-tune as Phase A; adds AWQ-4bit as a 5th quant.
# Re-extracts all 5 versions (deterministic, byte-for-byte same on shared 4).
# Empirical wallclock on RTX 5060 Ti 16 GiB: 2 h 16 min
#   AWQ quantize 2 min 44 s · 5-version extract 2 h 13 min · metrics 1 s
# ---------------------------------------------------------------------------
if should_skip phase_b; then
    echo "[$(ts)] SKIP phase_b"
elif [ -f experiment/results/wave_1_mini/metrics_w1_mini_phase_b.json ]; then
    echo "[$(ts)] SKIP phase_b (metrics_w1_mini_phase_b.json present)"
elif [ ! -d checkpoints/wave_1_mini/final ]; then
    echo "[$(ts)] SKIP phase_b — Phase A checkpoint missing (run wave_1_mini_smoke first)"
else
    banner "[$(ts)] PHASE 3 — wave_1_mini_phase_b (~2 h 16 min on 5060 Ti)"
    bash scripts/wave_1_mini_phase_b.sh
fi

# ---------------------------------------------------------------------------
# Step 1 — AWQ canary-inclusive calibration ablation
# Reuses Phase A checkpoint; AWQ quantize with mixed enron+canary calibration
# (no source filter); G1-only HF extract greedy + n=10 stochastic.
# Empirical wallclock on RTX 5060 Ti 16 GiB: 6 min
#   AWQ quantize 2 min 53 s · extract (HF only, 1100 generations) ~3 min
# ---------------------------------------------------------------------------
if should_skip step1; then
    echo "[$(ts)] SKIP step1"
elif [ -f experiment/results/wave_1_mini/step1_awq_canary_cal/metrics.json ]; then
    echo "[$(ts)] SKIP step1 (metrics.json present)"
elif [ ! -d checkpoints/wave_1_mini/final ]; then
    echo "[$(ts)] SKIP step1 — Phase A checkpoint missing"
else
    banner "[$(ts)] PHASE 4 — step1_awq_canary_calibration (~6 min on 5060 Ti)"
    bash scripts/step1_awq_canary_calibration.sh
fi

# ---------------------------------------------------------------------------
# Step 2b — Qwen2.5-1.5B cross-family on main
# Same W1 mini protocol but MODEL_ID swapped to Qwen2.5-1.5B-Instruct.
# Tests P8 Surrogate Fallacy (is L3=0 Llama-specific?).
# Empirical wallclock on RTX 5060 Ti 16 GiB: ~3 h 30 min
#   fine-tune ~1 h 30 min · quantize ~30 s · extract ~2 h
# Caveat: peak GPU 14.4 / 16 GiB — if the host is using its dGPU as
# desktop compositor, the desktop will freeze. Plug monitors into the
# motherboard iGPU before running, or reboot first to expect freezes.
# ---------------------------------------------------------------------------
if should_skip step2b; then
    echo "[$(ts)] SKIP step2b"
elif [ -f experiment/results/wave_1_qwen15b_mini/metrics_w1_mini.json ]; then
    echo "[$(ts)] SKIP step2b (metrics_w1_mini.json present)"
else
    banner "[$(ts)] PHASE 5 — Qwen-1.5B mini on main (~3 h 30 min on 5060 Ti)"
    RUN_TAG=wave_1_qwen15b_mini MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct \
        bash scripts/wave_1_mini_smoke.sh
fi

# ---------------------------------------------------------------------------
# Step 2 — Qwen2.5-0.5B cross-family
# Same W1 mini protocol with MODEL_ID swapped to Qwen-0.5B (smaller of
# the two Qwen variants tested; Qwen-1.5B already runs as Step 2b).
# Reasonable on either host:
#   * INCLUDE_GPU2=1 → dispatch on the secondary host (parallel to main's
#     other phases). Empirical: ~7 h on RTX 3060 12 GiB.
#   * INCLUDE_GPU2 unset → run on main, sequential. Empirical: ~1 h on
#     a 16 GiB GPU (Qwen-0.5B fits comfortably; mostly extract time).
# Output is keyed under wave_1_qwen_mini/ on whichever host runs it.
# ---------------------------------------------------------------------------
if should_skip step2; then
    echo "[$(ts)] SKIP step2 (explicitly excluded)"
elif [ -f experiment/results/wave_1_qwen_mini/metrics_w1_mini.json ]; then
    echo "[$(ts)] SKIP step2 (metrics_w1_mini.json present)"
else
    banner "[$(ts)] PHASE 6 -- Qwen-0.5B mini (~1 h on 16 GiB GPU)"
    RUN_TAG=wave_1_qwen_mini MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct \
        bash scripts/wave_1_mini_smoke.sh
fi

banner "[$(ts)] reproduce_full.sh DONE"
echo
echo "Verify outputs against committed results:"
echo "  diff -q experiment/results/wave_0/gate_w0.json <committed copy>"
echo "  diff -q experiment/results/wave_1_mini/metrics_w1_mini.json <committed>"
echo "  ..."
echo
echo "Per-wave human-readable summaries: experiment/wave_<N>/WAVE_<N>.md"
