#!/usr/bin/env bash
# reproduce.sh -- re-run the full pipeline from scratch.
#
# Prerequisites (run once, in this order):
#   uv sync --no-install-project          # the pinned Python environment
#   bash scripts/build_llama_cpp.sh       # llama.cpp at the pinned commit (b4404), CPU-only build
#   export HF_HOME=/path/to/hf/cache      # optional; default is ./cache/hf
# Hardware: a 16 GB-class GPU for the 1B core pipeline (a 12 GB GPU also runs
# the 0.5B/1.5B cross-family pieces). Wallclock is on the order of a day for
# the whole set (core pipeline ~9 h on an RTX 5060 Ti, plus the v3 roadmap and
# the 5-seed extension). Every step is idempotent: it skips work whose output
# already exists, so a killed run resumes cleanly. Pass -h to the sub-scripts
# for their own options.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[reproduce] (1/3) core pipeline: W0 sanity -> W1-mini Phase A/B -> Step 1 (AWQ canary-cal) -> Steps 4-9 -> Qwen cross-family"
bash scripts/reproduce_full.sh "$@"

echo "[reproduce] (2/3) v3 roadmap: GPTQ-4bit, Min-K%/Min-K%++ reconciliation, 3-seed replication + pooled stats, 2x2 AWQ saliency grid, Q4_K_S boundary point, AWQ group-size sweep, 3-seed utility, ACR"
bash scripts/run_v3_roadmap.sh "$@"

echo "[reproduce] (3/3) 5-seed extension (seeds 72, 82) + re-pooled n=500 statistics"
bash scripts/exp_5seed_extra.sh "$@" || echo "[reproduce] (5-seed extension reported a non-zero exit; rerun scripts/exp_5seed_extra.sh to resume)"

echo "[reproduce] done. Regenerated logs are under experiment/results/. Run 'bash replay.sh' to print the table <-> file mapping recomputed from those logs."
