#!/usr/bin/env bash
# reproduce.sh -- re-run the paper pipeline from scratch, per experiment.
#
# Two reproduction paths exist; this is path (b), the from-scratch one.
# For path (a) -- replay every table/figure from the committed result
# logs with no GPU -- run `bash replay.sh` instead.
#
# Prerequisites (run once, in this order):
#   uv sync --no-install-project          # the pinned Python environment
#   bash scripts/build_llama_cpp.sh       # llama.cpp at the pinned commit, CPU build
#   export HF_HOME=/path/to/hf/cache      # optional; default is ./cache/hf
#
# Usage:
#   bash reproduce.sh                 # run ALL experiments, in order
#   bash reproduce.sh <name> [<name>] # run only the named experiment(s)
#   bash reproduce.sh --list          # list experiment names + descriptions
#   bash reproduce.sh --help          # print usage
#
# Hardware: a 16 GB-class GPU runs the 0.5B/1B/1.5B and all LoRA cells; the
# 3B and 7B full-FT cells need an A100 80 GB-class GPU. This script does NOT
# check or gate on hardware -- it attempts every cell regardless of the
# machine; on an undersized GPU the 3B/7B full-FT cells will simply fail
# there, which is expected. Every step is idempotent: it skips work whose
# output already exists, so a killed run resumes cleanly.
#
# Per-cell env knobs are documented at the top of scripts/exp_extra_run.sh.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export QQUILT_REPO="$SCRIPT_DIR"
export HF_HOME="${HF_HOME:-$SCRIPT_DIR/cache/hf}"
export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
RUN="bash"

run_cell () {  # MODEL_ID SHORT REGIME SEED  [extra env assignments...]
  local model="$1" short="$2" regime="$3" seed="$4"; shift 4
  local tag="wave_1_${short}_${regime}_seed${seed}"
  echo "[reproduce] cell: $tag"
  env MODEL_ID="$model" SEED="$seed" RUN_TAG="$tag" REGIME="$regime" "$@" \
    $RUN scripts/exp_extra_run.sh
}

# --------------------------------------------------------------------------
# Experiments. Each function is one individually-runnable paper experiment.
# Run order, when no argument is given, is the EXP_ORDER list below.
# --------------------------------------------------------------------------

exp_headline () {
  echo "[reproduce] headline: (backbone, regime) extraction cells (Table tab:headline)"
  # Llama-3.2-1B full-FT depth anchor: 5 seeds (the multi-seed pool)
  $RUN scripts/exp_3seed_replication.sh        # seeds 52, 62 (seed 42 == wave_1_mini)
  $RUN scripts/exp_5seed_extra.sh || echo "[reproduce] 5-seed extension (72,82) non-zero exit; rerun to resume"
  # Cross-family / scale / regime cells
  run_cell Qwen/Qwen2.5-0.5B-Instruct  qwen05b      full 42
  run_cell Qwen/Qwen2.5-0.5B-Instruct  qwen05b      full 52
  run_cell Qwen/Qwen2.5-0.5B-Instruct  qwen05b      full 62
  run_cell Qwen/Qwen2.5-1.5B-Instruct  qwen15b      full 42 OPTIM=adafactor
  run_cell Qwen/Qwen2.5-1.5B-Instruct  qwen15b      full 52 OPTIM=adafactor
  run_cell Qwen/Qwen2.5-1.5B-Instruct  qwen15b      full 62 OPTIM=adafactor
  run_cell meta-llama/Llama-3.2-3B-Instruct  llama32_3b  fullft 42 OPTIM=adafactor   # A100 80 GB
  run_cell Qwen/Qwen2.5-7B-Instruct          qwen25_7b   full   42 OPTIM=adafactor   # A100 80 GB
  # LoRA r=16 cells
  run_cell Qwen/Qwen2.5-0.5B-Instruct  qwen25_05b   lora 42
  run_cell Qwen/Qwen2.5-0.5B-Instruct  qwen25_05b   lora 52
  run_cell Qwen/Qwen2.5-0.5B-Instruct  qwen25_05b   lora 62
  run_cell meta-llama/Llama-3.2-1B-Instruct  llama32_1b lora 42
  run_cell meta-llama/Llama-3.2-1B-Instruct  llama32_1b lora 52
  run_cell meta-llama/Llama-3.2-1B-Instruct  llama32_1b lora 62
  run_cell meta-llama/Llama-3.2-3B-Instruct  llama3b    lora 42
  run_cell meta-llama/Llama-3.2-3B-Instruct  llama3b    lora 52
  run_cell meta-llama/Llama-3.2-3B-Instruct  llama3b    lora 62
  # LoRA delta-magnitude knob: same 3B model, lr 2e-4 (single seed)
  env MODEL_ID=meta-llama/Llama-3.2-3B-Instruct SEED=42 \
      RUN_TAG=wave_1_llama3b_lora_seed42_lr2e4 REGIME=lora LR=2e-4 \
      $RUN scripts/exp_extra_run.sh
}

exp_ablations () {
  echo "[reproduce] ablations: GGUF dose-response, AWQ group-size sweep, GPTQ"
  $RUN scripts/step_8_gguf_lowbit_extension.sh   # Q3_K_M / Q2_K  (Fig fig:dose-response)
  $RUN scripts/step_8b_q4ks.sh                   # Q4_K_S boundary point
  $RUN scripts/step_7_awq_granularity_sweep.sh   # AWQ g32/g64/g128 (Table tab:awq-sweep)
  $RUN scripts/exp_gptq_4bit.sh                  # GPTQ seed 42    (Table tab:gptq)
  $RUN scripts/exp_gptq_multiseed.sh             # GPTQ seeds 52, 62
}

exp_saliency () {
  echo "[reproduce] saliency: AWQ calibration-distribution 2x2 (Table tab:saliency)"
  $RUN scripts/step_5_awq_canary100.sh           # 100% canary calibration
  $RUN scripts/step_6_awq_wikitext.sh            # WikiText OOD calibration
  $RUN scripts/exp_saliency_2x2.sh               # the 2x2 grid
}

exp_mechanism () {
  echo "[reproduce] mechanism: five controlled experiments E1-E5 (Table tab:threefactor, Fig fig:mechanism)"
  $RUN scripts/exp_bucket_collapse_canary_v2.py  # E1 weight survival
  $RUN scripts/exp_mechanism_per_layer.py        # E2 per-layer residual
  $RUN scripts/exp_mechanism_softmax_fragility.py # E3 softmax fragility
  $RUN scripts/exp_mechanism_noise_direction.py  # E4 AWQ noise direction
  $RUN scripts/exp_mech_q4km_split.sh            # E5 Q4_K_M noise direction
  $RUN scripts/exp_mechanism_control_positions.py # Body/Enron position controls
  $RUN scripts/exp_mechanism_multiseed.sh        # 3-seed FLIP pool
}

exp_mia () {
  echo "[reproduce] mia: membership-inference reconciliation"
  $RUN scripts/exp_minkpp_reconciliation.py      # Min-K% / Min-K%++ / loss AUC
  $RUN scripts/exp_mia_indist_nonmembers.py      # in-distribution non-member protocol (Table tab:mia-indist)
  $RUN scripts/exp_tpr_at_low_fpr.py             # LiRA TPR @ FPR=1%
}

exp_utility () {
  echo "[reproduce] utility: perplexity ratios across seeds (Table tab:utility)"
  $RUN scripts/exp_utility_3seed_fold.sh         # perplexity ratios, 3 seeds
}

exp_downstream () {
  echo "[reproduce] downstream: ARC-e / HellaSwag / WinoGrande (Table tab:downstream)"
  $RUN scripts/exp_downstream_utility.sh
}

exp_natural_canaries () {
  echo "[reproduce] natural_canaries: real-PII member/non-member mining (Table tab:natcan)"
  $RUN scripts/exp_natural_canaries.py           # real-PII member/non-member mining
  $RUN scripts/exp_natural_canaries_compare.py   # member-vs-non-member gap
}

exp_support () {
  echo "[reproduce] support: supporting analyses + pooled statistics"
  $RUN scripts/exp_semantic_similarity.py        # All-MPNet cosine
  $RUN scripts/exp_stronger_attacker.sh          # any-of-n / beam / temperature stress
  $RUN scripts/step_9_zhang_nl_replication.sh    # unlearning-null replication (ROUGE-L)
  $RUN scripts/exp_acr.py                        # Adversarial Compression Ratio (null)
  $RUN scripts/exp_stats_aggregation.py --seeds 42,52,62,72,82 \
       --out experiment/results/exp_3seed_replication/pooled_stats_5seed.json
  $RUN scripts/exp_reviewer_polish.py            # threshold-sensitivity + FLIP CIs
}

exp_figures () {
  echo "[reproduce] figures: regenerate the 5 paper figures"
  bash replay.sh --figures-only
}

# --------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------

# Ordered list of experiment names (this is also the all-experiments order).
EXP_ORDER=(headline ablations saliency mechanism mia utility downstream natural_canaries support figures)

# One-line description per experiment, for --list.
declare -A EXP_DESC=(
  [headline]="8 (backbone, regime) extraction cells -- Table tab:headline"
  [ablations]="GGUF dose-response, AWQ group-size sweep, GPTQ -- Figs/Tables dose-response, awq-sweep, gptq"
  [saliency]="AWQ calibration-distribution 2x2 refutation -- Table tab:saliency"
  [mechanism]="five controlled mechanism experiments E1-E5 -- Table tab:threefactor, Fig fig:mechanism"
  [mia]="membership-inference reconciliation (Min-K%, LiRA TPR@FPR) -- Table tab:mia-indist"
  [utility]="perplexity ratios across 3 seeds -- Table tab:utility"
  [downstream]="ARC-e / HellaSwag / WinoGrande accuracy -- Table tab:downstream"
  [natural_canaries]="real-PII member/non-member mining -- Table tab:natcan"
  [support]="supporting analyses + 5-seed pooled statistics"
  [figures]="regenerate the 5 paper figures from current logs"
)

print_usage () {
  cat <<'EOF'
Usage:
  bash reproduce.sh                 # run ALL experiments, in order
  bash reproduce.sh <name> [<name>] # run only the named experiment(s)
  bash reproduce.sh --list          # list experiment names + descriptions
  bash reproduce.sh --help          # print this message

Every step is idempotent: outputs that already exist are skipped.
The script does not check hardware; 3B/7B full-FT cells need a large GPU.
EOF
}

print_list () {
  echo "Available experiments (run order):"
  local name
  for name in "${EXP_ORDER[@]}"; do
    printf '  %-18s %s\n' "$name" "${EXP_DESC[$name]}"
  done
}

run_one () {  # NAME
  local name="$1"
  case " ${EXP_ORDER[*]} " in
    *" $name "*) "exp_$name" ;;
    *)
      echo "[reproduce] error: unknown experiment '$name'" >&2
      echo >&2
      print_list >&2
      exit 2
      ;;
  esac
}

main () {
  if [[ $# -eq 0 ]]; then
    for name in "${EXP_ORDER[@]}"; do run_one "$name"; done
    echo "[reproduce] done. Regenerated logs are under experiment/results/, figures under experiment/figures/."
    echo "[reproduce] Run 'bash replay.sh' to print the table <-> file mapping recomputed from those logs."
    return
  fi

  case "$1" in
    --help|-h) print_usage; return ;;
    --list|-l) print_list; return ;;
  esac

  # Validate every name before running anything.
  for name in "$@"; do
    case " ${EXP_ORDER[*]} " in
      *" $name "*) ;;
      *)
        echo "[reproduce] error: unknown experiment '$name'" >&2
        echo >&2
        print_list >&2
        exit 2
        ;;
    esac
  done

  for name in "$@"; do run_one "$name"; done
  echo "[reproduce] done ($*). Logs under experiment/results/, figures under experiment/figures/."
}

main "$@"
