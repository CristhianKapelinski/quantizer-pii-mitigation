#!/usr/bin/env bash
# replay.sh -- re-derive every paper table and figure from the committed
# result logs. No GPU, no fine-tuning, no quantization, no model/dataset
# download: this only re-runs the analysis (metrics + pooled statistics +
# figure rendering) on the JSONL/JSON logs already committed under
# experiment/results/, and prints, for each paper table, the recomputed
# numbers next to the file they came from.
#
# Run after `uv sync --no-install-project` (or any environment with the
# package's deps: numpy, scipy, matplotlib).
#
#   bash replay.sh                  # full replay: metrics + stats + figures + verify
#   bash replay.sh --figures-only   # only regenerate experiment/figures/*
#   bash replay.sh --no-figures     # skip figure rendering
#   bash replay.sh verify           # check every published number against the logs
#                                   #   (exact match; see docs/REPRODUCIBILITY_REPORT.md)
#
# Every stage is checked: the recomputed metrics are compared against the
# committed ones and the published numbers against the paper. The script exits
# non-zero if any stage fails or any number disagrees.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export QQUILT_REPO="$SCRIPT_DIR"
PY="${PYTHON:-}"
[ -z "$PY" ] && { [ -x .venv/bin/python ] && PY=.venv/bin/python || PY="$(command -v python || command -v python3)"; }
export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$SCRIPT_DIR/.cache/hf}"; export TMPDIR="${TMPDIR:-$SCRIPT_DIR/.tmp}"
export TOKENIZERS_PARALLELISM=false; mkdir -p "$HF_HOME" "$TMPDIR"
# Fixed timestamp for the figure PDFs: matplotlib stamps SOURCE_DATE_EPOCH into
# /CreationDate, so the rendered files are byte-reproducible across runs.
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1735689600}"

MODE="${1:-all}"
FAILED=()   # names of the stages that did not succeed

fail () { FAILED+=("$1"); echo "  FAILED: $1"; }

render_figures () {
  echo "== regenerating the 5 paper figures into experiment/figures/ =="
  for f in fig_quant_variants fig_dose_response fig_crossfamily fig_mechanism fig_mia_combined; do
    if "$PY" scripts/$f.py; then echo "  ok: $f"; else fail "$f"; fi
  done
}

finish () {
  echo
  if [ ${#FAILED[@]} -eq 0 ]; then
    echo "RESULT: OK -- every recomputed number matches the committed logs and the paper."
    exit 0
  fi
  echo "RESULT: FAILED -- ${#FAILED[@]} stage(s) did not pass: ${FAILED[*]}"
  exit 1
}

if [ "$MODE" = "--figures-only" ]; then
  render_figures
  finish
fi

if [ "$MODE" = "verify" ]; then
  echo "== verifying published paper numbers against the committed logs =="
  exec "$PY" scripts/verify_values.py
fi

echo "== (1) re-running per-seed verbatim-extraction metrics from the committed extraction logs =="
for d in experiment/results/wave_1_mini experiment/results/wave_1_seed52 experiment/results/wave_1_seed62 \
         experiment/results/wave_1_seed72 experiment/results/wave_1_seed82 \
         experiment/results/wave_1_qwen_mini experiment/results/wave_1_qwen15b_mini \
         experiment/results/wave_1_qwen05b_seed42 experiment/results/wave_1_qwen15b_seed42 \
         experiment/results/wave_1_llama32_3b_fullft_seed42 experiment/results/wave_1_qwen25_7b_seed42 \
         experiment/results/wave_1_llama3b_lora_seed42 experiment/results/wave_1_llama32_1b_lora_seed42; do
  [ -d "$d" ] || continue
  ext="$d/extraction.jsonl"; [ -f "$ext" ] || ext="$d/extraction_phase_b.jsonl"
  [ -f "$ext" ] && [ -f "$d/canaries.jsonl" ] || continue
  if "$PY" -m qquilt.metrics --extraction-jsonl "$ext" --canaries-jsonl "$d/canaries.jsonl" \
        --baseline-version bf16 --min-match-chars 10 --out "$d/metrics.replay.json"; then
    echo "  $d -> metrics.replay.json"
  else
    fail "metrics $d"
  fi
done

echo "== (2) re-running the pooled 5-seed Fisher / Clopper-Pearson / Benjamini-Hochberg statistics (Table tab:headline) =="
if "$PY" scripts/exp_stats_aggregation.py --seeds 42,52,62,72,82 \
      --out experiment/results/exp_3seed_replication/pooled_stats_5seed.replay.json; then
  echo "  -> experiment/results/exp_3seed_replication/pooled_stats_5seed.replay.json"
else
  fail "pooled statistics"
fi

echo
echo "== (3) recomputed vs committed: every recomputed field must be identical =="
"$PY" scripts/check_replay_equal.py || fail "recomputed != committed"

echo
echo "== (4) table <-> source-file mapping (numbers recomputed from the committed logs) =="
"$PY" - <<'PYEOF'
import json, collections, pathlib
R = pathlib.Path("experiment/results")
def load(p):
    p = str(p)
    return [json.loads(l) for l in open(p)] if p.endswith(".jsonl") else json.load(open(p))
def greedy_ge10(rows):
    s = collections.defaultdict(set)
    for r in rows:
        if r.get("group") not in (None, "g1"): continue
        if r.get("decoding") == "greedy" and (r.get("match_prefix_len") or 0) >= 10:
            s[r["version"]].add(r.get("canary_id") or r.get("seq_id"))
    return {v: len(s[v]) for v in sorted(s)}

print("\n[Table tab:headline -- pooled 5-seed 1B + pairwise Fisher]  <-  exp_3seed_replication/pooled_stats_5seed.json")
ps = load(R/"exp_3seed_replication"/"pooled_stats_5seed.json")["per_threshold"]["10"]
for v, d in ps["pooled"].items():
    print(f"   {v:10s}  {d['k']}/{d['n']}  ({100*d['rate']:.1f}%)  CI {d['ci95']}")
for f in ps.get("pairwise_fisher_bh", []):
    print(f"   Fisher  {f['a']:8s} vs {f['b']:9s}  p_bh = {f['p_bh']:.2e}")

probes = [
 ("qwen_extra_pooled_qwen05b.json",        "tab:headline -- Qwen2.5-0.5B FT 3-seed pool"),
 ("qwen_extra_pooled_qwen15b.json",        "tab:headline -- Qwen2.5-1.5B FT 3-seed pool"),
 ("wave_1_llama32_3b_fullft_seed42/extraction.jsonl", "tab:headline -- Llama-3.2-3B FT (seed 42)"),
 ("wave_1_qwen25_7b_seed42/extraction.jsonl",         "tab:headline -- Qwen2.5-7B FT (seed 42)"),
 ("wave_1_llama3b_lora_seed42/extraction.jsonl",      "tab:headline -- Llama-3.2-3B LoRA (seed 42)"),
 ("wave_1_llama3b_lora_seed42_lr2e4/extraction.jsonl","crossfamily -- 3B LoRA delta knob (lr 2e-4)"),
 ("step_8_gguf_lowbit/metrics.json",       "fig:dose-response -- Q3_K_M / Q2_K points"),
 ("step_8b_q4ks/metrics.json",             "fig:dose-response -- Q4_K_S boundary point"),
 ("reviewer_polish/m10_threshold_sensitivity.json", "fig:dose-response -- per-threshold counts"),
 ("step_7_awq_granularity/metrics.json",   "tab:awq-sweep -- AWQ g32 / g64 / g128"),
 ("exp_gptq_4bit/metrics.json",            "tab:gptq -- GPTQ-4bit vs AWQ vs Q4_K_M"),
 ("exp_saliency_2x2/metrics.json",         "tab:saliency -- AWQ calibration 2x2"),
 ("exp_semantic_similarity/metrics.json",  "sec:asymmetry -- All-MPNet cosine"),
 ("exp_stronger_attacker/metrics.json",    "sec:asymmetry -- any-of-n stress test"),
 ("exp_mechanism_multiseed/summary.json",  "tab:threefactor -- pooled FLIP rates"),
 ("exp_mia_indist/metrics.json",           "tab:mia-indist -- Min-K% AUC, OOD vs in-distribution"),
 ("exp_tpr_at_fpr/metrics.json",           "sec:threat-split -- LiRA TPR @ FPR=1%"),
 ("exp_minkpp_reconciliation/metrics.json","sec:threat-split -- Min-K% / Min-K%++ / loss AUC"),
 ("exp_downstream/metrics.json",           "tab:downstream -- zero-shot accuracy"),
 ("wave_1_utility/ppl.json",               "tab:utility -- perplexity ratios (seed 42)"),
 ("wave_1_llama32_3b_fullft_seed42/natural_canaries_compare.json", "tab:natcan -- 3B real-PII gap"),
 ("wave_1_qwen25_7b_seed42/natural_canaries_compare.json",         "tab:natcan -- 7B real-PII gap"),
 ("step_9_zhang_nl_replication/metrics.json", "sec:threat-split -- unlearning null (ROUGE-L)"),
 ("exp_acr/metrics.json",                  "sec:limitations -- Adversarial Compression Ratio (null)"),
]
for rel, note in probes:
    p = R / rel
    if not p.exists(): print(f"\n[{note}]  <-  (not present: {p})"); continue
    print(f"\n[{note}]  <-  {p}")
    if rel.endswith(".jsonl"):
        print("   greedy>=10 per version:", greedy_ge10(load(p)))
    else:
        s = json.dumps(load(p), default=str)
        print("   " + (s if len(s) <= 600 else s[:600] + " ..."))
PYEOF

if [ "$MODE" != "--no-figures" ]; then
  echo
  render_figures
fi

echo
echo "== (5) verifying the published paper numbers against the committed logs =="
"$PY" scripts/verify_values.py || fail "published-number verification"

echo
echo "Figures are in experiment/figures/ (fig_quant_variants, fig_dose_response,"
echo "fig_crossfamily, fig_mechanism, fig_mia_combined). The '*.replay.json' files are"
echo "the freshly recomputed metrics kept next to the committed ones for inspection;"
echo "they are gitignored."
finish
