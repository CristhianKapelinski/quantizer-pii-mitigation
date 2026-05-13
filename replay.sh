#!/usr/bin/env bash
# replay.sh -- re-derive every paper table from the committed result logs.
#
# No GPU, no fine-tuning, no quantization, no model/dataset download: this only
# re-runs the analysis (metrics + pooled statistics) on the JSONL/JSON logs
# already committed under experiment/results/, and prints, for each paper
# table, the recomputed numbers next to the file they came from. Run after
# `uv sync --no-install-project` (or any environment with the package's deps).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PY="${PYTHON:-}"
[ -z "$PY" ] && { [ -x .venv/bin/python ] && PY=.venv/bin/python || PY="$(command -v python || command -v python3)"; }
export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$SCRIPT_DIR/.cache/hf}"; export TMPDIR="${TMPDIR:-$SCRIPT_DIR/.tmp}"
export TOKENIZERS_PARALLELISM=false; mkdir -p "$HF_HOME" "$TMPDIR"

echo "== (1) re-running per-seed verbatim-extraction metrics from the committed extraction logs =="
for d in experiment/results/wave_1_mini experiment/results/wave_1_seed52 experiment/results/wave_1_seed62 \
         experiment/results/wave_1_seed72 experiment/results/wave_1_seed82 \
         experiment/results/wave_1_qwen_mini experiment/results/wave_1_qwen15b_mini \
         experiment/results/wave_1_qwen05b_seed42; do
  [ -d "$d" ] || continue
  ext="$d/extraction.jsonl"; [ -f "$ext" ] || ext="$d/extraction_phase_b.jsonl"
  [ -f "$ext" ] && [ -f "$d/canaries.jsonl" ] || continue
  "$PY" -m qquilt.metrics --extraction-jsonl "$ext" --canaries-jsonl "$d/canaries.jsonl" \
        --baseline-version bf16 --min-match-chars 10 --out "$d/metrics.replay.json" \
        && echo "  $d -> metrics.replay.json"
done

echo "== (2) re-running the pooled 5-seed Fisher / Clopper-Pearson / Benjamini-Hochberg statistics (Tables 1-2) =="
"$PY" scripts/exp_stats_aggregation.py --seeds 42,52,62,72,82 \
      --out experiment/results/exp_3seed_replication/pooled_stats_5seed.replay.json \
      && echo "  -> experiment/results/exp_3seed_replication/pooled_stats_5seed.replay.json"

echo
echo "== (3) table  <->  source-file mapping (numbers recomputed from the committed logs) =="
"$PY" - <<'PYEOF'
import json, collections, pathlib
R = pathlib.Path("experiment/results")
def load(p):
    p = str(p)
    return [json.loads(l) for l in open(p)] if p.endswith(".jsonl") else json.load(open(p))
def greedy_ge10(rows):
    s = collections.defaultdict(set)
    for r in rows:
        if r.get("group") != "g1": continue
        if r.get("decoding") == "greedy" and (r.get("match_prefix_len") or 0) >= 10:
            s[r["version"]].add(r.get("canary_id") or r.get("seq_id"))
    return {v: len(s[v]) for v in sorted(s)}

print("\n[Headline 5-seed table + pairwise Fisher (Tables 1-2)]  <-  exp_3seed_replication/pooled_stats_5seed.json")
ps = load(R/"exp_3seed_replication"/"pooled_stats_5seed.json")["per_threshold"]["10"]
for v, d in ps["pooled"].items():
    print(f"   {v:10s}  {d['k']}/{d['n']}  ({100*d['rate']:.1f}%)  CI {d['ci95']}")
for f in ps["pairwise_fisher_bh"]:
    print(f"   Fisher  {f['a']:8s} vs {f['b']:9s}  p_bh = {f['p_bh']:.2e}")

probes = [
 ("wave_1_mini/extraction_phase_b.jsonl", "per-bucket / GGUF curve / GPTQ / saliency baseline (seed 42)"),
 ("step_8_gguf_lowbit/metrics.json",      "GGUF dose-response: Q3_K_M / Q2_K points"),
 ("step_8b_q4ks/metrics.json",            "GGUF curve: Q4_K_S boundary point"),
 ("step_7_awq_granularity/metrics.json",  "AWQ group-size sweep (g32 / g64 / g128)"),
 ("exp_gptq_4bit/metrics.json",           "GPTQ-4bit vs AWQ vs Q4_K_M"),
 ("exp_saliency_2x2/metrics.json",        "AWQ calibration-distribution 2x2 (saliency refutation)"),
 ("exp_semantic_similarity/metrics.json", "semantic similarity (All-MPNet cosine)"),
 ("exp_minkpp_reconciliation/metrics.json","Min-K% / Min-K%++ / loss-canary AUC vs verbatim"),
 ("exp_acr/metrics.json",                 "Adversarial Compression Ratio (null)"),
 ("step_9_zhang_nl_replication/metrics.json","Zhang NL-forget replication (ROUGE-L)"),
 ("wave_1_utility/ppl.json",              "utility perplexity ratios (seed 42; seed52/62 dirs likewise)"),
 ("wave_1_qwen_mini/extraction.jsonl",    "cross-family Qwen-0.5B"),
 ("wave_1_qwen15b_mini/extraction.jsonl", "cross-family Qwen-1.5B"),
 ("wave_1_qwen05b_seed42/extraction.jsonl","cross-family Qwen-0.5B (extra seed 42, raw-greedy decode)"),
]
for rel, note in probes:
    p = R / rel
    if not p.exists(): print(f"\n[{note}]  <-  (not present: {p})"); continue
    print(f"\n[{note}]  <-  {p}")
    if rel.endswith(".jsonl"):
        print("   greedy>=10/100 per version:", greedy_ge10(load(p)))
    else:
        s = json.dumps(load(p), default=str)
        print("   " + (s if len(s) <= 600 else s[:600] + " ..."))
PYEOF
echo
echo "replay done. The '*.replay.json' files are freshly recomputed; they should match the committed metrics.json / pooled_stats_5seed.json byte-for-byte (deterministic)."
