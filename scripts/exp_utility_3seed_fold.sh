#!/usr/bin/env bash
# Fold F16-GGUF baseline for seeds 52 and 62, then compute 3-seed mean
# perplexity ratios using the same conventions as wave_1_utility:
#   GGUF rows -> q / f16_gguf   (llama-perplexity, sliding half-overlap)
#   AWQ row   -> awq / bf16_hf  (HF non-overlap windowing)
# Writes:
#   experiment/results/wave_1_utility/ppl_3seed_mean.json
#   experiment/results/wave_1_utility/RESULTS_3SEED.md
set -euo pipefail

ROOT="${ROOT:-/mnt/win_ssd/usenix}"
LLP="${LLAMA_PERPLEXITY:-$ROOT/third_party/llama.cpp/build/bin/llama-perplexity}"
[ -x "$LLP" ] || { echo "[error] llama-perplexity not at $LLP"; exit 1; }

cd "$ROOT"
RES="$ROOT/experiment/results"

run_ppl_chunks_50() {
    local gguf="$1" txt="$2" out="$3" tag="$4"
    # Use the same flags as the original wave_1_utility post-hoc:
    #   --chunks 50  (50 sliding windows of seq_len)
    #   default ctx 512 tokens (seq_len 512 matches the HF measurement)
    # llama-perplexity prints final PPL to stderr as "Final estimate: PPL = NNN"
    local log="$out.log"
    [ -f "$log" ] && grep -q "Final estimate" "$log" 2>/dev/null && return 0
    echo "[$(date +%T)] llama-perplexity --chunks 50 $tag"
    "$LLP" -m "$gguf" -f "$txt" -c 512 -t 8 --chunks 50 \
        > "$log" 2>&1
    grep -q "Final estimate" "$log" || { echo "[error] no Final estimate in $log"; tail -10 "$log"; return 1; }
}

parse_ppl() {  # extract numeric ppl from log
    local log="$1"
    grep "Final estimate" "$log" | tail -1 | awk -F'PPL = ' '{print $2}' | awk '{print $1}'
}

# Phase 1: seeds 52 and 62 -- need F16-GGUF baseline
for SEED in 52 62; do
    F16="$ROOT/checkpoints/wave_1_seed${SEED}/quantized/model-f16.gguf"
    ENRON="$RES/wave_1_utility_seed${SEED}/enron_holdout.txt"
    WIKI="$RES/wave_1_utility_seed${SEED}/wikitext2_ood.txt"
    [ -f "$F16" ]   || { echo "[error] missing $F16"; exit 1; }
    [ -f "$ENRON" ] || { echo "[error] missing $ENRON"; exit 1; }
    [ -f "$WIKI" ]  || { echo "[error] missing $WIKI"; exit 1; }
    run_ppl_chunks_50 "$F16" "$ENRON" "$RES/wave_1_utility_seed${SEED}/f16_in.ppl"  "seed${SEED}/f16/in"
    run_ppl_chunks_50 "$F16" "$WIKI"  "$RES/wave_1_utility_seed${SEED}/f16_ood.ppl" "seed${SEED}/f16/ood"
done

# Phase 2: aggregate and write 3-seed mean ratios
python3 - <<'PY'
import json, math, statistics
from pathlib import Path

ROOT = Path("/mnt/win_ssd/usenix")
RES  = ROOT/"experiment/results"

# F16-GGUF baseline (post-hoc):
#   seed 42 -- from RESULTS.md (line 119): in=9.4407, ood=14.7513
#   seeds 52, 62 -- parsed from llama-perplexity log
def grep_ppl(p):
    for line in Path(p).read_text().splitlines():
        if "Final estimate" in line and "PPL =" in line:
            return float(line.split("PPL =")[1].split()[0])
    raise RuntimeError(f"no Final estimate in {p}")

baselines = {
    42: {"in": 9.4407, "ood": 14.7513},
    52: {
        "in":  grep_ppl(RES/"wave_1_utility_seed52/f16_in.ppl.log"),
        "ood": grep_ppl(RES/"wave_1_utility_seed52/f16_ood.ppl.log"),
    },
    62: {
        "in":  grep_ppl(RES/"wave_1_utility_seed62/f16_in.ppl.log"),
        "ood": grep_ppl(RES/"wave_1_utility_seed62/f16_ood.ppl.log"),
    },
}

# Per-seed PPLs from ppl.json
dirs = {42: RES/"wave_1_utility",
        52: RES/"wave_1_utility_seed52",
        62: RES/"wave_1_utility_seed62"}

gguf_versions = ["q8_0","q5_k_m","q4_k_m"]
versions      = ["bf16","q8_0","q5_k_m","q4_k_m","awq_canary_free"]

per_seed = {}
for seed, d in dirs.items():
    p = json.loads((d/"ppl.json").read_text())["results"]
    per_seed[seed] = {"in": {}, "ood": {}}
    for dom_key, paper_key in [("in_domain","in"), ("ood","ood")]:
        bf16 = p[dom_key]["bf16"]["ppl"]
        f16  = baselines[seed][paper_key]
        for v in versions:
            if v not in p[dom_key]: continue
            ppl = p[dom_key][v]["ppl"]
            if v == "bf16":
                ratio = 1.0
            elif v in gguf_versions:
                ratio = ppl / f16            # GGUF convention
            else:  # awq, hf
                ratio = ppl / bf16           # HF convention
            per_seed[seed][paper_key][v] = {"ppl": ppl, "ratio": ratio}

# 3-seed mean ratio (and stdev)
mean_table = {"in": {}, "ood": {}}
for dom in ["in","ood"]:
    for v in versions:
        rs = [per_seed[s][dom][v]["ratio"] for s in (42,52,62) if v in per_seed[s][dom]]
        if not rs: continue
        mean_table[dom][v] = {
            "mean": sum(rs)/len(rs),
            "stdev": statistics.pstdev(rs) if len(rs) > 1 else 0.0,
            "n": len(rs),
            "per_seed": dict(zip([42,52,62], rs)),
        }

out = {
    "schema": "qquilt.utility.3seed.v1",
    "conventions": {
        "gguf_rows": "q_ppl / f16_gguf_ppl (llama-perplexity, sliding half-overlap, 50 chunks)",
        "hf_rows":   "version_ppl / bf16_hf_ppl (HF non-overlap, 50 windows)",
    },
    "f16_gguf_baseline": baselines,
    "per_seed": per_seed,
    "3seed_mean": mean_table,
}
(RES/"wave_1_utility/ppl_3seed_mean.json").write_text(json.dumps(out, indent=2))

# Markdown summary
md = ["# Utility -- 3-seed mean perplexity ratios", "",
      "Conventions: GGUF rows use llama-perplexity sliding-half-overlap with",
      "F16-GGUF as baseline; AWQ/HF rows use HF non-overlap with BF16 baseline.",
      "Per-seed F16-GGUF baselines (in / ood):",
      f"  * seed 42: {baselines[42]['in']:.4f} / {baselines[42]['ood']:.4f}",
      f"  * seed 52: {baselines[52]['in']:.4f} / {baselines[52]['ood']:.4f}",
      f"  * seed 62: {baselines[62]['in']:.4f} / {baselines[62]['ood']:.4f}", "",
      "## 3-seed mean ratios (n=3)", "",
      "| Version | In-domain ratio | OOD ratio |",
      "|---|---|---|"]
for v, label in [("bf16","BF16"), ("q8_0","Q8_0"), ("q5_k_m","Q5_K_M"),
                 ("q4_k_m","Q4_K_M"), ("awq_canary_free","AWQ-4bit")]:
    a = mean_table["in"].get(v); b = mean_table["ood"].get(v)
    if not a or not b: continue
    md.append(f"| {label} | {a['mean']:.3f} (σ={a['stdev']:.4f}) | {b['mean']:.3f} (σ={b['stdev']:.4f}) |")
(RES/"wave_1_utility/RESULTS_3SEED.md").write_text("\n".join(md) + "\n")
print("[3seed-fold] wrote ppl_3seed_mean.json + RESULTS_3SEED.md")
print()
print(open(RES/"wave_1_utility/RESULTS_3SEED.md").read())
PY
